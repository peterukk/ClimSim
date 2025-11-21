#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytorch model constructors based on vertical RNN architectures
"""
import os 
import gc
import sys
import torch
import torch.nn as nn
import torch.nn.parameter as Parameter
from layers import LayerPressure, LevelPressure
import torch.nn.functional as F
from typing import List, Tuple, Final, Optional
from torch import Tensor
from models_torch_kernels import GLU
from models_torch_kernels import *
import numpy as np 
from typing import Final 
import time

class BiRNN(nn.Module):
    """
    Basic model demonstrating the BiRNN approach from Ukkonen & Chantry (2025), Fig 1
    Not actually tested in this project
    """
    use_initial_mlp: Final[bool]
    def __init__(self, RNN_type='LSTM', 
                 nx = 9, nx_sfc=17, 
                 ny = 8, ny_sfc=8, 
                 nneur=(64,64), 
                 use_initial_mlp = True,
                 outputs_one_longer=False, # if True, inputs are a sequence
                 # of N and outputs a sequence of N+1 (e.g. predicting fluxes)
                 concat=False, 
                 out_scale=None, 
                 out_sfc_scale = None):
        # Simple bidirectional RNN (Either LSTM or GRU) for predicting column 
        # outputs shaped either (B, L, Ny) or (B, L+1, Ny) from column inputs
        # (B, L, Nx) and optionally surface inputs (B, Nx_sfc) 
        # If surface inputs exist, they are used to initialize first (upward) RNN 
        # Assumes top-of-atmosphere is first in memory i.e. at index 0 
        # if it's not the flip operations need to be moved!
        super(BiRNN, self).__init__()
        self.nx = nx
        self.ny = ny 
        self.nx_sfc = nx_sfc 
        self.ny_sfc = ny_sfc
        self.nneur = nneur 
        self.outputs_one_longer=outputs_one_longer
        if len(nneur) < 1 or len(nneur) > 3:
            sys.exit("Number of RNN layers and length of nneur should be 2 or 3")

        self.RNN_type=RNN_type
        if self.RNN_type=='LSTM':
            RNN_model = nn.LSTM
        elif self.RNN_type=='GRU':
            RNN_model = nn.GRU
        else:
            raise NotImplementedError()
                    
        self.concat=concat
        
        if out_scale is not None:
            cuda = torch.cuda.is_available() 
            device = torch.device("cuda" if cuda else "cpu")
            self.yscale_lev = torch.from_numpy(out_scale).to(device)
            self.yscale_sca = torch.from_numpy(out_sfc_scale).to(device)

        if self.nx_sfc > 0:
            self.mlp_surface1  = nn.Linear(nx_sfc, self.nneur[0])
            if self.RNN_type=="LSTM":
                self.mlp_surface2  = nn.Linear(nx_sfc, self.nneur[0])

        self.rnn1      = RNN_model(nx,            self.nneur[0], batch_first=True) # (input_size, hidden_size, num_layers=1
        self.rnn2      = RNN_model(self.nneur[0], self.nneur[1], batch_first=True)
        if len(self.nneur)==3:
            self.rnn3      = RNN_model(self.nneur[1], self.nneur[2], batch_first=True)

        # The final hidden variable is either the output from the last RNN, or
        # the  concatenated outputs from all RNNs
        if concat:
            nh_rnn = sum(nneur)
        else:
            nh_rnn = nneur[-1]

        self.mlp_output = nn.Linear(nh_rnn, self.ny)
        if self.ny_sfc>0:
            self.mlp_surface_output = nn.Linear(nneur[-1], self.ny_sfc)
            
    def postprocessing(self, out, out_sfc):
        out_denorm  = out / self.yscale_lev
        out_sfc_denorm  = out_sfc / self.yscale_sca
        return out_denorm, out_sfc_denorm
    
    def forward(self, inputs_main, inputs_aux):
            
        # Auxililiary surface-level variables are used to initialize the first, upward-iterating RNN (LSTM or GRU)
        sfc1 = self.mlp_surface1(inputs_aux)
        sfc1 = nn.Tanh()(sfc1)

        if self.RNN_type=="LSTM":
            sfc2 = self.mlp_surface2(inputs_aux)
            sfc2 = nn.Tanh()(sfc2)
            hidden = (sfc1.view(1,-1,self.nneur[0]), sfc2.view(1,-1,self.nneur[0])) # (h0, c0)
        else:
            hidden = (sfc1.view(1,-1,self.nneur[0]))

        # print(f'Using state1 {hidden}')
        # TOA is first in memory, so we need to flip the axis
        inputs_main = torch.flip(inputs_main, [1])
      
        out, hidden = self.rnn1(inputs_main, hidden)
        
        if self.outputs_one_longer:
            out = torch.cat((sfc1, out),axis=1)

        out = torch.flip(out, [1]) # the surface was processed first, but for
        # the second RNN (and the final output) we want TOA first, so flip again
        
        out2, hidden2 = self.rnn2(out) 
        
        (last_h, last_c) = hidden2

        if self.concat:
            rnnout = torch.cat((out2, out),axis=2)
        else:
            rnnout = out2
    
        out = self.mlp_output(rnnout)

        if self.ny_sfc>0:
            # use cell state or hidden state? likely doesn't matter
            out_sfc = self.mlp_surface_output(last_h.squeeze())
            return out, out_sfc
        else:
            return out 


class LSTM_autoreg_torchscript(nn.Module):
    """
    More advanced version of the biLSTM using a latent convective memory (Fig 10 in Ukkonen & Chantry,2025)
    with further options that include:
    separate_radiation: separate biLSTM at the end for radiation (although we don't have separate tendencies in ClimSim data), 
    physical_precip: attempt to incorporate some physics in the way precipitation is predicted,
    add_stochastic_layer: optional stochastic RNN after the deterministic layers 
    """
    use_initial_mlp: Final[bool]
    use_intermediate_mlp: Final[bool]
    add_pres: Final[bool]
    add_stochastic_layer: Final[bool]
    output_prune: Final[bool]
    separate_radiation: Final[bool]
    use_third_rnn: Final[bool]
    physical_precip: Final[bool]
    predict_liq_ratio: Final[bool]
    randomly_initialize_cellstate: Final[bool]
    output_sqrt_norm: Final[bool]
    concat: Final[bool]
    predict_background_precip: Final[bool]
    store_precip_and_include_evap: Final[bool]
    include_sedimentation_term: Final[bool]

    def __init__(self, hyam, hybm,  hyai, hybi,
                out_scale, out_sfc_scale, 
                xmean_lev, xmean_sca, xdiv_lev, xdiv_sca, lbd_qc, lbd_qi, lbd_qn,
                device,
                nlev=60, nx = 15, nx_sfc=24, ny = 5, ny0=5, ny_sfc=5, nneur=(192,192), 
                use_initial_mlp=False, # initial MLP layer (applied to each vertical level independently) 
                use_intermediate_mlp=True, # Intermediate MLP between RNNs and final MLP that predicts model outputs, 
                # useful for smaller convective memory (e.g. 16 per level)
                add_pres=False, # add layer pressure to inputs
                add_stochastic_layer=False,
                output_prune=False,
                repeat_mu=False, # repeat solar zenith angle to each vertical level?
                separate_radiation=False,
                use_third_rnn=False,
                mp_mode=0, # see train_rnn_rollout_torchscript_hydra
                physical_precip=False,
                predict_liq_ratio=False,
                randomly_initialize_cellstate=False, # introduce a smalld degree of randomness into deterministic LSTM
                concat=False,
                output_sqrt_norm=False,
                nh_mem=16):  # dimension of latent convective memory (per vertical level)
        super(LSTM_autoreg_torchscript, self).__init__()
        self.ny = ny 
        self.ny0 = ny  #for diagnose precip option, need to distinguish between model outputs (ny) and intermediate outputs (ny0)
        self.nlev = nlev 
        self.nlev_mem = nlev
        self.nx_sfc = nx_sfc 
        self.ny_sfc = ny_sfc
        self.ny_sfc0 = ny_sfc
        self.nneur = nneur 
        self.use_initial_mlp=use_initial_mlp
        self.output_prune = output_prune
        self.separate_radiation=separate_radiation
        self.physical_precip=physical_precip
        if self.physical_precip:
            #  predict autoconversion and evaporation tendencies separately (Perkins 2024) and add those to 
            # q tendency and precipitation source/sink. It would be nice to just diagnose the precipitation
            # but the issue is that in the CRM the precip doesn't down fall immediately, but we're not
            # tracking this in the training data. 
            # for this reason we try t predict precipitation "flux" and track of precip at each vertical level
            # P is then equal to P_old + dP_sourcesink + dP_flux
            print("warning: physical_precip is ON")
            self.ny0 = self.ny0 + 6   # , evaporation, autoconversion, and flux of precipitation
            if not self.separate_radiation:
              self.ny_sfc0 = self.ny_sfc0 - 2 # PRECC is computed from above using Eq 9.,
            # PRECSC is diagnosed using bottom temperature
            # learnable weight for t tendency from evaporation
            self.w1 = nn.Parameter(torch.randn(1), requires_grad=True)
            self.w2 = nn.Parameter(torch.randn(1), requires_grad=True)
            self.w3 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.predict_background_precip = False 
        self.predict_liq_ratio = predict_liq_ratio
        self.add_pres = add_pres
        self.index_q = nx - 1
        if self.add_pres:
            self.preslay = LayerPressure(hyam,hybm)
            nx = nx +1
            # self.preslay_nonorm = LayerPressure(hyam, hybm, norm=False)
            self.preslev_nonorm = LevelPressure(hyai, hybi)
        self.randomly_initialize_cellstate = randomly_initialize_cellstate
        self.nx = nx
        self.nh_rnn1 = self.nneur[0]
        self.nx_rnn2 = self.nneur[0]
        self.nh_rnn2 = self.nneur[1]
        self.use_third_rnn = use_third_rnn
        # if len(nneur)==3:
            # self.use_third_rnn = True 
        if self.use_third_rnn:
            self.nx_rnn3 = self.nneur[1]
            self.nh_rnn3 = self.nneur[2]
        # elif len(nneur)==2:
        #     self.use_third_rnn = False 
        # else:
        #     raise NotImplementedError()
        self.concat=concat
        self.output_sqrt_norm=output_sqrt_norm
        if self.output_sqrt_norm:
            print("Warning: output_sqrt_norm ON")
        self.repeat_mu = repeat_mu
        if self.repeat_mu:
            nx = nx + 1
        self.use_intermediate_mlp=use_intermediate_mlp  
        if self.use_intermediate_mlp:
            # self.nh_mem = self.nneur[1] // 4
            # if nh_mem is None:
            #     self.nh_mem = self.nneur[1] // 8
            # else:
            self.nh_mem = nh_mem
        else:
            self.nh_mem = self.nneur[1]

        # for physical_precip
        self.include_sedimentation_term = True 
        self.store_precip_and_include_evap = False
        if self.physical_precip and self.store_precip_and_include_evap:
          self.nh_mem0 = self.nh_mem - 1 
          self.ny_sfc0 = self.ny_sfc0 + 2
        else:
          self.nh_mem0 = self.nh_mem

        # in ClimSim config of E3SM, the CRM physics first computes 
        # moist physics on 50 levels, and then computes radiation on 60 levels!
        if self.separate_radiation:
            # self.nlev = 50
            self.nlev_mem = 50
            self.nlev_rad = 60
            # Rad inputs would in reality be the state variables (except winds) updated by the CRM, plus the gases:
            self.nx_rad_gas = 3 # 'pbuf_ozone', 'pbuf_CH4', 'pbuf_N2O'
            # We don't have the CRM state variables (T, qwv, qliq, qice) but pretend is in the output from the first BiRNN
            # mlp_output does a reduction from the RNN state to actual model outputs, it would be tempting to use the first 4 of these model outputs
            # (T, qwv, qlic, qice) but this would only be valid if radiation was calculated on the CRM domain mean.
            # However, it's probable that radiation was run on grouped CRM columns in climsim? https://doi.org/10.5194/gmd-2023-55 
            # Let's use the BiRNN  output (after intermediate reduction to nh_mem) and not final output so we don't make this assumption
            # that could be wrong
            self.nx_rad_crm = self.nh_mem0
            self.nx_rad_tot = self.nx_rad_gas + self.nx_rad_crm 
            nx = nx - 3
            self.nx_sfc_rad = 6 # 'pbuf_COSZRS' 'cam_in_ALDIF' 'cam_in_ALDIR' 'cam_in_ASDIF' 'cam_in_ASDIR' 'cam_in_LWUP' 
            self.nx_sfc = self.nx_sfc  - self.nx_sfc_rad
            self.ny_rad = 1
            self.ny_sfc_rad = self.ny_sfc0 - 2
            self.ny_sfc0 = 2
        
        if self.use_initial_mlp:
            self.nx_rnn1 = self.nneur[0]
            # should we include large-scale tendencies from the lowest level here?
        else:
            self.nx_rnn1 = nx
                
        self.add_stochastic_layer = add_stochastic_layer
        self.nonlin = nn.Tanh()
        self.relu = nn.ReLU()

        yscale_lev = torch.from_numpy(out_scale).to(device)
        yscale_sca = torch.from_numpy(out_sfc_scale).to(device)
        xmean_lev  = torch.from_numpy(xmean_lev).to(device)
        xmean_sca  = torch.from_numpy(xmean_sca).to(device)
        xdiv_lev   = torch.from_numpy(xdiv_lev).to(device)
        xdiv_sca   = torch.from_numpy(xdiv_sca).to(device)
        lbd_qc     = torch.from_numpy(lbd_qc).to(device)
        lbd_qi     = torch.from_numpy(lbd_qi).to(device)
        lbd_qn     = torch.from_numpy(lbd_qn).to(device)

        self.register_buffer('yscale_lev', yscale_lev)
        self.register_buffer('yscale_sca', yscale_sca)
        self.register_buffer('xmean_lev', xmean_lev)
        self.register_buffer('xmean_sca', xmean_sca)
        self.register_buffer('xdiv_lev', xdiv_lev)
        self.register_buffer('xdiv_sca', xdiv_sca)
        self.register_buffer('hyai', hyai)
        self.register_buffer('hybi', hybi)
        self.register_buffer("lbd_qc", lbd_qc)
        self.register_buffer("lbd_qi", lbd_qi)
        self.register_buffer("lbd_qn", lbd_qn)

        # self.rnn1_mem = None 
        print("Building RNN that feeds its hidden memory at t0,z0 to its inputs at t1,z0")
        print("Initial mlp: {}, intermediate mlp: {}".format(self.use_initial_mlp, self.use_intermediate_mlp))
 
        self.nx_rnn1 = self.nx_rnn1 + self.nh_mem0
        self.nh_rnn1 = self.nneur[0]
        
        print("nx rnn1", self.nx_rnn1, "nh rnn1", self.nh_rnn1)
        print("nx rnn2", self.nx_rnn2, "nh rnn2", self.nh_rnn2)  
        if self.use_third_rnn: print("nx rnn3", self.nx_rnn3, "nh rnn3", self.nh_rnn3) 
        print("nx sfc", self.nx_sfc)
        print("ny", self.ny, "ny0", self.ny0)

        if self.use_initial_mlp:
            self.mlp_initial = nn.Linear(nx, self.nneur[0])

        self.mlp_surface1  = nn.Linear(self.nx_sfc, self.nh_rnn1)
        if not self.randomly_initialize_cellstate:
            self.mlp_surface2  = nn.Linear(self.nx_sfc, self.nh_rnn1)

        # self.rnn1      = nn.LSTMCell(self.nx_rnn1, self.nh_rnn1)  # (input_size, hidden_size)
        # self.rnn2      = nn.LSTMCell(self.nx_rnn2, self.nh_rnn2)
        if self.use_third_rnn:

            self.rnn0   = nn.LSTM(self.nx_rnn1, self.nh_rnn1,  batch_first=True)
            self.rnn1   = nn.LSTM(self.nx_rnn2, self.nh_rnn2,  batch_first=True)  # (input_size, hidden_size)
            if self.add_stochastic_layer:
                use_bias=False
                self.rnn2 = MyStochasticLSTMLayer4(self.nx_rnn3, self.nh_rnn3, use_bias=use_bias)  
            else:
                self.rnn2   = nn.LSTM(self.nx_rnn3, self.nh_rnn3,  batch_first=True)
            self.rnn0.flatten_parameters()
        else:

            self.rnn1      = nn.LSTM(self.nx_rnn1, self.nh_rnn1,  batch_first=True)  # (input_size, hidden_size)
            if self.add_stochastic_layer:
                use_bias=False
                self.rnn2 = MyStochasticLSTMLayer4(self.nx_rnn2, self.nh_rnn2, use_bias=use_bias)  
            else:
                self.rnn2 = nn.LSTM(self.nx_rnn2, self.nh_rnn2,  batch_first=True)
                
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()
        self.sigmoid = nn.Sigmoid()

        if self.concat: 
            nh_rnn = self.nh_rnn1 + self.nh_rnn2
        else:
            nh_rnn = self.nh_rnn2

        if self.use_intermediate_mlp: 
            self.mlp_latent = nn.Linear(nh_rnn, self.nh_mem0)
            self.mlp_output = nn.Linear(self.nh_mem0, self.ny0)
        else:
            self.mlp_output = nn.Linear(nh_rnn, self.ny0)
            
        if self.physical_precip and self.store_precip_and_include_evap:
            self.mlp_precip_release = nn.Linear(nh_rnn, 1)
            self.softmax = nn.Softmax(dim=1)
        if self.predict_liq_ratio:
            self.mlp_predfrac = nn.Linear(nh_mem, 1)

        self.mlp_surface_output = nn.Linear(nneur[-1], self.ny_sfc0)
            
        if self.separate_radiation:
            self.nh_rnn1_rad = 96 
            self.nh_rnn2_rad = 96
            self.rnn1_rad      = nn.GRU(self.nx_rad_tot, self.nh_rnn1_rad,  batch_first=True)   # (input_size, hidden_size)
            self.rnn2_rad      = nn.GRU(self.nh_rnn1_rad, self.nh_rnn2_rad,  batch_first=True) 
            self.mlp_surface_rad = nn.Linear(self.nx_sfc_rad, self.nh_rnn1_rad)
            self.mlp_surface_output_rad = nn.Linear(self.nh_rnn2_rad, self.ny_sfc_rad)
            self.mlp_toa_rad  = nn.Linear(2, self.nh_rnn2_rad)
            self.mlp_output_rad = nn.Linear(self.nh_rnn2_rad, self.ny_rad)
            print("nx rnn1", self.nx_rnn1, "nh rnn1", self.nh_rnn1)
        else: 
            self.mlp_toa1  = nn.Linear(2, self.nh_rnn2)
            self.mlp_toa2  = nn.Linear(2, self.nh_rnn2)

    def temperature_scaling(self, T_raw):
        # T_denorm = T = T*(self.xmax_lev[:,0] - self.xmin_lev[:,0]) + self.xmean_lev[:,0]
        # T_denorm = T*(self.xcoeff_lev[2,:,0] - self.xcoeff_lev[1,:,0]) + self.xcoeff_lev[0,:,0]
        # liquid_ratio = (T_raw - 253.16) / 20.0 
        liquid_ratio = (T_raw - 253.16) * 0.05 
        liquid_ratio = F.hardtanh(liquid_ratio, 0.0, 1.0)
        return liquid_ratio
    
    def temperature_scaling_precip(self, temp_surface):
        snow_frac = (283.3 - temp_surface) /  14.6
        snow_frac = F.hardtanh(snow_frac, 0.0, 1.0)
        return snow_frac

    def postprocessing(self, out, out_sfc):
        out             = out / self.yscale_lev
        out_sfc         = out_sfc / self.yscale_sca
        if self.output_sqrt_norm:
            signs = torch.sign(out)
            out = signs*torch.pow(out, 4)
            # out = signs*torch.square(out)
        return out, out_sfc
        
    def pp_mp(self, out, out_sfc, x_denorm):

        out_denorm      = out / self.yscale_lev
        out_sfc_denorm  = out_sfc / self.yscale_sca
        # print("pp1 MEAN FRAC P", torch.mean(out_denorm[:,:,3]).item(), "MIN", torch.min(out_denorm[:,:,3]).item(),  "MAx", torch.max(out_denorm[:,:,3]).item() )

        if self.output_sqrt_norm:
            signs = torch.sign(out_denorm)
            out_denorm = signs*torch.pow(out_denorm, 4)
            # out_denorm = signs*torch.square(out_denorm)
        # print("pp_mp 1 frac100, ", out_denorm[100,:,3].detach().cpu().numpy())
        T_before        = x_denorm[:,:,0:1]
        qliq_before     = x_denorm[:,:,2:3]
        qice_before     = x_denorm[:,:,3:4]   
        qn_before       = qliq_before + qice_before 

        # print("shape x denorm", x_denorm.shape, "T", T_before.shape)
        T_new           = T_before  + out_denorm[:,:,0:1]*1200

        # T_new           = T_before  + out_denorm[:,:,0:1]*1200
        liq_frac_constrained    = self.temperature_scaling(T_new)

        if self.predict_liq_ratio:
            liq_frac_pred = out_denorm[:,:,3:4]
            # print("min max lfrac pred raw", torch.max(liq_frac_pred).item(), torch.min(liq_frac_pred).item())
            # Hu et al. Fig 2 b:
            max_frac = torch.clamp(liq_frac_constrained + 0.2, max=1.0)
            min_frac = torch.clamp(liq_frac_constrained - 0.2, min=0.0)
            # print("shape lfracpre", liq_frac_pred.shape, "con", liq_frac_constrained.shape)
            liq_frac_constrained = torch.clamp(liq_frac_pred, min=min_frac, max=max_frac)
            # print("pp2 MEAN FRAC P", torch.mean(liq_frac_constrained).item(), "MIN", torch.min(liq_frac_constrained).item(),  "MAx", torch.max(liq_frac_constrained).item() )

        # liq_frac_constrained = out_denorm[:,:,3:4]
        # print("pp_mp 2 frac100, ", liq_frac_constrained[100,:].detach().cpu().numpy())

        #                            dqn
        # print("mean DQN", torch.mean(torch.sum(out_denorm[:,:,2:3],1)))
        qn_new      = qn_before + out_denorm[:,:,2:3]*1200  
        qliq_new    = liq_frac_constrained*qn_new
        qice_new    = (1-liq_frac_constrained)*qn_new
        dqliq       = (qliq_new - qliq_before) * 0.0008333333333333334  #/1200  
        dqice       = (qice_new - qice_before) * 0.0008333333333333334  #/1200  
        sum = dqliq+dqice 
        # print("pp_mp mean dq liq", torch.mean(dqliq).item(), "dq ice", torch.mean(dqice).item(), "dq TOT", torch.mean( out_denorm[:,:,2:3]).item())
        # print(( "dq TOT", torch.mean( out_denorm[:,:,2:3]).item(), "dqliq+ice",torch.mean( sum).item() ))

        if self.predict_liq_ratio:           # replace    dqn,   liqfrac
            out_denorm  = torch.cat((out_denorm[:,:,0:2], dqliq, dqice, out_denorm[:,:,4:]),dim=2)
        else:
            out_denorm  = torch.cat((out_denorm[:,:,0:2], dqliq, dqice, out_denorm[:,:,3:]),dim=2)

        return out_denorm, out_sfc_denorm
    
    def forward(self, inp_list : List[Tensor]):
        inputs_main   = inp_list[0]
        inputs_aux    = inp_list[1]
        rnn1_mem      = inp_list[2]

        batch_size = inputs_main.shape[0]
        # print("shape inputs main", inputs_main.shape)
        # print("max q", torch.max(inputs_main[:,:,-1]), "shape", inputs_main.shape)

        if self.add_pres:
            sp = torch.unsqueeze(inputs_aux[:,0:1],1)
            # undo scaling
            # sp = sp*35451.17 + 98623.664 
            sp = sp*self.xdiv_sca[0] + self.xmean_sca[0]
            pres  = self.preslay(sp)
            # pres_nonorm  = torch.squeeze(self.preslay_nonorm(sp))
            pres_nonorm  = torch.squeeze(self.preslev_nonorm(sp))
            inputs_main = torch.cat((inputs_main,pres),dim=2)

        if self.repeat_mu:
            mu = torch.reshape(inputs_aux[:,6:7],(-1,1,1))
            mu_rep = torch.repeat_interleave(mu,self.nlev,dim=1)
            inputs_main = torch.cat((inputs_main,mu_rep),dim=2)
            
        if self.separate_radiation:
            # Do not use inputs -2,-3,-4 (O3, CH4, N2O) or first 10 levels
            inputs_main_crm = torch.cat((inputs_main[:,10:,0:-4], inputs_main[:,10:,-1:]),dim=2)
        else:
            inputs_main_crm = inputs_main
            
        if self.use_initial_mlp:
            inputs_main_crm = self.mlp_initial(inputs_main_crm)
            inputs_main_crm = self.nonlin(inputs_main_crm)  
            
        # if self.use_memory:
        # rnn1_input = torch.cat((rnn1_input,self.rnn1_mem), axis=2)
        if self.physical_precip and self.store_precip_and_include_evap: 
          P_old = rnn1_mem[:,-1,-1] 
        
        inputs_main_crm = torch.cat((inputs_main_crm,rnn1_mem[:,:,0:self.nh_mem0]), dim=2)
            
        if self.use_third_rnn: # use initial downward RNN
            hx0 = torch.randn((batch_size, self.nh_rnn1),device=inputs_main.device)  # (batch, hidden_size)
            cx0 = torch.randn((batch_size, self.nh_rnn1),device=inputs_main.device)
            hidden0 = (torch.unsqueeze(hx0,0), torch.unsqueeze(cx0,0))  
            rnn0out, states = self.rnn0(inputs_main_crm, hidden0)
            
            rnn1_input =  torch.flip(rnn0out, [1])
        else:
            # TOA is first in memory, so to start at the surface we need to go backwards
            rnn1_input = torch.flip(inputs_main_crm, [1])
        
        # The input (a vertical sequence) is concatenated with the
        # output of the RNN from the previous time step 

        if self.separate_radiation:
            inputs_sfc = torch.cat((inputs_aux[:,0:6],inputs_aux[:,12:]),dim=1)
        else:
            inputs_sfc = inputs_aux
        hx = self.mlp_surface1(inputs_sfc)
        hx = self.nonlin(hx)
        if self.randomly_initialize_cellstate:
            cx = torch.randn_like(hx)
        else:
            cx = self.mlp_surface2(inputs_sfc)
        # cx = self.nonlin(cx)
        hidden = (torch.unsqueeze(hx,0), torch.unsqueeze(cx,0))

        rnn1out, states = self.rnn1(rnn1_input, hidden)
        del rnn1_input

        # if self.predict_flux:
        #     rnn1out = torch.cat((torch.unsqueeze(hx,1),rnn1out),dim=1)

        rnn1out = torch.flip(rnn1out, [1])

        if self.separate_radiation:
          hx2 = torch.randn((batch_size, self.nh_rnn2),device=inputs_main.device)  # (batch, hidden_size)
          cx2 = torch.randn((batch_size, self.nh_rnn2),device=inputs_main.device)
        else: 
          inputs_toa = torch.cat((inputs_aux[:,1:2], inputs_aux[:,6:7]),dim=1) 
          hx2 = self.mlp_toa1(inputs_toa)
          cx2 = self.mlp_toa2(inputs_toa)
        
        if self.add_stochastic_layer:
            input_rnn2 = torch.transpose(rnn1out,0,1)
            hidden2 = (hx2, cx2)
        else:
            input_rnn2 = rnn1out
            hidden2 = (torch.unsqueeze(hx2,0), torch.unsqueeze(cx2,0))

        if self.add_stochastic_layer and self.use_third_rnn:
          z, states = self.rnn2(input_rnn2, hidden2)
          # SPPT
        #   z = F.hardtanh(z, 0.0, 2.0)
          rnn2out = z*rnn1out
          # rnn2out = rnn1out + 0.01*z 
        #   rnn2out = z
          rnn2out = torch.transpose(rnn2out,0,1)

        else:
          rnn2out, states = self.rnn2(input_rnn2, hidden2)

        del input_rnn2

        (last_h, last_c) = states
        final_sfc_inp = last_h.squeeze() 
            
        if self.concat:
            rnn2out = torch.cat((rnn1out, rnn2out), dim=2)
        
        if self.use_intermediate_mlp: 
            rnn2out = self.mlp_latent(rnn2out)
          
        # if self.use_memory:
        if self.use_third_rnn:
            rnn1_mem = rnn2out
        else:
            # rnn1_mem = torch.flip(rnn2out, [1])
            rnn1_mem = rnn2out

        out = self.mlp_output(rnn2out)

        if self.output_prune and (not self.separate_radiation):
            # Only temperature tendency is computed for the top 10 levels
            # if self.separate_radiation:
            #     out[:,0:12,:] = out[:,0:12,:].clone().zero_()
            # else:
            out[:,0:12,1:] = out[:,0:12,1:].clone().zero_()
        
        if not (self.physical_precip and self.separate_radiation):
            out_sfc = self.mlp_surface_output(final_sfc_inp)
                
        if self.separate_radiation:
            # out_crm = out.clone()
            out_new = torch.zeros(batch_size, self.nlev_rad, self.ny0, device=inputs_main.device)
            out_new[:,10:,:] = out
            # Start at surface again
            inputs_gas_rad =  inputs_main[:,:,12:15] # gases
            # # add dT from crm 
            # T_old =   inputs_main * (self.xcoeff_lev[2,:,0:1] - self.xcoeff_lev[1,:,0:1]) + self.xcoeff_lev[0,:,0:1] 
            # T_new = T_old + dT
            # inputs_rad =  torch.zeros(batch_size, self.nlev_rad, self.nh_mem+self.nx_rad,device=inputs_main.device)
            inputs_rad =  torch.zeros(batch_size, self.nlev_rad, self.nx_rad_tot, device=inputs_main.device)

            # inputs_rad[:,10:,0:self.nh_mem] = torch.flip(rnn2out, [1])
            # inputs_rad[:,:,self.nh_mem:] = inputs_main_rad
            inputs_rad[:,:,0:3] = inputs_gas_rad
            inputs_rad[:,10:,3:] = torch.flip(rnn1_mem, [1])
            # inputs_rad = torch.flip(inputs_rad, [1])

            inputs_sfc_rad = inputs_aux[:,6:12]
            hx = self.mlp_surface_rad(inputs_sfc_rad)
            hidden = (torch.unsqueeze(hx,0))
            rnn_out, states = self.rnn1_rad(inputs_rad, hidden)
            rnn_out = torch.flip(rnn_out, [1])

            inputs_toa = torch.cat((inputs_aux[:,1:2], inputs_aux[:,6:7]),dim=1) 
            hx2 = self.mlp_toa_rad(inputs_toa)

            rnn_out, last_h = self.rnn2_rad(rnn_out, hidden)
            out_rad = self.mlp_output_rad(rnn_out)

            out_sfc_rad = self.mlp_surface_output_rad(last_h)
            # dT_tot = dT_crm + dT_rad
            out_new[:,:,0:1] = out_new[:,:,0:1] + out_rad
            out = out_new
            out_sfc_rad = torch.squeeze(out_sfc_rad)
            # rad predicts everything except PRECSC, PRECC
            if not self.physical_precip:
                out_sfc =  torch.cat((out_sfc_rad[:,0:2], out_sfc, out_sfc_rad[:,2:]),dim=1)

        if self.physical_precip:
            if self.separate_radiation:
                ilev_crm = 10
            else:
                ilev_crm = 0
            #  ['ptend_t', 'ptend_q0001', 'ptend_qn', 'ptend_u', 'ptend_v']
            #  ['ptend_t', 'ptend_q0001', 'ptend_qn', 'liq_frac', 'ptend_u', 'ptend_v']
            out_neww = torch.zeros(batch_size, self.nlev, self.ny, device=inputs_main.device)
            out_neww[:,:,0] = out[:,:,0]
            out_neww[:,:,3:self.ny] = out[:,:,3:self.ny]
            dqv_evap_prec = out[:,ilev_crm:,1] # Evaporation of precipitation (precip to water vapor = sink of precip, source of qv)
            dqn_aa        = out[:,ilev_crm:,2] # Accretion/autoconversion (cloud water to precip. = source of precip, sink of qn)
            dqn_evap_cond_vapor = out[:,ilev_crm:,3] # evaporation - condensation  (cloud water to water vapor is positive) 
            
            flux_mult_coeff = 1.0e3
            if self.include_sedimentation_term:
                flux_net_qv   = flux_mult_coeff*out[:,ilev_crm:,self.ny]
                flux_net_qn   = flux_mult_coeff*out[:,ilev_crm:,self.ny+1]
                # FORCE FLUX TO SURFACE (sedimentation) TO BE POSITIVE 
                # if it's negative, we have negative precipitation and water is taken from nowhere and added to the atmosphere
                # OR should we not bother to include a sedimentation term, and set boundary fluxes to zero to avoid net transport 
                flux_net_qv[:,-1] = self.relu(flux_net_qv[:,-1]) # net downward flux is positive by convention
                flux_net_qn[:,-1] = self.relu(flux_net_qn[:,-1])
            else:
                flux_net_qv   = flux_mult_coeff*out[:,ilev_crm+1:,self.ny]
                flux_net_qn   = flux_mult_coeff*out[:,ilev_crm+1:,self.ny+1]

            g = torch.tensor(9.806650000)
            one_over_g = torch.tensor(0.1019716213)
            scaling_factor = -g # tendency equation in pressure coordinates has -g in front
            zeroes = torch.zeros(batch_size, 1, device=inputs_main.device)
              
            pres_diff = pres_nonorm[:,ilev_crm+1:] - pres_nonorm[:,ilev_crm:-1]
            if self.include_sedimentation_term:
                flux_net_qv = torch.cat((zeroes,flux_net_qv),dim=1)
                flux_net_qn = torch.cat((zeroes,flux_net_qn),dim=1)
            else:
                flux_net_qv = torch.cat((zeroes,flux_net_qv, zeroes),dim=1)
                flux_net_qn = torch.cat((zeroes,flux_net_qn, zeroes),dim=1)
            flux_diff_qv = flux_net_qv[:,1:] - flux_net_qv[:,0:-1]
            flux_diff_qn = flux_net_qn[:,1:] - flux_net_qn[:,0:-1]
            flux_qv_dp = scaling_factor*(flux_diff_qv / pres_diff) 
            flux_qn_dp = scaling_factor*(flux_diff_qn / pres_diff) 

            dqn_aa        = self.relu(dqn_aa) # force positive
            dqv_evap_prec = self.relu(dqv_evap_prec) # force positive

            if self.store_precip_and_include_evap:

                Pmax = 5e3
                pour_excess=False
                if not pour_excess:
                    # Cap the amount of evaporated precipitation to the stored amount
                    # This should help with preventing negative precipitation
                    # vert_sum_evap = torch.sum(dqv_evap_prec,1)
                    # fac_scale_down_evap = P_old / vert_sum_evap
                    # fac_scale_down_evap = torch.clamp(fac_scale_down_evap, max=1.0)
                    # dqv_evap_prec = torch.unsqueeze(fac_scale_down_evap,1) * dqv_evap_prec 
                    # above leads to NaN loss, use another method where we use softmax to get ratios (that sum to 1)
                    # which determine how the total amount should be distributed vertically
                    evap_prec_tot = out_sfc[:,-1]
                    evap_prec_tot = self.relu(evap_prec_tot) # force positive
                    # print("model mean min max evap_prec_tot",  torch.mean(evap_prec_tot).item(), torch.min(evap_prec_tot).item(), torch.max(evap_prec_tot).item())
                    evap_prec_tot = torch.clamp(evap_prec_tot, max=P_old)
                    # print("model mean min max Pold",  torch.mean(P_old).item(), torch.min(P_old).item(), torch.max(P_old).item())
                    # print("model mean min max evap_prec_tot2",  torch.mean(evap_prec_tot).item(), torch.min(evap_prec_tot).item(), torch.max(evap_prec_tot).item())
                    dqv_evap_prec = self.softmax(dqv_evap_prec)
                    dqv_evap_prec = dqv_evap_prec * torch.unsqueeze(evap_prec_tot,1)
                    # print("model mean dqv_evap_prec",  torch.mean(torch.sum(dqv_evap_prec,1)).item())

                    # If we're including evaporation of precipitation (a sink of precip.), we need to be careful
                    # about how large we allow its source (dqn_aa) to be if we also allow precipitation to be stored.
                    # (Which we should because the stored precipitation should determine the upper limit for evaporation)
                    # This is because to avoid stored precipitation to just grow indefinitely we (previously) clamped
                    # it to some maximum value. However, this can break water conservation by removing (Pnew - Pmax) water 
                    # and not adding it anywhere again. To avoid this, instead of capping Pnew at Pmax, we need to make sure 
                    # the source dqn_aa doesn't become so large that Pnew exceeds Pmax 
                    # max_dqn_aa = Pmax - Pold +  dqv_evap_prec
                    # dqn_aa      = self.relu(dqn_aa) # force positive
                    dqn_aa_tot  = out_sfc[:,-2]
                    # print("max Pold", torch.max(P_old).item(), "evap", torch.max(evap_prec_tot).item())
                    # print("min Pold", torch.min(P_old).item(), "evap", torch.min(evap_prec_tot).item())
                    max_dqn_aa  = Pmax - P_old + evap_prec_tot
                    # print("max min max_dqn_aa", torch.max(max_dqn_aa).item(), torch.min(max_dqn_aa).item())
                    # print("max min dqn_aa_tot", torch.max(dqn_aa_tot).item(), torch.min(dqn_aa_tot).item())
                    dqn_aa_tot  = torch.clamp(dqn_aa_tot, max=max_dqn_aa)
                    # print("max dqn_aa_tot 2", torch.max(dqn_aa_tot).item())
                    # print("max min dqn aa tot", torch.max(dqn_aa_tot).item(), torch.min(dqn_aa_tot).item())
                    dqn_aa      = self.softmax(dqn_aa)
                    dqn_aa      = dqn_aa * torch.unsqueeze(dqn_aa_tot,1)

                #                                    (evap-cond)>0 from vapor,  evap. from prec. both add water vapor  
                out_neww[:,ilev_crm:,1] = flux_qv_dp + dqn_evap_cond_vapor     + dqv_evap_prec     
                #                                    (evap-cond)>0 removes,     acc-au removes cldwater  
                out_neww[:,ilev_crm:,2] = flux_qn_dp - dqn_evap_cond_vapor     - dqn_aa  
                 #                      source,       sink   of precipitation  (note: signs already reversed w.r.t. above)
                d_precip_sourcesink    = dqn_aa      - dqv_evap_prec 

            else:
                dqn_aa        = self.relu(dqn_aa) # force positive

                out_neww[:,ilev_crm:,1] = flux_qv_dp + dqn_evap_cond_vapor 
                out_neww[:,ilev_crm:,2] = flux_qn_dp - dqn_evap_cond_vapor      - dqn_aa  
                d_precip_sourcesink    = dqn_aa           

                # min_dqn_evap_cond_vapor = qv_before + flux_qv_dp
                # max_dqn_evap_cond_vapor = qn_before + flux_qn_dp - dqn_aa

                # dqn_evap_cond_vapor = torch.clamp(dqn_evap_cond_vapor, min=min_dqn_evap_cond_vapor,max=max_dqn_evap_cond_vapor)

                # dqv_tot = flux_qv_dp + dqn_evap_cond_vapor 
                # dqn_tot = flux_qn_dp - dqn_evap_cond_vapor      - dqn_aa  

            # This should be positive to avoid negative precipitation!

            # print("model mean max min dqn_evap_cond_vapor", torch.mean(dqn_evap_cond_vapor).item(),  torch.max(dqn_evap_cond_vapor).item(), torch.min(dqn_evap_cond_vapor).item())
            # print("model mean max min dqv_evap_prec", torch.mean(dqv_evap_prec).item(),  torch.max(dqv_evap_prec).item(), torch.min(dqv_evap_prec).item())
            # print("model mean dqn_aa", torch.mean(dqn_aa).item())
            # print("model mean max min d_precip_sourcesink", torch.mean(d_precip_sourcesink).item(),  torch.max(d_precip_sourcesink).item(), torch.min(d_precip_sourcesink).item())

            out = out_neww 
            
            if self.include_sedimentation_term:
              sedimentation = ( flux_net_qv[:,-1] + flux_net_qn[:,-1] )
            else:
              sedimentation = 0

            dp_water = (one_over_g*pres_diff*d_precip_sourcesink)
          
            if self.store_precip_and_include_evap:
                water_new = torch.sum(dp_water,1)  
                water_new = self.relu(water_new)
                water_new = P_old + water_new
                precc_release_fraction = torch.sigmoid(self.mlp_precip_release(last_h)).squeeze()
                # print("precc_release_fraction ", precc_release_fraction.shape, "water_new", water_new.shape)
                water_released = precc_release_fraction*water_new
                water_stored  = water_new*(1-precc_release_fraction)
                # Just clipping the stored water here is incorrect, because we break the conservation! 
                # Instead compute the excess and add it to precipitation?
                if pour_excess:
                    water_excess = water_stored - Pmax
                    water_excess = self.relu(water_excess)
                    water_stored = water_stored - water_excess
                water_stored_lev  = torch.unsqueeze(water_stored,dim=1)
                water_stored_lev = torch.unsqueeze(torch.repeat_interleave(water_stored_lev,self.nlev,dim=1),dim=2)
                rnn1_mem = torch.cat((rnn1_mem[:,:,0:self.nh_mem], water_stored_lev),dim=2)
                if pour_excess:
                    precip=  sedimentation + water_released + water_excess
                else:
                    precip=  sedimentation + water_released 

                # print("model min max mean dpwater", torch.min(water_released).item(), torch.max(water_released).item(), torch.mean(water_released).item())
                # print("model min max mean P_old", torch.min(P_old).item(), torch.max(P_old).item(), torch.mean(P_old).item())
                # print("model min max mean P_new", torch.min(water_stored).item(), torch.max(water_stored).item(), torch.mean(water_stored).item())

            else:
                precip =  sedimentation + torch.sum(dp_water,1)  # <-- we already reversed signs in d_precip_sourcesink

            # d_water_tot = out[:,ilev_crm:,1] + out[:,ilev_crm:,2]
            # vint_q_change =  torch.sum(one_over_g*pres_diff*d_water_tot,1)
            # vint_q_sourcesink = torch.sum(one_over_g*pres_diff*d_precip_sourcesink,1)
            # print("model PREC", torch.mean(precip).item(), "VINT TOT Q CHANGE", torch.mean(vint_q_change).item(), "vint sourcesink", torch.mean(vint_q_sourcesink).item())
            # # print("model min max mean dpwater", torch.min(dp_water).item(), torch.max(dp_water).item())

            # ind = 100
            # print("model PREC", precip[100].item(), "VINT TOT Q CHANGE", vint_q_change[100].item())
            # print("P old", P_old[100].item(), "water_stored", water_stored1[100].item())
            # print("dqv evap", torch.sum(dqv_evap_prec[ind,:]).item())
            # print( "dqn_aa", torch.sum(dqn_aa[ind,:]).item(), "DPRECSS", torch.sum(d_precip_sourcesink[ind,:]).item() )
            # print("dqn_evap_cond_vapor", torch.sum(dqn_evap_cond_vapor[ind,:]).item() )
            # print("flux_qv_dp", torch.sum(flux_qv_dp[ind,:]).item(), "flux_qn_dp", torch.sum(flux_qn_dp[ind,:]).item()  )
            # print("sedimentation", sedimentation[ind].item(), "dpwater",  water_released[ind].item(), "prec", precip[ind].item())
            # precc = self.yscale_sca[3] * (precip_nonorm/1000) 
            precc = (precip/1000) 
            precc = precc.unsqueeze(1)

            temp_sfc = (inputs_main[:,-1,0:1]*self.xdiv_lev[-1,0:1]) + self.xmean_lev[-1,0:1]
            snowfrac = self.temperature_scaling_precip(temp_sfc)
            precsc = snowfrac*precc
            if self.store_precip_and_include_evap:
              ny_sfc = self.ny_sfc0 - 2
            else:
              ny_sfc = self.ny_sfc0 
            if self.separate_radiation:
                out_sfc_rad = self.relu(out_sfc_rad)
                # print("ny sfc", ny_sfc, "out sfc rad", out_sfc_rad.shape)
                out_sfc =  torch.cat((out_sfc_rad[:,0:2], precsc, precc, out_sfc_rad[:,2:]),dim=1)
            else:
                out_sfc = self.relu(out_sfc)
                out_sfc =  torch.cat((out_sfc[:,0:2], precsc, precc, out_sfc[:,2:ny_sfc]),dim=1) 
        else:
            out_sfc = self.relu(out_sfc)

        # out_sfc = self.relu(out_sfc)
        if self.predict_liq_ratio:
            temp = (inputs_main[:,:,0]*self.xdiv_lev[:,0]) + self.xmean_lev[:,0]  
            temp = temp + (out[:,:,0]/self.yscale_lev[:,0]) * 1200
            liq_frac_diagnosed0    = self.temperature_scaling(temp)
            liq_frac_diagnosed = liq_frac_diagnosed0[:,ilev_crm:]
            # liq_frac_pred = out[:,:,3] + liq_frac_diagnosed*self.yscale_lev[:,3]
            # liq_frac_pred[temp<250.0] = 0.0 
            # liq_frac_pred[temp>275.0] = 1.0 
            # out[:,:,3] = self.relu(liq_frac_pred)
            temp = temp[:,ilev_crm:]
            # mem = rnn1_mem[:,ilev_crm:]

            inds = (temp < 275.0) & (temp<250.0)
            x_predfrac = rnn1_mem[inds]
            liq_frac_pred = self.mlp_predfrac(x_predfrac)
            liq_frac_pred = torch.reshape(liq_frac_pred,(-1,))
            liq_frac_pred = liq_frac_pred.to(liq_frac_diagnosed.dtype)
            liq_frac_diagnosed[inds] = liq_frac_pred
            out[:,ilev_crm:,3] = self.relu(liq_frac_diagnosed)
            out[:,0:ilev_crm,3] = liq_frac_diagnosed0[:,0:ilev_crm]
        # print("shape out pred", out.shape, "sfc", out_sfc.shape)
        # print("model PREC fin2",torch.mean(out_sfc[:,3:4]).item())
        return out, out_sfc, rnn1_mem
    
    
class LSTM_autoreg_torchscript_perturb(nn.Module):
    """
    Deterministic biLSTM + single stochastic RNN that adds a multiplicative perturbation
    to the latent state, similar to SPPT
    """
    use_initial_mlp: Final[bool]
    add_pres: Final[bool]
    output_prune: Final[bool]
    separate_radiation: Final[bool]
    return_det: Final[bool]
    det_mode: Final[bool]
    use_stochastic_lstm: Final[bool]
    deterministic_mode: Final[bool]
    randomly_initialize_cellstate: Final[bool]
    
    def __init__(self, hyam, hybm,  hyai, hybi,
                out_scale, out_sfc_scale, 
                xmean_lev, xmean_sca, xdiv_lev, xdiv_sca,
                device,
                nlev=60, nx = 15, nx_sfc=24, ny = 5, ny_sfc=5, nneur=(192,192), 
                use_initial_mlp=False, 
                add_pres=False,
                output_prune=False,
                use_stochastic_lstm=True,
                deterministic_mode=False,
                separate_radiation=False,
                randomly_initialize_cellstate=False,
                return_det=True,
                nh_mem=16):
        super(LSTM_autoreg_torchscript_perturb, self).__init__()
        self.ny = ny 
        self.nlev = nlev 
        self.nlev_mem= nlev
        self.nx_sfc = nx_sfc 
        self.ny_sfc = ny_sfc
        self.ny_sfc0 = ny_sfc
        self.nneur = nneur 
        self.use_initial_mlp=use_initial_mlp
        self.output_prune = output_prune
        self.add_pres = add_pres
        if self.add_pres:
            self.preslay = LayerPressure(hyam,hybm)
            nx = nx +1
        self.return_det = return_det
        self.deterministic_mode=deterministic_mode
        self.nx = nx
        self.use_stochastic_lstm = use_stochastic_lstm
        self.separate_radiation=separate_radiation
        if self.separate_radiation:
            print("Model config separate_radiation is ON!")
            if self.return_det: raise NotImplementedError("separate_rad not compatible with return_det")
            # self.nlev = 50
            self.nlev_mem = 50
            self.nlev_rad = 60
            self.nx_rad = self.nx - 2
            # self.nx_rad = nx - 2
            # self.nx_rnn1 = self.nx_rnn1 - 3
            nx = nx - 3
            self.nx_sfc_rad = 5
            self.nx_sfc = self.nx_sfc  - self.nx_sfc_rad
            self.ny_rad = 1
            self.ny_sfc_rad = self.ny_sfc0 - 2
            self.ny_sfc0 = 2
        self.randomly_initialize_cellstate = randomly_initialize_cellstate
        self.nh_rnn1 = self.nneur[0]
        self.nx_rnn2 = self.nneur[0]
        self.nh_rnn2 = self.nneur[1]
        self.nx_rnn3 = self.nneur[1]
        self.nh_rnn3 = self.nneur[1]

        if self.use_initial_mlp:
            self.nx_rnn1 = self.nneur[0]
        else:
            self.nx_rnn1 = nx
                
        self.nonlin = nn.Tanh()
        self.relu = nn.ReLU()
        # self.hardtanh = nn.Hardtanh(0,2.0)

        yscale_lev = torch.from_numpy(out_scale).to(device)
        yscale_sca = torch.from_numpy(out_sfc_scale).to(device)
        xmean_lev  = torch.from_numpy(xmean_lev).to(device)
        xmean_sca  = torch.from_numpy(xmean_sca).to(device)
        xdiv_lev   = torch.from_numpy(xdiv_lev).to(device)
        xdiv_sca   = torch.from_numpy(xdiv_sca).to(device)
        
        self.register_buffer('yscale_lev', yscale_lev)
        self.register_buffer('yscale_sca', yscale_sca)
        self.register_buffer('xmean_lev', xmean_lev)
        self.register_buffer('xmean_sca', xmean_sca)
        self.register_buffer('xdiv_lev', xdiv_lev)
        self.register_buffer('xdiv_sca', xdiv_sca)
        self.register_buffer('hyai', hyai)
        self.register_buffer('hybi', hybi)
        self.use_intermediate_mlp = True
        self.nh_mem = nh_mem
        print("Building RNN that feeds its hidden memory at t0,z0 to its inputs at t1,z0")
        print("Initial mlp: {}, intermediate mlp: {}".format(self.use_initial_mlp, self.use_intermediate_mlp))
 
        self.nx_rnn1 = self.nx_rnn1 + self.nh_mem
        self.nh_rnn1 = self.nneur[0]
        
        print("nx rnn1", self.nx_rnn1, "nh rnn1", self.nh_rnn1)
        print("nx rnn2", self.nx_rnn2, "nh rnn2", self.nh_rnn2)  
        # if self.use_third_rnn: print("nx rnn3", self.nx_rnn3, "nh rnn3", self.nh_rnn3) 
        print("nx sfc", self.nx_sfc)

        if self.use_initial_mlp:
            self.mlp_initial = nn.Linear(nx, self.nneur[0])

        self.mlp_surface1  = nn.Linear(self.nx_sfc, self.nh_rnn1)
        self.mlp_surface2  = nn.Linear(self.nx_sfc, self.nh_rnn1)

        self.rnn1   = nn.LSTM(self.nx_rnn1, self.nh_rnn1,  batch_first=True)  # (input_size, hidden_size)
        self.rnn2   = nn.LSTM(self.nx_rnn2, self.nh_rnn2,  batch_first=True)
        use_bias=False
        if self.use_stochastic_lstm:
            stochastic_layer = MyStochasticLSTMLayer4 
        else:
        # stochastic_layer = MyStochasticGRULayer
            stochastic_layer = MyStochasticGRULayer5
        # stochastic_layer = MyStochasticLSTMLayer2

        if not self.deterministic_mode:
          self.rnn3 = stochastic_layer(self.nx_rnn3, self.nh_rnn3, use_bias=use_bias) 

        nh_rnn = self.nh_rnn2
        self.mlp_latent = nn.Linear(nh_rnn, self.nh_mem)
        self.mlp_output = nn.Linear(self.nh_mem, self.ny)

        self.mlp_surface_output = nn.Linear(nneur[-1], self.ny_sfc0)
        if self.separate_radiation:
            self.nh_rnn1_rad = 96 
            self.nh_rnn2_rad = 96
            # self.rnn1_rad      = nn.GRU(self.nh_mem+self.nx_rad, self.nh_rnn1_rad,  batch_first=True)   # (input_size, hidden_size)
            self.rnn1_rad      = nn.GRU(4+self.nx_rad, self.nh_rnn1_rad,  batch_first=True)   # (input_size, hidden_size)
            self.rnn2_rad      = nn.GRU(self.nh_rnn1_rad, self.nh_rnn2_rad,  batch_first=True) 
            self.mlp_surface_rad = nn.Linear(self.nx_sfc_rad, self.nh_rnn1_rad)
            self.mlp_surface_output_rad = nn.Linear(self.nh_rnn2_rad, self.ny_sfc_rad)
            self.mlp_toa_rad  = nn.Linear(2, self.nh_rnn2_rad)
            self.mlp_output_rad = nn.Linear(self.nh_rnn2_rad, self.ny_rad)
            print("nx rnn1", self.nx_rnn1, "nh rnn1", self.nh_rnn1)
        else: 
            self.mlp_toa1  = nn.Linear(2, self.nh_rnn2)
            if not self.randomly_initialize_cellstate:
                self.mlp_toa2  = nn.Linear(2, self.nh_rnn2)

    def temperature_scaling(self, T_raw):
        # T_denorm = T = T*(self.xmax_lev[:,0] - self.xmin_lev[:,0]) + self.xmean_lev[:,0]
        # T_denorm = T*(self.xcoeff_lev[2,:,0] - self.xcoeff_lev[1,:,0]) + self.xcoeff_lev[0,:,0]
        # liquid_ratio = (T_raw - 253.16) / 20.0 
        liquid_ratio = (T_raw - 253.16) * 0.05 
        liquid_ratio = F.hardtanh(liquid_ratio, 0.0, 1.0)
        return liquid_ratio
    
    def temperature_scaling_precip(self, temp_surface):
        snow_frac = (283.3 - temp_surface) /  14.6
        snow_frac = F.hardtanh(snow_frac, 0.0, 1.0)
        return snow_frac
    
    def forward_rad(self, inputs_main, inputs_aux, inputs_toa, out):
        batch_size = inputs_main.shape[0]  
        # out_crm = out.clone()
        out_new = torch.zeros(batch_size, self.nlev_rad, self.ny, device=inputs_main.device)
        out_new[:,10:,:] = out
        # Start at surface again
        # Do not use inputs 4,5 (winds)
        inputs_main_rad =  torch.cat((inputs_main[:,:,0:4], inputs_main[:,:,6:]),dim=2)
        # # add dT from crm 
        # T_old =   inputs_main * (self.xcoeff_lev[2,:,0:1] - self.xcoeff_lev[1,:,0:1]) + self.xcoeff_lev[0,:,0:1] 
        # T_new = T_old + dT
        # inputs_rad =  torch.zeros(batch_size, self.nlev_rad, self.nh_mem+self.nx_rad,device=inputs_main.device)
        inputs_rad =  torch.zeros(batch_size, self.nlev_rad, 4+self.nx_rad,device=inputs_main.device)

        # inputs_rad[:,10:,0:self.nh_mem] = torch.flip(rnn2out, [1])
        # inputs_rad[:,:,self.nh_mem:] = inputs_main_rad

        inputs_rad[:,10:,0:4] = torch.flip(out[:,:,0:4], [1])
        inputs_rad[:,:,4:] = inputs_main_rad
        # inputs_rad = torch.flip(inputs_rad, [1])

        inputs_sfc_rad = inputs_aux[:,6:11]
        hx = self.mlp_surface_rad(inputs_sfc_rad)
        hidden = (torch.unsqueeze(hx,0))
        rnn_out, states = self.rnn1_rad(inputs_rad, hidden)
        rnn_out = torch.flip(rnn_out, [1])

        inputs_toa = torch.cat((inputs_aux[:,1:2], inputs_aux[:,6:7]),dim=1) 
        hx2 = self.mlp_toa_rad(inputs_toa)

        rnn_out, last_h = self.rnn2_rad(rnn_out, hidden)
        out_rad = self.mlp_output_rad(rnn_out)

        out_sfc_rad = self.mlp_surface_output_rad(last_h)
        #1D (scalar) Output variables: ['cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 
        #'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']
        # print("shape 1", out_sfc_rad.shape, "2", out_sfc.shape)
        out_sfc_rad = torch.squeeze(out_sfc_rad)

        # dT_tot = dT_crm + dT_rad
        out_new[:,:,0:1] = out_new[:,:,0:1] + out_rad
        # rad predicts everything except PRECSC, PRECC
        out_sfc =  torch.cat((out_sfc_rad[:,0:2], out_sfc, out_sfc_rad[:,2:]),dim=1)

        return out_new, out_sfc

    def postprocessing(self, out, out_sfc):
        out             = out / self.yscale_lev
        out_sfc         = out_sfc / self.yscale_sca
        return out, out_sfc
        
    def pp_mp(self, out, out_sfc, x_denorm):

        out_denorm      = out / self.yscale_lev
        out_sfc_denorm  = out_sfc / self.yscale_sca

        T_before        = x_denorm[:,:,0:1]
        qliq_before     = x_denorm[:,:,2:3]
        qice_before     = x_denorm[:,:,3:4]   
        qn_before       = qliq_before + qice_before 

        T_new           = T_before  + out_denorm[:,:,0:1]*1200

        liq_frac_constrained    = self.temperature_scaling(T_new)

        #                            dqn
        qn_new      = qn_before + out_denorm[:,:,2:3]*1200  
        qliq_new    = liq_frac_constrained*qn_new
        qice_new    = (1-liq_frac_constrained)*qn_new
        dqliq       = (qliq_new - qliq_before) * 0.0008333333333333334 #/1200  
        dqice       = (qice_new - qice_before) * 0.0008333333333333334 #/1200   
        out_denorm  = torch.cat((out_denorm[:,:,0:2], dqliq, dqice, out_denorm[:,:,3:]),dim=2)
        
        return out_denorm, out_sfc_denorm
    
    def forward(self, inp_list : List[Tensor]):
        inputs_main   = inp_list[0]
        inputs_aux    = inp_list[1]
        rnn1_mem      = inp_list[2]
        batch_size = inputs_main.shape[0]
        # print("shape inputs main", inputs_main.shape)
        if self.add_pres:
            sp = torch.unsqueeze(inputs_aux[:,0:1],1)
            # undo scaling
            sp = sp*35451.17 + 98623.664
            pres  = self.preslay(sp)
            inputs_main = torch.cat((inputs_main,pres),dim=2)

        if self.separate_radiation:
            # Do not use inputs -2,-3,-4 (O3, CH4, N2O) or first 10 levels
            inputs_main_crm = torch.cat((inputs_main[:,10:,0:-4], inputs_main[:,10:,-1:]),dim=2)
        else:
            inputs_main_crm = inputs_main
            
        if self.use_initial_mlp:
            inputs_main_crm = self.mlp_initial(inputs_main_crm)
            inputs_main_crm = self.nonlin(inputs_main_crm)  
            
        # if self.use_memory:
        # rnn1_input = torch.cat((rnn1_input,self.rnn1_mem), axis=2)
        inputs_main_crm = torch.cat((inputs_main_crm,rnn1_mem), dim=2)
        
        # TOA is first in memory, so to start at the surface we need to go backwards
        rnn1_input = torch.flip(inputs_main_crm, [1])
    
        # The input (a vertical sequence) is concatenated with the
        # output of the RNN from the previous time step 

        if self.separate_radiation:
            inputs_sfc = torch.cat((inputs_aux[:,0:6],inputs_aux[:,11:]),dim=1)
        else:
            inputs_sfc = inputs_aux
        hx = self.mlp_surface1(inputs_sfc)
        hx = self.nonlin(hx)
        cx = self.mlp_surface2(inputs_sfc)
        # cx = self.nonlin(cx)
        hidden = (torch.unsqueeze(hx,0), torch.unsqueeze(cx,0))

        rnn1out, states = self.rnn1(rnn1_input, hidden)
        rnn1out = torch.flip(rnn1out, [1])

        if self.separate_radiation:
            hx2 = torch.randn((batch_size, self.nh_rnn2),device=inputs_main.device)  # (batch, hidden_size)
            cx2 = torch.randn((batch_size, self.nh_rnn2),device=inputs_main.device)
        else: 
            inputs_toa = torch.cat((inputs_aux[:,1:2], inputs_aux[:,6:7]),dim=1) 
            hx2 = self.mlp_toa1(inputs_toa)
            if self.randomly_initialize_cellstate: 
                cx2 = torch.randn_like(hx2)
            else:
                cx2 = self.mlp_toa2(inputs_toa)

        hidden2 = (torch.unsqueeze(hx2,0), torch.unsqueeze(cx2,0))
        
        input_rnn2 = rnn1out
        h_final, states = self.rnn2(input_rnn2, hidden2)

        (last_h, last_c) = states
        h_sfc = last_h.squeeze() 
        
        # Final steps: 
        # if self.return_det:
        #     rnn1_mem        = self.mlp_latent(h_final)
        #     out_det         = self.mlp_output(rnn1_mem)
        #     if self.output_prune:
        #         out_det[:,0:12,1:] = out_det[:,0:12,1:].clone().zero_()
        # out_sfc         = self.mlp_surface_output(h_sfc)
        # out_sfc         = self.relu(out_sfc)
        
        
        #  --------- STOCHASTIC RNN FOR PERTURBATION ----------
        if not self.deterministic_mode:
            if self.return_det:
                rnn1_mem        = self.mlp_latent(h_final)
                out_det         = self.mlp_output(rnn1_mem)
                if self.output_prune and (not self.separate_radiation):
                    out_det[:,0:12,1:] = out_det[:,0:12,1:].clone().zero_()

            input_rnn3 = torch.transpose(h_final,0,1)
            # input_rnn3 = torch.flip(input_rnn3, [0])

            hx = torch.randn((batch_size, self.nh_rnn2),device=inputs_main.device) 
            if self.use_stochastic_lstm:
              cx = torch.randn((batch_size, self.nh_rnn2),device=inputs_main.device) 
              hx = (hx, cx)
              srnn_out, last_state = self.rnn3(input_rnn3, hx)
            else:
              srnn_out = self.rnn3(input_rnn3, hx)
            h_sfc_perturb = srnn_out[-1,:,:]

            # srnn_out, state = self.rnn3(input_rnn3, (hx,cx))
            # h_sfc_perturb, dummy = state
            
            h_final_perturb = torch.transpose(srnn_out,0,1)
            
            # h_final = h_final + 0.01*h_final_perturb
            # h_sfc   = h_sfc + 0.01*h_sfc_perturb
            # h_final_perturb = self.hardtanh(h_final_perturb)
            # h_sfc_perturb = self.hardtanh(h_sfc_perturb)
            
            h_final = h_final*h_final_perturb
            # h_sfc   = h_sfc*h_sfc_perturb
            h_sfc   = h_sfc_perturb

          #  --------- STOCHASTIC RNN FOR PERTURBATION ----------

        # Final steps: 
        rnn1_mem        = self.mlp_latent(h_final)
        out             = self.mlp_output(rnn1_mem)
        if self.output_prune and (not self.separate_radiation):
            # Only temperature tendency is computed for the top 10 levels
            # if self.separate_radiation:
            #     out[:,0:12,:] = out[:,0:12,:].clone().zero_()
            # else:
            out[:,0:12,1:] = out[:,0:12,1:].clone().zero_()
        out_sfc         = self.mlp_surface_output(h_sfc)

        if self.separate_radiation:
            # # out_crm = out.clone()
            # out_new = torch.zeros(batch_size, self.nlev_rad, self.ny, device=inputs_main.device)
            # out_new[:,10:,:] = out
            # # Start at surface again
            # # Do not use inputs 4,5 (winds)
            # inputs_main_rad =  torch.cat((inputs_main[:,:,0:4], inputs_main[:,:,6:]),dim=2)
            # # # add dT from crm 
            # # T_old =   inputs_main * (self.xcoeff_lev[2,:,0:1] - self.xcoeff_lev[1,:,0:1]) + self.xcoeff_lev[0,:,0:1] 
            # # T_new = T_old + dT
            # # inputs_rad =  torch.zeros(batch_size, self.nlev_rad, self.nh_mem+self.nx_rad,device=inputs_main.device)
            # inputs_rad =  torch.zeros(batch_size, self.nlev_rad, 4+self.nx_rad,device=inputs_main.device)

            # # inputs_rad[:,10:,0:self.nh_mem] = torch.flip(rnn2out, [1])
            # # inputs_rad[:,:,self.nh_mem:] = inputs_main_rad

            # inputs_rad[:,10:,0:4] = torch.flip(out[:,:,0:4], [1])
            # inputs_rad[:,:,4:] = inputs_main_rad
            # # inputs_rad = torch.flip(inputs_rad, [1])

            # inputs_sfc_rad = inputs_aux[:,6:11]
            # hx = self.mlp_surface_rad(inputs_sfc_rad)
            # hidden = (torch.unsqueeze(hx,0))
            # rnn_out, states = self.rnn1_rad(inputs_rad, hidden)
            # rnn_out = torch.flip(rnn_out, [1])

            # inputs_toa = torch.cat((inputs_aux[:,1:2], inputs_aux[:,6:7]),dim=1) 
            # hx2 = self.mlp_toa_rad(inputs_toa)

            # rnn_out, last_h = self.rnn2_rad(rnn_out, hidden)
            # out_rad = self.mlp_output_rad(rnn_out)

            # out_sfc_rad = self.mlp_surface_output_rad(last_h)
            # #1D (scalar) Output variables: ['cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 
            # #'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']
            # # print("shape 1", out_sfc_rad.shape, "2", out_sfc.shape)
            # out_sfc_rad = torch.squeeze(out_sfc_rad)

            # # dT_tot = dT_crm + dT_rad
            # out_new[:,:,0:1] = out_new[:,:,0:1] + out_rad
            # out = out_new
            # # rad predicts everything except PRECSC, PRECC
            # out_sfc =  torch.cat((out_sfc_rad[:,0:2], out_sfc, out_sfc_rad[:,2:]),dim=1)
            out, out_sfc = self.forward_rad(inputs_main, inputs_aux, inputs_toa, out)

        out_sfc         = self.relu(out_sfc)
        if self.return_det:
            return out, out_sfc, rnn1_mem, out_det
        else:
            return out, out_sfc, rnn1_mem
        

class stochastic_RNN_autoreg_torchscript(nn.Module):
    use_initial_mlp: Final[bool]
    use_intermediate_mlp: Final[bool]
    add_pres: Final[bool]
    output_prune: Final[bool]
    use_memory: Final[bool]
    use_lstm: Final[bool]
    use_ar_noise: Final[bool]
    two_eps_variables: Final[bool]
    use_surface_memory: Final[bool]
    def __init__(self, hyam, hybm,  hyai, hybi,
                out_scale, out_sfc_scale, 
                xmean_lev, xmean_sca, xdiv_lev, xdiv_sca,
                device,
                nlev=60, nx = 4, nx_sfc=3, ny = 4, ny_sfc=3, nneur=(64,64), 
                use_initial_mlp=False, 
                use_intermediate_mlp=True,
                add_pres=False,
                output_prune=False,
                use_memory=False,
                use_lstm=True,
                nh_mem=64,
                ar_noise_mode=0,
                ar_tau = 0.85,
                use_surface_memory=False):
        super(stochastic_RNN_autoreg_torchscript, self).__init__()
        self.ny = ny 
        self.nlev = nlev 
        self.nlev_mem = nlev
        self.nx_sfc = nx_sfc 
        self.ny_sfc = ny_sfc
        self.nneur = nneur 
        self.use_initial_mlp=use_initial_mlp
        self.output_prune = output_prune
        self.add_pres = add_pres
        if self.add_pres:
            self.preslay = LayerPressure(hyam,hybm)
            nx = nx +1
        self.nx = nx
        self.nh_rnn1 = self.nneur[0]
        self.nx_rnn2 = self.nneur[0]
        self.nh_rnn2 = self.nneur[1]

        self.use_memory= use_memory
        if self.use_initial_mlp:
            self.nx_rnn1 = self.nneur[0]
        else:
            self.nx_rnn1 = nx
            
        self.nonlin = nn.Tanh()
        self.relu = nn.ReLU()
        self.use_lstm=use_lstm
        self.ar_noise_mode = ar_noise_mode 
        # 0 : Fully uncorrelated noise. The sampled noise (eps) used in the stochastic RNN has no temporal correlation, is
        # redrawn at every vertical level and for each RNN. Therefore no need to keep track of it outside the model
        # 1: Eps has temporal correlation (correlation time scale set by tau_t), 
        # but no vertical correlation (eps has a vertical dimension) and is not shared by the two RNNs 
        # 2: Eps has temporal correlation and no vertical correlation, but is shared between the two RNNs
        # 3: Fully correlated noise: temporal correlation, and eps is shared between the two RNN models and at all vertical levels
        if self.ar_noise_mode>0:
            self.use_ar_noise = True
        else:
            self.use_ar_noise = False
        if self.ar_noise_mode==1:
            self.two_eps_variables=True 
        else:
            self.two_eps_variables=False
        if self.use_ar_noise:
            print("Using autoregressive (AR) noise")
            tau_t = torch.tensor(ar_tau) #torch.tensor(0.85)
            tau_e = torch.sqrt(1 - tau_t**2)
            self.register_buffer('tau_t', tau_t)
            self.register_buffer('tau_e', tau_e)
        yscale_lev = torch.from_numpy(out_scale).to(device)
        yscale_sca = torch.from_numpy(out_sfc_scale).to(device)
        xmean_lev  = torch.from_numpy(xmean_lev).to(device)
        xmean_sca  = torch.from_numpy(xmean_sca).to(device)
        xdiv_lev   = torch.from_numpy(xdiv_lev).to(device)
        xdiv_sca   = torch.from_numpy(xdiv_sca).to(device)
        
        self.register_buffer('yscale_lev', yscale_lev)
        self.register_buffer('yscale_sca', yscale_sca)
        self.register_buffer('xmean_lev', xmean_lev)
        self.register_buffer('xmean_sca', xmean_sca)
        self.register_buffer('xdiv_lev', xdiv_lev)
        self.register_buffer('xdiv_sca', xdiv_sca)
        self.use_intermediate_mlp=use_intermediate_mlp

        if self.use_memory:
            if self.use_intermediate_mlp:
                self.nh_mem = nh_mem
            else:
                self.nh_mem = self.nneur[1]
            print("Building RNN that feeds its hidden memory at t0,z0 to its inputs at t1,z0")
        else: 
            self.nh_mem = 0
            print("Building RNN without convective memory")
        self.use_surface_memory = use_surface_memory
        if self.use_surface_memory:
            self.nh_mem = self.nh_mem + 2 

        print("Initial mlp: {}, intermediate mlp: {}".format(self.use_initial_mlp, self.use_intermediate_mlp))
 
        self.nx_rnn1 = self.nx_rnn1 + self.nh_mem
        self.nh_rnn1 = self.nneur[0]
        
        print("nx rnn1", self.nx_rnn1, "nh rnn1", self.nh_rnn1)
        print("nx rnn2", self.nx_rnn2, "nh rnn2", self.nh_rnn2)  
        print("nx sfc", self.nx_sfc)

        if self.use_initial_mlp:
            print("use initial mpl on, nx {} nh {}".format(nx,self.nneur[0]))
            self.mlp_initial = nn.Linear(nx, self.nneur[0])

        self.mlp_surface    = nn.Linear(self.nx_sfc, self.nh_rnn1)
        self.mlp_toa        = nn.Linear(2, self.nh_rnn2)
        
        use_bias=False
        if self.use_lstm:
            if self.use_ar_noise:
                # rnn_layer = MyStochasticLSTMLayer3_ar
                rnn_layer = MyStochasticLstmLayer4_ar
            else:
                # rnn_layer = MyStochasticLSTMLayer3
                rnn_layer = MyStochasticLSTMLayer4
            self.rnn1      = rnn_layer(self.nx_rnn1, self.nh_rnn1, use_bias=use_bias) 
            self.rnn2      = rnn_layer(self.nx_rnn2, self.nh_rnn2, use_bias=use_bias)
            self.mlp_surface2    = nn.Linear(self.nx_sfc, self.nh_rnn1)
            self.mlp_toa2        = nn.Linear(2, self.nh_rnn2)
        else:
            # srnn_layer = MyStochasticGRULayer
            srnn_layer = MyStochasticGRULayer5

            self.rnn1      = srnn_layer(self.nx_rnn1, self.nh_rnn1, use_bias=use_bias) 
            self.rnn2      = srnn_layer(self.nx_rnn2, self.nh_rnn2, use_bias=use_bias)        
            # if self.use_intermediate_mlp:
            #     self.rnn2      = MyStochasticGRULayer5_MLP_fused(self.nx_rnn2, self.nh_rnn2, self.nh_mem, use_bias=use_bias)        
            # else: 
            #     self.rnn2      = srnn_layer(self.nx_rnn2, self.nh_rnn2, use_bias=use_bias)        

        nh_rnn = self.nh_rnn2

        if self.use_intermediate_mlp: 
            self.mlp_latent = nn.Linear(nh_rnn, self.nh_mem)
            self.mlp_output = nn.Linear(self.nh_mem, self.ny)
        else:
            self.mlp_output = nn.Linear(nh_rnn, self.ny)
            
        self.ny_sfc_prec = 2
        self.ny_sfc_rad = self.ny_sfc - self.ny_sfc_prec
        self.mlp_surface_output = nn.Linear(nneur[-1], self.ny_sfc_rad)
        self.mlp_surface_output_mu = nn.Linear(nneur[-1], self.ny_sfc_prec)
        self.mlp_surface_output_sigma = nn.Linear(nneur[-1], self.ny_sfc_prec)

        
    def temperature_scaling(self, T_raw):
        liquid_ratio = (T_raw - 253.16) * 0.05 
        liquid_ratio = F.hardtanh(liquid_ratio, 0.0, 1.0)
        return liquid_ratio
    
    def postprocessing(self, out, out_sfc):
        out             = out / self.yscale_lev
        out_sfc         = out_sfc / self.yscale_sca
        return out, out_sfc
        
    def pp_mp(self, out, out_sfc, x_denorm):
        out_denorm      = out / self.yscale_lev
        out_sfc_denorm  = out_sfc / self.yscale_sca

        T_before        = x_denorm[:,:,0:1]
        qliq_before     = x_denorm[:,:,2:3]
        qice_before     = x_denorm[:,:,3:4]   
        qn_before       = qliq_before + qice_before 

        # print("shape x denorm", x_denorm.shape, "T", T_before.shape)
        T_new           = T_before  + out_denorm[:,:,0:1]*1200

        # T_new           = T_before  + out_denorm[:,:,0:1]*1200
        liq_frac_constrained    = self.temperature_scaling(T_new)

        #                            dqn
        qn_new      = qn_before + out_denorm[:,:,2:3]*1200  
        qliq_new    = liq_frac_constrained*qn_new
        qice_new    = (1-liq_frac_constrained)*qn_new
        dqliq       = (qliq_new - qliq_before) * 0.0008333333333333334 #/1200  
        dqice       = (qice_new - qice_before) * 0.0008333333333333334 #/1200   
        out_denorm  = torch.cat((out_denorm[:,:,0:2], dqliq, dqice, out_denorm[:,:,3:]),dim=2)
        
        return out_denorm, out_sfc_denorm
    
    def forward(self, inp_list : List[Tensor]):
        inputs_main   = inp_list[0]
        inputs_aux    = inp_list[1]
        rnn1_mem      = inp_list[2]
        if self.use_ar_noise:
            eps_prev  = inp_list[3]
            if self.two_eps_variables:
                if eps_prev.shape[0]==2:
                    eps_prev2 = eps_prev[1]
                    eps_prev = eps_prev[0]
                else:
                    raise NotImplementedError("two_eps_variables was set to True but only one was provided")

        if self.add_pres:
            sp = torch.unsqueeze(inputs_aux[:,0:1],1)
            # undo scaling
            sp = sp*35451.17 + 98623.664
            pres  = self.preslay(sp)
            inputs_main = torch.cat((inputs_main,pres),dim=2)

            
        if self.use_initial_mlp:
            inputs_main = self.mlp_initial(inputs_main)
            inputs_main = self.nonlin(inputs_main)  
            
        if self.use_memory:
            inputs_main = torch.cat((inputs_main,rnn1_mem), dim=2)
            
        inputs_main = torch.transpose(inputs_main,0,1)
        rnn1_input =  torch.flip(inputs_main, [0])

        inputs_sfc = inputs_aux
        hx = self.mlp_surface(inputs_sfc)
        hx = self.nonlin(hx)
        
        if self.use_lstm:
            cx = self.mlp_surface2(inputs_sfc)
            cx = self.nonlin(cx)
            hidden = (hx, cx)
            if self.use_ar_noise:
                rnn1out, state = self.rnn1(rnn1_input, hidden, eps_prev)
            else:
                rnn1out, state = self.rnn1(rnn1_input, hidden)
        else:
            hidden = hx
            rnn1out = self.rnn1(rnn1_input, hidden)
        del rnn1_input

        rnn1out = torch.flip(rnn1out, [0])
        if self.use_ar_noise and (not self.two_eps_variables):
          if (eps_prev.dim() == 3):
            eps_prev2 = torch.flip(eps_prev, [0])

        inputs_toa = torch.cat((inputs_aux[:,1:2], inputs_aux[:,6:7]),dim=1) 
        hx2 = self.mlp_toa(inputs_toa)
        hx2 = self.nonlin(hx2)
        
        if self.use_lstm:
            cx2 = self.mlp_toa2(inputs_toa)
            cx2 = self.nonlin(cx2)
            hidden = (hx2, cx2)
            if self.use_ar_noise:
                rnn2out, state = self.rnn2(rnn1out, hidden, eps_prev2)
                if not self.two_eps_variables:
                    del eps_prev2
            else:
                rnn2out, state = self.rnn2(rnn1out, hidden)
            last_hidden = rnn2out[-1,:]
        else:
            hidden = hx2
            # if self.use_intermediate_mlp: 
            #     rnn2out,last_hidden = self.rnn2(rnn1out, hidden) 
            # else:
            rnn2out = self.rnn2(rnn1out, hidden)
            last_hidden = rnn2out[-1,:]
        del rnn1out

        rnn2out = torch.transpose(rnn2out,0,1)

        if self.use_intermediate_mlp: #and self.use_lstm: 
            rnn2out = self.mlp_latent(rnn2out)
          
        rnn1_mem = rnn2out
      
        out = self.mlp_output(rnn2out)

        if self.output_prune:
            out[:,0:12,1:] = out[:,0:12,1:].clone().zero_()
        
        out_sfc_rad = self.mlp_surface_output(last_hidden)

        sfc_mean_ = self.mlp_surface_output_mu(last_hidden)
        sfc_logvar_ = self.mlp_surface_output_sigma(last_hidden)

        eps = torch.randn_like(sfc_mean_)
        sigma = torch.exp(0.5*sfc_logvar_)
        out_sfc = sfc_mean_ + eps * sigma
        if self.use_surface_memory:
            # print("rnn1 mem shape 0", rnn1_mem.shape)
            prec = torch.reshape(out_sfc,(-1,1,2))
            prec = torch.repeat_interleave(prec,self.nlev,dim=1)
            rnn1_mem = torch.cat((rnn1_mem[:,:,0:self.nh_mem-2:],prec),dim=2)
            # print("rnn1 mem shape 1", rnn1_mem.shape)

        out_sfc =  torch.cat((out_sfc_rad[:,0:2], out_sfc, out_sfc_rad[:,2:]),dim=1)

        out_sfc = self.relu(out_sfc)
        
        if self.use_ar_noise:
            eps_prev = self.tau_t * eps_prev + self.tau_e * torch.randn_like(eps_prev) #eps
            if self.two_eps_variables:
              eps_prev2 = self.tau_t * eps_prev2 + self.tau_e * torch.randn_like(eps_prev)
              eps_prev = torch.stack((eps_prev,eps_prev2))
            return out, out_sfc, rnn1_mem, eps_prev
        else:
            return out, out_sfc, rnn1_mem
    

class halfstochastic_RNN_autoreg_torchscript(nn.Module):
    use_initial_mlp: Final[bool]
    use_intermediate_mlp: Final[bool]
    add_pres: Final[bool]
    output_prune: Final[bool]
    separate_radiation: Final[bool]
    use_lstm: Final[bool]
    diagnose_precip: Final[bool]
    use_surface_memory: Final[bool]
    use_ar_noise: Final[bool]
    two_eps_variables: Final[bool]
    def __init__(self, hyam, hybm,  hyai, hybi,
                out_scale, out_sfc_scale, 
                xmean_lev, xmean_sca, xdiv_lev, xdiv_sca,
                device,
                nlev=60, nx = 4, nx_sfc=3, ny = 4, ny_sfc=3, nneur=(64,64), 
                use_initial_mlp=False, 
                use_intermediate_mlp=True,
                add_pres=False,
                output_prune=False,
                use_lstm=True,
                use_surface_memory=False,
                ar_noise_mode=0,
                ar_tau = 0.85,
                nh_mem=64):

        super(halfstochastic_RNN_autoreg_torchscript, self).__init__()
        self.ny = ny 
        self.ny0 = ny 
        self.nlev = nlev 
        self.nlev_mem = nlev
        self.nx_sfc = nx_sfc 
        self.ny_sfc = ny_sfc
        self.ny_sfc0 = ny_sfc
        self.nneur = nneur 
        self.use_initial_mlp=use_initial_mlp
        self.output_prune = output_prune
        self.add_pres = add_pres
        if self.add_pres:
            self.preslay = LayerPressure(hyam,hybm)
            nx = nx +1
        self.nx = nx
        self.nh_rnn1 = self.nneur[0]
        self.nx_rnn2 = self.nneur[0]
        self.nh_rnn2 = self.nneur[1]
        if self.use_initial_mlp:
            self.nx_rnn1 = self.nneur[0]
        else:
            self.nx_rnn1 = nx
            
        self.nonlin = nn.Tanh()
        self.relu = nn.ReLU()
        self.use_lstm=use_lstm
        self.ar_noise_mode = ar_noise_mode 
        # 0 : Fully uncorrelated noise. The sampled noise (eps) used in the stochastic RNN has no temporal correlation, is
        # redrawn at every vertical level and for each RNN. Therefore no need to keep track of it outside the model
        # 1: Eps has temporal correlation (correlation time scale set by tau_t), 
        # but no vertical correlation (eps has a vertical dimension) and is not shared by the two RNNs 
        # 2: Eps has temporal correlation and no vertical correlation, but is shared between the two RNNs
        # 3: Fully correlated noise: temporal correlation, and eps is shared between the two RNN models and at all vertical levels
        if self.ar_noise_mode>0:
            self.use_ar_noise = True
        else:
            self.use_ar_noise = False
        if self.ar_noise_mode==1:
            # self.two_eps_variables=True 
            raise NotImplementedError("two eps variables for halfstochastic model makes no sense") # need mlp here too for custom nh_mem
        else:
            self.two_eps_variables=False
        if self.use_ar_noise:
            print("Using autoregressive (AR) noise")
            tau_t = torch.tensor(ar_tau) #torch.tensor(0.85)
            tau_e = torch.sqrt(1 - tau_t**2)
            self.register_buffer('tau_t', tau_t)
            self.register_buffer('tau_e', tau_e)
        yscale_lev = torch.from_numpy(out_scale).to(device)
        yscale_sca = torch.from_numpy(out_sfc_scale).to(device)
        xmean_lev  = torch.from_numpy(xmean_lev).to(device)
        xmean_sca  = torch.from_numpy(xmean_sca).to(device)
        xdiv_lev   = torch.from_numpy(xdiv_lev).to(device)
        xdiv_sca   = torch.from_numpy(xdiv_sca).to(device)
        
        self.register_buffer('yscale_lev', yscale_lev)
        self.register_buffer('yscale_sca', yscale_sca)
        self.register_buffer('xmean_lev', xmean_lev)
        self.register_buffer('xmean_sca', xmean_sca)
        self.register_buffer('xdiv_lev', xdiv_lev)
        self.register_buffer('xdiv_sca', xdiv_sca)
        self.register_buffer('hyai', hyai)
        self.register_buffer('hybi', hybi)
        self.use_intermediate_mlp=use_intermediate_mlp
            
        if self.use_intermediate_mlp:
            self.nh_mem = nh_mem
        else:
            self.nh_mem = self.nneur[1]
        self.use_surface_memory = use_surface_memory
        if self.use_surface_memory:
            self.nh_mem = self.nh_mem + 2 
            print("Using surface memory, updated nh_mem", self.nh_mem)
        print("Building det + stochastic RNN that feeds its hidden memory at t0,z0 to its inputs at t1,z0")
 
        print("Initial mlp: {}, intermediate mlp: {}".format(self.use_initial_mlp, self.use_intermediate_mlp))
 
        self.nx_rnn1 = self.nx_rnn1 + self.nh_mem
        self.nh_rnn1 = self.nneur[0]
        
        print("nx rnn1", self.nx_rnn1, "nh rnn1", self.nh_rnn1)
        print("nx rnn2", self.nx_rnn2, "nh rnn2", self.nh_rnn2)  
        print("nx sfc", self.nx_sfc)
        print("nh mem", self.nh_mem)

        if self.use_initial_mlp:
            print("use initial mpl on, nx {} nh {}".format(nx,self.nneur[0]))
            self.mlp_initial = nn.Linear(nx, self.nneur[0])

        self.mlp_surface    = nn.Linear(self.nx_sfc, self.nh_rnn1)
        self.mlp_toa        = nn.Linear(2, self.nh_rnn2)
        
        use_bias=False
        self.rnn1      = nn.LSTM(self.nx_rnn1, self.nh_rnn1,  batch_first=False) 
        if self.use_lstm:
            if self.use_ar_noise:
                rnn_layer = MyStochasticLSTMLayer3_ar
            else:
                # rnn_layer = MyStochasticLSTMLayer3
                rnn_layer = MyStochasticLSTMLayer4
            self.rnn2      = rnn_layer(self.nx_rnn2, self.nh_rnn2, use_bias=use_bias)
            self.mlp_surface2    = nn.Linear(self.nx_sfc, self.nh_rnn1)
            self.mlp_toa2        = nn.Linear(2, self.nh_rnn2)
        else:
            raise NotImplementedError() # need mlp here too for custom nh_mem
            # srnn_layer = MyStochasticGRULayer
            srnn_layer = MyStochasticGRULayer5
            self.rnn2      = srnn_layer(self.nx_rnn2, self.nh_rnn2, use_bias=use_bias)        

        nh_rnn = self.nh_rnn2

        if self.use_intermediate_mlp: 
            self.mlp_latent = nn.Linear(nh_rnn, self.nh_mem)
            self.mlp_output = nn.Linear(self.nh_mem, self.ny0)
        else:
            self.mlp_output = nn.Linear(nh_rnn, self.ny0)
            
        # # self.mlp_surface_output = nn.Linear(nneur[-1], self.ny_sfc)
        # self.mlp_surface_output_mu = nn.Linear(nneur[-1], self.ny_sfc)
        # self.mlp_surface_output_sigma = nn.Linear(nneur[-1], self.ny_sfc)

        self.ny_sfc_prec = 2
        self.ny_sfc_rad = self.ny_sfc - self.ny_sfc_prec
        self.mlp_surface_output = nn.Linear(nneur[-1], self.ny_sfc_rad)
        self.mlp_surface_output_mu = nn.Linear(nneur[-1], self.ny_sfc_prec)
        self.mlp_surface_output_sigma = nn.Linear(nneur[-1], self.ny_sfc_prec)

    def temperature_scaling(self, T_raw):
        liquid_ratio = (T_raw - 253.16) * 0.05 
        liquid_ratio = F.hardtanh(liquid_ratio, 0.0, 1.0)
        return liquid_ratio

    def temperature_scaling_precip(self, temp_surface):
        snow_frac = (283.3 - temp_surface) /  14.6
        snow_frac = F.hardtanh(snow_frac, 0.0, 1.0)
        return snow_frac 
    
    def postprocessing(self, out, out_sfc):
        out             = out / self.yscale_lev
        out_sfc         = out_sfc / self.yscale_sca
        return out, out_sfc
        
    def pp_mp(self, out, out_sfc, x_denorm):
        out_denorm      = out / self.yscale_lev
        out_sfc_denorm  = out_sfc / self.yscale_sca

        T_before        = x_denorm[:,:,0:1]
        qliq_before     = x_denorm[:,:,2:3]
        qice_before     = x_denorm[:,:,3:4]   
        qn_before       = qliq_before + qice_before 

        # print("shape x denorm", x_denorm.shape, "T", T_before.shape)
        T_new           = T_before  + out_denorm[:,:,0:1]*1200

        # T_new           = T_before  + out_denorm[:,:,0:1]*1200
        liq_frac_constrained    = self.temperature_scaling(T_new)

        #                            dqn
        qn_new      = qn_before + out_denorm[:,:,2:3]*1200  
        qliq_new    = liq_frac_constrained*qn_new
        qice_new    = (1-liq_frac_constrained)*qn_new
        dqliq       = (qliq_new - qliq_before) * 0.0008333333333333334 #/1200  
        dqice       = (qice_new - qice_before) * 0.0008333333333333334 #/1200   
        out_denorm  = torch.cat((out_denorm[:,:,0:2], dqliq, dqice, out_denorm[:,:,3:]),dim=2)
        
        return out_denorm, out_sfc_denorm
    
    def forward(self, inp_list : List[Tensor]):
        inputs_main   = inp_list[0]
        inputs_aux    = inp_list[1]
        rnn1_mem      = inp_list[2]
        if self.use_ar_noise:
            eps_prev  = inp_list[3]
            if self.two_eps_variables:
                if eps_prev.shape[0]==2:
                    eps_prev2 = eps_prev[1]
                    eps_prev = eps_prev[0]
                else:
                    raise NotImplementedError("two_eps_variables was set to True but only one was provided")
                    
        if self.add_pres:
            sp = torch.unsqueeze(inputs_aux[:,0:1],1)
            # undo scaling
            sp = sp*35451.17 + 98623.664
            pres  = self.preslay(sp)
            inputs_main = torch.cat((inputs_main,pres),dim=2)

            
        if self.use_initial_mlp:
            inputs_main = self.mlp_initial(inputs_main)
            inputs_main = self.nonlin(inputs_main)  
            
        inputs_main = torch.cat((inputs_main,rnn1_mem), dim=2)
            
        inputs_main = torch.transpose(inputs_main,0,1)
        rnn1_input =  torch.flip(inputs_main, [0])

        inputs_sfc = inputs_aux
        hx = self.mlp_surface(inputs_sfc)
        hx = self.nonlin(hx)
        cx = self.mlp_surface2(inputs_sfc)
        # cx = self.nonlin(cx)
        # cx = torch.randn_like()

        hidden = (torch.unsqueeze(hx,0), torch.unsqueeze(cx,0))

        rnn1out, states = self.rnn1(rnn1_input, hidden)
        
        rnn1out = torch.flip(rnn1out, [0])

        inputs_toa = torch.cat((inputs_aux[:,1:2], inputs_aux[:,6:7]),dim=1) 
        hx2 = self.mlp_toa(inputs_toa)
        hx2 = self.nonlin(hx2)
        if self.use_lstm:
            cx2 = self.mlp_toa2(inputs_toa)
            cx2 = self.nonlin(cx2)
            hidden = (hx2, cx2)
            # out, states = self.rnn2(rnn1out, hidden)
            if self.use_ar_noise:
                out, states = self.rnn2(rnn1out, hidden, eps_prev)
            else:
                out, state = self.rnn2(rnn1out, hidden)
        else:
            # out = self.rnn2(rnn1out, hx2)
            if self.use_ar_noise:
                out = self.rnn2(rnn1out, hx2, eps_prev)
            else:
                out = self.rnn2(rnn1out, hx2)   
        del rnn1out 
        
        last_hidden = out[-1,:]

        out = torch.transpose(out,0,1)
        
        if self.use_intermediate_mlp: 
            out = self.mlp_latent(out)
          
        rnn1_mem = out
      
        out = self.mlp_output(out)

        if self.output_prune:
            out[:,0:12,1:] = out[:,0:12,1:].clone().zero_()
        
        out_sfc_rad = self.mlp_surface_output(last_hidden)

        sfc_mean_ = self.mlp_surface_output_mu(last_hidden)
        sfc_logvar_ = self.mlp_surface_output_sigma(last_hidden)

        eps = torch.randn_like(sfc_mean_)
        sigma = torch.exp(0.5*sfc_logvar_)
        out_sfc = sfc_mean_ + eps * sigma
        if self.use_surface_memory:
            # print("rnn1 mem shape 0", rnn1_mem.shape)
            prec = torch.reshape(out_sfc,(-1,1,2))
            prec = torch.repeat_interleave(prec,self.nlev,dim=1)
            rnn1_mem = torch.cat((rnn1_mem[:,:,0:self.nh_mem-2:],prec),dim=2)
            # print("rnn1 mem shape 1", rnn1_mem.shape)

        out_sfc =  torch.cat((out_sfc_rad[:,0:2], out_sfc, out_sfc_rad[:,2:]),dim=1)

        out_sfc = self.relu(out_sfc)

        if self.use_ar_noise:
            eps_prev = self.tau_t * eps_prev + self.tau_e * torch.randn_like(eps_prev) #eps
            if self.two_eps_variables:
              eps_prev2 = self.tau_t * eps_prev2 + self.tau_e * torch.randn_like(eps_prev)
              eps_prev = torch.stack((eps_prev,eps_prev2))
            return out, out_sfc, rnn1_mem, eps_prev
        else:
            return out, out_sfc, rnn1_mem
        
class detLSTM_stochastic_RNN_autoreg_torchscript(nn.Module):
    use_initial_mlp: Final[bool]
    use_intermediate_mlp: Final[bool]
    add_pres: Final[bool]
    output_prune: Final[bool]
    use_memory: Final[bool]
    separate_radiation: Final[bool]
    use_lstm: Final[bool]
    def __init__(self, hyam, hybm,  hyai, hybi,
                out_scale, out_sfc_scale, 
                xmean_lev, xmean_sca, xdiv_lev, xdiv_sca,
                device,
                nlev=60, nx = 4, nx_sfc=3, ny = 4, ny_sfc=3, nneur=(64,64), 
                use_initial_mlp=False, 
                use_intermediate_mlp=True,
                add_pres=False,
                output_prune=False,
                use_memory=False,
                use_lstm=True,
                nh_mem=64):
        super(detLSTM_stochastic_RNN_autoreg_torchscript, self).__init__()
        self.ny = ny 
        self.nlev = nlev 
        self.nx_sfc = nx_sfc 
        self.ny_sfc = ny_sfc
        self.nneur = nneur 
        self.use_initial_mlp=use_initial_mlp
        self.output_prune = output_prune
        self.add_pres = add_pres
        if self.add_pres:
            self.preslay = LayerPressure(hyam,hybm)
            nx = nx +1
        self.nx = nx
        self.nh_rnn1 = self.nneur[0]
        self.nx_rnn2 = self.nneur[0]
        self.nh_rnn2 = self.nneur[1]

        self.use_memory= use_memory
        if self.use_initial_mlp:
            self.nx_rnn1 = self.nneur[0]
        else:
            self.nx_rnn1 = nx
            
        self.nonlin = nn.Tanh()
        self.relu = nn.ReLU()
        self.use_lstm=use_lstm

        yscale_lev = torch.from_numpy(out_scale).to(device)
        yscale_sca = torch.from_numpy(out_sfc_scale).to(device)
        xmean_lev  = torch.from_numpy(xmean_lev).to(device)
        xmean_sca  = torch.from_numpy(xmean_sca).to(device)
        xdiv_lev   = torch.from_numpy(xdiv_lev).to(device)
        xdiv_sca   = torch.from_numpy(xdiv_sca).to(device)
        
        self.register_buffer('yscale_lev', yscale_lev)
        self.register_buffer('yscale_sca', yscale_sca)
        self.register_buffer('xmean_lev', xmean_lev)
        self.register_buffer('xmean_sca', xmean_sca)
        self.register_buffer('xdiv_lev', xdiv_lev)
        self.register_buffer('xdiv_sca', xdiv_sca)
        self.use_intermediate_mlp=use_intermediate_mlp
            
        if self.use_memory:
            if self.use_intermediate_mlp:
                self.nh_mem = nh_mem
            else:
                self.nh_mem = self.nneur[1]
            print("Building RNN that feeds its hidden memory at t0,z0 to its inputs at t1,z0")
        else: 
            self.nh_mem = 0
            print("Building RNN without convective memory")

        print("Initial mlp: {}, intermediate mlp: {}".format(self.use_initial_mlp, self.use_intermediate_mlp))
 
        self.nx_rnn1 = self.nx_rnn1 + self.nh_mem
        self.nh_rnn1 = self.nneur[0]
        
        print("nx rnn1", self.nx_rnn1, "nh rnn1", self.nh_rnn1)
        print("nx rnn2", self.nx_rnn2, "nh rnn2", self.nh_rnn2)  
        print("nx sfc", self.nx_sfc)

        if self.use_initial_mlp:
            print("use initial mpl on, nx {} nh {}".format(nx,self.nneur[0]))
            self.mlp_initial = nn.Linear(nx, self.nneur[0])

        self.mlp_surface    = nn.Linear(self.nx_sfc, self.nh_rnn1)
        self.mlp_toa        = nn.Linear(2, self.nh_rnn2)
        self.rnn1   = nn.LSTM(self.nx_rnn1, self.nh_rnn1,  batch_first=True)
        self.rnn2   = nn.LSTM(self.nx_rnn2, self.nh_rnn2,  batch_first=True)  # (input_size, hidden_size)
        use_bias=True
        if self.use_lstm:
            self.rnn3      = MyStochasticLSTMLayer(self.nh_rnn2+self.nx_rnn1, self.nh_rnn1) 
            self.rnn4      = MyStochasticLSTMLayer(self.nx_rnn2, self.nh_rnn2)
            self.mlp_surface2    = nn.Linear(self.nx_sfc, self.nh_rnn1)
            self.mlp_toa2        = nn.Linear(2, self.nh_rnn2)
        else:
            self.rnn3      = MyStochasticGRULayer(self.nh_rnn2+self.nx_rnn1, self.nh_rnn1, use_bias=use_bias) 
            self.rnn4      = MyStochasticGRULayer(self.nx_rnn2, self.nh_rnn2, use_bias=use_bias)

             
        

        nh_rnn = self.nh_rnn2

        if self.use_intermediate_mlp: 
            self.mlp_latent = nn.Linear(nh_rnn, self.nh_mem)
            self.mlp_output = nn.Linear(self.nh_mem, self.ny)
        else:
            self.mlp_output = nn.Linear(nh_rnn, self.ny)
            
        self.mlp_surface_output = nn.Linear(nneur[-1], self.ny_sfc)
      
        
    def temperature_scaling(self, T_raw):
        liquid_ratio = (T_raw - 253.16) * 0.05 
        liquid_ratio = F.hardtanh(liquid_ratio, 0.0, 1.0)
        return liquid_ratio
    
    def postprocessing(self, out, out_sfc):
        out             = out / self.yscale_lev
        out_sfc         = out_sfc / self.yscale_sca
        return out, out_sfc
        
    def pp_mp(self, out, out_sfc, x_denorm):
        out_denorm      = out / self.yscale_lev
        out_sfc_denorm  = out_sfc / self.yscale_sca

        T_before        = x_denorm[:,:,0:1]
        qliq_before     = x_denorm[:,:,2:3]
        qice_before     = x_denorm[:,:,3:4]   
        qn_before       = qliq_before + qice_before 

        # print("shape x denorm", x_denorm.shape, "T", T_before.shape)
        T_new           = T_before  + out_denorm[:,:,0:1]*1200

        # T_new           = T_before  + out_denorm[:,:,0:1]*1200
        liq_frac_constrained    = self.temperature_scaling(T_new)

        #                            dqn
        qn_new      = qn_before + out_denorm[:,:,2:3]*1200  
        qliq_new    = liq_frac_constrained*qn_new
        qice_new    = (1-liq_frac_constrained)*qn_new
        dqliq       = (qliq_new - qliq_before) * 0.0008333333333333334 #/1200  
        dqice       = (qice_new - qice_before) * 0.0008333333333333334 #/1200   
        out_denorm  = torch.cat((out_denorm[:,:,0:2], dqliq, dqice, out_denorm[:,:,3:]),dim=2)
        
        return out_denorm, out_sfc_denorm
    
    def forward(self, inputs_main, inputs_aux, rnn1_mem):

        if self.add_pres:
            sp = torch.unsqueeze(inputs_aux[:,0:1],1)
            # undo scaling
            sp = sp*35451.17 + 98623.664
            pres  = self.preslay(sp)
            inputs_main = torch.cat((inputs_main,pres),dim=2)

            
        if self.use_initial_mlp:
            inputs_main = self.mlp_initial(inputs_main)
            inputs_main = self.nonlin(inputs_main)  
            
        if self.use_memory:
            # rnn1_input = torch.cat((rnn1_input,self.rnn1_mem), axis=2)
            inputs_main = torch.cat((inputs_main,rnn1_mem), dim=2)
            
            
        inputs_main = torch.transpose(inputs_main,0,1)
        rnn1_input =  torch.flip(inputs_main, [0])

        inputs_sfc = inputs_aux
        hx = self.mlp_surface(inputs_sfc)
        hx = self.nonlin(hx)
        
        # print("shape inp", rnn1_input.shape, "hx", hx.shape)
        if self.use_lstm:
            cx = self.mlp_surface2(inputs_sfc)
            cx = self.nonlin(cx)
            hidden = (hx, cx)
            rnn1out, states = self.rnn1(rnn1_input, hidden)
        else:
            rnn1out = self.rnn1(rnn1_input, hx)
        
        rnn1out = torch.flip(rnn1out, [0])

        inputs_toa = torch.cat((inputs_aux[:,1:2], inputs_aux[:,6:7]),dim=1) 
        hx2 = self.mlp_toa(inputs_toa)
        hx2 = self.nonlin(hx2)
        if self.use_lstm:
            cx2 = self.mlp_toa2(inputs_toa)
            cx2 = self.nonlin(cx2)
            hidden = (hx2, cx2)
            rnn2out, states = self.rnn2(rnn1out, hidden)
        else:
            rnn2out = self.rnn2(rnn1out, hx2)
        
        last_hidden = rnn2out[-1,:]

        rnn2out = torch.transpose(rnn2out,0,1)

        
        if self.use_intermediate_mlp: 
            rnn2out = self.mlp_latent(rnn2out)
          
        if self.use_memory:
            rnn1_mem = torch.flip(rnn2out, [1])
            
      
        out = self.mlp_output(rnn2out)

        if self.output_prune:
            # Only temperature tendency is computed for the top 10 levels
            # if self.separate_radiation:
            #     out[:,0:12,:] = out[:,0:12,:].clone().zero_()
            # else:
            out[:,0:12,1:] = out[:,0:12,1:].clone().zero_()
        
        out_sfc = self.mlp_surface_output(last_hidden)


        out_sfc = self.relu(out_sfc)

        return out, out_sfc, rnn1_mem

class LSTM_torchscript(nn.Module):
    use_initial_mlp: Final[bool]
    use_intermediate_mlp: Final[bool]
    add_pres: Final[bool]
    add_thetae: Final[bool]
    add_stochastic_layer: Final[bool]
    output_prune: Final[bool]
    concat: Final[bool]
    def __init__(self, hyam, hybm,  hyai, hybi,
                out_scale, out_sfc_scale, 
                xmean_lev, xmean_sca, xdiv_lev, xdiv_sca,
                device,
                nlev=60, nx = 4, nx_sfc=3, ny = 4, ny_sfc=3, nneur=(64,64), 
                use_initial_mlp=False, 
                use_intermediate_mlp=True,
                add_pres=False,
                add_thetae=False,
                concat=False,
                add_stochastic_layer=False,
                output_prune=False,
                nh_mem=16):
        super(LSTM_torchscript, self).__init__()
        self.ny = ny 
        self.nlev = nlev 
        self.nx_sfc = nx_sfc 
        self.ny_sfc = ny_sfc
        self.nneur = nneur 
        self.use_initial_mlp=use_initial_mlp
        self.output_prune = output_prune
        self.add_pres = add_pres
        if self.add_pres:
            self.preslay = LayerPressure(hyam,hybm)
            nx = nx +1
        self.add_thetae = add_thetae
        if self.add_thetae:
            self.preslay_nonorm = LayerPressure(hyam,hybm,norm=False)
            nx = nx +1
        self.nx = nx
        self.nh_rnn1 = self.nneur[0]
        self.nx_rnn2 = self.nneur[0]
        self.nh_rnn2 = self.nneur[1]
        self.concat=concat
        self.add_stochastic_layer = add_stochastic_layer
        self.nonlin = nn.Tanh()
        self.relu = nn.ReLU()
        
        yscale_lev = torch.from_numpy(out_scale).to(device)
        yscale_sca = torch.from_numpy(out_sfc_scale).to(device)
        xmean_lev  = torch.from_numpy(xmean_lev).to(device)
        xmean_sca  = torch.from_numpy(xmean_sca).to(device)
        xdiv_lev   = torch.from_numpy(xdiv_lev).to(device)
        xdiv_sca   = torch.from_numpy(xdiv_sca).to(device)
        
        self.register_buffer('yscale_lev', yscale_lev)
        self.register_buffer('yscale_sca', yscale_sca)
        self.register_buffer('xmean_lev', xmean_lev)
        self.register_buffer('xmean_sca', xmean_sca)
        self.register_buffer('xdiv_lev', xdiv_lev)
        self.register_buffer('xdiv_sca', xdiv_sca)
        self.register_buffer('hyai', hyai)
        self.register_buffer('hybi', hybi)
            
        # in ClimSim config of E3SM, the CRM physics first computes 
        # moist physics on 50 levels, and then computes radiation on 60 levels!
        self.use_intermediate_mlp=use_intermediate_mlp
        
        if self.use_initial_mlp:
            self.nx_rnn1 = self.nneur[0]
        else:
            self.nx_rnn1 = nx
            
        print("Building RNN without convective memory")

        print("Initial mlp: {}, intermediate mlp: {}".format(self.use_initial_mlp, self.use_intermediate_mlp))
 
        self.nh_rnn1 = self.nneur[0]
        
        print("nx rnn1", self.nx_rnn1, "nh rnn1", self.nh_rnn1)
        print("nx rnn2", self.nx_rnn2, "nh rnn2", self.nh_rnn2)  
        print("nx sfc", self.nx_sfc)

        if self.use_initial_mlp:
            self.mlp_initial = nn.Linear(nx, self.nneur[0])

        self.mlp_surface1  = nn.Linear(nx_sfc, self.nh_rnn1)
        self.mlp_surface2  = nn.Linear(nx_sfc, self.nh_rnn1)

        # self.rnn1      = nn.LSTMCell(self.nx_rnn1, self.nh_rnn1)  # (input_size, hidden_size)
        # self.rnn2      = nn.LSTMCell(self.nx_rnn2, self.nh_rnn2)
        self.rnn1      = nn.LSTM(self.nx_rnn1, self.nh_rnn1,  batch_first=True)  # (input_size, hidden_size)
        self.rnn2      = nn.LSTM(self.nx_rnn2, self.nh_rnn2,  batch_first=True)

        if self.add_stochastic_layer:
            from models_torch_kernels import MyStochasticGRU, MyStochasticGRULayer
            nx_srnn = self.nh_rnn2
            # self.rnn_stochastic = StochasticGRUCell(nx_srnn, self.nh_rnn2)  # (input_size, hidden_size)
            self.rnn_stochastic = MyStochasticGRU(nx_srnn, self.nh_rnn2)  # (input_size, hidden_size)

        if self.concat:
            nh_rnn = self.nh_rnn1 + self.nh_rnn2
        else:
            nh_rnn = self.nh_rnn2

        if self.use_intermediate_mlp: 
            self.mlp_latent = nn.Linear(nh_rnn, self.nh_rnn2)
            self.mlp_output = nn.Linear(self.nh_rnn2, self.ny)
        else:
            self.mlp_output = nn.Linear(nh_rnn, self.ny)
            
        self.mlp_surface_output = nn.Linear(nneur[-1], self.ny_sfc)
            

    def temperature_scaling(self, T_raw):
        # T_denorm = T = T*(self.xmax_lev[:,0] - self.xmin_lev[:,0]) + self.xmean_lev[:,0]
        # T_denorm = T*(self.xcoeff_lev[2,:,0] - self.xcoeff_lev[1,:,0]) + self.xcoeff_lev[0,:,0]
        # liquid_ratio = (T_raw - 253.16) / 20.0 
        liquid_ratio = (T_raw - 253.16) * 0.05 
        liquid_ratio = F.hardtanh(liquid_ratio, 0.0, 1.0)
        return liquid_ratio
        
    def pp_mp(self, out, out_sfc, x_denorm):

        out_denorm      = out / self.yscale_lev.to(device=out.device)
        
        T_before        = x_denorm[:,:,0:1]
        qliq_before     = x_denorm[:,:,2:3]
        qice_before     = x_denorm[:,:,3:4]   
        qn_before       = qliq_before + qice_before 

        # print("shape x denorm", x_denorm.shape, "T", T_before.shape)
        T_new           = T_before  + out_denorm[:,:,0:1]*1200

        # T_new           = T_before  + out_denorm[:,:,0:1]*1200
        liq_frac_constrained    = self.temperature_scaling(T_new)

        #                            dqn
        qn_new      = qn_before + out_denorm[:,:,2:3]*1200  
        qliq_new    = liq_frac_constrained*qn_new
        qice_new    = (1-liq_frac_constrained)*qn_new
        dqliq       = (qliq_new - qliq_before) * 0.0008333333333333334 #/1200  
        dqice       = (qice_new - qice_before) * 0.0008333333333333334 #/1200   
        out_denorm  = torch.cat((out_denorm[:,:,0:2], dqliq, dqice, out_denorm[:,:,3:]),dim=2)
        
        out_sfc_denorm  = out_sfc / self.yscale_sca.to(device=out.device)
        
        return out_denorm, out_sfc_denorm
    
    def forward(self, inputs_main, inputs_aux):

        batch_size = inputs_main.shape[0]
        # print("shape inputs main", inputs_main.shape)
        if self.add_pres:
            sp = torch.unsqueeze(inputs_aux[:,0:1],1)
            # undo scaling
            sp = sp*35451.17 + 98623.664
            pres  = self.preslay(sp)
            inputs_main = torch.cat((inputs_main,pres),dim=2)
            
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.add_thetae:
            tk =  inputs_main[:,:,0:1]*(self.xdiv_lev[:,0:1]) + self.xmean_lev[:,0:1]
            rh = inputs_main[:,:,1:2]
            pres_nonorm  = self.preslay_nonorm(sp)
            thetae = thermo_rh(tk,pres_nonorm,rh) / 3000.0
            thetae = torch.where(torch.isnan(thetae), torch.tensor(0.0, device=thetae.device), thetae)
            thetae  = torch.where(torch.isinf(thetae),  torch.tensor(0.0, device=thetae.device),  thetae)
            # print("T {:.2f}   rh {:.2f}  thetae {:.2f}  thetae0 {:.2f} ".format( 
            #                                                 # x_denorm[200,35,4].item(),
            #                                                 tk[200,55,0].item(), 
            #                                                 rh[200,55,0].item(), 
            #                                                 thetae[200,55,0].item(),
            #                                                 thetae[200,15,0].item()))
            
            
            inputs_main = torch.cat((inputs_main,thetae),dim=2)

        hx = self.mlp_surface1(inputs_aux)
        # hx = self.nonlin(hx)

        # TOA is first in memory, so to start at the surface we need to go backwards
        inputs_main = torch.flip(inputs_main, [1])
        
        # The input (a vertical sequence) is concatenated with the
        # output of the RNN from the previous time step 
        cx = self.mlp_surface2(inputs_aux)
        cx = self.nonlin(cx)
        hidden = (torch.unsqueeze(hx,0), torch.unsqueeze(cx,0))

        if self.use_initial_mlp:
            rnn1_input = self.mlp_initial(inputs_main)
            rnn1_input = self.nonlin(rnn1_input)
    
        else:
            rnn1_input = inputs_main

        rnn1out, states = self.rnn1(rnn1_input, hidden)

        # hx2 = torch.randn((batch_size, self.nh_rnn2),dtype=self.dtype,device=device)  # (batch, hidden_size)
        # cx2 = torch.randn((batch_size, self.nh_rnn2),dtype=self.dtype,device=device)
        hx2 = torch.randn((batch_size, self.nh_rnn2),device=inputs_main.device)  # (batch, hidden_size)
        cx2 = torch.randn((batch_size, self.nh_rnn2),device=inputs_main.device)
        
        hidden2 = (torch.unsqueeze(hx2,0), torch.unsqueeze(cx2,0))

        input_rnn2 = rnn1out
            
        rnn2out, states = self.rnn2(input_rnn2, hidden2)

        rnn1out = torch.flip(rnn1out, [1])

        input_rnn2 = rnn1out
    
        rnn2out, states = self.rnn2(input_rnn2, hidden2)

        # rnn1_input = torch.flip(rnn1_input, [1])
        # rnn2out, states = self.rnn2(rnn1_input, hidden2)

        (last_h, last_c) = states
        (last_h, last_c) = states
        
        if self.use_intermediate_mlp: 
            rnn2out = self.mlp_latent(rnn2out)
          
        # Add a stochastic perturbation
        # Convective memory is still based on the deterministic model,
        # and does not include the stochastic perturbation
        # concat and use_intermediate_mlp should be set to false
        if self.add_stochastic_layer:
            srnn_input = torch.transpose(rnn2out,0,1)

            # transpose is needed because this layer assumes seq. dim first
            z = self.rnn_stochastic(srnn_input)
            z = torch.flip(z, [0])

            z = torch.transpose(z,0,1)
            # z = torch.flip(z, [1])
            # rnn2out = z
            # z is a perburbation added to the hidden state
            rnn2out = rnn2out + 0.01*z 

        if self.concat:
            rnn2out = torch.cat((rnn1out, rnn2out), dim=2)
        
        out = self.mlp_output(rnn2out)

        if self.output_prune:
            out[:,0:12,1:] = out[:,0:12,1:].clone().zero_()
        
        out_sfc = self.mlp_surface_output(last_h.squeeze())
        out_sfc = self.relu(out_sfc)

        return out, out_sfc
