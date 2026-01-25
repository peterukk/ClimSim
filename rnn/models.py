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
    use_lstm: Final[bool]
    separate_radiation: Final[bool]
    do_fluxrad: Final[bool]
    physical_precip: Final[bool]
    predict_liq_frac: Final[bool]
    randomly_initialize_cellstate: Final[bool]
    output_sqrt_norm: Final[bool]
    concat: Final[bool]
    store_precip: Final[bool]
    do_heat_advection: Final[bool]
    include_sedimentation_term: Final[bool]
    include_diffusivity: Final[bool]
    pour_excess: Final[bool]
    pred_total_water: Final[bool]
    conserve_water: Final[bool]
    constrain_precip: Final[bool]
    return_neg_precip: Final[bool]
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
                use_lstm=True,
                output_prune=False,
                repeat_mu=False, # repeat solar zenith angle to each vertical level?
                separate_radiation=False,
                mp_mode=0, # see train_rnn_rollout_torchscript_hydra
                physical_precip=False,
                predict_liq_frac=False,
                randomly_initialize_cellstate=False, # introduce a smalld degree of randomness into deterministic LSTM
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
        self.mp_mode=mp_mode
        if self.mp_mode==-2:
          self.pred_total_water=True 
        else:
          self.pred_total_water=False
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

        self.predict_liq_frac = predict_liq_frac
        self.add_pres = add_pres
        if self.add_pres:
            self.preslay = LayerPressure(hyam,hybm)
            self.preslay_nonorm = LayerPressure(hyam,hybm, norm=False)
            nx = nx +1
            self.preslev_nonorm = LevelPressure(hyai, hybi)
        self.use_lstm = use_lstm
        self.randomly_initialize_cellstate = randomly_initialize_cellstate
        self.add_stochastic_layer = add_stochastic_layer
        self.nx = nx
        self.nh_rnn1 = self.nneur[0]
        self.nx_rnn2 = self.nneur[0]
        self.nh_rnn2 = self.nneur[1]
        if len(nneur) != 2 and not self.add_stochastic_layer:
            raise NotImplementedError()
        if add_stochastic_layer:
            if len(nneur) == 3:        
                self.nx_rnn3 = self.nneur[1]
                self.nh_rnn3 = self.nneur[2]   
            elif len(nneur) == 2:
                self.nx_rnn3 = self.nneur[0]
                self.nh_rnn3 = self.nneur[1]     
            else:
                raise NotImplementedError()
    
        self.output_sqrt_norm=output_sqrt_norm
        if self.output_sqrt_norm:
            print("Warning: output_sqrt_norm ON")
        self.repeat_mu = repeat_mu
        if self.repeat_mu:
            nx = nx + 1
        self.use_intermediate_mlp=use_intermediate_mlp  
        if self.use_intermediate_mlp:
            self.nh_mem = nh_mem
        else:
            self.nh_mem = self.nneur[1]

        # lots of options for physical_precip (ignored if physical_precip=false!)
        self.conserve_water = True
        self.store_precip = True
        self.constrain_precip = False
        self.include_sedimentation_term = True
        self.pour_excess = True
        self.do_heat_advection = True 
        self.include_diffusivity= True
        self.return_neg_precip = True
        if self.pred_total_water:
          self.influde_evap = False 
        if self.constrain_precip:
          self.pour_excess = False
        if not self.physical_precip:
            self.store_precip=False
        if self.store_precip: 
            self.nh_mem0 = self.nh_mem - 1 
        else:
            self.nh_mem0 = self.nh_mem
        
        if self.physical_precip:
            print("conserve_water:",self.conserve_water, "storeprec:", self.store_precip, "constrainprec:", self.constrain_precip, 
                "pour:", self.pour_excess, "do heat:", self.do_heat_advection, "diffusivity:", self.include_diffusivity)

            self.mp_ncol = 16
            if self.pred_total_water: 
              self.mlp_qtot_crm      = nn.Linear(self.nh_rnn2, self.mp_ncol)
              self.mlp_flux_qtot_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
              self.mlp_mp_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
            else:
              self.mlp_qv_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
              self.mlp_flux_qv_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
              self.mlp_flux_qn_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
              self.mlp_evap_cond_vapor_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
            self.mlp_evap_prec_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
            self.mlp_qn_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
            self.mlp_mp_aa_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
            self.mlp_mp_aa2_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)

            if self.do_heat_advection:
              self.mlp_t_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
              self.mlp_flux_crm_t = nn.Linear(self.nh_rnn2, self.mp_ncol)

            self.softmax = nn.Softmax(dim=2)
            self.softmax_dim1 = nn.Softmax(dim=1)
        else: 
            self.return_neg_precip = False 
            self.include_diffusivity = False 
            
        # in ClimSim config of E3SM, the CRM physics first computes 
        # moist physics on 50 levels, and then computes radiation on 60 levels!
        self.do_fluxrad=False  
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
            if not self.do_fluxrad: 
              self.ny_rad = 1
            self.ny_sfc_rad = self.ny_sfc0 - 2
            self.ny_sfc0 = 2
        
        if self.use_initial_mlp:
            self.nx_rnn1 = self.nneur[0]
            # should we include large-scale tendencies from the lowest level here?
        else:
            self.nx_rnn1 = nx
                
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

        # self.rnn_mem = None 
        print("Building RNN that feeds its hidden memory at t0,z0 to its inputs at t1,z0")
        print("Initial mlp: {}, intermediate mlp: {}".format(self.use_initial_mlp, self.use_intermediate_mlp))
 
        self.nx_rnn1 = self.nx_rnn1 + self.nh_mem0
        self.nh_rnn1 = self.nneur[0]
        
        print("nx rnn1", self.nx_rnn1, "nh rnn1", self.nh_rnn1)
        print("nx rnn2", self.nx_rnn2, "nh rnn2", self.nh_rnn2)  
        if self.add_stochastic_layer: print("nx rnn3", self.nx_rnn3, "nh rnn3", self.nh_rnn3) 
        print("nx sfc", self.nx_sfc)
        print("ny", self.ny, "ny0", self.ny0)

        if self.use_initial_mlp:
            self.mlp_initial = nn.Linear(nx, self.nneur[0])

        self.mlp_surface1  = nn.Linear(self.nx_sfc, self.nh_rnn1)
        if not self.randomly_initialize_cellstate and self.use_lstm:
            self.mlp_surface2  = nn.Linear(self.nx_sfc, self.nh_rnn1)

        if self.use_lstm:
            rnn_layer = nn.LSTM 
        else:
            rnn_layer = nn.GRU

        if self.add_stochastic_layer:
            self.rnn0   = rnn_layer(self.nx_rnn1, self.nh_rnn1,  batch_first=True)
            self.rnn1   = rnn_layer(self.nx_rnn2, self.nh_rnn2,  batch_first=True)  # (input_size, hidden_size)
            use_bias=False
            if self.use_lstm:
                self.rnn2 = MyStochasticLSTMLayer4(self.nx_rnn3, self.nh_rnn3, use_bias=use_bias)  
            else: 
                self.rnn2 = MyStochasticGRULayer5(self.nx_rnn3, self.nh_rnn3, use_bias=use_bias)   
        else:
            self.rnn1   = rnn_layer(self.nx_rnn1, self.nh_rnn1,  batch_first=True)  # (input_size, hidden_size)
            self.rnn2   = rnn_layer(self.nx_rnn2, self.nh_rnn2,  batch_first=True)
                
        self.sigmoid = nn.Sigmoid()

        # if self.concat: 
        #     nh_rnn = self.nh_rnn1 + self.nh_rnn2
        # else:
        nh_rnn = self.nh_rnn2

        if self.use_intermediate_mlp: 
          self.mlp_latent = nn.Linear(nh_rnn, self.nh_mem0)
          self.mlp_output = nn.Linear(self.nh_mem0, self.ny0)
        else:
          self.mlp_output = nn.Linear(nh_rnn, self.ny0)
            
        if self.store_precip:
          self.mlp_precip_release = nn.Linear(nh_rnn, 1)

        if self.predict_liq_frac:
          self.mlp_predfrac = nn.Linear(nh_mem, 1)

        if self.physical_precip and self.include_diffusivity:
          self.conv_diff = nn.Conv1d(self.nx_rnn1, 1, 3, stride=1, padding="same")

        self.mlp_surface_output = nn.Linear(nneur[-1], self.ny_sfc0)
            
        if self.separate_radiation:
          if self.do_fluxrad: 
            self.ng_sw = 48
            self.ng_lw = 48
            self.mlp_rad_reftrans = nn.Linear(self.nx_rad_tot, self.ng_sw)
            self.mlp_rad_surface_sw = nn.Linear(9, self.ng_sw)
            self.rnn1_rad_sw = nn.GRU(self.ng_sw, self.ng_sw,    batch_first=True)  
            self.rnn2_rad_sw = nn.GRU(self.ng_sw, 2*self.ng_sw,  batch_first=True)  
            self.mlp_rad_sw =   nn.Linear(2*self.ng_sw, 2)
            self.mlp_rad_reftrans = nn.Linear(self.nx_rad_tot, self.ng_lw)
            self.mlp_rad_surface_lw = nn.Linear(10, self.ng_lw)
            # self.mlp_rad_lw_flux_scale = nn.Linear(self.nx_rad_tot, 1)
            self.mlp_rad_lw_flux_scale = nn.Linear(self.ng_lw, 1)
            self.rnn1_rad_lw = nn.GRU(self.ng_lw, self.ng_lw,    batch_first=True)  
            self.rnn2_rad_lw = nn.GRU(self.ng_lw, 2*self.ng_lw,  batch_first=True)  
          else:
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
        # liquid_frac = (T_raw - 253.16) / 20.0 
        liquid_frac = (T_raw - 253.16) * 0.05 
        liquid_frac = F.hardtanh(liquid_frac, 0.0, 1.0)
        return liquid_frac
    
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
        return out, out_sfc
        
    def forward(self, inp_list : List[Tensor]):
        inputs_main   = inp_list[0]
        inputs_aux    = inp_list[1]
        rnn_mem      = inp_list[2]
        if self.physical_precip:
          inputs_denorm = inp_list[3]

        batch_size, seq_size, feature_size = inputs_main.shape

        ilev_crm = 0

        if self.add_pres:
            sp = torch.unsqueeze(inputs_aux[:,0:1],1)
            # surface pressure, undo scaling
            sp = sp*self.xdiv_sca[0] + self.xmean_sca[0]
            pres  = self.preslay(sp)
            preslev_nonorm  = torch.squeeze(self.preslev_nonorm(sp))
            preslay_nonorm  = torch.squeeze(self.preslay_nonorm(sp))
            inputs_main = torch.cat((inputs_main,pres),dim=2)
            del pres

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
            inputs_main_crm = self.nonlin(self.mlp_initial(inputs_main_crm))
            
        # if self.use_memory:
        if self.physical_precip and self.store_precip: 
          P_old = rnn_mem[:,-1,-1] 
        
        inputs_main_crm = torch.cat((inputs_main_crm,rnn_mem[:,:,0:self.nh_mem0]), dim=2)
        del rnn_mem
            
        if self.add_stochastic_layer: 
            # LSTM downwards --> LSTM upwards --> stochastic RNN downwards
            hx0 = torch.randn((batch_size, self.nh_rnn1),device=inputs_main.device)  # (batch, hidden_size)
            if self.use_lstm: 
                cx0 = torch.randn((batch_size, self.nh_rnn1),device=inputs_main.device)
                hidden0 = (torch.unsqueeze(hx0,0), torch.unsqueeze(cx0,0))  
            else:
                hidden0 = torch.unsqueeze(hx0,0)
            rnn0out, states = self.rnn0(inputs_main_crm, hidden0)
            
            rnn1_input =  torch.flip(rnn0out, [1])
        else:
            # LSTM upwards --> LSTM downwards
            # TOA is first in memory, so to start at the surface we need to go backwards
            rnn1_input = torch.flip(inputs_main_crm, [1])

        # del inputs_main_crm

        # The input (a vertical sequence) is concatenated with the
        # output of the RNN from the previous time step 
        if self.separate_radiation:
            inputs_sfc = torch.cat((inputs_aux[:,0:6],inputs_aux[:,12:]),dim=1)
        else:
            inputs_sfc = inputs_aux

        hx = self.nonlin(self.mlp_surface1(inputs_sfc))

        if self.use_lstm:
            if self.randomly_initialize_cellstate:
                cx = torch.randn_like(hx)
            else:
                cx = self.mlp_surface2(inputs_sfc)
            hidden = (torch.unsqueeze(hx,0), torch.unsqueeze(cx,0))
        else: 
            hidden = torch.unsqueeze(hx,0)

        rnn1out, states = self.rnn1(rnn1_input, hidden)
        del rnn1_input, states

        rnn1out = torch.flip(rnn1out, [1])

        if self.separate_radiation:
          hx2 = torch.randn((batch_size, self.nh_rnn2),device=inputs_main.device)  # (batch, hidden_size)
          if self.use_lstm: 
              cx2 = torch.randn((batch_size, self.nh_rnn2),device=inputs_main.device)
        else: 
          inputs_toa = torch.cat((inputs_aux[:,1:2], inputs_aux[:,6:7]),dim=1) 
          hx2 = self.mlp_toa1(inputs_toa)
          if self.use_lstm: 
              cx2 = self.mlp_toa2(inputs_toa)
        
        if self.add_stochastic_layer:
            rnn1out = torch.transpose(rnn1out,0,1)
            if self.use_lstm:
                hidden = (hx2, cx2)
            else:
                hidden = hx2
        else:
            if self.use_lstm:
                hidden = (torch.unsqueeze(hx2,0), torch.unsqueeze(cx2,0))
            else:
                hidden = torch.unsqueeze(hx2,0)

        if self.add_stochastic_layer:
            if self.use_lstm:
                z, states = self.rnn2(rnn1out, hidden)
                (last_h, last_c) = states
            else: 
                z = self.rnn2(rnn1out, hidden)
                states =  z[-1,:,:]
            # SPPT
            rnn2outt = z
            #   z = F.hardtanh(z, 0.0, 2.0)
            #   rnn2outt = z*rnn1out
            # rnn2outt = rnn1out + 0.01*z 
            rnn2outt = torch.transpose(rnn2outt,0,1)
        else:
            rnn2outt, states = self.rnn2(rnn1out, hidden)

        del rnn1out, hidden

        if self.use_lstm:
            (last_h, last_c) = states
        else:
            last_h = states

        final_sfc_inp = last_h.squeeze() 
            
        # if self.concat:
        #   rnn2outt = torch.cat((rnn1out, rnn2outt), dim=2)
        
        if self.use_intermediate_mlp: 
            rnn_mem = self.mlp_latent(rnn2outt)
        else:
            rnn_mem = rnn2outt 

        out = self.mlp_output(rnn_mem)

        if self.output_prune and (not self.separate_radiation):
            # Only temperature tendency is computed for the top 10 levels, by the radiation scheme, after CRM runs on lowest 50 levels
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
            inputs_rad[:,10:,3:] = rnn_mem
            inputs_rad = torch.flip(inputs_rad, [1])

            if self.do_fluxrad:
                inputs_sfc_rad_sw =  torch.cat((inputs_aux[:,9:11], inputs_aux[:,12:]),dim=1) 
                inputs_sfc_rad_lw =  torch.cat((inputs_aux[:,7:9], inputs_aux[:,11:]),dim=1) 

                # Separate shortwave (SW) and longwave (LW),  both follow the same recipe:
                # 1) local MLP to compute reflectance-transmittance variables 
                # 2.1) mlp to compute surface albedo variables 
                # 2.2) upward RNN to compute albedo variables, initialized with surface vars
                # 2.3) concatenate surface and layer albedos to get (nlev+1) levels/layers
                # 3.1) downward RNN to compute upward and downward fluxes (nlev+1, nhidden)
                # 4.1) (SW) get specific required surface components from the spectral flux at surface
                # 4.2) sum over hidden dimension to get broadband fluxes (nlev+1)
                # Then using both SW and LW:
                # 5) compute net broadband flux as (flux_sw_dn - flux_sw_up) + (flux_lw_dn - flux_lw_up) 
                # 
                # ---------- SHORTWAVE -----------

                # 1)
                reftrans = self.mlp_rad_reftrans(inputs_rad)
                # 2.1)
                albedos_sfc = self.mlp_rad_surface_sw(inputs_sfc_rad_sw)
                hidden = (torch.unsqueeze(albedos_sfc,0))
                # 2.2)
                albedos, states = self.rnn1_rad_sw(reftrans, hidden)
                # 2.3)
                albedos = torch.cat((torch.reshape(albedos_sfc,(batch_size,1,-1)), albedos),dim=1) 
                # flip back so that TOA is first
                albedos = torch.flip(albedos, [1])
                # 3.1)
                flux_sw, states = self.rnn2_rad_sw(albedos)
                flux_sw_dn,flux_sw_up = flux_sw.chunk(2,2)
                del flux_sw, states
                # flux_sw = self.sigmoid(flux_sw)
                flux_sw_tot = self.sigmoid(self.mlp_rad_sw(flux_sw))
                flux_sw_dn_tot,flux_sw_up_tot = flux_sw_tot.chunk(2,2)
                flux_sw_dn = self.softmax(flux_sw_dn)*flux_sw_dn_tot
                flux_sw_up = self.softmax(flux_sw_up)*flux_sw_up_tot

                incflux = torch.unsqueeze(inputs_aux[:,1:2],1)
                incflux = torch.reshape(incflux*self.xdiv_sca[1] + self.xmean_sca[1], (-1,1,1))

                # flux_sw = flux_sw * incflux
                flux_sw_dn = flux_sw_dn * incflux
                flux_sw_up = flux_sw_up * incflux

                # print("flux sw max", torch.max(flux_sw).item())

                # 4.1)
                # flux_sw_dn,flux_sw_up = flux_sw.chunk(2,2)
                SOLS      = flux_sw_dn[:,-1,0:1]  
                SOLL      = flux_sw_dn[:,-1,1:2,] 
                SOLSD       = flux_sw_dn[:,-1,2:3] 
                SOLLD       = flux_sw_dn[:,-1,3:4]
                # 4.2)
                flux_sw_dn  = torch.sum(flux_sw_dn,dim=2)
                flux_sw_up  = torch.sum(flux_sw_up,dim=2)
                flux_sw_net = flux_sw_dn - flux_sw_up
                flux_sw_net_sfc = flux_sw_net[:,-1].unsqueeze(1)

                # 1)
                reftrans = self.mlp_rad_reftrans(inputs_rad)
                # 2.1)
                albedos_sfc = self.mlp_rad_surface_lw(inputs_sfc_rad_lw)
                hidden = (torch.unsqueeze(albedos_sfc,0))
                # 2.2)
                albedos, states = self.rnn1_rad_lw(reftrans, hidden)
                # 2.3)
                albedos = torch.cat((torch.reshape(albedos_sfc,(batch_size,1,-1)), albedos),dim=1) 
                albedos = torch.flip(albedos, [1])
                # 3.1)
                flux_lw, states = self.rnn2_rad_lw(albedos)
                flux_lw = self.relu(flux_lw)
                del reftrans, albedos, states

                # lw_flux_scale = self.mlp_rad_lw_flux_scale(torch.flip(inputs_rad, [1]))
                lw_flux_scale = self.mlp_rad_lw_flux_scale(albedos)
                lw_flux_scale = self.relu(lw_flux_scale)
                flux_lw      = flux_lw * torch.reshape(lw_flux_scale,(batch_size,-1,1))

                # 4)
                flux_lw_dn,flux_lw_up = flux_lw.chunk(2,2)
                flux_lw_dn      = torch.sum(flux_lw_dn,dim=2)
                flux_lw_up      = torch.sum(flux_lw_up,dim=2)
                flux_lw_dn_sfc  = flux_lw_dn[:,-1].unsqueeze(1)
                flux_lw_net     = flux_lw_dn - flux_lw_up

                # print("inc toa  min max", torch.min(incflux).item(), torch.max(incflux).item())

                # 4)
                flux_net = flux_lw_net + flux_sw_net
                flux_diff = flux_net[:,1:] - flux_net[:,0:-1]

                pres_diff = preslev_nonorm[:,1:] - preslev_nonorm[:,0:-1]
                dT_rad = -(flux_diff / pres_diff) * 0.009767579681 
                dT_rad = self.yscale_lev[:,0]*dT_rad
                out_new[:,:,0:1] = out_new[:,:,0:1] + dT_rad.unsqueeze(2)
                # print("out new shape", out_new.shape)
                out_sfc_rad = torch.cat((flux_sw_net_sfc, flux_lw_dn_sfc, SOLS, SOLL, SOLSD, SOLLD),dim=-1)
                yscale_sca = torch.cat((self.yscale_sca[0:2], self.yscale_sca[4:]))
                out_sfc_rad = yscale_sca*out_sfc_rad
            else:
                hidden = self.mlp_surface_rad(inputs_aux[:,6:12])
                hidden = (torch.unsqueeze(hidden,0))
                rnn_out, states = self.rnn1_rad(inputs_rad, hidden)
                rnn_out = torch.flip(rnn_out, [1])
                del inputs_rad, states

                inputs_toa = torch.cat((inputs_aux[:,1:2], inputs_aux[:,6:7]),dim=1) 
                hidden = self.mlp_toa_rad(inputs_toa)
                hidden = (torch.unsqueeze(hidden,0))

                rnn_out, last_h_rad = self.rnn2_rad(rnn_out, hidden)

                # with torch.autocast(device_type="cuda", enabled=False):
                out_rad = self.mlp_output_rad(rnn_out.float())
                del rnn_out
                
                out_sfc_rad = self.mlp_surface_output_rad(last_h_rad.float())
                # dT_tot = dT_crm + dT_rad
                # out_new[:,:,0:1] = out_new[:,:,0:1] + out_rad
                out_new[:,:,0:1] =  out_rad

                out_sfc_rad = torch.squeeze(out_sfc_rad)
                # rad predicts everything except PRECSC, PRECC
                if not self.physical_precip:
                    out_sfc =  torch.cat((out_sfc_rad[:,0:2], out_sfc, out_sfc_rad[:,2:]),dim=1)
                    
            out = out_new

        # with torch.autocast(device_type="cuda", enabled=False):
        if self.physical_precip:
          rnn2outt = rnn2outt.float()
          last_h = last_h.float()
          # ilev_crm = 12
          if self.separate_radiation:
            ilev_crm = 10
            # rnn2outt = rnn2outt[:,2:]
          else:
            # ilev_crm = 0
            ilev_crm = 12
            rnn2outt = rnn2outt[:,ilev_crm:]
          # ilev_crm = 12
          
          out_neww = torch.zeros(batch_size, self.nlev, self.ny, device=inputs_main.device)
          g = torch.tensor(9.806650000)
          one_over_g = torch.tensor(0.1019716213)
          scaling_factor = -g # tendency equation in pressure coordinates has -g in front
          pres_diff = preslev_nonorm[:,ilev_crm+1:] - preslev_nonorm[:,ilev_crm:-1]
          zeroes = torch.zeros(batch_size, 1, device=inputs_main.device)

          # Pmax = 1e4
          Tsfc = inputs_denorm[:,-1,0]
          #Pmax = 0.8 * 1000 *  self.yscale_sca[3] * 5.58e-18*torch.exp(0.077*Tsfc)
          Pmax = 1000 *  self.yscale_sca[3] * 5.58e-18*torch.exp(0.077*Tsfc)
          # print("min max pmax", Pmax.min().item(), Pmax.max().item())
          if self.pred_total_water:
            #  model outputs ['ptend_t', 'dqtot', 'cld_water_frac', 'liq_frac', 'ptend_u', 'ptend_v']

            subcolumns=True
          
            out_neww[:,:,0] = out[:,:,0]
            out_neww[:,:,2:self.ny] = out[:,:,2:self.ny] 

            qv_old = inputs_denorm[:,ilev_crm-1:,-1]
            qliq_old = inputs_denorm[:,ilev_crm-1:,2]
            qice_old = inputs_denorm[:,ilev_crm-1:,3]
            qtot_old = qliq_old + qice_old + qv_old 
            qtot_old0 = qtot_old
            qtot_old = qtot_old[:,1:]

            flux_mult_coeff = 1.0e3

            # # -----------
            if subcolumns:
              dqtot_mp = self.mlp_mp_crm(rnn2outt)
              flux = self.mlp_flux_qtot_crm(rnn2outt)
              # FORCE FLUX TO SURFACE (sedimentation) TO BE POSITIVE 
              # if it's negative, we have negative precipitation and water is taken from nowhere and added to the atmosphere
              # OR should we not bother to include a sedimentation term, and set boundary fluxes to zero to avoid net transport 
              flux[:,-1] = self.relu(flux[:,-1]) # net downward flux cannot be negative 
              p_crm_qtot_old = self.softmax(self.mlp_qtot_crm(rnn2outt))
              crm_qtot_old = self.mp_ncol*qtot_old.unsqueeze(2)*p_crm_qtot_old 
              qtot_old = crm_qtot_old 
              flux_net_qtot = flux_mult_coeff*flux*crm_qtot_old
              zeroes_crm = torch.zeros(batch_size, 1, self.mp_ncol, device=inputs_main.device)
              flux_net_qtot = torch.cat((zeroes_crm,flux_net_qtot),dim=1)
              sedimentation = torch.mean( flux_net_qtot[:,-1] + flux_net_qtot[:,-1], 1)
              flux_qtot_dp = scaling_factor*( (flux_net_qtot[:,1:] - flux_net_qtot[:,0:-1]) / pres_diff.unsqueeze(2)) 
            else:
              dqtot_mp = out[:,ilev_crm:,1] # Intermediate output, net microphysics tendency
              flux_net_qtot   = flux_mult_coeff*out[:,ilev_crm:,self.ny]  # Total water flux
              # FORCE FLUX TO SURFACE (sedimentation) TO BE POSITIVE 
              # if it's negative, we have negative precipitation and water is taken from nowhere and added to the atmosphere
              # OR should we not bother to include a sedimentation term, and set boundary fluxes to zero to avoid net transport 
              flux_net_qtot[:,-1] = self.relu(flux_net_qtot[:,-1]) # net downward flux is positive by convention
              flux_net_qtot = torch.cat((zeroes,flux_net_qtot),dim=1)
              sedimentation = flux_net_qtot[:,-1] 
              flux_diff_qtot = flux_net_qtot[:,1:] - flux_net_qtot[:,0:-1]
              flux_qtot_dp = scaling_factor*(flux_diff_qtot / pres_diff) 

            # ------------
            if self.include_diffusivity:
              preslay_diff = preslay_nonorm[:,ilev_crm+1:] - preslay_nonorm[:,ilev_crm:-1]
              # D = out[:,ilev_crm:,1]
              if not self.separate_radiation:
                inputs_main_crm = inputs_main_crm[:,ilev_crm:]
              inputs_main_crm = torch.transpose(inputs_main_crm,1,2)
              D = self.conv_diff(inputs_main_crm)
              D = torch.squeeze(torch.transpose(D,1,2))
              
              T_old2 = self.yscale_lev[ilev_crm-1:,0].unsqueeze(0) * inputs_denorm[:,ilev_crm-1:,0]
              diff_t =  D * ( (T_old2[:,1:] - T_old2[:,0:-1]) / pres_diff) 
              diff_t =  (diff_t[:,1:] - diff_t[:,0:-1]) / preslay_diff
              diff_qtot = D *( (qtot_old0[:,1:] - qtot_old0[:,0:-1]) / pres_diff) 
              diff_qtot = (diff_qtot[:,1:] - diff_qtot[:,0:-1]) / preslay_diff

            if self.do_heat_advection: 
              T_old = 5e-2*inputs_denorm[:,ilev_crm:,0]
              flux3 = self.mlp_flux_crm_t(rnn2outt)
              p_crm_t_old = self.softmax(self.mlp_t_crm(rnn2outt))
              crm_t_old = self.mp_ncol*T_old.unsqueeze(2)*p_crm_t_old
              flux_net_t = flux3*crm_t_old
              flux_net_t = torch.cat((zeroes_crm,flux_net_t),dim=1)
              flux_t_dp = scaling_factor*( (flux_net_t[:,1:] - flux_net_t[:,0:-1]) / pres_diff.unsqueeze(2)) 
              del flux3, p_crm_t_old, crm_t_old, flux_net_t 

            if subcolumns:
              alpha = self.relu(self.mlp_mp_aa_crm(rnn2outt))

              qn_old = (inputs_denorm[:,ilev_crm:,2] + inputs_denorm[:,ilev_crm:,3])
              p_crm_qn_old = self.softmax(self.mlp_qn_crm(rnn2outt)) 
              crm_qn_old = self.mp_ncol*qn_old.unsqueeze(2)*p_crm_qn_old 
              
              dqn_aa = alpha*torch.square(crm_qn_old)*torch.reshape(self.yscale_lev[ilev_crm:,1],(1,-1,1)) 
              dqv_evap_prec = self.relu(self.mlp_evap_prec_crm(rnn2outt))
            #   dqv_evap_prec = self.relu(dqv_evap_prec) + 1.0e-6 # force positive
              # Relate evaporated precipitation to stored precipitation here?
              P_old_vertical = out[:,ilev_crm:,2] # self.mlp_prec_vertical(rnn2outt)
              # P_old_vertical = rnn_mem[:,ilev_crm:,0] 
              P_old_vertical = self.softmax_dim1(P_old_vertical) * P_old.unsqueeze(1) # sums to P_old
              dqv_evap_prec = dqv_evap_prec*P_old_vertical.unsqueeze(2) # P_old.unsqueeze(1)
              del rnn2outt
              dqtot_mp = dqn_aa - dqv_evap_prec

            if self.conserve_water:
                # When we predict total water and diagnose precipitation from the vertical integral of the microphysical tendency,
                # we would otherwise always conserve water, BUT if the predicted total tendency would make the amount of water
                # negative at some level, we lose the water conservation because this will be clipped to zero online 

                # qtot_old + dqtot_denorm*1200 > 0 
                # dqtot_denorm*1200 > -qtot_old
                # dqtot_denorm > -qtot_old/1200
                # dqtot/self.yscale_lev[:,1] > -qtot_old/1200 
                # dqtot > -(self.yscale_lev[:,1]*qtot_old/(1200) 
                # flux_qtot_dp - dqtot_mp  > -(self.yscale_lev[:,1]*qtot_old/(1200) 
                #   - dqtot_mp  > -(self.yscale_lev[:,1]*qtot_old/(1200)  - flux_qtot_dp
                # dqtot_mp  < (self.yscale_lev[:,1]*qtot_old/(1200)  + flux_qtot_dp

                # print("min max dqtot_mp0", torch.min(dqtot_mp), torch.max(dqtot_mp))
                # print("min max flux_qtot_dp", torch.min(flux_qtot_dp), torch.max(flux_qtot_dp))
                # print("min max qtot scale", torch.min((self.yscale_lev[:,1]*qtot_old/1200)), torch.max((self.yscale_lev[:,1]*qtot_old/1200)))
                if subcolumns:
                  max_dqtot_mp = (self.yscale_lev[ilev_crm:,1:2]*qtot_old/1200) + flux_qtot_dp
                else:
                  max_dqtot_mp = (self.yscale_lev[ilev_crm:,1]*qtot_old/1200) + flux_qtot_dp
                dqtot_mp = torch.clamp(dqtot_mp, max=max_dqtot_mp)
                

            # if the diagnosed precip is negative this reflects a net SOURCE of column water 
            # if we are storing precipitation, then we can allow this to be negative with a minimum value set by the 
            # amount of stored precipitation: new precipitation P1 = P0 + P_diagnosed will then be zero
            # this reflects evaporation of precipitation that hasn't fallen out yet
            

            if self.constrain_precip:
              # Precipitation constraints:
              #  1) dqtot_mp => -P0 (positivity)
              #  2) dqtot_mp <= Pmax -P0 (stored precip. can't get larger than Pmax)
              predicted_sum = torch.sum(one_over_g*pres_diff*dqtot_mp,1) 
              max_sum = Pmax - P_old 
              min_sum = -P_old
              target_sum = torch.clamp(predicted_sum, min=min_sum, max=max_sum)
              dqtot_mp = dqtot_mp * (target_sum.unsqueeze(1) / predicted_sum.unsqueeze(1))

            if subcolumns:
              out_neww[:,ilev_crm:,1] =  torch.mean(flux_qtot_dp - dqtot_mp,2)
              d_precip_sourcesink    = torch.mean(dqtot_mp,2)    
            else:
              out_neww[:,ilev_crm:,1] = flux_qtot_dp - dqtot_mp   
              d_precip_sourcesink    = dqtot_mp    


            if self.include_diffusivity:
              out_neww[:,ilev_crm:-1,0] = out_neww[:,ilev_crm:-1,0] + diff_t
              out_neww[:,ilev_crm:-1,1] = out_neww[:,ilev_crm:-1,1] + diff_qtot


            if self.do_heat_advection: 
              out_neww[:,ilev_crm:,0] = out_neww[:,ilev_crm:,0] + torch.mean(flux_t_dp, 2)
              
          else:                      # !!! PREDICT WATER VAPOR AND CLOUD WATER TENDENCIES SEPARATELY !!!!
          # if self.mp_mode==-1:
            #  ['ptend_t', 'dqv', 'dqn', 'liq_frac', 'ptend_u', 'ptend_v']
            out_neww[:,:,0] = out[:,:,0]
            out_neww[:,:,3:self.ny] = out[:,:,3:self.ny]

            # dqv_evap_prec = out[:,ilev_crm:,1] # Evaporation of precipitation (precip to water vapor = sink of precip, source of qv)
            dqn_aa        = out[:,ilev_crm:,2] # Accretion/autoconversion (cloud water to precip. = source of precip, sink of qn)
            # dqn_evap_cond_vapor = out[:,ilev_crm:,3] # evaporation - condensation  (cloud water to water vapor is positive) 

            # rnn2outt = rnn2outt[:,ilev_crm:]
            dqv_evap_prec       = self.mlp_evap_prec_crm(rnn2outt)
            dqn_evap_cond_vapor = self.mlp_evap_cond_vapor_crm(rnn2outt)

            # mp_crm_ar = self.mlp_microphysics(rnn2out[:,ilev_crm:])
            # dqv_evap_prec, dqn_aa, dqn_evap_cond_vapor  = mp_crm_ar.chunk(3,2)

            flux_mult_coeff = 3.0e5

            qv_old0 =  inputs_denorm[:,ilev_crm:,-1]
            qn_old0 = (inputs_denorm[:,ilev_crm:,2] + inputs_denorm[:,ilev_crm:,3])

            if self.include_diffusivity:
              # D = out[:,ilev_crm:,1]
              if not self.separate_radiation:
                inputs_main_crm = inputs_main_crm[:,ilev_crm:]
              inputs_main_crm = torch.transpose(inputs_main_crm,1,2)
              D = self.conv_diff(inputs_main_crm)
              D = torch.squeeze(torch.transpose(D,1,2))
              qv_old2 =  self.yscale_lev[ilev_crm-1:,1].unsqueeze(0) * inputs_denorm[:,ilev_crm-1:,-1]
              qn_old2 = self.yscale_lev[ilev_crm-1:,2].unsqueeze(0) * (inputs_denorm[:,ilev_crm-1:,2] + inputs_denorm[:,ilev_crm-1:,3])
              T_old2 = self.yscale_lev[ilev_crm-1:,0].unsqueeze(0) * inputs_denorm[:,ilev_crm-1:,0]
              diff_qv = D *( (qv_old2[:,1:] - qv_old2[:,0:-1]) / pres_diff) 
              diff_qn = D * ( (qn_old2[:,1:] - qn_old2[:,0:-1]) / pres_diff) 
              diff_t =  D * ( (T_old2[:,1:] - T_old2[:,0:-1]) / pres_diff) 
              preslay_diff = preslay_nonorm[:,ilev_crm+1:] - preslay_nonorm[:,ilev_crm:-1]
              diff_qv = (diff_qv[:,1:] - diff_qv[:,0:-1]) / preslay_diff
              diff_qn = (diff_qn[:,1:] - diff_qn[:,0:-1]) / preslay_diff
              diff_t =  (diff_t[:,1:] - diff_t[:,0:-1]) / preslay_diff

            # crm_flux = True 
            # if crm_flux: 
            flux1 = self.mlp_flux_qv_crm(rnn2outt)
            flux2 = self.mlp_flux_qn_crm(rnn2outt)

            if self.include_sedimentation_term:
              # FORCE FLUX TO SURFACE (sedimentation) TO BE POSITIVE 
              # if it's negative, we have negative precipitation and water is taken from nowhere and added to the atmosphere
              # OR should we not bother to include a sedimentation term, and set boundary fluxes to zero to avoid net transport 
              flux1[:,-1] = self.relu(flux1[:,-1]) # net downward flux cannot be negative 
              flux2[:,-1] = self.relu(flux2[:,-1])

            p_crm_qv_old = self.softmax(self.mlp_qv_crm(rnn2outt))
            p_crm_qn_old = self.softmax(self.mlp_qn_crm(rnn2outt)) 

            crm_qn_old = self.mp_ncol*qn_old0.unsqueeze(2)*p_crm_qn_old 
            crm_qv_old = self.mp_ncol*qv_old0.unsqueeze(2)*p_crm_qv_old 
            qn_old = crm_qn_old 
            qv_old = crm_qv_old

            flux_net_qv = flux_mult_coeff*flux1*crm_qv_old#*torch.reshape(self.yscale_lev[ilev_crm:,1],(1,-1,1))
            # flux_net_qv = torch.mean( flux_net_qv , 2)
            
            flux_net_qn = flux_mult_coeff*flux2*crm_qn_old#*torch.reshape(self.yscale_lev[ilev_crm:,1],(1,-1,1))
            # flux_net_qn = torch.mean( flux_net_qn , 2)
            del flux1, flux2, p_crm_qn_old, p_crm_qv_old

            # else:
            #     flux_net_qv   = flux_mult_coeff*out[:,ilev_crm:,self.ny]*qv_old*300#*self.yscale_lev[ilev_crm:,1]
            #     flux_net_qn   = flux_mult_coeff*out[:,ilev_crm:,self.ny+1]*qn_old*300#*self.yscale_lev[ilev_crm:,2]

            # # FORCE FLUX TO SURFACE (sedimentation) TO BE POSITIVE 
            # # if it's negative, we have negative precipitation and water is taken from nowhere and added to the atmosphere
            # # OR should we not bother to include a sedimentation term, and set boundary fluxes to zero to avoid net transport 
            # flux_net_qv[:,-1] = self.relu(flux_net_qv[:,-1]) # net downward flux cannot be negative 
            # flux_net_qn[:,-1] = self.relu(flux_net_qn[:,-1])

            # flux_net_qv = torch.cat((zeroes,flux_net_qv),dim=1)
            # flux_net_qn = torch.cat((zeroes,flux_net_qn),dim=1)
            zeroes_crm = torch.zeros(batch_size, 1, self.mp_ncol, device=inputs_main.device)
            if self.include_sedimentation_term:
              flux_net_qv = torch.cat((zeroes_crm,flux_net_qv[:,0:-1], zeroes_crm),dim=1)
              flux_net_qn = torch.cat((zeroes_crm,flux_net_qn[:,0:-1], zeroes_crm),dim=1)
              sedimentation = 0
            else:
              flux_net_qv = torch.cat((zeroes_crm,flux_net_qv),dim=1)
              flux_net_qn = torch.cat((zeroes_crm,flux_net_qn),dim=1)
              sedimentation = torch.mean( flux_net_qv[:,-1] + flux_net_qn[:,-1], 1)

            # flux_qv_dp = scaling_factor*( (flux_net_qv[:,1:] - flux_net_qv[:,0:-1]) / pres_diff) 
            # flux_qn_dp = scaling_factor*( (flux_net_qn[:,1:] - flux_net_qn[:,0:-1]) / pres_diff) 
            flux_qv_dp = scaling_factor*( (flux_net_qv[:,1:] - flux_net_qv[:,0:-1]) / pres_diff.unsqueeze(2)) 
            flux_qn_dp = scaling_factor*( (flux_net_qn[:,1:] - flux_net_qn[:,0:-1]) / pres_diff.unsqueeze(2)) 
            del flux_net_qv, flux_net_qn

            out_neww[:,ilev_crm:,1] = torch.mean(flux_qv_dp, 2)  
            out_neww[:,ilev_crm:,2] = torch.mean(flux_qn_dp, 2)    

            # dqn_aa        = self.relu(dqn_aa) + 1.0e-6 # force positive
            dqv_evap_prec = self.relu(dqv_evap_prec) + 1.0e-6 # force positive
            # Relate evaporated precipitation to stored precipitation here?
            P_old_vertical = out[:,ilev_crm:,2] # self.mlp_prec_vertical(rnn2outt)
            # P_old_vertical = rnn_mem[:,ilev_crm:,0] 
            P_old_vertical = self.softmax_dim1(P_old_vertical) * P_old.unsqueeze(1) # sums to P_old
            dqv_evap_prec = dqv_evap_prec*P_old_vertical.unsqueeze(2) # P_old.unsqueeze(1)

            if self.do_heat_advection: 
              T_old = inputs_denorm[:,ilev_crm:,0] #5e-2*inputs_denorm[:,ilev_crm:,0]
              flux3 = self.mlp_flux_crm_t(rnn2outt)
              p_crm_t_old = self.softmax(self.mlp_t_crm(rnn2outt))
              crm_t_old = self.mp_ncol*T_old.unsqueeze(2)*p_crm_t_old
              flux_net_t = flux3*crm_t_old
              flux_net_t = torch.cat((zeroes_crm,flux_net_t),dim=1)
              flux_t_dp = scaling_factor*( (flux_net_t[:,1:] - flux_net_t[:,0:-1]) / pres_diff.unsqueeze(2)) 
              del flux3, p_crm_t_old, crm_t_old, flux_net_t 

            if self.conserve_water: 
            #   dqn_aa = out[:,ilev_crm:,1:2]
              alpha = self.relu(self.mlp_mp_aa_crm(rnn2outt)) #+ 1.0e-6 # force positive

              q_old_sum = torch.sum(qn_old0,dim=1)
              qn_old_redist = out[:,ilev_crm:,1]
              qn_old2 = torch.reshape(self.softmax_dim1(qn_old_redist) * q_old_sum.unsqueeze(1), (batch_size, -1, 1))
              crm_qn_old = qn_old2
              dqn_aa = alpha*crm_qn_old*torch.reshape(self.yscale_lev[ilev_crm:,2],(1,-1,1)) 
              # dqn_aa = alpha*torch.square(crm_qn_old)*torch.reshape(self.yscale_lev[ilev_crm:,2],(1,-1,1)) 
              # dqn_aa = alpha*torch.pow(crm_qn_old, 1.5)*torch.reshape(self.yscale_lev[ilev_crm:,2],(1,-1,1)) 

              # thresh = self.sigmoid(self.w1)*0.001 # 1e-4 # 5e-4
            #   thresh = 4.0e-4
            #   qscale = torch.reshape(self.yscale_lev[ilev_crm:,2],(1,-1,1))
            #   dqn_aa = self.relu(alpha*(crm_qn_old-thresh))
            #   # dqn_aa = alpha*crm_qn_old
            #   alpha2 = self.mlp_mp_aa2_crm(rnn2outt)
            #   alpha2 = self.relu(alpha2) #  + 1.0e-6 # force positive
            #   dqn_accretion = alpha2*crm_qn_old*P_old_vertical.unsqueeze(2)
            #   print("min max Pvert", P_old_vertical.min().item(), P_old_vertical.max().item())
            #   dqn_accretion = alpha2*crm_qn_old*torch.pow(P_old_vertical.unsqueeze(2)+1e-4,0.875)
            #   dqn_aa = qscale*dqn_aa + qscale*dqn_accretion
              del rnn2outt

              if True:
                # first ensure positive qn by clamping dqn_evap_cond_vapor
                minval = -(self.yscale_lev[ilev_crm:,2:3]*qn_old/1200) - flux_qn_dp + dqn_aa
                dqn_evap_cond_vapor = torch.clamp(dqn_evap_cond_vapor, min=minval)
                # minval = -(self.yscale_lev[:,2]*qn_old/1200) - flux_qn_dp + dqn_aa - dqn_evap_cond_vapor 
                # dqn_pos_fix = torch.relu(minval)
            
              if True:
                # then ensure positive qv by clamping dqv_evap_prec
                minval = -(self.yscale_lev[ilev_crm:,1:2]*qv_old/1200) - flux_qv_dp + dqn_evap_cond_vapor
                dqv_evap_prec = torch.clamp(dqv_evap_prec, min=minval)
                # minval = -(self.yscale_lev[:,1]*qv_old/1200) - flux_qv_dp + dqn_evap_cond_vapor - dqv_evap_prec
                # dqv_pos_fix = torch.relu(minval)

              if True:
                # maximum cloud water limit, make sure qn_new doesn't exceed qn_max
                # dqn_normed < yscale*(qnmax - qnold)/1200
                # flux_qn_dp + dqn_evap_cond_vapor   - dqn_aa < yscale*(qnmax - qnold)/1200
                # flux_qn_dp + dqn_evap_cond_vapor < yscale*(qnmax - qnold)/1200 + dqn_aa
                #  dqn_aa > flux_qn_dp + dqn_evap_cond_vapor -yscale*(qnmax - qnold)/1200 
                qn_max = 0.0006
                minval = flux_qn_dp + dqn_evap_cond_vapor -(self.yscale_lev[ilev_crm:,2:3]*(qn_max - qn_old)/1200) 
                dqn_aa = torch.clamp(dqn_aa, min=minval)

            # note: can't both conserve water and do the following!
            elif self.constrain_precip and self.store_precip:
              # We have two constraints for (d_precip_sourcesink = dqn_aa - dqv_evap_prec), which is vertically
              # integrated to get precipitation:
              # 1/g*vint(dqn_aa - dqv_evap_prec) => -P0,
              # 1/g*vint(dqn_aa - dqv_evap_prec) <= Pmax; where P0 is the precip. stopage and Pmax is the maximum allowed stored precipitation
              # Lets leave dqv_evap_prec unconstrained and constrain dqn_aa:
              # Pmax + 1/g*vint(dqv_evap_prec) => 1/g*vint(dqn_aa) => -P0 + 1/g*vint(dqv_evap_prec) 
              predicted_sum = torch.sum(one_over_g*pres_diff*dqn_aa,1) 
              y =  torch.sum(one_over_g*pres_diff*dqv_evap_prec,1) 
              # print("y", y[100].item(), y.min().item(), y.max().item(), y.mean().item())
              # print("predicted_sum", predicted_sum[100].item(),predicted_sum.min().item(), predicted_sum.max().item(), predicted_sum.mean().item())
              # print("P_old", P_old[100].item(), P_old.min().item(), P_old.max().item(), P_old.mean().item())
              max_sum = Pmax - P_old + y
              min_sum = -P_old + y
              # print("max_sum", max_sum[100].item(), max_sum.min().item(), max_sum.max().item(), max_sum.mean().item())
              # print("min_sum", min_sum[100].item(), min_sum.min().item(), min_sum.max().item(), min_sum.mean().item())
              target_sum = torch.clamp(predicted_sum, min=min_sum, max=max_sum)
              # print("target_sum", target_sum[100].item(), target_sum.min().item(), target_sum.max().item())
              dqn_aa = dqn_aa * (target_sum.unsqueeze(1) / predicted_sum.unsqueeze(1))
              # new_sum = torch.sum(one_over_g*pres_diff*dqn_aa,1) 
              # print("adjusted",  new_sum[100].item(),new_sum.min().item(), new_sum.max().item(), new_sum.mean().item())
              # fac = target_sum.unsqueeze(1) / predicted_sum.unsqueeze(1)
              # print("scalefac", fac[100].item() ,  "min", torch.min(fac).item(), "max", torch.max(fac).item(), "mean", torch.mean(fac).item())
    
          #   #                                    (cond-evap)<0 from vapor,  evap. from prec. both add water vapor  
          #   out_neww[:,ilev_crm:,1] = flux_qv_dp - dqn_evap_cond_vapor     + dqv_evap_prec 
          #   #                                    (cond-evap)>0 adds,     acc-au removes cldwater  
          #   out_neww[:,ilev_crm:,2] = flux_qn_dp + dqn_evap_cond_vapor     - dqn_aa        

            #                                                (cond-evap)<0 from vapor,  evap. from prec. both add water vapor  
            out_neww[:,ilev_crm:,1] = torch.mean(flux_qv_dp - dqn_evap_cond_vapor     + dqv_evap_prec, 2)  
            #                                                 (cond-evap)>0 adds,     acc-au removes cldwater  
            out_neww[:,ilev_crm:,2] = torch.mean(flux_qn_dp + dqn_evap_cond_vapor     - dqn_aa, 2)        

            if self.include_diffusivity:
              out_neww[:,ilev_crm:-1,0] = out_neww[:,ilev_crm:-1,0] + diff_t
              out_neww[:,ilev_crm:-1,1] = out_neww[:,ilev_crm:-1,1] + diff_qv
              out_neww[:,ilev_crm:-1,2] = out_neww[:,ilev_crm:-1,2] + diff_qn

            if self.do_heat_advection: 
              out_neww[:,ilev_crm:,0] = out_neww[:,ilev_crm:,0] + torch.mean(flux_t_dp, 2)
              

            net_condensation = torch.mean(dqn_evap_cond_vapor - dqv_evap_prec, 2)
            Lv_over_cp =  2490.04 # 2.5e6/1004 = 2.5 × 10⁶ J/kg at 0°C  divided by 1004 J/kg/K
            net_condensation = Lv_over_cp*(net_condensation/self.yscale_lev[ilev_crm:,1]) * self.yscale_lev[ilev_crm:,0]
          #   print("max dt0", torch.max(out_neww[:,ilev_crm:,0]).item(), "cond", net_condensation.max().item())
            out_neww[:,ilev_crm:,0] = out_neww[:,ilev_crm:,0] + net_condensation

            #                      source,       sink   of precipitation  (note: signs already reversed w.r.t. above)
            d_precip_sourcesink    = torch.mean(dqn_aa      - dqv_evap_prec,2)                    # - dqv_pos_fix  - dqn_pos_fix
            # This should be positive to avoid negative precipitation!

        #   print("model mean max min dqn_evap_cond_vapor", torch.mean(dqn_evap_cond_vapor).item(),  torch.max(dqn_evap_cond_vapor).item(), torch.min(dqn_evap_cond_vapor).item())
        #   print("model mean max min dqv_evap_prec", torch.mean(dqv_evap_prec).item(),  torch.max(dqv_evap_prec).item(), torch.min(dqv_evap_prec).item())
        #   print("model mean dqn_aa", torch.mean(dqn_aa).item())
        #   print("model mean max min d_precip_sourcesink", torch.mean(d_precip_sourcesink).item(),  torch.max(d_precip_sourcesink).item(), torch.min(d_precip_sourcesink).item())

          out = out_neww   

        #   dp_water = (one_over_g*pres_diff*d_precip_sourcesink)
          water_new = torch.sum((one_over_g*pres_diff*d_precip_sourcesink),1)  

          if self.store_precip:
            # prec_negative = self.relu(-water_new) # punish model for diagnosing negative precip from column water changes?
            # water_new = self.relu(water_new)
            # print("model min max mean water_new",  torch.min(water_new).item(), torch.max(water_new).item(),torch.mean(water_new).item())
            water_new = P_old + water_new
            # print("model min max mean water_new 2",  torch.min(water_new).item(), torch.max(water_new).item(),torch.mean(water_new).item())
            prec_negative = self.relu(-water_new) # punish model for diagnosing negative precip from column water changes?
            water_new = self.relu(water_new)
            precc_release_fraction = torch.sigmoid(self.mlp_precip_release(last_h)).squeeze()
            # precc_release_fraction = 1.0
            # print("precc_release_fraction min max ", precc_release_fraction.min().item(), precc_release_fraction.max().item())
            water_released = precc_release_fraction*water_new
            water_stored  = water_new*(1-precc_release_fraction)

            # water_stored = self.relu(water_stored)

            # Just clipping the stored water here is incorrect, because we break the conservation! 
            # Instead compute the excess and add it to precipitation?
            if self.pour_excess:
                water_excess = water_stored - Pmax
                water_excess = self.relu(water_excess)
                water_stored = water_stored - water_excess
            else:
                water_excess = 0
            # if torch.any(water_stored>1.1*Pmax):
            #   raise Exception("P went above Pmax") 
            # print("mean prec_negative", prec_negative.mean().item(), "max", prec_negative.max().item())
            # if torch.any(prec_negative>0.5*Pmax):
            #   raise Exception("Negative precipitation") 
            water_stored_lev  = torch.unsqueeze(water_stored,dim=1)
            water_stored_lev = torch.unsqueeze(torch.repeat_interleave(water_stored_lev,self.nlev_mem,dim=1),dim=2)
            rnn_mem = torch.cat((rnn_mem[:,:,0:self.nh_mem], water_stored_lev),dim=2)
            if self.pour_excess:
                precip=  sedimentation + water_released + water_excess # - prec_negative
                # print("model  water_stored", torch.mean(water_stored).item(), "Pmax", torch.mean(Pmax).item(), "precip", precip.mean().item(), "ecess", water_excess.mean().item())

            else:
                precip=  sedimentation + water_released # - prec_negative
                # print("model  water_stored", torch.mean(water_stored).item(), "Pmax", torch.mean(Pmax).item(), "precip", precip.mean().item())

            # print("model min max mean dpwater", torch.min(water_released).item(), torch.max(water_released).item(), torch.mean(water_released).item())
            # print("model min max mean P_old", torch.min(P_old).item(), torch.max(P_old).item(), torch.mean(P_old).item())

            # print("model sed", torch.mean(sedimentation).item())

          else:
            prec_negative = self.relu(-water_new) # punish model for diagnosing negative precip from column water changes?
            water_new = self.relu(water_new)
            precip =  sedimentation + torch.sum((one_over_g*pres_diff*d_precip_sourcesink),1)  # <-- we already reversed signs in d_precip_sourcesink

          # print("model min max mean precip", torch.min(precip).item(), torch.max(precip).item(), torch.mean(precip).item())

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
          precc = (precip/1000).unsqueeze(1)

          temp_sfc = (inputs_main[:,-1,0:1]*self.xdiv_lev[-1,0:1]) + self.xmean_lev[-1,0:1]
          snowfrac = self.temperature_scaling_precip(temp_sfc)
          precsc = snowfrac*precc
          if self.separate_radiation:
              out_sfc_rad = self.relu(out_sfc_rad)
            #   print("ny sfc", ny_sfc, "out sfc rad", out_sfc_rad.shape)
              out_sfc =  torch.cat((out_sfc_rad[:,0:2], precsc, precc, out_sfc_rad[:,2:]),dim=1)
          else:
              out_sfc = self.relu(out_sfc)
              out_sfc =  torch.cat((out_sfc[:,0:2], precsc, precc, out_sfc[:,2:]),dim=1) 
        else:
            out_sfc = self.relu(out_sfc)

        if self.mp_mode==-2:
          out[:,:,2] = self.relu(out[:,:,2])

        if self.predict_liq_frac:
          temp = (inputs_main[:,:,0]*self.xdiv_lev[:,0]) + self.xmean_lev[:,0]  
          temp = temp + (out[:,:,0]/self.yscale_lev[:,0]) * 1200
          liq_frac_diagnosed0    = self.temperature_scaling(temp)
          liq_frac_diagnosed = liq_frac_diagnosed0[:,ilev_crm:]
          # liq_frac_pred = out[:,:,3] + liq_frac_diagnosed*self.yscale_lev[:,3]
          # liq_frac_pred[temp<250.0] = 0.0 
          # liq_frac_pred[temp>275.0] = 1.0 
          # out[:,:,3] = self.relu(liq_frac_pred)
          temp = temp[:,ilev_crm:]
          # mem = rnn_mem[:,ilev_crm:]

          inds = (temp < 275.0) & (temp<250.0)
        #   x_predfrac = rnn_mem[inds]
          if self.separate_radiation:
              mem = rnn_mem 
          else:
              mem = rnn_mem[:,ilev_crm:]
          
          x_predfrac = mem[inds]
          liq_frac_pred = self.mlp_predfrac(x_predfrac)
          liq_frac_pred = torch.reshape(liq_frac_pred,(-1,))
          liq_frac_pred = liq_frac_pred.to(liq_frac_diagnosed.dtype)
          liq_frac_diagnosed[inds] = liq_frac_pred
          out[:,ilev_crm:,3] = self.relu(liq_frac_diagnosed)
          out[:,0:ilev_crm,3] = liq_frac_diagnosed0[:,0:ilev_crm]
        # print("shape out pred", out.shape, "sfc", out_sfc.shape)
        # print("model PREC fin2",torch.mean(out_sfc[:,3:4]).item())

        # return out, out_sfc, rnn_mem
        if self.physical_precip and self.return_neg_precip:
          return out, out_sfc, rnn_mem, prec_negative
        else:
          return out, out_sfc, rnn_mem
    
    @torch.jit.export
    def pp_mp(self, out, out_sfc, x_denorm):

        out_denorm      = out / self.yscale_lev
        out_sfc_denorm  = out_sfc / self.yscale_sca
        # print("pp1 MEAN FRAC P", torch.mean(out_denorm[:,:,3]).item(), "MIN", torch.min(out_denorm[:,:,3]).item(),  "MAx", torch.max(out_denorm[:,:,3]).item() )

        if self.output_sqrt_norm:
          signs = torch.sign(out_denorm)
          out_denorm = signs*torch.pow(out_denorm, 4)
          # out_denorm = signs*torch.square(out_denorm)
        # print("pp_mp 1 frac100, ", out_denorm[100,:,3].detach().cpu().numpy())
        T_old        = x_denorm[:,:,0:1]
        qliq_old     = x_denorm[:,:,2:3]
        qice_old     = x_denorm[:,:,3:4]   
        qn_old       = qliq_old + qice_old 

        if self.pred_total_water:
          # Predicting total water tendency and fraction of total water that is clouds 
          dqtot                 = out_denorm[:,:,1:2]
          cld_water_frac        = out_denorm[:,:,2:3].clone()
          cld_water_frac        = torch.square(torch.square(cld_water_frac)) 
          cld_water_frac        = torch.clamp(cld_water_frac, min=0.0, max=1.0)
          qv_old                = x_denorm[:,:,-1:]
          qtot_old              = qn_old + qv_old
          qtot_new              = qtot_old + dqtot*1200  
          #   qtot_new[qtot_new<0.0] = 0.0
          qv_new                = (1-cld_water_frac)*qtot_new
          qn_new                = (cld_water_frac)*qtot_new
          dqv                   = (qv_new - qv_old) * 0.0008333333333333334  #/1200  
          dqn                   = (qn_new - qn_old) * 0.0008333333333333334  #/1200  
          out_denorm[:,:,1:2]   = dqv 
          out_denorm[:,:,2:3]   = dqn 

        # print("shape x denorm", x_denorm.shape, "T", T_old.shape)
        T_new           = T_old  + out_denorm[:,:,0:1]*1200

        # T_new           = T_old  + out_denorm[:,:,0:1]*1200
        liq_frac_constrained    = self.temperature_scaling(T_new)

        if self.predict_liq_frac:
            liq_frac_pred = out_denorm[:,:,3:4]
            # print("min max lfrac pred raw", torch.max(liq_frac_pred).item(), torch.min(liq_frac_pred).item())
            # Hu et al. Fig 2 b:
            max_frac = torch.clamp(liq_frac_constrained + 0.2, max=1.0)
            min_frac = torch.clamp(liq_frac_constrained - 0.2, min=0.0)
            # print("shape lfracpre", liq_frac_pred.shape, "con", liq_frac_constrained.shape)
            liq_frac_constrained = torch.clamp(liq_frac_pred, min=min_frac, max=max_frac)
            # print("pp2 MEAN FRAC P", torch.mean(liq_frac_constrained).item(), "MIN", torch.min(liq_frac_constrained).item(),  "MAx", torch.max(liq_frac_constrained).item() )

        #                            dqn
        # print("mean DQN", torch.mean(torch.sum(out_denorm[:,:,2:3],1)))
        qn_new      = qn_old + out_denorm[:,:,2:3]*1200  
        # qn_new[qn_new<0.0] = 0.0
        qliq_new    = liq_frac_constrained*qn_new
        qice_new    = (1-liq_frac_constrained)*qn_new
        dqliq       = (qliq_new - qliq_old) * 0.0008333333333333334  #/1200  
        dqice       = (qice_new - qice_old) * 0.0008333333333333334  #/1200  
        sum = dqliq+dqice 
        # print("pp_mp mean dq liq", torch.mean(dqliq).item(), "dq ice", torch.mean(dqice).item(), "dq TOT", torch.mean( out_denorm[:,:,2:3]).item())
        # print(( "dq TOT", torch.mean( out_denorm[:,:,2:3]).item(), "dqliq+ice",torch.mean( sum).item() ))

        if self.predict_liq_frac:           # replace    dqn,   liqfrac
            out_denorm  = torch.cat((out_denorm[:,:,0:2], dqliq, dqice, out_denorm[:,:,4:]),dim=2)
        else:
            out_denorm  = torch.cat((out_denorm[:,:,0:2], dqliq, dqice, out_denorm[:,:,3:]),dim=2)

        return out_denorm, out_sfc_denorm


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
        # liquid_frac = (T_raw - 253.16) / 20.0 
        liquid_frac = (T_raw - 253.16) * 0.05 
        liquid_frac = F.hardtanh(liquid_frac, 0.0, 1.0)
        return liquid_frac
    
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

        T_old        = x_denorm[:,:,0:1]
        qliq_old     = x_denorm[:,:,2:3]
        qice_old     = x_denorm[:,:,3:4]   
        qn_old       = qliq_old + qice_old 

        T_new           = T_old  + out_denorm[:,:,0:1]*1200

        liq_frac_constrained    = self.temperature_scaling(T_new)

        #                            dqn
        qn_new      = qn_old + out_denorm[:,:,2:3]*1200  
        qliq_new    = liq_frac_constrained*qn_new
        qice_new    = (1-liq_frac_constrained)*qn_new
        dqliq       = (qliq_new - qliq_old) * 0.0008333333333333334 #/1200  
        dqice       = (qice_new - qice_old) * 0.0008333333333333334 #/1200   
        out_denorm  = torch.cat((out_denorm[:,:,0:2], dqliq, dqice, out_denorm[:,:,3:]),dim=2)
        
        return out_denorm, out_sfc_denorm
    
    def forward(self, inp_list : List[Tensor]):
        inputs_main   = inp_list[0]
        inputs_aux    = inp_list[1]
        rnn_mem      = inp_list[2]
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
        # rnn1_input = torch.cat((rnn1_input,self.rnn_mem), axis=2)
        inputs_main_crm = torch.cat((inputs_main_crm,rnn_mem), dim=2)
        
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
        #     rnn_mem        = self.mlp_latent(h_final)
        #     out_det         = self.mlp_output(rnn_mem)
        #     if self.output_prune:
        #         out_det[:,0:12,1:] = out_det[:,0:12,1:].clone().zero_()
        # out_sfc         = self.mlp_surface_output(h_sfc)
        # out_sfc         = self.relu(out_sfc)
        
        
        #  --------- STOCHASTIC RNN FOR PERTURBATION ----------
        if not self.deterministic_mode:
            if self.return_det:
                rnn_mem        = self.mlp_latent(h_final)
                out_det         = self.mlp_output(rnn_mem)
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
        rnn_mem        = self.mlp_latent(h_final)
        out             = self.mlp_output(rnn_mem)
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
            return out, out_sfc, rnn_mem, out_det
        else:
            return out, out_sfc, rnn_mem
        

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
        liquid_frac = (T_raw - 253.16) * 0.05 
        liquid_frac = F.hardtanh(liquid_frac, 0.0, 1.0)
        return liquid_frac
    
    def postprocessing(self, out, out_sfc):
        out             = out / self.yscale_lev
        out_sfc         = out_sfc / self.yscale_sca
        return out, out_sfc
        
    def pp_mp(self, out, out_sfc, x_denorm):
        out_denorm      = out / self.yscale_lev
        out_sfc_denorm  = out_sfc / self.yscale_sca

        T_old        = x_denorm[:,:,0:1]
        qliq_old     = x_denorm[:,:,2:3]
        qice_old     = x_denorm[:,:,3:4]   
        qn_old       = qliq_old + qice_old 

        # print("shape x denorm", x_denorm.shape, "T", T_old.shape)
        T_new           = T_old  + out_denorm[:,:,0:1]*1200

        # T_new           = T_old  + out_denorm[:,:,0:1]*1200
        liq_frac_constrained    = self.temperature_scaling(T_new)

        #                            dqn
        qn_new      = qn_old + out_denorm[:,:,2:3]*1200  
        qliq_new    = liq_frac_constrained*qn_new
        qice_new    = (1-liq_frac_constrained)*qn_new
        dqliq       = (qliq_new - qliq_old) * 0.0008333333333333334 #/1200  
        dqice       = (qice_new - qice_old) * 0.0008333333333333334 #/1200   
        out_denorm  = torch.cat((out_denorm[:,:,0:2], dqliq, dqice, out_denorm[:,:,3:]),dim=2)
        
        return out_denorm, out_sfc_denorm
    
    def forward(self, inp_list : List[Tensor]):
        inputs_main   = inp_list[0]
        inputs_aux    = inp_list[1]
        rnn_mem      = inp_list[2]
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
            inputs_main = torch.cat((inputs_main,rnn_mem), dim=2)
            
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
          
        rnn_mem = rnn2out
      
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
            # print("rnn1 mem shape 0", rnn_mem.shape)
            prec = torch.reshape(out_sfc,(-1,1,2))
            prec = torch.repeat_interleave(prec,self.nlev,dim=1)
            rnn_mem = torch.cat((rnn_mem[:,:,0:self.nh_mem-2:],prec),dim=2)
            # print("rnn1 mem shape 1", rnn_mem.shape)

        out_sfc =  torch.cat((out_sfc_rad[:,0:2], out_sfc, out_sfc_rad[:,2:]),dim=1)

        out_sfc = self.relu(out_sfc)
        
        if self.use_ar_noise:
            eps_prev = self.tau_t * eps_prev + self.tau_e * torch.randn_like(eps_prev) #eps
            if self.two_eps_variables:
              eps_prev2 = self.tau_t * eps_prev2 + self.tau_e * torch.randn_like(eps_prev)
              eps_prev = torch.stack((eps_prev,eps_prev2))
            return out, out_sfc, rnn_mem, eps_prev
        else:
            return out, out_sfc, rnn_mem
    

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
        liquid_frac = (T_raw - 253.16) * 0.05 
        liquid_frac = F.hardtanh(liquid_frac, 0.0, 1.0)
        return liquid_frac

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

        T_old        = x_denorm[:,:,0:1]
        qliq_old     = x_denorm[:,:,2:3]
        qice_old     = x_denorm[:,:,3:4]   
        qn_old       = qliq_old + qice_old 

        # print("shape x denorm", x_denorm.shape, "T", T_old.shape)
        T_new           = T_old  + out_denorm[:,:,0:1]*1200

        # T_new           = T_old  + out_denorm[:,:,0:1]*1200
        liq_frac_constrained    = self.temperature_scaling(T_new)

        #                            dqn
        qn_new      = qn_old + out_denorm[:,:,2:3]*1200  
        qliq_new    = liq_frac_constrained*qn_new
        qice_new    = (1-liq_frac_constrained)*qn_new
        dqliq       = (qliq_new - qliq_old) * 0.0008333333333333334 #/1200  
        dqice       = (qice_new - qice_old) * 0.0008333333333333334 #/1200   
        out_denorm  = torch.cat((out_denorm[:,:,0:2], dqliq, dqice, out_denorm[:,:,3:]),dim=2)
        
        return out_denorm, out_sfc_denorm
    
    def forward(self, inp_list : List[Tensor]):
        inputs_main   = inp_list[0]
        inputs_aux    = inp_list[1]
        rnn_mem      = inp_list[2]
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
            
        inputs_main = torch.cat((inputs_main,rnn_mem), dim=2)
            
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
          
        rnn_mem = out
      
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
            # print("rnn1 mem shape 0", rnn_mem.shape)
            prec = torch.reshape(out_sfc,(-1,1,2))
            prec = torch.repeat_interleave(prec,self.nlev,dim=1)
            rnn_mem = torch.cat((rnn_mem[:,:,0:self.nh_mem-2:],prec),dim=2)
            # print("rnn1 mem shape 1", rnn_mem.shape)

        out_sfc =  torch.cat((out_sfc_rad[:,0:2], out_sfc, out_sfc_rad[:,2:]),dim=1)

        out_sfc = self.relu(out_sfc)

        if self.use_ar_noise:
            eps_prev = self.tau_t * eps_prev + self.tau_e * torch.randn_like(eps_prev) #eps
            if self.two_eps_variables:
              eps_prev2 = self.tau_t * eps_prev2 + self.tau_e * torch.randn_like(eps_prev)
              eps_prev = torch.stack((eps_prev,eps_prev2))
            return out, out_sfc, rnn_mem, eps_prev
        else:
            return out, out_sfc, rnn_mem
        
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
        liquid_frac = (T_raw - 253.16) * 0.05 
        liquid_frac = F.hardtanh(liquid_frac, 0.0, 1.0)
        return liquid_frac
    
    def postprocessing(self, out, out_sfc):
        out             = out / self.yscale_lev
        out_sfc         = out_sfc / self.yscale_sca
        return out, out_sfc
        
    def pp_mp(self, out, out_sfc, x_denorm):
        out_denorm      = out / self.yscale_lev
        out_sfc_denorm  = out_sfc / self.yscale_sca

        T_old        = x_denorm[:,:,0:1]
        qliq_old     = x_denorm[:,:,2:3]
        qice_old     = x_denorm[:,:,3:4]   
        qn_old       = qliq_old + qice_old 

        # print("shape x denorm", x_denorm.shape, "T", T_old.shape)
        T_new           = T_old  + out_denorm[:,:,0:1]*1200

        # T_new           = T_old  + out_denorm[:,:,0:1]*1200
        liq_frac_constrained    = self.temperature_scaling(T_new)

        #                            dqn
        qn_new      = qn_old + out_denorm[:,:,2:3]*1200  
        qliq_new    = liq_frac_constrained*qn_new
        qice_new    = (1-liq_frac_constrained)*qn_new
        dqliq       = (qliq_new - qliq_old) * 0.0008333333333333334 #/1200  
        dqice       = (qice_new - qice_old) * 0.0008333333333333334 #/1200   
        out_denorm  = torch.cat((out_denorm[:,:,0:2], dqliq, dqice, out_denorm[:,:,3:]),dim=2)
        
        return out_denorm, out_sfc_denorm
    
    def forward(self, inputs_main, inputs_aux, rnn_mem):

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
            # rnn1_input = torch.cat((rnn1_input,self.rnn_mem), axis=2)
            inputs_main = torch.cat((inputs_main,rnn_mem), dim=2)
            
            
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
            rnn_mem = torch.flip(rnn2out, [1])
            
      
        out = self.mlp_output(rnn2out)

        if self.output_prune:
            # Only temperature tendency is computed for the top 10 levels
            # if self.separate_radiation:
            #     out[:,0:12,:] = out[:,0:12,:].clone().zero_()
            # else:
            out[:,0:12,1:] = out[:,0:12,1:].clone().zero_()
        
        out_sfc = self.mlp_surface_output(last_hidden)


        out_sfc = self.relu(out_sfc)

        return out, out_sfc, rnn_mem

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
        # liquid_frac = (T_raw - 253.16) / 20.0 
        liquid_frac = (T_raw - 253.16) * 0.05 
        liquid_frac = F.hardtanh(liquid_frac, 0.0, 1.0)
        return liquid_frac
        
    def pp_mp(self, out, out_sfc, x_denorm):

        out_denorm      = out / self.yscale_lev.to(device=out.device)
        
        T_old        = x_denorm[:,:,0:1]
        qliq_old     = x_denorm[:,:,2:3]
        qice_old     = x_denorm[:,:,3:4]   
        qn_old       = qliq_old + qice_old 

        # print("shape x denorm", x_denorm.shape, "T", T_old.shape)
        T_new           = T_old  + out_denorm[:,:,0:1]*1200

        # T_new           = T_old  + out_denorm[:,:,0:1]*1200
        liq_frac_constrained    = self.temperature_scaling(T_new)

        #                            dqn
        qn_new      = qn_old + out_denorm[:,:,2:3]*1200  
        qliq_new    = liq_frac_constrained*qn_new
        qice_new    = (1-liq_frac_constrained)*qn_new
        dqliq       = (qliq_new - qliq_old) * 0.0008333333333333334 #/1200  
        dqice       = (qice_new - qice_old) * 0.0008333333333333334 #/1200   
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
            preslev_nonorm  = self.preslay_nonorm(sp)
            thetae = thermo_rh(tk,preslev_nonorm,rh) / 3000.0
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
