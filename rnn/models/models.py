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
from layers import * #LayerPressure, LevelPressure
import torch.nn.functional as F
from typing import List, Tuple, Final, Optional
from torch import Tensor
from models_torch_kernels import *
import numpy as np 
from typing import Final 
from omegaconf import DictConfig, OmegaConf

# class BiRNN(nn.Module):
#     """
#     Demonstrating the basic BiRNN model from Ukkonen & Chantry (2025) Figure 1
#     Not actually used in this project but can be adapted to various sub-grid parameterization problems
#     """
#     use_initial_mlp: Final[bool]
#     def __init__(self, RNN_type='LSTM', 
#                  nx = 9, nx_sfc=17, 
#                  ny = 8, ny_sfc=8, 
#                  nneur=(64,64), 
#                  outputs_one_longer=False, 
#                  concat=False, 
#                  out_scale=None, 
#                  out_sfc_scale = None):
#         # Simple bidirectional RNN (Either LSTM or GRU) for predicting column 
#         # outputs shaped either (B, L, Ny) or (B, L+1, Ny) from column inputs
#         # (B, L, Nx) and optionally surface inputs (B, Nx_sfc) 
#         # If surface inputs exist, they are used to initialize first (upward) RNN 
#         # Differs from usual biRNN in that first RNN is connected to second
#         # Assumes top-of-atmosphere is first in memory i.e. at index 0 - otherwise move the flip operations
#         super(BiRNN, self).__init__()
#         self.nx = nx
#         self.ny = ny 
#         self.nx_sfc = nx_sfc 
#         self.ny_sfc = ny_sfc
#         self.nneur = nneur 
#         self.outputs_one_longer=outputs_one_longer
#         # if True, inputs are have sequence length N and outputs have length N+1
#         # could be useful when predicting fluxes
#         if len(nneur) < 1 or len(nneur) > 3:
#             sys.exit("Number of RNN layers and length of nneur should be 2 or 3")

#         self.RNN_type=RNN_type
#         if self.RNN_type=='LSTM':
#             RNN_model = nn.LSTM
#             self.use_lstm =  True
#         elif self.RNN_type=='GRU':
#             RNN_model = nn.GRU
#             self.use_lstm = False
#         else:
#             raise NotImplementedError()
                    
#         self.concat=concat
#         self.nonlin = nn.Tanh()
        
#         if out_scale is not None:
#             cuda = torch.cuda.is_available() 
#             device = torch.device("cuda" if cuda else "cpu")
#             self.yscale_lev = torch.from_numpy(out_scale).to(device)
#             self.yscale_sca = torch.from_numpy(out_sfc_scale).to(device)

#         if self.nx_sfc > 0:
#             self.mlp_surface1  = nn.Linear(nx_sfc, self.nneur[0])
#             if self.RNN_type=="LSTM":
#                 self.mlp_surface2  = nn.Linear(nx_sfc, self.nneur[0])

#         self.rnn1      = RNN_model(nx,            self.nneur[0], batch_first=True) # (input_size, hidden_size, num_layers=1
#         self.rnn2      = RNN_model(self.nneur[0], self.nneur[1], batch_first=True)
#         if len(self.nneur)==3:
#             self.rnn3      = RNN_model(self.nneur[1], self.nneur[2], batch_first=True)

#         # The final latent variable is either the output from the last RNN, or the concatenated outputs from both RNNs
#         if concat:
#             nh_rnn = sum(nneur)
#         else:
#             nh_rnn = nneur[-1]

#         self.mlp_output = nn.Linear(nh_rnn, self.ny)
#         if self.ny_sfc>0:
#             self.mlp_surface_output = nn.Linear(nneur[-1], self.ny_sfc)
            
#     def postprocessing(self, out, out_sfc):
#         out_denorm  = out / self.yscale_lev
#         out_sfc_denorm  = out_sfc / self.yscale_sca
#         return out_denorm, out_sfc_denorm
    
#     def forward(self, inputs_main, inputs_aux):
#         # Auxililiary surface-level variables are used to initialize the first, upward-iterating RNN (LSTM or GRU)
#         sfc1 = self.nonlin(self.mlp_surface1(inputs_aux))

#         if self.RNN_type=="LSTM":
#             sfc2 = self.nonlin(self.mlp_surface2(inputs_aux))
#             hidden = (sfc1.view(1,-1,self.nneur[0]), sfc2.view(1,-1,self.nneur[0])) # (h0, c0)
#         else:
#             hidden = (sfc1.view(1,-1,self.nneur[0]))

#         # TOA is first in memory, so we need to flip the axis to iterate from surface to TOA with the first RNN
#         inputs_main = torch.flip(inputs_main, [1])
      
#         out, hidden = self.rnn1(inputs_main, hidden)
        
#         if self.outputs_one_longer:
#             out = torch.cat((sfc1, out),axis=1)

#         out = torch.flip(out, [1]) # for the second RNN (and the final output) we want TOA first, so flip again
        
#         out2, hidden2 = self.rnn2(out) 
        
#         if self.use_lstm:
#             (last_h, last_c) = hidden2
#         else:
#             last_h = hidden2 

#         if self.concat:
#             rnnout = torch.cat((out2, out),axis=2)
#         else:
#             rnnout = out2
    
#         out = self.mlp_output(rnnout)

#         if self.ny_sfc>0:
#             out_sfc = self.mlp_surface_output(last_h.squeeze())  # use cell state or hidden state? likely doesn't matter
#             return out, out_sfc
#         else:
#             return out 

class Base_RNN_autoreg(nn.Module):
    """
    Base class for a more advanced version of the biLSTM using a latent convective memory (Fig 10 in Ukkonen & Chantry,2025)
    """
    # Shared TorchScript attributes
    use_initial_mlp: Final[bool]
    use_intermediate_mlp: Final[bool]
    add_pres: Final[bool]
    add_stochastic_layer: Final[bool]
    use_lstm: Final[bool]
    output_prune: Final[bool]
    predict_liq_frac: Final[bool] # Predict fraction of cloud that is liquid 
    predict_total_water: Final[bool] # Predict total water tendency, fraction of total water that is cloud, and liq_frac 
    use_mp_constraint: Final[bool]
    mp_mode: Final[int]
    use_ensemble: Final[bool]
    def __init__(self, cfg: DictConfig,
                coeffs: dict,
                device: torch.device,
                batch_first=False,
                ):  
        super().__init__()
        self.register_buffer('yscale_lev', torch.from_numpy(coeffs['yscale_lev']).to(device))
        self.register_buffer('yscale_sca', torch.from_numpy(coeffs['yscale_sca']).to(device))
        self.register_buffer('xmean_lev', torch.from_numpy(coeffs['xmean_lev']).to(device))
        self.register_buffer('xmean_sca', torch.from_numpy(coeffs['xmean_sca']).to(device))
        self.register_buffer('xdiv_lev',  torch.from_numpy(coeffs['xdiv_lev']).to(device))
        self.register_buffer('xdiv_sca', torch.from_numpy(coeffs['xdiv_sca']).to(device))
        self.register_buffer('hyai', torch.from_numpy(coeffs['hyai']).to(device))
        self.register_buffer('hybi', torch.from_numpy(coeffs['hybi']).to(device))
        self.register_buffer('hyam', torch.from_numpy(coeffs['hyam']).to(device))
        self.register_buffer('hybm', torch.from_numpy(coeffs['hybm']).to(device))
        self.register_buffer('lbd_qc', torch.from_numpy(coeffs['lbd_qc']).to(device))
        self.register_buffer('lbd_qi', torch.from_numpy(coeffs['lbd_qi']).to(device))
        self.register_buffer('lbd_qn', torch.from_numpy(coeffs['lbd_qn']).to(device))       
        self.ny = cfg.ny 
        self.ny0 = cfg.ny  #for diagnose precip option, need to distinguish between model outputs (ny) and intermediate outputs (ny0)
        self.nlev = cfg.nlev 
        self.nlev_mem = cfg.nlev
        # Note that in E3SM-MMF (that we are emulating), the CRM which handles moist physics is only active
        # in the bottom 50 levels; after moist physics computations radiation is computed on the full 60 levels 
        # This matters for us too, for instance it only makes sense to track convective memory on bottom 50 levels;
        # level-wise tendencies should be 0 in the top 10 levels for all variables except temperature
        self.nx_sfc = cfg.nx_sfc 
        self.ny_sfc = cfg.ny_sfc
        self.ny_sfc0 = cfg.ny_sfc
        self.nneur = cfg.nneur 
        self.nh_rnn1 = self.nneur[0]
        self.nx_rnn2 = self.nneur[0]
        self.nh_rnn2 = self.nneur[1]
        self.nh_mem = cfg.nh_mem
        self.nh_mem0 = self.nh_mem
        self.use_initial_mlp=cfg.use_initial_mlp
        self.add_pres = cfg.add_pres
        self.output_prune = cfg.output_prune
        self.use_lstm = cfg.use_lstm
        self.add_stochastic_layer = cfg.add_stochastic_layer
        self.predict_liq_frac = False
        self.predict_total_water = False
        if cfg.ensemble_size>1:
          self.use_ensemble = True 
        else:
          self.use_ensemble = False
        self.mp_mode=cfg.mp_mode
        # SELECT MICROPHYSICS MODE which determines model outputs
        # Temperature and wind tendencies are predicted in each case, but humidity/water outputs differ
        # mp_mode = 0   # regular 6 outputs: tendencies of (T), qv, qliq, qice, (u), (v)
        # Temperature tendency, q-wv tendency, cloud liquid tendency, cloud ice tendency, wind tendencies
        #          ['ptend_t', 'ptend_q0001', 'ptend_q0002',             'ptend_q0003',  'ptend_u', 'ptend_v']
        # mp_mode = 1   # 5 outputs: qv and qn; liq_frac DIAGNOSED from temperature (Hu et al.)
        # mp_mode = -1  # 6 outputs: qv, qn and liq_frac 
        # mp_mode = -2  # 6 outputs: qtot=qv+qn, liq_frac (fraction of cloud that is liquid) and cld_water_frac (fraction of total water that is cloud)
        self.use_mp_constraint = False
        print("mp_mode was set to {}, meaning..".format(self.mp_mode))
        if self.mp_mode==0:
            print("Predicting regular 6 outputs (tendencies of T, qv, qliq, qice, u, v)")
        elif self.mp_mode==1:
            print("Predicting tendencies of qv and qn; diagnosing qn into liq. and ice based on grid-scale mean temperature (Hu et al. 2025)")
            print("Number of outputs should be one less than usual (5 not 6), checking and quitting if not")
            if self.ny != 5:
                raise NotImplementedError("Number of outputs ny is incorrect, was {} but should be 5".format(self.ny))
            self.use_mp_constraint = True
        elif self.mp_mode==-1:
            print("Predicting tendencies of qv, qn and liquid cloud fraction")
            self.predict_liq_frac = True
            self.use_mp_constraint = True
        elif self.mp_mode==-2:
            print("Predicting tendency of qtot, fraction of cloud that is liquid, and fraction of total water that is cloud")
            self.predict_liq_frac  = True
            self.predict_total_water = True
            self.use_mp_constraint = True 

        nx = cfg.nx
        if self.add_pres:
            self.preslay = LayerPressure(self.hyam, self.hybm, batch_first=batch_first)
            self.preslay_nonorm = LayerPressure(self.hyam, self.hybm, norm=False)
            self.preslev_nonorm = LevelPressure(self.hyai, self.hybi)
            nx = nx +1
        self.nx = nx

        if self.add_stochastic_layer:
            if len(self.nneur) == 3:        
                self.nx_rnn3 = self.nneur[1]
                self.nh_rnn3 = self.nneur[2]   
            elif len(self.nneur) == 2:
                self.nx_rnn3 = self.nneur[0]
                self.nh_rnn3 = self.nneur[1]     
            else:
                raise NotImplementedError()
        else:
            if len(self.nneur) != 2:
                raise NotImplementedError()   
        if self.nh_mem != self.nneur[-1]:
            self.use_intermediate_mlp = True 
        else:
            self.use_intermediate_mlp = False
        if self.use_initial_mlp:
            self.nx_rnn1 = self.nneur[0]
            # should we include large-scale tendencies from the lowest level here?
        else:
            self.nx_rnn1 = nx
        self.nx_rnn1 = self.nx_rnn1 + self.nh_mem0

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

    @torch.jit.export
    def postprocessing(self, out, out_sfc, x_denorm):
        out_denorm      = out / self.yscale_lev
        out_sfc_denorm  = out_sfc / self.yscale_sca

        if not self.use_mp_constraint:
          return out, out_sfc 
        else:
          T_old        = x_denorm[:,:,0:1]
          qliq_old     = x_denorm[:,:,2:3]
          qice_old     = x_denorm[:,:,3:4]   
          qn_old       = qliq_old + qice_old 

          if self.predict_total_water:
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
            liq_frac_constrained = out_denorm[:,:,3:4]

          #                            dqn
          # print("mean DQN", torch.mean(torch.sum(out_denorm[:,:,2:3],1)))
          qn_new      = qn_old + out_denorm[:,:,2:3]*1200  
          # qn_new[qn_new<0.0] = 0.0
          qliq_new    = liq_frac_constrained*qn_new
          qice_new    = (1-liq_frac_constrained)*qn_new
          # print("qliq new min", qliq_new.min(), "max", qliq_new.max())
          dqliq       = (qliq_new - qliq_old) * 0.0008333333333333334  #/1200  
          dqice       = (qice_new - qice_old) * 0.0008333333333333334  #/1200  
          sum = dqliq+dqice 
          # print("pp_mp mean dq liq", torch.mean(dqliq).item(), "dq ice", torch.mean(dqice).item(), "dq TOT", torch.mean( out_denorm[:,:,2:3]).item())
          # print(( "dq TOT", torch.mean( out_denorm[:,:,2:3]).item(), "dqliq+ice",torch.mean( sum).item() ))

          if self.predict_liq_frac:      # replace       dqn,   liqfrac
            out_denorm  = torch.cat((out_denorm[:,:,0:2], dqliq, dqice, out_denorm[:,:,4:]),dim=2)
          else:
            out_denorm  = torch.cat((out_denorm[:,:,0:2], dqliq, dqice, out_denorm[:,:,3:]),dim=2)

          return out_denorm, out_sfc_denorm
    
    def forward(self, inp_list : List[Tensor]):
        raise NotImplementedError("Subclasses must implement forward()")


class RNN_autoreg(Base_RNN_autoreg):
    separate_radiation: Final[bool]
    batch_first: Final[bool]
    add_pres: Final[bool]
    add_stochastic_layer: Final[bool]
    use_lstm: Final[bool]
    def __init__(self, 
                 cfg: DictConfig, 
                 coeffs: dict, 
                 device: torch.device):
        self.batch_first = False 
        super().__init__(cfg, coeffs, device, batch_first=self.batch_first)
        self.separate_radiation  = cfg.separate_radiation
        self.add_pres = cfg.add_pres
        self.add_stochastic_layer = cfg.add_stochastic_layer
        self.nonlin = nn.Tanh()
        self.relu = nn.ReLU()
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
            self.nx_rad_crm = self.nh_mem0
            self.nx_rad_tot = self.nx_rad_gas + self.nx_rad_crm 
            self.nx = self.nx - 3
            self.nx_sfc_rad = 6 # 'pbuf_COSZRS' 'cam_in_ALDIF' 'cam_in_ALDIR' 'cam_in_ASDIF' 'cam_in_ASDIR' 'cam_in_LWUP' 
            self.nx_sfc = self.nx_sfc  - self.nx_sfc_rad
            self.ny_rad = 1
            self.ny_sfc_rad = self.ny_sfc0 - 2
            self.ny_sfc0 = 2
            self.nh_rnn1_rad = 96 
            self.nh_rnn2_rad = 96
            self.rnn1_rad      = nn.GRU(self.nx_rad_tot, self.nh_rnn1_rad,  batch_first=self.batch_first)   # (input_size, hidden_size)
            self.rnn2_rad      = nn.GRU(self.nh_rnn1_rad, self.nh_rnn2_rad,  batch_first=self.batch_first) 
            self.mlp_surface_rad = nn.Linear(self.nx_sfc_rad, self.nh_rnn1_rad)
            self.mlp_surface_output_rad = nn.Linear(self.nh_rnn2_rad, self.ny_sfc_rad)
            self.mlp_toa_rad  = nn.Linear(2, self.nh_rnn2_rad)
            self.mlp_output_rad = nn.Linear(self.nh_rnn2_rad, self.ny_rad)
        else:
            self.mlp_toa1  = nn.Linear(2, self.nh_rnn2)
            if self.use_lstm: self.mlp_toa2  = nn.Linear(2, self.nh_rnn2)

        if self.use_initial_mlp:
            self.mlp_initial = nn.Linear(self.nx, self.nneur[0])

        self.mlp_surface1  = nn.Linear(self.nx_sfc, self.nh_rnn1)
        if self.use_lstm:
          self.mlp_surface2  = nn.Linear(self.nx_sfc, self.nh_rnn1)

        if self.use_lstm:
            rnn_layer = nn.LSTM 
        else:
            rnn_layer = nn.GRU

        if self.add_stochastic_layer:
            self.rnn0   = rnn_layer(self.nx_rnn1, self.nh_rnn1,  batch_first=self.batch_first)
            self.rnn1   = rnn_layer(self.nx_rnn2, self.nh_rnn2,  batch_first=self.batch_first)  # (input_size, hidden_size)
            use_bias=False
            if self.use_lstm:
                self.rnn2 = MyStochasticLSTMLayer4(self.nx_rnn3, self.nh_rnn3, use_bias=use_bias)  
            else: 
                self.rnn2 = MyStochasticGRULayer5(self.nx_rnn3, self.nh_rnn3, use_bias=use_bias)   
        else:
            self.rnn1   = rnn_layer(self.nx_rnn1, self.nh_rnn1,  batch_first=self.batch_first)  # (input_size, hidden_size)
            self.rnn2   = rnn_layer(self.nx_rnn2, self.nh_rnn2,  batch_first=self.batch_first)
                
        self.sigmoid = nn.Sigmoid()

        nh_rnn = self.nh_rnn2

        if self.use_intermediate_mlp: 
          self.mlp_latent = nn.Linear(nh_rnn, self.nh_mem0)
          self.mlp_output = nn.Linear(self.nh_mem0, self.ny0)
        else:
          self.mlp_output = nn.Linear(nh_rnn, self.ny0)
            
        if self.predict_liq_frac:
          self.mlp_predfrac = nn.Linear(self.nh_mem, 1)

        self.mlp_surface_output = nn.Linear(self.nneur[-1], self.ny_sfc0)

    def forward(self, inp_list : List[Tensor]):

        inputs_main = inp_list[0]
        inputs_aux  = inp_list[1]
        rnn_mem     = inp_list[2]

        batch_size, seq_size, feature_size = inputs_main.shape
        if self.batch_first:
          flipdim=1 
        else:
          flipdim=0
          inputs_main = torch.transpose(inputs_main,0,1).contiguous()
    
        if self.add_pres:
            sp = torch.unsqueeze(inputs_aux[:,0:1],flipdim)
            # surface pressure, undo scaling
            sp = sp*self.xdiv_sca[0] + self.xmean_sca[0]
            # print("shape inp main", inputs_main.shape, "sp", sp.shape )
            pres  = self.preslay(sp)
            # preslev_nonorm  = torch.squeeze(self.preslev_nonorm(sp))
            # preslay_nonorm  = torch.squeeze(self.preslay_nonorm(sp))
            inputs_main = torch.cat((inputs_main,pres),dim=2)

        inputs_main_crm = inputs_main
            
        if self.use_initial_mlp:
            inputs_main_crm = self.nonlin(self.mlp_initial(inputs_main_crm))

        # The input (a vertical sequence) is concatenated with the output of the RNN from the previous time step 
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
            
            rnn1_input =  torch.flip(rnn0out, [flipdim])
        else:
            # LSTM upwards --> LSTM downwards
            # TOA is first in memory, so to start at the surface we need to go backwards
            rnn1_input = torch.flip(inputs_main_crm, [flipdim])

        if self.separate_radiation:
            inputs_sfc = torch.cat((inputs_aux[:,0:6],inputs_aux[:,12:]),dim=1)
        else:
            inputs_sfc = inputs_aux

        hx = self.nonlin(self.mlp_surface1(inputs_sfc))

        if self.use_lstm:
            cx = self.mlp_surface2(inputs_sfc)
            hidden = (torch.unsqueeze(hx,0), torch.unsqueeze(cx,0))
        else: 
            hidden = torch.unsqueeze(hx,0)

        rnn1out, states = self.rnn1(rnn1_input, hidden)
        del rnn1_input, states

        rnn1out = torch.flip(rnn1out, [flipdim])

        if self.separate_radiation:
          hx2 = torch.randn((batch_size, self.nh_rnn2),device=inputs_main.device)  # (batch, hidden_size)
          if self.use_lstm or self.add_stochastic_layer: 
              cx2 = torch.randn((batch_size, self.nh_rnn2),device=inputs_main.device)
        else: 
          inputs_toa = torch.cat((inputs_aux[:,1:2], inputs_aux[:,6:7]),dim=1) 
          hx2 = self.mlp_toa1(inputs_toa)
          if self.use_lstm or self.add_stochastic_layer: 
              cx2 = self.mlp_toa2(inputs_toa)
        
        if self.add_stochastic_layer:
            if self.batch_first:
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
            if self.batch_first:
              rnn2outt = torch.transpose(rnn2outt,0,1)
        else:
            rnn2outt, states = self.rnn2(rnn1out, hidden)

        del rnn1out, hidden

        if self.use_lstm:
            (last_h, last_c) = states
        else:
            last_h = states

        final_sfc_inp = last_h.squeeze() 
            
        if self.use_intermediate_mlp: 
            rnn_mem = self.mlp_latent(rnn2outt)
        else:
            rnn_mem = rnn2outt 

        out = self.mlp_output(rnn_mem)

        if self.output_prune and (not self.separate_radiation):
            # Only temperature tendency is computed for the top 10 levels, by the radiation scheme, after CRM runs on lowest 50 levels
            if self.batch_first:
              out[:,0:12,1:] = out[:,0:12,1:].clone().zero_()
            else:
              out[0:12,:,1:] = torch.zeros_like(out[0:12,:,1:])
        out_sfc = self.mlp_surface_output(final_sfc_inp)

        if self.separate_radiation:
          # out_crm = out.clone()
          if self.batch_first:
            out_new = torch.zeros(batch_size, self.nlev_rad, self.ny0, device=inputs_main.device)
            out_new[:,10:,:] = out
          else:
            out_new = torch.zeros(batch_size, self.nlev_rad, self.ny0, device=inputs_main.device)
            out_new[10:] = out
          # Start at surface again
          inputs_gas_rad =  inputs_main[:,:,12:15] # gases
          # # add dT from crm 
          # T_old =   inputs_main * (self.xcoeff_lev[2,:,0:1] - self.xcoeff_lev[1,:,0:1]) + self.xcoeff_lev[0,:,0:1] 
          # T_new = T_old + dT
          # inputs_rad =  torch.zeros(batch_size, self.nlev_rad, self.nh_mem+self.nx_rad,device=inputs_main.device)
          if self.batch_first:
            inputs_rad =  torch.zeros(batch_size, self.nlev_rad, self.nx_rad_tot, device=inputs_main.device)
          else:
            inputs_rad =  torch.zeros(self.nlev_rad, batch_size, self.nx_rad_tot, device=inputs_main.device)
          # inputs_rad[:,10:,0:self.nh_mem] = torch.flip(rnn2out, [1])
          # inputs_rad[:,:,self.nh_mem:] = inputs_main_rad
          inputs_rad[:,:,0:3] = inputs_gas_rad
          inputs_rad[:,10:,3:] = torch.flip(rnn_mem, [1])
          # inputs_rad = torch.flip(inputs_rad, [1])

          inputs_sfc_rad = inputs_aux[:,6:12]
          hx = self.mlp_surface_rad(inputs_sfc_rad)
          hidden = (torch.unsqueeze(hx,0))
          rnn_out, states = self.rnn1_rad(inputs_rad, hidden)
          rnn_out = torch.flip(rnn_out, [flipdim])

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
          out_sfc =  torch.cat((out_sfc_rad[:,0:2], out_sfc, out_sfc_rad[:,2:]),dim=1)

        if not self.batch_first:
          out = torch.transpose(out,0,1).contiguous()
        return out, out_sfc, rnn_mem 

