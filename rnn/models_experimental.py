#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytorch model constructors (old experimental stuff)
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
from metrics import specific_to_relative_humidity_torch_cc


def saturation_vapor_pressure_liquid(T):
    #if T < 123 || T > 332   
    #	@warn "Temperature out of range [123-332] K."
    #end
    temp = (54.842763 - (6763.22 / T) - 4.210  * torch.log(T) + 0.000367  * T) + (torch.tanh( 0.0415  * (T - 218.8 ) )*(53.878  - (1331.22 /T) - 9.44523 *torch.log(T) + 0.014025  * T))
    es = torch.exp(temp)
    return es

def thermo_rh(tk,p,rh):
    # thermo_rh generates thermodynamic variables from t(K) p(Pa)  rh(0-1) 
    # changed rsat denominator from p to p-es (3 may 07)
    # Realised thetae calculation is incorrect (17 Nov 2009)
    # Changed to used formulae of Bolton (1980), for pseudo-equivalent p.t.

    #convert t,p to SI units
    #p = p.*100f0; #Pa

    p0 = 100000.0	#reference pressure in Pa
    R = 287.0		#gas constant
    K = R/1004.0
    epsi = 0.621980

    # a = ((p/p0)**(-K))
    a = torch.pow((p/p0),(-K))

    theta = tk*a

    es = saturation_vapor_pressure_liquid(tk) # Saturation vapor pressure
    e = rh*es                            # vapour pressure (Pa)

    # Calculate water vapour mixing ratio r and q specific humidity
    r = (epsi*e)/(p-e)
    #rsat = (epsi*es)/(p-es)
    #ri = r - rsat; #liquid water content of air (Zhang et al 1990)
    # k = 0.28540 * (1 - 0.28*r)
    # change units from g/g to g/kg
    rg = r*1.0e3
    # rsat = rsat*1.0e3

    # calculate pseudo-equivalent potential temperature, from Bolton, Mon Wea Rev, 1980
    # r = is g/kg
    # Firstly calculate Temp at LCL, note e must be in mb.
    tk_lcl = ( 2840 / (3.50 * torch.log(tk) - torch.log(e/100.0) - 4.8050) ) + 55              # eqn 21, Bolton
    thetae = theta*torch.exp(( (3.3760/tk_lcl) - 0.002540)*rg*(1+0.810 * rg *1.0e-3))  # eqn 38, Bolton
    #thetaes = theta*exp(( (3.3760/tk_lcl) - 0.002540)*rsat*(1+0.810 * rg *1.0e-3))   # eqn 38, Bolton

    # #LCL height using Poisson's equation
    # p_lcl =  0.01*p*((tk_lcl/tk)^(1.0/k))

    # return theta, thetae, thetaes, p_lcl, tk_lcl, r 
    return thetae
    

lbd_qi = np.array([10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        ,  7556905.1099813 ,
        3240294.53811436,  4409304.29170528,  5388911.78320826,
        1414189.69398583,   444847.03675674,   550036.71076073,
         452219.47765234,   243545.07231263,   163264.17204164,
         128850.88117789,   108392.13699281,    96868.6539061 ,
          90154.39383647,    83498.67423248,    76720.52614694,
          70937.87706283,    66851.27198026,    64579.78345685,
          64987.05874437,    68963.77227883,    75498.91605962,
          82745.37660119,    89624.52634008,    96373.41157796,
         102381.42808207,   102890.33417304,    96849.77123401,
          92727.78368907,    91320.9721545 ,    91240.30382044,
          91448.65004889,    91689.26513737,    91833.1829058 ,
          91941.15859653,    92144.1029509 ,    92628.38565183,
          93511.1538428 ,    94804.20080999,    96349.5878153 ,
          98174.89731264,   100348.81479455,   102750.86508174,
         105013.71207426,   106732.83687405,   107593.00387448,
         108022.91061398,   109634.8552567 ,   112259.85403167], dtype=np.float32).reshape(60,1)


lbd_qc = np.array([10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        ,  2410793.53754872,
        3462644.65436088,  1594172.20270602,   328086.13752288,
         154788.55435228,   118712.37335602,   104208.42410058,
          95801.11739569,    89619.52961093,    83709.51800851,
          78846.75613935,    74622.76219094,    70555.95112947,
          66436.67119096,    61797.61126943,    56926.03823691,
          51838.00818631,    46355.21691466,    40874.23574077,
          36196.39550842,    32935.40953052,    31290.83140741,
          30908.27330462,    31386.06558422,    32606.7350768 ,
          34631.09245739,    37847.88977875,    42878.24049123,
          50560.90175672,    61294.98389768,    72912.41450047,
          80998.32102651,    88376.7321416 ,   135468.13760583], dtype=np.float32).reshape(60,1)
        
class LSTM_autoreg_torchscript_physprec2(nn.Module):
    """
    More advanced version of the biLSTM using a latent convective memory (Fig 10 in Ukkonen & Chantry,2025),
    attempt to incorporate some physics in the way precipitation is predicted,
    with further options that include:
    separate_radiation: separate biLSTM at the end for radiation (although we don't have separate tendencies in ClimSim data), 
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
                predict_liq_frac=False,
                randomly_initialize_cellstate=False, # introduce a smalld degree of randomness into deterministic LSTM
                output_sqrt_norm=False,
                nh_mem=16):  # dimension of latent convective memory (per vertical level)
        super(LSTM_autoreg_torchscript_physprec2, self).__init__()
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
        self.pred_total_water=True 

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
        self.influde_evap = False 
        if self.constrain_precip:
          self.pour_excess = False
        if self.store_precip: 
            self.nh_mem0 = self.nh_mem - 1 
        else:
            self.nh_mem0 = self.nh_mem
        
        print("conserve_water:",self.conserve_water, "storeprec:", self.store_precip, "constrainprec:", self.constrain_precip, 
            "pour:", self.pour_excess, "do heat:", self.do_heat_advection, "diffusivity:", self.include_diffusivity)

        self.mp_ncol = 16
        self.mlp_qtot_crm      = nn.Linear(self.nh_rnn2, self.mp_ncol)
        self.mlp_flux_qtot_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
        self.mlp_mp_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
        self.mlp_evap_prec_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
        self.mlp_qn_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
        self.mlp_mp_aa_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
        self.mlp_mp_aa2_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)

        if self.do_heat_advection:
          self.mlp_t_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
          self.mlp_flux_crm_t = nn.Linear(self.nh_rnn2, self.mp_ncol)

        self.softmax = nn.Softmax(dim=2)
        self.softmax_dim1 = nn.Softmax(dim=1)
            
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

        if self.include_diffusivity:
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
        if self.store_precip: 
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
        
        if not self.separate_radiation:
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

            out = out_new

        # with torch.autocast(device_type="cuda", enabled=False):
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

        # if self.pred_total_water:

        #  model outputs ['ptend_t', 'dqtot', 'cld_water_frac', 'liq_frac', 'ptend_u', 'ptend_v']
      
        out_neww[:,:,0] = out[:,:,0]
        out_neww[:,:,2:self.ny] = out[:,:,2:self.ny] 

        qv_old = inputs_denorm[:,ilev_crm-1:,-1]
        qliq_old = inputs_denorm[:,ilev_crm-1:,2]
        qice_old = inputs_denorm[:,ilev_crm-1:,3]
        qtot_old = qliq_old + qice_old + qv_old 
        qtot_old0 = qtot_old
        qtot_old = qtot_old[:,1:]

        flux_mult_coeff = 1.0e3

        # ------------
        dqtot_mp = self.mlp_mp_crm(rnn2outt)
        flux = self.mlp_flux_qtot_crm(rnn2outt)     # Total water flux

        T_old = inputs_denorm[:,ilev_crm:,0]
        p_T_crm_old = self.softmax(self.mlp_t_crm(rnn2outt))
        T_crm_old = self.mp_ncol*T_old.unsqueeze(2)*p_T_crm_old
        liq_frac_diagnosed_old   = self.temperature_scaling(T_crm_old)

        p_qtot_crm_old = self.softmax(self.mlp_qtot_crm(rnn2outt))
        qtot_crm_old = self.mp_ncol*qtot_old.unsqueeze(2)*p_qtot_crm_old 

        pressure = preslay_nonorm[:,ilev_crm:].unsqueeze(2)

        # print("qtot_crm_old max", qtot_crm_old.max().item(), "shape", qtot_crm_old.shape, "T_crm_old max", T_crm_old.max().item(), "shape", T_crm_old.shape)
        # print("max q_old",  qtot_old.max().item(), "temp", T_old.max().item(), "shape", T_old.shape, "P max", pressure.max().item(), "shape", pressure.shape)
        
        rh_crm, q_excess_crm = specific_to_relative_humidity_torch_cc(qtot_crm_old, T_crm_old, pressure, return_q_excess=True)
        # print("min max q_excess crm", q_excess_crm.min().item(), q_excess_crm.max().item())
        qn_crm_old = q_excess_crm


        # FORCE FLUX TO SURFACE (sedimentation) TO BE POSITIVE 
        # if it's negative, we have negative precipitation and water is taken from nowhere and added to the atmosphere
        # OR should we not bother to include a sedimentation term, and set boundary fluxes to zero to avoid net transport 
        flux[:,-1] = self.relu(flux[:,-1]) # net downward flux cannot be negative 

        qtot_old = qtot_crm_old 

        flux_net_qtot = flux_mult_coeff*flux*qtot_crm_old

        zeroes_crm = torch.zeros(batch_size, 1, self.mp_ncol, device=inputs_main.device)
        flux_net_qtot = torch.cat((zeroes_crm,flux_net_qtot),dim=1)
        sedimentation = torch.mean( flux_net_qtot[:,-1] + flux_net_qtot[:,-1], 1)
        flux_qtot_dp = scaling_factor*( (flux_net_qtot[:,1:] - flux_net_qtot[:,0:-1]) / pres_diff.unsqueeze(2)) 
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
          flux3 = self.mlp_flux_crm_t(rnn2outt)
          flux_net_t = flux3*5e-2*T_crm_old
          flux_net_t = torch.cat((zeroes_crm,flux_net_t),dim=1)
          flux_t_dp = scaling_factor*( (flux_net_t[:,1:] - flux_net_t[:,0:-1]) / pres_diff.unsqueeze(2)) 
          del flux3, p_T_crm_old, T_crm_old, flux_net_t 

        if True:
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
            max_dqtot_mp = (self.yscale_lev[ilev_crm:,1:2]*qtot_old/1200) + flux_qtot_dp
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


        qtot_crm_new = qtot_crm_old + flux_qtot_dp - dqtot_mp 
        
        rh_crm_new, q_excess_crm_new = specific_to_relative_humidity_torch_cc(crm_qtot_new, T_crm_new, pressure, return_q_excess=True)
        qv_crm_new = crm_qtot_new - q_excess_crm_new
        qn_crm_new = q_excess_crm_new

        out_neww[:,ilev_crm:,1] =  torch.mean(flux_qtot_dp - dqtot_mp,2)
        d_precip_sourcesink    = torch.mean(dqtot_mp,2)    

        liq_frac_diagnosed    = self.temperature_scaling(temp)
        # qn_new      = qn_old + out_denorm[:,:,2:3]*1200  
        qliq_crm_new    = liq_frac_diagnosed*qn_crm_new
        qice_crm_new    = (1-liq_frac_diagnosed)*qn_crm_new
        qliq_crm_old    = liq_frac_diagnosed_old*qn_crm_old
        qice_crm_old   = (1-liq_frac_diagnosed_old)*qn_crm_old


        dqliq       = (qliq_crm_new - qliq_crm_old) * 0.0008333333333333334  #/1200  
        dqice       = (qice_crm_new - qice_crm_old) * 0.0008333333333333334  #/1200  



        if self.include_diffusivity:
          out_neww[:,ilev_crm:-1,0] = out_neww[:,ilev_crm:-1,0] + diff_t
          out_neww[:,ilev_crm:-1,1] = out_neww[:,ilev_crm:-1,1] + diff_qtot

        if self.do_heat_advection: 
          out_neww[:,ilev_crm:,0] = out_neww[:,ilev_crm:,0] + torch.mean(flux_t_dp, 2)
          
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
       
        if self.mp_mode==-2:
          out[:,:,2] = self.relu(out[:,:,2])

        # if self.predict_liq_frac:
        temp = (inputs_main[:,:,0]*self.xdiv_lev[:,0]) + self.xmean_lev[:,0]  
        temp = temp + (out[:,:,0]/self.yscale_lev[:,0]) * 1200
        liq_frac_diagnosed    = self.temperature_scaling(temp)
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
        if self.return_neg_precip:
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

class LSTM_autoreg_torchscript_radflux(nn.Module):
    use_initial_mlp: Final[bool]
    use_intermediate_mlp: Final[bool]
    add_pres: Final[bool]
    add_stochastic_layer: Final[bool]
    output_prune: Final[bool]
    use_memory: Final[bool]
    mp_constraint: Final[bool]

    def __init__(self, hyam, hybm,  hyai, hybi,
                out_scale, out_sfc_scale, 
                xmean_lev, xmean_sca, xdiv_lev, xdiv_sca,
                device,
                nlev=60, nx = 4, nx_sfc=3, ny = 4, ny_sfc=3, nneur=(64,64), 
                use_initial_mlp=False, 
                use_intermediate_mlp=True,
                add_pres=False,
                add_stochastic_layer=False,
                output_prune=False,
                use_memory=False,
                coeff_stochastic = 0.0,
                nh_mem=16,
                mp_mode=0):
        super(LSTM_autoreg_torchscript_radflux, self).__init__()
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
        self.preslay_nonorm = LayerPressure(hyam,hybm,name="LayerPressure_nonorm", norm=False)
        self.preslev = LevelPressure(hyai,hybi)
        self.nx = nx
        self.nh_rnn1 = self.nneur[0]
        self.nx_rnn2 = self.nneur[0]
        self.nh_rnn2 = self.nneur[1]

        self.use_memory= use_memory
        if mp_mode==0:
          self.mp_constraint=False 
        elif mp_mode==1:
          self.mp_constraint=True
        else:
          raise NotImplementedError("model requires mp_mode>=0")

        # if self.predict_flux:
        self.preslev = LayerPressure(hyai, hybi, norm=False)
        if self.use_initial_mlp:
            self.nx_rnn1 = self.nneur[0]
        else:
            self.nx_rnn1 = nx

        # if self.separate_radiation:
        self.nlev = 60
        self.nlev_rad = 60
        self.nlev_mem = 50
        self.nx_crm = self.nx - 3
        self.nx_rad = self.nx - 2
        # self.nx_rad = nx - 2
        self.nx_rnn1 = self.nx_rnn1 - 3
        self.nx_sfc_rad = 5

        self.nx_sfc = self.nx_sfc  - self.nx_sfc_rad
        self.ny_crm = 4
        self.ny_rad = 4
        self.ny_sfc_rad = 4
        self.ny_sfc_crm = 2
            
        self.add_stochastic_layer = add_stochastic_layer
        self.coeff_stochastic = coeff_stochastic
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
        self.lbd_qc     = torch.tensor(lbd_qc, dtype=torch.float32, device=device)
        self.lbd_qi     = torch.tensor(lbd_qi, dtype=torch.float32, device=device)
        # in ClimSim config of E3SM, the CRM physics first computes 
        # moist physics on 50 levels, and then computes radiation on 60 levels!
        self.use_intermediate_mlp=use_intermediate_mlp
            
        if self.use_memory:
            if self.use_intermediate_mlp:
                # self.nh_mem = self.nneur[1] // 4
                # if nh_mem is None:
                #     self.nh_mem = self.nneur[1] // 8
                # else:
                self.nh_mem = nh_mem

            else:
                self.nh_mem = self.nneur[1]
            # self.rnn1_mem = None 
            print("Building RNN that feeds its hidden memory at t0,z0 to its inputs at t1,z0")
        else: 
            self.nh_mem = 0
            print("Building RNN without convective memory")

        print("Initial mlp: {}, intermediate mlp: {}".format(self.use_initial_mlp, self.use_intermediate_mlp))
 
        self.nx_rnn1 = self.nx_rnn1 + self.nh_mem
        self.nh_rnn1 = self.nneur[0]
        


        if self.use_initial_mlp:
            self.mlp_initial = nn.Linear(self.nx_crm, self.nneur[0])
            self.nx_rnn1 = self.nneur[0] + self.nh_mem
            
        print("nx rnn1", self.nx_rnn1, "nh rnn1", self.nh_rnn1)
        print("nx rnn2", self.nx_rnn2, "nh rnn2", self.nh_rnn2)  
        print("nx sfc", self.nx_sfc)
        
        self.mlp_surface1  = nn.Linear(self.nx_sfc, self.nh_rnn1)
        self.mlp_surface2  = nn.Linear(self.nx_sfc, self.nh_rnn1)

        # self.rnn1      = nn.LSTMCell(self.nx_rnn1, self.nh_rnn1)  # (input_size, hidden_size)
        # self.rnn2      = nn.LSTMCell(self.nx_rnn2, self.nh_rnn2)

        self.mlp_toa1  = nn.Linear(2, self.nh_rnn2)
        self.mlp_toa2  = nn.Linear(2, self.nh_rnn2)

        self.rnn1      = nn.LSTM(self.nx_rnn1, self.nh_rnn1,  batch_first=True)  # (input_size, hidden_size)
        if self.add_stochastic_layer:
            use_bias=False
            self.rnn2 = MyStochasticGRULayer(self.nx_rnn2, self.nh_rnn2, use_bias=use_bias)  
        else:
            self.rnn2 = nn.LSTM(self.nx_rnn2, self.nh_rnn2,  batch_first=True)

        nh_rnn = self.nh_rnn2

        if self.use_intermediate_mlp: 
            self.mlp_latent = nn.Linear(nh_rnn, self.nh_mem)
            self.mlp_output = nn.Linear(self.nh_mem, self.ny)
        else:
            self.mlp_output = nn.Linear(nh_rnn, self.ny)
            
        self.mlp_surface_output = nn.Linear(nneur[-1], self.ny_sfc_crm)
            
        # if self.separate_radiation:
            # self.nh_rnn1_rad = self.nh_rnn1 
            # self.nh_rnn2_rad = self.nh_rnn2 
        self.nh_rnn1_rad = 128 
        self.nh_rnn2_rad = 128
        self.rnn1_rad      = nn.GRU(9, self.nh_rnn1_rad,  batch_first=True)   # (input_size, hidden_size)
        self.rnn2_rad      = nn.GRU(self.nh_rnn1_rad, self.nh_rnn2_rad,  batch_first=True) 
        self.mlp_surface_rad = nn.Linear(self.nx_sfc_rad, self.nh_rnn1_rad)
        self.mlp_flux_scale = nn.Linear(self.nx_sfc_rad, 32)
        self.mlp_flux_scale2 = nn.Linear(32,1)

        self.mlp_surface_output_rad = nn.Linear(self.nh_rnn2_rad, self.ny_sfc_rad)
        # self.mlp_toa_rad  = nn.Linear(2, self.nh_rnn2_rad)
        self.mlp_output_rad = nn.Linear(self.nh_rnn2_rad, self.ny_rad)
        
        
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

    def temperature_scaling(self, T_raw):
        liquid_ratio = (T_raw - 253.16) * 0.05 
        liquid_ratio = F.hardtanh(liquid_ratio, 0.0, 1.0)
        return liquid_ratio
    
    def postprocessing(self, out, out_sfc):
        out             = out / self.yscale_lev
        out_sfc         = out_sfc / self.yscale_sca
        return out, out_sfc

    def relative_to_specific_humidity(self, rh, temp, pressure):
        """
        Convert relative humidity to specific humidity using PyTorch tensors.
        
        Args:
            rh (torch.Tensor): Relative humidity as a fraction (0-1)
            temp (torch.Tensor): Temperature in Kelvin
            pressure (torch.Tensor): Pressure in Pa (Pascals)
        
        Returns:
            torch.Tensor: Specific humidity (kg water vapor / kg total air)
        
        Notes:
            - Uses Clausius-Clapeyron relation for saturation vapor pressure
            - Computes temperature-dependent latent heat of vaporization
            - All calculations performed in Kelvin (no temperature conversion)
        """
        
        # Constants
        es0 = 611.2  # Reference saturation vapor pressure at T0 (Pa)
        T0 = 273.15  # Reference temperature (K) - triple point of water
        Rv = 461.5   # Specific gas constant for water vapor (J/(kg·K))
        
        # Gas constant ratio (water vapor / dry air)
        epsilon = 0.622  # kg/kg
        
        # Temperature-dependent latent heat of vaporization (J/kg)
        # Linear relationship: Lv = Lv0 + a * (T - T0)
        # where Lv0 = 2.501e6 J/kg at 273.15K, and a ≈ -2370 J/(kg·K)
        Lv0 = 2.501e6  # Latent heat at reference temperature (J/kg)
        a = -2370.0    # Temperature coefficient (J/(kg·K))
        
        Lv = Lv0 + a * (temp - T0)
        
        # Calculate saturation vapor pressure using Clausius-Clapeyron relation
        # es = es0 * exp((Lv/Rv) * (1/T0 - 1/T))
        e_sat = es0 * torch.exp((Lv / Rv) * (1/T0 - 1/temp))
        
        # Calculate actual vapor pressure
        e_actual = rh * e_sat
        
        # Calculate specific humidity
        # q = (epsilon * e) / (p - e * (1 - epsilon))
        specific_humidity = (epsilon * e_actual) / (pressure - e_actual * (1 - epsilon))
        
        return specific_humidity

    def forward(self, inp_list : List[Tensor]):
        inputs_main     = inp_list[0]
        inputs_aux      = inp_list[1]
        rnn1_mem        = inp_list[2]
        x_denorm        = inp_list[3]
        
        incflux = inputs_aux[:,1:2] # TOA flux * cos_sza

        batch_size = inputs_main.shape[0]
        # print("shape inputs main", inputs_main.shape)
        if self.add_pres:
            sp = torch.unsqueeze(inputs_aux[:,0:1],1)
            # undo scaling
            sp = sp*35451.17 + 98623.664
            pres  = self.preslay(sp)
            inputs_main = torch.cat((inputs_main,pres),dim=2)
            play = self.preslay_nonorm(sp)
            plev = self.preslev(sp)
            
                    # Do not use inputs -2,-3,-4 (O3, CH4, N2O) or first 10 levels
        inputs_main_crm = torch.cat((inputs_main[:,10:,0:-4], inputs_main[:,10:,-1:]),dim=2)
            
        if self.use_initial_mlp:
            inputs_main_crm = self.mlp_initial(inputs_main_crm)
            inputs_main_crm = self.nonlin(inputs_main_crm)  
            
        # print("shape inputs main", inputs_main_crm.shape, "mem shape", rnn1_mem.shape)

        if self.use_memory:
            # rnn1_input = torch.cat((rnn1_input,self.rnn1_mem), axis=2)
            inputs_main_crm = torch.cat((inputs_main_crm,rnn1_mem), dim=2)
            
        # TOA is first in memory, so to start at the surface we need to go backwards
        rnn1_input = torch.flip(inputs_main_crm, [1])
    
        # The input (a vertical sequence) is concatenated with the
        # output of the RNN from the previous time step 
        
        # CRM aux inputs do not include 'cam_in_ALDIF' 'cam_in_ALDIR' 'cam_in_ASDIF' 'cam_in_ASDIR'
        #  'cam_in_LWUP' 
        inputs_sfc = torch.cat((inputs_aux[:,0:6],inputs_aux[:,11:]),dim=1)

        hx = self.mlp_surface1(inputs_sfc)
        hx = self.nonlin(hx)
        cx = self.mlp_surface2(inputs_sfc)
        cx = self.nonlin(cx)
        hidden = (torch.unsqueeze(hx,0), torch.unsqueeze(cx,0))

        rnn1out, states = self.rnn1(rnn1_input, hidden)
        
        # For now do NOT predict fluxes for convection, only radiation
        # if self.predict_flux:
        # rnn1out = torch.cat((torch.unsqueeze(hx,1),rnn1out),dim=1)

        rnn1out = torch.flip(rnn1out, [1])

        inputs_toa = torch.cat((inputs_aux[:,1:2], inputs_aux[:,6:7]),dim=1) 
        hx2 = self.mlp_toa1(inputs_toa)
        if not self.add_stochastic_layer: 
            cx2 = self.mlp_toa2(inputs_toa)
        
        if self.add_stochastic_layer:
            hidden2 = hx2
        else:
            hidden2 = (torch.unsqueeze(hx2,0), torch.unsqueeze(cx2,0))

        
        if self.add_stochastic_layer:
            input_rnn2 = torch.transpose(rnn1out,0,1)
            
            rnn2out = self.rnn2(input_rnn2, hidden2)
            final_sfc_inp = rnn2out[-1,:,:]

            rnn2out = torch.transpose(rnn2out,0,1)

        else:
            input_rnn2 = rnn1out
            rnn2out, states = self.rnn2(input_rnn2, hidden2)

            (last_h, last_c) = states
            final_sfc_inp = last_h.squeeze() 
            
        # Add a stochastic perturbation
        # Convective memory is still based on the deterministic model,
        # and does not include the stochastic perturbation
        # concat and use_intermediate_mlp should be set to false
        # if self.add_stochastic_layer:
        #     srnn_input = torch.transpose(rnn2out,0,1)
        #     srnn_input = torch.flip(srnn_input, [0])
        #     # srnn_input = torch.transpose(self.rnn1_mem,0,1)
        #     # srnn_input = torch.transpose(rnn1_mem,0,1)
        #     # transpose is needed because this layer assumes seq. dim first
            
        #     hx2 = torch.randn((batch_size, self.nh_rnn2),device=inputs_main.device)  # (batch, hidden_size)
        #     z = self.rnn_stochastic(srnn_input, hx2)
        #     z = torch.flip(z, [0])

        #     z = torch.transpose(z,0,1)
        #     # z = torch.flip(z, [1])
        #     # rnn2out = z
        #     # z is a perburbation added to the hidden state
        #     rnn2out = rnn2out + 0.01*z 
        #     # rnn2out = rnn2out + self.coeff_stochastic*z 
        
        if self.use_intermediate_mlp: 
            rnn2out = self.mlp_latent(rnn2out)
          
        if self.use_memory:
            rnn1_mem = torch.flip(rnn2out, [1])
            
        # # Add a stochastic perturbation
        # # Convective memory is still based on the deterministic model,
        # # and does not include the stochastic perturbation
        # # concat and use_intermediate_mlp should be set to false

        out = self.mlp_output(rnn2out)

        out_sfc = self.mlp_surface_output(final_sfc_inp)
        

        out_new = torch.zeros(batch_size, self.nlev_rad, self.ny, device=inputs_main.device)
        out_new[:,10:,:] = out
        out_denorm      = out_new / self.yscale_lev
        T_before        = x_denorm[:,:,0:1]
        rh_before       = x_denorm[:,:,1:2]
        # rh_before =  torch.clamp(rh_before, min=0.1, max=1.4)
        q_before        = self.relative_to_specific_humidity(rh_before, T_before, play)
        q_before        = torch.clamp(q_before, min=0.0, max=0.5)
        qliq_before     = x_denorm[:,:,2:3]
        qice_before     = x_denorm[:,:,3:4]   
        qn_before       = qliq_before + qice_before 
        T_new           = T_before  + out_denorm[:,:,0:1]*1200
        if self.mp_constraint:
          liq_frac_constrained    = self.temperature_scaling(T_new)
          #                            dqn
          qn_new      = qn_before + out_denorm[:,:,2:3]*1200  
          qliq_new    = liq_frac_constrained*qn_new
          qice_new    = (1-liq_frac_constrained)*qn_new
        else:
          qliq_new    = qliq_before + out_denorm[:,:,2:3]*1200 
          qice_new    = qice_before + out_denorm[:,:,3:4]*1200 

        dq          = out_denorm[:,:,1:2]
        q_new       = q_before + dq*1200 
        q_new       = torch.clamp(q_new, min=0.0)


        vmr_h2o = q_new * 1.608079364 # (28.97 / 18.01528) # mol_weight_air / mol_weight_gas
        #                             avogad)
        # T_new = (T_new - self.xmean_lev[:,0:1] ) / (self.xdiv_lev[:,0:1])
        temp = (T_new - 160 ) / (180)
        pressure = (torch.log(play) - 0.00515) / (11.59485)
        vmr_h2o = (torch.sqrt(torch.sqrt(vmr_h2o))  - 0.0101) / 0.497653
        # print("q new  min max", torch.min(q_new[:,:]).item(), torch.max(q_new[:,:]).item())
        # print("q nans", torch.nonzero(torch.isnan(q_new.view(-1))))
        # assert not torch.isnan(q_new).any()
        # 0. INPUT SCALING - clouds
        qliq_new = 1 - torch.exp(-qliq_new * self.lbd_qc)
        qice_new = 1 - torch.exp(-qice_new * self.lbd_qi)
        
        # Radiation inputs: pressure, temperature, water vapor, cloud ice and liquid, O3, CH4, N2O,, cld heterogeneity
        cloudfrac  = torch.zeros(batch_size, self.nlev, 1, device=inputs_main.device)
        cloudfrac[:,10:] = rnn1_mem[:,:,0:1]

        # inputs_rad =  torch.zeros(batch_size, self.nlev_rad, 9, device=inputs_main.device)
        # inputs_rad[:,:,0:6] =  torch.cat((pressure, temp, vmr_h2o, qliq_new, qice_new, cloudfrac),dim=2)
        # inputs_rad[:,:,6:] =  inputs_main[:,:,12:15]
        inputs_rad = torch.cat((pressure, temp, vmr_h2o, qliq_new, qice_new, cloudfrac, inputs_main[:,:,12:15] ),dim=2)
        # inputs_rad = torch.cat((pressure, temp, vmr_h2o, qliq_new, qice_new, inputs_main[:,:,12:15], cloudfrac),dim=2)
        

        # RADIATION
        # if self.separate_radiation:
        # out_crm = out.clone()
        # out_new = torch.zeros(batch_size, self.nlev_rad, self.ny, device=inputs_main.device)
        # out_new[:,10:,:] = out
        # # Start at surface again
        # # Do not use inputs 4,5 (winds)
        # inputs_main_rad =  torch.cat((inputs_main[:,:,0:4], inputs_main[:,:,6:]),dim=2)
        # # # add dT, dq, dq_cldliq, dq_cldice from crm 
        # # inputs_main_rad[:,:,0:1] = inputs_main_rad[:,:,0:1] + tend
        
        # # T_old =   inputs_main * (self.xcoeff_lev[2,:,0:1] - self.xcoeff_lev[1,:,0:1]) + self.xcoeff_lev[0,:,0:1] 
        # # T_new = T_old + dT
        # inputs_rad =  torch.zeros(batch_size, self.nlev_rad, self.ny_crm+self.nx_rad,device=inputs_main.device)
        # inputs_rad[:,10:,0:self.ny_crm] = out[:,:,0:self.ny_crm] #torch.flip(rnn2out, [1])
        # inputs_rad[:,:,self.ny_crm:] = inputs_main_rad

        inputs_rad = torch.flip(inputs_rad, [1])

        inputs_sfc_rad = inputs_aux[:,6:11]
        # inputs_sfc_rad = inputs_aux[:,7:11]

        hx = self.mlp_surface_rad(inputs_sfc_rad)
        hidden = (torch.unsqueeze(hx,0))
        rnn_out, last_h = self.rnn1_rad(inputs_rad, hidden)
        hx = torch.unsqueeze(hx,1)
        # print("shape out rnnrad", rnn_out.shape, "hidden", hx.shape)
        rnn_out = torch.cat((hx,rnn_out),dim=1)

        rnn_out = torch.flip(rnn_out, [1])

        # inputs_toa = torch.cat((inputs_aux[:,1:2], inputs_aux[:,6:7]),dim=1) 
        # hx2 = self.mlp_toa_rad(inputs_toa)
        # hidden2 = (torch.unsqueeze(hx2,0))
        hx2 = torch.randn((batch_size, self.nh_rnn2_rad),device=inputs_main.device)  # (batch, hidden_size)
        hidden2 = (torch.unsqueeze(hx2,0))
        rnn_out, last_h = self.rnn2_rad(rnn_out, hidden2)
        out_rad = self.mlp_output_rad(rnn_out) # 4: LW_down, LW_up, SW_down, SW_yp
        out_rad = self.relu(out_rad)

        lw_flux_scale = self.mlp_flux_scale(inputs_sfc_rad)
        lw_flux_scale = self.mlp_flux_scale2(lw_flux_scale)

        lw_down = out_rad[:,:,0:1] * torch.reshape(lw_flux_scale,(-1,1,1))
        lw_down_sfc = lw_down[:,-1]
        lw_up = out_rad[:,:,1:2] * torch.reshape(lw_flux_scale,(-1,1,1))
        lw_net = lw_down - lw_up

        # print("inc toa  min max", torch.min(incflux).item(), torch.max(incflux).item())

        sw_down = out_rad[:,:,2:3]*torch.reshape(incflux,(-1,1,1))
        sw_up = out_rad[:,:,3:4]*torch.reshape(incflux,(-1,1,1))
        sw_net = sw_down - sw_up       
        sw_net_sfc = sw_net[:,-1]
        
        rad_net = lw_net + sw_net
        flux_diff = rad_net[:,1:] - rad_net[:,0:-1]
        
        preslev  = self.preslev(sp)
        pres_diff = preslev[:,1:] - preslev[:,0:-1]
        dT_rad = -(flux_diff / pres_diff) * 0.009767579681 # * g/cp = 9.80665 / 1004
        # normalize heating rate output
        dT_rad = dT_rad * self.yscale_lev[:,0:1] 
        
        # dT_tot = dT_crm + dT_rad
        out_new[:,:,0:1] = out_new[:,:,0:1] + dT_rad   
        # print("shape out new", out_new.shape)

        out_sfc_rad = self.mlp_surface_output_rad(last_h) 
        out_sfc_rad = torch.squeeze(out_sfc_rad) * incflux 
        # normalize SW flux outputs
        out_sfc_rad = out_sfc_rad * self.yscale_sca[4:]
        sw_net_sfc  = sw_net_sfc  * self.yscale_sca[0:1]
        
        # lw_down_sfc = lw_down_sfc * self.yscale_sca[1:2]

        #1D (scalar) Output variables: ['cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 
        #'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']
        # print("shape 1", out_sfc_rad.shape, "2", out_sfc.shape)
        # rad predicts everything except PRECSC, PRECC
        out_sfc =  torch.cat((sw_net_sfc, lw_down_sfc, out_sfc, out_sfc_rad),dim=1)
        # print("shape out sfc", out_sfc.shape, "swn", sw_net_sfc.shape, "lwd", lw_down.shape, "outsfcrad", out_sfc_rad.shape)
        out_sfc = self.relu(out_sfc)

        return out_new, out_sfc, rnn1_mem
    
class SpaceStateModel_autoreg(nn.Module):
    # use_initial_mlp: Final[bool]
    use_intermediate_mlp: Final[bool]
    add_pres: Final[bool]
    output_prune: Final[bool]
    concat: Final[bool]
    pres_step_scale: Final[bool]
    nx_mlp0: Final[int]
    # autoregressive: Final[bool]
    model_type: Final[str]
    use_glu_layers: Final[bool]
    reduce_dim_with_mlp: Final[bool]
    model_is_lru: Final[bool]
    model_is_mingru: Final[bool]
    model_is_s5: Final[bool]
    model_is_mamba: Final[bool]
    model_is_gss: Final[bool]
    model_is_qrnn: Final[bool]
    model_is_sru: Final[bool]
    model_is_sru2d: Final[bool]
    model_is_gateloop: Final[bool]
        
    def __init__(self,  hyam, hybm,  hyai, hybi,
                out_scale, out_sfc_scale, 
                xmean_lev, xmean_sca, xdiv_lev, xdiv_sca,
                device,
                nlev=60, nx = 4, nx_sfc=3, ny = 4, ny_sfc=3, nneur=(64,64), 
                model_type='LRU', 
                # use_initial_mlp=False, 
                use_intermediate_mlp=True,
                add_pres=False,
                output_prune=False,
                concat=False,
                nh_mem=16
                ):
        super(SpaceStateModel_autoreg, self).__init__()
        self.nx = nx
        self.ny = ny 
        self.nlev = nlev 
        self.nx_sfc = nx_sfc 
        self.ny_sfc = ny_sfc
        self.nneur = nneur 
        self.add_pres = add_pres
        if self.add_pres:
            self.preslay = LayerPressure(hyam,hybm)
            nx = nx +1
        self.output_prune = output_prune
        self.nh_rnn1 = self.nneur[0]
        self.nh_rnn2 = self.nneur[1]
        self.ny_rnn1 = self.nh_rnn1
        # self.use_initial_mlp=use_initial_mlp
        self.model_type=model_type
        self.device=device

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
            
        print("Using model type {}".format(model_type))
        if self.model_type in ["Mamba","GSS"]:#,"QRNN"]:  #Mamba doesnt support providing the state
            print("WARNING: this SSM type doesn't support providing the state, so the surface variables are instead added to each point in the sequence")
            
            # concatenate the vertical (sequence) inputs with tiled scalars
            # self.use_initial_mlp=True
            nx = nx + nx_sfc
        # if self.use_initial_mlp: 
        # self.nh_mlp1 = self.nh_rnn1
        self.nx_rnn1 = self.nh_rnn1
        # else:
        #     self.nx_rnn1 = nx
        self.use_intermediate_mlp = False
        # if memory == 'None':
        #     print("Building non-autoregressive SSM, but may be stateful")
        #     self.autoregressive = False
        # elif memory == 'Hidden':
        # self.autoregressive = True
        print("Building autoregressive SSM that feeds a hidden memory at t0,z0 to SSM1 at t1,z0")
        self.rnn1_mem = None
        # self.rnn2_mem = None
        # self.use_intermediate_mlp = True
            
        # if self.use_intermediate_mlp:
        #     self.nh_mem = self.nh_rnn2
        # self.nh_mem = self.nh_rnn2
        self.nh_mem = nh_mem
        glu_expand_factor = 2
        # if model_type in ['S5','GSS','QRNN','Mamba','GateLoop','SRU','LRU']:
        if model_type in ['S5','GSS','QRNN','Mamba','GateLoop','SRU','SRU_2D','LRU']:
            # self.use_initial_mlp = True
            self.use_intermediate_mlp = True
            # self.nh_mem = self.nh_rnn2
            self.nx_rnn1 = self.nh_rnn1 
            self.nx_rnn2 = self.nh_rnn2
            # if self.autoregressive and model_type != 'SRU_2D':
            # This is a bit tricky with these SSMs as they expect 
            # num_inputs = num_hidden. Therefore we need MLPs that halve 
            # the size of both inputs and the final hidden variable, e.g.
            # SSM = 128 neurons, Xlay = 64 neurons, Xhidden = 64 neurons,
            # X = concat(Xlay,Xhidden)
            # Similarly, to feed the latent variable to RNN2, we would need an MLP before it
            # that halves the size of  the output from RNN1
            #    mlp1      concat(mlp1,hfin=64)     RNN1(128)     MLP2     concat(mlp2,hfin=64)      RNN2(128)     mlpfin
            # nx ----> 64 ---------------------> 128 -------> 128 ---> 64 ---------------------> 128 --------> 128 ----->  64
            #   OR let second SSMs hidden size be bigger
            #    mlp1      concat(mlp1,hfin=64)     RNN1(128)      concat(hfin=64)           RNN2(192)     mlpfin
            # nx ----> 64 ---------------------> 128 -------> 128 ---------------------> 192 --------> 192 ----->  64  
            # or Dont bother making RNN2 autoregressive?
            # mlp1        concat(hfin=128)          RNN1(256)     GLU      RNN2(128)    GLU      RNN3
            # nx ---> 128 ---------------------> 256 -------> 256 ---> 128 -------> 128 ---> 128 --->  128

            # self.nh_mem = self.nh_rnn1//2
            # self.ny_rnn1 = self.nh_rnn1//2
            # self.nh_mlp1  = self.nh_rnn1//2 
            # self.nx_rnn2 = self.nh_rnn2
            # glu_expand_factor = 1
            #                  mlp            RNN
            #  concat(nx,16)  ---> 128  ---> 
            self.ny_rnn1 = self.nh_rnn1
            # self.nh_mlp1  = self.nh_rnn1
            self.nx_rnn2 = self.nh_rnn2
            glu_expand_factor = 2

        else:
            raise NotImplementedError() # need mlp here too for custom nh_mem

            self.nx_rnn2 = self.nh_rnn1
            # if self.autoregressive:
            self.nx_rnn1 = self.nx_rnn1 + self.nh_mem 
            # self.nx_rnn2 = self.nh_rnn1 + self.nh_mem 

        # if self.use_initial_mlp:
            # self.mlp_initial = nn.Linear(nx, self.nh_mlp1 )
        self.nx_mlp0 = self.nh_mem + nx 
        print("nx_mlp0", self.nx_mlp0, "nx", nx, "nh_rnn1", self.nh_rnn1)
        print("initial mLP dims:", self.nx_mlp0, self.nh_rnn1)
        self.mlp_initial = nn.Linear(self.nx_mlp0, self.nh_rnn1 )

        print("nx rnn1", self.nx_rnn1, "nh rnn1", self.nh_rnn1, "ny rnn1", self.ny_rnn1)
        print("nx rnn2", self.nx_rnn2, "nh rnn2", self.nh_rnn2)  
        # if self.autoregressive: 
        print("nh_memory", self.nh_mem)
        self.concat = concat

        self.mlp_surface1  = nn.Linear(nx_sfc, self.nh_rnn1)
        self.pres_step_scale = False

        self.model_is_lru = False
        self.model_is_mingru = False 
        self.model_is_s5 = False
        self.model_is_mamba = False
        self.model_is_gss = False
        self.model_is_qrnn = False
        self.model_is_sru = False
        self.model_is_sru2d = False
        self.model_is_gateloop = False
        if model_type == 'LRU':
            self.model_is_lru = True
            from models_torch_kernels_LRU import LRU
            self.rnn1= LRU(in_features=self.nx_rnn1,out_features=self.nh_rnn1,state_features=self.nh_rnn1)
            self.rnn2= LRU(in_features=self.nx_rnn2,out_features=self.nh_rnn2,state_features=self.nh_rnn2)
        elif model_type == "MinGRU":
            self.model_is_mingru = True
            # MinGRU = models_torch_kernels.MinGRU
            from models_torch_kernels import MinGRU as MinGRU
            self.rnn1= MinGRU(self.nx_rnn1,self.nh_rnn1)
            self.rnn2= MinGRU(self.nx_rnn2,self.nh_rnn2)
            # from models_torch_kernels import minGRU as MinGRU
            # self.rnn1= MinGRU(self.nh_rnn1)
            # self.rnn2= MinGRU(self.nh_rnn2)
        elif model_type == 'S5':
            self.model_is_s5 = True
            from s5 import S5
            self.liquid=False
            self.pres_step_scale = False
            # S5 outputs shape equals inputs?
            # self.rnn1= S5(self.nx_rnn1,self.nh_rnn1,liquid=self.liquid)
            # self.rnn2= S5(self.nx_rnn2,self.nh_rnn2,liquid=self.liquid)   
            # self.rnn1= S5(self.nh_rnn1,liquid=self.liquid)
            # self.rnn2= S5(self.nh_rnn2,liquid=self.liquid)   
            self.rnn1= S5(self.nx_rnn1,liquid=self.liquid)
            self.rnn2= S5(self.nx_rnn2,liquid=self.liquid)  
        elif model_type == 'Mamba':
            self.model_is_mamba = True
            from mamba_ssm import Mamba
            self.rnn1 = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=self.nx_rnn1, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
                )
            self.rnn2 = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=self.nx_rnn2, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
                )
        elif model_type == 'GSS':
            self.model_is_gss = True
            from gated_state_spaces_pytorch import GSS
            self.rnn1= GSS(dim=self.nx_rnn1,dss_kernel_N=self.nh_rnn1,dss_kernel_H=self.nh_rnn1)
            self.rnn2= GSS(dim=self.nx_rnn2,dss_kernel_N=self.nh_rnn2,dss_kernel_H=self.nh_rnn2)
        elif model_type == 'QRNN':
            self.model_is_qrnn = True
            from models_torch_kernels import QRNNLayer, QRNNLayer_noncausal
            kernelsize = 3
            self.rnn1= QRNNLayer_noncausal(self.nx_rnn1,self.nh_rnn1, kernel_size=3, pad =(0,1))
            self.rnn2= QRNNLayer_noncausal(self.nx_rnn2,self.nh_rnn2, kernel_size=3) 
            
            self.mlp_surface2  = nn.Linear(nx_sfc, self.nh_rnn1)  
        elif model_type == 'SRU':
            self.model_is_sru = True
            from models_torch_kernels import SRU
            self.rnn1= SRU(self.nx_rnn1,self.nh_rnn1)
            self.rnn2= SRU(self.nx_rnn2,self.nh_rnn2)  
            # from sru import SRU
            # self.rnn1= SRU(self.nx_rnn1,self.nh_rnn1,num_layers=1)
            # self.rnn2= SRU(self.nx_rnn2,self.nh_rnn2,num_layers=1)
        elif model_type == "SRU_2D":
            self.model_is_sru2d = True
            from models_torch_kernels import SRU, SRU2
            self.rnn1= SRU2(self.nx_rnn1,self.nh_rnn1)
            self.rnn2= SRU(self.nx_rnn2,self.nh_rnn2)  
        elif model_type == 'GateLoop':
            self.model_is_gateloop = True
            from gateloop_transformer import SimpleGateLoopLayer
            self.rnn1= SimpleGateLoopLayer(self.nx_rnn1) #, use_jax_associative_scan=True)
            self.rnn2= SimpleGateLoopLayer(self.nx_rnn2) #, use_jax_associative_scan=True)
            self.mlp_surface2  = nn.Linear(nx_sfc, self.nh_rnn1)
        else:
            
            raise NotImplementedError()

        # if model_type in ['LRU','S5','GSS','QRNN','Mamba']:
        # if model_type in ['LRU','SRU','S5','GSS','QRNN','GateLoop','MinGRU']:
        # if model_type in ['LRU','SRU','S5','GSS','QRNN','GateLoop','MinGRU','Mamba']:
        if model_type in ['LRU','SRU','SRU_2D','S5','QRNN','GateLoop','MinGRU',]:

            self.use_glu_layers = True 
        else:
            self.use_glu_layers = False
            
        self.reduce_dim_with_mlp=False
        if self.use_glu_layers:
            print("Using GLU layers in between for nonlinearity, expand factor (1 means halving output) for GLU1 is ", glu_expand_factor)
            glu_layernorm=True
            # glu_layernorm=False

            # self.rnn1= LRU(in_features=nx_rnn1,out_features=self.nneur[0],state_features=self.nneur[0])
            # self.mlp  = nn.Linear(self.nneur[0], self.nneur[0])
            # self.SeqLayer1 = SequenceLayer(nlev=self.nlev,nneur=self.nneur[0],layernorm=True)
            self.SeqLayer1 = GLU(nseq=self.nlev,nneur=self.nh_rnn1,layernorm=glu_layernorm, expand_factor=glu_expand_factor)
    
            # self.rnn2= LRU(in_features=nx_rnn2,out_features=self.nneur[1],state_features=self.nneur[0])
            # self.mlp2  = nn.Linear(self.nneur[1], self.nneur[1])
            # self.SeqLayer2 = SequenceLayer(nlev=self.nlev,nneur=self.nneur[1],layernorm=True)
            self.SeqLayer2 = GLU(nseq=self.nlev,nneur=self.nh_rnn2,layernorm=glu_layernorm)

            # self.SeqLayer15 = GLU(nlev=self.nlev,nneur=self.nneur[0],layernorm=glu_layernorm)
        else:
            if model_type in ['S5','GSS','QRNN','Mamba','GateLoop','SRU','SRU_2D''LRU']:
                self.reduce_dim_with_mlp=True
                self.reduce_dim_mlp = nn.Linear(self.nh_rnn1, self.nx_rnn2)
                print("Reduce MLP dims", self.nh_rnn1, self.nx_rnn2)
        if self.concat:
            nx_last = self.ny_rnn1 + self.nh_rnn2
        else:
            nx_last = self.nh_rnn2
            
        if self.use_intermediate_mlp:
            print("intermediate MLP dims", nx_last, self.nh_mem)
            self.mlp_latent = nn.Linear(nx_last, self.nh_mem)
            nx_last = self.nh_mem

        self.mlp_output        = nn.Linear(nx_last, self.ny)
        print("MLP surface dims", nneur[-1], self.ny_sfc)
        self.mlp_surface_output = nn.Linear(nneur[-1], self.ny_sfc)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh() 
        
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
    
    # def forward(self, inputs_main, inputs_aux):
        

    def forward(self, inp_list : List[Tensor]):
        inputs_main   = inp_list[0]
        inputs_aux    = inp_list[1]
        rnn1_mem      = inp_list[2]
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device=self.device

        if self.add_pres:
            sp = torch.unsqueeze(inputs_aux[:,0:1],1)
            # undo scaling
            sp = sp*35451.17 + 98623.664
            pres  = self.preslay(sp)
            inputs_main = torch.cat((inputs_main,pres),dim=2)
        
        # if self.model_type=="S5" and self.pres_step_scale:
        if self.model_is_sru and self.pres_step_scale:
            refp = pres

        inputs_main = torch.flip(inputs_main, [1])
        
        if self.model_is_sru and self.pres_step_scale:
            refp_rev = inputs_main[:,:,-1]

        if self.model_is_mamba or self.model_is_gss:#,,"QRNN"]:
            #Mamba doesnt support providing the state, so as a hack we instead
            # concatenate the vertical (sequence) inputs with tiled scalars
            inputs_aux_tiled = torch.tile(torch.unsqueeze(inputs_aux,1),(1,self.nlev,1))
            inputs_main = torch.cat((inputs_main,inputs_aux_tiled), dim=2)
        else:
            init_inputs = inputs_aux
            init_states = self.mlp_surface1(init_inputs)
            # init_states = nn.Softsign()(init_states)
            init_states = self.tanh(init_states)
            # print("shape init states" , init_states.shape)
        # print("shape inp main inp", inputs_main.shape)

        # if self.autoregressive and not self.model_is_sru2d:# "SRU_2D":
        inputs_main = torch.cat((inputs_main,rnn1_mem), dim=2)
        # print("shape inp main inp", inputs_main.shape)

        # if self.use_initial_mlp:
        rnn1_input = self.mlp_initial(inputs_main)
        # print("shape rnn1_input", rnn1_input.shape)

        rnn1_input = self.tanh(rnn1_input)
        # else:
        #     rnn1_input = inputs_main
        # print("shape rnn1 inp", rnn1_input.shape)

        # if self.autoregressive and self.model_type != "SRU_2D":
        #     rnn1_input = torch.cat((rnn1_input,rnn1_mem), axis=2)

        # print("shape rnn1 inp", rnn1_input.shape)
            
        # B_tilde, C_tilde = self.rnn1.seq.get_BC_tilde()
        # print("shape B tild", B_tilde.shape, "C tild", C_tilde.shape)
        # shape B tild torch.Size([96, 96]) C tild torch.Size([96, 96])
        
        # if self.layernorm:
        #     rnn1_input = self.norm(rnn1_input)
        if self.model_is_s5: #"S5":
            if self.pres_step_scale:
                out = self.rnn1(rnn1_input,state=init_states, step_scale=refp_rev)
                # pres_rev = torch.flip(pres, [1])        
                # out = self.rnn1(rnn1_input,state=init_states, step_scale=pres_rev)
            else:
                out = self.rnn1(rnn1_input,state=init_states)
                # out,h = self.rnn1(rnn1_input,state=init_states,return_state=True)
                # print("OUT", out[0,-1,0], "STATE", h[0,0])
                # OUT tensor(-0.2206, device='cuda:0', grad_fn=<SelectBackward0>) STATE tensor(0.0096+0.0025j
        elif self.model_is_mamba or self.model_is_gss:# ["Mamba","GSS"]:#,"QRNN"]:
            out = self.rnn1(rnn1_input) 
        elif self.model_is_qrnn:
            init_states2 = self.mlp_surface2(init_inputs)
            init_states2 = self.tanh(init_states2)
            init_states = (init_states, init_states2)
            out = self.rnn1(rnn1_input,init_states) 

        elif self.model_is_sru:
            # print("init shape", init_states.shape)
            # init_states = init_states.view((1,batch_size,-1)) 
            out,c = self.rnn1(rnn1_input,init_states) 
        elif self.model_is_sru2d:
            # print("init shape", init_states.shape)
            out,c = self.rnn1(rnn1_input,init_states, rnn1_mem) 
        elif self.model_is_mingru:
            init_states = init_states.view((batch_size,1, -1)) 
            out = self.rnn1(rnn1_input,init_states)      
        elif self.model_is_gateloop:
            init_states2 = self.mlp_surface2(init_inputs)
            init_states2 = self.tanh(init_states2)
            cache = [init_states.view(batch_size*self.nh_rnn1,1), init_states2.view(batch_size*self.nh_rnn1,1)]
            out = self.rnn1(rnn1_input,cache=cache)          
            # out = self.rnn1(rnn1_input)          
        else:
            # out = self.rnn1(rnn1_input,state=init_states) 
            out = self.rnn1(rnn1_input,init_states)     
            
        init_states2 = None 
            
        out = torch.flip(out, [1])
        # print("shape rnn1 out", out.shape)

        if self.use_glu_layers: 
            out = self.SeqLayer1(out)
            # out = self.SeqLayer15(out)
        elif self.reduce_dim_with_mlp:
            out = self.reduce_dim_mlp(out)

        rnn2_input = out  
        # print("shape rnn2 inp", rnn2_input.shape)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        if self.model_is_s5:
            if self.pres_step_scale:
                out2 = self.rnn2(rnn2_input, state=init_states2,step_scale=refp)
            else:
                out2 = self.rnn2(rnn2_input, state=init_states2)
        elif self.model_is_mamba or self.model_is_gss:#,,"QRNN"]:
            out2 = self.rnn2(rnn2_input)    
        elif self.model_is_sru or self.model_is_sru2d: # in ["SRU","SRU_2D"]:
            out2,c = self.rnn2(rnn2_input,init_states2) 
        elif self.model_is_mingru:#=="MinGRU":
            init_states2 = torch.randn((batch_size, 1, self.nh_rnn2),device=device)
            out2 = self.rnn2(rnn2_input,init_states2)   
        elif self.model_is_gateloop:# =="GateLoop":
            out2 = self.rnn2(rnn2_input)      
        else:
            out2 = self.rnn2(rnn2_input,init_states2)
        # if self.layernorm:
        #     out2 = self.normrnn2(out2)                                                                                                                                                                                                                                                                                                                            
        # self.state2 = self._detach_state(state2)
        #self.state = state
        
        if self.use_glu_layers: 
            out2 = self.SeqLayer2(out2)

        if self.concat:
            outs  = torch.cat((out,out2),dim=2)
        else:
            outs = out2
            
        out_sfc = self.mlp_surface_output(outs[:,-1])

            
        if self.use_intermediate_mlp:
            outs = self.mlp_latent(outs)
            
        # if self.autoregressive:
        
        rnn1_mem = torch.flip(outs, [1])


        # out_sfc = self.mlp_surface_output(outs[:,-1])
        out_sfc = self.relu(out_sfc)
        
        outs = self.mlp_output(outs)

        return outs, out_sfc, rnn1_mem
        

class LiquidNN_autoreg_torchscript(nn.Module):
    use_initial_mlp: Final[bool]
    use_intermediate_mlp: Final[bool]
    add_pres: Final[bool]
    add_stochastic_layer: Final[bool]
    output_prune: Final[bool]
    separate_radiation: Final[bool]
    use_third_rnn: Final[bool]
    diagnose_precip: Final[bool]
    concat: Final[bool]
    mixed_memory: Final[bool]

    def __init__(self, hyam, hybm,  hyai, hybi,
                out_scale, out_sfc_scale, 
                xmean_lev, xmean_sca, xdiv_lev, xdiv_sca,
                device,
                nlev=60, nx = 15, nx_sfc=24, ny = 5, ny0=5, ny_sfc=5, nneur=(192,192), 
                nout_cfc=32,
                use_initial_mlp=False, 
                use_intermediate_mlp=True,
                add_pres=False,
                add_stochastic_layer=False,
                output_prune=False,
                repeat_mu=False,
                concat=False,
                coeff_stochastic = 0.0,
                nh_mem=16):
        super(LiquidNN_autoreg_torchscript, self).__init__()
        # from ncps.torch import CfC
        # from ncps.wirings import AutoNCP
        from ncp import CfC
        from ncp import AutoNCP
        self.ny = ny 
        self.ny0 = ny  #for diagnose precip option, need to distinguish between model outputs (ny) and intermediate outputs (ny0)
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
        self.nx = nx
        self.nh_rnn1 = self.nneur[0]
        self.nx_rnn2 = self.nneur[0]
        self.nh_rnn2 = self.nneur[1]
        self.nout_cfc = nout_cfc
        self.concat=concat
        self.repeat_mu = repeat_mu
        if self.repeat_mu:
            nx = nx + 1

        if self.use_initial_mlp:
            self.nx_rnn1 = self.nneur[0]
        else:
            self.nx_rnn1 = nx
                
        self.add_stochastic_layer = add_stochastic_layer
        self.coeff_stochastic = coeff_stochastic
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
            
        if self.use_intermediate_mlp:
            # self.nh_mem = self.nneur[1] // 4
            # if nh_mem is None:
            #     self.nh_mem = self.nneur[1] // 8
            # else:
            self.nh_mem = nh_mem
        else:
            self.nh_mem = self.nneur[1]
        # self.rnn1_mem = None 
        print("Building RNN that feeds its hidden memory at t0,z0 to its inputs at t1,z0")
        print("Initial mlp: {}, intermediate mlp: {}".format(self.use_initial_mlp, self.use_intermediate_mlp))
 
        self.nx_rnn1 = self.nx_rnn1 + self.nh_mem
        self.nh_rnn1 = self.nneur[0]
        
        print("nx rnn1", self.nx_rnn1, "nh rnn1", self.nh_rnn1)
        print("nx rnn2", self.nx_rnn2, "nh rnn2", self.nh_rnn2)  
        print("nx sfc", self.nx_sfc)

        if self.use_initial_mlp:
            self.mlp_initial = nn.Linear(nx, self.nneur[0])
            self.mlp_initial2 = nn.Linear(self.nneur[0], self.nneur[0])
            self.mlp_initial3 = nn.Linear(self.nneur[0], self.nneur[0])

        self.mixed_memory = True

        self.mlp_surface1  = nn.Linear(self.nx_sfc, self.nh_rnn1)
        if self.mixed_memory: self.mlp_surface2  = nn.Linear(self.nx_sfc, self.nh_rnn1)

        self.mlp_toa1  = nn.Linear(2, self.nh_rnn2)
        if self.mixed_memory: self.mlp_toa2  = nn.Linear(2, self.nh_rnn2)
        
        
        wiring1 = AutoNCP(self.nh_rnn1, self.nout_cfc) 
        wiring2 = AutoNCP(self.nh_rnn2, self.nout_cfc)
        self.rnn1      = CfC(self.nx_rnn1, wiring1,  batch_first=False, mixed_memory=self.mixed_memory)  # (input_size, hidden_size)
        if self.add_stochastic_layer:
            use_bias=False
            self.rnn2 = MyStochasticLSTMLayer2(self.nx_rnn2, self.nh_rnn2, use_bias=use_bias)  
        else:
            self.rnn2      = CfC(self.nout_cfc, wiring2,  batch_first=False, mixed_memory=self.mixed_memory)  # (input_size, hidden_size)

        if self.concat: 
            nh_rnn = self.nout_cfc + self.nout_cfc
        else:
            nh_rnn = self.nout_cfc

        if self.use_intermediate_mlp: 
            self.mlp_latent = nn.Linear(nh_rnn, self.nh_mem)
            self.mlp_output = nn.Linear(self.nh_mem, self.ny0)
        else:
            self.mlp_output = nn.Linear(nh_rnn, self.ny0)
            
        self.mlp_surface_output = nn.Linear(nneur[-1], self.ny_sfc0)
            
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
    
    # def forward(self, inputs_main, inputs_aux, rnn1_mem):
    def forward(self, inp_list : List[Tensor]):
        inputs_main   = inp_list[0]
        inputs_aux    = inp_list[1]
        rnn1_mem      = inp_list[2]
                    
        batch_size = inputs_main.shape[1]
        if self.add_pres:
            sp = torch.unsqueeze(inputs_aux[:,0:1],1)
            # undo scaling
            sp = sp*35451.17 + 98623.664
            pres  = self.preslay(sp)
            inputs_main = torch.cat((inputs_main,pres),dim=2)

        if self.repeat_mu:
            mu = torch.reshape(inputs_aux[:,6:7],(-1,1,1))
            mu_rep = torch.repeat_interleave(mu,self.nlev,dim=1)
            inputs_main = torch.cat((inputs_main,mu_rep),dim=2)
            
        inputs_main_crm = inputs_main

        if self.use_initial_mlp:
            inputs_main_crm = self.mlp_initial(inputs_main_crm)
            inputs_main_crm = self.nonlin(inputs_main_crm)  
            inputs_main_crm = self.mlp_initial2(inputs_main_crm)
            inputs_main_crm = self.nonlin(inputs_main_crm) 
            inputs_main_crm = self.mlp_initial3(inputs_main_crm)
            inputs_main_crm = self.nonlin(inputs_main_crm)       
        
        inputs_main_crm = torch.cat((inputs_main_crm,rnn1_mem), dim=2)
        inputs_main_crm = torch.transpose(inputs_main_crm,0,1)

        # TOA is first in memory, so to start at the surface we need to go backwards
        rnn1_input = torch.flip(inputs_main_crm, [0])
    
        # The input (a vertical sequence) is concatenated with the
        # output of the RNN from the previous time step 
        inputs_sfc = inputs_aux
        hx = self.mlp_surface1(inputs_sfc)
        hx = self.nonlin(hx)
        if self.mixed_memory:
            cx = self.mlp_surface2(inputs_sfc)
            # cx = self.nonlin(cx)
            hidden = (hx, cx)
        else:
            hidden = hx
        rnn1out, states = self.rnn1(rnn1_input, hidden)
        
        rnn1out = torch.flip(rnn1out, [0])

        inputs_toa = torch.cat((inputs_aux[:,1:2], inputs_aux[:,6:7]),dim=1) 
        hx2 = self.mlp_toa1(inputs_toa)
      
        if self.add_stochastic_layer or self.mixed_memory:
            cx2 = self.mlp_toa2(inputs_toa)
            hidden2 = (hx2, cx2)
        else:
            hidden2 = hx2

        # if self.add_stochastic_layer:
        #     input_rnn2 = torch.transpose(rnn1out,0,1)
        # else:
        #     input_rnn2 = rnn1out
        input_rnn2 = rnn1out

        rnn2out, states = self.rnn2(input_rnn2, hidden2)

        if self.mixed_memory:
            (last_h, last_c) = states
            final_sfc_inp = last_h.squeeze() 
        else:
            final_sfc_inp = states
            
        if self.concat:
            rnn2out = torch.cat((rnn1out, rnn2out), dim=2)
        
        if self.use_intermediate_mlp: 
            rnn2out = self.mlp_latent(rnn2out)
          
        rnn2out = torch.transpose(rnn2out,0,1)
        rnn1_mem = rnn2out 
        
        out = self.mlp_output(rnn2out)

        if self.output_prune:
            out[:,0:12,1:] = out[:,0:12,1:].clone().zero_()

        out_sfc = self.mlp_surface_output(final_sfc_inp)
        
        out_sfc = self.relu(out_sfc)

        return out, out_sfc, rnn1_mem