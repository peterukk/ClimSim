#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytorch model constructors for hybrid models incorporating radiative transfer equations / domain knowledge

"""
import os 
import gc
import sys
import torch
import torch.nn as nn
import torch.nn.parameter as Parameter
from layers import LayerPressure, LayerPressureThickness, LevelPressure
import torch.nn.functional as F
from typing import List, Tuple, Final, Optional
from torch import Tensor
from models_torch_kernels import GLU
from models_torch_kernels import *
import numpy as np 
from typing import Final 
import time 

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

class LSTM_autoreg_torchscript_physrad(nn.Module):
    use_initial_mlp: Final[bool]
    use_intermediate_mlp: Final[bool]
    add_pres: Final[bool]
    add_stochastic_layer: Final[bool]
    output_prune: Final[bool]
    # ensemble_size: Final[int]
    use_ensemble: Final[bool]
    # predict_flux: Final[bool]
    use_third_rnn: Final[bool]
    concat: Final[bool]
    mp_constraint: Final[bool]
    def __init__(self, hyam, hybm,  hyai, hybi,
                out_scale, out_sfc_scale, 
                xmean_lev, xmean_sca, xdiv_lev, xdiv_sca,
                device,
                nlev=60, nx = 15, nx_sfc=24, ny = 5, ny0=5, ny_sfc=5, nneur=(192,192), 
                use_initial_mlp=False, 
                use_intermediate_mlp=True,
                add_pres=False,
                add_stochastic_layer=False,
                output_prune=False,
                repeat_mu=False,
                use_third_rnn=False,
                use_ensemble=False,
                concat=False,
                # predict_flux=False,
                # ensemble_size=1,
                coeff_stochastic = 0.0,
                nh_mem=16,
                mp_mode=0):
        super(LSTM_autoreg_torchscript_physrad, self).__init__()
        self.ny = ny 
        self.nlev = nlev 
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
        self.preslay_nonorm = LayerPressure(hyam,hybm,name="LayerPressure_nonorm", norm=False)
        self.presdelta = LayerPressureThickness(hyai,hybi)
        self.preslev = LevelPressure(hyai,hybi)
        self.nx = nx
        self.nh_rnn1 = self.nneur[0]
        self.nx_rnn2 = self.nneur[0]
        self.nh_rnn2 = self.nneur[1]
        self.use_third_rnn = use_third_rnn
        if self.use_third_rnn:
            self.nx_rnn3 = self.nneur[1]
            self.nh_rnn3 = self.nneur[2]
        self.concat=concat
        self.repeat_mu = repeat_mu
        if self.repeat_mu:
            nx = nx + 1
        self.use_ensemble = use_ensemble
        if mp_mode==0:
          self.mp_constraint=False 
        elif mp_mode==1:
          self.mp_constraint=True
        else:
          raise NotImplementedError("model requires mp_mode>=0")
        self.nlev = 60
        self.nlev_crm = 50
        self.nlev_mem = self.nlev_crm
        self.nx_rad = 8
        nx = nx - 3
        self.nx_sfc_tot = self.nx_sfc
        self.nx_sfc_rad = 5
        self.nx_sfc = self.nx_sfc  - self.nx_sfc_rad
        self.ny_rad = 1
        self.ny_sfc_rad = self.ny_sfc0 - 2
        self.ny_sfc0 = 2
        
        if self.use_initial_mlp:
            self.nx_rnn1 = self.nneur[0]
        else:
            self.nx_rnn1 = nx
                
        # self.ensemble_size = ensemble_size
        self.add_stochastic_layer = add_stochastic_layer
        self.coeff_stochastic = coeff_stochastic
        self.nonlin = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)
        self.softmax_dim1 = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        yscale_lev = torch.from_numpy(out_scale).to(device)
        yscale_sca = torch.from_numpy(out_sfc_scale).to(device)
        yscale_sca_rad = yscale_sca[[0,1,4,5,6,7]]
        xmean_lev  = torch.from_numpy(xmean_lev).to(device)
        xmean_sca  = torch.from_numpy(xmean_sca).to(device)
        xdiv_lev   = torch.from_numpy(xdiv_lev).to(device)
        xdiv_sca   = torch.from_numpy(xdiv_sca).to(device)
        
        self.register_buffer('yscale_lev', yscale_lev)
        self.register_buffer('yscale_sca', yscale_sca)
        self.register_buffer('yscale_sca_rad', yscale_sca_rad)
        self.register_buffer('xmean_lev', xmean_lev)
        self.register_buffer('xmean_sca', xmean_sca)
        self.register_buffer('xdiv_lev', xdiv_lev)
        self.register_buffer('xdiv_sca', xdiv_sca)
        self.register_buffer('hyai', hyai)
        self.register_buffer('hybi', hybi)
        
        self.lbd_qc     = torch.tensor(lbd_qc, dtype=torch.float32, device=device)
        self.lbd_qi     = torch.tensor(lbd_qi, dtype=torch.float32, device=device)
        # in ClimSim config of E3SM, the CRM physics first computes 
        # moist physics on 50 levels, and then computes radiation on 60 levels!
        self.use_intermediate_mlp=use_intermediate_mlp
            
        if self.use_intermediate_mlp:
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
        if self.use_third_rnn: print("nx rnn3", self.nx_rnn3, "nh rnn3", self.nh_rnn3) 
        print("nx sfc", self.nx_sfc)

        if self.use_initial_mlp:
            self.mlp_initial = nn.Linear(nx, self.nneur[0])

        self.mlp_surface1  = nn.Linear(self.nx_sfc, self.nh_rnn1)
        self.mlp_surface2  = nn.Linear(self.nx_sfc, self.nh_rnn1)

        if self.use_third_rnn:
            self.mlp_toa1  = nn.Linear(2, self.nh_rnn1)
            self.mlp_toa2  = nn.Linear(2, self.nh_rnn1)
            
            self.rnn0   = nn.LSTM(self.nx_rnn1, self.nh_rnn1,  batch_first=True)
            self.rnn1   = nn.LSTM(self.nx_rnn2, self.nh_rnn2,  batch_first=True)  # (input_size, hidden_size)
            if self.add_stochastic_layer:
                use_bias=False
                self.rnn2 = MyStochasticLSTMLayer2(self.nx_rnn3, self.nh_rnn3, use_bias=use_bias)  
            else:
                self.rnn2   = nn.LSTM(self.nx_rnn3, self.nh_rnn3,  batch_first=True)
            self.rnn0.flatten_parameters()
        else:

            self.mlp_toa1  = nn.Linear(2, self.nh_rnn2)
            self.mlp_toa2  = nn.Linear(2, self.nh_rnn2)

            self.rnn1      = nn.LSTM(self.nx_rnn1, self.nh_rnn1,  batch_first=True)  # (input_size, hidden_size)
            if self.add_stochastic_layer:
                use_bias=False
                self.rnn2 = MyStochasticLSTMLayer2(self.nx_rnn2, self.nh_rnn2, use_bias=use_bias)  
            else:
                self.rnn2 = nn.LSTM(self.nx_rnn2, self.nh_rnn2,  batch_first=True)
                
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()

        if self.concat: 
            nh_rnn = self.nh_rnn1 + self.nh_rnn2
        else:
            nh_rnn = self.nh_rnn2

        if self.use_intermediate_mlp: 
            self.mlp_latent = nn.Linear(nh_rnn, self.nh_mem)
            self.mlp_output = nn.Linear(self.nh_mem, self.ny)
        else:
            self.mlp_output = nn.Linear(nh_rnn, self.ny)
            
        self.mlp_surface_output = nn.Linear(nneur[-1], self.ny_sfc0)
            
        self.ng_sw              = 16
        self.ng_lw              = 16
        self.ng_tot             = self.ng_sw + self.ng_lw
        
        self.ny_optprops_sw     = 3 # tau_abs, tau_sca, g
        self.nx_optprops_sw     = 9 
        # T, p, clw, cli, H2O, O3, CH4, N2O, +  CLOUD FRAC / HIDDEN VARIABLE to account for cloud heterogeneity
        self.ny_optprops_lw     = 2  # tau_abs, planck_frac
        # not g or lw because aerosols were transparent and LW cloud scattering is ignored in original runs
        self.nx_optprops_lw     = 9
        self.mlp_optprops_sw    = nn.Linear(self.nx_optprops_sw, self.ny_optprops_sw*self.ng_sw)
        self.mlp_optprops_lw    = nn.Linear(self.nx_optprops_sw, self.ny_optprops_lw*self.ng_lw)

        # self.ny_gasopt_sw       = self.ng_sw*2 # tau_abs, tau_ray
        # self.ny_gasopt_lw       = self.ng_lw*2 # tau_abs, planck_frac
        # self.nx_gasopt_sw       = 6 # T, p, H2O, O3, CH4, N2O
        # self.nx_gasopt_lw       = 6
        # self.mlp_gasopt_sw      = nn.Linear(self.nx_gasopt_sw, self.ny_gasopt_sw)
        # self.mlp_gasopt_lw      = nn.Linear(self.nx_gasopt_lw, self.ny_gasopt_lw)

        # self.ny_cloudopt_sw     = self.ng_sw*3 # tau_tot, ssa, g
        # self.ny_cloudopt_lw     = self.ng_sw*1 # assuming no scattering: only tau
        # self.mlp_cloudopt_sw      = nn.Linear(2, self.ny_cloudopt_sw) # inputs: T, clw (scale tau by cloud water path?)
        # self.mlp_cloudopt_lw      = nn.Linear(2, self.ny_cloudopt_lw) # inputs: T, cli (scale tau by cloud ice path?)
        
        self.ny_reftra_sw       = 4 # ref_diff, trans_diff, ref_dir, trans_dir_diff
        self.ny_reftra_lw       = 4 # reflectance, transmittance, source_up, source_dn
        self.mlp_reftra_sw      = nn.Linear(3, self.ny_reftra_sw)
        self.mlp_reftra_lw      = nn.Linear(3, self.ny_reftra_lw)
        
        self.mlp_sfc_albedo_lw  = nn.Linear(self.nx_sfc_tot, self.ng_lw)
        self.sw_solar_weights   = nn.Parameter(torch.randn(1, self.ng_sw))

    def temperature_scaling(self, T_raw):
        liquid_ratio = (T_raw - 253.16) * 0.05 
        liquid_ratio = F.hardtanh(liquid_ratio, 0.0, 1.0)
        return liquid_ratio

    def postprocessing(self, out, out_sfc):
        out             = out / self.yscale_lev
        out_sfc         = out_sfc / self.yscale_sca
        return out, out_sfc
        
    
    def outgoing_lw(self, temp):
        # Stefan-Boltzmann constant (W/m²/K⁴)
        sigma = 5.670374419e-8
        
        # Assuming emissivity = 1 (blackbody approximation)
        olr_exact = sigma * temp**4
        return olr_exact
    
    def interpolate_tlev(self, tlay, play, plev):
        nlay, ncol = tlay.shape
        device = tlay.device
        dtype = tlay.dtype
        # Initialize output arrays
        tlev = torch.zeros(nlay + 1, ncol, dtype=dtype, device=device)
        
        tlev[0,:] = tlay[0,:] + (plev[0,:]-play[0,:])*(tlay[1,:]-tlay[0,:]) / (play[1,:]-play[0,:])
        for ilay in range(1, nlay-1):
          tlev[ilay,:] = (play[ilay-1,:]*tlay[ilay-1,:]*(plev[ilay,:]-play[ilay,:]) \
                + play[ilay,:]*tlay[ilay,:]*(play[ilay-1,:]-plev[ilay,:])) /  (plev[ilay,:]*(play[ilay-1,:] - play[ilay,:]))
                                  
        tlev[nlay,:] = tlay[nlay-1,:] + (plev[nlay,:]-play[nlay-1,:])*(tlay[nlay-1,:]-tlay[nlay-2,:])  \
                / (play[nlay-1,:]-play[nlay-2,:])
                                 
        return tlev
    
    def interpolate_tlev_batchfirst(self, tlay, play, plev):
        ncol, nlay = tlay.shape
        device = tlay.device
        dtype = tlay.dtype
        # Initialize output arrays
        tlev = torch.zeros(ncol, nlay + 1, dtype=dtype, device=device)
        
        tlev[:,0] = tlay[:,0] + (plev[:,0]-play[:,0])*(tlay[:,1]-tlay[:,0]) / (play[:,1]-play[:,0])
        for ilay in range(1, nlay-1):
          tlev[:,ilay] = (play[:,ilay-1]*tlay[:,ilay-1]*(plev[:,ilay]-play[:,ilay]) \
                + play[:,ilay]*tlay[:,ilay]*(play[:,ilay-1]-plev[:,ilay])) /  (plev[:,ilay]*(play[:,ilay-1] - play[:,ilay]))
                                  
        tlev[:,nlay] = tlay[:,nlay-1] + (plev[:,nlay]-play[:,nlay-1])*(tlay[:,nlay-1]-tlay[:,nlay-2])  \
                / (play[:,nlay-1]-play[:,nlay-2])
                                 
        return tlev
            
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

    @torch.compile
    def reftrans_lw(self, source_lev, tau_lw):
        planck_top = source_lev[:,:,0:-1]
        planck_bot = source_lev[:,:,1:]
        
        od_loc = 1.66 * tau_lw
        trans_lw = torch.exp(-od_loc)
        # Calculate coefficient for Padé approximant (vectorized)
        coeff = 0.2 * od_loc
        # Calculate mean Planck function (vectorized)
        planck_fl = 0.5 * (planck_top + planck_bot)
        # Calculate source terms using Padé approximant (vectorized)
        one_minus_trans = 1.0 - trans_lw
        one_plus_coeff = 1.0 + coeff
        source_dn = one_minus_trans * (planck_fl + coeff * planck_bot) / one_plus_coeff
        source_up = one_minus_trans * (planck_fl + coeff * planck_top) / one_plus_coeff
        return source_up, source_dn, trans_lw
    
    @torch.compile
    def calc_reflectance_transmittance_sw(self, mu0, od, ssa, asymmetry ):
        """
        Calculate reflectance and transmittance for shortwave radiation using two-stream approximation.
        
        Args:
            mu0: Cosine of solar zenith angle (scalar)
            od: Optical depth (tensor of shape [ng])
            ssa: Single scattering albedo (tensor of shape [ng])
            asymmetry: asymmetry factor (tensor of shape [ng])
        
        Returns:
            tuple: (ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir)
        """
      
        zeroes =  torch.zeros_like(od)

        factor = 0.75*asymmetry
    
        gamma1 = 2 - ssa * (1.25 + factor)
        gamma2 = ssa * (0.75 - factor)
        gamma3 = 0.5 - mu0*factor
        # Calculate gamma4
        gamma4 = 1.0 - gamma3
        
        # Calculate alpha1 and alpha2 (Equations 16 and 17)
        alpha1 = gamma1 * gamma4 + gamma2 * gamma3
        alpha2 = gamma1 * gamma3 + gamma2 * gamma4
        
        # Calculate k_exponent (Equation 18)
        k_exponent = torch.sqrt(torch.clamp((gamma1 - gamma2) * (gamma1 + gamma2), min=1e-4))
        
        # Calculate various terms
        k_mu0 = k_exponent * mu0
        k_gamma3 = k_exponent * gamma3
        k_gamma4 = k_exponent * gamma4
        
        # Calculate optical depth over mu0
        od_over_mu0 = torch.clamp(od / mu0, min=0.0)
        od_over_mu0 = torch.clamp(-od_over_mu0, min=-50.0)
    
        # Calculate exponential terms
        trans_dir_dir = torch.exp(od_over_mu0)
        exponential = torch.exp(-k_exponent * od)
        exponential2 = exponential * exponential
        k_2_exponential = 2.0 * k_exponent * exponential
        
        # Calculate reftrans_factor for diffuse terms
        reftrans_factor = 1.0 / (k_exponent + gamma1 + (k_exponent - gamma1) * exponential2)
        # assert not torch.isnan(reftrans_factor).any()

        # Meador & Weaver (1980) Eq. 25
        ref_diff = gamma2 * (1.0 - exponential2) * reftrans_factor
        
        # Meador & Weaver (1980) Eq. 26
        trans_diff = k_2_exponential * reftrans_factor
        
        # Calculate reftrans_factor for direct terms
        one_minus_kmu0_sqr = (1.0 - k_mu0 * k_mu0)
        one_minus_kmu0_sqr = torch.clamp(one_minus_kmu0_sqr, min=1e-6) 
        # print("one_minus_kmu0_sqr  min max", torch.min(one_minus_kmu0_sqr).item(), torch.max(one_minus_kmu0_sqr).item())
        # reftrans_factor_dir = mu0 * ssa * reftrans_factor / one_minus_kmu0_sqr
        reftrans_factor_dir =  ssa * reftrans_factor / one_minus_kmu0_sqr

        # print("reftrans_factor_dir  min max", torch.min(reftrans_factor_dir).item(), torch.max(reftrans_factor_dir).item())
        # assert not torch.isnan(reftrans_factor_dir).any()

        # Meador & Weaver (1980) Eq. 14
        ref_dir = reftrans_factor_dir * (
            (1.0 - k_mu0) * (alpha2 + k_gamma3) -
            (1.0 + k_mu0) * (alpha2 - k_gamma3) * exponential2 -
            k_2_exponential * (gamma3 - alpha2 * mu0) * trans_dir_dir
        )
        
        # Meador & Weaver (1980) Eq. 15
        trans_dir_diff = reftrans_factor_dir * (
            k_2_exponential * (gamma4 + alpha1 * mu0) -
            trans_dir_dir * (
                (1.0 + k_mu0) * (alpha1 + k_gamma4) -
                (1.0 - k_mu0) * (alpha1 - k_gamma4) * exponential2
            )
        )
        # assert not torch.isnan(trans_dir_dir).any()
        # assert not torch.isnan(ref_diff).any()
        # assert not torch.isnan(trans_diff).any()
        # assert not torch.isnan(ref_dir).any()
        # assert not torch.isnan(trans_dir_diff).any()

        # Final bounds checking
        ref_diff = torch.clamp(ref_diff, min=0.0, max=1.0)
        trans_diff = torch.clamp(trans_diff, min=0.0, max=1.0)
    
        # ref_dir = torch.clamp(ref_dir, min=zeroes, max=mu0*(1.0 - trans_dir_dir))
        # trans_dir_diff = torch.clamp(trans_dir_diff, min=zeroes, max=mu0*(1.0 - trans_dir_dir - ref_dir))
        ref_dir = torch.clamp(ref_dir, min=zeroes, max=(1.0 - trans_dir_dir))
        trans_dir_diff = torch.clamp(trans_dir_diff, min=zeroes, max=(1.0 - trans_dir_dir - ref_dir))
        
        return ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir
    
    def calc_noscat_trans_lw_pade(self, od, planck_top, planck_bot, LwDiffusivity=1.66):
        """
        Calculate longwave transmittance and source terms using Padé approximant method.
        
        This function implements the alternative source computation using a Padé approximant
        for the linear-in-tau solution, following Clough et al. (1992), doi:10.1029/92JD01419, Eq 15.
        This method requires no conditional statements but introduces some approximation error.
        
        Args:
            od (torch.Tensor): Optical depth [ng]
            planck_top (torch.Tensor): Planck function at layer top [ng]
            planck_bot (torch.Tensor): Planck function at layer bottom [ng]
            LwDiffusivity (float): Longwave diffusivity factor (default 1.66)
        
        Returns:
            tuple: (transmittance, source_up, source_dn)
                - transmittance (torch.Tensor): Diffuse transmittance [ng]
                - source_up (torch.Tensor): Upward emission at layer top [ng]
                - source_dn (torch.Tensor): Downward emission at layer bottom [ng]
        """
            
        # Calculate local optical depth and transmittance using Beer's law
        od_loc = LwDiffusivity * od
        transmittance = torch.exp(-od_loc)
        
        # Calculate coefficient for Padé approximant (vectorized)
        coeff = 0.2 * od_loc
        
        # Calculate mean Planck function (vectorized)
        planck_fl = 0.5 * (planck_top + planck_bot)
        
        # Calculate source terms using Padé approximant (vectorized)
        one_minus_trans = 1.0 - transmittance
        one_plus_coeff = 1.0 + coeff
        
        source_dn = one_minus_trans * (planck_fl + coeff * planck_bot) / one_plus_coeff
        source_up = one_minus_trans * (planck_fl + coeff * planck_top) / one_plus_coeff
        
        return transmittance, source_up, source_dn
    
    @torch.compile
    def adding_ica_sw_batchfirst(self, incoming_toa, albedo_surf_diffuse, albedo_surf_direct, cos_sza,
                    reflectance, transmittance, ref_dir, trans_dir_diff, trans_dir_dir):
            """
            Adding method for shortwave radiation with Independent Column Approximation (ICA).
            
            Args:
                incoming_toa: Incoming downwelling solar radiation at TOA (tensor of shape [ncol])
                albedo_surf_diffuse: Surface albedo to diffuse radiation (tensor of shape [ncol])
                albedo_surf_direct: Surface albedo to direct radiation (tensor of shape [ncol])
                cos_sza: Cosine of solar zenith angle (tensor of shape [ncol])
                reflectance: Diffuse reflectance of each layer (tensor of shape [ncol, nlev])
                transmittance: Diffuse transmittance of each layer (tensor of shape [ncol, nlev])
                ref_dir: Direct beam reflectance of each layer (tensor of shape [ncol, nlev])
                trans_dir_diff: Direct beam to diffuse transmittance (tensor of shape [ncol, nlev])
                trans_dir_dir: Direct transmittance of each layer (tensor of shape [ncol, nlev])
            
            Returns:
                tuple: (flux_up, flux_dn_diffuse, flux_dn_direct) each of shape [ncol, nlev+1]
            """
            
            ncol, nlev = reflectance.shape
            device = reflectance.device
            dtype = reflectance.dtype
            
            # Initialize output arrays
            flux_up = torch.zeros(ncol, nlev + 1, dtype=dtype, device=device)
            flux_dn_diffuse = torch.zeros(ncol, nlev + 1, dtype=dtype, device=device)
            flux_dn_direct = torch.zeros(ncol, nlev + 1, dtype=dtype, device=device)
            
            # Initialize working arrays
            albedo = torch.zeros(ncol, nlev + 1, dtype=dtype, device=device)
            source = torch.zeros(ncol, nlev + 1, dtype=dtype, device=device)
            inv_denominator = torch.zeros(ncol, nlev, dtype=dtype, device=device)
            
            # Compute profile of direct (unscattered) solar fluxes at each half-level
            # by working down through the atmosphere
            flux_dn_direct[:, 0] = incoming_toa
            for jlev in range(nlev):
                flux_dn_direct[:, jlev + 1] = flux_dn_direct[:, jlev].clone() * trans_dir_dir[:, jlev]
            
            # Set surface albedo
            albedo[:, nlev] = albedo_surf_diffuse
            
            # At the surface, the direct solar beam is reflected back into the diffuse stream
            source[:, nlev] = albedo_surf_direct * flux_dn_direct[:, nlev] #* cos_sza
            
            # Work back up through the atmosphere and compute the albedo of the entire
            # earth/atmosphere system below that half-level, and also the "source"
            for jlev in range(nlev - 1, -1, -1):  # nlev down to 1 in Fortran indexing
                # Lacis and Hansen (1974) Eq 33, Shonk & Hogan (2008) Eq 10:
                # inv_denominator[:, jlev] = 1.0 / (1.0 - albedo[:, jlev + 1].clone() * reflectance[:, jlev])
                
                albedoplusone = albedo[:, jlev + 1].clone()
                inv_denominator[:, jlev] = 1.0 / (1.0 - albedoplusone * reflectance[:, jlev])
    
                # Shonk & Hogan (2008) Eq 9, Petty (2006) Eq 13.81:
                # albedo[:, jlev] = (reflectance[:, jlev] + 
                #                   transmittance[:, jlev] * transmittance[:, jlev] * 
                #                   albedo[:, jlev + 1] * inv_denominator[:, jlev])
                
                inv_denom = inv_denominator[:, jlev].clone()
                albedo[:, jlev] = (reflectance[:, jlev] + 
                                  torch.square(transmittance[:, jlev]) * 
                                  albedoplusone * inv_denom)

                # TripleClouds :
                # ng_nreg = self.nreg//3 
                # A[0:ng_nreg]           = A[0:ng_nreg]*V[1,1] + A[ng_nreg:2*ng_nreg]*V[2,1] + A[2*ng_nreg:3*ng_nreg]*V[3,1]
                # A[ng_nreg:2*ng_nreg]   = A[0:ng_nreg]*V[1,2] + A[ng_nreg:2*ng_nreg]*V[2,2] + A[2*ng_nreg:3*ng_nreg]*V[3,2]
                # A[2*ng_nreg:3*ng_nreg] = A[0:ng_nreg]*V[1,3] + A[ng_nreg:2*ng_nreg]*V[2,3] + A[2*ng_nreg:3*ng_nreg]*V[3,3]         
                                
                # Shonk & Hogan (2008) Eq 11:
                fluxdndir = flux_dn_direct[:, jlev].clone()
                source[:, jlev] = (ref_dir[:, jlev] * fluxdndir +
                                  transmittance[:, jlev] * 
                                  (source[:, jlev + 1] + 
                                  albedoplusone * trans_dir_diff[:, jlev] * fluxdndir) *
                                  inv_denom)
            
            # At top-of-atmosphere there is no diffuse downwelling radiation
            flux_dn_diffuse[:, 0] = 0.0
            
            # At top-of-atmosphere, all upwelling radiation is due to scattering
            # by the direct beam below that level
            flux_up[:, 0] = source[:, 0]
            
            # Work back down through the atmosphere computing the fluxes at each half-level
            for jlev in range(nlev):  # 1 to nlev in Fortran indexing
                # Shonk & Hogan (2008) Eq 14 (after simplification):
                fluxdndir = flux_dn_direct[:, jlev].clone()
                fluxdndiff = flux_dn_diffuse[:, jlev].clone()
    
                flux_dn_diffuse[:, jlev + 1] = ((transmittance[:, jlev] * fluxdndiff +  
                                                 reflectance[:, jlev] * source[:, jlev + 1] + 
                                                 trans_dir_diff[:, jlev] * fluxdndir) *
                                               inv_denominator[:, jlev])      
          
                # Shonk & Hogan (2008) Eq 12:
                flux_up[:, jlev + 1] = (albedo[:, jlev + 1] * flux_dn_diffuse[:, jlev + 1].clone() +
                                       source[:, jlev + 1])
                
                # Apply cosine correction to direct flux
                # flux_dn_direct[:, jlev] = fluxdndir * cos_sza
            
            # Final cosine correction for surface direct flux
            # flux_dn_direct[:, nlev] = flux_dn_direct[:, nlev] * cos_sza
            
            return flux_up, flux_dn_diffuse, flux_dn_direct
   
    @torch.compile
    def adding_tc_sw_batchfirst(self, incoming_toa, albedo_surf_diffuse, albedo_surf_direct, cos_sza,
                    reflectance, transmittance, ref_dir, trans_dir_diff, trans_dir_dir, V):
            """
            Adding method for shortwave radiation with TripleClouds
            
            Args:
                incoming_toa: Incoming downwelling solar radiation at TOA (tensor of shape [ncol])
                albedo_surf_diffuse: Surface albedo to diffuse radiation (tensor of shape [ncol])
                albedo_surf_direct: Surface albedo to direct radiation (tensor of shape [ncol])
                cos_sza: Cosine of solar zenith angle (tensor of shape [ncol])
                reflectance: Diffuse reflectance of each layer (tensor of shape [ncol, nlev])
                transmittance: Diffuse transmittance of each layer (tensor of shape [ncol, nlev])
                ref_dir: Direct beam reflectance of each layer (tensor of shape [ncol, nlev])
                trans_dir_diff: Direct beam to diffuse transmittance (tensor of shape [ncol, nlev])
                trans_dir_dir: Direct transmittance of each layer (tensor of shape [ncol, nlev])
                v_matrix: Overlap matrix [ncol, 3, 3]
            Returns:
                tuple: (flux_up, flux_dn_diffuse, flux_dn_direct) each of shape [ncol, nlev+1]
            """
            
            ncol, nlev = reflectance.shape
            device = reflectance.device
            dtype = reflectance.dtype
            
            # Initialize output arrays
            flux_up = torch.zeros(ncol, nlev + 1, dtype=dtype, device=device)
            flux_dn_diffuse = torch.zeros(ncol, nlev + 1, dtype=dtype, device=device)
            flux_dn_direct = torch.zeros(ncol, nlev + 1, dtype=dtype, device=device)
            
            # Initialize working arrays
            A = torch.zeros(ncol, nlev + 1, dtype=dtype, device=device) # albedo
            source = torch.zeros(ncol, nlev + 1, dtype=dtype, device=device)
            inv_denominator = torch.zeros(ncol, nlev, dtype=dtype, device=device)
            
            # Compute profile of direct (unscattered) solar fluxes at each half-level
            # by working down through the atmosphere
            flux_dn_direct[:, 0] = incoming_toa
            for jlev in range(nlev):
                flux_dn_direct[:, jlev + 1] = flux_dn_direct[:, jlev].clone() * trans_dir_dir[:, jlev]
            
            # Set surface albedo
            A[:, nlev] = albedo_surf_diffuse
            
            # At the surface, the direct solar beam is reflected back into the diffuse stream
            source[:, nlev] = albedo_surf_direct * flux_dn_direct[:, nlev] #* cos_sza
            
            # Work back up through the atmosphere and compute the albedo of the entire
            # earth/atmosphere system below that half-level, and also the "source"
            for jlev in range(nlev - 1, -1, -1):  # nlev down to 1 in Fortran indexing
                # Lacis and Hansen (1974) Eq 33, Shonk & Hogan (2008) Eq 10:
                # inv_denominator[:, jlev] = 1.0 / (1.0 - albedo[:, jlev + 1].clone() * reflectance[:, jlev])
                
                Aplusone = A[:, jlev + 1].clone()
                Adirplusone = Adir[:, jlev + 1].clone()

                inv_denominator[:, jlev] = 1.0 / (1.0 - Aplusone * reflectance[:, jlev])
    
                # Shonk & Hogan (2008) Eq 9, Petty (2006) Eq 13.81:
                # albedo[:, jlev] = (reflectance[:, jlev] + 
                #                   transmittance[:, jlev] * transmittance[:, jlev] * 
                #                   albedo[:, jlev + 1] * inv_denominator[:, jlev])
                
                inv_denom = inv_denominator[:, jlev].clone()
                A[:,jlev] = (reflectance[:, jlev] + torch.square(transmittance[:,jlev]) * Aplusone * inv_denom)
                Adir[:,jlev] = (ref_dir[:, jlev] + (trans_dir_dir[:,jlev]*Adirplusone + trans_dir_diff[:,jlev]*Aplusone)  * 
                            transmittance[:,jlev] * inv_denom)

                # TripleClouds :
                # ng_nreg = self.nreg//3 
                A[0:ng_nreg]           = A[0:ng_nreg]*V[1,1] + A[ng_nreg:2*ng_nreg]*V[2,1] + A[2*ng_nreg:3*ng_nreg]*V[3,1]
                A[ng_nreg:2*ng_nreg]   = A[0:ng_nreg]*V[1,2] + A[ng_nreg:2*ng_nreg]*V[2,2] + A[2*ng_nreg:3*ng_nreg]*V[3,2]
                A[2*ng_nreg:3*ng_nreg] = A[0:ng_nreg]*V[1,3] + A[ng_nreg:2*ng_nreg]*V[2,3] + A[2*ng_nreg:3*ng_nreg]*V[3,3]         
                                
                # Shonk & Hogan (2008) Eq 11:
                fluxdndir = flux_dn_direct[:, jlev].clone()
                source[:, jlev] = (ref_dir[:, jlev] * fluxdndir +
                                  transmittance[:, jlev] * 
                                  (source[:, jlev + 1] + 
                                  Aplusone * trans_dir_diff[:, jlev] * fluxdndir) *
                                  inv_denom)
            
            # At top-of-atmosphere there is no diffuse downwelling radiation
            flux_dn_diffuse[:, 0] = 0.0
            
            # At top-of-atmosphere, all upwelling radiation is due to scattering
            # by the direct beam below that level
            flux_up[:, 0] = source[:, 0]
            
            # Work back down through the atmosphere computing the fluxes at each half-level
            for jlev in range(nlev):  # 1 to nlev in Fortran indexing
                # Shonk & Hogan (2008) Eq 14 (after simplification):
                fluxdndir = flux_dn_direct[:, jlev].clone()
                fluxdndiff = flux_dn_diffuse[:, jlev].clone()
    
                flux_dn_diffuse[:, jlev + 1] = ((transmittance[:, jlev] * fluxdndiff +  
                                                 reflectance[:, jlev] * source[:, jlev + 1] + 
                                                 trans_dir_diff[:, jlev] * fluxdndir) *
                                               inv_denominator[:, jlev])      
          
                # Shonk & Hogan (2008) Eq 12:
                flux_up[:, jlev + 1] = (A[:, jlev + 1] * flux_dn_diffuse[:, jlev + 1].clone() +
                                       source[:, jlev + 1])
                
                # Apply cosine correction to direct flux
                # flux_dn_direct[:, jlev] = fluxdndir * cos_sza
            
            # Final cosine correction for surface direct flux
            # flux_dn_direct[:, nlev] = flux_dn_direct[:, nlev] * cos_sza
            
            return flux_up, flux_dn_diffuse, flux_dn_direct

    @torch.compile
    def lw_solver_noscat_batchfirst(self, flux_lw_dn, flux_lw_up, trans_lw, 
                              source_dn, source_up, source_sfc, albedo_surf):
        
        nlev = trans_lw.shape[1]
        
        # flux_lw_dn = torch.zeros(nlev+1,  device=trans_lw.device)
        # flux_lw_up = torch.zeros(nlev+1,  device=trans_lw.device)

        # At top-of-atmosphere there is no diffuse downwelling radiation
        flux_lw_dn[:,0] = 0.0
        # Work down through the atmosphere computing the downward fluxes
        # at each half-level (vectorized over columns)
        for jlev in range(nlev):
            flux_lw_dn[:,jlev + 1] = (trans_lw[:,jlev] * flux_lw_dn[:,jlev].clone()  + 
                                   source_dn[:,jlev])
        
        flux_lw_up[:,nlev] = source_sfc + albedo_surf * flux_lw_dn[:,nlev]
        
        # Work back up through the atmosphere computing the upward fluxes
        # at each half-level (vectorized over columns)
        for jlev in range(nlev - 1, -1, -1):
            flux_lw_up[:,jlev] = (trans_lw[:,jlev] * flux_lw_up[:,jlev + 1].clone()  + 
                               source_up[:,jlev])
            
        return flux_lw_dn, flux_lw_up
    
    def forward(self, inp_list : List[Tensor]):
        inputs_main     = inp_list[0]
        inputs_aux      = inp_list[1]
        rnn1_mem        = inp_list[2]
        x_denorm        = inp_list[3]

        if self.use_ensemble:
            inputs_main = inputs_main.unsqueeze(0)
            inputs_aux = inputs_aux.unsqueeze(0)
            inputs_main = torch.repeat_interleave(inputs_main,repeats=2,dim=0)
            inputs_aux = torch.repeat_interleave(inputs_aux,repeats=2,dim=0)
            inputs_main = inputs_main.flatten(0,1)
            inputs_aux = inputs_aux.flatten(0,1)
                    
        batch_size  = inputs_main.shape[0]
        nlev        = inputs_main.shape[1]
        device      = inputs_main.device

        if self.add_pres:
            sp = torch.unsqueeze(inputs_aux[:,0:1],1)
            # undo scaling
            sp = sp*35451.17 + 98623.664
            pres  = self.preslay(sp)
            inputs_main = torch.cat((inputs_main,pres),dim=2)
            play = self.preslay_nonorm(sp)
            delta_plev = self.presdelta(sp)
            plev = self.preslev(sp)
        if self.repeat_mu:
            mu = torch.reshape(inputs_aux[:,6:7],(-1,1,1))
            mu_rep = torch.repeat_interleave(mu,nlev,dim=1)
            inputs_main = torch.cat((inputs_main,mu_rep),dim=2)
            
        
        # if self.separate_radiation:
        # Do not use inputs -2,-3,-4 (O3, CH4, N2O) or first 10 levels
        inputs_main_crm = torch.cat((inputs_main[:,10:,0:-4], inputs_main[:,10:,-1:]),dim=2)

        if self.use_initial_mlp:
            inputs_main_crm = self.mlp_initial(inputs_main_crm)
            inputs_main_crm = self.nonlin(inputs_main_crm)  
            
        inputs_main_crm = torch.cat((inputs_main_crm,rnn1_mem), dim=2)
            
        if self.use_third_rnn: # use initial downward RNN
            inputs_toa = torch.cat((inputs_aux[:,1:2], inputs_aux[:,6:7]),dim=1) #  pbuf_SOLIN and COSZRS
            cx0 = self.mlp_toa1(inputs_toa)
            # cx0 = self.nonlin(cx0)
            hx0 = self.mlp_toa2(inputs_toa)
            # hx0 = self.nonlin(hx0)
            hidden0 = (torch.unsqueeze(hx0,0), torch.unsqueeze(cx0,0))  
            rnn0out, states = self.rnn0(inputs_main_crm, hidden0)
            
            rnn1_input =  torch.flip(rnn0out, [1])
            del rnn0out
        else:
            # TOA is first in memory, so to start at the surface we need to go backwards
            rnn1_input = torch.flip(inputs_main_crm, [1])
            del inputs_main_crm
        
        # The input (a vertical sequence) is concatenated with the
        # output of the RNN from the previous time step 

        # if self.separate_radiation:
        inputs_sfc = torch.cat((inputs_aux[:,0:6],inputs_aux[:,11:]),dim=1)

        hx = self.mlp_surface1(inputs_sfc)
        hx = self.nonlin(hx)
        cx = self.mlp_surface2(inputs_sfc)
        # cx = self.nonlin(cx)
        hidden = (torch.unsqueeze(hx,0), torch.unsqueeze(cx,0))
        rnn1out, states = self.rnn1(rnn1_input, hidden)
        del rnn1_input

        # if self.predict_flux:
        #     rnn1out = torch.cat((torch.unsqueeze(hx,1),rnn1out),dim=1)

        rnn1out = torch.flip(rnn1out, [1])

        if self.use_third_rnn:
          hx2 = torch.randn((batch_size, self.nh_rnn2),device=device)  # (batch, hidden_size)
          cx2 = torch.randn((batch_size, self.nh_rnn2),device=device)
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
          rnn2out, states = self.rnn2(input_rnn2, hidden2)

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
          
        if self.use_third_rnn:
            rnn1_mem = rnn2out
        else:
            rnn1_mem = rnn2out

        out = self.mlp_output(rnn2out)

        out_sfc = self.mlp_surface_output(final_sfc_inp)
        
        out_new = torch.zeros(batch_size, nlev, self.ny, device=device)
        out_new[:,10:,:] = out
        
        out_denorm      = out_new / self.yscale_lev

        T_before        = x_denorm[:,:,0:1]
        rh_before       = x_denorm[:,:,1:2]

        # rh_before =  torch.clamp(rh_before, min=0.1, max=1.4)
        # q_before        = self.relative_to_specific_humidity(rh_before, T_before, play)
        # q_before        = torch.clamp(q_before, min=0.0, max=0.5)
        q_before        = x_denorm[:,:,-1:]

        qliq_before     = x_denorm[:,:,2:3]
        qice_before     = x_denorm[:,:,3:4]   
        qn_before       = qliq_before + qice_before 
        
        T_new           = T_before  + out_denorm[:,:,0:1]*1200

        if self.mp_constraint:
          liq_frac_constrained    = self.temperature_scaling(T_new)
          #                            dqn
          qn_new      = qn_before + out_denorm[:,:,2:3]*1200  
          qn_new      = torch.clamp(qn_new, min=0.0)
          qliq_new    = liq_frac_constrained*qn_new
          qice_new    = (1-liq_frac_constrained)*qn_new
          out_new[:,:,2:3] = (qn_new - qn_before)/1200 * self.yscale_lev[:,2:3]
        else:
          qliq_new    = qliq_before + out_denorm[:,:,2:3]*1200 
          qice_new    = qice_before + out_denorm[:,:,3:4]*1200 
          qliq_new = torch.clamp(qliq_new, min=0.0)
          qice_new = torch.clamp(qice_new, min=0.0)
          out_new[:,:,2:3] = (qliq_new - qliq_before)/1200 * self.yscale_lev[:,2:3]
          out_new[:,:,3:4] = (qice_new - qice_before)/1200 * self.yscale_lev[:,3:4]


        dq          = out_denorm[:,:,1:2]
        q_new       = q_before + dq*1200 
        q_new       = torch.clamp(q_new, min=0.0)
        out_new[:,:,1:2] = (q_new - q_before)/1200 * self.yscale_lev[:,1:2]

        # assert not torch.isnan(q_new).any()
        # print("q new raw min max",  torch.min(q_new[:,:]).item(), torch.max(q_new[:,:]).item())

        # 0. INPUT SCALING - gases

        # nn_inputs_lw = "tlay", "play", "h2o", "o3", "co2", "ch4", "n2o", "cfc11", "cfc12", 
        # "co", "ccl4", "cfc22", "hfc143a", "hfc125", "hfc23", "hfc32", "hfc134a", "cf4" ;
        # nn_input_coeffs_max = 340, 11.6, 0.507753, 0.06316834, 0.0028, 4.2e-06, 5.813521e-07, 2e-09, 6e-10, 2.4e-06, 1.03168e-10, 2.384533e-10, 7.791439e-10, 9.888e-10, 3.106764e-11, 1.364208e-11, 4.233e-10, 1.670263e-10 ;
        # nn_input_coeffs_min = 160, 0.00515, 0.0101, 0.00436, 0.000141, 2e-09, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ;
        # x(i) = (x(i) - xmin(i)) / (xmax(i) - xmin(i))
        
        # Water vapor specific humidity to volume mixing ratio                                      REMEMBER TO FIX THIS!!!!!!!!!!!!!!!!!!!!!!!!!!
        # vmr_h2o = (q_new / ( 1.0 - q_new))  *  28.97 / 18.01528 # mol_weight_air / mol_weight_gas
        vmr_h2o = q_new * 1.608079364 # (28.97 / 18.01528) # mol_weight_air / mol_weight_gas

        fact = 1/ (1 + vmr_h2o)
        m_air = (0.04698 + vmr_h2o)*fact #(m_dry + m_h2o * vmr_h2o) * fact 
        #                             avogad
        col_dry = 10.0 * delta_plev * 6.02214076e23 * fact/(1000.0*m_air*100.0*9.80665) 
        col_dry = torch.reshape(col_dry,(batch_size,1, nlev))
        # col_dry = torch.transpose(col_dry, 0, 1)
        
        # T_new = (T_new - self.xmean_lev[:,0:1] ) / (self.xdiv_lev[:,0:1])
        temp = (T_new - 160 ) / (180)
        pressure = (torch.log(play) - 0.00515) / (11.59485)
        vmr_h2o = (torch.sqrt(torch.sqrt(vmr_h2o))  - 0.0101) / 0.497653
        # print("q new  min max", torch.min(vmr_h2o[:,:]).item(), torch.max(vmr_h2o[:,:]).item())
        # print("q nans", torch.nonzero(torch.isnan(q_new.view(-1))))
        # assert not torch.isnan(q_new).any()
        
        # 0. INPUT SCALING - clouds
        qliq_new = 1 - torch.exp(-qliq_new * self.lbd_qc)
        qice_new = 1 - torch.exp(-qice_new * self.lbd_qi)
        
        # Radiation inputs: pressure, temperature, water vapor, cloud ice and liquid, O3, CH4, N2O,, cld heterogeneity
        cloudfrac  = torch.zeros(batch_size, nlev, 1, device=device)
        cloudfrac[:,10:] = rnn1_mem[:,:,0:1]

        inputs_rad = torch.cat((pressure, temp, vmr_h2o, qliq_new, qice_new, inputs_main[:,:,12:15], cloudfrac),dim=2)
        # inputs_rad = torch.transpose(inputs_rad, 0, 1) 
        # print("pressure min max", torch.min(pressure[:,:]).item(), torch.max(pressure[:,:]).item())
        # print("temp min max", torch.min(temp[:,:]).item(), torch.max(temp[:,:]).item())
        # print("vmr_h2o min max", torch.min(vmr_h2o[:,:]).item(), torch.max(vmr_h2o[:,:]).item())
        # print("qliq_new min max", torch.min(qliq_new[:,:]).item(), torch.max(qliq_new[:,:]).item())
        # print("qice_new min max", torch.min(qice_new[:,:]).item(), torch.max(qice_new[:,:]).item())
        # print("inputs_rad min max", torch.min(inputs_rad[:,:]).item(), torch.max(inputs_rad[:,:]).item())

        # 1. MLP TO PREDICT OPTICAL PROPERTIES
        

        # --------------------------- LONGWAVE ---------------------------
        optprops_lw = self.mlp_optprops_lw(inputs_rad) 
        # optprops_lw = torch.reshape(optprops_lw,(nlev, batch_size,self.ng_lw, self.ny_optprops_lw))

        # (nb, nlev, ng*ny ) --> (nb, ng*ny, nlev) --> (nb, ng, ny, nlev)
        optprops_lw = torch.transpose(optprops_lw, 1, 2)
        optprops_lw = torch.reshape(optprops_lw,(batch_size,self.ng_lw, self.ny_optprops_lw, nlev))

        # tau_lw, pfrac = optprops_lw.chunk(2, 3)
        tau_lw, pfrac = optprops_lw.chunk(2, 2)
        tau_lw      = torch.pow(tau_lw.squeeze(), 8) # (ncol, ng_lw, nlev)

        tau_lw      = tau_lw*(1e-24*col_dry)
        tau_lw      = torch.clamp(tau_lw, min=1e-6, max=400.0)
        pfrac       = pfrac.squeeze() # 
        # print("pfrac0 min max", torch.min(pfrac[:,:]).item(), torch.max(pfrac[:,:]).item())

        pfrac       = self.softmax(pfrac)
        # T_new       = torch.transpose(T_new.squeeze(),0,1)
        # play        = torch.transpose(play.squeeze(),0,1)
        # plev        = torch.transpose(plev.squeeze(),0,1) 
        # tlev        = self.interpolate_tlev(T_new, play, plev) # (nlev+1, nb)
        T_new       = T_new.squeeze()
        play        = play.squeeze()
        plev        = plev.squeeze()
        tlev        = self.interpolate_tlev_batchfirst(T_new, play, plev) # (nb, nlev+1)
        # print("shape tlev", tlev.shape)
        lwup_sfc    = (inputs_aux[:,11]*self.xdiv_sca[11]) + self.xmean_sca[11]
        # source_sfc  = pfrac[-1,:,:]*lwup_sfc.unsqueeze(-1)
        source_sfc  = pfrac[:,:,-1]*lwup_sfc.unsqueeze(1)

        # lwup_lev    = torch.unsqueeze(self.outgoing_lw(tlev),-1)
        # source_lev  = torch.zeros(nlev+1, batch_size, self.ng_lw, device=pfrac.device)
        # source_lev[-1,:,:] = pfrac[-1,:,:] * lwup_lev[-1,:,:]
        # source_lev[0:-1,:] = pfrac[:,:,:]  * lwup_lev[0:-1,:,:]
        lwup_lev    = torch.unsqueeze(self.outgoing_lw(tlev),1) # (nb, ng, nlev+1)
        # print("pfrac min max", torch.min(pfrac[:,:]).item(), torch.max(pfrac[:,:]).item())
        # print("lwup_lev min max", torch.min(lwup_lev[:,:]).item(), torch.max(lwup_lev[:,:]).item())

        source_lev  = torch.zeros(batch_size, self.ng_lw, nlev+1, device=device)
        source_lev[:,:,-1] = pfrac[:,:,-1] * lwup_lev[:,:,-1]
        source_lev[:,:,0:-1] = pfrac[:,:,:]  * lwup_lev[:,:,0:-1]
        del pfrac, optprops_lw
        
        # ---- REFTRANS LW ----
        # planck_top = source_lev[:,:,0:-1]
        # planck_bot = source_lev[:,:,1:]
        
        # od_loc = 1.66 * tau_lw
        # trans_lw = torch.exp(-od_loc)
        # # Calculate coefficient for Padé approximant (vectorized)
        # coeff = 0.2 * od_loc
        # # Calculate mean Planck function (vectorized)
        # planck_fl = 0.5 * (planck_top + planck_bot)
        # # Calculate source terms using Padé approximant (vectorized)
        # one_minus_trans = 1.0 - trans_lw
        # one_plus_coeff = 1.0 + coeff
        # source_dn = one_minus_trans * (planck_fl + coeff * planck_bot) / one_plus_coeff
        # source_up = one_minus_trans * (planck_fl + coeff * planck_top) / one_plus_coeff
        # print("source_lev min max", torch.min(source_lev[:,:]).item(), torch.max(source_lev[:,:]).item())

        source_up, source_dn, trans_lw = self.reftrans_lw(source_lev, tau_lw)

        # print("shape source_up dn trans_lw", source_up.shape, source_dn.shape, trans_lw.shape)
        # source_up       = torch.reshape(source_up, (nlev,-1))
        # source_dn       = torch.reshape(source_dn, (nlev,-1))
        # trans_lw        = torch.reshape(trans_lw, (nlev,-1))
        source_up       = torch.reshape(source_up, (-1,nlev))
        source_dn       = torch.reshape(source_dn, (-1,nlev))
        trans_lw        = torch.reshape(trans_lw, (-1,nlev))
        # print("trans_lw min max", torch.min(trans_lw[:,:]).item(), torch.max(trans_lw[:,:]).item())
        # print("source_up min max", torch.min(source_up[:,:]).item(), torch.max(source_up[:,:]).item())

        # calc_fluxes_no_scattering_lw
        # At top-of-atmosphere there is no diffuse downwelling radiation
        # flux_lw_dn = torch.zeros(nlev+1, batch_size*self.ng_lw,  device=pfrac.device)
        # flux_lw_up = torch.zeros(nlev+1, batch_size*self.ng_lw,  device=pfrac.device)

        # # At top-of-atmosphere there is no diffuse downwelling radiation
        # flux_lw_dn[0, :] = 0.0
        
        # Surface reflection and emission (vectorized)
        albedo_surf = self.mlp_sfc_albedo_lw(inputs_aux)
        albedo_surf = self.sigmoid(albedo_surf)
        albedo_surf = torch.flatten(albedo_surf)
        source_sfc  = torch.flatten(source_sfc)
        
        flux_lw_dn_gpt = torch.zeros(batch_size*self.ng_lw, nlev+1,  device=device)
        flux_lw_up_gpt = torch.zeros(batch_size*self.ng_lw, nlev+1,  device=device)

        flux_lw_dn, flux_lw_up = self.lw_solver_noscat_batchfirst(flux_lw_dn_gpt, flux_lw_up_gpt, trans_lw, source_dn, source_up, source_sfc,
                                                        albedo_surf)

        flux_lw_up_gpt = torch.reshape(flux_lw_up_gpt, (batch_size, self.ng_lw, nlev+1))
        flux_lw_up = torch.sum(flux_lw_up_gpt,dim=1)
        flux_lw_dn_gpt = torch.reshape(flux_lw_dn_gpt, (batch_size, self.ng_lw, nlev+1))
        flux_lw_dn = torch.sum(flux_lw_dn_gpt,dim=1)
        del flux_lw_dn_gpt, flux_lw_up_gpt
        
        # print("flux lw up  min max", torch.min(flux_lw_up[:,:]).item(), torch.max(flux_lw_up[:,:]).item())

        # -------------------------- SHORTWAVE -----------------------------
        optprops_sw = self.mlp_optprops_sw(inputs_rad)
        # optprops_sw = torch.reshape(optprops_sw,(nlev, batch_size,self.ng_sw, self.ny_optprops_sw))
        
        # (nb, nlev, ng*ny ) --> (nb, ng*ny, nlev) --> (nb, ng, ny, nlev)
        optprops_sw = torch.transpose(optprops_sw, 1, 2)
        optprops_sw = torch.reshape(optprops_sw,(batch_size,self.ng_sw, self.ny_optprops_sw, nlev))

        # tau_sw, tau_sw_scat, g_sw = optprops_sw.chunk(3, 3)
        tau_sw, tau_sw_scat, g_sw = optprops_sw.chunk(3,2 )
        tau_sw      = torch.pow(tau_sw.squeeze(), 8)

        tau_sw      = tau_sw*(1e-24*col_dry)
        tau_sw      = torch.clamp(tau_sw, min=1e-6, max=40.0) 
        tau_sw_scat = torch.pow(tau_sw_scat.squeeze(), 8)
        tau_sw_scat = tau_sw_scat*(1e-24*col_dry)
        tau_sw_scat = torch.clamp(tau_sw_scat, min=1e-6, max=40.0) 
        tau_sw      = tau_sw + tau_sw_scat
        ssa_sw      = tau_sw_scat / tau_sw
        g_sw        = self.sigmoid(g_sw.squeeze())
        del optprops_sw, tau_sw_scat
        
        # ---- REFTRANS SW ------
        mu0 = torch.reshape(inputs_aux[:,6:7],(-1,1,1))
        mu0 = torch.clamp(mu0, min=1e-3) 
        # since changing to batch first, reftrans will operate on arrays shaped (ncol*ng,nlev)
        # we need to repeat mu0 to shape (ncol*ng,1) from (ncol)
        # mu0 = torch.reshape(torch.repeat_interleave(mu0, self.ng_sw, dim=1),(batch_size*self.ng_sw,1))
        mu0_rep = torch.reshape(torch.repeat_interleave(mu0, self.ng_sw*nlev, dim=1),(batch_size*self.ng_sw,nlev))

        # print("mu0 min max", torch.min(mu0).item(), torch.max(mu0).item())
        # print("tau_sw min max", torch.min(tau_sw).item(), torch.max(tau_sw).item())
        # print("ssa_sw min max", torch.min(ssa_sw).item(), torch.max(ssa_sw).item())
        # print("g_sw min max", torch.min(g_sw).item(), torch.max(g_sw).item())

        # t0 = time.time()
        
        # batched_reftra = torch.func.vmap(self.calc_reflectance_transmittance_sw)
        
        tau_sw = torch.reshape(tau_sw, (-1, nlev))
        ssa_sw = torch.reshape(ssa_sw, (-1, nlev))
        g_sw   = torch.reshape(g_sw, (-1, nlev))
        # print("shape mu0", mu0.shape, "tausw", tau_sw.shape, "ssasw", ssa_sw.shape, "gsw", g_sw.shape)
        ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir = self.calc_reflectance_transmittance_sw(mu0_rep, tau_sw, ssa_sw, g_sw)
        
        # ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir = batched_reftra(mu0, tau_sw, ssa_sw, g_sw)
        
        ref_diff            = torch.reshape(ref_diff, (-1, nlev))
        trans_diff          = torch.reshape(trans_diff, (-1, nlev))
        ref_dir             = torch.reshape(ref_dir, (-1, nlev))
        trans_dir_diff      = torch.reshape(trans_dir_diff, (-1, nlev))
        trans_dir_dir       = torch.reshape(trans_dir_dir, (-1, nlev))

        # print ("shape ref_diff", ref_diff.shape)
        # print("elapsed reftra {}".format(time.time() - t0))
        del tau_sw, ssa_sw, g_sw, mu0_rep

        
        incoming_toa    = inputs_aux[:,1:2]
        # print("inc toa 0  min max", torch.min(incoming_toa).item(), torch.max(incoming_toa).item())
        # incoming_toa    = torch.clone(torch.div(incoming_toa, mu0))
        # print("inc toa 1  min max", torch.min(incoming_toa).item(), torch.max(incoming_toa).item())

        toa_spectral = self.softmax_dim1(torch.square(self.sw_solar_weights))
        # print("toa_spectral  min max", torch.min(toa_spectral).item(), torch.max(toa_spectral).item())

        # Here we apply torch.square to ensure the values are positive, then softmax so that they sum to 0
        incoming_toa = torch.flatten(incoming_toa*toa_spectral)
        
        albedo_surf_diff_sw     = inputs_aux[:,9]
        albedo_surf_dir_sw      = inputs_aux[:,10]

        # cos_sza = torch.repeat_interleave(mu0.flatten().unsqueeze(1),self.ng_sw).flatten()
        # albedo_surf_diff_sw = self.mlp_sfc_albedo_sw(inputs_aux)
        albedo_surf_diff_sw = torch.repeat_interleave(albedo_surf_diff_sw.flatten().unsqueeze(1),self.ng_sw).flatten()
        albedo_surf_dir_sw = torch.repeat_interleave(albedo_surf_dir_sw.flatten().unsqueeze(1),self.ng_sw).flatten()

        # print("inc toa  min max", torch.min(incoming_toa).item(), torch.max(incoming_toa).item())
        # print("albedo_surf_dir_sw  min max", torch.min(albedo_surf_dir_sw).item(), torch.max(albedo_surf_dir_sw).item())
        # print("albedo_surf_dif_sw  min max", torch.min(albedo_surf_diff_sw).item(), torch.max(albedo_surf_diff_sw).item())
        # print("cos_sza  min max", torch.min(cos_sza).item(), torch.max(cos_sza).item())
        # print("ref_diff min max", torch.min(ref_diff).item(), torch.max(ref_diff).item())
        # print("trans_diff min max", torch.min(trans_diff).item(), torch.max(trans_diff).item())
        # print("ref_dir min max", torch.min(ref_dir).item(), torch.max(ref_dir).item())
        # print("trans_dir_diff min max", torch.min(trans_dir_diff).item(), torch.max(trans_dir_diff).item())
        # print("trans_dir_dir min max", torch.min(trans_dir_dir).item(), torch.max(trans_dir_dir).item())

        # flux_sw_up, flux_sw_dn_diffuse, flux_sw_dn_direct = self.adding_ica_sw(incoming_toa, 
        #             albedo_surf_diff_sw, albedo_surf_dir_sw, cos_sza, ref_diff, 
        #             trans_diff, ref_dir, trans_dir_diff, trans_dir_dir)
        
        mu0_rep_ng = torch.flatten(torch.repeat_interleave(mu0, self.ng_sw, dim=1))

        # print("shape args", incoming_toa.shape, albedo_surf_diff_sw.shape, albedo_surf_dir_sw.shape, 
        #       mu0_rep.shape, ref_diff.shape, trans_diff.shape, ref_dir.shape, trans_dir_diff.shape, trans_dir_dir.shape)
        # incoming_toa        = incoming_toa.unsqueeze(1)
        # albedo_surf_diff_sw = albedo_surf_diff_sw.unsqueeze(1)
        # albedo_surf_dir_sw  = albedo_surf_dir_sw.unsqueeze(1)

        flux_sw_up, flux_sw_dn_diffuse, flux_sw_dn_direct = self.adding_ica_sw_batchfirst(incoming_toa, 
                    albedo_surf_diff_sw, albedo_surf_dir_sw, mu0_rep_ng, ref_diff, 
                    trans_diff, ref_dir, trans_dir_diff, trans_dir_dir)
        del ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir
        
        # print("flux_sw_up  min max", torch.min(flux_sw_up[:,:]).item(), torch.max(flux_sw_up[:,:]).item())
        # print("flux_sw_dn_diffuse  min max", torch.min(flux_sw_dn_diffuse[:,:]).item(), torch.max(flux_sw_dn_diffuse[:,:]).item())
        # print("flux_sw_dn_direct  min max", torch.min(flux_sw_dn_direct[:,:]).item(), torch.max(flux_sw_dn_direct[:,:]).item())
        
        # nlev, ncol = ref_dir.shape
        # device = ref_dir.device
        # dtype = ref_dir.dtype
        
        # Initialize output arrays
        # flux_sw_up = torch.zeros((batch_size, self.ng_sw, nlev + 1), device=device)
        # flux_sw_dn_diffuse = torch.zeros((batch_size, self.ng_sw, nlev + 1),  device=device)
        # flux_sw_dn_direct = torch.zeros((batch_size, self.ng_sw, nlev + 1),  device=device)
        
        flux_sw_up = self.relu(flux_sw_up)
        flux_sw_dn_diffuse = self.relu(flux_sw_dn_diffuse)
        flux_sw_dn_direct = self.relu(flux_sw_dn_direct)

        flux_sw_up = torch.reshape(flux_sw_up, (batch_size, self.ng_sw, nlev+1))
        flux_sw_dn_diffuse = torch.reshape(flux_sw_dn_diffuse, (batch_size, self.ng_sw, nlev+1))
        flux_sw_dn_direct = torch.reshape(flux_sw_dn_direct, (batch_size, self.ng_sw, nlev+1))
        

        # flux is (ncol, ng, nlev)
        SOLS = flux_sw_dn_direct[:,0,-1].unsqueeze(1)   # SOLS (ncol, 1)
        SOLL = flux_sw_dn_direct[:,1,-1].unsqueeze(1)   # SOLL
        SOLSD = flux_sw_dn_diffuse[:,0,-1].unsqueeze(1) # SOLSD
        SOLLD = flux_sw_dn_diffuse[:,1,-1].unsqueeze(1) # SOLLD

        flux_sw_up          = torch.sum(flux_sw_up,dim=1)
        flux_sw_dn_diffuse  = torch.sum(flux_sw_dn_diffuse,dim=1)
        flux_sw_dn_direct   = torch.sum(flux_sw_dn_direct,dim=1)
        
        flux_sw_dn          = flux_sw_dn_diffuse + flux_sw_dn_direct
        flux_sw_dn_sfc      = flux_sw_dn[:,-1].unsqueeze(1)             # NETSW

        flux_sw_net         = flux_sw_dn - flux_sw_up
        mu = torch.reshape(inputs_aux[:,6:7],(-1,1))
        mu_rep = torch.repeat_interleave(mu,nlev+1,dim=1)
        inds_zero = mu_rep < 1e-3
        flux_sw_net[inds_zero] = 0.0
        
        flux_lw_dn_sfc      = flux_lw_dn[:,-1].unsqueeze(1)             # FLWDS 

        flux_lw_net         = flux_lw_dn - flux_lw_up
        # print("flux_lw_up  min max", torch.min(flux_lw_up[:,:]).item(), torch.max(flux_lw_up[:,:]).item())
        # print("flux_lw_net  min max", torch.min(flux_lw_net[:,:]).item(), torch.max(flux_lw_net[:,:]).item())
        # print("flux_sw_net  min max", torch.min(flux_sw_net[:,:]).item(), torch.max(flux_sw_net[:,:]).item())
        # print("incflux 100", inputs_aux[100,1:2].item())
        # print("mu0 100", mu0[100].item())
        # print("flux_sw_dn 100", flux_sw_dn[100,:].detach().cpu().numpy())

        # print("flux_sw 100", flux_sw_net[100,:].detach().cpu().numpy())

        flux_net = flux_lw_net + flux_sw_net
        flux_diff = flux_net[:,1:] - flux_net[:,0:-1]

        pres_diff = plev[:,1:] - plev[:,0:-1]
        dT_rad = -(flux_diff / pres_diff) * 0.009767579681 # * g/cp = 9.80665 / 1004
        # print("flux_diff  min max", torch.min(flux_diff[:,:]).item(), torch.max(flux_diff[:,:]).item())
        # print("pres_diff  min max", torch.min(pres_diff[:,:]).item(), torch.max(pres_diff[:,:]).item())
        # print("dT_rad  min max", torch.min(dT_rad[:,:]).item(), torch.max(dT_rad[:,:]).item())
        
        # normalize heating rate output 
        dT_rad = dT_rad * self.yscale_lev[:,0].unsqueeze(0)

        # print("dT_rad 2  min max", torch.min(dT_rad[:,:]).item(), torch.max(dT_rad[:,:]).item())
        out_new[:,:,0:1] = out_new[:,:,0:1] + dT_rad.unsqueeze(2)

        # #1D (scalar) Output variables: ['cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 
        # #'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']
        out_sfc_rad = torch.cat((flux_sw_dn_sfc, flux_lw_dn_sfc,  SOLS, SOLL, SOLSD, SOLLD ), dim=1)
        out_sfc_rad =  out_sfc_rad * self.yscale_sca_rad
        
        # # rad predicts everything except PRECSC, PRECC
        out_sfc =  torch.cat((out_sfc_rad[:,0:2], out_sfc, out_sfc_rad[:,2:]),dim=1)
        
        # -------------- SEPARATE GAS AND CLOUD OPTICAL PROPERTIES ---------------
        # inputs_rad_gas =    torch.cat((pressure, temp, vmr_h2o, inputs_main[:,:,12:15]),dim=2)
        # inputs_rad_gas =    torch.transpose(inputs_rad_gas, 0, 1) 
        # # TOA is first in memory, so to start at the surface we need to go backwards
        # # inputs_rad_gas =    torch.flip(inputs_rad_gas, [0])
        
        # optprops_gas_sw = self.mlp_gasopt_sw(inputs_rad_gas) # (nlev, nb, ng_sw*2)
        # # optprops_gas_sw = self.relu(optprops_gas_sw)
        # optprops_gas_sw = torch.reshape(optprops_gas_sw,(nlev, batch_size*self.ng_sw, 2))
        # optprops_gas_sw = torch.pow(optprops_gas_sw,8)
        # # print("shape opt", optprops_gas_sw.shape)
        # tau_sw, tau_sw_ray = optprops_gas_sw.chunk(2, 2)

        # optprops_gas_lw = self.mlp_gasopt_lw(inputs_rad_gas) # (nlev, nb, ng_lw*2)
        # optprops_gas_lw = self.relu(optprops_gas_lw)
        # # optprops_gas_lw = torch.reshape(optprops_gas_lw,(nlev, batch_size*self.ng_lw, 2))
        # optprops_gas_lw = torch.reshape(optprops_gas_lw,(nlev, batch_size, self.ng_lw, 2))
        # tau_lw, pfrac   = optprops_gas_lw.chunk(2, 3) # (nlev, nb, ng_lw, 1)
        # tau_lw          = torch.pow(tau_lw, 8)
        # pfrac = pfrac.squeeze() # (nlev, nb, ng_lw)
        # pfrac = self.softmax(pfrac)

        # T_new = torch.transpose(T_new.squeeze(),0,1)
        # play = torch.transpose(play.squeeze(),0,1)
        # plev = torch.transpose(plev.squeeze(),0,1) 
        
        # tlev = self.interpolate_tlev(T_new, play, plev) # (nlev+1, nb)
        # lwup_sfc = (inputs_aux[:,11]*self.xdiv_sca[11]) + self.xmean_sca[11]
        # source_sfc = pfrac[-1,:,:]*lwup_sfc.unsqueeze(-1)
        # lwup_lev = torch.unsqueeze(self.outgoing_lw(tlev),-1)

        # source_lev = torch.zeros(nlev+1, batch_size, self.ng_lw, device=pfrac.device)
        # source_lev[-1,:,:] = pfrac[-1,:,:] * lwup_lev[-1,:,:]
        # source_lev[0:-1,:] = pfrac[:,:,:]  * lwup_lev[0:-1,:,:]
        # # print("shape sfc source", source_sfc.shape, "shape lev source", source_lev.shape)
        # # print("source_sfc min max", torch.min(source_sfc).item(), torch.max(source_sfc).item())
        # # print("source_lev min max", torch.min(source_lev).item(), torch.max(source_lev).item())
        
        # optprops_cloud_aer_sw = self.mlp_cloudopt_sw(inputs_rad_cld) # (nlev, nb, ng_sw*3)
        # optprops_cloud_aer_lw = self.mlp_cloudopt_lw(inputs_rad_cld) # (nlev, nb, ng_sw*3)
        
        # optprops_cloud_aer_sw = self.relu(optprops_cloud_aer_sw)
        # optprops_cloud_aer_lw = self.relu(optprops_cloud_aer_lw)
        
        # optprops_cloud_aer_sw = torch.reshape(optprops_cloud_aer_sw,(nlev, batch_size, self.ng_sw, 3))
        # optprops_cloud_aer_lw = torch.reshape(optprops_cloud_aer_lw,(nlev, batch_size, self.ng_sw, 3))

        # tau_cld_sw, ssa_cld_sw, g_cld_sw   = optprops_cloud_aer_sw.chunk(2, 3) #
        # tau_cld_lw, ssa_cld_lw, g_cld_lw   = optprops_cloud_aer_lw.chunk(2, 3) #
        
        # ssa_cld_sw  = self.sigmoid(ssa_cld_sw)
        # ssa_cld_lw  = self.sigmoid(ssa_cld_lw)

        # g_cld_sw    = self.sigmoid(g_cld_sw)
        # g_cld_lw    = self.sigmoid(g_cld_lw)
        
        # # 1.b TRANSFORM PREDICTED OPTICAL PROPERTIES INTO OD, SSA, G
        # # GAS
        # tau_lw = tau_lw*(1e-24*col_dry)
        # tau_lw = torch.clamp(tau_lw, min=1e-6, max=400.0)
        
        # tau_sw = tau_sw*(1e-24*col_dry)
        # tau_sw = torch.clamp(tau_sw, min=1e-6, max=40.0) 
        # tau_sw_ray = tau_sw_ray*(1e-24*col_dry)
        # tau_sw_ray = torch.clamp(tau_sw_ray, min=1e-6, max=40.0) 

        # tau_sw = tau_sw + tau_sw_ray
        # ssa_sw_gas = tau_sw_ray / tau_sw
        
        # # CLOUD + AEROSOL
        # tau_cld_sw = tau_cld_sw*(1e-24*col_dry)
        # tau_cld_lw = tau_cld_lw*(1e-24*col_dry)
        
        # tau_cld_sw = torch.clamp(tau_cld_sw, max=500.0)
    
        return out_new, out_sfc, rnn1_mem
    
    @torch.jit.export
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