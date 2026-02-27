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
from layers import LayerPressure, PressureThickness, LevelPressure
import torch.nn.functional as F
from typing import List, Tuple, Final, Optional
from torch import Tensor
from models_torch_kernels import GLU
from models_torch_kernels import *
import numpy as np 
from typing import Final 
import time 
from norm_coefficients import lbd_qi_lev, lbd_qc_lev, lbd_qn_lev
from settings import disable_compile

class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)

    def forward(self, input):
        return nn.functional.linear(input, self.log_weight.exp())
        
class LSTM_autoreg_torchscript_physrad(nn.Module):
    use_initial_mlp: Final[bool]
    use_intermediate_mlp: Final[bool]
    add_pres: Final[bool]
    add_stochastic_layer: Final[bool]
    output_prune: Final[bool]
    # ensemble_size: Final[int]
    use_ensemble: Final[bool]
    concat: Final[bool]
    use_lstm: Final[bool]
    mp_constraint: Final[bool]
    experimental_rad: Final[bool]
    physical_precip: Final[bool]
    predict_liq_frac: Final[bool]
    include_diffusivity: Final[bool]
    return_neg_precip: Final[bool]
    store_precip: Final[bool]
    ice_sedimentation: Final[bool]
    allow_extra_heating: Final[bool]
    use_existing_gas_optics_lw: Final[bool]
    use_existing_gas_optics_sw: Final[bool]
    reduce_lw_gas_optics: Final[bool]
    reduce_sw_gas_optics: Final[bool]
    update_q_for_rad: Final[bool]
    rad_cloud_masking: Final[bool]
    use_cloud_overlap_rnn: Final[bool]
    def __init__(self, hyam, hybm,  hyai, hybi,
                out_scale, out_sfc_scale, 
                xmean_lev, xmean_sca, xdiv_lev, xdiv_sca,
                device,
                nlev=60, nx = 15, nx_sfc=24, ny = 5, ny0=5, ny_sfc=5, nneur=(192,192), 
                gas_optics_model_lw=None,
                gas_optics_model_sw1=None,gas_optics_model_sw2=None,
                use_initial_mlp=False, 
                use_intermediate_mlp=True,
                add_pres=False,
                add_stochastic_layer=False,
                output_prune=False,
                repeat_mu=False,
                use_ensemble=False,
                concat=False,
                physical_precip=False,
                predict_liq_frac=False,
                # predict_flux=False,
                # ensemble_size=1,
                coeff_stochastic = 0.0,
                nh_mem=16,
                mp_mode=0):
        super(LSTM_autoreg_torchscript_physrad, self).__init__()
        self.ny = ny 
        self.ny0 = ny  #for physical precip option, need to distinguish between model outputs (ny) and intermediate outputs (ny0)
        self.nlev = nlev 
        self.nx_sfc = nx_sfc 
        self.ny_sfc = ny_sfc
        self.ny_sfc0 = ny_sfc
        self.nneur = nneur 
        self.use_initial_mlp=use_initial_mlp
        self.output_prune = output_prune
        self.add_pres = add_pres
        self.nx0 = nx
        self.nx_sfc0 = nx_sfc
        if self.add_pres:
            self.preslay = LayerPressure(hyam,hybm)
            nx = nx +1
        self.preslay_nonorm = LayerPressure(hyam,hybm,name="LayerPressure_nonorm", norm=False)
        self.presdelta = PressureThickness(hyai,hybi)
        self.preslev_nonorm = LevelPressure(hyai,hybi)
        # in ClimSim config of E3SM, the CRM physics first computes 
        # moist physics on 50 levels, and then computes radiation on 60 levels!
        self.use_intermediate_mlp=use_intermediate_mlp
        if self.use_intermediate_mlp:
            self.nh_mem = nh_mem
        else:
            self.nh_mem = self.nneur[1]
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
            self.store_precip=True
            # self.store_precip = False
        else:
            self.store_precip=False
        if self.store_precip: 
            self.nh_mem0 = self.nh_mem - 1 
        else:
            self.nh_mem0 = self.nh_mem 
        self.predict_liq_frac = predict_liq_frac 
        self.nx = nx
        self.nh_rnn1 = self.nneur[0]
        self.nx_rnn2 = self.nneur[0]
        self.nh_rnn2 = self.nneur[1]
        self.concat=concat
        self.repeat_mu = repeat_mu
        if self.repeat_mu:
            nx = nx + 1
        self.use_ensemble = use_ensemble
        if mp_mode==0:
          self.mp_constraint=False 
        elif mp_mode==1:
          self.mp_constraint=True
        elif mp_mode==-1 and self.physical_precip:
          print("Warning: combining physical_precip with physical radiation, experimental_rad")
        else:
          raise NotImplementedError("model requires mp_mode>=0")
        self.nlev = 60
        self.nlev_crm = 50
        self.ilev_crm = self.nlev-self.nlev_crm
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
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=2)
        self.softmax_dim1 = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.softsign =  nn.Softsign()

        yscale_lev = torch.from_numpy(out_scale).to(device)
        yscale_sca = torch.from_numpy(out_sfc_scale).to(device)
        yscale_sca_rad = yscale_sca[[0,1,4,5,6,7]]
        xmean_lev  = torch.from_numpy(xmean_lev).to(device)
        xmean_sca  = torch.from_numpy(xmean_sca).to(device)
        xdiv_lev   = torch.from_numpy(xdiv_lev).to(device)
        xdiv_sca   = torch.from_numpy(xdiv_sca).to(device)
 
        if gas_optics_model_sw1 is not None:
          self.gas_optics_model_sw1 = gas_optics_model_sw1
          self.gas_optics_model_sw2 = gas_optics_model_sw2
          self.use_existing_gas_optics_sw = True 
          print("Existing shortwave gas optics (absorption) model LOADED! Number of g-points: {}".format(self.gas_optics_model_sw1.ng))
          print("Existing shortwave gas optics (Rayleigh) model LOADED! Number of g-points: {}".format(self.gas_optics_model_sw2.ng))
        else:
          self.use_existing_gas_optics_sw = False 

        if gas_optics_model_lw is not None:
          self.gas_optics_model_lw = gas_optics_model_lw
          self.use_existing_gas_optics_lw = True 
          print("Existing longwave gas optics model LOADED! Number of g-points: {}".format(self.gas_optics_model_lw.ng))
        else:
          self.use_existing_gas_optics_lw = False 
        self.update_q_for_rad = True

        self.register_buffer('yscale_lev', yscale_lev)
        self.register_buffer('yscale_sca', yscale_sca)
        self.register_buffer('yscale_sca_rad', yscale_sca_rad)
        self.register_buffer('xmean_lev', xmean_lev)
        self.register_buffer('xmean_sca', xmean_sca)
        self.register_buffer('xdiv_lev', xdiv_lev)
        self.register_buffer('xdiv_sca', xdiv_sca)
        self.register_buffer('hyai', hyai)
        self.register_buffer('hybi', hybi)

        lbd_qc     = torch.tensor(lbd_qc_lev.reshape(60,1), device=device)
        lbd_qi     = torch.tensor(lbd_qi_lev.reshape(60,1), device=device)
        lbd_qn     = torch.tensor(lbd_qn_lev.reshape(60,1), device=device)
        self.register_buffer('lbd_qc', lbd_qc)
        self.register_buffer('lbd_qi', lbd_qi)
        self.register_buffer('lbd_qn', lbd_qn)
        # self.rnn_mem = None 
        print("Building RNN that feeds its hidden memory at t0,z0 to its inputs at t1,z0")
        print("Initial mlp: {}, intermediate mlp: {}".format(self.use_initial_mlp, self.use_intermediate_mlp))
 
        self.nx_rnn1 = self.nx_rnn1 + self.nh_mem0
        self.nh_rnn1 = self.nneur[0]
        
        print("nx rnn1", self.nx_rnn1, "nh rnn1", self.nh_rnn1)
        print("nx rnn2", self.nx_rnn2, "nh rnn2", self.nh_rnn2)  
        print("nx sfc", self.nx_sfc)

        if self.use_initial_mlp:
            self.mlp_initial = nn.Linear(nx, self.nneur[0])

        self.mlp_surface1  = nn.Linear(self.nx_sfc, self.nh_rnn1)

        self.use_lstm = False
        if self.use_lstm:
            rnn_layer = nn.LSTM 
            self.mlp_surface2  = nn.Linear(self.nx_sfc, self.nh_rnn1)
        else:
            rnn_layer = nn.GRU

        self.rnn1   = rnn_layer(self.nx_rnn1, self.nh_rnn1,  batch_first=True)  # (input_size, hidden_size)
        self.rnn2   = rnn_layer(self.nx_rnn2, self.nh_rnn2,  batch_first=True)
        if self.add_stochastic_layer:
            use_bias=False
            if self.use_lstm:
              self.rnn2 = MyStochasticLSTMLayer4(self.nh_rnn2, self.nh_rnn2, use_bias=use_bias)  
            else:
              self.rnn3 = MyStochasticGRULayer5(self.nh_rnn2, self.nh_rnn2, use_bias=use_bias)   

        nh_rnn = self.nh_rnn2

        self.rad_cloud_masking = True 
        self.use_cloud_overlap_rnn = False
        self.experimental_rad = False   
        self.allow_extra_heating=False
        self.ice_sedimentation = True
        # self.do_heat_advection = True 
        self.include_diffusivity= False
        if self.physical_precip:
            if self.store_precip:
              self.mlp_precip_release = nn.Linear(nh_rnn, 1)
            self.return_neg_precip = True
            self.mp_ncol = 16
            self.mlp_qv_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
            self.mlp_flux_qv_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
            self.mlp_flux_qn_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
            self.mlp_evap_cond_vapor_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
            if self.ice_sedimentation:
              self.mlp_sed_qn_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
            self.mlp_qtot_crm      = nn.Linear(self.nh_rnn2, self.mp_ncol)

            self.mlp_evap_prec_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)

            self.mlp_qn_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
            self.mlp_qi_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)

            self.mlp_mp_aa_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
            self.mlp_mp_aa2_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)

            # if self.do_heat_advection:
            self.mlp_t_crm = nn.Linear(self.nh_rnn2, self.mp_ncol)
            self.mlp_flux_crm_t = nn.Linear(self.nh_rnn2, self.mp_ncol)

            if self.include_diffusivity:
              self.conv_diff = nn.Conv1d(self.nx_rnn1, 1, 3, stride=1, padding="same")
        else: 
            self.return_neg_precip = False 
            self.include_diffusivity = False 
            self.mlp_surface_output = nn.Linear(nneur[-1], self.ny_sfc0)

        if self.predict_liq_frac:
          self.mlp_predfrac = nn.Linear(nh_mem, 1)

        if self.use_intermediate_mlp: 
            self.mlp_latent = nn.Linear(nh_rnn, self.nh_mem0)
            self.mlp_output = nn.Linear(self.nh_mem0, self.ny)
        else:
            self.mlp_output = nn.Linear(nh_rnn, self.ny)
        self.g = 9.806650000
        self.one_over_g = 0.1019716213
        self.cp =  1004.64
        # ORIGINAL RRTMGP-NN WITHOUT REDUCTION/DECODER (if equations are correct, should guarantee accurate radiation and associated heating at least in top 10 levels where there's no clouds)
        # self.ng_lw = 128
        self.ng_lw              = 16
        # self.ng_sw              = 112 
        self.ng_sw              = 16
        self.ng_tot             = self.ng_sw + self.ng_lw
        
        self.ny_sw_optprops     = 3 # tau_abs, tau_sca, g
        self.ny_lw_optprops     = 2  # tau_abs, planck_frac
        if self.physical_precip:
          self.nx_sw_optprops     = 4 + 6 + self.nh_mem #3*self.mp_ncol + 8# expanded ("CRM") qtot+qv+temp + gases + l.s. forcing?..
          self.nx_lw_optprops     = self.nx_sw_optprops
          self.reduce_sw_gas_optics = False
          self.reduce_lw_gas_optics = False
          if self.use_existing_gas_optics_lw or self.use_existing_gas_optics_sw:
            if self.use_existing_gas_optics_lw:
              # We already have a pre-trained NN for gas optics (RRTMGP-NN)
              # But we need MLPs to reduce the spectral dimension..
              if self.gas_optics_model_lw.ng != self.ng_lw:
                print("Number of spectral points in existing LW gas optics model ({}) doesn't match desired ({})," \
                          "using another MLP to map to desired".format(self.gas_optics_model_lw.ng, self.ng_lw))
                self.reduce_lw_gas_optics = True 
                self.gas_optics_lw_reduce1 = nn.Linear(self.gas_optics_model_lw.ng, self.ng_lw)
                self.gas_optics_lw_reduce2 = nn.Linear(self.gas_optics_model_lw.ng, self.ng_lw)
              # and we still need MLPs for cloud optical properties (only abs. optical depth)
              self.cloud_optics_lw = nn.Linear(2*self.mp_ncol+self.nh_mem, self.ng_lw)
              # self.cloud_optics_lw = nn.Conv1d(2*self.mp_ncol, self.ng_lw, 3, stride=1, padding=1)
              if self.use_cloud_overlap_rnn:
                self.cloud_mcica_scaling = nn.GRU(2*self.mp_ncol, 16,  batch_first=False) 
                self.cloud_mcica_scaling2 = nn.Linear(16, self.ng_lw)

            if self.use_existing_gas_optics_sw:
              if self.gas_optics_model_sw1.ng !=  self.ng_sw:
                print("Number of spectral points in existing SW gas optics model ({}) doesn't match desired ({})," \
                        "using another MLP to map to desired".format(self.gas_optics_model_sw1.ng, self.ng_sw))
                self.gas_optics_sw_reduce1 = nn.Linear(self.gas_optics_model_sw1.ng, self.ng_sw)
                self.gas_optics_sw_reduce2 = nn.Linear(self.gas_optics_model_sw2.ng, self.ng_sw)
                self.reduce_sw_gas_optics = True 
              else:
                from norm_coefficients import rrtmgp_sw_solar_source
                rrtmgp_sw_solar_source = rrtmgp_sw_solar_source/np.sum(rrtmgp_sw_solar_source)
                sw_solar_weights = torch.tensor(rrtmgp_sw_solar_source, device=device).unsqueeze(0)
                self.register_buffer('sw_solar_weights', sw_solar_weights)
              # and we still need MLPs for cloud optical properties (optical depth, ssa, g)
              self.cloud_optics_sw = nn.Linear(3*self.mp_ncol, 3*self.ng_sw)
              # self.cloud_optics_sw = nn.Conv1d(2*self.mp_ncol, 3*self.ng_sw, 3, stride=1, padding=1)
              # self.cloud_optics_sw = nn.GRU(2*self.mp_ncol, 3*self.ng_lw,  batch_first=False) 
              if self.use_cloud_overlap_rnn:
                self.cloud_mcica_scaling_sw = nn.GRU(2*self.mp_ncol, 16,  batch_first=False) 
                self.cloud_mcica_scaling_sw2 = nn.Linear(16, self.ng_sw)


              # self.cloud_optics_sw1 = nn.Linear(2*self.mp_ncol, 2*self.ng_sw)
              # self.cloud_optics_sw2 = nn.Linear(2*self.ng_sw, 3*self.ng_sw)

            print("Reduce LW: {} SW: {}".format(self.reduce_lw_gas_optics, self.reduce_sw_gas_optics))
        else:
          self.nx_sw_optprops     = 8 + self.nh_mem0 + 3 #1 #self.nh_mem 
          self.nx_lw_optprops     = 8 + self.nh_mem0 + 3 # 1 #  self.nh_mem #
          if self.use_existing_gas_optics_lw:
             raise NotImplementedError("use of existing gas optics model only implemented for physical_precip=True")

        # T, p, clw, cli, H2O, O3, CH4, N2O, +  CLOUD FRAC / HIDDEN VARIABLE(s) to account for cloud heterogeneity
        # In the longwave, the optical properties consist of absorption (gas + cloud) and emission (gas)
        # cloud LW scattering ignored like in most climate models 

        # not g or lw because aerosols were transparent and LW cloud scattering is ignored in original runs
        if not self.use_existing_gas_optics_sw:
          self.mlp_sw_optprops    = nn.Linear(self.nx_sw_optprops, self.ny_sw_optprops*self.ng_sw)
          # self.mlp_sw_optprops1    = nn.Linear(self.nx_sw_optprops, 2*self.ng_sw)
          # self.mlp_sw_optprops2    = nn.Linear(2*self.ng_sw, self.ny_sw_optprops*self.ng_sw)

        if not self.use_existing_gas_optics_lw:
          self.mlp_lw_optprops    = nn.Linear(self.nx_lw_optprops, self.ny_lw_optprops*self.ng_lw)
         

        if self.experimental_rad:
          # self.conv_vmat = nn.Conv1d(self.nh_rnn2, self.ng_sw, 2, stride=1)
          self.conv_vmat = nn.Conv1d(self.nh_mem, self.ng_sw, 2, stride=1)
        
        self.mlp_sfc_albedo_sw1  = nn.Linear(4, self.ng_sw)
        self.mlp_sfc_albedo_sw2  = nn.Linear(4, self.ng_sw)

        self.mlp_sfc_albedo_lw  = nn.Linear(2, self.ng_lw)
        if not (self.use_existing_gas_optics_sw and (not self.reduce_sw_gas_optics)): 
          self.sw_solar_weights   = nn.Parameter(torch.randn(1, self.ng_sw))

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
        
    # @torch.compile(dynamic=False, disable=disable_compile)
    def outgoing_lw(self, temp):
        # Stefan-Boltzmann constant (W/m²/K⁴)
        # sigma = 5.670374419e-8
        
        # Assuming emissivity = 1 (blackbody approximation)
        olr_exact = 5.670374419e-8 * torch.pow(temp,4)
        return olr_exact
  
    def interpolate_tlev_batchfirst(self, tlay, play, plev):
        ncol, nlay = tlay.shape
        device = tlay.device
        dtype = tlay.dtype
        # Initialize output arrays
        tlev = torch.zeros(ncol, nlay + 1, dtype=dtype, device=device)
        
        tlev[:,0] = tlay[:,0] + (plev[:,0]-play[:,0])*(tlay[:,1]-tlay[:,0]) / (play[:,1]-play[:,0])
        for ilay in range(1, nlay):
          tlev[:,ilay] = (play[:,ilay-1]*tlay[:,ilay-1]*(plev[:,ilay]-play[:,ilay]) \
                + play[:,ilay]*tlay[:,ilay]*(play[:,ilay-1]-plev[:,ilay])) /  (plev[:,ilay]*(play[:,ilay-1] - play[:,ilay]))
                                  
        tlev[:,nlay] = tlay[:,nlay-1] + (plev[:,nlay]-play[:,nlay-1])*(tlay[:,nlay-1]-tlay[:,nlay-2])  \
                / (play[:,nlay-1]-play[:,nlay-2])
                                 
        return tlev
    
    def interpolate_tlev_batchlast(self, tlay, play, plev):
        nlay, ncol = tlay.shape
        device = tlay.device
        dtype = tlay.dtype
        # Initialize output arrays
        tlev = torch.zeros(nlay + 1, ncol, dtype=dtype, device=device)
        
        tlev[0] = tlay[0] + (plev[0]-play[0])*(tlay[1]-tlay[0]) / (play[1]-play[0])
        for ilay in range(1, nlay):
          tlev[ilay] = (play[ilay-1]*tlay[ilay-1]*(plev[ilay]-play[ilay]) \
                + play[ilay]*tlay[ilay]*(play[ilay-1]-plev[ilay])) /  (plev[ilay]*(play[ilay-1] - play[ilay]))
                                  
        tlev[nlay] = tlay[nlay-1] + (plev[nlay]-play[nlay-1])*(tlay[nlay-1]-tlay[nlay-2])  \
                / (play[nlay-1]-play[nlay-2])
                                 
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

    @torch.compile(dynamic=False, disable=disable_compile)
    def reftrans_lw(self, planck_top, planck_bot, od_loc):
        # planck_top = source_lev[:,:,0:-1]
        # planck_bot = source_lev[:,:,1:]
        # planck_top = source_lev[0:-1,:,:]
        # planck_bot = source_lev[1:,:,:]
      
        od_loc = 1.66 * od_loc
        trans_lw = torch.exp(-od_loc)
        # Calculate coefficient for Padé approximant (vectorized)
        coeff = 0.2 * od_loc
        # Calculate mean Planck function (vectorized)
        planck_fl = 0.5 * (planck_top + planck_bot)
        # Calculate source terms using Padé approximant (vectorized)
        # one_minus_trans = 1.0 - trans_lw
        # one_plus_coeff = 1.0 + coeff
        source_dn = (1.0 - trans_lw) * (planck_fl + coeff * planck_bot) / (1.0 + coeff)
        source_up = (1.0 - trans_lw) * (planck_fl + coeff * planck_top) / (1.0 + coeff)
        return source_up, source_dn, trans_lw
    
    @torch.compile(dynamic=False, disable=disable_compile)
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
        od_over_mu0 = od/mu0 # torch.clamp(od / mu0, min=0.0)
        od_over_mu0 = torch.clamp(-od_over_mu0, min=-100.0)
    
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
    
    @torch.compile(dynamic=False, disable=disable_compile)
    def calc_ref_trans_sw(self,mu0,od,ssa,asymmetry):
        """
        Two-stream shortwave reflectance and transmittance calculation.
        Implements Meador & Weaver (1980) equations.

        Args:
            mu0:       Cosine of solar zenith angle (ncol*ng) (expanded in outer code)
            od:        Optical depth, shape (ncol*ng)
            ssa:       Single scattering albedo, shape (ncol*ng)
            asymmetry: Asymmetry factor, shape (ncol*ng)

        Returns:
            ref_diff:       Diffuse reflectance.
            trans_diff:     Diffuse transmittance.
            ref_dir:        Direct reflectance.
            trans_dir_diff: Direct-to-diffuse transmittance.
            trans_dir_dir:  Direct unscattered transmittance.
        """
        eps = torch.finfo(od.dtype).eps
        
        # ------------------------------------------------------------------ #
        # Unscattered direct transmittance
        # ------------------------------------------------------------------ #
        trans_dir_dir = torch.exp(-od / mu0)

        # ------------------------------------------------------------------ #
        # Two-stream gamma coefficients
        # ------------------------------------------------------------------ #
        factor  = 0.75 * asymmetry
        gamma1  = 2.0  - ssa * (1.25 + factor)
        gamma2  = ssa  * (0.75 - factor)
        gamma3  = 0.5  - mu0 * factor
        gamma4  = 1.0  - gamma3

        # alpha1 / alpha2  (Eqs. 16-17)
        alpha1 = gamma1 * gamma4 + gamma2 * gamma3
        alpha2 = gamma1 * gamma3 + gamma2 * gamma4

        # ------------------------------------------------------------------ #
        # Diffuse reflectance / transmittance  (Eqs. 25-26)
        # ------------------------------------------------------------------ #
        # k_exponent  (Eq. 18) — clamped for numerical safety
        k = torch.sqrt(torch.clamp((gamma1 - gamma2) * (gamma1 + gamma2), min=1.0e-4))

        exponential   = torch.exp(-k * od)
        exponential2  = exponential ** 2
        k_2_exp       = 2.0 * k * exponential

        reftrans_factor = 1.0 / (k + gamma1 + (k - gamma1) * exponential2)

        ref_diff   = gamma2 * (1.0 - exponential2) * reftrans_factor

        zeros=torch.zeros_like(ref_diff)
        trans_diff = torch.clamp(
            k_2_exp * reftrans_factor,
            min=zeros,
            max=1.0 - ref_diff,          # never exceeds 1 − ref_diff
        )
        trans_diff = torch.clamp(trans_diff, min=0.0)

        # ------------------------------------------------------------------ #
        # Direct reflectance / transmittance  (Eqs. 14-15)
        # ------------------------------------------------------------------ #
        k_mu0              = k * mu0
        one_minus_kmu0_sqr = 1.0 - k_mu0 ** 2
        k_gamma3           = k * gamma3
        k_gamma4           = k * gamma4

        # Guard against one_minus_kmu0_sqr ≈ 0 (mirrors Fortran's merge/epsilon)
        safe_denom = torch.where(
            one_minus_kmu0_sqr.abs() > eps,
            one_minus_kmu0_sqr,
            torch.full_like(one_minus_kmu0_sqr, eps),
        )
        # safe_denom = one_minus_kmu0_sqr.abs().clamp(min=eps) * one_minus_kmu0_sqr.sign()

        # reftrans_factor = mu0 * ssa * reftrans_factor / safe_denom
        reftrans_factor = ssa * reftrans_factor / safe_denom

        # Eq. 14
        ref_dir = reftrans_factor * (
              (1.0 - k_mu0) * (alpha2 + k_gamma3)
            - (1.0 + k_mu0) * (alpha2 - k_gamma3) * exponential2
            - k_2_exp * (gamma3 - alpha2 * mu0) * trans_dir_dir
        )

        # Eq. 15 (minus the direct unscattered term)
        trans_dir_diff = reftrans_factor * (
              k_2_exp * (gamma4 + alpha1 * mu0)
            - trans_dir_dir * (
                  (1.0 + k_mu0) * (alpha1 + k_gamma4)
                - (1.0 - k_mu0) * (alpha1 - k_gamma4) * exponential2
              )
        )

        # ------------------------------------------------------------------ #
        # Final clipping so that ref_dir + trans_dir_diff ≤ mu0*(1−T_dir_dir)
        # ------------------------------------------------------------------ #
        # max_direct = mu0 * (1.0 - trans_dir_dir)
        max_direct = (1.0 - trans_dir_dir)

        ref_dir        = torch.clamp(ref_dir,        min=zeros, max=max_direct)
        trans_dir_diff = torch.clamp(trans_dir_diff, min=zeros, max=max_direct - ref_dir)

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
    
    @torch.compile(dynamic=False, disable=disable_compile)
    def adding_ica_sw_batchlast(self, incoming_toa, emissivity_surf_diffuse, emissivity_surf_direct,
                    reflectance, transmittance, ref_dir, trans_dir_diff, trans_dir_dir):
            """
            Adding method for shortwave radiation with Independent Column Approximation (ICA).

            Returns:
                tuple: (flux_up, flux_dn_diffuse, flux_dn_direct) each of shape [ncol, nlev+1]
            """
            
            nlev, ncol = reflectance.shape
            device = reflectance.device
            dtype = reflectance.dtype
            
            # Initialize output arrays
            flux_up = torch.zeros(nlev + 1, ncol, dtype=dtype, device=device)
            flux_dn_diffuse = torch.zeros(nlev + 1, ncol, dtype=dtype, device=device)
            flux_dn_direct = torch.zeros(nlev + 1, ncol, dtype=dtype, device=device)
            
            # Initialize working arrays
            albedo = torch.zeros(nlev + 1, ncol, dtype=dtype, device=device)
            source = torch.zeros(nlev + 1, ncol, dtype=dtype, device=device)
            inv_denominator = torch.zeros(nlev, ncol,  dtype=dtype, device=device)
            
            # Compute profile of direct (unscattered) solar fluxes at each half-level  by working down through the atmosphere
            flux_dn_direct[0, :] = incoming_toa
            for jlev in range(nlev):
                flux_dn_direct[jlev + 1,:] = flux_dn_direct[jlev,:].clone() * trans_dir_dir[jlev,:]
            
            # Set surface albedo
            albedo[nlev] = emissivity_surf_diffuse
            
            # At the surface, the direct solar beam is reflected back into the diffuse stream
            source[nlev] = emissivity_surf_direct * flux_dn_direct[nlev] #* cos_sza
            
            # Work back up through the atmosphere and compute the albedo of the entire
            # earth/atmosphere system below that half-level, and also the "source"
            for jlev in range(nlev - 1, -1, -1):  # nlev down to 1 in Fortran indexing
                # Lacis and Hansen (1974) Eq 33, Shonk & Hogan (2008) Eq 10:
                # inv_denominator[:, jlev] = 1.0 / (1.0 - albedo[:, jlev + 1].clone() * reflectance[:, jlev])
                
                albedoplusone = albedo[jlev + 1].clone()
                inv_denominator[jlev] = 1.0 / (1.0 - albedoplusone * reflectance[jlev])
    
                # Shonk & Hogan (2008) Eq 9, Petty (2006) Eq 13.81:
                # albedo[:, jlev] = (reflectance[:, jlev] + 
                #                   transmittance[:, jlev] * transmittance[:, jlev] * 
                #                   albedo[:, jlev + 1] * inv_denominator[:, jlev])
                
                inv_denom = inv_denominator[jlev].clone()
                albedo[jlev] = (reflectance[jlev] + 
                                  torch.square(transmittance[jlev]) * 
                                  albedoplusone * inv_denom)
    
                # Shonk & Hogan (2008) Eq 11:
                fluxdndir = flux_dn_direct[jlev].clone()
                source[jlev] = (ref_dir[jlev] * fluxdndir +
                                  transmittance[jlev] * 
                                  (source[jlev + 1] + 
                                  albedoplusone * trans_dir_diff[jlev] * fluxdndir) *
                                  inv_denom)
            
            # At top-of-atmosphere there is no diffuse downwelling radiation
            flux_dn_diffuse[0] = 0.0
            
            # At top-of-atmosphere, all upwelling radiation is due to scattering
            # by the direct beam below that level
            flux_up[0] = source[0]
            
            # Work back down through the atmosphere computing the fluxes at each half-level
            for jlev in range(nlev):  # 1 to nlev in Fortran indexing
                # Shonk & Hogan (2008) Eq 14 (after simplification):
                fluxdndir = flux_dn_direct[jlev].clone()
                fluxdndiff = flux_dn_diffuse[jlev].clone()
    
                flux_dn_diffuse[jlev + 1] = ((transmittance[jlev] * fluxdndiff +  
                                                 reflectance[jlev] * source[jlev + 1] + 
                                                 trans_dir_diff[jlev] * fluxdndir) *inv_denominator[jlev])      
          
                # Shonk & Hogan (2008) Eq 12:
                flux_up[jlev + 1] = (albedo[jlev + 1] * flux_dn_diffuse[jlev + 1].clone() +
                                       source[jlev + 1])
                
                # Apply cosine correction to direct flux
                # flux_dn_direct[:, jlev] = fluxdndir * cos_sza
            
            # Final cosine correction for surface direct flux
            # flux_dn_direct[:, nlev] = flux_dn_direct[:, nlev] * cos_sza
            
            return flux_up, flux_dn_diffuse, flux_dn_direct
       
    @torch.compile(dynamic=False, disable=disable_compile)
    def adding_ica_sw_batchlast_opt(self, incoming_toa, emissivity_surf_diffuse, emissivity_surf_direct,
                    reflectance, transmittance, ref_dir, trans_dir_diff, trans_dir_dir):
            """
            Adding method for shortwave radiation

            Returns:
                tuple: (flux_up, flux_dn_diffuse, flux_dn_direct) each of shape [ncol, nlev+1]
            """
            
            nlev, ncol = reflectance.shape
            device = reflectance.device
            
            # Set surface albedo
            albedo = torch.jit.annotate(List[Tensor], [])
            albedo0 = emissivity_surf_diffuse
            albedo += [albedo0]

            albedodir = torch.jit.annotate(List[Tensor], [])
            albedodir0 = emissivity_surf_direct
            albedodir += [albedodir0]

            # Work up through the atmosphere and compute the albedo of the entire earth/atmosphere system below that half-level
            for jlev in range(nlev-1, -1, -1):  # nlev down to 1 in Fortran indexing

              # comparing ecRad Tripleclouds code to the McICA code, "source" variable  is like fluxdndir*albedodir
              # If we use albedodir instead (like in TripleClouds), we dont need to precompute fluxdndir in a separate loop, so just two vertical loops
              # Adapted from https://github.com/ecmwf-ifs/ecrad/blob/master/radiation/radiation_tripleclouds_sw.F90
              albedodir0 = (ref_dir[jlev] +
                                      (trans_dir_dir[jlev]*albedodir0 + trans_dir_diff[jlev]*albedo0) *
                                      transmittance[jlev] / (1.0 - albedo0 * reflectance[jlev])) #* inv_denom)  
              albedodir  += [albedodir0]  
              
              albedo0 = (reflectance[jlev] + torch.square(transmittance[jlev]) * albedo0  / (1.0 - albedo0 * reflectance[jlev])) 
              albedo  += [albedo0]

            albedo.reverse(); albedodir.reverse()
            
            # At top-of-atmosphere, all upwelling radiation is due to scattering by the direct beam below that level
            fluxup = incoming_toa*albedodir[0]
            flux_up = torch.jit.annotate(List[Tensor], [])
            flux_up += [fluxup]

            fluxdndir = incoming_toa
            flux_dn_direct = torch.jit.annotate(List[Tensor], [])
            flux_dn_direct += [fluxdndir]

            # At top-of-atmosphere there is no diffuse downwelling radiation
            fluxdndiff = torch.zeros_like(incoming_toa)
            flux_dn_diffuse = torch.jit.annotate(List[Tensor], [])
            flux_dn_diffuse += [fluxdndiff]

            # Work back down through the atmosphere computing the fluxes at each half-level
            for jlev in range(nlev):  # 1 to nlev in Fortran indexing

                fluxdndiff = (transmittance[jlev]*fluxdndiff + fluxdndir 
                     * (transmittance[jlev]*albedodir[jlev + 1]*reflectance[jlev]  + trans_dir_diff[jlev] ) 
                     / (1.0-  reflectance[jlev]*albedo[jlev + 1])) 

                # flux_dn_direct[jlev + 1] = fluxdndir * trans_dir_dir[jlev,:]
                fluxdndir =  fluxdndir * trans_dir_dir[jlev,:]
                flux_dn_direct  += [fluxdndir]
                flux_dn_diffuse += [fluxdndiff]

                # flux_up[jlev+1] =  fluxdndir*albedodir[jlev+1] + fluxdndiff* albedo[jlev + 1]
                fluxup = fluxdndir*albedodir[jlev+1] + fluxdndiff* albedo[jlev + 1]
                flux_up += [fluxup]       
                # Apply cosine correction to direct flux
                # flux_dn_direct[:, jlev] = fluxdndir * cos_sza
            
            flux_dn_direct  = torch.stack(flux_dn_direct)
            flux_dn_diffuse = torch.stack(flux_dn_diffuse)
            flux_up = torch.stack(flux_up)

            # Final cosine correction for surface direct flux
            # flux_dn_direct[:, nlev] = flux_dn_direct[:, nlev] * cos_sza
            
            return flux_up, flux_dn_diffuse, flux_dn_direct
  
    @torch.compile(dynamic=False, disable=disable_compile)
    def adding_tc_sw_batchlast(self, incoming_toa, emissivity_surf_diffuse, emissivity_surf_direct,
                    reflectance, transmittance, ref_dir, trans_dir_diff, trans_dir_dir, V):
            """
            Adding method for shortwave radiation with TripleClouds
            
            Args:
                incoming_toa: Incoming downwelling solar radiation at TOA (tensor of shape [ncol])
                emissivity_surf_diffuse: Surface albedo to diffuse radiation (tensor of shape [ncol])
                emissivity_surf_direct: Surface albedo to direct radiation (tensor of shape [ncol])
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
            
            nlev, ncol= reflectance.shape
            device = reflectance.device
            dtype = reflectance.dtype
            
            # Initialize output arrays
            flux_up = torch.zeros(nlev + 1, ncol, dtype=dtype, device=device)
            flux_dn_diffuse = torch.zeros(nlev + 1, ncol, dtype=dtype, device=device)
            flux_dn_direct = torch.zeros(nlev + 1, ncol, dtype=dtype, device=device)
            
            # Initialize working arrays
            A = torch.zeros(nlev + 1, ncol, dtype=dtype, device=device) # albedo
            Adir = torch.zeros(nlev + 1, ncol, dtype=dtype, device=device) # albedo

            source = torch.zeros(nlev + 1, ncol, dtype=dtype, device=device)
            inv_denominator = torch.zeros(nlev, ncol, dtype=dtype, device=device)
            
            # Compute profile of direct (unscattered) solar fluxes at each half-level
            # by working down through the atmosphere
            flux_dn_direct[0] = incoming_toa
            for jlev in range(nlev):
                flux_dn_direct[jlev + 1] = flux_dn_direct[jlev].clone() * trans_dir_dir[jlev]
            
            # Set surface albedo
            A[nlev] = emissivity_surf_diffuse
            Adir[nlev] = emissivity_surf_direct

            # At the surface, the direct solar beam is reflected back into the diffuse stream
            source[nlev] = emissivity_surf_direct * flux_dn_direct[nlev] #* cos_sza
            
            # Work back up through the atmosphere and compute the albedo of the entire
            # earth/atmosphere system below that half-level, and also the "source"
            for jlev in range(nlev - 1, -1, -1):  # nlev down to 1 in Fortran indexing
                # Lacis and Hansen (1974) Eq 33, Shonk & Hogan (2008) Eq 10:
                # inv_denominator[jlev] = 1.0 / (1.0 - albedo[jlev + 1].clone() * reflectance[jlev])
                
                Aplusone = A[jlev + 1].clone()
                Adirplusone = Adir[jlev + 1].clone()

                inv_denominator[jlev] = 1.0 / (1.0 - Aplusone * reflectance[jlev])
    
                # Shonk & Hogan (2008) Eq 9, Petty (2006) Eq 13.81:
                # albedo[jlev] = (reflectance[jlev] + 
                #                   transmittance[jlev] * transmittance[jlev] * 
                #                   albedo[jlev + 1] * inv_denominator[jlev])
                
                inv_denom = inv_denominator[jlev].clone()
                A[jlev] = (reflectance[jlev] + torch.square(transmittance[jlev]) * Aplusone * inv_denom)*V[jlev]
                Adir[jlev] = (ref_dir[jlev] + (trans_dir_dir[jlev]*Adirplusone + trans_dir_diff[jlev]*Aplusone)  * 
                            transmittance[jlev] * inv_denom)*V[jlev]

                # TripleClouds :
                # ng_nreg = self.nreg//3 
                # A[0:ng_nreg,jlev]           = A[0:ng_nreg,jlev]*V[1,1] + A[ng_nreg:2*ng_nreg,jlev]*V[2,1] + A[2*ng_nreg:3*ng_nreg,jlev]*V[3,1]
                # A[ng_nreg:2*ng_nreg,jlev]   = A[0:ng_nreg,jlev]*V[1,2] + A[ng_nreg:2*ng_nreg,jlev]*V[2,2] + A[2*ng_nreg:3*ng_nreg,jlev]*V[3,2]
                # A[2*ng_nreg:3*ng_nreg,jlev] = A[0:ng_nreg,jlev]*V[1,3] + A[ng_nreg:2*ng_nreg,jlev]*V[2,3] + A[2*ng_nreg:3*ng_nreg,jlev]*V[3,3]         
                # A[jlev]           = A[jlev]*V[jlev]
            
                # Shonk & Hogan (2008) Eq 11:
                fluxdndir = flux_dn_direct[jlev].clone()
                source[jlev] = (ref_dir[jlev] * fluxdndir +
                                  transmittance[jlev] * 
                                  (source[jlev + 1] + 
                                  Aplusone * trans_dir_diff[jlev] * fluxdndir) *
                                  inv_denom)
            
            # At top-of-atmosphere there is no diffuse downwelling radiation
            flux_dn_diffuse[0] = 0.0
            
            # At top-of-atmosphere, all upwelling radiation is due to scattering
            # by the direct beam below that level
            flux_up[0] = source[0]
            
            # Work back down through the atmosphere computing the fluxes at each half-level
            for jlev in range(nlev):  # 1 to nlev in Fortran indexing
                # Shonk & Hogan (2008) Eq 14 (after simplification):
                fluxdndir = flux_dn_direct[jlev].clone()
                fluxdndiff = flux_dn_diffuse[jlev].clone()
    
                flux_dn_diffuse[jlev + 1] = ((transmittance[jlev] * fluxdndiff +  
                                                 reflectance[jlev] * source[jlev + 1] + 
                                                 trans_dir_diff[jlev] * fluxdndir) *
                                               inv_denominator[jlev])      
          
                # Shonk & Hogan (2008) Eq 12:
                flux_up[jlev + 1] = (A[jlev + 1] * flux_dn_diffuse[jlev + 1].clone() +
                                       source[jlev + 1])
                
                # Apply cosine correction to direct flux
                # flux_dn_direct[jlev] = fluxdndir * cos_sza
            
            # Final cosine correction for surface direct flux
            # flux_dn_direct[nlev] = flux_dn_direct[nlev] * cos_sza
            
            return flux_up, flux_dn_diffuse, flux_dn_direct

    @torch.compile(dynamic=False, disable=disable_compile)
    def lw_solver_noscat_batchlast(self, trans_lw, source_dn, source_up, source_sfc, emissivity_surf):
        
        nlev = trans_lw.shape[0]
        
        # At top-of-atmosphere there is no diffuse downwelling radiation
        flux_lw_dn0 = torch.zeros_like(emissivity_surf)
        flux_lw_dn = torch.jit.annotate(List[Tensor], [])
        flux_lw_dn += [flux_lw_dn0]

        # Work down through the atmosphere computing the downward fluxes
        # at each half-level (vectorized over columns)
        for jlev in range(nlev):
            # flux_lw_dn[jlev + 1] = (trans_lw[jlev] * flux_lw_dn[jlev].clone()  + 
            #                        source_dn[jlev])
            flux_lw_dn0 = (trans_lw[jlev] * flux_lw_dn0 + source_dn[jlev])
            flux_lw_dn += [flux_lw_dn0]

        # flux_lw_up[nlev] = source_sfc + albedo_surf * flux_lw_dn[nlev]
        #                                              albedo
        flux_lw_up0   = emissivity_surf*source_sfc +  (1-emissivity_surf) * flux_lw_dn[nlev]
        flux_lw_up    = torch.jit.annotate(List[Tensor], [])
        flux_lw_up    += [flux_lw_up0]

        flux_lw_dn = torch.stack(flux_lw_dn)

        # Work back up through the atmosphere computing the upward fluxes
        # at each half-level (vectorized over columns)
        for jlev in range(nlev - 1, -1, -1):
            # flux_lw_up[jlev] = (trans_lw[jlev] * flux_lw_up[jlev + 1].clone()  + 
            #                    source_up[jlev])
            flux_lw_up0 = (trans_lw[jlev] * flux_lw_up0  + source_up[jlev])    
            flux_lw_up += [flux_lw_up0]

        flux_lw_up.reverse()
        flux_lw_up  = torch.stack(flux_lw_up)
        return flux_lw_dn, flux_lw_up
    
    def microphysics_decode(self, inputs_main, inputs_denorm, delta_plev, play, plev, P_old, out, rnn_mem, rnn2out, last_h, out_new):
        # self.ilev_crm = 10
        # self.ilev_crm = 12
        batch_size, nlev, ny = out_new.shape 

        # out_new = torch.zeros(batch_size, self.nlev, self.ny, device=inputs_main.device)
        # g = torch.tensor(9.806650000)
        # one_over_g = torch.tensor(0.1019716213)
        scaling_factor = -self.g # tendency equation in pressure coordinates has -g in front
        # pres_diff = plev[:,self.ilev_crm+1:] - plev[:,self.ilev_crm:-1]
        pres_diff = delta_plev[:,self.ilev_crm:]
        zeroes = torch.zeros(batch_size, 1, device=out_new.device)

        Tsfc = inputs_denorm[:,-1,0]
        Pmax = 1000 *  self.yscale_sca[3] * 5.58e-18*torch.exp(0.077*Tsfc)
        
        #  ['ptend_t', 'dqv', 'dqn', 'liq_frac', 'ptend_u', 'ptend_v']
        if self.allow_extra_heating:
          out_new[:,self.ilev_crm+2:,0] = out[:,2:,0]
        out_new[:,self.ilev_crm+2:,3:self.ny] = out[:,2:,3:self.ny]

        dqv_evap_prec       = self.mlp_evap_prec_crm(rnn2out)

        dq_cond_evap_vapor = self.mlp_evap_cond_vapor_crm(rnn2out)

        flux_mult_coeff = 3.0e5

        qv_old0 =  inputs_denorm[:,self.ilev_crm:,-1]
        qn_old0 = (inputs_denorm[:,self.ilev_crm:,2] + inputs_denorm[:,self.ilev_crm:,3])
        qi_old0 = (inputs_denorm[:,self.ilev_crm:,3])

        if self.include_diffusivity:
          # D = out[:,self.ilev_crm:,1]
          # if not self.separate_radiation:
          #   inputs_main_crm = inputs_main_crm[:,self.ilev_crm:]
          inputs_main_crm = torch.transpose(inputs_main_crm,1,2)
          D = self.conv_diff(inputs_main_crm)
          D = torch.squeeze(torch.transpose(D,1,2))
          qv_old2 =  self.yscale_lev[self.ilev_crm-1:,1].unsqueeze(0) * inputs_denorm[:,self.ilev_crm-1:,-1]
          qn_old2 = self.yscale_lev[self.ilev_crm-1:,2].unsqueeze(0) * (inputs_denorm[:,self.ilev_crm-1:,2] + inputs_denorm[:,self.ilev_crm-1:,3])
          T_old2 = self.yscale_lev[self.ilev_crm-1:,0].unsqueeze(0) * inputs_denorm[:,self.ilev_crm-1:,0]
          diff_qv = D *( (qv_old2[:,1:] - qv_old2[:,0:-1]) / pres_diff.squeeze()) 
          diff_qn = D * ( (qn_old2[:,1:] - qn_old2[:,0:-1]) / pres_diff.squeeze()) 
          diff_t =  D * ( (T_old2[:,1:] - T_old2[:,0:-1]) / pres_diff.squeeze()) 
          preslay_diff = play[:,self.ilev_crm+1:] - play[:,self.ilev_crm:-1]
          diff_qv = (diff_qv[:,1:] - diff_qv[:,0:-1]) / preslay_diff.squeeze()
          diff_qn = (diff_qn[:,1:] - diff_qn[:,0:-1]) / preslay_diff.squeeze()
          diff_t =  (diff_t[:,1:] - diff_t[:,0:-1]) / preslay_diff.squeeze()

        flux1 = self.mlp_flux_qv_crm(rnn2out)
        flux2 = self.mlp_flux_qn_crm(rnn2out)

        p_qv_crm = self.softmax(self.mlp_qv_crm(rnn2out)) # here p is the fraction of the grid-scale total q that is in each pseudo-CRM column
        p_qn_crm = self.softmax(self.mlp_qn_crm(rnn2out)) 

        qn_crm = self.mp_ncol*qn_old0.unsqueeze(2)*p_qn_crm 
        qv_crm = self.mp_ncol*qv_old0.unsqueeze(2)*p_qv_crm 

        qn_old = qn_crm 
        qv_old = qv_crm

        zeroes_crm = torch.zeros(batch_size, 1, self.mp_ncol, device=inputs_main.device)

        # T_old = inputs_denorm[:,self.ilev_crm:,0] #5e-2*inputs_denorm[:,self.ilev_crm:,0]
        tlev  = self.interpolate_tlev_batchfirst(inputs_denorm[:,:,0], play.squeeze(), plev.squeeze()) # (nb, nlev+1)
        T_old = tlev[:,self.ilev_crm+1:]
        preslay_diff0 = play[:,self.ilev_crm:] - play[:,self.ilev_crm-1:-1]
        flux3 = self.mlp_flux_crm_t(rnn2out)
        p_crm_t_new = self.softmax(self.mlp_t_crm(rnn2out))
        crm_t_new = self.mp_ncol*T_old.unsqueeze(2)*p_crm_t_new
        # if self.do_heat_advection: 
        # flux_net_t = flux3*crm_t_new
        flux_net_t = flux3*102.34*crm_t_new*preslay_diff0 #  cp/g · T · Δp 
        flux_net_t[:,-1] = -self.relu(flux_net_t[:,-1]) # net downward flux at sfc must be upwards or zero
        # flux_net_t = torch.cat((zeroes_crm,flux_net_t[:,0:-1], zeroes_crm),dim=1)
        flux_net_t = torch.cat((zeroes_crm,flux_net_t),dim=1)
        flux_t_dp = (scaling_factor/self.cp)*( (flux_net_t[:,1:] - flux_net_t[:,0:-1]) / pres_diff) 
        del flux3, flux_net_t 

        flux_net_qv = flux_mult_coeff*flux1*qv_crm#*torch.reshape(self.yscale_lev[self.ilev_crm:,1],(1,-1,1))              
        flux_net_qn = flux_mult_coeff*flux2*qn_crm#*torch.reshape(self.yscale_lev[self.ilev_crm:,1],(1,-1,1))  
        if self.ice_sedimentation:
          p_crm_qi_old = self.softmax(self.mlp_qi_crm(rnn2out))  
          crm_qi_old = self.mp_ncol*qi_old0.unsqueeze(2)*p_crm_qi_old 
          sed = self.relu(self.mlp_sed_qn_crm(rnn2out))
          sed = sed*self.g*crm_qi_old*torch.reshape(self.yscale_lev[self.ilev_crm:,2],(1,-1,1))
          sedimentation = torch.mean( sed[:,-1], 1)
          sed = torch.cat((zeroes_crm,sed),dim=1)
          sed_qn_dp = scaling_factor*( (sed[:,1:] - sed[:,0:-1]) / pres_diff) 
        else:
          sedimentation = 0

        # ------------------------ experimental -------------------------------------
        # rh = inputs_denorm[:,self.ilev_crm:,1:2]
        # dq_cond_evap_vapor = dq_cond_evap_vapor*rh
        # ------------------------ experimental -------------------------------------

        flux_net_qv = torch.cat((zeroes_crm,flux_net_qv[:,0:-1], zeroes_crm),dim=1)
        flux_net_qn = torch.cat((zeroes_crm,flux_net_qn[:,0:-1], zeroes_crm),dim=1)

        del flux1, flux2, p_qn_crm, p_qv_crm, p_crm_qi_old 

        flux_qv_dp = scaling_factor*( (flux_net_qv[:,1:] - flux_net_qv[:,0:-1]) / pres_diff) 
        flux_qn_dp = scaling_factor*( (flux_net_qn[:,1:] - flux_net_qn[:,0:-1]) / pres_diff) 
        del flux_net_qv, flux_net_qn

        dqv_evap_prec = self.relu(dqv_evap_prec) + 1.0e-6 # force positive
        if self.store_precip:
          # Relate evaporated precipitation to stored precipitation here?
          P_old_vertical = out[:,:,2] # self.mlp_prec_vertical(rnn2out)
          # P_old_vertical = rnn_mem[:,self.ilev_crm:,0] 
          P_old_vertical = self.softmax_dim1(P_old_vertical) * P_old.unsqueeze(1) # sums to P_old
          dqv_evap_prec = dqv_evap_prec*P_old_vertical.unsqueeze(2) # P_old.unsqueeze(1)

        alpha = self.relu(self.mlp_mp_aa_crm(rnn2out)) #+ 1.0e-6 # force positive
        if True: # if we want the acc-au term to be proportional to qn, option to "redistribute" qn first vertically
          q_old_sum = torch.sum(qn_old0,dim=1)
          qn_old_redist = out[:,:,1]
          qn_old2 = torch.reshape(self.softmax_dim1(qn_old_redist) * q_old_sum.unsqueeze(1), (batch_size, -1, 1))
          # qn_crm = qn_old2
          dqn_aa = alpha*qn_old2*torch.reshape(self.yscale_lev[self.ilev_crm:,2],(1,-1,1)) 
        else:
          dqn_aa = alpha*qn_crm*torch.reshape(self.yscale_lev[self.ilev_crm:,2],(1,-1,1)) 
          # dqn_aa = alpha*torch.square(qn_crm)*torch.reshape(self.yscale_lev[self.ilev_crm:,2],(1,-1,1)) 
          # dqn_aa = alpha*torch.pow(qn_crm, 1.5)*torch.reshape(self.yscale_lev[self.ilev_crm:,2],(1,-1,1)) 

        if True:
          # first ensure positive qn by clamping dq_cond_evap_vapor
          if self.ice_sedimentation:
              minval = -(self.yscale_lev[self.ilev_crm:,2:3]*qn_old/1200) - flux_qn_dp + dqn_aa - sed_qn_dp
          else:
              minval = -(self.yscale_lev[self.ilev_crm:,2:3]*qn_old/1200) - flux_qn_dp + dqn_aa
          dq_cond_evap_vapor = torch.clamp(dq_cond_evap_vapor, min=minval)
  
          # then ensure positive qv by clamping dqv_evap_prec
          minval = -(self.yscale_lev[self.ilev_crm:,1:2]*qv_old/1200) - flux_qv_dp + dq_cond_evap_vapor
          dqv_evap_prec = torch.clamp(dqv_evap_prec, min=minval)

          # maximum cloud water limit, make sure qn_new doesn't exceed qn_max
          # dqn_normed < yscale*(qnmax - qnold)/1200
          # flux_qn_dp + dq_cond_evap_vapor   - dqn_aa < yscale*(qnmax - qnold)/1200
          # flux_qn_dp + dq_cond_evap_vapor < yscale*(qnmax - qnold)/1200 + dqn_aa
          #  dqn_aa > flux_qn_dp + dq_cond_evap_vapor -yscale*(qnmax - qnold)/1200 
          qn_max = 0.0006
          if self.ice_sedimentation:
              minval = flux_qn_dp + dq_cond_evap_vapor + sed_qn_dp -(self.yscale_lev[self.ilev_crm:,2:3]*(qn_max - qn_old)/1200) 
          else:
              minval = flux_qn_dp + dq_cond_evap_vapor -(self.yscale_lev[self.ilev_crm:,2:3]*(qn_max - qn_old)/1200) 
          dqn_aa = torch.clamp(dqn_aa, min=minval)    

        #                      (cond-evap)<0 from vapor,  evap. from prec. both add water vapor  
        dqv_crm =   flux_qv_dp - dq_cond_evap_vapor     + dqv_evap_prec 
        dqn_crm =   flux_qn_dp + dq_cond_evap_vapor     - dqn_aa

        if self.ice_sedimentation: 
          dqn_crm =  dqn_crm + sed_qn_dp 
          
        out_new[:,self.ilev_crm:,1] = torch.mean(dqv_crm, 2)
        out_new[:,self.ilev_crm:,2] = torch.mean(dqn_crm, 2)
                
        if self.include_diffusivity:
          out_new[:,self.ilev_crm:-1,0] = out_new[:,self.ilev_crm:-1,0] + diff_t
          out_new[:,self.ilev_crm:-1,1] = out_new[:,self.ilev_crm:-1,1] + diff_qv
          out_new[:,self.ilev_crm:-1,2] = out_new[:,self.ilev_crm:-1,2] + diff_qn

        # if self.do_heat_advection: 
        out_new[:,self.ilev_crm:,0] = out_new[:,self.ilev_crm:,0] + torch.mean(flux_t_dp, 2)

        # temp_dyn = forcing_factor*inputs_denorm[:,self.ilev_crm:,6]*self.yscale_lev[self.ilev_crm:,0]
        # out_new[:,self.ilev_crm:,0] = out_new[:,self.ilev_crm:,0] + temp_dyn
        # # print("temp dyn mean", temp_dyn[:,self.ilev_crm].abs().mean().item(), "tot", out_new[:,self.ilev_crm:,0].abs().mean().item())
        # print("temp dyn max", temp_dyn[:,self.ilev_crm].abs().max().item(), "tot", out_new[:,self.ilev_crm:,0].abs().max().item())

        # net_condensation = torch.mean(dq_cond_evap_vapor - dqv_evap_prec, 2)
        # Lv_over_cp =  2490.04 # 2.5e6/self.cp = 2.5 × 10⁶ J/kg at 0°C  divided by self.cp J/kg/K
        # net_condensation = Lv_over_cp*(net_condensation/self.yscale_lev[self.ilev_crm:,1]) * self.yscale_lev[self.ilev_crm:,0]
        temp = inputs_denorm[:,self.ilev_crm:,0:1].squeeze() + (out_new[:,self.ilev_crm:,0]/self.yscale_lev[self.ilev_crm:,0]) * 1200
        liq_frac    = torch.unsqueeze(self.temperature_scaling(temp),2)
        ice_frac = 1 - liq_frac
        dq_cond_evap_vapor = (liq_frac*(2.5104e6/self.cp) + ice_frac*(2.8440e6/self.cp))*dq_cond_evap_vapor
        net_condensation = torch.mean(dq_cond_evap_vapor- (2.5104e6/self.cp)*dqv_evap_prec, 2)
        net_condensation = (net_condensation/self.yscale_lev[self.ilev_crm:,1]) * self.yscale_lev[self.ilev_crm:,0]
        out_new[:,self.ilev_crm:,0] = out_new[:,self.ilev_crm:,0] + net_condensation

        #                      source,       sink   of precipitation  (note: signs already reversed w.r.t. above)
        d_precip_sourcesink    = torch.mean(dqn_aa      - dqv_evap_prec,2)  
        # This should be positive to avoid negative precipitation!
        # out = out_new   

        #   dp_water = (one_over_g*pres_diff*d_precip_sourcesink)
        water_new = torch.sum((self.one_over_g*pres_diff.squeeze()*d_precip_sourcesink),1)  

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
          # if self.pour_excess:
          water_excess = water_stored - Pmax
          water_excess = self.relu(water_excess)
          water_stored = water_stored - water_excess
          water_stored_lev  = torch.unsqueeze(water_stored,dim=1)
          water_stored_lev = torch.unsqueeze(torch.repeat_interleave(water_stored_lev,self.nlev_mem,dim=1),dim=2)
          rnn_mem = torch.cat((rnn_mem[:,:,0:self.nh_mem], water_stored_lev),dim=2)
          # if self.pour_excess:
          precip=  sedimentation + water_released + water_excess # - prec_negative
          # else:
          #     precip=  sedimentation + water_released # - prec_negative
        else:
          prec_negative = self.relu(-water_new) # punish model for diagnosing negative precip from column water changes?
          water_new = self.relu(water_new)
          precip =  sedimentation + water_new  # <-- we already reversed signs in d_precip_sourcesink

        precc = (precip/1000).unsqueeze(1)

        temp_sfc = (inputs_main[:,-1,0:1]*self.xdiv_lev[-1,0:1]) + self.xmean_lev[-1,0:1]
        snowfrac = self.temperature_scaling_precip(temp_sfc)
        precsc = snowfrac*precc

        qv_crm = self.relu(qv_crm + dqv_crm/self.yscale_lev[self.ilev_crm:,1:2])
        qn_crm = self.relu(qn_crm + dqn_crm/self.yscale_lev[self.ilev_crm:,2:3])

        return out_new, precc, precsc, rnn_mem, qn_crm, qv_crm, p_crm_t_new, prec_negative
    
    def radiative_transfer(self, inputs_main, inputs_aux, inputs_denorm, play, plev, delta_plev, rnn_mem, rnn2out, qn_crm, qv_crm, p_crm_t_new, out_new):

      
      batch_size, nlev, ny = out_new.shape 
      device = out_new.device
      
      T_before        = inputs_denorm[:,:,0:1] 

      # ----------------- PHYSICAL RADIATIVE TRANSFER ----------------

      out_denorm    = out_new / self.yscale_lev
      qv_before     = inputs_denorm[:,:,-1:]
      # print("qbef [0:10] min max mean",qv_before[:,0:10].min().item(), qv_before[:,0:10].max().item(), qv_before[:,0:10].mean().item())

      qn_before     = inputs_denorm[:,:,2:3] + inputs_denorm[:,:,3:4]   
      if self.update_q_for_rad:
        qn_new        = self.relu(qn_before + out_denorm[:,:,2:3].detach().clone()*1200)
      else:
        qn_new        = self.relu(qn_before)
      cldpath       = torch.transpose((delta_plev[:,self.ilev_crm:]/self.g)*qn_new[:,self.ilev_crm:],0,1).contiguous()
      # print("cldpath min max mean", cldpath.min().item(), cldpath.max().item(), cldpath.mean().item())
      # inds_do_clouds = (cldpath.squeeze() > 1e-5)
      if self.rad_cloud_masking:
        qn_new_vint = qn_new.sum(axis=1)
        inds_do_clouds = (qn_new_vint > 1e-6).squeeze() 
        # inds_do_clouds = (cldpath.squeeze() > 1e-10)

      if self.update_q_for_rad:
        qv_new           = qv_before + out_denorm[:,:,1:2].detach().clone()*1200
      else:
        qv_new           = qv_before
      qv_new           = torch.clamp(qv_new, min=0.0)
      T_new           = T_before #+ out_denorm[:,:,0:1]*1200

      # 0. INPUT SCALING - gases
      # Water vapor specific humidity to volume mixing ratio                                      REMEMBER TO FIX THIS!!!!!!!!!!!!!!!!!!!!!!!!!!
      # vmr_h2o = (qv_new / ( 1.0 - qv_new))  *  28.97 / 18.01528 # mol_weight_air / mol_weight_gas
      vmr_h2o = qv_new * 1.608079364 # (28.97 / 18.01528) # mol_weight_air / mol_weight_gas
      # print("vmr_h2o  min max mean", torch.min(vmr_h2o).item(), torch.max(vmr_h2o).item(), torch.mean(vmr_h2o[:,:]).item())

      fact = 1/ (1 + vmr_h2o)
      m_air = (0.04698 + vmr_h2o)*fact #(m_dry + m_h2o * vmr_h2o) * fact 
      #                             avogad
      col_dry = 10.0 * delta_plev * 6.02214076e23 * fact/(1000.0*m_air*100.0*9.80665) 
      col_dry = torch.transpose(col_dry,0,1).contiguous() # (nlev, ncol)
      col_dry = torch.reshape(col_dry,(nlev,batch_size,1))
      
      # T_new = (T_new - self.xmean_lev[:,0:1] ) / (self.xdiv_lev[:,0:1])
      temp = (T_new - 160 ) / (180)
      pressure = (torch.log(play) - 0.00515) / (11.59485)
      # print("q new  min max", torch.min(vmr_h2o[:,:]).item(), torch.max(vmr_h2o[:,:]).item())
      # print("q nans", torch.nonzero(torch.isnan(qv_new.view(-1))))
      # assert not torch.isnan(qv_new).any()

      printdebug=False

      # 1. MLP TO PREDICT OPTICAL PROPERTIES
      if self.physical_precip:
        zeroes_crm_toa = torch.zeros(batch_size, self.ilev_crm, self.mp_ncol, device=device)

        # qn_before     = inputs_denorm[:,:,2:3] + inputs_denorm[:,:,3:4]   
        # qn_before     = torch.repeat_interleave(qn_before,self.mp_ncol,dim=2)
        # dqn           = torch.cat((zeroes_crm_toa, 1200*(dqn_crm/self.yscale_lev[self.ilev_crm:,2:3])),dim=1)
        # qn_new        = qn_before + self.relu(dqn)
        qn_new_crm = torch.cat((zeroes_crm_toa, qn_crm),dim=1)
        # qv_new_crm = torch.cat((zeroes_crm_toa, qv_crm),dim=1)

        t_new_crm0 = self.mp_ncol*T_new[:,self.ilev_crm:]*p_crm_t_new
        tlay_strat = torch.repeat_interleave(T_new[:,0:self.ilev_crm],self.mp_ncol,dim=2)
        t_new_crm = torch.cat((tlay_strat, t_new_crm0),dim=1)
        t_new_crm = t_new_crm / 300

        if self.use_existing_gas_optics_lw or self.use_existing_gas_optics_sw:
          pres1 = torch.log(play)
          vmr_h2o = (torch.sqrt(torch.sqrt(vmr_h2o)))
          o3 =  inputs_denorm[:,:,12:13] 
          o3 = torch.sqrt(torch.sqrt(o3))
          ch4 =  inputs_denorm[:,:,13:14]
          n2o =  inputs_denorm[:,:,14:15]
          co2 =  torch.full((batch_size, self.nlev, 1), 388.7e-6, device=device)
          qn_new_crm0 = 3.5 * torch.sqrt((delta_plev[:,self.ilev_crm:]/self.g)*qn_crm)

          # qn_new1        = 1 - torch.exp(-qn_new * self.lbd_qn)
          # print("qn_new1  min max", torch.min(qn_new1).item(), torch.max(qn_new1).item(), qn_new1.mean(), qn_new1.std() )
          # x_cld = torch.cat((qn_new_crm0, t_new_crm0,rnn_mem), dim=2)
          x_cld = torch.cat((qn_new_crm0, t_new_crm0, rnn_mem), dim=2)
          x_cld = torch.transpose(x_cld,0,1).contiguous()
          if self.rad_cloud_masking:
            # print("shape do cld", x_cld[inds_do_clouds,:].shape, "not", x_cld[~inds_do_clouds,:].shape)
            x_cld = x_cld[:,inds_do_clouds,:]
            # print("shape x cld", x_cld.shape)
          # x_cld = torch.cat((qn_new_crm1,liq_frac_diagnosed.unsqueeze(2)), dim=2)
          if self.use_existing_gas_optics_lw:  
            zero_gases = torch.zeros(batch_size, self.nlev, 11, device=device)
            x_gas = torch.cat((T_new, pres1, vmr_h2o, o3, co2, ch4, n2o, zero_gases), dim=2)
            x_gas = (x_gas - self.gas_optics_model_lw.xmin) / (self.gas_optics_model_lw.xmax - self.gas_optics_model_lw.xmin)
            # print("x_gas T  min max", torch.min(x_gas[:,:,0]).item(), torch.max(x_gas[:,:,0]).item())
            # print("x_gas P min max", torch.min(x_gas[:,:,1]).item(), torch.max(x_gas[:,:,1]).item())
            # print("x_gas h2o min max", torch.min(x_gas[:,:,2]).item(), torch.max(x_gas[:,:,2]).item())
            # print("x_gas o3 min max", torch.min(x_gas[:,:,3]).item(), torch.max(x_gas[:,:,3]).item())
            # print("x_gas co2 min max", torch.min(x_gas[:,:,4]).item(), torch.max(x_gas[:,:,4]).item())
            # print("x_gas  min max", torch.min(x_gas).item(), torch.max(x_gas).item(), "shape", x_gas.shape)

            x_gas = torch.transpose(x_gas,0,1).contiguous()
            tau_lw, pfrac = self.gas_optics_model_lw(x_gas, col_dry)
            if printdebug:
              print("tau_lw0 min max mean", tau_lw.min().item(), tau_lw.max().item(), tau_lw.mean().item())
              print("pfrac min max mean", pfrac.min().item(), pfrac.max().item(), pfrac.mean().item())

            if self.reduce_lw_gas_optics:
              tau_lw = self.gas_optics_lw_reduce1(tau_lw)
              pfrac = self.gas_optics_lw_reduce2(pfrac) # (nlev, nb, g)
              tau_lw = self.softplus(tau_lw)
              # tau_lw = self.relu(tau_lw)
              # tau_lw = 0.1*tau_lw
              tau_lw = 0.01*tau_lw
              # tau_lw = 0.005*tau_lw
              # pfrac = torch.cat((pfrac,x_gas),dim=2)
              pfrac = self.softmax(pfrac)
            
            if printdebug:
              print("tau_lw1 min max mean", tau_lw.min().item(), tau_lw.max().item(), tau_lw.mean().item())
              print("pfrac1 min max mean", pfrac.min().item(), pfrac.max().item(), pfrac.mean().item())

            tau_lw_cld = self.cloud_optics_lw(x_cld)
            if self.use_cloud_overlap_rnn:
              scaling_factor, dummy = self.cloud_mcica_scaling(x_cld)
              scaling_factor = self.cloud_mcica_scaling2(scaling_factor)
              tau_lw_cld = tau_lw_cld*self.relu(scaling_factor)

            # x_cld = torch.permute(x_cld, (1, 2, 0)).contiguous()  # (nlev, nb, ng ) --> (nb, ng, nlev)
            # tau_lw_cld = self.cloud_optics_lw(x_cld) # (nb, ng, nlev)
            # tau_lw_cld = torch.permute(tau_lw_cld, (2, 0, 1)).contiguous()

            zeroes = torch.zeros(self.ilev_crm, batch_size, self.ng_lw, device=device)
            # ----
            if self.rad_cloud_masking:
              # tau_lw_cld = torch.reshape(tau_lw_cld,(-1, self.ng_lw))
              # print("shape tau lw cld", tau_lw_cld.shape)
              tau_lw_cld0 = torch.zeros(self.nlev_crm, batch_size, self.ng_lw, device=device)
              tau_lw_cld0[:,inds_do_clouds,:] = tau_lw_cld
              tau_lw_cld = tau_lw_cld0
            # ---
            tau_lw_cld = cldpath*self.relu(tau_lw_cld)# 0.01*self.relu(tau_lw_cld)
            # tau_lw_cld = 0.01*self.relu(tau_lw_cld)
            tau_lw_cld = torch.cat((zeroes, tau_lw_cld),dim=0)

            # tau_lw_cld      = torch.clamp(tau_lw_cld, max=5.0)
            tau_lw = tau_lw + tau_lw_cld # shape is (nlev, ncol, ng)
            # tau_lw      = torch.clamp(tau_lw, min=1e-7, max=100.0)

          if self.use_existing_gas_optics_sw:
              # inputs: ['tlay' 'play' 'h2o' 'o3' 'co2' 'n2o' 'ch4']
            x_gas = torch.cat((T_new, pres1, vmr_h2o, o3, co2, n2o, ch4), dim=2)
            x_gas = (x_gas - self.gas_optics_model_sw1.xmin) / (self.gas_optics_model_sw1.xmax - self.gas_optics_model_sw1.xmin)
            x_gas = torch.transpose(x_gas,0,1).contiguous()
          
            tau_sw     = self.gas_optics_model_sw1(x_gas, col_dry)
            tau_sw_scat= self.gas_optics_model_sw2(x_gas, col_dry)

            if self.reduce_sw_gas_optics:
              tau_sw      = self.gas_optics_sw_reduce1(tau_sw)
              tau_sw_scat = self.gas_optics_sw_reduce2(tau_sw_scat)
              tau_sw      = 0.01*self.softplus(tau_sw) + 1.0e-7
              tau_sw_scat = 0.01*self.softplus(tau_sw_scat)

            # Total (gas) optical depth
            tau_sw  = tau_sw + tau_sw_scat

            sw_optprops = self.cloud_optics_sw(x_cld) 
            # sw_optprops = torch.permute(sw_optprops, (2, 0, 1)).contiguous()
            if self.use_cloud_overlap_rnn:
              scaling_factor, dummy = self.cloud_mcica_scaling_sw(x_cld)
              scaling_factor = self.cloud_mcica_scaling_sw2(scaling_factor)

            # sw_optprops = self.cloud_optics_sw1(x_cld)
            # sw_optprops = self.softsign(sw_optprops)
            # sw_optprops = self.cloud_optics_sw2(sw_optprops)
            # (nlev, nb, ng*ny ) -->  (nlev, nb, ny, ng)
            # sw_optprops = torch.reshape(sw_optprops,(50, batch_size, self.ny_sw_optprops, self.ng_sw))
            if self.rad_cloud_masking:
              # sw_optprops = torch.reshape(sw_optprops,(-1, self.ny_sw_optprops, self.ng_sw))
              # sw_optprops0 = torch.zeros(self.nlev_crm, batch_size, 3, self.ng_sw, device=device)
              sw_optprops0 = torch.zeros(self.nlev_crm, batch_size, 3*self.ng_sw, device=device)
              # print("shape sw_optprops", sw_optprops.shape)
              sw_optprops0[:,inds_do_clouds,:] = sw_optprops
              sw_optprops = sw_optprops0
              if self.use_cloud_overlap_rnn:
                scaling_factor0 = torch.ones(self.nlev_crm, batch_size, self.ng_sw, device=device)
                scaling_factor0[:,inds_do_clouds,:] = scaling_factor
                scaling_factor = scaling_factor0

            # tau_sw_cld, tau_sw_scat_cld, g_sw_cld = sw_optprops.chunk(3,2 )
            tau_sw_cld, ssa_sw_cld, g_sw_cld = sw_optprops.chunk(3,2 )
            if self.use_cloud_overlap_rnn:
              tau_sw_cld = tau_sw_cld*self.relu(scaling_factor)

            if not self.ng_sw==self.ng_lw:
                zeroes = torch.zeros(10, batch_size, self.ng_sw, device=device)
            if self.reduce_sw_gas_optics:
              # tau_sw_cld      = 0.002*self.relu(tau_sw_cld.squeeze())
              tau_sw_cld      = cldpath*self.relu(tau_sw_cld.squeeze())
              # tau_sw_scat_cld = 0.01*self.relu(tau_sw_scat_cld.squeeze())
              # tau_sw_scat_cld = self.relu(tau_sw_cld*tau_sw_scat_cld.squeeze())

            else:
              # tau_sw_cld      = 0.005*self.relu(tau_sw_cld.squeeze())
              tau_sw_cld      = cldpath*self.relu(tau_sw_cld.squeeze())
              # tau_sw_scat_cld = 0.005*self.relu(tau_sw_scat_cld.squeeze())

            # tau_sw_scat_cld = self.relu(tau_sw_cld*tau_sw_scat_cld.squeeze()) 

            ssa_sw_cld      = self.sigmoid(ssa_sw_cld.squeeze())
            # ssa_sw_cld      = torch.clamp(ssa_sw_cld.squeeze(), min=0.0, max=1.0) 
            # ssa_sw_cld = 0.5*self.tanh(ssa_sw_cld.squeeze()) + 0.5
            g_sw_cld        = self.sigmoid(g_sw_cld.squeeze())

            # print("tau_sw_cld min max mean", tau_sw_cld.min().item(), tau_sw_cld.max().item(), tau_sw.mean().item())
            # print("tau_sw_scat_cld min max mean", tau_sw_scat_cld.min().item(), tau_sw_scat_cld.max().item(), tau_sw_cld.mean().item())
            # print("ssa_sw_cld min max mean", ssa_sw_cld.min().item(), ssa_sw_cld.max().item(), ssa_sw_cld.mean().item())
            
            tau_sw_cld      = torch.cat((zeroes, tau_sw_cld),dim=0)
            # tau_sw_scat_cld = torch.cat((zeroes, tau_sw_scat_cld),dim=0)
            ssa_sw_cld      = torch.cat((zeroes, ssa_sw_cld),dim=0)
            tau_sw_scat_cld = ssa_sw_cld * tau_sw_cld
            g_sw_cld        = torch.cat((zeroes, g_sw_cld.squeeze()),dim=0)

        # else:
        if not (self.use_existing_gas_optics_sw and self.use_existing_gas_optics_lw):
          # qv_before = torch.repeat_interleave(qv_before,self.mp_ncol,dim=2)
          # dqv       = torch.cat((zeroes_crm_toa, 1200*(dqv_crm/self.yscale_lev[self.ilev_crm:,1:2])),dim=1)
          # qv_new    = qv_before + self.relu(dqv)
          qv_new    = qv_new * 1.608079364 # (28.97 / 18.01528) # mol_weight_air / mol_weight_gas
          qv_new    = (torch.sqrt(torch.sqrt(qv_new))) / 0.497653

          mem  = torch.zeros(batch_size, nlev, self.nh_mem, device=device)
          mem[:, self.ilev_crm:] = rnn_mem #[:,:,0:1]
          qn_new        = 1 - torch.exp(-qn_new * self.lbd_qn)
          inputs_rad = torch.cat((pressure, temp, qv_new, qn_new, inputs_main[:,:,6:9], inputs_main[:,:,12:15], mem),dim=2)
          inputs_rad = torch.transpose(inputs_rad,0,1).contiguous()

      else: # not self.physical_precip
        vmr_h2o = (torch.sqrt(torch.sqrt(vmr_h2o))  - 0.0101) / 0.497653

        qliq_before     = inputs_denorm[:,:,2:3]
        qice_before     = inputs_denorm[:,:,3:4]   
        qn_before       = qliq_before + qice_before 
        if self.predict_liq_frac:
          liq_frac_constrained    = out_new[:,:,3]
          qn_new      = qn_before + out_denorm[:,:,2:3]*1200  
          qn_new      = torch.clamp(qn_new, min=0.0)
          qliq_new    = liq_frac_constrained*qn_new
          qice_new    = (1-liq_frac_constrained)*qn_new
        elif self.mp_constraint:
          liq_frac_constrained    = self.temperature_scaling(T_new)
          qn_new      = qn_before + out_denorm[:,:,2:3]*1200  
          qn_new      = torch.clamp(qn_new, min=0.0)
          qliq_new    = liq_frac_constrained*qn_new
          qice_new    = (1-liq_frac_constrained)*qn_new
        else:
          qliq_new    = qliq_before + out_denorm[:,:,2:3]*1200 
          qice_new    = qice_before + out_denorm[:,:,3:4]*1200 
          qliq_new = torch.clamp(qliq_new, min=0.0)
          qice_new = torch.clamp(qice_new, min=0.0)

        # 0. INPUT SCALING - clouds
        qliq_new = 1 - torch.exp(-qliq_new * self.lbd_qc)
        qice_new = 1 - torch.exp(-qice_new * self.lbd_qi)
        
        # Radiation inputs: pressure, temperature, water vapor, cloud ice and liquid, O3, CH4, N2O,, cld heterogeneity
        # conv_mem  = torch.zeros(batch_size, nlev, 1, device=device)
        conv_mem  = torch.zeros(batch_size, nlev, self.nh_mem, device=device)

        # conv_mem[:, self.ilev_crm:] = rnn_mem[:,:,0:1]
        conv_mem[:, self.ilev_crm:] = rnn_mem[:,:,:]
        # inputs_rad = torch.cat((pressure, temp, vmr_h2o, qliq_new, qice_new, inputs_main[:,:,12:15], conv_mem),dim=2)
        inputs_rad = torch.cat((pressure, temp, vmr_h2o, qliq_new, qice_new, inputs_main[:,:,6:9], inputs_main[:,:,12:15], conv_mem),dim=2)
        inputs_rad = torch.transpose(inputs_rad,0,1).contiguous()

      if not self.use_existing_gas_optics_lw:
        # --------------------------- LONGWAVE ---------------------------
        lw_optprops = self.mlp_lw_optprops(inputs_rad) 
        # (nlev, nb, ng*ny ) -->  (nlev, nb, ny, ng)
        lw_optprops = torch.reshape(lw_optprops,(nlev, batch_size, self.ny_lw_optprops, self.ng_lw))
        tau_lw, pfrac = lw_optprops.chunk(2, 2)
        tau_lw      = torch.pow(tau_lw.squeeze(), 8) # (nlev,batch,ng)
        tau_lw      = tau_lw*(1e-24*col_dry)
        tau_lw      = torch.clamp(tau_lw, min=1e-6, max=400.0)
        pfrac       = pfrac.squeeze() # 
        pfrac       = self.softmax(pfrac)
        del lw_optprops
        
      # print("pfrac1 min max mean", torch.min(pfrac[:,:]).item(), torch.max(pfrac[:,:]).item(), torch.mean(pfrac[:,:]).item())
      # print("tau_lw1 min max mean", torch.min(tau_lw[:,:]).item(), torch.max(tau_lw[:,:]).item(), torch.mean(tau_lw[:,:]).item())
      T_new       = torch.transpose(T_new.squeeze(),0,1).contiguous()
      play        = torch.transpose(play.squeeze(),0,1).contiguous()
      plev        = torch.transpose(plev.squeeze(),0,1).contiguous()
      tlev        = self.interpolate_tlev_batchlast(T_new.squeeze(), play.squeeze(), plev.squeeze()) # (nlev+1, nb)
      # print("tlev min max mean", tlev.min().item(), tlev.max().item(), tlev.mean().item())
      lwup_sfc    = (inputs_aux[:,11]*self.xdiv_sca[11]) + self.xmean_sca[11]

      # BELOW IS INCORRECT IF we just used the raw Planck fractions predicted by RRTMGP-NN, without reduction/changing last layer.
      # RRTMGP planck fracs sum to 1 only within bands, not across all g-points. 
      # We should be multiplying with the band-wise Planck emissions, not broadband Planck emission. 
      source_sfc  = pfrac[-1,:,:]*lwup_sfc.unsqueeze(1)
      if printdebug:
        print("lup sfc min max mean", lwup_sfc.min().item(), lwup_sfc.max().item(), lwup_sfc.mean().item())
        print("pfrac shape", pfrac.shape, "sfc vals", pfrac[-1,100,:])
        print("source_sfc min max mean", source_sfc.min().item(), source_sfc.max().item(), source_sfc.mean().item(), "shape", source_sfc.shape)

      # if self.physical_precip:
      #   # p_t_new_crm = self.softmax(self.mlp_t_crm(rnn2out))
      #   # print("shape tlev", tlev.shape, "p", p_t_new_crm.shape)
      #   tlev0 = tlev.transpose(0,1)
      #   t_new_crm = self.ng_lw*tlev0[:,11:].unsqueeze(2)*p_t_new_crm
      #   tlev_strat = torch.repeat_interleave(tlev0[:,0:11].unsqueeze(2),self.ng_lw,dim=2)
      #   t_new_crm = torch.cat((tlev_strat, t_new_crm),dim=1) # (nb, nlev, ng)
      #   lwup_lev    = self.outgoing_lw(t_new_crm).transpose(0,1).contiguous() # (nlev+1, nb, ng)
      # else:
      lwup_lev    = torch.unsqueeze(self.outgoing_lw(tlev),2) # (nlev+1, nb, ng)
      # print("pfrac min max", torch.min(pfrac[:,:]).item(), torch.max(pfrac[:,:]).item())
      if printdebug:
        print("lup lev min max mean", lwup_lev.min().item(), lwup_lev.max().item(), lwup_lev.mean().item(), "shape", lwup_lev.shape)
        print("lwup_lev", lwup_lev[:,0,:])
      source_lev  = torch.zeros(nlev+1, batch_size, self.ng_lw, device=device)
      source_lev[-1,:,:] = pfrac[-1,:,:] * lwup_lev[-1,:,:]
      source_lev[0:-1,:,:] = pfrac[:,:,:]  * lwup_lev[0:-1,:,:]
      if printdebug:
        print("source_lev min max mean", source_lev.min().item(), source_lev.max().item(), source_lev.mean().item())
        print("source_lev 100", source_lev[:,100,0])

      del pfrac
      
      # ---- REFTRANS LW ----
      planck_top = source_lev[0:-1,:,:].view(-1)
      planck_bot = source_lev[1:,:,:].view(-1)
      tau_lw = tau_lw.view(-1)
      source_up, source_dn, trans_lw = self.reftrans_lw(planck_top,planck_bot, tau_lw)
      del tau_lw, planck_top, planck_bot, source_lev
      source_up = source_up.view(nlev, -1)
      source_dn = source_up.view(nlev, -1)
      trans_lw = trans_lw.view(nlev, -1)

      if printdebug:
        print("source_up min max mean", source_up.min().item(), source_up.max().item(), source_up.mean().item())
        print("source_dn min max mean", source_dn.min().item(), source_dn.max().item(), source_dn.mean().item())
        print("trans_lw min max mean", trans_lw.min().item(), trans_lw.max().item(), trans_lw.mean().item(), "shape", trans_lw.shape)
        print("source_up min max", torch.min(source_up[:,:]).item(), torch.max(source_up[:,:]).item())

      # Surface reflection and emission (vectorized)
      # albedos_lw = inputs_aux[:,7:9]
      # albedos_lw = self.mlp_sfc_albedo_lw(albedos_lw)
      # albedos_lw = self.sigmoid(albedos_lw)
      emissivity_surf = torch.ones_like(source_sfc)
      emissivity_surf = emissivity_surf.view(-1)
      source_sfc  = source_sfc.view(-1)
      
      # flux_lw_dn_gpt = torch.zeros(nlev+1, batch_size*self.ng_lw,  device=device)
      # flux_lw_up_gpt = torch.zeros(nlev+1, batch_size*self.ng_lw, device=device)
      if printdebug: 
        print("emissivity_surf min max mean", emissivity_surf.min().item(), emissivity_surf.max().item(), emissivity_surf.mean().item())
      flux_lw_dn, flux_lw_up = self.lw_solver_noscat_batchlast(trans_lw, source_dn, source_up, source_sfc, emissivity_surf)
      if printdebug: print("flux lw gpt up  min max mean", flux_lw_up.min().item(), flux_lw_up.max().item(),flux_lw_up.mean().item())

      flux_lw_up = flux_lw_up.view(nlev+1,batch_size, self.ng_lw)
      flux_lw_dn = flux_lw_dn.view(nlev+1,batch_size, self.ng_lw)
      flux_lw_up = torch.sum(flux_lw_up,dim=2)
      flux_lw_dn = torch.sum(flux_lw_dn,dim=2)
      if printdebug:
        print("flux lw up ", flux_lw_up[:,100])
        print("flux lw up  min max mean", flux_lw_up.min().item(), flux_lw_up.max().item(),flux_lw_up.mean().item())

      # -------------------------- SHORTWAVE -----------------------------
      if not self.use_existing_gas_optics_sw:
        sw_optprops = self.mlp_sw_optprops(inputs_rad)
        # sw_optprops = self.mlp_sw_optprops1(inputs_rad)
        # sw_optprops = self.softsign(sw_optprops)
        # sw_optprops = self.mlp_sw_optprops2(sw_optprops)

        # (nlev, nb, ng*ny ) -->  (nlev, nb, ny, ng)
        sw_optprops = torch.reshape(sw_optprops,(nlev, batch_size, self.ny_sw_optprops, self.ng_sw))
        # tau_sw, tau_sw_scat, g_sw = sw_optprops.chunk(3,2 )
        tau_sw, ssa_sw, g_sw = sw_optprops.chunk(3,2 )
        g_sw        = self.sigmoid(g_sw.squeeze())
        ssa_sw      = self.sigmoid(ssa_sw.squeeze())

        # print("tau sha", tau_sw.shape, "coldry", col_dry.shape)
        tau_sw      = torch.pow(tau_sw.squeeze(), 8)
        tau_sw      = tau_sw*(1e-24*col_dry)
        tau_sw      = torch.clamp(tau_sw, min=1e-6, max=40.0) 
        # tau_sw_scat  = torch.pow(tau_sw_scat.squeeze(), 8)
        # tau_sw_scat  = tau_sw_scat*(1e-24*col_dry)
        # tau_sw_scat  = torch.clamp(tau_sw_scat, min=1e-6, max=40.0) 

        # tau_sw      = tau_sw + tau_sw_scat
        # ssa_sw      = tau_sw_scat / tau_sw
        # # g_sw        = self.sigmoid(g_sw.squeeze())
        # g_sw        = torch.clamp(g_sw.squeeze(), min=0.0, max=1.0) 
        # # g_sw        = (g_sw*tau_sw_scat_cld) / torch.maximum(tau_sw_scat, torch.tensor(1.0e-12))
        del sw_optprops
      else:
        # Compute total optical depth as gas (total) optical depth + cloud (total) optical depth
        tau_sw_tot = tau_sw + tau_sw_cld
        # Compute single-scattering albedo of the cloud-gas mixture as (total scattering optical depth)/(total optical depth)
        tau_sw_scat_tot = tau_sw_scat + tau_sw_scat_cld # total scattering optical depth 
        # Combine cloud and gas asymmetry factors, weighting by scattering optical depth 
        #        gas g*tau_scat         cloud g*tau_scat        total tau_scat
        # g_sw    = (g_sw*tau_sw_scat + g_sw_cld*tau_sw_scat_cld) / (tau_sw_scat_tot) 
        # HOWEVER g_sw from gases is zero, so lose first term
        g_sw = g_sw_cld*tau_sw_scat_cld / tau_sw_scat_tot
        tau_sw  = tau_sw_tot 
        ssa_sw  = tau_sw_scat_tot / tau_sw_tot
        # print("min max g_sw_cld", g_sw_cld.min().item(), g_sw_cld.max().item())
        # print("min max mean tau_sw_cld", tau_sw_cld.min().item(), tau_sw_cld.max().item(), tau_sw_cld.mean().item())
        # print("min max mean tau_sw_scat", tau_sw_scat.min().item(), tau_sw_scat.max().item(), tau_sw_scat.mean().item())
        # print("min max mean tau_sw_scat_tot", tau_sw_scat_tot.min().item(), tau_sw_scat_tot.max().item(), tau_sw_scat_tot.mean().item())        # print("min max mean g sw cld", g_sw_cld.min().item(), g_sw_cld.max().item(), g_sw_cld.max().item())
        # print("min max mean g sw", g_sw.min().item(), g_sw.max().item(), g_sw.max().item())
        # print("min max ssa", ssa_sw.min().item(), ssa_sw.max().item(), "tau", tau_sw.min().item(), tau_sw.max().item())

        del tau_sw_scat, tau_sw_scat_tot, tau_sw_scat_cld
      
      # ---- REFTRANS SW ------
      mu0 = torch.reshape(inputs_aux[:,6:7],(-1,1,1))
      min_mu = 1e-6 #1e-3
      mu0 = torch.clamp(mu0, min=min_mu) 

      # (ncol) -> (nlev, ncol) -> (nlev,ng*ncol)
      mu0_rep = torch.repeat_interleave(torch.repeat_interleave(mu0.reshape((1,-1)), nlev, dim=0),self.ng_sw,dim=1)

      # print("mu0 min max", torch.min(mu0).item(), torch.max(mu0).item())
      # print("tau_sw min max", torch.min(tau_sw).item(), torch.max(tau_sw).item())
      # print("ssa_sw min max", torch.min(ssa_sw).item(), torch.max(ssa_sw).item())
      # print("g_sw min max", torch.min(g_sw).item(), torch.max(g_sw).item())

      # t0 = time.time()
      # batched_reftra = torch.func.vmap(self.calc_reflectance_transmittance_sw)
      
      # print("shape mu0", mu0.shape, "tausw", tau_sw.shape, "ssasw", ssa_sw.shape, "gsw", g_sw.shape)
      # ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir = self.calc_reflectance_transmittance_sw(mu0_rep.view(-1), tau_sw.view(-1), ssa_sw.view(-1), g_sw.view(-1))
      ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir = self.calc_ref_trans_sw(mu0_rep.view(-1), tau_sw.view(-1), ssa_sw.view(-1), g_sw.view(-1))
      # ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir = batched_reftra(mu0, tau_sw, ssa_sw, g_sw)
      ref_diff            = ref_diff.view(nlev, -1)
      trans_diff          = trans_diff.view(nlev, -1)
      ref_dir             = ref_dir.view(nlev, -1)
      trans_dir_diff      = trans_dir_diff.view(nlev, -1)
      trans_dir_dir       = trans_dir_dir.view(nlev, -1)
      
      # print ("shape ref_diff", ref_diff.shape)
      # print("elapsed reftra {}".format(time.time() - t0))
      del tau_sw, ssa_sw, g_sw, mu0_rep

      incoming_toa    = inputs_aux[:,1:2]

      if printdebug:
        print("incoming toa", incoming_toa[500:560])

      if (self.use_existing_gas_optics_sw and (not self.reduce_sw_gas_optics)):
        toa_spectral = self.sw_solar_weights 
      else:
        # Here we apply torch.square to ensure the weights are positive, then softmax so that they sum to 1
        toa_spectral = self.softmax_dim1(torch.square(self.sw_solar_weights))
    
      # toa_spectral = torch.repeat_interleave(toa_spectral,self.ng_sw)

      incoming_toa = incoming_toa*toa_spectral
      incoming_toa = torch.flatten(incoming_toa)

      # emissivity_surf_diff_sw_sca   = inputs_aux[:,9]
      # emissivity_surf_dir_sw_sca    = inputs_aux[:,10]
      # emissivity_surf_diff_sw = torch.repeat_interleave(emissivity_surf_diff_sw_sca,self.ng_sw)
      # emissivity_surf_dir_sw = torch.repeat_interleave(emissivity_surf_dir_sw_sca,self.ng_sw)

      # albedos_sw = inputs_aux[:,7:11]
      # emissivity_surf_diff_sw = self.sigmoid(self.mlp_sfc_albedo_sw1(albedos_sw)).view(-1)
      # emissivity_surf_dir_sw = self.sigmoid(self.mlp_sfc_albedo_sw2(albedos_sw)).view(-1)

      aldif = inputs_aux[:,7].view(1,-1)
      aldir = inputs_aux[:,8].view(1,-1)
      asdif = inputs_aux[:,9].view(1,-1)
      asdir  = inputs_aux[:,10].view(1,-1)

      # Extract spectral points associated with near-IR and visible radiation
      # If we are using RRTMGP(-NN), then this should exactly match how things are done in subroutine set_albedo in E3SM/components/eam/src/physics/rrtmgp/radiation.F90
      # If we are learning a new gas optics module on the fly, or a decoder to shrink the spectral (=hidden) dimension after RRTMGP-NN, then the lines below assume that it's a good
      # idea to follow RRTMGP in how much of the spectral space is allocated to near-IR versus visible. Might be a poor assumption but don't know what else to do!
      iend_ir = int(np.round((80/112)*self.ng_sw)) # RRTMGP bands 1-9 (g-points 1-88) encompass 820-12850 cm-1 (near-ir), see data/rrtmgp-data-sw-g112-210809.nc
      iend_mix=int(np.round((89/112)*self.ng_sw)) # RRTMGP band 10 is in between UV/visible and near-IR, and bands 11-14 (89-112) are fully in visible range (> 14286 ! cm^-1)
      
      emissivity_surf_dir_sw   = torch.ones(self.ng_sw, batch_size, device=device)
      emissivity_surf_diff_sw  = torch.ones(self.ng_sw, batch_size, device=device)
      ng_ir  = iend_ir
      ng_mix = iend_mix - iend_ir
      ng_vis = self.ng_sw - iend_mix

      aldir1 = torch.repeat_interleave(aldir,ng_ir,dim=0)
      aldir2 = torch.repeat_interleave(aldir,ng_mix,dim=0)
      asdir1 = torch.repeat_interleave(asdir,ng_vis,dim=0)
      asdir2 = torch.repeat_interleave(asdir,ng_mix,dim=0)

      aldif1 = torch.repeat_interleave(aldif,ng_ir,dim=0)
      aldif2 = torch.repeat_interleave(aldif,ng_mix,dim=0)
      asdif1 = torch.repeat_interleave(asdif,ng_vis,dim=0)
      asdif2 = torch.repeat_interleave(asdif,ng_mix,dim=0)

      emissivity_surf_dir_sw[0:iend_ir]         =  aldir1
      emissivity_surf_dir_sw[iend_ir:iend_mix]  =  0.5*(aldir2 + asdir2)
      emissivity_surf_dir_sw[iend_mix:]         =  asdir1

      emissivity_surf_diff_sw[0:iend_ir]         =  aldif1
      emissivity_surf_diff_sw[iend_ir:iend_mix]  =  0.5*(aldif2 + asdif2)
      emissivity_surf_diff_sw[iend_mix:]         =  asdif1

      emissivity_surf_dir_sw = torch.transpose(emissivity_surf_dir_sw,0,1).contiguous().view(-1)
      emissivity_surf_diff_sw = torch.transpose(emissivity_surf_diff_sw,0,1).contiguous().view(-1)

      # if printdebug:
      #   print("emissivity_surf_dir", emissivity_surf_dir_sw[500], "min", emissivity_surf_dir_sw.min().item(), "max", emissivity_surf_dir_sw.max().item())
      #   print("emissivity_surf_diff_sw", emissivity_surf_diff_sw[500], "min", emissivity_surf_diff_sw.min().item(), "max", emissivity_surf_diff_sw.max().item())

      # cos_sza = torch.repeat_interleave(mu0.flatten().unsqueeze(1),self.ng_sw).flatten()

      if self.experimental_rad:
        # xx = torch.transpose(rnn2out,1,2)
        xx = torch.transpose(qn_crm,1,2)
        v_mat = self.sigmoid(self.conv_vmat(xx)) # (nb, ng, nlev)
        v_mat = torch.transpose(v_mat,1,2) # (nb, nlev, ng)
        v_mat = torch.transpose(v_mat,0,1) # (nlev, nb, ng)
        ones  = torch.ones(1, batch_size, self.ng_sw, device=device)
        oness  = torch.ones(11, batch_size, self.ng_sw, device=device)
        v_mat = torch.cat((oness, v_mat, ones), dim=0)
        # print("shape v mat", v_mat.shape)
        v_mat       = torch.reshape(v_mat, (nlev+1, -1))

        flux_sw_up, flux_sw_dn_diffuse, flux_sw_dn_direct = self.adding_tc_sw_batchlast(incoming_toa, 
                    emissivity_surf_diff_sw, emissivity_surf_dir_sw, ref_diff, 
                    trans_diff, ref_dir, trans_dir_diff, trans_dir_dir, v_mat)
        flux_sw_dn_diffuse = flux_sw_dn_diffuse*v_mat
        flux_sw_dn_direct = flux_sw_dn_direct*v_mat
      else:

        # flux_sw_up, flux_sw_dn_diffuse, flux_sw_dn_direct = self.adding_ica_sw_batchlast(incoming_toa, 
        flux_sw_up, flux_sw_dn_diffuse, flux_sw_dn_direct = self.adding_ica_sw_batchlast_opt(incoming_toa, 
                    emissivity_surf_diff_sw, emissivity_surf_dir_sw, 
                    ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir)
      
      del ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir

      # mu0_rep = torch.repeat_interleave(torch.repeat_interleave(inputs_aux[:,6:7].reshape((1,-1)), nlev, dim=0),self.ng_sw,dim=1)
      # inds_zero = mu0_rep < min_mu
      # flux_sw_up[inds_zero] = 0.0
      # flux_sw_dn_diffuse[inds_zero] = 0.0
      # flux_sw_dn_direct[inds_zero] = 0.0

      # print("flux_sw_up  min max", torch.min(flux_sw_up[:,:]).item(), torch.max(flux_sw_up[:,:]).item())
      # print("flux_sw_dn_diffuse  min max", torch.min(flux_sw_dn_diffuse[:,:]).item(), torch.max(flux_sw_dn_diffuse[:,:]).item())
      # print("flux_sw_dn_direct  min max", torch.min(flux_sw_dn_direct[:,:]).item(), torch.max(flux_sw_dn_direct[:,:]).item())
      
      flux_sw_up = self.relu(flux_sw_up)
      flux_sw_dn_diffuse = self.relu(flux_sw_dn_diffuse)
      flux_sw_dn_direct = self.relu(flux_sw_dn_direct)

      flux_sw_up = torch.reshape(flux_sw_up, (nlev+1, batch_size, self.ng_sw))
      flux_sw_dn_diffuse = torch.reshape(flux_sw_dn_diffuse, (nlev+1, batch_size, self.ng_sw))
      flux_sw_dn_direct = torch.reshape(flux_sw_dn_direct, (nlev+1, batch_size, self.ng_sw))
      
      # Extract spectral points associated with near-IR and visible radiation
      # If we are using RRTMGP(-NN), then this should exactly match how things are done in subroutine export_surface_fluxes in E3SM/components/eam/src/physics/rrtmgp/radiation.F90
      # If we are learning a new gas optics module on the fly, or a decoder to shrink the spectral (=hidden) dimension after RRTMGP-NN, then the lines below assume that it's a good
      # idea to follow RRTMGP in how much of the spectral space is allocated to near-IR versus visible. Might be a poor assumption but don't know what else to do!
      # iend_ir = int(np.round((80/112)*self.ng_sw)) # RRTMGP bands 1-9 (g-points 1-88) encompass 820-12850 cm-1 (near-ir), see data/rrtmgp-data-sw-g112-210809.nc
      # iend_mix=int(np.round((89/112)*self.ng_sw)) # RRTMGP band 10 is in between UV/visible and near-IR, and bands 11-14 (89-112) are fully in visible range (> 14286 ! cm^-1) near-IR, and bands 12-14 (96-112) are in visible range
      
      sw_dir_dn_mixband = torch.sum(flux_sw_dn_direct[-1,:,iend_ir:iend_mix],dim=1,keepdim=True)
      SOLL = torch.sum(flux_sw_dn_direct[-1,:,0:iend_ir],dim=1,keepdim=True) + 0.5*sw_dir_dn_mixband
      SOLS = torch.sum(flux_sw_dn_direct[-1,:,iend_mix:],dim=1,keepdim=True) + 0.5*sw_dir_dn_mixband

      sw_diff_dn_mixband = torch.sum(flux_sw_dn_diffuse[-1,:,iend_ir:iend_mix],dim=1,keepdim=True)
      SOLLD = torch.sum(flux_sw_dn_diffuse[-1,:,0:iend_ir],dim=1,keepdim=True) + 0.5*sw_diff_dn_mixband
      SOLSD = torch.sum(flux_sw_dn_diffuse[-1,:,iend_mix:],dim=1,keepdim=True) + 0.5*sw_diff_dn_mixband

      flux_sw_up          = torch.sum(flux_sw_up,dim=2)
      flux_sw_dn_diffuse  = torch.sum(flux_sw_dn_diffuse,dim=2)
      flux_sw_dn_direct   = torch.sum(flux_sw_dn_direct,dim=2)
      
      flux_sw_dn          = flux_sw_dn_diffuse + flux_sw_dn_direct
      flux_sw_dn_sfc      = flux_sw_dn[-1,:].unsqueeze(1)             # NETSW  

      flux_sw_net = flux_sw_dn - flux_sw_up
      # mu_rep = torch.repeat_interleave(inputs_aux[:,6:7].reshape(1,-1),nlev+1,dim=0)
      inds_zero = inputs_aux[:,6] < min_mu
      flux_sw_net[:,inds_zero] = 0.0
      SOLL[inds_zero] = 0.0
      SOLS[inds_zero] = 0.0
      SOLLD[inds_zero] = 0.0
      SOLSD[inds_zero] = 0.0


      if printdebug:
        print("flux sw dn ", flux_sw_dn[:,500])
        print("flux sw up ", flux_sw_up[:,500])

      flux_lw_dn_sfc      = flux_lw_dn[-1,:].unsqueeze(1)             # FLWDS 
      flux_lw_net         = flux_lw_dn - flux_lw_up
      # print("flux_lw_up  min max", torch.min(flux_lw_up[:,:]).item(), torch.max(flux_lw_up[:,:]).item())
      # print("flux_lw_net  min max", torch.min(flux_lw_net[:,:]).item(), torch.max(flux_lw_net[:,:]).item())
      # print("flux_sw_net  min max", torch.min(flux_sw_net[:,:]).item(), torch.max(flux_sw_net[:,:]).item())
      # print("incflux 100", inputs_aux[100,1:2].item())
      # print("mu0 100", mu0[100].item())
      # print("flux_sw_dn 100", flux_sw_dn[100,:].detach().cpu().numpy())
      # print("flux_sw 100", flux_sw_net[100,:].detach().cpu().numpy())

      delta_plev = torch.transpose(delta_plev,0,1).contiguous().squeeze()
      # flux_net = flux_lw_net
      flux_net = flux_lw_net + flux_sw_net
      flux_diff = flux_net[1:,:] - flux_net[0:-1,:]
      dT_rad = -(flux_diff / delta_plev) * 0.009761357302 # * g/cp = 9.80665 / 1004.64
      # # print("flux_diff  min max", torch.min(flux_diff[:,:]).item(), torch.max(flux_diff[:,:]).item())
      # # print("pres_diff  min max", torch.min(delta_plev[:,:]).item(), torch.max(pres_diff[:,:]).item())
      # # print("dT_rad  min max", torch.min(dT_rad[:,:]).item(), torch.max(dT_rad[:,:]).item())

      # flux_net =  flux_sw_net
      # flux_diff = flux_net[1:,:] - flux_net[0:-1,:]
      # dT_rad_sw = -(flux_diff / delta_plev) * 0.009767579681

      # # dT_rad1 = dT_rad* self.yscale_lev[:,0].unsqueeze(0)
      # # dT_rad2 = dT_rad_sw* self.yscale_lev[:,0].unsqueeze(0)
      # dT_rad = dT_rad + dT_rad_sw 
      # print("mean dt rad lw", dT_rad1[:,0:2].mean().item(), "SW", dT_rad2[:,0:2].mean().item(), "tot", dT_rad[:,0:2].mean().item())

      dT_rad = torch.transpose(dT_rad,0,1).contiguous()
      # normalize heating rate output 
      dT_rad = dT_rad * self.yscale_lev[:,0].unsqueeze(0)
      # print("mean dt rad tot", dT_rad[:,0:2].mean().item())
      # print("plev", plev[:,0:2].mean())
      
      # #1D (scalar) Output variables: ['cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 
      # #'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']
      out_sfc_rad = torch.cat((flux_sw_dn_sfc, flux_lw_dn_sfc,  SOLS, SOLL, SOLSD, SOLLD ), dim=1)
      out_sfc_rad =  out_sfc_rad * self.yscale_sca_rad

      return dT_rad, out_sfc_rad
      # return out_new, out_sfc_rad 
        
    
    def forward(self, inp_list : List[Tensor]):
        inputs_main     = inp_list[0]
        inputs_aux      = inp_list[1]
        rnn_mem         = inp_list[2]
        inputs_denorm   = inp_list[3]
        if self.training:
          out_new_true = inp_list[4]
        # print("shape inp main", inputs_main.shape, "aux", inputs_aux.shape, "mem", rnn_mem.shape, "denorm", inputs_denorm.shape)
        if self.physical_precip and self.store_precip: 
          P_old = rnn_mem[:,-1,-1] 
          
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
            sp = sp*self.xdiv_sca[0:1] + self.xmean_sca[0:1]
            pres  = self.preslay(sp)
            inputs_main = torch.cat((inputs_main,pres),dim=2)
            play = self.preslay_nonorm(sp)
            delta_plev = self.presdelta(sp)
            plev = self.preslev_nonorm(sp)
        if self.repeat_mu:
            mu = torch.reshape(inputs_aux[:,6:7],(-1,1,1))
            mu_rep = torch.repeat_interleave(mu,nlev,dim=1)
            inputs_main = torch.cat((inputs_main,mu_rep),dim=2)
            
        
        # if self.separate_radiation:
        # Do not use inputs -2,-3,-4 (O3, CH4, N2O) or first 10 levels
        inputs_main_crm = torch.cat((inputs_main[:, self.ilev_crm:,0:-4], inputs_main[:, self.ilev_crm:,-1:]),dim=2)

        if self.use_initial_mlp:
            inputs_main_crm = self.mlp_initial(inputs_main_crm)
            inputs_main_crm = self.tanh(inputs_main_crm)  
            
        # inputs_main_crm = torch.cat((inputs_main_crm,rnn_mem), dim=2)
        inputs_main_crm = torch.cat((inputs_main_crm,rnn_mem[:,:,0:self.nh_mem0]), dim=2)

        # TOA is first in memory, so to start at the surface we need to go backwards
        rnn1_input = torch.flip(inputs_main_crm, [1])
        if not self.physical_precip:
          del inputs_main_crm
        
        # The input (a vertical sequence) is concatenated with the
        # output of the RNN from the previous time step 

        # if self.separate_radiation:
        inputs_sfc = torch.cat((inputs_aux[:,0:6],inputs_aux[:,11:]),dim=1)

        hx = self.mlp_surface1(inputs_sfc)
        hx = self.tanh(hx)
        if self.use_lstm: 
            cx = self.mlp_surface2(inputs_sfc)
            # cx = self.tanh(cx)
            hidden = (torch.unsqueeze(hx,0), torch.unsqueeze(cx,0))  
        else:
            hidden = torch.unsqueeze(hx,0)
        rnn1out, states = self.rnn1(rnn1_input, hidden)
        del rnn1_input

        rnn1out = torch.flip(rnn1out, [1])

        hx2 = torch.randn((batch_size, self.nh_rnn2),device=device)  # (batch, hidden_size)
        if self.use_lstm: 
          cx2 = torch.randn((batch_size, self.nh_rnn2),device=device)
          hidden2 = (torch.unsqueeze(hx2,0), torch.unsqueeze(cx2,0))
        else:
          hidden2 = torch.unsqueeze(hx2,0)

        rnn2out, last_h = self.rnn2(rnn1out, hidden2)
        if self.use_lstm: 
          (last_h, last_c) = last_h

        if self.add_stochastic_layer:
          input_srnn = torch.transpose(rnn2out,0,1)

          hx = torch.randn((batch_size, self.nh_rnn2),device=inputs_main.device) 
          if self.use_lstm:
            cx = torch.randn((batch_size, self.nh_rnn2),device=inputs_main.device) 
            hx = (hx, cx)
            srnn_out, last_state = self.rnn3(input_srnn, hx)
          else:
            srnn_out = self.rnn3(input_srnn, hx)
          h_sfc_perturb = srnn_out[-1,:,:]

          h_final_perturb = torch.transpose(srnn_out,0,1)

          # h_final = h_final + 0.01*h_final_perturb
          # h_sfc   = h_sfc + 0.01*h_sfc_perturb
          # h_final_perturb = self.hardtanh(h_final_perturb)
          # h_sfc_perturb = self.hardtanh(h_sfc_perturb)

          rnn2out = rnn2out*h_final_perturb
          # h_sfc   = h_sfc*h_sfc_perturb
          last_h   = h_sfc_perturb
  
            
        # if self.concat:
        #     rnn2out = torch.cat((rnn1out, rnn2out), dim=2)
        
        if self.use_intermediate_mlp: 
            rnn_mem = self.mlp_latent(rnn2out)
        else:
            rnn_mem = rnn2out 

        out = self.mlp_output(rnn_mem)

        out_new = torch.zeros(batch_size, self.nlev, self.ny, device=device)

        if self.physical_precip:
           if self.training:
              with torch.autocast(device_type=rnn2out.device.type, enabled=False):
                rnn2out = rnn2out.float()
                last_h = last_h.float()
                out_new, precc, precsc, rnn_mem, qn_crm, qv_crm, p_crm_t_new,prec_negative = self.microphysics_decode(inputs_main, 
                                                    inputs_denorm, delta_plev, play, plev, P_old, out, rnn_mem, rnn2out, last_h, out_new)
           else:
              out_new, precc, precsc, rnn_mem, qn_crm, qv_crm, p_crm_t_new,prec_negative = self.microphysics_decode(inputs_main, 
                                                      inputs_denorm, delta_plev, play, plev, P_old, out, rnn_mem, rnn2out, last_h, out_new)
                
        else:
          qn_crm, p_crm_t_new = None, None


        if self.training:
          with torch.autocast(device_type=rnn2out.device.type, enabled=False):
            dT_rad, out_sfc_rad = self.radiative_transfer(inputs_main, inputs_aux, inputs_denorm, play, plev, delta_plev, 
                                      rnn_mem, rnn2out, qn_crm, qv_crm, p_crm_t_new, out_new_true)
        else:
          dT_rad, out_sfc_rad = self.radiative_transfer(inputs_main, inputs_aux, inputs_denorm, play, plev, delta_plev, 
                                      rnn_mem, rnn2out, qn_crm, qv_crm, p_crm_t_new, out_new)

        # print("dT_rad 2  min max", torch.min(dT_rad[:,:]).item(), torch.max(dT_rad[:,:]).item())
        out_new[:,:,0:1] = out_new[:,:,0:1] + dT_rad.unsqueeze(2)


        # # rad predicts everything except PRECSC, PRECC
        if self.physical_precip:
            out_sfc =  torch.cat((out_sfc_rad[:,0:2], precsc, precc, out_sfc_rad[:,2:]),dim=1)
        else:
            final_sfc_inp = last_h.squeeze() 
            out_sfc = self.mlp_surface_output(final_sfc_inp)
            out_sfc = self.relu(out_sfc)    
            out_sfc =  torch.cat((out_sfc_rad[:,0:2], out_sfc, out_sfc_rad[:,2:]),dim=1)

        if self.predict_liq_frac:
          temp = (inputs_main[:,:,0]*self.xdiv_lev[:,0]) + self.xmean_lev[:,0]  
          temp = temp + (out_new[:,:,0]/self.yscale_lev[:,0]) * 1200
          liq_frac_diagnosed0    = self.temperature_scaling(temp)
          liq_frac_diagnosed = liq_frac_diagnosed0[:,self.ilev_crm:]
          temp = temp[:,self.ilev_crm:]
          inds = (temp < 275.0) & (temp<250.0)

          x_predfrac = rnn_mem[inds]
          liq_frac_pred = self.mlp_predfrac(x_predfrac)
          liq_frac_pred = torch.reshape(liq_frac_pred,(-1,))
          liq_frac_pred = liq_frac_pred.to(liq_frac_diagnosed.dtype)
          liq_frac_diagnosed[inds] = liq_frac_pred
          out_new[:,self.ilev_crm:,3] = self.relu(liq_frac_diagnosed)
          out_new[:,0:self.ilev_crm,3] = liq_frac_diagnosed0[:,0:self.ilev_crm]

        if self.physical_precip and self.return_neg_precip:
          return out_new, out_sfc, rnn_mem, prec_negative
        else:
          return out_new, out_sfc, rnn_mem

    @torch.jit.export
    def pp_mp(self, out, out_sfc, x_denorm):

        out_denorm      = out / self.yscale_lev
        out_sfc_denorm  = out_sfc / self.yscale_sca
        # print("pp1 MEAN FRAC P", torch.mean(out_denorm[:,:,3]).item(), "MIN", torch.min(out_denorm[:,:,3]).item(),  "MAx", torch.max(out_denorm[:,:,3]).item() )

        # print("pp_mp 1 frac100, ", out_denorm[100,:,3].detach().cpu().numpy())
        T_old        = x_denorm[:,:,0:1]
        qliq_old     = x_denorm[:,:,2:3]
        qice_old     = x_denorm[:,:,3:4]   
        qn_old       = qliq_old + qice_old 

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
        # print("qliq new min", qliq_new.min(), "max", qliq_new.max())
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