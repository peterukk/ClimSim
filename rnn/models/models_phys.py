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
from .models import Base_RNN_autoreg
from models_torch_kernels import GLU
from models_torch_kernels import *
from .physics_rad import outgoing_lw, reftrans_lw, lw_solver_noscat_batchlast, calc_overlap_matrices
from .physics_rad import calc_ref_trans_sw, adding_ica_sw_batchlast_opt, adding_ica_sw, adding_tc_sw_batchlast_opt
from .physics_rad import stratified_sample, interpolate_tlev_batchfirst, interpolate_tlev_batchlast
from .physics_rad_e3sm import reitab, reltab, slingo_liq_cloud_optics_sw, ec_ice_optics_sw
from metrics import specific_to_relative_humidity_torch_cc
import numpy as np 
from typing import Final 
import time
from omegaconf import DictConfig

class physical_RNN_autoreg(Base_RNN_autoreg):
    """
    Physics-informed version of the BiRNN
    Predicts vertical fluxes and microphysical process rates (conserving moisture) for "sub-grid regions" (nreg)
    Precipitation that hasn't yet fallen to the surface can be stored as a scalar to the next time step
    Optionally, can be combined with physical equations for radiative transfer (use_physrad)
    """
    use_initial_mlp: Final[bool]
    use_intermediate_mlp: Final[bool]
    add_pres: Final[bool]
    add_stochastic_layer: Final[bool]
    use_ensemble: Final[bool]
    use_lstm: Final[bool]
    output_prune: Final[bool]
    predict_liq_frac: Final[bool] # Predict fraction of cloud that is liquid 
    predict_total_water: Final[bool] # Predict total water tendency, fraction of total water that is cloud, and liq_frac 
    use_mp_constraint: Final[bool] # see train_rnn_rollout_torchscript_hydra.py
    mp_mode: Final[int]
    # --- Options for physics-based moist physics based on predicting fluxes and microphysics process rates ---
    return_neg_precip: Final[bool] # Here precip is (semi-)diagnosed and can have negative values - we can penalize this with a loss 
    store_precip: Final[bool] # In the semi-prognostic method, precip that hasn't yet reached the surface can be stored to the next t-step
    ice_sedimentation: Final[bool] 
    allow_extra_heating: Final[bool] # Allow extra (predicted) dT besides latent heating from microphysics and radiative heating/cooling?
    condense_supersaturated_water: Final[bool] # Remove supersatured water and put it into cloud water
    use_clear_sky_region: Final[bool] # Force one of the "regions" (latent/sub-grid dim) to be clear-sky, as in the TripleClouds radiation scheme
    separate_radiation: Final[bool] # If physrad is False, we can still try to separate radiation (use separate BiGRU and vertical grid)
    pred_subgrid_liq_frac: Final[bool] 
    pred_subgrid_temp: Final[bool]
    # --- Options for physical radiation ---
    use_physrad: Final[bool] # Turn on physical radiation
    experimental_rad: Final[bool] # Use the TripleClouds method (Shonk&Hogan 2010) for sub-grid clouds in radiation - experimental and perhaps buggy!
    use_mcica: Final[bool] # Alternative should be McICA...but here both can be turned off, which weirdly conflates the spectral and sub-grid region dimensions
    include_qv_variability: Final[bool] # Account for sub-grid variability of water vapor for SW by doing two passes with the SW gas optics
    update_states_for_rad: Final[bool] # 
    use_e3sm_cloud_optics: Final[bool] 
    map_e3sm_cloud_optics: Final[bool]
    use_existing_gas_optics_lw: Final[bool] # Use existing RRTMGP-NN-LW
    use_existing_gas_optics_sw: Final[bool] # Use existing RRTMGP-NN-SW
    reduce_lw_gas_optics: Final[bool] # ...which may be combined with another MLP to shrink the spectral dim to e.g. 16 (otherwise expensive)
    reduce_sw_gas_optics: Final[bool]
    use_new_sw_gas_optics: Final[bool]
    printdebug: Final[bool]

    def __init__(self, 
                cfg: DictConfig, 
                coeffs: dict, 
                device: torch.device,
                gas_optics_model_lw=None,
                gas_optics_model_sw1=None,gas_optics_model_sw2=None,
                ):
        super().__init__(cfg, coeffs, device)
        if self.mp_mode==0:
            raise NotImplementedError("Cfg=0 not supported by model")
        self.use_physrad = cfg.use_physrad
        self.separate_radiation = cfg.separate_radiation
        self.nlev_crm = 50
        self.ilev_crm = self.nlev-self.nlev_crm
        self.nlev_mem = self.nlev_crm
        if self.use_physrad or self.separate_radiation:
          self.nx_sfc_rad = 5
          self.nx = self.nx - 3 # remove 'pbuf_ozone', 'pbuf_CH4', 'pbuf_N2O' from main inputs
        else:
          self.nx_sfc_rad = 0
          self.nx = self.nx
        self.nx_sfc = self.nx_sfc  - self.nx_sfc_rad
        self.ny_rad = 1
        self.ny_sfc_rad = self.ny_sfc0 - 2
        self.ny_sfc0 = 2    
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=2)
        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim0 = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()
        self.softsign =  nn.Softsign()
        print("Building RNN model that feeds its hidden memory at t0,z0 to its inputs at t1,z0")
        print("Initial mlp: {}, intermediate mlp: {}".format(self.use_initial_mlp, self.use_intermediate_mlp))
        print("nx rnn1", self.nx_rnn1, "nh rnn1", self.nh_rnn1)
        print("nx rnn2", self.nx_rnn2, "nh rnn2", self.nh_rnn2)  
        print("nx sfc", self.nx_sfc)
        self.presdelta = PressureThickness(self.hyai,self.hybi)
        self.reitab = reitab 
        self.reltab = reltab
        self.stratified_sample = stratified_sample
        self.interpolate_tlev_batchlast = interpolate_tlev_batchlast
        self.outgoing_lw = outgoing_lw
        self.slingo_liq_cloud_optics_sw = slingo_liq_cloud_optics_sw 
        self.ec_ice_optics_sw = ec_ice_optics_sw
        self.specific_to_relative_humidity_torch_cc = specific_to_relative_humidity_torch_cc
        self.printdebug = False
        # --------- Moist physics options configured in cfg ---------
        # 
        # Number of sub-grid regions
        self.nreg = cfg.nreg
        # -----------------------------------------------------------
        # 
        # --------- Moist physics options configured here ---------
        # 
        # Allow falling precip to be stored to the next time step and predict a fraction of total
        # (diagnosed + stored) that is released, or simply diagnose?
        self.store_precip         = True
        # The diagnosed precipitation can be negative, which is unphysical. Penalize this with a loss term,
        # which requires the negative precipitation to be outputted (should be set to True)
        if self.store_precip:
          self.return_neg_precip  = True
          # ^Currently always true with this model, if a method to guarantee positive precipitation is 
          # devised then any associated lines can be removed
          self.mlp_precip_release = nn.Linear(self.nh_rnn2, 1)
          self.nh_mem0 = self.nh_mem - 1 
          self.nx_rnn1 = self.nx_rnn1 - 1
        else:
          self.nh_mem0 = self.nh_mem 
          self.return_neg_precip  = False
        # Include sedimentation of ice which contributes to the precip. falling to the surface?
        self.ice_sedimentation    = cfg.ice_sedimentation # True
        # Allow extra heating/cooling besides that from microphysics, vertical fluxes and radiation (if physrad=true)?
        self.allow_extra_heating  = False
        if not self.use_physrad:
          self.allow_extra_heating = True
        self.pred_subgrid_temp = cfg.pred_subgrid_temp #False
        self.use_clear_sky_region = cfg.use_clear_sky_region #True
        self.condense_supersaturated_water = False
        print("ice_sedimentation: {}, allow_extra_heating {}, pred_subgrid_temp: {}, use_clearskyreg: {}, condense_supersat: {}".format(self.ice_sedimentation, 
                        self.allow_extra_heating, self.pred_subgrid_temp, self.use_clear_sky_region, self.condense_supersaturated_water))
        if self.use_clear_sky_region:
          num_reg_cld  = self.nreg -1 
        else:
          num_reg_cld = self.nreg    
        # 
        # -------------------------------------------------------------------------------------------------
        # ---- Linear layers predicting variables needed in moist physics module from RNN hidden state ----
        # --- Flux terms ---
        self.mlp_massflux   = nn.Linear(self.nh_rnn2, self.nreg)
        # if self.do_heat_advection:
        if self.pred_subgrid_temp:
          self.mlp_eddy_diff  = nn.Linear(self.nh_rnn2, self.nreg)
        else:
          self.mlp_eddy_diff  = nn.Linear(self.nh_rnn2, 1)

        # --- Microphysical process rates ---
        self.mlp_evap_cond_vapor_crm  = nn.Linear(self.nh_rnn2, num_reg_cld)
        self.mlp_evap_prec_crm        = nn.Linear(self.nh_rnn2, self.nreg)
        self.mlp_mp_aa_crm            = nn.Linear(self.nh_rnn2, self.nreg)
        if self.ice_sedimentation:
          self.mlp_sed_qn_crm         = nn.Linear(self.nh_rnn2, self.nreg)        
        
        # --- Decoders for "downscaling" grid mean values to sub-grid values ---
        # nx_decoder = self.nh_mem0 
        nx_decoder = self.nh_rnn2
        self.mlp_qv_crm     = nn.Linear(nx_decoder, self.nreg)
        self.mlp_qn_crm     = nn.Linear(nx_decoder, num_reg_cld)

        if self.pred_subgrid_temp:
          self.mlp_t_crm      = nn.Linear(nx_decoder, self.nreg)
        if self.use_mp_constraint:
          self.mlp_qice_crm = nn.Linear(nx_decoder, self.nreg)
        else:
          self.mlp_qice_crm = nn.Linear(nx_decoder, self.nreg)
          self.mlp_qliq_crm = nn.Linear(nx_decoder, self.nreg)

        self.pred_subgrid_liq_frac = cfg.pred_subgrid_liq_frac #True
        if self.pred_subgrid_liq_frac:
          self.mlp_liq_frac_crm = nn.Linear(nx_decoder, self.nreg)

        self.mlp_subgrid_area_frac = nn.Linear(self.nh_rnn2, self.nreg)
            
        if self.predict_liq_frac:
          self.mlp_predfrac = nn.Linear(self.nh_mem, 1)

        # Physical coefficients
        self.g = 9.806650000
        self.one_over_g = 0.1019716213
        self.cp =  1004.64
        self.Ls = 2.8440e6
        self.Lv = 2.5104e6

        # --------- Physical radiation options configured from inputs or cfg ---------
        # 
        if self.use_physrad:
          yscale_sca_rad = self.yscale_sca[[0,1,4,5,6,7]]
          self.register_buffer('yscale_sca_rad', yscale_sca_rad)
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
        # Define number of spectral intervals (g-points) used in radiation computations
        # If using pre-trained gas optics models and
        #   If ng_lw != self.gas_optics_model_lw.ng, needs a decoder to shrink the spectral space after LW gas optics
        #   If ng_sw != self.gas_optics_model_sw.ng, needs a decoder to shrink the spectral space after SW gas optics
        self.ng_lw              = cfg.ng_lw #16
        self.ng_sw              = cfg.ng_sw #16
        # -----------------------------------------------------------
        # 
        # --------- Physical radiation options configured here ---------
        # 
        self.update_states_for_rad  = True  
        if self.use_physrad:
          # Update (latent/"sub-grid") variables for radiation after moist physics module ?
          self.update_states_for_rad  = True
          self.use_e3sm_cloud_optics  = cfg.use_e3sm_cloud_optics
          self.map_e3sm_cloud_optics  = cfg.map_e3sm_cloud_optics
          # When using separate gas optics module, include water vapor variability by sampling from sub-grid states and doing two passes? 
          self.include_qv_variability = True
          # Option for using TripleClouds-style solver where fluxes are computed in each sub-grid region (nreg) and g-point,
          # then summed over the regions
          # The vertical overlap between regions is handled by a matrix (v_matrix) which is here predicted with a convolutional layer
          self.experimental_rad       = False   
          # McICA-style sampling of sub-grid cloud states for each g-point
          self.use_mcica              = cfg.use_mcica # False 
          if self.experimental_rad:
            self.use_mcica=False
            if not (self.use_existing_gas_optics_sw and self.use_existing_gas_optics_lw):
              raise NotImplementedError("existing SW and LW gas optics models must be provided for experimental_rad=True")
          else:
            if self.use_e3sm_cloud_optics:
              if (self.ng_lw != self.nreg) or (self.ng_lw != self.nreg):
                print("Warning: use_e3sm_cloud_optics was true, and number of g-points did not equal number of sub-grid regions. For this to work, McICA needs to be ON")
                self.use_mcica = True
              if not self.use_mcica:
                print("Warning: McICA was off, so the radiation treats cloud sub-grid variability deterministically by each g-point being associated with a specific sub-grid region")
                print("This is a bit weird conceptually but has the advantage of keeping all gradients alive, has low computational cost, and seems to work!")

          if not (self.use_existing_gas_optics_lw or self.use_existing_gas_optics_sw):
            self.use_e3sm_cloud_optics = False

          print("use_e3sm_cld: {}, exp_rad {}, mcica: {}, include_qv_var: {}, updstates4rad {} liqfracmlp: {}".format(self.use_e3sm_cloud_optics, 
                          self.experimental_rad, self.use_mcica, self.include_qv_variability, self.update_states_for_rad, self.pred_subgrid_liq_frac))
          
          self.ny_sw_optprops     = 3  # tau_abs, tau_sca, g
          self.ny_lw_optprops     = 2  # tau_abs, planck_fraction (Fraction of Planck source associated with each g-point)
          # ^In the longwave, the radiative processes consist of absorption (gas + cloud) and emission (gas)
          # (cloud LW scattering ignored like in most climate models)
          self.nx_sw_optprops     = 4 + 3 + self.nh_mem0  + 2 
          self.nx_lw_optprops     = self.nx_sw_optprops
          self.reduce_sw_gas_optics = False
          self.reduce_lw_gas_optics = False
          self.use_new_sw_gas_optics = False
          if self.use_existing_gas_optics_lw or self.use_existing_gas_optics_sw:
            if self.use_e3sm_cloud_optics:
              print("For cloud optics, using the simple E3SM cloud optics scheme")
              if self.map_e3sm_cloud_optics and self.use_existing_gas_optics_sw:
                print("..in the SW, this includes a learned spectral mapping from e3SM cloud bands")
                self.cloud_optics_sw_expand = PositiveLinear(4, self.ng_sw) # nn.Identity()# for debugging
            if self.use_existing_gas_optics_lw:
              # We already have a pre-trained NN for gas optics (RRTMGP-NN)
              # But to reduce cost we likely want to reduce the spectral dimension..
              if self.gas_optics_model_lw.ng != self.ng_lw:
                print("Number of spectral points in existing LW gas optics model ({}) doesn't match configured ({})," \
                          "using another MLP to map between".format(self.gas_optics_model_lw.ng, self.ng_lw))
                self.reduce_lw_gas_optics = True 
                self.gas_optics_lw_reduce1 = nn.Linear(self.gas_optics_model_lw.ng, self.ng_lw)
                self.gas_optics_lw_reduce2 = nn.Linear(self.gas_optics_model_lw.ng, self.ng_lw)
              # and we still need MLPs for cloud optical properties (only abs. optical depth)
              if not self.use_e3sm_cloud_optics:
                self.cloud_optics_lw = nn.Linear(3 + self.nh_mem, self.ng_lw)

            if self.use_existing_gas_optics_sw:
              if self.gas_optics_model_sw1.ng !=  self.ng_sw:
                print("Number of spectral points in existing SW gas optics model ({}) doesn't match configured ({})," \
                        "using another MLP to map between".format(self.gas_optics_model_sw1.ng, self.ng_sw))
                self.gas_optics_sw_reduce1 = nn.Linear(self.gas_optics_model_sw1.ng, self.ng_sw)
                self.gas_optics_sw_reduce2 = nn.Linear(self.gas_optics_model_sw2.ng, self.ng_sw)
                # self.gas_optics_sw_reduce = PositiveLinear(self.gas_optics_model_sw1.ng, self.ng_sw)

                self.reduce_sw_gas_optics = True 
              elif self.ng_sw==112: # RRTMGP-NN 
                from norm_coefficients import rrtmgp_sw_solar_source
                rrtmgp_sw_solar_source = rrtmgp_sw_solar_source/np.sum(rrtmgp_sw_solar_source)
                sw_solar_weights = torch.tensor(rrtmgp_sw_solar_source, device=device).unsqueeze(0)
                self.register_buffer('sw_solar_weights', sw_solar_weights)
              else:
                self.use_new_sw_gas_optics=True
                print("Using NEW gas optics model trained from scratch on RRTMGP fluxes, but not optical properties!")
              # and we still need MLPs for cloud optical properties (optical depth, ssa, g)
              if not self.use_e3sm_cloud_optics:
                # self.cloud_optics_sw = nn.Linear(2*self.nreg+self.nh_mem +2, 3*self.ng_sw)
                self.cloud_optics_sw = nn.Linear(3 + self.nh_mem, 2*self.ng_sw)
                self.cloud_optics_sw2 = nn.Linear(2*self.ng_sw, 3*self.ng_sw)

            print("Reduce LW: {} SW: {}".format(self.reduce_lw_gas_optics, self.reduce_sw_gas_optics))
            
            if self.experimental_rad:
              # self.conv_vmat = nn.Conv1d(self.nreg, self.nreg*self.nreg, 2, stride=1)
              self.conv_vmat    = nn.Conv1d(self.nh_mem, self.nreg*self.nreg, 2, stride=1)
              self.mlp_overlap  = nn.Linear(self.nh_mem, 1)
              self.mlp_vmat     = nn.Linear(self.nreg + 1,  self.nreg*self.nreg)
          if not (self.use_existing_gas_optics_sw and (not self.reduce_sw_gas_optics)): 
            # self.sw_solar_weights   = nn.Parameter(torch.randn(1, self.ng_sw))
            self.sw_solar_weights = nn.Parameter(torch.zeros(1, self.ng_sw)) 

          if not self.use_existing_gas_optics_sw:
            # self.mlp_sw_optprops    = nn.Linear(self.nx_sw_optprops, self.ny_sw_optprops*self.ng_sw)
            self.mlp_sw_optprops1    = nn.Linear(self.nx_sw_optprops, 2*self.ng_sw)
            self.mlp_sw_optprops2    = nn.Linear(2*self.ng_sw, self.ny_sw_optprops*self.ng_sw)   
            print("Using physical radiation, but not existing SW gas optics, instead a combined MLP for cloud+gas")

          if not self.use_existing_gas_optics_lw:
            self.mlp_lw_optprops    = nn.Linear(self.nx_lw_optprops, self.ny_lw_optprops*self.ng_lw)
            print("Using physical radiation, but not existing LW gas optics, instead a combined MLP for cloud+gas")

        else:
          if self.separate_radiation:
              # Rad inputs would in reality be the state variables (except winds) updated by the CRM, plus the gases:
            self.nx_rad_gas = 3 # 'pbuf_ozone', 'pbuf_CH4', 'pbuf_N2O'
            self.nx_rad_crm = self.nh_mem0
            self.nx_rad_tot = self.nx_rad_gas + self.nx_rad_crm 
            # if not self.do_fluxrad: 
            #   self.ny_rad = 1
            self.nh_rnn1_rad = 96 
            self.nh_rnn2_rad = 96
            self.rnn1_rad   = nn.GRU(self.nx_rad_tot, self.nh_rnn1_rad,  batch_first=False) 
            self.rnn2_rad   = nn.GRU(self.nh_rnn1_rad, self.nh_rnn2_rad,  batch_first=False)  
            self.mlp_surface_init_rad = nn.Linear(5, self.nh_rnn1_rad)
            self.mlp_toa_rad  = nn.Linear(2, self.nh_rnn2_rad)
            self.nx_mlp_sfc_out_rad = self.nh_rnn2_rad
            self.nx_mlp_lev_out_rad = self.nh_rnn2_rad
          else:
            self.nx_mlp_sfc_out_rad = self.nh_rnn2
            self.nx_mlp_lev_out_rad = self.nh_rnn2

          self.mlp_surface_output_rad = nn.Linear(self.nx_mlp_sfc_out_rad, 6)
          # ['cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']
          self.mlp_output_rad = nn.Linear(self.nx_mlp_lev_out_rad, 1)

        if self.use_initial_mlp:
            self.mlp_initial = nn.Linear(self.nx, self.nneur[0])

        self.mlp_surface1  = nn.Linear(self.nx_sfc, self.nh_rnn1)

        if self.use_lstm:
            rnn_layer = nn.LSTM 
            self.mlp_surface2  = nn.Linear(self.nx_sfc, self.nh_rnn1)
        else:
            rnn_layer = nn.GRU

        self.rnn1   = rnn_layer(self.nx_rnn1, self.nh_rnn1,  batch_first=False)  # (input_size, hidden_size)
        self.rnn2   = rnn_layer(self.nx_rnn2, self.nh_rnn2,  batch_first=False)
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()
        if self.add_stochastic_layer:
            use_bias=False
            # nx_srnn = self.nh_rnn2 + self.nx_rnn1
            nx_srnn = self.nh_rnn2
            if self.use_lstm:
              self.rnn2 = MyStochasticLSTMLayer4(nx_srnn, self.nh_rnn2, use_bias=use_bias)  
            else:
              self.rnn3 = MyStochasticGRULayer5(nx_srnn, self.nh_rnn2, use_bias=use_bias)   
            # self.rnn3.flatten_parameters()
        if self.separate_radiation:
          if self.use_physrad:
            raise NotImplementedError("separate_radiation=true incompatible with use_physrad=true")
              
        if self.use_intermediate_mlp: 
          self.mlp_latent = nn.Linear(self.nh_rnn2, self.nh_mem0)
          self.mlp_output = nn.Linear(self.nh_mem0, self.ny)
        else:
          self.mlp_output = nn.Linear(self.nh_rnn2, self.ny)

    @torch.compile(dynamic=False)
    def microphysics_decode(self, inputs_denorm, delta_plev, play, P_old, out, rnn_mem, rnn2out, last_h):
        nlev, batch_size, nx  = inputs_denorm.shape 
        device                = inputs_denorm.device
        
        scaling_factor        = -self.g # tendency equation in pressure coordinates has -g in front
        pres_diff             = delta_plev[self.ilev_crm:]
        one_over_pres_diff    = 1/pres_diff

        #  mp_mode ==-1  ['ptend_t', 'dqv', 'dqn', 'liq_frac', 'ptend_u', 'ptend_v']
        #  mp_mode ==0  ['ptend_t', 'dqv', 'dqliq', "dqice", 'ptend_u', 'ptend_v']
        #  mp_mode ==1  ['ptend_t', 'dqv', 'dqn',            'ptend_u', 'ptend_v']

        qv_gcm      = inputs_denorm[self.ilev_crm:,:,-1:]
        T_gcm       = inputs_denorm[self.ilev_crm:,:,0:1]
        qliq_gcm    = inputs_denorm[self.ilev_crm:,:,2:3] 
        qice_gcm    = inputs_denorm[self.ilev_crm:,:,3:4] 
        qn_gcm      = qliq_gcm + qice_gcm
        # tlev        = interpolate_tlev_batchfirst(inputs_denorm[:,:,0], play.squeeze(), plev.squeeze()) # (nb, nlev+1)

        # --------------- 1. Expand current state-variables into "sub-grid" values, preserving mean -------------------
        #   1.1 Predict with local MLPs
        latent_state = rnn2out 
        # latent_state =  rnn_mem # rnn_mem_prev 
        qv_crm    = self.softplus(self.mlp_qv_crm(latent_state)) 
        qn_crm    = self.softplus(self.mlp_qn_crm(latent_state))
        if self.use_clear_sky_region:
          zeroes_lev = torch.zeros(self.nlev_crm, batch_size, 1, device=inputs_denorm.device)
          qn_crm = torch.cat((zeroes_lev, qn_crm),dim=-1)

        #   1.2 Scale with GCM values 
        area_frac = self.softmax(self.mlp_subgrid_area_frac(rnn2out))

        qv_mean_old = (qv_crm * area_frac).sum(dim=-1, keepdim=True)
        scale = torch.where(qv_mean_old == 0, torch.ones_like(qv_mean_old), qv_gcm / qv_mean_old)
        qv_crm = qv_crm * scale

        qn_mean_old = (qn_crm * area_frac).sum(dim=-1, keepdim=True)
        scale = torch.where(qn_mean_old == 0, torch.ones_like(qn_mean_old), qn_gcm / qn_mean_old)
        qn_crm = qn_crm * scale

        if self.pred_subgrid_temp:
          # shift, don't scale?
          deltaT = self.mlp_t_crm(latent_state)
          deltaT = deltaT - (deltaT * area_frac).sum(dim=-1, keepdim=True) # enforce zero mean
          T_crm = T_gcm + deltaT                             # add back grid mean
          # print("min max T crm", T_crm.min().item(), T_crm.max().item(), "mean", T_crm.mean().item())
          # print("min max T crm -1", T_crm[100,-1,:].min().item(), T_crm[100,-1,:].max().item())
          # print("min max T gcm", T_gcm.min().item(), T_gcm.max().item(), " ean", T_gcm.mean().item())
        else:
          T_crm = T_gcm 

        if not self.use_mp_constraint:
          qliq_crm = self.softplus(self.mlp_qliq_crm(latent_state)) 
          qice_crm = self.softplus(self.mlp_qice_crm(latent_state)) 

          qliq_mean_old = (qliq_crm * area_frac).sum(dim=-1, keepdim=True)
          scale = torch.where(qliq_mean_old == 0, torch.ones_like(qliq_mean_old), qliq_gcm / qliq_mean_old)
          qliq_crm = qliq_crm * scale

          qice_mean_old = (qice_crm * area_frac).sum(dim=-1, keepdim=True)
          scale = torch.where(qice_mean_old == 0, torch.ones_like(qice_mean_old), qice_gcm / qice_mean_old)
          qice_crm = qice_crm * scale

        # --------------- 2. Compute vertical fluxes -------------------
        flux1     = self.mlp_massflux(rnn2out)
        eddy_diff = self.mlp_eddy_diff(rnn2out)

        zeroes_single_level = torch.zeros(1, batch_size, self.nreg, device=inputs_denorm.device)
        preslay_diff0 = play[self.ilev_crm:] - play[self.ilev_crm-1:-1]
        flux_net_H = eddy_diff*(self.cp/self.g)*T_crm*preslay_diff0 #  cp/g · T · Δp 
        # flux_net_H[-1] = -self.relu(flux_net_H[-1]) # net downward flux at sfc must be upwards or zero
        if self.pred_subgrid_temp:
          zer = zeroes_single_level
        else:
          zer = torch.zeros(1, batch_size, 1, device=inputs_denorm.device)
        # flux_net_H = torch.cat((zer,flux_net_H),dim=0)
        flux_net_H = torch.cat((zer,flux_net_H[0:-1], zer),dim=0)
        flux_t_dp = (scaling_factor/self.cp)*( (flux_net_H[1:] - flux_net_H[0:-1]) * one_over_pres_diff) 
        del flux_net_H, eddy_diff

        flux_mult_coeff = 3.0e5 # just some scaling to help learning
        flux_net_qv = flux_mult_coeff*flux1*qv_crm         
        flux_net_qn = flux_mult_coeff*flux1*qn_crm
        if self.ice_sedimentation:
          if self.use_mp_constraint:
            qice_crm  = self.softplus(self.mlp_qice_crm(latent_state))
            qi_mean_old = (qice_crm * area_frac).sum(dim=-1, keepdim=True)
            scale = torch.where( #  rescale to enforce constraint exactly
                qi_mean_old == 0, torch.ones_like(qi_mean_old), qice_gcm / qi_mean_old)
            qice_crm = qice_crm * scale       

          sed = self.relu(self.mlp_sed_qn_crm(rnn2out))
          sed = sed*self.g*qice_crm*torch.reshape(self.yscale_lev[self.ilev_crm:,2],(-1,1,1))
          # sedimentation = torch.mean( sed[:,-1], 1)
          sedimentation = torch.sum( area_frac[-1]*sed[-1], -1)
          sed = torch.cat((zeroes_single_level,sed),dim=0)
          sed_qn_dp = scaling_factor*( (sed[1:] - sed[0:-1]) * one_over_pres_diff) 
          # del qice_crm
        else:
          sedimentation = 0

        flux_net_qv = torch.cat((zeroes_single_level,flux_net_qv[0:-1], zeroes_single_level),dim=0)
        flux_net_qn = torch.cat((zeroes_single_level,flux_net_qn[0:-1], zeroes_single_level),dim=0)

        flux_qv_dp = scaling_factor*( (flux_net_qv[1:] - flux_net_qv[0:-1]) * one_over_pres_diff) 
        flux_qn_dp = scaling_factor*( (flux_net_qn[1:] - flux_net_qn[0:-1]) * one_over_pres_diff) 
        del flux1, flux_net_qv, flux_net_qn

        # --------------- 3. Compute MP rates -------------------
        #   3.1: Predict rates with nonlocal MLPs
        dqv_evap_prec       = self.mlp_evap_prec_crm(rnn2out)
        dqv_evap_prec       = self.relu(dqv_evap_prec) + 1.0e-6 # force positive
        dq_cond_evap_vapor  = self.mlp_evap_cond_vapor_crm(rnn2out)
        if self.use_clear_sky_region:
          dq_cond_evap_vapor = torch.cat((zeroes_lev, dq_cond_evap_vapor),dim=-1)

        if self.store_precip: # Relate evaporated precipitation to stored precipitation here?
          # P_old_vertical = rnn_mem[:,self.ilev_crm:,0] 
          P_old_vertical = out[:,:,2] # self.mlp_prec_vertical(rnn2out)
          P_old_vertical = self.softmax_dim0(P_old_vertical) * P_old.unsqueeze(0) # sums to P_old
          dqv_evap_prec = dqv_evap_prec*P_old_vertical.unsqueeze(2) # P_old.unsqueeze(1)

        # Accretion + autoconversion. 
        # To enforce dependence on existing cloud water (more water --> more conversion), 
        # we predict "alpha" (usually a tunable parameter in MP schemes) and multiply with existing cloud water
        alpha = self.relu(self.mlp_mp_aa_crm(rnn2out)) #+ 1.0e-6 # force positive

        dqn_aa = alpha*qn_crm*torch.reshape(self.yscale_lev[self.ilev_crm:,2],(-1,1,1)) 
        # If simulations have too much cloud water, might help to enforce nonlinear dependence on qn
        # dqn_aa = alpha*torch.pow(qn_crm, 1.5)*torch.reshape(self.yscale_lev[self.ilev_crm:,2],(-1,1,1)) 

        #   3.2: Clamp values to enforce positivity of state variables 
        # first ensure positive qn by clamping dq_cond_evap_vapor
        if self.ice_sedimentation:
            ice_sed_term = sed_qn_dp
        else:
            ice_sed_term = 0 
            
        minval = -(self.yscale_lev[self.ilev_crm:,2:3].unsqueeze(2)*qn_crm/1200) - flux_qn_dp + dqn_aa - ice_sed_term
        dq_cond_evap_vapor = torch.clamp(dq_cond_evap_vapor, min=minval)

        # then ensure positive qv by clamping dqv_evap_prec
        minval = -(self.yscale_lev[self.ilev_crm:,1:2].unsqueeze(2)*qv_crm/1200) - flux_qv_dp + dq_cond_evap_vapor
        dqv_evap_prec = torch.clamp(dqv_evap_prec, min=minval)

        # maximum cloud water limit, make sure qn_new doesn't exceed qn_max
        # dqn_normed < yscale*(qnmax - qnold)/1200
        # flux_qn_dp + dq_cond_evap_vapor   - dqn_aa < yscale*(qnmax - qnold)/1200
        # flux_qn_dp + dq_cond_evap_vapor < yscale*(qnmax - qnold)/1200 + dqn_aa
        #  dqn_aa > flux_qn_dp + dq_cond_evap_vapor -yscale*(qnmax - qnold)/1200 
        qn_max = 0.0006
        if self.ice_sedimentation:
            minval = flux_qn_dp + dq_cond_evap_vapor + sed_qn_dp -(self.yscale_lev[self.ilev_crm:,2:3].unsqueeze(2)*(qn_max - qn_crm)/1200) 
        else:
            minval = flux_qn_dp + dq_cond_evap_vapor -(self.yscale_lev[self.ilev_crm:,2:3].unsqueeze(2)*(qn_max - qn_crm)/1200) 
        dqn_aa = torch.clamp(dqn_aa, min=minval)    

        # --------------- 4. Conservation equations -------------------
        # First the sub-grid tendencies are computed, and then the GCM tendencies are computed using area fractions

        #                 (cond-evap)<0 from vapor,      evap. from prec. both add water vapor  
        dqv_crm =  flux_qv_dp - dq_cond_evap_vapor     + dqv_evap_prec
        dqn_crm =  flux_qn_dp + dq_cond_evap_vapor     - dqn_aa # Autoconversion removes cloud water

        if self.ice_sedimentation: 
          dqn_crm =  dqn_crm + sed_qn_dp 

        # Temperature tendency: moist physics contribution is due to flux...
        dT_crm  =  flux_t_dp
        # ...and condensation - evaporation:
        if self.pred_subgrid_liq_frac or self.pred_subgrid_temp:
          # How should we compute latent heat release?  
          # The sub-grid liquid fraction will matter for radiation (due to very different optical properties for ice vs liquid cloud),
          # to which we pass this variable if available, and it also matters slightly for the latent heating computed here
          # it may be difficult to diagnose it via T_crm which is poorly constrained - instead, predict liq_frac_crm directly with a NN?
          if self.pred_subgrid_liq_frac:
            liq_frac_crm = self.sigmoid(self.mlp_liq_frac_crm(rnn2out))
          elif self.pred_subgrid_temp:
            # temp = T_gcm.squeeze() + (torch.sum(area_frac*dT_crm, 2)/self.yscale_lev[self.ilev_crm:,0:1]) * 1200
            # liq_frac    = torch.unsqueeze(self.temperature_scaling(temp),2); ice_frac = 1 - liq_frac
            temp = T_crm + dT_crm/self.yscale_lev[self.ilev_crm:,0:1].unsqueeze(2) * 1200
            liq_frac_crm    = self.temperature_scaling(temp)
          ice_frac_crm = 1 - liq_frac_crm
          net_condensation_crm = (1/self.cp)*((liq_frac_crm*self.Lv + ice_frac_crm*self.Ls)*dq_cond_evap_vapor - self.Lv*dqv_evap_prec)  
        else:
          # Just use the grid-mean liquid fraction when computing latent heat. This is an approximation because it ignores sub-grid variability in liquid 
          # cloud fraction, but should not matter too much for the purposes of latent heat from phase changes as Ls and Lv are quite similar
          temp = T_gcm.squeeze() + dT_crm.squeeze()/self.yscale_lev[self.ilev_crm:,0:1] * 1200
          liq_frac    = torch.unsqueeze(self.temperature_scaling(temp),2); ice_frac = 1 - liq_frac
          dq_cond_evap_vapor_s = torch.sum(area_frac*dq_cond_evap_vapor, 2, keepdim=True)
          dqv_evap_prec_s      = torch.sum(area_frac*dqv_evap_prec, 2, keepdim=True)
          net_condensation_crm = (1/self.cp)*((liq_frac*self.Lv + ice_frac*self.Ls)*dq_cond_evap_vapor_s - self.Lv*dqv_evap_prec_s)
          liq_frac_crm = liq_frac

        # undo humidity-scaling (terms like dq_cond_evap have this) and apply temperature-scaling to get heating
        net_condensation_crm = (net_condensation_crm/self.yscale_lev[self.ilev_crm:,1:2].unsqueeze(2)) * self.yscale_lev[self.ilev_crm:,0:1].unsqueeze(2)
        # print("mean max cond crm", net_condensation_crm.abs().mean().item(), net_condensation_crm.abs().max().item())
        # print("mean max dT crm 1", dT_crm.abs().mean().item(), dT_crm.abs().max().item())
        dT_crm = dT_crm + net_condensation_crm 

        dT = torch.sum(area_frac*dT_crm, 2, keepdim=True)
          
        # Water vapor
        dqv = torch.sum(area_frac*dqv_crm, 2, keepdim=True)
        # Cloud tendencies
        if self.use_mp_constraint:
          # Predicting dqn
          dqn = torch.sum(area_frac*dqn_crm, 2, keepdim=True) 
        else:
          # Predicting dqliq, dqice
          T_crm = T_crm + dT_crm/self.yscale_lev[self.ilev_crm:,0:1].unsqueeze(2)
          liq_frac_crm_new = self.temperature_scaling(T_crm)
          # liq_frac_crm_new = liq_frac_crm_old
          # print("mean liq fracm gcm", liq_frac.mean().item(), "crm", liq_frac_crm_new.mean().item())
          # print("std liq fracm gcm", liq_frac.std().item(), "crm", liq_frac_crm_new.std().item())
          # print("mean qn_crm", qn_crm.mean().item(), "dqn crm", dqn_crm.mean().item(), "max", dqn_crm.max().item())
          qn_crm        = qn_crm + dqn_crm/self.yscale_lev[self.ilev_crm:,2:3].unsqueeze(2)
          # print("mean qn_crm 2", qn_crm.mean().item())
          qliq_crm_new  = qn_crm * liq_frac_crm_new
          qice_crm_new  = qn_crm * (1-liq_frac_crm_new)
          dqliq = ((qliq_crm_new - qliq_crm)*area_frac).sum(dim=-1)/1200 
          dqice = ((qice_crm_new - qice_crm)*area_frac).sum(dim=-1)/1200 
          dqliq = dqliq*self.yscale_lev[self.ilev_crm:,2].unsqueeze(2) 
          dqice = dqice*self.yscale_lev[self.ilev_crm:,3].unsqueeze(2)

        #                              precipitation  source,       sink    (signs already reversed w.r.t. above)
        d_precip_sourcesink    = torch.sum(area_frac*(dqn_aa      - dqv_evap_prec),2)
        # This *should* be positive to avoid negative precipitation!

        if self.condense_supersaturated_water and self.use_mp_constraint:
            qv_new = self.relu(qv_gcm + 1200*dqv/self.yscale_lev[self.ilev_crm:,1:2].unsqueeze(2))
            temp = self.relu(T_gcm + 1200*dT/self.yscale_lev[self.ilev_crm:,0:1].unsqueeze(2))
            qv_excess = (1/1200) * self.specific_to_relative_humidity_torch_cc(qv_new, temp, play[self.ilev_crm:], return_excess=True)
            # print("mean max qv ex", qv_excess.mean().item(), qv_excess.max().item())
            dqv = dqv -  qv_excess*self.yscale_lev[self.ilev_crm:,1:2].unsqueeze(2)
            dqn = dqn +  qv_excess*self.yscale_lev[self.ilev_crm:,2:3].unsqueeze(2)
            if self.pred_subgrid_liq_frac or self.pred_subgrid_temp:
              liq_frac    = self.temperature_scaling(temp); ice_frac = 1 - liq_frac
            net_condensation =  (1/self.cp)*((liq_frac*self.Lv + ice_frac*self.Ls)*qv_excess)* self.yscale_lev[self.ilev_crm:,0:1].unsqueeze(2)
            # print("mean max dT", out_new[:,self.ilev_crm:,0:1].mean().item(), out_new[:,self.ilev_crm:,0:1].max().item())
            # print("mean max cond", net_condensation.mean().item(), net_condensation.max().item())
            dT  = dT  + net_condensation

        # --------------- 5. Precipitation -------------------
        water_new = torch.sum((self.one_over_g*pres_diff.squeeze()*d_precip_sourcesink),0)  

        if self.store_precip:
          water_new = P_old + water_new
          prec_negative = self.relu(-water_new) # punish model for diagnosing negative precip from column water changes?
          water_new = self.relu(water_new)
          precc_release_fraction = torch.sigmoid(self.mlp_precip_release(last_h)).squeeze()
          water_released = precc_release_fraction*water_new
          water_stored  = water_new*(1-precc_release_fraction)
          # Just clipping the stored water here is incorrect, because we break the mass conservation! 
          # Instead compute the excess and add it to precipitation?
          # We need some physical limit on the amount of falling precipitation in the column
          Tsfc = inputs_denorm[-1,:,0]
          Pmax = 1000 *  self.yscale_sca[3] * 5.58e-18*torch.exp(0.077*Tsfc) # approximate fit from data
          water_excess = water_stored - Pmax
          water_excess = self.relu(water_excess)
          water_stored = water_stored - water_excess
          water_stored_lev  = torch.unsqueeze(water_stored,dim=0)
          water_stored_lev = torch.unsqueeze(torch.repeat_interleave(water_stored_lev,self.nlev_mem,dim=0),dim=2)
          rnn_mem = torch.cat((rnn_mem[:,:,0:self.nh_mem], water_stored_lev),dim=2)
          precip=  sedimentation + water_released + water_excess # - prec_negative
        else:
          prec_negative = self.relu(-water_new) # punish model for diagnosing negative precip from column water changes?
          water_new = self.relu(water_new)
          precip =  sedimentation + water_new  # <-- we already reversed signs in d_precip_sourcesink

        precc = (precip/1000).unsqueeze(1) # Total precip (rain+snow)
        temp_sfc = inputs_denorm[-1,:,0:1]
        snowfrac = self.temperature_scaling_precip(temp_sfc)
        precsc = snowfrac*precc # Snow

        # Finally, update subgrid values for radiation
        if self.use_physrad:
          if self.update_states_for_rad:
            qv_crm = self.relu(qv_crm + 1200*dqv_crm/self.yscale_lev[self.ilev_crm:,1:2].unsqueeze(2))
            qn_crm = self.relu(qn_crm + 1200*dqn_crm/self.yscale_lev[self.ilev_crm:,2:3].unsqueeze(2))
            # T_crm  = self.relu(T_crm  + 1200*dT_crm/self.yscale_lev[self.ilev_crm:,0:1].unsqueeze(2))

        out_new               = torch.zeros(self.nlev, batch_size, self.ny, device=device)
        #  mp_mode ==-1  ['ptend_t', 'dqv', 'dqn', 'liq_frac', 'ptend_u', 'ptend_v']
        #  mp_mode ==0   ['ptend_t', 'dqv', 'dqliq', "dqice", 'ptend_u', 'ptend_v']
        #  mp_mode ==1   ['ptend_t', 'dqv', 'dqn',            'ptend_u', 'ptend_v']
        out_new[self.ilev_crm+2:,:,-2:] = out[2:,:,-2:] # winds are predicted with pure ML

        # if self.allow_extra_heating:
        #   out_new[self.ilev_crm+2:,:,0] = out[2:,:,0]
        # ignore top two layers of the CRM here? These are ignored when adding CRM tendencies to GCM in original simulations?
        if self.allow_extra_heating:
          out_new[self.ilev_crm:,:,0:1] = out[:,:,0:1] + dT
        else:
          out_new[self.ilev_crm:,:,0:1] = dT
        out_new[self.ilev_crm:,:,1:2] = dqv
        if self.use_mp_constraint: 
          out_new[self.ilev_crm:,:,2:3] = dqn
        else:
          out_new[self.ilev_crm:,:,2] = dqliq
          out_new[self.ilev_crm:,:,3] = dqice

        return out_new, precc, precsc, rnn_mem, liq_frac_crm, qv_crm, qn_crm, area_frac, prec_negative

    @torch.compile(dynamic=False)
    def expand_slingo_to_bands(self,x4):
        """
        Map Slingo 4-band optical property (..., 4) to ng g-points (..., ng).

        Slingo coeffs index mapping (from wavmax Fortran cutoffs):
          index 3 (band 4): < 4,200   cm-1  → your band 1: bb[0]:bb[1]
          index 2 (band 3): 4,200–8,000 cm-1  → your band 2 lower half: bb[1]:i_b32
          index 1 (band 2): 8,000–14,286 cm-1 → your band 2 upper half: i_b32:bb[2]
          index 0 (band 1): > 14,286 cm-1     → your bands 3+4+5: bb[2]:bb[5]

        The Slingo band 3/2 boundary at 8,000 cm-1 sits at fraction
        (8000-4200)/(14500-4200) = 0.369 through your band 2.
        """
        bb = self.gas_optics_model_sw1.band_bounds  # [0, 4, 11, 13, 15, 16] for ng=16

        # Slingo band 3/2 boundary within your band 2
        # i_b32 = bb[1] + int(round(((8000 - 4200) / (14500 - 4200)) * (bb[2] - bb[1])))
        # For ng=16: 4 + round(0.369 * 7) = 4 + 3 = 7

        shape = x4.shape[:-1]
        out = torch.empty(*shape, self.ng_sw, dtype=x4.dtype, device=x4.device)

        # Your band 1 (250–4200) ← Slingo band 4 (index 3)
        out[..., bb[0]:bb[1]] = x4[..., 3:4].expand(*shape, bb[1] - bb[0])

        # # Your band 2 lower (4200–8000) ← Slingo band 3 (index 2)
        # out[..., bb[1]:i_b32] = x4[..., 2:3].expand(*shape, i_b32 - bb[1])

        # # Your band 2 upper (8000–14500) ← Slingo band 2 (index 1)
        # out[..., i_b32:bb[2]] = x4[..., 1:2].expand(*shape, bb[2] - i_b32)

        slingo_mix =  0.5*(x4[..., 2:3] + x4[..., 1:2]).expand(*shape, bb[2] - bb[1])
        out[..., bb[1]:bb[2]] = 0.5*slingo_mix

        # Your bands 3+4+5 (14500–50000) ← Slingo band 1 (index 0)
        out[..., bb[2]:bb[5]] = x4[..., 0:1].expand(*shape, bb[5] - bb[2])

        return out

    @torch.compile(dynamic=False)
    def rad_optical_props(self, inputs_main, inputs_aux0, inputs_denorm, play, plev, delta_plev, rnn_mem, 
                liq_frac_crm, qv_crm, qn_crm, T_new, qv_new, qn_new, area_frac, rnn2out):
      """
      Prediction of atmospheric (gas + cloud) optical properties for SW and LW using correlated-k or similar methods
      Fully parallelizable (no vertical loops)
      This module is a bit complicated mainly because we are lacking pre-trained spectrally reduced gas and cloud optics
      NNs. We can use pre-existing RRTMGP-NN models (Ukkonen & Hogan 2022) but these have high spectral resolution (112-118)
      so we probably want to combine this with training a new MLP to shrink the dimension (reduce_LW_gas_optics)
      Furthermore there various options for:
        - using or not using the cloud optics scheme from E3SM 
        - representing water vapor sub-grid variability in gas optics (include_qv_variability)
        - representing cloud sub-grid variability : 
            - use_mcica=True : McICA, used in most GCMs
            - use_mcica=False: a deterministic approach assuming ng=nreg which seems to work here but is not very interpretable
            - experimental_rad=True : TripleClouds-like but we don't have the cloud overlap matrices so experimental at best
      """
      nlev, batch_size, nx = inputs_main.shape 
      device = inputs_main.device
      inputs_aux =  (inputs_aux0*self.xdiv_sca) + self.xmean_sca
      
      # Water vapor specific humidity to volume mixing ratio - needed by gas optics
      vmr_h2o = (qv_new / ( 1.0 - qv_new))  *  1.608079364 # 28.97 / 18.01528 # mol_weight_air / mol_weight_gas
      fact = 1/ (1 + vmr_h2o)
      m_air = (0.04698 + vmr_h2o)*fact # (m_dry + m_h2o * vmr_h2o) * fact 
      # Compute path number of dry air molecules in a column N (mol m−2) - used for scaling predicted optical properties
      #                             avogad
      col_dry = 10.0 * delta_plev * 6.02214076e23 * fact/(1000.0*m_air*100.0*9.80665) 
      col_dry = torch.reshape(col_dry,(nlev,batch_size,1))
      log_play = torch.log(play)

      if self.use_existing_gas_optics_lw or self.use_existing_gas_optics_sw:
        # inputs needed for gas optical properties
        pres1 = log_play
        vmr_h2o = (torch.sqrt(torch.sqrt(vmr_h2o)))
        o3 =  inputs_denorm[:,:,12:13] 
        o3 = torch.sqrt(torch.sqrt(o3))
        ch4 =  inputs_denorm[:,:,13:14]
        n2o =  inputs_denorm[:,:,14:15]
        co2 =  torch.full((self.nlev, batch_size, 1), 388.7e-6, device=device)

        # inputs needed for cloud optical properties: ice and liquid cloud effective radius using the LUT interpolation routine from E3SM-MMF
        T_new1 = T_new[self.ilev_crm:].view(-1)
        ice_eff_rad = self.reitab(T_new1)
        ice_eff_rad = ice_eff_rad.view((self.nlev_crm, batch_size,1))
        icefrac   =  torch.repeat_interleave(inputs_aux[:,12].view(1,-1),self.nlev_crm,dim=0)
        landfrac  =  torch.repeat_interleave(inputs_aux[:,13].view(1,-1),self.nlev_crm,dim=0)
        snowh     =  torch.repeat_interleave(inputs_aux[:,15].view(1,-1),self.nlev_crm,dim=0)
        liq_eff_rad = self.reltab(T_new1, landfrac.view(-1), icefrac.view(-1), snowh.view(-1))
        liq_eff_rad = liq_eff_rad.view((self.nlev_crm, batch_size,1))

        # print("min max mean ice_eff_rad", ice_eff_rad.min().item(), ice_eff_rad.max().item(), ice_eff_rad.mean().item())
        # print("min max mean liq_eff_rad", liq_eff_rad.min().item(), liq_eff_rad.max().item(), liq_eff_rad.mean().item())
        # print("min max mean T_new1", T_new1.min().item(), T_new1.max().item(), T_new1.mean().item())

        # -------- ACCOUNTING FOR CLOUD SUB-GRID VARIABILITY ---------
        # 
        if self.use_e3sm_cloud_optics: 
          #  ------------------- McICA style randomization of which cloud state each g-point sees -------------------
          if self.use_mcica:
            # Three different options for McICA-inspired stochastic cloud sampling:
            #   1) Just shuffle the sub-grid cloud states along the hidden dimension (ncol_mp for cloud variables, ng for rad variables),
            #      requires  ncol_mp = ng_sw = ng_lw and doesn't account for area fractions
            # oops, code below uses same permutation for each sample, we don't want that!
            # idx               = torch.randperm(self.nreg) 
            # qn_crm            = qn_crm[:,:,idx]
            # T_crm             = T_crm[:,:,idx]
            #   2) True McICA? Sample from sub-grid states according to their probability p (area_frac) until we have ng samples (g-points)
            #  For a given sample, a low-p cloud state may not selected at all, but because we do random draws we will occasionally
            #  sample also low-p states and the estimate will be unbiased (but noisy) 

            if self.use_clear_sky_region: # the first index (area_frac[:,:,0]) is a clear-sky region without clouds
              p_flat = area_frac[:,:,1:]
              p_flat = p_flat * 1/torch.sum(p_flat,dim=-1,keepdim=True) # renorm so that it sums to 1
              qn_crm = qn_crm[:,:,1:]
              # if self.pred_subgrid_temp and not self.pred_subgrid_liq_frac:
              #   T_crm = T_crm[:,:,1:]
              if self.pred_subgrid_liq_frac or self.pred_subgrid_temp:
                liq_frac_crm = liq_frac_crm[:,:,1:]
              nreg = self.nreg - 1 
            else:
              nreg = self.nreg
              p_flat = area_frac 

            p_flat = p_flat.view(-1, nreg)
            qn_crm_flat = qn_crm.view(-1, nreg)
            # if self.pred_subgrid_temp and not self.pred_subgrid_liq_frac:
            #   T_crm_flat = T_crm.view(-1, nreg)
            if self.pred_subgrid_liq_frac or self.pred_subgrid_temp:
              liq_frac_crm_flat = liq_frac_crm.view(-1, nreg)

            # indices = torch.multinomial(p_flat, num_samples=self.ng_sw, replacement=True)
            # T_crm = torch.gather(T_crm_flat, dim=-1, index=indices).reshape(self.nlev_crm, batch_size, self.ng_sw)
            # qn_crm = torch.gather(qn_crm_flat, dim=-1, index=indices).reshape(self.nlev_crm, batch_size, self.ng_sw)

            # #  3) Rather than purely random assignment, deterministically partition g-points among states proportional to p, then shuffle.
            # # Same cost as McICA, zero bias, lower variance
            indices = self.stratified_sample(p_flat, self.ng_lw) #, shuffle=False)
            # if self.pred_subgrid_temp and not self.pred_subgrid_liq_frac:
            #   T_crm = torch.gather(T_crm_flat, dim=-1, index=indices).view(self.nlev_crm, batch_size, self.ng_lw)
            qn_crm = torch.gather(qn_crm_flat, dim=-1, index=indices).view(self.nlev_crm, batch_size, self.ng_lw)
            if self.pred_subgrid_liq_frac or self.pred_subgrid_temp:
              liq_frac_crm = torch.gather(liq_frac_crm_flat, dim=-1, index=indices).view(self.nlev_crm, batch_size, self.ng_lw)

          # This would be how things are done in the underlying CRM: temperature on CRM grid is used to diagnose cloud into liquid and ice
          # if not self.pred_subgrid_liq_frac:
          #   liq_frac_crm          = self.temperature_scaling(T_crm.squeeze())
          # # However it may be difficult to optimize T_crm - instead, predict liq_frac_crm with a NN?
          # else:
          #   liq_frac_crm = self.sigmoid(self.mlp_liq_frac_crm(rnn2out))
          #   if self.use_mcica:
          #     liq_frac_crm = torch.gather(liq_frac_crm.view(-1,self.nreg), dim=-1, index=indices).view(self.nlev_crm, batch_size, self.ng_lw)

          cldpath_tot       = 1000*(delta_plev[self.ilev_crm:]/self.g)*qn_crm
          cldpath_liq       = liq_frac_crm * cldpath_tot 
          cldpath_ice       = (1-liq_frac_crm) * cldpath_tot

        else: # if not self.use_e3sm_cloud_optics:
          # t_new_crm   = (T_crm - 160 ) / (180)
          t_new_crm   = (T_new - 160 ) / (180)
          liq_eff_rad_normed  = liq_eff_rad / 13.5
          ice_eff_rad_normed   = ice_eff_rad / 125.0
          x_cld =  torch.cat((t_new_crm, ice_eff_rad_normed, liq_eff_rad_normed, rnn_mem), dim=2)
          # the cloud optical depth is scaled with cloud path 
          cldpath    = 1000*(delta_plev[self.ilev_crm:]/self.g)*qn_crm
          # print("cldpath min max mean", cldpath.min().item(), cldpath.max().item(), cldpath.mean().item()

        # ------ LONGWAVE OPTICAL PROPERTIES -------
        if self.use_existing_gas_optics_lw:  
          zero_gases = torch.zeros(self.nlev, batch_size, 11, device=device)
          x_gas = torch.cat((T_new, pres1, vmr_h2o, o3, co2, ch4, n2o, zero_gases), dim=2)
          x_gas = (x_gas - self.gas_optics_model_lw.xmin) / self.gas_optics_model_lw.xdiv
          # gasopt_vars = ["T","p","h2o","o3","co2","ch4","n20"]
          # for i in range(len(gasopt_vars)):
          #   print("x_gas {} min {} max {}".format(gasopt_vars[i],torch.min(x_gas[:,:,i]).item(), torch.max(x_gas[:,:,i]).item()))
          x_gas = self.relu(x_gas)
          
          tau_lw, pfrac = self.gas_optics_model_lw(x_gas, col_dry)

          if self.reduce_lw_gas_optics: # we used an existing gas optics scheme, but want to shrink the spectral dimension with an MLP
            pfrac   = self.softmax(self.gas_optics_lw_reduce2(pfrac)) # (nlev, nb, g)
            tau_lw  = self.gas_optics_lw_reduce1(tau_lw)
            tau_lw  = 0.01*self.softplus(tau_lw)
          
          if self.printdebug:
            print("tau_lw min max mean", tau_lw.min().item(), tau_lw.max().item(), tau_lw.mean().item())
            print("pfrac min max mean", pfrac.min().item(), pfrac.max().item(), pfrac.mean().item())

          zeroes = torch.zeros(self.ilev_crm, batch_size, self.ng_lw, device=device)
        
          if self.use_e3sm_cloud_optics:
            # slingo_liq_optics_lw: https://github.com/NVlabs/E3SM/blob/main/components/eam/src/physics/rrtmgp/slingo.F90#L147
            # Strange that there's no spectral dependence, cannot be very accurate!
            icefrac         = cldpath_ice / torch.clamp(cldpath_tot, min=1.0e-8)
            tau_lw_cld_liq  =  cldpath_tot*0.090361*(1-icefrac)

            # ec_ice_optics_lw : https://github.com/NVlabs/E3SM/blob/main/components/eam/src/physics/rrtmgp/ebert_curry.F90#L132
            tau_lw_cld_ice  = cldpath_tot*icefrac*(0.005 + 1.0 / ice_eff_rad.clamp(13.0, 130.0))
            tau_lw_cld      = tau_lw_cld_liq + tau_lw_cld_ice        
          else:
            tau_lw_cld = cldpath*self.relu(self.cloud_optics_lw(x_cld)) # 0.01*self.relu(tau_lw_cld)

          if not self.use_mcica and (self.nreg != self.ng_lw): 
            # We have gray LW gas optics so we can just repeat the nreg until we have ng
            tau_lw_cld = torch.repeat_interleave(tau_lw_cld, self.ng_lw//self.nreg, dim=2)

          tau_lw_cld  = torch.cat((zeroes, tau_lw_cld),dim=0)
          tau_lw      = tau_lw + tau_lw_cld # shape is (nlev, ncol, ng)

        # ------ SHORTWAVE OPTICAL PROPERTIES -------
        if self.use_existing_gas_optics_sw:
            # inputs: ['tlay' 'play' 'h2o' 'o3' 'co2' 'n2o' 'ch4']
          if self.include_qv_variability:
            # Account for sub-grid variability of water vapor by taking the two most likely values and doing two passes with gas opt MLP
            qv_crm = torch.clamp(qv_crm, max=0.05)
            vmr_h2o_crm = (qv_crm / (1.0-qv_crm))*1.608079364 

            k = 2
            i0 = torch.arange (self.nlev_crm).unsqueeze (-1).unsqueeze (-1).expand(self.nlev_crm, batch_size, k)
            i1 = torch.arange (batch_size).unsqueeze (0).unsqueeze (-1).expand(self.nlev_crm, batch_size, k)
            i2 =  torch.topk (area_frac, k, dim = -1).indices
            vmr_sorted = vmr_h2o_crm[i0,i1,i2]
            vmr_h2o_1,vmr_h2o_2 = vmr_sorted.chunk(k,2)
            vmr_h2o_top = vmr_h2o[0:10]
            vmr_h2o_1 = torch.cat((vmr_h2o_top, vmr_h2o_1),dim=0).contiguous()
            vmr_h2o_2 = torch.cat((vmr_h2o_top, vmr_h2o_2),dim=0).contiguous()

            fact = 1/ (1 + vmr_h2o_1)
            col_dry_crm_1 = delta_plev * 60.2214076e23 * fact/(((0.04698 + vmr_h2o_1)*fact)*980665) 
            vmr_h2o_1 = torch.sqrt(torch.sqrt(vmr_h2o_1))

            fact = 1/ (1 + vmr_h2o_2)
            col_dry_crm_2 = delta_plev * 60.2214076e23 * fact/(((0.04698 + vmr_h2o_2)*fact)*980665) 
            vmr_h2o_2 = torch.sqrt(torch.sqrt(vmr_h2o_2))

            x_gas_1 = torch.cat((T_new, pres1, vmr_h2o_1, o3, co2, n2o, ch4), dim=2)
            x_gas_1 = (x_gas_1- self.gas_optics_model_sw1.xmin) / self.gas_optics_model_sw1.xdiv

            x_gas_2 = x_gas_1.clone()
            x_gas_2[:, :, 2:3] = (vmr_h2o_2 - self.gas_optics_model_sw1.xmin[2:3]) / self.gas_optics_model_sw1.xdiv[2:3]
            # Use SW gas optics absorption NN : two passes to sample water vapor sub-grid variability, then merge
            # print("x_gas_1 min max", x_gas_1.min(), x_gas_1.max(), "col dry mean", col_dry_crm_1.mean())
            tau_sw1     = self.gas_optics_model_sw1(x_gas_1, col_dry_crm_1) # Absorption optical depth
            tau_sw_scat1= self.gas_optics_model_sw2(x_gas_1, col_dry_crm_1) # Scattering optical depth
            tau_sw2     = self.gas_optics_model_sw1(x_gas_2, col_dry_crm_2)
            tau_sw_scat2= self.gas_optics_model_sw2(x_gas_2, col_dry_crm_2)

            # mask          = torch.rand_like(tau_sw1) < 0.5
            # tau_sw        = torch.where(mask, tau_sw1, tau_sw2)
            # tau_sw_scat   = torch.where(mask, tau_sw_scat1, tau_sw_scat2)
            tau_sw        = 0.5*(tau_sw1+tau_sw2)
            tau_sw_scat   = 0.5*(tau_sw_scat1+tau_sw_scat2)

          else:

            x_gas = torch.cat((T_new, pres1, vmr_h2o, o3, co2, n2o, ch4), dim=2)
            x_gas = (x_gas - self.gas_optics_model_sw1.xmin) / self.gas_optics_model_sw1.xdiv
            # for i in range(x_gas.shape[-1]):
            #  print("x_gas {} min {} max {} mean {}".format(i,torch.min(x_gas[:,:,i]).item(), torch.max(x_gas[:,:,i]).item(), torch.mean(x_gas[:,:,i]).item()))  
            tau_sw     = self.gas_optics_model_sw1(x_gas, col_dry)
            tau_sw_scat= self.gas_optics_model_sw2(x_gas, col_dry)

          if self.use_new_sw_gas_optics:
            tau_sw      = torch.clamp(tau_sw,min=1e-9)

          if self.reduce_sw_gas_optics:
            tau_sw      = 0.01*self.softplus(self.gas_optics_sw_reduce1(tau_sw)) + 1.0e-9
            tau_sw_scat = 0.01*self.softplus(self.gas_optics_sw_reduce2(tau_sw_scat)) # + 1.0e-9
            
          # print("col dry mean", col_dry.mean().item())
          # print("GAS tau_sw min max mean", tau_sw.min().item(), tau_sw.max().item(), tau_sw.mean().item())
          # print("GAS tau_sw_scat min max mean", tau_sw_scat.min().item(), tau_sw_scat.max().item(), tau_sw_scat.mean().item())

          # Total (gas) optical depth is the absorption optical depth plus scattering optical depth
          tau_sw  = tau_sw + tau_sw_scat

          if self.experimental_rad:
              tau_sw       = torch.repeat_interleave(tau_sw.unsqueeze(3),self.nreg,dim=3)
              tau_sw_scat  = torch.repeat_interleave(tau_sw_scat.unsqueeze(3),self.nreg,dim=3)
              zeroes = torch.zeros(10, batch_size, self.ng_sw, self.nreg, device=device)
          else:
              if not self.ng_sw==self.ng_lw:
                zeroes = torch.zeros(10, batch_size, self.ng_sw, device=device)
            
          if self.use_e3sm_cloud_optics:
            # E3SM uses cloud optics in 4 bands. Here we need to map them to our spectral discretization. 

            if self.map_e3sm_cloud_optics: 
              # use an MLP to map E3SM cloud optics to our spectral discretization. This option is used when
              # training new gas optics on the fly.
              k_sw_cld_liq0, ssa_sw_cld_liq0, g_sw_cld_liq0 = self.slingo_liq_cloud_optics_sw(liq_eff_rad)
              ksca_sw_cld_liq0  = k_sw_cld_liq0*ssa_sw_cld_liq0
              kscag_sw_cld_liq0 = ksca_sw_cld_liq0*g_sw_cld_liq0
              k_sw_cld_liq      = self.cloud_optics_sw_expand(k_sw_cld_liq0)
              ksca_sw_cld_liq   = self.cloud_optics_sw_expand(ksca_sw_cld_liq0) # / k_sw_cld_liq 
              kscag_sw_cld_liq  = self.cloud_optics_sw_expand(kscag_sw_cld_liq0) # / (k_sw_cld_liq*ssa_sw_cld_liq)
              k_sw_cld_ice0, ssa_sw_cld_ice0, g_sw_cld_ice0 = self.ec_ice_optics_sw(ice_eff_rad)
              ksca_sw_cld_ice0  = k_sw_cld_ice0*ssa_sw_cld_ice0
              kscag_sw_cld_ice0 = ksca_sw_cld_ice0*g_sw_cld_ice0
              k_sw_cld_ice    = self.cloud_optics_sw_expand(k_sw_cld_ice0)
              ksca_sw_cld_ice = self.cloud_optics_sw_expand(ksca_sw_cld_ice0)
              kscag_sw_cld_ice  = self.cloud_optics_sw_expand(kscag_sw_cld_ice0)
            elif self.use_new_sw_gas_optics:
                # Get cloud optics in Slingo's native 4 bands (no learned mapping needed)
                # slingo returns (k, ssa, g) each of shape (..., 4)
                k_sw_cld_liq4,   ssa_sw_cld_liq4,   g_sw_cld_liq4   = self.slingo_liq_cloud_optics_sw(liq_eff_rad, ng=4)
                k_sw_cld_ice4,   ssa_sw_cld_ice4,   g_sw_cld_ice4   = self.ec_ice_optics_sw(ice_eff_rad, ng=4)

                bb = self.gas_optics_model_sw1.band_bounds  # [0, b1, b2, b3, b4, ng_sw]
                ng = self.ng_sw

                k_sw_cld_liq   = self.expand_slingo_to_bands(k_sw_cld_liq4)
                ssa_sw_cld_liq = self.expand_slingo_to_bands(ssa_sw_cld_liq4)
                g_sw_cld_liq   = self.expand_slingo_to_bands(g_sw_cld_liq4)
                ksca_sw_cld_liq  = k_sw_cld_liq  * ssa_sw_cld_liq
                kscag_sw_cld_liq = ksca_sw_cld_liq * g_sw_cld_liq

                k_sw_cld_ice   = self.expand_slingo_to_bands(k_sw_cld_ice4)
                ssa_sw_cld_ice = self.expand_slingo_to_bands(ssa_sw_cld_ice4)
                g_sw_cld_ice   = self.expand_slingo_to_bands(g_sw_cld_ice4)
                ksca_sw_cld_ice  = k_sw_cld_ice  * ssa_sw_cld_ice
                kscag_sw_cld_ice = ksca_sw_cld_ice * g_sw_cld_ice
            else:
              k_sw_cld_liq, ksca_sw_cld_liq, kscag_sw_cld_liq = self.slingo_liq_cloud_optics_sw(liq_eff_rad, self.ng_sw)
              ksca_sw_cld_liq  = k_sw_cld_liq*ksca_sw_cld_liq # ksca_sw_cld_liq is ssa before this line
              kscag_sw_cld_liq = ksca_sw_cld_liq*kscag_sw_cld_liq # kscag_sw_cld_liq is g before this line
              k_sw_cld_ice, ksca_sw_cld_ice, kscag_sw_cld_ice = self.ec_ice_optics_sw(ice_eff_rad, self.ng_sw)
              ksca_sw_cld_ice  = k_sw_cld_ice*ksca_sw_cld_ice
              kscag_sw_cld_ice = ksca_sw_cld_ice*kscag_sw_cld_ice
              

            if self.experimental_rad: 
                cldpath_liq = torch.repeat_interleave(cldpath_liq.unsqueeze(2), self.ng_sw,dim=2)
                cldpath_ice = torch.repeat_interleave(cldpath_ice.unsqueeze(2), self.ng_sw,dim=2)
                k_sw_cld_liq = torch.repeat_interleave(k_sw_cld_liq.unsqueeze(3), self.nreg,dim=3)
                k_sw_cld_ice = torch.repeat_interleave(k_sw_cld_ice.unsqueeze(3), self.nreg,dim=3)
                ksca_sw_cld_liq = torch.repeat_interleave(ksca_sw_cld_liq.unsqueeze(3), self.nreg,dim=3)
                ksca_sw_cld_ice = torch.repeat_interleave(ksca_sw_cld_ice.unsqueeze(3), self.nreg,dim=3)
                kscag_sw_cld_liq = torch.repeat_interleave(kscag_sw_cld_liq.unsqueeze(3), self.nreg,dim=3)
                kscag_sw_cld_ice = torch.repeat_interleave(kscag_sw_cld_ice.unsqueeze(3), self.nreg,dim=3)

            cldeps = 1e-7
            # optical depth (dimensionless) = cldpath (1000 kg/m2) *  k (m2/g) =  (kg/m2) *  k (1000 g/kg m2/g)
            tau_sw_cld_liq  = cldpath_liq*k_sw_cld_liq
            tau_sw_cld_ice  = cldpath_ice*k_sw_cld_ice
            tau_sw_scat_cld_liq  = cldpath_liq*ksca_sw_cld_liq
            tau_sw_scat_cld_ice  = cldpath_ice*ksca_sw_cld_ice
            g_tau_scat_sw_cld_liq  = cldpath_liq*kscag_sw_cld_liq
            g_tau_scat_sw_cld_ice  = cldpath_ice*kscag_sw_cld_ice

            tau_sw_cld          = tau_sw_cld_ice + tau_sw_cld_liq
            tau_sw_scat_cld     = tau_sw_scat_cld_liq + tau_sw_scat_cld_ice 
            g_tau_scat_sw_cld   = g_tau_scat_sw_cld_liq + g_tau_scat_sw_cld_ice 
            # g_sw_cld = torch.where(tau_sw_scat_cld == 0, torch.zeros_like(tau_sw_scat_cld), g_tau_scat_sw_cld / tau_sw_scat_cld)
            # g_sw_cld            =  g_tau_scat_sw_cld / tau_sw_scat_cld
            g_sw_cld            = g_tau_scat_sw_cld / (tau_sw_scat_cld + cldeps)
            if self.printdebug:
              print("max min mean gswcld", g_sw_cld.max().item(), g_sw_cld.min().item(),  g_sw_cld.mean().item())
              print("min max mean tau_sw", tau_sw.min().item(), tau_sw.max().item(), tau_sw.mean().item())
              print("min max mean tau_sw_cld", tau_sw_cld.min().item(), tau_sw_cld.max().item(), tau_sw_cld.mean().item())
              print("max mean tau_sw_cld*ssa", tau_sw_scat_cld.max().item(), tau_sw_scat_cld.min().item(), tau_sw_scat_cld.mean().item())
              print("max mean tau_sw_cld*ssa*g", g_tau_scat_sw_cld.max().item(), g_tau_scat_sw_cld.min().item(),g_tau_scat_sw_cld.mean().item())

            del tau_sw_cld_ice, tau_sw_cld_liq, tau_sw_scat_cld_liq, tau_sw_scat_cld_ice, g_tau_scat_sw_cld_liq
            del g_tau_scat_sw_cld_ice, g_tau_scat_sw_cld, k_sw_cld_liq, ksca_sw_cld_liq, k_sw_cld_ice, ksca_sw_cld_ice

          else: 
            sw_optprops = self.cloud_optics_sw(x_cld)
            sw_optprops = self.cloud_optics_sw2(sw_optprops) 
            # if we're not using E3SM cloud optics, we predict cloud single-scattering albedo and asymmetry factor directly
            tau_sw_cld, ssa_sw_cld, g_sw_cld = sw_optprops.chunk(3,2 )
            tau_sw_cld      = cldpath*self.relu(tau_sw_cld.squeeze()) # / 1000
            ssa_sw_cld      = self.sigmoid(ssa_sw_cld.squeeze())
            g_sw_cld        = self.sigmoid(g_sw_cld.squeeze())
            tau_sw_scat_cld = ssa_sw_cld * tau_sw_cld

          # ------ COMBINE SHORTWAVE GAS AND CLOUD OPTICAL PROPERTIES -----
          tau_sw_scat_cld = torch.cat((zeroes, tau_sw_scat_cld),dim=0)
          tau_sw_cld      = torch.cat((zeroes, tau_sw_cld),dim=0)
          g_sw_cld        = torch.cat((zeroes, g_sw_cld.squeeze()),dim=0)
          # Compute total optical depth as gas (total) optical depth + cloud (total) optical depth
          tau_sw_tot      = tau_sw + tau_sw_cld
          # Compute total scattering optical depth
          tau_sw_scat_tot = tau_sw_scat + tau_sw_scat_cld # total scattering optical depth 
          tau_sw_scat_tot = torch.clamp(tau_sw_scat_tot, min=1e-9)

          # print("min max tau_Sw_scat_tot", tau_sw_scat_tot.min(), tau_sw_scat_tot.max())
          # Combine cloud and gas asymmetry factors, weighting by scattering optical depth 
          #        gas g*tau_scat         cloud g*tau_scat        total tau_scat
          # g_sw    = (g_sw*tau_sw_scat + g_sw_cld*tau_sw_scat_cld) / (tau_sw_scat_tot) 
          # HOWEVER g_sw from gases is zero, so lose first term
            # g_sw_cld = torch.where(tau_sw_scat_cld == 0, torch.zeros_like(tau_sw_scat_cld), g_tau_scat_sw_cld / tau_sw_scat_cld)

          g_sw = g_sw_cld*tau_sw_scat_cld / tau_sw_scat_tot              
          tau_sw  = tau_sw_tot 
          ssa_sw  = tau_sw_scat_tot / tau_sw_tot
          del tau_sw_scat, tau_sw_scat_tot, tau_sw_scat_cld      

      if not (self.use_existing_gas_optics_sw and self.use_existing_gas_optics_lw): 
        # at least one is False, so SW and/or LW optical properties are predicted with a new MLP (that does gas+cld)
        
        # Humidity, T and p with simple scaling
        qv_new    = qv_new * 1.608079364 # (28.97 / 18.01528) # mol_weight_air / mol_weight_gas
        qv_new    = (torch.sqrt(torch.sqrt(qv_new))) / 0.497653
        temp      = (T_new - 160 ) / (180)
        pressure  = (log_play - 0.00515) / (11.59485)
        if not (self.use_existing_gas_optics_sw or self.use_existing_gas_optics_lw):
        # if neither are True ( both are False), effective radiuses have not been computed yet
          T_new1 = T_new[self.ilev_crm:].view(-1) 
          ice_eff_rad = self.reitab(T_new1).view((self.nlev_crm, batch_size, 1))
          icefrac   =  torch.repeat_interleave(inputs_aux[:,12:13].unsqueeze(0),self.nlev_crm,dim=0)
          landfrac  =  torch.repeat_interleave(inputs_aux[:,13:14].unsqueeze(0),self.nlev_crm,dim=0)
          snowh     =  torch.repeat_interleave(inputs_aux[:,15:16].unsqueeze(0),self.nlev_crm,dim=0)
          liq_eff_rad = self.reltab(T_new1, landfrac.view(-1), icefrac.view(-1), snowh.view(-1)).view((self.nlev_crm, batch_size, 1))

        liq_eff_rad  = liq_eff_rad / 13.5  # simple scaling 
        ice_eff_rad  = ice_eff_rad / 250.0 # simple scaling
        zer_1 = torch.zeros(10, batch_size, 1, device=device)
        ice_eff_rad = torch.cat((zer_1, ice_eff_rad),dim=0)
        liq_eff_rad = torch.cat((zer_1, liq_eff_rad),dim=0)  

        # we use a latent convective memory as input here, but need to concatenate with zeros near the top where moist physics is inactive
        mem                 = torch.zeros(nlev, batch_size, self.nh_mem0, device=device)
        mem[self.ilev_crm:] = rnn_mem[:,:,0:self.nh_mem0] #[:,:,0:1]
        qn_new              = 1 - torch.exp(-qn_new * self.lbd_qn.reshape(-1,1,1))
        # inputs_rad = torch.cat((pressure, temp, qv_new, qn_new, inputs_main[:,:,6:9], inputs_main[:,:,12:15], mem),dim=2)
        inputs_rad = torch.cat((pressure, temp, qv_new, qn_new, inputs_main[:,:,12:15], liq_eff_rad, ice_eff_rad, mem),dim=2)

        # --- SHORTWAVE ---
        if not self.use_existing_gas_optics_sw:
          # sw_optprops = self.mlp_sw_optprops(inputs_rad)
          sw_optprops = self.mlp_sw_optprops1(inputs_rad)
          sw_optprops = self.softsign(sw_optprops)
          sw_optprops = self.mlp_sw_optprops2(sw_optprops)

          # (nlev, nb, ng*ny ) -->  (nlev, nb, ny, ng)
          sw_optprops = torch.reshape(sw_optprops,(self.nlev, batch_size, self.ny_sw_optprops, self.ng_sw))
          tau_sw, ssa_sw, g_sw = sw_optprops.chunk(3,2 )
          g_sw        = self.sigmoid(g_sw.squeeze())
          ssa_sw      = self.sigmoid(ssa_sw.squeeze())

          # print("tau sha", tau_sw.shape, "coldry", col_dry.shape)
          tau_sw      = torch.pow(tau_sw.squeeze(), 8)
          tau_sw      = tau_sw*(1e-23*col_dry)
          tau_sw      = torch.clamp(tau_sw, min=1e-6, max=40.0) 
          del sw_optprops

        # --- LONGWAVE ---
        if not self.use_existing_gas_optics_lw:
          # --------------------------- LONGWAVE ---------------------------
          lw_optprops = self.mlp_lw_optprops(inputs_rad) 
          # (nlev, nb, ng*ny ) -->  (nlev, nb, ny, ng)
          lw_optprops = torch.reshape(lw_optprops,(nlev, batch_size, self.ny_lw_optprops, self.ng_lw))
          tau_lw, pfrac = lw_optprops.chunk(2, 2) # LW optical depth and Planck fraction 
          tau_lw      = torch.pow(tau_lw.squeeze(), 8) # (nlev,batch,ng)
          tau_lw      = tau_lw*(1e-24*col_dry)
          tau_lw      = torch.clamp(tau_lw, min=1e-6, max=400.0)
          pfrac       = pfrac.squeeze() # 
          pfrac       = self.softmax(pfrac)
          del lw_optprops
        
      # print("pfrac1 min max mean", torch.min(pfrac[:,:]).item(), torch.max(pfrac[:,:]).item(), torch.mean(pfrac[:,:]).item())
      # print("tau_lw1 min max mean", torch.min(tau_lw[:,:]).item(), torch.max(tau_lw[:,:]).item(), torch.mean(tau_lw[:,:]).item())
      tlev        = self.interpolate_tlev_batchlast(T_new.squeeze(), play.squeeze(), plev.squeeze()) # (nlev+1, nb)
      # print("tlev min max mean", tlev.min().item(), tlev.max().item(), tlev.mean().item())
      # lwup_sfc    = (inputs_aux[:,11]*self.xdiv_sca[11]) + self.xmean_sca[11]
      lwup_sfc    = inputs_aux[:,11]

      # BELOW IS INCORRECT IF we just used the raw Planck fractions predicted by RRTMGP-NN, without reduction/changing last layer.
      # RRTMGP planck fracs sum to 1 only within bands, not across all g-points. 
      # We should be multiplying with the band-wise Planck emissions, not broadband Planck emission. 
      source_sfc  = pfrac[-1,:,:]*lwup_sfc.unsqueeze(1)
      lwup_lev    = torch.unsqueeze(self.outgoing_lw(tlev),2) # (nlev+1, nb, ng)
      source_lev  = torch.zeros(nlev+1, batch_size, self.ng_lw, device=device)
      source_lev[-1,:,:] = pfrac[-1,:,:] * lwup_lev[-1,:,:]
      source_lev[0:-1,:,:] = pfrac[:,:,:]  * lwup_lev[0:-1,:,:]
      if self.printdebug:
        print("lup sfc min max mean", lwup_sfc.min().item(), lwup_sfc.max().item(), lwup_sfc.mean().item())
        print("source_sfc min max mean", source_sfc.min().item(), source_sfc.max().item(), source_sfc.mean().item())
        print("pfrac min max", torch.min(pfrac[:,:]).item(), torch.max(pfrac[:,:]).item(), "sfc", pfrac[-1,100,:])
        print("lup lev min max mean", lwup_lev.min().item(), lwup_lev.max().item(), lwup_lev.mean().item())
        # print("lwup_lev", lwup_lev[:,0,0]) 
        print("source_lev min max mean", source_lev.min().item(), source_lev.max().item(), source_lev.mean().item())
        # print("source_lev 100", source_lev[100,0,0])
      del pfrac

      return tau_lw, source_lev, source_sfc, tau_sw, ssa_sw, g_sw
    

    def radiative_transfer(self, inputs_aux0, delta_plev, area_frac, rnn_mem, tau_lw, source_lev, source_sfc, tau_sw, ssa_sw, g_sw):
      """
      Two-stream longwave and shortwave radiative transfer with correlated-k (or other spectral discretization)
      Arrays are structured (level, column, spectral) where we can collapse the column (batch) and spectral dims
      """

      nlev, batch_size, ng = tau_lw.shape 
      device = tau_lw.device
      inputs_aux =  (inputs_aux0*self.xdiv_sca) + self.xmean_sca # undo norm

      # -------------------------- LONGWAVE -----------------------------
      # 
      # Computation of layer-wise LW transmittances and source terms 
      planck_top = source_lev[0:-1,:,:]
      planck_bot = source_lev[1:,:,:]
      source_up, source_dn, trans_lw = reftrans_lw(planck_top.view(-1),planck_bot.view(-1), tau_lw.view(-1))
      del tau_lw, planck_top, planck_bot, source_lev

      emissivity_surf = torch.ones_like(source_sfc)
      
      # call LW solver to predict downward and upward spectral fluxes at each half-level. LW scattering is ignored here (as in most ESMs)
      flux_lw_dn, flux_lw_up = lw_solver_noscat_batchlast(trans_lw.view(nlev, -1), source_dn.view(nlev, -1), source_up.view(nlev, -1), 
                                                          source_sfc.view(-1), emissivity_surf.view(-1))

      flux_lw_up = flux_lw_up.view(nlev+1,batch_size, self.ng_lw)
      flux_lw_dn = flux_lw_dn.view(nlev+1,batch_size, self.ng_lw)
      # finally spectral reduction to get broadband flux
      flux_lw_up = torch.sum(flux_lw_up,dim=2) 
      flux_lw_dn = torch.sum(flux_lw_dn,dim=2)
      if self.printdebug:
        print("source_up min max mean", source_up.min().item(), source_up.max().item(), source_up.mean().item())
        print("source_dn min max mean", source_dn.min().item(), source_dn.max().item(), source_dn.mean().item())
        print("trans_lw min max mean", trans_lw.min().item(), trans_lw.max().item(), trans_lw.mean().item())
        print("source_up min max", torch.min(source_up).item(), torch.max(source_up).item())   
        print("emissivity_surf min max mean", emissivity_surf.min().item(), emissivity_surf.max().item(), emissivity_surf.mean().item())
        print("flux lw gpt up  min max mean", flux_lw_up.min().item(), flux_lw_up.max().item(),flux_lw_up.mean().item())
        # print("flux lw up ", flux_lw_up[:,100])
        print("flux lw up  min max mean", flux_lw_up.min().item(), flux_lw_up.max().item(),flux_lw_up.mean().item())
      del trans_lw, source_dn, source_up, source_sfc, emissivity_surf

      # -------------------------- END LONGWAVE -----------------------------

      # -------------------------- SHORTWAVE -----------------------------
      # 
      mu0 = torch.reshape(inputs_aux[:,6:7],(-1,1,1))
      min_mu = 1e-6 #1e-3
      mu0 = torch.clamp(mu0, min=min_mu) 

      # expand mu0 (cosine of solar zenith angle) from (ncol) -> (nlev,ncol,ng_sw)
      mu0_rep = mu0.reshape((1,-1,1)).expand(self.nlev, -1, self.ng_sw)
      if self.experimental_rad:
        mu0_rep = mu0_rep.unsqueeze(3).expand(-1, -1, -1, self.nreg)

      # SW REFLECTANCE-TRANSMITTANCE COMPUTATIONS
      if self.printdebug:
        print("min max mean tau", tau_sw.min().item(), tau_sw.max().item(),  tau_sw.mean().item())
        print("min max ssa_sw", ssa_sw.min().item(), ssa_sw.max().item(), ssa_sw.mean().item())
        print("min max g_sw", g_sw.min().item(), g_sw.max().item(), g_sw.mean().item())

      ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir = calc_ref_trans_sw(mu0_rep, tau_sw, ssa_sw, g_sw)

      ref_diff            = ref_diff.view(nlev, -1)
      trans_diff          = trans_diff.view(nlev, -1)
      ref_dir             = ref_dir.view(nlev, -1)
      trans_dir_diff      = trans_dir_diff.view(nlev, -1)
      trans_dir_dir       = trans_dir_dir.view(nlev, -1)
      del tau_sw, ssa_sw, g_sw, mu0_rep

      incoming_toa    = inputs_aux[:,1:2]

      # if self.printdebug: print("incoming toa", incoming_toa[500:530])

      if (self.use_existing_gas_optics_sw and (not self.reduce_sw_gas_optics)):
        if self.use_new_sw_gas_optics: # new spectrally reduced (ng=12..32) gas optics models
          toa_spectral =  self.gas_optics_model_sw1.get_solar_weights()
          # print("toa spectral shape", toa_spectral.shape, "sum -1", toa_spectral.sum(dim=-1), "raw", toa_spectral)
        else:
          toa_spectral = self.sw_solar_weights 
      else:
        # Here we apply softmax to ensure the solar weights sum to 1 (and are positive)
        toa_spectral = self.softmax_dim1(self.sw_solar_weights)

      incoming_toa = incoming_toa*toa_spectral
      if self.experimental_rad:
         incoming_toa = torch.repeat_interleave(incoming_toa.unsqueeze(2),self.nreg,dim=2)
        #  area_frac_toa = torch.repeat_interleave(area_frac[0].unsqueeze(1),self.ng_sw,dim=1)
         area_frac_toa  = torch.zeros(batch_size, self.nreg, device=device)
         area_frac_toa[:,0] = 1.0
         area_frac_toa = torch.repeat_interleave(area_frac_toa.unsqueeze(1),self.ng_sw,dim=1)
         incoming_toa = incoming_toa*area_frac_toa
      # print("inc toa max sum 1", incoming_toa.sum(dim=-1).max())
      incoming_toa = incoming_toa.view(-1)


      # ------- COMPUTE SURFACE SPECTRAL ALBEDO TO DIRECT AND DIFFUSE RADIATION --
      aldif  = inputs_aux[:, 7].view(1, -1)
      aldir  = inputs_aux[:, 8].view(1, -1)
      asdif  = inputs_aux[:, 9].view(1, -1)
      asdir  = inputs_aux[:, 10].view(1, -1)

      # print("mean aldif", aldif.mean(), "aldir", aldir.mean(), "asdif", asdif.mean(), asdir.mean())
      albedo_surf_dir_sw  = torch.ones(self.ng_sw, batch_size, device=device)
      albedo_surf_diff_sw = torch.ones(self.ng_sw, batch_size, device=device)

      if self.use_new_sw_gas_optics:
        # New 5-band gas optics with exact spectral boundaries:
        #   band 1 (bb[0]:bb[1]): 250–4200   cm-1  → pure NIR  → aldir/aldif
        #   band 2 (bb[1]:bb[2]): 4200–14500 cm-1  → pure NIR  → aldir/aldif
        #   band 3 (bb[2]:bb[3]): 14500–16000 cm-1 → transition (~0.69–0.625 µm) → blend
        #   band 4 (bb[3]:bb[4]): 16000–22000 cm-1 → pure vis   → asdir/asdif
        #   band 5 (bb[4]:bb[5]): 22000–50000 cm-1 → pure UV    → asdir/asdif
        bb = self.gas_optics_model_sw1.band_bounds  # List[int], length 6

        ng_b1  = bb[1] - bb[0]
        ng_b2  = bb[2] - bb[1]
        ng_b3  = bb[3] - bb[2]
        ng_b4  = bb[4] - bb[3]
        ng_b5  = bb[5] - bb[4]

        # Bands 1+2: near-IR
        albedo_surf_dir_sw [bb[0]:bb[2]] = aldir.expand(ng_b1 + ng_b2, -1)
        albedo_surf_diff_sw[bb[0]:bb[2]] = aldif.expand(ng_b1 + ng_b2, -1)
        # Band 3: transition — blend 50/50, matching the 0.7 µm boundary intent
        albedo_surf_dir_sw [bb[2]:bb[3]] = 0.5 * (aldir.expand(ng_b3, -1) + asdir.expand(ng_b3, -1))
        albedo_surf_diff_sw[bb[2]:bb[3]] = 0.5 * (aldif.expand(ng_b3, -1) + asdif.expand(ng_b3, -1))
        # Bands 4+5: visible/UV
        albedo_surf_dir_sw [bb[3]:bb[5]] = asdir.expand(ng_b4 + ng_b5, -1)
        albedo_surf_diff_sw[bb[3]:bb[5]] = asdif.expand(ng_b4 + ng_b5, -1)

        # print("mean albedo_surf_dir_sw", albedo_surf_dir_sw.mean().item(), "albedo_surf_diff_sw ", albedo_surf_diff_sw.mean().item())
      else:
        # ------- COMPUTE SURFACE SPECTRAL ALBEDO TO DIRECT AND DIFFUSE RADIATION --
        # 
        # Here the input surface albedos are given only in 4 bands and don't match our k-distribution, so we need to do a simple mapping
        # If we are using RRTMGP(-NN), the mapping below should match the subroutine set_albedo in E3SM/components/eam/src/physics/rrtmgp/radiation.F90
        # If we are learning a new gas optics NN on the fly, or an extraNN to shrink the spectral dim of RRTMGP-NN, then below its assumed to be a good
        # idea to follow RRTMGP in how much of the spectral space is allocated to near-IR versus visible
        # band g-points:  1,10 | 11,18 | 19,29 | 30,37 | 38,46 | 47,56 | 57,67 | 68,71 | 72,80 | 81,89 | 90, 96 | 97, 102 | 103, 109 | 110, 112 
        # wavenum limits: 820, 2680 | 2680, 3250 | 3250, 4000 | 4000, 4650 | 4650, 5150  | 5150, 6150 | 6150, 7700 | 7700, 8050  | 12850, 16000 | 
        # 16000, 22650 | 22650, 29000 | 29000, 38000 | 38000, 50000 |
        # ! sols(pcols)      Direct solar rad on surface (< 0.7)
        # ! soll(pcols)      Direct solar rad on surface (>= 0.7)
        # Old proportional mapping following RRTMGP band 10 transition
        iend_ir = int(round((80/112)*self.ng_sw)) # RRTMGP bands 1-9 (g-points 1-80) encompass 820-12850 cm-1 (near-ir), see data/rrtmgp-data-sw-g112-210809.nc
        iend_mix= int(round((89/112)*self.ng_sw)) # RRTMGP band 10 is in between UV/visible and near-IR, and bands 11-14 (89-112) are fully in visible range (> 14286 ! cm^-1)
        ng_ir    = iend_ir
        ng_mix   = iend_mix - iend_ir
        ng_vis   = self.ng_sw - iend_mix

        albedo_surf_dir_sw [0:iend_ir]        = aldir.expand(ng_ir,  -1)
        albedo_surf_dir_sw [iend_ir:iend_mix] = 0.5 * (aldir.expand(ng_mix, -1) + asdir.expand(ng_mix, -1))
        albedo_surf_dir_sw [iend_mix:]        = asdir.expand(ng_vis, -1)
        albedo_surf_diff_sw[0:iend_ir]        = aldif.expand(ng_ir,  -1)
        albedo_surf_diff_sw[iend_ir:iend_mix] = 0.5 * (aldif.expand(ng_mix, -1) + asdif.expand(ng_mix, -1))
        albedo_surf_diff_sw[iend_mix:]        = asdif.expand(ng_vis, -1)

      albedo_surf_dir_sw    = torch.transpose(albedo_surf_dir_sw,0,1).contiguous()
      albedo_surf_diff_sw   = torch.transpose(albedo_surf_diff_sw,0,1).contiguous()

      if self.experimental_rad:
        # Repeat to sub-grid regions (TripleClouds scheme)
        albedo_surf_dir_sw   = torch.repeat_interleave(albedo_surf_dir_sw.unsqueeze(2), self.nreg, dim=2)
        albedo_surf_diff_sw  = torch.repeat_interleave(albedo_surf_diff_sw.unsqueeze(2), self.nreg, dim=2)

      albedo_surf_dir_sw    = albedo_surf_dir_sw.view(-1)
      albedo_surf_diff_sw   = albedo_surf_diff_sw.view(-1)

      # print("albedo_surf_dir_sw mean", albedo_surf_dir_sw.mean(), "albedo_surf_diff_sw", albedo_surf_diff_sw.mean())

      # --------- SW RADIATIVE TRANSFER USING ADDING METHOD -----------
      if self.experimental_rad:
        if True:
          # xx = torch.permute(qn_crm,(1,2,0)) # qn_crm(nlev,nb,nreg) --> (nb,nreg,nlev)
          xx = torch.permute(rnn_mem,(1,2,0)) # qn_crm(nlev,nb,nreg) --> (nb,nreg,nlev)
    
          # v_mat = self.softmax_dim1(self.conv_vmat(xx)) # (nb, nreg*nreg, nlev)
          # v_mat = torch.permute(v_mat,(2,0,1)).contiguous()
          # v_mat = v_mat.view(self.nlev_crm-1, batch_size, self.nreg, self.nreg)

          frac_th = 1e-14
          conv_vmat = self.conv_vmat(xx) # (nb, nreg*nreg, nlev)
          conv_vmat = torch.permute(conv_vmat,(2,0,1)).contiguous()
          conv_vmat = conv_vmat.view(self.nlev_crm-1, batch_size, self.nreg, self.nreg)
          # Predict "conditional" overlap matrices (will sum to 1 across nreg*nreg after multiplying with area_frac)
          conv_vmat = torch.softmax(conv_vmat,dim=-1)
          empty = area_frac[0:-1].unsqueeze(-1) < frac_th  # (nlev, nbatch, nreg, 1)
          v_mat = conv_vmat.masked_fill(empty, 0.0)      # (nlev, nbatch, jupper, jlower)
          v_mat = v_mat.transpose(-1, -2).contiguous()
          v_mat = torch.repeat_interleave(v_mat.unsqueeze(2), self.ng_sw, dim=2)

          ones  = torch.zeros(1, batch_size, self.ng_sw,self.nreg,self.nreg, device=device)
          oness  = torch.zeros(11, batch_size, self.ng_sw,self.nreg,self.nreg, device=device)
          ones[:,:,:,0,0] = 1.0; oness[:,:,:,0,0] = 1.0
          ones = ones.view(1, batch_size, self.ng_sw*self.nreg*self.nreg)
          oness = oness.view(11, batch_size, self.ng_sw*self.nreg*self.nreg)
          v_mat = torch.cat((oness, v_mat.view(self.nlev_crm-1, batch_size, self.ng_sw*self.nreg*self.nreg), ones), dim=0)

          del ones, oness
        else:
          overlap_param = self.sigmoid(self.mlp_overlap(rnn_mem[0:-1])).squeeze()
          region_fracs = area_frac.transpose(1,2).contiguous()
          # print("shape regfrac", region_fracs.shape, "op", overlap_param.shape)
          v_mat = calc_overlap_matrices(region_fracs, overlap_param)
          # print("shape v mat", v_mat.shape) # (3,3,nlev,nb) -> (nlev,nb,3,3)
          v_mat = torch.permute(v_mat,(2,3,0,1)).contiguous()
          oness  = torch.zeros(10, batch_size, self.nreg, self.nreg, device=device)
          oness[:,:,0,0] = 1.0
          v_mat = torch.cat((oness, v_mat), dim=0)
          v_mat = v_mat.unsqueeze(2).expand(self.nlev+1, batch_size, self.ng_sw, 3, 3).contiguous()
          # print("shape v mat 2", v_mat.shape) 
                
        flux_sw_up, flux_sw_dn_diffuse, flux_sw_dn_direct = adding_tc_sw_batchlast_opt(incoming_toa, 
                    albedo_surf_diff_sw, albedo_surf_dir_sw, ref_diff, 
                    trans_diff, ref_dir, trans_dir_diff, trans_dir_dir, v_mat, self.nreg)
      else:
        flux_sw_up, flux_sw_dn_diffuse, flux_sw_dn_direct = adding_ica_sw(
                    incoming_toa, albedo_surf_diff_sw, albedo_surf_dir_sw, 
                    ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir)
            
      # print("Min max inc",incoming_toa.min().item(), incoming_toa.max().item())
      # print("Min max albedo_surf_diff_sw",albedo_surf_diff_sw.min().item(), albedo_surf_diff_sw.max().item())
      # print("Min max albedo_surf_dir_sw",albedo_surf_dir_sw.min().item(), albedo_surf_dir_sw.max().item())
      # print("Min max ref_diff",ref_diff.min().item(), ref_diff.max().item())
      # print("Min max flux_sw_up",flux_sw_up.min().item(), flux_sw_up.max().item())
      # print("Min max flux_sw_dn_diffuse",flux_sw_dn_diffuse.min().item(), flux_sw_dn_diffuse.max().item())
      # print("Min max albedo_surf_dir_sw",albedo_surf_dir_sw.min().item(), albedo_surf_dir_sw.max().item())

      del ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir

      if self.experimental_rad:
        # sum over region dimension
        flux_sw_up          = torch.sum(flux_sw_up.view(nlev+1, batch_size, self.ng_sw, self.nreg),dim=-1)
        flux_sw_dn_diffuse  = torch.sum(flux_sw_dn_diffuse.view(nlev+1, batch_size, self.ng_sw, self.nreg),dim=-1)
        flux_sw_dn_direct   = torch.sum(flux_sw_dn_direct.view(nlev+1, batch_size, self.ng_sw, self.nreg),dim=-1)
      else:
        flux_sw_up = torch.reshape(flux_sw_up, (nlev+1, batch_size, self.ng_sw))
        flux_sw_dn_diffuse = torch.reshape(flux_sw_dn_diffuse, (nlev+1, batch_size, self.ng_sw))
        flux_sw_dn_direct = torch.reshape(flux_sw_dn_direct, (nlev+1, batch_size, self.ng_sw))
                
      # Sum over whole / parts of spectral dimension to get broadband / band-wise fluxes
      if self.use_new_sw_gas_optics:
        bb = self.gas_optics_model_sw1.band_bounds  # List[int], length 6

        # Band 3 is the transition band (~0.69–0.625 µm), split 50/50 into NIR and visible
        sw_dir_dn_mixband  = torch.sum(flux_sw_dn_direct [-1, :, bb[2]:bb[3]], dim=-1, keepdim=True)
        sw_diff_dn_mixband = torch.sum(flux_sw_dn_diffuse[-1, :, bb[2]:bb[3]], dim=-1, keepdim=True)

        # SOLL/SOLLD: near-IR (bands 1+2 + half of band 3)
        SOLL  = torch.sum(flux_sw_dn_direct [-1, :, bb[0]:bb[2]], dim=-1, keepdim=True) + 0.5 * sw_dir_dn_mixband
        SOLLD = torch.sum(flux_sw_dn_diffuse[-1, :, bb[0]:bb[2]], dim=-1, keepdim=True) + 0.5 * sw_diff_dn_mixband

        # SOLS/SOLSD: visible/UV (bands 4+5 + half of band 3)
        SOLS  = torch.sum(flux_sw_dn_direct [-1, :, bb[3]:bb[5]], dim=-1, keepdim=True) + 0.5 * sw_dir_dn_mixband
        SOLSD = torch.sum(flux_sw_dn_diffuse[-1, :, bb[3]:bb[5]], dim=-1, keepdim=True) + 0.5 * sw_diff_dn_mixband

      else:
        sw_dir_dn_mixband  = torch.sum(flux_sw_dn_direct [-1, :, iend_ir:iend_mix], dim=-1, keepdim=True)
        SOLL  = torch.sum(flux_sw_dn_direct [-1, :, 0:iend_ir],   dim=-1, keepdim=True) + 0.5 * sw_dir_dn_mixband
        SOLS  = torch.sum(flux_sw_dn_direct [-1, :, iend_mix:],   dim=-1, keepdim=True) + 0.5 * sw_dir_dn_mixband
        sw_diff_dn_mixband = torch.sum(flux_sw_dn_diffuse[-1, :, iend_ir:iend_mix], dim=-1, keepdim=True)
        SOLLD = torch.sum(flux_sw_dn_diffuse[-1, :, 0:iend_ir],   dim=-1, keepdim=True) + 0.5 * sw_diff_dn_mixband
        SOLSD = torch.sum(flux_sw_dn_diffuse[-1, :, iend_mix:],   dim=-1, keepdim=True) + 0.5 * sw_diff_dn_mixband
        
      flux_sw_up          = torch.sum(flux_sw_up,dim=2)
      flux_sw_dn_diffuse  = torch.sum(flux_sw_dn_diffuse,dim=2)
      flux_sw_dn_direct   = torch.sum(flux_sw_dn_direct,dim=2)
      
      flux_sw_dn          = flux_sw_dn_diffuse + flux_sw_dn_direct
      flux_sw_dn_sfc      = flux_sw_dn[-1,:].unsqueeze(1)             # NETSW  

      flux_sw_net = flux_sw_dn - flux_sw_up
      # We did SW computations in all columns, so now we need to zero out nighttime columns
      inds_zero = inputs_aux[:,6] < min_mu
      flux_sw_net[:,inds_zero] = 0.0
      flux_sw_dn_sfc[inds_zero] = 0.0
      SOLL[inds_zero] = 0.0
      SOLS[inds_zero] = 0.0
      SOLLD[inds_zero] = 0.0
      SOLSD[inds_zero] = 0.0

      # print("SOLL", SOLL.mean().item(), "SOLLD", SOLLD.mean().item(), "SOLS", SOLS.mean().item(), "SOLSD", SOLSD.mean().item())
      # SOLL[:] = 0.0 
      # SOLLD[:] = 0.0 
      # SOLS[:] = 0.0
      # SOLSD[:] = 0.0
      

      if self.printdebug:
        print("flux sw dn ", flux_sw_dn.mean(dim=1))
        print("flux sw dndir ", flux_sw_dn_direct.mean(dim=1))
        print("flux sw up ", flux_sw_up.mean(dim=1))
        # print("mu0", inputs_aux[500,6:7])

      flux_lw_dn_sfc  = flux_lw_dn[-1,:].unsqueeze(1)             # FLWDS 
      flux_lw_net     = flux_lw_dn - flux_lw_up
      flux_net        = flux_lw_net + flux_sw_net
      flux_diff       = flux_net[1:,:] - flux_net[0:-1,:]
      dT_rad          = -(flux_diff / delta_plev.squeeze()) * 0.009761357302 # * g/cp = 9.80665 / 1004.64
      dT_rad          = dT_rad * self.yscale_lev[:,0:1] # scaling (physical computations gave us unscaled tendency, but target outputs are scaled)
      
      out_sfc_rad = torch.cat((flux_sw_dn_sfc, flux_lw_dn_sfc,  SOLS, SOLL, SOLSD, SOLLD ), dim=1)
      out_sfc_rad = out_sfc_rad * self.yscale_sca_rad

      return dT_rad, out_sfc_rad

    def forward(self, inp_list : List[Tensor]):
        inputs_main     = inp_list[0]
        inputs_aux      = inp_list[1]
        rnn_mem         = inp_list[2]
        inputs_denorm   = inp_list[3]
        if self.training:
          out_new_true = inp_list[4]
        else:
          out_new_true = rnn_mem # dummy

        # print("shape inp main", inputs_main.shape, "aux", inputs_aux.shape, "mem", rnn_mem.shape, "denorm", inputs_denorm.shape)
        # if self.store_precip: 
        P_old = rnn_mem[-1,:,-1] # if self.store_precip is False, this is a dummy variable not used for anything 

        batch_size  = inputs_main.shape[0]
        nlev        = inputs_main.shape[1]
        device      = inputs_main.device

        # t0 = time.time()
        inputs_main = torch.transpose(inputs_main,0,1).contiguous()
        inputs_denorm = torch.transpose(inputs_denorm,0,1).contiguous()

        if self.add_pres:
            sp = torch.unsqueeze(inputs_aux[:,0:1],0)
            # undo scaling
            sp = sp*self.xdiv_sca[0:1] + self.xmean_sca[0:1]
            pres  = self.preslay(sp)
            inputs_main = torch.cat((inputs_main,pres),dim=2)
            play = self.preslay_nonorm(sp)
            delta_plev = self.presdelta(sp)
            plev = self.preslev_nonorm(sp)

        if self.use_physrad or self.separate_radiation:
          # Do not use inputs -2,-3,-4 (O3, CH4, N2O) or first 10 levels
          inputs_main_crm = torch.cat((inputs_main[self.ilev_crm:,:,0:-4], inputs_main[self.ilev_crm:,:,-1:]),dim=2)
        else:
          inputs_main_crm = inputs_main 

        if self.use_initial_mlp:
            inputs_main_crm = self.mlp_initial(inputs_main_crm)
            inputs_main_crm = self.tanh(inputs_main_crm)  
            
        # The input (a vertical sequence) is concatenated with the (latent) convective memory
        if self.store_precip:
          rnn_mem = rnn_mem[:,:,0:self.nh_mem0] # remove stored precip from array

        if self.use_physrad or self.separate_radiation:
          rnn_mem0 = rnn_mem 
        else:
          zer = torch.zeros(self.ilev_crm, batch_size, self.nh_mem0, device=device)
          rnn_mem0 = torch.cat((zer, rnn_mem), dim=0)

        inputs_main_crm = torch.cat((inputs_main_crm,rnn_mem0), dim=2)

        # TOA is first in memory, so to start at the surface we need to go backwards
        rnn1_input = torch.flip(inputs_main_crm, [0])
        
        if self.use_physrad or self.separate_radiation:
          inputs_sfc = torch.cat((inputs_aux[:,0:6],inputs_aux[:,11:]),dim=1)
        else:
          inputs_sfc = inputs_aux

        hx = self.mlp_surface1(inputs_sfc)
        hx = self.tanh(hx)
        if self.use_lstm: 
            cx = self.mlp_surface2(inputs_sfc)
            # cx = self.tanh(cx)
            hidden = (torch.unsqueeze(hx,0), torch.unsqueeze(cx,0))  
        else:
            hidden = torch.unsqueeze(hx,0)
        rnn1out, states = self.rnn1(rnn1_input, hidden)

        rnn1out = torch.flip(rnn1out, [0])

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
          input_srnn = rnn2out
          # input_srnn = torch.flip(rnn2out, [0]) 
          # input_srnn = torch.cat((input_srnn, rnn1_input),dim=2)

          hx = torch.randn((batch_size, self.nh_rnn2),device=inputs_main.device) 
          if self.use_lstm:
            cx = torch.randn((batch_size, self.nh_rnn2),device=inputs_main.device) 
            hx = (hx, cx)
            srnn_out, last_state = self.rnn3(input_srnn, hx)
          else:
            srnn_out = self.rnn3(input_srnn, hx)

          # srnn_out = torch.flip(srnn_out, [0]) 

          h_sfc_perturb = srnn_out[-1,:,:]

          h_final_perturb = srnn_out

          # h_final = h_final + 0.01*h_final_perturb
          # h_sfc   = h_sfc + 0.01*h_sfc_perturb
          rnn2out = rnn2out*h_final_perturb
          # h_sfc   = h_sfc*h_sfc_perturb
          last_h   = h_sfc_perturb
  
        if self.use_intermediate_mlp: 
            rnn_mem = self.mlp_latent(rnn2out)
        else:
            rnn_mem = rnn2out 

        if not self.use_physrad: # Need to predict surface radiation variables with an MLP
            if self.separate_radiation:
              hidden = self.mlp_surface_init_rad(inputs_aux[:,6:11])
              hidden = (torch.unsqueeze(hidden,0))

              inputs_gas_rad =  inputs_main[:,:,12:15] # gases
              zer = torch.zeros(self.ilev_crm,batch_size,  self.nh_mem0, device=device)
              rnn_mem0 = torch.cat((zer, rnn_mem), dim=0)
              inputs_rad = torch.cat((inputs_gas_rad, rnn_mem0),dim=2)
              rnn_out, states = self.rnn1_rad(inputs_rad, hidden)
              rnn_out = torch.flip(rnn_out, [0])
              del inputs_rad, states

              inputs_toa = torch.cat((inputs_aux[:,1:2], inputs_aux[:,6:7]),dim=1) 
              hidden = self.mlp_toa_rad(inputs_toa)
              hidden = (torch.unsqueeze(hidden,0))

              last_rnn, last_h_rad = self.rnn2_rad(rnn_out, hidden)                   
            else:
              last_rnn = rnn2out 
              last_h_rad = last_h
              rnn_mem = rnn_mem[self.ilev_crm:]
              rnn2out = rnn2out[self.ilev_crm:]

            out_sfc_rad = self.mlp_surface_output_rad(last_h_rad.squeeze())
            out_sfc_rad = self.relu(out_sfc_rad)
            dT_rad = self.mlp_output_rad(last_rnn)

        out = self.mlp_output(rnn_mem)

        # print("elapsed RNNs {}".format(time.time() - t0))

        # ------ MOIST PHYSICS MODULE ------
        #
        # t0 = time.time()
        # if torch.is_autocast_enabled() and self.training:
        #   # print("autocast ON!!")
        #   with torch.autocast(device_type=rnn2out.device.type, enabled=False):
        #     rnn2out = rnn2out.float()
        #     last_h = last_h.float()
        #     # rnn_mem = rnn_mem.float()
        #     out_new, precc, precsc, rnn_mem, T_crm, qv_crm, qn_crm, area_frac, prec_negative = self.microphysics_decode(inputs_denorm, 
        #                                         delta_plev, play, P_old, out, rnn_mem, rnn2out, last_h)
        # else:
        out_new, precc, precsc, rnn_mem, liq_frac_crm, qv_crm, qn_crm, area_frac, prec_negative = self.microphysics_decode(inputs_denorm, 
                                              delta_plev, play, P_old, out, rnn_mem, rnn2out, last_h)
        # area_frac = area_frac.detach()
        # print("elapsed microphysics_decode {}".format(time.time() - t0))

        # ------ RADIATION MODULE ------
        if self.use_physrad:
          
          T     = inputs_denorm[:,:,0:1]
          qliq  = inputs_denorm[:,:,2:3]
          qice  = inputs_denorm[:,:,3:4] 
          qn    = qliq+qice
          qv    = inputs_denorm[:,:,-1:]

          if self.update_states_for_rad:

            if self.training: 
              out_denorm    = torch.transpose(out_new_true,0,1) / self.yscale_lev.unsqueeze(1)
            else:
              out_denorm    = out_new / self.yscale_lev.unsqueeze(1)
            dT = 1200*out_denorm[:,:,0:1]
            dqv = 1200*out_denorm[:,:,1:2]
            if self.use_ensemble:
                nens = int(qv.shape[0] // dqv.shape[0])
                # dT = torch.repeat_interleave(dT,repeats=nens,dim=1)
                dqv = torch.repeat_interleave(dqv,repeats=nens,dim=1)
            T   = self.relu(T + dT)
            qv  = self.relu(qv + dqv)
            if self.use_mp_constraint:
              dqn = 1200*out_denorm[:,:,2:3]
              qn = self.relu(qn + dqn)

          # t0 = time.time()
          tau_lw, source_lev, source_sfc, tau_sw, ssa_sw, g_sw = self.rad_optical_props(inputs_main, inputs_aux, inputs_denorm, play, plev, delta_plev, 
                                      rnn_mem, liq_frac_crm, qv_crm, qn_crm, T, qv, qn, area_frac, rnn2out)
          # print("elapsed rad_optical_props {}".format(time.time() - t0))

          # t0 = time.time()
          dT_rad, out_sfc_rad = self.radiative_transfer(inputs_aux, delta_plev, area_frac, rnn_mem, tau_lw, source_lev, source_sfc, 
                                      tau_sw, ssa_sw, g_sw)
          dT_rad = dT_rad.unsqueeze(2)
          # print("elapsed radiative_transfer {}".format(time.time() - t0))

        # print("dT_rad 2  min max", torch.min(dT_rad[:,:]).item(), torch.max(dT_rad[:,:]).item())
        # print("shape out", out_new.shape, "dt", dT_rad.shape)
        out_new[:,:,0:1] = out_new[:,:,0:1] + dT_rad

        # # rad predicts everything except PRECSC, PRECC
        out_sfc =  torch.cat((out_sfc_rad[:,0:2], precsc, precc, out_sfc_rad[:,2:]),dim=1)

        if self.predict_liq_frac:
          # T = T + out_new[:,:,0:1] / self.yscale_lev[0:1]
          liq_frac_diagnosed0    = self.temperature_scaling(T.squeeze())
          liq_frac = liq_frac_diagnosed0

          # liq_frac_diagnosed = liq_frac_diagnosed0[:,self.ilev_crm:]
          # temp = T[:,self.ilev_crm:].squeeze()
          # inds = (temp > 250.0) & (temp < 275.0)
          # x_predfrac = rnn_mem[inds]
          # # liq_frac_pred = self.relu(self.mlp_predfrac(x_predfrac))
          # liq_frac_pred = 0.1*self.mlp_predfrac(x_predfrac)
          # liq_frac_pred = torch.reshape(liq_frac_pred,(-1,))
          # liq_frac_pred = liq_frac_pred.to(liq_frac_diagnosed.dtype)
          # liq_frac_diagnosed[inds] = liq_frac_pred
          # # liq_frac_diagnosed[inds] = self.relu(liq_frac_diagnosed[inds] + liq_frac_pred)
          # liq_frac = torch.cat((liq_frac_diagnosed0[:,0:self.ilev_crm],liq_frac_diagnosed),dim=1)
    
          # max_frac = torch.clamp(liq_frac_diagnosed0 + 0.2, max=1.0)
          # min_frac = torch.clamp(liq_frac_diagnosed0 - 0.2, min=0.0)
          # liq_frac = torch.clamp(liq_frac, min=min_frac, max=max_frac)
    
          out_new[:,:,3]     = liq_frac

        out_new = torch.transpose(out_new,0,1).contiguous()
  
        if self.return_neg_precip:
          return out_new, out_sfc, rnn_mem, prec_negative
        else:
          return out_new, out_sfc, rnn_mem
