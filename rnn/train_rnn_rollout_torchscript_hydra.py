#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch code for training ClimSim emulators, utilizing Hydra for easy configuration and WandB for tracking results.
Messy but flexible (LOTS of options!) 
Here we train autoregressively on a continuous time series, which enables learning a latent convective memory

Other stuff
- Several options for 
    -- models (RNN and SSM-based, found in models.py)
    -- loss functions (hybrid loss, optionally adding terms for e.g. energy conservation), see metrics.py
    -- optimizers (I recommend SOAP)
    -- schedulers
- Fast data loader based on chunking and prefetching data, with on-the-fly preprocessing (generator_xy in utils.py )
- Convective memory using previous tendencies: option to mix with predictions (cfg.train_replay, cfg.val_replay, cfg.gradual_mixing_end_epoch)
- Save models when new minimum in validation loss found, also same some validation plots while at it
    -- models are saved as both H5 and JIT-scripted model. These include normalization coefficients, but
       a wrapper is still needed for implementation in E3SM (see notebook save_wrapper_mem_prevtend_ftorch_test)
       should perhaps move everything to the main model classes so a wrapper is not needed

Actual per-epoch training code is in train_or_eval_one_epoch in utils.py

@author: Peter Ukkonen

"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import inspect
import gc
import time 
import psutil
from omegaconf import OmegaConf
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
# from numba import config, njit, threading_layer, set_num_threads
# set_num_threads(1)
# config.THREADING_LAYER = 'threadsafe'
from climsim_utils.data_utils import data_utils
import numpy as np
import xarray as xr
import h5py
import torch
import torch.nn as nn
cuda = torch.cuda.is_available() 
device = torch.device("cuda" if cuda else "cpu")
print(device)

from torch.utils.data import DataLoader
from torchinfo import summary
from models import  *
from models_experimental import *
from utils import train_or_eval_one_epoch, generator_xy, BatchSampler, plot_bias_diff
import metrics as metrics
from torchmetrics.regression import R2Score
import wandb
from omegaconf import DictConfig
import hydra
import matplotlib.pyplot as plt
from random import randrange

@hydra.main(version_base="1.2", config_path="conf", config_name="autoreg_LSTM")
def main(cfg: DictConfig):
        
    grid_path = '../grid_info/ClimSim_low-res_grid-info.nc'
    norm_path = '../preprocessing/normalizations/'
    tr_data_path = cfg.tr_data_dir + cfg.tr_data_fname
    val_data_path = cfg.val_data_dir + cfg.val_data_fname

    #torch.set_float32_matmul_precision("medium")
    #torch.backends.cuda.matmul.allow_tf32 = True    
    # torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    print("Allow TF32:", torch.backends.cuda.matmul.allow_tf32)
    
    print('RAM memory % used:', psutil.virtual_memory()[2], flush=True)
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush=True)
    
    # torch.autograd.set_detect_anomaly(True)
    # print("backends:", torch._dynamo.list_backends())
    # cfg = OmegaConf.load("conf/autoreg_LSTM.yaml")

    # torch.cuda.memory._record_memory_history(enabled='all')
    
    # SELECT OUTPUTS / MICROPHYSICS CONSTRAINT
    # mp_mode = 0   # regular 6 outputs
    # mp_mode = 1   # 5 outputs, predict qn, liq_ratio DIAGNOSED from temperature (mp_constraint) (Hu et al.)
    # mp_mode = -1  # 6 outputs, predict qn + liq_ratio PREDICTED
    # physical_precip: attempt to incorporate some physics in the way precip is predicted (see models.py)
    if cfg.physical_precip and cfg.mp_mode==0:
      raise NotImplementedError("Physical_precip=true as it not compatible with mp_mode=0 as it requires qn")

    if cfg.memory=="None":
        cfg.autoregressive=False
        cfg.use_intermediate_mlp = False
    else:
        cfg.shuffle_data = False 
        
    if cfg.mp_autocast:
        print(torch.cuda.get_device_name(0))
        dtype=torch.float16
        cfg.use_scaler = True 
    else:
        dtype=torch.float32
        cfg.use_scaler = False
        
    # --------------------------------------
    
    grid_info = xr.open_dataset(grid_path)
    level = grid_info.lev.values
    input_mean = xr.open_dataset(norm_path + 'inputs/input_mean_v4_pervar.nc').astype(np.float32)
    input_max = xr.open_dataset(norm_path + 'inputs/input_max_v4_pervar.nc').astype(np.float32)
    input_min = xr.open_dataset(norm_path + 'inputs/input_min_v4_pervar.nc').astype(np.float32)
    
    if cfg.v4_to_v5_inputs:
        input_mean = xr.open_dataset(norm_path + 'inputs/input_mean_v5_pervar.nc').astype(np.float32)
        input_max = xr.open_dataset(norm_path + 'inputs/input_max_v5_pervar.nc').astype(np.float32)
        input_min = xr.open_dataset(norm_path + 'inputs/input_min_v5_pervar.nc').astype(np.float32)
    
    output_scale = xr.open_dataset(norm_path + 'outputs//output_scale_std_lowerthred_v5.nc').astype(np.float32)
    # output_scale = xr.open_dataset(norm_path + 'outputs/output_scale_std_nopenalty.nc').astype(np.float32)
    
    ml_backend = 'pytorch'
    input_abbrev = 'mlexpand'
    output_abbrev = 'mlo'
    data = data_utils(grid_info = grid_info, 
                      input_mean = input_mean, 
                      input_max = input_max, 
                      input_min = input_min, 
                      output_scale = output_scale,
                      ml_backend = ml_backend,
                      normalize = True,
                      input_abbrev = input_abbrev,
                      output_abbrev = output_abbrev,
                      save_h5=True,
                      save_npy=False,
                      )
    
    print("data utils initialized", flush=True)
    data.set_to_v4_rnn_vars()
    
    hyam = torch.from_numpy(data.grid_info['hyam'].values).to(device, torch.float32)
    hybm = torch.from_numpy(data.grid_info['hybm'].values).to(device, torch.float32)
    hyai = torch.from_numpy(data.grid_info['hyai'].values).to(device, torch.float32)
    hybi = torch.from_numpy(data.grid_info['hybi'].values).to(device, torch.float32)
    sp_max = torch.from_numpy(data.input_max['state_ps'].values).to(device, torch.float32)
    sp_min = torch.from_numpy(data.input_min['state_ps'].values).to(device, torch.float32)
    sp_mean = torch.from_numpy(data.input_mean['state_ps'].values).to(device, torch.float32)
    
    hf = h5py.File(tr_data_path, 'r')
    print(hf.keys()) # <KeysViewHDF5 ['input_lev', 'input_sca', 'output_lev', 'output_sca']>
    print(hf['input_lev'].attrs.keys())
    dims = hf['input_lev'].shape
    if len(dims)==4:
        ns, nloc, nlev, nx = dims 
    else:
        ns, nlev, nx = dims

    # ------------------------------------------------------------------------------------------------
    # -------------------------------  INPUTS/OUTPUTS AND NORMALIZATON -------------------------------
    
    vars_2D_inp = hf['input_lev'].attrs.get('varnames').tolist()
    # ['state_t', 'state_rh', 'state_q0002', 'state_q0003', 'state_u',
    #  'state_v', 'state_t_dyn', 'state_q0_dyn', 'state_u_dyn', 'tm_state_t_dyn',
    #  'tm_state_q0_dyn', 'tm_state_u_dyn', 'pbuf_ozone', 'pbuf_CH4', 'pbuf_N2O']
    
    dims = hf['input_sca'].shape; nx_sfc = dims[-1]
    vars_1D_inp = hf['input_sca'].attrs.get('varnames').tolist()
    #1D (scalar) Input variables:
    #   'ps' 'pbuf_SOLIN' 'pbuf_LHFLX' 'pbuf_SHFLX' 'pbuf_TAUX' 'pbuf_TAUY'
    #  'pbuf_COSZRS' 'cam_in_ALDIF' 'cam_in_ALDIR' 'cam_in_ASDIF' 'cam_in_ASDIR'
    #  'cam_in_LWUP' 'cam_in_ICEFRAC' 'cam_in_LANDFRAC' 'cam_in_OCNFRAC'
    #  'cam_in_SNOWHICE' 'cam_in_SNOWHLAND'  (removed:)'tm_state_ps'  'tm_pbuf_SOLIN'
    #  'tm_pbuf_LHFLX' 'tm_pbuf_SHFLX' 'tm_pbuf_COSZRS'(<-removed)  'clat' 'slat']
    
    dims = hf['output_lev'].shape; ny = dims[-1]
    vars_2D_outp = hf['output_lev'].attrs.get('varnames').tolist()
    #     ['ptend_t', 'ptend_q0001', 'ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v']
    # Temperature tendency, q-wv tendency, cloud liquid tendency, cloud ice tendency, wind tendencies
    # Internally, the CRM has only 50 levels, but radiation is computed using the full 60 levels
    # MP constraint means --> ['ptend_t', 'ptend_q0001', 'ptend_qn', 'ptend_u', 'ptend_v']
    print(vars_2D_outp)
    
    dims = hf['output_sca'].shape; ny_sfc = dims[-1]
    vars_1D_outp = hf['output_sca'].attrs.get('varnames').tolist()
    #1D (scalar) Output variables: ['cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 
    #'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']
    
    hf.close()
    
    if cfg.v4_to_v5_inputs:
        vars_2D_inp.remove('state_q0002') # liq
        vars_2D_inp.remove('state_q0003')
        vars_2D_inp.insert(2,"state_qn")
        vars_2D_inp.insert(3,"liq_partition")
    
    if abs(cfg.mp_mode)>0:
        vars_2D_outp.remove('ptend_q0002')
        vars_2D_outp.remove('ptend_q0003')
        vars_2D_outp.insert(2,"ptend_qn")
        # if not cfg.output_norm_per_level:
        #     raise NotImplementedError("output_norm_per_level=false not compatible with mp_mode !=0")

    if cfg.include_prev_outputs:
        # if not cfg.output_norm_per_level:
        #     raise NotImplementedError("Only level-specific norm coefficients saved for previous tendency outputs")

        vars_2D_inp.append('state_t_prvphy')
        vars_2D_inp.append('state_q0001_prvphy')
        vars_2D_inp.append('state_q0002_prvphy')
        vars_2D_inp.append('state_q0003_prvphy')
        vars_2D_inp.append('state_u_prvphy')
        nx = nx + 5
    
    if cfg.include_prev_inputs:
        nx = nx + 6 

    if cfg.include_prev_inputs or cfg.include_prev_outputs:
        skip_first_index=True 
    else:
        skip_first_index=False

    if cfg.add_refpres:
        nx = nx + 1

    # if cfg.use_surface_memory:
    #     nx_sfc = nx_sfc + 2
        
    # if use_mp_constraint:
    if cfg.mp_mode>0:
        ny_pp = ny # The 6 original outputs will be after postprocessing
        ny = ny - 1 # The model itself only has 5 outputs (total cloud water)
    else:
        ny_pp = ny 
        
    if cfg.mp_mode<0:
        predict_liq_ratio=True
        print("mp mode was <0, we are PREDICTING liquid fraction")
    else:
        predict_liq_ratio=False

    print("ns", ns, "nloc", nloc, "nlev", nlev,  "nx", nx, "nx_sfc", nx_sfc, "ny", ny, "ny_sfc", ny, flush=True)

    yscale_sca = output_scale[vars_1D_outp].to_dataarray(dim='features', name='outputs_sca').transpose().values

    if cfg.output_norm_per_level:
        yscale_lev = output_scale[vars_2D_outp].to_dataarray(dim='features', name='outputs_lev').transpose().values

        if cfg.mp_mode<0:
            ones = np.ones((nlev),dtype=np.float32).reshape(nlev,1)
            yscale_lev = np.concatenate((yscale_lev[:,0:3], ones, yscale_lev[:,3:]), axis=1)
            print("Padded y norm coefficients with ones, new shape: {}".format(yscale_lev.shape))

    else:
        if not cfg.new_nolev_scaling:
            if cfg.mp_mode==1:
                yscale_lev = np.repeat(np.array([1.87819239e+04, 3.25021485e+07, 1.58085550e+08, 5.00182069e+04,
                        6.21923225e+04], dtype=np.float32).reshape((1,-1)),nlev,axis=0)
            elif cfg.mp_mode==0:
                yscale_lev = np.repeat(np.array([1.87819239e+04, 3.25021485e+07, 1.91623978e+08, 3.23919949e+08, 
                    5.00182069e+04, 6.21923225e+04], dtype=np.float32).reshape((1,-1)),nlev,axis=0)
            else:
                raise NotImplementedError()
        else:
            if cfg.mp_mode==0:
                 yscale_lev = np.repeat(np.array([24.4, 22.8, 25.1, 30.1, 10.1, 11.3], dtype=np.float32).reshape((1,-1)),nlev,axis=0)
                # yscale_lev = np.repeat(np.array([1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float32).reshape((1,-1)),nlev,axis=0)
            else:
                raise NotImplementedError()

            if cfg.input_norm_per_level:
                raise NotImplementedError()
            
    if cfg.input_norm_per_level:
        xmax_lev = input_max[vars_2D_inp].to_dataarray(dim='features', name='inputs_lev').transpose().values
        xmin_lev = input_min[vars_2D_inp].to_dataarray(dim='features', name='inputs_lev').transpose().values
        xmean_lev = input_mean[vars_2D_inp].to_dataarray(dim='features', name='inputs_lev').transpose().values
    else:
        # xmax_lev = np.repeat(np.array([2.9577750e+02, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
        #        7.6608604e+01, 5.4018818e+01, 2.3003130e-03, 3.2301173e-07,
        #        4.1381191e-03, 2.3003130e-03, 3.2301173e-07, 4.1381191e-03,
        #        2.7553122e-06, 8.6128614e-07, 4.0667697e-07], dtype=np.float32).reshape((1,-1)),nlev,axis=0)
        # xmin_lev = np.repeat(np.array([ 1.9465800e+02,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
        #        -4.5055122e+01, -5.4770996e+01, -2.7256422e-03, -2.8697787e-07,
        #        -3.2421835e-03, -2.7256422e-03, -2.8697787e-07, -3.2421835e-03,
        #         1.0383576e-06,  6.6867170e-07,  3.3562134e-07], dtype=np.float32).reshape((1,-1)),nlev,axis=0)
        # xmean_lev = np.repeat(np.array([ 2.47143495e+02,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #         9.02305768e+00, -1.85453092e-02,  0.00000000e+00,  0.00000000e+00,
        #         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #         1.95962779e-06,  8.22708859e-07,  3.88440249e-07], dtype=np.float32).reshape((1,-1)),nlev,axis=0)
    
        # if cfg.new_nolev_scaling:
        xmax_lev = np.repeat(np.array([3.21864136e+02, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
                2.13669708e+02, 1.41469925e+02, 6.18724059e-03, 8.70866188e-07,
                4.59552743e-02, 6.18724059e-03, 8.70866188e-07, 4.59552743e-02,
                1.80104525e-05, 9.98605856e-07, 4.90858383e-07], dtype=np.float32).reshape((1,-1)),nlev,axis=0)
        
        xmin_lev = np.repeat(np.array([ 1.56582825e+02,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                -1.46704926e+02, -2.35915283e+02, -4.92735580e-03, -1.11688621e-06,
                -4.69117053e-02, -4.92735580e-03, -1.11688621e-06, -4.69117053e-02,
                9.70113589e-09,  1.78764156e-10,  3.65223324e-10], dtype=np.float32).reshape((1,-1)),nlev,axis=0)
        xmean_lev = np.repeat(np.zeros((15), dtype=np.float32).reshape((1,-1)),nlev,axis=0)

        if cfg.include_prev_outputs:
            xmax_lev_prevout = np.repeat(np.array([1.6314061e-03, 8.9240649e-07, 2.0485280e-07, 4.7505119e-07,
                   1.1354494e-03], dtype=np.float32).reshape((1,-1)),nlev,axis=0)
            
            xmin_lev_prevout = np.repeat(np.array([-1.3457362e-03, -9.0609535e-07, -1.1949140e-07, -2.0962088e-07,
                   -1.3081296e-03], dtype=np.float32).reshape((1,-1)),nlev,axis=0)
            xmean_lev_prevout = np.repeat(np.zeros((5), dtype=np.float32).reshape((1,-1)),nlev,axis=0)
            xmax_lev = np.concatenate((xmax_lev, xmax_lev_prevout),axis=1)
            xmin_lev = np.concatenate((xmin_lev, xmin_lev_prevout),axis=1)
            xmean_lev = np.concatenate((xmean_lev, xmean_lev_prevout),axis=1)

    weights=None 
    
    ycoeffs = (yscale_lev, yscale_sca)
    print("Y coeff shapes:", yscale_lev.shape, yscale_sca.shape)
    
    xdiv_lev = xmax_lev - xmin_lev
    if xdiv_lev[-1,-1] == 0.0:
        # the division coefficients for N2O, CH4 are zero in lower atmosphere, fix:
        xdiv1 = xdiv_lev[:,-1]
        xdiv1_min = np.min(xdiv1[xdiv1>0.0])
        xdiv1[xdiv1==0.0] = xdiv1_min
        xdiv_lev[:,-1] = xdiv1 
        
        xdiv1 = xdiv_lev[:,-2]
        xdiv1_min = np.min(xdiv1[xdiv1>0.0])
        xdiv1[xdiv1==0.0] = xdiv1_min
        xdiv_lev[:,-2] = xdiv1 
        
    if cfg.include_prev_inputs:
        xmean_lev = np.concatenate((xmean_lev, xmean_lev[:,0:6]), axis=1)
        xdiv_lev =  np.concatenate((xdiv_lev, xdiv_lev[:,0:6]), axis=1)
                 
    if cfg.add_refpres:
        xmean_lev = np.concatenate((xmean_lev, np.zeros(nlev).reshape(nlev,1)), axis=1)
        xdiv_lev =  np.concatenate((xdiv_lev, np.ones(nlev).reshape(nlev,1)), axis=1)
            
    xmean_sca = input_mean[vars_1D_inp].to_dataarray(dim='features', name='inputs_sca').transpose().values
    xmax_sca = input_max[vars_1D_inp].to_dataarray(dim='features', name='inputs_sca').transpose().values
    xmin_sca = input_min[vars_1D_inp].to_dataarray(dim='features', name='inputs_sca').transpose().values
    xdiv_sca = xmax_sca - xmin_sca

    if cfg.remove_past_sfc_inputs:
        nx_sfc = nx_sfc - 5
        sfc_vars_remove = (17, 18, 19, 20, 21)
        xdiv_sca =  np.delete(xdiv_sca,sfc_vars_remove)
        xmean_sca =  np.delete(xmean_sca,sfc_vars_remove)

    if cfg.model_type=="radflux":
        xdiv_sca[1] = 1.0 
        xdiv_sca[6] = 1.0 
        xmean_sca[1] = 0.0
        xmean_sca[6] = 0.0
        # yscale_sca[0]  = 1.0 
        # yscale_sca[4:] = 1.0 
    elif cfg.model_type=="physrad":
        xdiv_sca[1] = 1.0; xmean_sca[1] = 0.0
        xdiv_sca[6] = 1.0; xmean_sca[6] = 0.0
        xdiv_sca[9] = 1.0; xmean_sca[9] = 0.0
        xdiv_sca[10] = 1.0; xmean_sca[10] = 0.0

    if cfg.snowhice_fix:
        xmean_sca[15] = 0.0 
        xdiv_sca[15] = 1.0 
        
    xcoeff_sca = np.stack((xmean_sca, xdiv_sca))
    xcoeff_lev = np.stack((xmean_lev, xdiv_lev))
    xcoeffs = np.float32(xcoeff_lev), np.float32(xcoeff_sca)

    # ------------------------------------------------------------------------------------------------

    
    if cfg.ensemble_size>1:
        use_ensemble = True 
        is_stochastic = True
        if cfg.loss_fn_type not in ["CRPS","variogram_score","energy_score","ds_score"]:
            raise NotImplementedError("To train stochastic RNN, use CRPS or variogram loss")
    else:
        use_ensemble = False
        is_stochastic = False 

    # ---------------------------------------- SELECT MODEL  -----------------------------------------

    print("Setting up RNN model using nx={}, nx_sfc={}, ny={}, ny_sfc={}".format(nx,nx_sfc,ny,ny_sfc))
    if len(cfg.nneur)==3:
      use_third_rnn = True 
    elif len(cfg.nneur)==2:
      use_third_rnn = False 
    else: 
      raise NotImplementedError()

    if cfg.model_type=="LSTM":
        if cfg.autoregressive:
            model = LSTM_autoreg_torchscript(hyam,hybm,hyai,hybi,
                        out_scale = yscale_lev,
                        out_sfc_scale = yscale_sca, 
                        xmean_lev = xmean_lev, xmean_sca = xmean_sca, 
                        xdiv_lev = xdiv_lev, xdiv_sca = xdiv_sca,
                        device=device,
                        nx = nx, nx_sfc=nx_sfc, 
                        ny = ny, ny_sfc=ny_sfc, 
                        nneur=cfg.nneur, 
                        use_initial_mlp = cfg.use_initial_mlp,
                        use_intermediate_mlp = cfg.use_intermediate_mlp,
                        add_pres = cfg.add_pres,
                        add_stochastic_layer = cfg.add_stochastic_layer, 
                        output_prune = cfg.output_prune,
                        # repeat_mu = cfg.repeat_mu,
                        separate_radiation = cfg.separate_radiation,
                        physical_precip = cfg.physical_precip,
                        # diagnose_precip = diagnose_precip,
                        # diagnose_precip_v2 = diagnose_precip_v2,
                        predict_liq_ratio=predict_liq_ratio,
                        randomly_initialize_cellstate=cfg.randomly_initialize_cellstate,
                        output_sqrt_norm=cfg.new_nolev_scaling,
                        use_third_rnn = use_third_rnn,
                        concat = cfg.concat,
                        nh_mem = cfg.nh_mem)#,
                        #ensemble_size = ensemble_size)
        else:
            model = LSTM_torchscript(hyam,hybm,hyai,hybi,
                        out_scale = yscale_lev,
                        out_sfc_scale = yscale_sca, 
                        xmean_lev = xmean_lev, xmean_sca = xmean_sca, 
                        xdiv_lev = xdiv_lev, xdiv_sca = xdiv_sca,
                        device=device,
                        nx = nx, nx_sfc=nx_sfc, 
                        ny = ny, ny_sfc=ny_sfc, 
                        nneur = cfg.nneur, 
                        use_initial_mlp = cfg.use_initial_mlp,
                        use_intermediate_mlp = cfg.use_intermediate_mlp,
                        add_pres = cfg.add_pres, output_prune = cfg.output_prune)
    elif cfg.model_type=="CFC":
        if cfg.autoregressive:
            model = LiquidNN_autoreg_torchscript(hyam,hybm,hyai,hybi,
                        out_scale = yscale_lev,
                        out_sfc_scale = yscale_sca, 
                        xmean_lev = xmean_lev, xmean_sca = xmean_sca, 
                        xdiv_lev = xdiv_lev, xdiv_sca = xdiv_sca,
                        device=device,
                        nx = nx, nx_sfc=nx_sfc, 
                        ny = ny, ny_sfc=ny_sfc, 
                        nneur=cfg.nneur, 
                        nout_cfc=cfg.cfc_nout,
                        use_initial_mlp = cfg.use_initial_mlp,
                        use_intermediate_mlp = cfg.use_intermediate_mlp,
                        add_pres = cfg.add_pres,
                        add_stochastic_layer = cfg.add_stochastic_layer, 
                        output_prune = cfg.output_prune,
                        concat = cfg.concat,
                        nh_mem = cfg.nh_mem)#,
    elif cfg.model_type in ["SLSTM", "SGRU"]: #cfg.model_type=="SRNN":
        if cfg.model_type=="SLSTM":
            use_lstm=True 
        else:
            use_lstm=False 
        model = stochastic_RNN_autoreg_torchscript(hyam,hybm,hyai,hybi,
                    out_scale = yscale_lev,
                    out_sfc_scale = yscale_sca, 
                    xmean_lev = xmean_lev, xmean_sca = xmean_sca, 
                    xdiv_lev = xdiv_lev, xdiv_sca = xdiv_sca,
                    device=device,
                    nx = nx, nx_sfc=nx_sfc, 
                    ny = ny, ny_sfc=ny_sfc, 
                    nneur=cfg.nneur, 
                    use_lstm=use_lstm,
                    use_initial_mlp = cfg.use_initial_mlp,
                    use_intermediate_mlp = cfg.use_intermediate_mlp,
                    add_pres = cfg.add_pres,
                    output_prune = cfg.output_prune,
                    use_memory = cfg.autoregressive,
                    nh_mem = cfg.nh_mem,
                    ar_noise_mode = cfg.ar_noise_mode,
                    ar_tau = cfg.ar_tau,
                    use_surface_memory=cfg.use_surface_memory)#,
    elif cfg.model_type=="LSTM_autoreg_torchscript_perturb":
        # if cfg.autoregressive:
        model =  LSTM_autoreg_torchscript_perturb(hyam,hybm,hyai,hybi,
                                    out_scale = yscale_lev,
                                    out_sfc_scale = yscale_sca, 
                                    xmean_lev = xmean_lev, xmean_sca = xmean_sca, 
                                    xdiv_lev = xdiv_lev, xdiv_sca = xdiv_sca,
                                    device=device,
                                    nx = nx, nx_sfc=nx_sfc, 
                                    ny = ny, ny_sfc=ny_sfc, 
                                    nneur=cfg.nneur, 
                                    use_initial_mlp=cfg.use_initial_mlp, 
                                    separate_radiation=cfg.separate_radiation,
                                    randomly_initialize_cellstate=cfg.randomly_initialize_cellstate,
                                    deterministic_mode=not is_stochastic,
                                    add_pres = cfg.add_pres,
                                    output_prune = cfg.output_prune,
                                    nh_mem=cfg.nh_mem)
    elif cfg.model_type=="halfstochasticRNN":
        model = halfstochastic_RNN_autoreg_torchscript(hyam,hybm,hyai,hybi,
                    out_scale = yscale_lev,
                    out_sfc_scale = yscale_sca, 
                    xmean_lev = xmean_lev, xmean_sca = xmean_sca, 
                    xdiv_lev = xdiv_lev, xdiv_sca = xdiv_sca,
                    device=device,
                    nx = nx, nx_sfc=nx_sfc, 
                    ny = ny, ny_sfc=ny_sfc, 
                    nneur=cfg.nneur, 
                    use_initial_mlp = cfg.use_initial_mlp,
                    use_intermediate_mlp = cfg.use_intermediate_mlp,
                    add_pres = cfg.add_pres,
                    output_prune = cfg.output_prune,
                    nh_mem = cfg.nh_mem,
                    ar_noise_mode = cfg.ar_noise_mode,
                    ar_tau = cfg.ar_tau,
                    use_surface_memory=cfg.use_surface_memory)#,
    elif cfg.model_type=="physrad":
        from models_rad import LSTM_autoreg_torchscript_physrad
        model = LSTM_autoreg_torchscript_physrad(hyam,hybm,hyai,hybi,
                    out_scale = yscale_lev,
                    out_sfc_scale = yscale_sca, 
                    xmean_lev = xmean_lev, xmean_sca = xmean_sca, 
                    xdiv_lev = xdiv_lev, xdiv_sca = xdiv_sca,
                    device=device,
                    nx = nx, nx_sfc=nx_sfc, 
                    ny = ny, ny_sfc=ny_sfc, 
                    nneur=cfg.nneur, 
                    use_initial_mlp = cfg.use_initial_mlp,
                    use_intermediate_mlp = cfg.use_intermediate_mlp,
                    add_pres = cfg.add_pres,
                    add_stochastic_layer = cfg.add_stochastic_layer, 
                    output_prune = cfg.output_prune,
                    use_third_rnn = use_third_rnn,
                    concat = cfg.concat,
                    nh_mem = cfg.nh_mem,
                    mp_mode = cfg.mp_mode)
    elif cfg.model_type=="radflux":
        model = LSTM_autoreg_torchscript_radflux(hyam,hybm,hyai,hybi,
                    out_scale = yscale_lev,
                    out_sfc_scale = yscale_sca, 
                    xmean_lev = xmean_lev, xmean_sca = xmean_sca, 
                    xdiv_lev = xdiv_lev, xdiv_sca = xdiv_sca,
                    device=device,
                    nx = nx, nx_sfc=nx_sfc, 
                    ny = ny, ny_sfc=ny_sfc, 
                    nneur=cfg.nneur, 
                    use_initial_mlp = cfg.use_initial_mlp,
                    use_intermediate_mlp = cfg.use_intermediate_mlp,
                    add_pres = cfg.add_pres,
                    output_prune = cfg.output_prune,
                    use_memory = cfg.autoregressive,
                    nh_mem = cfg.nh_mem,
                    mp_mode = cfg.mp_mode)
    elif cfg.model_type=="LSTM_sepmp":
        model = LSTM_autoreg_torchscript_mp(hyam,hybm,hyai,hybi,
                        out_scale = yscale_lev,
                        out_sfc_scale = yscale_sca, 
                        xmean_lev = xmean_lev, xmean_sca = xmean_sca, 
                        xdiv_lev = xdiv_lev, xdiv_sca = xdiv_sca,
                        device=device,
                        nx = nx, nx_sfc=nx_sfc, 
                        ny = ny, ny_sfc=ny_sfc, 
                        nneur=cfg.nneur, 
                        use_initial_mlp = cfg.use_initial_mlp,
                        use_intermediate_mlp = cfg.use_intermediate_mlp,
                        add_pres = cfg.add_pres,
                        add_stochastic_layer = cfg.add_stochastic_layer, 
                        output_prune = cfg.output_prune,
                        use_memory = cfg.autoregressive,
                        # separate_radiation = cfg.separate_radiation,
                        # diagnose_precip = diagnose_precip,
                        # use_third_rnn = use_third_rnn,
                        concat = cfg.concat,
                        nh_mem = cfg.nh_mem)#,
                        #ensemble_size = ensemble_size)               
    else:
      print("using SSM")
      model = SpaceStateModel_autoreg(hyam,hybm,hyai,hybi,
                    out_scale = yscale_lev,
                    out_sfc_scale = yscale_sca, 
                    xmean_lev = xmean_lev, xmean_sca = xmean_sca, 
                    xdiv_lev = xdiv_lev, xdiv_sca = xdiv_sca,
                    device=device,
                    nx = nx, nx_sfc=nx_sfc, 
                    ny = ny, ny_sfc=ny_sfc, 
                    nneur=cfg.nneur, 
                    model_type=cfg.model_type,
                    use_initial_mlp = cfg.use_initial_mlp,
                    use_intermediate_mlp = cfg.use_intermediate_mlp,
                    add_pres = cfg.add_pres,
                    output_prune = cfg.output_prune,
                    nh_mem = cfg.nh_mem)#,
    
    model = model.to(device)
    
    infostr = summary(model)
    num_params = infostr.total_params

    # ------------------------------------------------------------------------------------------------
    # --------------------------------------- DATA I/O -----------------------------------------------

    if cfg.autoregressive:
        batch_size_tr = nloc
    else:
        batch_size_tr = cfg.batch_size_tr

    pin = False
    persistent=False
    
    if cfg.num_workers==0:
        no_multiprocessing=True
        prefetch_factor = None
    else:
        no_multiprocessing=False
        prefetch_factor = 1


    # To improve IO, which is a bottleneck, increase the batch size by a factor of chunk_factor and load this many
    # batches at once. These chk then need to be manually split into batches within the data iteration loop  
    train_data = generator_xy(tr_data_path, cache = cfg.cache, nloc = nloc, add_refpres = cfg.add_refpres, 
                    remove_past_sfc_inputs = cfg.remove_past_sfc_inputs, mp_mode = cfg.mp_mode,
                    v4_to_v5_inputs = cfg.v4_to_v5_inputs, rh_prune = cfg.rh_prune,  
                    input_norm_per_level=cfg.input_norm_per_level,
                    output_sqrt_norm=cfg.new_nolev_scaling,
                    qinput_prune = cfg.qinput_prune, output_prune = cfg.output_prune, 
                    include_prev_inputs=cfg.include_prev_inputs, include_prev_outputs=cfg.include_prev_outputs,
                    ycoeffs=ycoeffs, xcoeffs=xcoeffs, no_multiprocessing=no_multiprocessing)
    
    train_batch_sampler = BatchSampler(cfg.chunksize_train, # samples per chunk 
                                       # num_samples=train_data.ntimesteps*nloc, shuffle=shuffle_data)
                                       num_samples = train_data.ntimesteps, shuffle = cfg.shuffle_data, 
                                       skip_first=skip_first_index)
    
    train_loader = DataLoader(dataset = train_data, num_workers = cfg.num_workers, 
                              sampler = train_batch_sampler, 
                              batch_size=None, batch_sampler=None, 
                              prefetch_factor = prefetch_factor, 
                              pin_memory=pin, persistent_workers=persistent)
    

    val_data = generator_xy(val_data_path, cache=cfg.val_cache, add_refpres = cfg.add_refpres, 
                    remove_past_sfc_inputs = cfg.remove_past_sfc_inputs, mp_mode = cfg.mp_mode,
                    v4_to_v5_inputs = cfg.v4_to_v5_inputs, rh_prune = cfg.rh_prune, 
                    input_norm_per_level=cfg.input_norm_per_level,
                    output_sqrt_norm=cfg.new_nolev_scaling,
                    qinput_prune = cfg.qinput_prune, output_prune = cfg.output_prune, 
                    include_prev_inputs=cfg.include_prev_inputs, include_prev_outputs=cfg.include_prev_outputs,
                    ycoeffs=ycoeffs, xcoeffs=xcoeffs, no_multiprocessing=no_multiprocessing)
    
    val_batch_sampler = BatchSampler(cfg.chunksize_val, 
                                       # num_samples=val_data.ntimesteps*nloc_val, shuffle=shuffle_data)
                                       num_samples=val_data.ntimesteps, shuffle = cfg.shuffle_data,
                                       skip_first=skip_first_index)
    
    val_loader = DataLoader(dataset=val_data, num_workers = cfg.num_workers, 
                              sampler = val_batch_sampler, 
                              batch_size=None, batch_sampler=None, 
                              prefetch_factor = prefetch_factor, 
                              pin_memory=pin, persistent_workers=persistent)
    if cfg.autoregressive:
        nloc_val = batch_size_val = val_data.nloc
    else:
        nloc_val = val_data.nloc
        batch_size_val = cfg.batch_size_val 

    print("batch_size_tr: {}, batch_size_val: {}".format(batch_size_tr, batch_size_val))


    # -------------------------------------------------------------------------------------------------------
    # ----------------------------------- METRICS AND LOSS FUNCTIONS  ---------------------------------------

    metric_h_con = metrics.get_energy_metric(hyai, hybi)
    metric_water_con = metrics.get_water_conservation(hyai, hybi)

    # mse = metrics.get_mse_flatten(weights)
    metrics_det = metrics.get_metrics_flatten(weights)
    
    if cfg.loss_fn_type == "mse":
        loss_fn = metrics_det
    elif cfg.loss_fn_type == "huber":
        loss_fn = metrics_det
    elif cfg.loss_fn_type == "CRPS":
        loss_fn = metrics.get_CRPS(cfg.crps_sumvar)
        print("CRPS sum variables first:", cfg.crps_sumvar)
    elif cfg.loss_fn_type == "variogram_score":
        loss_fn = metrics.variogram_score   
    elif cfg.loss_fn_type == "energy_score":
        loss_fn = metrics.energy_score   
    elif cfg.loss_fn_type == "ds_score":
        loss_fn = metrics.ds_score   
    else:
        raise NotImplementedError("loss_fn {} not implemented".format(cfg.loss_fn_type))
        
    # -------------------------------------------------------------------------------------------------------
    # --------------------------------------------- OPTIMIZER -----------------------------------------------

    if cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = cfg.lr)
    elif cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr = cfg.lr)
    elif cfg.optimizer == "adamwschedulefree":
        import schedulefree
        optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=cfg.lr)
    elif cfg.optimizer == "soap":
        from soap import SOAP
        optimizer = SOAP(model.parameters(), lr = cfg.lr, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
    elif cfg.optimizer == "muon":
        from muon import SingleDeviceMuonWithAuxAdam
        hidden_weights = [p for p in model.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.parameters() if p.ndim < 2]
        rnn_params = [*model.rnn1.parameters(), *model.rnn2.parameters()]
        param_groups = [
            dict(params=hidden_weights, use_muon=True,
                lr=cfg.lr, weight_decay=0.01),
            dict(params=hidden_gains_biases, use_muon=False,
                lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)  
    else:
        raise NotImplementedError(("optimizer {} not implemented".format(cfg.optimizer)))

    # -------------------------------------------------------------------------------------------------------
    # -------------------------------- ROLLOUT SCHEDULE AND LR SCHEDULER ------------------------------------

    timewindow_default = 1
    rollout_schedule = cfg.rollout_schedule #timestep_schedule[0:10].tolist()
    timestep_schedule = np.arange(1000)
    timestep_schedule[0:len(rollout_schedule)] = rollout_schedule
    timestep_schedule[len(rollout_schedule):] = rollout_schedule[len(rollout_schedule)-1]

    if cfg.lr_scheduler=="OneCycleLR":
        #   https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        max_lr = cfg.scheduler_max_lr
        min_lr = cfg.scheduler_min_lr
        final_div_factor = cfg.lr/min_lr
        scheduler_end_epoch = cfg.scheduler_end_epoch
        pct_start = cfg.scheduler_peak_epoch / scheduler_end_epoch
        annealing = cfg.scheduler_annealing # "linear" #"cos"
        print("Using OneCycleLR with max_lr={} total steps={}, min_lr={}, pct_start={}, anneal={}".format(max_lr, scheduler_end_epoch, min_lr, pct_start, annealing))
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, 
                                                            div_factor=max_lr/cfg.lr,
                                                            final_div_factor=final_div_factor,
                                                            total_steps=scheduler_end_epoch,
                                                            pct_start=pct_start, 
                                                            anneal_strategy=annealing)                                                         
    elif cfg.lr_scheduler=="StepLR":
      print("Using StepLR with gamma={}".format(cfg.lr_gamma))
      lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg.lr_gamma, last_epoch=-1)
    elif cfg.lr_scheduler=="None":
      print("not using a LR scheduler")
      lr_scheduler=None
    else:
      raise NotImplementedError(("scheduler {} not supported".format(cfg.lr_scheduler)))
    
    conf = OmegaConf.to_container(cfg)
    dtypestr = "{}".format(dtype)
    cwd = os.getcwd()
    conf["dtypestr"] = dtypestr
    conf["cwd"] = cwd
    conf["model_num"] = randrange(10,99999)
    conf["num_params"] = num_params 
    
    if cfg.use_wandb:
        os.environ["WANDB__SERVICE_WAIT"]="400"
        run = wandb.init(
            project="climsim",
            config=conf
        )       
    
    train_runner = train_or_eval_one_epoch(train_loader, model, device, dtype, cfg, metrics_det, 
                                           metric_h_con, metric_water_con, batch_size_tr,  train=True, model_is_stochastic=is_stochastic)
    val_runner = train_or_eval_one_epoch(val_loader, model, device, dtype, cfg, metrics_det,
                                           metric_h_con, metric_water_con, batch_size_val, train=False, model_is_stochastic=is_stochastic)
    
    # Strings for saving model 
    inpstr = "v5" if cfg.v4_to_v5_inputs else "v4"
    MODEL_STR =  '{}-{}_lr{}.neur{}-{}_x{}_mp{}_num{}'.format(cfg.model_type,
                                                                     cfg.memory, cfg.lr, 
                                                                     cfg.nneur[0], cfg.nneur[1], 
                                                                     inpstr, cfg.mp_mode,
                                                                     conf["model_num"] )

    SAVE_PATH       = "saved_models/" + MODEL_STR + ".pt"
    save_file_torch = "saved_models/" + MODEL_STR + "_script.pt"
    
    best_val_loss = np.inf
    # best_val_loss = 0.0
    
    new_lr = cfg.lr
    
    start_epoch=0
    # load from checkpoint of it exists

    if len(cfg.model_file_checkpoint)>0:
        print("lading existing model from {}".format(cfg.model_file_checkpoint))
        checkpoint = torch.load("saved_models/"+cfg.model_file_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        if not cfg.only_load_model:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']

    # --------------------------------------------------------------------------------------------------------
    # ----------------------------------------- START TRAINING -----------------------------------------------

    for epoch in range(start_epoch, cfg.num_epochs):
        t0 = time.time()
        
        if cfg.timestep_scheduling:
            timesteps=timestep_schedule[epoch]            
        else:
            timesteps=timewindow_default
            
        print("Epoch {} Training rollout timesteps: {} ".format(epoch+1, timesteps))

        train_runner.eval_one_epoch(loss_fn, optimizer, epoch, timesteps, lr_scheduler)

        if cfg.lr_scheduler != "None":
          # lr_scheduler.step()
          # # debugging purpose
          print("last LR was", lr_scheduler.get_last_lr()) # will print last learning rate.
          
        if cfg.lr_scheduler=="StepLR":
            print("Calling scheduler .step()")
            lr_scheduler.step()
        elif cfg.lr_scheduler=="OneCycleLR" and epoch<(cfg.scheduler_end_epoch-1):
            print("Calling scheduler .step()")
            lr_scheduler.step()
        if train_runner.loader.dataset.cache:
            train_runner.loader.dataset.cache_loaded = True
        
        if cfg.use_wandb: 
            logged_metrics = train_runner.metrics.copy()
            logged_metrics.pop("rmse_perlev"); logged_metrics.pop("bias_perlev")
            nan_inds = np.isnan(logged_metrics['R2_lev']); num_nans = np.sum(nan_inds)
            nan_inds_moistening = np.isnan(logged_metrics['R2_lev'][:,1])
            # num_nans_moist = np.sum(nan_inds_moistening)
            logged_metrics['R2_lev'][nan_inds] = 0.0
            inf_inds = np.isinf(logged_metrics['R2_lev'])
            logged_metrics['R2_lev'][inf_inds] = 0.0
            # logged_metrics['R2_lev'][:,1][nan_inds_moistening] = 0.0
            logged_metrics['num_nans'] = num_nans
            # logged_metrics['num_nans_moist'] = num_nans_moist
            logged_metrics = {"train_"+k:v for k, v in logged_metrics.items()}
            logged_metrics['epoch'] = epoch
            wandb.log(logged_metrics)

        if (np.isnan(logged_metrics['train_loss']) or (logged_metrics['train_R2'] < 0.0)):
            sys.exit("Loss was NaN or R-squared was below 0 - something's wrong, stopping training") 
        
        # if (bool(epoch%2) and (epoch>=cfg.val_epoch_start)):
        if (epoch>=cfg.val_epoch_start):
            print("VALIDATION..")
            val_runner.eval_one_epoch(loss_fn, optimizer, epoch, timesteps)
            if val_runner.loader.dataset.cache:
                val_runner.loader.dataset.cache_loaded = True

            if cfg.use_wandb: 
                logged_metrics = val_runner.metrics.copy()
                logged_metrics.pop("rmse_perlev"); logged_metrics.pop("bias_perlev")
                nan_inds = np.isnan(logged_metrics['R2_lev']); num_nans = np.sum(nan_inds)
                nan_inds_moistening = np.isnan(logged_metrics['R2_lev'][:,1])
                # num_nans_moist = np.sum(nan_inds_moistening)
                logged_metrics['R2_lev'][nan_inds] = 0.0
                # logged_metrics['R2_lev'][:,1][nan_inds_moistening] = 0.0
                inf_inds = np.isinf(logged_metrics['R2_lev'])
                logged_metrics['R2_lev'][inf_inds] = 0.0
                logged_metrics['num_nans'] = num_nans
                # logged_metrics['num_nans_moist'] = num_nans_moist
                logged_metrics = {"val_"+k:v for k, v in logged_metrics.items()}
                logged_metrics['epoch'] = epoch
                wandb.log(logged_metrics)
    
            val_loss = val_runner.metrics["loss"]
            # val_loss = val_runner.metrics["R2"]

            # MODEL CHECKPOINT IF VALIDATION LOSS IMPROVED
            # if True:
            if cfg.save_model and val_loss < best_val_loss:
            # if cfg.save_model and val_loss > best_val_loss:
                save_file_torch1 = "saved_models/" + MODEL_STR + "_script_gpu.pt"
                save_file_torch2 = "saved_models/" + MODEL_STR + "_script_cpu.pt"
                print("saving model to", SAVE_PATH)
                # print("New best validation result obtained, saving model to", SAVE_PATH)
                if is_stochastic:
                    model.use_ensemble=False
                # model = model.to("cpu")
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': lr_scheduler.state_dict(),
                            'val_loss': val_loss,
                            }, SAVE_PATH)  
                scripted_model = torch . jit . script ( model )
                scripted_model = scripted_model.eval()
                scripted_model.save(save_file_torch1)
                model = model.to("cpu")
                scripted_model = torch . jit . script ( model )
                scripted_model = scripted_model.eval()
                scripted_model.save(save_file_torch2)
                best_val_loss = val_loss 
                if is_stochastic:
                    model.use_ensemble=True
                model = model.to(device)
                print("model saved!")

                R2 = val_runner.metrics["R2_lev"]
                labels = ["dT/dt", "dq/dt", "dqliq/dt", "dqice/dt", "dU/dt", "dV/dt"]
                ncols, nrows = 3,2
                fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(9.5, 4.5),
                                        layout="constrained")
                j = 0
                for irow in range(2):
                    for icol in range(3):
                        axs[irow,icol].plot(R2[:,j],level)
                        axs[irow,icol].set_ylim(0,1000)
                        axs[irow,icol].invert_yaxis()
                        axs[irow,icol].set_xlim(0,1)
                        axs[irow,icol].set_title(labels[j])
                        j = j + 1
                    
                fig.subplots_adjust(hspace=0)
                plt.savefig('saved_models/val_eval/' + MODEL_STR + 'val_R2.pdf')

            plt.clf()
            rmse = val_runner.metrics["rmse_perlev"]
            fig, axs = plt.subplots(ncols=1, nrows=6, figsize=(7.0, 12.0)) #layout="constrained")
            for i in range(6):
                axs[i].plot(np.arange(60), rmse[:,i]); 
                axs[i].set_title(labels[i])
                axs[i].set_xlim(0,60)
                axs[i].axvspan(0, 30, facecolor='0.2', alpha=0.2)

            fig.subplots_adjust(hspace=0.6)                                                     
            plt.savefig('saved_models/val_eval/' + MODEL_STR + 'val_rmse.pdf')
            
            if batch_size_val==384:
                dt_diff = val_runner.epoch_bias_collev[:,:,0]
                q_diff = val_runner.epoch_bias_collev[:,:,1]
                clw_diff = val_runner.epoch_bias_collev[:,:,2]
                cli_diff = val_runner.epoch_bias_collev[:,:,3]
                v_diff = val_runner.epoch_bias_collev[:,:,5]

                vars_stacked = [dt_diff, q_diff, v_diff, clw_diff, cli_diff]
                                
                lat = grid_info.lat
                grid_area =  grid_info.area
                fig = plot_bias_diff(vars_stacked, grid_area, lat, level)
                fig.savefig('saved_models/val_eval/' + MODEL_STR + 'val_zonalmeanbias.png')


        print('Epoch {}/{} complete, took {:.2f} seconds, autoreg window was {}'.format(epoch+1,cfg.num_epochs,time.time() - t0,timesteps))
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush=True)

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()