#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 10:15:11 2025

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
from models import  MyRNN, LSTM_autoreg_torchscript, SpaceStateModel, LSTM_torchscript, SRNN_autoreg_torchscript, LSTM_autoreg_torchscript_radflux
from utils import generator_xy, BatchSampler
# from metrics import get_energy_metric, get_hybrid_loss, my_mse_flatten
import metrics as metrics
from torchmetrics.regression import R2Score
import wandb
from omegaconf import DictConfig
import hydra
import matplotlib.pyplot as plt


@hydra.main(version_base="1.2", config_path="conf", config_name="autoreg_LSTM")
def main(cfg: DictConfig):
        
    grid_path = '../grid_info/ClimSim_low-res_grid-info.nc'
    norm_path = '../preprocessing/normalizations/'
    tr_data_path = cfg.tr_data_dir + cfg.tr_data_fname
    val_data_path = cfg.val_data_dir + cfg.val_data_fname

    #torch.set_float32_matmul_precision("medium")
    #torch.backends.cuda.matmul.allow_tf32 = True    

    print('RAM memory % used:', psutil.virtual_memory()[2], flush=True)
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush=True)

    # cfg = OmegaConf.load("conf/autoreg_LSTM.yaml")
    
    # mp_mode = 0   # regular 6 outputs
    # mp_mode = 1   # 5 outputs, pred qn, liq_ratio diagnosed (mp_constraint)
    # mp_mode = 2   # 6 outputs, pred qn and liq_ratio
    if cfg.mp_mode>0:
        use_mp_constraint=True
    else:
        use_mp_constraint=False 
        
    
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
    
    # ns, nlev, nx, nx_sfc, ny, ny_sfc = get_input_output_shapes(tr_data_path)

    hf = h5py.File(tr_data_path, 'r')
    print(hf.keys()) # <KeysViewHDF5 ['input_lev', 'input_sca', 'output_lev', 'output_sca']>
    print(hf['input_lev'].attrs.keys())
    dims = hf['input_lev'].shape
    if len(dims)==4:
        ns, nloc, nlev, nx = dims 
    else:
        ns, nlev, nx = dims
    
    vars_2D_inp = hf['input_lev'].attrs.get('varnames').tolist()
    # ['state_t', 'state_rh', 'state_q0002', 'state_q0003', 'state_u',
    #  'state_v', 'state_t_dyn', 'state_q0_dyn', 'state_u_dyn', 'tm_state_t_dyn',
    #  'tm_state_q0_dyn', 'tm_state_u_dyn', 'pbuf_ozone', 'pbuf_CH4', 'pbuf_N2O']
    
    dims = hf['input_sca'].shape; nx_sfc = dims[-1]
    print("nx_sfc:", nx_sfc)
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
    
    if cfg.mp_mode>0:
        vars_2D_outp.remove('ptend_q0002')
        vars_2D_outp.remove('ptend_q0003')
        vars_2D_outp.insert(2,"ptend_qn")
    
    if cfg.include_prev_outputs:
        if not cfg.output_norm_per_level:
            raise NotImplementedError("Only level-specific norm coefficients saved for previous tendency outputs")

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
        
    # if use_mp_constraint:
    if cfg.mp_mode==1:
        ny_pp = ny # The 6 original outputs will be after postprocessing
        ny = ny - 1 # The model itself only has 5 outputs (total cloud water)
    else:
        ny_pp = ny 
    # ny_pp = ny 

    print("ns", ns, "nloc", nloc, "nlev", nlev,  "nx", nx, "nx_sfc", nx_sfc, "ny", ny, "ny_sfc", ny, flush=True)

    yscale_sca = output_scale[vars_1D_outp].to_dataarray(dim='features', name='outputs_sca').transpose().values

    if cfg.output_norm_per_level:
        yscale_lev = output_scale[vars_2D_outp].to_dataarray(dim='features', name='outputs_lev').transpose().values
    else:
        yscale_lev = np.repeat(np.array([2.3405453e+04, 2.3265182e+08, 1.4898973e+08, 6.4926711e+04,
                7.8328773e+04], dtype=np.float32).reshape((1,-1)),nlev,axis=0)
        if cfg.new_nolev_scaling:
            if use_mp_constraint:
                
                yscale_lev = np.repeat(np.array([1.87819239e+04, 3.25021485e+07, 1.58085550e+08, 5.00182069e+04,
                       6.21923225e+04], dtype=np.float32).reshape((1,-1)),nlev,axis=0)
            else:
                yscale_lev = np.repeat(np.array([1.87819239e+04, 3.25021485e+07, 1.91623978e+08, 3.23919949e+08, 
                    5.00182069e+04, 6.21923225e+04], dtype=np.float32).reshape((1,-1)),nlev,axis=0)
                
    if cfg.input_norm_per_level:
        xmax_lev = input_max[vars_2D_inp].to_dataarray(dim='features', name='inputs_lev').transpose().values
        xmin_lev = input_min[vars_2D_inp].to_dataarray(dim='features', name='inputs_lev').transpose().values
        xmean_lev = input_mean[vars_2D_inp].to_dataarray(dim='features', name='inputs_lev').transpose().values
    else:
        xmax_lev = np.repeat(np.array([2.9577750e+02, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
               7.6608604e+01, 5.4018818e+01, 2.3003130e-03, 3.2301173e-07,
               4.1381191e-03, 2.3003130e-03, 3.2301173e-07, 4.1381191e-03,
               2.7553122e-06, 8.6128614e-07, 4.0667697e-07], dtype=np.float32).reshape((1,-1)),nlev,axis=0)
        xmin_lev = np.repeat(np.array([ 1.9465800e+02,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
               -4.5055122e+01, -5.4770996e+01, -2.7256422e-03, -2.8697787e-07,
               -3.2421835e-03, -2.7256422e-03, -2.8697787e-07, -3.2421835e-03,
                1.0383576e-06,  6.6867170e-07,  3.3562134e-07], dtype=np.float32).reshape((1,-1)),nlev,axis=0)
        xmean_lev = np.repeat(np.array([ 2.47143495e+02,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                9.02305768e+00, -1.85453092e-02,  0.00000000e+00,  0.00000000e+00,
                0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                1.95962779e-06,  8.22708859e-07,  3.88440249e-07], dtype=np.float32).reshape((1,-1)),nlev,axis=0)
    
        if cfg.new_nolev_scaling:
            xmax_lev = np.repeat(np.array([3.21864136e+02, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
                   2.13669708e+02, 1.41469925e+02, 6.18724059e-03, 8.70866188e-07,
                   4.59552743e-02, 6.18724059e-03, 8.70866188e-07, 4.59552743e-02,
                   1.80104525e-05, 9.98605856e-07, 4.90858383e-07], dtype=np.float32).reshape((1,-1)),nlev,axis=0)
            
            xmin_lev = np.repeat(np.array([ 1.56582825e+02,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
                   -1.46704926e+02, -2.35915283e+02, -4.92735580e-03, -1.11688621e-06,
                   -4.69117053e-02, -4.92735580e-03, -1.11688621e-06, -4.69117053e-02,
                    9.70113589e-09,  1.78764156e-10,  3.65223324e-10], dtype=np.float32).reshape((1,-1)),nlev,axis=0)
            xmean_lev = np.repeat(np.zeros((15), dtype=np.float32).reshape((1,-1)),nlev,axis=0)

    weights=None 
    
    if cfg.mp_mode==2:
        liq_frac_scale = np.ones(nlev,dtype=np.float32).reshape(nlev,1)
        liq_frac_scale[:] = 2.4
        yscale_lev = np.concatenate((yscale_lev[:,0:3], liq_frac_scale, yscale_lev[:,3:]), axis=1)
    
            
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

    if cfg.snowhice_fix:
        xmean_sca[15] = 0.0 
        xdiv_sca[15] = 1.0 
        
    xcoeff_sca = np.stack((xmean_sca, xdiv_sca))
    xcoeff_lev = np.stack((xmean_lev, xdiv_lev))
    xcoeffs = np.float32(xcoeff_lev), np.float32(xcoeff_sca)
    
    
    if cfg.add_stochastic_layer:
        use_ensemble = True 
        ensemble_size = 2
        if cfg.loss_fn_type != "CRPS":
            raise NotImplementedError("To train stochastic RNN, use CRPS loss")
    else:
        use_ensemble = False
        ensemble_size = 1
        
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
                        use_memory = cfg.autoregressive,
                        separate_radiation = cfg.separate_radiation,
                        use_ensemble = use_ensemble,
                        use_third_rnn = use_third_rnn,
                        concat = cfg.concat,
                        nh_mem = cfg.nh_mem)#,
                        #ensemble_size = ensemble_size)
        else:
            model = LSTM_torchscript(hyam,hybm,
                        out_scale = yscale_lev,
                        out_sfc_scale = yscale_sca, 
                        nx = nx, nx_sfc=nx_sfc, 
                        ny = ny, ny_sfc=ny_sfc, 
                        nneur = cfg.nneur, 
                        use_initial_mlp = cfg.use_initial_mlp,
                        use_intermediate_mlp = cfg.use_intermediate_mlp,
                        add_pres = cfg.add_pres, output_prune = cfg.output_prune)
    elif cfg.model_type=="SRNN":
        model = SRNN_autoreg_torchscript(hyam,hybm,hyai,hybi,
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
                    use_ensemble = use_ensemble,
                    nh_mem = cfg.nh_mem)#,
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
                    use_ensemble = use_ensemble,
                    nh_mem = cfg.nh_mem)#,
    else:
        model = SpaceStateModel(hyam, hybm, 
                    out_scale = yscale_lev,
                    out_sfc_scale = yscale_sca,  
                    nlev=60, nx = nx, nx_sfc=nx_sfc, 
                    ny = ny, ny_sfc=ny_sfc, 
                    nneur = cfg.nneur, model_type = cfg.model_type, 
                    use_initial_mlp = cfg.use_initial_mlp, add_pres=cfg.add_pres,  concat=cfg.concat)
    
    model = model.to(device)
    
    infostr = summary(model)
    num_params = infostr.total_params
    
    if cfg.use_scaler:
        # scaler = torch.amp.GradScaler(autocast = True)
        scaler = torch.amp.GradScaler(device.type)
            
    batch_size_tr = nloc
    
    # To improve IO, which is a bottleneck, increase the batch size by a factor of chunk_factor and 
    # load this many batches at once. These chk then need to be manually split into batches 
    # within the data iteration loop   
    pin = False
    persistent=False
    
    # if type(tr_data_path)==list:  
    #     num_files = len(tr_data_path)
    #     batch_size_tr = num_files*batch_size_tr 
    #     chunk_size_tr = cfg.chunksiz // num_files
    # else:
    # num_files = 1
    # chunk size in number of batches
    # 720 = 10 days (3 time steps in an hour, 72 in a day)
    # chunk_size_tr = cfg.chunksize_train
    
    if cfg.num_workers==0:
        no_multiprocessing=True
        prefetch_factor = None
    else:
        no_multiprocessing=False
        prefetch_factor = 1

    
    train_data = generator_xy(tr_data_path, cache = cfg.cache, nloc = nloc, add_refpres = cfg.add_refpres, 
                    remove_past_sfc_inputs = cfg.remove_past_sfc_inputs, mp_mode = cfg.mp_mode,
                    v4_to_v5_inputs = cfg.v4_to_v5_inputs, rh_prune = cfg.rh_prune, 
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
    
    nloc_val = batch_size_val = val_data.nloc
    
    metric_h_con = metrics.get_energy_metric(hyai, hybi)
    metric_water_con = metrics.get_water_conservation(hyai, hybi)

    # mse = metrics.get_mse_flatten(weights)
    det_metrics = metrics.get_metrics_flatten(weights)
    
    if cfg.loss_fn_type == "mse":
        loss_fn = det_metrics
    elif cfg.loss_fn_type == "huber":
        loss_fn = det_metrics
    elif cfg.loss_fn_type == "CRPS":
        loss_fn = metrics.get_CRPS(cfg.beta)
    else:
        raise NotImplementedError()
        
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
    else:
        raise NotImplementedError()
  
    timewindow_default = 1
    rollout_schedule = cfg.rollout_schedule #timestep_schedule[0:10].tolist()
    timestep_schedule = np.arange(1000)
    timestep_schedule[0:len(rollout_schedule)] = rollout_schedule
    timestep_schedule[len(rollout_schedule):] = rollout_schedule[len(rollout_schedule)-1]

            
    from random import randrange
    model_num = randrange(10,99999)
    
    conf = OmegaConf.to_container(cfg)
    dtypestr = "{}".format(dtype)
    cwd = os.getcwd()
    conf["dtypestr"] = dtypestr
    conf["cwd"] = cwd
    conf["model_num"] = model_num
    conf["num_params"] = num_params 
    
    # OmegaConf.save(sorted(config), "conf/autoreg_LSTM.yaml")
    if cfg.use_wandb:
        os.environ["WANDB__SERVICE_WAIT"]="400"
        run = wandb.init(
            project="climsim",
            config=conf
        )       
    
    class model_train_eval:
        def __init__(self, dataloader, model, batch_size = 384, train=True):
            super().__init__()
            self.loader = dataloader
            self.dtype = dtype
            self.train = train
            self.report_freq = 800
            self.batch_size = batch_size
            self.model = model 
            # self.metric_R2 =  R2Score(num_outputs=ny_pp).to(device) 
            self.metric_R2 =  R2Score().to(device) 

            self.metrics = {}
            
        def eval_one_epoch(self, optim, epoch, timesteps=1):
            report_freq = self.report_freq
            running_loss = 0.0; running_energy = 0.0; running_water = 0.0
            running_var=0.0
            epoch_loss = 0.0; epoch_mse = 0.0; epoch_huber = 0.0; epoch_mae = 0.0
            epoch_R2precc = 0.0
            epoch_hcon = 0.0; epoch_wcon = 0.0
            epoch_ens_var = 0.0; epoch_det_skill = 0.0; epoch_spreadskill = 0.0
            epoch_r2_lev = 0.0
            epoch_bias_lev = 0.0; epoch_bias_sfc = 0.0; epoch_bias_heating = 0.0
            epoch_bias_clw = 0.0; epoch_bias_cli = 0.0
            epoch_bias_lev_tot = 0.0; epoch_bias_perlev= 0.0
            epoch_mae_lev_clw = 0.0; epoch_mae_lev_cli = 0.0
            epoch_rmse_perlev = 0.0 
            t_comp =0 
            t0_it = time.time()
            j = 0; k = 0; k2=2    
            
            if cfg.optimizer == "adamwschedulefree":
                if self.train:
                    optim.train()
                else:
                    optim.eval()
            
            if cfg.autoregressive:
                preds_lay = []; preds_sfc = []
                targets_lay = []; targets_sfc = [] 
                x_sfc = [];  x_lay_raw = []
                yto_lay = []; yto_sfc = []
                rnn1_mem = torch.zeros(self.batch_size*ensemble_size, model.nlev, model.nh_mem, device=device)
                loss_update_start_index = 60
            else:
                loss_update_start_index = 0
                
            for i,data in enumerate(self.loader):
                # print("shape mem 2 {}".format(model.rnn1_mem.shape))
                x_lay_chk, x_sfc_chk, targets_lay_chk, targets_sfc_chk, x_lay_raw_chk  = data
    
                # print(" x lay raw 0,50::", x_lay_raw_chk[0,50:,0])
                x_lay_chk       = x_lay_chk.to(device)
                x_lay_raw_chk   = x_lay_raw_chk.to(device)
                x_sfc_chk       = x_sfc_chk.to(device)
                targets_sfc_chk = targets_sfc_chk.to(device)
                targets_lay_chk = targets_lay_chk.to(device)
                
                x_lay_chk       = torch.split(x_lay_chk, self.batch_size)
                x_lay_raw_chk   = torch.split(x_lay_raw_chk, self.batch_size)
                x_sfc_chk       = torch.split(x_sfc_chk, self.batch_size)
                targets_sfc_chk = torch.split(targets_sfc_chk, self.batch_size)
                targets_lay_chk = torch.split(targets_lay_chk, self.batch_size)
                
                # to speed-up IO, we loaded chk=many batches, which now need to be divided into batches
                # each batch is one time step
                for ichunk in range(len(x_lay_chk)):
                    x_lay0 = x_lay_chk[ichunk]
                    x_lay_raw0 = x_lay_raw_chk[ichunk]
                    x_sfc0 = x_sfc_chk[ichunk]
                    target_lay0 = targets_lay_chk[ichunk]
                    target_sfc0 = targets_sfc_chk[ichunk]
                        
                    tcomp0= time.time()               
                    
                    with torch.autocast(device_type=device.type, dtype=self.dtype, enabled=cfg.mp_autocast):
                        if cfg.autoregressive:
                            preds_lay0, preds_sfc0, rnn1_mem = self.model(x_lay0, x_sfc0, rnn1_mem)
                        else:
                            preds_lay0, preds_sfc0 = self.model(x_lay0, x_sfc0)
    
                    if cfg.autoregressive:
                        # In the autoregressive training case are gathering many time steps before computing loss
                        preds_lay.append(preds_lay0); preds_sfc.append(preds_sfc0)
                        targets_lay.append(target_lay0); targets_sfc.append(target_sfc0)
                        x_sfc.append(x_sfc0)
                        x_lay_raw.append(x_lay_raw0)
                    else:
                        preds_lay = preds_lay0; preds_sfc = preds_sfc0
                        targets_lay = target_lay0; targets_sfc = target_sfc0
                        x_lay_raw = x_lay_raw0
                        x_sfc = x_sfc0
                        
                    if (not cfg.autoregressive) or (cfg.autoregressive and (j+1) % timesteps==0):
                
                        if cfg.autoregressive:
                            preds_lay   = torch.cat(preds_lay)
                            preds_sfc   = torch.cat(preds_sfc)
                            targets_lay = torch.cat(targets_lay)
                            targets_sfc = torch.cat(targets_sfc)
                            x_sfc       = torch.cat(x_sfc)
                            x_lay_raw   = torch.cat(x_lay_raw)
                            
                        with torch.autocast(device_type=device.type, dtype=self.dtype, enabled=cfg.mp_autocast):
                            
                            loss = loss_fn(targets_lay, targets_sfc, preds_lay, preds_sfc)
                            if cfg.loss_fn_type == "CRPS":
                                loss, det_skill, ens_var = loss 
                            else:
                                huber, mse, mae = loss 
                                if cfg.loss_fn_type == "huber":
                                    loss = huber
                                else:
                                    loss = mse
                                
                            if use_ensemble:
                                # print("preds shape", preds_lay)
                                # use only first member from here on 
                                preds_lay = torch.reshape(preds_lay, (timesteps, 2, self.batch_size, nlev, ny))
                                preds_lay = torch.reshape(preds_lay[:,0,:,:], shape=targets_lay.shape)
                                preds_sfc = torch.reshape(preds_sfc, (timesteps, 2, self.batch_size, ny_sfc))
                                preds_sfc = torch.reshape(preds_sfc[:,0,:], shape=targets_sfc.shape)
                                              
                            if use_mp_constraint:
                                ypo_lay, ypo_sfc = model.pp_mp(preds_lay, preds_sfc, x_lay_raw)
                                with torch.no_grad(): 
                                    yto_lay, yto_sfc = model.pp_mp(targets_lay, targets_sfc, x_lay_raw )
                                # ypo_lay, ypo_sfc, yto_lay, yto_sfc = model.pp_mp(preds_lay, preds_sfc, targets_lay, targets_sfc, x_lay_raw )
                                # if i>10: print ("yto lay true lev 35  dqliq {:.2e} ".format(ypo_lay[200,35,2].item()))
                                # if i>10: print ("yto lay pp-true lev 35  dqliq {:.2e} ".format(ypo_lay[200,35,2].item()))

                            else:
                                ypo_lay, ypo_sfc = model.postprocessing(preds_lay, preds_sfc)
                                yto_lay, yto_sfc = model.postprocessing(targets_lay, targets_sfc)
                                
                            with torch.no_grad():
                                x_sfc = x_sfc*model.xdiv_sca + model.xmean_sca 
                                surf_pres_denorm = x_sfc[:,0:1]
                                lhf = x_sfc[:,2] 

                            h_con = metric_h_con(yto_lay, ypo_lay, surf_pres_denorm)

                            water_con_p     = metric_water_con(ypo_lay, ypo_sfc, surf_pres_denorm, lhf)
                            water_con_t     = metric_water_con(yto_lay, yto_sfc, surf_pres_denorm, lhf)
                            water_con       = torch.mean(torch.square(water_con_p - water_con_t))
                            del water_con_p, water_con_t
                            
                            if cfg.use_energy_loss: 
                                loss = loss + cfg._lambda*h_con

                            if cfg.use_water_loss:
                                loss = loss + cfg._alpha * water_con
                                
                        if self.train:
                            if cfg.use_scaler:
                                scaler.scale(loss).backward()
                                scaler.step(optim)
                                scaler.update()
                            else:
                                loss.backward()       
                                optim.step()
                
                            optim.zero_grad()
                            loss = loss.detach()
                            if cfg.loss_fn_type == "CRPS":
                                det_skill = det_skill.detach(); ens_var = ens_var.detach()
                            else: 
                                huber = huber.detach(); mse = mse.detach(); mae = mae.detach()
                            h_con = h_con.detach() 
                            water_con = water_con.detach()
                            
                        ypo_lay = ypo_lay.detach(); ypo_sfc = ypo_sfc.detach()
                        yto_lay = yto_lay.detach(); yto_sfc = yto_sfc.detach()
                            
                        running_loss    += loss.item()
                        running_energy  += h_con.item()
                        running_water   += water_con.item()
                        if cfg.loss_fn_type == "CRPS": running_var += ens_var.item() 
                        #mae             = metrics.mean_absolute_error(targets_lay, preds_lay)
                        if j>loss_update_start_index:
                            with torch.no_grad():
                                epoch_loss      += loss.item()
                                if cfg.loss_fn_type =="CRPS":
                                    huber, mse, mae       = det_metrics(targets_lay, targets_sfc, preds_lay, preds_sfc)

                                epoch_huber += huber.item()
                                epoch_mse += mse.item()
                                epoch_mae += mae.item()
                                # epoch_mae       += metrics.mean_absolute_error(targets_lay, preds_lay)
                            
                                epoch_hcon  += h_con.item()
                                epoch_wcon  += water_con.item()
                                
                                if cfg.loss_fn_type == "CRPS":
                                    epoch_ens_var += ens_var.item()
                                    epoch_det_skill += det_skill.item()
                                    epoch_spreadskill += ens_var.item() / det_skill.item()

                                # print("shape ypo", ypo_lay.shape, "yto", yto_lay.shape)
                                # water_con       = metric_water_con(ypo_lay, ypo_sfc, surf_pres_denorm, lhf)
                                # # print("true:")
                                # water_con_true  = metric_water_con(yto_lay, yto_sfc, surf_pres_denorm, lhf)
                                # water_con_loss = torch.mean(torch.square(water_con - water_con_true))
                                # print("water con loss", water_con_loss)
                                # print("pred con", water_con, "true con", water_con_true)

                                biases_lev, biases_sfc = metrics.compute_absolute_biases(yto_lay, yto_sfc, ypo_lay, ypo_sfc)
                                epoch_bias_lev += np.mean(biases_lev)
                                epoch_bias_heating += biases_lev[0]
                                epoch_bias_clw += biases_lev[2]
                                epoch_bias_cli += biases_lev[3]
                                epoch_bias_sfc += np.mean(biases_sfc)

                                biases_nolev, biases_perlev = metrics.compute_biases(yto_lay, ypo_lay)
                                epoch_bias_lev_tot += np.mean(biases_nolev)
                                epoch_bias_perlev += biases_perlev

                                epoch_rmse_perlev += metrics.rmse(yto_lay, ypo_lay)


                                self.metric_R2.update(ypo_lay.reshape((-1,ny_pp)), yto_lay.reshape((-1,ny_pp)))
                                       
                                r2_np = np.corrcoef((ypo_sfc.reshape(-1,ny_sfc)[:,3].cpu().numpy(),yto_sfc.reshape(-1,ny_sfc)[:,3].detach().cpu().numpy()))[0,1]
                                epoch_R2precc += r2_np
                                #print("R2 numpy", r2_np, "R2 torch", self.metric_R2_precc(ypo_sfc[:,3:4], yto_sfc[:,3:4]) )
    
                                ypo_lay = ypo_lay.reshape(-1,nlev,ny_pp).cpu().numpy()
                                yto_lay = yto_lay.reshape(-1,nlev,ny_pp).cpu().numpy()
    
                                epoch_r2_lev += metrics.corrcoeff_pairs_batchfirst(ypo_lay, yto_lay)**2

                                epoch_mae_lev_clw +=  np.nanmean(np.abs(ypo_lay[:,:,2] - yto_lay[:,:,2]),axis=0)
                                epoch_mae_lev_cli +=  np.nanmean(np.abs(ypo_lay[:,:,3] - yto_lay[:,:,3]),axis=0)
                               # if track_ks:
                               #     if (j+1) % max(timesteps*4,12)==0:
                               #         epoch_ks += kolmogorov_smirnov(yto,ypo).item()
                               #         k2 += 1
                                k += 1
                        if cfg.autoregressive:
                            preds_lay = []; preds_sfc = []
                            targets_lay = []; targets_sfc = [] 
                            x_lay_raw = []; yto_lay = []; yto_sfc = []
                            x_sfc = []
                            rnn1_mem = rnn1_mem.detach()
                            
                    t_comp += time.time() - tcomp0
                    # # print statistics 
                    if j % report_freq == (report_freq-1): # print every 200 minibatches
                        elaps = time.time() - t0_it
                        fac = report_freq/timesteps
                        running_loss = running_loss / fac
                        running_energy = running_energy / fac
                        running_water = running_water / fac

                        r2raw = self.metric_R2.compute()
    
                        #r2_np = np.corrcoef((ypo_sfc.reshape(-1,ny_sfc)[:,3].detach().cpu().numpy(),yto_sfc.reshape(-1,ny_sfc)[:,3].detach().cpu().numpy()))[0,1]
                        if cfg.loss_fn_type == "CRPS": 
                            running_var = running_var / fac

                            print("[{:d}, {:d}] Loss: {:.2e} var {:.2e} h-con: {:.2e}  w-con: {:.2e}  runR2: {:.2f}, took {:.1f}s (comp. {:.1f})" .format(epoch + 1, 
                                                            j+1, running_loss,running_var, running_energy,running_water, r2raw, elaps, t_comp), flush=True)
                            running_var = 0.0
                        else:
                            print("[{:d}, {:d}] Loss: {:.2e}  h-con: {:.2e}  w-con: {:.2e}  runR2: {:.2f},  elapsed {:.1f}s (compute {:.1f})" .format(epoch + 1, 
                                                            j+1, running_loss,running_energy,running_water, r2raw, elaps, t_comp), flush=True)
                        running_loss = 0.0
                        running_energy = 0.0; running_water=0.0
                        t0_it = time.time()
                        t_comp = 0
                    j += 1
                    
                del x_lay_chk, x_sfc_chk, targets_lay_chk, targets_sfc_chk, x_lay_raw_chk
                    
            self.metrics['loss'] =  epoch_loss / k
            self.metrics['mean_squared_error'] = epoch_mse / k
            self.metrics['huber'] = epoch_huber / k

            if cfg.loss_fn_type == "CRPS": 
                self.metrics['ens_var'] =  epoch_ens_var / k
                self.metrics['det_skill'] =  epoch_det_skill / k
                self.metrics['spread_skill_ratio'] =  epoch_spreadskill / k

            self.metrics["h_conservation"] =  epoch_hcon / k
            self.metrics["water_conservation"] =  epoch_wcon / k

            self.metrics["bias_lev"] = epoch_bias_lev / k 
            self.metrics["bias_lev_noabs"] = epoch_bias_lev_tot / k 

            self.metrics["bias_sfc"] = epoch_bias_sfc / k 
            self.metrics["bias_heating"] = epoch_bias_heating / k 
            self.metrics["bias_cldliq"] = epoch_bias_clw / k 
            self.metrics["bias_cldice"] = epoch_bias_cli / k 

            self.metrics["bias_perlev"] = epoch_bias_perlev / k 
            self.metrics["rmse_perlev"] = epoch_rmse_perlev / k 

            self.metrics['mean_absolute_error'] = epoch_mae / k
            #self.metrics['ks'] =  epoch_ks / k2
            self.metrics['R2'] = self.metric_R2.compute()
            
            
            self.metrics['R2_lev'] = epoch_r2_lev / k
            self.metrics['R2_heating'] = epoch_r2_lev[:,0].mean() / k
            # self.metrics['R2_moistening'] =  epoch_r2_lev[:,1].mean() / k
            R2_moistening = epoch_r2_lev[:,1].mean() / k
            R2 = epoch_r2_lev[:,2] / k
            R2[np.isnan(R2)] = 0.0; R2[np.isinf(R2)] = 0.0
            # self.metrics['R2_lev_clw'] = R2
            self.metrics['R2_clw'] = np.mean(R2)
            R2 = epoch_r2_lev[:,3] / k
            R2[np.isnan(R2)] = 0.0; R2[np.isinf(R2)] = 0.0
            # self.metrics['R2_lev_cli'] = R2
            self.metrics['R2_cli'] = np.mean(R2)

            #self.metrics['R2_precc'] = self.metric_R2_precc.compute()
            self.metrics['R2_precc'] = epoch_R2precc / k
        
            self.metrics['mae_clw'] = np.nanmean(epoch_mae_lev_clw / k)
            self.metrics['mae_cli'] = np.nanmean(epoch_mae_lev_cli / k)
            
            self.metric_R2.reset() 
            #self.metric_R2_heating.reset(); self.metric_R2_precc.reset()
            # if self.autoregressive:
            #     # self.model.reset_states()
            #     # model.rnn1_mem = torch.randn_like(model.rnn1_mem)
            #     # model.rnn1_mem = torch.randn(self.batch_size, nlev, model.nh_mem, device=device)
            #     model.rnn1_mem = torch.zeros(self.batch_size, nlev, model.nh_mem, device=device)
    
            datatype = "TRAIN" if self.train else "VAL"
            print('Epoch {} {} loss: {:.2e}  MSE: {:.2e}  h-con:  {:.2e}   R2: {:.2f}  R2-dT/dt: {:.2f}   R2-dq/dt: {:.2f}   R2-precc: {:.3f}'.format(epoch+1, datatype, 
                                                                self.metrics['loss'], 
                                                                self.metrics['mean_squared_error'], 
                                                                self.metrics['h_conservation'],
                                                                self.metrics['R2'],
                                                                self.metrics['R2_heating'],
                                                                R2_moistening, # self.metrics['R2_moistening'],                                                              
                                                                self.metrics['R2_precc'] ))
            
            del loss, h_con, water_con
            if cfg.autoregressive:
                del rnn1_mem
            if cuda: 
                torch.cuda.empty_cache()
            gc.collect()
    
    train_runner = model_train_eval(train_loader, model, batch_size_tr,  train=True)
    val_runner = model_train_eval(val_loader, model, batch_size_val, train=False)
    
    inpstr = "v5" if cfg.v4_to_v5_inputs else "v4"
    outpstr = "v5" if use_mp_constraint else "v4"
    SAVE_PATH =  'saved_models/{}-{}_lr{}.neur{}-{}_x{}_y{}_num{}.pt'.format(cfg.model_type,
                                                                     cfg.memory, cfg.lr, 
                                                                     cfg.nneur[0], cfg.nneur[1], 
                                                                     inpstr, outpstr,
                                                                     model_num)


    save_file_torch = "saved_models/" + SAVE_PATH.split("/")[1].split(".pt")[0] + "_script.pt"
    # best_val_loss = np.inf
    best_val_loss = 0.0
    
    tsteps_old = 1
    new_lr = cfg.lr
    for epoch in range(cfg.num_epochs):
        t0 = time.time()
        
        if cfg.timestep_scheduling:
            timesteps=timestep_schedule[epoch]            
        else:
            timesteps=timewindow_default
            
        print("Epoch {} Training rollout timesteps: {} ".format(epoch+1, timesteps))

        if cfg.timestepped_optimizer and (timesteps==(tsteps_old+1)):
            # print("Timestepped optimizer turned on, doubling learning rate upon increased time window")
            # for g in optimizer.param_groups:
            #     g['lr'] = 2*g['lr']
            new_lr = (timesteps/(tsteps_old)) * new_lr
            print("Timestepped optimizer turned on,setting rate to {}".format(new_lr))

            if cfg.optimizer == "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr = new_lr)
            elif cfg.optimizer == "adamw":
                optimizer = torch.optim.AdamW(model.parameters(), lr = new_lr)
            elif cfg.optimizer == "adamwschedulefree":
                import schedulefree
                optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=new_lr)
            elif cfg.optimizer == "soap":
                from soap import SOAP
                optimizer = SOAP(model.parameters(), lr = new_lr, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
            else:
                raise NotImplementedError()

        tsteps_old = timesteps

        train_runner.eval_one_epoch(optimizer, epoch, timesteps)

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
        
        if epoch%2:
            print("VALIDATION..")
            val_runner.eval_one_epoch(optimizer, epoch, timesteps)
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
    
            # val_loss = val_runner.metrics["loss"]
            val_loss = val_runner.metrics["R2"]

            # MODEL CHECKPOINT IF VALIDATION LOSS IMPROVED
            # if cfg.save_model and val_loss < best_val_loss:
            if cfg.save_model and val_loss > best_val_loss:
                print("New best validation result obtained, saving model to", SAVE_PATH)
                if cfg.loss_fn_type == "CRPS":
                    model.use_ensemble=False
                model = model.to("cpu")
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                            }, SAVE_PATH)  
                scripted_model = torch . jit . script ( model )
                scripted_model = scripted_model.eval()
                scripted_model.save(save_file_torch)
                best_val_loss = val_loss 
                if cfg.loss_fn_type == "CRPS":
                    model.use_ensemble=True
                model = model.to(device)

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
                plt.savefig(os.path.join('saved_models/val_eval/', '{}-{}_lr{}.neur{}-{}_x{}_y{}_num{}_val_R2.pdf'.format(cfg.model_type,
                                                                                cfg.memory, cfg.lr, cfg.nneur[0], cfg.nneur[1], 
                                                                                inpstr, outpstr, model_num)))
                plt.clf()
                bias = val_runner.metrics["bias_perlev"]
                fig, axs = plt.subplots(ncols=1, nrows=6, figsize=(7.0, 12.0)) #layout="constrained")
                for i in range(6):
                    axs[i].plot(np.arange(60), bias[:,i]); 
                    axs[i].set_title(labels[i])
                    axs[i].set_xlim(0,60)
                    axs[i].axvspan(0, 30, facecolor='0.2', alpha=0.2)

                fig.subplots_adjust(hspace=0.6)                                                     
                plt.savefig(os.path.join('saved_models/val_eval/', '{}-{}_lr{}.neur{}-{}_x{}_y{}_num{}_val_bias.pdf'.format(cfg.model_type,
                                                                                cfg.memory, cfg.lr, cfg.nneur[0], cfg.nneur[1], 
                                                                                inpstr, outpstr, model_num)))

                plt.clf()
                rmse = val_runner.metrics["rmse_perlev"]
                fig, axs = plt.subplots(ncols=1, nrows=6, figsize=(7.0, 12.0)) #layout="constrained")
                for i in range(6):
                    axs[i].plot(np.arange(60), rmse[:,i]); 
                    axs[i].set_title(labels[i])
                    axs[i].set_xlim(0,60)
                    axs[i].axvspan(0, 30, facecolor='0.2', alpha=0.2)

                fig.subplots_adjust(hspace=0.6)                                                     
                plt.savefig(os.path.join('saved_models/val_eval/', '{}-{}_lr{}.neur{}-{}_x{}_y{}_num{}_val_rmse.pdf'.format(cfg.model_type,
                                                                                cfg.memory, cfg.lr, cfg.nneur[0], cfg.nneur[1], 
                                                                                inpstr, outpstr, model_num)))



        print('Epoch {}/{} complete, took {:.2f} seconds, autoreg window was {}'.format(epoch+1,cfg.num_epochs,time.time() - t0,timesteps))
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush=True)
    
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
