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
from models import RNN_autoreg, MyRNN, LSTM_autoreg_torchscript, SpaceStateModel, LSTM_torchscript
from utils import generator_xy, BatchSampler
# from metrics import get_energy_metric, get_hybrid_loss, my_mse_flatten
import metrics as metrics
from torchmetrics.regression import R2Score
import wandb
from omegaconf import DictConfig
import hydra

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
        
        if (not cfg.output_norm_per_level):
            raise NotImplementedError()
    
    if cfg.memory=="None":
        cfg.autoregressive=False
        cfg.use_intermediate_mlp = False
    else:
        cfg.shuffle_data = False 
        
        
    if cuda:
        print(torch.cuda.get_device_name(0))
        cfg.mp_autocast = True 
        print(torch.cuda.is_bf16_supported())
        # if torch.cuda.is_bf16_supported(): 
        #     dtype=torch.bfloat16 
        #     use_scaler = False
        # else:
        #     dtype=torch.float16
        #     use_scaler = True 
        dtype=torch.float16
        cfg.use_scaler = True 
    else:
        dtype=torch.float32
        cfg.mp_autocast = False
        cfg.use_scaler = False
        
        
    # --------------------------------------
    
    grid_info = xr.open_dataset(grid_path)
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
    print("ns", ns, "nloc", nloc, "nlev", nlev,  "nx", nx, "nx_sfc", nx_sfc, "ny", ny, "ny_sfc", ny, flush=True)
    
    if cfg.v4_to_v5_inputs:
        vars_2D_inp.remove('state_q0002') # liq
        vars_2D_inp.remove('state_q0003')
        vars_2D_inp.insert(2,"state_qn")
        vars_2D_inp.insert(3,"liq_partition")
    
    if cfg.mp_mode>0:
        vars_2D_outp.remove('ptend_q0002')
        vars_2D_outp.remove('ptend_q0003')
        vars_2D_outp.insert(2,"ptend_qn")
    
    
    yscale_sca = output_scale[vars_1D_outp].to_dataarray(dim='features', name='outputs_sca').transpose().values
    
    if cfg.output_norm_per_level:
        yscale_lev = output_scale[vars_2D_outp].to_dataarray(dim='features', name='outputs_lev').transpose().values
    else:
        yscale_lev = np.repeat(np.array([2.3405453e+04, 2.3265182e+08, 1.4898973e+08, 6.4926711e+04,
                7.8328773e+04], dtype=np.float32).reshape((1,-1)),nlev,axis=0)
        
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
    
    weights=None 
    
    if cfg.mp_mode==2:
        liq_frac_scale = np.ones(nlev,dtype=np.float32).reshape(nlev,1)
        liq_frac_scale[:] = 2.4
        # liq_frac_scale = np.array([1.0000000e+04, 1.0000000e+04, 1.0000000e+04, 1.0000000e+04,
        #        1.0000000e+04, 1.0000000e+04, 1.0000000e+04, 1.0000000e+04,
        #        1.0000000e+04, 1.0000000e+04, 1.0000000e+04, 1.0000000e+04,
        #        1.0000000e+04, 5.7569299e+02, 1.0000000e+04, 1.0000000e+04,
        #        3.7180832e+02, 3.7180832e+02, 1.6628116e+02, 1.2394203e+02,
        #        4.6134792e+01, 2.2092237e+01, 7.7577863e+00, 5.6359839e+00,
        #        5.4808850e+00, 6.4823780e+00, 6.6720495e+00, 6.2344809e+00,
        #        6.8770485e+00, 8.4653273e+00, 1.1100765e+01, 1.2499543e+01,
        #        9.9253035e+00, 7.3044314e+00, 5.6595097e+00, 4.5972705e+00,
        #        3.8331578e+00, 3.2858577e+00, 2.9477441e+00, 2.7470703e+00,
        #        2.6018870e+00, 2.4938695e+00, 2.3889098e+00, 2.3040833e+00,
        #        2.2352726e+00, 2.2033858e+00, 2.1932576e+00, 2.1929729e+00,
        #        2.1951184e+00, 2.1974444e+00, 2.1985419e+00, 2.1974182e+00,
        #        2.1818981e+00, 2.1499007e+00, 2.1246915e+00, 2.1189482e+00,
        #        2.1840427e+00, 2.3258853e+00, 2.5237758e+00, 3.0374336e+00],
        #       dtype=np.float32).reshape(nlev,1)
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
    
    if cfg.snowhice_fix:
        xmean_sca[15] = 0.0 
        xdiv_sca[15] = 1.0 
        
    xcoeff_sca = np.stack((xmean_sca, xdiv_sca))
    xcoeff_lev = np.stack((xmean_lev, xdiv_lev))
    xcoeffs = np.float32(xcoeff_lev), np.float32(xcoeff_sca)
    
    if cfg.add_refpres:
        nx = nx + 1
        
    # if use_mp_constraint:
    if cfg.mp_mode==1:
        ny_pp = ny # The 6 original outputs will be after postprocessing
        ny = ny - 1 # The model itself only has 5 outputs (total cloud water)
    else:
        ny_pp = ny 
    # ny_pp = ny 
    
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
                        nx = nx, nx_sfc=nx_sfc, 
                        ny = ny, ny_sfc=ny_sfc, 
                        nneur=cfg.nneur, 
                        use_initial_mlp = cfg.use_initial_mlp,
                        use_intermediate_mlp = cfg.use_intermediate_mlp,
                        add_pres = cfg.add_pres,
                        output_prune = cfg.output_prune,
                        use_memory = cfg.autoregressive,
                        separate_radiation = cfg.separate_radiation,
                        use_third_rnn = use_third_rnn)
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
    else:
        model = SpaceStateModel(hyam, hybm, 
                    out_scale = yscale_lev,
                    out_sfc_scale = yscale_sca,  
                    nlay=60, nx = nx, nx_sfc=nx_sfc, 
                    ny = ny, ny_sfc=ny_sfc, 
                    nneur = cfg.nneur, model_type = cfg.model_type, 
                    use_initial_mlp = cfg.use_initial_mlp, add_pres=cfg.add_pres,  concat=cfg.concat)
    
    model = model.to(device)
    
    if cfg.autoregressive:
        # model.rnn1_mem = torch.randn(nloc, nlay, model.nh_mem, device=device)
        rnn1_mem = torch.zeros(nloc, model.nlay, model.nh_mem, device=device)
    
    infostr = summary(model)
    num_params = infostr.total_params
    # print(infostr)
    
    if cfg.use_scaler:
        # scaler = torch.amp.GradScaler(autocast = True)
        scaler = torch.amp.GradScaler(device.type)
        
    # scripted_model = torch . jit . script ( model )
    
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
                    ycoeffs=ycoeffs, xcoeffs=xcoeffs, no_multiprocessing=no_multiprocessing)
    
    train_batch_sampler = BatchSampler(cfg.chunksize_train, # samples per chunk 
                                       # num_samples=train_data.ntimesteps*nloc, shuffle=shuffle_data)
                                       num_samples = train_data.ntimesteps, shuffle = cfg.shuffle_data)
    
    train_loader = DataLoader(dataset = train_data, num_workers = cfg.num_workers, 
                              sampler = train_batch_sampler, 
                              batch_size=None, batch_sampler=None, 
                              prefetch_factor = prefetch_factor, 
                              pin_memory=pin, persistent_workers=persistent)
    
        
    
    val_data = generator_xy(val_data_path, cache=cfg.cache, add_refpres = cfg.add_refpres, 
                    remove_past_sfc_inputs = cfg.remove_past_sfc_inputs, mp_mode = cfg.mp_mode,
                    v4_to_v5_inputs = cfg.v4_to_v5_inputs, rh_prune = cfg.rh_prune, 
                    qinput_prune = cfg.qinput_prune, output_prune = cfg.output_prune,
                    ycoeffs=ycoeffs, xcoeffs=xcoeffs, no_multiprocessing=no_multiprocessing)
    
    val_batch_sampler = BatchSampler(cfg.chunksize_val, 
                                       # num_samples=val_data.ntimesteps*nloc_val, shuffle=shuffle_data)
                                       num_samples=val_data.ntimesteps, shuffle = cfg.shuffle_data)
    
    val_loader = DataLoader(dataset=val_data, num_workers = cfg.num_workers, 
                              sampler = val_batch_sampler, 
                              batch_size=None, batch_sampler=None, 
                              prefetch_factor = prefetch_factor, 
                              pin_memory=pin, persistent_workers=persistent)
    
    nloc_val = batch_size_val = val_data.nloc
    
    metric_h_con = metrics.get_energy_metric(hyai, hybi)
    mse = metrics.get_mse_flatten(weights)
    
    if cfg.loss_fn_type == "mse":
        loss_fn = mse
    elif cfg.loss_fn_type == "huber":
        loss_fn = metrics.get_huber_flatten(weights)
    else:
        raise NotImplementedError()
    
    hybrid_loss = metrics.get_hybrid_loss(torch.tensor(cfg._lambda))
    
    if cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = cfg.lr)
    elif cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr = cfg.lr)
    elif cfg.optimizer == "adamwschedulefree":
        import schedulefree
        optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=cfg.lr)
    else:
        raise NotImplementedError()

    # if not cfg.autoregressive:
    #     timewindow = 1
    #     timestep_scheduling=False
    #     timestep_schedule = np.arange(1000)
    #     timestep_schedule[:] = timewindow
    # else:
    #     timewindow = 3
    #     timestep_scheduling=True
    #     timestep_schedule = np.arange(1000)
    #     timestep_schedule[:] = timewindow
    
    #     if timestep_scheduling:
    #         timestep_schedule[0:3] = 1
    #         timestep_schedule[3:4] = timewindow-1
    #         timestep_schedule[4:] = timewindow
    #         timestep_schedule[5:] = timewindow+1
    #         timestep_schedule[6:] = timewindow+2
            
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
        # def __init__(self, conf, dataloader, model, autoregressive, loss_fn, dtype, batch_size = 384, train=True):
        def __init__(self, dataloader, model, batch_size = 384, train=True):
 
            super().__init__()
            # self.cfg = conf
            self.loader = dataloader
            self.dtype = dtype
            self.train = train
            self.report_freq = 800
            self.batch_size = batch_size
            self.model = model 
            # self.autoregressive = autoregressive
            # self.loss_fn = loss_fn
            # self.metric_R2 =  R2Score(num_outputs=ny_pp).to(device) 
            self.metric_R2 =  R2Score().to(device) 

            self.metrics = {}
            
        def eval_one_epoch(self, epoch, timewindow=1):
            report_freq = self.report_freq
            running_loss = 0.0; running_energy = 0.0
            epoch_loss = 0.0
            epoch_mse = 0.0; epoch_mae = 0.0
            epoch_R2precc = 0.0
            epoch_hcon = 0.0
            epoch_r2_lev = 0.0
            epoch_bias_lev = 0.0; epoch_bias_sfc = 0.0; epoch_bias_heating = 0.0
            epoch_bias_clw = 0.0; epoch_bias_cli = 0.0
            epoch_bias_lev_tot = 0.0 
            t_comp =0 
            t0_it = time.time()
            j = 0; k = 0; k2=2    
            
            if cfg.optimizer == "adamwschedulefree":
                if self.train:
                    optimizer.train()
                else:
                    optimizer.eval()
            
            if cfg.autoregressive:
                preds_lay = []; preds_sfc = []
                targets_lay = []; targets_sfc = [] 
                surf_pres = [];  x_lay_raw = []
                yto_lay = []; yto_sfc = []
                # model.rnn1_mem = torch.randn(self.batch_size, nlay, model.nh_mem, device=device)
                # model.rnn1_mem = torch.zeros(self.batch_size, nlay, model.nh_mem, device=device)
                rnn1_mem = torch.zeros(self.batch_size, model.nlay, model.nh_mem, device=device)
                loss_update_start_index = 60
            else:
                loss_update_start_index = 0
                
            for i,data in enumerate(self.loader):
                # print("shape mem 2 {}".format(model.rnn1_mem.shape))
    
                # x_lay_chk, x_sfc_chk, targets_lay_chk, targets_sfc_chk, ytos_lay_chk, ytos_sfc_chk, x_lay_raw_chk  = data
                # x_lay_chk, x_sfc_chk, targets_lay_chk, targets_sfc_chk, x_lay_raw_chk, ytos_lay_chk, ytos_sfc_chk  = data
                x_lay_chk, x_sfc_chk, targets_lay_chk, targets_sfc_chk, x_lay_raw_chk  = data
    
                # print(" x lay raw 0,50::", x_lay_raw_chk[0,50:,0])
                x_lay_chk       = x_lay_chk.to(device)
                x_lay_raw_chk   = x_lay_raw_chk.to(device)
                x_sfc_chk       = x_sfc_chk.to(device)
                targets_sfc_chk = targets_sfc_chk.to(device)
                targets_lay_chk = targets_lay_chk.to(device)
                # ytos_lay_chk    = ytos_lay_chk.to(device) # these are the raw (unnormalized) full outputs
                # ytos_sfc_chk    = ytos_sfc_chk.to(device)
                
                x_lay_chk       = torch.split(x_lay_chk, self.batch_size)
                x_lay_raw_chk   = torch.split(x_lay_raw_chk, self.batch_size)
                x_sfc_chk       = torch.split(x_sfc_chk, self.batch_size)
                targets_sfc_chk = torch.split(targets_sfc_chk, self.batch_size)
                targets_lay_chk = torch.split(targets_lay_chk, self.batch_size)
                # ytos_lay_chk    = torch.split(ytos_lay_chk, self.batch_size)
                # ytos_sfc_chk    = torch.split(ytos_sfc_chk, self.batch_size)
                
                # to speed-up IO, we loaded chk=many batches, which now need to be divided into batches
                # each batch is one time step
                for ichunk in range(len(x_lay_chk)):
                    x_lay0 = x_lay_chk[ichunk]
                    x_lay_raw0 = x_lay_raw_chk[ichunk]
                    x_sfc0 = x_sfc_chk[ichunk]; sp0 = x_sfc0[:,0:1] 
                    target_lay0 = targets_lay_chk[ichunk]
                    target_sfc0 = targets_sfc_chk[ichunk]
    
                    # yto_lay0 = ytos_lay_chk[ichunk]
                    # yto_sfc0 = ytos_sfc_chk[ichunk]
                    
                    tcomp0= time.time()
                        
                    with torch.autocast(device_type=device.type, dtype=self.dtype, enabled=cfg.mp_autocast):
                        if cfg.autoregressive:
                            preds_lay0, preds_sfc0, rnn1_mem = self.model(x_lay0, x_sfc0, rnn1_mem)
                        else:
                            preds_lay0, preds_sfc0 = self.model(x_lay0, x_sfc0)
    
                    if cfg.autoregressive:
                        # In the autoregressive training case are gathering many time steps before computing loss
                        preds_lay.append(preds_lay0)
                        preds_sfc.append(preds_sfc0)
                        targets_lay.append(target_lay0)
                        targets_sfc.append(target_sfc0)
                        surf_pres.append(sp0) 
                        x_lay_raw.append(x_lay_raw0)
                        # yto_lay.append(yto_lay0)
                        # yto_sfc.append(yto_sfc0)                    
                    else:
                        preds_lay = preds_lay0
                        preds_sfc = preds_sfc0
                        targets_lay = target_lay0
                        targets_sfc = target_sfc0
                        surf_pres = sp0
                        x_lay_raw = x_lay_raw0
                        # yto_lay = yto_lay0
                        # yto_sfc = yto_sfc0
                        
                    if (not cfg.autoregressive) or (cfg.autoregressive and (j+1) % timewindow==0):
                
                        if cfg.autoregressive:
                            preds_lay   = torch.cat(preds_lay)
                            preds_sfc   = torch.cat(preds_sfc)
                            targets_lay = torch.cat(targets_lay)
                            targets_sfc = torch.cat(targets_sfc)
                            surf_pres   = torch.cat(surf_pres)
                            x_lay_raw   = torch.cat(x_lay_raw)
                            # yto_lay     = torch.cat(yto_lay)
                            # yto_sfc     = torch.cat(yto_sfc)   
                            
                        with torch.autocast(device_type=device.type, dtype=self.dtype, enabled=cfg.mp_autocast):
                            
                            main_loss = loss_fn(targets_lay, targets_sfc, preds_lay, preds_sfc)
                            
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
                            surf_pres_denorm = surf_pres*(sp_max - sp_min) + sp_mean
                            h_con = metric_h_con(yto_lay, ypo_lay, surf_pres_denorm)
                            
                            if cfg.use_energy_loss: 
                                loss = hybrid_loss(main_loss, h_con)
                            else:
                                loss = main_loss
                                                
                        if self.train:
                            if cfg.use_scaler:
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                loss.backward()       
                                optimizer.step()
                
                            optimizer.zero_grad()
                            loss = loss.detach()
                            h_con = h_con.detach() 
                            
                        ypo_lay = ypo_lay.detach(); ypo_sfc = ypo_sfc.detach()
                        yto_lay = yto_lay.detach(); yto_sfc = yto_sfc.detach()
                        
                            
                        running_loss    += loss.item()
                        running_energy  += h_con.item()
                        #mae             = metrics.mean_absolute_error(targets_lay, preds_lay)
                        if j>loss_update_start_index:
                            with torch.no_grad():
                                epoch_loss      += loss.item()
                                if cfg.loss_fn_type =="mse":
                                    epoch_mse       += main_loss.item()
                                else:
                                    epoch_mse       += mse(targets_lay, targets_sfc, preds_lay, preds_sfc)
                                #epoch_mae       += mae.item()
                            
                                epoch_hcon  += h_con.item()
                                # print("shape ypo", ypo_lay.shape, "yto", yto_lay.shape)
                                
                                biases_lev, biases_sfc = metrics.compute_absolute_biases(yto_lay, yto_sfc, ypo_lay, ypo_sfc)
                                epoch_bias_lev += np.mean(biases_lev)
                                epoch_bias_heating += biases_lev[0]
                                epoch_bias_clw += biases_lev[2]
                                epoch_bias_cli += biases_lev[3]

                                epoch_bias_sfc += np.mean(biases_sfc)

                                biases_lev = metrics.compute_biases(yto_lay, ypo_lay)
                                epoch_bias_lev_tot += np.mean(biases_lev)

                                self.metric_R2.update(ypo_lay.reshape((-1,ny_pp)), yto_lay.reshape((-1,ny_pp)))
                                       
                                r2_np = np.corrcoef((ypo_sfc.reshape(-1,ny_sfc)[:,3].cpu().numpy(),yto_sfc.reshape(-1,ny_sfc)[:,3].detach().cpu().numpy()))[0,1]
                                epoch_R2precc += r2_np
                                #print("R2 numpy", r2_np, "R2 torch", self.metric_R2_precc(ypo_sfc[:,3:4], yto_sfc[:,3:4]) )
    
                                ypo_lay = ypo_lay.reshape(-1,nlev,ny_pp).cpu().numpy()
                                yto_lay = yto_lay.reshape(-1,nlev,ny_pp).cpu().numpy()
    
                                epoch_r2_lev += metrics.corrcoeff_pairs_batchfirst(ypo_lay, yto_lay) 
                               # if track_ks:
                               #     if (j+1) % max(timewindow*4,12)==0:
                               #         epoch_ks += kolmogorov_smirnov(yto,ypo).item()
                               #         k2 += 1
                                k += 1
                        if cfg.autoregressive:
                            preds_lay = []; preds_sfc = []
                            targets_lay = []; targets_sfc = [] 
                            surf_pres = []; x_lay_raw = []
                            yto_lay = []; yto_sfc = []
                            rnn1_mem = rnn1_mem.detach()
                            
                    t_comp += time.time() - tcomp0
                    # # print statistics 
                    if j % report_freq == (report_freq-1): # print every 200 minibatches
                        elaps = time.time() - t0_it
                        running_loss = running_loss / (report_freq/timewindow)
                        running_energy = running_energy / (report_freq/timewindow)
                        
                        r2raw = self.metric_R2.compute()
                        #r2raw_prec = self.metric_R2_precc.compute()
    
                        #ypo_lay, ypo_sfc = model.postprocessing(preds_lay, preds_sfc)
                        #yto_lay, yto_sfc = model.postprocessing(targets_lay, targets_sfc) 
                        #r2_np = np.corrcoef((ypo_sfc.reshape(-1,ny_sfc)[:,3].detach().cpu().numpy(),yto_sfc.reshape(-1,ny_sfc)[:,3].detach().cpu().numpy()))[0,1]
    
                        print("[{:d}, {:d}] Loss: {:.2e}  h-con: {:.2e}  runR2: {:.2f},  elapsed {:.1f}s (compute {:.1f})" .format(epoch + 1, 
                                                        j+1, running_loss,running_energy, r2raw, elaps, t_comp), flush=True)
                        running_loss = 0.0
                        running_energy = 0.0
                        t0_it = time.time()
                        t_comp = 0
                    j += 1
                    
                del x_lay_chk, x_sfc_chk, targets_lay_chk, targets_sfc_chk, x_lay_raw_chk
                
            #if self.loader.dataset.cache and epoch==0:
            #    self.loader.dataset.cache_loaded = True
    
            self.metrics['loss'] =  epoch_loss / k
            self.metrics['mean_squared_error'] = epoch_mse / k
            self.metrics["h_conservation"] =  epoch_hcon / k
            
            self.metrics["bias_lev"] = epoch_bias_lev / k 
            self.metrics["bias_lev_noabs"] = epoch_bias_lev_tot / k 

            self.metrics["bias_sfc"] = epoch_bias_sfc / k 
            self.metrics["bias_heating"] = epoch_bias_heating / k 
            self.metrics["bias_cldliq"] = epoch_bias_clw / k 
            self.metrics["bias_cldice"] = epoch_bias_cli / k 
    
            #self.metrics['energymetric'] = epoch_energy / k
            #self.metrics['mean_absolute_error'] = epoch_mae / k
            #self.metrics['ks'] =  epoch_ks / k2
            self.metrics['R2'] = self.metric_R2.compute()
            
            
            self.metrics['R2_heating'] = epoch_r2_lev[:,0].mean() / k
            # self.metrics['R2_moistening'] =  epoch_r2_lev[:,1].mean() / k
            R2_moistening = epoch_r2_lev[:,1].mean() / k
            #self.metrics['R2_precc'] = self.metric_R2_precc.compute()
            self.metrics['R2_precc'] = epoch_R2precc / k
            
            self.metrics['R2_lev'] = epoch_r2_lev / k
            
    
            self.metric_R2.reset() 
            #self.metric_R2_heating.reset(); self.metric_R2_precc.reset()
            # if self.autoregressive:
            #     # self.model.reset_states()
            #     # model.rnn1_mem = torch.randn_like(model.rnn1_mem)
            #     # model.rnn1_mem = torch.randn(self.batch_size, nlay, model.nh_mem, device=device)
            #     model.rnn1_mem = torch.zeros(self.batch_size, nlay, model.nh_mem, device=device)
    
            datatype = "TRAIN" if self.train else "VAL"
            print('Epoch {} {} loss: {:.2e}  MSE: {:.2e}  h-con:  {:.2e}   R2: {:.2f}  R2-dT/dt: {:.2f}   R2-dq/dt: {:.2f}   R2-precc: {:.3f}'.format(epoch+1, datatype, 
                                                                self.metrics['loss'], 
                                                                self.metrics['mean_squared_error'], 
                                                                self.metrics['h_conservation'],
                                                                self.metrics['R2'],
                                                                self.metrics['R2_heating'],
                                                                R2_moistening, # self.metrics['R2_moistening'],                                                              
                                                                self.metrics['R2_precc'] ))
            
            del loss, main_loss, h_con
            if cfg.autoregressive:
                del rnn1_mem
            if cuda: 
                torch.cuda.empty_cache()
            gc.collect()
    
    # 160 160
    # autoreg, hybrid-loss, 2 years concat
    
    # train_runner = model_train_eval(train_loader, model, batch_size_tr, cfg.autoregressive, train=True)
    # val_runner = model_train_eval(val_loader, model, batch_size_val, cfg.autoregressive, train=False)
    
    train_runner = model_train_eval(train_loader, model, batch_size_tr,  train=True)
    val_runner = model_train_eval(val_loader, model, batch_size_val, train=False)
    
    SAVE_PATH =  'saved_models/{}-{}_lr{}.neur{}-{}.num{}.pt'.format(cfg.model_type,
                                                                     cfg.memory, cfg.lr, 
                                                                     cfg.nneur[0], cfg.nneur[1], model_num)


    save_file_torch = "saved_models/" + SAVE_PATH.split("/")[1].split(".pt")[0] + "_script.pt"
    # best_val_loss = np.inf
    best_val_loss = 0.0

    # w
    
    for epoch in range(cfg.num_epochs):
        t0 = time.time()
        
        if cfg.timestep_scheduling:
            timewindoww=timestep_schedule[epoch]            
        else:
            timewindoww=timewindow_default
            
        print("Epoch {} Training rollout timesteps: {} ".format(epoch+1, timewindoww))
        train_runner.eval_one_epoch(epoch, timewindoww)
        if train_runner.loader.dataset.cache:
            train_runner.loader.dataset.cache_loaded = True
        
        if cfg.use_wandb: 
            logged_metrics = train_runner.metrics.copy()
            nan_inds = np.isnan(logged_metrics['R2_lev']); num_nans = np.sum(nan_inds)
            nan_inds_moistening = np.isnan(logged_metrics['R2_lev'][:,1])
            num_nans_moist = np.sum(nan_inds_moistening)
            logged_metrics['R2_lev'][nan_inds] = 0.0
            inf_inds = np.isinf(logged_metrics['R2_lev'])
            logged_metrics['R2_lev'][inf_inds] = 0.0
            # logged_metrics['R2_lev'][:,1][nan_inds_moistening] = 0.0
            logged_metrics['num_nans'] = num_nans
            logged_metrics['num_nans_moist'] = num_nans_moist
            logged_metrics = {"train_"+k:v for k, v in logged_metrics.items()}
            logged_metrics['epoch'] = epoch
            wandb.log(logged_metrics)
        
        if epoch%2:
            print("VALIDATION..")
            val_runner.eval_one_epoch(epoch, timewindoww)
            if val_runner.loader.dataset.cache:
                val_runner.loader.dataset.cache_loaded = True

            if cfg.use_wandb: 
                logged_metrics = val_runner.metrics.copy()
                nan_inds = np.isnan(logged_metrics['R2_lev']); num_nans = np.sum(nan_inds)
                nan_inds_moistening = np.isnan(logged_metrics['R2_lev'][:,1])
                num_nans_moist = np.sum(nan_inds_moistening)
                logged_metrics['R2_lev'][nan_inds] = 0.0
                # logged_metrics['R2_lev'][:,1][nan_inds_moistening] = 0.0
                inf_inds = np.isinf(logged_metrics['R2_lev'])
                logged_metrics['R2_lev'][inf_inds] = 0.0
                logged_metrics['num_nans'] = num_nans
                logged_metrics['num_nans_moist'] = num_nans_moist
                logged_metrics = {"val_"+k:v for k, v in logged_metrics.items()}
                logged_metrics['epoch'] = epoch
                wandb.log(logged_metrics)
    
            # val_loss = val_runner.metrics["loss"]
            val_loss = val_runner.metrics["R2"]

            # MODEL CHECKPOINT IF VALIDATION LOSS IMPROVED
            # if cfg.save_model and val_loss < best_val_loss:
            if cfg.save_model and val_loss > best_val_loss:
                print("New best validation result obtained, saving model to", SAVE_PATH)
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
                  
        print('Epoch {}/{} complete, took {:.2f} seconds, autoreg window was {}'.format(epoch+1,cfg.num_epochs,time.time() - t0,timewindoww))
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush=True)
    
    if cfg.use_wandb:
        wandb.finish()
    
if __name__ == "__main__":
    main()
