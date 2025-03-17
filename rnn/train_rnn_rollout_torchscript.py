#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:27:32 2025

@author: peter
"""
import os
import sys
import inspect
import gc
import time 

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
# from numba import config, njit, threading_layer, set_num_threads
# set_num_threads(1)
# config.THREADING_LAYER = 'threadsafe'
from climsim_utils.data_utils import *
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

data_dir = "/media/peter/CrucialP1/data/ClimSim/low_res_expanded/"
grid_path = '../grid_info/ClimSim_low-res_grid-info.nc'
norm_path = '../preprocessing/normalizations/'

tr_data_path = data_dir + "train_v4_rnn_nonorm_febtofeb_y1-5_nocompress_chunk3.h5"
nloc = 1920 

tr_data_path = data_dir + "train_v4_rnn_nonorm_febtofeb_y1_3_5_7_nocompress_chunk3.h5"
nloc = 1536 

val_data_fname = "data_v4_rnn_nonorm_year8_nocompress_chunk3.h5"
val_data_dir = "/media/peter/CrucialBX500/data/ClimSim/low_res_expanded/"
nloc_val = 384 
val_data_path = val_data_dir + val_data_fname


use_val = False 
use_val = True
shuffle_data = False 
# shuffle_data = True 

output_norm_per_level = True
# output_norm_per_level = False

input_norm_per_level = True
# input_norm_per_level = False

rh_prune = True; qinput_prune=True
output_prune = True

predict_flux = False 
# predict_flux = True 

loss_fn = "mse"
# loss_fn = "huber"
use_energy_loss = False
use_energy_loss = True

mp_mode = 0   # regular 6 outputs
mp_mode = 1   # 5 outputs, pred qn, liq_ratio diagnosed (mp_constraint)
# mp_mode = 2   # 6 outputs, pred qn and liq_ratio
if mp_mode>0:
    use_mp_constraint=True
else:
    use_mp_constraint=False 
if (not output_norm_per_level) and mp_mode==0:
    raise NotImplementedError()

v4_to_v5_inputs = True
v4_to_v5_inputs = False

remove_past_sfc_inputs = True # remove pbuf_* 
preprocess_on_the_fly = False
# Predict total cloud water instead of liquid and ice, optimize for this during training,
# and use physical  microphysics constraint in postprocessing to get the liquid and ice
# use_mp_constraint = False
# use_mp_constraint = True
if use_mp_constraint:
    # output_norm_per_level = True 
    preprocess_on_the_fly = False 
    
# pred_liq_ratio = True
# if pred_liq_ratio:
#     use_mp_constraint=False

# Loss and training
num_epochs = 20
save_model = True 
if output_norm_per_level:
    _lambda = torch.tensor(1.0e-6) 
else:
    # _lambda = torch.tensor(1.0e-7) 
    _lambda = torch.tensor(1.0e-5) 

# _lambda = torch.tensor(5.0e-5) 

lr = 1e-3
use_wandb = True
# use_wandb = False

# RNN Model configuration
separate_radiation = True
separate_radiation = False

model_type = "LSTM"
# model_type = "QRNN"

autoregressive = True
memory = "Hidden"
# memory = "None"

concat = False 
add_refpres = False
use_initial_mlp = False 
use_intermediate_mlp = False
use_initial_mlp = True 
use_intermediate_mlp = True

if memory=="None":
    autoregressive=False
    use_intermediate_mlp = False
else:
    shuffle_data = False 
    
use_memory = autoregressive


add_pres = True 
ensemble_size = 1
add_stochastic_layer = False
reverse_scaling = False
# use_torchscript = True

nlay = 60 
batch_first = True 

# nneur = (64,64)
# nneur = (96,96)

nneur = (128,128)
# nneur = (192,96)

# nneur = (160, 160)
# nneur = (144,144)

if separate_radiation:
    nneur = (128, 128)
    nneur = (128, 128)
    
    
if cuda:
    print(torch.cuda.get_device_name(0))
    mp_autocast = True 
    print(torch.cuda.is_bf16_supported())
    # if torch.cuda.is_bf16_supported(): 
    #     dtype=torch.bfloat16 
    #     use_scaler = False
    # else:
    #     dtype=torch.float16
    #     use_scaler = True 
    dtype=torch.float16
    use_scaler = True 
else:
    dtype=torch.float32
    mp_autocast = False
    use_scaler = False
    
    

# --------------------------------------

grid_info = xr.open_dataset(grid_path)
input_mean = xr.open_dataset(norm_path + 'inputs/input_mean_v4_pervar.nc').astype(np.float32)
input_max = xr.open_dataset(norm_path + 'inputs/input_max_v4_pervar.nc').astype(np.float32)
input_min = xr.open_dataset(norm_path + 'inputs/input_min_v4_pervar.nc').astype(np.float32)


if v4_to_v5_inputs:
    # input_mean_ref = input_mean 
    # input_max_ref = input_max
    # input_min_ref = input_min
    input_mean = xr.open_dataset(norm_path + 'inputs/input_mean_v5_pervar.nc').astype(np.float32)
    input_max = xr.open_dataset(norm_path + 'inputs/input_max_v5_pervar.nc').astype(np.float32)
    input_min = xr.open_dataset(norm_path + 'inputs/input_min_v5_pervar.nc').astype(np.float32)

# if output_norm_per_level:
output_scale = xr.open_dataset(norm_path + 'outputs//output_scale_std_lowerthred_v5.nc').astype(np.float32)
# output_scale = xr.open_dataset(norm_path + 'outputs/output_scale_std_nopenalty.nc').astype(np.float32)

# else:
#     output_scale = xr.open_dataset(norm_path + 'outputs/output_scale.nc').astype(np.float32)


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



# set inputs and outputs to V1 subset
#data.set_to_v1_vars()
#data.set_to_v2_vars()
#data.set_to_v4_vars()
data.set_to_v4_rnn_vars()

hyam = torch.from_numpy(data.grid_info['hyam'].values).to(device, torch.float32)
hybm = torch.from_numpy(data.grid_info['hybm'].values).to(device, torch.float32)
hyai = torch.from_numpy(data.grid_info['hyai'].values).to(device, torch.float32)
hybi = torch.from_numpy(data.grid_info['hybi'].values).to(device, torch.float32)
sp_max = torch.from_numpy(data.input_max['state_ps'].values).to(device, torch.float32)
sp_min = torch.from_numpy(data.input_min['state_ps'].values).to(device, torch.float32)
sp_mean = torch.from_numpy(data.input_mean['state_ps'].values).to(device, torch.float32)


# ns, nlev, nx, nx_sfc, ny, ny_sfc = get_input_output_shapes(tr_data_path)
testfile = tr_data_path[0] if type(tr_data_path)==list else tr_data_path
hf = h5py.File(testfile, 'r')
print(hf.keys())
# <KeysViewHDF5 ['input_lev', 'input_sca', 'output_lev', 'output_sca']>
#print(hf.attrs.keys())
print(hf['input_lev'].attrs.keys())
dims = hf['input_lev'].shape
if len(dims)==4:
    ns, nloc, nlev, nx = dims 
else:
    ns, nlev, nx = dims

# xlev =  hf['input_lev'][0:1024,:,:]
print("ns", ns, "nlev", nlev,  "nx", nx)
# vars_2D_inp = hf['input_lev'].attrs.get('varnames').tolist()
vars_2D_inp = ['state_t',
  'state_rh',
  'state_q0002',
  'state_q0003',
  'state_u',
  'state_v',
  'state_t_dyn',
  'state_q0_dyn',
  'state_u_dyn',
  'tm_state_t_dyn',
  'tm_state_q0_dyn',
  'tm_state_u_dyn',
  'pbuf_ozone',
  'pbuf_CH4',
  'pbuf_N2O']
print(vars_2D_inp)

# inds = np.arange(0,240).tolist()
# t0_it = time.time()
# # x = hf['input_lev'][inds]
# x = hf['input_lev'][0:240]
# elaps = time.time() - t0_it
# print("Runtime load {:.2f}s".format(elaps))
# t0_it = time.time()
# # Runtime load 7.30s
# # Runtime load 4.74s

# t0_it = time.time()
# x = hf['input_lev'][0:480]
# elaps = time.time() - t0_it
# print("Runtime load {:.2f}s".format(elaps))
# t0_it = time.time()
# # Runtime load 12.42s
# # 480 / ns * (244)  =  4.45 GB 
# # x.size * 4 / 1e9 = 3.31776 GB
# # 4.45 / 12.42 = 0.358 GB /s
# # 3.1  / 12.42 = 0.2665 GB / s
#          # CHUNKED ( 822, 60, 4, 1 )
         
# t0_it = time.time()
# x = hf['input_lev'][0:822, 0:60, 0:4, 0:1]
# elaps = time.time() - t0_it
# print("Runtime load {:.4f}s".format(elaps))
# t0_it = time.time()
# # Runtime load 0.0008s

# t0_it = time.time()
# # x = hf['input_lev'][inds]
# x = hf['input_lev'][0:128]
# elaps = time.time() - t0_it
# print("Runtime load {:.2f}s".format(elaps))
# t0_it = time.time()
# # Runtime load 2.15s

# t0_it = time.time()
# yy = hf['output_sca'][:]
# elaps = time.time() - t0_it
# print("Runtime load {:.2f}s".format(elaps))
# t0_it = time.time()
# # (26277, 1920, 8)
# # Runtime load 4.66s



# Runtime load 5.92s 
# compressed, WD Runtime load 13.67s
# non compressed, WD Runtime load 8.46s


# future training data should have a "varnames" attribute for each dataset type 
#                                              
#2D Input variables: ['state_t', 'state_q0001', 'state_q0002', 'state_q0003', 'state_u', 'state_v', 
# 'pbuf_ozone', 'pbuf_CH4', 'pbuf_N2O']
# NEW::
#['state_t' 'state_rh' 'state_q0002' 'state_q0003' 'state_u' 'state_v'
# 'state_t_dyn' 'state_q0_dyn' 'state_u_dyn' 'tm_state_t_dyn'
# 'tm_state_q0_dyn' 'tm_state_u_dyn' 'pbuf_ozone' 'pbuf_CH4' 'pbuf_N2O']
# state_q0001 lev, ncol kg/kg Specific humidity
# state_q0002 lev, ncol kg/kg Cloud liquid mixing ratio
# state_q0003 lev, ncol kg/kg Cloud ice mixing ratio
# We need pressure!

#1D (scalar) Input variables:
#   'ps' 'pbuf_SOLIN' 'pbuf_LHFLX' 'pbuf_SHFLX' 'pbuf_TAUX' 'pbuf_TAUY'
#  'pbuf_COSZRS' 'cam_in_ALDIF' 'cam_in_ALDIR' 'cam_in_ASDIF' 'cam_in_ASDIR'
#  'cam_in_LWUP' 'cam_in_ICEFRAC' 'cam_in_LANDFRAC' 'cam_in_OCNFRAC'
#  'cam_in_SNOWHICE' 'cam_in_SNOWHLAND' 'tm_state_ps' 'tm_pbuf_SOLIN'
#  'tm_pbuf_LHFLX' 'tm_pbuf_SHFLX' 'tm_pbuf_COSZRS' 'clat' 'slat']
dims = hf['input_sca'].shape; nx_sfc = dims[-1]
print("nx_sfc:", nx_sfc)
# vars_1D_inp = hf['input_sca'].attrs.get('varnames').tolist()
vars_1D_inp = ['state_ps',
  'pbuf_SOLIN',
  'pbuf_LHFLX',
  'pbuf_SHFLX',
  'pbuf_TAUX',
  'pbuf_TAUY',
  'pbuf_COSZRS',
  'cam_in_ALDIF',
  'cam_in_ALDIR',
  'cam_in_ASDIF',
  'cam_in_ASDIR',
  'cam_in_LWUP',
  'cam_in_ICEFRAC',
  'cam_in_LANDFRAC',
  'cam_in_OCNFRAC',
  'cam_in_SNOWHICE',
  'cam_in_SNOWHLAND',
  'tm_state_ps',
  'tm_pbuf_SOLIN',
  'tm_pbuf_LHFLX',
  'tm_pbuf_SHFLX',
  'tm_pbuf_COSZRS',
  'clat',
  'slat']
print(vars_1D_inp)

# Temperature tendency, q-wv tendency, cloud liquid tendency, cloud ice tendency, wind tendencies
# Internally, the CRM has only 50 levels, but radiation is compute using the full 60 levels
# this means computation of dT_rad is done on 60 level grid but everything else
# the full 60 levels
#2D Output variables: ['ptend_t', 'ptend_q0001', 'ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v']
# ['ptend_t', 'ptend_q0001', 'ptend_qn', 'ptend_u', 'ptend_v']
dims = hf['output_lev'].shape; ny = dims[-1]
print("ny:", ny)
# vars_2D_outp = hf['output_lev'].attrs.get('varnames').tolist()
vars_2D_outp = ['ptend_t', 'ptend_q0001', 'ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v']
print(vars_2D_outp)

#1D (scalar) Output variables: ['cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 
#'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']
dims = hf['output_sca'].shape; ny_sfc = dims[-1]
print("ny_sfc:", ny_sfc)
# vars_1D_outp = hf['output_sca'].attrs.get('varnames').tolist()
vars_1D_outp = ['cam_out_NETSW',
  'cam_out_FLWDS',
  'cam_out_PRECSC',
  'cam_out_PRECC',
  'cam_out_SOLS',
  'cam_out_SOLL',
  'cam_out_SOLSD',
  'cam_out_SOLLD']
print(vars_1D_outp)

if reverse_scaling:
    input_mean_ref = np.copy(input_mean) 
    input_max_ref = np.copy(input_max) 
    input_min_ref = np.copy(input_min) 
    
    xmean_lev_ref = input_mean_ref[vars_2D_inp].to_dataarray(dim='features', name='inputs_lev').transpose().values
    xmax_lev_ref = input_max_ref[vars_2D_inp].to_dataarray(dim='features', name='inputs_lev').transpose().values
    xmin_lev_ref = input_min_ref[vars_2D_inp].to_dataarray(dim='features', name='inputs_lev').transpose().values
    # if add_refpres:
    #     xmean_lev_ref = np.concatenate((xmean_lev_ref, np.zeros(nlev).reshape(nlev,1)), axis=1)
    #     xmax_lev_ref = np.concatenate((xmax_lev_ref, np.ones(nlev).reshape(nlev,1)), axis=1)
    #     xmin_lev_ref = np.concatenate((xmin_lev_ref, np.zeros(nlev).reshape(nlev,1)), axis=1)
    # xcoeff_lev_ref = np.stack((xmean_lev_ref, xmin_lev_ref, xmax_lev_ref))
    xdiv_lev_ref = xmax_lev_ref - xmin_lev_ref
    xcoeff_lev_ref = np.stack((xmean_lev_ref, xdiv_lev_ref))

    xmean_sca_ref = input_mean_ref[vars_1D_inp].to_dataarray(dim='features', name='inputs_sca').transpose().values
    xmax_sca_ref = input_max_ref[vars_1D_inp].to_dataarray(dim='features', name='inputs_sca').transpose().values
    xmin_sca_ref = input_min_ref[vars_1D_inp].to_dataarray(dim='features', name='inputs_sca').transpose().values
    # xcoeff_sca_ref = np.stack((xmean_sca_ref, xmin_sca_ref, xmax_sca_ref))
    xdiv_sca_ref = xmax_sca_ref - xmin_sca_ref
    xcoeff_sca_ref = np.stack((xmean_sca_ref, xdiv_sca_ref))

    xcoeffs_ref = np.float32(xcoeff_lev_ref), np.float32(xcoeff_sca_ref)
    
    
    yscale_lev_ref  = np.array([[1.00464e+03, 2.83470e+06, 5.66940e+06, 2.83470e+06, 2.50000e+02,
            5.00000e+02]], dtype=np.float32).repeat(60,axis=0)
    yscale_sca_ref = np.array([2.40000e-03, 5.00000e-03, 1.24416e+07, 1.31328e+06, 5.00000e-03,
            4.60000e-03, 6.10000e-03, 9.50000e-03], dtype=np.float32)
    
    xcoeffs_gen_ref = xcoeffs_ref
    ycoeffs_gen_ref = np.float32(yscale_lev_ref, yscale_sca_ref )

else:
    xcoeffs_gen_ref = None
    ycoeffs_gen_ref = None

if v4_to_v5_inputs:
    # xmean_lev_ref = input_mean_ref[vars_2D_inp].to_dataarray(dim='features', name='inputs_lev').transpose().values
    # xmax_lev_ref = input_max_ref[vars_2D_inp].to_dataarray(dim='features', name='inputs_lev').transpose().values
    # xmin_lev_ref = input_min_ref[vars_2D_inp].to_dataarray(dim='features', name='inputs_lev').transpose().values
    # # if add_refpres:
    # #     xmean_lev_ref = np.concatenate((xmean_lev_ref, np.zeros(nlev).reshape(nlev,1)), axis=1)
    # #     xmax_lev_ref = np.concatenate((xmax_lev_ref, np.ones(nlev).reshape(nlev,1)), axis=1)
    # #     xmin_lev_ref = np.concatenate((xmin_lev_ref, np.zeros(nlev).reshape(nlev,1)), axis=1)
    # # xcoeff_lev_ref = np.stack((xmean_lev_ref, xmin_lev_ref, xmax_lev_ref))
    # xdiv_lev_ref = xmax_lev_ref - xmin_lev_ref
    # xcoeff_lev_ref = np.stack((xmean_lev_ref, xdiv_lev_ref))

    # xmean_sca_ref = input_mean_ref[vars_1D_inp].to_dataarray(dim='features', name='inputs_sca').transpose().values
    # xmax_sca_ref = input_max_ref[vars_1D_inp].to_dataarray(dim='features', name='inputs_sca').transpose().values
    # xmin_sca_ref = input_min_ref[vars_1D_inp].to_dataarray(dim='features', name='inputs_sca').transpose().values
    # # xcoeff_sca_ref = np.stack((xmean_sca_ref, xmin_sca_ref, xmax_sca_ref))
    # xdiv_sca_ref = xmax_sca_ref - xmin_sca_ref
    # xcoeff_sca_ref = np.stack((xmean_sca_ref, xdiv_sca_ref))

    # xcoeffs_ref = np.float32(xcoeff_lev_ref), np.float32(xcoeff_sca_ref)

    #  old array has  T, rh, qliq, qice, X..,.
    ##  new array has  T, rh, qn,   qice, liqratio, X ...
    #  new array has  T, rh, qn,   liqratio, X ...
    vars_2D_inp.remove('state_q0002') # liq
    vars_2D_inp.remove('state_q0003')
    vars_2D_inp.insert(2,"state_qn")
    vars_2D_inp.insert(3,"liq_partition")
    # vars_2D_inp.insert(4,"liq_partition")
    # nx = nx + 1
    
    # ['state_t',
    #  'state_rh',
    #  'state_qn',
    #  'state_q0003',
    #  'liq_partition',
    #  'state_u',
    #  'state_v',
    #  'state_t_dyn',
    #  'state_q0_dyn',
    #  'state_u_dyn',
    #  'tm_state_t_dyn',
    #  'tm_state_q0_dyn',
    #  'tm_state_u_dyn',
    #  'pbuf_ozone',
    #  'pbuf_CH4',
    #  'pbuf_N2O']


hf.close()

# if use_mp_constraint or pred_liq_ratio:
if mp_mode>0:
    vars_2D_outp.remove('ptend_q0002')
    vars_2D_outp.remove('ptend_q0003')
    vars_2D_outp.insert(2,"ptend_qn")

# if pred_liq_ratio:
#     vars_2D_outp.insert(2,"liq_frac")   

yscale_lev = output_scale[vars_2D_outp].to_dataarray(dim='features', name='outputs_lev').transpose().values
yscale_sca = output_scale[vars_1D_outp].to_dataarray(dim='features', name='outputs_sca').transpose().values

# yscale_lev[20:,:].mean(axis=0)
# Out[52]: 
# array([2.3405453e+04, 2.3265182e+08, 1.4898973e+08, 6.4926711e+04,
#        7.8328773e+04], dtype=float32)
if not output_norm_per_level:
    yscale_lev = np.repeat(np.array([2.3405453e+04, 2.3265182e+08, 1.4898973e+08, 6.4926711e+04,
            7.8328773e+04], dtype=np.float32).reshape((1,-1)),nlev,axis=0)
    # _lambda = torch.tensor(5.0e-7) 

    # weights = 1 / (yscale_lev / yscale_lev_mean)
    # weights = torch.from_numpy(weights).to(device).unsqueeze(dim=0)
weights=None 

if mp_mode==2:
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

        
ycoeffs_gen = (yscale_lev, yscale_sca)
print("Y coeff shapes:", yscale_lev.shape, yscale_sca.shape)

# if not output_norm_per_level:
#     yscale_lev_gen, yscale_sca_gen = None, None
# else:
#     yscale_lev_gen, yscale_sca_gen = yscale_lev, yscale_sca

xmean_lev = input_mean[vars_2D_inp].to_dataarray(dim='features', name='inputs_lev').transpose().values
xmax_lev = input_max[vars_2D_inp].to_dataarray(dim='features', name='inputs_lev').transpose().values
xmin_lev = input_min[vars_2D_inp].to_dataarray(dim='features', name='inputs_lev').transpose().values

if not input_norm_per_level:
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
    
xdiv_lev = xmax_lev - xmin_lev
if xdiv_lev[-1,-1] == 0.0:
    xdiv1 = xdiv_lev[:,-1]
    xdiv1_min = np.min(xdiv1[xdiv1>0.0])
    xdiv1[xdiv1==0.0] = xdiv1_min
    xdiv_lev[:,-1] = xdiv1 
    
    xdiv1 = xdiv_lev[:,-2]
    xdiv1_min = np.min(xdiv1[xdiv1>0.0])
    xdiv1[xdiv1==0.0] = xdiv1_min
    xdiv_lev[:,-2] = xdiv1 

if add_refpres:
    xmean_lev = np.concatenate((xmean_lev, np.zeros(nlev).reshape(nlev,1)), axis=1)
    # xmin_lev = np.concatenate((xmin_lev, np.zeros(nlev).reshape(nlev,1)), axis=1)
    # xmax_lev = np.concatenate((xmax_lev, np.ones(nlev).reshape(nlev,1)), axis=1)
    xdiv_lev =  np.concatenate((xdiv_lev, np.ones(nlev).reshape(nlev,1)), axis=1)
    
# xcoeff_lev = np.stack((xmean_lev, xmin_lev, xmax_lev))
xcoeff_lev = np.stack((xmean_lev, xdiv_lev))

# for i in range(xcoeff_lev.shape[-1]):
#     for j in range(nlev):
#         div = (xcoeff_lev[2,j,i] - xcoeff_lev[1,j,i])
#         print( i,j, "scaling div x",  div)
#         if div==0:
#             break
#     else:
#         continue  # only executed if the inner loop did NOT break
#     break  # only executed if the inner loop DID break
xmean_sca = input_mean[vars_1D_inp].to_dataarray(dim='features', name='inputs_sca').transpose().values
xmax_sca = input_max[vars_1D_inp].to_dataarray(dim='features', name='inputs_sca').transpose().values
xmin_sca = input_min[vars_1D_inp].to_dataarray(dim='features', name='inputs_sca').transpose().values
# xcoeff_sca = np.stack((xmean_sca, xmin_sca, xmax_sca))
xdiv_sca = xmax_sca - xmin_sca
xcoeff_sca = np.stack((xmean_sca, xdiv_sca))
xcoeffs = np.float32(xcoeff_lev), np.float32(xcoeff_sca)


# if v4_to_v5_inputs:
#     xcoeffs_gen = xcoeffs
#     # xcoeffs_gen_ref = xcoeffs_ref
# else:
#     xcoeffs_gen = None
#     # xcoeffs_gen_ref = None
xcoeffs_gen = xcoeffs

if preprocess_on_the_fly: xcoeffs_gen = None

if remove_past_sfc_inputs:
    nx_sfc = nx_sfc - 5
    
# nx = 15
if add_refpres:
    nx = nx + 1
    
# if use_mp_constraint:
if mp_mode==1:
    ny_pp = ny # The 6 original outputs will be after postprocessing
    ny = ny - 1 # The model itself only has 5 outputs (total cloud water)
else:
    ny_pp = ny 
# ny_pp = ny 

print("Setting up RNN model using nx={}, nx_sfc={}, ny={}, ny_sfc={}".format(nx,nx_sfc,ny,ny_sfc))

if model_type=="LSTM":
    if autoregressive:
        model = LSTM_autoreg_torchscript(hyam,hybm,hyai,hybi,
                    out_scale = yscale_lev,
                    out_sfc_scale = yscale_sca, 
                    nx = nx, nx_sfc=nx_sfc, 
                    ny = ny, ny_sfc=ny_sfc, 
                    nneur=nneur, 
                    use_initial_mlp = use_initial_mlp,
                    use_intermediate_mlp=use_intermediate_mlp,
                    add_pres=add_pres,
                    output_prune=output_prune,
                    use_memory=use_memory,
                    separate_radiation=separate_radiation,
                    predict_flux=predict_flux)
    else:
        model = LSTM_torchscript(hyam,hybm,
                    out_scale = yscale_lev,
                    out_sfc_scale = yscale_sca, 
                    nx = nx, nx_sfc=nx_sfc, 
                    ny = ny, ny_sfc=ny_sfc, 
                    nneur=nneur, 
                    use_initial_mlp = use_initial_mlp,
                    use_intermediate_mlp=use_intermediate_mlp,
                    add_pres=add_pres, output_prune=output_prune)
else:
    model = SpaceStateModel(hyam, hybm, 
                out_scale = yscale_lev,
                out_sfc_scale = yscale_sca,  
                nlay=60, nx = nx, nx_sfc=nx_sfc, 
                ny = ny, ny_sfc=ny_sfc, 
                nneur=nneur,model_type=model_type, 
                use_initial_mlp=use_initial_mlp, add_pres=add_pres,  concat=concat)
# elif autoregressive:
#     model = RNN_autoreg(hyam,hybm,
#             cell_type=model_type, 
#             nlay = nlev, 
#             nx = nx, nx_sfc=nx_sfc, 
#             ny = ny, ny_sfc=ny_sfc, 
#             nneur=nneur,
#             memory=memory,
#             concat=concat,
#             use_initial_mlp=use_initial_mlp,
#             use_intermediate_mlp=use_intermediate_mlp,
#             add_pres=add_pres,
#             ensemble_size=ensemble_size,
#             add_stochastic_layer=add_stochastic_layer,     
#             # use_mp_constraint = use_mp_constraint,
#             mp_mode = mp_mode,
#             preprocess=preprocess_on_the_fly,
#             out_scale = yscale_lev,
#             out_sfc_scale = yscale_sca, 
#             xcoeffs=xcoeffs, 
#             v4_to_v5_inputs = v4_to_v5_inputs,
#             separate_radiation=separate_radiation)

model = model.to(device)

if autoregressive:
    # model.rnn1_mem = torch.randn(nloc, nlay, model.nh_mem, device=device)
    # model.rnn1_mem = torch.zeros(nloc, nlay, model.nh_mem, device=device)
    rnn1_mem = torch.zeros(nloc, model.nlay, model.nh_mem, device=device)

infostr = summary(model)
num_params = infostr.total_params
print(infostr)


    
if use_scaler:
    # scaler = torch.amp.GradScaler(autocast = True)
    scaler = torch.amp.GradScaler(device.type)
    
test_with_real_data = False

if test_with_real_data:
    testfile = tr_data_path[0] if type(tr_data_path)==list else tr_data_path

    hf = h5py.File(testfile, 'r')
    bsize = nloc 
    nb = 10
    x_lay = hf['input_lev'][0:nb*bsize]
    x_sfc = hf['input_sca'][0:nb*bsize]
    y_lay = hf['output_lev'][0:nb*bsize]
    y_sfc = hf['output_sca'][0:nb*bsize]
    hf.close()

    print(x_lay.shape, x_lay.min(), x_lay.max())
    for i in range(nx-1):
        print("x={}, min {} max {}".format(i, x_lay[:,:,i].min(), x_lay[:,:,i].max()))
    print(x_sfc.shape, x_sfc.min(), x_sfc.max())
    print(y_lay.shape, y_lay.min(), y_lay.max())
    print(y_sfc.shape, y_sfc.min(), y_sfc.max())

    # Test the model with real data 
    x_lay0 = torch.from_numpy(x_lay).to(device)
    x_sfc0 = torch.from_numpy(x_sfc).to(device)
    y_lay0 = torch.from_numpy(y_lay).to(device)
    y_sfc0 = torch.from_numpy(y_sfc).to(device)

    print(x_lay0.shape, x_sfc0.shape, y_lay0.shape, y_sfc0.shape)

    out, out_sfc = model(x_lay0, x_sfc0)
    print(out.shape, out_sfc.shape)
else:
    # Test the model with dummy data 
    bsize = nloc 
    nb = 1
    x_lay = torch.zeros((nb*bsize, nlev, nx))
    x_sfc = torch.zeros((nb*bsize, nx_sfc))

    x_lay = x_lay.to(device)
    x_sfc = x_sfc.to(device)

    if autoregressive:
        out, out_sfc, rnn1_mem = model(x_lay, x_sfc, rnn1_mem)
    else:
        out, out_sfc = model(x_lay, x_sfc)

    print(out.shape, out_sfc.shape)
    

# scripted_model = torch . jit . script ( model )


batch_size_tr = nloc

# To improve IO, which is a bottleneck, increase the batch size by a factor of chunk_factor and 
# load this many batches at once. These chk then need to be manually split into batches 
# within the data iteration loop   

chunksiz = 720
chunksiz = 360
chunksiz = 60
chunksiz = 120
chunksiz = 240
chunksiz = 144

# chunksiz = 220
# chunksiz = 380
chunksiz_val = 340
chunksiz_val = 14

# num_workers = 1
# num_workers = 4
num_workers = 7
# num_workers = 5
# num_workers = 8
# num_workers = 10

prefetch_factor = 1
pin = False
persistent=False


if type(tr_data_path)==list:  
    num_files = len(tr_data_path)
    batch_size_tr = num_files*batch_size_tr 
    chunk_size_tr = chunksiz // num_files

    # chunk_size_tr = 720 
else:
    num_files = 1
    # chunk size in number of batches240
    chunk_size_tr = chunksiz # 720 = 10 days (3 time steps in an hour, 72 in a day)
    
    
# chunk size in number of elements
# num_samples_per_chunk_tr = chunk_size_tr*batch_size_tr
# num_samples_per_chunk_tr = chunk_size_tr*nloc
num_samples_per_chunk_tr = chunk_size_tr

if use_val:
    batch_size_val = nloc_val
    if type(val_data_path)==list:  
        num_files = len(tr_data_path)
        batch_size_val = num_files*batch_size_val
        chunk_size_val = chunksiz_val // num_files
    else:
        chunk_size_val = chunksiz_val # 10 days (3 time steps in an hour, 72 in a day)
        
    # num_samples_per_chunk_val = chunk_size_val*batch_size_val
    # num_samples_per_chunk_val = chunk_size_val*nloc_val
    num_samples_per_chunk_val = chunk_size_val


train_data = generator_xy(tr_data_path, nloc=nloc, add_refpres=add_refpres, remove_past_sfc_inputs=remove_past_sfc_inputs, 
                              mp_mode=mp_mode,v4_to_v5_inputs=v4_to_v5_inputs,
                              rh_prune=rh_prune, qinput_prune=qinput_prune,output_prune=output_prune,
                             ycoeffs=ycoeffs_gen, xcoeffs=xcoeffs_gen, ycoeffs_ref=ycoeffs_gen_ref, xcoeffs_ref=xcoeffs_gen_ref)
train_batch_sampler = BatchSampler(num_samples_per_chunk_tr, 
                                   # num_samples=train_data.ntimesteps*nloc, shuffle=shuffle_data)
                                   num_samples=train_data.ntimesteps, shuffle=shuffle_data)

train_loader = DataLoader(dataset=train_data, num_workers=num_workers, sampler=train_batch_sampler, 
                          batch_size=None,batch_sampler=None,prefetch_factor=prefetch_factor, 
                          pin_memory=pin, persistent_workers=persistent)



# print(x_lay_chk[0,:,14])   
# tensor([0.0604, 0.0629, 0.0650, 0.0666, 0.0676, 0.0679, 0.0676, 0.0665, 0.0649,
#         0.0627, 0.0600, 0.0569, 0.0534, 0.0497, 0.0457, 0.0415, 0.0372, 0.0328,
#         0.0285, 0.0321, 0.0499, 0.0547, 0.0324, 0.0162, 0.0061, 0.0012, 0.0001,
#         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, ..
        
if use_val:
    
    val_data = generator_xy(val_data_path, nloc=nloc_val, add_refpres=add_refpres, remove_past_sfc_inputs=remove_past_sfc_inputs, 
                            mp_mode=mp_mode,v4_to_v5_inputs=v4_to_v5_inputs,
                            rh_prune=rh_prune, qinput_prune=qinput_prune,output_prune=output_prune,
                            ycoeffs=ycoeffs_gen, xcoeffs=xcoeffs_gen, ycoeffs_ref=ycoeffs_gen_ref, xcoeffs_ref=xcoeffs_gen_ref)

    val_batch_sampler = BatchSampler(num_samples_per_chunk_val, 
                                       # num_samples=val_data.ntimesteps*nloc_val, shuffle=shuffle_data)
                                       num_samples=val_data.ntimesteps, shuffle=shuffle_data)

    val_loader = DataLoader(dataset=val_data, num_workers=num_workers,sampler=val_batch_sampler,
                            batch_size=None,batch_sampler=None,prefetch_factor=prefetch_factor, 
                            pin_memory=pin, persistent_workers=persistent)



metric_h_con = metrics.get_energy_metric(hyai, hybi)
mse = metrics.get_mse_flatten(weights)

if loss_fn == "mse":
    regular_loss = mse
elif loss_fn == "huber":
    regular_loss = metrics.get_huber_flatten(weights)
else:
    raise NotImplementedError()

#loss_fn = my_mse_flatten
# regular_loss = metrics.my_huber_loss

loss_fn = metrics.get_hybrid_loss(_lambda)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if not autoregressive:
    timewindow = 1
    timestep_scheduling=False
    timestep_schedule = np.arange(1000)
    timestep_schedule[:] = timewindow
else:
    timewindow = 3
    timestep_scheduling=True
    timestep_schedule = np.arange(1000)
    timestep_schedule[:] = timewindow

    if timestep_scheduling:
        timestep_schedule[0:3] = 1
        timestep_schedule[3:4] = timewindow-1
        timestep_schedule[4:] = timewindow
        timestep_schedule[5:] = timewindow+1
        timestep_schedule[6:] = timewindow+2
        
    # if timestep_scheduling:
    #     timestep_schedule[0:3] = 1
    #     timestep_schedule[3:5] = timewindow-1
    #     timestep_schedule[5:7] = timewindow
    #     timestep_schedule[7:] = timewindow+1
    #     # timestep_schedule[6:] = timewindow+2
        
from random import randrange
model_num = randrange(10,99999)

dtypestr = "{}".format(dtype)
cwd = os.getcwd()
config = dict(((k, eval(k)) for k in ("num_files",
                                      "model_type",
                                      "add_stochastic_layer",
                                      "val_data_fname",
                                      "autoregressive",
                                      # "stateful",
                                      "v4_to_v5_inputs",
                                      "remove_past_sfc_inputs",
                                      "memory",
                                      "model_num",
                                      "num_params",
                                      "separate_radiation",
                                      "num_workers",
                                      "mp_autocast",
                                      "shuffle_data",
                                      "concat",
                                      "nx", 
                                      "nx_sfc",
                                      "ny",
                                      "ny_pp",
                                      "ny_sfc",
                                      "rh_prune",
                                      "qinput_prune",
                                      "output_prune",
                                      "loss_fn",
                                      "use_energy_loss",
                                      "use_intermediate_mlp",
                                      "use_initial_mlp",
                                      "lr",
                                      "_lambda",
                                      "add_pres",
                                      "batch_size_tr",
                                      "nneur",
                                      # "norm",
                                      "timestep_scheduling",
                                      "timestep_schedule",
                                      "preprocess_on_the_fly",
                                      "dtypestr",
                                      "use_scaler",
                                      "output_norm_per_level",
                                      "input_norm_per_level",
                                      "cwd",
)
))
if use_wandb:
    os.environ["WANDB__SERVICE_WAIT"]="400"
    config['timestep_schedule'] = timestep_schedule[0:10]
    run = wandb.init(
        project="climsim",
        config=config
    )       

class model_train_eval:
    def __init__(self, dataloader, model, batch_size = 384, autoregressive=True, train=True):
        super().__init__()
        self.loader = dataloader
        self.train = train
        self.report_freq = 800
        self.batch_size = batch_size
        self.model = model 
        self.autoregressive = autoregressive
        # if self.autoregressive:
        #     # self.model.reset_states()
        #     # model.rnn1_mem = torch.randn_like(model.rnn1_mem)
        #     model.rnn1_mem = torch.randn(self.batch_size, nlay, nneur[1], device=device)
        # print("Initializing trainer, is train: {} nloc: {}".format(self.train, self.batch_size))
        # print("shape mem {}˝".format(model.rnn1_mem.shape))

        self.metric_R2 =  R2Score(num_outputs=ny_pp).to(device) 
        # self.metric_R2_heating =  R2Score().to(device) 
        # self.metric_R2_precc =  R2Score().to(device) 
        # self.metric_R2_moistening =  R2Score().to(device) 

        self.metrics = {}
        # self.metrics= {'loss': 0, 'mean_squared_error': 0,  # the latter is just MSE
        #                 'mean_absolute_error': 0, 'R2' : 0, 'R2_heating' : 0,
        #                 # 'R2_moistening' : 0,  
        #                 'R2_precc' : 0, 'R2_lev' : np.zeros((nlev,ny_pp)),
        #                 'h_conservation' : 0,
        #                 "bias_lev" : 0, "bias_heating" : 0, "bias_sfc" : 0}

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

        t_comp =0 
        if self.autoregressive:
            preds_lay = []; preds_sfc = []
            targets_lay = []; targets_sfc = [] 
            surf_pres = []; x_lay = []; x_lay_raw = []
            yto_lay = []; yto_sfc = []
        t0_it = time.time()
        j = 0; k = 0; k2=2    
        if self.autoregressive:
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
                    
                if mp_autocast:
                    with torch.autocast(device_type=device.type, dtype=dtype):
                        if autoregressive:
                            preds_lay0, preds_sfc0, rnn1_mem = self.model(x_lay0, x_sfc0, rnn1_mem)
                        else:
                            preds_lay0, preds_sfc0 = self.model(x_lay0, x_sfc0)

                else:
                    if autoregressive:
                        preds_lay0, preds_sfc0, rnn1_mem = self.model(x_lay0, x_sfc0, rnn1_mem)
                    else:
                        preds_lay0, preds_sfc0 = self.model(x_lay0, x_sfc0)

                if self.autoregressive:
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
                    
                if (not self.autoregressive) or (self.autoregressive and (j+1) % timewindow==0):
            
                    if self.autoregressive:
                        preds_lay   = torch.cat(preds_lay)
                        preds_sfc   = torch.cat(preds_sfc)
                        targets_lay = torch.cat(targets_lay)
                        targets_sfc = torch.cat(targets_sfc)
                        surf_pres   = torch.cat(surf_pres)
                        x_lay_raw   = torch.cat(x_lay_raw)
                        # yto_lay     = torch.cat(yto_lay)
                        # yto_sfc     = torch.cat(yto_sfc)   
                        
                    if mp_autocast:
                        with torch.autocast(device_type=device.type, dtype=dtype):
                            #loss = loss_fn(targets_lay, targets_sfc, preds_lay, preds_sfc)
                            
                            main_loss = regular_loss(targets_lay, targets_sfc, preds_lay, preds_sfc)
                            
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
                            
                            if use_energy_loss: 
                                loss = loss_fn(main_loss, h_con)
                            else:
                                loss = main_loss
                    else:
                        #loss = loss_fn(targets_lay, targets_sfc, preds_lay, preds_sfc)
                        
                        main_loss = regular_loss(targets_lay, targets_sfc, preds_lay, preds_sfc)
                        if use_mp_constraint:
                            ypo_lay, ypo_sfc = model.pp_mp(preds_lay, preds_sfc, x_lay_raw )
                            with torch.no_grad(): 
                                yto_lay, yto_sfc = model.pp_mp(targets_lay, targets_sfc, x_lay_raw )
                            # ypo_lay, ypo_sfc, yto_lay, yto_sfc = model.pp_mp(preds_lay, preds_sfc, targets_lay, targets_sfc, x_lay_raw )

                        else:
                            ypo_lay, ypo_sfc = model.postprocessing(preds_lay, preds_sfc)
                            yto_lay, yto_sfc = model.postprocessing(targets_lay, targets_sfc)
                        surf_pres_denorm = surf_pres*(sp_max - sp_min) + sp_mean
                        h_con = metric_h_con(yto_lay, ypo_lay, surf_pres_denorm)
                        if use_energy_loss: 
                            loss = loss_fn(main_loss, h_con)
                        else:
                            loss = main_loss
                    
                    # print("shape ypo lay", ypo_lay.shape, "yto", yto_lay.shape)
                    # print("minmax ypo lay", ypo_lay.min(), ypo_lay.max())
                    # print("minmax yto lay", yto_lay.min(), yto_lay.max())
                    # for i in range(6):
                    #     print("OOOO", i," minmax yp ", ypo_lay[:,50,i].min().item(), ypo_lay[:,50,i].max().item())
                    #     print("OOOO", i," minmax yt ", yto_lay[:,50,i].min().item(), yto_lay[:,50,i].max().item())

                    if self.train:
                        if use_scaler:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()       
                            optimizer.step()
            
                        optimizer.zero_grad()
                            
                    running_loss    += loss.item()
                    running_energy  += h_con.item()
                    #mae             = metrics.mean_absolute_error(targets_lay, preds_lay)
                    if j>loss_update_start_index:
                        with torch.no_grad():
                            epoch_loss      += loss.item()
                            if loss_fn =="huber":
                                epoch_mse       += main_loss.item()
                            else:
                                epoch_mse       += mse(targets_lay, targets_sfc, preds_lay, preds_sfc)
                            #epoch_mae       += mae.item()
                        
                            epoch_hcon  += h_con.item()
                            # print("shape ypo", ypo_lay.shape, "yto", yto_lay.shape)
                            
                            biases_lev, biases_sfc = metrics.compute_biases(yto_lay, yto_sfc, ypo_lay, ypo_sfc)
                            epoch_bias_lev += np.mean(biases_lev)
                            epoch_bias_heating += biases_lev[0]
                            epoch_bias_clw += biases_lev[2]
                            epoch_bias_cli += biases_lev[3]

                            epoch_bias_sfc += np.mean(biases_sfc)

                            self.metric_R2.update(ypo_lay.reshape((-1,ny_pp)), yto_lay.reshape((-1,ny_pp)))
                                   
                            r2_np = np.corrcoef((ypo_sfc.reshape(-1,ny_sfc)[:,3].detach().cpu().numpy(),yto_sfc.reshape(-1,ny_sfc)[:,3].detach().cpu().numpy()))[0,1]
                            epoch_R2precc += r2_np
                            #print("R2 numpy", r2_np, "R2 torch", self.metric_R2_precc(ypo_sfc[:,3:4], yto_sfc[:,3:4]) )

                            ypo_lay = ypo_lay.reshape(-1,nlev,ny_pp).detach().cpu().numpy()
                            yto_lay = yto_lay.reshape(-1,nlev,ny_pp).detach().cpu().numpy()

                            epoch_r2_lev += metrics.corrcoeff_pairs_batchfirst(ypo_lay, yto_lay) 
                           # if track_ks:
                           #     if (j+1) % max(timewindow*4,12)==0:
                           #         epoch_ks += kolmogorov_smirnov(yto,ypo).item()
                           #         k2 += 1
                            k += 1
                    if self.autoregressive:
                        preds_lay = []; preds_sfc = []
                        targets_lay = []; targets_sfc = [] 
                        surf_pres = []; x_lay_raw = []
                        yto_lay = []; yto_sfc = []
                        
                    if self.autoregressive: 
                        # model.detach_states()
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
                                                    j+1, running_loss,running_energy, r2raw, elaps, t_comp))
                    running_loss = 0.0
                    running_energy = 0.0
                    t0_it = time.time()
                    t_comp = 0
                j += 1

        self.metrics['loss'] =  epoch_loss / k
        self.metrics['mean_squared_error'] = epoch_mse / k
        self.metrics["h_conservation"] =  epoch_hcon / k
        
        self.metrics["bias_lev"] = epoch_bias_lev / k 
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

        if cuda: torch.cuda.empty_cache()
        gc.collect()


# 160 160
# autoreg, hybrid-loss, 2 years concat

train_runner = model_train_eval(train_loader, model, batch_size_tr, autoregressive, train=True)
if use_val: val_runner = model_train_eval(val_loader, model, batch_size_val, autoregressive, train=False)

SAVE_PATH =  'saved_models/{}-{}_lr{}.neur{}-{}.num{}.pt'.format(model_type,memory,lr, nneur[0],nneur[1], model_num)
# best_val_loss = np.inf
best_val_loss = 0

# w

for epoch in range(num_epochs):
    t0 = time.time()
    
    if timestep_scheduling:
        timewindoww=timestep_schedule[epoch]            
    else:
        timewindoww=timewindow
        
    print("Epoch {} Training rollout timesteps: {} ".format(epoch+1, timewindoww))
    train_runner.eval_one_epoch(epoch, timewindoww)
    
    if use_wandb: 
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
    
    if use_val:
        if epoch%2:
            print("VALIDATION..")
            val_runner.eval_one_epoch(epoch, timewindoww)

            if use_wandb: 
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
            val_loss = val_runner.metrics["R2˝"]

            # MODEL CHECKPOINT IF VALIDATION LOSS IMPROVED
            # if save_model and val_loss < best_val_loss:
            if save_model and val_loss > best_val_loss:

                print("New best validation result, saving model to ", SAVE_PATH)
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                            }, SAVE_PATH)  
                best_val_loss = val_loss 
              
    print('Epoch {}/{} complete, took {:.2f} seconds, autoreg window was {}'.format(epoch+1,num_epochs,time.time() - t0,timewindoww))

if use_wandb:
    wandb.finish()
    
loader = val_loader 
runner = val_runner

# loader = train_loader
# runner = train_runner

R2 = runner.metrics["R2_lev"]

import matplotlib.pyplot as plt


labels = ["dT/dt", "dq/dt", "dqliq/dt", "dqice/dt", "dU/dt", "dV/dt"]
# y = np.arange(60)
# ncols, nrows = 1,6
# fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5.5, 3.5),
#                         layout="constrained")
# for i in range(6):
#     axs[i].plot(R2[:,i],y)
#     axs[i].invert_yaxis()
#     axs[i].set_xlim(0,1)
#     axs[i].set_title(labels[i])
    

x = np.arange(60)
ncols, nrows = 6,1
fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5.5, 3.5),
                        gridspec_kw = {'wspace':0}) #layout="constrained")
for i in range(6):
    axs[i].plot(x, R2[:,i]); 
    axs[i].set_title(labels[i])
    axs[i].set_ylim(0,1)
    axs[i].set_xlim(0,60)
    axs[i].axvspan(0, 30, facecolor='0.2', alpha=0.2)
    if i>0:
        axs[i].set_yticklabels([])
    axs[i].set_xticklabels([])

fig.subplots_adjust(hspace=0)
    

for i,testdata in enumerate(loader):
    x_lay_chk, x_sfc_chk, targets_lay_chk, targets_sfc_chk, x_lay_raw_chk  = testdata
    print("i, x shape", i, x_lay_chk.shape)

    x_lay_chk       = x_lay_chk.to(device)
    x_lay_raw_chk   = x_lay_raw_chk.to(device)
    x_sfc_chk       = x_sfc_chk.to(device)
    targets_sfc_chk = targets_sfc_chk.to(device)
    targets_lay_chk = targets_lay_chk.to(device)
    # ytos_lay_chk    = ytos_lay_chk.to(device) # these are the raw (unnormalized) full outputs
    # ytos_sfc_chk    = ytos_sfc_chk.to(device)
    break

if autoregressive: 
    # model.reset_states()
    # model.rnn1_mem = torch.randn_like(model.rnn1_mem)
    # model.rnn1_mem = torch.zeros_like(model.rnn1_mem)
    rnn1_mem = torch.zeros_like(rnn1_mem)

    
j = 0
ntime = 10


for jj in range(ntime):
    jend = j + nloc
    print(j, jend)
    x_lay       = x_lay_chk[j:jend]
    x_lay_raw   = x_lay_raw_chk[j:jend]
    x_sfc       = x_sfc_chk[j:jend]
    targets_sfc = targets_sfc_chk[j:jend]
    targets_lay = targets_lay_chk[j:jend]
    

    with torch.no_grad(): 
        # preds_lay, preds_sfc = model(x_lay, x_sfc)
        preds_lay, preds_sfc, rnn1_mem = model(x_lay, x_sfc, rnn1_mem)

    j= j + nloc 
    

with torch.no_grad(): 
    ypo_lay, ypo_sfc = model.pp_mp(preds_lay,   preds_sfc,   x_lay_raw )
    yto_lay, yto_sfc = model.pp_mp(targets_lay, targets_sfc, x_lay_raw )
    
    
ypo_lay = ypo_lay.detach().cpu().numpy()
yto_lay = yto_lay.detach().cpu().numpy()
preds_lay = preds_lay.detach().cpu().numpy()
targets_lay = targets_lay.detach().cpu().numpy()



icol = 100
ivar = 2
y = np.arange(60)
fig, ax = plt.subplots(1,1)
ax.plot(yto_lay[icol,:,ivar], y , 'k', ypo_lay[icol,:,ivar], y, 'b')
ax.invert_yaxis()
ax.legend(["true","pred"])


out = preds_lay 
x_denorm = x_lay_raw.detach().cpu().numpy()
x = x_lay.detach().cpu().numpy()

out_denorm      = out / yscale_lev

T_before        = x_denorm[:,:,0:1]
qliq_before     = x_denorm[:,:,2:3]
qice_before     = x_denorm[:,:,3:4]   
qn_before       = qliq_before + qice_before 

# print("shape x denorm", x_denorm.shape, "T", T_before.shape)
T_new           = T_before  + out_denorm[:,:,0:1]*1200
# T_new_tar       = T_before  + tar_denorm[:,:,0:1]*1200

import torch.nn.functional as F
# T_denorm = T = T*(self.xmax_lev[:,0] - self.xmin_lev[:,0]) + self.xmean_lev[:,0]
# T_denorm = T*(self.xcoeff_lev[2,:,0] - self.xcoeff_lev[1,:,0]) + self.xcoeff_lev[0,:,0]
# liquid_ratio = (T_raw - 253.16) / 20.0 
T_raw = torch.from_numpy(T_new)
liquid_ratio = (T_raw - 253.16) * 0.05 
liquid_ratio = F.hardtanh(liquid_ratio, 0.0, 1.0)
liq_frac = liquid_ratio.detach().cpu().numpy()

qn_new       = qn_before + out_denorm[:,:,2:3]*1200  
qliq_new    = liq_frac*qn_new
qice_new    = (1-liq_frac)*qn_new

dqliq       = (qliq_new - qliq_before) * 0.0008333333333333334 #/1200  
dqice       = (qice_new - qice_before) * 0.0008333333333333334 #/1200   
        
icol = 240
icol = icol+10
for j in range(nlev):
    print("{} : dqn-raw  {:.3f} - {:.3f}  | dqLIQ {:.2e} - {:.2e}  | dqICE {:.2e} - {:.2e}  | liqFRAC {:.2f} |    ".format(j, preds_lay[icol,j,2],
                                                                targets_lay[icol,j,2],
                                                                ypo_lay[icol,j,2],
                                                                yto_lay[icol,j,2],
                                                                ypo_lay[icol,j,3],
                                                                  yto_lay[icol,j,3],
                                                                liq_frac[icol,j][0]))
    
# 0 : dqn-raw  0.000 - 0.000  | dqLIQ -1.81e-38 - -1.81e-38  | dqICE 0.00e+00 - 0.00e+00  | liqFRAC 0.00 |    
# 1 : dqn-raw  0.000 - 0.000  | dqLIQ -1.76e-38 - -1.76e-38  | dqICE 0.00e+00 - 0.00e+00  | liqFRAC 0.00 |    
# 2 : dqn-raw  0.000 - 0.000  | dqLIQ -1.63e-38 - -1.63e-38  | dqICE 0.00e+00 - 0.00e+00  | liqFRAC 0.00 |    
# 3 : dqn-raw  0.000 - 0.000  | dqLIQ -1.48e-38 - -1.48e-38  | dqICE 0.00e+00 - 0.00e+00  | liqFRAC 0.00 |    
# 4 : dqn-raw  0.000 - 0.000  | dqLIQ -1.38e-38 - -1.38e-38  | dqICE 0.00e+00 - 0.00e+00  | liqFRAC 0.00 |    
# 5 : dqn-raw  0.000 - 0.000  | dqLIQ -1.31e-38 - -1.31e-38  | dqICE 0.00e+00 - 0.00e+00  | liqFRAC 0.00 |    
# 6 : dqn-raw  0.000 - 0.000  | dqLIQ -1.27e-38 - -1.27e-38  | dqICE 0.00e+00 - 0.00e+00  | liqFRAC 0.00 |    
# 7 : dqn-raw  0.000 - 0.000  | dqLIQ -1.24e-38 - -1.24e-38  | dqICE 0.00e+00 - 0.00e+00  | liqFRAC 0.00 |    
# 8 : dqn-raw  0.000 - 0.000  | dqLIQ -1.07e-38 - -1.07e-38  | dqICE 0.00e+00 - 0.00e+00  | liqFRAC 0.00 |    
# 9 : dqn-raw  0.000 - 0.000  | dqLIQ -9.05e-39 - -9.05e-39  | dqICE 0.00e+00 - 0.00e+00  | liqFRAC 0.00 |    
# 10 : dqn-raw  0.000 - 0.000  | dqLIQ -7.24e-39 - -7.24e-39  | dqICE 0.00e+00 - 0.00e+00  | liqFRAC 0.00 |    
# 11 : dqn-raw  0.000 - 0.000  | dqLIQ -2.59e-39 - -2.59e-39  | dqICE 0.00e+00 - 0.00e+00  | liqFRAC 0.00 |    
# 12 : dqn-raw  0.000 - 0.000  | dqLIQ -5.11e-43 - -5.11e-43  | dqICE 0.00e+00 - 0.00e+00  | liqFRAC 0.00 |    
# 13 : dqn-raw  0.000 - 0.000  | dqLIQ -0.00e+00 - -0.00e+00  | dqICE 0.00e+00 - 0.00e+00  | liqFRAC 0.00 |    
# 14 : dqn-raw  0.000 - 0.000  | dqLIQ 0.00e+00 - 0.00e+00  | dqICE 0.00e+00 - 0.00e+00  | liqFRAC 0.00 |    
# 15 : dqn-raw  0.031 - -0.000  | dqLIQ 0.00e+00 - 0.00e+00  | dqICE 9.22e-12 - -8.57e-30  | liqFRAC 0.00 |    
# 16 : dqn-raw  0.007 - -0.000  | dqLIQ 0.00e+00 - 0.00e+00  | dqICE 2.10e-12 - -6.31e-32  | liqFRAC 0.00 |    
# 17 : dqn-raw  -0.005 - 0.000  | dqLIQ -0.00e+00 - 0.00e+00  | dqICE -4.51e-12 - 0.00e+00  | liqFRAC 0.00 |    
# 18 : dqn-raw  -0.008 - -0.000  | dqLIQ -0.00e+00 - 0.00e+00  | dqICE -1.04e-11 - -8.81e-19  | liqFRAC 0.00 |    
# 19 : dqn-raw  0.005 - 0.000  | dqLIQ 0.00e+00 - 0.00e+00  | dqICE 1.39e-11 - 0.00e+00  | liqFRAC 0.00 |    
# 20 : dqn-raw  -0.001 - -0.000  | dqLIQ -0.00e+00 - 0.00e+00  | dqICE -4.80e-12 - -6.67e-13  | liqFRAC 0.00 |    
# 21 : dqn-raw  -0.018 - -0.003  | dqLIQ -0.00e+00 - 0.00e+00  | dqICE -8.01e-11 - -1.14e-11  | liqFRAC 0.00 |    
# 22 : dqn-raw  0.009 - 0.013  | dqLIQ 0.00e+00 - 0.00e+00  | dqICE 4.52e-11 - 6.59e-11  | liqFRAC 0.00 |    
# 23 : dqn-raw  -0.003 - 0.016  | dqLIQ 0.00e+00 - 0.00e+00  | dqICE -1.89e-11 - 9.16e-11  | liqFRAC 0.00 |    
# 24 : dqn-raw  -0.001 - 0.016  | dqLIQ -0.00e+00 - -0.00e+00  | dqICE -3.72e-12 - 9.89e-11  | liqFRAC 0.00 |    
# 25 : dqn-raw  -0.017 - 0.004  | dqLIQ -8.24e-43 - -8.24e-43  | dqICE -1.09e-10 - 2.27e-11  | liqFRAC 0.00 |    
# 26 : dqn-raw  -0.022 - -0.043  | dqLIQ -2.88e-39 - -2.88e-39  | dqICE -1.45e-10 - -2.81e-10  | liqFRAC 0.00 |    
# 27 : dqn-raw  -0.057 - -0.165  | dqLIQ -1.27e-33 - -1.27e-33  | dqICE -3.75e-10 - -1.08e-09  | liqFRAC 0.00 |    
# 28 : dqn-raw  -0.101 - -0.237  | dqLIQ -1.40e-29 - -1.40e-29  | dqICE -6.56e-10 - -1.55e-09  | liqFRAC 0.00 |    
# 29 : dqn-raw  -0.105 - -0.218  | dqLIQ -6.50e-26 - -6.50e-26  | dqICE -6.80e-10 - -1.42e-09  | liqFRAC 0.00 |    
# 30 : dqn-raw  -0.049 - -0.195  | dqLIQ 0.00e+00 - 0.00e+00  | dqICE -3.21e-10 - -1.27e-09  | liqFRAC 0.00 |    
# 31 : dqn-raw  0.028 - -0.188  | dqLIQ -5.15e-17 - -5.15e-17  | dqICE 1.85e-10 - -1.24e-09  | liqFRAC 0.00 |    
# 32 : dqn-raw  0.160 - 0.031  | dqLIQ -5.06e-13 - -5.06e-13  | dqICE 1.07e-09 - 2.06e-10  | liqFRAC 0.00 |    
# 33 : dqn-raw  0.327 - 0.252  | dqLIQ -1.75e-10 - -1.75e-10  | dqICE 2.40e-09 - 1.89e-09  | liqFRAC 0.00 |    
# 34 : dqn-raw  0.271 - 0.172  | dqLIQ -5.59e-10 - -5.59e-10  | dqICE 2.46e-09 - 1.76e-09  | liqFRAC 0.00 |    
# 35 : dqn-raw  0.277 - 0.177  | dqLIQ -9.80e-10 - -9.85e-10  | dqICE 3.01e-09 - 2.29e-09  | liqFRAC 0.01 |    
# 36 : dqn-raw  0.869 - 1.195  | dqLIQ 1.56e-09 - 1.93e-09  | dqICE 5.19e-09 - 7.34e-09  | liqFRAC 0.15 |    
# 37 : dqn-raw  0.318 - 0.995  | dqLIQ 1.00e-09 - 2.65e-09  | dqICE 1.59e-09 - 5.45e-09  | liqFRAC 0.31 |    


fnames = ["E3SM-MMF.mlexpand.0001-02-01-02400.nc", 
          "E3SM-MMF.mlexpand.0001-12-01-00000.nc",
          "E3SM-MMF.mlexpand.0002-01-01-00000.nc", 
          "E3SM-MMF.mlexpand.0002-02-01-00000.nc", 
          "E3SM-MMF.mlexpand.0002-12-01-00000.nc"]

# Year 1 Month 2-12, Year 2 month 1 
import re

 # [0-5][0-9]
 # 0[2-9]|1[0-2]     02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12
year = 1
# regexp0 = 'E3SM-MMF.mlexpand.000{}-[02-12]-*-*.nc'.format(year)

# regexp0 = 'E3SM-MMF.mlexpand.000{}-0[1-9]|1[0-2]-*-*.nc'.format(year)

# Year i Month 2-12, Year i+1 month 1 

regexp0 = 'E3SM-MMF.mlexpand.000{}-0[2-9]-*-*.nc'.format(year)
regexp1 = 'E3SM-MMF.mlexpand.000{}-1[0-2]-*-*.nc'.format(year)
regexp2 = 'E3SM-MMF.mlexpand.000{}-01-*-*.nc'.format(year+1)
regexps = [regexp0, regexp1, regexp2]

mypath = "/media/peter/CrucialP1/data/ClimSim/low_res_expanded/test"

filelist = []


for regexp in regexps:
    filelist = filelist + glob.glob(mypath + "*/" + regexp)
    # r = re.compile(regexp)
    # filelist = filelist + list(filter(r.match, fnames)) # Read Note below
    # for string in fnames:
    #     print(string)
    #     if re.match(regexp, string):
    #         print(string, "matches the pattern")
        
print(sorted(filelist))