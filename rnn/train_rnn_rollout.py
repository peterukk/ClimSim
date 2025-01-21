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
from climsim_utils.data_utils import *
import xarray as xr
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
from models import RNN_autoreg, MyRNN
from utils import generator_xy, BatchSampler
# from metrics import get_energy_metric, get_hybrid_loss, my_mse_flatten
import metrics as metrics
from torchmetrics.regression import R2Score

def get_input_output_shapes(filepath):
    # inspect data
    testfile = filepath[0] if type(filepath)==list else filepath
    hf = h5py.File(testfile, 'r')
    print(hf.keys())
    # <KeysViewHDF5 ['input_lev', 'input_sca', 'output_lev', 'output_sca']>
    #print(hf.attrs.keys())
    print(hf['input_lev'].attrs.keys())
    ns, nlev, nx = hf['input_lev'].shape
    print("ns", ns, "nlev", nlev,  "nx", nx)
    print(hf['input_lev'].attrs.get('varnames'))
    # future training data should have a "varnames" attribute for each dataset type 
    
    #                                               CLOUD LIQUID mr  CLOUD ICE 
    #2D Input variables: ['state_t', 'state_q0001', 'state_q0002', 'state_q0003', 'state_u', 'state_v', 
    # 'pbuf_ozone', 'pbuf_CH4', 'pbuf_N2O']
    # NEW::
    #['state_t' 'state_rh' 'state_q0002' 'state_q0003' 'state_u' 'state_v'
    # 'state_t_dyn' 'state_q0_dyn' 'state_u_dyn' 'tm_state_t_dyn'
    # 'tm_state_q0_dyn' 'tm_state_u_dyn' 'pbuf_ozone' 'pbuf_CH4' 'pbuf_N2O']
    # We need pressure!
    
    #1D (scalar) Input variables: ['state_ps', 'pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX', 'pbuf_TAUX', 
    # 'pbuf_TAUY', 'pbuf_COSZRS', 'cam_in_ALDIF', 'cam_in_ALDIR', 'cam_in_ASDIF', 'cam_in_ASDIR', 
    # 'cam_in_LWUP', 'cam_in_ICEFRAC', 'cam_in_LANDFRAC', 'cam_in_OCNFRAC', 'cam_in_SNOWHICE', 
    # 'cam_in_SNOWHLAND', 'lat', 'lon']
    ns, nx_sfc = hf['input_sca'].shape
    print("nx_sfc:", nx_sfc)
    print(hf['input_sca'].attrs.get('varnames'))
    
    #2D Output variables: ['ptend_t', 'ptend_q0001', 'ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v']
    ns, nlev, ny = hf['output_lev'].shape
    print("ny:", ny)
    print(hf['output_lev'].attrs.get('varnames'))
    
    #1D (scalar) Output variables: ['cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 
    #'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']
    (ns, ny_sfc) = hf['output_sca'].shape
    print("ny_sfc:", ny_sfc)
    print(hf['output_sca'].attrs.get('varnames'))
    
    hf.close()
    return ns, nlev, nx, nx_sfc, ny, ny_sfc


#state_q0001 lev, ncol kg/kg Specific humidity
#state_q0002 lev, ncol kg/kg Cloud liquid mixing ratio
#state_q0003 lev, ncol kg/kg Cloud ice mixing ratio


data_dir = "/network/group/aopp/predict/HMC009_UKKONEN_CLIMSIM/ClimSim_data/ClimSim_low-res-expanded/train/preprocessed/"
data_dir = "/media/peter/CrucialBX500/data/ClimSim/"
data_dir = "/media/peter/CrucialP1/data/ClimSim/low_res_expanded/"

grid_path = '../grid_info/ClimSim_low-res_grid-info.nc'
norm_path = '../preprocessing/normalizations/'

use_val = False 
#use_val = True
shuffle_data = False 
nloc = 384

# Loss and training
num_epochs = 10
save_model = False 
_lambda = torch.tensor(1.0e-7) 
#_lambda = torch.tensor(1.0e-6) 
lr = 1e-3
use_wandb = False
if use_wandb: import wandb

# RNN Model configuration
autoregressive = True
memory = "Hidden"
concat = False 
use_initial_mlp = False 
use_intermediate_mlp = False
# add_pres = False 
ensemble_size = 1
add_stochastic_layer = False

nneur = (64,64)
nneur = (128,128)
nneur = (160, 160)

# --------------------------------------

grid_info = xr.open_dataset(grid_path)
input_mean = xr.open_dataset(norm_path + 'inputs/input_mean_v4_pervar.nc').astype(np.float32)
input_max = xr.open_dataset(norm_path + 'inputs/input_max_v4_pervar.nc').astype(np.float32)
input_min = xr.open_dataset(norm_path + 'inputs/input_min_v4_pervar.nc').astype(np.float32)
output_scale = xr.open_dataset(norm_path + 'outputs/output_scale.nc').astype(np.float32)

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


vars_1D_outp = []; vars_2D_outp = []
all_vars = list(data.output_scale.keys())
for var in all_vars:
    if 'lev' in data.output_scale[var].dims:
        vars_2D_outp.append(var)
    else:
        vars_1D_outp.append(var)  
        
yscale_lev = data.output_scale[vars_2D_outp].to_dataarray(dim='features', name='outputs_lev').transpose().values
yscale_sca = data.output_scale[vars_1D_outp].to_dataarray(dim='features', name='outputs_sca').transpose().values

print(yscale_lev.shape, yscale_sca.shape)

# tr_data_fname = "data_v4_rnn_year2.h5"
# tr_data_path = data_dir + tr_data_fname

tr_data_fname = ["data_v4_rnn_year2.h5", 
                "data_v4_rnn_year3.h5",
                "data_v4_rnn_year4.h5"]
tr_data_path = [data_dir + b for b in tr_data_fname]

if use_val:
    val_data_fname = "data_v4_rnn_year5.h5"
    val_data_path = data_dir + val_data_fname

ns, nlev, nx, nx_sfc, ny, ny_sfc = get_input_output_shapes(tr_data_path)


nx = 15
add_refpres = True
if add_refpres:
    nx = nx + 1

print("Setting up RNN model using nx={}, nx_sfc={}, ny={}, ny_sfc={}".format(nx,nx_sfc,ny,ny_sfc))

if autoregressive:
    model = RNN_autoreg(cell_type='LSTM', 
            nlay = nlev, 
            nx = nx, nx_sfc=nx_sfc, 
            ny = ny, ny_sfc=ny_sfc, 
            nneur=nneur,
            memory=memory,
            concat=concat,
            use_initial_mlp=use_initial_mlp,
            use_intermediate_mlp=use_intermediate_mlp,
            # add_pres=add_pres,
            ensemble_size=ensemble_size,
            add_stochastic_layer=add_stochastic_layer,      
            out_scale = yscale_lev,
            out_sfc_scale = yscale_sca)
else:

    model = MyRNN(RNN_type='LSTM', 
                 nx = nx, nx_sfc=nx_sfc, 
                 ny = ny, ny_sfc=ny_sfc, 
                 nneur=nneur,
                 out_scale = yscale_lev,
                 out_sfc_scale = yscale_sca)

cuda = torch.cuda.is_available() 
device = torch.device("cuda" if cuda else "cpu")
print(device)

model = model.to(device)


infostr = summary(model)
num_params = infostr.total_params
print(infostr)

if cuda:
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
    mp_autocast = False
    use_scaler = False
    
    
if use_scaler:
    # scaler = torch.amp.GradScaler(autocast = True)
    scaler = torch.amp.GradScaler(device.type)
    
test_with_real_data = False

if test_with_real_data:
    
    hf = h5py.File(tr_data_path, 'r')
    bsize = 384 
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
    bsize = 384 
    nb = 10
    x_lay = torch.zeros((nb*bsize, nlev, nx))
    x_sfc = torch.zeros((nb*bsize, nx_sfc))

    x_lay = x_lay.to(device)
    x_sfc = x_sfc.to(device)

    out, out_sfc = model(x_lay, x_sfc)
    print(out.shape, out_sfc.shape)
    
    
batch_size_tr = nloc

# To improve IO, which is a bottleneck, increase the batch size by a factor of chunk_factor and 
# load this many batches at once. These chunks then need to be manually split into batches 
# within the data iteration loop   

if type(tr_data_path)==list:  
    num_files = len(tr_data_path)
    batch_size_tr = num_files*batch_size_tr 
    chunk_size_tr = 720 // num_files

    # chunk_size_tr = 720 
else:
    # chunk size in number of batches
    chunk_size_tr = 720 # 10 days (3 time steps in an hour, 72 in a day)
    
# chunk size in number of elements
# num_samples_per_chunk_tr = chunk_size_tr*batch_size_tr
num_samples_per_chunk_tr = chunk_size_tr*nloc

if use_val:
    batch_size_val = nloc
    if type(val_data_path)==list:  
        num_files = len(tr_data_path)
        batch_size_val = num_files*batch_size_val
        chunk_size_val = 720 // num_files
    else:
        chunk_size_val = 720 # 10 days (3 time steps in an hour, 72 in a day)
        
    # num_samples_per_chunk_val = chunk_size_val*batch_size_val
    num_samples_per_chunk_val = chunk_size_val*nloc


num_workers = 4
prefetch_factor = 1
pin = False
persistent=False


train_data = generator_xy(tr_data_path, nloc=nloc, add_refpres=add_refpres)

train_batch_sampler = BatchSampler(num_samples_per_chunk_tr, 
                                   num_samples=train_data.ntimesteps*nloc, shuffle=shuffle_data)

train_loader = DataLoader(dataset=train_data, num_workers=num_workers, sampler=train_batch_sampler, 
                          batch_size=None,batch_sampler=None,prefetch_factor=prefetch_factor, 
                          pin_memory=pin, persistent_workers=persistent)

if use_val:
    
    val_data = generator_xy(val_data_path, nloc=nloc, add_refpres=add_refpres)

    val_batch_sampler = BatchSampler(num_samples_per_chunk_val, 
                                     num_samples=val_data.ntimesteps*nloc, shuffle=shuffle_data)

    val_loader = DataLoader(dataset=val_data, num_workers=num_workers,sampler=val_batch_sampler,
                            batch_size=None,batch_sampler=None,prefetch_factor=prefetch_factor, 
                            pin_memory=pin, persistent_workers=persistent)


hybi = torch.from_numpy(data.grid_info['hybi'].values).to(device)
hyai = torch.from_numpy(data.grid_info['hyai'].values).to(device)
sp_max = torch.from_numpy(data.input_max['state_ps'].values).to(device)
sp_min = torch.from_numpy(data.input_min['state_ps'].values).to(device)
sp_mean = torch.from_numpy(data.input_mean['state_ps'].values).to(device)


metric_h_con = metrics.get_energy_metric(hyai, hybi)
#loss_fn = my_mse_flatten

loss_fn = metrics.get_hybrid_loss(_lambda)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if not autoregressive:
    timewindow = 1
    timestep_scheduling=False
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

    
class model_train_eval:
    def __init__(self, dataloader, model, batch_size = 384, autoregressive=True, train=True):
        super().__init__()
        self.loader = dataloader
        self.train = train
        self.report_freq = 800
        self.batch_size = batch_size
        self.model = model 
        self.autoregressive = autoregressive
        if self.autoregressive:
            self.model.reset_states()
        self.metric_R2 =  R2Score(num_outputs=ny).to(device) 
        # self.metric_R2_heating =  R2Score().to(device) 
        # self.metric_R2_precc =  R2Score().to(device) 
        # self.metric_R2_moistening =  R2Score().to(device) 

        self.metrics= {'loss': 0, 'mean_squared_error': 0,  # the latter is just MSE
                        'mean_absolute_error': 0, 'R2' : 0, 'R2_heating' : 0,'R2_moistening' : 0,  
                        'R2_precc' : 0, 'R2_lev' : np.zeros((nlev,ny)),
                        'h_conservation' : 0 }

    def eval_one_epoch(self, epoch, timewindow=1):
        report_freq = self.report_freq
        running_loss = 0.0 
        epoch_loss = 0.0
        epoch_mse = 0.0; epoch_mae = 0.0
        epoch_R2precc = 0.0
        epoch_hcon = 0.0
        epoch_r2_lev = 0.0
        t_comp =0 
        if self.autoregressive:
            preds_lay = []; preds_sfc = []
            targets_lay = []; targets_sfc = [] 
            sps = []
        t0_it = time.time()
        j = 0; k = 0; k2=2    
        if self.autoregressive:
            loss_update_start_index = 60
        else:
            loss_update_start_index = 0
        for i,data in enumerate(self.loader):
            inputs_lay_chunks, inputs_sfc_chunks, targets_lay_chunks, targets_sfc_chunks = data
            inputs_lay_chunks   = inputs_lay_chunks.to(device)
            inputs_sfc_chunks   = inputs_sfc_chunks.to(device)
            targets_sfc_chunks  = targets_sfc_chunks.to(device)
            targets_lay_chunks  = targets_lay_chunks.to(device)
            
            inputs_lay_chunks    = torch.split(inputs_lay_chunks, self.batch_size)
            inputs_sfc_chunks    = torch.split(inputs_sfc_chunks, self.batch_size)
            targets_sfc_chunks   = torch.split(targets_sfc_chunks, self.batch_size)
            targets_lay_chunks   = torch.split(targets_lay_chunks, self.batch_size)
         
            # to speed-up IO, we loaded chunks=many batches, which now need to be divided into batches
            for ichunk in range(len(inputs_lay_chunks)):
                inputs_lay = inputs_lay_chunks[ichunk]
                inputs_sfc = inputs_sfc_chunks[ichunk]
                target_lay = targets_lay_chunks[ichunk]
                target_sfc = targets_sfc_chunks[ichunk]
                sp = inputs_sfc[:,0:1] # surface pressure


                tcomp0= time.time()
                    
                if mp_autocast:
                    with torch.autocast(device_type=device.type, dtype=dtype):
                        pred_lay, pred_sfc = self.model(inputs_lay, inputs_sfc)
                else:
                    pred_lay, pred_sfc = self.model(inputs_lay, inputs_sfc)
                    
                if self.autoregressive:
                    # In the autoregressive training case are gathering many time steps before computing loss
                    preds_lay.append(pred_lay)
                    preds_sfc.append(pred_sfc)
                    targets_lay.append(target_lay)
                    targets_sfc.append(target_sfc)
                    sps.append(sp) 
                    
                else:
                    preds_lay = pred_lay
                    preds_sfc = pred_sfc 
                    targets_lay = target_lay
                    targets_sfc = target_sfc
                    sps = sp
                    
                if (not self.autoregressive) or (self.autoregressive and (j+1) % timewindow==0):
            
                    if self.autoregressive:
                        preds_lay   = torch.stack(preds_lay)
                        preds_sfc   = torch.stack(preds_sfc)
                        targets_lay = torch.stack(targets_lay)
                        targets_sfc = torch.stack(targets_sfc)
                        sps         = torch.stack(sps)
                                
                    if mp_autocast:
                        with torch.autocast(device_type=device.type, dtype=dtype):
                            #loss = loss_fn(targets_lay, targets_sfc, preds_lay, preds_sfc)
                            
                            mse = metrics.my_mse_flatten(targets_lay, targets_sfc, preds_lay, preds_sfc)
                            
                            ypo_lay, ypo_sfc = model.postprocessing(preds_lay, preds_sfc)
                            yto_lay, yto_sfc = model.postprocessing(targets_lay, targets_sfc)
                            sps_denorm = sp = sps*(sp_max - sp_min) + sp_mean
                            h_con = metric_h_con(yto_lay, ypo_lay, sps_denorm)
                            
                            loss = loss_fn(mse, h_con)
                    else:
                        #loss = loss_fn(targets_lay, targets_sfc, preds_lay, preds_sfc)
                        
                        mse = metrics.my_mse_flatten(targets_lay, targets_sfc, preds_lay, preds_sfc)
                        ypo_lay, ypo_sfc = model.postprocessing(preds_lay, preds_sfc)
                        yto_lay, yto_sfc = model.postprocessing(targets_lay, targets_sfc)
                        sps_denorm = sp = sps*(sp_max - sp_min) + sp_mean
                        h_con = metric_h_con(yto_lay, ypo_lay, sps_denorm)
                        loss = loss_fn(mse, h_con)
                        
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
                    #mae             = metrics.mean_absolute_error(targets_lay, preds_lay)
                    if j>loss_update_start_index:
                        with torch.no_grad():
                            epoch_loss      += loss.item()
                            #epoch_energy    += energy.item()
                            epoch_mse       += mse.item()
                            #epoch_mae       += mae.item()
                        
                           # yto, ypo =  denorm_func(targets_lay, preds_lay)
                            # -------------- TO-DO:  DE-NORM OUTPUT --------------
                            #yto, ypo = targets_lay, preds_lay

                            epoch_hcon  += h_con.item()
                            # print("shape ypo", ypo_lay.shape, "yto", yto_lay.shape)
                            self.metric_R2.update(ypo_lay.reshape((-1,ny)), yto_lay.reshape((-1,ny)))
                            # self.metric_R2_heating.update(ypo_lay[:,:,0].reshape(-1,1), yto_lay[:,:,0].reshape(-1,1))
                            # self.metric_R2_moistening.update(ypo_lay[:,:,1].reshape(-1,1), yto_lay[:,:,1].reshape(-1,1))

                            # self.metric_R2_precc.update(ypo_sfc[:,3].reshape(-1,1), yto_sfc[:,3].reshape(-1,1))
                            
                            r2_np = np.corrcoef((ypo_sfc.reshape(-1,ny_sfc)[:,3].detach().cpu().numpy(),yto_sfc.reshape(-1,ny_sfc)[:,3].detach().cpu().numpy()))[0,1]
                            epoch_R2precc += r2_np
                            #print("R2 numpy", r2_np, "R2 torch", self.metric_R2_precc(ypo_sfc[:,3:4], yto_sfc[:,3:4]) )

                            ypo_lay = ypo_lay.reshape(-1,nlev,ny).detach().cpu().numpy()
                            yto_lay = yto_lay.reshape(-1,nlev,ny).detach().cpu().numpy()

                            epoch_r2_lev += metrics.corrcoeff_pairs_batchfirst(ypo_lay, yto_lay) 
                           # if track_ks:
                           #     if (j+1) % max(timewindow*4,12)==0:
                           #         epoch_ks += kolmogorov_smirnov(yto,ypo).item()
                           #         k2 += 1
                            k += 1
                    if self.autoregressive:
                        preds_lay = []; preds_sfc = []
                        targets_lay = []; targets_sfc = [] 
                        sps = []
                    if self.autoregressive: 
                        model.detach_states()
                
                t_comp += time.time() - tcomp0
                # # print statistics 
                if j % report_freq == (report_freq-1): # print every 200 minibatches
                    elaps = time.time() - t0_it
                    running_loss = running_loss / (report_freq/timewindow)
                    #running_energy = running_energy / (report_freq/timewindow)
                    
                    r2raw = self.metric_R2.compute()
                    #r2raw_prec = self.metric_R2_precc.compute()

                    #ypo_lay, ypo_sfc = model.postprocessing(preds_lay, preds_sfc)
                    #yto_lay, yto_sfc = model.postprocessing(targets_lay, targets_sfc) 
                    #r2_np = np.corrcoef((ypo_sfc.reshape(-1,ny_sfc)[:,3].detach().cpu().numpy(),yto_sfc.reshape(-1,ny_sfc)[:,3].detach().cpu().numpy()))[0,1]

                   # print(torch.mean(ypo_sfc[:,3] -  yto_sfc[:,3]))
                   # print(torch.mean(preds_sfc[:,3] - targets_sfc[:,3]))
                    

                    print("[{:d}, {:d}] Loss: {:.2e}  runR2: {:.2f},  elapsed {:.1f}s (compute {:.1f})" .format(epoch + 1, 
                                                    j+1, running_loss, r2raw, elaps, t_comp))
                    running_loss = 0.0
                    running_energy = 0.0
                    t0_it = time.time()
                    t_comp = 0
                j += 1

        self.metrics['loss'] =  epoch_loss / k
        self.metrics['mean_squared_error'] = epoch_mse / k
        self.metrics["h_conservation"] =  epoch_hcon / k

        #self.metrics['energymetric'] = epoch_energy / k
        #self.metrics['mean_absolute_error'] = epoch_mae / k
        #self.metrics['ks'] =  epoch_ks / k2
        self.metrics['R2'] = self.metric_R2.compute()
        # self.metrics['R2_heating'] = self.metric_R2_heating.compute()
        # self.metrics['R2_moistening'] = self.metric_R2_moistening.compute()


        self.metrics['R2_heating'] = epoch_r2_lev[:,0].mean() / k
        self.metrics['R2_moistening'] =  epoch_r2_lev[:,1].mean() / k

        #self.metrics['R2_precc'] = self.metric_R2_precc.compute()
        self.metrics['R2_precc'] = epoch_R2precc / k
        
        self.metrics['R2_lev'] = epoch_r2_lev / k

        self.metric_R2.reset() 
        #self.metric_R2_heating.reset(); self.metric_R2_precc.reset()
        if self.autoregressive:
            self.model.reset_states()
        
        datatype = "TRAIN" if self.train else "VAL"
        print('Epoch {} {} loss: {:.2e}  MSE: {:.2e}  h-con:  {:.2e}   R2: {:.2f}  R2-dT/dt: {:.2f}   R2-dq/dt: {:.2f}   R2-precc: {:.3f}'.format(epoch+1, datatype, 
                                                            self.metrics['loss'], 
                                                            self.metrics['mean_squared_error'], 
                                                            self.metrics['h_conservation'],
                                                            self.metrics['R2'],
                                                            self.metrics['R2_heating'],
                                                            self.metrics['R2_moistening'],                                                              
                                                            self.metrics['R2_precc'] ))

    if cuda: torch.cuda.empty_cache()
    gc.collect()


# 160 160
# autoreg, hybrid-loss, 2 years concat

train_runner = model_train_eval(train_loader, model, batch_size_tr, autoregressive, train=True)
if use_val: val_runner = model_train_eval(val_loader, model, batch_size_val, autoregressive, train=False)

for epoch in range(num_epochs):
    t0 = time.time()
    
    if timestep_scheduling:
        timewindoww=timestep_schedule[epoch]            
    else:
        timewindoww=timewindow
        
    print("Epoch {} Training rollout timesteps: {} ".format(epoch+1, timewindoww))
    train_runner.eval_one_epoch(epoch, timewindoww)
    
    if use_wandb: wandb.log(train_runner.metrics)
    
    if use_val:
        if epoch%2:
            print("VALIDATION..")
            val_runner.eval_one_epoch(epoch, timewindoww)

            losses_val = {"val_"+k: v for k, v in val_runner.metrics.items()}
            if use_wandb: wandb.log(losses_val)

            val_loss = losses_val["val_loss"]

            # MODEL CHECKPOINT IF VALIDATION LOSS IMPROVED
            if save_model and val_loss < best_val_loss:
              torch.save({
                          'epoch': epoch,
                          'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'val_loss': val_loss,
                          }, SAVE_PATH)  
              best_val_loss = val_loss 
              
    print('Epoch {}/{} complete, took {:.2f} seconds, autoreg window was {}'.format(epoch+1,num_epochs,time.time() - t0,timewindoww))
    
    
R2 = val_runner.metrics["R2_lev"]

import matplotlib.pyplot as plt


labels = ["dT/dt", "dq/dt", "dqliq/dt", "dqice/dt", "dU/dt", "dV/dt"]
y = np.arange(60)
ncols, nrows = 1,6
fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5.5, 3.5),
                        layout="constrained")
for i in range(6):
    axs[i].plot(R2[:,i],y)
    axs[i].invert_yaxis()
    axs[i].set_xlim(0,1)
    axs[i].set_title(labels[i])
    

x = np.arange(60)
ncols, nrows = 6,1
fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5.5, 3.5),
                        layout="constrained")
for i in range(6):
    axs[i].plot(x, R2[:,i]); 
    axs[i].set_title(labels[i])
    axs[i].set_ylim(0,1)