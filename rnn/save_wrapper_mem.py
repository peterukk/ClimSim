#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:34:47 2025

@author: peter
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Final 
from models import RNN_autoreg, MyRNN, LSTM_autoreg_torchscript
import h5py
from utils import apply_input_norm_numba, cloud_exp_norm_numba, apply_output_norm_numba
import matplotlib.pyplot as plt
import time 


nneur = (128,128)


fdir = "/media/peter/CrucialBX500/data/ClimSim/ClimSim/rnn/saved_models/"
model_path = fdir + "LSTM-None_lr0.001.neur128-128.num17436.pt"
model_path = fdir + "LSTM-None_lr0.001.neur128-128.num17826.pt"
model_path = fdir + "LSTM-None_lr0.001.neur128-128.num56493.pt"
model_path = fdir + "LSTM-None_lr0.001.neur128-128.num81220.pt"
memory = "None"
use_initial_mlp = True
use_intermediate_mlp = False

model_path = fdir + "LSTM-Hidden_lr0.001.neur128-128.num93823.pt"  # nh_mem = 16
use_initial_mlp = True
use_intermediate_mlp = True
memory = "Hidden"

# model_path = fdir + "LSTM-Hidden_lr0.001.neur160-160.num32703.pt"  # nh_mem = 16
# nneur = (160,160)





nx, nx_sfc, ny, ny_sfc = 15, 19, 5, 8

model_type = "LSTM"
autoregressive = True
if memory=="None":
    autoregressive=False
use_memory = autoregressive
separate_radiation = False
concat = False 
add_refpres = False
add_pres = True 
ensemble_size = 1
add_stochastic_lever = False

sfc_vars_remove = (17, 18, 19, 20, 21)


fpath_yscale_lev = "/media/peter/CrucialBX500/data/ClimSim/ClimSim/rnn/saved_models/out_scale_lev.txt"
fpath_yscale_sca = "/media/peter/CrucialBX500/data/ClimSim/ClimSim/rnn/saved_models/out_scale_sca.txt"
fpath_hyam = "/media/peter/CrucialBX500/data/ClimSim/ClimSim/rnn/saved_models/hyam.txt"
fpath_hybm = "/media/peter/CrucialBX500/data/ClimSim/ClimSim/rnn/saved_models/hybm.txt"
fpath_lbd_qn = "/media/peter/CrucialBX500/data/ClimSim/ClimSim/rnn/saved_models/lbd_qn.txt"
fpath_lbd_qc = "/media/peter/CrucialBX500/data/ClimSim/ClimSim/rnn/saved_models/lbd_qc.txt"
fpath_lbd_qi = "/media/peter/CrucialBX500/data/ClimSim/ClimSim/rnn/saved_models/lbd_qi.txt"


fpath_xmean_lev = "/media/peter/CrucialBX500/data/ClimSim/ClimSim/rnn/saved_models/in_mean_lev.txt"
fpath_xmean_sca = "/media/peter/CrucialBX500/data/ClimSim/ClimSim/rnn/saved_models/in_mean_sca.txt"
fpath_xdiv_lev = "/media/peter/CrucialBX500/data/ClimSim/ClimSim/rnn/saved_models/in_div_lev.txt"
fpath_xdiv_sca = "/media/peter/CrucialBX500/data/ClimSim/ClimSim/rnn/saved_models/in_div_sca.txt"

# np.savetxt(fpath_yscale_lev, yscale_lev, fmt='%.18e', delimiter=',')
# np.savetxt(fpath_yscale_sca, yscale_sca, fmt='%.18e', delimiter=',')

# np.savetxt(fpath_hyam, hyam_np, fmt='%.18e', delimiter=',')
# np.savetxt(fpath_hybm, hybm_np, fmt='%.18e', delimiter=',')

# np.savetxt(fpath_lbd_qn, lbd_qn, fmt='%.18e', delimiter=',')
# np.savetxt(fpath_lbd_qc, lbd_qc, fmt='%.18e', delimiter=',')
# np.savetxt(fpath_lbd_qi, lbd_qi, fmt='%.18e', delimiter=',')


# np.savetxt(fpath_xmean_lev, xmean_lev, fmt='%.18e', delimiter=',')
# np.savetxt(fpath_xmean_sca, xmean_sca, fmt='%.18e', delimiter=',')

# np.savetxt(fpath_xdiv_lev, xdiv_lev, fmt='%.18e', delimiter=',')
# np.savetxt(fpath_xdiv_sca, xdiv_sca, fmt='%.18e', delimiter=',')


yscale_lev = np.loadtxt(fpath_yscale_lev, delimiter=",", dtype=np.float32)
yscale_sca = np.loadtxt(fpath_yscale_sca, delimiter=",", dtype=np.float32)

xmean_lev = np.loadtxt(fpath_xmean_lev, delimiter=",", dtype=np.float32)
xmean_sca0 = np.loadtxt(fpath_xmean_sca, delimiter=",", dtype=np.float32)

xdiv_lev = np.loadtxt(fpath_xdiv_lev, delimiter=",", dtype=np.float32)
xdiv_sca0 = np.loadtxt(fpath_xdiv_sca, delimiter=",", dtype=np.float32)

xdiv_sca =  np.delete(xdiv_sca0,sfc_vars_remove)
xmean_sca =  np.delete(xmean_sca0,sfc_vars_remove)


hyam  =  np.loadtxt(fpath_hyam, delimiter=",", dtype=np.float32)
hybm  =  np.loadtxt(fpath_hybm, delimiter=",", dtype=np.float32)

hyam = torch.from_numpy(hyam)
hybm = torch.from_numpy(hybm)

lbd_qn  =  np.loadtxt(fpath_lbd_qn, delimiter=",", dtype=np.float32)
lbd_qc  =  np.loadtxt(fpath_lbd_qc, delimiter=",", dtype=np.float32)
lbd_qi  =  np.loadtxt(fpath_lbd_qi, delimiter=",", dtype=np.float32)


# model = LSTM_autoreg_torchscript(hyam,hybm,
#             out_scale = yscale_lev,
#             out_sfc_scale = yscale_sca, 
#             nx = nx, nx_sfc=nx_sfc, 
#             ny = ny, ny_sfc=ny_sfc, 
#             nneur=nneur, 
#             use_initial_mlp = use_initial_mlp,
#             use_intermediate_mlp=use_intermediate_mlp,
#             add_pres=add_pres,
#             use_memory=use_memory)

# from torchinfo import summary
# infostr = summary(model)
# num_params = infostr.total_params
# print(infostr)

# checkpoint = torch.load(model_path, weights_only=True)
# model.load_state_dict(checkpoint['model_state_dict'])

# model_path_script = 'saved_models/LSTM-Hidden_lr0.001.neur128-128.num66745_script.pt'
# model = torch.jit.load(model_path_script)

model_path_script = 'saved_models/LSTM-Hidden_lr0.001.neur128-128.num68516_script.pt'
model_path_script = "saved_models/LSTM-Hidden_lr0.001.neur96-96.num67132_script.pt"
model_path_script = "saved_models/LSTM-Hidden_lr0.001.neur96-96.num31564_script.pt"
qinput_prune, snowhice_fix, v5_input = True, True, False

model_path_script = "saved_models/LSTM-Hidden_lr0.001.neur144-144.num9826_script.pt"
model = torch.jit.load(model_path_script)
qinput_prune, snowhice_fix, v5_input = True, True, True

model_path_script = "saved_models/LSTM-Hidden_lr0.001.neur128-128.num68516_script.pt"
model = torch.jit.load(model_path_script)
qinput_prune, snowhice_fix, v5_input = True, True, False

model_path_script = "saved_models/LSTM-Hidden_lr0.001.neur96-96_xv4_yv5_num23192_script.pt"
model = torch.jit.load(model_path_script)
qinput_prune, snowhice_fix, v5_input = True, True, False
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model_path_script = "saved_models/LSTM-Hidden_lr0.001.neur160-160.num26608_script.pt"
qinput_prune, snowhice_fix, v5_input,  mp_constraint = True, True, False, True

model_path_script = "saved_models/LSTM-Hidden_lr0.001.neur96-96_xv4_yv4_num83377_script.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint = True, True, False, False

model_path_script = "saved_models/LSTM-Hidden_lr0.001.neur160-160.num44482_script.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint = True, True, False, True

model_path_script = "saved_models/LSTM-Hidden_lr0.001.neur96-96_xv4_yv5_num29116_script.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint = True, True, False, True

model_path_script = "saved_models/LSTM-Hidden_lr0.001.neur96-96_xv4_yv5_num77672_script.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint = True, True, False, True

model = torch.jit.load(model_path_script)

class NewModel_constraint(nn.Module):
    qinput_prune: Final[bool]
    snowhice_fix: Final[bool]
    v5_input: Final[bool]
    mp_constraint: Final[bool]

    def __init__(self, original_model, 
                 lbd_qc, lbd_qi, lbd_qn,
                 qinput_prune, snowhice_fix, v5_input, mp_constraint):
        
        super(NewModel_constraint, self).__init__()
        self.original_model = original_model
        self.lbd_qc     = torch.tensor(lbd_qc, dtype=torch.float32)
        self.lbd_qi     = torch.tensor(lbd_qi, dtype=torch.float32)
        self.lbd_qn     = torch.tensor(lbd_qn, dtype=torch.float32)

        self.hardtanh = nn.Hardtanh(0.0, 1.0)
        self.qinput_prune = qinput_prune
        self.snowhice_fix = snowhice_fix
        self.v5_input = v5_input
        self.mp_constraint = mp_constraint 
        
    def preprocessing(self, x_main0, x_sfc0):
        # v4 input array
        x_main = x_main0.clone()
        x_sfc = x_sfc0.clone()
        
        # for i in range(x_main.shape[-1]):
        #     print("i", i, "min max", np.min(x_main[:,:,i]), np.max(x_main[:,:,i]))
        # for i in range(x_sfc.shape[-1]):
        #     print("i", i, "min max sfc", np.min(x_sfc[:,i]), np.max(x_sfc[:,i]))
        if self.snowhice_fix:
            x_sfc = torch.where(torch.ge(x_sfc,1e10), torch.tensor(-1.0), x_sfc)
            
        if self.v5_input:
            # v5 inputs
            qn   = x_main[:,:,2]  + x_main[:,:,3]
            if self.qinput_prune:
                qn[:,0:15] = 0.0
            qn = torch.exp(-qn * self.lbd_qn)
            x_main[:,:,2] = qn
            liq_frac_constrained  = self.temperature_scaling(x_main[:,:,0])
            x_main[:,:,3] = liq_frac_constrained

            #                            mean     max - min
            # x_main = (x_main - self.xmean_lev)/(self.xdiv_lev)
            # x_sfc =  (x_sfc -  self.xmean_sca)/(self.xdiv_sca)
            x_main = (x_main - self.original_model.xmean_lev)/(self.original_model.xdiv_lev)
            x_sfc =  (x_sfc -  self.original_model.xmean_sca)/(self.original_model.xdiv_sca)
            
        else:
            # v4 inputs
            x_main[:,:,2] = 1 - torch.exp(-x_main[:,:,2] * self.lbd_qc)
            x_main[:,:,3] = 1 - torch.exp(-x_main[:,:,3] * self.lbd_qi)   
            
            #                            mean     max - min
            # x_main = (x_main - self.xmean_lev)/(self.xdiv_lev)
            # x_sfc =  (x_sfc -  self.xmean_sca)/(self.xdiv_sca)
            x_main = (x_main - self.original_model.xmean_lev)/(self.original_model.xdiv_lev)
            x_sfc =  (x_sfc -  self.original_model.xmean_sca)/(self.original_model.xdiv_sca)
            
            if self.qinput_prune:
                x_main[:,0:15,2:3] = 0.0
        # clip RH 
        x_main[:,:,1] = torch.clamp(x_main[:,:,1], 0, 1.2)

        x_main = torch.where(torch.isnan(x_main), torch.tensor(0.0, device=x_main.device), x_main)
        x_main = torch.where(torch.isinf(x_main), torch.tensor(0.0, device=x_main.device), x_main)
        return x_main, x_sfc 
    
    # def postprocessing(self, out_lev, out_sfc):
    #     out_lev[:,0:12,1:] = 0
    #     out_lev      = out_lev / self.yscale_lev
    #     out_sfc     = out_sfc / self.yscale_sca

    #     return out_lev, out_sfc
    
    def temperature_scaling(self, T_raw):
        # T_denorm = T = T*(self.xmax_lev[:,0] - self.xmin_lev[:,0]) + self.xmean_lev[:,0]
        # T_denorm = T*(self.xcoeff_lev[2,:,0] - self.xcoeff_lev[1,:,0]) + self.xcoeff_lev[0,:,0]
        # liquid_ratio = (T_raw - 253.16) / 20.0 
        liquid_ratio = (T_raw - 253.16) * 0.05 
        # liquid_ratio = F.hardtanh(liquid_ratio, 0.0, 1.0)
        liquid_ratio = self.hardtanh(liquid_ratio)

        return liquid_ratio
        
    def forward(self, x_main, x_sfc, rnn1_mem):
        # x_denorm = x_main.clone()
        
        T_before        = x_main[:,:,0:1].clone()
        qliq_before     = x_main[:,:,2:3].clone()
        qice_before     = x_main[:,:,3:4].clone()
        qn_before       = qliq_before + qice_before 
        # print("shape xsfc", x_sfc.shape)

        # print("xmain 0", torch.sum(x_main[200,:,:]))

        x_main, x_sfc = self.preprocessing(x_main, x_sfc)

        out_lev, out_sfc, rnn1_mem = self.original_model(x_main, x_sfc, rnn1_mem)

        out_lev      = out_lev / self.original_model.yscale_lev
        out_sfc      = out_sfc / self.original_model.yscale_sca
        
        nlev_mem = rnn1_mem.shape[1]
        
        if self.mp_constraint:
            T_new           = T_before  + out_lev[:,:,0:1]*1200
            # print("T_new min", T_new.min(), "max", T_new.max())
            
            liq_frac_constrained    = self.temperature_scaling(T_new)
            # liq_frac_constrained    = self.original_model.temperature_scaling(T_new)
    
            # #                            dqn
            qn_new      = qn_before + out_lev[:,:,2:3]*1200  
            qliq_new    = liq_frac_constrained*qn_new
            qice_new    = (1-liq_frac_constrained)*qn_new
            dqliq       = (qliq_new - qliq_before) * 0.0008333333333333334 #/1200  
            dqice       = (qice_new - qice_before) * 0.0008333333333333334 #/1200   
            
            batch_size = out_lev.shape[0] 
            # (nb, nlev, ny) --> (nb, ny, nlev)
            out_lev = torch.transpose(out_lev, 1, 2).reshape(batch_size,300)
    
            yout = torch.zeros((batch_size,368+nlev_mem*16), device=x_main.device)
            yout[:,0:120] = out_lev[:,0:120]
            yout[:,120:180] = torch.reshape(dqliq, (batch_size, 60))
            yout[:,180:240] = torch.reshape(dqice, (batch_size, 60))
            yout[:,240:360] = out_lev[:,180:360]
            yout[:,360:368] = out_sfc
            yout[:,368:] = torch.reshape(rnn1_mem,(-1,rnn1_mem.shape[1]*16))
            
        else:
            batch_size = out_lev.shape[0] 
            out_lev = torch.transpose(out_lev, 1, 2).reshape(batch_size,360)
            yout = torch.zeros((batch_size,368+nlev_mem*16), device=x_main.device)
            yout[:,0:360] = out_lev
            yout[:,360:368] = out_sfc
            yout[:,368:] = torch.reshape(rnn1_mem,(-1,nlev_mem*16))
                      
        yout = torch.where(torch.isnan(yout), torch.tensor(0.0, device=x_main.device), yout)

        # return yout, rnn1_mem
        return yout

        # directly write output so that columns are fastest varying dim,
        # features are slow varying
        # yout = torch.zeros((368+60*16, batch_size))

        # return yout, rnn1_mem
        # return yout


# new_model = NewModel(model, xmean_lev, xmean_sca, 
#                     xdiv_lev, xdiv_sca,
#                     yscale_lev, yscale_sca, 
#                     lbd_qc, lbd_qi)


new_model = NewModel_constraint(model, lbd_qc, lbd_qi, lbd_qn, 
                                qinput_prune, snowhice_fix, v5_input, mp_constraint)

device = torch.device("cpu")
new_model = new_model.to(device)

scripted_model = torch.jit.script(new_model)
scripted_model = scripted_model.eval()


# save_file_torch = "v4_rnn-memory_wrapper_constrained_huber_160.pt"
# save_file_torch = "v4_rnn-memory_wrapper_constrained_huber_160.pt"
# save_file_torch = "wrappers/v4_rnn-memory_wrapper_constrained_huber_energy_num26608_160.pt"  
save_file_torch = "wrappers/" + model_path_script.split("/")[1].split("_script.pt")[0] + ".pt"
print("saving to ", save_file_torch)
# scripted_model.save(save_file_torch)

h

# fpath_data = '/media/peter/samsung/data/ClimSim_low_res_expanded/data_v4_rnn_nonorm_year8.h5'
fpath_data = '/media/peter/CrucialBX500/data/ClimSim/low_res_expanded/data_v4_rnn_nonorm_year8_nocompress_chunk3.h5'
# (26278, 384, 60, 15)

hf = h5py.File(fpath_data, 'r')
bsize = 384 
# nb = 2160
nb = 1400

nlev = 60
# nlev = 50

offset = 0
# offset = 10000*bsize
offset = 22000 

ns = nb * bsize
# x_lev_np = hf['input_lev'][offset:offset+ns]
# x_sfc_np = hf['input_sca'][offset:offset+ns]
# x_sfc_np = np.delete(x_sfc_np,sfc_vars_remove,axis=1)
# y_lev_np = hf['output_lev'][offset:offset+ns]
# y_sfc_np = hf['output_sca'][offset:offset+ns]

x_lev_np = hf['input_lev'][offset:offset+nb]
x_sfc_np = hf['input_sca'][offset:offset+nb]
x_sfc_np = np.delete(x_sfc_np,sfc_vars_remove,axis=2)
y_lev_np = hf['output_lev'][offset:offset+nb]
y_sfc_np = hf['output_sca'][offset:offset+nb]

hf.close()

y_lev = np.copy(y_lev_np); y_sfc = np.copy(y_sfc_np)
# v5 outputs: qn instead of qi,qc
# eval_v5_outputs = True
# if eval_v5_outputs:
#     # y_lev[:,:,2] = y_lev[:,:,2] + y_lev[:,:,3] 
#     # y_lev = np.delete(y_lev, 3, axis=2) 
#     y_lev[:,:,:,2] = y_lev[:,:,:,2] + y_lev[:,:,:,3] 
#     y_lev = np.delete(y_lev, 3, axis=3) 

xlev = torch.from_numpy(x_lev_np)
xsfc = torch.from_numpy(x_sfc_np)


nrep = 10
t0 = time.time()
j = 0 

# for i in range(nrep):
#     # out_test = scripted_model(xlev,xsfc)
#     out_test = scripted_model(xlev,xsfc)

# print("Inferencex100 took {:.2f}s".format(time.time() - t0))

ntime = nb 
j = 0 
rnn1_mem = torch.zeros((bsize, nlev, 16))

outs = []
for jj in range(ntime):
    jend = j + bsize
    # out_test = scripted_model(xlev,xsfc)
    # x0 = xlev[j:jend]
    # x1 = xsfc[j:jend]
    x0 = xlev[jj,:]
    x1 = xsfc[jj,:]
    # print(rnn1_mem[0,0,0], rnn1_mem[4,30,10])
    
    with torch.no_grad(): 
        # out_test, rnn1_mem = scripted_model(x0,x1,rnn1_mem)
        out_test = scripted_model(x0,x1,rnn1_mem)
        rnn1_mem = torch.reshape(out_test[:,368:], (bsize,nlev,16))
        outs.append(out_test[:,0:368])
    j = j + bsize
    
outs = torch.stack(outs)
outs_lev = outs[:,:,0:360].reshape(-1,384,6,60).transpose(2,3).detach().numpy()
outs_sfc = outs[:,:,360:368].detach().numpy()
# y_lev_np = y_lev_np.reshape(-1,60,6)
import gc 
gc.collect()
# for j in range(368):
#     print("max j", j, out_test[:,j].max())

# out_lev, out_sfc = scripted_model(xlev,xsfc)

# with torch.no_grad(): 

#     out_lev, out_sfc = new_model(xlev,xsfc)

# out_lev = torch.transpose(out_test[:,0:360].reshape(-1,6,60), 1, 2)
# out_lev = torch.transpose(outs[:,0:360].reshape(-1,6,60), 1, 2)

# out_lev_np = out_lev.detach().numpy()
# out_sfc_np = out_sfc.detach().numpy()


R2 = np.zeros((60,6))
bias = np.zeros((60,6))

for i in range(6): 
    for j in range(60):
        # R2[j,i] = np.corrcoef(y_lev_np[:,j,i].flatten(),out_lev_np[:,j,i].flatten())[0,1]
        # R2[j,i] = np.corrcoef(y_lev_np[ns-bsize:ns,j,i].flatten(),out_lev_np[:,j,i].flatten())[0,1]
        # truth, pred = y_lev_np[:,j,i].flatten(),out_lev_np[:,j,i].flatten()
        truth, pred = y_lev_np[:,:,j,i].flatten(),outs_lev[:,:,j,i].flatten()

        R2[j,i] = np.corrcoef(truth, pred)[0,1]**2
        bias[j,i] = np.nanmean(truth - pred)
        
R2[np.isnan(R2)] = 0


labels = ["dT/dt", "dq/dt", "dqliq/dt", "dqice/dt", "dU/dt", "dV/dt"]

x = np.arange(60)
ncols, nrows = 6,1


fig, axs = plt.subplots(ncols=nrows, nrows=ncols, figsize=(5.5, 3.5)) #layout="constrained")
for i in range(6):
    axs[i].plot(x, bias[:,i]); 
    axs[i].set_title(labels[i])
    # axs[i].set_ylim(0,1)
    axs[i].set_xlim(0,60)
    axs[i].axvspan(0, 30, facecolor='0.2', alpha=0.2)
    # axs[i].set_yticklabels([])
    # axs[i].set_xticklabels([])

fig.subplots_adjust(hspace=0.6)


    

# fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5.5, 3.5),
#                         gridspec_kw = {'wspace':0}) #layout="constrained")
# for i in range(6):
#     axs[i].plot(x, R2[:,i]); 
#     axs[i].set_title(labels[i])
#     axs[i].set_ylim(0,1)
#     axs[i].set_xlim(0,60)
#     axs[i].axvspan(0, 30, facecolor='0.2', alpha=0.2)
#     if i>0:
#         axs[i].set_yticklabels([])
#     axs[i].set_xticklabels([])

# fig.subplots_adjust(hspace=0)


# fig.suptitle("Bias for val first 2160 batches")

# y_true = y_lev_np.reshape(-1,384,60,6)
# y_pred = outs_lev.reshape(-1,384,60,6)
y_true = y_lev_np 
y_pred = outs_lev

dt_t_mean = y_true[:,:,:,0].mean(axis=0)
dt_p_mean = y_pred[:,:,:,0].mean(axis=0)
q_t_mean = y_true[:,:,:,1].mean(axis=0)
q_p_mean = y_pred[:,:,:,1].mean(axis=0)
clw_t_mean = y_true[:,:,:,2].mean(axis=0)
clw_p_mean = y_pred[:,:,:,2].mean(axis=0)
cli_t_mean = y_true[:,:,:,3].mean(axis=0)
cli_p_mean = y_pred[:,:,:,3].mean(axis=0)
u_t_mean = y_true[:,:,:,4].mean(axis=0)
u_p_mean = y_pred[:,:,:,4].mean(axis=0)
v_t_mean = y_true[:,:,:,5].mean(axis=0)
v_p_mean = y_pred[:,:,:,5].mean(axis=0)

colors = ['b','g']
fig, ax1 = plt.subplots()
ax1.hist([clw_t_mean.flatten(),clw_p_mean.flatten()],color=colors, label=['clw', 'clw-pred'])
ax1.set_yscale('log')
ax1.legend()

colors = ['b','g']
fig, ax2 = plt.subplots()
ax2.hist([clw_t_mean.flatten(),clw_p_mean.flatten()],color=colors, label=['cli', 'cli-pred'])
ax2.set_yscale('log')
ax2.legend()

# import mpl_scatter_density # adds projection='scatter_density'
# from matplotlib.colors import LinearSegmentedColormap

# # "Viridis-like" colormap with white background
# white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
#     (0, '#ffffff'),
#     (1e-20, '#440053'),
#     (0.2, '#404388'),
#     (0.4, '#2a788e'),
#     (0.6, '#21a784'),
#     (0.8, '#78d151'),
#     (1, '#fde624'),
# ], N=256)

# def using_mpl_scatter_density(fig, x, y):
#     ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
#     density = ax.scatter_density(x, y, cmap=white_viridis)
#     fig.colorbar(density, label='Number of points per pixel')

# fig = plt.figure()
# using_mpl_scatter_density(fig, x, y)



import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import glob

data_path = '/media/peter/CrucialBX500/data/ClimSim/hu_etal2024_data/'

ds_grid = xr.open_dataset(data_path+'data_grid/ne4pg2_scrip.nc')
grid_area = ds_grid['grid_area']

def zonal_mean_area_weighted(data, grid_area, lat):
    # Define latitude bins ranging from -90 to 90, each bin spans 10 degrees
    bins = np.arange(-90, 91, 10)  # Create edges for 10 degree bins

    # Get indices for each lat value indicating which bin it belongs to
    bin_indices = np.digitize(lat.values, bins) - 1

    # Initialize a list to store the zonal mean for each latitude bin
    data_zonal_mean = []

    # Iterate through each bin to calculate the weighted average
    for i in range(len(bins)-1):
        # Filter data and grid_area for current bin
        mask = (bin_indices == i)
        data_filtered = data[mask]
        grid_area_filtered = grid_area[mask]

        # Check if there's any data in this bin
        if data_filtered.size > 0:
            # Compute area-weighted average for the current bin
            weighted_mean = np.average(data_filtered, axis=0, weights=grid_area_filtered)
        else:
            # If no data in bin, append NaN or suitable value
            weighted_mean = np.nan

        # Append the result to the list
        data_zonal_mean.append(weighted_mean)

    # Convert list to numpy array
    data_zonal_mean = np.array(data_zonal_mean)

    # The mid points of the bins are used as the representative latitudes
    lats_mid = bins[:-1] + 5

    return data_zonal_mean, lats_mid

ds2 = xr.open_dataset(data_path+'data_grid/E3SM_ML.GNUGPU.F2010-MMF1.ne4pg2_ne4pg2.eam.h0.0001-01.nc')
lat = ds2.lat
lon = ds2.lon
level = ds2.lev.values


y = np.arange(60)
ncols, nrows = 3,2
fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5.5, 3.5),
                        layout="constrained")
j = 0
for irow in range(2):
    for icol in range(3):
        axs[irow,icol].plot(R2[:,j],level)
        axs[irow,icol].invert_yaxis()
        axs[irow,icol].set_xlim(0,1)
        axs[irow,icol].set_title(labels[j])
        j = j + 1
    
fig.subplots_adjust(hspace=0)


def zonal_mean(var):
    var_re = var.reshape(-1,384,var.shape[-1])
    var_re = np.transpose(var_re, (1,0,2))
    var_zonal_mean, lats_sorted = zonal_mean_area_weighted(var_re, grid_area, lat)
    return var_zonal_mean, lats_sorted


variables = [
    {'var': 'T', 'var_title': 'T', 'scaling': 1., 'unit': 'K', 'diff_scale': 0.9, 'max_diff': 5},
    {'var': 'Q', 'var_title': 'Q', 'scaling': 1000., 'unit': 'g/kg', 'diff_scale': 1, 'max_diff': 1},
    {'var': 'U', 'var_title': 'U', 'scaling': 1., 'unit': 'm/s', 'diff_scale': 0.2, 'max_diff': 4},
    {'var': 'CLDLIQ', 'var_title': 'Liquid cloud', 'scaling': 1e6, 'unit': 'mg/kg', 'diff_scale': 1, 'max_diff': 40},
    {'var': 'CLDICE', 'var_title': 'Ice cloud', 'scaling': 1e6, 'unit': 'mg/kg', 'diff_scale': 1, 'max_diff': 5}
]

vars_stacked = [[dt_t_mean,dt_p_mean], 
                [q_t_mean,q_p_mean],
                # [u_t_mean,u_p_mean], 
                [v_t_mean,v_p_mean],
                [clw_t_mean,clw_p_mean],
                [cli_t_mean,cli_p_mean]]


# labels=["Heating","U","V","Cloud water", "cloud ice"]
# scalings = [1,1,1,1e6,1e6]

labels=["Heating","Moistening","V","Cloud water", "cloud ice"]
scalings = [1,1000,1,1e6,1e6]

# max_diffs = [40,5]

latitude_ticks = [-60, -30, 0, 30, 60]
latitude_labels = ['60S', '30S', '0', '30N', '60N']

fig, axs = plt.subplots(len(vars_stacked), 3, figsize=(14, 12.5)) 

for idx in range(len(vars_stacked)):
    var_t, var_p = vars_stacked[idx]
    
    sp_zm, lats_sorted = zonal_mean_area_weighted(var_t, grid_area, lat)
    nn_zm, lats_sorted = zonal_mean_area_weighted(var_p, grid_area, lat)
    
    # data_sp, data_nn = 1e6*sp_zm.T, 1e6*nn_zm.T
    
    scaling = scalings[idx]
    data_sp = scaling * xr.DataArray(sp_zm[:, :].T, dims=["hybrid pressure (hPa)", "latitude"],
                                     coords={"hybrid pressure (hPa)": level, "latitude": lats_sorted})
    data_nn = scaling * xr.DataArray(nn_zm[:, :].T, dims=["hybrid pressure (hPa)", "latitude"],
                                     coords={"hybrid pressure (hPa)": level, "latitude": lats_sorted})
    data_diff = data_nn - data_sp
        
    # Determine color scales
    vmax = max(abs(data_sp).max(), abs(data_nn).max())
    vmin = min(abs(data_sp).min(), abs(data_nn).min())
    # if var_info['diff_scale']:
    #     vmax_diff = abs(data_diff).max() * diff_scale
    #     vmin_diff = -vmax_diff
    # vmax_diff =  max_diffs[idx]
    # vmin_diff = -vmax_diff
    vmax_diff = max(abs(data_diff).max(), abs(data_diff).max())
    vmin_diff = min(abs(data_diff).min(), abs(data_diff).min())
    # Plot each variable in its row
    
    
    data_sp.plot(ax=axs[idx, 0], add_colorbar=True, cmap='viridis', vmin=vmin, vmax=vmax)
    # axs[idx, 0].set_title(f'{labels[idx * 3]} {var_title} ({unit}): MMF')
    axs[idx, 0].set_title("{} , {} ".format(labels[idx],'MMF'))
    axs[idx, 0].invert_yaxis()
    
    data_nn.plot(ax=axs[idx, 1], add_colorbar=True, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[idx, 1].set_title("{} , {} ".format(labels[idx],'RNN'))
    axs[idx, 1].invert_yaxis()
    axs[idx, 1].set_ylabel('')  # Clear the y-label to clean up plot
    
    data_diff.plot(ax=axs[idx, 2], add_colorbar=True, cmap='RdBu_r', vmin=vmin_diff, vmax=vmax_diff)
    axs[idx, 2].set_title("{} , {} ".format(labels[idx],'diff'))
    axs[idx, 2].invert_yaxis()
    axs[idx, 2].set_ylabel('')  # Clear the y-label to clean up plot
    
    # axs[idx, 0].set_xlabel('')
    # axs[idx, 1].set_xlabel('')
    # axs[idx, 2].set_xlabel('')

# Set these ticks and labels for each subplot
for ax_row in axs:
    for ax in ax_row:
        ax.set_xticks(latitude_ticks)  # Set the positions for the ticks
        ax.set_xticklabels(latitude_labels)  # Set the custom text labels
plt.tight_layout()
plt.show() 



# zonal plots of BIAS, MAE 
qn_true = y_true[:,:,:,2]  + y_true[:,:,:,3]
qn_pred = y_pred[:,:,:,2]  + y_pred[:,:,:,3]

mae_dt = np.mean(np.abs(y_true[:,:,:,0] - y_pred[:,:,:,0]),axis=0)
mae_dq = np.mean(np.abs(y_true[:,:,:,1] - y_pred[:,:,:,1]),axis=0)
mae_clw = np.mean(np.abs(y_true[:,:,:,2] - y_pred[:,:,:,2]),axis=0)
mae_cli = np.mean(np.abs(y_true[:,:,:,3] - y_pred[:,:,:,3]),axis=0)
mae_qn = np.mean(np.abs(qn_true - qn_pred),axis=0)

R2_dt = np.zeros((384,60))
R2_dq = np.zeros((384,60))
R2_clw = np.zeros((384,60))
R2_cli = np.zeros((384,60))
R2_qn = np.zeros((384,60))

for i in range(384): 
    for j in range(60):
        R2_dt[i,j] =  np.corrcoef(y_true[:,i,j,0], y_pred[:,i,j,0])[0,1]**2
        R2_dq[i,j] =  np.corrcoef(y_true[:,i,j,1], y_pred[:,i,j,1])[0,1]**2
        R2_clw[i,j] =  np.corrcoef(y_true[:,i,j,2], y_pred[:,i,j,2])[0,1]**2
        R2_cli[i,j] =  np.corrcoef(y_true[:,i,j,3], y_pred[:,i,j,3])[0,1]**2
        R2_qn[i,j] =  np.corrcoef(qn_true[:,i,j], qn_pred[:,i,j])[0,1]**2


vars_stacked2 = [[mae_dt,R2_dt], 
                 [mae_dq,R2_dq], 
                [mae_clw,R2_clw], 
                [mae_cli,R2_cli],
                 [mae_qn,R2_qn]]

labels2=["Heating", "Moistening", "Cloud water", "cloud ice", "qn"]

fig, axs = plt.subplots(len(vars_stacked2), 2, figsize=(14, 12.5)) 

for idx in range(len(vars_stacked2)):
    var_mae, var_r2 = vars_stacked2[idx]
    
    mae_zm, lats_sorted = zonal_mean_area_weighted(var_mae, grid_area, lat)
    r2_zm, lats_sorted = zonal_mean_area_weighted(var_r2, grid_area, lat)
    
    # data_sp, data_nn = 1e6*sp_zm.T, 1e6*nn_zm.T
    
    scaling = 1
    data_mae = scaling * xr.DataArray(mae_zm[:, :].T, dims=["hybrid pressure (hPa)", "latitude"],
                                     coords={"hybrid pressure (hPa)": level, "latitude": lats_sorted})
    data_r2 = scaling * xr.DataArray(r2_zm[:, :].T, dims=["hybrid pressure (hPa)", "latitude"],
                                     coords={"hybrid pressure (hPa)": level, "latitude": lats_sorted})
        
    # Determine color scales
    vmax1 = data_mae.max()
    vmin1 = data_mae.min()
    vmax2 = 1.0
    vmin2 = 0.0

    
    data_mae.plot(ax=axs[idx, 0], add_colorbar=True, cmap='plasma', vmin=vmin1, vmax=vmax1)
    # axs[idx, 0].set_title(f'{labels[idx * 3]} {var_title} ({unit}): MMF')
    axs[idx, 0].set_title("{} , {} ".format(labels2[idx],'MAE'))
    axs[idx, 0].invert_yaxis()
    
    data_r2.plot(ax=axs[idx, 1], add_colorbar=True, cmap='plasma', vmin=vmin2, vmax=vmax2)
    axs[idx, 1].set_title("{} , {} ".format(labels2[idx],'R**2'))
    axs[idx, 1].invert_yaxis()
    axs[idx, 1].set_ylabel('')  # Clear the y-label to clean up plot
    

    # axs[idx, 0].set_xlabel('')
    # axs[idx, 1].set_xlabel('')
    # axs[idx, 2].set_xlabel('')

# Set these ticks and labels for each subplot
for ax_row in axs:
    for ax in ax_row:
        ax.set_xticks(latitude_ticks)  # Set the positions for the ticks
        ax.set_xticklabels(latitude_labels)  # Set the custom text labels
plt.tight_layout()
plt.show() 

