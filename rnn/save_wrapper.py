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
# from models import RNN_autoreg, MyRNN, LSTM_autoreg_torchscript
import h5py
from utils import apply_input_norm_numba, cloud_exp_norm_numba, apply_output_norm_numba
import matplotlib.pyplot as plt
import time 




fdir = "/media/peter/CrucialBX500/data/ClimSim/ClimSim/rnn/saved_models/"
model_path = fdir + "LSTM-None_lr0.001.neur128-128.num17436.pt"
model_path = fdir + "LSTM-None_lr0.001.neur128-128.num17826.pt"
model_path = fdir + "LSTM-None_lr0.001.neur128-128.num56493.pt"
model_path = fdir + "LSTM-None_lr0.001.neur128-128.num81220.pt"
memory = "None"
use_initial_mlp = True
use_intermediate_mlp = False

# model_path = fdir + "LSTM-Hidden_lr0.001.neur128-128.num93033.pt"  # nh_mem = 16
# use_initial_mlp = True
# use_intermediate_mlp = True
# memory = "Hidden"



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
nneur = (128,128)

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
# # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
class NewModel_constraint(nn.Module):
    qinput_prune: Final[bool]
    snowhice_fix: Final[bool]
    v5_input: Final[bool]
    mp_constraint: Final[bool]
    is_stochastic: Final[bool]
    return_det: Final[bool]
    
    def __init__(self, original_model, 
                 lbd_qc, lbd_qi, lbd_qn,
                 qinput_prune, snowhice_fix, v5_input, mp_constraint, 
                 is_stochastic, return_det):
        
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
        self.is_stochastic = is_stochastic
        self.return_det = return_det
        
        self.xmean_lev      = self.original_model.xmean_lev.to("cpu")
        self.xdiv_lev       = self.original_model.xdiv_lev.to("cpu")
        self.xmean_sca      = self.original_model.xmean_sca.to("cpu")
        self.xdiv_sca       = self.original_model.xdiv_sca.to("cpu")
        self.yscale_lev     = self.original_model.yscale_lev.to("cpu")
        self.yscale_sca     = self.original_model.yscale_sca.to("cpu")

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
            qn = 1 - torch.exp(-qn * self.lbd_qn)
            x_main[:,:,2] = qn
            liq_frac_constrained  = self.temperature_scaling(x_main[:,:,0])
            x_main[:,:,3] = liq_frac_constrained

            #                            mean     max - min
            # x_main = (x_main - self.xmean_lev)/(self.xdiv_lev)
            # x_sfc =  (x_sfc -  self.xmean_sca)/(self.xdiv_sca)
            x_main = (x_main - self.xmean_lev)/(self.xdiv_lev)
            x_sfc =  (x_sfc -  self.xmean_sca)/(self.xdiv_sca)
            
            # if self.qinput_prune:
            #     x_main[:,0:15,2] = 0.0
                
        else:
            # v4 inputs
            x_main[:,:,2] = 1 - torch.exp(-x_main[:,:,2] * self.lbd_qc)
            x_main[:,:,3] = 1 - torch.exp(-x_main[:,:,3] * self.lbd_qi)   
            
            #                            mean     max - min
            # x_main = (x_main - self.xmean_lev)/(self.xdiv_lev)
            # x_sfc =  (x_sfc -  self.xmean_sca)/(self.xdiv_sca)
            x_main = (x_main - self.xmean_lev)/(self.xdiv_lev)
            x_sfc =  (x_sfc -  self.xmean_sca)/(self.xdiv_sca)
            
            if self.qinput_prune:
                x_main[:,0:15,2:3] = 0.0
        # clip RH 
        x_main[:,:,1] = torch.clamp(x_main[:,:,1], 0, 1.2)

        x_main = torch.where(torch.isnan(x_main), torch.tensor(0.0, device=x_main.device), x_main)
        x_main = torch.where(torch.isinf(x_main), torch.tensor(0.0, device=x_main.device), x_main)
        return x_main, x_sfc 
    
    
    def temperature_scaling(self, T_raw):
        # T_denorm = T = T*(self.xmax_lev[:,0] - self.xmin_lev[:,0]) + self.xmean_lev[:,0]
        # T_denorm = T*(self.xcoeff_lev[2,:,0] - self.xcoeff_lev[1,:,0]) + self.xcoeff_lev[0,:,0]
        # liquid_ratio = (T_raw - 253.16) / 20.0 
        liquid_ratio = (T_raw - 253.16) * 0.05 
        # liquid_ratio = F.hardtanh(liquid_ratio, 0.0, 1.0)
        liquid_ratio = self.hardtanh(liquid_ratio)

        return liquid_ratio
    
    def mp_postprocessing(self, T_before, qn_before, qliq_before, qice_before,
                          out_lev, out_sfc):
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

        yout = torch.zeros((batch_size,368), device=T_before.device)
        yout[:,0:120] = out_lev[:,0:120]
        yout[:,120:180] = torch.reshape(dqliq, (batch_size, 60))
        yout[:,180:240] = torch.reshape(dqice, (batch_size, 60))
        yout[:,240:360] = out_lev[:,180:360]
        yout[:,360:368] = out_sfc
        return yout
    
    def forward(self, x_main, x_sfc):
        # x_denorm = x_main.clone()
        
        T_before        = x_main[:,:,0:1].clone()
        qliq_before     = x_main[:,:,2:3].clone()
        qice_before     = x_main[:,:,3:4].clone()
        qn_before       = qliq_before + qice_before 
        # print("shape xsfc", x_sfc.shape)

        # print("xmain 0", torch.sum(x_main[200,:,:]))

        x_main, x_sfc = self.preprocessing(x_main, x_sfc)

        if self.is_stochastic:
            out_lev, out_sfc, out_lev_det = self.original_model(x_main, x_sfc)
            out_lev_det      = out_lev_det / self.yscale_lev

        else:
            out_lev, out_sfc = self.original_model(x_main, x_sfc)

        out_lev      = out_lev / self.yscale_lev
        out_sfc      = out_sfc / self.yscale_sca
                
        if self.mp_constraint:
            yout = self.mp_postprocessing(T_before, qn_before, qliq_before, qice_before, 
                                  out_lev, out_sfc)
            
            if self.is_stochastic and self.return_det:
                yout_det = self.mp_postprocessing(T_before, qn_before, qliq_before, qice_before, 
                                      out_lev_det, out_sfc)
            
        else:
            batch_size = out_lev.shape[0] 
            out_lev = torch.transpose(out_lev, 1, 2).reshape(batch_size,360)
            yout = torch.zeros((batch_size,368), device=x_main.device)
            yout[:,0:360] = out_lev
            yout[:,360:368] = out_sfc
                      
        yout = torch.where(torch.isnan(yout), torch.tensor(0.0, device=x_main.device), yout)
        if self.is_stochastic and self.return_det:
            yout_det = torch.where(torch.isnan(yout_det), torch.tensor(0.0, device=x_main.device), yout_det)
            return yout, yout_det 
        else:
            return yout
    
model_path_script = "saved_models/LSTM-None_lr0.0004.neur144-144_xv4_yv5_num49430_script.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint, is_stochastic, return_det = True, True, False, True, False, False

model_path_script = "saved_models/LSTM-None_lr0.0004.neur144-144_xv4_yv5_num91940_script.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint, is_stochastic, return_det = True, True, False, True, False, False

model_path_script = "saved_models/LSTM-None_lr0.0004.neur160-160_xv4_yv5_num22056_script.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint, is_stochastic, return_det = True, True, False, True, False, False

model_path_script = "saved_models/LSTM-None_lr0.0004.neur144-144_xv4_yv5_num3150_script.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint, is_stochastic, return_det = True, True, False, True, False, False



existing_wrapper=True

# new_model = NewModel(model, xmean_lev, xmean_sca, 
#                     xdiv_lev, xdiv_sca,
#                     yscale_lev, yscale_sca, 
#                     lbd_qc, lbd_qi)

# NewModel.device = "cpu"
# device = torch.device("cpu")

# scripted_model = torch.jit.script(new_model)
# scripted_model = scripted_model.eval()

# if existing_wrapper:
model = torch.jit.load(model_path_script)

new_model = NewModel_constraint(model, lbd_qc, lbd_qi, lbd_qn, 
                                qinput_prune, snowhice_fix, v5_input, mp_constraint, 
                                is_stochastic, return_det)

device = torch.device("cpu")
new_model = new_model.to(device)

scripted_model = torch.jit.script(new_model)
scripted_model = scripted_model.eval()


# save_file_torch = "v4_rnn-memory_wrapper_constrained_huber_160.pt"
# save_file_torch = "v4_rnn-memory_wrapper_constrained_huber_160.pt"
# save_file_torch = "wrappers/v4_rnn-memory_wrapper_constrained_huber_energy_num26608_160.pt"  
save_file_torch = "wrappers/" + model_path_script.split("/")[1].split("_script.pt")[0] + ".pt"
print("saving to ", save_file_torch)
# 
scripted_model.save(save_file_torch)
f



fpath_data = '/media/peter/samsung/data/ClimSim_low_res_expanded/data_v4_rnn_nonorm_year8.h5'

hf = h5py.File(fpath_data, 'r')
bsize = 384 
nb = 10
x_lev_np = hf['input_lev'][0:nb*bsize]
x_sfc_np = hf['input_sca'][0:nb*bsize]
x_sfc_np = np.delete(x_sfc_np,sfc_vars_remove,axis=1)

x_lev = np.copy(x_lev_np)
x_sfc = np.copy(x_sfc_np)

y_lev_np = hf['output_lev'][0:nb*bsize]
y_sfc_np = hf['output_sca'][0:nb*bsize]
hf.close()

y_lev = np.copy(y_lev_np); y_sfc = np.copy(y_sfc_np)
y_lev[:,:,2] = y_lev[:,:,2] + y_lev[:,:,3] 
y_lev = np.delete(y_lev, 3, axis=2) 

xlev = torch.from_numpy(x_lev)
xsfc = torch.from_numpy(x_sfc)

nrep = 10
t0 = time.time()

for i in range(nrep):
    out_test = scripted_model(xlev,xsfc)

print("Inferencex100 took {:.2f}s".format(time.time() - t0))

out_test = scripted_model(xlev,xsfc)

# for j in range(368):
#     print("max j", j, out_test[:,j].max())

# out_lev, out_sfc = scripted_model(xlev,xsfc)

# with torch.no_grad(): 

#     out_lev, out_sfc = new_model(xlev,xsfc)

out_lev = torch.transpose(out_test[:,0:360].reshape(-1,6,60), 1, 2)
out_lev_np = out_lev.detach().numpy()
# out_sfc_np = out_sfc.detach().numpy()


R2 = np.zeros((60,6))

for i in range(6): 
    for j in range(60):
        R2[j,i] = np.corrcoef(y_lev_np[:,j,i].flatten(),out_lev_np[:,j,i].flatten())[0,1]
        
R2[np.isnan(R2)] = 0


labels = ["dT/dt", "dq/dt", "dqliq/dt", "dqice/dt", "dU/dt", "dV/dt"]

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

save_file_torch = "v4_rnn_wrapper_constrained_huber.pt"
scripted_model.save(save_file_torch)
