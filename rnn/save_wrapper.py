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


model = LSTM_autoreg_torchscript(hyam,hybm,
            out_scale = yscale_lev,
            out_sfc_scale = yscale_sca, 
            nx = nx, nx_sfc=nx_sfc, 
            ny = ny, ny_sfc=ny_sfc, 
            nneur=nneur, 
            use_initial_mlp = use_initial_mlp,
            use_intermediate_mlp=use_intermediate_mlp,
            add_pres=add_pres,
            use_memory=use_memory)

from torchinfo import summary
infostr = summary(model)
num_params = infostr.total_params
print(infostr)

checkpoint = torch.load(model_path, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class NewModel(nn.Module):
    def __init__(self, original_model, 
                 xmean_lev, xmean_sca, 
                 xdiv_lev, xdiv_sca,
                 yscale_lev, yscale_sca, 
                 # lbd_qn, 
                 lbd_qc, lbd_qi):
        
        super(NewModel, self).__init__()
        self.original_model = original_model
        self.xmean_lev  = torch.tensor(xmean_lev, dtype=torch.float32)
        self.xmean_sca  = torch.tensor(xmean_sca, dtype=torch.float32)
        self.xdiv_lev   = torch.tensor(xdiv_lev, dtype=torch.float32)
        self.xdiv_sca   = torch.tensor(xdiv_sca, dtype=torch.float32)
        self.yscale_lev = torch.tensor(yscale_lev, dtype=torch.float32)
        self.yscale_sca   = torch.tensor(yscale_sca, dtype=torch.float32)
        # self.lbd_qn     = torch.tensor(lbd_qn, dtype=torch.float32)
        self.lbd_qc     = torch.tensor(lbd_qc, dtype=torch.float32)
        self.lbd_qi     = torch.tensor(lbd_qi, dtype=torch.float32)

    # def temperature_scaling(self, T_raw):
    #     # T_denorm = T = T*(self.xmax_lev[:,0] - self.xmin_lev[:,0]) + self.xmean_lev[:,0]
    #     # T_denorm = T*(self.xcoeff_lev[2,:,0] - self.xcoeff_lev[1,:,0]) + self.xcoeff_lev[0,:,0]
    #     # liquid_ratio = (T_raw - 253.16) / 20.0 
    #     liquid_ratio = (T_raw - 253.16) * 0.05 
    #     liquid_ratio = F.hardtanh(liquid_ratio, 0.0, 1.0)
    #     return liquid_ratio
    
    # def preprocessing(self, x_main, x_sfc):
    #     # convert v4 input array to v5 input array:
    #     # ['state_t',
    #     # 'state_rh',
    #     # 'state_q0002' = qliq   -->  qn 
    #     # 'state_q0003' = qice   --> liquid ratio
    #     x_main[:,:,2]   =  x_main[:,:,2] +  x_main[:,:,3]  
    #     x_main[:,:,3]   = self.temperature_scaling(x_main[:,:,0])      

    #     #                            mean     max - min
    #     x_main = (x_main - self.xmean_lev)/(self.xdiv_lev)
    #     x_sfc =  (x_sfc  - self.xmean_sca)/(self.xdiv_sca)
    #     x_main[:,:,2] = 1 - torch.exp(-x_main[:,:,2] * self.lbd_qn)
    #     x_main = torch.where(torch.isnan(x_main), torch.tensor(0.0, device=x_main.device), x_main)
    #     x_sfc  = torch.where(torch.isinf(x_sfc),  torch.tensor(0.0, device=x_sfc.device),  x_sfc)
    #     return x_main, x_sfc 

    def preprocessing(self, x_main0, x_sfc0):
        # v4 input array
        x_main = x_main0.clone()
        x_sfc = x_sfc0.clone()

        x_main[:,:,2] = 1 - torch.exp(-x_main[:,:,2] * self.lbd_qc)
        x_main[:,:,3] = 1 - torch.exp(-x_main[:,:,3] * self.lbd_qi)   

        #                            mean     max - min
        x_main = (x_main - self.xmean_lev)/(self.xdiv_lev)
        x_sfc =  (x_sfc -  self.xmean_sca)/(self.xdiv_sca)
        
        x_main = torch.where(torch.isnan(x_main), torch.tensor(0.0, device=x_main.device), x_main)
        # x_sfc  = torch.where(torch.isinf(x_sfc),  torch.tensor(0.0, device=x_sfc.device),  x_sfc)
        return x_main, x_sfc 
    
    def postprocessing(self, out_lev, out_sfc):
        out_lev[:,0:12,1:] = 0
        out_lev      = out_lev / self.yscale_lev
        out_sfc     = out_sfc / self.yscale_sca
        # concatenate  
        # out =  torch.cat((out_lev.flatten(start_dim=1),out_sfc),dim=1)

        return out_lev, out_sfc
    

    def forward(self, x_main, x_sfc):
        # x_denorm = x_main.clone()
        
        T_before        = x_main[:,:,0:1].clone()
        qliq_before     = x_main[:,:,2:3].clone()
        qice_before     = x_main[:,:,3:4].clone()
        qn_before       = qliq_before + qice_before 
        # print("shape xsfc", x_sfc.shape)

        # print("xmain 0", torch.sum(x_main[200,:,:]))

        x_main, x_sfc = self.preprocessing(x_main, x_sfc)
        # print("shape xsfc 2", x_sfc.shape)
        # print("xmain ", torch.sum(x_main[200,:,:]), "min", x_main.min(), "max", x_main.max())
        # print("x_sfc ", torch.sum(x_sfc[200,:]))


        # outlev = torch.zeros((1000,36,3))
        # print("dqn raw0", outlev[200,35,0:1])

        # outlev, outsfc = self.original_model(x_main, x_sfc)
        out_lev, out_sfc = self.original_model(x_main, x_sfc)

        
        # out_lev, out_sfc = self.original_model.pp_mp(outlev, outsfc, x_denorm)
            
        # out_lev, out_sfc = self.postprocessing(outlev, outsfc)
        # out_lev      = outlev / self.original_model.yscale_lev
        # out_sfc      = outsfc / self.original_model.yscale_sca
        out_lev      = out_lev / self.original_model.yscale_lev
        out_sfc      = out_sfc / self.original_model.yscale_sca
        
        # # print("shape x denorm", x_denorm.shape, "T", T_before.shape)
        T_new           = T_before  + out_lev[:,:,0:1]*1200
        # print("T_new min", T_new.min(), "max", T_new.max())
        
        # # liq_frac_constrained    = self.temperature_scaling(T_new)
        liq_frac_constrained    = self.original_model.temperature_scaling(T_new)


        # #                            dqn
        qn_new      = qn_before + out_lev[:,:,2:3]*1200  
        qliq_new    = liq_frac_constrained*qn_new
        qice_new    = (1-liq_frac_constrained)*qn_new
        dqliq       = (qliq_new - qliq_before) * 0.0008333333333333334 #/1200  
        dqice       = (qice_new - qice_before) * 0.0008333333333333334 #/1200   
        
        # print("Tbef {:.2f}   T {:.2f}  dt {:.2e}  liqfrac {:.2f}   dqn {:.2e}  qbef {:.2e}  qnew {:.2e}  dqliq {:.2e}  dqice {:.2e} ".format( 
        #                                                 # x_denorm[200,35,4].item(),
        #                                                 T_before[200,35].item(), 
        #                                                 T_new[200,35].item(), 
        #                                                 (out_lev[200,35,0:1]*1200).item(),
        #                                                 liq_frac_constrained[200,35].item(), 
        #                                                 (out_lev[200,35,2]*1200).item(), 
        #                                                 qn_before[200,35].item(),
        #                                                 qn_new[200,35].item(),
        #                                                 dqliq[200,35].item(),
        #                                                 dqice[200,35].item()))
 
        # out_lev     = torch.cat((out_lev[:,:,0:2], dqliq, dqice, out_lev[:,:,3:]),dim=2)
        # return out_lev, out_sfc, outlev, outsfc
        # return out_lev, out_sfc


        # out_lev = torch.transpose(out_lev, 1, 2) # (nb,nlev,ny) --> (nb, ny, nlev)

        # print("shape 1, ", out_lev[:,0:2].shape, "dq", dqliq.shape)
        # out_lev = torch.cat((out_lev[:,0:2], dqliq.reshape(-1,1,60), dqice.reshape(-1,1,60), out_lev[:,3:]),dim=1)
        
        # out =  torch.cat((out_lev.flatten(start_dim=1),out_sfc),dim=1)
        
        batch_size = out_lev.shape[0] 
        out_lev = torch.transpose(out_lev, 1, 2).reshape(batch_size,300)

        yout = torch.zeros((batch_size,368))
        yout[:,0:120] = out_lev[:,0:120]
        yout[:,120:180] = torch.reshape(dqliq, (batch_size, 60))
        yout[:,180:240] = torch.reshape(dqice, (batch_size, 60))
        yout[:,240:360] = out_lev[:,180:360]
        yout[:,360:368] = out_sfc
        return yout 
    
# def utils_preprocess(x_lev, x_sfc, y_lev, y_sfc):
#     x_lev_b = np.copy(x_lev)
#     x_sfc_b = np.copy(x_sfc)
#     y_lev_b = np.copy(y_lev)
#     y_sfc_b = np.copy(y_sfc)
    
#     x_lev_b_denorm = np.copy(x_lev_b[:,:,0:4])
#     cloud_exp_norm_numba(x_lev_b)
#     apply_input_norm_numba(x_lev_b, xmean_lev, xdiv_lev)  
#     x_sfc_b = (x_sfc_b - xmean_sca0 ) / (xdiv_sca0)
    
#     x_lev_b[np.isnan(x_lev_b)] = 0
#     x_lev_b[np.isinf(x_lev_b)] = 0 
    
#     y_lev_b[:,:,2] = y_lev_b[:,:,2] + y_lev_b[:,:,3] 
#     y_lev_b = np.delete(y_lev_b, 3, axis=2) 
    
#     apply_output_norm_numba(y_lev_b, yscale_lev)
#     y_sfc_b  = y_sfc_b * yscale_sca    

#     x_sfc_b = np.delete(x_sfc_b,sfc_vars_remove,axis=1)
#     print("orig minmax x", x_lev_b.min(), x_lev_b.max())
    
#     x_lev_b_denorm = torch.from_numpy(x_lev_b_denorm)

#     x_lev_b = torch.from_numpy(x_lev_b)
#     x_sfc_b = torch.from_numpy(x_sfc_b)
#     y_lev_b = torch.from_numpy(y_lev_b)
#     y_sfc_b = torch.from_numpy(y_sfc_b)
#     return x_lev_b, x_sfc_b, y_lev_b, y_sfc_b, x_lev_b_denorm


new_model = NewModel(model, xmean_lev, xmean_sca, 
                    xdiv_lev, xdiv_sca,
                    yscale_lev, yscale_sca, 
                    lbd_qc, lbd_qi)

NewModel.device = "cpu"
device = torch.device("cpu")

scripted_model = torch.jit.script(new_model)
scripted_model = scripted_model.eval()



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
