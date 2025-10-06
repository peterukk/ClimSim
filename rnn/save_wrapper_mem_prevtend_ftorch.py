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
import h5py
import matplotlib.pyplot as plt
import time 
import xarray as xr
from utils import plot_bias
import gc 

sfc_vars_remove = (17, 18, 19, 20, 21)


fpath_lbd_qn = "saved_models/lbd_qn.txt"
fpath_lbd_qc = "saved_models/lbd_qc.txt"
fpath_lbd_qi = "saved_models/lbd_qi.txt"

#save_dir = "/media/peter/samlinux/soft/ClimSim-Online/climsim-online/shared_e3sm/saved_models/wrapper_ftorch/"
save_dir = "/data/climsim-online/shared_e3sm/saved_models/wrapper_ftorch/"

lbd_qn  =  np.loadtxt(fpath_lbd_qn, delimiter=",", dtype=np.float32)
lbd_qc  =  np.loadtxt(fpath_lbd_qc, delimiter=",", dtype=np.float32)
lbd_qi  =  np.loadtxt(fpath_lbd_qi, delimiter=",", dtype=np.float32)

model_path_script = "saved_models/LSTM_autoreg_torchscript_perturb-Hidden_lr0.0004.neur160-160_xv4_yv5_num50210_ep0_script.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint, nmem, nlev_mem, perturb = True, True, False, True, 16, 60, True
return_det = False


model_path_script = "saved_models/LSTM_autoreg_torchscript_perturb-Hidden_lr0.0004.neur160-160_xv4_yv5_num665_ep1_script.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint, nmem, nlev_mem, perturb = True, True, False, True, 16, 60, True
return_det = False

rh_prune = True

model_path_script = "saved_models/LSTM-Hidden_lr0.001.neur192-192_xv4_yv5_num96160_script.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint, nmem, nlev_mem, perturb = True, True, False, True, 16, 60, False

model_path_script = "saved_models/LSTM_sepmp-Hidden_lr0.001.neur160-160_xv4_yv5_num2134_script.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint, nmem, nlev_mem, perturb = True, True, False, True, 16, 60, False

model_path_script = "saved_models/LSTM-Hidden_lr0.001.neur144-144_xv4_yv5_num34897_script.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint, nmem, nlev_mem, perturb = True, True, False, True, 16, 60, False

model_path_script = "saved_models/LSTM-Hidden_lr0.0005.neur144-144_xv4_yv5_num86705_script_ep8.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint, nmem, nlev_mem, perturb = True, True, False, True, 16, 60, False

model_path_script = "saved_models/LSTM-Hidden_lr0.0004.neur144-144_xv4_yv5_num43301_script.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint, nmem, nlev_mem, perturb = True, True, False, True, 16, 60, False

model_path_script = "saved_models/LSTM-Hidden_lr0.0005.neur144-144_xv4_yv5_num16992_script.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint, nmem, nlev_mem, perturb = True, True, False, True, 16, 60, False

model_path_script = "saved_models/LSTM-Hidden_lr0.0005.neur144-144_xv4_yv5_num99576_script.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint, nmem, nlev_mem, perturb = True, True, False, True, 16, 60, False

model_path_script = "saved_models/SRNN-Hidden_lr0.001.neur144-144_xv4_yv5_num79482_script.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint, nmem, nlev_mem, perturb = True, True, False, True, 16, 60, False

model_path_script = "saved_models/LSTM-Hidden_lr0.0007.neur144-144_xv4_yv5_num8415_script.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint, nmem, nlev_mem, perturb = True, True, False, True, 16, 60, False

model_path_script = "saved_models/LSTM-Hidden_lr0.0005.neur144-144_xv4_yv5_num92002_script.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint, nmem, nlev_mem, perturb = True, True, False, True, 16, 60, False

model_path_script = "saved_models/LSTM-Hidden_lr0.0005.neur144-144_xv4_yv5_num10438_34_script.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint, nmem, nlev_mem, perturb = True, True, False, True, 16, 60, False

# model_path_script = "saved_models/partiallystochasticRNN-Hidden_lr0.0005.neur144-144_xv4_yv5_num63117_script.pt"
# qinput_prune, snowhice_fix, v5_input, mp_constraint, nmem, nlev_mem, perturb = True, True, False, True, 16, 60, False

model_path_script = "saved_models/partiallystochasticRNN-Hidden_lr0.0005.neur144-144_xv4_yv5_num55290_ep14_script.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint, nmem, nlev_mem, perturb = True, True, False, True, 16, 60, False

# model_path_script = "saved_models/LSTM_sepmp-Hidden_lr0.0005.neur144-144_xv4_yv5_num58938_ep1_val0.4534_script.pt"
# qinput_prune, rh_prune, snowhice_fix, v5_input, mp_constraint, nmem, nlev_mem, perturb = True, False, True, False, True, 16, 60, False

# model_path_script = "saved_models/SRNN-Hidden_lr0.0005.neur144-144_xv4_yv4_num51100_ep1_val0.6111_script.pt"
# qinput_prune, rh_prune, snowhice_fix, v5_input, mp_constraint, nmem, nlev_mem, perturb = True, False, True, False, False, 16, 60, False

fdir_bsc = "/media/peter/CrucialBX500/data/BSC_mount/ClimSim/rnn/saved_models/"
model_path_script = fdir_bsc + "LSTM-Hidden_lr0.0007.neur144-144_xv4_yv4_num2515_ep79_val0.0489_script.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint, nmem, nlev_mem, perturb = True, True, False, True, 16, 60, False

model_path_script = "saved_models/LSTM-Hidden_lr0.001.neur128-128_xv4_yv5_num25949_script_cuda.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint, nmem, nlev_mem, perturb = True, True, False, True, 16, 60, False

model_path_script = fdir_bsc + "LSTM-Hidden_lr0.0007.neur144-144_xv4_yv4_num36534_script_cpu.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint, nmem, nlev_mem, perturb = True, True, False, False, 16, 60, False

model_path_script = fdir_bsc + "LSTM-Hidden_lr0.0007.neur144-144_xv4_mp0_num62905_script_cpu.pt"
qinput_prune, snowhice_fix, v5_input, mp_constraint, nmem, nlev_mem, perturb = True, True, False, False, 16, 60, False

model_path_script = fdir_bsc + "LSTM-Hidden_lr0.0007.neur144-144_xv4_mp0_num51230_script_cpu.pt" 
qinput_prune, snowhice_fix, v5_input, mp_constraint, nmem, nlev_mem, perturb = True, True, False, False, 16, 60, False

model_path_script = fdir_bsc + "LSTM-Hidden_lr0.0007.neur144-144_xv4_mp-1_num88067_script_cpu.pt" 
qinput_prune, snowhice_fix, v5_input, nmem, nlev_mem, perturb = True, True, False, 16, 60, False

model_path_script = fdir_bsc + "SRNN-Hidden_lr0.0007.neur144-144_xv4_mp0_num72844_script_cpu.pt"  # ar1_0.7
qinput_prune, snowhice_fix, v5_input, nmem, nlev_mem, perturb = True, True, False, 16, 60, False

model_path_script = "saved_models/partiallystochasticRNN-Hidden_lr0.0007.neur144-144_xv4_mp0_num70753_script_cpu.pt"
qinput_prune, snowhice_fix, v5_input, nmem, nlev_mem, perturb = True, True, False, 16, 60, False

# model_path_script = "saved_models/partiallystochasticRNN-Hidden_lr0.0007.neur144-144_xv4_mp0_num88532_script_cpu.pt"
# qinput_prune, snowhice_fix, v5_input, nmem, nlev_mem, perturb = True, True, False, 16, 60, False

mp_mode = int(model_path_script.split('mp')[-1][0:3].split('_')[0])

if mp_mode==0: # predict qliq, qice
    mp_constraint = False 
    pred_liq_ratio = False
elif mp_mode>0: # predict qn, DIAGNOSE liquid fraction
    mp_constraint = True 
    pred_liq_ratio = False
else: # < 0  predict qn and liquid fraction
    mp_constraint = True       
    pred_liq_ratio = True 
    
if 'SRNN' or 'stochastic' in model_path_script:
    nens=5
else:
    nens=1

# use_gpu = True 
if "gpu" in model_path_script or "cuda" in model_path_script:
    use_gpu = True 
else:
    use_gpu = False 
    
model = torch.jit.load(model_path_script)

# include_prev_outputs = True
save = True
# save = False

try:
    use_ar_noise = model.use_ar_noise
    print("use_ar_noise IS defined in model, setting to True")

except:
    print("use_ar_noise not defined in model, setting to false")
    use_ar_noise = False



class NewModel_constraint(nn.Module):
    qinput_prune: Final[bool]
    rh_prune: Final[bool]
    snowhice_fix: Final[bool]
    v5_input: Final[bool]
    mp_constraint: Final[bool]
    predict_liq_ratio: Final[bool]
    perturb: Final[bool]
    return_det: Final[bool]
    
    def __init__(self, original_model, 
                 lbd_qc, lbd_qi, lbd_qn,
                 qinput_prune, rh_prune,
                 snowhice_fix, v5_input, mp_constraint, 
                 predict_liq_ratio,
                 perturb, return_det, device):
        
        super(NewModel_constraint, self).__init__()
        self.original_model = original_model
        self.lbd_qc     = torch.tensor(lbd_qc, dtype=torch.float32, device=device)
        self.lbd_qi     = torch.tensor(lbd_qi, dtype=torch.float32, device=device)
        self.lbd_qn     = torch.tensor(lbd_qn, dtype=torch.float32, device=device)

        self.hardtanh = nn.Hardtanh(0.0, 1.0)
        self.qinput_prune = qinput_prune
        self.rh_prune = rh_prune
        self.snowhice_fix = snowhice_fix
        self.v5_input = v5_input
        self.mp_constraint = mp_constraint  
        self.perturb = perturb
        self.return_det = return_det
        self.nmem = nmem
        self.predict_liq_ratio = predict_liq_ratio
        self.xmean_lev      = self.original_model.xmean_lev.to(device)
        self.xdiv_lev       = self.original_model.xdiv_lev.to(device)
        self.xmean_sca      = self.original_model.xmean_sca.to(device)
        self.xdiv_sca       = self.original_model.xdiv_sca.to(device)
        self.yscale_lev     = self.original_model.yscale_lev.to(device)
        self.yscale_sca     = self.original_model.yscale_sca.to(device)

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
        if self.rh_prune:
            x_main[:,:,1] = torch.clamp(x_main[:,:,1], 0, 1.2)

        x_main = torch.where(torch.isnan(x_main), torch.tensor(0.0, device=x_main.device), x_main)
        x_main = torch.where(torch.isinf(x_main), torch.tensor(0.0, device=x_main.device), x_main)
        return x_main, x_sfc 
    
    def temperature_scaling(self, T_raw):
        liquid_ratio = (T_raw - 253.16) * 0.05 
        liquid_ratio = F.hardtanh(liquid_ratio, 0.0, 1.0)
        return liquid_ratio
    
    def postprocessing(self, out, out_sfc):
        out             = out / self.yscale_lev
        out_sfc         = out_sfc / self.yscale_sca
        return out, out_sfc
        
    # def pp_mp(self, out, out_sfc, x_denorm):
    #     out_denorm      = out / self.yscale_lev
    #     out_sfc_denorm  = out_sfc / self.yscale_sca

    #     T_before        = x_denorm[:,:,0:1]
    #     qliq_before     = x_denorm[:,:,2:3]
    #     qice_before     = x_denorm[:,:,3:4]   
    #     qn_before       = qliq_before + qice_before 

    #     # print("shape x denorm", x_denorm.shape, "T", T_before.shape)
    #     T_new           = T_before  + out_denorm[:,:,0:1]*1200

    #     # T_new           = T_before  + out_denorm[:,:,0:1]*1200
    #     liq_frac_constrained    = self.temperature_scaling(T_new)

    #     #                            dqn
    #     qn_new      = qn_before + out_denorm[:,:,2:3]*1200  
    #     qliq_new    = liq_frac_constrained*qn_new
    #     qice_new    = (1-liq_frac_constrained)*qn_new
    #     dqliq       = (qliq_new - qliq_before) * 0.0008333333333333334 #/1200  
    #     dqice       = (qice_new - qice_before) * 0.0008333333333333334 #/1200   
    #     out_denorm  = torch.cat((out_denorm[:,:,0:2], dqliq, dqice, out_denorm[:,:,3:]),dim=2)
    
    #     return out_denorm, out_sfc_denorm
    def pp_mp(self, out, out_sfc, x_denorm):

        # out_denorm      = out / self.yscale_lev.to(device=out.device)
        # out_sfc_denorm  = out_sfc / self.yscale_sca.to(device=out.device)
        out_denorm      = out / self.yscale_lev
        out_sfc_denorm  = out_sfc / self.yscale_sca

        T_before        = x_denorm[:,:,0:1]
        qliq_before     = x_denorm[:,:,2:3]
        qice_before     = x_denorm[:,:,3:4]   
        qn_before       = qliq_before + qice_before 

        # print("shape x denorm", x_denorm.shape, "T", T_before.shape)
        T_new           = T_before  + out_denorm[:,:,0:1]*1200

        # T_new           = T_before  + out_denorm[:,:,0:1]*1200
        liq_frac_constrained    = self.temperature_scaling(T_new)

        if self.predict_liq_ratio:
            liq_frac_pred = out_denorm[:,:,3:4]
            # print("min max lfrac pred raw", torch.max(liq_frac_pred).item(), torch.min(liq_frac_pred).item())
            # Hu et al. Fig 2 b:
            max_frac = torch.clamp(liq_frac_constrained + 0.2, max=1.0)
            min_frac = torch.clamp(liq_frac_constrained - 0.2, min=0.0)
            # print("shape lfracpre", liq_frac_pred.shape, "con", liq_frac_constrained.shape)
            liq_frac_constrained = torch.clamp(liq_frac_pred, min=min_frac, max=max_frac)
            
            # print("min max lfrac pred pp", torch.max(liq_frac_constrained).item(), torch.min(liq_frac_constrained).item())

        #                            dqn
        qn_new      = qn_before + out_denorm[:,:,2:3]*1200  
        qliq_new    = liq_frac_constrained*qn_new
        qice_new    = (1-liq_frac_constrained)*qn_new
        # print("min max qice_new pred pp", torch.max(qice_new).item(), torch.min(qice_new).item())
        # print("min max qice_bf pred pp", torch.max(qice_before).item(), torch.min(qice_before).item())

        dqliq       = (qliq_new - qliq_before) * 0.0008333333333333334 #/1200  
        dqice       = (qice_new - qice_before) * 0.0008333333333333334 #/1200   
        if self.predict_liq_ratio:           # replace    dqn,   liqfrac
            out_denorm  = torch.cat((out_denorm[:,:,0:2], dqliq, dqice, out_denorm[:,:,4:]),dim=2)
        else:
            out_denorm  = torch.cat((out_denorm[:,:,0:2], dqliq, dqice, out_denorm[:,:,3:]),dim=2)

        
        return out_denorm, out_sfc_denorm
    
    
    if use_ar_noise:
        def forward(self, x_main0, x_sfc0, rnn1_mem, eps_prev):            

            x_main, x_sfc = self.preprocessing(x_main0, x_sfc0)
        
            if self.perturb:
                out_lev, out_sfc, rnn1_mem, out_lev_det, eps_prev = self.original_model(x_main, x_sfc, rnn1_mem, eps_prev)
            else:
                out_lev, out_sfc, rnn1_mem, eps_prev = self.original_model(x_main, x_sfc, rnn1_mem, eps_prev)
        

            if self.mp_constraint:
                out_lev, out_sfc = self.pp_mp(out_lev, out_sfc, x_main0)
                if self.perturb:
                    out_lev_det, out_sfc_tmp = self.pp_mp(out_lev_det, out_sfc, x_main0)
            else:
                out_lev      = out_lev / self.yscale_lev
                out_sfc      = out_sfc / self.yscale_sca
                if self.perturb and self.return_det:
                    out_lev_det      = out_lev_det / self.yscale_lev
                    

            out_lev = torch.where(torch.isnan(out_lev), torch.tensor(0.0, device=x_main.device), out_lev)
            if self.perturb and self.return_det:
                out_lev_det = torch.where(torch.isnan(out_lev_det), torch.tensor(0.0, device=x_main.device), out_lev_det)
                return out_lev, out_lev_det, out_sfc, rnn1_mem, eps_prev
            else:
                return out_lev, out_sfc, rnn1_mem, eps_prev
    else:
        def forward(self, x_main0, x_sfc0, rnn1_mem):
            
            x_main, x_sfc = self.preprocessing(x_main0, x_sfc0)
        
            if self.perturb:
                out_lev, out_sfc, rnn1_mem, out_lev_det = self.original_model(x_main, x_sfc, rnn1_mem)
            else:
                out_lev, out_sfc, rnn1_mem = self.original_model(x_main, x_sfc, rnn1_mem)
        

            if self.mp_constraint:
                out_lev, out_sfc = self.pp_mp(out_lev, out_sfc, x_main0)
                if self.perturb:
                    out_lev_det, out_sfc_tmp = self.pp_mp(out_lev_det, out_sfc, x_main0)
            else:
                out_lev      = out_lev / self.yscale_lev
                out_sfc      = out_sfc / self.yscale_sca
                if self.perturb and self.return_det:
                    out_lev_det      = out_lev_det / self.yscale_lev
                    

            out_lev = torch.where(torch.isnan(out_lev), torch.tensor(0.0, device=x_main.device), out_lev)
            if self.perturb and self.return_det:
                out_lev_det = torch.where(torch.isnan(out_lev_det), torch.tensor(0.0, device=x_main.device), out_lev_det)
                return out_lev, out_lev_det, out_sfc, rnn1_mem
            else:
                return out_lev, out_sfc, rnn1_mem

if use_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = model.to(device)

nx = model.xmean_lev.shape[1]
if nx == 20:
    print("Model has 20 level wise inputs, assuming previous tendencies are used")
    use_previous_tendencies = True 
elif nx == 15:
    print("Model has 15 level wise inputs, assuming previous tendencies are NOT used")
    use_previous_tendencies = False 
else:
    raise NotImplementedError("check number of level-wise inputs: {}, possibly not supported".format(nx))

new_model = NewModel_constraint(model, lbd_qc, lbd_qi, lbd_qn, 
                                qinput_prune, rh_prune, 
                                snowhice_fix, v5_input, mp_constraint, pred_liq_ratio,
                                perturb, return_det, device)
# device = torch.device("cpu")
new_model = new_model.to(device)

# with torch.jit.optimized_execution(True):
scripted_model = torch.jit.script(new_model)
scripted_model = scripted_model.eval()

# save_file_torch = "v4_rnn-memory_wrapper_constrained_huber_160.pt"
# save_file_torch = "v4_rnn-memory_wrapper_constrained_huber_160.pt"
# save_file_torch = "wrappers/v4_rnn-memory_wrapper_constrained_huber_energy_num26608_160.pt"
if use_gpu:
    save_file_torch = save_dir + model_path_script.split("/")[-1].split(".pt")[0] + "_cuda.pt"
else:
    save_file_torch = save_dir + model_path_script.split("/")[-1].split(".pt")[0] + ".pt"

if save:
    print("saving to ", save_file_torch)
    scripted_model.save(save_file_torch)
    print("success")
d
# fpath_data = '/media/peter/samsung/data/ClimSim_low_res_expanded/data_v4_rnn_nonorm_year8.h5'
# fpath_data = "/media/peter/CrucialP1/data/ClimSim/low_res_expanded/train_v4_rnn_nonorm_febtofeb_y1-7_stackedhalfyear_gzip8_chunk1_subset.h5"
fpath_data = '/media/peter/CrucialBX500/data/ClimSim/low_res_expanded/data_v4_rnn_nonorm_year8_nocompress_chunk3.h5'
# (26278, 384, 60, 15)

hf = h5py.File(fpath_data, 'r')
bsize = 384 
# nb = 2160

nb = 1400
# nb = 500

nlev = 60
# nlev = 50
offset = 0
offset = 22000 

ns = nb * bsize

x_lev_np = hf['input_lev'][offset:offset+nb]
x_sfc_np = hf['input_sca'][offset:offset+nb]
x_sfc_np = np.delete(x_sfc_np,sfc_vars_remove,axis=2)
y_lev_np = hf['output_lev'][offset:offset+nb]
y_sfc_np = hf['output_sca'][offset:offset+nb]

if use_previous_tendencies:
    prev_outputs = hf['output_lev'][offset-1:offset+nb-1]
    prev_outputs = prev_outputs[:,:,:,[0,1,2,3,4]]
    x_lev_np = np.concatenate((x_lev_np, prev_outputs),axis=-1)

hf.close()

xlev = torch.from_numpy(x_lev_np).to(device)
xsfc = torch.from_numpy(x_sfc_np).to(device)

if use_ar_noise:
    # eps_prev = torch.rand(nlev, bsize, model.nh_rnn1,device=xlev.device)
    eps_prev = torch.rand(2, nlev, bsize, model.nh_rnn1,device=xlev.device)

    eps_prev.requires_grad = False 

outs_lev = []
outs_sfc = []
outs_lev_det = []

t0_it = time.time()

for i in range(nens):
    ntime = nb 
    j = 0 
    rnn1_mem = torch.zeros((bsize, nlev_mem, nmem),device=xlev.device)
    
    for jj in range(ntime):
        jend = j + bsize
        # out_test = scripted_model(xlev,xsfc)
        # x0 = xlev[j:jend]
        # x1 = xsfc[j:jend]
        x0 = xlev[jj,:]
        x1 = xsfc[jj,:]
        # print(rnn1_mem[0,0,0], rnn1_mem[4,30,10])
        
        with torch.no_grad(): 
            if use_ar_noise:
                if perturb:
                    out_lev, out_lev_det, out_sfc, rnn1_mem, eps_prev = scripted_model(x0,x1,rnn1_mem,eps_prev)
                    outs_lev_det.append(out_lev_det)
                else:
                    out_lev, out_sfc, rnn1_mem, eps_prev = scripted_model(x0,x1,rnn1_mem,eps_prev)   
            else:
                if perturb:
                    out_lev, out_lev_det, out_sfc, rnn1_mem = scripted_model(x0,x1,rnn1_mem)
                    outs_lev_det.append(out_lev_det)
                else:
                    out_lev, out_sfc, rnn1_mem = scripted_model(x0,x1,rnn1_mem)
            outs_lev.append(out_lev)
            outs_sfc.append(out_sfc)
            # rnn1_mem = torch.reshape(out_test[:,368:], (bsize,nlev_mem,nmem))
        j = j + bsize
    
elaps = time.time() - t0_it
print(" took {:.1f}s".format(elaps))

outs_lev = torch.stack(outs_lev)
outs_sfc = torch.stack(outs_sfc)

if perturb:
    outs_lev_det = torch.stack(outs_lev_det)
    outs_lev_det = outs_lev_det.detach().numpy().reshape(nens,-1,384,60,6)

outs_lev = outs_lev.detach().numpy().reshape(nens,-1,384,60,6)
outs_sfc = outs_sfc.detach().numpy().reshape(nens,-1,384,8)
prec_true = y_sfc_np[:,:,3]
prec_pred = outs_sfc[0,:,:,3]


outs_lev_std = np.std(outs_lev,axis=0)

gc.collect()

# y_true = y_lev_np.reshape(-1,384,60,6)
# y_pred = outs_lev.reshape(-1,384,60,6)
y_true = y_lev_np 
y_pred = outs_lev[0]

colors = ['b','g']
fig, ax1 = plt.subplots()
ax1.hist([prec_pred.flatten(),prec_true.flatten()],color=colors)
ax1.set_yscale('log')
ax1.legend(['Pred','True'])

colors = ['b','g']
fig, ax1 = plt.subplots()
ax1.hist([y_pred[:,:,:,2].flatten(),y_true[:,:,:,2].flatten()],color=colors)
ax1.set_yscale('log')
ax1.legend(['Pred','True'])



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
gc.collect()


vars_stacked = [[dt_t_mean,dt_p_mean], 
                [q_t_mean,q_p_mean],
                # [u_t_mean,u_p_mean], 
                [v_t_mean,v_p_mean],
                [clw_t_mean,clw_p_mean],
                [cli_t_mean,cli_p_mean]]


data_path = '/media/peter/CrucialBX500/data/ClimSim/hu_etal2024_data/'

ds_grid = xr.open_dataset(data_path+'data_grid/ne4pg2_scrip.nc')
grid_area = ds_grid['grid_area']

ds2 = xr.open_dataset(data_path+'data_grid/E3SM_ML.GNUGPU.F2010-MMF1.ne4pg2_ne4pg2.eam.h0.0001-01.nc')
lat = ds2.lat
lon = ds2.lon
level = ds2.lev.values

plot_bias(vars_stacked, grid_area, lat, level)

dd

# How correlated are the errors of the different members?            
dqt = y_true[:,:,:,1]
dqt_mean = np.mean(dqt,axis=0)
dqp1 = outs_lev[0,:,:,:,1]
dqp2 = outs_lev[1,:,:,:,1]
dqp3 = outs_lev[2,:,:,:,1]

dqpm = np.mean(outs_lev,axis=0)
# Error = true - pred AFTER removing bias
dqp1_er = dqt - dqp1 
dqp2_er = dqt - dqp2
dqp3_er = dqt - dqp3
np.corrcoef(dqp1_er.flatten(),dqp2_er.flatten())
# Out[15]: 
# array([[1.        , 0.59954648],
#        [0.59954648, 1.        ]])

# partiallystochastic 40289 trained on scoringrules-CRPS
# array([[1.        , 0.51269304],
#        [0.51269304, 1.        ]])

dqt = y_true[:,:,:,1] 
dqt = dqt - np.mean(dqt,axis=0)
dqp1 = outs_lev[0,:,:,:,1]; dqp2 = outs_lev[1,:,:,:,1]; dqp3 = outs_lev[2,:,:,:,1]

dqp1 = dqp1 - np.mean(dqp1,axis=0)
dqp2 = dqp2 - np.mean(dqp1,axis=0)
dqp3 = dqp3 - np.mean(dqp1,axis=0)

# Error = true - pred AFTER removing bias
dqp1_er = dqt - dqp1 
dqp2_er = dqt - dqp2
dqp3_er = dqt - dqp3
np.corrcoef(dqp1_er.flatten(),dqp2_er.flatten())


labels = ["dT/dt", "dq/dt", "dqliq/dt", "dqice/dt", "dU/dt", "dV/dt"]



import matplotlib.pyplot as plt
import pandas as pd
import glob
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap


# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

def using_mpl_scatter_density(fig, x, y):
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(x, y, cmap=white_viridis)
    fig.colorbar(density, label='Number of points per pixel')
    
data_path = '/media/peter/CrucialBX500/data/ClimSim/hu_etal2024_data/'

ds_grid = xr.open_dataset(data_path+'data_grid/ne4pg2_scrip.nc')
grid_area = ds_grid['grid_area']

ds2 = xr.open_dataset(data_path+'data_grid/E3SM_ML.GNUGPU.F2010-MMF1.ne4pg2_ne4pg2.eam.h0.0001-01.nc')
lat = ds2.lat
lon = ds2.lon
level = ds2.lev.values

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



def plot_dq_profile(y,yp,i,j,k):
    min_hpa = 300
    colorlist = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
    alpha=0.5
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(9.5, 4.5),
                            layout="constrained")
    icol=0
    axs[icol].plot(y[i,j,:,1],level,'k',linewidth=2)
    ymean = np.mean(yp[:,i,j,:,1],axis=0)
    axs[icol].plot(ymean,level,'--b',linewidth=2)

    for im in range(10):
        axs[icol].plot(yp[im,i,j,:,1],level,color=colorlist[im],alpha=alpha)
    axs[icol].legend(['True', 'Pred (ensemble mean)', 'Pred'])
    axs[icol].set_ylim(min_hpa,1000)
    # axs[icol].invert_yaxis()
    axs[icol].set_ylabel("Pressure (hPa)")
    axs[icol].set_xlabel("Water vapor tendency (kg/kg/s)")
    axs[icol].invert_yaxis()

    icol=1
    # axs[icol].plot(y[i,j,:,1],level,'k')
    axs[icol].plot(y[i,j,:,1]-ymean,level,'--b',linewidth=2)
    for im in range(10):
        axs[icol].plot(y[i,j,:,1]-yp[im,i,j,:,1],level,color=colorlist[im],alpha=alpha)
    im=0
    # axs[icol].plot(y[i,j,:,1]-yp[im,i,j,:,1],level,"r")
    axs[icol].legend(['True - Pred (ensemble mean)','True - Pred'])
    axs[icol].set_ylim(min_hpa,1000)
    # axs[icol].invert_yaxis
    axs[icol].set_ylabel("Pressure (hPa)")
    axs[icol].set_xlabel("Water vapor tendency (kg/kg/s)")
    axs[icol].invert_yaxis()

    # axs[icol].set_yticks(level, minor=True)
    # axs[icol].grid(which="minor")
    


dq_pred = y_pred[:,:,:,1]
dq_true = y_true[:,:,:,1]

ind = np.unravel_index(dq_true.argmax(), dq_true.shape)
ind = np.unravel_index(dq_pred.argmax(), dq_pred.shape)

i,j,k = ind

plot_dq_profile(y_true,outs_lev,i,j,k)
plot_dq_profile(y_true,outs_lev,i+1,j,k)
plot_dq_profile(y_true,outs_lev,i+2,j,k)


import matplotlib.animation as animation

fig, ax = plt.subplots()

im=0
jtime = 0 
timestep = i + jtime
line, = ax.plot(y_true[i,j,:,1]-outs_lev[im,i,j,:,1],level,"r")
ax.set_title('Member=0, Timestep = {}'.format(0))
ax.invert_yaxis()

def animate(jtime):
    line.set_xdata(y_true[i+jtime,j,:,1]-outs_lev[im,i+jtime,j,:,1])  # update the data.
    ax.set_title('Error for Member=0, Timestep = {}'.format(jtime+1))
    return line,

ani = animation.FuncAnimation(
    fig, animate, frames=10,interval=1000) # interval=20, blit=True, save_count=50)
plt.show()


fig, ax = plt.subplots()
im=0
jtime = 0 
timestep = i + jtime
line, = ax.plot(y_true[i,j,:,1],level,"r")
ax.set_title('dq-true, Timestep = {}'.format(0))
ax.invert_yaxis()

def animate(jtime):
    line.set_xdata(y_true[i+jtime,j,:,1])  # update the data.
    ax.set_title('dq-true, Timestep = {}'.format(jtime+1))
    return line,

ani = animation.FuncAnimation(
    fig, animate, frames=10,interval=1000) # interval=20, blit=True, save_count=50)
plt.show()




fig, ax = plt.subplots(ncols=3)
fig.suptitle("Water vapor tendency errors")
im=0
jtime = 0 
min_hpa = 200
line, = ax[0].plot(y_true[i,j,:,1]-outs_lev[0,i,j,:,1],level,"r")
line2, = ax[1].plot(y_true[i,j,:,1]-outs_lev[1,i,j,:,1],level,"r")
line3, = ax[2].plot(y_true[i,j,:,1]-outs_lev[2,i,j,:,1],level,"r")

ax[0].set_title('member={}, t={}'.format(0,jtime))
ax[1].set_title('member={}, t={}'.format(1,jtime))
ax[2].set_title('member={}, t={}'.format(2,jtime))

ax[0].set_ylim(min_hpa,1000); ax[1].set_ylim(min_hpa,1000); ax[2].set_ylim(min_hpa,1000)
ax[0].invert_yaxis(); ax[1].invert_yaxis(); ax[2].invert_yaxis()
ax[0].set_xlim([-4.0e-7,4.0e-7]); ax[1].set_xlim([-4.0e-7,4.0e-7]); ax[2].set_xlim([-4.0e-7,4.0e-7])

def animate(jtime):
    line.set_xdata(y_true[i+jtime,j,:,1]-outs_lev[0,i+jtime,j,:,1])  # update the data.
    ax[0].set_title('member={}, t={}'.format(0,jtime+1))
    line2.set_xdata(y_true[i+jtime,j,:,1]-outs_lev[1,i+jtime,j,:,1])  # update the data.
    ax[1].set_title('member={}, t={}'.format(1,jtime+1))
    line3.set_xdata(y_true[i+jtime,j,:,1]-outs_lev[2,i+jtime,j,:,1])  # update the data.
    ax[2].set_title('member={}, t={}'.format(2,jtime+1))
    return line,line2, line3

# ani = animation.FuncAnimation(
#     fig, animate, frames=10,interval=1000) # interval=20, blit=True, save_count=50)

ani = animation.FuncAnimation(
    fig, animate, frames=20,interval=1500) # interval=20, blit=True, save_count=50)

writergif = animation.PillowWriter(fps=1)
ani.save('/media/peter/samlinux/gdrive/postdoc/results/anim_srnn_errors_newcrps.gif',writer=writergif)


plt.show()



fig, ax = plt.subplots(ncols=3,figsize=(10, 4))
 
im=0
jtime = 0 
min_hpa = 200
xmin, xmax = -3e-7, 3e-7

line, = ax[0].plot(outs_lev[0,i,j,:,1],level,"r")
line2, = ax[1].plot(y_true[i,j,:,1],level,"k")
line3, = ax[2].plot(outs_lev[1,i,j,:,1],level,"r")
ax[0].set_title("Pred member 1")
ax[1].set_title("True")
ax[2].set_title("Pred member 2")
fig.suptitle("Timestep = {}".format(jtime))
for j in range(3):
    ax[j].set_xlabel("q tendency (kg/kg/s)")
    ax[j].set_ylim(min_hpa,1000)
    ax[j].invert_yaxis()
    ax[j].set_xlim([xmin,xmax])
    ax[j].set_ylabel("hPa")

# fig.tight_layout()
fig.subplots_adjust(top=0.88,
bottom=0.11,
left=0.1,
right=0.965,
hspace=0.205,
wspace=0.33)    

def animate(jtime):
    line.set_xdata(outs_lev[0,i+jtime,j,:,1])  # update the data.
    line2.set_xdata(y_true[i+jtime,j,:,1])  # update the data.
    line3.set_xdata(outs_lev[1,i+jtime,j,:,1])  # update the data.
    fig.suptitle("Timestep = {}".format(jtime+1))
    return line,line2,line3

ani = animation.FuncAnimation(
    fig, animate, frames=20,interval=1500) # interval=20, blit=True, save_count=50)

writergif = animation.PillowWriter(fps=1)
ani.save('/media/peter/samlinux/gdrive/postdoc/results/anim.gif',writer=writergif)

# ani.save("movie.mp4")

# or

# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)
plt.show()







# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)
# plt.show()


# def ar_noise(eps_tprev):
#     tau_t = 0.5
#     tau_e = np.sqrt(1-tau_t**2)
#     eps = np.random.randn(1)
#     eps_t = tau_t * eps_tprev + tau_e * eps
#     return eps_t

# epss = []
# eps = np.random.randn(1)
# for jrep in range(1000):
#    eps = ar_noise(eps) 
#    epss.append(eps)
   
# epss = np.array(epss)



labels=["Heating","Moistening","V","Cloud water", "cloud ice"]
scalings = [1,1000,1,1e6,1e6]

# max_diffs = [40,5]

latitude_ticks = [-60, -30, 0, 30, 60]
latitude_labels = ['60S', '30S', '0', '30N', '60N']

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
#R2_qn = np.zeros((384,60))
if perturb:
    z_mean_fac_dt = np.zeros((384,60)) 
    z_mean_fac_dq = np.zeros((384,60)) 
    z_mean_fac_clw = np.zeros((384,60)) 
    z_mean_fac_cli = np.zeros((384,60)) 

mean_abs_dt = np.zeros((384,60))
mean_abs_dq = np.zeros((384,60))
mean_abs_clw = np.zeros((384,60))
mean_abs_cli = np.zeros((384,60))

for i in range(384): 
    for j in range(60):
        R2_dt[i,j] =  np.corrcoef(y_true[:,i,j,0], y_pred[:,i,j,0])[0,1]**2
        R2_dq[i,j] =  np.corrcoef(y_true[:,i,j,1], y_pred[:,i,j,1])[0,1]**2
        R2_clw[i,j] =  np.corrcoef(y_true[:,i,j,2], y_pred[:,i,j,2])[0,1]**2
        R2_cli[i,j] =  np.corrcoef(y_true[:,i,j,3], y_pred[:,i,j,3])[0,1]**2
      #  R2_qn[i,j] =  np.corrcoef(qn_true[:,i,j], qn_pred[:,i,j])[0,1]**2
      
        mean_abs_dt[i,j] = np.mean(np.abs(y_true[:,i,j,0]))
        mean_abs_dq[i,j] = np.mean(np.abs(y_true[:,i,j,1]))
        mean_abs_clw[i,j] = np.mean(np.abs(y_true[:,i,j,2]))
        mean_abs_cli[i,j] = np.mean(np.abs(y_true[:,i,j,3])) 
        
        if perturb:
            z_mean_fac_dt[i,j] =  np.mean(outs_lev_std[:,i,j,0]) / np.mean(np.abs(y_pred[:,i,j,0]))
            z_mean_fac_dq[i,j] =  np.mean(outs_lev_std[:,i,j,1])  / np.mean(np.abs(y_pred[:,i,j,1]))
            z_mean_fac_clw[i,j] =  np.mean(outs_lev_std[:,i,j,2])  / np.mean(np.abs(y_pred[:,i,j,2]))
            z_mean_fac_cli[i,j] =  np.mean(outs_lev_std[:,i,j,3])  / np.mean(np.abs(y_pred[:,i,j,3]))

# z = outs_lev - outs_lev_det

if perturb:
    z_mean_fac_clw[z_mean_fac_clw>100] = np.nan
    z_mean_fac_cli[z_mean_fac_cli>100] = np.nan


fs_label = 15


if perturb:
    vars_stacked2 = [[mae_dt,R2_dt, z_mean_fac_dt], 
                     [mae_dq,R2_dq, z_mean_fac_dq], 
                    [mae_clw,R2_clw, z_mean_fac_clw], 
                    [mae_cli,R2_cli, z_mean_fac_cli]]#,
                     # [mae_qn,R2_qn]]
    
    labels2=[r"$dT/dt$", r"$dQ_v/dt$", r"$dQ_{liq}/dt$", r"$dQ_{ice}/dt$"]
    
    fig, axs = plt.subplots(len(vars_stacked2), 3, figsize=(14, 12.5)) 
    
    for idx in range(len(vars_stacked2)):
        # var_mae, var_r2, var_y, var_z = vars_stacked2[idx]
        var_mae, var_r2, var_z = vars_stacked2[idx]

        mae_zm, lats_sorted = zonal_mean_area_weighted(var_mae, grid_area, lat)
        r2_zm, lats_sorted = zonal_mean_area_weighted(var_r2, grid_area, lat)
        # y_zm, lats_sorted = zonal_mean_area_weighted(var_y, grid_area, lat)
        z_zm, lats_sorted = zonal_mean_area_weighted(var_z, grid_area, lat)

        # data_sp, data_nn = 1e6*sp_zm.T, 1e6*nn_zm.T
        
        scaling = 1
        data_mae = scaling * xr.DataArray(mae_zm[:, :].T, dims=["hybrid pressure (hPa)", "latitude"],
                                         coords={"hybrid pressure (hPa)": level, "latitude": lats_sorted})
        data_r2 = scaling * xr.DataArray(r2_zm[:, :].T, dims=["hybrid pressure (hPa)", "latitude"],
                                         coords={"hybrid pressure (hPa)": level, "latitude": lats_sorted})
        # data_y = scaling * xr.DataArray(y_zm[:, :].T, dims=["hybrid pressure (hPa)", "latitude"],
        #                                  coords={"hybrid pressure (hPa)": level, "latitude": lats_sorted})        
        data_z = scaling * xr.DataArray(z_zm[:, :].T, dims=["hybrid pressure (hPa)", "latitude"],
                                         coords={"hybrid pressure (hPa)": level, "latitude": lats_sorted})           
        # Determine color scales
        vmax1 = data_mae.max()
        vmin1 = data_mae.min()
        vmax2 = 1.0
        vmin2 = 0.0
    
        
        data_mae.plot(ax=axs[idx, 0], add_colorbar=True, cmap='plasma', vmin=vmin1, vmax=vmax1)
        # axs[idx, 0].set_title(f'{labels[idx * 3]} {var_title} ({unit}): MMF')
        axs[idx, 0].set_title("{} , {} ".format(labels2[idx],'MAE'),fontsize=fs_label)
        axs[idx, 0].invert_yaxis()

        data_r2.plot(ax=axs[idx, 1], add_colorbar=True, cmap='plasma', vmin=vmin2, vmax=vmax2)
        axs[idx, 1].set_title("{} , {} ".format(labels2[idx], r'$R^2$'),fontsize=fs_label)
        axs[idx, 1].invert_yaxis()
        axs[idx, 1].set_ylabel('')  # Clear the y-label to clean up plot
        
        # data_y.plot(ax=axs[idx, 2], add_colorbar=True, cmap='plasma')#, vmin=vmin2, vmax=vmax2)
        # axs[idx, 2].set_title("{} , {} ".format(labels2[idx], r'mean(y)'))
        # axs[idx, 2].invert_yaxis()
        # axs[idx, 2].set_ylabel('')  # Clear the y-label to clean up plot
        
        data_z.plot(ax=axs[idx, 2], add_colorbar=True, cmap='plasma')#, vmin=vmin2, vmax=vmax2)
        axs[idx, 2].set_title("{} , {} ".format(labels2[idx], r"$\frac{\overline{std(y)}}{\overline{|y|}}$"),fontsize=fs_label)
        axs[idx, 2].invert_yaxis()
        axs[idx, 2].set_ylabel('')  # Clear the y-label to clean up plot

        # if idx < (len(vars_stacked2)-1):
        axs[idx, 0].set_xlabel('')  # Clear the y-label to clean up plot
        axs[idx, 1].set_xlabel('')  # Clear the y-label to clean up plot
        axs[idx, 2].set_xlabel('')  # Clear the y-label to clean up plot

        axs[idx, 1].set_yticklabels('')
        axs[idx, 2].set_yticklabels('')

    
    # Set these ticks and labels for each subplot
    for ax_row in axs:
        for ax in ax_row:
            ax.set_xticks(latitude_ticks)  # Set the positions for the ticks
            ax.set_xticklabels(latitude_labels)  # Set the custom text labels
    plt.tight_layout()
    plt.show() 


