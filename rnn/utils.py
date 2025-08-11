#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch data loader based on indexing HDF5 file, and other utils
"""
import ctypes
import os
import multiprocessing as mp
import psutil
import h5py
import torch
import torch.nn as nn
import numpy as np 
from numba import config, njit, threading_layer
import numpy as np
import gc
import metrics as metrics
from torchmetrics.regression import R2Score
import time
from metrics import corrcoeff_pairs_batchfirst 
import matplotlib
import matplotlib.pyplot as plt
lbd_qi_lev = np.array([10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        ,  7556905.1099813 ,
        3240294.53811436,  4409304.29170528,  5388911.78320826,
        1414189.69398583,   444847.03675674,   550036.71076073,
         452219.47765234,   243545.07231263,   163264.17204164,
         128850.88117789,   108392.13699281,    96868.6539061 ,
          90154.39383647,    83498.67423248,    76720.52614694,
          70937.87706283,    66851.27198026,    64579.78345685,
          64987.05874437,    68963.77227883,    75498.91605962,
          82745.37660119,    89624.52634008,    96373.41157796,
         102381.42808207,   102890.33417304,    96849.77123401,
          92727.78368907,    91320.9721545 ,    91240.30382044,
          91448.65004889,    91689.26513737,    91833.1829058 ,
          91941.15859653,    92144.1029509 ,    92628.38565183,
          93511.1538428 ,    94804.20080999,    96349.5878153 ,
          98174.89731264,   100348.81479455,   102750.86508174,
         105013.71207426,   106732.83687405,   107593.00387448,
         108022.91061398,   109634.8552567 ,   112259.85403167], dtype=np.float32)
lbd_qi_mean = np.repeat(np.float32(2291547.5),60)

lbd_qc_lev = np.array([10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        ,  2410793.53754872,
        3462644.65436088,  1594172.20270602,   328086.13752288,
         154788.55435228,   118712.37335602,   104208.42410058,
          95801.11739569,    89619.52961093,    83709.51800851,
          78846.75613935,    74622.76219094,    70555.95112947,
          66436.67119096,    61797.61126943,    56926.03823691,
          51838.00818631,    46355.21691466,    40874.23574077,
          36196.39550842,    32935.40953052,    31290.83140741,
          30908.27330462,    31386.06558422,    32606.7350768 ,
          34631.09245739,    37847.88977875,    42878.24049123,
          50560.90175672,    61294.98389768,    72912.41450047,
          80998.32102651,    88376.7321416 ,   135468.13760583], dtype=np.float32)
lbd_qc_mean = np.repeat(np.float32(4496518.0),60)

lbd_qn_lev = np.array([10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        , 10000000.        ,
       10000000.        , 10000000.        ,  7556905.1099813 ,
        3240294.53811436,  4409304.29170528,  5388911.78320826,
        1414189.69398583,   444847.03675674,   550036.71076073,
         452219.47765234,   243545.07231263,   163264.17204164,
         128850.88117789,   108392.13699281,    96868.6539061 ,
          90154.39383647,    83498.67423248,    76720.52614694,
          70937.79468155,    66821.0327278 ,    63916.46591524,
          61597.41430156,    60417.96523765,    60359.64347926,
          60430.76970212,    59696.934318  ,    58222.94889662,
          56637.11031175,    54844.45378425,    52735.80221775,
          50450.11987115,    47895.00010132,    45134.95219383,
          42075.52757738,    38557.91174999,    34843.47468245,
          31537.88963513,    29179.71520305,    28016.06440645,
          27844.86770893,    28377.06256804,    29532.22068928,
          31360.65252559,    34174.61235695,    38452.69084769,
          44777.29680978,    53238.52542881,    61797.74325549,
          66939.83519617,    70867.57480034,    94733.63482142], dtype=np.float32) 
lbd_qn_mean = np.repeat(np.float32(2268406.8),60)


# qliq = 1 - np.exp(-xlev[:,:,2] * lbd_qc)
# qice = 1 - np.exp(-xlev[:,:,3] * lbd_qi)

# plt.hist(xlev[:,:,2].flatten())
# plt.hist(qliq.flatten())

# plt.hist(xlev[:,:,3].flatten(), bins=20)
# plt.hist(qice.flatten(), bins=20)

# x_lev_b = x_lev_b * (xmax_lev - xmin_lev) + xmean_lev 
# x_sfc_b = x_sfc_b * (xmax_sca - xmin_sca) + xmean_sca 


# T_denorm = T = T*(xmax_lev[:,0] - xmin_lev[:,0]) + xmax_lev[:,0]
# T_denorm = T
# liquid_ratio = (T_denorm - 253.16) / 20.0 
# liquid_ratio = F.hardtanh(liquid_ratio, 0.0, 1.0)
        


class train_or_eval_one_epoch:
    def __init__(self, dataloader, 
                 model, 
                 device, dtype,
                 cfg, 
                 metrics_det, metric_h_con, metric_water_con, 
                 batch_size = 384, train=True,):
        super().__init__()
        self.loader = dataloader
        self.batch_size = batch_size
        self.train = train
        self.report_freq = 800
        self.model = model
        self.device = device
        self.dtype = dtype
        self.cfg = cfg 
        if self.cfg.use_scaler:
            # scaler = torch.amp.GradScaler(autocast = True)
            self.scaler = torch.amp.GradScaler(self.device.type)
        # if self.cfg.mp_mode>0:
        self.ny_pp = 6 # Regardless of MP constraint, 6 postprocessed output features per level
        # Loss functions to be computed, possibly part of overall loss
        # or just monitored
        self.metrics_det = metrics_det
        self.metric_h_con = metric_h_con
        self.metric_water_con = metric_water_con
        self.metric_R2 =  R2Score().to(self.device) 
        # The computed values of various metrics are stored at the end
        self.metrics = {}
        
        if self.cfg.ensemble_size>1:
            self.use_ensemble=True
        else:
            self.use_ensemble=False
            
        if self.cfg.mp_mode>0:
            self.use_mp_constraint=True
        else:
            self.use_mp_constraint=False 
                
    def eval_one_epoch(self, lossf, optim, epoch, timesteps=1, lr_scheduler=None):
        report_freq = self.report_freq
        running_loss = 0.0; running_energy = 0.0; running_water = 0.0
        running_var=0.0; running_bias = 0.0
        epoch_loss = 0.0; epoch_mse = 0.0; epoch_huber = 0.0; epoch_mae = 0.0
        epoch_R2precc = 0.0; epoch_R2netsw = 0.0; epoch_R2flwds = 0.0
        epoch_hcon = 0.0; epoch_wcon = 0.0
        epoch_ens_var = 0.0; epoch_det_skill = 0.0; epoch_spreadskill = 0.0
        epoch_r2_lev = 0.0
        epoch_bias_lev = 0.0; epoch_bias_sfc = 0.0; epoch_bias_heating = 0.0
        epoch_bias_clw = 0.0; epoch_bias_cli = 0.0
        epoch_bias_lev_tot = 0.0; epoch_bias_perlev= 0.0
        epoch_mae_lev_clw = 0.0; epoch_mae_lev_cli = 0.0
        epoch_rmse_perlev = 0.0; epoch_prec_std_frac = 0.0
        t_comp =0 
        t0_it = time.time()
        j = 0; k = 0; k2=2    
        device = self.device
        
        if self.cfg.optimizer == "adamwschedulefree":
            if self.train:
                optim.train()
            else:
                optim.eval()
        
        if self.cfg.autoregressive:
            preds_lay = []; preds_sfc = []
            targets_lay = []; targets_sfc = [] 
            x_sfc = [];  x_lay_raw = []
            yto_lay = []; yto_sfc = []
            rnn1_mem = torch.zeros(self.batch_size*self.cfg.ensemble_size, self.model.nlev_mem, self.model.nh_mem, device=device)
            if self.cfg.use_surface_memory:
                sfc_mem = torch.zeros(self.batch_size, self.model.ny_sfc, device=device)
            loss_update_start_index = 60
        else:
            loss_update_start_index = 0
            
        # torch.autograd.set_detect_anomaly(True)
            
        # Option ar_noise_mode for training stochastic RNNs
        # 0 : Fully uncorrelated noise. The sampled noise (eps) used in the stochastic RNN has no temporal correlation, is
        # redrawn at every vertical level and for each RNN. Therefore no need to keep track of it outside the model
        # 1: Eps has temporal correlation (correlation time scale set by tau_t), 
        # but no vertical correlation (eps has a vertical dimension) and is not shared by the two RNNs 
        # 2: Eps has temporal correlation and no vertical correlation, and is shared between the two RNNs
        # 3: Fully correlated noise: temporal correlation, and eps is shared between the two RNN models and at all vertical levels
        if self.cfg.ar_noise_mode>0:
            use_ar_noise = True
            if self.cfg.ar_noise_mode==1:
                eps_prev = torch.rand(self.model.nlev, self.batch_size*self.cfg.ensemble_size, self.cfg.nneur[0], device=device)
                eps_prev2 = torch.rand(self.model.nlev, self.batch_size*self.cfg.ensemble_size, self.cfg.nneur[1], device=device)
                eps_prev = torch.stack((eps_prev,eps_prev2))
            elif self.cfg.ar_noise_mode==2:
                eps_prev = torch.rand(self.model.nlev, self.batch_size*self.cfg.ensemble_size, self.cfg.nneur[0], device=device)
            elif self.cfg.ar_noise_mode==3:
                eps_prev = torch.rand(self.batch_size*self.cfg.ensemble_size, self.cfg.nneur[0], device=device)
            eps_prev.requires_grad = False 
        else:
            use_ar_noise = False 
        for i,data in enumerate(self.loader):
            # print("shape mem 2 {}".format(model.rnn1_mem.shape))
            x_lay_chk, x_sfc_chk, targets_lay_chk, targets_sfc_chk, x_lay_raw_chk  = data 
            
            # print("shape x lay", x_lay_chk.shape)

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
                # print("shape xlay 0", x_lay0.shape)

                if self.cfg.use_surface_memory:
                    x_sfc0 = torch.cat((x_sfc0, sfc_mem),dim=-1)

                tcomp0= time.time()               
                
                with torch.autocast(device_type=device.type, dtype=self.dtype, enabled=self.cfg.mp_autocast):
                    inp_list = [x_lay0, x_sfc0]
                    if self.cfg.autoregressive:
                        inp_list.append(rnn1_mem)
                    if use_ar_noise:
                        inp_list.append(eps_prev)
                    if self.cfg.model_type=="physrad":
                        inp_list.append(x_lay_raw0)
                        
                    outs = self.model(inp_list)
                        
                    if self.cfg.autoregressive:
                        if use_ar_noise:
                            if self.cfg.model_type=="LSTM_autoreg_torchscript_perturb":
                                preds_lay0, preds_sfc0, rnn1_mem, eps_prev, dummy = outs
                            else:
                                preds_lay0, preds_sfc0, rnn1_mem, eps_prev = outs
                        else:
                            if self.cfg.model_type=="LSTM_autoreg_torchscript_perturb":
                                preds_lay0, preds_sfc0, rnn1_mem, dummy = outs
                            else:
                                preds_lay0, preds_sfc0, rnn1_mem = outs
                    else:
                        preds_lay0, preds_sfc0 = outs

                if self.cfg.use_surface_memory:
                    sfc_mem = preds_sfc0

                if self.cfg.autoregressive:
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
                    
                if (not self.cfg.autoregressive) or (self.cfg.autoregressive and (j+1) % timesteps==0):
            
                    if self.cfg.autoregressive:
                        preds_lay   = torch.cat(preds_lay)
                        preds_sfc   = torch.cat(preds_sfc)
                        targets_lay = torch.cat(targets_lay)
                        targets_sfc = torch.cat(targets_sfc)
                        x_sfc       = torch.cat(x_sfc)
                        x_lay_raw   = torch.cat(x_lay_raw)
                        
                    with torch.autocast(device_type=device.type, dtype=self.dtype, enabled=self.cfg.mp_autocast):
                        
                        if self.cfg.loss_fn_type == "CRPS":
                            loss = lossf(targets_lay, targets_sfc, preds_lay, preds_sfc, timesteps)
                            loss, det_skill, ens_var = loss
                        else:
                            loss = lossf(targets_lay, targets_sfc, preds_lay, preds_sfc)
                            huber, mse, mae = loss 
                            if self.cfg.loss_fn_type == "huber":
                                loss = huber
                            else:
                                loss = mse
                            
                        if self.use_ensemble:
                            # print("preds shape", preds_lay)
                            # use only first member from here on 
                            preds_lay = torch.reshape(preds_lay, (timesteps, self.cfg.ensemble_size, self.batch_size,  self.model.nlev,  self.model.ny))
                            preds_lay = torch.reshape(preds_lay[:,0,:,:], shape=targets_lay.shape)
                            preds_sfc = torch.reshape(preds_sfc, (timesteps, self.cfg.ensemble_size, self.batch_size,  self.model.ny_sfc))
                            preds_sfc = torch.reshape(preds_sfc[:,0,:], shape=targets_sfc.shape)
                                          
                        if self.use_mp_constraint:
                            ypo_lay, ypo_sfc = self.model.pp_mp(preds_lay, preds_sfc, x_lay_raw)
                            with torch.no_grad(): 
                                yto_lay, yto_sfc = self.model.pp_mp(targets_lay, targets_sfc, x_lay_raw )
                            # ypo_lay, ypo_sfc, yto_lay, yto_sfc = model.pp_mp(preds_lay, preds_sfc, targets_lay, targets_sfc, x_lay_raw )
                            # if i>10: print ("yto lay true lev 35  dqliq {:.2e} ".format(ypo_lay[200,35,2].item()))
                            # if i>10: print ("yto lay pp-true lev 35  dqliq {:.2e} ".format(ypo_lay[200,35,2].item()))

                        else:
                            ypo_lay, ypo_sfc = self.model.postprocessing(preds_lay, preds_sfc)
                            yto_lay, yto_sfc = self.model.postprocessing(targets_lay, targets_sfc)
                            
                        with torch.no_grad():
                            x_sfc = x_sfc*self.model.xdiv_sca + self.model.xmean_sca 
                            surf_pres_denorm = x_sfc[:,0:1]
                            lhf = x_sfc[:,2] 

                        # Energy conservation metric
                        h_con           = self.metric_h_con(yto_lay, ypo_lay, surf_pres_denorm, 1) #timesteps)
                        
                        # Water conservation metric (computed over multiple timesteps since the CRM has precipitation storage not exposed to NN)
                        water_con_p     = self.metric_water_con(ypo_lay, ypo_sfc, surf_pres_denorm, lhf, x_lay_raw, timesteps)
                        water_con_t     = self.metric_water_con(yto_lay, yto_sfc, surf_pres_denorm, lhf, x_lay_raw, timesteps)#,printdebug=True)
                        water_con       = torch.mean(torch.square(water_con_p - water_con_t))
                        del water_con_p, water_con_t
                        
                        # if cfg.use_bias_loss:
                        raw_bias_lev, raw_bias_sfc = metrics.compute_absolute_biases(yto_lay, yto_sfc, ypo_lay, ypo_sfc )
                        raw_bias_lev = torch.nanmean(raw_bias_lev)
                        if self.cfg.use_bias_loss:
                            loss = loss + self.cfg.w_bias*raw_bias_lev
                        del raw_bias_sfc

                        # if epoch>4:
                            
                        if self.cfg.use_energy_loss: 
                            loss = loss + self.cfg.w_hcon*h_con

                        if self.cfg.use_water_loss:
                            loss = loss + self.cfg.w_wcon * water_con
                            
                    if self.train:
                        if self.cfg.use_scaler:
                            self.scaler.scale(loss).backward(retain_graph=True)
                            self.scaler.step(optim)
                            self.scaler.update()
                        else:
                            loss.backward(retain_graph=True)       
                            optim.step()
   
                        optim.zero_grad()
                        loss = loss.detach()
                        if self.cfg.loss_fn_type == "CRPS":
                            det_skill = det_skill.detach(); ens_var = ens_var.detach()
                        else: 
                            huber = huber.detach(); mse = mse.detach(); mae = mae.detach()
                        h_con = h_con.detach() 
                        water_con = water_con.detach()
                        # if self.cfg.lr_scheduler=="OneCycleLR":
                        #     lr_scheduler.step()
                          
                    ypo_lay = ypo_lay.detach(); ypo_sfc = ypo_sfc.detach()
                    yto_lay = yto_lay.detach(); yto_sfc = yto_sfc.detach()
                        
                    running_loss    += loss.item()
                    running_energy  += h_con.item()
                    running_water   += water_con.item()
                    running_bias    += raw_bias_lev.item() 
                    if self.cfg.loss_fn_type == "CRPS": running_var += ens_var.item() 
                    #mae             = metrics.mean_absolute_error(targets_lay, preds_lay)
                    if j>loss_update_start_index:
                        with torch.no_grad():
                            epoch_loss      += loss.item()
                            if self.cfg.loss_fn_type =="CRPS":
                                huber, mse, mae       = self.metrics_det(targets_lay, targets_sfc, preds_lay, preds_sfc)

                            epoch_huber += huber.item()
                            epoch_mse += mse.item()
                            epoch_mae += mae.item()
                            # epoch_mae       += metrics.mean_absolute_error(targets_lay, preds_lay)
                        
                            epoch_hcon  += h_con.item()
                            epoch_wcon  += water_con.item()
                            
                            if self.cfg.loss_fn_type == "CRPS":
                                epoch_ens_var += ens_var.item()
                                epoch_det_skill += det_skill.item()
                                epoch_spreadskill += ens_var.item() / det_skill.item()

                            biases_lev, biases_sfc = metrics.compute_absolute_biases(yto_lay, yto_sfc, ypo_lay, ypo_sfc, numpy=True)
                            epoch_bias_lev += np.mean(biases_lev)
                            epoch_bias_heating += biases_lev[0]
                            epoch_bias_clw += biases_lev[2]
                            epoch_bias_cli += biases_lev[3]
                            epoch_bias_sfc += np.mean(biases_sfc)

                            biases_nolev, biases_perlev = metrics.compute_biases(yto_lay, ypo_lay)
                            epoch_bias_lev_tot += np.mean(biases_nolev)
                            epoch_bias_perlev += biases_perlev

                            epoch_rmse_perlev += metrics.rmse(yto_lay, ypo_lay)

                            self.metric_R2.update(ypo_lay.reshape((-1,self.ny_pp)), yto_lay.reshape((-1,self.ny_pp)))
                                   
                            sfc_pred = ypo_sfc.reshape(-1,self.model.ny_sfc).cpu().numpy().transpose()
                            sfc_true = yto_sfc.reshape(-1,self.model.ny_sfc).cpu().numpy().transpose()

                            epoch_R2netsw += np.corrcoef((sfc_pred[0,:],sfc_true[0,:]))[0,1]
                            epoch_R2flwds += np.corrcoef((sfc_pred[1,:],sfc_true[1,:]))[0,1]

                            prec_pred = sfc_pred[3,:]
                            prec_true = sfc_true[3,:]
                            epoch_R2precc += np.corrcoef((prec_pred,prec_true))[0,1]
                            epoch_prec_std_frac += prec_pred.std()/prec_true.std()
                            # epoch_prec_perc_frac += np.percentile(prec_pred,99.9)/np.percentile(prec_true,99.9)

                            #print("R2 numpy", r2_np, "R2 torch", self.metric_R2_precc(ypo_sfc[:,3:4], yto_sfc[:,3:4]) )

                            ypo_lay = ypo_lay.reshape(-1,self.model.nlev,self.ny_pp).cpu().numpy()
                            yto_lay = yto_lay.reshape(-1,self.model.nlev,self.ny_pp).cpu().numpy()

                            epoch_r2_lev += metrics.corrcoeff_pairs_batchfirst(ypo_lay, yto_lay)**2

                            epoch_mae_lev_clw +=  np.nanmean(np.abs(ypo_lay[:,:,2] - yto_lay[:,:,2]),axis=0)
                            epoch_mae_lev_cli +=  np.nanmean(np.abs(ypo_lay[:,:,3] - yto_lay[:,:,3]),axis=0)
                           # if track_ks:
                           #     if (j+1) % max(timesteps*4,12)==0:
                           #         epoch_ks += kolmogorov_smirnov(yto,ypo).item()
                           #         k2 += 1
                            k += 1
                    if self.cfg.autoregressive:
                        preds_lay = []; preds_sfc = []
                        targets_lay = []; targets_sfc = [] 
                        x_lay_raw = []; yto_lay = []; yto_sfc = []
                        x_sfc = []
                        rnn1_mem = rnn1_mem.detach()
                        if self.cfg.use_surface_memory:
                            sfc_mem = sfc_mem.detach()
                        # rnn1_mem.requires_grad=True
                        
                t_comp += time.time() - tcomp0
                # # print statistics 
                if j % report_freq == (report_freq-1): # print every 200 minibatches
                    elaps = time.time() - t0_it
                    fac = report_freq/timesteps
                    running_loss = running_loss / fac
                    running_energy = running_energy / fac
                    running_water = running_water / fac
                    running_bias = running_bias / fac

                    r2raw = self.metric_R2.compute()

                    #r2_np = np.corrcoef((ypo_sfc.reshape(-1,ny_sfc)[:,3].detach().cpu().numpy(),yto_sfc.reshape(-1,ny_sfc)[:,3].detach().cpu().numpy()))[0,1]
                    if self.cfg.loss_fn_type == "CRPS": 
                        running_var = running_var / fac

                        print("[{:d}, {:d}] Loss: {:.2e} var {:.2e} h-con: {:.2e}  w-con: {:.2e}  MBE: {:.2e}  R2: {:.2f}, took {:.1f}s (comp. {:.1f})" .format(epoch + 1, 
                                                        j+1, running_loss,running_var, running_energy,running_water, running_bias,r2raw, elaps, t_comp), flush=True)
                        running_var = 0.0
                    else:
                        print("[{:d}, {:d}] Loss: {:.2e}  h-con: {:.2e}  w-con: {:.2e}  MBE: {:.2e}  R2: {:.2f},  took {:.1f}s (compute {:.1f})" .format(epoch + 1, 
                                                        j+1, running_loss,running_energy,running_water,running_bias, r2raw, elaps, t_comp), flush=True)
                    running_loss = 0.0
                    running_energy = 0.0; running_water=0.0
                    running_bias = 0.0
                    t0_it = time.time()
                    t_comp = 0
                j += 1
                
            del x_lay_chk, x_sfc_chk, targets_lay_chk, targets_sfc_chk, x_lay_raw_chk
                
        self.metrics['loss'] =  epoch_loss / k
        self.metrics['mean_squared_error'] = epoch_mse / k
        self.metrics['huber'] = epoch_huber / k

        if self.cfg.loss_fn_type == "CRPS": 
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
        self.metrics['R2_FLWDS'] = epoch_R2flwds / k
        self.metrics['R2_NETSW'] = epoch_R2netsw / k
        self.metrics['prec_std_frac'] = epoch_prec_std_frac / k 
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
        if self.cfg.autoregressive:
            del rnn1_mem
        if self.cfg.use_surface_memory:
            del sfc_mem
        if device.type=="cuda": 
            torch.cuda.empty_cache()
        gc.collect()
    
@njit(fastmath=True)    
def v4_to_v5_inputs_numba(x_lev_b, liq_ratio, lbd_qn):
    (ns,nlev,nx) = x_lev_b.shape
    for ii in range(ns):
        for jj in range(nlev):
            qn = x_lev_b[ii,jj,2] + x_lev_b[ii,jj,3] 
            qn = 1 - np.exp(-qn * lbd_qn[jj])
            x_lev_b[ii,jj,2] = qn
            x_lev_b[ii,jj,3] = liq_ratio[ii,jj]
            
@njit(fastmath=True)    
def cloud_exp_norm_numba(x_lev_b, lbd_qc, lbd_qi):
    (ns,nlev,nx) = x_lev_b.shape
    for ii in range(ns):
        for jj in range(nlev):
            x_lev_b[ii,jj,2] = 1 - np.exp(-x_lev_b[ii,jj,2] * lbd_qc[jj])
            x_lev_b[ii,jj,3] = 1 - np.exp(-x_lev_b[ii,jj,3] * lbd_qi[jj])
                    
            
@njit()    
# def reverse_input_norm_numba(x, xcoeff_mean, xcoeff_min, xcoeff_max):
def reverse_input_norm_numba(x, xcoeff_mean, xcoeff_div):
    (ns,nlev,nx) = x.shape
    for ii in range(ns):
        for jj in range(nlev):
            for kk in range(nx):
                # x[ii,jj,kk] = x[ii,jj,kk] * (xcoeff_max[jj,kk] - xcoeff_min[jj,kk]) + xcoeff_mean[jj,kk]
                x[ii,jj,kk] = x[ii,jj,kk] * (xcoeff_div[jj,kk]) + xcoeff_mean[jj,kk]

@njit(error_model="numpy")    
# def apply_input_norm_numba(x, xcoeff_mean, xcoeff_min, xcoeff_max):
def apply_input_norm_numba(x, xcoeff_mean, xcoeff_div):

    (ns,nlev,nx) = x.shape
    for ii in range(ns):
        for jj in range(nlev):
            for kk in range(nx):           #  mean                  max                       min 
                # x[ii,jj,kk] = (x[ii,jj,kk] - xcoeff_mean[jj,kk] ) / (xcoeff_max[jj,kk] - xcoeff_min[jj,kk])     
                x[ii,jj,kk] = (x[ii,jj,kk] - xcoeff_mean[jj,kk] ) / (xcoeff_div[jj,kk])     

@njit(error_model="numpy")    
# def apply_input_norm_numba(x, xcoeff_mean, xcoeff_min, xcoeff_max):
def apply_output_norm_numba(y, ycoeff):
    (ns,nlev,ny) = y.shape
    for ii in range(ns):
        for jj in range(nlev):
            for kk in range(ny): 
                y[ii,jj,kk] = y[ii,jj,kk] * ycoeff[jj,kk]
    
class generator_xy(torch.utils.data.Dataset):
    def __init__(self, filepath, nloc=384, cache=False, add_refpres=True,
                 ycoeffs=None, xcoeffs=None, 
                 xcoeffs_ref=None,
                 ycoeffs_ref=None,
                 v4_to_v5_inputs=False,
                 cloud_exp_norm=True,
                 input_norm_per_level=True,
                 remove_past_sfc_inputs=False,
                 qinput_prune=False,
                 rh_prune=False,
                 output_prune=False,
                 mp_mode=0,
                 include_prev_inputs=False, # include inputs 9,10,11 of the previous time step
                 include_prev_outputs=False, # include previous TRUE tendencies in inputs as in Hu et al.
                 no_multiprocessing=False,
                 snowhice_fix=True): # 0 = regular outputs, 1 = mp constraint, 2 = pred liq ratio
                 # use_mp_constraint=False):
        self.filepath = filepath
        # The file list will be divided into chunks (a list of lists)eg [[12,4,32],[1,9,3]..]
        # where the length of each item is the chunk size; i.e. how many files 
        # are loaded at once (in this example 3 files)
        # self.chunk_size = chunk_size # how many batches are loaded at once in getitem
        self.cache = cache
        self.use_numba = True
        self.nloc = nloc
        # self.cloud_exp_norm = True
        self.cloud_exp_norm = cloud_exp_norm
        self.remove_past_sfc_inputs = remove_past_sfc_inputs
        self.mp_mode = mp_mode
        self.no_multiprocessing=no_multiprocessing
        self.snowhice_fix = snowhice_fix
        self.qinput_prune = qinput_prune
        self.rh_prune = rh_prune 
        self.output_prune=output_prune
        self.include_prev_inputs = include_prev_inputs
        self.include_prev_outputs = include_prev_outputs
        if self.mp_mode>0:
            self.use_mp_constraint = True 
        else:
            self.use_mp_constraint = False    
        
        self.v4_to_v5_inputs    = v4_to_v5_inputs
        if input_norm_per_level:
            self.lbd_qc = lbd_qc_lev
            self.lbd_qi = lbd_qi_lev
            self.lbd_qn = lbd_qn_lev
        else:
            self.lbd_qc = lbd_qc_mean
            self.lbd_qi = lbd_qi_mean
            self.lbd_qn = lbd_qn_mean

        if xcoeffs_ref is None:
            # self.v4_to_v5_inputs    = False
            self.reverse_input_norm = False
        else:
            # self.v4_to_v5_inputs    = True
            self.reverse_input_norm = True 
            self.xcoeff_lev_ref, self.xcoeff_sca_ref  = xcoeffs_ref
        if ycoeffs_ref is None:
            # self.v4_to_v5_inputs    = False
            self.reverse_output_norm = False
        else:
            # self.v4_to_v5_inputs    = True
            self.reverse_output_norm = True 
            self.yscale_lev_ref, self.yscale_sca_ref  = ycoeffs_ref   
            
        # if ref coefficients are provided, undo scaling
        # if new coefficients are provided, apply new scaling, assume v4_to_V5 inputs=True
        # could be that reference but not new coefficients are provided,
        # in this case the preprocessing is done inside the model,
        # and the old scaling should be reversed (only)
        
        if xcoeffs is not None:
            # self.reverse_input_norm = True 
      
            # self.v4_to_v5_inputs    = True
            self.xcoeff_lev, self.xcoeff_sca = xcoeffs
            self.apply_new_input_scaling = True
        else:
            self.apply_new_input_scaling = False
            
        if ycoeffs is not None:
            self.yscale_lev, self.yscale_sca = ycoeffs
       
        # if type(self.filepath)==list:
        #     # In this case, each time a chunk is fetched, all the files are opened and the
        #     # data is concatenated along the column dimension
        #     self.num_files = len(self.filepath)
        #     print("Number of files: {}".format(self.num_files))
        #     hdf = h5py.File(self.filepath[0], 'r')
        # else:
        self.num_files = 1
        hdf = h5py.File(self.filepath, 'r')
            
        dims = hdf['input_lev'].shape
        if len(dims)==4:
            self.ntimesteps, self.nloc, self.nlev, self.nx = dims
            self.separate_timedim=True
        else:
            _, self.nlev, self.nx = dims
            self.separate_timedim=False 
            self.ntimesteps = hdf['input_lev'].shape[0]//self.nloc

        if self.include_prev_inputs:
            self.nx = self.nx + 6

        if self.include_prev_outputs:
            self.nx = self.nx + 5 

        if self.include_prev_inputs or self.include_prev_outputs:
            self.ntimesteps = self.ntimesteps - 1
            if not self.separate_timedim: 
                raise NotImplementedError()


        print("Data shape: {} for data in {}".format(dims, filepath))
        self.nx_sfc = hdf['input_sca'].shape[-1]
        self.ny = hdf['output_lev'].shape[-1]
        self.ny_sfc = hdf['output_sca'].shape[-1]
        
        # if type(self.filepath)==list:
        #     self.ncol = self.num_files * nloc
        # else:
        self.ncol = nloc

        self.cache_loaded = False
        if self.cache:

            if not self.separate_timedim:
                raise NotImplementedError()
            ns, nloc, nlev, nx = dims
            
            if self.no_multiprocessing:
                print("Using only one worker, loading all data to RAM", flush=True)

                self.input_lev = np.empty((ns,nloc,nlev,nx),dtype=np.float32)
                print("input_lev initialized", flush=True)
                print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush=True)
                os.system("df -h /dev/shm/")
                
                self.input_sca = np.empty((ns,nloc,self.nx_sfc),dtype=np.float32)
                self.output_lev = np.empty((ns,nloc,nlev,self.ny),dtype=np.float32)
                print("output_lev initialized", flush=True)
    
                self.output_sca = np.empty((ns,nloc,self.ny_sfc),dtype=np.float32)
                print("output_sca initialized", flush=True)
            else:
                print("Using Shared memory between workers, loading all data to RAM", flush=True)
                os.system("df -h /dev/shm/")

                input_lev_base = mp.Array(ctypes.c_float, ns*nloc*nlev*nx)
                input_lev = np.ctypeslib.as_array(input_lev_base.get_obj())
                self.input_lev = input_lev.reshape(ns,nloc,nlev,nx)
                # self.input_lev[:,:,:,:] = hdf['input_lev'][:]
                print("input_lev initialized", flush=True)
                print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush=True)
                os.system("df -h /dev/shm/")

                input_sca_base = mp.Array(ctypes.c_float, ns*nloc*self.nx_sfc)
                input_sca = np.ctypeslib.as_array(input_sca_base.get_obj())
                self.input_sca = input_sca.reshape(ns,nloc,self.nx_sfc)
                # self.input_sca[:,:,:] = hdf['input_sca'][:]
    
                # self.shared_array = torch.from_numpy(shared_array)
                output_lev_base = mp.Array(ctypes.c_float, ns*nloc*nlev*self.ny)
                output_lev = np.ctypeslib.as_array(output_lev_base.get_obj())
                self.output_lev = output_lev.reshape(ns,nloc,nlev,self.ny)
                # self.output_lev[:,:,:,:] = hdf['output_lev'][:]
                print("output_lev initialized", flush=True)
    
                output_sca_base = mp.Array(ctypes.c_float, ns*nloc*self.ny_sfc)
                output_sca = np.ctypeslib.as_array(output_sca_base.get_obj())
                self.output_sca = output_sca.reshape(ns,nloc,self.ny_sfc)
                # self.output_sca[:,:,:] = hdf['output_sca'][:]
                print("output_sca initialized", flush=True)
            
            # Getting usage of virtual_memory in GB ( 4th field)
            print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush=True)
            os.system("df -h /dev/shm/")

        hdf.close()
        # self.nloc = int(os.path.basename(self.filepath).split('_')[-1])
        # self.stateful = stateful
        self.refpres = np.array([7.83478113e-02,1.41108318e-01,2.52923297e-01,4.49250635e-01,
                    7.86346161e-01,1.34735576e+00,2.24477729e+00,3.61643148e+00,
                    5.61583643e+00,8.40325322e+00,1.21444894e+01,1.70168280e+01,
                    2.32107981e+01,3.09143463e+01,4.02775807e+01,5.13746323e+01,
                    6.41892284e+01,7.86396576e+01,9.46300920e+01,1.12091274e+02,
                    1.30977804e+02,1.51221318e+02,1.72673905e+02,1.95087710e+02,
                    2.18155935e+02,2.41600379e+02,2.65258515e+02,2.89122322e+02,
                    3.13312087e+02,3.38006999e+02,3.63373492e+02,3.89523338e+02,
                    4.16507922e+02,4.44331412e+02,4.72957206e+02,5.02291917e+02,
                    5.32152273e+02,5.62239392e+02,5.92149276e+02,6.21432841e+02,
                    6.49689897e+02,6.76656485e+02,7.02242188e+02,7.26498589e+02,
                    7.49537645e+02,7.71445217e+02,7.92234260e+02,8.11856675e+02,
                    8.30259643e+02,8.47450653e+02,8.63535902e+02,8.78715875e+02,
                    8.93246018e+02,9.07385213e+02,9.21354397e+02,9.35316717e+02,
                    9.49378056e+02,9.63599599e+02,9.78013432e+02,9.92635544e+02],dtype=np.float32)
        # self.refpres_norm = self.refpres
        self.refpres_norm = np.log(self.refpres)
        # self.refpres_norm = (self.refpres-self.refpres.min())/(self.refpres.max()-self.refpres.min())*2 - 1
        
        # self.yscale_lev_ref  = np.array([[1.00464e+03, 2.83470e+06, 5.66940e+06, 2.83470e+06, 2.50000e+02,
        #         5.00000e+02]], dtype=np.float32).repeat(60,axis=0)
        # self.yscale_sca_ref = np.array([2.40000e-03, 5.00000e-03, 1.24416e+07, 1.31328e+06, 5.00000e-03,
        #        4.60000e-03, 6.10000e-03, 9.50000e-03], dtype=np.float32)
        
        #if 'train' in self.filepath:
        #    self.is_validation = False
        #    print("Training dataset, path is: {}".format(self.filepath))
        #else:
        #    self.is_validation = True
        #    print("Validation dataset, path is: {}".format(self.filepath))

        self.add_refpres = add_refpres
        # batch_idx_expanded =  [0,1,2,3...ntime*1024]

        print("Number of locations {}; colums {}, time steps {}".format(self.nloc,self.ncol, self.ntimesteps), flush=True)
        # indices_all = list(np.arange(self.ntimesteps*self.nloc))
        # chunksize_tot = self.nloc*self.chunk_size
        # indices_chunked = self.chunkize(indices_all,chunksize_tot,False) 
        # self.hdf = h5py.File(self.filepath, 'r')

    # def set_cache_status(self, cache_loaded):
    #     self.cache_loaded = cache_loaded
        


    def compute_liq_ratio(self, T):
        liquid_ratio = (T - 253.16) * 0.05 
        liquid_ratio = np.clip(liquid_ratio, 0.0, 1.0)
        return liquid_ratio 
    
    def __len__(self):
        # return self.ntimesteps*self.nloc
        return self.ntimesteps*self.ncol

        
                
    def __getitem__(self, indices):
        # t0_it = time.time()
        # print("inds ", indices[0:3], "...", indices[-1])

        if self.include_prev_inputs or self.include_prev_outputs:
            if indices[0]>0:
                inds_prev = indices[0:-1]
                inds_prev.insert(0,inds_prev[0]-1)
            else:
                raise NotImplementedError("First time index cannot be zero as it's used for memory")
                # inds_prev = indices[0:-1]
                # inds_prev.insert(0,inds_prev[0])
        # print(indices, "|||||||||", inds_prev)
          
        if self.cache and self.cache_loaded: 
            x_lev_b = self.input_lev[indices,:]
            x_sfc_b = self.input_sca[indices,:] 
            y_lev_b = self.output_lev[indices,:]  
            y_sfc_b = self.output_sca[indices,:] 

            if self.include_prev_outputs:
                prev_outputs = self.output_lev[inds_prev,:,:,:]
                prev_outputs = prev_outputs[:,:,:,[0,1,2,3,4]]

            if self.include_prev_inputs:
                # prev_inputs = self.input_lev[inds_prev,:,:,[0,1,2,3,4,5]]
                prev_inputs = self.input_lev[inds_prev,:,:,:]
                prev_inputs = prev_inputs[:,:,:,[0,1,2,3,4,5]]
        else:
            hdf = h5py.File(self.filepath, 'r')
            # hdf = self.hdf

            x_lev_b = hdf['input_lev'][indices,:]
            x_sfc_b = hdf['input_sca'][indices,:]
            y_lev_b = hdf['output_lev'][indices,:]
            y_sfc_b = hdf['output_sca'][indices,:]
            # x_lev_b = hdf['input_lev'][indices[0]:indices[-1]+1,:]
            # x_sfc_b = hdf['input_sca'][indices[0]:indices[-1]+1,:]
            # y_lev_b = hdf['output_lev'][indices[0]:indices[-1]+1,:]
            # y_sfc_b = hdf['output_sca'][indices[0]:indices[-1]+1,:]
            if self.include_prev_outputs:
                prev_outputs = hdf['output_lev'][inds_prev,:,:,:]
                prev_outputs = prev_outputs[:,:,:,[0,1,2,3,4]]

            if self.include_prev_inputs:
                # prev_inputs = hdf['input_lev'][inds_prev,:,:,[0,1,2,3,4,5]]
                prev_inputs = hdf['input_lev'][inds_prev]
                prev_inputs = prev_inputs[:,:,:,[0,1,2,3,4,5]]

            if self.cache and (not self.cache_loaded):
                self.input_lev[indices,:] = x_lev_b
                self.input_sca[indices,:] = x_sfc_b
                self.output_lev[indices,:] = y_lev_b
                self.output_sca[indices,:] = y_sfc_b

            hdf.close() 

        if self.include_prev_outputs:
            x_lev_b = np.concatenate((x_lev_b, prev_outputs),axis=-1)

        if self.include_prev_inputs:
            # if indices[0]==0:
            #     prev_inputs = np.concatenate((np.zeros_like(prev_inputs[0:1]),prev_inputs),axis=0)
            x_lev_b = np.concatenate((x_lev_b, prev_inputs),axis=-1)

        if self.separate_timedim:
            # print("inds", indices)
            # x_lev_b = x_lev_b.reshape(-1,self.nlev, self.nx)
            # x_sfc_b = x_sfc_b.reshape(-1,self.nx_sfc)
            # y_lev_b = y_lev_b.reshape(-1,self.nlev, self.ny)
            # y_sfc_b = y_sfc_b.reshape(-1,self.ny_sfc)
            x_lev_b.shape = (-1,self.nlev, self.nx)
            x_sfc_b.shape = (-1,self.nx_sfc)
            y_lev_b.shape = (-1,self.nlev, self.ny)
            y_sfc_b.shape = (-1,self.ny_sfc)
                
        #hdf.close()
  
        if self.reverse_input_norm:
            if self.use_numba:
                reverse_input_norm_numba(x_lev_b, self.xcoeff_lev_ref[0], self.xcoeff_lev_ref[1])

            else:
                x_lev_b = x_lev_b * (self.xcoeff_lev_ref[1]) + self.xcoeff_lev_ref[0] 

            x_sfc_b = x_sfc_b * (self.xcoeff_sca_ref[1]) + self.xcoeff_sca_ref[0] 
            
        if self.remove_past_sfc_inputs:
            x_sfc_b = np.delete(x_sfc_b,(17, 18, 19, 20, 21),axis=1)
          
        if self.snowhice_fix:
            x_sfc_b[x_sfc_b>1.0e10] = -1.0
        # elaps = time.time() - t0_it
        # print("Runtime load {:.2f}s".format(elaps))
        # t0_it = time.time()
  
        x_lev_b_denorm = np.copy(x_lev_b[:,:,0:8])

        if self.add_refpres:
            dim0,dim1,dim2 = x_lev_b.shape
            # if self.norm=="minmax":
            refpres_norm = self.refpres_norm.reshape((1,-1,1))
            refpres_norm = np.repeat(refpres_norm, dim0,axis=0)
            #self.x[:,:,nx-1] = refpres_norm
            x_lev_b = np.concatenate((x_lev_b, refpres_norm),axis=2)
            # self.x  = torch.cat((self.x,refpres_norm),dim=3)
            del refpres_norm 
            
        # elaps = time.time() - t0_it
        # print("Runtime refp {:.2f}s".format(elaps))
        # t0_it = time.time()

        if self.v4_to_v5_inputs:
            # qn and liq_ratio instead of cloud water and ice mixing ratios
            #  old array has  T, rh, qliq, qice, X..,.
            ##  new array has  T, rh, qn,   qice, liqratio, X ...
            #  new array has  T, rh, qn,   liqratio, X ...
            # print("doing v4 to v5")
            liq_frac_constrained = self.compute_liq_ratio(x_lev_b[:,:,0])
            # numba optimized
            if self.cloud_exp_norm and self.use_numba:
                v4_to_v5_inputs_numba(x_lev_b, liq_frac_constrained) 
                if self.qinput_prune:
                    x_lev_b[:,0:15,2] = 0.0
            else:
                qn   = x_lev_b[:,:,2]  + x_lev_b[:,:,3]
                if self.qinput_prune:
                    qn[:,0:15] = 0.0
                x_lev_b[:,:,2] = qn
                x_lev_b[:,:,3] = liq_frac_constrained
                if self.cloud_exp_norm:
                    x_lev_b[:,:,2] = 1 - np.exp(-x_lev_b[:,:,2] * self.lbd_qn)

        else:
            if self.cloud_exp_norm:
                if self.use_numba:
                    cloud_exp_norm_numba(x_lev_b, self.lbd_qc, self.lbd_qi)
                else:
                    x_lev_b[:,:,2] = 1 - np.exp(-x_lev_b[:,:,2] * self.lbd_qc)
                    x_lev_b[:,:,3] = 1 - np.exp(-x_lev_b[:,:,3] * self.lbd_qi)
                    # print("min max liq", x_lev_b[:,:,2].min(), x_lev_b[:,:,2].max() )
                    # print("min max ice", x_lev_b[:,:,3].min(), x_lev_b[:,:,3].max() )
            if self.qinput_prune:
                x_lev_b[:,0:15,2:3] = 0.0
                
        if self.rh_prune:
            x_lev_b[:,:,1] = np.clip(x_lev_b[:,:,1], 0.0, 1.2)

        # elaps = time.time() - t0_it
        # print("Runtime cloudnorm {:.2f}s".format(elaps))         
        # t0_it = time.time()

        if self.apply_new_input_scaling:
        
            if self.use_numba:
                apply_input_norm_numba(x_lev_b, self.xcoeff_lev[0], self.xcoeff_lev[1])  
            else:
                x_lev_b = (x_lev_b - self.xcoeff_lev[0] ) / (self.xcoeff_lev[1])

            # but note, for some variables  mean and min is set to 0, and the scaling is actually x/std!! confusing..
            # x_sfc_b = (x_sfc_b - self.xcoeff_sca[0] ) / (self.xcoeff_sca[2] - self.xcoeff_sca[1])
            x_sfc_b = (x_sfc_b - self.xcoeff_sca[0] ) / (self.xcoeff_sca[1])


        # elaps = time.time() - t0_it
        # print("Runtime in norm {:.2f}s".format(elaps))
        
        # x_lev_b[np.isinf(x_lev_b)] = 0 
        x_lev_b[np.isnan(x_lev_b)] = 0
        
        # for i in range(x_lev_b.shape[-1]):
        #     print("after new scaling", i, "minmax x", x_lev_b[:,:,i].min(), x_lev_b[:,:,i].max())
        #     # for j in range(60):
        #     #     print("liq rat",  x_lev_b[0,j,3])
        # for i in range(x_sfc_b.shape[-1]):
        #     print("after new scaling", i, "minmax xsfc", x_sfc_b[:,i].min(), x_sfc_b[:,i].max())
                    
          
        # x_lev_b = torch.from_numpy(x_lev_b)
        # x_sfc_b = torch.from_numpy(x_sfc_b)
        
        # for i in range(6):
        #     print("O", i, " minmax y ", y_lev_b[:,:,i].min(), y_lev_b[:,:,i].max())
        # for i in range(5):
        #     print("O", i," minmax ysca ", y_sfc_b[:,i].min(), y_sfc_b[:,i].max())

    
        if self.reverse_output_norm:
            y_lev_b = y_lev_b / self.yscale_lev_ref
            y_sfc_b = y_sfc_b / self.yscale_sca_ref
            

        # for i in range(6):
        #     print("OO", i, "minmax y ", y_lev_b[:,:,i].min(), y_lev_b[:,:,i].max())
        
        # if not self.reverse_input_norm:    
        # y_lev_b_denorm = np.copy(y_lev_b)
        # y_sfc_b_denorm = np.copy(y_sfc_b)
        
        # t0_it = time.time()

        if self.use_mp_constraint:
            y_lev_b[:,:,2] = y_lev_b[:,:,2] + y_lev_b[:,:,3] 
            y_lev_b = np.delete(y_lev_b, 3, axis=2) 
        # elif self.pred_liq_ratio:
        #     # total = y_lev_b[:,:,2] + y_lev_b[:,:,3] 
        #     # liq_frac = y_lev_b[:,:,2] / total 
        #     # liq_frac[np.isinf(liq_ratio)] = 0 
        #     # liq_frac[np.isnan(liq_ratio)] = 0
        #     # # print("liq ratio", liq_frac[200,35], "min", liq_frac.min(), "max", liq_frac.max())
            
        #     qliq_before     = x_lev_b_denorm[:,:,2]
        #     qice_before     = x_lev_b_denorm[:,:,3]   
        #     qn_before       = qliq_before + qice_before 
            
        #     dqliq = y_lev_b[:,:,2]
        #     dqice = y_lev_b[:,:,3]  
        #     dqn =  dqliq + dqice
            
        #     qn_new          = qn_before + dqn*1200  
        #     qliq_new        = qliq_before + dqliq*1200
        #     # qice_new        = qice_before + dqice*1200
        #     liq_frac = qliq_new / qn_new
        #     liq_frac[np.isinf(liq_frac)] = 0 
        #     liq_frac[np.isnan(liq_frac)] = 0
        #     liq_frac[liq_frac<0.0] = 0.0
        #     liq_frac[liq_frac>1.0] = 1.0
            
        #     liq_frac = liq_frac**(1/16)
            
        #     # print("liq frac", liq_frac[200,35], "min", liq_frac.min(), "max", liq_frac.max())
        #     # print("len", liq_frac.size, "< -0.1 ", liq_frac[liq_frac<-0.1].size, ">1.1 ", liq_frac[liq_frac>1.1].size)
        #     y_lev_b[:,:,2] = dqn
        #     y_lev_b[:,:,3] = liq_frac 

        # elaps = time.time() - t0_it
        # print("Runtime mp {:.2f}s".format(elaps))
        # t0_it = time.time()
     
        # for i in range(5):
        #     print("OOO", i, "minmax y ", y_lev_b[:,50,i].min(), y_lev_b[:,50,i].max())
        if self.use_numba:
            apply_output_norm_numba(y_lev_b, self.yscale_lev)
        else:
            y_lev_b  = y_lev_b * self.yscale_lev
        
        if self.output_prune:
            y_lev_b[:,0:12,1:] = 0.0
            
        y_sfc_b  = y_sfc_b * self.yscale_sca    

        
        # elaps = time.time() - t0_it
        # print("Runtime out norm {:.2f}s".format(elaps))
        
        # for i in range(5):
        #     print("OOOO", i," minmax y ", y_lev_b[:,50,i].min(), y_lev_b[:,50,i].max(), "std", y_lev_b[:,50,i].std())
        # # for i in range(5):
        #     print("OOOO", i," minmax ysca ", y_sfc_b[:,i].min(), y_sfc_b[:,i].max())
  
        # if self.remove_past_sfc_inputs:
        #     x_sfc_b = np.delete(x_sfc_b,(17, 18, 19, 20, 21),axis=1)
          
        x_lev_b = torch.from_numpy(x_lev_b)
        x_sfc_b = torch.from_numpy(x_sfc_b)
        
        y_lev_b = torch.from_numpy(y_lev_b)
        y_sfc_b = torch.from_numpy(y_sfc_b)
        # y_lev_b_denorm = torch.from_numpy(y_lev_b_denorm)
        # y_sfc_b_denorm = torch.from_numpy(y_sfc_b_denorm)

        # print("gen x_lev shape", x_lev_b.shape)
        gc.collect()

        # return x_lev_b, x_sfc_b, y_lev_b, y_sfc_b, x_lev_b_denorm, y_lev_b_denorm, y_sfc_b_denorm
        return x_lev_b, x_sfc_b, y_lev_b, y_sfc_b, x_lev_b_denorm
        
        # return x_lev_b, x_sfc_b, y_lev_b, y_sfc_b, y_lev_b_denorm, y_sfc_b_denorm
 

def chunkize(filelist, chunk_size, shuffle_before_chunking=False, shuffle_after_chunking=True):
    import random
    # Takes a list, shuffles it (optional), and divides into chunks of length n
    # (no concept of batches within this function, chunk size is given in number of samples)
    def divide(filelist,chunk_size):
        # looping till length l
        for i in range(0, len(filelist), chunk_size): 
            yield filelist[i:i + chunk_size]  
    if shuffle_before_chunking:
        random.shuffle(filelist)
        # # we need the indices to be sorted within a chunk because these indices
        # # are used to index into the first dimension of a H5 file
        # for i in range(filelist):
        #     filelist[i] = sorted(filelist[i])
            
    mylist = list(divide(filelist,chunk_size))
    if shuffle_after_chunking:
        random.shuffle(mylist)  
        
    if shuffle_before_chunking:
        # random.shuffle(filelist)
        # # we need the indices to be sorted within a chunk because these indices
        # # are used to index into the first dimension of a H5 file
        for i in range(len(mylist)):
            mylist[i] = sorted(mylist[i])
    return mylist


class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, num_samples_per_chunk, num_samples, shuffle=False, skip_first=False):
        self.num_samples_per_chunk = num_samples_per_chunk
        self.num_samples = num_samples
        self.skip_first = skip_first 
        if self.skip_first:
            indices_all = list(range(1,self.num_samples))
        else:
            indices_all = list(range(self.num_samples))
        print("Shuffling the indices: {}".format(shuffle))
        self.indices_chunked = chunkize(indices_all,self.num_samples_per_chunk,
                                        shuffle_before_chunking=False,
                                        shuffle_after_chunking=shuffle)
        # print("indices chunked [-1]", self.indices_chunked[-1])
        # one item is one chunk, consisting of chunk_factor*batch_size samples
        
    # def __len__(self):
    #     return self.num_samples // self.batch_size

    def __iter__(self):
        return iter(self.indices_chunked)
        # for batch in self.indices_chunked:
        #     yield batch
