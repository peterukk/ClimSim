#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss functions
"""

from __future__ import absolute_import 
from __future__ import print_function
import numpy as np 
import torch 
import torch.nn as nn


def corrcoeff_pairs_batchfirst(A, B):
    nb, nlev, nx = A.shape
    # A and B are (N,M1,M2) vectors. 
    # Reshape to (N,M1*M2), then compute corrcoef for each (N,m),(N,m) pair
    # reshape back at the end
    A = A.reshape(nb,-1)
    B = B.reshape(nb,-1)

    A_mA = A - A.mean(0)[None,:]
    B_mB = B - B.mean(0)[None,:] # (N, M)  

    # Sum of squares across rows
    ssA = (A_mA**2).sum(0)
    ssB = (B_mB**2).sum(0)

    # Finally get corr coeff
    # dividend = np.dot(A_mA, B_mB.T) # (M,M)
    dividend = np.sum(A_mA*B_mB, axis=0)  # (M)
    
    # divisor = np.sqrt(np.dot(ssA[:, None],ssB[None])) # (M, M)
    divisor =  np.sqrt(ssA*ssB)
    corrcoef = dividend / divisor
    return corrcoef.reshape(nlev, nx)


def my_mse(y_true_lay, y_true_sfc, y_pred_lay, y_pred_sfc):
    mse1 = torch.mean(torch.square(y_pred_lay - y_true_lay))
    mse2 = torch.mean(torch.square(y_pred_sfc - y_true_sfc))
    return (mse1+mse2)/2


# def compute_biases(y_true_lay, y_true_sfc, y_pred_lay, y_pred_sfc):
#     # y_true_lay = torch.abs(y_true_lay); y_pred_lay = torch.abs(y_pred_lay)
#     # y_true_sfc = torch.abs(y_true_sfc); y_pred_sfc = torch.abs(y_pred_sfc) 
    
#     # mean_t_lay = torch.nanmean(y_true_lay,dim=(0,1))
#     # mean_p_lay = torch.nanmean(y_pred_lay,dim=(0,1))
#     mean_t_lay = torch.nanmean(y_true_lay,dim=(0))
#     mean_p_lay = torch.nanmean(y_pred_lay,dim=(0))
    
#     mean_t_sfc = torch.nanmean(y_true_sfc,dim=(0))
#     mean_p_sfc = torch.nanmean(y_pred_sfc,dim=(0))
    
#     mean_t_lay = torch.abs(mean_t_lay); mean_p_lay = torch.abs(mean_p_lay) 
#     mean_t_sfc = torch.abs(mean_t_sfc); mean_p_sfc = torch.abs(mean_p_sfc)
#     # Mean across levels and variables only after taking torch.abs(),
#     # so that compensating biases are not hidden (Zeyuan Hu)
#     mean_t_lay = torch.nanmean(mean_t_lay,dim=(0)); mean_p_lay = torch.nanmean(mean_p_lay,dim=(0))
    
#     biases_lev = mean_t_lay - mean_p_lay
#     biases_sfc = mean_t_sfc - mean_p_sfc

#     return biases_lev.detach().cpu().numpy(), biases_sfc.detach().cpu().numpy()


def compute_biases(y_true_lay, y_true_sfc, y_pred_lay, y_pred_sfc):
    # 1) Mean across batches
    mean_t_lay  = torch.nanmean(y_true_lay,dim=(0))
    mean_p_lay  = torch.nanmean(y_pred_lay,dim=(0))
    mean_t_sfc  = torch.nanmean(y_true_sfc,dim=(0))
    mean_p_sfc  = torch.nanmean(y_pred_sfc,dim=(0))
    # 2) Diff
    diff_lay    = mean_t_lay - mean_p_lay
    diff_sfc    = mean_t_sfc - mean_p_sfc
    # 3) Abs, so that compensating biases are not hidden 
    biases_lev = torch.abs(diff_lay) # (levels, features)
    biases_sfc = torch.abs(diff_sfc) # (features)
    # 4) Mean again across levels / features as needed to distill into scalar metric
    biases_lev = torch.nanmean(biases_lev, dim=(0)) 
    
    return biases_lev.detach().cpu().numpy(), biases_sfc.detach().cpu().numpy()

# def my_mse_flatten(y_true_lay, y_true_sfc, y_pred_lay, y_pred_sfc):

#     if len(y_true_lay.shape)==4: # autoregressive, time dimension included 
#         y_pred_flat =  torch.cat(( y_pred_lay.flatten(start_dim=0,end_dim=1).flatten(start_dim=1) , y_pred_sfc.flatten(start_dim=0,end_dim=1) ),dim=1)
#         y_true_flat =  torch.cat(( y_true_lay.flatten(start_dim=0,end_dim=1).flatten(start_dim=1) , y_true_sfc.flatten(start_dim=0,end_dim=1) ),dim=1)
#     else:
#         y_pred_flat =  torch.cat((y_pred_lay.flatten(start_dim=1),y_pred_sfc),dim=1)
#         y_true_flat =  torch.cat((y_true_lay.flatten(start_dim=1),y_true_sfc),dim=1)

#     mse = torch.mean(torch.square(y_pred_flat - y_true_flat))
#     return mse

def mse_flatten(y_true_lay, y_true_sfc, y_pred_lay, y_pred_sfc, weights=None):

    if weights is not None:
        y_pred_lay = weights*y_pred_lay
        y_true_lay = weights*y_true_lay
        
    y_pred_flat =  torch.cat((y_pred_lay.flatten(start_dim=1),y_pred_sfc),dim=1)
    y_true_flat =  torch.cat((y_true_lay.flatten(start_dim=1),y_true_sfc),dim=1)

    mse = torch.mean(torch.square(y_pred_flat - y_true_flat))
    return mse

def huber_flatten(y_true_lay, y_true_sfc, y_pred_lay, y_pred_sfc, weights=None):
    
    if weights is not None:
        y_pred_lay = weights*y_pred_lay
        y_true_lay = weights*y_true_lay

    y_pred_flat =  torch.cat((y_pred_lay.flatten(start_dim=1),y_pred_sfc),dim=1)
    y_true_flat =  torch.cat((y_true_lay.flatten(start_dim=1),y_true_sfc),dim=1)
    
    criterion = nn.SmoothL1Loss()
    err =  criterion(y_pred_flat, y_true_flat)
    return err

def get_mse_flatten(weights):
    def my_mse_flatten(y_true_lay, y_true_sfc, y_pred_lay, y_pred_sfc):
        return mse_flatten(y_true_lay, y_true_sfc, y_pred_lay, y_pred_sfc, weights=weights)
    return my_mse_flatten

def get_huber_flatten(weights):
    def my_huber_flatten(y_true_lay, y_true_sfc, y_pred_lay, y_pred_sfc):
        return huber_flatten(y_true_lay, y_true_sfc, y_pred_lay, y_pred_sfc, weights=weights)
    return my_huber_flatten

def energy_metric(yto, ypo, sp, hyai,hybi):
    
    cp = torch.tensor(1004.0)
    Lv = torch.tensor(2.5104e6)
    one_over_grav = torch.tensor(0.1020408163) # 1/9.8
    if len(yto.shape)==3:
        thick= one_over_grav*(sp * (hybi[1:61].view(1,-1)-hybi[0:60].view(1,-1)) 
                             + torch.tensor(100000)*(hyai[1:61].view(1,-1)-hyai[0:60].view(1,-1)))
    
        dT_pred = ypo[:,:,0]
        dq_pred = ypo[:,:,1] 
        
        dT_true = yto[:,:,0]
        dq_true = yto[:,:,1]
        
        energy=torch.mean(torch.square(torch.sum(dq_pred*thick*(Lv)+dT_pred*thick*cp,1)
                                -      torch.sum(dq_true*thick*(Lv)+dT_true*thick*cp,1)))
    else: 
        # time dimension included
                                       #      batch,time,1     (1,1,30)
        thick= one_over_grav *(sp * (hybi[1:61].view(1,1,-1)-hybi[0:60].view(1,1,-1)) 
             + torch.tensor(100000)*(hyai[1:61].view(1,1,-1)-hyai[0:60].view(1,1,-1)))
        dT_pred = ypo[:,:,:,0]
        dq_pred = ypo[:,:,:,1] 
        
        dT_true = yto[:,:,:,0]
        dq_true = yto[:,:,:,1] 
        
        energy=torch.mean(torch.square(torch.sum(dq_pred*thick*Lv + dT_pred*thick*cp,2)
                                -      torch.sum(dq_true*thick*Lv + dT_true*thick*cp,2)))
    return energy

def get_energy_metric(hyai, hybi):
    def energy(y_true, y_pred, sp):
        return energy_metric(y_true, y_pred, sp, hyai, hybi)
    return energy

def mse(y_true, y_pred):
    mse = torch.mean(torch.square(y_pred- y_true))
    return mse



# def loss_con(y_true_norm, y_pred_norm, y_true, y_pred, sp, _lambda):

#     energy = energy_metric(y_true, y_pred, sp, hyai,hybi)
#     #mse = torch.mean(torch.square(y_pred- y_true))
#     mse = my_mse_flatten(y_true_norm, y_pred_norm)
#     loss = mse + _lambda*energy
#     return loss, energy, mse

# def get_loss_con(hyai, hybi, _lambda, denorm_func):
#     def hybrid_loss(y_true_norm, y_pred_norm, y_true, y_pred, sp):
#         return loss_con(y_true_norm, y_pred_norm, y_true, y_pred, sp, _lambda)
#     return hybrid_loss

def my_hybrid_loss(mse, energy, _lambda):
    loss = mse + _lambda*energy
    return loss 

def get_hybrid_loss(_lambda):
    def hybrid_loss(mse, energy):
        return my_hybrid_loss(mse, energy, _lambda)
    return hybrid_loss
