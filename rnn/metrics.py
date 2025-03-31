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


def my_mse(y_true_lev, y_true_sfc, y_pred_lev, y_pred_sfc):
    mse1 = torch.mean(torch.square(y_pred_lev - y_true_lev))
    mse2 = torch.mean(torch.square(y_pred_sfc - y_true_sfc))
    return (mse1+mse2)/2

def rmse(y_true_lev, y_pred_lev):
    val_rmse = torch.sqrt(torch.mean(torch.square(y_pred_lev - y_true_lev),dim=(0)))
    return val_rmse.detach().cpu().numpy()


def compute_biases(y_true_lev, y_pred_lev):

    # mean_t_lev = torch.nanmean(y_true_lev,dim=(0,1))
    # mean_p_lev = torch.nanmean(y_pred_lev,dim=(0,1))
  
    # biases_lev = mean_t_lev - mean_p_lev

    mean_t_lev = torch.nanmean(y_true_lev,dim=(0))
    mean_p_lev = torch.nanmean(y_pred_lev,dim=(0))

    biases_perlev = mean_t_lev - mean_p_lev

    mean_t_nolev = torch.nanmean(mean_t_lev,dim=(0))
    mean_p_nolev = torch.nanmean(mean_p_lev,dim=(0))
  
    biases_nolev = mean_t_nolev - mean_p_nolev

    return biases_nolev.detach().cpu().numpy(), biases_perlev.detach().cpu().numpy()


def compute_absolute_biases(y_true_lev, y_true_sfc, y_pred_lev, y_pred_sfc):
    # 1) Mean across batches
    mean_t_lev  = torch.nanmean(y_true_lev,dim=(0))
    mean_p_lev  = torch.nanmean(y_pred_lev,dim=(0))
    mean_t_sfc  = torch.nanmean(y_true_sfc,dim=(0))
    mean_p_sfc  = torch.nanmean(y_pred_sfc,dim=(0))
    # 2) Diff
    diff_lev    = mean_t_lev - mean_p_lev
    diff_sfc    = mean_t_sfc - mean_p_sfc
    # 3) Abs, so that compensating biases are not hidden 
    biases_lev = torch.abs(diff_lev) # (levels, features)
    biases_sfc = torch.abs(diff_sfc) # (features)
    # 4) Mean again across levels / features as needed to distill into scalar metric
    biases_lev = torch.nanmean(biases_lev, dim=(0)) 
    
    return biases_lev.detach().cpu().numpy(), biases_sfc.detach().cpu().numpy()



# def my_mse_flatten(y_true_lev, y_true_sfc, y_pred_lev, y_pred_sfc):

#     if len(y_true_lev.shape)==4: # autoregressive, time dimension included 
#         y_pred_flat =  torch.cat(( y_pred_lev.flatten(start_dim=0,end_dim=1).flatten(start_dim=1) , y_pred_sfc.flatten(start_dim=0,end_dim=1) ),dim=1)
#         y_true_flat =  torch.cat(( y_true_lev.flatten(start_dim=0,end_dim=1).flatten(start_dim=1) , y_true_sfc.flatten(start_dim=0,end_dim=1) ),dim=1)
#     else:
#         y_pred_flat =  torch.cat((y_pred_lev.flatten(start_dim=1),y_pred_sfc),dim=1)
#         y_true_flat =  torch.cat((y_true_lev.flatten(start_dim=1),y_true_sfc),dim=1)

#     mse = torch.mean(torch.square(y_pred_flat - y_true_flat))
#     return mse

def mse_flatten(y_true_lev, y_true_sfc, y_pred_lev, y_pred_sfc, weights=None):

    if weights is not None:
        y_pred_lev = weights*y_pred_lev
        y_true_lev = weights*y_true_lev
        
    y_pred_flat =  torch.cat((y_pred_lev.flatten(start_dim=1),y_pred_sfc),dim=1)
    y_true_flat =  torch.cat((y_true_lev.flatten(start_dim=1),y_true_sfc),dim=1)

    mse = torch.mean(torch.square(y_pred_flat - y_true_flat))
    return mse

def huber_flatten(y_true_lev, y_true_sfc, y_pred_lev, y_pred_sfc, weights=None):
    
    if weights is not None:
        y_pred_lev = weights*y_pred_lev
        y_true_lev = weights*y_true_lev

    y_pred_flat =  torch.cat((y_pred_lev.flatten(start_dim=1),y_pred_sfc),dim=1)
    y_true_flat =  torch.cat((y_true_lev.flatten(start_dim=1),y_true_sfc),dim=1)
    
    criterion = nn.SmoothL1Loss()
    err =  criterion(y_pred_flat, y_true_flat)

    mae = nn.L1Loss()
    mae_err = mae(y_pred_flat, y_true_flat)
    return err, mae_err


def get_mse_flatten(weights):
    def my_mse_flatten(y_true_lev, y_true_sfc, y_pred_lev, y_pred_sfc):
        return mse_flatten(y_true_lev, y_true_sfc, y_pred_lev, y_pred_sfc, weights=weights)
    return my_mse_flatten

def get_huber_flatten(weights):
    def my_huber_flatten(y_true_lev, y_true_sfc, y_pred_lev, y_pred_sfc):
        return huber_flatten(y_true_lev, y_true_sfc, y_pred_lev, y_pred_sfc, weights=weights)
    return my_huber_flatten

def energy_metric(yto, ypo, sp, hyai,hybi):
    
    cp = torch.tensor(1004.0)
    Lv = torch.tensor(2.5104e6)
    Lf = torch.tensor(3.34e5)
    one_over_grav = torch.tensor(0.1020408163) # 1/9.8
    if len(yto.shape)==3:
        thick= one_over_grav*(sp * (hybi[1:61].view(1,-1)-hybi[0:60].view(1,-1)) 
                             + torch.tensor(100000)*(hyai[1:61].view(1,-1)-hyai[0:60].view(1,-1)))
    
        dT_pred = ypo[:,:,0]
        dq_pred = ypo[:,:,1] 
        
        dT_true = yto[:,:,0]
        dq_true = yto[:,:,1]
        
        dql_pred = ypo[:,:,2] 
        dql_true = yto[:,:,2] 
        
        energy=torch.mean(torch.square(torch.sum(thick*(dq_pred*Lv + dT_pred*cp + dql_pred*Lf),1)
                                -      torch.sum(thick*(dq_true*Lv + dT_true*cp + dql_true*Lf),1)))
    else: 
        # time dimension included
                                       #      batch,time,1     (1,1,30)
        thick= one_over_grav *(sp * (hybi[1:61].view(1,1,-1)-hybi[0:60].view(1,1,-1)) 
             + torch.tensor(100000)*(hyai[1:61].view(1,1,-1)-hyai[0:60].view(1,1,-1)))
        dT_pred = ypo[:,:,:,0]
        dq_pred = ypo[:,:,:,1] 
        
        dT_true = yto[:,:,:,0]
        dq_true = yto[:,:,:,1] 
        
        dql_pred = ypo[:,:,:,2] 
        dql_true = yto[:,:,:,2] 
        
        energy=torch.mean(torch.square(torch.sum(thick*(dq_pred*Lv + dT_pred*cp + dql_pred*Lf),2)
                                -      torch.sum(thick*(dq_true*Lv + dT_true*cp + dql_true*Lf),2)))
    return energy

def get_energy_metric(hyai, hybi):
    def energy(y_true, y_pred, sp):
        return energy_metric(y_true, y_pred, sp, hyai, hybi)
    return energy

def mse(y_true, y_pred):
    mse = torch.mean(torch.square(y_pred- y_true))
    return mse

def get_water_conservation(hyai, hybi):
    def wc(pred_lev, pred_sfc, sp, LHF):
        Lv = torch.tensor(2.5104e6)
        precip = (pred_sfc[:,2] + pred_sfc[:,3]) * 1000.0 # density of water. m s-1 * 1000 kg m-3 = kg m-2 s-1 
        one_over_grav = torch.tensor(0.1020408163) # 1/9.8
        thick= one_over_grav*(sp * (hybi[1:61].view(1,-1)-hybi[0:60].view(1,-1)) 
                        + torch.tensor(100000)*(hyai[1:61].view(1,-1)-hyai[0:60].view(1,-1)))

        # pressure_grid_p1 = 1e5 * hyai.reshape(1,-1) # (1, 61)
        # pressure_grid_p2 = hybi.reshape(1,-1) * sp # (batch_size, 61)
        # pressure_grid = pressure_grid_p1 + pressure_grid_p2 # (batch_size, 61)
        # dp = pressure_grid[:,1:] - pressure_grid[:,:-1] # (batch_size, 60)
        # thick2 = one_over_grav * dp 

        qv = pred_lev[:,:,1]
        ql = pred_lev[:,:,2]
        qi = pred_lev[:,:,3]

        lhs = torch.sum(thick*(qv + ql + qi),1)
        rhs = LHF / Lv - precip
        # print("lhs", lhs[100].item(), "rhs", rhs[100].item(), "precip", precip[100].item())
        # print( "sp", sp[100].item(), "thick[30]", thick[100,30], "thick2", thick2[100,30])
        # diff = torch.mean(lhs - rhs)
        diff = lhs - rhs
        return diff 
    return wc

# def water_conservation(pred_lev, pred_sfc):




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

def CRPS(y, y_sfc, y_pred, y_sfc_pred, beta=1, return_low_var_inds=False):
    """
    Calculate Continuous Ranked Probability Score.

    Parameters:
    - y_pred (torch.Tensor): Prediction tensor.   (nens*batch,30,4)
      Needs to be transposed to (batch, nens, 30*4)
    - y (torch.Tensor): Ground truth tensor.  (batch,30,4)
      needs to be reshaped to (batch,1,30*4) 

    Returns:
    - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: CRPS and its components.
    """ 
    
    # print("shape ypred", y_pred.shape, "y true", y.shape)

    # if len(y.shape)==4:
    #     #autoreg training with time windows,
    #     # y shape: (ntime, nb, nlev, ny)
    #     # ypred:  (ntime, nens*nb, nlev, ny)    
    #     time_steps,batch_size,seq_size,feature_size = y.shape
    #     ens_size = y_pred.shape[1] // batch_size
    #     y_pred = torch.reshape(y_pred, (time_steps, ens_size, batch_size, seq_size*feature_size))
        
        
        
    #     y_pred = torch.transpose(y_pred, 1, 2) # time, batch, ens, seq_size*feature_size))
    #     y_pred = torch.reshape(y_pred, (time_steps*batch_size, ens_size, seq_size*feature_size))
    #     batch_size_new = y_pred.shape[0]
    #     y = torch.reshape(y, (batch_size_new, 1, seq_size*feature_size))
    #     # print("shape ypred", y_pred.shape, "y true", y.shape)
    # else:
        
    batch_size,seq_size,feature_size = y.shape
    ens_size = y_pred.shape[0] // batch_size           # seq_size*feature_size
    y_pred = torch.reshape(y_pred, (ens_size, batch_size, -1))
    # NEW: ADD SFC
    y_sfc_pred = torch.reshape(y_sfc_pred, (ens_size, batch_size, -1))
    y_pred = torch.cat((y_pred, y_sfc_pred), axis=-1)
    y_pred = torch.transpose(y_pred, 0, 1) # batch_size, ens_size, nlev*ny
    y_sfc = torch.reshape(y_sfc,(batch_size,1,-1))
    y = torch.reshape(y, (batch_size, 1, -1))
    y = torch.cat((y, y_sfc),axis=-1)

    # cdist
    # x1 (Tensor) – input tensor of shape B×P×M
    # x2 (Tensor) – input tensor of shape B×R×M
    
    # ypred (batch,nens,30*4)   y  (batch,1,30*4)   
    # cdist out term1 (batch,1,nens)
    # cdist out term2 (batch, nens, nens)   
    # print("shape ytrue", y.shape, "y pred", y_pred.shape )
    MAE     = torch.cdist(y, y_pred).mean() 
    # ens_var = torch.cdist(y_pred, y_pred).mean(0).sum() / (self.ens_size * (self.ens_size - 1))
    
    cmean_out = torch.cdist(y_pred, y_pred) # B, nens, nens

    ens_var = cmean_out.mean(0).sum() / (ens_size * (ens_size - 1))
    
    MAE         /= y_pred.size(-1) ** 0.5
    ens_var     /= y_pred.size(-1) ** 0.5
    
    CRPS =  beta * 2 * MAE - ens_var # beta should be 1
    
    if return_low_var_inds:
        x = cmean_out[:,0,1]
        q = torch.quantile(x, 0.05,  keepdim=False)
        inds = x <= q

        # print("shape cmean out", cmean_out.shape, "shape inds", inds.shape)
        # print("cmean out 0, 01,   2048, 01", cmean_out[0,0,1], cmean_out[2048,0,1])
        # return beta * 2 * MSE - ens_var, MSE, ens_var, inds
        return CRPS,  MAE, ens_var, inds

    else:
        return CRPS, MAE, ens_var

def get_CRPS(beta): 
    def customCRPS(y_true, y_true_sfc, y_pred, y_pred_sfc):
        return CRPS(y_true, y_true_sfc, y_pred, y_pred_sfc, beta)
    return customCRPS