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
import torch.nn.functional as F

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


def compute_absolute_biases(y_true_lev, y_true_sfc, y_pred_lev, y_pred_sfc, numpy=False):
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
    
    if numpy:
        return biases_lev.detach().cpu().numpy(), biases_sfc.detach().cpu().numpy()
    else:
        return biases_lev, biases_sfc



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

    return err

def metrics_flatten(y_true_lev, y_true_sfc, y_pred_lev, y_pred_sfc, weights=None):
    
    if weights is not None:
        y_pred_lev = weights*y_pred_lev
        y_true_lev = weights*y_true_lev

    y_pred_flat =  torch.cat((y_pred_lev.flatten(start_dim=1),y_pred_sfc),dim=1)
    y_true_flat =  torch.cat((y_true_lev.flatten(start_dim=1),y_true_sfc),dim=1)
    
    func_huber = nn.SmoothL1Loss()
    huber =  func_huber(y_pred_flat, y_true_flat)

    func_mae = nn.L1Loss()
    mae = func_mae(y_pred_flat, y_true_flat)

    mse = torch.mean(torch.square(y_pred_flat - y_true_flat))

    return huber, mse, mae

def get_mse_flatten(weights):
    def my_mse_flatten(y_true_lev, y_true_sfc, y_pred_lev, y_pred_sfc):
        return mse_flatten(y_true_lev, y_true_sfc, y_pred_lev, y_pred_sfc, weights=weights)
    return my_mse_flatten

def get_huber_flatten(weights):
    def my_huber_flatten(y_true_lev, y_true_sfc, y_pred_lev, y_pred_sfc):
        return huber_flatten(y_true_lev, y_true_sfc, y_pred_lev, y_pred_sfc, weights=weights)
    return my_huber_flatten

def get_metrics_flatten(weights):
    def my_loss_flatten(y_true_lev, y_true_sfc, y_pred_lev, y_pred_sfc):
        return metrics_flatten(y_true_lev, y_true_sfc, y_pred_lev, y_pred_sfc, weights=weights)
    return my_loss_flatten 

def mse(y_true, y_pred):
    mse = torch.mean(torch.square(y_pred- y_true))
    return mse

# def energy_metric(yto, ypo, sp, hyai,hybi):
    
#     cp = torch.tensor(1004.0)
#     Lv = torch.tensor(2.5104e6)
#     Lf = torch.tensor(3.34e5)
#     one_over_grav = torch.tensor(0.1020408163) # 1/9.8
#     if len(yto.shape)==3:
#         thick= one_over_grav*(sp * (hybi[1:61].view(1,-1)-hybi[0:60].view(1,-1)) 
#                              + torch.tensor(100000)*(hyai[1:61].view(1,-1)-hyai[0:60].view(1,-1)))
    
#         dT_pred = ypo[:,:,0]
#         dq_pred = ypo[:,:,1] 
        
#         dT_true = yto[:,:,0]
#         dq_true = yto[:,:,1]
        
#         dql_pred = ypo[:,:,2] 
#         dql_true = yto[:,:,2] 
        
#         energy=torch.mean(torch.square(torch.sum(thick*(dq_pred*Lv + dT_pred*cp + dql_pred*Lf),1)
#                                 -      torch.sum(thick*(dq_true*Lv + dT_true*cp + dql_true*Lf),1)))
#     else: 
#         # time dimension included
#                                        #      batch,time,1     (1,1,30)
#         thick= one_over_grav *(sp * (hybi[1:61].view(1,1,-1)-hybi[0:60].view(1,1,-1)) 
#              + torch.tensor(100000)*(hyai[1:61].view(1,1,-1)-hyai[0:60].view(1,1,-1)))
#         dT_pred = ypo[:,:,:,0]
#         dq_pred = ypo[:,:,:,1] 
        
#         dT_true = yto[:,:,:,0]
#         dq_true = yto[:,:,:,1] 
        
#         dql_pred = ypo[:,:,:,2] 
#         dql_true = yto[:,:,:,2] 
        
#         energy=torch.mean(torch.square(torch.sum(thick*(dq_pred*Lv + dT_pred*cp + dql_pred*Lf),2)
#                                 -      torch.sum(thick*(dq_true*Lv + dT_true*cp + dql_true*Lf),2)))
#     return energy

# def get_energy_metric(hyai, hybi):
#     def energy(y_true, y_pred, sp):
#         return energy_metric(y_true, y_pred, sp, hyai, hybi)
#     return energy



def get_energy_metric(hyai, hybi):
    def em(yto, ypo, sp, timesteps):
        #  y: (batch*timesteps, lev, ny)
        cp = torch.tensor(1004.0)
        Lv = torch.tensor(2.5104e6)
        Lf = torch.tensor(3.34e5)
        one_over_grav = torch.tensor(0.1020408163) # 1/9.8
        
        thick= one_over_grav*(sp * (hybi[1:61].view(1,-1)-hybi[0:60].view(1,-1)) 
                             + torch.tensor(100000)*(hyai[1:61].view(1,-1)-hyai[0:60].view(1,-1)))
    
        dT_pred = ypo[:,:,0]
        dT_true = yto[:,:,0]

        dq_pred = ypo[:,:,1] 
        dq_true = yto[:,:,1]
        
        dql_pred = ypo[:,:,2] 
        dql_true = yto[:,:,2] 
            
        energy_pred = torch.sum(thick*(dq_pred*Lv + dT_pred*cp + dql_pred*Lf),1)
        energy_true = torch.sum(thick*(dq_true*Lv + dT_true*cp + dql_true*Lf),1)
        # (batch)
        
        energy_pred = torch.reshape(energy_pred,(timesteps, -1))
        energy_pred = torch.mean(energy_pred,dim=0)
        
        energy_true = torch.reshape(energy_true,(timesteps, -1))
        energy_true = torch.mean(energy_true,dim=0)

        energy_mse=torch.mean(torch.square(energy_pred - energy_true))
        return energy_mse 
    
    return em
    
def get_water_conservation(hyai, hybi):
    def wc(pred_lev, pred_sfc, sp, LHF, xlay, timesteps, printdebug=False): #, xlay, printdebug=False):
        Lv = torch.tensor(2.5104e6)
        # precip = (pred_sfc[:,2] + pred_sfc[:,3]) * 1000.0 # density of water. m s-1 * 1000 kg m-3 = kg m-2 s-1 
        precip = (pred_sfc[:,3]) * 1000.0 # density of water. m s-1 * 1000 kg m-3 = kg m-2 s-1 

        one_over_grav = torch.tensor(0.1020408163) # 1/9.8
        thick= one_over_grav*(sp * (hybi[1:61].view(1,-1)-hybi[0:60].view(1,-1)) 
            + torch.tensor(100000)*(hyai[1:61].view(1,-1)-hyai[0:60].view(1,-1)))

        # qv = pred_lev[:,:,1]
        # ql = pred_lev[:,:,2]
        # qi = pred_lev[:,:,3]
        # dp_water = thick*(qv + ql + qi)
        dp_water = thick*(torch.sum(pred_lev[:,:,1:4],dim=2))
        lhs = torch.sum(dp_water,1)
        # rhs = LHF / Lv - precip
        rhs = - precip # Latent heat flux should not be part of loss, as land model is not directly coupled to CRM?

        # (batch)
        rhs = torch.reshape(rhs,(timesteps, -1))
        rhs = torch.mean(rhs,dim=0)
        lhs = torch.reshape(lhs,(timesteps, -1))
        lhs = torch.mean(lhs,dim=0)

        if printdebug: 
            total_water_dyn = torch.sum(thick*xlay[:,:,7],1)
            total_water_dyn = torch.reshape(total_water_dyn,(timesteps, -1))
            total_water_dyn = torch.mean(total_water_dyn,dim=0)

            # print("mean fac2", torch.mean(lhs[precip>0.0] / precip[precip>0.0]).item(), "std fac", torch.std(lhs[precip>0.0]/precip[precip>0.0]).item())
            # print("fac precip / lhs", torch.nanmean(precip/lhs).item(), "fac rhs / lhs", torch.nanmean(rhs/lhs).item())

            # print("rhs", torch.mean(rhs).item(), "precip ", torch.mean(precip).item(), "lhs = dp_waterwater ", torch.mean(lhs).item())
            # print("rhs", torch.mean(rhs).item(), "rhs with dyn", torch.mean(rhs + total_water_dyn).item()) #, "LHF/lv", torch.mean(LHF/Lv).item(), "P ", torch.mean(precip).item(), "LHF/Lv + P", torch.mean(LHF/Lv+precip).item())
            # print("lhs", torch.mean(lhs).item(), "lhs with dyn", torch.mean(lhs-total_water_dyn).item())

            print("mean fac ", torch.nanmean((lhs) / rhs).item())
            print("mean fac with dyn", torch.nanmean((lhs) / (rhs-total_water_dyn)).item()) #, "std", torch.std((lhs) / (rhs+total_water_dyn)).item())

            # print("fac", lhs[100].item()/rhs[100].item(), "lhs", lhs[100].item(), "rhs", rhs[100].item(), 
            #       "rhs1", LHF[100].item()/Lv.item(), "precip", precip[100].item(), "lfh", LHF[100].item(), "qdyn", total_water_dyn[100].item())
            # print("fac with dyn",(lhs[100].item() -total_water_dyn[100].item()  ) /rhs[100].item() )
            # if precip[100].item()>1e-8:
            #     print("fac without LHF", lhs[100].item()/(-precip[100]).item())
        # print( "sp", sp[100].item(), "thick[30]", thick[100,30], "thick2", thick2[100,30])
        # diff = torch.mean(lhs - rhs)
        diff = lhs - rhs
        # print("diff mean", torch.mean(diff).item(), "diff2 mean", torch.mean(rhs-lhs).item())
        return diff 
    return wc

def get_dprec_ddlwp(hyai, hybi):
    def metric(pred_lev, pred_sfc, true_lev, true_sfc, sp):
        # precip = (pred_sfc[:,2] + pred_sfc[:,3]) * 1000.0 # density of water. m s-1 * 1000 kg m-3 = kg m-2 s-1 
        precip_pred = (pred_sfc[:,3]) * 1000.0 # density of water. m s-1 * 1000 kg m-3 = kg m-2 s-1 
        precip_true = (true_sfc[:,3]) * 1000.0 # density of water. m s-1 * 1000 kg m-3 = kg m-2 s-1 

        one_over_grav = torch.tensor(0.1020408163) # 1/9.8
        thick= one_over_grav*(sp * (hybi[1:61].view(1,-1)-hybi[0:60].view(1,-1)) 
            + torch.tensor(100000)*(hyai[1:61].view(1,-1)-hyai[0:60].view(1,-1)))
        
        water_pred = pred_lev[:,:,1] + pred_lev[:,:,2] +  pred_lev[:,:,3]
        lwp_pred = torch.sum(thick*(water_pred),1)
        
        water_true = true_lev[:,:,1] + true_lev[:,:,2] +  true_lev[:,:,3]
        lwp_true = torch.sum(thick*(water_true),1)
        
        dprec_dlwp_pred = precip_pred / lwp_pred
        dprec_dlwp_pred = torch.sqrt(torch.sqrt(torch.sqrt(dprec_dlwp_pred)))
        dprec_dlwp_true = precip_true / lwp_true
        dprec_dlwp_true = torch.sqrt(torch.sqrt(torch.sqrt(dprec_dlwp_true)))
        
        diff = torch.nanmean(torch.abs(dprec_dlwp_true - dprec_dlwp_pred))

        return diff 
    return metric


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

def CRPS(y, y_sfc, y_pred, y_sfc_pred, timesteps, beta=1, alpha=1.0, return_low_var_inds=False):
    """
    Calculate Continuous Ranked Probability Score (CRPS)
    or Almost fair CRPS if alpha<1.0

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
    # y: ntime*nbatch,      nseq, ny 
    # yp: ntime*nbatch*nens, nseq, ny 
    ns,seq_size,feature_size = y.shape
    batch_size = ns // timesteps
    ens_size = y_pred.shape[0] // (timesteps*batch_size)          
    # y_pred = torch.reshape(y_pred, (ens_size, batch_size, -1))
    y_pred = torch.reshape(y_pred, (timesteps, ens_size, batch_size, seq_size*feature_size))
    y_pred = torch.transpose(y_pred, 1, 2) # time, batch, ens, seq_size*feature_size))
    y_pred = torch.reshape(y_pred, (timesteps*batch_size, ens_size, seq_size*feature_size))

    y_sfc_pred = torch.reshape(y_sfc_pred, (timesteps, ens_size, batch_size, -1))
    y_sfc_pred = torch.transpose(y_sfc_pred, 1, 2) # time, batch, ens, nx_sfc))
    y_sfc_pred = torch.reshape(y_sfc_pred, (timesteps*batch_size, ens_size, -1))
    y_pred = torch.cat((y_pred, y_sfc_pred), axis=-1)

    y = torch.reshape(y, (timesteps*batch_size, 1, seq_size*feature_size))
    y_sfc = torch.reshape(y_sfc, (timesteps*batch_size, 1, -1))
    y = torch.cat((y, y_sfc),axis=-1)

    eps = (1-alpha) / ens_size

    # cdist
    # x1 (Tensor) – input tensor of shape B×P×M
    # x2 (Tensor) – input tensor of shape B×R×M
    
    # ypred (batch,nens,30*4)   y  (batch,1,30*4)   
    # cdist out term1 (batch,1,nens)
    # cdist out term2 (batch, nens, nens)   
    # print("shape ytrue", y.shape, "y pred", y_pred.shape )

    MSE       = torch.cdist(y, y_pred).mean()  # B, 1, nens  --> (1)
    cmean_out = torch.cdist(y_pred, y_pred) # B, nens, nens
    ens_var = ( (1-eps)* cmean_out.mean(0).sum() ) / (ens_size * (ens_size - 1)) 


     # Mean over batch, then summed over ens dim. Should this instead be first sum over ens, then mean over batch?

    ## ens_var = cmean_out.mean() / (ens_size * (ens_size - 1))
    # ens_var = torch.mean(torch.sum(cmean_out,dim=[1,2])) / (ens_size * (ens_size - 1)) 
    # MSE     = torch.mean(torch.sum(torch.cdist(y, y_pred),dim=2)) # B, 1, nens  --> (1)

    MSE         /= y_pred.size(-1) ** 0.5
    ens_var     /= y_pred.size(-1) ** 0.5
    
    CRPS =  beta * 2 * MSE - ens_var # beta should be 1

    if return_low_var_inds:
        x = cmean_out[:,0,1]
        q = torch.quantile(x, 0.05,  keepdim=False)
        inds = x <= q

        # print("shape cmean out", cmean_out.shape, "shape inds", inds.shape)
        # print("cmean out 0, 01,   2048, 01", cmean_out[0,0,1], cmean_out[2048,0,1])
        # return beta * 2 * MSE - ens_var, MSE, ens_var, inds
        return CRPS,  MSE, ens_var, inds

    else:
        return CRPS, MSE, ens_var


def CRPS_scoringrules(y, y_sfc, y_pred, y_sfc_pred, timesteps):
    """
    Calculate Continuous Ranked Probability Score (CRPS)

    Parameters:
    - y_pred (torch.Tensor): Prediction tensor.   (nens*batch,30,4)
      Needs to be transposed to (batch, nens, 30*4)
    - y (torch.Tensor): Ground truth tensor.  (batch,30,4)
      needs to be reshaped to (batch,1,30*4) 

    Returns:
    - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: CRPS and its components.
    """ 
    # y:  ntime*nbatch,      nseq, ny 
    # yp: ntime*nbatch*nens, nseq, ny 
    import scoringrules as sr
    ns,seq_size,feature_size = y.shape
    batch_size = ns // timesteps
    ens_size = y_pred.shape[0] // (timesteps*batch_size)          
    y_pred = torch.reshape(y_pred, (timesteps, ens_size, batch_size, seq_size*feature_size))
    y_pred = torch.transpose(y_pred, 1, 2) # time, batch, ens, seq_size*feature_size))
    y_pred = torch.reshape(y_pred, (timesteps*batch_size, ens_size, seq_size*feature_size))

    y_sfc_pred = torch.reshape(y_sfc_pred, (timesteps, ens_size, batch_size, -1))
    y_sfc_pred = torch.transpose(y_sfc_pred, 1, 2) # time, batch, ens, nx_sfc))
    y_sfc_pred = torch.reshape(y_sfc_pred, (timesteps*batch_size, ens_size, -1))
    y_pred = torch.cat((y_pred, y_sfc_pred), axis=-1)

    y     = torch.reshape(y,     (timesteps*batch_size, seq_size*feature_size))
    y_sfc = torch.reshape(y_sfc, (timesteps*batch_size, -1))
    y = torch.cat((y, y_sfc), axis=-1)
    # print("y shape and dev", y.shape, y.device, "pred", y_pred.shape, y_pred.device)
    CRPS = sr.crps_ensemble(y, y_pred, m_axis=1, backend='torch', estimator='fair')
    # CRPS = CRPS.mean()
    CRPS = CRPS.sum(dim=-1).mean()

    # print("CRPS shape", CRPS.shape)
    y       = y.reshape((timesteps*batch_size,1,-1))
    y_pred  = y_pred.reshape((timesteps*batch_size,ens_size,-1))

    eps = 1 / ens_size

    # cdist
    # x1 (Tensor) – input tensor of shape B×P×M
    # x2 (Tensor) – input tensor of shape B×R×M
    # ypred (batch,nens,30*4)   y  (batch,1,30*4)   
    # cdist out term1 (batch,1,nens)
    # cdist out term2 (batch, nens, nens)   
    MSE       = torch.cdist(y, y_pred).mean()  # B, 1, nens  --> (1)
    cmean_out = torch.cdist(y_pred, y_pred) # B, nens, nens
    ens_var = ( (1-eps)* cmean_out.mean(0).sum() ) / (ens_size * (ens_size - 1)) 
    MSE         /= y_pred.size(-1) ** 0.5
    ens_var     /= y_pred.size(-1) ** 0.5
    

    return CRPS, MSE, ens_var

def CRPS_l1(y, y_sfc, y_pred, y_sfc_pred, timesteps, beta=1):
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
    # y: ntime*nbatch,      nseq, ny 
    # yp: ntime*nbatch*nens, nseq, ny 
    ns,seq_size,feature_size = y.shape
    batch_size = ns // timesteps
    ens_size = y_pred.shape[0] // (timesteps*batch_size)          
    # y_pred = torch.reshape(y_pred, (ens_size, batch_size, -1))
    y_pred = torch.reshape(y_pred, (timesteps, ens_size, batch_size, seq_size*feature_size))
    y_pred = torch.transpose(y_pred, 1, 2) # time, batch, ens, seq_size*feature_size))
    y_pred = torch.reshape(y_pred, (timesteps*batch_size, ens_size, seq_size*feature_size))

    y_sfc_pred = torch.reshape(y_sfc_pred, (timesteps, ens_size, batch_size, -1))
    y_sfc_pred = torch.transpose(y_sfc_pred, 1, 2) # time, batch, ens, nx_sfc))
    y_sfc_pred = torch.reshape(y_sfc_pred, (timesteps*batch_size, ens_size, -1))
    y_pred = torch.cat((y_pred, y_sfc_pred), axis=-1)

    y = torch.reshape(y, (timesteps*batch_size, 1, seq_size*feature_size))
    y_sfc = torch.reshape(y_sfc, (timesteps*batch_size, 1, -1))
    y = torch.cat((y, y_sfc),axis=-1)

    # ypred (batch,nens,30*4)   y  (batch,1,30*4)   

    # Args:
    #     x: Predicted ensemble values [batch, seq_len, x, y].
    #     y: Target values [batch, 1, x, y].

    # Returns:
    #     Skill, spread, and spectral losses.
    # """
    # # Compute magnitude of spectral terms
    # x_k = torch.fft.fft2(x, norm='ortho').abs()
    # y_k = torch.fft.fft2(y, norm='ortho').abs()

    # # Compute loss components L1 loss for CRPS, L2 for Energy Score
    skill  = F.l1_loss(y_pred, y)              # mean L1 loss over ensemble members
    spread = F.l1_loss(y_pred[:, 0], y_pred[:, 1])  # Ensemble spread (assumes ens_size = 2)
    # spec   = F.l1_loss(x_k, y_k)          # Spectral loss over ensemble members

    # # Compute weighted loss (spec_weight = 1 by default)
    loss = skill - 0.5 * spread
    return loss, skill, spread

def CRPS_anemoi(y, y_sfc, y_pred, y_sfc_pred, timesteps, beta=1):
    """
    Calculate Continuous Ranked Probability Score.

    Parameters:
    - y_pred (torch.Tensor): Prediction tensor.   (ntime*nens*batch,nlev,ny)
      Needs to be transposed to  (nens, ntime*batch, nlev*ny)
    - y (torch.Tensor): Ground truth tensor.  (ntime*batch,nlev,ny)
      needs to be reshaped to (1, ntime*batch, nlev*ny) 

    Returns:
    - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: CRPS and its components.
    """ 
    
    ns,seq_size,feature_size = y.shape
    batch_size = ns // timesteps
    ens_size = y_pred.shape[0] // (timesteps*batch_size)          
    # y_pred = torch.reshape(y_pred, (ens_size, batch_size, -1))
    y_pred = torch.reshape(y_pred, (timesteps, ens_size, batch_size, seq_size*feature_size))
    y_pred = torch.transpose(y_pred, 0, 1) # ens, time, batch, seq_size*feature_size))
    y_pred = torch.reshape(y_pred, (ens_size, timesteps*batch_size, seq_size*feature_size))

    y_sfc_pred = torch.reshape(y_sfc_pred, (timesteps, ens_size, batch_size, -1))
    y_sfc_pred = torch.transpose(y_sfc_pred, 0, 1) # ens, time, batch, nx_sfc))
    y_sfc_pred = torch.reshape(y_sfc_pred, (ens_size, timesteps*batch_size, -1))
    y_pred = torch.cat((y_pred, y_sfc_pred), axis=-1)

    y = torch.reshape(y, (1, timesteps*batch_size, seq_size*feature_size))
    y_sfc = torch.reshape(y_sfc, (1, timesteps*batch_size, -1))
    y = torch.cat((y, y_sfc),axis=-1) 

    # def _kernel_crps(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # """Kernel (ensemble) CRPS.
    mae = torch.mean(torch.abs(y - y_pred), dim=0)
    # mae = torch.cat( [(target - mem).abs().mean().unsqueeze(0) for mem in ens], 0).mean()
    # https://github.com/clessig/atmorep/blob/055f858e68f5e0151eb6fece9f6b574d3da4af8d/atmorep/utils/utils.py#L374

    # assert ens_size > 1, "Ensemble size must be greater than 1."
    fair = True
    # coef = -1.0 / (ens_size * (ens_size - 1)) if fair else -1.0 / (ens_size**2)
    coef = -1.0 / (2.0 * ens_size * (ens_size - 1)) if fair else -1.0 / (2.0 * ens_size**2)

    ens_var = torch.zeros(size=y.shape, device=y.device).squeeze(0)
    for i in range(ens_size):  # loop version to reduce memory usage
        ens_var += torch.sum(torch.abs(y_pred[i:i+1,:,:] - y_pred[i + 1 :,:,:]), dim=0)
    ens_var = coef * ens_var

    loss = mae + ens_var
    # (batch, n_vars)

    skill = mae.mean()
    spread = (-1*ens_var).mean()
    # In ECMWF ANEMOI code ( https://github.com/ecmwf/anemoi-core/blob/main/training/src/anemoi/training/losses/base.py ),
    # the loss is at this point SUMMED over the variable dimension
    # loss = torch.sum(loss,dim=1)  # --> (batch)
    loss = torch.mean(loss)
   
    return loss, skill, spread


def CRPS4(y, y_sfc, y_pred, y_sfc_pred, timesteps, beta=1, return_low_var_inds=False):
    
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
    # y: ntime*nbatch,      nseq, ny 
    # yp: ntime*nbatch*nens, nseq, ny 
    ns,seq_size,feature_size = y.shape
    batch_size = ns // timesteps
    ens_size = y_pred.shape[0] // (timesteps*batch_size)          
    # y_pred = torch.reshape(y_pred, (ens_size, batch_size, -1))
    y_pred = torch.reshape(y_pred, (timesteps, ens_size, batch_size, seq_size*feature_size))
    y_pred = torch.transpose(y_pred, 1, 2) # time, batch, ens, seq_size*feature_size))
    y_pred = torch.reshape(y_pred, (timesteps*batch_size, ens_size, seq_size*feature_size))

    y_sfc_pred = torch.reshape(y_sfc_pred, (timesteps, ens_size, batch_size, -1))
    y_sfc_pred = torch.transpose(y_sfc_pred, 1, 2) # time, batch, ens, nx_sfc))
    y_sfc_pred = torch.reshape(y_sfc_pred, (timesteps*batch_size, ens_size, -1))
    y_pred = torch.cat((y_pred, y_sfc_pred), axis=-1)

    y = torch.reshape(y, (timesteps*batch_size, 1, seq_size*feature_size))
    y_sfc = torch.reshape(y_sfc, (timesteps*batch_size, 1, -1))
    y = torch.cat((y, y_sfc),axis=-1)

    # cdist
    # x1 (Tensor) – input tensor of shape B×P×M
    # x2 (Tensor) – input tensor of shape B×R×M
    
    # ypred (batch,nens,30*4)   y  (batch,1,30*4)   
    # cdist out term1 (batch,1,nens)
    # cdist out term2 (batch, nens, nens)   
    # print("shape ytrue", y.shape, "y pred", y_pred.shape )

    # MSE     = torch.cdist(y, y_pred).mean()  # B, 1, nens  --> (1)
    cmean_out = torch.cdist(y_pred, y_pred) # B, nens, nens
    # ens_var = cmean_out.mean(0).sum() / (ens_size * (ens_size - 1)) 
     # Mean over batch, then summed over ens dim. Should this instead be first sum over ens, then mean over batch?


    # ens_var = cmean_out.sum(1).mean() / (ens_size * (ens_size - 1)) 
    ens_var = torch.mean(cmean_out) / (ens_size * (ens_size - 1)) 

    MSE     = torch.mean(torch.cdist(y, y_pred)) # B, 1, nens  --> (1)


    # ens_var = cmean_out.mean() / (ens_size * (ens_size - 1))

    MSE         /= y_pred.size(-1) ** 0.5
    ens_var     /= y_pred.size(-1) ** 0.5
    
    CRPS =  beta * 2 * MSE - ens_var # beta should be 1

    if return_low_var_inds:
        x = cmean_out[:,0,1]
        q = torch.quantile(x, 0.05,  keepdim=False)
        inds = x <= q

        # print("shape cmean out", cmean_out.shape, "shape inds", inds.shape)
        # print("cmean out 0, 01,   2048, 01", cmean_out[0,0,1], cmean_out[2048,0,1])
        # return beta * 2 * MSE - ens_var, MSE, ens_var, inds
        return CRPS,  MSE, ens_var, inds

    else:
        return CRPS, MSE, ens_var

def variogram_score(y_true, y_true_sfc, y_pred, y_pred_sfc, timesteps):
    # y:  ntime*nbatch,      nseq, ny 
    # yp: ntime*nbatch*nens, nseq, ny 
    # sr.vs_ensemble needs 
    # obs: "Array",  # (... D)
    # fct: "Array",  # (... nens D) 
    import scoringrules as sr
    ns,seq_size,feature_size = y.shape
    batch_size = ns // timesteps
    ens_size = y_pred.shape[0] // (timesteps*batch_size)          
    y_pred = torch.reshape(y_pred, (timesteps, ens_size, batch_size, seq_size*feature_size))
    y_pred = torch.transpose(y_pred, 1, 2) # time, batch, ens, seq_size*feature_size))
    y_pred = torch.reshape(y_pred, (timesteps*batch_size, ens_size, seq_size*feature_size))
    y_sfc_pred = torch.reshape(y_sfc_pred, (timesteps, ens_size, batch_size, -1))
    y_sfc_pred = torch.transpose(y_sfc_pred, 1, 2) # time, batch, ens, nx_sfc))
    y_sfc_pred = torch.reshape(y_sfc_pred, (timesteps*batch_size, ens_size, -1))
    y_pred = torch.cat((y_pred, y_sfc_pred), axis=-1)

    y     = torch.reshape(y,     (timesteps*batch_size, seq_size*feature_size))
    y_sfc = torch.reshape(y_sfc, (timesteps*batch_size, -1))
    y = torch.cat((y, y_sfc), axis=-1)

    # CRPS = sr.vs_ensemble(y, y_pred, p=0.5, backend='torch', estimator='fair')
    CRPS = sr.vs_ensemble(y, y_pred, p=0.5, backend='torch', estimator='nrg')

    # CRPS = CRPS.mean()
    CRPS = CRPS.sum(dim=-1).mean()

    # print("CRPS shape", CRPS.shape)
    y       = y.reshape((timesteps*batch_size,1,-1))
    # y_pred  = y_pred.reshape((timesteps*batch_size,ens_size,-1))
    eps = 1 / ens_size
    # cdist
    # x1 (Tensor) – input tensor of shape B×P×M
    # x2 (Tensor) – input tensor of shape B×R×M
    # ypred (batch,nens,30*4)   y  (batch,1,30*4)   
    # cdist out term1 (batch,1,nens)
    # cdist out term2 (batch, nens, nens)   
    MSE       = torch.cdist(y, y_pred).mean()  # B, 1, nens  --> (1)
    cmean_out = torch.cdist(y_pred, y_pred) # B, nens, nens
    ens_var = ( (1-eps)* cmean_out.mean(0).sum() ) / (ens_size * (ens_size - 1)) 
    MSE         /= y_pred.size(-1) ** 0.5
    ens_var     /= y_pred.size(-1) ** 0.5
    
    return CRPS, MSE, ens_var


def get_CRPS(beta): 
    def customCRPS(y_true, y_true_sfc, y_pred, y_pred_sfc, timesteps):
        # return CRPS_anemoi(y_true, y_true_sfc, y_pred, y_pred_sfc, timesteps, beta)
        # return CRPS(y_true, y_true_sfc, y_pred, y_pred_sfc, timesteps, beta)
        return CRPS_scoringrules(y_true, y_true_sfc, y_pred, y_pred_sfc, timesteps)

    return customCRPS

def get_GEL(_lambda):
  def GEL(y_true, y_true_sfc, y_pred, y_pred_sfc, _lambda):
    # implement GEL from 
    # https://www.sciencedirect.com/science/article/pii/S0169809525004119
    ntot = y_true.nelement()
    expterm = 1 / ntot * 1
    loss = torch.pow(2, expterm)
    return loss 
  return GEL 