#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytorch layers
"""
import os 
import gc
import sys
import torch
import torch.nn as nn
import torch.nn.parameter as Parameter
# from layers_callbacks_torch import LayerPressure 
import torch.nn.functional as F
from typing import List, Tuple, Final, Optional
from torch import Tensor
from models_torch_kernels import GLU
from models_torch_kernels import *
import numpy as np 
from typing import Final 
import time 

class LayerPressure(nn.Module):
    def __init__(self,hyam, hybm, name='LayerPressure',
                 norm=True,
                  # sp_min=62532.977,sp_max=104871.82,
                 ):
        super(LayerPressure, self).__init__()

        # self.sp_min = sp_min
        # self.sp_max = sp_max
        # self.pres_min = 36.434
        # self.pres_max = self.sp_max
        self.nlev = hyam.shape[0]
        
        # hyam = torch.from_numpy(hyam)
        # hybm = torch.from_numpy(hybm)
        # self.hyam = torch.reshape(hyam,(1,self.nlev,1))
        # self.hybm = torch.reshape(hybm,(1,self.nlev,1))
        hyam = torch.reshape(hyam,(1,self.nlev,1))
        hybm = torch.reshape(hybm,(1,self.nlev,1))
        self.register_buffer('hyam', hyam)
        self.register_buffer('hybm', hybm)

        self.norm = norm
        
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.hyam = torch.reshape(hyam,(1,self.nlev,1)).to(device, torch.float32)
        # self.hybm = torch.reshape(hybm,(1,self.nlev,1)).to(device, torch.float32)
        
    def forward(self, sp):
        # unnormalize
        # sp = (sp + 1.0)*(self.sp_max - self.sp_min)/2 + self.sp_min

        pres = self.hyam*100000.0 + sp*self.hybm
        if self.norm:
            pres = torch.sqrt(pres) / 314.0
        # print(pres[0,:])
        # pres = (pres-self.pres_min)/(self.pres_max-self.pres_min)*2 - 1.0

        return pres
    
class PressureThickness(nn.Module):
    def __init__(self,hyai, hybi, name='PressureThickness',
                  # sp_min=62532.977,sp_max=104871.82,
                  ):
        super(PressureThickness, self).__init__()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.sp_min = sp_min
        # self.sp_max = sp_max
        # self.pres_min = 36.434
        # self.pres_max = self.sp_max
        self.nlev = hyai.shape[0] - 1
        hyai = torch.reshape(hyai,(1,self.nlev+1,1))
        hybi = torch.reshape(hybi,(1,self.nlev+1,1))
        self.register_buffer('hyai', hyai)
        self.register_buffer('hybi', hybi)

    def forward(self, sp):
        # sp  (batch, 1, 1)   
        # hyai,hybi  (1, nlev+1, 1)
        
        dp = sp*(self.hybi[:,1:self.nlev+1]-self.hybi[:,0:self.nlev]) + 100000.0*(self.hyai[:,1:self.nlev+1]-self.hyai[:,0:self.nlev])

        return dp
    
class LevelPressure(nn.Module):
    def __init__(self,hyai, hybi, name='LevelPressure',
                  # sp_min=62532.977,sp_max=104871.82,
                  ):
        super(LevelPressure, self).__init__()

        self.nlev = hyai.shape[0] 
        hyai = torch.reshape(hyai,(1,self.nlev,1))
        hybi = torch.reshape(hybi,(1,self.nlev,1))
        self.register_buffer('hyai', hyai)
        self.register_buffer('hybi', hybi)

    def forward(self, sp):
        # sp  (batch, 1, 1)   
        # hyai,hybi  (1, nlev+1, 1)
        plev = sp*(self.hybi) + 100000.0*self.hyai

        return plev

def interpolate_tlev_batchfirst(tlay, play, plev):
    ncol, nlay = tlay.shape
    device = tlay.device
    dtype = tlay.dtype
    # Initialize output arrays
    tlev = torch.zeros(ncol, nlay + 1, dtype=dtype, device=device)
    
    tlev[:,0] = tlay[:,0] + (plev[:,0]-play[:,0])*(tlay[:,1]-tlay[:,0]) / (play[:,1]-play[:,0])
    for ilay in range(1, nlay-1):
        tlev[:,ilay] = (play[:,ilay-1]*tlay[:,ilay-1]*(plev[:,ilay]-play[:,ilay]) \
            + play[:,ilay]*tlay[:,ilay]*(play[:,ilay-1]-plev[:,ilay])) /  (plev[:,ilay]*(play[:,ilay-1] - play[:,ilay]))
                                
    tlev[:,nlay] = tlay[:,nlay-1] + (plev[:,nlay]-play[:,nlay-1])*(tlay[:,nlay-1]-tlay[:,nlay-2])  \
            / (play[:,nlay-1]-play[:,nlay-2])
                                
    return tlev

class gasopt_mlp(nn.Module):
    # additional_mlp: Final[bool]
    is_longwave: Final[bool]
    lock_weights: Final[bool]
    change_last_layer: Final[bool]
    def __init__(self, device,
                xmin, xmax, ymean, ystd,
                nn_w1, nn_w2, nn_w3,
                nn_b1, nn_b2, nn_b3, 
                # lock_weights=True,
                num_outputs_desired,
                is_longwave=True):
        super(gasopt_mlp, self).__init__()
        self.nx = xmin.shape[0]
        self.ny = ymean.shape[0]
        self.is_longwave=is_longwave
        if self.is_longwave:
            self.ng = self.ny//2
        else:
            self.ng = self.ny
        self.change_last_layer = False
        if self.ng != num_outputs_desired:
            print("Number of g-points in existing gas optics model doesn't match desired, changing last layer so that ng={}".format(num_outputs_desired))
            self.ng = num_outputs_desired
            self.change_last_layer=True
            if self.is_longwave:
                self.ny = 2*self.ng
                self.softmax = nn.Softmax(dim=2)
            else:
                self.ny = self.ng
        self.nh = nn_w1.shape[1]
        xmin  = torch.from_numpy(xmin)
        xmax  = torch.from_numpy(xmax)
        ymean = torch.from_numpy(ymean[0:self.ng])
        ystd  = torch.from_numpy(ystd[0:self.ng])
        if self.change_last_layer:
            ymean[:] = 0.0
            ystd[:] = 1.0# 0.0005#1.0
        self.register_buffer('xmin', xmin)
        self.register_buffer('xmax', xmax)
        self.register_buffer('ymean', ymean)
        self.register_buffer('ystd', ystd)
        self.softsign =  nn.Softsign()
        self.mlp1 = nn.Linear(self.nx, self.nh)
        self.mlp2 = nn.Linear(self.nh, self.nh)
        self.mlp3 = nn.Linear(self.nh, self.ny)
        self.lock_weights=True #lock_weights
        # if self.ng != self.ng0:
        #     self.additional_mlp = True
        #     self.mlp4 = nn.Linear(self.ny, self.ng*2)
        #     print("Number of spectral points in model doesn't match desired, adding additional mLP")
        # else:
        #     self.additional_mlp = False 
        print("gasopt_mlp_lw number of g-points: {}, hidden neurons: {}, inputs: {}".format(self.ng, self.nh, self.nx)) 

        self.mlp1.weight = torch.nn.Parameter(torch.from_numpy(nn_w1.T))
        self.mlp2.weight = torch.nn.Parameter(torch.from_numpy(nn_w2.T))
        self.mlp1.bias = torch.nn.Parameter(torch.from_numpy(nn_b1.T))
        self.mlp2.bias = torch.nn.Parameter(torch.from_numpy(nn_b2.T))
        if not self.change_last_layer:
            self.mlp3.weight = torch.nn.Parameter(torch.from_numpy(nn_w3.T))
            self.mlp3.bias = torch.nn.Parameter(torch.from_numpy(nn_b3.T))
        if self.lock_weights:
          self.mlp1.weight.requires_grad = False; self.mlp1.bias.requires_grad = False
          self.mlp2.weight.requires_grad = False; self.mlp2.bias.requires_grad = False
          if not self.change_last_layer:
            self.mlp3.weight.requires_grad = False; self.mlp3.bias.requires_grad = False

        self.to(device)

    def forward(self, x, col_dry):
        x = self.mlp1(x)
        x = self.softsign(x)
        x = self.mlp2(x)
        x = self.softsign(x)
        x = self.mlp3(x)
        # if self.additional_mlp:
        #     x = self.mlp4(x)

        if self.is_longwave:
            tau, pfrac = x.chunk(2,-1)
            pfrac = torch.square(pfrac)
            if self.change_last_layer:
                pfrac = self.softmax(pfrac)
        else:
            tau = x 

        # if col_dry is not None:
        # print("shape coldry", col_dry.shape, "tau", tau.shape, "ystd", self.ystd.shape)
        tau = col_dry * torch.pow(self.ystd*tau + self.ymean,8)
        if self.change_last_layer:
            tau = 1e-19*tau 
    #    ! Postprocess absorption output: reverse standard scaling and square root scaling
    #    tau(igpt,ilay,icol) = (ystd(igpt) * outp_both(igpt,ilay,icol) + ymeans(igpt))**8
    #    ! Optical depth from cross-sections
    #    tau(igpt,ilay,icol) = tau(igpt,ilay,icol)*col_dry_wk(ilay,icol)
        if self.is_longwave:
            return tau, pfrac
        else:
            return tau
