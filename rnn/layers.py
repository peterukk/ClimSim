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