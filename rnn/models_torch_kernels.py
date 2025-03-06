#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 20:22:43 2025

@author: peter
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.nn.parameter as Parameter
from torch.nn import Parameter
import numpy as np 

import numbers
import warnings
from collections import namedtuple
from typing import List, Tuple, Final
import torch.jit as jit
from torch import Tensor


class GLU(nn.Module):
    """ The static nonlinearity used in the S4 paper"""
    def __init__(self,  nlay, nneur, layernorm=True, dropout=0, expand_factor=2):
        super(GLU, self).__init__()
        self.activation = nn.GELU()
        self.nlay = nlay
        self.nneur = nneur
        self.layernorm = layernorm
        self.expand_factor=expand_factor

        if self.layernorm:
            self.normalization = nn.LayerNorm((self.nlay,self.nneur))

        if self.layernorm:
            self.normalization = nn.LayerNorm((self.nlay,self.nneur))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Linear(nneur, self.expand_factor * nneur),#nn.Conv1d(config.d_model, 2 * config.d_model, kernel_size=1),
            nn.GLU(dim=-1),
        )

    def forward(self, x):
        if self.layernorm:
            x = self.normalization(x)  # pre normalization
        x = self.dropout(self.activation(x))
        x = self.output_linear(x)
        return x
    
    
class QRNNLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, kernel_size: int, mode: str = "f", zoneout: float = 0.0):
        super(QRNNLayer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.mode = mode
        self.zoneout = zoneout

        self.zoneout_distribution = torch.distributions.Bernoulli(probs=self.zoneout)
        self.pad = nn.ConstantPad1d((self.kernel_size-1, 0), value=0.0)
        self.z_conv = nn.Conv1d(input_size, hidden_size, kernel_size)
        self.f_conv = nn.Conv1d(input_size, hidden_size, kernel_size)

        if self.mode == "fo" or self.mode == "ifo":
            self.o_conv = nn.Conv1d(input_size, hidden_size, kernel_size)

        if self.mode == "ifo":
            self.i_conv = nn.Conv1d(input_size, hidden_size, kernel_size)
    
    def forward(self, inputs: torch.Tensor, init_state:torch.Tensor) -> torch.Tensor:
        # inputs = shape: [batch x timesteps x features]
        batch, timesteps, _ = inputs.shape
        
        # Apply convolutions
        inputs = inputs.transpose(1, 2)
        inputs = self.pad(inputs)
        raw_f = self.f_conv(inputs).transpose(1, 2)
        raw_z = self.z_conv(inputs).transpose(1, 2)

        if self.mode == "ifo":
            raw_i = self.i_conv(inputs).transpose(1, 2)
            log_one_minus_f = F.logsigmoid(raw_i)
        else:
            log_one_minus_f = F.logsigmoid(-raw_f)
        
        # Get log values of activations
        log_z = F.logsigmoid(raw_z)  # Use sigmoid activation
        log_f = F.logsigmoid(raw_f)

        # Optionally apply zoneout
        if self.zoneout > 0.0:
            zoneout_mask = self.zoneout_distribution.sample(sample_shape=log_f.shape).bool()
            zoneout_mask = zoneout_mask.to(log_f.device)
            log_f = torch.masked_fill(input=log_f, mask=zoneout_mask, value=0.0)
            log_one_minus_f = torch.masked_fill(input=log_one_minus_f, mask=zoneout_mask, value=-1e8)
        
        # Precalculate recurrent gate values by reverse cumsum
        recurrent_gates = log_f[:, 1:, :]
        recurrent_gates_cumsum = torch.cumsum(recurrent_gates, dim=1)
        recurrent_gates = recurrent_gates - recurrent_gates_cumsum + recurrent_gates_cumsum[:, -1:, :]
        
        # Pad last timestep
        padding = torch.zeros([batch, 1, self.hidden_size], device=recurrent_gates.device)
        recurrent_gates = torch.cat([recurrent_gates, padding], dim=1)
        
        # Calculate expanded recursion by cumsum (logcumsumexp in log space)
        log_hidden = torch.logcumsumexp(log_z + log_one_minus_f + recurrent_gates, dim=1)
        hidden = torch.exp(log_hidden - recurrent_gates)

        # Optionally multiply by output gate
        if self.mode == "fo" or self.mode == "ifo":
            o = torch.sigmoid(self.o_conv(inputs)).transpose(1, 2)
            hidden = hidden * o
        
        return hidden
    

class QRNNLayer_noncausal(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, kernel_size: int, 
                 mode: str = "f",  pad="same"):
        super(QRNNLayer_noncausal, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.mode = mode
        
        self.fastconv = False

        # self.pad = nn.ConstantPad1d((self.kernel_size-1, 0), value=0.0)
        # pad = (0,1)
        if type(pad) == tuple:
            self.use_padding=True
            self.pad = nn.ConstantPad1d(pad, value=0.0)
            pad = "valid"
        else:
            self.use_padding=False
            pad = "same"
        if self.fastconv:
            self.z_conv = nn.Conv1d(input_size, hidden_size, kernel_size, padding=pad,dtype=torch.bfloat16)
            self.f_conv = nn.Conv1d(input_size, hidden_size, kernel_size, padding=pad,dtype=torch.bfloat16)
        else:
            self.z_conv = nn.Conv1d(input_size, hidden_size, kernel_size, padding=pad)
            self.f_conv = nn.Conv1d(input_size, hidden_size, kernel_size, padding=pad)
            
        if self.mode == "fo" or self.mode == "ifo":
            self.o_conv = nn.Conv1d(input_size, hidden_size, kernel_size, padding=pad)

        if self.mode == "ifo":
            self.i_conv = nn.Conv1d(input_size, hidden_size, kernel_size, padding=pad)
            
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
    # @jit.script_method
    @torch.compile
    def forward(self, inputs, init_state):
        # inputs = shape: [batch x timesteps x features]
        batch, timesteps, _ = inputs.shape
        
        # Apply convolutions
        # if init_state is None:
        #     inputs = inputs.transpose(1, 2)
        # else:
        #     init_state = torch.unsqueeze(init_state,1)
        #     inputs = torch.cat((init_state, inputs), axis=1)
        #     inputs = inputs.transpose(1, 2)
        inputs = inputs.transpose(1, 2)
        
        if self.use_padding:
            inputs = self.pad(inputs)
        # print("shape inp", inputs.shape)

        if self.fastconv:
            inputs = inputs.to(torch.bfloat16)
        
        raw_f = self.f_conv(inputs).transpose(1, 2)
        raw_z = self.z_conv(inputs).transpose(1, 2)
        
        if init_state is not None:
            init_state1 = torch.unsqueeze(init_state[0],1)
            init_state2 = torch.unsqueeze(init_state[1],1)
            raw_f = torch.cat((init_state1, raw_f), axis=1)
            raw_z = torch.cat((init_state2, raw_z), axis=1)
            
            
        # print("shape raw_f", raw_f.shape)
        
        if self.mode == "ifo":
            raw_i = self.i_conv(inputs).transpose(1, 2)
            log_one_minus_f = F.logsigmoid(raw_i)
        else:
            log_one_minus_f = F.logsigmoid(-raw_f)
        
        # Get log values of activations
        if self.fastconv:
            raw_z = raw_z.to(torch.float32)
            raw_f = raw_f.to(torch.float32)
        log_z = F.logsigmoid(raw_z)  # Use sigmoid activation
        log_f = F.logsigmoid(raw_f)
    
        # Precalculate recurrent gate values by reverse cumsum
        # recurrent_gates = log_f[:, 1:, :]
        recurrent_gates = log_f[:, :, :]

        # dtype = recurrent_gates.dtype 
        # if dtype==torch.bfloat16:
        #     recurrent_gates_cumsum = torch.cumsum(recurrent_gates, dim=1, dtype=torch.float32)
        #     recurrent_gates_cumsum = recurrent_gates_cumsum.to(dtype)
        #     #     outs = outs.to(torch.float32)
        # else:
        recurrent_gates_cumsum = torch.cumsum(recurrent_gates, dim=1)

        recurrent_gates = recurrent_gates - recurrent_gates_cumsum + recurrent_gates_cumsum[:, -1:, :]
        
        # Pad last timestep
        
        log_hidden = torch.logcumsumexp(log_z + log_one_minus_f + recurrent_gates, dim=1)
        hidden = torch.exp(log_hidden - recurrent_gates)

        # Optionally multiply by output gate
        if self.mode == "fo" or self.mode == "ifo":
            o = torch.sigmoid(self.o_conv(inputs)).transpose(1, 2)
            hidden = hidden * o
        
        return hidden
    