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
from typing import List, Tuple, Final, Optional
import torch.jit as jit
from torch import Tensor
import gc


class SRU(nn.Module):
    """ Simple Recurrent Unit https://arxiv.org/pdf/1709.02755.pdf """

    def __init__(self, input_size, hidden_size, activation=nn.Sigmoid()):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.linear_transform = nn.Linear(input_size, hidden_size, bias=False)
        self.gate = nn.Linear(input_size, 2 * hidden_size)
        self.activation = activation

        self.gate_ln = nn.LayerNorm(2 * hidden_size)
        self.act_ln = nn.LayerNorm(hidden_size)
        
    @torch.compile
    def forward(self, x, c):
        if c is None:
            c = torch.zeros((x.shape[0], self.hidden_size), dtype=x.dtype, device=x.device)

        x_tilde = self.linear_transform(x)
        # gate = F.sigmoid(self.gate_ln(self.gate(x)))
        gate = nn.Sigmoid()(self.gate_ln(self.gate(x)))

        f = gate[:, :, :self.hidden_size]
        r = gate[:, :, self.hidden_size:]
        new_data = (1 - f) * x_tilde

        cell_states = []
        for t in range(x.size(1)):
            # Every timestep
            c = f[:, t] * c + new_data[:, t]
            cell_states.append(c)

        all_c = torch.stack(cell_states, dim=1)
        h = r * self.activation(self.act_ln(all_c)) + (1 - r) * x
        # h = r * self.activation()(self.act_ln(all_c)) + (1 - r) * x

        return h, c
    
class MyStochasticGRU(nn.Module):

    """
    stochastic GRU

    """

    def __init__(self, input_size, hidden_size, go_backwards=False, bias=True):
        super(MyStochasticGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.go_backwards = go_backwards
        self.bias = bias
        print("input size", input_size, "hidden size", hidden_size)
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.z2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        self.encoder_mlp_mean =  nn.Linear(input_size+hidden_size, hidden_size)
        self.encoder_mlp_sigma =  nn.Linear(input_size+hidden_size, hidden_size)
        # self.encoder_mlp_dist =  nn.Linear(input_size+hidden_size, 2*hidden_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
    @torch.compile
    def forward(self, x_seq): #, hidden=None):
        
        nseq = x_seq.shape[0]
        batch_size = x_seq.shape[1]
        
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.go_backwards:
            istart = nseq-1 
            istop = -1
            istep = -1 
        else:
            istart = 0
            istop = nseq
            istep = 1
            
        hidden_seq = []
        
        # if hidden==None:
        hidden = torch.randn((batch_size, self.hidden_size), device=x_seq.device)  # (batch, hidden_size)
        
        for i in range(istart,istop,istep):    
            # print(i)
            x = x_seq[i]
            # print("x shape", x.shape, "hidden shape", hidden.shape)
            inp = torch.cat((x,hidden), dim=1)
            mean_ = self.encoder_mlp_mean(inp)
            logvar_ = self.encoder_mlp_sigma(inp)
            
            # gates = (
            #     torch.mm(x, weight_ih)
            #     + self.bias_ih
            #     + torch.mm(hx, weight_hh)
            #     + self.bias_hh
            # )
            # ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            eps = torch.randn_like(mean_)
            sigma = torch.exp(0.5*logvar_)
            z = mean_ + eps * sigma

            gate_x = self.x2h(x) 
            gate_h = self.h2h(hidden)
            gate_z = self.z2h(z)
    
            gate_x = gate_x.squeeze()
            gate_h = gate_h.squeeze()
            gate_z = gate_z.squeeze()
    
            i_r, i_i, i_n = gate_x.chunk(3, 1)
            h_r, h_i, h_n = gate_h.chunk(3, 1)
            z_r, z_i, z_n = gate_z.chunk(3, 1)
    
            resetgate = torch.sigmoid((i_r + h_r + z_r))
            inputgate = torch.sigmoid((i_i + h_i + z_i))
            newgate = torch.tanh((i_n + z_n + (resetgate * h_n)))
            
            hidden = newgate + inputgate * (hidden - newgate)
            
            hidden_seq.append(hidden)
        
        return torch.stack(hidden_seq, dim=0) 
    

    
class MyStochasticGRULayer(jit.ScriptModule):
    use_bias: Final[bool]

    def __init__(self, input_size, hidden_size, dtype=torch.float32, use_bias=False):
        super().__init__()
        # self.cell = cell(*cell_args)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn((input_size, 3 * hidden_size),dtype=dtype))
        self.weight_hh = Parameter(torch.randn((hidden_size, 3 * hidden_size),dtype=dtype))
        self.weight_zh = Parameter(torch.randn((hidden_size, 3 * hidden_size),dtype=dtype))

        self.use_bias = use_bias
        if self.use_bias:
            self.bias_ih = Parameter(torch.randn((3 * hidden_size),dtype=dtype))
            self.bias_hh = Parameter(torch.randn((3 * hidden_size),dtype=dtype))
            self.bias_zh = Parameter(torch.randn((3 * hidden_size),dtype=dtype))

        self.weight_encoder =  Parameter(torch.randn((input_size + hidden_size, 2*hidden_size),dtype=dtype))
        # self.weight_encoder_sigma =  Parameter(torch.randn((input_size + hidden_size, hidden_size),dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
    # @torch.jit.script          
    @jit.script_method
    # @torch.compile
    def forward(
        self, input_seq: Tensor, hidden: Tensor) -> Tensor: #Tuple[Tensor, Tensor]:
        
        nseq, batch_size, nx = input_seq.shape

        # epss = torch.randn_like(input_seq)
        epss = torch.randn((nseq, batch_size, self.hidden_size),device=input_seq.device)
        epss = epss.unbind(0)
        
        inputs = input_seq.unbind(0)


        outputs = torch.jit.annotate(List[Tensor], [])
        
        
        for i in range(len(input_seq)):
            x = inputs[i]
            eps = epss[i]
            
            # print("shape x", x.shape, "shape hid", hidden.shape)
            inp = torch.cat((x,hidden), dim=1)
            predicted_distribution = torch.mm(inp, self.weight_encoder) 
            mean_, logvar_ = predicted_distribution.chunk(2,1)
            
            # eps = torch.randn_like(mean_)
            z = mean_ + eps * torch.exp(0.5*logvar_)
            
            if self.use_bias:
                x_results = torch.mm(x, self.weight_ih) + self.bias_ih
                h_results = torch.mm(hidden, self.weight_hh)  + self.bias_hh
                z_results = torch.mm(z, self.weight_zh)  + self.bias_zh
            else:
                x_results = torch.mm(x, self.weight_ih) 
                h_results = torch.mm(hidden, self.weight_hh) 
                z_results = torch.mm(z, self.weight_zh) 

            i_r, i_z, i_n = x_results.chunk(3, 1)
            h_r, h_z, h_n = h_results.chunk(3, 1)
            z_r, z_z, z_n = z_results.chunk(3, 1)

            r = torch.sigmoid(i_r + h_r + z_r)
            z = torch.sigmoid(i_z + h_z + z_z)
            n = torch.tanh(i_n + z_n + r * h_n)
                
            # hidden =  n - torch.mul(n, z) + torch.mul(z, hidden)
            hidden = n + torch.mul(z, (hidden - n))
            # hidden = newgate + inputgate * (hidden - newgate)

            outputs += [hidden]

        return torch.stack(outputs)
    
    
class MyStochasticGRULayer2(jit.ScriptModule):
    # the one in the paper "Stochastic Recurrent Neural Network for
    # Multistep Time Series Forecasting" by Yin et al;
    # difference to MyStochasticGRULayer is that the distribution of the 
    # latent variable z is predicted only using the previous hidden state,
    # not also the input
    def __init__(self, input_size, hidden_size, dtype=torch.float32, use_bias=False):
        super().__init__()
        # self.cell = cell(*cell_args)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn((input_size, 3 * hidden_size),dtype=dtype))
        self.weight_hh = Parameter(torch.randn((hidden_size, 3 * hidden_size),dtype=dtype))
        self.use_bias = use_bias
        # if self.use_bias:
        #     self.bias_ih = Parameter(torch.randn((3 * hidden_size),dtype=dtype))
        #     self.bias_hh = Parameter(torch.randn((3 * hidden_size),dtype=dtype))
        #     self.bias_zh = Parameter(torch.randn((3 * hidden_size),dtype=dtype))

        self.weight_encoder =  Parameter(torch.randn((hidden_size, 2*hidden_size),dtype=dtype))
        # self.weight_encoder_sigma =  Parameter(torch.randn((input_size + hidden_size, hidden_size),dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
    # @torch.jit.script          
    @jit.script_method
    # @torch.compile
    def forward(
        self, input_seq: Tensor, hidden: Tensor) -> Tensor: #Tuple[Tensor, Tensor]:
        
        nseq, batch_size, nx = input_seq.shape

        # epss = torch.randn_like(input_seq)
        epss = torch.randn((nseq, batch_size, self.hidden_size),device=input_seq.device)
        epss = epss.unbind(0)
        
        inputs = input_seq.unbind(0)


        outputs = torch.jit.annotate(List[Tensor], [])
        
        
        for i in range(len(input_seq)):
            x = inputs[i]
            eps = epss[i]
            
            # print("shape x", x.shape, "shape hid", hidden.shape)
            predicted_distribution = torch.mm(hidden, self.weight_encoder) 
            mean_, logvar_ = predicted_distribution.chunk(2,1)
            
            # eps = torch.randn_like(mean_)
            z = mean_ + eps * torch.exp(0.5*logvar_)
            
            # if self.use_bias:
            #     x_results = torch.mm(x, self.weight_ih) + self.bias_ih
            #     h_results = torch.mm(hidden, self.weight_hh)  + self.bias_hh
            #     z_results = torch.mm(z, self.weight_hh)  + self.bias_zh
            # else:
            x_results = torch.mm(x, self.weight_ih) 
            h_results = torch.mm(hidden, self.weight_hh) 
            z_results = torch.mm(z, self.weight_hh) 

            i_r, i_z, i_n = x_results.chunk(3, 1)
            h_r, h_z, h_n = h_results.chunk(3, 1)
            z_r, z_z, z_n = z_results.chunk(3, 1)

            r = torch.sigmoid(i_r + h_r + z_r)
            z = torch.sigmoid(i_z + h_z + z_z)
            n = torch.tanh(i_n + z_n + r * h_n)
                
            # hidden =  n - torch.mul(n, z) + torch.mul(z, hidden)
            hidden = n + torch.mul(z, (hidden - n))
            # hidden = newgate + inputgate * (hidden - newgate)

            outputs += [hidden]

        return torch.stack(outputs)
    


class MyStochasticGRULayer3(jit.ScriptModule):
    use_bias: Final[bool]

    def __init__(self, input_size, hidden_size, dtype=torch.float32, use_bias=False):
        super().__init__()
        # self.cell = cell(*cell_args)
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.weight_ih = Parameter(torch.randn((input_size, 3 * hidden_size),dtype=dtype))
        self.weight_hh = Parameter(torch.randn((hidden_size, 3 * hidden_size),dtype=dtype))
        self.weight_zh = Parameter(torch.randn((hidden_size, 3 * hidden_size),dtype=dtype))

        self.use_bias = use_bias
        if self.use_bias:
            # self.bias_ih = Parameter(torch.randn((3 * hidden_size),dtype=dtype))
            self.bias_hh = Parameter(torch.randn((3 * hidden_size),dtype=dtype))
            self.bias_zh = Parameter(torch.randn((3 * hidden_size),dtype=dtype))

        self.weight_encoder =  Parameter(torch.randn((input_size + hidden_size, 2*hidden_size),dtype=dtype))
        # self.weight_encoder_sigma =  Parameter(torch.randn((input_size + hidden_size, hidden_size),dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
    # @torch.jit.script          
    @jit.script_method
    # @torch.compile
    def forward(
        self, input_seq: Tensor, hidden: Tensor) -> Tensor: #Tuple[Tensor, Tensor]:
        
        nseq, batch_size, nx = input_seq.shape

        # epss = torch.randn_like(input_seq)
        epss = torch.randn((nseq, batch_size, self.hidden_size),device=input_seq.device)
        epss = epss.unbind(0)
        
        inputs = input_seq.unbind(0)


        outputs = torch.jit.annotate(List[Tensor], [])
        
        
        for i in range(len(input_seq)):
            x = inputs[i]
            eps = epss[i]
            
            # print("shape x", x.shape, "shape hid", hidden.shape)
            inp = torch.cat((x,hidden), dim=1)
            predicted_distribution = torch.mm(inp, self.weight_encoder) 
            mean_, logvar_ = predicted_distribution.chunk(2,1)
            
            # eps = torch.randn_like(mean_)
            z = mean_ + eps * torch.exp(0.5*logvar_)
            
            if self.use_bias:
                # x_results = torch.mm(x, self.weight_ih) + self.bias_ih
                h_results = torch.mm(hidden, self.weight_hh)  + self.bias_hh
                z_results = torch.mm(z, self.weight_zh)  + self.bias_zh
            else:
                # x_results = torch.mm(x, self.weight_ih) 
                h_results = torch.mm(hidden, self.weight_hh) 
                z_results = torch.mm(z, self.weight_zh) 

            # i_r, i_z, i_n = x_results.chunk(3, 1)
            h_r, h_z, h_n = h_results.chunk(3, 1)
            z_r, z_z, z_n = z_results.chunk(3, 1)

            r = torch.sigmoid(h_r + z_r)
            z = torch.sigmoid(h_z + z_z)
            n = torch.tanh(z_n + r * h_n)
                
            # hidden =  n - torch.mul(n, z) + torch.mul(z, hidden)
            hidden = n + torch.mul(z, (hidden - n))
            # hidden = newgate + inputgate * (hidden - newgate)

            outputs += [hidden]

        return torch.stack(outputs)


class MyStochasticGRULayer4(jit.ScriptModule):
    use_bias: Final[bool]

    def __init__(self, input_size, hidden_size, dtype=torch.float32, use_bias=False):
        super().__init__()
        # self.cell = cell(*cell_args)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn((input_size, 3 * hidden_size),dtype=dtype))
        # self.weight_hh = Parameter(torch.randn((hidden_size, 3 * hidden_size),dtype=dtype))
        self.weight_zh = Parameter(torch.randn((hidden_size, 3 * hidden_size),dtype=dtype))

        self.use_bias = use_bias
        if self.use_bias:
            self.bias_ih = Parameter(torch.randn((3 * hidden_size),dtype=dtype))
            # self.bias_hh = Parameter(torch.randn((3 * hidden_size),dtype=dtype))
            self.bias_zh = Parameter(torch.randn((3 * hidden_size),dtype=dtype))

        self.weight_encoder =  Parameter(torch.randn((input_size + hidden_size, 2*hidden_size),dtype=dtype))
        # self.weight_encoder_sigma =  Parameter(torch.randn((input_size + hidden_size, hidden_size),dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
    # @torch.jit.script          
    @jit.script_method
    # @torch.compile
    def forward(
        self, input_seq: Tensor, hidden: Tensor) -> Tensor: #Tuple[Tensor, Tensor]:
        
        nseq, batch_size, nx = input_seq.shape

        # epss = torch.randn_like(input_seq)
        epss = torch.randn((nseq, batch_size, self.hidden_size),device=input_seq.device)
        epss = epss.unbind(0)
        
        inputs = input_seq.unbind(0)


        outputs = torch.jit.annotate(List[Tensor], [])
        
        
        for i in range(len(input_seq)):
            x = inputs[i]
            eps = epss[i]
            
            # print("shape x", x.shape, "shape hid", hidden.shape)
            inp = torch.cat((x,hidden), dim=1)
            predicted_distribution = torch.mm(inp, self.weight_encoder) 
            mean_, logvar_ = predicted_distribution.chunk(2,1)
            
            # eps = torch.randn_like(mean_)
            z = mean_ + eps * torch.exp(0.5*logvar_)
            
            if self.use_bias:
                x_results = torch.mm(x, self.weight_ih) + self.bias_ih
                # h_results = torch.mm(hidden, self.weight_hh)  + self.bias_hh
                z_results = torch.mm(z, self.weight_zh)  + self.bias_zh
            else:
                x_results = torch.mm(x, self.weight_ih) 
                # h_results = torch.mm(hidden, self.weight_hh) 
                z_results = torch.mm(z, self.weight_zh) 

            i_r, i_z, i_n = x_results.chunk(3, 1)
            # h_r, h_z, h_n = h_results.chunk(3, 1)
            z_r, z_z, z_n = z_results.chunk(3, 1)

            r = torch.sigmoid(i_r + z_r)
            z = torch.sigmoid(i_z + z_z)
            n = torch.tanh(i_n + r * z_n)
                
            # hidden =  n - torch.mul(n, z) + torch.mul(z, hidden)
            hidden = n + torch.mul(z, (hidden - n))
            # hidden = newgate + inputgate * (hidden - newgate)

            outputs += [hidden]

        return torch.stack(outputs)
    
class MyStochasticGRULayer5(jit.ScriptModule):
    use_bias: Final[bool]

    def __init__(self, input_size, hidden_size, use_bias=False):
        super().__init__()
        # self.cell = cell(*cell_args)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn((input_size, 3 * hidden_size)))
        self.weight_zh = Parameter(torch.randn((hidden_size, 3 * hidden_size)))

        self.use_bias = use_bias
        if self.use_bias:
            self.bias_ih = Parameter(torch.randn((3 * hidden_size)))
            self.bias_zh = Parameter(torch.randn((3 * hidden_size)))

        self.weight_encoder =  Parameter(torch.randn((hidden_size, 2*hidden_size)))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
    @torch.compile
    def forward(
        self, input_seq: Tensor, hidden: Tensor) -> Tensor: #Tuple[Tensor, Tensor]:
        
        nseq, batch_size, nx = input_seq.shape

        epss = torch.randn_like(input_seq)
        epss = torch.randn((nseq, batch_size, self.hidden_size),device=input_seq.device)
        epss = epss.unbind(0)
        
        inputs = input_seq.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        
        for i in range(len(input_seq)):
            x = inputs[i]
            eps = epss[i]
            
            predicted_distribution = torch.mm(hidden, self.weight_encoder) 
            mean_, z = predicted_distribution.chunk(2,1)
            
            z = mean_ + eps * torch.exp(0.5*z)
            
            if self.use_bias:
                x_results = torch.mm(x, self.weight_ih) + self.bias_ih
                z_results = torch.mm(z, self.weight_zh)  + self.bias_zh
            else:
                x_results = torch.mm(x, self.weight_ih) 
                z_results = torch.mm(z, self.weight_zh) 

            # i_r, i_z, i_n = x_results.chunk(3, 1)
            # z_r, z_z, z_n = z_results.chunk(3, 1)
            r, z, n = x_results.chunk(3, 1)
            z_r, z_z, z_n = z_results.chunk(3, 1)
            
            r = torch.sigmoid(r + z_r)
            z = torch.sigmoid(z + z_z)
            n = torch.tanh(n + r * z_n)
                
            hidden = n + torch.mul(z, (hidden - n))

            outputs += [hidden]

        return torch.stack(outputs)
    
class MyStochasticGRULayer5_MLP_fused(jit.ScriptModule):
    use_bias: Final[bool]

    def __init__(self, input_size, hidden_size, final_size, use_bias=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn((input_size, 3 * hidden_size)))
        self.weight_zh = Parameter(torch.randn((hidden_size, 3 * hidden_size)))

        self.use_bias = use_bias
        if self.use_bias:
            self.bias_ih = Parameter(torch.randn((3 * hidden_size)))
            self.bias_zh = Parameter(torch.randn((3 * hidden_size)))
        
        self.weight_encoder =  Parameter(torch.randn((hidden_size, 2*hidden_size)))

        self.weight_fh = Parameter(torch.randn((hidden_size, final_size)))
        self.bias_fh = Parameter(torch.randn((final_size)))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
    @torch.compile
    def forward(
        self, input_seq: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        
        nseq, batch_size, nx = input_seq.shape

        epss = torch.randn_like(input_seq)
        epss = torch.randn((nseq, batch_size, self.hidden_size),device=input_seq.device)
        epss = epss.unbind(0)
        
        inputs = input_seq.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        
        for i in range(len(input_seq)):
            x = inputs[i]
            eps = epss[i]
            
            predicted_distribution = torch.mm(hidden, self.weight_encoder) 
            mean_, z = predicted_distribution.chunk(2,1)
            
            z = mean_ + eps * torch.exp(0.5*z)
            
            if self.use_bias:
                x_results = torch.mm(x, self.weight_ih) + self.bias_ih
                z_results = torch.mm(z, self.weight_zh)  + self.bias_zh
            else:
                x_results = torch.mm(x, self.weight_ih) 
                z_results = torch.mm(z, self.weight_zh) 
                # torch.mm(x, self.weight_ih, out=x_results) 
                # torch.mm(z, self.weight_zh, out=z_results) 

            r, z, n = x_results.chunk(3, 1)
            z_r, z_z, z_n = z_results.chunk(3, 1)
            
            # r = torch.sigmoid(r + z_r)
            z = torch.sigmoid(z + z_z)
            n = torch.tanh(n + torch.sigmoid(r + z_r) * z_n)
                
            hidden = n + torch.mul(z, (hidden - n))

            output = torch.addmm(self.bias_fh, hidden, self.weight_fh)

            outputs += [output]
            gc.collect()

        return torch.stack(outputs), hidden

class LayerNorm(nn.Module):
    def __init__(self, nb_eps, nb_features, eps = 1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gain = Parameter(torch.ones(nb_eps, nb_features))
        self.bias = Parameter(torch.zeros(nb_eps, nb_features))

    @torch.compile
    def forward(self, x, noise):
        mean = x.mean(1,keepdim=True).expand_as(x)
        std = x.std(1,keepdim=True).expand_as(x)
        # x = (x - mean) / (std + self.eps)
        # x = x * self.gain.expand_as(x) + self.bias.expand_as(x)
        #  eps is (nb, neps) where neps is e.g. 16
        #  x is (nb, nh)
        #  multiply eps with an MLP (neps, nh) to get to (nb, nh) for the gain and bias parameters 
        # e.g. (nb,nh)*(16,128)
        x = (x - mean) / (std + self.eps) * torch.mm(noise, self.gain) + torch.mm(noise, self.bias)
        return x

class StochasticLayerNormLSTMLayer(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, eps_size):
        super().__init__()
        # self.cell = cell(*cell_args)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.eps_size = eps_size
        self.weight_ih = Parameter(torch.randn(( input_size, 4 * hidden_size)))
        self.weight_hh = Parameter(torch.randn(( hidden_size, 4 * hidden_size)))
        
        self.bias_ih = Parameter(torch.randn((4 * hidden_size)))
        self.bias_hh = Parameter(torch.randn((4 * hidden_size)))
        
        self.ln_ih = LayerNorm(eps_size, 4 * hidden_size)
        self.ln_hh = LayerNorm(eps_size, 4 * hidden_size)
        self.ln_ho = LayerNorm(eps_size, hidden_size)

        # self.weight_encoder =  Parameter(torch.randn((input_size + hidden_size, 2*hidden_size),dtype=dtype))
        # self.weight_zh = Parameter(torch.randn((hidden_size, hidden_size),dtype=dtype))
        # self.bias_zh = Parameter(torch.randn((3 * hidden_size),dtype=dtype))
        # self.bias_zh = Parameter(torch.randn((hidden_size),dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
    @torch.compile
    def forward(
        self, input_seq: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        nseq, batch_size, nx = input_seq.shape
        # epss = torch.randn((nseq, batch_size, self.hidden_size),device=input_seq.device)
        # epss = epss.unbind(0)
        inputs = input_seq.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        hx, cx = state
        
        # weight_ih = self.weight_ih.t()
        # weight_hh = self.weight_hh.t()
        epss = torch.randn((nseq, batch_size, self.eps_size),device=input_seq.device, dtype=input_seq.dtype)

        for i in range(len(inputs)):
            x = inputs[i]
            eps = epss[i]
            # eps = torch.randn((batch_size, self.eps_size),device=input_seq.device, dtype=input_seq.dtype)

            # gates = self.ln_ih(F.linear(x, self.weight_ih, self.bias_ih)) + self.ln_hh(F.linear(hx, self.weight_hh, self.bias_hh))
            gates = self.ln_ih((torch.mm(x, self.weight_ih) + self.bias_ih), eps) + self.ln_hh((torch.mm(hx, self.weight_hh) + self.bias_hh), eps)
                       
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cx = (forgetgate * cx) + (ingate * cellgate)
            hx = outgate * torch.tanh(self.ln_ho(cx, eps))
            
            outputs += [hx]

        state =  (hx, cx)

        return torch.stack(outputs), state

class MyStochasticLSTMLayer(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, dtype=torch.float32):
        super().__init__()
        # self.cell = cell(*cell_args)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(( input_size, 3 * hidden_size),dtype=dtype))
        self.weight_hh = Parameter(torch.randn(( hidden_size, 3 * hidden_size),dtype=dtype))
        
        self.bias_ih = Parameter(torch.randn((3 * hidden_size),dtype=dtype))
        self.bias_hh = Parameter(torch.randn((3 * hidden_size),dtype=dtype))
        
        self.weight_encoder =  Parameter(torch.randn((input_size + hidden_size, 2*hidden_size),dtype=dtype))
        self.weight_zh = Parameter(torch.randn((hidden_size, hidden_size),dtype=dtype))
        # self.bias_zh = Parameter(torch.randn((3 * hidden_size),dtype=dtype))
        self.bias_zh = Parameter(torch.randn((hidden_size),dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
    # @torch.jit.script          
    @jit.script_method
    def forward(
        self, input_seq: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        nseq, batch_size, nx = input_seq.shape
        epss = torch.randn((nseq, batch_size, self.hidden_size),device=input_seq.device)
        epss = epss.unbind(0)
        inputs = input_seq.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        hx, cx = state
        
        # weight_ih = self.weight_ih.t()
        # weight_hh = self.weight_hh.t()
        
        for i in range(len(inputs)):
            x = inputs[i]
            
            eps = epss[i]
            # print("shape x", x.shape, "shape hid", hidden.shape)
            inp = torch.cat((x, hx), dim=1)
            predicted_distribution = torch.mm(inp, self.weight_encoder) 
            mean_, logvar_ = predicted_distribution.chunk(2,1)
            z = mean_ + eps * torch.exp(0.5*logvar_)
            z = torch.mm(z, self.weight_zh)  + self.bias_zh
            outgate = torch.sigmoid(z)
            
            # hx, cx = state
            gates = (
                torch.mm(x, self.weight_ih)
                + self.bias_ih
                + torch.mm(hx, self.weight_hh)
                + self.bias_hh
            )
            # ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            ingate, forgetgate, cellgate = gates.chunk(3, 1)
            
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            # outgate = torch.sigmoid(outgate)
    
            cx = (forgetgate * cx) + (ingate * cellgate)

            hx = outgate * torch.tanh(cx)
            # hx = torch.tanh(cx)
            
            # state =  (hy, cy)
            # outputs += [hy]
            outputs += [hx]

        # state =  (hy, cy)
        state =  (hx, cx)

        return torch.stack(outputs), state
    
class MyStochasticLSTMLayer2(jit.ScriptModule):
    use_bias: Final[bool]

    def __init__(self, input_size, hidden_size, use_bias):
        super().__init__()
        # self.cell = cell(*cell_args)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(( input_size, 3 * hidden_size)))
        self.weight_hh = Parameter(torch.randn(( hidden_size, 3 * hidden_size)))
        
        self.weight_encoder =  Parameter(torch.randn((hidden_size, 2*hidden_size)))
        self.weight_zh = Parameter(torch.randn((hidden_size, hidden_size)))
        self.use_bias = use_bias
        if self.use_bias:
          self.bias_zh = Parameter(torch.randn((hidden_size)))
          self.bias_ih = Parameter(torch.randn((3 * hidden_size)))
          self.bias_hh = Parameter(torch.randn((3 * hidden_size)))
          
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
    # @torch.jit.script          
    # @jit.script_method
    @torch.compile
    def forward(
        self, input_seq: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        nseq, batch_size, nx = input_seq.shape
        # epss = torch.randn((nseq, batch_size, self.hidden_size),device=input_seq.device)
        # epss = torch.randn_like(input_seq)
        # epss = epss.unbind(0)
        inputs = input_seq.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        hx, cx = state
        
        # weight_ih = self.weight_ih.t()
        # weight_hh = self.weight_hh.t()
        
        for i in range(len(inputs)):
            x = inputs[i]
            
            # eps = epss[i]
            eps = torch.randn_like(hx)
            # print("shape x", x.shape, "shape hid", hidden.shape)
            # inp = torch.cat((x, hx), dim=1)
            predicted_distribution = torch.mm(hx, self.weight_encoder) 
            mean_, logvar_ = predicted_distribution.chunk(2,1)
            z = mean_ + eps * torch.exp(0.5*logvar_)
            if self.use_bias:
                z = torch.mm(z, self.weight_zh)  + self.bias_zh
            else:
                z = torch.mm(z, self.weight_zh)
            outgate = torch.sigmoid(z)
            
            # hx, cx = state
            if self.use_bias:
              gates = (
                  torch.mm(x, self.weight_ih)
                  + self.bias_ih
                  + torch.mm(hx, self.weight_hh)
                  + self.bias_hh
              )
            else:
              gates = (
                torch.mm(x, self.weight_ih)
                + torch.mm(hx, self.weight_hh)
              )
            # ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            ingate, forgetgate, cellgate = gates.chunk(3, 1)
            
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            # outgate = torch.sigmoid(outgate)
    
            cx = (forgetgate * cx) + (ingate * cellgate)

            hx = outgate * torch.tanh(cx)
            # hx = torch.tanh(cx)
            
            # state =  (hy, cy)
            # outputs += [hy]
            outputs += [hx]

        # state =  (hy, cy)
        state =  (hx, cx)

        return torch.stack(outputs), state

class MyStochasticLSTMLayer3(jit.ScriptModule):
    use_bias: Final[bool]

    def __init__(self, input_size, hidden_size, use_bias):
        super().__init__()
        # self.cell = cell(*cell_args)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(( input_size, 3 * hidden_size)))
        self.weight_hh = Parameter(torch.randn(( hidden_size, 3 * hidden_size)))
        self.weight_encoder =  Parameter(torch.randn((hidden_size, 2*hidden_size)))
        self.use_bias = use_bias
        self.tau_t = torch.tensor(0.5)
        self.tau_e = torch.sqrt(1 - self.tau_t**2)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
    # def ar_noise(self, eps_prev, eps):
    #     # tau_t = 0.5
    #     # tau_e = torch.sqrt(1 - tau_t**2)
    #     # eps = np.random.randn(1)
    #     eps_t = self.tau_t * eps_prev + self.tau_e * eps
    #     return eps_t       
                
    # @torch.jit.script          
    # @jit.script_method
    @torch.compile
    def forward(
        self, input_seq: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        nseq, batch_size, nx = input_seq.shape
        epss = torch.randn((nseq, batch_size, self.hidden_size),device=input_seq.device, dtype=input_seq.dtype)
        # epss = epss.unbind(0)
        
        # eps_prev = eps_prev.unbind(0)
        inputs = input_seq.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        hx, cx = state
        
        for i in range(len(inputs)):
            # x = inputs[i]
            eps = epss[i]
            # eps = torch.randn_like(hx)
            # inp = torch.cat((x, hx), dim=1)
            predicted_distribution = torch.mm(hx, self.weight_encoder) 
            mean_, logvar_ = predicted_distribution.chunk(2,1)
            
            z = mean_ + eps * torch.exp(0.5*logvar_)
            outgate = torch.sigmoid(z)

            x = inputs[i]
            gates = (
                torch.mm(x, self.weight_ih) + torch.mm(hx, self.weight_hh)
              )
            ingate, forgetgate, cellgate = gates.chunk(3, 1)
            
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
    
            cx = (forgetgate * cx) + (ingate * cellgate)

            hx = outgate * torch.tanh(cx)
        
            outputs += [hx]

        state =  (hx, cx)

        return torch.stack(outputs), state
    
    
class MyStochasticLSTMLayer3_ar(jit.ScriptModule):
    use_bias: Final[bool]

    def __init__(self, input_size, hidden_size, use_bias):
        super().__init__()
        # self.cell = cell(*cell_args)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(( input_size, 3 * hidden_size)))
        self.weight_hh = Parameter(torch.randn(( hidden_size, 3 * hidden_size)))
        self.weight_encoder =  Parameter(torch.randn((hidden_size, 2*hidden_size)))
        self.use_bias = use_bias
        # self.tau_t = torch.tensor(0.5)
        # self.tau_e = torch.sqrt(1 - self.tau_t**2)
        # tau_t = torch.tensor(0.5)
        # tau_e = torch.sqrt(1 - tau_t**2)
        # self.register_buffer('tau_t', tau_t)
        # self.register_buffer('tau_e', tau_e)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
                  
                
    # @torch.jit.script          
    # @jit.script_method
    @torch.compile()
    def forward(
        self, input_seq: Tensor, state: Tuple[Tensor, Tensor], eps_t: Tensor
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        nseq, batch_size, nx = input_seq.shape
        # epss = torch.randn((nseq, batch_size, self.hidden_size),device=input_seq.device, dtype=input_seq.dtype)
        # epss = epss.unbind(0)
        # if eps_t_seq.dim() == 3:
        #     eps_t = eps_t_seq.unbind(0)
        #     eps_has_vertical_dim=True 
        # else:
        #     eps_t = eps_t_seq
        #     eps_has_vertical_dim=False
        if eps_t.dim() == 3:
            eps_has_vertical_dim=True 
        else:
            eps_has_vertical_dim=False
        # eps_prev = eps_prev.unbind(0)
        inputs = input_seq.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        hx, cx = state
        
        for i in range(len(inputs)):
            # x = inputs[i]
            # eps = epss[i]
            # eps = torch.randn_like(hx)
            # print("shape x", x.shape, "shape hid", hidden.shape)
            # inp = torch.cat((x, hx), dim=1)
            predicted_distribution = torch.mm(hx, self.weight_encoder) 
            # mean_, logvar_ = predicted_distribution.chunk(2,1)
            z, logvar_ = predicted_distribution.chunk(2,1)

            # eps_t = self.tau_t * eps_prev[i] + self.tau_e * eps
            if eps_has_vertical_dim:
                eps = eps_t[i]
            else:
                eps = eps_t 
            z = z + eps * torch.exp(0.5*logvar_)
            # z = mean_ + eps * torch.exp(0.5*logvar_)
            z = torch.sigmoid(z)

            x = inputs[i]
            gates = (
                torch.mm(x, self.weight_ih) + torch.mm(hx, self.weight_hh)
              )
            ingate, forgetgate, cellgate = gates.chunk(3, 1)
            
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
    
            cx = (forgetgate * cx) + (ingate * cellgate)

            hx = z * torch.tanh(cx)
            # hx = outgate * torch.tanh(cx)
            outputs += [hx]

        state =  (hx, cx)

        return torch.stack(outputs), state
    
class MyStochasticLSTMLayer3_ar_mlp_fused(jit.ScriptModule):

    def __init__(self, input_size, hidden_size, final_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.final_size = final_size
        self.weight_ih = Parameter(torch.randn(( input_size, 3 * hidden_size)))
        self.weight_hh = Parameter(torch.randn(( hidden_size, 3 * hidden_size)))
        self.weight_encoder =  Parameter(torch.randn((hidden_size, 2*hidden_size)))
        self.weight_decoder =  Parameter(torch.randn((hidden_size, final_size)))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
                  
    @torch.compile
    def forward(
        self, input_seq: Tensor, state: Tuple[Tensor, Tensor], eps_t: Tensor
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        nseq, batch_size, nx = input_seq.shape
        # epss = torch.randn((nseq, batch_size, self.hidden_size),device=input_seq.device, dtype=input_seq.dtype)
        # epss = epss.unbind(0)
        # if eps_t_seq.dim() == 3:
        #     eps_t = eps_t_seq.unbind(0)
        #     eps_has_vertical_dim=True 
        # else:
        #     eps_t = eps_t_seq
        #     eps_has_vertical_dim=False
        if eps_t.dim() == 3:
            eps_has_vertical_dim=True 
        else:
            eps_has_vertical_dim=False
        # eps_prev = eps_prev.unbind(0)
        inputs = input_seq.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        hx, cx = state
        
        for i in range(len(inputs)):
            # x = inputs[i]
            # eps = epss[i]
            # eps = torch.randn_like(hx)
            # print("shape x", x.shape, "shape hid", hidden.shape)
            # inp = torch.cat((x, hx), dim=1)
            predicted_distribution = torch.mm(hx, self.weight_encoder) 
            # mean_, logvar_ = predicted_distribution.chunk(2,1)
            z, logvar_ = predicted_distribution.chunk(2,1)
            
            # eps_t = self.tau_t * eps_prev[i] + self.tau_e * eps
            if eps_has_vertical_dim:
                eps = eps_t[i]
            else:
                eps = eps_t 
            # z = mean_ + eps * torch.exp(0.5*logvar_)
            # outgate = torch.sigmoid(z)
            z = torch.sigmoid(z + eps * torch.exp(0.5*logvar_))

            x = inputs[i]
            gates = (
                torch.mm(x, self.weight_ih) + torch.mm(hx, self.weight_hh)
              )
            ingate, forgetgate, cellgate = gates.chunk(3, 1)
            
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
    
            cx = (forgetgate * cx) + (ingate * cellgate)

            # hx = outgate * torch.tanh(cx)
            hx = z * torch.tanh(cx)

            outputs += [hx]

        state =  (hx, cx)

        return torch.stack(outputs), state

class MyStochasticLSTMLayer4(jit.ScriptModule):
    use_bias: Final[bool]

    def __init__(self, input_size, hidden_size, use_bias):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_encoder =  Parameter(torch.randn((hidden_size+input_size, 5*hidden_size)))
        self.use_bias = use_bias
        self.tau_t = torch.tensor(0.5)
        self.tau_e = torch.sqrt(1 - self.tau_t**2)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
    # @torch.compile(options={"epilogue_fusion": True})
    @torch.compile
    def forward(
        self, input_seq: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        nseq, batch_size, nx = input_seq.shape
        epss = torch.randn((nseq, batch_size, self.hidden_size),device=input_seq.device, dtype=input_seq.dtype)
        
        inputs = input_seq.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        hx, cx = state
        for i in range(len(inputs)):
            eps = epss[i]
            x = inputs[i]
            
            inp = torch.cat((x, hx), dim=1)
            # inp = inp.unsqueeze(0)
            # inp = inp.expand((5,-1,-1))
            # yy = torch.bmm(inp, self.weight_encoder) 
            # yy = yy.unbind(0)
            # mean_, logvar_, ingate, forgetgate, cellgate = yy
            
            yy = torch.mm(inp, self.weight_encoder) 
            mean_, logvar_, ingate, forgetgate, cellgate = yy.chunk(5,1)
            # z = mean_ + eps * torch.exp(0.5*logvar_)
            # outgate = torch.sigmoid(z)
            hx = torch.sigmoid(mean_ + eps * torch.exp(0.5*logvar_))
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
    
            cx = (forgetgate * cx) + (ingate * cellgate)

            hx = hx * torch.tanh(cx)
            # hx = outgate * torch.tanh(cx)

            outputs += [hx]

        state =  (hx, cx)

        return torch.stack(outputs), state

class GLU(nn.Module):
    """ The static nonlinearity used in the S4 paper"""
    def __init__(self,  nseq, nneur, layernorm=True, dropout=0, expand_factor=2):
        super(GLU, self).__init__()
        self.activation = nn.GELU()
        self.nseq = nseq
        self.nneur = nneur
        self.layernorm = layernorm
        self.expand_factor=expand_factor

        if self.layernorm:
            self.normalization = nn.LayerNorm((self.nseq,self.nneur))

        if self.layernorm:
            self.normalization = nn.LayerNorm((self.nseq,self.nneur))
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
    fastconv: Final[bool]
    use_padding: Final[bool]
    
    def __init__(self, input_size: int, hidden_size: int, kernel_size: int, 
                 mode: str = "f",  pad="same"):
        super(QRNNLayer_noncausal, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.mode = mode
        
        self.fastconv = False
        # print("QRNN nx", input_size, "nh", hidden_size)
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
            
        # if self.mode == "fo" or self.mode == "ifo":
        #     self.o_conv = nn.Conv1d(input_size, hidden_size, kernel_size, padding=pad)

        # if self.mode == "ifo":
        #     self.i_conv = nn.Conv1d(input_size, hidden_size, kernel_size, padding=pad)
            
        self.logsigmoid = nn.LogSigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
    # @jit.script_method
    @torch.compile
    def forward(self, inputs:Tensor, init_state:Optional[Tuple[Tensor, Tensor]]):
        # inputs = shape: [batch x timesteps x features]
        batch, timesteps, _ = inputs.shape
        # print("qrn inputs shape 0", inputs.shape)

        # Apply convolutions
        # if init_state is None:
        #     inputs = inputs.transpose(1, 2)
        # else:
        #     init_state = torch.unsqueeze(init_state,1)
        #     inputs = torch.cat((init_state, inputs), dim=1)
        #     inputs = inputs.transpose(1, 2)
        inputs = inputs.transpose(1, 2)
        
        if self.use_padding:
            inputs = self.pad(inputs)
        # print("shape inp", inputs.shape)

        if self.fastconv:
            inputs = inputs.to(torch.bfloat16)
        
        # print("qrn inputs shape", inputs.shape)
        raw_f = self.f_conv(inputs).transpose(1, 2)
        raw_z = self.z_conv(inputs).transpose(1, 2)
        
        if init_state is not None:
            init_state1 = torch.unsqueeze(init_state[0],1)
            init_state2 = torch.unsqueeze(init_state[1],1)
            raw_f = torch.cat((init_state1, raw_f), dim=1)
            raw_z = torch.cat((init_state2, raw_z), dim=1)
            
            
        # print("shape raw_f", raw_f.shape)
        
        # if self.mode == "ifo":
        #     raw_i = self.i_conv(inputs).transpose(1, 2)
        #     log_one_minus_f = self.logsigmoid(raw_i)
        # else:
        log_one_minus_f = self.logsigmoid(-raw_f)
        
        # Get log values of activations
        if self.fastconv:
            raw_z = raw_z.to(torch.float32)
            raw_f = raw_f.to(torch.float32)
        log_z = self.logsigmoid(raw_z)  # Use sigmoid activation
        log_f = self.logsigmoid(raw_f)
    
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
        # if self.mode == "fo" or self.mode == "ifo":
        #     o = torch.sigmoid(self.o_conv(inputs)).transpose(1, 2)
        #     hidden = hidden * o
        
        return hidden
    
