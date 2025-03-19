#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytorch model constructors
"""
import os 
import gc
import sys
import torch
import torch.nn as nn
import torch.nn.parameter as Parameter
# from layers_callbacks_torch import LayerPressure 
import torch.nn.functional as F
from models_torch_kernels import GLU
import numpy as np 
from typing import Final 

lbd_qi = np.array([10000000.        , 10000000.        , 10000000.        ,
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


lbd_qc = np.array([10000000.        , 10000000.        , 10000000.        ,
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
    
class LayerPressureThickness(nn.Module):
    def __init__(self,hyai, hybi, name='LayerPressureThickness',
                  # sp_min=62532.977,sp_max=104871.82,
                  ):
        super(LayerPressureThickness, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.sp_min = sp_min
        # self.sp_max = sp_max
        # self.pres_min = 36.434
        # self.pres_max = self.sp_max
        self.nlev = hyai.shape[0] - 1
        self.hyai = torch.reshape(hyai,(1,self.nlev+1,1)).to(device)
           
        self.hybi = torch.reshape(hybi,(1,self.nlev+1,1)).to(device)
        
    def forward(self, sp):
        # sp  (batch, 1, 1)   
        # hyai,hybi  (1, nlev+1, 1)
        
        dp = sp*(self.hybi[:,1:self.nlev+1]-self.hybi[:,0:self.nlev]) + 100000.0*(self.hyai[:,1:self.nlev+1]-self.hyai[:,0:self.nlev])

        return dp
    
    
class MyRNN(nn.Module):
    def __init__(self, RNN_type='LSTM', 
                 nx = 9, nx_sfc=17, 
                 ny = 8, ny_sfc=8, 
                 nneur=(64,64), 
                 outputs_one_longer=False, # if True, inputs are a sequence
                 # of N and outputs a sequence of N+1 (e.g. predicting fluxes)
                 concat=False, out_scale=None, out_sfc_scale = None):
        # Simple bidirectional RNN (Either LSTM or GRU) for predicting column 
        # outputs shaped either (B, L, Ny) or (B, L+1, Ny) from column inputs
        # (B, L, Nx) and optionally surface inputs (B, Nx_sfc) 
        # If surface inputs exist, they are used to initialize first (upward) RNN 
        # Assumes top-of-atmosphere is first in memory i.e. at index 0 
        # if it's not the flip operations need to be moved!
        super(MyRNN, self).__init__()
        self.nx = nx
        self.ny = ny 
        self.nx_sfc = nx_sfc 
        self.ny_sfc = ny_sfc
        self.nneur = nneur 
        self.outputs_one_longer=outputs_one_longer
        if len(nneur) < 1 or len(nneur) > 3:
            sys.exit("Number of RNN layers and length of nneur should be 2 or 3")

        self.RNN_type=RNN_type
        if self.RNN_type=='LSTM':
            RNN_model = nn.LSTM
        elif self.RNN_type=='GRU':
            RNN_model = nn.GRU
        else:
            raise NotImplementedError()
                    
        self.concat=concat
        
        if out_scale is not None:
            cuda = torch.cuda.is_available() 
            device = torch.device("cuda" if cuda else "cpu")
            self.yscale_lev = torch.from_numpy(out_scale).to(device)
            self.yscale_sca = torch.from_numpy(out_sfc_scale).to(device)

        if self.nx_sfc > 0:
            self.mlp_surface1  = nn.Linear(nx_sfc, self.nneur[0])
            if self.RNN_type=="LSTM":
                self.mlp_surface2  = nn.Linear(nx_sfc, self.nneur[0])

        self.rnn1      = RNN_model(nx,            self.nneur[0], batch_first=True) # (input_size, hidden_size, num_layers=1
        self.rnn2      = RNN_model(self.nneur[0], self.nneur[1], batch_first=True)
        if len(self.nneur)==3:
            self.rnn3      = RNN_model(self.nneur[1], self.nneur[2], batch_first=True)

        # The final hidden variable is either the output from the last RNN, or
        # the  concatenated outputs from all RNNs
        if concat:
            nh_rnn = sum(nneur)
        else:
            nh_rnn = nneur[-1]

        self.mlp_output = nn.Linear(nh_rnn, self.ny)
        if self.ny_sfc>0:
            self.mlp_surface_output = nn.Linear(nneur[-1], self.ny_sfc)
            
    def postprocessing(self, out, out_sfc):
        out_denorm = out / self.yscale_lev
        out_sfc_denorm  = out_sfc / self.yscale_sca
        return out_denorm, out_sfc_denorm
    
    def forward(self, inputs_main, inputs_aux):
            
        # batch_size = inputs_main.shape[0]
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
        sfc1 = self.mlp_surface1(inputs_aux)
        sfc1 = nn.Tanh()(sfc1)

        if self.RNN_type=="LSTM":
            sfc2 = self.mlp_surface2(inputs_aux)
            sfc2 = nn.Tanh()(sfc2)
            hidden = (sfc1.view(1,-1,self.nneur[0]), sfc2.view(1,-1,self.nneur[0])) # (h0, c0)
        else:
            hidden = (sfc1.view(1,-1,self.nneur[0]))

        # print(f'Using state1 {hidden}')
        # TOA is first in memory, so we need to flip the axis
        inputs_main = torch.flip(inputs_main, [1])
      
        out, hidden = self.rnn1(inputs_main, hidden)
        
        if self.outputs_one_longer:
            out = torch.cat((sfc1, out),axis=1)

        out = torch.flip(out, [1]) # the surface was processed first, but for
        # the second RNN (and the final output) we want TOA first
        
        out2, hidden2 = self.rnn2(out) 
        
        (last_h, last_c) = hidden2

        if len(self.nneur)==3:
            rnn3_input = torch.flip(out2, [1])
            
            out3, hidden3 = self.rnn3(rnn3_input) 
            
            out3 = torch.flip(out3, [1])
            
            if self.concat:
                rnnout = torch.cat((out3, out2, out),axis=2)
            else:
                rnnout = out3
        else:
            if self.concat:
                rnnout = torch.cat((out2, out),axis=2)
            else:
                rnnout = out2
        
        out = self.mlp_output(rnnout)

        if self.ny_sfc>0:
            #print("shape last_c", last_c.shape)
            # use cell state or hidden state?
            out_sfc = self.mlp_surface_output(last_h.squeeze())
            return out, out_sfc
        else:
            return out 


class RNN_autoreg(nn.Module):
    def __init__(self, hyam, hybm, nlay=60, nx = 4, nx_sfc=3, ny = 4, ny_sfc=3, nneur=(64,64), 
                cell_type="LSTM",
                memory="None", # "None", "Hidden", or "Output",
                concat=False,
                use_initial_mlp=False, 
                ensemble_size=1,
                random_init_cx=False,
                use_intermediate_mlp=True,
                add_pres=False,
                third_rnn=False,
                add_stochastic_layer=False,
                coeff_stochastic = 0.0,
                dtype=torch.float32,
                # use_mp_constraint = False,
                mp_mode=0,
                preprocess=False,
                out_scale=None, out_sfc_scale=None,
                xcoeffs=None,
                v4_to_v5_inputs=False,
                separate_radiation=False,
                inference_mode=False):
        super(RNN_autoreg, self).__init__()
        self.mp_mode=mp_mode
        if self.mp_mode==1:
            self.use_mp_constraint = True 
            self.pred_liq_frac = False
        elif self.mp_mode==2:
            self.use_mp_constraint = False 
            self.pred_liq_frac = True
        else:
            self.use_mp_constraint = False 
            self.pred_liq_frac = False
        # self.add_liqfrac_inp = True
        # if self.add_liqfrac_inp:
        #     nx = nx + 1
        # self.nx = nx

        # if self.use_mp_constraint:
        #     self.ny = ny - 1
        # else:
        #     self.ny = ny 
        self.ny = ny 
        self.nlay = nlay 
        self.nx_sfc = nx_sfc 
        self.ny_sfc = ny_sfc
        self.nneur = nneur 
        self.use_initial_mlp=use_initial_mlp
        self.v4_to_v5_inputs = v4_to_v5_inputs
        self.output_prune = True
        self.add_pres = add_pres
        if self.add_pres:
            self.preslay = LayerPressure(hyam,hybm)
            nx = nx +1
        self.nx = nx

        self.nh_rnn1 = self.nneur[0]
        self.nx_rnn2 = self.nneur[0]
        self.nh_rnn2 = self.nneur[1]
        self.concat = concat
        self.model_type=cell_type
        self.ensemble_size = ensemble_size
        self.random_init_cx= random_init_cx
        self.third_rnn=third_rnn
        self.add_stochastic_layer = add_stochastic_layer
        self.coeff_stochastic = coeff_stochastic
        self.separate_radiation = separate_radiation
        self.inference_mode=inference_mode
        # in ClimSim config of E3SM, the CRM physics first computes 
        # moist physics on 50 levels, and then computes radiation on 60 levels!
        # self.dtype=dtype
        if self.ensemble_size>1 and not add_stochastic_layer:
        # In this case the stochasticity comes purely from random initialization
        # of hidden states (probably not enough)
            self.random_init_cx=True
        self.use_intermediate_mlp=use_intermediate_mlp
        self.share_weights = False
        if self.share_weights:
            self.use_intermediate_mlp=False
        if self.use_initial_mlp:
            self.nx_rnn1 = self.nneur[0]
        else:
            self.share_weights=False
            self.nx_rnn1 = nx
        if self.separate_radiation:
            self.nlay = 50
            self.nlay_rad = 60
            self.nx_rad = self.nx - 2
            # self.nx_rad = nx - 2
            self.nx_sfc_rad = 15

            self.nx_rnn1 = self.nx_rnn1 - 3
            self.ny_rad = 1
            self.ny_sfc_rad = self.ny_sfc - 2
            self.ny_sfc = 2
        self.memory = memory
        self.preprocess=preprocess 
        if self.preprocess:
            print("On the fly preprocessing turned ON")

        cuda = torch.cuda.is_available() 
        device = torch.device("cuda" if cuda else "cpu")
        if out_scale is not None:
            self.yscale_lev = torch.from_numpy(out_scale).to(device)
            self.yscale_sca = torch.from_numpy(out_sfc_scale).to(device)
        if xcoeffs is not None:
            xcoeff_lev, xcoeff_sca = xcoeffs
            # xmean_lev, xmin_lev, xmax_lev, xmean_sca, xmin_sca, xmax_sca = xcoeff
            # xcoeff_lev = np.stack((xmean_lev, xmin_lev, xmax_lev))
            self.xcoeff_lev = torch.from_numpy(xcoeff_lev).to(device)
            self.xcoeff_sca = torch.from_numpy(xcoeff_sca).to(device)
            self.lbd_qi = torch.from_numpy(lbd_qi).to(device)
            self.lbd_qc = torch.from_numpy(lbd_qc).to(device)
        else:
            if self.use_mp_constraint or self.preprocess:
                raise NotImplementedError("input scaling coefficients needed for MP constraint or on-the-fly norm")
            
        if memory == 'None':
            raise NotImplementedError()
        elif memory == 'Output':
            print("Building RNN that feeds its output t0,z0 to its inputs at t1,z0")
            self.rnn1_mem = None 
            self.nh_mem = self.ny
            self.nx_rnn1 = self.nx_rnn1 + self.nh_mem
            self.nh_rnn1 = self.nneur[0]
        elif memory == 'Hidden':
            if use_intermediate_mlp:
                self.nh_mem = self.nneur[1] // 2
            else:
                self.nh_mem = self.nneur[1]
            print("Building RNN that feeds its hidden memory at t0,z0 to its inputs at t1,z0")
            print("Initial mlp: {}, intermediate mlp: {}".format(self.use_initial_mlp, self.use_intermediate_mlp))
 
            self.rnn1_mem = None 
            self.nx_rnn1 = self.nx_rnn1 + self.nh_mem
            self.nh_rnn1 = self.nneur[0]
        elif memory == 'CustomLSTM':
            raise NotImplementedError()
        else:
            sys.exit("memory argument must equal one of : 'None', 'Hidden', or 'CustomLSTM'")
            
        print("nx rnn1", self.nx_rnn1, "nh rnn1", self.nh_rnn1)
        print("nx rnn2", self.nx_rnn2, "nh rnn2", self.nh_rnn2)  
        print("nx sfc", self.nx_sfc)

        print("Cell type:", cell_type)
        if self.use_initial_mlp:
            self.mlp_initial = nn.Linear(nx, self.nneur[0])

        self.mlp_surface1  = nn.Linear(nx_sfc, self.nh_rnn1)
        if not self.random_init_cx:
            self.mlp_surface2  = nn.Linear(nx_sfc, self.nh_rnn1)

        if self.model_type=="LSTM":
            # self.rnn1      = nn.LSTMCell(self.nx_rnn1, self.nh_rnn1)  # (input_size, hidden_size)
            # self.rnn2      = nn.LSTMCell(self.nx_rnn2, self.nh_rnn2)
            self.rnn1      = nn.LSTM(self.nx_rnn1, self.nh_rnn1,  batch_first=True)  # (input_size, hidden_size)
            if not self.share_weights:
                self.rnn2      = nn.LSTM(self.nx_rnn2, self.nh_rnn2,  batch_first=True)
                if self.third_rnn:
                    self.rnn3      = nn.LSTM(self.nh_rnn2, self.nneur[2],  batch_first=True)
                
        elif self.model_type=="GRU":
              self.rnn1      = nn.GRU(self.nx_rnn1, self.nh_rnn1,  batch_first=True)   # (input_size, hidden_size)
              self.rnn2      = nn.GRU(self.nx_rnn2, self.nh_rnn2,  batch_first=True) 
        elif self.model_type=="RNN":
              self.rnn1      = nn.RNN(self.nx_rnn1, self.nh_rnn1,  batch_first=True)   # (input_size, hidden_size)
              self.rnn2      = nn.RNN(self.nx_rnn2, self.nh_rnn2,  batch_first=True) 
        else:
            raise NotImplementedError()
            
            
        if self.add_stochastic_layer:
            from models_torch_kernels import StochasticGRUCell, MyStochasticGRUCell
            nx_srnn = self.nh_rnn2
            # self.rnn_stochastic = StochasticGRUCell(nx_srnn, self.nh_rnn2)  # (input_size, hidden_size)
            self.rnn_stochastic = MyStochasticGRUCell(nx_srnn, self.nh_rnn2)  # (input_size, hidden_size)
        if concat:
            nh_rnn = self.nh_rnn2+self.nh_rnn1
        else:
            nh_rnn = self.nh_rnn2

        if self.use_intermediate_mlp: 
            self.mlp_latent = nn.Linear(nh_rnn, self.nh_mem)
            self.mlp_output = nn.Linear(self.nh_mem, self.ny)
        else:
            self.mlp_output = nn.Linear(nh_rnn, self.ny)
            
        if self.ny_sfc>0:
            self.mlp_surface_output = nn.Linear(nneur[-1], self.ny_sfc)
            
        if self.separate_radiation:
            # self.nh_rnn1_rad = self.nh_rnn1 
            # self.nh_rnn2_rad = self.nh_rnn2 
            self.nh_rnn1_rad = 96 
            self.nh_rnn2_rad = 96
            self.rnn1_rad      = nn.GRU(self.nh_rnn2+self.nx_rad, self.nh_rnn1_rad,  batch_first=True)   # (input_size, hidden_size)
            self.rnn2_rad      = nn.GRU(self.nh_rnn1_rad, self.nh_rnn2_rad,  batch_first=True) 
            self.mlp_surface_rad = nn.Linear(self.nx_sfc_rad, self.nh_rnn1_rad)
            self.mlp_toa_rad = nn.Linear(1, self.nh_rnn1_rad)
            self.mlp_surface_output_rad = nn.Linear(self.nh_rnn2_rad, self.ny_sfc_rad)
            self.mlp_output_rad = nn.Linear(self.nh_rnn2_rad, self.ny_rad)

    def temperature_scaling(self, T_raw):
        # T_denorm = T = T*(self.xmax_lev[:,0] - self.xmin_lev[:,0]) + self.xmean_lev[:,0]
        # T_denorm = T*(self.xcoeff_lev[2,:,0] - self.xcoeff_lev[1,:,0]) + self.xcoeff_lev[0,:,0]
        # liquid_ratio = (T_raw - 253.16) / 20.0 
        liquid_ratio = (T_raw - 253.16) * 0.05 
        liquid_ratio = F.hardtanh(liquid_ratio, 0.0, 1.0)
        return liquid_ratio
    
    # def preprocessing(self, x_main, x_sfc):
    #     #                                mean                 max             min
    #     x_main = (x_main - self.xcoeff_lev[0])/(self.xcoeff_lev[2] - self.xcoeff_lev[1])
    #     x_sfc =  (x_sfc  - self.xcoeff_sca[0])/(self.xcoeff_sca[2] - self.xcoeff_sca[1])
    #     x_main[:,:,2] = 1 - torch.exp(-x_main[:,:,2] * self.lbd_qc)
    #     x_main[:,:,3] = 1 - torch.exp(-x_main[:,:,3] * self.lbd_qi)        
    #     x_main = torch.where(torch.isnan(x_main), torch.tensor(0.0, device=x_main.device), x_main)
    #     x_sfc = torch.where(torch.isinf(x_sfc), torch.tensor(0.0, device=x_sfc.device), x_sfc)
    #     return x_main, x_sfc 
    
    def preprocessing(self, x_main, x_sfc):
        # convert v4 input array to v5 input array:
        # ['state_t',
        # 'state_rh',
        # 'state_q0002' = qliq   -->  qn 
        # 'state_q0003' = qice   --> liquid ratio
        x_main[:,:,2]   =  x_main[:,:,2] +  x_main[:,:,3]  
        x_main[:,:,3]   = self.temperature_scaling(x_main[:,:,0])      

        #                                mean                 max             min
        x_main = (x_main - self.xcoeff_lev[0])/(self.xcoeff_lev[2] - self.xcoeff_lev[1])
        x_sfc =  (x_sfc  - self.xcoeff_sca[0])/(self.xcoeff_sca[2] - self.xcoeff_sca[1])
        x_main[:,:,2] = 1 - torch.exp(-x_main[:,:,2] * self.lbd_qn)
        x_main = torch.where(torch.isnan(x_main), torch.tensor(0.0, device=x_main.device), x_main)
        x_sfc  = torch.where(torch.isinf(x_sfc),  torch.tensor(0.0, device=x_sfc.device), x_sfc)
        return x_main, x_sfc 
    
    def reverse_input_scaling(self, x_main, x_sfc=None):
        x_main_raw = x_main * (self.xcoeff_lev[2] - self.xcoeff_lev[1]) + self.xcoeff_lev[0] 
        if x_sfc is None:
            return x_main_raw
        else:
            x_sfc_raw  = x_sfc * (self.xcoeff_sca[2] - self.xcoeff_sca[1]) + self.xcoeff_sca[0] 
            return x_main_raw, x_sfc_raw 
        
    def postprocessing(self, out, out_sfc):
        out_denorm      = out / self.yscale_lev
        out_sfc_denorm  = out_sfc / self.yscale_sca
        return out_denorm, out_sfc_denorm

    # @torch.compile
    # def pp_mp(self, out, out_sfc, tar_true, tar_sfc, x_denorm):
    def pp_mp(self, out, out_sfc, x_denorm):

        if self.output_prune:
            # out[:,0:15,1] = 0.0 # rh
            out[:,0:15,2] = 0.0 # qn
            # out[:,0:15,4] = 0.0 # liqratio"
        out_denorm      = out / self.yscale_lev
        # tar_denorm      = tar_true / self.yscale_lev

        T_before        = x_denorm[:,:,0:1]
        qliq_before     = x_denorm[:,:,2:3]
        qice_before     = x_denorm[:,:,3:4]   
        qn_before       = qliq_before + qice_before 

        # print("shape x denorm", x_denorm.shape, "T", T_before.shape)
        T_new           = T_before  + out_denorm[:,:,0:1]*1200
        # T_new_tar       = T_before  + tar_denorm[:,:,0:1]*1200

        # T_new           = T_before  + out_denorm[:,:,0:1]*1200
        liq_frac_constrained        = self.temperature_scaling(T_new)
        # liq_frac_tar    = self.temperature_scaling(T_new_tar)


        if self.pred_liq_frac:
            # dqn         = out_denorm[:,:,2:3].clone()
            # liq_frac    = out_denorm[:,:,3:4].clone()
            liq_frac    = out_denorm[:,:,3:4].clone()**16

        
            # dqliq       = liq_frac * dqn
            # dqice       = (1-liq_frac)*dqn
            # out_denorm[:,:,2:3] = dqliq
            # out_denorm[:,:,3:4] = dqice
            # total = out_true[:,:,2] + out_true[:,:,3] 
            # liq_frac_real = out_true[:,:,2] / total 
            
            qn_new          = qn_before + out_denorm[:,:,2:3]*1200  
            # qn_new          = nn.ReLU()(qn_new)

            # qn_new_tar      = qn_before + tar_denorm[:,:,2:3]*1200  
            qliq_new    = liq_frac*qn_new
            qice_new    = (1-liq_frac)*qn_new
            dqliq       = (qliq_new - qliq_before) * 0.0008333333333333334 #/1200  
            dqice       = (qice_new - qice_before) * 0.0008333333333333334 #/1200   
            # dqliq_tar   = (qliq_new_tar - qliq_before) * 0.0008333333333333334 #/1200  
            # dqice_tar   = (qice_new_tar - qice_before) * 0.0008333333333333334 #/1200  
            
            out_denorm[:,:,2:3] = dqliq
            out_denorm[:,:,3:4] = dqice
            # liq_frac_real = out_true[:,:,3]
            # print("T-old {:.1f} T {:.1f}  liqfrac {:.2f} constrained {:.2f} real {:.2f}  max {:.2f}  qbef {:.2e}  dqn {:.2e}".format( 
            #                                                 # x_denorm[200,35,4].item(),
            #                                                 T_before[200,35].item(),
            #                                                 T_new[200,35].item(), 
            #                                                 liq_frac[200,35].item(), 
            #                                                 liq_frac_constrained[200,35].item(),
            #                                                 liq_frac_real[200,35].item(),
            #                                                 liq_frac_real[200,35].max().item(),
            #                                                 qn_before[200,35].item(),
            #                                                 (out_denorm[200,35,2]*1200).item() ))
        else:
            #                              dqn
            qn_new          = qn_before + out_denorm[:,:,2:3]*1200  
            # qn_new_tar      = qn_before + tar_denorm[:,:,2:3]*1200  
            qliq_new    = liq_frac_constrained*qn_new
            qice_new    = (1-liq_frac_constrained)*qn_new
            dqliq       = (qliq_new - qliq_before) * 0.0008333333333333334 #/1200  
            dqice       = (qice_new - qice_before) * 0.0008333333333333334 #/1200   
            # dqliq_tar   = (qliq_new_tar - qliq_before) * 0.0008333333333333334 #/1200  
            # dqice_tar   = (qice_new_tar - qice_before) * 0.0008333333333333334 #/1200  
            out_denorm  = torch.cat((out_denorm[:,:,0:2], dqliq, dqice, out_denorm[:,:,3:]),dim=2)
            
            # print("Tbef {:.2f}   T {:.2f}  dt {:.2e}  liqfrac {:.2f}   dqn {:.2e}  qbef {:.2e}  qnew {:.2e}  dqliq {:.2e}  dqice {:.2e} ".format( 
            #                                                 # x_denorm[200,35,4].item(),
            #                                                 T_before[200,35].item(), 
            #                                                 T_new[200,35].item(), 
            #                                                 (out_denorm[200,35,0:1]*1200).item(),
            #                                                 liq_frac_constrained[200,35].item(), 
            #                                                 (out_denorm[200,35,2]*1200).item(), 
            #                                                 qn_before[200,35].item(),
            #                                                 qn_new[200,35].item(),
            #                                                 dqliq[200,35].item(),
            #                                                 dqice[200,35].item()))

        # print("T-old {:.1f} T {:.1f}  dt {:.2e}  liqfrac {:.2f}   qbef {:.2e}  dqn {:.2e}  qnew {:.2e}  dqliq {:.2e}".format( 
        #                                                 # x_denorm[200,35,4].item(),
        #                                                 T_before[200,35].item(),
        #                                                 T_new[200,35].item(), 
        #                                                 (out_denorm[200,35,0:1]*1200).item(),
        #                                                 liq_frac[200,35].item(), 
        #                                                 qn_before[200,35].item(),
        #                                                 (out_denorm[200,35,2]*1200).item(), 
        #                                                 qn_new[200,35].item(),
        #                                                 dqliq[200,35].item()))
        # somethings wrong with q values, way too big
        #    FORGOT TO REVERSE EXP-NORM OF CLOUD INPUTS 

        # tar_denorm  = torch.cat((tar_denorm[:,:,0:2], dqliq_tar, dqice_tar, tar_denorm[:,:,3:]),dim=2)

        out_sfc_denorm  = out_sfc / self.yscale_sca
        # tar_sfc_denorm  = tar_sfc / self.yscale_sca
        return out_denorm, out_sfc_denorm

        # return out_denorm, out_sfc_denorm, tar_denorm, tar_sfc_denorm
    
    def reset_states(self):
        self.rnn1_mem = None

    def detach_states(self):
        self.rnn1_mem = self.rnn1_mem.detach()
   
    def get_states(self):
        return self.rnn1_mem.detach()

    def set_states(self, states):
        self.rnn1_mem = states 
        
    def forward(self, inputs_main, inputs_aux):
        if self.ensemble_size>0:
            inputs_main = inputs_main.unsqueeze(0)
            inputs_aux = inputs_aux.unsqueeze(0)
            inputs_main = torch.repeat_interleave(inputs_main,repeats=self.ensemble_size,dim=0)
            inputs_aux = torch.repeat_interleave(inputs_aux,repeats=self.ensemble_size,dim=0)
            inputs_main = inputs_main.flatten(0,1)
            inputs_aux = inputs_aux.flatten(0,1)
                    
        batch_size = inputs_main.shape[0]
        # print("shape inputs main", inputs_main.shape)
        if self.add_pres:
            sp = torch.unsqueeze(inputs_aux[:,0:1],1)
            # undo scaling
            sp = sp*35451.17 + 98623.664
            pres  = self.preslay(sp)
            inputs_main = torch.cat((inputs_main,pres),dim=2)
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # if self.preprocess:
        if self.inference_mode:
            T_before    = inputs_main[:,:,0:1]
            qliq_before = inputs_main[:,:,2:3]
            qice_before = inputs_main[:,:,3:4]
            qn_before   = qliq_before + qice_before
            # print("min max x1 before", inputs_main[:,:,0].min(), inputs_main[:,:,0].max())
            inputs_main, inputs_aux = self.preprocessing(inputs_main, inputs_aux)
            # print("min max x1 after", inputs_main[:,:,0].min(), inputs_main[:,:,0].max())
        
        # if self.add_liqfrac_inp:
        #     x_denorm        = self.reverse_input_scaling(inputs_main)
        #     liq_frac        = self.temperature_scaling(x_denorm[:,:,0:1])
        #     inputs_main     = torch.cat((inputs_main, liq_frac),axis=2)
            
        if self.rnn1_mem is None: 
            self.rnn1_mem = torch.randn(batch_size, self.nlay, self.nh_mem,device=device)
            # self.rnn1_mem = torch.randn((batch_size, self.nlay, self.nh_mem),dtype=self.dtype,device=device)

        hx = self.mlp_surface1(inputs_aux)
        hx = nn.Tanh()(hx)

        # TOA is first in memory, so to start at the surface we need to go backwards
        inputs_main = torch.flip(inputs_main, [1])
        
        if self.separate_radiation:
            # Do not use inputs -2,-3,-4 (O3, CH4, N2O) or first 10 levels (last 10 cos flipped)
            inputs_main_crm = torch.cat((inputs_main[:,0:50,0:-4], inputs_main[:,0:50,-1:]),dim=2)
        else:
            inputs_main_crm = inputs_main

        # The input (a vertical sequence) is concatenated with the
        # output of the RNN from the previous time step 
        if self.model_type in ["LSTM"]:
            if self.random_init_cx:
                cx = torch.randn(batch_size, self.nh_rnn1,device=device)
                # cx = torch.randn((batch_size, self.nh_rnn1),dtype=self.dtype,device=device)
            else:
                cx = self.mlp_surface2(inputs_aux)
                cx = nn.Tanh()(cx)
            hidden = (torch.unsqueeze(hx,0), torch.unsqueeze(cx,0))
        else:
            hidden = (torch.unsqueeze(hx,0))

        if self.use_initial_mlp:
            rnn1_input = self.mlp_initial(inputs_main_crm)
            rnn1_input = nn.Tanh()(rnn1_input)
        else:
            rnn1_input = inputs_main_crm 

        # print("shape rnn1 inp", rnn1_input.shape, "shape rnn1mem", self.rnn1_mem.shape)            
        rnn1_input = torch.cat((rnn1_input,self.rnn1_mem), axis=2)

        rnn1out, states = self.rnn1(rnn1_input, hidden)

        rnn1out = torch.flip(rnn1out, [1])

        # hx2 = torch.randn((batch_size, self.nh_rnn2),dtype=self.dtype,device=device)  # (batch, hidden_size)
        hx2 = torch.randn((batch_size, self.nh_rnn2), device=device)  # (batch, hidden_size)

        if self.model_type in ["LSTM"]:
            # cx2 = torch.randn((batch_size, self.nh_rnn2),dtype=self.dtype,device=device)
            cx2 = torch.randn((batch_size, self.nh_rnn2),device=device)
            hidden2 = (torch.unsqueeze(hx2,0), torch.unsqueeze(cx2,0))
        else:
            hidden2 = (torch.unsqueeze(hx2,0))

        input_rnn2 = rnn1out
            
        if self.share_weights:
            rnn2out, states = self.rnn1(input_rnn2, hidden2)
        else:
            rnn2out, states = self.rnn2(input_rnn2, hidden2)

        (last_h, last_c) = states

        if self.third_rnn:
            rnn3in = torch.flip(rnn2out, [1])
            rnn2out, states = self.rnn3(rnn3in)

        if self.concat:
            rnn2out = torch.cat((rnn2out,rnn1out),axis=2)
        
        if self.use_intermediate_mlp: 
            rnn2out = self.mlp_latent(rnn2out)
          
        if self.memory=="Hidden": 
            if not self.third_rnn: 
                self.rnn1_mem = torch.flip(rnn2out, [1])
            
        # Add a stochastic perturbation
        # Convective memory is still based on the deterministic model,
        # and does not include the stochastic perturbation
        # concat and use_intermediate_mlp should be set to false
        if self.add_stochastic_layer:
            # srnn_input = torch.transpose(rnn2out,0,1)
            srnn_input = torch.transpose(self.rnn1_mem,0,1)
            # transpose is needed because this layer assumes seq. dim first
            z = self.rnn_stochastic(srnn_input)
            z = torch.flip(z, [0])

            z = torch.transpose(z,0,1)
            # z = torch.flip(z, [1])
            # rnn2out = z
            # z is a perburbation added to the hidden state
            # rnn2out = rnn2out + 0.01*z 
            rnn2out = rnn2out + self.coeff_stochastic*z 

        out = self.mlp_output(rnn2out)
        if self.memory=="Output":
            if not self.third_rnn: self.rnn1_mem = torch.flip(out, [1])
            
        
        if self.output_prune:
        #     # out[:,0:15,1:] = out[:,0:15,1:]*1e-6
        #     # out[:,0:15,1:] = out[:,0:15,1:].clone().zero_()
            # if inputs_main.shape[-1]==16:
                
            #     pres_ref = torch.exp(inputs_main[:,:,15:16]) / 992.63556
            #     out = out*pres_ref
              
            # Only temperature tendency is computed for the top 10 levels
            # if self.separate_radiation:
            #     out[:,0:12,:] = out[:,0:12,:].clone().zero_()
            # else:
            if not self.separate_radiation:
                out[:,0:12,1:] = out[:,0:12,1:].clone().zero_()
        
        # if self.pred_liq_frac:
        #     liquid_ratio  = out[:,:,3:4].clone()
        #     liquid_ratio = F.hardtanh(liquid_ratio, 0.0, 1.0)
        #     # liquid_ratio = F.hardtanh(liquid_ratio, 0.0, 0.041666666666666664)
        #     out         = torch.cat((out[:,:,0:3], liquid_ratio, out[:,:,4:]),dim=2)

        # use cell state or hidden state?
        out_sfc = self.mlp_surface_output(last_h.squeeze())
        
        if self.separate_radiation:
            # out_crm = out.clone()
            out_new = torch.zeros(batch_size, self.nlay_rad, self.ny, device=device)
            out_new[:,10:,:] = out
            # Start at surface again
            # Do not use inputs 4,5 (winds)
            inputs_main_rad =  torch.cat((inputs_main[:,:,0:4], inputs_main[:,:,6:]),dim=2)
            # # add dT from crm 
            # T_old =   inputs_main * (self.xcoeff_lev[2,:,0:1] - self.xcoeff_lev[1,:,0:1]) + self.xcoeff_lev[0,:,0:1] 
            # T_new = T_old + dT
            inputs_rad =  torch.zeros(batch_size, self.nlay_rad, self.nh_rnn2+self.nx_rad, device=device)
            inputs_rad[:,10:,0:self.nh_rnn2] = torch.flip(rnn2out, [1])
            inputs_rad[:,:,self.nh_rnn2:] = inputs_main_rad
            # inputs_rad = torch.flip(inputs_rad, [1])

            hx = self.mlp_surface_rad(inputs_aux)
            hidden = (torch.unsqueeze(hx,0))
            rnn_out, states = self.rnn1_rad(inputs_rad, hidden)
            rnn_out = torch.flip(rnn_out, [1])
            rnn_out, last_h = self.rnn2_rad(rnn_out, hidden)
            out_rad = self.mlp_output_rad(rnn_out)

            out_sfc_rad = self.mlp_surface_output_rad(last_h)
            # dT_tot = dT_crm + dT_rad
            out_new[:,:,0:1] = out_new[:,:,0:1] + out_rad
            out = out_new
            #1D (scalar) Output variables: ['cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 
            #'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']
            # print("shape 1", out_sfc[:,0:1].shape, "2", out_sfc_rad.shape)
            out_sfc =  torch.cat((out_sfc[:,0:1], out_sfc_rad.squeeze(), out_sfc[:,1:]),dim=1)
            
            
        if self.inference_mode:
            out,out_sfc = self.postprocessing(out, out_sfc)
            T_new       = T_before  + out[:,:,0:1]*1200
            liq_frac    = self.temperature_scaling(T_new)
            qn_new      = qn_before + out[:,:,2:3]*1200  
            qliq_new    = liq_frac*qn_new
            qice_new    = (1-liq_frac)*qn_new
            dqliq       = (qliq_new - qliq_before) * 0.0008333333333333334 #/1200  
            dqice       = (qice_new - qice_before) * 0.0008333333333333334 #/1200  
            out         = torch.cat((out[:,:,0:2], dqliq, dqice, out[:,:,3:]),dim=2)

        return out, out_sfc



class LSTM_autoreg_torchscript(nn.Module):
    use_initial_mlp: Final[bool]
    use_intermediate_mlp: Final[bool]
    add_pres: Final[bool]
    add_stochastic_layer: Final[bool]
    output_prune: Final[bool]
    ensemble_size: Final[int]
    use_memory: Final[bool]
    separate_radiation: Final[bool]
    # predict_flux: Final[bool]
    use_third_rnn: Final[bool]

    def __init__(self, hyam, hybm,  hyai, hybi,
                out_scale, out_sfc_scale, 
                xmean_lev, xmean_sca, xdiv_lev, xdiv_sca,
                nlay=60, nx = 4, nx_sfc=3, ny = 4, ny_sfc=3, nneur=(64,64), 
                use_initial_mlp=False, 
                use_intermediate_mlp=True,
                add_pres=False,
                add_stochastic_layer=False,
                output_prune=False,
                use_memory=False,
                separate_radiation=False,
                use_third_rnn=False,
                # predict_flux=False,
                ensemble_size=1,
                coeff_stochastic = 0.0,
                nh_mem=16):
        super(LSTM_autoreg_torchscript, self).__init__()
        self.ny = ny 
        self.nlay = nlay 
        self.nx_sfc = nx_sfc 
        self.ny_sfc = ny_sfc
        self.nneur = nneur 
        self.use_initial_mlp=use_initial_mlp
        self.output_prune = output_prune
        self.add_pres = add_pres
        if self.add_pres:
            self.preslay = LayerPressure(hyam,hybm)
            nx = nx +1
        self.nx = nx
        self.nh_rnn1 = self.nneur[0]
        self.nx_rnn2 = self.nneur[0]
        self.nh_rnn2 = self.nneur[1]
        self.use_third_rnn = use_third_rnn
        # if len(nneur)==3:
            # self.use_third_rnn = True 
        if self.use_third_rnn:
            self.nx_rnn3 = self.nneur[1]
            self.nh_rnn3 = self.nneur[2]
        # elif len(nneur)==2:
        #     self.use_third_rnn = False 
        # else:
        #     raise NotImplementedError()

        self.use_memory= use_memory
        self.separate_radiation=separate_radiation
        # self.predict_flux=predict_flux
        # if self.predict_flux:
        #     self.preslev = LayerPressure(hyai, hybi, norm=False)
        if self.use_initial_mlp:
            self.nx_rnn1 = self.nneur[0]
        else:
            self.nx_rnn1 = nx

        if self.separate_radiation:
            self.nlay = 50
            self.nlay_rad = 60
            self.nx_rad = self.nx - 2
            # self.nx_rad = nx - 2
            self.nx_rnn1 = self.nx_rnn1 - 3
            self.nx_sfc_rad = 5
            self.nx_sfc = self.nx_sfc  - self.nx_sfc_rad
            self.ny_rad = 1
            self.ny_sfc_rad = self.ny_sfc - 2
            self.ny_sfc = 2
        self.ensemble_size = ensemble_size
        self.add_stochastic_layer = add_stochastic_layer
        self.coeff_stochastic = coeff_stochastic
        self.nonlin = nn.Tanh()
        self.relu = nn.ReLU()

        self.yscale_lev = torch.from_numpy(out_scale)
        self.yscale_sca = torch.from_numpy(out_sfc_scale)
        self.xmean_lev  = torch.from_numpy(xmean_lev)
        self.xmean_sca  = torch.from_numpy(xmean_sca)
        self.xdiv_lev   = torch.from_numpy(xdiv_lev)
        self.xdiv_sca   = torch.from_numpy(xdiv_sca)
        
        # in ClimSim config of E3SM, the CRM physics first computes 
        # moist physics on 50 levels, and then computes radiation on 60 levels!
        self.use_intermediate_mlp=use_intermediate_mlp
            
        if self.use_memory:
            if self.use_intermediate_mlp:
                # self.nh_mem = self.nneur[1] // 4
                # if nh_mem is None:
                #     self.nh_mem = self.nneur[1] // 8
                # else:
                self.nh_mem = nh_mem

            else:
                self.nh_mem = self.nneur[1]
            # self.rnn1_mem = None 
            print("Building RNN that feeds its hidden memory at t0,z0 to its inputs at t1,z0")
        else: 
            self.nh_mem = 0
            print("Building RNN without convective memory")

        print("Initial mlp: {}, intermediate mlp: {}".format(self.use_initial_mlp, self.use_intermediate_mlp))
 
        self.nx_rnn1 = self.nx_rnn1 + self.nh_mem
        self.nh_rnn1 = self.nneur[0]
        
        print("nx rnn1", self.nx_rnn1, "nh rnn1", self.nh_rnn1)
        print("nx rnn2", self.nx_rnn2, "nh rnn2", self.nh_rnn2)  
        if self.use_third_rnn: print("nx rnn3", self.nx_rnn3, "nh rnn3", self.nh_rnn3) 
        print("nx sfc", self.nx_sfc)

        if self.use_initial_mlp:
            self.mlp_initial = nn.Linear(nx, self.nneur[0])

        self.mlp_surface1  = nn.Linear(nx_sfc, self.nh_rnn1)
        self.mlp_surface2  = nn.Linear(nx_sfc, self.nh_rnn1)

        # self.rnn1      = nn.LSTMCell(self.nx_rnn1, self.nh_rnn1)  # (input_size, hidden_size)
        # self.rnn2      = nn.LSTMCell(self.nx_rnn2, self.nh_rnn2)
        if self.use_third_rnn:
            self.mlp_toa1  = nn.Linear(1, self.nh_rnn1)
            self.mlp_toa2  = nn.Linear(1, self.nh_rnn1)
            
            self.rnn0   = nn.LSTM(self.nx_rnn1, self.nh_rnn1,  batch_first=True)
            self.rnn1   = nn.LSTM(self.nx_rnn2, self.nh_rnn2,  batch_first=True)  # (input_size, hidden_size)
            self.rnn2   = nn.LSTM(self.nx_rnn3, self.nh_rnn3,  batch_first=True)
        else:

            self.mlp_toa1  = nn.Linear(1, self.nh_rnn2)
            self.mlp_toa2  = nn.Linear(1, self.nh_rnn2)

            self.rnn1      = nn.LSTM(self.nx_rnn1, self.nh_rnn1,  batch_first=True)  # (input_size, hidden_size)
            self.rnn2      = nn.LSTM(self.nx_rnn2, self.nh_rnn2,  batch_first=True)

        if self.add_stochastic_layer:
            from models_torch_kernels import StochasticGRUCell, MyStochasticGRUCell
            nx_srnn = self.nh_rnn2
            # self.rnn_stochastic = StochasticGRUCell(nx_srnn, self.nh_rnn2)  # (input_size, hidden_size)
            self.rnn_stochastic = MyStochasticGRUCell(nx_srnn, self.nh_rnn2)  # (input_size, hidden_size)
        nh_rnn = self.nh_rnn2

        if self.use_intermediate_mlp: 
            self.mlp_latent = nn.Linear(nh_rnn, self.nh_mem)
            self.mlp_output = nn.Linear(self.nh_mem, self.ny)
        else:
            self.mlp_output = nn.Linear(nh_rnn, self.ny)
            
        self.mlp_surface_output = nn.Linear(nneur[-1], self.ny_sfc)
            
        if self.separate_radiation:
            # self.nh_rnn1_rad = self.nh_rnn1 
            # self.nh_rnn2_rad = self.nh_rnn2 
            self.nh_rnn1_rad = 96 
            self.nh_rnn2_rad = 96
            self.rnn1_rad      = nn.GRU(self.nh_mem+self.nx_rad, self.nh_rnn1_rad,  batch_first=True)   # (input_size, hidden_size)
            self.rnn2_rad      = nn.GRU(self.nh_rnn1_rad, self.nh_rnn2_rad,  batch_first=True) 
            self.mlp_surface_rad = nn.Linear(self.nx_sfc_rad, self.nh_rnn1_rad)
            self.mlp_surface_output_rad = nn.Linear(self.nh_rnn2_rad, self.ny_sfc_rad)
            self.mlp_output_rad = nn.Linear(self.nh_rnn2_rad, self.ny_rad)
    # def reset_states(self):
    #     self.rnn1_mem = None

    # def detach_states(self):
    #     self.rnn1_mem = self.rnn1_mem.detach()
   
    # def get_states(self):
    #     return self.rnn1_mem.detach()

    # def set_states(self, states):
    #     self.rnn1_mem = states 
        
    def temperature_scaling(self, T_raw):
        # T_denorm = T = T*(self.xmax_lev[:,0] - self.xmin_lev[:,0]) + self.xmean_lev[:,0]
        # T_denorm = T*(self.xcoeff_lev[2,:,0] - self.xcoeff_lev[1,:,0]) + self.xcoeff_lev[0,:,0]
        # liquid_ratio = (T_raw - 253.16) / 20.0 
        liquid_ratio = (T_raw - 253.16) * 0.05 
        liquid_ratio = F.hardtanh(liquid_ratio, 0.0, 1.0)
        return liquid_ratio
        
    def pp_mp(self, out, out_sfc, x_denorm):

        out_denorm      = out / self.yscale_lev.to(device=out.device)
        
        T_before        = x_denorm[:,:,0:1]
        qliq_before     = x_denorm[:,:,2:3]
        qice_before     = x_denorm[:,:,3:4]   
        qn_before       = qliq_before + qice_before 

        # print("shape x denorm", x_denorm.shape, "T", T_before.shape)
        T_new           = T_before  + out_denorm[:,:,0:1]*1200

        # T_new           = T_before  + out_denorm[:,:,0:1]*1200
        liq_frac_constrained    = self.temperature_scaling(T_new)

        #                            dqn
        qn_new      = qn_before + out_denorm[:,:,2:3]*1200  
        qliq_new    = liq_frac_constrained*qn_new
        qice_new    = (1-liq_frac_constrained)*qn_new
        dqliq       = (qliq_new - qliq_before) * 0.0008333333333333334 #/1200  
        dqice       = (qice_new - qice_before) * 0.0008333333333333334 #/1200   
        out_denorm  = torch.cat((out_denorm[:,:,0:2], dqliq, dqice, out_denorm[:,:,3:]),dim=2)
        
        # print("Tbef {:.2f}   T {:.2f}  dt {:.2e}  liqfrac {:.2f}   dqn {:.2e}  qbef {:.2e}  qnew {:.2e}  dqliq {:.2e}  dqice {:.2e} ".format( 
        #                                                 # x_denorm[200,35,4].item(),
        #                                                 T_before[200,35].item(), 
        #                                                 T_new[200,35].item(), 
        #                                                 (out_denorm[200,35,0:1]*1200).item(),
        #                                                 liq_frac_constrained[200,35].item(), 
        #                                                 (out_denorm[200,35,2]*1200).item(), 
        #                                                 qn_before[200,35].item(),
        #                                                 qn_new[200,35].item(),
        #                                                 dqliq[200,35].item(),
        #                                                 dqice[200,35].item()))

        
        out_sfc_denorm  = out_sfc / self.yscale_sca.to(device=out.device)
        
        return out_denorm, out_sfc_denorm
    
    def forward(self, inputs_main, inputs_aux, rnn1_mem):
        if self.ensemble_size>0:
            inputs_main = inputs_main.unsqueeze(0)
            inputs_aux = inputs_aux.unsqueeze(0)
            inputs_main = torch.repeat_interleave(inputs_main,repeats=self.ensemble_size,dim=0)
            inputs_aux = torch.repeat_interleave(inputs_aux,repeats=self.ensemble_size,dim=0)
            inputs_main = inputs_main.flatten(0,1)
            inputs_aux = inputs_aux.flatten(0,1)
                    
        batch_size = inputs_main.shape[0]
        # print("shape inputs main", inputs_main.shape)
        if self.add_pres:
            sp = torch.unsqueeze(inputs_aux[:,0:1],1)
            # undo scaling
            sp = sp*35451.17 + 98623.664
            pres  = self.preslay(sp)
            inputs_main = torch.cat((inputs_main,pres),dim=2)
            
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # if self.use_memory and self.rnn1_mem is None: 
        #     self.rnn1_mem = torch.randn(batch_size, self.nlay, self.nh_mem,device=inputs_main.device)
        #     # self.rnn1_mem = torch.randn((batch_size, self.nlay, self.nh_mem),dtype=self.dtype,device=device)

        
        if self.separate_radiation:
            # Do not use inputs -2,-3,-4 (O3, CH4, N2O) or first 10 levels
            inputs_main_crm = torch.cat((inputs_main[:,10:,0:-4], inputs_main[:,10:,-1:]),dim=2)
        else:
            inputs_main_crm = inputs_main
            
        if self.use_initial_mlp:
            inputs_main_crm = self.mlp_initial(inputs_main_crm)
            inputs_main_crm = self.nonlin(inputs_main_crm)  
            
        if self.use_memory:
            # rnn1_input = torch.cat((rnn1_input,self.rnn1_mem), axis=2)
            inputs_main_crm = torch.cat((inputs_main_crm,rnn1_mem), dim=2)
            
        if self.use_third_rnn: # use initial downward RNN
            inputs_toa = inputs_aux[:,1:2] # only pbuf_SOLIN
            cx0 = self.mlp_toa1(inputs_toa)
            # cx0 = self.nonlin(cx0)
            hx0 = self.mlp_toa2(inputs_toa)
            # hx0 = self.nonlin(hx0)
            hidden0 = (torch.unsqueeze(hx0,0), torch.unsqueeze(cx0,0))  
            rnn0out, states = self.rnn0(inputs_main_crm, hidden0)
            
            rnn1_input =  torch.flip(rnn0out, [1])
        else:
            # TOA is first in memory, so to start at the surface we need to go backwards
            rnn1_input = torch.flip(inputs_main_crm, [1])
        
        # The input (a vertical sequence) is concatenated with the
        # output of the RNN from the previous time step 

        if self.separate_radiation:
            inputs_sfc = torch.cat((inputs_aux[:,0:5],inputs_aux[:,11:]),dim=1)
        else:
            inputs_sfc = inputs_aux
        hx = self.mlp_surface1(inputs_sfc)
        hx = self.nonlin(hx)
        cx = self.mlp_surface2(inputs_sfc)
        cx = self.nonlin(cx)
        hidden = (torch.unsqueeze(hx,0), torch.unsqueeze(cx,0))

        # if self.use_initial_mlp:
        #     rnn1_input = self.mlp_initial(rnn1_input)
        #     rnn1_input = self.nonlin(rnn1_input)


        # print("shape rnn1 inp", rnn1_input.shape, "shape rnn1mem", self.rnn1_mem.shape)     
        # if self.use_memory:
        #     # rnn1_input = torch.cat((rnn1_input,self.rnn1_mem), axis=2)
        #     rnn1_input = torch.cat((rnn1_input,rnn1_mem), dim=2)

        rnn1out, states = self.rnn1(rnn1_input, hidden)
        
        # if self.predict_flux:
        #     rnn1out = torch.cat((torch.unsqueeze(hx,1),rnn1out),dim=1)

        rnn1out = torch.flip(rnn1out, [1])

        if self.use_third_rnn:
          hx2 = torch.randn((batch_size, self.nh_rnn2),device=inputs_main.device)  # (batch, hidden_size)
          cx2 = torch.randn((batch_size, self.nh_rnn2),device=inputs_main.device)
        else: 
          inputs_toa = inputs_aux[:,1:2] # only pbuf_SOLIN
          hx2 = self.mlp_toa1(inputs_toa)
          cx2 = self.mlp_toa2(inputs_toa)
        hidden2 = (torch.unsqueeze(hx2,0), torch.unsqueeze(cx2,0))

        input_rnn2 = rnn1out
            
        rnn2out, states = self.rnn2(input_rnn2, hidden2)

        (last_h, last_c) = states
        
        if self.use_intermediate_mlp: 
            rnn2out = self.mlp_latent(rnn2out)
          
        if self.use_memory:
            # self.rnn1_mem = torch.flip(rnn2out, [1])
            # if self.predict_flux:
            #     rnn1_mem = torch.flip(rnn2out[:,1:,:], [1])
            # else:
            #     rnn1_mem = torch.flip(rnn2out, [1])
            if self.use_third_rnn:
                rnn1_mem = rnn2out
            else:
                rnn1_mem = torch.flip(rnn2out, [1])
            
        # Add a stochastic perturbation
        # Convective memory is still based on the deterministic model,
        # and does not include the stochastic perturbation
        # concat and use_intermediate_mlp should be set to false
        if self.add_stochastic_layer:
            # srnn_input = torch.transpose(rnn2out,0,1)
            # srnn_input = torch.transpose(self.rnn1_mem,0,1)
            srnn_input = torch.transpose(rnn1_mem,0,1)

            # transpose is needed because this layer assumes seq. dim first
            z = self.rnn_stochastic(srnn_input)
            z = torch.flip(z, [0])

            z = torch.transpose(z,0,1)
            # z = torch.flip(z, [1])
            # rnn2out = z
            # z is a perburbation added to the hidden state
            # rnn2out = rnn2out + 0.01*z 
            rnn2out = rnn2out + self.coeff_stochastic*z 

        out = self.mlp_output(rnn2out)

        if self.output_prune and (not self.separate_radiation):
            # Only temperature tendency is computed for the top 10 levels
            # if self.separate_radiation:
            #     out[:,0:12,:] = out[:,0:12,:].clone().zero_()
            # else:
            out[:,0:12,1:] = out[:,0:12,1:].clone().zero_()
        
        out_sfc = self.mlp_surface_output(last_h.squeeze())
        
        if self.separate_radiation:
            # out_crm = out.clone()
            out_new = torch.zeros(batch_size, self.nlay_rad, self.ny, device=inputs_main.device)
            out_new[:,10:,:] = out
            # Start at surface again
            # Do not use inputs 4,5 (winds)
            inputs_main_rad =  torch.cat((inputs_main[:,:,0:4], inputs_main[:,:,6:]),dim=2)
            # # add dT from crm 
            # T_old =   inputs_main * (self.xcoeff_lev[2,:,0:1] - self.xcoeff_lev[1,:,0:1]) + self.xcoeff_lev[0,:,0:1] 
            # T_new = T_old + dT
            inputs_rad =  torch.zeros(batch_size, self.nlay_rad, self.nh_mem+self.nx_rad,device=inputs_main.device)
            inputs_rad[:,10:,0:self.nh_mem] = torch.flip(rnn2out, [1])
            inputs_rad[:,:,self.nh_mem:] = inputs_main_rad
            # inputs_rad = torch.flip(inputs_rad, [1])

            inputs_sfc_rad = inputs_aux[:,6:11]
            hx = self.mlp_surface_rad(inputs_sfc_rad)
            hidden = (torch.unsqueeze(hx,0))
            rnn_out, states = self.rnn1_rad(inputs_rad, hidden)
            rnn_out = torch.flip(rnn_out, [1])

            inputs_toa = inputs_aux[:,1:2] # only pbuf_SOLIN
            hx2 = self.mlp_toa_rad(inputs_toa)

            rnn_out, last_h = self.rnn2_rad(rnn_out, hidden)
            out_rad = self.mlp_output_rad(rnn_out)

            out_sfc_rad = self.mlp_surface_output_rad(last_h)
            # dT_tot = dT_crm + dT_rad
            out_new[:,:,0:1] = out_new[:,:,0:1] + out_rad
            out = out_new
            #1D (scalar) Output variables: ['cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 
            #'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']
            # print("shape 1", out_sfc[:,0:1].shape, "2", out_sfc_rad.shape)
            # rad predicts everything except PRECSC, PRECC
            out_sfc =  torch.cat((out_sfc_rad[:,0:2], out_sfc, out_sfc_rad[:,2:]),dim=1)
            
        out_sfc = self.relu(out_sfc)
        # if self.predict_flux:
        #     gcp = 9.80665 / 1004
        #     preslev  = self.preslev(sp)
        #     pres_diff = preslev[:,1:] - preslev[:,0:-1]
        #     flux_diff = out[:,1:] - out[:,0:-1]
        #     out = gcp * flux_diff / pres_diff
        return out, out_sfc, rnn1_mem


class LSTM_torchscript(nn.Module):
    use_initial_mlp: Final[bool]
    use_intermediate_mlp: Final[bool]
    add_pres: Final[bool]
    add_stochastic_layer: Final[bool]
    output_prune: Final[bool]
    ensemble_size: Final[int]
    def __init__(self, hyam, hybm, 
                out_scale, out_sfc_scale, 
                nlay=60, nx = 4, nx_sfc=3, ny = 4, ny_sfc=3, nneur=(64,64), 
                use_initial_mlp=False, 
                use_intermediate_mlp=True,
                add_pres=False,
                add_stochastic_layer=False,
                output_prune=False,
                ensemble_size=1,
                coeff_stochastic = 0.0,
                nh_mem=16):
        super(LSTM_torchscript, self).__init__()
        self.ny = ny 
        self.nlay = nlay 
        self.nx_sfc = nx_sfc 
        self.ny_sfc = ny_sfc
        self.nneur = nneur 
        self.use_initial_mlp=use_initial_mlp
        self.output_prune = output_prune
        self.add_pres = add_pres
        if self.add_pres:
            self.preslay = LayerPressure(hyam,hybm)
            nx = nx +1
        self.nx = nx
        self.nh_rnn1 = self.nneur[0]
        self.nx_rnn2 = self.nneur[0]
        self.nh_rnn2 = self.nneur[1]
        self.ensemble_size = ensemble_size
        self.add_stochastic_layer = add_stochastic_layer
        self.coeff_stochastic = coeff_stochastic
        self.nonlin = nn.Tanh()
        self.relu = nn.ReLU()
        
        self.yscale_lev = torch.from_numpy(out_scale)
        self.yscale_sca = torch.from_numpy(out_sfc_scale)
            
        # in ClimSim config of E3SM, the CRM physics first computes 
        # moist physics on 50 levels, and then computes radiation on 60 levels!
        self.use_intermediate_mlp=use_intermediate_mlp
        
        if self.use_initial_mlp:
            self.nx_rnn1 = self.nneur[0]
        else:
            self.nx_rnn1 = nx
            
        print("Building RNN without convective memory")

        print("Initial mlp: {}, intermediate mlp: {}".format(self.use_initial_mlp, self.use_intermediate_mlp))
 
        self.nh_rnn1 = self.nneur[0]
        
        print("nx rnn1", self.nx_rnn1, "nh rnn1", self.nh_rnn1)
        print("nx rnn2", self.nx_rnn2, "nh rnn2", self.nh_rnn2)  
        print("nx sfc", self.nx_sfc)

        if self.use_initial_mlp:
            self.mlp_initial = nn.Linear(nx, self.nneur[0])

        self.mlp_surface1  = nn.Linear(nx_sfc, self.nh_rnn1)
        self.mlp_surface2  = nn.Linear(nx_sfc, self.nh_rnn1)

        # self.rnn1      = nn.LSTMCell(self.nx_rnn1, self.nh_rnn1)  # (input_size, hidden_size)
        # self.rnn2      = nn.LSTMCell(self.nx_rnn2, self.nh_rnn2)
        self.rnn1      = nn.LSTM(self.nx_rnn1, self.nh_rnn1,  batch_first=True)  # (input_size, hidden_size)
        self.rnn2      = nn.LSTM(self.nx_rnn2, self.nh_rnn2,  batch_first=True)

        if self.add_stochastic_layer:
            from models_torch_kernels import StochasticGRUCell, MyStochasticGRUCell
            nx_srnn = self.nh_rnn2
            # self.rnn_stochastic = StochasticGRUCell(nx_srnn, self.nh_rnn2)  # (input_size, hidden_size)
            self.rnn_stochastic = MyStochasticGRUCell(nx_srnn, self.nh_rnn2)  # (input_size, hidden_size)
        nh_rnn = self.nh_rnn2

        if self.use_intermediate_mlp: 
            self.mlp_latent = nn.Linear(nh_rnn, self.nh_rnn2)
            self.mlp_output = nn.Linear(self.nh_rnn2, self.ny)
        else:
            self.mlp_output = nn.Linear(nh_rnn, self.ny)
            
        self.mlp_surface_output = nn.Linear(nneur[-1], self.ny_sfc)
            

    def temperature_scaling(self, T_raw):
        # T_denorm = T = T*(self.xmax_lev[:,0] - self.xmin_lev[:,0]) + self.xmean_lev[:,0]
        # T_denorm = T*(self.xcoeff_lev[2,:,0] - self.xcoeff_lev[1,:,0]) + self.xcoeff_lev[0,:,0]
        # liquid_ratio = (T_raw - 253.16) / 20.0 
        liquid_ratio = (T_raw - 253.16) * 0.05 
        liquid_ratio = F.hardtanh(liquid_ratio, 0.0, 1.0)
        return liquid_ratio
        
    def pp_mp(self, out, out_sfc, x_denorm):

        out_denorm      = out / self.yscale_lev.to(device=out.device)
        
        T_before        = x_denorm[:,:,0:1]
        qliq_before     = x_denorm[:,:,2:3]
        qice_before     = x_denorm[:,:,3:4]   
        qn_before       = qliq_before + qice_before 

        # print("shape x denorm", x_denorm.shape, "T", T_before.shape)
        T_new           = T_before  + out_denorm[:,:,0:1]*1200

        # T_new           = T_before  + out_denorm[:,:,0:1]*1200
        liq_frac_constrained    = self.temperature_scaling(T_new)

        #                            dqn
        qn_new      = qn_before + out_denorm[:,:,2:3]*1200  
        qliq_new    = liq_frac_constrained*qn_new
        qice_new    = (1-liq_frac_constrained)*qn_new
        dqliq       = (qliq_new - qliq_before) * 0.0008333333333333334 #/1200  
        dqice       = (qice_new - qice_before) * 0.0008333333333333334 #/1200   
        out_denorm  = torch.cat((out_denorm[:,:,0:2], dqliq, dqice, out_denorm[:,:,3:]),dim=2)
        
        # print("Tbef {:.2f}   T {:.2f}  dt {:.2e}  liqfrac {:.2f}   dqn {:.2e}  qbef {:.2e}  qnew {:.2e}  dqliq {:.2e}  dqice {:.2e} ".format( 
        #                                                 # x_denorm[200,35,4].item(),
        #                                                 T_before[200,35].item(), 
        #                                                 T_new[200,35].item(), 
        #                                                 (out_denorm[200,35,0:1]*1200).item(),
        #                                                 liq_frac_constrained[200,35].item(), 
        #                                                 (out_denorm[200,35,2]*1200).item(), 
        #                                                 qn_before[200,35].item(),
        #                                                 qn_new[200,35].item(),
        #                                                 dqliq[200,35].item(),
        #                                                 dqice[200,35].item()))

        
        out_sfc_denorm  = out_sfc / self.yscale_sca.to(device=out.device)
        
        return out_denorm, out_sfc_denorm
    
    def forward(self, inputs_main, inputs_aux):
        if self.ensemble_size>0:
            inputs_main = inputs_main.unsqueeze(0)
            inputs_aux = inputs_aux.unsqueeze(0)
            inputs_main = torch.repeat_interleave(inputs_main,repeats=self.ensemble_size,dim=0)
            inputs_aux = torch.repeat_interleave(inputs_aux,repeats=self.ensemble_size,dim=0)
            inputs_main = inputs_main.flatten(0,1)
            inputs_aux = inputs_aux.flatten(0,1)
                    
        batch_size = inputs_main.shape[0]
        # print("shape inputs main", inputs_main.shape)
        if self.add_pres:
            sp = torch.unsqueeze(inputs_aux[:,0:1],1)
            # undo scaling
            sp = sp*35451.17 + 98623.664
            pres  = self.preslay(sp)
            inputs_main = torch.cat((inputs_main,pres),dim=2)
            
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        hx = self.mlp_surface1(inputs_aux)
        hx = self.nonlin(hx)

        # TOA is first in memory, so to start at the surface we need to go backwards
        inputs_main = torch.flip(inputs_main, [1])
        
        # The input (a vertical sequence) is concatenated with the
        # output of the RNN from the previous time step 
        cx = self.mlp_surface2(inputs_aux)
        cx = self.nonlin(cx)
        hidden = (torch.unsqueeze(hx,0), torch.unsqueeze(cx,0))

        if self.use_initial_mlp:
            rnn1_input = self.mlp_initial(inputs_main)
            rnn1_input = self.nonlin(rnn1_input)
    
        else:
            rnn1_input = inputs_main

        rnn1out, states = self.rnn1(rnn1_input, hidden)

        rnn1out = torch.flip(rnn1out, [1])

        # hx2 = torch.randn((batch_size, self.nh_rnn2),dtype=self.dtype,device=device)  # (batch, hidden_size)
        # cx2 = torch.randn((batch_size, self.nh_rnn2),dtype=self.dtype,device=device)
        hx2 = torch.randn((batch_size, self.nh_rnn2),device=inputs_main.device)  # (batch, hidden_size)
        cx2 = torch.randn((batch_size, self.nh_rnn2),device=inputs_main.device)
        
        hidden2 = (torch.unsqueeze(hx2,0), torch.unsqueeze(cx2,0))

        input_rnn2 = rnn1out
            
        rnn2out, states = self.rnn2(input_rnn2, hidden2)

        (last_h, last_c) = states
        
        if self.use_intermediate_mlp: 
            rnn2out = self.mlp_latent(rnn2out)
          
        # Add a stochastic perturbation
        # Convective memory is still based on the deterministic model,
        # and does not include the stochastic perturbation
        # concat and use_intermediate_mlp should be set to false
        if self.add_stochastic_layer:
            srnn_input = torch.transpose(rnn2out,0,1)

            # transpose is needed because this layer assumes seq. dim first
            z = self.rnn_stochastic(srnn_input)
            z = torch.flip(z, [0])

            z = torch.transpose(z,0,1)
            # z = torch.flip(z, [1])
            # rnn2out = z
            # z is a perburbation added to the hidden state
            # rnn2out = rnn2out + 0.01*z 
            rnn2out = rnn2out + self.coeff_stochastic*z 

        out = self.mlp_output(rnn2out)

        if self.output_prune:
            # Only temperature tendency is computed for the top 10 levels
            # if self.separate_radiation:
            #     out[:,0:12,:] = out[:,0:12,:].clone().zero_()
            # else:
            out[:,0:12,1:] = out[:,0:12,1:].clone().zero_()
        
        out_sfc = self.mlp_surface_output(last_h.squeeze())
        out_sfc = self.relu(out_sfc)

        return out, out_sfc


class SpaceStateModel(nn.Module):
    def __init__(self, hyam, hybm, 
                out_scale, out_sfc_scale,  
                nlay=30, nx = 4, nx_sfc=3, ny = 4,  ny_sfc=1,
                nneur=(64,64),model_type='LRU', 
                use_initial_mlp=True, add_pres=False,  concat=False,
                device=None):
        super(SpaceStateModel, self).__init__()
        self.nx = nx
        self.ny = ny 
        self.nlay = nlay 
        self.nx_sfc = nx_sfc 
        self.ny_sfc = ny_sfc
        self.nneur = nneur 
        self.add_pres = add_pres
        if self.add_pres:
            self.preslay = LayerPressure(hyam,hybm)
            nx = nx +1
        self.nh_rnn1 = self.nneur[0]
        self.nh_rnn2 = self.nneur[1]
        self.ny_rnn1 = self.nh_rnn1
        self.use_initial_mlp=use_initial_mlp
        self.model_type=model_type
        self.device=device
        self.yscale_lev = torch.from_numpy(out_scale)
        self.yscale_sca = torch.from_numpy(out_sfc_scale)
            
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using model type {}".format(model_type))
        if self.model_type in ["Mamba","GSS"]:#,"QRNN"]:  #Mamba doesnt support providing the state
            print("WARNING: this SSM type doesn't support providing the state, so the surface variables are instead added to each point in the sequence")
            
            # concatenate the vertical (sequence) inputs with tiled scalars
            self.use_initial_mlp=True
            nx = nx + nx_sfc
        if self.use_initial_mlp: 
            self.nh_mlp1 = self.nh_rnn1
            self.nx_rnn1 = self.nh_rnn1
        else:
            self.nx_rnn1 = nx
        self.use_intermediate_mlp = False
        # if memory == 'None':
        #     print("Building non-autoregressive SSM, but may be stateful")
        #     self.autoregressive = False
        # elif memory == 'Hidden':
        print("Building autoregressive SSM that feeds a hidden memory at t0,z0 to SSM1 at t1,z0")
        self.autoregressive = True
        self.rnn1_mem = None
        # self.rnn2_mem = None
        # self.use_intermediate_mlp = True
            
        # if self.use_intermediate_mlp:
        #     self.nh_latent = self.nh_rnn2
        self.nh_latent = self.nh_rnn2
        glu_expand_factor = 2
        # if model_type in ['S5','GSS','QRNN','Mamba','GateLoop','SRU','LRU']:
        if model_type in ['S5','GSS','QRNN','Mamba','GateLoop','SRU','SRU_2D','LRU']:
            self.use_initial_mlp = True
            self.use_intermediate_mlp = True; self.nh_latent = self.nh_rnn2
            self.nx_rnn1 = self.nh_rnn1 
            self.nx_rnn2 = self.nh_rnn2
            if self.autoregressive and model_type != 'SRU_2D':
                # This is a bit tricky with these SSMs as they expect 
                # num_inputs = num_hidden. Therefore we need MLPs that halve 
                # the size of both inputs and the final hidden variable, e.g.
                # SSM = 128 neurons, Xlay = 64 neurons, Xhidden = 64 neurons,
                # X = concat(Xlay,Xhidden)
                # Similarly, to feed the latent variable to RNN2, we would need an MLP before it
                # that halves the size of  the output from RNN1
                #    mlp1      concat(mlp1,hfin=64)     RNN1(128)     MLP2     concat(mlp2,hfin=64)      RNN2(128)     mlpfin
                # nx ----> 64 ---------------------> 128 -------> 128 ---> 64 ---------------------> 128 --------> 128 ----->  64
                #   OR let second SSMs hidden size be bigger
                #    mlp1      concat(mlp1,hfin=64)     RNN1(128)      concat(hfin=64)           RNN2(192)     mlpfin
                # nx ----> 64 ---------------------> 128 -------> 128 ---------------------> 192 --------> 192 ----->  64  
                # or Dont bother making RNN2 autoregressive?
                # mlp1        concat(hfin=128)          RNN1(256)     GLU      RNN2(128)    GLU      RNN3
                # nx ---> 128 ---------------------> 256 -------> 256 ---> 128 -------> 128 ---> 128 --->  128

                self.nh_latent = self.nh_rnn1//2
                self.ny_rnn1 = self.nh_rnn1//2
                self.nh_mlp1  = self.nh_rnn1//2 
                self.nh_mlp2  = self.nh_rnn1
                # self.nh_rnn2 = self.nh_rnn1  + self.nh_latent
                # self.nx_rnn2 = self.nh_rnn2
                # self.nh_rnn2 = self.nh_rnn1 
                self.nx_rnn2 = self.nh_rnn2
                glu_expand_factor = 1
        else:
            self.nx_rnn2 = self.nh_rnn1
            if self.autoregressive:
                self.nx_rnn1 = self.nx_rnn1 + self.nh_latent 
                # self.nx_rnn2 = self.nh_rnn1 + self.nh_latent 

        if self.use_initial_mlp:
            self.mlp_initial = nn.Linear(nx, self.nh_mlp1 )

        print("nx rnn1", self.nx_rnn1, "nh rnn1", self.nh_rnn1, "ny rnn1", self.ny_rnn1)
        print("nx rnn2", self.nx_rnn2, "nh rnn2", self.nh_rnn2)  
        if self.autoregressive: 
            print("nh_memory", self.nh_latent)
        self.concat = concat

        self.mlp_surface1  = nn.Linear(nx_sfc, self.nh_rnn1)

        if model_type == 'LRU':
            from models_torch_kernels_LRU import LRU
            self.rnn1= LRU(in_features=self.nx_rnn1,out_features=self.nh_rnn1,state_features=self.nh_rnn1)
            self.rnn2= LRU(in_features=self.nx_rnn2,out_features=self.nh_rnn2,state_features=self.nh_rnn2)
        elif model_type == "MinGRU":
            # MinGRU = models_torch_kernels.MinGRU
            from models_torch_kernels import MinGRU as MinGRU
            self.rnn1= MinGRU(self.nx_rnn1,self.nh_rnn1)
            self.rnn2= MinGRU(self.nx_rnn2,self.nh_rnn2)
            # from models_torch_kernels import minGRU as MinGRU
            # self.rnn1= MinGRU(self.nh_rnn1)
            # self.rnn2= MinGRU(self.nh_rnn2)
        elif model_type == 'S5':
            from s5 import S5
            self.liquid=False
            self.pres_step_scale = False
            # S5 outputs shape equals inputs?
            # self.rnn1= S5(self.nx_rnn1,self.nh_rnn1,liquid=self.liquid)
            # self.rnn2= S5(self.nx_rnn2,self.nh_rnn2,liquid=self.liquid)   
            # self.rnn1= S5(self.nh_rnn1,liquid=self.liquid)
            # self.rnn2= S5(self.nh_rnn2,liquid=self.liquid)   
            self.rnn1= S5(self.nx_rnn1,liquid=self.liquid)
            self.rnn2= S5(self.nx_rnn2,liquid=self.liquid)  
        elif model_type == 'Mamba':
            from mamba_ssm import Mamba
            self.rnn1 = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=self.nx_rnn1, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
                )
            self.rnn2 = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=self.nx_rnn2, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
                )
        elif model_type == 'GSS':
            from gated_state_spaces_pytorch import GSS
            self.rnn1= GSS(dim=self.nx_rnn1,dss_kernel_N=self.nh_rnn1,dss_kernel_H=self.nh_rnn1)
            self.rnn2= GSS(dim=self.nx_rnn2,dss_kernel_N=self.nh_rnn2,dss_kernel_H=self.nh_rnn2)
        elif model_type == 'QRNN':
            from models_torch_kernels import QRNNLayer, QRNNLayer_noncausal
            kernelsize = 3
            self.rnn1= QRNNLayer_noncausal(self.nx_rnn1,self.nh_rnn1, kernel_size=3, pad =(0,1))
            self.rnn2= QRNNLayer_noncausal(self.nx_rnn2,self.nh_rnn2, kernel_size=3) 
            
            self.mlp_surface2  = nn.Linear(nx_sfc, self.nh_rnn1)  
        elif model_type == 'SRU':
            from models_torch_kernels import SRU
            self.rnn1= SRU(self.nx_rnn1,self.nh_rnn1)
            self.rnn2= SRU(self.nx_rnn2,self.nh_rnn2)  
            # from sru import SRU
            # self.rnn1= SRU(self.nx_rnn1,self.nh_rnn1,num_layers=1)
            # self.rnn2= SRU(self.nx_rnn2,self.nh_rnn2,num_layers=1)
        elif model_type == "SRU_2D":
            from models_torch_kernels import SRU, SRU2
            self.rnn1= SRU2(self.nx_rnn1,self.nh_rnn1)
            self.rnn2= SRU(self.nx_rnn2,self.nh_rnn2)  
        elif model_type == 'GateLoop':
            from gateloop_transformer import SimpleGateLoopLayer
            self.rnn1= SimpleGateLoopLayer(self.nx_rnn1) #, use_jax_associative_scan=True)
            self.rnn2= SimpleGateLoopLayer(self.nx_rnn2) #, use_jax_associative_scan=True)
            self.mlp_surface2  = nn.Linear(nx_sfc, self.nh_rnn1)
        else:
            
            raise NotImplementedError()

        # if model_type in ['LRU','S5','GSS','QRNN','Mamba']:
        # if model_type in ['LRU','SRU','S5','GSS','QRNN','GateLoop','MinGRU']:
        # if model_type in ['LRU','SRU','S5','GSS','QRNN','GateLoop','MinGRU','Mamba']:
        if model_type in ['LRU','SRU','SRU_2D','S5','QRNN','GateLoop','MinGRU',]:

            self.use_glu_layers = True 
        else:
            self.use_glu_layers = False
            
        self.reduce_dim_with_mlp=False
        if self.use_glu_layers:
            print("Using GLU layers in between for nonlinearity, expand factor (1 means halving output) for GLU1 is ", glu_expand_factor)
            glu_layernorm=True
            # glu_layernorm=False

            # self.rnn1= LRU(in_features=nx_rnn1,out_features=self.nneur[0],state_features=self.nneur[0])
            # self.mlp  = nn.Linear(self.nneur[0], self.nneur[0])
            # self.SeqLayer1 = SequenceLayer(nlay=self.nlay,nneur=self.nneur[0],layernorm=True)
            self.SeqLayer1 = GLU(nlay=self.nlay,nneur=self.nh_rnn1,layernorm=glu_layernorm, expand_factor=glu_expand_factor)
    
            # self.rnn2= LRU(in_features=nx_rnn2,out_features=self.nneur[1],state_features=self.nneur[0])
            # self.mlp2  = nn.Linear(self.nneur[1], self.nneur[1])
            # self.SeqLayer2 = SequenceLayer(nlay=self.nlay,nneur=self.nneur[1],layernorm=True)
            self.SeqLayer2 = GLU(nlay=self.nlay,nneur=self.nh_rnn2,layernorm=glu_layernorm)

            # self.SeqLayer15 = GLU(nlay=self.nlay,nneur=self.nneur[0],layernorm=glu_layernorm)
        else:
            if self.autoregressive and model_type in ['S5','GSS','QRNN','Mamba','GateLoop','SRU','SRU_2D''LRU']:
                   self.reduce_dim_with_mlp=True
                   self.reduce_dim_mlp = nn.Linear(self.nh_rnn1, self.nx_rnn2)

        if self.concat:
            nx_last = self.ny_rnn1 + self.nh_rnn2
        else:
            nx_last = self.nh_rnn2
            
        if self.use_intermediate_mlp:
            self.mlp_latent = nn.Linear(nx_last, self.nh_latent)
            nx_last = self.nh_latent

        self.mlp_output        = nn.Linear(nx_last, self.ny)
        self.mlp_surface_output = nn.Linear(nneur[-1], self.ny_sfc)

    def reset_states(self):
        if self.autoregressive:
            self.rnn1_mem = None
            # self.rnn2_mem = None

    def detach_states(self):
        if self.autoregressive:
            self.rnn1_mem = self.rnn1_mem.detach()
            # self.rnn2_mem = self.rnn2_mem.detach()
        
    def temperature_scaling(self, T_raw):
        # T_denorm = T = T*(self.xmax_lev[:,0] - self.xmin_lev[:,0]) + self.xmean_lev[:,0]
        # T_denorm = T*(self.xcoeff_lev[2,:,0] - self.xcoeff_lev[1,:,0]) + self.xcoeff_lev[0,:,0]
        # liquid_ratio = (T_raw - 253.16) / 20.0 
        liquid_ratio = (T_raw - 253.16) * 0.05 
        liquid_ratio = F.hardtanh(liquid_ratio, 0.0, 1.0)
        return liquid_ratio
        
    def pp_mp(self, out, out_sfc, x_denorm):

        out_denorm      = out / self.yscale_lev.to(device=out.device)
        
        T_before        = x_denorm[:,:,0:1]
        qliq_before     = x_denorm[:,:,2:3]
        qice_before     = x_denorm[:,:,3:4]   
        qn_before       = qliq_before + qice_before 

        # print("shape x denorm", x_denorm.shape, "T", T_before.shape)
        T_new           = T_before  + out_denorm[:,:,0:1]*1200

        # T_new           = T_before  + out_denorm[:,:,0:1]*1200
        liq_frac_constrained    = self.temperature_scaling(T_new)

        #                            dqn
        qn_new      = qn_before + out_denorm[:,:,2:3]*1200  
        qliq_new    = liq_frac_constrained*qn_new
        qice_new    = (1-liq_frac_constrained)*qn_new
        dqliq       = (qliq_new - qliq_before) * 0.0008333333333333334 #/1200  
        dqice       = (qice_new - qice_before) * 0.0008333333333333334 #/1200   
        out_denorm  = torch.cat((out_denorm[:,:,0:2], dqliq, dqice, out_denorm[:,:,3:]),dim=2)
        
        out_sfc_denorm  = out_sfc / self.yscale_sca.to(device=out.device)
        
        return out_denorm, out_sfc_denorm
    
    def forward(self, inputs_main, inputs_aux):

        # print("Shape inputs main", inputs_main.shape)
        batch_size = inputs_main.shape[0]
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device=self.device

        if self.add_pres:
            sp = torch.unsqueeze(inputs_aux[:,0:1],1)
            # undo scaling
            sp = sp*35451.17 + 98623.664
            pres  = self.preslay(sp)
            inputs_main = torch.cat((inputs_main,pres),dim=2)
            
        if self.model_type=="S5" and self.pres_step_scale:
                refp = inputs_main[:,:,-1].clone().detach()
        
        inputs_main = torch.flip(inputs_main, [1])
        
        if self.model_type=="S5" and self.pres_step_scale:
                refp_rev = inputs_main[:,:,-1].clone().detach()

        if self.model_type in ["Mamba","GSS"]:#,,"QRNN"]:
            #Mamba doesnt support providing the state, so as a hack we instead
            # concatenate the vertical (sequence) inputs with tiled scalars
            inputs_aux_tiled = torch.tile(torch.unsqueeze(inputs_aux,1),(1,self.nlay,1))
            inputs_main = torch.cat((inputs_main,inputs_aux_tiled), dim=2)
        else:
            init_inputs = inputs_aux
                
            init_states = self.mlp_surface1(init_inputs)
            # init_states = nn.Softsign()(init_states)
            init_states = nn.Tanh()(init_states)
            # print("shape init states" , init_states.shape)
        # print("shape inp main inp", inputs_main.shape)

        if self.use_initial_mlp:
            # print("shape inp main", inputs_main.shape)
            rnn1_input = self.mlp_initial(inputs_main)
            # print("shape rnn1_input", rnn1_input.shape)

            rnn1_input = nn.Tanh()(rnn1_input)
        else:
            rnn1_input = inputs_main
        # print("shape rnn1 inp", rnn1_input.shape)

        if self.autoregressive and self.model_type != "SRU_2D":
            rnn1_input = torch.cat((rnn1_input,self.rnn1_mem), axis=2)

        # print("shape rnn1 inp", rnn1_input.shape)
            
        # B_tilde, C_tilde = self.rnn1.seq.get_BC_tilde()
        # print("shape B tild", B_tilde.shape, "C tild", C_tilde.shape)
        # shape B tild torch.Size([96, 96]) C tild torch.Size([96, 96])
        
        # if self.layernorm:
        #     rnn1_input = self.norm(rnn1_input)
        if self.model_type=="S5":
            if self.pres_step_scale:
                out = self.rnn1(rnn1_input,state=init_states, step_scale=refp_rev)
                # pres_rev = torch.flip(pres, [1])        
                # out = self.rnn1(rnn1_input,state=init_states, step_scale=pres_rev)
            else:
                out = self.rnn1(rnn1_input,state=init_states)
                # out,h = self.rnn1(rnn1_input,state=init_states,return_state=True)
                # print("OUT", out[0,-1,0], "STATE", h[0,0])
                # OUT tensor(-0.2206, device='cuda:0', grad_fn=<SelectBackward0>) STATE tensor(0.0096+0.0025j
        elif self.model_type in ["Mamba","GSS"]:#,"QRNN"]:
            out = self.rnn1(rnn1_input) 
        elif self.model_type == "QRNN":
            init_states2 = self.mlp_surface2(init_inputs)
            init_states2 = nn.Tanh()(init_states2)
            init_states = (init_states, init_states2)
            out = self.rnn1(rnn1_input,init_states) 

        elif self.model_type=="SRU":
            # print("init shape", init_states.shape)
            # init_states = init_states.view((1,batch_size,-1)) 
            out,c = self.rnn1(rnn1_input,init_states) 
        elif self.model_type=="SRU_2D":
            # print("init shape", init_states.shape)
            out,c = self.rnn1(rnn1_input,init_states, self.rnn1_mem) 
        elif self.model_type=="MinGRU":
            init_states = init_states.view((batch_size,1, -1)) 
            out = self.rnn1(rnn1_input,init_states)      
        elif self.model_type=="GateLoop":
            init_states2 = self.mlp_surface2(init_inputs)
            init_states2 = nn.Tanh()(init_states2)
            cache = [init_states.view(batch_size*self.nh_rnn1,1), init_states2.view(batch_size*self.nh_rnn1,1)]
            out = self.rnn1(rnn1_input,cache=cache)          
            # out = self.rnn1(rnn1_input)          

        else:
            # out = self.rnn1(rnn1_input,state=init_states) 
            out = self.rnn1(rnn1_input,init_states) 
        # out = self.rnn1(rnn1_input)
    
            
        init_states2 = None 
            
        out = torch.flip(out, [1])
        # print("shape rnn1 out", out.shape)

        if self.use_glu_layers: 
            out = self.SeqLayer1(out)
            # out = self.SeqLayer15(out)
        elif self.reduce_dim_with_mlp:
            out = self.reduce_dim_mlp(out)
        # if self.autoregressive:
        #     rnn2_input = torch.cat((out,self.rnn2_mem), axis=2)
        # else:
        #     rnn2_input = out 
        rnn2_input = out  
        # print("shape rnn2 inp", rnn2_input.shape)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        if self.model_type=="S5":
            if self.pres_step_scale:
                out2 = self.rnn2(rnn2_input, state=init_states2,step_scale=refp)
            else:
                out2 = self.rnn2(rnn2_input, state=init_states2)
        elif self.model_type in ["Mamba","GSS"]:#,,"QRNN"]:
            out2 = self.rnn2(rnn2_input)    
        elif self.model_type in ["SRU","SRU_2D"]:
            out2,c = self.rnn2(rnn2_input,init_states2) 
        elif self.model_type=="MinGRU":
            init_states2 = torch.randn((batch_size, 1, self.nh_rnn2),device=device)
            out2 = self.rnn2(rnn2_input,init_states2)   
        elif self.model_type=="GateLoop":
            out2 = self.rnn2(rnn2_input)      
        else:
            out2 = self.rnn2(rnn2_input,init_states2)
        # if self.layernorm:
        #     out2 = self.normrnn2(out2)                                                                                                                                                                                                                                                                                                                            
        # self.state2 = self._detach_state(state2)
        #self.state = state
        
        if self.use_glu_layers: 
            out2 = self.SeqLayer2(out2)

        if self.concat:
            outs  = torch.cat((out,out2),axis=2)
        else:
            outs = out2
            
        if self.use_intermediate_mlp:
            outs = self.mlp_latent(outs)
            
        if self.autoregressive:
        
            self.rnn1_mem = torch.flip(outs, [1])


        out_sfc = self.mlp_surface_output(outs[:,-1])
        
        outs = self.mlp_output(outs)

        return outs, out_sfc
        