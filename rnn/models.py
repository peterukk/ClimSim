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
    
    def forward(self, inputs_main, inputs_sfc):
            
        # batch_size = inputs_main.shape[0]
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
        sfc1 = self.mlp_surface1(inputs_sfc)
        sfc1 = nn.Tanh()(sfc1)

        if self.RNN_type=="LSTM":
            sfc2 = self.mlp_surface2(inputs_sfc)
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
    def __init__(self, nlay=60, nx = 4, nx_sfc=3, ny = 4, ny_sfc=3, nneur=(64,64), 
                cell_type="LSTM",
                memory="None", # "None", "Hidden", or "Output",
                concat=False,
                use_initial_mlp=False, ensemble_size=1,
                random_init_cx=False,
                use_intermediate_mlp=True,
                # add_pres=False,
                third_rnn=False,
                add_stochastic_layer=False,
                coeff_stochastic = 0.0,
                dtype=torch.float32,
                out_scale=None, out_sfc_scale=None):
        super(RNN_autoreg, self).__init__()
        self.nx = nx
        self.ny = ny 
        self.nlay = nlay 
        self.nx_sfc = nx_sfc 
        self.ny_sfc = ny_sfc
        self.nneur = nneur 
        self.use_initial_mlp=use_initial_mlp
        # self.add_pres = add_pres
        # if self.add_pres:
            # self.preslay = LayerPressure()
            # nx = nx +1
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
        self.dtype=dtype
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
        self.memory = memory
        if out_scale is not None:
            cuda = torch.cuda.is_available() 
            device = torch.device("cuda" if cuda else "cpu")
            self.yscale_lev = torch.from_numpy(out_scale).to(device)
            self.yscale_sca = torch.from_numpy(out_sfc_scale).to(device)
        if memory == 'None':
            raise NotImplementedError()
        elif memory == 'Output':
            print("Building RNN that feeds its output t0,z0 to its inputs at t1,z0")
            self.rnn1_mem = None 
            self.nh_mem = self.ny
            self.nx_rnn1 = self.nx_rnn1 + self.nh_mem
            self.nh_rnn1 = self.nneur[0]
        elif memory == 'Hidden':
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
            self.mlp_latent = nn.Linear(nh_rnn, self.nh_rnn1)
            self.mlp_output = nn.Linear(self.nh_rnn1, self.ny)
        else:
            self.mlp_output = nn.Linear(nh_rnn, self.ny)
            
        if self.ny_sfc>0:
            self.mlp_surface_output = nn.Linear(nneur[-1], self.ny_sfc)
            
    def postprocessing(self, out, out_sfc):
        out_denorm = out / self.yscale_lev
        out_sfc_denorm  = out_sfc / self.yscale_sca
        return out_denorm, out_sfc_denorm
    
    def reset_states(self):
        self.rnn1_mem = None

    def detach_states(self):
        self.rnn1_mem = self.rnn1_mem.detach()
   
    def get_states(self):
        return self.rnn1_mem.detach()

    def set_states(self, states):
        self.rnn1_mem = states 
        
    def forward(self, inputs_main, inputs_sfc):
        if self.ensemble_size>0:
            inputs_main = inputs_main.unsqueeze(0)
            inputs_sfc = inputs_sfc.unsqueeze(0)
            inputs_main = torch.repeat_interleave(inputs_main,repeats=self.ensemble_size,dim=0)
            inputs_sfc = torch.repeat_interleave(inputs_sfc,repeats=self.ensemble_size,dim=0)
            inputs_main = inputs_main.flatten(0,1)
            inputs_sfc = inputs_sfc.flatten(0,1)
                    
        batch_size = inputs_main.shape[0]
        # if self.add_pres:
        #     sp = inputs_sfc[:,-1]
        #     pres  = self.preslay(sp)
        #     inputs_main = torch.cat((inputs_main,torch.unsqueeze(pres,2)),dim=2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
        if self.rnn1_mem is None: 
            # self.rnn1_mem = torch.randn(batch_size, self.nlay, self.nh_mem,device=device)
            self.rnn1_mem = torch.randn((batch_size, self.nlay, self.nh_mem),dtype=self.dtype,device=device)

        hx = self.mlp_surface1(inputs_sfc)
        hx = nn.Tanh()(hx)

        # TOA is first in memory, so to start at the surface we need to go backwards
        inputs_main = torch.flip(inputs_main, [1])

        # The input (a vertical sequence) is concatenated with the
        # output of the RNN from the previous time step 
        if self.model_type in ["LSTM"]:
            if self.random_init_cx:
                # cx = torch.randn(batch_size, self.nh_rnn1,device=device)
                cx = torch.randn((batch_size, self.nh_rnn1),dtype=self.dtype,device=device)
            else:
                cx = self.mlp_surface2(inputs_sfc)
                cx = nn.Tanh()(cx)
            hidden = (torch.unsqueeze(hx,0), torch.unsqueeze(cx,0))
        else:
            hidden = (torch.unsqueeze(hx,0))

        if self.use_initial_mlp:
            rnn1_input = self.mlp_initial(inputs_main)
            rnn1_input = nn.Tanh()(rnn1_input)
        else:
            rnn1_input = inputs_main 
            
        rnn1_input = torch.cat((rnn1_input,self.rnn1_mem), axis=2)

        rnn1out, states = self.rnn1(rnn1_input, hidden)

        rnn1out = torch.flip(rnn1out, [1])

        hx2 = torch.randn((batch_size, self.nh_rnn2),dtype=self.dtype,device=device)  # (batch, hidden_size)
        if self.model_type in ["LSTM"]:
            cx2 = torch.randn((batch_size, self.nh_rnn2),dtype=self.dtype,device=device)
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
            if not self.third_rnn: self.rnn1_mem = torch.flip(rnn2out, [1])
            
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

        if self.ny_sfc>0:
            #print("shape last_c", last_c.shape)
            # use cell state or hidden state?
            out_sfc = self.mlp_surface_output(last_h.squeeze())
            return out, out_sfc
        else:
            return out 
        

class SpaceStateModel(nn.Module):
    def __init__(self, nlay=30, nx = 4, nx_sfc=3, ny = 4, nneur=(64,64),model_type='LRU', 
                 use_initial_mlp=True, add_pres=False,  concat=False,
                 third_rnn=False, init_with_state_and_sfc=False, memory=None,
                 dtype=torch.float32,
                 device=None):
        super(SpaceStateModel, self).__init__()
        self.nx = nx
        self.ny = ny 
        self.nlay = nlay 
        self.nx_sfc = nx_sfc 
        self.nneur = nneur 
        self.add_pres = add_pres
        if self.add_pres:
            self.preslay = LayerPressure()
            nx = nx +1
        self.nh_rnn1 = self.nneur[0]
        self.nh_rnn2 = self.nneur[1]
        self.ny_rnn1 = self.nh_rnn1
        self.use_initial_mlp=use_initial_mlp
        self.init_with_state_and_sfc = init_with_state_and_sfc
        self.model_type=model_type
        self.dtype=dtype
        self.device=device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using model type {}".format(model_type))
        if self.model_type in ["Mamba","GSS"]:#,"QRNN"]:  #Mamba doesnt support providing the state
            print("WARNING: this SSM type doesn't support providing the state, so the surface variables are instead added to each point in the sequence")
            
            # concatenate the vertical (sequence) inputs with tiled scalars
            self.init_with_state_and_sfc=False
            self.use_initial_mlp=True
            nx = nx + nx_sfc
        if self.use_initial_mlp: 
            self.nh_mlp1 = self.nh_rnn1
            self.nx_rnn1 = self.nh_rnn1
        else:
            self.nx_rnn1 = nx
        self.use_intermediate_mlp = False
        if memory == 'None':
            print("Building non-autoregressive SSM, but may be stateful")
            self.autoregressive = False
        elif memory == 'Hidden':
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
        self.third_rnn=third_rnn

        print("nx rnn1", self.nx_rnn1, "nh rnn1", self.nh_rnn1, "ny rnn1", self.ny_rnn1)
        print("nx rnn2", self.nx_rnn2, "nh rnn2", self.nh_rnn2)  
        if self.autoregressive: 
            print("nh_memory", self.nh_latent)
        self.concat = concat

        if self.init_with_state_and_sfc:
            self.state1 = None
            # self.state2 = None 
            self.mlp_surface1  = nn.Linear(nx_sfc + self.nh_rnn1, self.nh_rnn1)
        else:
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
            if self.third_rnn:
                self.rnn3= MinGRU(self.nx_rnn2,self.nh_rnn2)
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
            
            if self.init_with_state_and_sfc:
                self.mlp_surface2  = nn.Linear(nx_sfc + self.nh_rnn1, self.nh_rnn1)
            else:
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
            if self.third_rnn:
                self.rnn3= SimpleGateLoopLayer(self.nneur[2])
                
            if self.init_with_state_and_sfc:
                self.mlp_surface2  = nn.Linear(nx_sfc + self.nh_rnn1, self.nh_rnn1)
            else:
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
                   
        if self.third_rnn:
            # self.rnn3= LRU(in_features=nx_rnn2,out_features=self.nneur[1],state_features=self.nneur[0])
            # self.rnn3= LRU(in_features=self.nneur[1],out_features=self.nneur[1],state_features=self.nneur[1])
            # self.mlp3  = nn.Linear(self.nneur[1], self.nneur[1])
            # self.SeqLayer3 = SequenceLayer(nlay=self.nlay,nneur=self.nneur[1],layernorm=True)
            if self.use_glu_layers: self.SeqLayer3 = GLU(nlay=self.nlay,nneur=self.nneur[2],layernorm=glu_layernorm)

        if self.concat:
            nx_last = self.ny_rnn1 + self.nh_rnn2
            if self.third_rnn:  
                nx_last += self.nh_rnn2
        else:
            nx_last = self.nh_rnn2
            
        if self.use_intermediate_mlp:
            self.mlp_latent = nn.Linear(nx_last, self.nh_latent)
            nx_last = self.nh_latent

        self.mlp_output        = nn.Linear(nx_last, self.ny)

    def reset_states(self):
        if self.init_with_state_and_sfc:
            self.state1 = None
        if self.autoregressive:
            self.rnn1_mem = None
            # self.rnn2_mem = None

    def detach_states(self):
        if self.init_with_state_and_sfc:
            self.state1 = self.state1.detach()
        if self.autoregressive:
            self.rnn1_mem = self.rnn1_mem.detach()
            # self.rnn2_mem = self.rnn2_mem.detach()
        
    def forward(self, inputs_main, inputs_sfc):

        # print("Shape inputs main", inputs_main.shape)
        batch_size = inputs_main.shape[0]
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device=self.device
        if self.autoregressive:
            if self.rnn1_mem is None: 
                self.rnn1_mem = torch.randn((batch_size, self.nlay, self.nh_latent),dtype=self.dtype,device=device)
    
            # if self.rnn2_mem is None: 
            #     self.rnn2_mem = torch.randn(batch_size, self.nlay, self.nh_latent ,device=device)
        if self.add_pres:
            sp = inputs_sfc[:,-1]
            pres  = self.preslay(sp)
            inputs_main = torch.cat((inputs_main,torch.unsqueeze(pres,2)),dim=2)
        
        if self.model_type=="S5" and self.pres_step_scale:
                refp = inputs_main[:,:,-1].clone().detach()
        
        inputs_main = torch.flip(inputs_main, [1])
        
        if self.model_type=="S5" and self.pres_step_scale:
                refp_rev = inputs_main[:,:,-1].clone().detach()

        if self.model_type in ["Mamba","GSS"]:#,,"QRNN"]:
            #Mamba doesnt support providing the state, so as a hack we instead
            # concatenate the vertical (sequence) inputs with tiled scalars
            inputs_sfc_tiled = torch.tile(torch.unsqueeze(inputs_sfc,1),(1,self.nlay,1))
            inputs_main = torch.cat((inputs_main,inputs_sfc_tiled), dim=2)
        else:
            # Initial states can be a vector of shape (state_features,) or a matrix of shape (batch_size, state_features)
            if self.init_with_state_and_sfc:
                if self.state1==None:
                    self.state1 =  torch.randn((batch_size, self.nh_rnn1),dtype=self.dtype,device=device)
    
                init_inputs = torch.cat((inputs_sfc,self.state1), dim=1)
            else:
                init_inputs = inputs_sfc
                
            init_states = self.mlp_surface1(init_inputs)
            # init_states = nn.Softsign()(init_states)
            init_states = nn.Tanh()(init_states)
            # print("shape init states" , init_states.shape)
        # print("shape inp main inp", inputs_main.shape)

        if self.use_initial_mlp:
            rnn1_input = self.mlp_initial(inputs_main)
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
        
        if self.init_with_state_and_sfc: 
            self.state1 = out[:,-1,:].clone()
            
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
            init_states2 = torch.randn((batch_size, 1, self.nh_rnn2),dtype=self.dtype,device=device)
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
            
            # self.rnn2_mem = torch.clone(hidden_fin)
            # if self.third_rnn: 
            #     self.rnn1_mem = torch.clone(hidden_fin, [1])
            # else:
            #     out3 = torch.flip(out3, [1])
            self.rnn1_mem = torch.flip(outs, [1])
        
        outs = self.mlp_output(outs)
        
        if self.dtype==torch.bfloat16:
            outs = outs.to(torch.float32)
            
        return outs
        