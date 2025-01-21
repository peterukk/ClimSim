#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch data loader based on indexing HDF5 file, and other utils
"""
import h5py
import torch
import torch.nn as nn
import numpy as np 
import gc
from torchmetrics.regression import R2Score
import time
from metrics import corrcoeff_pairs_batchfirst 
import matplotlib
import matplotlib.pyplot as plt
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

# qliq = 1 - np.exp(-xlay[:,:,2] * lbd_qc)
# qice = 1 - np.exp(-xlay[:,:,3] * lbd_qi)

# plt.hist(xlay[:,:,2].flatten())
# plt.hist(qliq.flatten())

# plt.hist(xlay[:,:,3].flatten(), bins=20)
# plt.hist(qice.flatten(), bins=20)


class model_train_eval:
    def __init__(self, dataloader, model, batch_size, nlev, ny, 
                 autoregressive, train, scaler, optimizer, mp_autocast):
        super().__init__()

        self.loader = dataloader
        self.train = train
        self.report_freq = 800
        self.batch_size = batch_size
        self.model = model 
        self.scaler = scaler
        self.optimizer = optimizer
        self.mp_autocast = mp_autocast
        self.autoregressive = autoregressive
        if self.autoregressive:
            self.model.reset_states()
        self.nlev = nlev 
        self.ny = ny
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metric_R2 =  R2Score().to(self.device) 
        self.metric_R2_heating =  R2Score().to(self.device) 
        self.metric_R2_precc =  R2Score().to(self.device) 
        self.metric_R2_moistening =  R2Score().to(self.device) 

        self.metrics= {'loss': 0, 'mean_squared_error': 0,  # the latter is just MSE
                        'mean_absolute_error': 0, 'R2' : 0, 'R2_heating' : 0,'R2_moistening' : 0,  
                        'R2_precc' : 0, 'R2_lev' : np.zeros((self.nlev,self.ny)),
                        'h_conservation' : 0 }

    def eval_one_epoch(self, epoch, timewindow=1):
        report_freq = self.report_freq
        running_loss = 0.0 
        epoch_loss = 0.0
        epoch_mse = 0.0; epoch_mae = 0.0
        epoch_R2precc = 0.0
        epoch_hcon = 0.0
        epoch_r2_lev = 0.0
        t_comp =0 
        if self.autoregressive:
            preds_lay = []; preds_sfc = []
            targets_lay = []; targets_sfc = [] 
            sps = []
        t0_it = time.time()
        j = 0; k = 0; k2=2    
        if self.autoregressive:
            loss_update_start_index = 60
        else:
            loss_update_start_index = 0
        for i,data in enumerate(self.loader):
            inputs_lay_chunks, inputs_sfc_chunks, targets_lay_chunks, targets_sfc_chunks = data
            inputs_lay_chunks   = inputs_lay_chunks.to(self.device)
            inputs_sfc_chunks   = inputs_sfc_chunks.to(self.device)
            targets_sfc_chunks  = targets_sfc_chunks.to(self.device)
            targets_lay_chunks  = targets_lay_chunks.to(self.device)
            
            inputs_lay_chunks    = torch.split(inputs_lay_chunks, self.batch_size)
            inputs_sfc_chunks    = torch.split(inputs_sfc_chunks, self.batch_size)
            targets_sfc_chunks   = torch.split(targets_sfc_chunks, self.batch_size)
            targets_lay_chunks   = torch.split(targets_lay_chunks, self.batch_size)
         
            # to speed-up IO, we loaded chunks=many batches, which now need to be divided into batches
            for ichunk in range(len(inputs_lay_chunks)):
                inputs_lay = inputs_lay_chunks[ichunk]
                inputs_sfc = inputs_sfc_chunks[ichunk]
                target_lay = targets_lay_chunks[ichunk]
                target_sfc = targets_sfc_chunks[ichunk]
                sp = inputs_sfc[:,0:1] # surface pressure


                tcomp0= time.time()
                    
                if self.mp_autocast:
                    with torch.autocast(device_type=self.device.type, dtype=dtype):
                        pred_lay, pred_sfc = self.model(inputs_lay, inputs_sfc)
                else:
                    pred_lay, pred_sfc = self.model(inputs_lay, inputs_sfc)
                    
                if self.autoregressive:
                    # In the autoregressive training case are gathering many time steps before computing loss
                    preds_lay.append(pred_lay)
                    preds_sfc.append(pred_sfc)
                    targets_lay.append(target_lay)
                    targets_sfc.append(target_sfc)
                    sps.append(sp) 
                    
                else:
                    preds_lay = pred_lay
                    preds_sfc = pred_sfc 
                    targets_lay = target_lay
                    targets_sfc = target_sfc
                    sps = sp
                    
                if (not self.autoregressive) or (self.autoregressive and (j+1) % timewindow==0):
            
                    if self.autoregressive:
                        preds_lay   = torch.stack(preds_lay)
                        preds_sfc   = torch.stack(preds_sfc)
                        targets_lay = torch.stack(targets_lay)
                        targets_sfc = torch.stack(targets_sfc)
                        sps         = torch.stack(sps)
                                
                    if self.mp_autocast:
                        with torch.autocast(device_type=self.device.type, dtype=dtype):
                            #loss = loss_fn(targets_lay, targets_sfc, preds_lay, preds_sfc)
                            
                            mse = my_mse_flatten(targets_lay, targets_sfc, preds_lay, preds_sfc)
                            
                            ypo_lay, ypo_sfc = self.model.postprocessing(preds_lay, preds_sfc)
                            yto_lay, yto_sfc = self.model.postprocessing(targets_lay, targets_sfc)
                            sps_denorm = sp = sps*(sp_max - sp_min) + sp_mean
                            h_con = metric_h_con(yto_lay, ypo_lay, sps_denorm)
                            
                            loss = loss_fn(mse, h_con)
                    else:
                        #loss = loss_fn(targets_lay, targets_sfc, preds_lay, preds_sfc)
                        
                        mse = my_mse_flatten(targets_lay, targets_sfc, preds_lay, preds_sfc)
                        ypo_lay, ypo_sfc = self.model.postprocessing(preds_lay, preds_sfc)
                        yto_lay, yto_sfc = self.model.postprocessing(targets_lay, targets_sfc)
                        sps_denorm = sp = sps*(sp_max - sp_min) + sp_mean
                        h_con = metric_h_con(yto_lay, ypo_lay, sps_denorm)
                        loss = loss_fn(mse, h_con)
                        
                    if self.train:
                        if use_scaler:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()       
                            optimizer.step()
            
                        optimizer.zero_grad()
                            
                    running_loss    += loss.item()
                    #mae             = metrics.mean_absolute_error(targets_lay, preds_lay)
                    if j>loss_update_start_index:
                        with torch.no_grad():
                            epoch_loss      += loss.item()
                            #epoch_energy    += energy.item()
                            epoch_mse       += mse.item()
                            #epoch_mae       += mae.item()
                        
                           # yto, ypo =  denorm_func(targets_lay, preds_lay)
                            # -------------- TO-DO:  DE-NORM OUTPUT --------------
                            #yto, ypo = targets_lay, preds_lay

                            epoch_hcon  += h_con.item()
                            
                            self.metric_R2.update(ypo_lay.reshape((-1,ny)), yto_lay.reshape((-1,ny)))
                            self.metric_R2_heating.update(ypo_lay[:,:,0].reshape(-1,1), yto_lay[:,:,0].reshape(-1,1))
                            self.metric_R2_moistening.update(ypo_lay[:,:,1].reshape(-1,1), yto_lay[:,:,1].reshape(-1,1))

                            self.metric_R2_precc.update(ypo_sfc[:,3].reshape(-1,1), yto_sfc[:,3].reshape(-1,1))
                            
                            r2_np = np.corrcoef((ypo_sfc.reshape(-1,ny_sfc)[:,3].detach().cpu().numpy(),yto_sfc.reshape(-1,ny_sfc)[:,3].detach().cpu().numpy()))[0,1]
                            epoch_R2precc += r2_np
                            #print("R2 numpy", r2_np, "R2 torch", self.metric_R2_precc(ypo_sfc[:,3:4], yto_sfc[:,3:4]) )

                            ypo_lay = ypo_lay.reshape(-1,self.nlev,self.ny).detach().cpu().numpy()
                            yto_lay = yto_lay.reshape(-1,self.nlev,self.ny).detach().cpu().numpy()

                            epoch_r2_lev += corrcoeff_pairs_batchfirst(ypo_lay, yto_lay) 
                           # if track_ks:
                           #     if (j+1) % max(timewindow*4,12)==0:
                           #         epoch_ks += kolmogorov_smirnov(yto,ypo).item()
                           #         k2 += 1
                            k += 1
                    if self.autoregressive:
                        preds_lay = []; preds_sfc = []
                        targets_lay = []; targets_sfc = [] 
                        sps = []
                    if self.autoregressive: 
                        self.model.detach_states()
                
                t_comp += time.time() - tcomp0
                # # print statistics 
                if j % report_freq == (report_freq-1): # print every 200 minibatches
                    elaps = time.time() - t0_it
                    running_loss = running_loss / (report_freq/timewindow)
                    #running_energy = running_energy / (report_freq/timewindow)
                    r2raw = self.metric_R2.compute()
                    #r2raw_prec = self.metric_R2_precc.compute()


                    print("[{:d}, {:d}] Loss: {:.2e}  runR2: {:.2f},  elapsed {:.1f}s (compute {:.1f})" .format(epoch + 1, 
                                                    j+1, running_loss, r2raw, elaps, t_comp))
                    running_loss = 0.0
                    running_energy = 0.0
                    t0_it = time.time()
                    t_comp = 0
                j += 1

        self.metrics['loss'] =  epoch_loss / k
        self.metrics['mean_squared_error'] = epoch_mse / k
        self.metrics["h_conservation"] =  epoch_hcon / k

        #self.metrics['energymetric'] = epoch_energy / k
        #self.metrics['mean_absolute_error'] = epoch_mae / k
        #self.metrics['ks'] =  epoch_ks / k2
        self.metrics['R2'] = self.metric_R2.compute()
        self.metrics['R2_heating'] = self.metric_R2_heating.compute()
        self.metrics['R2_moistening'] = self.metric_R2_moistening.compute()

        #self.metrics['R2_precc'] = self.metric_R2_precc.compute()
        self.metrics['R2_precc'] = epoch_R2precc / k
        
        self.metrics['R2_lev'] = epoch_r2_lev / k

        self.metric_R2.reset(); self.metric_R2_heating.reset(); self.metric_R2_precc.reset()
        if self.autoregressive:
            self.model.reset_states()
        
        datatype = "TRAIN" if self.train else "VAL"
        print('Epoch {} {} loss: {:.2e}  MSE: {:.2e}  h-con:  {:.2e}   R2: {:.2f}  R2-dT/dt: {:.2f}   R2-dq/dt: {:.2f}   R2-precc: {:.3f}'.format(epoch+1, datatype, 
                                                            self.metrics['loss'], 
                                                            self.metrics['mean_squared_error'], 
                                                            self.metrics['h_conservation'],
                                                            self.metrics['R2'],
                                                            self.metrics['R2_heating'],
                                                            self.metrics['R2_moistening'],                                                              
                                                            self.metrics['R2_precc'] ))

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()

class generator_xy(torch.utils.data.Dataset):
    def __init__(self, filepath, nloc=384, add_refpres=True, cuda=False):
        self.filepath = filepath
        # The file list will be divided into chunks (a list of lists)eg [[12,4,32],[1,9,3]..]
        # where the length of each item is the chunk size; i.e. how many files 
        # are loaded at once (in this example 3 files)
        # self.chunk_size = chunk_size # how many batches are loaded at once in getitem
        self.nloc = nloc
        self.cloud_exp_norm = True
        if type(self.filepath)==list:
            # In this case, each time a chunk is fetched, all the files are opened and the
            # data is concatenated along the column dimension
            self.num_files = len(self.filepath)
            self.ncol = self.num_files * nloc
            print("Number of files: {}".format(self.num_files))
            hdf = h5py.File(self.filepath[0], 'r')
        else:
            self.num_files = 1
            self.ncol = nloc
            hdf = h5py.File(self.filepath, 'r')
        print("Shape dim 0",hdf['input_lev'].shape[0] )
        self.ntimesteps = hdf['input_lev'].shape[0]//self.nloc
        self.nlev = hdf['input_lev'].shape[1]; self.nx = hdf['input_lev'].shape[2]
        self.nx_sfc = hdf['input_sca'].shape[1]
        self.ny = hdf['output_lev'].shape[2]
        self.ny_sfc = hdf['output_sca'].shape[1]
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
        self.refpres_norm = (self.refpres-self.refpres.min())/(self.refpres.max()-self.refpres.min())*2 - 1

        #if 'train' in self.filepath:
        #    self.is_validation = False
        #    print("Training dataset, path is: {}".format(self.filepath))
        #else:
        #    self.is_validation = True
        #    print("Validation dataset, path is: {}".format(self.filepath))
        self.cuda = cuda

        self.add_refpres = add_refpres
        # batch_idx_expanded =  [0,1,2,3...ntime*1024]

        print("Number of locations {}; colums {}, time steps {}".format(self.nloc,self.ncol, self.ntimesteps))
        # indices_all = list(np.arange(self.ntimesteps*self.nloc))
        # chunksize_tot = self.nloc*self.chunk_size
        # indices_chunked = self.chunkize(indices_all,chunksize_tot,False) 
        # self.hdf = h5py.File(self.filepath, 'r')

    def __len__(self):
        # return self.ntimesteps*self.nloc
        return self.ntimesteps*self.ncol

    def __getitem__(self, batch_indices):
        if self.num_files>1:
            i = 0
            for filepath in self.filepath:
                hdf = h5py.File(filepath, 'r')
                # hdf = self.hdf

                x_lay_b0 = hdf['input_lev'][batch_indices,:]
                x_sfc_b0 = hdf['input_sca'][batch_indices,:]
                y_lay_b0 = hdf['output_lev'][batch_indices,:]
                y_sfc_b0 = hdf['output_sca'][batch_indices,:]
                # print("1shape x file i ", x_lay_b0.shape)
                x_lay_b0 = x_lay_b0.reshape(-1,self.nloc,self.nlev,self.nx)
                x_sfc_b0 = x_sfc_b0.reshape(-1,self.nloc,self.nx_sfc)
                y_lay_b0 = y_lay_b0.reshape(-1,self.nloc,self.nlev,self.ny)
                y_sfc_b0 = y_sfc_b0.reshape(-1,self.nloc,self.ny_sfc)
                # print("2 shape x file i ", x_lay_b0.shape)

                # if self.cloud_exp_norm:
                #     x_lay_b0[:,:,2] = 1 - np.exp(-x_lay_b0[:,:,2] * lbd_qc)
                #     x_lay_b0[:,:,3] = 1 - np.exp(-x_lay_b0[:,:,3] * lbd_qi)

                # if self.add_refpres:
                #     dim0,dim1,dim2 = x_lay_b0.shape
                #     refpres_norm = self.refpres_norm.reshape((1,-1,1))
                #     refpres_norm = np.repeat(refpres_norm, dim0,axis=0)
                #     x_lay_b0 = np.concatenate((x_lay_b0, refpres_norm),axis=2)
                #     del refpres_norm 

                if i==0:
                    x_lay_b = x_lay_b0
                    x_sfc_b = x_sfc_b0
                    y_lay_b = y_lay_b0
                    y_sfc_b = y_sfc_b0
                else:
                    x_lay_b = np.concatenate((x_lay_b, x_lay_b0), axis=1)
                    x_sfc_b = np.concatenate((x_sfc_b, x_sfc_b0), axis=1)
                    y_lay_b = np.concatenate((y_lay_b, y_lay_b0), axis=1)
                    y_sfc_b = np.concatenate((y_sfc_b, y_sfc_b0), axis=1)
                    
                i = i + 1
                
            x_lay_b = x_lay_b.reshape(-1,self.nlev,self.nx)
            x_sfc_b = x_sfc_b.reshape(-1,self.nx_sfc)
            y_lay_b = y_lay_b.reshape(-1,self.nlev,self.ny)
            y_sfc_b = y_sfc_b.reshape(-1,self.ny_sfc)

        else:       
            hdf = h5py.File(self.filepath, 'r')
            # hdf = self.hdf

            x_lay_b = hdf['input_lev'][batch_indices,:]
            x_sfc_b = hdf['input_sca'][batch_indices,:]
            y_lay_b = hdf['output_lev'][batch_indices,:]
            y_sfc_b = hdf['output_sca'][batch_indices,:]
            # if self.cloud_exp_norm:
            #     x_lay_b[:,:,2] = 1 - np.exp(-x_lay_b[:,:,2] * lbd_qc)
            #     x_lay_b[:,:,3] = 1 - np.exp(-x_lay_b[:,:,3] * lbd_qi)
                
            # if self.add_refpres:
            #     dim0,dim1,dim2 = x_lay_b.shape
            #     # if self.norm=="minmax":
            #     refpres_norm = self.refpres_norm.reshape((1,-1,1))
            #     refpres_norm = np.repeat(refpres_norm, dim0,axis=0)
            #     #self.x[:,:,nx-1] = refpres_norm
            #     x_lay_b = np.concatenate((x_lay_b, refpres_norm),axis=2)
            #     # self.x  = torch.cat((self.x,refpres_norm),dim=3)
            #     del refpres_norm 
       
        if self.cloud_exp_norm:
            x_lay_b[:,:,2] = 1 - np.exp(-x_lay_b[:,:,2] * lbd_qc)
            x_lay_b[:,:,3] = 1 - np.exp(-x_lay_b[:,:,3] * lbd_qi)
                    
        if self.add_refpres:
            dim0,dim1,dim2 = x_lay_b.shape
            # if self.norm=="minmax":
            refpres_norm = self.refpres_norm.reshape((1,-1,1))
            refpres_norm = np.repeat(refpres_norm, dim0,axis=0)
            #self.x[:,:,nx-1] = refpres_norm
            x_lay_b = np.concatenate((x_lay_b, refpres_norm),axis=2)
            # self.x  = torch.cat((self.x,refpres_norm),dim=3)
            del refpres_norm 

        hdf.close()

        x_lay_b = torch.from_numpy(x_lay_b)
        x_sfc_b = torch.from_numpy(x_sfc_b)
        y_lay_b = torch.from_numpy(y_lay_b)
        y_sfc_b = torch.from_numpy(y_sfc_b)

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # x_lay_b = x_lay_b.to(device)
        # x_sfc_b = x_sfc_b.to(device)
        # y_lay_b = y_lay_b.to(device)
        # sp = sp.to(device)
        # print("gen x_lay shape", x_lay_b.shape)
        gc.collect()

        return x_lay_b, x_sfc_b, y_lay_b, y_sfc_b

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
        # we need the indices to be sorted within a chunk because these indices
        # are used to index into the first dimension of a H5 file
        for i in range(filelist):
            filelist[i] = sorted(filelist[i])
            
    mylist = list(divide(filelist,chunk_size))
    if shuffle_after_chunking:
        random.shuffle(mylist)  
    return mylist


class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, num_samples_per_chunk, num_samples, shuffle=False):
        self.num_samples_per_chunk = num_samples_per_chunk
        self.num_samples = num_samples
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