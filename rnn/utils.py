#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch data loader based on indexing HDF5 file, and other utils
"""

class generator_xy(torch.utils.data.Dataset):
    def __init__(self, filepath, nloc=384, nlev=60, add_refpres=True, cuda=False):
        self.filepath = filepath
        # The file list will be divided into chunks (a list of lists)eg [[12,4,32],[1,9,3]..]
        # where the length of each item is the chunk size; i.e. how many files 
        # are loaded at once (in this example 3 files)
        # self.chunk_size = chunk_size # how many batches are loaded at once in getitem
        self.nloc = nloc
        self.nlev = nlev
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

        if 'train' in self.filepath:
            self.is_validation = False
            print("Training dataset, path is: {}".format(self.filepath))
        else:
            self.is_validation = True
            print("Validation dataset, path is: {}".format(self.filepath))
        self.cuda = cuda

        self.add_refpres = add_refpres
        # batch_idx_expanded =  [0,1,2,3...ntime*1024]
        hdf = h5py.File(self.filepath, 'r')
        self.ntimesteps = hdf['input_lev'].shape[0]//self.nloc
        hdf.close()
        print("Number of locations {}; time steps {}".format(self.nloc, self.ntimesteps))
        # indices_all = list(np.arange(self.ntimesteps*self.nloc))
        # chunksize_tot = self.nloc*self.chunk_size
        # indices_chunked = self.chunkize(indices_all,chunksize_tot,False) 
        # self.hdf = h5py.File(self.filepath, 'r')

    def __len__(self):
        return self.ntimesteps*self.nloc
    
    def __getitem__(self, batch_indices):
        hdf = h5py.File(self.filepath, 'r')
        # hdf = self.hdf
        
        x_lay_b = hdf['input_lev'][batch_indices,:]
        x_sfc_b = hdf['input_sca'][batch_indices,:]
        y_lay_b = hdf['output_lev'][batch_indices,:]
        y_sfc_b = hdf['output_sca'][batch_indices,:]
        
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
        #print("indices chunked [0]", self.indices_chunked[0])
        # one item is one chunk, consisting of chunk_factor*batch_size samples
        
    def __len__(self):
        return self.num_samples // self.batch_size

    def __iter__(self):
        return iter(self.indices_chunked)
        # for batch in self.indices_chunked:
        #     yield batch