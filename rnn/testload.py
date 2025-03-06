#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:18:13 2025

@author: peter
"""
import tables
import h5py


# x.shape
# Out[22]: (480, 1920, 60, 15)

x = x[0:461]
new_data_path = "/media/peter/CrucialP1/data/ClimSim/low_res_expanded/testfile.h5"


with h5py.File(new_data_path, 'w') as hdf:
    # ysfc = hdf.create_dataset("output_sca", (ns, nloc, ny_sfc), maxshape=(None, nloc, ny_sfc),
    #                                    dtype='float32')#, compression=compression, compression_opts=comp_level)
    xx = hdf.create_dataset("input_lev", (461, nloc, nlev, nx), maxshape=(None, nloc, nlev, nx),
                               chunks = (3,nloc,nlev,nx), compression="lzf",
                                       dtype='float32')
    xx[:] = x
    
    
    
hf = h5py.File(new_data_path, 'r')

t0_it = time.time()
# x = hf['input_lev'][inds]
xx = hf['input_lev'][0:240]
elaps = time.time() - t0_it
print("Runtime load {:.2f}s".format(elaps))
t0_it = time.time()
hf.close()
#                               lzf     gzip
# REF:          3.92s  4.29s 
# chunk3:       1.70s  1.97s    3.04    14.67
# chunk12       1.66s  1.93s
# chunk 120     1.66            5.97

# 461, chunk3 

(xx.size * 4 / 1e9)  / 1.97
# Out[40]: 1.6841421319796954
outfile = "/media/peter/CrucialP1/data/ClimSim/low_res_expanded/testfile2.npz"

np.savez(outfile, yy)

_ = outfile.seek(0) # Only needed to simulate closing & reopening file

npzfile = np.load(outfile)


t0_it = time.time()
yy2 = npzfile['arr_0']
elaps = time.time() - t0_it
print("Runtime load {:.2f}s".format(elaps))
t0_it = time.time()
# Runtime load 2.04s



outfile = "/media/peter/CrucialP1/data/ClimSim/low_res_expanded/testfile3.npz"
np.savez(outfile, xlev=x)

npzfile = np.load(outfile)

t0_it = time.time()
xx2 = npzfile['xlev']
elaps = time.time() - t0_it
print("Runtime load {:.2f}s".format(elaps))
t0_it = time.time()
# 240: 
# Runtime load 2.03s
# Runtime load 3.32s
# 480:
# Runtime load 4.93s
# Runtime load 11.72s
# Runtime load 7.22s