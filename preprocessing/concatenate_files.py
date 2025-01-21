
import h5py

out_file_name = "train_y1-2.h5"

file_list = ["train_y1.h5", "train_y2.h5"]
file_dir = "/network/group/aopp/predict/HMC009_UKKONEN_CLIMSIM/ClimSim_data/ClimSim_low-res-expanded/train/preprocessed/"

output_file = file_dir + out_file_name

#keep track of the total number of rows
nrows = 0

compression = "gzip"
compression = "lzf"
comp_level = 8

nlev = 60 
nx = 9
nx_sfc = 17
ny = 6
ny_sfc = 8

with h5py.File(output_file, 'w') as hdf:

    for n, f in enumerate(file_list):
        filepath = file_dir + f
        hdf0  = h5py.File(filepath, 'r')
        x_lay = hdf0['input_lev'][:]
        x_sfc = hdf0['input_sca'][:]
        y_lay = hdf0['output_lev'][:]
        y_sfc = hdf0['output_sca'][:]

        nrows = nrows + x_lay.shape[0]


        if n == 0:
            #first file; create the dummy dataset with no max shape
            xlay = hdf.create_dataset("input_lev", (nrows, nlev, nx), maxshape=(None, nlev, nx),
                                               compression=compression, dtype='float32')#, compression_opts=comp_level)
            xsfc = hdf.create_dataset("input_sca", (nrows, nx_sfc), maxshape=(None, nx_sfc),
                                               compression=compression, dtype='float32')#, compression_opts=comp_level)

            ylay = hdf.create_dataset("output_lev", (nrows, nlev, ny), maxshape=(None, nlev, ny),
                                               compression=compression, dtype='float32')#, compression_opts=comp_level)
            ysfc = hdf.create_dataset("output_sca", (nrows, ny_sfc), maxshape=(None, ny_sfc),
                                               compression=compression, dtype='float32')#, compression_opts=comp_level)       

            #fill the first section of the dataset
            xlay[:] = x_lay; xlay.attrs['varnames'] = hdf0['input_lev'].attrs.get('varnames')
            xsfc[:] = x_sfc; xsfc.attrs['varnames'] = hdf0['input_sca'].attrs.get('varnames')

            ylay[:] = y_lay; ylay.attrs['varnames'] = hdf0['output_lev'].attrs.get('varnames')
            ysfc[:] = y_sfc; ysfc.attrs['varnames'] = hdf0['output_sca'].attrs.get('varnames')

            i0 = nrows

        else:
            #resize the dataset to accomodate the new data
            xlay =  hdf['input_lev']
            xsfc =  hdf['input_sca']
            ylay =  hdf['output_lev']
            ysfc =  hdf['output_sca']


            xlay.resize(nrows, axis=0); xlay[i0:nrows, :] = x_lay
            xsfc.resize(nrows, axis=0); xsfc[i0:nrows, :] = x_sfc
            ylay.resize(nrows, axis=0); ylay[i0:nrows, :] = y_lay
            ysfc.resize(nrows, axis=0); ysfc[i0:nrows, :] = y_sfc

            i0 = nrows
