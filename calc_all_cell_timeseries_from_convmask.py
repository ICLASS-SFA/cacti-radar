import os
import glob, time
import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
import xarray as xr
# import pandas as pd


if __name__ == "__main__":

    datadir = "/global/cscratch1/sd/feng045/iclass/cacti/arm/csapr/taranis_corcsapr2cfrppiqcM1_gridded_convmask.c1/"
    outdir = "/global/project/projectdirs/m1657/zfeng/cacti/arm/csapr/taranis_corcsapr2cfrppiqcM1_celltracking.c1.new/stats/"
    basename = "taranis_corcsapr2cfrppiqcM1_convmask.c1."
    outbasename = "csapr2_cellcounts_timeseries_"

    outfilename = f"{outdir}{outbasename}20181015_20190303.nc"

    # Find all files
    # files = sorted(glob.glob(f"{datadir}{basename}*201901??.*.nc"))
    files = sorted(glob.glob(f"{datadir}{basename}*.nc"))
    nfiles = len(files)
    print(f"Number of files: {nfiles}")

    # Read data
    ds = xr.open_mfdataset(files, concat_dim='time', combine='nested')
    print(f"Finished reading data.")
    nx = ds.sizes['x']
    ny = ds.sizes['y']
    ntimes = ds.sizes['time']
    dx = ds.attrs['dx'] / 1000
    dy = ds.attrs['dy'] / 1000

    conv_mask = ds.conv_mask.values
    # cell_num, cell_npix = np.unique(conv_mask.sel(time='2019-01-25T18:00').values.reshape((1,nx*ny)), return_counts=True, axis=1)
    # cell_num, cell_npix = np.unique(conv_mask.values.reshape((ntimes,nx*ny)), return_counts=True, axis=1)

    nmaxcell = 200
    ncells_all = np.full(ntimes, np.NaN, dtype=int)
    cell_area = np.full((ntimes, nmaxcell), np.NaN, dtype=float)
    # Loop over time
    for tt in range(0, ntimes):
        print(ds.time.isel(time=tt).values)
        # Get the number of unique values in this scene
        icellnum, icounts = np.unique(conv_mask[tt,:,:], return_counts=True)
        # The maximum number is the number of cells
        max_cellnum = np.max(icellnum)    
        ncells_all[tt] = max_cellnum
        # 0: not a cell, 1+: cell. Get the number of pixel for each cell
        if (max_cellnum > 0):
            cell_area[tt,0:max_cellnum] = icounts[1:max_cellnum+1]

    # Multiply pixel size to get cell area
    cell_area = cell_area * dx * dy

    # Define xarray dataset for Map
    print(f'Writing output ...')
    varlist = {'cell_count': (['time'], ncells_all), \
               'cell_area': (['time', 'cell'], cell_area), \
              }
    coordlist = {'time': (['time'], ds.time), \
                 'cell': (['cell'], np.arange(0, nmaxcell)), \
                }
    attrlist = {'title': 'CSAPR2 convective cell counts time series', \
                'contact':'Zhe Feng, zhe.feng@pnnl.gov', \
                'created_on':time.ctime(time.time())}    

    dsout = xr.Dataset(varlist, coords=coordlist, attrs=attrlist)

    dsout['cell'].attrs['long_name'] = 'Cell number'
    dsout['cell'].attrs['units'] = 'unitless'

    dsout['cell_count'].attrs['long_name'] = 'Number of cells at each time'
    dsout['cell_count'].attrs['units'] = 'counts'

    dsout['cell_area'].attrs['long_name'] = 'Aarea of cells at each time'
    dsout['cell_area'].attrs['units'] = 'km2'

    # Set encoding/compression for all variables
    comp = dict(zlib=True, dtype='float32')
    encoding = {var: comp for var in dsout.data_vars}

    fillvalue = np.nan
    dsout.to_netcdf(path=outfilename, mode='w', format='NETCDF4', 
                    unlimited_dims='time', encoding=encoding)
    print('Output saved as: ', outfilename)

    import pdb; pdb.set_trace()