"""
Resamples mergesonde/interpsonde data to a coarser resolution netCDF data.
The input mergesonde/interpsonde data is produced by concatenating daily data using NCO.
"""
import numpy as np
import xarray as xr
import pandas as pd

if __name__ == '__main__':

    out_freq = "1H"

    # period = '20181001_20190430'
    # # period = '20181010_20190302'
    # interpsondedir = f'/global/cscratch1/sd/feng045/iclass/cacti/arm/interpsondeM1/'
    # interpsonde_file = f'{interpsondedir}corinterpolatedsondeM1.c1.{period}.nc'
    # out_dir = f'/global/project/projectdirs/m1657/zfeng/cacti/arm/sounding_stats/'
    # out_file = f'{out_dir}corinterpolatedsondeM1.c1.{period}_resample{out_freq}.nc'

    period = '20181005_20190430'
    interpsondedir = f'/global/cscratch1/sd/feng045/iclass/cacti/arm/mergesondeM1/'
    interpsonde_file = f'{interpsondedir}cormergesonde2maceM1.c1.{period}.nc'
    out_dir = f'/global/project/projectdirs/m1657/zfeng/cacti/arm/sounding_stats/'
    out_file = f'{out_dir}cormergesonde2maceM1.c1.{period}_resample{out_freq}.nc'

    # Read data
    print(f'Reading input data: {interpsonde_file}')
    ds = xr.open_dataset(interpsonde_file)
    print('Finished reading input data.')

    # Resample data
    print('Resampling dataset ...')
    dsout = ds.resample(time=out_freq, keep_attrs=True).nearest(tolerance='1min')
    # Copy attributes
    dsout['height'].attrs = ds['height'].attrs

    # Write netCDF file:
    # Set encoding/compression for all variables
    comp = dict(zlib=True, dtype='float32')
    encoding = {var: comp for var in dsout.data_vars}
    # Update base_time variable dtype as 'double' for better precision
    bt_dict = {'base_time': {'zlib':True, 'dtype':'float64'},
    #            'time_offset': {'zlib':True, 'dtype':'float64'},
                'time': {'zlib':True, 'dtype':'float64'}}
    encoding.update(bt_dict)

    # Write to netcdf file
    dsout.to_netcdf(path=out_file, mode='w', format='NETCDF4_CLASSIC', unlimited_dims='time', encoding=encoding)
    print(f'Output saved as: {out_file}')