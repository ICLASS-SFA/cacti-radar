import os
import glob, time
import numpy as np
import xarray as xr
import pandas as pd

if __name__ == "__main__":

    start_datetime = '2018-10-01T00:00'
    end_datetime = '2019-04-01T00:00'

    datadir = "/global/project/projectdirs/m1657/zfeng/cacti/arm/csapr/taranis_corcsapr2cfrppiqcM1_celltracking.c1.new/stats/"
    datafile = f"{datadir}csapr2_cellcounts_timeseries_20181015_20190303.nc"
    outfilename = f"{datadir}csapr2_cellcounts_timeseries_20181015_20190303_interp.nc"

    # Read input time series data
    ds = xr.open_dataset(datafile)

    # Replace bins with NaN value to 0
    ds['cell_area'] = ds['cell_area'].where(~np.isnan(ds['cell_area']), other=0)

    # Generate standard 15min time marks within the start/end datetime
    std_times = pd.date_range(start=start_datetime, end=end_datetime, freq='15min')

    # Interpolate to match standard time
    dsout = ds.interp(time=std_times, method='nearest')

    # Mask out missing period values
    cell_count = dsout.cell_count.values
    cell_area = dsout.cell_area.values
    outtimes1 = [pd.Timestamp('2018-12-27T00'), pd.Timestamp('2019-1-21T00')]
    outtimes2 = [pd.Timestamp('2019-2-9T00'), pd.Timestamp('2019-2-24T00')]
    # Replace output time period with NaN
    outidx1 = np.where((std_times >= outtimes1[0]) & (std_times <= outtimes1[1]))
    outidx2 = np.where((std_times >= outtimes2[0]) & (std_times <= outtimes2[1]))
    cell_count[outidx1] = np.nan
    cell_count[outidx2] = np.nan
    cell_area[outidx1,:] = np.nan
    cell_area[outidx2,:] = np.nan
    # Convert to Xarray
    cell_count = xr.DataArray(cell_count, coords={'time':dsout.time}, dims=('time'), attrs=dsout.cell_count.attrs)
    cell_area = xr.DataArray(cell_area, coords={'time':dsout.time, 'cell':dsout.cell}, dims=('time','cell'), attrs=dsout.cell_area.attrs)
    # Replace arrays in output dataset
    dsout['cell_count'] = cell_count
    dsout['cell_area'] = cell_area

    # Global attributes
    dsout.attrs['start_datetime'] = start_datetime
    dsout.attrs['end_datetime'] = end_datetime
    dsout.attrs['created_on'] = time.ctime(time.time())

    # Set encoding/compression for all variables
    comp = dict(zlib=True, dtype='float32')
    encoding = {var: comp for var in dsout.data_vars}

    # Write output
    dsout.to_netcdf(path=outfilename, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
    print(f"Output saved as: {outfilename}")