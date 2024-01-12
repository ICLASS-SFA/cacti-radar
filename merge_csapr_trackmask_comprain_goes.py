import numpy as np
import xarray as xr
import pandas as pd
import glob, os, sys
import yaml
import dask
from dask.distributed import Client, LocalCluster

def get_datetime_from_filename(
    filenames,
    data_basename,
    time_format="yyyymodd.hhmmss",
):
    """
    Calculate Pandas DatetimeIndex from filenames.

    Args:
        filenames: list
            A list of filenames.
        data_basename: string
            Data base name.
        time_format: string (optional, default="yyyymodd_hhmm")
            Specify file time format to extract date/time.
    Returns:
        files_datetime: Pandas DatetimeIndex
            DatetimeIndex for the files.
    """
    # Get string positions for each date/time item
    yyyy_idx = time_format.find("yyyy")
    mo_idx = time_format.find("mo")
    dd_idx = time_format.find("dd")
    hh_idx = time_format.find("hh")
    mm_idx = time_format.find("mm")
    ss_idx = time_format.find("ss")

    # Add basename character counts to get the actual date/time string positions
    nleadingchar = len(data_basename)
    yyyy_idx = nleadingchar + yyyy_idx
    mo_idx = nleadingchar + mo_idx
    dd_idx = nleadingchar + dd_idx
    hh_idx = nleadingchar + hh_idx if (hh_idx != -1) else None
    mm_idx = nleadingchar + mm_idx if (mm_idx != -1) else None
    ss_idx = nleadingchar + ss_idx if (ss_idx != -1) else None

    files_datetime = []
    # Loop over each file
    for ii, ifile in enumerate(filenames):
        fn = os.path.basename(ifile)
        year = fn[yyyy_idx:yyyy_idx+4]
        month = fn[mo_idx:mo_idx+2]
        day = fn[dd_idx:dd_idx+2]
        # If hour, minute, second is not in time_format, assume 0
        hour = fn[hh_idx:hh_idx+2] if (hh_idx is not None) else '00'
        minute = fn[mm_idx:mm_idx+2] if (mm_idx is not None) else '00'
        second = fn[ss_idx:ss_idx+2] if (ss_idx is not None) else '00'
        
        files_datetime.append(pd.to_datetime(f'{year}{month}{day}{hour}{minute}{second}', format='%Y%m%d%H%M%S'))

    # Convert list to Pandas DatetimeIndex
    files_datetime = pd.to_datetime(files_datetime)
    return files_datetime


def merge_files(trackmask_file, rain_file, out_file, sat_file=''):
    """
    Combine several sets of netCDF files into a same file.

    Parameters:
    ===========
    trackmask_file: string
        Cell track mask filename.
    rain_file: string
        Rain rate filename.
    out_file: string
        Output filename.
    sat_file: string <optional>
        Satellite cloud retrieval filename.

    Returns:
    ===========
    status: boolean
        True or False.
    """
    
    # Read convective mask file
    ds_trackmask = xr.open_dataset(trackmask_file)
        
    # Check if rain file exist
    if os.path.isfile(rain_file):

        # Read rain rate file
        drop_rainvars = ['time', 'point_x', 'point_y', 'point_y', 'point_latitude', 'point_longitude',
                        'origin_latitude', 'origin_longitude', 'lat', 'lon']
        ds_rain = xr.open_dataset(rain_file, drop_variables=drop_rainvars)

        # Combine the datasets
        dsout = xr.combine_by_coords([ds_trackmask, ds_rain], join='outer', combine_attrs='override')
        # Replace global attributes
        dsout.attrs = ds_trackmask.attrs
        dsout.attrs['rainrate_file'] = 1
        status = True
    else:
        # Make output DataSet the same as convective mask DataSet
        dsout = ds_trackmask

        # Create empty arrays for rain rate variables
        var_tmp = xr.DataArray(ds_trackmask['comp_ref'].data * np.nan, dims=('time','lat','lon'))
        dsout['taranis_composite_rain_rate'] = var_tmp
        dsout['taranis_composite_rain_rate_height'] = var_tmp
        dsout.attrs['rainrate_file'] = 0
        status = False

    # Add attributes to rain_rate variables
    dsout['taranis_composite_rain_rate'].attrs['units'] = 'mm/h'
    dsout['taranis_composite_rain_rate_height'].attrs['units'] = 'm'

    # Check if satellite file exist
    if os.path.isfile(sat_file):
        drop_satvars = ['time', 'lon', 'lat']
        ds_sat = xr.open_dataset(sat_file, drop_variables=drop_satvars)

        # Combine the datasets
        dsout = xr.combine_by_coords([dsout, ds_sat], join='outer', combine_attrs='override')
        dsout.attrs['satellite_file'] = 1
    else:
        dsout.attrs['satellite_file'] = 0


    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}
    # Update base_time variable dtype as 'double' for better precision
    # bt_dict = {'base_time': {'zlib':True, 'dtype':'float64'}}
    # encoding.update(bt_dict)

    # Write to netcdf file
    dsout.to_netcdf(path=out_file, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
    print(f'File saved: {out_file}')

    return status


if __name__ == '__main__':

    # Get configuration file name from input
    config_file = sys.argv[1]

    # Read configuration from yaml file
    stream = open(config_file, 'r')
    config = yaml.full_load(stream)
    run_parallel = config['run_parallel']
    n_workers = config['n_workers']
    threads_per_worker = config['threads_per_worker']
    time_window = config['time_window']
    start_datetime = config['start_datetime']
    end_datetime = config['end_datetime']
    trackmask_path = config['trackmask_path']
    rain_path = config['rain_path']
    sat_path = config['sat_path']
    out_path = config['out_path']
    trackmask_basename = config['trackmask_basename']
    rain_basename = config['rain_basename']
    sat_basename = config['sat_basename']
    out_basename = config['out_basename']

    os.makedirs(out_path, exist_ok=True)

    # Generate time marks within the start/end datetime
    file_datetimes = pd.date_range(start=start_datetime, end=end_datetime, freq='1D').strftime('%Y%m%d')
    # Find all convective mask files
    trackmask_files = []
    sat_files = []
    rain_files = []
    for tt in range(0, len(file_datetimes)):
        trackmask_files.extend(sorted(glob.glob(f'{trackmask_path}{trackmask_basename}{file_datetimes[tt]}*.nc')))
        rain_files.extend(sorted(glob.glob(f'{rain_path}{rain_basename}{file_datetimes[tt]}*.nc')))
        sat_files.extend(sorted(glob.glob(f'{sat_path}{sat_basename}{file_datetimes[tt]}*.nc')))

    # Get datetimes for all the files
    trackmask_datetimes = get_datetime_from_filename(
        trackmask_files,
        trackmask_basename,
        time_format="yyyymodd_hhmm",
    )
    rain_datetimes = get_datetime_from_filename(
        rain_files,
        rain_basename,
        time_format="yyyymodd.hhmmss",
    )
    sat_datetimes = get_datetime_from_filename(
        sat_files,
        sat_basename,
        time_format="yyyymodd.hhmmss",
    )

    # Initialize dask local cluster
    if run_parallel==1:
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
        client = Client(cluster)
        client.wait_for_workers(n_workers=n_workers, timeout=60)


    rain_index = pd.Index(rain_datetimes)
    sat_index = pd.Index(sat_datetimes)

    final_results = []
    # Loop over each radar file
    for ii in range(0, len(trackmask_files)):
        trackmask_file = trackmask_files[ii]

        # Make matching time and filename
        itime = trackmask_datetimes[ii]
        itimestr_radar = itime.strftime('%Y%m%d.%H%M%S')

        # Find the nearest matching rain file
        # itime_rain = rain_datetimes[rain_datetimes.get_loc(itime, method='nearest')]
        itime_rain = rain_datetimes[rain_index.get_indexer([itime], method='nearest')]
        # Calculate time difference in minutes
        ds_rain = abs(itime - itime_rain).total_seconds()/60.
        if ds_rain < time_window:
            itimesstr_rain = itime_rain.strftime('%Y%m%d.%H%M%S').item()
            rain_file = f'{rain_path}{rain_basename}{itimesstr_rain}.nc'
        else:
            rain_file = ''
            print(f'Missing rain file at this time: {itimestr_radar}')

        # Find the nearest matching satellite time
        # itime_sat = sat_datetimes[sat_datetimes.get_loc(itime, method='nearest')]
        itime_sat = sat_datetimes[sat_index.get_indexer([itime], method='nearest')]
        # Calculate time difference in minutes
        dt_sat = abs(itime - itime_sat).total_seconds()/60.
        # Cloeset satellite file must be within specified time window
        if dt_sat < time_window:
            itimesstr_sat = itime_sat.strftime('%Y%m%d.%H%M%S').item()
            sat_file = f'{sat_path}{sat_basename}{itimesstr_sat}.nc'
        else:
            sat_file = ''
            print(f'Missing satellite file at this time: {itimestr_radar}')

        # Output filename
        out_file = f'{out_path}{out_basename}{itimestr_radar}.nc'

        # Call function to merge files
        if run_parallel==0:
            result = merge_files(trackmask_file, rain_file, out_file, sat_file=sat_file)
        elif run_parallel==1:
            result = dask.delayed(merge_files)(trackmask_file, rain_file, out_file, sat_file=sat_file)
            final_results.append(result)
        else:
            sys.exit('Error: Valid run_parallel option not set!')
    
    if run_parallel==1:
        # Dask compute
        final_results = dask.compute(*final_results)