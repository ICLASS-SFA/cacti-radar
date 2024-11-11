import numpy as np
import xarray as xr
import pandas as pd
import glob, os, sys
import yaml
import dask
from dask.distributed import Client, LocalCluster


def get_file_datetime(filenames, basename):
    """
    Make Pandas datetime index from filenames.

    Parameters:
    ===========
    filenames: list
        List of input filenames
    basename: string
        File basename

    Returns:
    ===========
    files_datetime: Pandas DatetimeIndex
        DatetimeIndex for the input files.
    """
    nfiles = len(filenames)
    nleadingchar = np.array(len(basename)).astype(int)

    files_datetime = []
    for ii in range(0, nfiles):
        # Get filename, date, time, filename format: basenameYYYYMMDD.HHMMSS
        fname = os.path.basename(filenames[ii])
        yyyymmdd = fname[nleadingchar:nleadingchar+8]
        hhmmss = fname[nleadingchar+9:nleadingchar+15]
        files_datetime.append(pd.to_datetime(yyyymmdd + hhmmss,  format='%Y%m%d%H%M%S'))
    
    # Convert list to Pandas DatetimeIndex
    files_datetime = pd.to_datetime(files_datetime)
    
    return files_datetime

def merge_files(convmask_file, rain_file, out_file, sat_file=None):
    """
    Combine several sets of netCDF files into a same file.

    Parameters:
    ===========
    convmask_file: string
        Convective mask filename.
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
    ds_convmask = xr.open_dataset(convmask_file)
        
    # Check if rain file exist
    if os.path.isfile(rain_file) == True:      

        # Read rain rate file
        drop_rainvars = ['time', 'point_x', 'point_y', 'point_y', 'point_latitude', 'point_longitude',
                        'origin_latitude', 'origin_longitude', 'lat', 'lon']
        ds_rain = xr.open_dataset(rain_file, drop_variables=drop_rainvars)

        # Check if satellite file exist
        if sat_file is not None:
            drop_satvars = ['time', 'lon', 'lat']
            ds_sat = xr.open_dataset(sat_file, drop_variables=drop_satvars)

            # Combine the datasets
            dsout = xr.combine_by_coords([ds_convmask, ds_rain, ds_sat], join='outer', combine_attrs='override')
        else:
            # Combine the datasets
            dsout = xr.combine_by_coords([ds_convmask, ds_rain], join='outer', combine_attrs='override')

        # Replace global attributes
        dsout.attrs = ds_convmask.attrs

        dsout['taranis_composite_rain_rate'].attrs['units'] = 'mm/h'
        dsout['taranis_composite_rain_rate_height'].attrs['units'] = 'm'
        # attr_list = ['transform_history', 'history', 'source_filename', 'output_filename']
        # for ii in attr_list:
        #     del dsout.attrs[ii]

        # import pdb; pdb.set_trace()

        status = True
    else:
        print(f'Missing rain file: {rain_file}')
        # Make output DataSet the same as convective mask DataSet
        dsout = ds_convmask
        status = False

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
    convmask_path = config['convmask_path']
    rain_path = config['rain_path']
    sat_path = config['sat_path']
    out_path = config['out_path']
    convmask_basename = config['convmask_basename']
    rain_basename = config['rain_basename']
    sat_basename = config['sat_basename']
    out_basename = config['out_basename']

    os.makedirs(out_path, exist_ok=True)

    # Generate time marks within the start/end datetime
    file_datetimes = pd.date_range(start=start_datetime, end=end_datetime, freq='1D').strftime('%Y%m%d')
    # Find all convective mask files
    convmask_files = []
    sat_files = []
    for tt in range(0, len(file_datetimes)):
        convmask_files.extend(sorted(glob.glob(f'{convmask_path}{convmask_basename}{file_datetimes[tt]}*.nc')))
        sat_files.extend(sorted(glob.glob(f'{sat_path}{sat_basename}{file_datetimes[tt]}*.nc')))

    # Get datetimes for all the files
    convmask_datetimes = get_file_datetime(convmask_files, convmask_basename)
    sat_datetimes = get_file_datetime(sat_files, sat_basename)

    if run_parallel==0:
        # Loop over each convective mask file
        for ii in range(0, len(convmask_files)):
            convmask_file = convmask_files[ii]

            # Make matching time and filename
            itime = convmask_datetimes[ii]
            itimestr_radar = itime.strftime('%Y%m%d.%H%M%S')
            rain_file = f'{rain_path}{rain_basename}{itimestr_radar}.nc'

            # Find the nearest matching satellite time
            itime_sat = sat_datetimes[sat_datetimes.get_loc(itime, method='nearest')]
            # Calculate time difference in minutes
            dt_sat = abs(itime - itime_sat).total_seconds()/60.
            # Cloeset satellite file must be within specified time window
            if dt_sat < time_window:
                itimesstr_sat = itime_sat.strftime('%Y%m%d.%H%M%S')
                sat_file = f'{sat_path}{sat_basename}{itimesstr_sat}.nc'
            else:
                sat_file = None
                print(f'Missing satellite file at this time: {itimestr_radar}')

            # Output filename
            out_file = f'{out_path}{out_basename}{itimestr_radar}.nc'

            # Call function to merge files
            result = merge_files(convmask_file, rain_file, out_file, sat_file=sat_file)

    elif run_parallel==1:
        print(f'Parallel version by dask')
        
        # Initialize dask
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
        client = Client(cluster)

        final_results = []
        # Loop over each convective mask file
        for ii in range(0, len(convmask_files)):
            convmask_file = convmask_files[ii]

            # Make matching time and filename
            itime = convmask_datetimes[ii]
            itimestr_radar = itime.strftime('%Y%m%d.%H%M%S')
            rain_file = f'{rain_path}{rain_basename}{itimestr_radar}.nc'

            # Find the nearest matching satellite time
            itime_sat = sat_datetimes[sat_datetimes.get_loc(itime, method='nearest')]
            # Calculate time difference in minutes
            dt_sat = abs(itime - itime_sat).total_seconds()/60.
            # Cloeset satellite file must be within specified time window
            if dt_sat < time_window:
                itimesstr_sat = itime_sat.strftime('%Y%m%d.%H%M%S')
                sat_file = f'{sat_path}{sat_basename}{itimesstr_sat}.nc'
            else:
                sat_file = None
                print(f'Missing satellite file at this time: {itimestr_radar}')

            # Output filename
            out_file = f'{out_path}{out_basename}{itimestr_radar}.nc'

            # Call function to merge files
            result = dask.delayed(merge_files)(convmask_file, rain_file, out_file, sat_file=sat_file)
            final_results.append(result)

        # Collect results from Dask
        final_results = dask.compute(*final_results)