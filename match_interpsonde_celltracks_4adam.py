import numpy as np
import sys
import yaml
import time, datetime, calendar
import pytz
import xarray as xr

if __name__ == '__main__':

    # Get configuration file name from input
    config_file = sys.argv[1]
    # Read configuration from yaml file
    stream = open(config_file, 'r')
    config = yaml.full_load(stream)

    startdate = config['startdate']
    enddate = config['enddate']
    stats_path = config['stats_path']
    sonde_path = config['sonde_path']

    # Maximum time difference allowed to match the datasets
    time_window = 10  # [second]
    print(f'Max time window allowed to match the datasets: {time_window} s')
  
    # Input file basenames
    stats_filebase = 'stats_tracknumbersv1.0_'

    # Output statistics filename
    output_path = stats_path
    output_filename = f'{output_path}interpsonde_celltrack_{startdate}_{enddate}.nc'

    # Track statistics file dimension names
    trackdimname = 'tracks'
    timedimname = 'times'
    relative_time_dimname = 'reltime'

    # Track statistics file
    trackstats_file = f'{stats_path}{stats_filebase}{startdate}_{enddate}.nc'
    muparcel_file = f'{sonde_path}CACTI_M1_interpsonde_muparcel_stats.nc'
    uvq_file = f'{sonde_path}CACTI_M1_interpsonde_wind_humidity_indices.nc'


    # Read track statistics file
    print(trackstats_file)
    dsstats = xr.open_dataset(trackstats_file, decode_times=False)
    ntracks = dsstats.dims[trackdimname]
    # ntimes = dsstats.dims[timedimname]
    stats_basetime = dsstats['basetime']
    basetime_units = dsstats['basetime'].units
    # Get cell initiation time
    stats_basetime0 = stats_basetime.sel(times=0).data
    dsstats.close()

    print(f'Total Number of Tracks: {ntracks}')


    # Read sonde MU parcel file
    dsmup = xr.open_dataset(muparcel_file, decode_times=False)
    ntimes_mup = dsmup.dims['time']
    # sonde_basetime = dsmup.time.values

    # Read sonde wind/humidity file
    dsuvq = xr.open_dataset(uvq_file, decode_times=False)
    ntimes_uvq = dsuvq.dims['time']

    # Double check to make sure the number of times between the sonde files is the same
    # The assumption is that both files have the exact same times
    if (ntimes_uvq != ntimes_mup):
        print('Error: number of times not the same between the two sonde files!')
        sys.exti()

    # Merge the two sonde datasets
    dsmup = xr.merge([dsmup, dsuvq], join='outer')

    # Calculate sonde basetime since it is not in the sonde data file
    year_sonde = dsmup.year.data
    month_sonde = dsmup.month.data
    day_sonde = dsmup.day.data
    hour_sonde = dsmup.hour.data
    minute_sonde = dsmup.minute.data
    seconds_sonde = dsmup.seconds.data
    sonde_basetime = np.full(ntimes_mup, dtype=float, fill_value=np.nan)
    for tt in range(0, ntimes_mup):
        sonde_basetime[tt] = calendar.timegm(datetime.datetime(
            year_sonde[tt], month_sonde[tt], day_sonde[tt], hour_sonde[tt], minute_sonde[tt], 0, tzinfo=pytz.UTC).timetuple())
    
    # Replace sonde dataset time variable
    sonde_basetime_xr = xr.DataArray(sonde_basetime, coords={'time':sonde_basetime}, dims=('time'), attrs={'units':basetime_units})
    dsmup['time'] = sonde_basetime

    # Create a variable list
    sonde_var_names = list(dsmup.data_vars.keys())
    # Add time to the list (since time is a coordinate, it is not included in the data_vars)
    sonde_var_names.append('time')
    # Drop yyyymmdd, hhmmss variables from the list (not sure how to handle char arrays yet)
    sonde_var_names.remove('year')
    sonde_var_names.remove('month')
    sonde_var_names.remove('day')
    sonde_var_names.remove('hour')
    sonde_var_names.remove('minute')
    sonde_var_names.remove('seconds')
    # import pdb; pdb.set_trace()

    # Number of relative sonde time to track initiation (-3, -2, -1, 0 hour)
    nreltime = 4

    # Create a dictionary with variable name as key, and output arrays as values
    # Define the first dictionary entry as 'time'
    out_vars = {'time': np.full((ntracks, nreltime), dtype=np.float, fill_value=np.nan)}
    # Loop over variable list to create the dictionary entry
    for ivar in sonde_var_names:
        out_vars[ivar] = np.full((ntracks, nreltime), dtype=np.float, fill_value=np.nan)

    # Create match time index arrays
    matchindex = np.zeros(ntracks, dtype=np.int)
    matchindex_1h = np.zeros(ntracks, dtype=np.int)
    matchindex_2h = np.zeros(ntracks, dtype=np.int)
    matchindex_3h = np.zeros(ntracks, dtype=np.int)
    skip_tracks = []
    skip_tracks_1h = []
    skip_tracks_2h = []
    skip_tracks_3h = []

    one_hour = 3600  # seconds in one hour
    counts = 0

    # Loop over each cell track to find match sonde time
    for tt in range(0, ntracks):
        # Tracks at initiation time
        matchindex[tt] = np.argmin(np.abs(sonde_basetime - stats_basetime0[tt]))
        # Initiation time -1, -2, -3 hour 
        matchindex_1h[tt] = np.argmin(np.abs(sonde_basetime[0:matchindex[tt]] - (stats_basetime0[tt] - one_hour*1))) 
        matchindex_2h[tt] = np.argmin(np.abs(sonde_basetime[0:matchindex[tt]] - (stats_basetime0[tt] - one_hour*2)))
        matchindex_3h[tt] = np.argmin(np.abs(sonde_basetime[0:matchindex[tt]] - (stats_basetime0[tt] - one_hour*3)))
        
        # Check if time difference is larger than defined time_window, if so save the track index
        if (sonde_basetime[matchindex[tt]] - stats_basetime0[tt] > time_window):
            print(f'No match sonde time found: {stats_basetime0[tt]}')
            skip_tracks.append(tt)
            continue
        
        if (sonde_basetime[matchindex_1h[tt]] - stats_basetime0[tt] + one_hour > time_window):
            print(f'No match sonde time (-1h) found: {stats_basetime0[tt]}')
            skip_tracks_1h.append(tt)
            continue
            
        if (sonde_basetime[matchindex_2h[tt]] - stats_basetime0[tt] + 2*one_hour > time_window):
            print(f'No match sonde time (-2h) found: {stats_basetime0[tt]}')
            skip_tracks_2h.append(tt)
            continue
            
        if (sonde_basetime[matchindex_3h[tt]] - stats_basetime0[tt] + 3*one_hour > time_window):
            print(f'No match sonde time (-3h) found: {stats_basetime0[tt]}')
            skip_tracks_3h.append(tt)
            continue

    nskip_tracks = len(skip_tracks)
    nskip_tracks_1h = len(skip_tracks_1h)
    nskip_tracks_2h = len(skip_tracks_2h)
    nskip_tracks_3h = len(skip_tracks_3h)
    print(f'Number of skipped tracks: {nskip_tracks}')
    print(f'Number of skipped tracks (-1h): {nskip_tracks_1h}')
    print(f'Number of skipped tracks (-2h): {nskip_tracks_2h}')
    print(f'Number of skipped tracks (-3h): {nskip_tracks_3h}')

    # Apply the matched time index to each of the variables
    for ivar in sonde_var_names:
        out_vars[ivar][:,0] = dsmup[ivar].values[matchindex]
        out_vars[ivar][:,1] = dsmup[ivar].values[matchindex_1h]
        out_vars[ivar][:,2] = dsmup[ivar].values[matchindex_2h]
        out_vars[ivar][:,3] = dsmup[ivar].values[matchindex_3h]

    # Filter skip_tracks if there is any
    for ivar in sonde_var_names:
        if nskip_tracks > 0:
            out_vars[ivar][skip_tracks,0] = np.NaN
        if nskip_tracks_1h > 0:
            out_vars[ivar][skip_tracks_1h,1] = np.NaN
        if nskip_tracks_2h > 0:
            out_vars[ivar][skip_tracks_2h,1] = np.NaN
        if nskip_tracks_3h > 0:
            out_vars[ivar][skip_tracks_3h,1] = np.NaN
    
    print(f'Writing output netCDF file ...')
    varlist = {}
    # Define output variable dictionary
    for key, value in out_vars.items():
        varlist[key] = ([trackdimname, relative_time_dimname], value)
        
    # Define coordinate list
    coordlist = {trackdimname: ([trackdimname], np.arange(0, ntracks)), \
                relative_time_dimname: ([relative_time_dimname], [0,-1,-2,-3])
                }

    # Define global attributes
    gattrlist = {'title':  'InterpSonde parameters matched to tracked cells', \
                'Institution': 'Pacific Northwest National Laboratoy', \
                'Contact': 'Zhe Feng, zhe.feng@pnnl.gov', \
                'Created_on':  time.ctime(time.time()), \
                'source_track_file': trackstats_file, \
                'source_sonde_file': muparcel_file, \
                'startdate': startdate, \
                'enddate': enddate, \
                }

    # Define xarray dataset
    dsout = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)

    # Define coordinate attributes
    dsout[trackdimname].attrs = dsstats[trackdimname].attrs

    # Copy original variable attributes
    for ivar in sonde_var_names:
        dsout[ivar].attrs = dsmup[ivar].attrs

    dsout['time'].attrs['long_name'] = 'Epoch time of closest interpsonde'
    dsout['time'].attrs['units'] = basetime_units

    dsout[relative_time_dimname].attrs['long_name'] = 'Relative sonde time prior to track initiation'
    dsout[relative_time_dimname].attrs['units'] = 'hour'

    # Set encoding/compression for all variables
    comp = dict(zlib=True, dtype='float32')
    encodelist = {var: comp for var in dsout.data_vars}

    # Write netcdf file
    dsout.to_netcdf(path=output_filename, mode='w', format='NETCDF4_CLASSIC', unlimited_dims=trackdimname, encoding=encodelist)
    print(f'Output saved: {output_filename}')




