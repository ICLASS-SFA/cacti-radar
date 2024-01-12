import numpy as np
import sys
import yaml
import time
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
    sonde_file = config['sonde_file']

    # Maximum time difference allowed to match the datasets
    time_window = 30  # [second]
    print(f'Max time window allowed to match the datasets: {time_window} s')

    # Input file basenames
    # stats_filebase = 'stats_tracknumbersv1.0_'
    stats_filebase = 'trackstats_'

    # Output statistics filename
    output_path = stats_path
    output_filename = f'{output_path}interpsonde_parameters_celltrack_{startdate}_{enddate}.nc'

    # Track statistics file dimension names
    trackdimname = 'tracks'
    timedimname = 'times'
    relative_time_dimname = 'reltime'

    # Track statistics file
    trackstats_file = f'{stats_path}{stats_filebase}{startdate}_{enddate}.nc'

    # Read track statistics file
    print(trackstats_file)
    dsstats = xr.open_dataset(trackstats_file, decode_times=False)
    ntracks = dsstats.dims[trackdimname]
    # ntimes = dsstats.dims[timedimname]
    stats_basetime = dsstats['base_time']
    basetime_units = dsstats['base_time'].units
    # Get cell initiation time
    stats_basetime0 = stats_basetime.sel(times=0).data
    dsstats.close()

    print(f'Total Number of Tracks: {ntracks}')


    # Read sonde MU parcel file
    dsmup = xr.open_dataset(sonde_file, decode_times=False)
    sonde_basetime = dsmup.time.values

    # Create a variable list
    sonde_var_names = list(dsmup.data_vars.keys())
    # Add time to the list (since time is a coordinate, it is not included in the data_vars)
    sonde_var_names.append('time')
    # Drop yyyymmdd, hhmmss variables from the list (not sure how to handle char arrays yet)
    sonde_var_names.remove('yyyymmdd')
    sonde_var_names.remove('hhmmss')


    # Number of relative sonde time to track initiation (-3, -2, -1, 0 hour)
    nreltime = 4

    # Create a dictionary with variable name as key, and output arrays as values
    # Define the first dictionary entry as 'time'
    out_vars = {'time': np.full((ntracks, nreltime), dtype=np.float64, fill_value=np.nan)}
    # Loop over variable list to create the dictionary entry
    for ivar in sonde_var_names:
        out_vars[ivar] = np.full((ntracks, nreltime), dtype=np.float64, fill_value=np.nan)

    # Create match time index arrays
    matchindex = np.zeros(ntracks, dtype=np.int64)
    matchindex_1h = np.zeros(ntracks, dtype=np.int64)
    matchindex_2h = np.zeros(ntracks, dtype=np.int64)
    matchindex_3h = np.zeros(ntracks, dtype=np.int64)
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
                'source_sonde_file': sonde_file, \
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




