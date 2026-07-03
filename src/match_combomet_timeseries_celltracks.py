import numpy as np
import sys
import yaml
import time
import xarray as xr
import pandas as pd

if __name__ == '__main__':

    # Get configuration file name from input
    config_file = sys.argv[1]
    # Read configuration from yaml file
    stream = open(config_file, 'r')
    config = yaml.full_load(stream)

    startdate = config['startdate']
    enddate = config['enddate']
    stats_path = config['stats_path']
    combomet_file = config['combomet_file']

    # Number of relative hours prior to track start to make time series
    nhours = 3
    # Frequency of time series in minutes
    freq_min = 15

    # Maximum time difference allowed to match the datasets
    time_window = config['time_window']  # [second]
    print(f'Max time window allowed to match the datasets: {time_window} s')

    # Input file basenames
    stats_filebase = 'trackstats_'

    # Output statistics filename
    output_path = stats_path
    output_filename = f'{output_path}combomet_parameters_celltrack_{startdate}_{enddate}.nc'

    # Track statistics file dimension names
    trackdimname = 'tracks'
    timedimname = 'times'
    relative_time_dimname = 'reltime'

    # Track statistics file
    trackstats_file = f'{stats_path}{stats_filebase}{startdate}_{enddate}.nc'

    # Read track statistics file
    print(trackstats_file)
    dsstats = xr.open_dataset(trackstats_file, decode_times=True)
    ntracks = dsstats.sizes[trackdimname]
    tracks_coord = dsstats.coords[trackdimname]
    stats_basetime = dsstats['base_time']
    # Get cell initiation time
    stats_basetime0 = stats_basetime.sel(times=0).data
    dsstats.close()

    print(f'Total Number of Tracks: {ntracks}')


    # Read combomet file
    dscombomet = xr.open_dataset(combomet_file, decode_times=True)
    combomet_basetime = dscombomet.time.values
    # Convert combomet times to Pandas datetime
    combomet_times = pd.to_datetime(combomet_basetime)

    # Create a variable list, excluding variables with no time dimension
    combomet_var_names = [v for v in dscombomet.data_vars if 'time' in dscombomet[v].dims]
    # Add time to the list (since time is a coordinate, it is not included in the data_vars)
    combomet_var_names.append('time')


    # Calculate the number of times to save prior to initiation
    ntimes_per_hour = np.round(60. / freq_min).astype(int)
    ntimes_prior = np.round(nhours * ntimes_per_hour + 1).astype(int)

    # Make relative time coordinate
    relative_time_coord = np.linspace(-1*(ntimes_prior-1), 0, ntimes_prior, dtype=int)

    # Create match time index arrays
    matchindex = np.zeros((ntracks, ntimes_prior), dtype=np.int64)

    # Create a dictionary with variable name as key, and output arrays as values
    # Define the first dictionary entry as 'time'
    out_vars = {'time': np.full((ntracks, ntimes_prior), dtype=np.float64, fill_value=np.nan)}
    # Loop over variable list to create the dictionary entry
    for ivar in combomet_var_names:
        out_vars[ivar] = np.full((ntracks, ntimes_prior), dtype=np.float64, fill_value=np.nan)


    # Loop over each cell track to find match combomet time
    for itrack in range(0, ntracks):
        # Calculate start time prior to initiation
        time0_start = stats_basetime0[itrack] - pd.offsets.Hour(nhours)
        # End time is at initiation
        time0_end = stats_basetime0[itrack]
        # Generate time series leading up to initiation
        prior_times = np.array(pd.date_range(time0_start, time0_end, freq=f'{freq_min:.0f}min'))

        # Initialize a list to store the time indices
        closest_indices = []
        # Loop through each time in prior_times
        for prior_time in prior_times:
            # Calculate the absolute time difference in seconds
            time_diffs = np.abs((combomet_times - prior_time).total_seconds())

            # Find the index of the minimum time difference
            min_diff_index = time_diffs.argmin()

            # Check if the minimum difference is within the specified threshold
            if time_diffs[min_diff_index] <= time_window:
                closest_indices.append(min_diff_index)
            else:
                closest_indices.append(-1)  # Append -1 if no match within threshold

        # Save match indices
        matchindex[itrack, :] = np.array(closest_indices)


    # Get indices of non-matched tracks
    skip_tracks, skip_times = np.where(matchindex == -1)
    nskip_tracks = len(skip_tracks)
    print(f'Number of no matched tracks: {nskip_tracks}')

    # Apply the matched time index to each of the variables
    for ivar in combomet_var_names:
        # Loop over each prior times
        for itime in range(ntimes_prior):
            # Find valid indices (!= -1)
            _index = matchindex[:,itime]
            valid = (_index != -1)
            if len(valid) > 0:
                if ivar == 'time':
                    # Convert datetime64[ns] dtype to np.float64 representing
                    # seconds since the Unix epoch (1970-01-01 00:00:00)
                    _itimes = dscombomet[ivar].values[_index]
                    _itimes[~valid] = np.datetime64('NaT')
                    _itimes_float64 = _itimes.astype('datetime64[s]').astype(np.float64)
                    out_vars[ivar][:,itime] = _itimes_float64
                else:
                    out_vars[ivar][:,itime] = dscombomet[ivar].values[_index]
                    out_vars[ivar][~valid,itime] = np.nan


    print(f'Writing output netCDF file ...')
    var_dict = {}
    # Define output variable dictionary
    for key, value in out_vars.items():
        var_dict[key] = ([trackdimname, relative_time_dimname], value)

    # Define coordinate dictionary
    coord_dict = {
        trackdimname: ([trackdimname], tracks_coord.data, tracks_coord.attrs),
        relative_time_dimname: ([relative_time_dimname], relative_time_coord),
    }

    # Define global attributes
    gattr_dict = {
        'title':  'Combomet-derived surface parameters matched to tracked cells',
        'Institution': 'Pacific Northwest National Laboratory',
        'Contact': 'Enoch Jo, enochjo2009@gmail.com',
        'Created_on':  time.ctime(time.time()),
        'source_track_file': trackstats_file,
        'source_combomet_file': combomet_file,
        'startdate': startdate,
        'enddate': enddate,
    }

    # Define xarray dataset
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Copy original variable attributes
    for ivar in combomet_var_names:
        dsout[ivar].attrs = dscombomet[ivar].attrs

    dsout['time'].attrs['long_name'] = 'Epoch time of closest combomet observation'
    dsout['time'].attrs['units'] = 'seconds since 1970-01-01 00:00:00'

    dsout[relative_time_dimname].attrs['long_name'] = 'Relative combomet time prior to track initiation'

    # Set encoding/compression for all variables
    comp = dict(zlib=True, dtype='float32')
    encoding = {var: comp for var in dsout.data_vars}
    # Replace time encoding
    encoding['time'] = {'dtype': 'double', 'zlib': True}

    # Write netcdf file
    dsout.to_netcdf(path=output_filename, mode='w', format='NETCDF4', unlimited_dims=trackdimname, encoding=encoding)
    print(f'Output saved: {output_filename}')
