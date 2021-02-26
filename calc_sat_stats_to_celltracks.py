"""
Calculates cloud statistics from satellite data for tracked convective cells.
The cloud statistics are written to netCDF file matching the cell track statistics file format.
"""
import numpy as np
import os, sys, glob
import time, datetime, calendar
from pytz import utc
import yaml
import xarray as xr
import dask
from dask.distributed import Client, LocalCluster
from calc_sat_cellstats_singlefile import calc_sat_cellstats_singlefile

#########################################################
def calc_basetime(filelist, filebase):
    """
    Calculates basetime (Epoch Time) and a filename dictionary from a list of filenames.
    The basetime uses year, month, day, hour, minute but set all seconds to 0.

    Parameters:
    ===========
    filelist: list
        A list of input file names
    filebase: string
        Basename of the files

    Returns:
    ===========
    basetime: int array
        Epoch time corresponding to the input files.
    file_dict: dictionary
        Direction key by basetime and value is the file names
    """
    nfiles = len(filelist)
    prelength = len(filebase)
    basetime = np.full(nfiles, -999, dtype=int)
    file_dict = {}
    for ifile in range(nfiles):
        fname = os.path.basename(filelist[ifile])
        # File name format: basename_yyyymmdd.hhmm
        TEMP_filetime = datetime.datetime(int(fname[prelength:(prelength+4)]), 
                                            int(fname[prelength+4:(prelength+6)]), 
                                            int(fname[prelength+6:(prelength+8)]),
                                            int(fname[prelength+9:(prelength+11)]), 
                                            int(fname[prelength+11:(prelength+13)]), 0, tzinfo=utc)
        basetime[ifile] = calendar.timegm(TEMP_filetime.timetuple())
        file_dict[basetime[ifile]] = filelist[ifile]
    return basetime, file_dict


if __name__ == '__main__':

    #########################################################
    # Load MCS track stats
    # print('Loading track stats file')
    print((time.ctime()))

    # Get configuration file name from input
    config_file = sys.argv[1]
    # Read configuration from yaml file
    stream = open(config_file, 'r')
    config = yaml.full_load(stream)

    run_parallel = config['run_parallel']
    n_workers = config['n_workers']
    threads_per_worker = config['threads_per_worker']
    startdate = config['startdate']
    enddate = config['enddate']
    time_window = config['time_window']
    stats_path = config['stats_path']
    pixelfile_path = config['pixelfile_path']
    satfile_path = config['satfile_path']

    output_path = stats_path

    # Input file basenames
    stats_filebase = 'stats_tracknumbersv1.0_'
    pixel_filebase = 'celltracks_'
    sat_filebase = 'corvisstpx2drectg16v4minnisX1.regrid2csapr2gridded.c1.'

    # Output statistics filename
    output_filename = f'{output_path}stats_goes16_{startdate}_{enddate}.nc'

    # Track statistics file dimension names
    trackdimname = 'tracks'
    timedimname = 'times'

    # Track statistics file
    trackstats_file = f'{stats_path}{stats_filebase}{startdate}_{enddate}.nc'
    # Find all pixel-level files
    pixelfilelist = sorted(glob.glob(f'{pixelfile_path}{pixel_filebase}*.nc'))
    nfiles = len(pixelfilelist)
    # Find all satellite files
    satfilelist = sorted(glob.glob(f'{satfile_path}{sat_filebase}*.nc'))
    nsatfiles = len(satfilelist)
    
    # Get basetime and from the satellite and pixel files
    sat_basetime, satfile_dict = calc_basetime(satfilelist, sat_filebase)
    pixel_basetime, pixelfile_dict = calc_basetime(pixelfilelist, pixel_filebase)

    # Find matching satellite files for each pixel file
    match_satfilelist = [''] * nfiles
    # match_pixelbasetime = np.full(nfiles, np.NaN, dtype=np.float)
    for ifile in range(nfiles):
        # # Check if the pixel file basetime key is in the satellite dictionary
        # # This assumes the pixel file basetime is exactly the same with the satellite basetime
        # if pixel_basetime[ifile] in satfile_dict:
        #     match_satfilelist[ifile] = satfile_dict[pixel_basetime[ifile]]

        # Find satellite time closest to the pixel file time and get the index
        # Save the filename if time difference is < time_window
        idx = np.argmin(np.abs(sat_basetime - pixel_basetime[ifile]))        
        if np.abs(sat_basetime[idx] - pixel_basetime[ifile]) < time_window:
            match_satfilelist[ifile] = satfilelist[idx]
        else:
            print(f'No match file found for: {pixelfilelist[ifile]}')

    # import pdb; pdb.set_trace()


    # Read track statistics file
    print(trackstats_file)
    dsstats = xr.open_dataset(trackstats_file, decode_times=False)
    ntracks = dsstats.dims[trackdimname]
    ntimes = dsstats.dims[timedimname]
    stats_basetime = dsstats['basetime'].values
    basetime_units = dsstats['basetime'].units
    cell_area = dsstats['cell_area'].values
    pixel_radius = dsstats.attrs['pixel_radius_km']
    dsstats.close()

    print(f'Total Number of Tracks: {ntracks}')
    

    # Create new statistics variables
    # cell_area_2 is to double check with that from the track statistics file to make sure they match exactly
    cell_area_2 = np.full((ntracks, ntimes), np.nan, dtype=float)
    ctt_min = np.full((ntracks, ntimes), np.nan, dtype=float)
    tir_min = np.full((ntracks, ntimes), np.nan, dtype=float)
    cth_max = np.full((ntracks, ntimes), np.nan, dtype=float)
    ctp_min = np.full((ntracks, ntimes), np.nan, dtype=float)
    area_liq = np.full((ntracks, ntimes), np.nan, dtype=float)
    area_ice = np.full((ntracks, ntimes), np.nan, dtype=float)
    lwp_max = np.full((ntracks, ntimes), np.nan, dtype=float)
    iwp_max = np.full((ntracks, ntimes), np.nan, dtype=float)

    ##############################################################
    # Call function to calculate statistics
    nmatchcloud_all = []
    matchindices_all = []
    final_results = []
    if run_parallel==0:
        # Loop over each pixel-file and call function to calculate
        for ifile in range(nfiles):
            # Find all matching time indices from track stats file to the current pixel file
            matchindices = np.array(np.where(np.abs(stats_basetime - pixel_basetime[ifile]) < time_window))
            # The returned match indices are for [tracks, times] dimensions respectively
            idx_track = matchindices[0]
            idx_time = matchindices[1]
            # Save matchindices for the current pixel file to the overall list
            nmatchcloud_all.append(len(idx_track))
            matchindices_all.append(matchindices)

            # iresult = calc_sat_cellstats_singlefile(pixelfilelist[ifile], match_satfilelist[ifile], pixel_filebase, stats_basetime, pixel_radius)
            iresult = calc_sat_cellstats_singlefile(
                        pixelfilelist[ifile], 
                        match_satfilelist[ifile], 
                        pixel_filebase, 
                        idx_track, 
                        pixel_radius
                        )
            final_results.append(iresult)
            # import pdb; pdb.set_trace()

    elif run_parallel==1:
        print(f'Parallel version by dask')
        
        # Initialize dask
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
        client = Client(cluster)

        # Loop over each pixel-file and call function to calculate
        for ifile in range(nfiles):
            # Find all matching time indices from robust MCS stats file to the current pixel file
            matchindices = np.array(np.where(np.abs(stats_basetime - pixel_basetime[ifile]) < 1))
            # The returned match indices are for [tracks, times] dimensions respectively
            idx_track = matchindices[0]
            idx_time = matchindices[1]
            # Save matchindices for the current pixel file to the overall list
            nmatchcloud_all.append(len(idx_track))
            matchindices_all.append(matchindices)

            iresult = dask.delayed(calc_sat_cellstats_singlefile)(
                        pixelfilelist[ifile], 
                        match_satfilelist[ifile], 
                        pixel_filebase, 
                        idx_track, 
                        pixel_radius
                        )
            # iresult = delayed(calc_sat_cellstats_singlefile)(pixelfilelist[ifile], match_satfilelist[ifile], pixel_filebase, stats_basetime, pixel_radius)
            final_results.append(iresult)
            
        # Collect results from Dask
        print("Computing statistics ...")
        final_results = dask.compute(*final_results)

    # import pdb; pdb.set_trace()

    # Now that all calculations for each pixel file is done, put the results back to the tracks format
    # Loop over the returned statistics list, organized by file
    for ifile in range(nfiles):
        # Get the results from the current file
        vars = final_results[ifile]
        if (vars is not None):
            # # The returned results from calc_sat_cellstats_singlefile are:
            # # nmatchcloud, matchindices, var1, [var2, var3, ...]
            # # Get the returned variables in order
            # nmatchcloudtmp = vars[0]
            # matchindicestmp = vars[1]
            # Get the match track indices (matchindicestmp contains: [track_index, time_index]) for this pixel file
            nmatchcloudtmp = nmatchcloud_all[ifile]
            matchindicestmp = matchindices_all[ifile]
            # import pdb; pdb.set_trace()

            # if nmatchcloudtmp > 0:

            # Loop over each matched cloud, and put them back in the track variable
            for imatch in range(nmatchcloudtmp):
                # matchindices are in [tracks, times]
                cell_area_2[matchindicestmp[0,imatch],matchindicestmp[1,imatch]] = vars[1][imatch]
                ctt_min[matchindicestmp[0,imatch],matchindicestmp[1,imatch]] = vars[2][imatch]
                tir_min[matchindicestmp[0,imatch],matchindicestmp[1,imatch]] = vars[3][imatch]
                cth_max[matchindicestmp[0,imatch],matchindicestmp[1,imatch]] = vars[4][imatch]
                ctp_min[matchindicestmp[0,imatch],matchindicestmp[1,imatch]] = vars[5][imatch]
                area_liq[matchindicestmp[0,imatch],matchindicestmp[1,imatch]] = vars[6][imatch]
                area_ice[matchindicestmp[0,imatch],matchindicestmp[1,imatch]] = vars[7][imatch]
                lwp_max[matchindicestmp[0,imatch],matchindicestmp[1,imatch]] = vars[8][imatch]
                iwp_max[matchindicestmp[0,imatch],matchindicestmp[1,imatch]] = vars[9][imatch]


    ##################################
    # Write to netcdf
    print('Writing output netcdf ... ')
    t0_write = time.time()

    # Define variable list
    varlist = {'basetime': ([trackdimname, timedimname], stats_basetime), \
               'cell_area': ([trackdimname, timedimname], cell_area_2), \
               'cloud_top_temperature_min': ([trackdimname, timedimname], ctt_min), \
                'temperature_ir_min': ([trackdimname, timedimname], tir_min), \
                'cloud_top_height_max': ([trackdimname, timedimname], cth_max), \
                'cloud_top_pressure_min': ([trackdimname, timedimname], ctp_min), \
                'area_liquid': ([trackdimname, timedimname], area_liq), \
                'area_ice': ([trackdimname, timedimname], area_ice), \
                'lwp_max': ([trackdimname, timedimname], lwp_max), \
                'iwp_max': ([trackdimname, timedimname], iwp_max), \
              }

    # Define coordinate list
    coordlist = {trackdimname: ([trackdimname], np.arange(0, ntracks)), \
                 timedimname: ([timedimname], np.arange(0, ntimes)), \
                }

    # Define global attributes
    gattrlist = {'title':  'Track statistics', \
                 'Institution': 'Pacific Northwest National Laboratoy', \
                 'Contact': 'Zhe Feng, zhe.feng@pnnl.gov', \
                 'Created_on':  time.ctime(time.time()), \
                 'source_trackfile': trackstats_file, \
                 'startdate': startdate, \
                 'enddate': enddate, \
                }
    # Define xarray dataset
    dsout = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)

    dsout.basetime.attrs['long_name'] = 'Epoch time of each cell in a track'
    dsout.basetime.attrs['standard_name'] = 'time'
    dsout.basetime.attrs['units'] = basetime_units
    dsout.cell_area.attrs['long_name'] = 'Area of the convective cell in a track'
    dsout.cell_area.attrs['units'] = 'km^2'
    dsout.cloud_top_temperature_min.attrs['long_name'] = 'Minimum cloud top temperature in a track'
    dsout.cloud_top_temperature_min.attrs['units'] = 'K'
    dsout.temperature_ir_min.attrs['long_name'] = 'Minimum IR temperature in a track'
    dsout.temperature_ir_min.attrs['units'] = 'K'
    dsout.cloud_top_height_max.attrs['long_name'] = 'Maximum cloud top height in a track'
    dsout.cloud_top_height_max.attrs['units'] = 'km'
    dsout.cloud_top_pressure_min.attrs['long_name'] = 'Minimum cloud top pressure in a track'
    dsout.cloud_top_pressure_min.attrs['units'] = 'hPa'
    dsout.area_liquid.attrs['long_name'] = 'Area of liquid cloud-top in a track'
    dsout.area_liquid.attrs['units'] = 'km^2'
    dsout.area_ice.attrs['long_name'] = 'Area of ice cloud-top in a track'
    dsout.area_ice.attrs['units'] = 'km^2'
    dsout.lwp_max.attrs['long_name'] = 'Maximum liquid water path in a track'
    dsout.lwp_max.attrs['units'] = 'g/m^2'
    dsout.iwp_max.attrs['long_name'] = 'Maximum ice water path in a track'
    dsout.iwp_max.attrs['units'] = 'g/m^2'

    # Specify encoding list
    fillval = -9999
    var_float_encode = {'dtype':'float32', 'zlib':True, '_FillValue': np.nan}
    var_int_encode = {'dtype': 'int', 'zlib':True, '_FillValue': fillval}
    encodelist = {#'lifetime': var_int_encode, \
                  #     'basetime': {'zlib':True, 'units': basetime_units}, \
                'basetime': {'zlib':True}, \
                'cell_area': var_float_encode, \
                'cloud_top_temperature_min': var_float_encode, \
                'temperature_ir_min': var_float_encode, \
                'cloud_top_height_max': var_float_encode, \
                'cloud_top_pressure_min': var_float_encode, \
                'area_liquid': var_float_encode, \
                'area_ice': var_float_encode, \
                'lwp_max': var_float_encode, \
                'iwp_max': var_float_encode, \
                 }

    # Write netcdf file
    dsout.to_netcdf(path=output_filename, mode='w', format='NETCDF4_CLASSIC', unlimited_dims=trackdimname, encoding=encodelist)
    print(f'Output saved: {output_filename}')