"""
Calculates 3D radar cell statistics from gridded PPI data for tracked convective cells.
The 3D cell statistics are written to netCDF file matching the cell track statistics file format.
"""
import numpy as np
import os, sys, glob
import time, datetime, calendar
from pytz import utc
import yaml
import xarray as xr
import dask
from dask.distributed import Client, LocalCluster
from calc_3d_cellstats_singlefile import calc_3d_cellstats_singlefile

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
    ppifile_path = config['ppifile_path']

    output_path = stats_path

    # Input file basenames
    stats_filebase = 'stats_tracknumbersv1.0_'
    pixel_filebase = 'celltracks_'
    ppi_filebase = 'taranis_corcsapr2cfrppiqcM1.c1.'

    # Output statistics filename
    output_filename = f'{output_path}stats_3d_ppi_{startdate}_{enddate}.nc'

    # Track statistics file dimension names
    trackdimname = 'tracks'
    timedimname = 'times'
    zdimname = 'z'

    # Track statistics file
    trackstats_file = f'{stats_path}{stats_filebase}{startdate}_{enddate}.nc'
    # Find all pixel-level files
    pixelfilelist = sorted(glob.glob(f'{pixelfile_path}{pixel_filebase}*.nc'))
    nfiles = len(pixelfilelist)
    # Find all PPI files
    ppifilelist = sorted(glob.glob(f'{ppifile_path}{ppi_filebase}*.nc'))
    nppifiles = len(ppifilelist)
    
    # Get basetime and from the satellite and pixel files
    ppi_basetime, ppifile_dict = calc_basetime(ppifilelist, ppi_filebase)
    pixel_basetime, pixelfile_dict = calc_basetime(pixelfilelist, pixel_filebase)

    # Find matching satellite files for each pixel file
    match_ppifilelist = [''] * nfiles
    match_ppibasetime = np.full(nfiles, np.NaN, dtype=np.float)
    for ifile in range(nfiles):
        # Find PPI time closest to the pixel file time and get the index
        # Save the filename if time difference is < time_window
        idx = np.argmin(np.abs(ppi_basetime - pixel_basetime[ifile]))
        if np.abs(ppi_basetime[idx] - pixel_basetime[ifile]) < time_window:
            match_ppifilelist[ifile] = ppifilelist[idx]
            match_ppibasetime[ifile] = ppi_basetime[idx]
        else:
            print(f'No match file found for: {pixelfilelist[ifile]}')


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

    # Read a PPI file to get vertical coordinates
    dsv = xr.open_dataset(match_ppifilelist[0])
    zdim = dsv.dims['z']
    height = dsv['z'].values
    dsv.close()


    # Create new statistics variables
    # cell_area_2 is to double check with that from the track statistics file to make sure they match exactly
    # zdim = 45
    cell_area_2 = np.full((ntracks, ntimes), np.nan, dtype=float)
    max_dbz = np.full((ntracks, ntimes, zdim), np.nan, dtype=float)
    npix_dbz0 = np.full((ntracks, ntimes, zdim), np.nan, dtype=float)
    npix_dbz10 = np.full((ntracks, ntimes, zdim), np.nan, dtype=float)
    npix_dbz20 = np.full((ntracks, ntimes, zdim), np.nan, dtype=float)
    npix_dbz30 = np.full((ntracks, ntimes, zdim), np.nan, dtype=float)
    npix_dbz40 = np.full((ntracks, ntimes, zdim), np.nan, dtype=float)
    npix_dbz50 = np.full((ntracks, ntimes, zdim), np.nan, dtype=float)
    npix_dbz60 = np.full((ntracks, ntimes, zdim), np.nan, dtype=float)
    max_zdr = np.full((ntracks, ntimes, zdim), np.nan, dtype=float)
    max_kdp = np.full((ntracks, ntimes, zdim), np.nan, dtype=float)
    max_rainrate = np.full((ntracks, ntimes, zdim), np.nan, dtype=float)
    max_Dm = np.full((ntracks, ntimes, zdim), np.nan, dtype=float)
    max_lwc = np.full((ntracks, ntimes, zdim), np.nan, dtype=float)
    volrain = np.full((ntracks, ntimes, zdim), np.nan, dtype=float)
    

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
            # import pdb; pdb.set_trace() 

            iresult = calc_3d_cellstats_singlefile(
                        pixelfilelist[ifile], 
                        match_ppifilelist[ifile], 
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

            # iresult = delayed(calc_sat_cellstats_singlefile)(pixelfilelist[ifile], match_ppifilelist[ifile], pixel_filebase, stats_basetime, pixel_radius)
            iresult = dask.delayed(calc_3d_cellstats_singlefile)(
                        pixelfilelist[ifile], 
                        match_ppifilelist[ifile], 
                        pixel_filebase, 
                        idx_track, 
                        pixel_radius
                        )
            final_results.append(iresult)
            
        # Collect results from Dask
        print("Computing statistics ...")
        final_results = dask.compute(*final_results)

    # import pdb; pdb.set_trace()

    # Now that all calculations for each pixel file is done, put the results back to the tracks format
    # Loop over the returned statistics list, organized by file
    for ifile in range(nfiles):
        # Get the return results from the current file
        vars = final_results[ifile]
        if (vars is not None):
            # Get the match track indices (matchindicestmp contains: [track_index, time_index]) for this pixel file
            nmatchcloudtmp = nmatchcloud_all[ifile]
            matchindicestmp = matchindices_all[ifile]
            # import pdb; pdb.set_trace()

            # Loop over each matched cloud, and put them back in the track variable
            for imatch in range(nmatchcloudtmp):
                # matchindices are in [tracks, times], or [tracks, times, height]
                cell_area_2[matchindicestmp[0,imatch], matchindicestmp[1,imatch]] = vars[1][imatch]
                max_dbz[matchindicestmp[0,imatch], matchindicestmp[1,imatch], :] = vars[2][imatch, :]
                npix_dbz0[matchindicestmp[0,imatch], matchindicestmp[1,imatch], :] = vars[3][imatch, :]
                npix_dbz10[matchindicestmp[0,imatch], matchindicestmp[1,imatch], :] = vars[4][imatch, :]
                npix_dbz20[matchindicestmp[0,imatch], matchindicestmp[1,imatch], :] = vars[5][imatch, :]
                npix_dbz30[matchindicestmp[0,imatch], matchindicestmp[1,imatch], :] = vars[6][imatch, :]
                npix_dbz40[matchindicestmp[0,imatch], matchindicestmp[1,imatch], :] = vars[7][imatch, :]
                npix_dbz50[matchindicestmp[0,imatch], matchindicestmp[1,imatch], :] = vars[8][imatch, :]
                npix_dbz60[matchindicestmp[0,imatch], matchindicestmp[1,imatch], :] = vars[9][imatch, :]

                max_zdr[matchindicestmp[0,imatch], matchindicestmp[1,imatch], :] = vars[10][imatch, :]
                max_kdp[matchindicestmp[0,imatch], matchindicestmp[1,imatch], :] = vars[11][imatch, :]
                max_rainrate[matchindicestmp[0,imatch], matchindicestmp[1,imatch], :] = vars[12][imatch, :]
                max_Dm[matchindicestmp[0,imatch], matchindicestmp[1,imatch], :] = vars[13][imatch, :]
                max_lwc[matchindicestmp[0,imatch], matchindicestmp[1,imatch], :] = vars[14][imatch, :]
                volrain[matchindicestmp[0,imatch], matchindicestmp[1,imatch], :] = vars[15][imatch, :]


    ##################################
    # Write to netcdf
    print('Writing output netcdf ... ')
    t0_write = time.time()

    # Define variable list
    varlist = {'basetime': ([trackdimname, timedimname], stats_basetime), \
                'cell_area2': ([trackdimname, timedimname], cell_area_2), \
                'max_reflectivity': ([trackdimname, timedimname, zdimname], max_dbz), \
                'npix_dbz0': ([trackdimname, timedimname, zdimname], npix_dbz0), \
                'npix_dbz10': ([trackdimname, timedimname, zdimname], npix_dbz10), \
                'npix_dbz20': ([trackdimname, timedimname, zdimname], npix_dbz20), \
                'npix_dbz30': ([trackdimname, timedimname, zdimname], npix_dbz30), \
                'npix_dbz40': ([trackdimname, timedimname, zdimname], npix_dbz40), \
                'npix_dbz50': ([trackdimname, timedimname, zdimname], npix_dbz50), \
                'npix_dbz60': ([trackdimname, timedimname, zdimname], npix_dbz60), \
                'max_zdr': ([trackdimname, timedimname, zdimname], max_zdr), \
                'max_kdp': ([trackdimname, timedimname, zdimname], max_kdp), \
                'max_rainrate': ([trackdimname, timedimname, zdimname], max_rainrate), \
                'max_Dm': ([trackdimname, timedimname, zdimname], max_Dm), \
                'max_lwc': ([trackdimname, timedimname, zdimname], max_lwc), \
                'volrain': ([trackdimname, timedimname, zdimname], volrain), \
              }

    # Define coordinate list
    coordlist = {trackdimname: ([trackdimname], np.arange(0, ntracks)), \
                 timedimname: ([timedimname], np.arange(0, ntimes)), \
                 zdimname: ([zdimname], height), \
                }

    # Define global attributes
    gattrlist = {'title':  'Track 3D statistics', \
                 'Institution': 'Pacific Northwest National Laboratoy', \
                 'Contact': 'Zhe Feng, zhe.feng@pnnl.gov', \
                 'Created_on':  time.ctime(time.time()), \
                 'source_trackfile': trackstats_file, \
                 'startdate': startdate, \
                 'enddate': enddate, \
                }
    # Define xarray dataset
    dsout = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)

    dsout['basetime'].attrs['long_name'] = 'Epoch time of each cell in a track'
    dsout['basetime'].attrs['standard_name'] = 'time'
    dsout['basetime'].attrs['units'] = basetime_units
    dsout['cell_area2'].attrs['long_name'] = 'Area of the convective cell in a track'
    dsout['cell_area2'].attrs['units'] = 'km^2'
    dsout['max_reflectivity'].attrs['long_name'] = 'Maximum reflectivity profile in a track'
    dsout['max_reflectivity'].attrs['units'] = 'dBZ'
    dsout['npix_dbz0'].attrs['long_name'] = 'Number of pixel greater than 0 dBZ profile in a track'
    dsout['npix_dbz0'].attrs['units'] = 'counts'
    dsout['npix_dbz10'].attrs['long_name'] = 'Number of pixel greater than 10 dBZ profile in a track'
    dsout['npix_dbz10'].attrs['units'] = 'counts'
    dsout['npix_dbz20'].attrs['long_name'] = 'Number of pixel greater than 20 dBZ profile in a track'
    dsout['npix_dbz20'].attrs['units'] = 'counts'
    dsout['npix_dbz30'].attrs['long_name'] = 'Number of pixel greater than 30 dBZ profile in a track'
    dsout['npix_dbz30'].attrs['units'] = 'counts'
    dsout['npix_dbz40'].attrs['long_name'] = 'Number of pixel greater than 40 dBZ profile in a track'
    dsout['npix_dbz40'].attrs['units'] = 'counts'
    dsout['npix_dbz50'].attrs['long_name'] = 'Number of pixel greater than 50 dBZ profile in a track'
    dsout['npix_dbz50'].attrs['units'] = 'counts'
    dsout['npix_dbz60'].attrs['long_name'] = 'Number of pixel greater than 60 dBZ profile in a track'
    dsout['npix_dbz60'].attrs['units'] = 'counts'
    dsout['max_zdr'].attrs['long_name'] = 'Maximum ZDR profile in a track'
    dsout['max_zdr'].attrs['units'] = 'dB'
    dsout['max_kdp'].attrs['long_name'] = 'Maximum KDP profile in a track'
    dsout['max_kdp'].attrs['units'] = 'degrees/km'
    dsout['max_rainrate'].attrs['long_name'] = 'Maximum rain rate profile in a track'
    dsout['max_rainrate'].attrs['units'] = 'mm/hr'
    dsout['max_Dm'].attrs['long_name'] = 'Maximum mass weighted mean diameter profile in a track'
    dsout['max_Dm'].attrs['units'] = 'mm'
    dsout['max_lwc'].attrs['long_name'] = 'Maximum liquid water content profile in a track'
    dsout['max_lwc'].attrs['units'] = 'g/m^3'
    dsout['volrain'].attrs['long_name'] = 'Volumetric rainfall profile in a track'
    dsout['volrain'].attrs['units'] = 'm^3/h^1'

    # Specify encoding list
    fillval = -9999
    var_float_encode = {'dtype':'float32', 'zlib':True, '_FillValue': np.nan}
    var_int_encode = {'dtype': 'int', 'zlib':True, '_FillValue': fillval}
    encodelist = {#'lifetime': var_int_encode, \
                  #     'basetime': {'zlib':True, 'units': basetime_units}, \
                    'basetime': {'zlib':True}, \
                    'cell_area2': var_float_encode, \
                    'max_reflectivity': var_float_encode, \
                    'npix_dbz0': var_float_encode, \
                    'npix_dbz10': var_float_encode, \
                    'npix_dbz20': var_float_encode, \
                    'npix_dbz30': var_float_encode, \
                    'npix_dbz40': var_float_encode, \
                    'npix_dbz50': var_float_encode, \
                    'npix_dbz60': var_float_encode, \
                    'max_zdr': var_float_encode, \
                    'max_kdp': var_float_encode, \
                    'max_rainrate': var_float_encode, \
                    'max_Dm': var_float_encode, \
                    'max_lwc': var_float_encode, \
                    'volrain': var_float_encode, \
                 }

    # Write netcdf file
    dsout.to_netcdf(path=output_filename, mode='w', format='NETCDF4_CLASSIC', unlimited_dims=trackdimname, encoding=encodelist)
    print(f'Output saved: {output_filename}')