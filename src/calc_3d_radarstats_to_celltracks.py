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
import warnings
import dask
from dask.distributed import Client, LocalCluster
from scipy.stats import mode

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

#########################################################
def calc_3d_cellstats_singlefile(
    pixel_filename, 
    ppi_filename, 
    idx_track, 
    pixel_radius
    ):
    """
    Calculates cell statistics from matching satellite data in a single pixel file.

    Parameters:
    ===========
    pixel_filename: string
        Input cell pixel file name
    ppi_filename: string
        Input 3D PPI gridded radar file name
    idx_track: array
        Track indices in the current pixel file
    pixel_radius: float
        Pixel size.

    Returns:
    ===========
    nmatchcloud: <int>
        Number of matched cells in this file
    cell_area: <array>
        Cell area for each matched cell
    max_dbz: <array>
        Maximum reflectivity profile for each matched cell

    """

    # Check if PPI file exist
    if os.path.isfile(ppi_filename):
        print(ppi_filename)

        # Read 3D PPI file
        dsv = xr.open_dataset(ppi_filename)
        zdim = dsv.sizes['z']
        height = dsv['z'].values
        dbz = dsv['taranis_attenuation_corrected_reflectivity'].squeeze()
        zdr = dsv['taranis_attenuation_corrected_differential_reflectivity'].squeeze()
        kdp = dsv['kdp_pos_lp_reg'].squeeze()
        rainrate = dsv['taranis_rain_rate'].squeeze()
        Dm = dsv['taranis_Dm'].squeeze()
        rwc = dsv['lwc_combined'].squeeze()
        hid = dsv['hydrometeor_identification_post_grid'].squeeze()
        # dsv.close()

        # Read pixel-level track file
        ds = xr.open_dataset(pixel_filename, decode_times=False)
        # Replace lon/lat coordinates with x/y
        ds['lon'] = dsv['x'].values
        ds['lat'] = dsv['y'].values
        # Drop original variables (x, y), then rename lat/lon to x/y
        ds = ds.drop_vars(['x', 'y']).rename({'lat':'y', 'lon':'x'})
        # ds = ds.rename({'lat':'y', 'lon':'x'})
        # Read variables
        cmask = ds['conv_mask'].squeeze()
        tracknumbermap = ds['tracknumber'].squeeze()
        # Get cell tracknumber mask
        # Convert convective cell mask to binary, then multiply by tracknumber
        tracknumbermap_cmask = (cmask > 0) * tracknumbermap
        # Replace background values with NaN
        tracknumbermap_cmask = tracknumbermap_cmask.where(tracknumbermap_cmask > 0, other=np.NaN)
        ds.close()

        # Create arrays for output statistics
        nmatchcloud = len(idx_track)
        cell_area = np.full((nmatchcloud), np.nan, dtype=np.float32)
        max_dbz = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        npix_dbz0 = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        npix_dbz10 = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        npix_dbz20 = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        npix_dbz30 = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        npix_dbz40 = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        npix_dbz50 = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        npix_dbz60 = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        max_zdr = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        max_kdp = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        max_rainrate = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        max_Dm = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        max_rwc = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        volrain = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        mode_hid = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        hid_01 = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        hid_02 = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        hid_03 = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        hid_04 = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        hid_05 = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        hid_06 = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        hid_07 = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        hid_08 = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        hid_09 = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        hid_10 = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        numdbz = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        numrwc = np.full((nmatchcloud, zdim), np.nan, dtype=np.float32)
        
        pct = [0,10,25,50,75,90] # Percentiles to calculate
        pdim = len(pct)
        pct_dbz = np.full((nmatchcloud, zdim, pdim), np.nan, dtype=np.float32)
        pct_zdr = np.full((nmatchcloud, zdim, pdim), np.nan, dtype=np.float32)
        pct_kdp = np.full((nmatchcloud, zdim, pdim), np.nan, dtype=np.float32)
        pct_rainrate = np.full((nmatchcloud, zdim, pdim), np.nan, dtype=np.float32)
        pct_Dm = np.full((nmatchcloud, zdim, pdim), np.nan, dtype=np.float32)
        pct_rwc = np.full((nmatchcloud, zdim, pdim), np.nan, dtype=np.float32)

        if (nmatchcloud > 0):
            
            # Loop over each match tracked cloud
            for imatchcloud in range(nmatchcloud):

                # Intialize array for keeping data associated with the tracked cell
                # filtered_dbz = np.full((zdim, ydim, xdim), np.nan, dtype=float)

                # Track number needs to add 1
                itracknum = idx_track[imatchcloud] + 1

                # Count the number of pixels for the original cell mask
                inpix_cloud = np.count_nonzero(tracknumbermap_cmask == itracknum)
                
                # Get location indices of the cell (original), used to calculate cell_area
                # icloudlocationy_orig, icloudlocationx_orig = np.where(tracknumbermap_cmask == itracknum)
                # Get location indices of the cell (inflated)
                # icloudlocationy, icloudlocationx = np.where(tracknumbermap == itracknum)
                
                # Count the number of pixels for the original cell mask
                # inpix_cloud = len(icloudlocationy_orig)

                # Proceed if the number matching cloud pixel > 0
                if inpix_cloud > 0:

                    # Subset 3D variables to the current cell mask using Xarray
                    sub_dbz = dbz.where(tracknumbermap == itracknum, drop=True).values
                    sub_zdr = zdr.where(tracknumbermap == itracknum, drop=True).values
                    sub_kdp = kdp.where(tracknumbermap == itracknum, drop=True).values
                    sub_rainrate = rainrate.where(tracknumbermap == itracknum, drop=True).values
                    sub_Dm = Dm.where(tracknumbermap == itracknum, drop=True).values
                    sub_rwc = rwc.where(tracknumbermap == itracknum, drop=True).values
                    sub_hid = hid.where(tracknumbermap == itracknum, drop=True).values
                    
                    cell_area[imatchcloud] = inpix_cloud * pixel_radius**2
                    
                    # Calculate new statistics of the cloud
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        # Max profile
                        max_dbz[imatchcloud,:] = np.nanmax(sub_dbz, axis=(1,2))
                        max_zdr[imatchcloud,:] = np.nanmax(sub_zdr, axis=(1,2))
                        max_kdp[imatchcloud,:] = np.nanmax(sub_kdp, axis=(1,2))
                        max_rainrate[imatchcloud,:] = np.nanmax(sub_rainrate, axis=(1,2))
                        max_Dm[imatchcloud,:] = np.nanmax(sub_Dm, axis=(1,2))
                        max_rwc[imatchcloud,:] = np.nanmax(sub_rwc, axis=(1,2))
                        
                        pct_dbz[imatchcloud,:,:] = np.nanpercentile(sub_dbz, pct, axis=(1,2)).T
                        pct_zdr[imatchcloud,:,:] = np.nanpercentile(sub_zdr, pct, axis=(1,2)).T
                        pct_kdp[imatchcloud,:,:] = np.nanpercentile(sub_kdp, pct, axis=(1,2)).T
                        pct_rainrate[imatchcloud,:,:] = np.nanpercentile(sub_rainrate, pct, axis=(1,2)).T
                        pct_Dm[imatchcloud,:,:] = np.nanpercentile(sub_Dm, pct, axis=(1,2)).T
                        pct_rwc[imatchcloud,:,:] = np.nanpercentile(sub_rwc, pct, axis=(1,2)).T
                        mode_hid[imatchcloud,:],_ = mode(sub_hid,axis=(1,2),nan_policy='omit')
                        hid_01[imatchcloud,:] = np.count_nonzero(sub_hid == 1, axis=(1,2))
                        hid_02[imatchcloud,:] = np.count_nonzero(sub_hid == 2, axis=(1,2))
                        hid_03[imatchcloud,:] = np.count_nonzero(sub_hid == 3, axis=(1,2))
                        hid_04[imatchcloud,:] = np.count_nonzero(sub_hid == 4, axis=(1,2))
                        hid_05[imatchcloud,:] = np.count_nonzero(sub_hid == 5, axis=(1,2))
                        hid_06[imatchcloud,:] = np.count_nonzero(sub_hid == 6, axis=(1,2))
                        hid_07[imatchcloud,:] = np.count_nonzero(sub_hid == 7, axis=(1,2))
                        hid_08[imatchcloud,:] = np.count_nonzero(sub_hid == 8, axis=(1,2))
                        hid_09[imatchcloud,:] = np.count_nonzero(sub_hid == 9, axis=(1,2))
                        hid_10[imatchcloud,:] = np.count_nonzero(sub_hid == 10, axis=(1,2))
                        numdbz[imatchcloud,:] = np.count_nonzero(abs(sub_dbz)>0, axis=(1,2))
                        numrwc[imatchcloud,:] = np.count_nonzero(sub_rwc>0, axis=(1,2))

                    # Count number of pixels > X dBZ at each level
                    npix_dbz0[imatchcloud,:] = np.count_nonzero(sub_dbz > 0, axis=(1,2))
                    npix_dbz10[imatchcloud,:] = np.count_nonzero(sub_dbz > 10, axis=(1,2))
                    npix_dbz20[imatchcloud,:] = np.count_nonzero(sub_dbz > 20, axis=(1,2))
                    npix_dbz30[imatchcloud,:] = np.count_nonzero(sub_dbz > 30, axis=(1,2))
                    npix_dbz40[imatchcloud,:] = np.count_nonzero(sub_dbz > 40, axis=(1,2))
                    npix_dbz50[imatchcloud,:] = np.count_nonzero(sub_dbz > 50, axis=(1,2))
                    npix_dbz60[imatchcloud,:] = np.count_nonzero(sub_dbz > 60, axis=(1,2))

                    # volume rainrate: area total rainfall depth x area of a pixel
                    # h * area = mm h^(-1) * km^2 = 10^(-3) m h^(-1) * (10^3 m)^2 = 10^3 m^3 h^(-1)
                    # h * area * 1e3 has unit of [m^3 h^(-1)]
                    volrain[imatchcloud,:] = np.nansum(sub_rainrate, axis=(1,2)) * pixel_radius**2 * 1e3

                else:
                    print(f'No cell matching track # {itracknum}')

            # Group outputs in dictionaries
            out_dict = {
                # nmatchcloud, 
                "cell_area": cell_area,
                "p50_reflectivity": pct_dbz[:,:,3],
                "max_reflectivity": max_dbz,
                "npix_dbz0": npix_dbz0,
                "npix_dbz10": npix_dbz10,
                "npix_dbz20": npix_dbz20,
                "npix_dbz30": npix_dbz30,
                "npix_dbz40": npix_dbz40,
                "npix_dbz50": npix_dbz50,
                "npix_dbz60": npix_dbz60,
                "min_zdr": pct_zdr[:,:,0],
                "p10_zdr": pct_zdr[:,:,1],
                "p25_zdr": pct_zdr[:,:,2],
                "p50_zdr": pct_zdr[:,:,3],
                "p75_zdr": pct_zdr[:,:,4],
                "p90_zdr": pct_zdr[:,:,5],
                "max_zdr": max_zdr,
                "min_kdp": pct_kdp[:,:,0],
                "p10_kdp": pct_kdp[:,:,1],
                "p25_kdp": pct_kdp[:,:,2],
                "p50_kdp": pct_kdp[:,:,3],
                "p75_kdp": pct_kdp[:,:,4],
                "p90_kdp": pct_kdp[:,:,5],
                "max_kdp": max_kdp,
                "p50_rainrate": pct_rainrate[:,:,3],
                "max_rainrate": max_rainrate,
                "p50_Dm": pct_Dm[:,:,3],
                "max_Dm": max_Dm,
                "p50_rwc": pct_rwc[:,:,3],
                "max_rwc": max_rwc,
                "volrain": volrain,
                "species": mode_hid,
                "drizzle": hid_01,
                "rain": hid_02,
                "ice_crystals": hid_03,
                "aggregates": hid_04,
                "wet_snow": hid_05,
                "vertical_ice": hid_06,
                "low_density_graupel": hid_07,
                "high_density_graupel": hid_08,
                "hail": hid_09,
                "big_drops": hid_10,
                "num_dbz": numdbz,
                "num_rwc": numrwc,
            }
            out_dict_attrs = {
                "cell_area": {
                    "long_name": "Area of the convective cell in a track",
                    "units": "km^2",
                }, 
                'p50_reflectivity': {
                    "long_name": '50th pct reflectivity profile in a track',
                    "units": "dBZ",
                }, 
                'max_reflectivity': {
                    "long_name": 'Maximum reflectivity profile in a track',
                    "units": "dBZ",
                },
                'npix_dbz0': {
                    "long_name": "Number of pixel greater than 0 dBZ profile in a track",
                    "units": "counts",
                },
                'npix_dbz10': {
                    "long_name": "Number of pixel greater than 10 dBZ profile in a track",
                    "units": "counts",
                },
                'npix_dbz20': {
                    "long_name": "Number of pixel greater than 20 dBZ profile in a track",
                    "units": "counts",
                },
                'npix_dbz30': {
                    "long_name": "Number of pixel greater than 30 dBZ profile in a track",
                    "units": "counts",
                },
                'npix_dbz40': {
                    "long_name": "Number of pixel greater than 40 dBZ profile in a track",
                    "units": "counts",
                },
                'npix_dbz50': {
                    "long_name": "Number of pixel greater than 50 dBZ profile in a track",
                    "units": "counts",
                },
                'npix_dbz60': {
                    "long_name": "Number of pixel greater than 60 dBZ profile in a track",
                    "units": "counts",
                },
                'min_zdr': {
                    "long_name": "Minimum ZDR profile in a track",
                    "units": "dB",
                },
                'p10_zdr': {
                    "long_name": "10th pct ZDR profile in a track",
                    "units": "dB",
                },
                'p25_zdr': {
                    "long_name": "25th pct ZDR profile in a track",
                    "units": "dB",
                },
                'p50_zdr': {
                    "long_name": "50th pct ZDR profile in a track",
                    "units": "dB",
                },
                'p75_zdr': {
                    "long_name": "75th pct ZDR profile in a track",
                    "units": "dB",
                },
                'p90_zdr': {
                    "long_name": "90th pct ZDR profile in a track",
                    "units": "dB",
                },
                'max_zdr': {
                    "long_name": "Maximum ZDR profile in a track",
                    "units": "dB",
                },
                'min_kdp': {
                    "long_name": "Minimum KDP profile in a track",
                    "units": "degrees/km",
                },
                'p10_kdp': {
                    "long_name": "10th pct KDP profile in a track",
                    "units": "degrees/km",
                },
                'p25_kdp': {
                    "long_name": "25th pct KDP profile in a track",
                    "units": "degrees/km",
                },
                'p50_kdp': {
                    "long_name": "50th pct KDP profile in a track",
                    "units": "degrees/km",
                },
                'p75_kdp': {
                    "long_name": "75th pct KDP profile in a track",
                    "units": "degrees/km",
                },
                'p90_kdp': {
                    "long_name": "90th pct KDP profile in a track",
                    "units": "degrees/km",
                },
                'max_kdp': {
                    "long_name": "Maximum KDP profile in a track",
                    "units": "degrees/km",
                },
                'p50_rainrate': {
                    "long_name": "50th pct rain rate profile in a track",
                    "units": "mm/hr",
                },
                'max_rainrate': {
                    "long_name": "Maximum rain rate profile in a track",
                    "units": "mm/hr",
                },
                'p50_Dm': {
                    "long_name": "50th pct mass weighted mean diameter profile in a track",
                    "units": "mm",
                },
                'max_Dm': {
                    "long_name": "Maximum mass weighted mean diameter profile in a track",
                    "units": "mm",
                },
                'p50_rwc': {
                    "long_name": "50th pct rain water content profile in a track",
                    "units": "g/m^3",
                },
                'max_rwc': {
                    "long_name": "Maximum liquid water content profile in a track",
                    "units": "g/m^3",
                },
                'volrain': {
                    "long_name": "Volumetric rainfall profile in a track",
                    "units": "m^3/h^1",
                },
                'species': {
                    "long_name": "Predominant hydrometeor class",
                    "units": "Drizzle (1), Rain (2), Ice Crystals (3), Aggregates (4), Wet Snow (5), Vertical Ice (6), Low Density Graupel (7), High Density Graupel (8), Hail (9), Big Drops (10)",
                },
                'drizzle': {
                    "long_name": "Number of pixels corresponding to drizzle in a cell",
                    "units": "counts",
                },
                'rain': {
                    "long_name": "Number of pixels corresponding to rain in a cell",
                    "units": "counts",
                },
                'ice_crystals': {
                    "long_name": "Number of pixels corresponding to ice crystals in a cell",
                    "units": "counts",
                },
                'aggregates': {
                    "long_name": "Number of pixels corresponding to aggregates in a cell",
                    "units": "counts",
                },
                'wet_snow': {
                    "long_name": "Number of pixels corresponding to wet snow in a cell",
                    "units": "counts",
                },
                'vertical_ice': {
                    "long_name": "Number of pixels corresponding to vertical ice in a cell",
                    "units": "counts",
                },
                'low_density_graupel': {
                    "long_name": "Number of pixels corresponding to low density graupel in a cell",
                    "units": "counts",
                },
                'high_density_graupel': {
                    "long_name": "Number of pixels corresponding to high density graupel in a cell",
                    "units": "counts",
                },
                'hail': {
                    "long_name": "Number of pixels corresponding to hail in a cell",
                    "units": "counts",
                },
                'big_drops': {
                    "long_name": "Number of pixels corresponding to big drops in a cell",
                    "units": "counts",
                },
                'num_dbz': {
                    "long_name": "Number of pixels in dBZ field containing non-zero values",
                    "units": "counts",
                },
                'num_rwc': {
                    "long_name": "Number of pixels in rwc field containing non-zero values",
                    "units": "counts",
                },
            }
            return out_dict, out_dict_attrs


#########################################################
if __name__ == '__main__':

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
    stats_filebase = 'trackstats_'
    pixel_filebase = 'celltracks_'
    ppi_filebase = 'taranis_corcsapr2cfrppiqcM1.c1.'

    # Output statistics filename
    output_filename = f'{output_path}stats_3d_ppi_{startdate}_{enddate}.nc'

    # Track statistics file dimension names
    tracks_dimname = 'tracks'
    times_dimname = 'times'
    z_dimname = 'z'

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
    match_ppibasetime = np.full(nfiles, np.NaN, dtype=np.float64)
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
    ntracks = dsstats.sizes[tracks_dimname]
    ntimes = dsstats.sizes[times_dimname]
    tracks_coord = dsstats.coords[tracks_dimname]
    times_coord = dsstats.coords[times_dimname]
    stats_basetime = dsstats['base_time']
    # basetime_units = dsstats['base_time'].units
    cell_area = dsstats['cell_area'].values
    pixel_radius = dsstats.attrs['pixel_radius_km']
    dsstats.close()

    print(f'Total Number of Tracks: {ntracks}')

    # Read a PPI file to get vertical coordinates
    dsv = xr.open_dataset(match_ppifilelist[0])
    zdim = dsv.sizes['z']
    height_coord = dsv['z']
    dsv.close()


    ##############################################################
    # Call function to calculate statistics
    trackindices_all = []
    timeindices_all = []
    final_results = []
    if run_parallel==0:
        # Loop over each pixel-file and call function to calculate
        for ifile in range(nfiles):
#         for ifile in range(147,149):
            # Find all matching time indices from track stats file to the current pixel file
            matchindices = np.array(
                np.where(np.abs(stats_basetime.values - pixel_basetime[ifile]) < time_window)
            )
            # The returned match indices are for [tracks, times] dimensions respectively
            idx_track = matchindices[0]
            idx_time = matchindices[1]

            if len(idx_track) > 0:
                print(ifile)
                # Save matchindices for the current pixel file to the overall list
                trackindices_all.append(idx_track)
                timeindices_all.append(idx_time)

                iresult = calc_3d_cellstats_singlefile(
                            pixelfilelist[ifile], 
                            match_ppifilelist[ifile], 
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
            matchindices = np.array(
                np.where(np.abs(stats_basetime.values - pixel_basetime[ifile]) < time_window)
            )
            # The returned match indices are for [tracks, times] dimensions respectively
            idx_track = matchindices[0]
            idx_time = matchindices[1]

            if len(idx_track) > 0:
                # Save matchindices for the current pixel file to the overall list
                trackindices_all.append(idx_track)
                timeindices_all.append(idx_time)

                iresult = dask.delayed(calc_3d_cellstats_singlefile)(
                            pixelfilelist[ifile], 
                            match_ppifilelist[ifile], 
                            idx_track, 
                            pixel_radius
                            )
                final_results.append(iresult)
            
        # Collect results from Dask
        print("Computing statistics ...")
        final_results = dask.compute(*final_results)


    # Make a variable list from one of the returned dictionaries
    var_names = list(final_results[0][0].keys())
    # Get variable attributes from one of the returned dictionaries
    var_attrs = final_results[0][1]

    # Loop over variable list to create the dictionary entry
    out_dict = {}
    out_dict_attrs = {}
    for ivar in var_names:
        if ivar == "cell_area":
            out_dict[ivar] = np.full((ntracks, ntimes), np.nan, dtype=np.float32)
        else:
            out_dict[ivar] = np.full((ntracks, ntimes, zdim), np.nan, dtype=np.float32)
        out_dict_attrs[ivar] = var_attrs[ivar]

    # The number of returned results
    nresults = len(final_results)

    # Now that all calculations for each pixel file is done, put the results back to the tracks format
    # Loop over the returned statistics list
    for ifile in range(nresults):
        # Get the results from the current file
        vars = final_results[ifile]
        if (vars is not None):
            # Get the return results for this pixel file
            # The result is a tuple: (out_dict, out_dict_attrs)
            # The first entry is the dictionary containing the variables
            iResult = final_results[ifile][0]

            # Get trackindices and timeindices for this file
            trackindices = trackindices_all[ifile]
            timeindices = timeindices_all[ifile]

            # Loop over each variable and assign values to output dictionary
            for ivar in var_names:
                if iResult[ivar].ndim == 2:
                    out_dict[ivar][trackindices,timeindices] = iResult[ivar]
                if iResult[ivar].ndim == 3:
                    out_dict[ivar][trackindices,timeindices,:] = iResult[ivar]


    ##################################
    # Write to netcdf
    print('Writing output netcdf ... ')
    t0_write = time.time()
    
    import pdb;pdb.set_trace()
    # Define output variable dictionary
    var_dict = {}
    for key, value in out_dict.items():
        if value.ndim == 2:
            var_dict[key] = ([tracks_dimname, times_dimname], value, out_dict_attrs[key])
        if value.ndim == 3:
            var_dict[key] = ([tracks_dimname, times_dimname, z_dimname], value, out_dict_attrs[key])
    # Define coordinate dictionary
    coord_dict = {
        tracks_dimname: ([tracks_dimname], tracks_coord.data, tracks_coord.attrs),
        times_dimname: ([times_dimname], times_coord.data, times_coord.attrs),
        z_dimname: ([z_dimname], height_coord.data, height_coord.attrs),
    }
    # Define global attributes
    gattr_dict = {
        'title':  'Track 3D statistics', \
        'Institution': 'Pacific Northwest National Laboratoy', \
        'Contact': 'Zhe Feng, zhe.feng@pnnl.gov', \
        'Created_on':  time.ctime(time.time()), \
        'source_trackfile': trackstats_file, \
        'startdate': startdate, \
        'enddate': enddate, \
    }
    # Define xarray dataset
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Add variables from cell track stats to the output
    dsout['base_time'] = stats_basetime

    # Delete file if it already exists
    if os.path.isfile(output_filename):
        os.remove(output_filename)
        
    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}

    # Write to netcdf file
    dsout.to_netcdf(path=output_filename, mode="w",
                    format="NETCDF4", unlimited_dims=tracks_dimname, encoding=encoding)
    print(f'Output saved: {output_filename}')