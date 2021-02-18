import numpy as np
import os, glob
import time, datetime, calendar
from math import pi
import xarray as xr
import pandas as pd
# from netCDF4 import Dataset, num2date, chartostring
# from scipy.ndimage import label, binary_dilation, generate_binary_structure
# from skimage.measure import regionprops
# from math import pi
# from scipy.stats import skew


#########################################################
def calc_sat_cellstats_singlefile(
    pixel_filename, 
    sat_filename, 
    pixel_filebase, 
    idx_track, 
    pixel_radius
    ):
    """
    Calculates cell statistics from matching satellite data in a single pixel file.

    Parameters:
    ===========
    pixel_filename: string
        Input cell pixel file name
    sat_filename: string
        Input satellite pixel file name
    pixel_filebase: string
        Input cell pixel file basename
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
    ctt_min: <array>
        Minimum cloud-top temperature for each matched cell
    
    If no matched cell exists in the file, returns None.
    """

    # pixelfile_path = os.path.dirname(pixel_filename)
    # prelength = len(pixelfile_path)+1 + len(pixel_filebase)
    # ittdatetimestring = pixel_filename[prelength:(prelength+13)]

    # Check if satellite data file exist
    # if os.path.isfile(pixel_filename):
    if os.path.isfile(sat_filename):
        # print(pixel_filename)
        print(sat_filename)

        # Read satellite file
        dsv = xr.open_dataset(sat_filename)
        tir = dsv['temperature_ir'].squeeze().values
        ctt = dsv['cloud_top_temperature'].squeeze().values
        cth = dsv['cloud_top_height'].squeeze().values
        ctp = dsv['cloud_top_pressure'].squeeze().values
        # cep = dsv['cloud_effective_pressure'].squeeze().values
        phase = dsv['cloud_phase'].squeeze().values
        lwp_iwp = dsv['cloud_lwp_iwp'].squeeze().values
        dsv.close()

        # Read pixel-level track file
        ds = xr.open_dataset(pixel_filename, decode_times=False)
        lon = ds['longitude'].values
        lat = ds['latitude'].values
        cloudid_basetime = ds['basetime'].values
        tracknumbermap = ds['tracknumber'].squeeze().values
        tracknumbermap_cmask = ds['tracknumber_cmask2'].squeeze().values
        xdim = ds.dims['lon']
        ydim = ds.dims['lat']
        ds.close()

        # # Find all matching time indices from track stats file to the pixel file
        # # The returned match indices are for [tracks, times] dimensions respectively
        # matchindices = np.array(np.where(np.abs(stats_basetime - cloudid_basetime) < 1))
        # idx_track = matchindices[0]
        # idx_time = matchindices[1]

        nmatchcloud = len(idx_track)

        # Create arrays for output statistics
        cell_area = np.full((nmatchcloud), np.nan, dtype=float)
        ctt_min = np.full((nmatchcloud), np.nan, dtype=float)
        tir_min = np.full((nmatchcloud), np.nan, dtype=float)
        cth_max = np.full((nmatchcloud), np.nan, dtype=float)
        ctp_min = np.full((nmatchcloud), np.nan, dtype=float)
        area_liq = np.full((nmatchcloud), np.nan, dtype=float)
        area_ice = np.full((nmatchcloud), np.nan, dtype=float)
        lwp_max = np.full((nmatchcloud), np.nan, dtype=float)
        iwp_max = np.full((nmatchcloud), np.nan, dtype=float)

        if (nmatchcloud > 0):
            
            # Loop over each match tracked cloud
            for imatchcloud in range(nmatchcloud):

                # Intialize array for keeping data associated with the tracked cell
                # filtered_ctt = np.full((ydim, xdim), np.nan, dtype=float)

                # Track number needs to add 1
                itracknum = idx_track[imatchcloud] + 1

                # Count the number of pixels for the original cell mask
                inpix_cloud = np.count_nonzero(tracknumbermap_cmask == itracknum)

                # Index location of original cell mask
                idx_cloudorig = np.where(tracknumbermap_cmask == itracknum)
                # Index location of dilated cell mask
                idx_clouddilated = np.where(tracknumbermap == itracknum)
                # Cell mask with liquid/ice phase
                idx_liq = np.where((tracknumbermap == itracknum) & (phase == 1))
                idx_ice = np.where((tracknumbermap == itracknum) & (phase == 2))

                # Get location indices of the cloud
                # icloudlocationy, icloudlocationx = np.where(tracknumbermap_cmask == itracknum)
                # inpix_cloud = len(icloudlocationy)

                # Proceed if the number matching cloud pixel > 0
                if inpix_cloud > 0:

                    # Subset variables to the current cell mask (original)
                    sub_ctt = ctt[idx_cloudorig]
                    sub_tir = tir[idx_cloudorig]
                    sub_cth = cth[idx_cloudorig]
                    sub_ctp = ctp[idx_cloudorig]
                    sub_phase = phase[idx_cloudorig]

                    # Subset variables to the current cell mask (dilated)
                    # sub_ctt = ctt[idx_clouddilated]
                    # sub_tir = tir[idx_clouddilated]
                    # sub_cth = cth[idx_clouddilated]
                    # sub_ctp = ctp[idx_clouddilated]
                    # sub_phase = phase[idx_clouddilated]
                    
                    # Fill array with data
                    # filtered_ctt[icloudlocationy, icloudlocationx] = np.copy(ctt[icloudlocationy,icloudlocationx])

                    # # Set edges of boundary
                    # miny = np.nanmin(icloudlocationy)
                    # if miny <= 10:
                    #     miny = 0
                    # else:
                    #     miny = miny - 10

                    # maxy = np.nanmax(icloudlocationy)
                    # if maxy >= ydim - 10:
                    #     maxy = ydim
                    # else:
                    #     maxy = maxy + 11

                    # minx = np.nanmin(icloudlocationx)
                    # if minx <= 10:
                    #     minx = 0
                    # else:
                    #     minx = minx - 10

                    # maxx = np.nanmax(icloudlocationx)
                    # if maxx >= xdim - 10:
                    #     maxx = xdim
                    # else:
                    #     maxx = maxx + 11

                    # # Isolate smaller region around cloud, this should speed up calculations
                    # sub_ctt = np.copy(filtered_ctt[miny:maxy, minx:maxx])
                    
                    cell_area[imatchcloud] = inpix_cloud * pixel_radius**2

                    # Calculate new statistics of the cloud
                    # Minimum/Maximum cloud-top variables
                    ctt_min[imatchcloud] = np.nanmin(sub_ctt)
                    tir_min[imatchcloud] = np.nanmin(sub_tir)
                    cth_max[imatchcloud] = np.nanmax(sub_cth)
                    ctp_min[imatchcloud] = np.nanmin(sub_ctp)
                    # Area with liquid/ice from cloud phase flags
                    # 0=clear with snow/ice, 1=water, 2=ice, 3=no retrieval, 4=clear, 5=bad retrieval, 6=weak water, 7=weak ice
                    area_liq[imatchcloud] = np.count_nonzero(sub_phase == 1) * pixel_radius**2
                    area_ice[imatchcloud] = np.count_nonzero(sub_phase == 2) * pixel_radius**2

                    if (len(idx_liq[0]) > 0):
                        lwp_max[imatchcloud] = np.nanmax(lwp_iwp[idx_liq])
                    if (len(idx_ice[0]) > 0):
                        iwp_max[imatchcloud] = np.nanmax(lwp_iwp[idx_ice])

                    # import pdb; pdb.set_trace()

            return (
                nmatchcloud, 
                cell_area, 
                ctt_min,
                tir_min,
                cth_max,
                ctp_min,
                area_liq,
                area_ice,
                lwp_max,
                iwp_max
                )