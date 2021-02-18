import numpy as np
import os, glob
import time, datetime, calendar
from math import pi
import xarray as xr
import pandas as pd
np.warnings.filterwarnings('ignore')
# from netCDF4 import Dataset, num2date, chartostring
# from scipy.ndimage import label, binary_dilation, generate_binary_structure
# from skimage.measure import regionprops
# from math import pi
# from scipy.stats import skew


#########################################################
def calc_3d_cellstats_singlefile(
    pixel_filename, 
    ppi_filename, 
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
    ppi_filename: string
        Input 3D PPI gridded radar file name
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
    max_dbz: <array>
        Maximum reflectivity profile for each matched cell

    """

    # Check if PPI file exist
    if os.path.isfile(ppi_filename):
        print(ppi_filename)

        # Read 3D PPI file
        dsv = xr.open_dataset(ppi_filename)
        zdim = dsv.dims['z']
        height = dsv['z'].values
        dbz = dsv['taranis_attenuation_corrected_reflectivity'].squeeze()
        zdr = dsv['taranis_attenuation_corrected_differential_reflectivity'].squeeze()
        kdp = dsv['kdp_pos_lp_reg'].squeeze()
        rainrate = dsv['taranis_rain_rate'].squeeze()
        Dm = dsv['taranis_Dm'].squeeze()
        lwc = dsv['lwc_combined'].squeeze()
        # hid = dsv['hydrometeor_identification_post_grid'].squeeze()
        # dsv.close()
        # import pdb; pdb.set_trace()

        # Read pixel-level track file
        ds = xr.open_dataset(pixel_filename, decode_times=False)
        xdim = ds.dims['lon']
        ydim = ds.dims['lat']
        lon = ds['longitude'].values
        lat = ds['latitude'].values
        # Replace lon/lat coordinates with x/y
        ds['lon'] = dsv['x'].values
        ds['lat'] = dsv['y'].values
        ds = ds.rename({'lat':'y', 'lon':'x'})
        # Read variables
        cloudid_basetime = ds['basetime'].values
        tracknumbermap = ds['tracknumber'].squeeze()
        tracknumbermap_cmask = ds['tracknumber_cmask2'].squeeze()
        # tracknumbermap = ds['tracknumber'].squeeze().values
        # tracknumbermap_cmask = ds['tracknumber_cmask2'].squeeze().values
        # import pdb; pdb.set_trace()
        # ds.close()

        # Create arrays for output statistics
        nmatchcloud = len(idx_track)
        cell_area = np.full((nmatchcloud), np.nan, dtype=float)
        max_dbz = np.full((nmatchcloud, zdim), np.nan, dtype=float)
        npix_dbz0 = np.full((nmatchcloud, zdim), np.nan, dtype=float)
        npix_dbz10 = np.full((nmatchcloud, zdim), np.nan, dtype=float)
        npix_dbz20 = np.full((nmatchcloud, zdim), np.nan, dtype=float)
        npix_dbz30 = np.full((nmatchcloud, zdim), np.nan, dtype=float)
        npix_dbz40 = np.full((nmatchcloud, zdim), np.nan, dtype=float)
        npix_dbz50 = np.full((nmatchcloud, zdim), np.nan, dtype=float)
        npix_dbz60 = np.full((nmatchcloud, zdim), np.nan, dtype=float)
        max_zdr = np.full((nmatchcloud, zdim), np.nan, dtype=float)
        max_kdp = np.full((nmatchcloud, zdim), np.nan, dtype=float)
        max_rainrate = np.full((nmatchcloud, zdim), np.nan, dtype=float)
        max_Dm = np.full((nmatchcloud, zdim), np.nan, dtype=float)
        max_lwc = np.full((nmatchcloud, zdim), np.nan, dtype=float)
        volrain = np.full((nmatchcloud, zdim), np.nan, dtype=float)

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
                    sub_lwc = lwc.where(tracknumbermap == itracknum, drop=True).values
                    # sub_hid = hid.where(tracknumbermap == itracknum, drop=True).values

                    # # Fill array with data
                    # import pdb; pdb.set_trace()
                    # filtered_dbz[:,icloudlocationy, icloudlocationx] = np.copy(dbz[:,icloudlocationy,icloudlocationx])
                    # filtered_zdr[:,icloudlocationy, icloudlocationx] = np.copy(zdr[:,icloudlocationy,icloudlocationx])
                    # filtered_kdp[:,icloudlocationy, icloudlocationx] = np.copy(kdp[:,icloudlocationy,icloudlocationx])
                    # filtered_rainrate[:,icloudlocationy, icloudlocationx] = np.copy(rainrate[:,icloudlocationy,icloudlocationx])
                    # filtered_Dm[:,icloudlocationy, icloudlocationx] = np.copy(Dm[:,icloudlocationy,icloudlocationx])
                    # filtered_lwc[:,icloudlocationy, icloudlocationx] = np.copy(lwc[:,icloudlocationy,icloudlocationx])
                    # filtered_hid[:,icloudlocationy, icloudlocationx] = np.copy(hid[:,icloudlocationy,icloudlocationx])

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
                    # sub_dbz = np.copy(filtered_dbz[:, miny:maxy, minx:maxx])
                    
                    cell_area[imatchcloud] = inpix_cloud * pixel_radius**2
                    
                    # Calculate new statistics of the cloud
                    # Max profile
                    max_dbz[imatchcloud,:] = np.nanmax(sub_dbz, axis=(1,2))
                    max_zdr[imatchcloud,:] = np.nanmax(sub_zdr, axis=(1,2))
                    max_kdp[imatchcloud,:] = np.nanmax(sub_kdp, axis=(1,2))
                    max_rainrate[imatchcloud,:] = np.nanmax(sub_rainrate, axis=(1,2))
                    max_Dm[imatchcloud,:] = np.nanmax(sub_Dm, axis=(1,2))
                    max_lwc[imatchcloud,:] = np.nanmax(sub_lwc, axis=(1,2))

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

                    # Loop over each vertical level
                    # for iz in range(zdim):
                    #     # print(height[iz])
                    #     # Count number of pixels > X dBZ
                    #     npix_dbz0[imatchcloud,iz] = np.count_nonzero(sub_dbz[iz,:,:] > 0)
                    #     npix_dbz10[imatchcloud,iz] = np.count_nonzero(sub_dbz[iz,:,:] > 10)
                    #     npix_dbz20[imatchcloud,iz] = np.count_nonzero(sub_dbz[iz,:,:] > 20)
                    #     npix_dbz30[imatchcloud,iz] = np.count_nonzero(sub_dbz[iz,:,:] > 30)
                    #     npix_dbz40[imatchcloud,iz] = np.count_nonzero(sub_dbz[iz,:,:] > 40)
                    #     npix_dbz50[imatchcloud,iz] = np.count_nonzero(sub_dbz[iz,:,:] > 50)
                    #     npix_dbz60[imatchcloud,iz] = np.count_nonzero(sub_dbz[iz,:,:] > 60)

                    # import matplotlib.pyplot as plt
                    # import pdb; pdb.set_trace()
                else:
                    print(f'No cell matching track # {itracknum}')

            return (
                nmatchcloud, 
                cell_area, 
                max_dbz,
                npix_dbz0,
                npix_dbz10,
                npix_dbz20,
                npix_dbz30,
                npix_dbz40,
                npix_dbz50,
                npix_dbz60,
                max_zdr,
                max_kdp,
                max_rainrate,
                max_Dm,
                max_lwc,
                volrain
                )