"""
Combines cell track statistics file with GOES-16 and sounding parameter file
into a single netCDF file for the PI dataset.
"""
import os, sys
import time
import yaml
import xarray as xr

#########################################################
if __name__ == '__main__':

    # Get configuration file name from input
    config_file = sys.argv[1]
    # Read configuration from yaml file
    stream = open(config_file, 'r')
    config = yaml.full_load(stream)

    startdate = config['startdate']
    enddate = config['enddate']
    time_window = config['time_window']
    stats_path = config['stats_path']

    # Output filename
    output_path = stats_path
    output_filename = f'{output_path}celltrack_database_{startdate}_{enddate}.nc'

    # Input file basenames
    trackstats_file = f'{stats_path}/trackstats_{startdate}_{enddate}.nc'
    trackstats_3dfile = f'{stats_path}/stats_3d_ppi_{startdate}_{enddate}.nc'
    sat_file = f'{stats_path}/stats_goes16_{startdate}_{enddate}.nc'
    sonde_file = f'{stats_path}/interpsonde_parameters_celltrack_{startdate}_{enddate}.nc'

    # Track statistics file dimension names
    tracks_dimname = 'tracks'
    times_dimname = 'times'

    # Read 2D track data
    stats2d = xr.open_dataset(trackstats_file, drop_variables=['cloudnumber'], mask_and_scale=False, decode_times=True)
    time_res = stats2d.attrs['time_resolution_hour']
    pixel_radius = stats2d.attrs['pixel_radius_km']
    print(f"Number of cell tracks (2D): {stats2d.sizes['tracks']}")

    # Read 3D track data
    stats3d = xr.open_dataset(trackstats_3dfile, drop_variables=['base_time', 'cell_area'], mask_and_scale=False, decode_times=True)
    print(f"Number of cell tracks (3D): {stats3d.sizes['tracks']}")

    # Read satellite data
    sat = xr.open_dataset(sat_file, drop_variables=['base_time', 'cell_area'], mask_and_scale=False, decode_times=True)
    print(f"Number of tracks in satellite file: {sat.sizes['tracks']}")

    # Read sonde data
    sonde = xr.open_dataset(sonde_file, drop_variables=['time'], mask_and_scale=False, decode_times=True)
    print(f"Number of tracks in sounding file: {sonde.sizes['tracks']}")

    # # Subset sonde variables
    # sonde_varlist = [
    #     'CAPE_mu', 'CIN_IB_mu', 'LCL_height_mu', 'LFC_height_mu', 'EL_height_mu', 'initial_ht_parcel_mu', 
    #     'CAPE_sfc', 'CIN_IB_sfc', 'LCL_height_sfc', 'LFC_height_sfc', 'EL_height_sfc', 
    #     'rvap_850mb', 'rvap_700mb', 'rh_700mb', 'rh_500mb',
    #     'shear_mag_bulk_0to1km', 'shear_mag_bulk_0to3km', 'shear_mag_bulk_0to6km', 
    #     'shear_mag_bulk_0to9km', 'shear_mag_bulk_0to12km', 'shear_mag_bulk_BL',
    #     'U_850mb', 'V_850mb', 'U_700mb', 'V_700mb', 
    # ]
    # sonde = sonde[sonde_varlist]
    # Only keep reltime at 0 hour lag
    sonde = sonde.sel(reltime=0)
    # Remove 'reltime' coordinate
    sonde = sonde.reset_coords('reltime', drop=True)

    # Combine datasets by coordinates
    dsout = xr.combine_by_coords([stats2d, stats3d, sat, sonde], combine_attrs='override').compute()

    # Assign global attributes
    gattr_dict = {
        'title': 'CACTI C-SAPR2 cell tracking database',
        'Institution': 'Pacific Northwest National Laboratory',
        'Contact': 'Zhe Feng, zhe.feng@pnnl.gov',
        'Created_on':  time.ctime(time.time()),
        'startdate': startdate,
        'enddate': enddate,
        'time_resolution_hour': time_res,
        'pixel_radius_km': pixel_radius,
    }
    dsout.attrs = gattr_dict

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