import os, glob
import numpy as np
import xarray as xr
import xesmf as xe
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster

def regrid_goes16(file_in, dir_output, basename_output, ds_dst, weight_file):
    """
    Regrids a GOES-16 file to an output grid using a pre-generated weight file and 
    write the output to a new netCDF file.

    Parameters:
    ===========
    file_in: string
        Input file name
    dir_output: string
        Output file directory
    basename_output: string
        Output file basename
    ds_dst: xarray Dataset
        Output file Dataset containing lat, lon variables
    weight_file: string
        Regridding weight file name.

    Returns:
    ===========
    status: <int>
        Returns 1
    """
    # Read input data
    ds = xr.open_dataset(file_in, drop_variables=('scanline_time','base_time','time_offset'))
    ds = ds.rename({'longitude':'lon', 'latitude':'lat'})
    ds = ds.rename({'element':'x', 'line':'y'})
    ds = ds.assign_coords({'time':ds.time})
    
    # Read weight file
    regridder_s2d = xe.Regridder(ds, ds_dst, 'nearest_s2d', reuse_weights=True, filename=weight_file)

    # Regrid the entire dataset
    ds_out = regridder_s2d(ds)

    # Expand dataset to create a time dimension
    ds_out = ds_out.expand_dims(dim='time', axis=0)
    # Change dataset time encoding
    time_unit = ds_out.time.dt.strftime('seconds since %Y-%m-%dT00:00:00.0').values.item()
    ds_out.time.encoding['units'] = time_unit

    # Create output file name
    outfile_timestr = ds_out.time.dt.strftime('%Y%m%d.%H%M%S').values.item()
    file_out = f'{dir_output}{basename_output}{outfile_timestr}.nc'

    # Set encoding/compression for all variables
    comp = dict(zlib=True, dtype='float32')
    encoding = {var: comp for var in ds_out.data_vars}
    # Update time variable dtype as 'double' for better precision
    time_dict = {'time': {'zlib':True, 'dtype':'float64', 'units':time_unit}}
    encoding.update(time_dict)

    # Write output netCDF file
    ds_out.to_netcdf(path=f'{file_out}', unlimited_dims='time', encoding=encoding)
    print(f'Output saved: {file_out}')

    status = 1
    return status


if __name__ == '__main__':

    # Set flag to run in serial (0) or parallel (1)
    run_parallel = 1
    # Number of workers for Dask
    n_workers = 35
    # Threads per worker
    threads_per_worker = 1

    # Input file directory
    year_month = '201902'
    dir_input = os.path.expandvars('$ICLASS') + f'cacti/corvisstpx2drectg16v4minnisX1.parallaxcorrected.c1/{year_month}/'
    files_in = sorted(glob.glob(f'{dir_input}corvisstpx2drectg16v4minnisX1.parallaxcorrected.c1.{year_month}*.*.cdf'))
    # files_in = sorted(glob.glob(f'{dir_input}corvisstpx2drectg16v4minnisX1.parallaxcorrected.c1.20190125.173034.cdf'))

    # Destination grid (CSAPR grid file)
    file_radar = os.path.expandvars('$ICLASS') + f'/cacti/radar_processing/corgridded_terrain.c0/CSAPR2_Taranis_Gridded_500m.Terrain_RangeMask.nc'

    # Pre-generated weight file (produced by test_regrid_goes16_to_csapr.ipynb)
    weight_file_dir = os.path.expandvars('$ICLASS') + f'cacti/corvisstpx2drectg16v4minnisX1.regridweights.c1/'
    # weight_file = f'{weight_file_dir}bilinear_728x672_441x441.nc'
    weight_file = f'{weight_file_dir}nearest_s2d_728x672_441x441.nc'

    # Output file directory
    dir_output = os.path.expandvars('$ICLASS') + f'cacti/corvisstpx2drectg16v4minnisX1.parallaxcorrected_regrid2csapr2gridded.c1/'
    basename_output = 'corvisstpx2drectg16v4minnisX1.regrid2csapr2gridded.c1.'


    # Read destination grid (CSAPR)
    ds_dst = xr.open_dataset(file_radar)
    ds_dst = ds_dst.drop(labels=('x','y'))
    ds_dst = ds_dst.rename({'latitude':'lat', 'longitude':'lon'})
    ds_dst = ds_dst.assign_coords({'lon':ds_dst.lon, 'lat':ds_dst.lat})

    nfile = len(files_in)

    ######################################################################################
    if run_parallel==0:
        # serial version
        for ifile in files_in:
            print(f'serial run')
            # print(reflectivity_files[ifile])
            status = regrid_goes16(ifile, dir_output, basename_output, ds_dst, weight_file)
            
    elif run_parallel==1:
        # parallel version
        print(f'parallel version by dask')

        # Initialize dask
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
        client = Client(cluster)

        results = []
        for ifile in files_in:
            print(f"Adding delayed file {ifile}")
            status = delayed(regrid_goes16)(ifile, dir_output, basename_output, ds_dst, weight_file)
            results.append(status)
        
        # Collect results from Dask
        print("Precompute step")
        results = dask.compute(*results)
        # print(results)