import os, sys, glob
import yaml
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

    # Get configuration file name from input
    config_file = sys.argv[1]
    # year_month = sys.argv[2]
    
    # Read configuration from yaml file
    stream = open(config_file, 'r')
    config = yaml.full_load(stream)
    run_parallel = config['run_parallel']
    n_workers = config['n_workers']
    threads_per_worker = config['threads_per_worker']
    dates_input = config['dates_input']
    dir_input = config['dir_input']
    dir_output = config['dir_output']
    file_radar = config['file_radar']
    weight_file = config['weight_file']
    basename_input = config['basename_input']
    basename_output = config['basename_output']

    # Find all input files
    files_in = sorted(glob.glob(f'{dir_input}{basename_input}{dates_input}*cdf'))

    # Read destination grid (CSAPR)
    ds_dst = xr.open_dataset(file_radar)
    ds_dst = ds_dst.drop(labels=('x','y'))
    ds_dst = ds_dst.rename({'latitude':'lat', 'longitude':'lon'})
    ds_dst = ds_dst.assign_coords({'lon':ds_dst.lon, 'lat':ds_dst.lat})

    nfile = len(files_in)
    print(f'Total number of files: {nfile}')

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