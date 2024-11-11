import os, sys, glob
import yaml
import xarray as xr
import xesmf as xe
import dask
from dask.distributed import Client, LocalCluster
from pyflextrkr.ft_regrid_func import make_grid4regridder, make_weight_file

def regrid_goes16(file_in):
    """
    Regrids a GOES-16 file to an output grid using a pre-generated weight file and 
    write the output to a new netCDF file.

    Parameters:
    ===========
    file_in: string
        Input file name

    Returns:
    ===========
    status: <int>
        Returns 1
    """
    # Read input data
    ds = xr.open_dataset(file_in, drop_variables=('scanline_time','base_time','time_offset'))
    # ds = ds.rename({'longitude':'lon', 'latitude':'lat'})
    # ds = ds.rename({'element':'x', 'line':'y'})
    ds = ds.assign_coords({'time':ds.time})
    
    # Read weight file
    regridder_s2d = xe.Regridder(grid_src, grid_dst, regrid_method, reuse_weights=True, filename=weight_filename)

    # Regrid the entire dataset
    ds_out = regridder_s2d(ds, keep_attrs=True)

    # Expand dataset to create a time dimension
    ds_out = ds_out.expand_dims(dim='time', axis=0)
    # Change dataset time encoding
    time_unit = ds_out.time.dt.strftime('seconds since %Y-%m-%dT00:00:00.0').values.item()
    ds_out.time.encoding['units'] = time_unit

    # Rename coordinates and dimensions
    ds_out = ds_out.rename({'lon':'longitude', 'lat':'latitude'})
    ds_out = ds_out.rename({'x':'lon', 'y':'lat'})

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
    ds_out.to_netcdf(path=f'{file_out}', mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
    print(f'Output saved: {file_out}')

    status = 1
    return status


if __name__ == '__main__':

    # Get configuration file name from input
    config_file = sys.argv[1]
    
    # Read configuration from yaml file
    stream = open(config_file, 'r')
    config = yaml.full_load(stream)
    run_parallel = config['run_parallel']
    n_workers = config['n_workers']
    threads_per_worker = config['threads_per_worker']
    dates_input = config['dates_input']
    dir_input = config['dir_input']
    dir_output = config['dir_output']
    gridfile_dst = config['gridfile_dst']
    weight_filename = config['weight_filename']
    basename_input = config['basename_input']
    basename_output = config['basename_output']
    regrid_method = config['regrid_method']

    # Create output directory
    os.makedirs(dir_output, exist_ok=True)

    # Find all input files
    files_in = sorted(glob.glob(f'{dir_input}{basename_input}{dates_input}*cdf'))
    nfile = len(files_in)
    print(f'Total number of files: {nfile}')

    # Get a source file for regridder
    gridfile_src = files_in[0]
    # Make source & destination grid data for regridder
    grid_src, grid_dst = make_grid4regridder(gridfile_src, config)
    # Build Regridder
    weight_filename = make_weight_file(gridfile_src, config)


    ######################################################################################
    if run_parallel==0:
        # serial version
        for ifile in files_in:
            print(f'serial run')
            # print(reflectivity_files[ifile])
            status = regrid_goes16(ifile)
            
    elif run_parallel==1:
        # parallel version
        print(f'parallel version by dask')

        # Initialize dask
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
        client = Client(cluster)

        results = []
        for ifile in files_in:
            print(f"Adding delayed file {ifile}")
            status = dask.delayed(regrid_goes16)(ifile)
            results.append(status)
        
        # Collect results from Dask
        # print("Processing")
        results = dask.compute(*results)