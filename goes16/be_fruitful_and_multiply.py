import xarray as xr
import os, sys, glob
import yaml
import dask
import warnings
# from xarray.backends.netCDF4_ import SerializationWarning
from dask.distributed import Client, LocalCluster

# Code did not end up being used.

def fruitful_and_multiply(file_in):

    # Loop through each NetCDF file in the input directory
    for file in os.listdir(dir_input):
        if file.endswith(".cdf"):
            # Load the NetCDF file
            filepath = os.path.join(dir_input, file)
            ds = xr.open_dataset(filepath)

            # Loop through the time dimension
            for i, timestamp in enumerate(ds.time):
                # Extract hour, minute, and second from the timestamp
                hour = timestamp.dt.hour.item()
                minute = timestamp.dt.minute.item()
                second = timestamp.dt.second.item()

                # Format hour, minute, and second as a 6-digit string (HHMMSS)
                time_str = f"{hour:02d}{minute:02d}{second:02d}"

                # Select the specific time slice
                ds_single_time = ds.isel(time=i)
                
                # Construct a filename for the output file
                base_filename = os.path.splitext(file)[0][:-7]
                output_file = f"{base_filename}.{time_str}.nc"
                output_filepath = os.path.join(dir_output, output_file)
                # import pdb;pdb.set_trace()
                # Save the single time slice to a new NetCDF file
                ds_single_time.to_netcdf(output_filepath,format="NETCDF3_CLASSIC")
            
            # Close the dataset to free resources
            # ds.close()


if __name__ == '__main__':

    # Get configuration file name from input
    config_file = sys.argv[1]
    
    # Read configuration from yaml file
    stream = open(config_file, 'r')
    config = yaml.full_load(stream)

    run_parallel = config['run_parallel']
    n_workers = config['n_workers']
    threads_per_worker = config['threads_per_worker']

    dir_input = config['dir_input']
    dir_output = config['dir_output']

    # Create output directory
    os.makedirs(dir_output, exist_ok=True)

    # Find all input files
    files_in = sorted(glob.glob(f'{dir_input}*cdf'))
    nfile = len(files_in)
    print(f'Total number of files: {nfile}')

    # Suppress SerializationWarning
    warnings.filterwarnings("ignore")

    ######################################################################################
    if run_parallel==0:
        # serial version
        for ifile in files_in:
            print(f'serial run')
            # print(reflectivity_files[ifile])
            status = fruitful_and_multiply(ifile)
            
    elif run_parallel==1:
        # parallel version
        print(f'parallel version by dask')

        # Initialize dask
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
        client = Client(cluster)

        results = []
        for ifile in files_in:
            print(f"Adding delayed file {ifile}")
            status = dask.delayed(fruitful_and_multiply)(ifile)
            results.append(status)
        
        # Collect results from Dask
        # print("Processing")
        results = dask.compute(*results)