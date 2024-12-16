import xarray as xr
import os
import glob
import yaml
import dask
import warnings
from dask.distributed import Client, LocalCluster

# 90% sure code worked.
# But ended up not being used as Zhe and Enoch found out ARM already has these individual satellite files available for order


def process_file(filepath, dir_output):
    """
    Process a single NetCDF file: split it into individual time slices 
    and save each slice as a separate file.
    """
    # Load the NetCDF file lazily with Dask
    ds = xr.open_dataset(filepath, chunks={"time": 1})  # Load one time step at a time

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
        base_filename = os.path.splitext(os.path.basename(filepath))[0][:-7]
        output_file = f"{base_filename}.{time_str}.nc"
        output_filepath = os.path.join(dir_output, output_file)

        # For some reason, .to_netcdf() wants to output lat and lon as integers.
        # Need to override that.
        ds_single_time['latitude'].encoding = {'dtype': 'float32', '_FillValue': None}
        ds_single_time['longitude'].encoding = {'dtype': 'float32', '_FillValue': None}
        
        # Save the single time slice to a new NetCDF file
        ds_single_time.to_netcdf(output_filepath)


def main(config_file):
    # Read configuration from yaml file
    with open(config_file, 'r') as stream:
        config = yaml.full_load(stream)

    run_parallel = config['run_parallel']
    n_workers = config['n_workers']
    threads_per_worker = config['threads_per_worker']
    dates_input = config['dates_input']
    basename_input = config['basename_input']
    dir_input = config['dir_input']
    dir_output = config['dir_output']

    # Create output directory
    os.makedirs(dir_output, exist_ok=True)

    # Find all input files
    files_in = sorted(glob.glob(f'{dir_input}{basename_input}{dates_input}*.cdf'))
    nfile = len(files_in)
    print(f'Total number of files: {nfile}')

    # Suppress SerializationWarning
    # warnings.filterwarnings("ignore")

    ######################################################################################
    if run_parallel == 0:
        # Serial version
        print(f"Running in serial mode...")
        for filepath in files_in:
            process_file(filepath, dir_output)

    elif run_parallel == 1:
        # Parallel version with Dask
        print(f"Running in parallel mode with Dask...")

        # Initialize Dask
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
        client = Client(cluster)
        print(client)

        # Submit tasks to Dask
        futures = [client.submit(process_file, filepath, dir_output) for filepath in files_in]

        # Wait for all tasks to complete
        dask.distributed.wait(futures)

        print(f"All files processed in parallel.")

    else:
        raise ValueError("Invalid value for 'run_parallel'. Use 0 for serial or 1 for parallel.")


if __name__ == '__main__':
    import sys
    config_file = sys.argv[1]
    main(config_file)
