#!/bin/bash
#SBATCH -N 4                # Number of nodes
#SBATCH -n 64               # Total number of tasks
#SBATCH -t 04:00:00
#SBATCH -q regular
#SBATCH -C cpu              # Use CPU nodes
#SBATCH -A m1657            
#SBATCH -J sat_stats
#SBATCH --mail-user=enoch.jo@pnnl.gov
#SBATCH --mail-type=ALL
#SBATCH -o dask_output.log  # Output log file
#SBATCH -e dask_error.log   # Error log file

conda activate mypy

# Define the scheduler file path
scheduler_file=$SCRATCH/scheduler_file.json
rm -f $scheduler_file  # Remove any existing scheduler file

# Start the Dask scheduler
echo "Starting Dask scheduler..."
DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=3600s \
DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=3600s \
dask-scheduler --interface hsn0 --scheduler-file $scheduler_file &

dask_pid=$!  # Store the PID of the scheduler process

# Wait for the scheduler to start
sleep 5
until [ -f $scheduler_file ]; do
    sleep 5
done

# Start Dask workers
echo "Starting Dask workers..."
DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=3600s \
DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=3600s \
srun --ntasks=$SLURM_NTASKS dask-worker --scheduler-file $scheduler_file --interface hsn0 &

# Wait for workers to initialize
sleep 10

# Run your Python script
echo "Running Python script..."
cd /global/homes/e/enochjo/github/cacti-radar/src
python calc_sat_stats_to_celltracks.py config_goamazon.yaml

# Kill the Dask scheduler process
echo "Killing Dask scheduler..."
kill -9 $dask_pid