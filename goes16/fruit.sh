#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -J fruit_multiply
#SBATCH --mail-user=enoch.jo@pnnl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

conda activate mypy
cd /global/homes/e/enochjo/github/cacti-radar/goes16/

#run the application:
srun -n 1 -c 1 --cpu_bind=cores python fruit_multiply.py config_singletime.yaml
