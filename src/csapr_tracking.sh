#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J cart
#SBATCH --mail-user=enoch.jo@pnnl.gov
#SBATCH --mail-type=ALL
#SBATCH -A m1657
#SBATCH -t 18:00:00

# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export PODMANHPC_MOUNT_PROGRAM=/global/common/shared/das/podman/bin/fuse-overlayfs-wrap

module load python
conda activate mypy

# PyFlextrkr
# python /global/homes/e/enochjo/github/PyFLEXTRKR/runscripts/run_celltracking.py /global/homes/e/enochjo/github/PyFLEXTRKR/config/config_csapr500m_example.yml
# Aerosol
# python /global/homes/e/enochjo/github/cacti-radar/src/match_aerosol_timeseries_celltracks.py /global/homes/e/enochjo/github/cacti-radar/src/config_csapr500m_lasso.yaml
# Interpsonde
# python /global/homes/e/enochjo/github/cacti-radar/src/match_interpsonde_timeseries_celltracks.py /global/homes/e/enochjo/github/cacti-radar/src/config_csapr500m_lasso.yaml
# GOES 16
# python /global/homes/e/enochjo/github/cacti-radar/src/calc_sat_stats_to_celltracks.py /global/homes/e/enochjo/github/cacti-radar/src/config_csapr500m_lasso.yaml
# Radar
python /global/homes/e/enochjo/github/cacti-radar/src/calc_3d_radarstats_to_celltracks.py /global/homes/e/enochjo/github/cacti-radar/src/config_csapr500m_lasso.yaml