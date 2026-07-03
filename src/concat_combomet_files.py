"""
Builds a single combomet-derived surface-parameter file for the cell-track matching.

Reads the monthly COR_15min_aerosol_meteorology_*.nc combomet files and keeps ONLY the
six surface variables P_met, T_aosmet, RH_met, DP_combomet, U_met, V_met. T_aosmet is
carried on the 'time_aosmet' axis in the source files but is identical to 'time_met', so
all six variables are collapsed onto a single 'time' coordinate (T reindexed onto the met
grid as a safety step). The concatenated record is de-duplicated in time and linearly
interpolated across NaNs (reproducing the method used in tmp_replace_surface_met.py).

From the interpolated surface data, four thermodynamic parameters are computed with MetPy
(same .data*units pattern as tmp_replace_surface_met.py):
    mixing_ratio                      [kg/kg]  from RH
    potential_temperature             [K]
    equivalent_potential_temperature  [K]      dewpoint = DP_combomet
    virtual_potential_temperature     [K]      from mixing_ratio

The output contains the six base variables (original names/attrs) plus the four computed
variables, all on a single 'time' coordinate, ready to be matched to cell tracks by
match_combomet_timeseries_celltracks.py.

Run with: module load python && conda activate mypy
"""
import os
import glob
import time
import numpy as np
import pandas as pd
import xarray as xr
from metpy.calc import (mixing_ratio_from_relative_humidity, potential_temperature,
                        equivalent_potential_temperature, virtual_potential_temperature)
from metpy.units import units

# Input combomet directory and output directory (stats_path from the config)
input_dir = '/global/cfs/projectdirs/m1657/enochjo/arm/combomet'
output_dir = '/global/cfs/cdirs/m1657/enochjo/taranis/taranis_corcsapr2cfrppiqcM1_celltracking.c1/stats/'

# Six surface variables to read from the combomet files.
# P_met, RH_met, DP_combomet, U_met, V_met are on 'time_met'; T_aosmet is on 'time_aosmet'.
MET_VARS = ['P_met', 'RH_met', 'DP_combomet', 'U_met', 'V_met']
AOS_VARS = ['T_aosmet']
BASE_VARS = ['P_met', 'T_aosmet', 'RH_met', 'DP_combomet', 'U_met', 'V_met']


if __name__ == '__main__':

    # Sorted list of monthly combomet files (filenames sort chronologically)
    filelist = sorted(glob.glob(f'{input_dir}/COR_15min_aerosol_meteorology_*.nc'))
    print(f'Number of input files: {len(filelist)}')

    # Read each file, collapse the six variables onto a single 'time' coordinate
    records = []
    for f in filelist:
        ds = xr.open_dataset(f, decode_times=True)

        # met-axis variables: rename their time dimension to 'time'
        rec = {v: ds[v].rename({'time_met': 'time'}) for v in MET_VARS}
        met_time = rec['P_met']['time']

        # aos-axis variables: rename then reindex onto the met time grid (they are
        # identical in these files; tolerance guards against any future mismatch)
        for v in AOS_VARS:
            da = ds[v].rename({'time_aosmet': 'time'})
            da = da.reindex(time=met_time, method='nearest',
                            tolerance=np.timedelta64(60, 's'))
            rec[v] = da

        records.append(xr.Dataset(rec))
        ds.close()

    # Concatenate all months along time
    combined = xr.concat(records, dim='time')
    print(f'Number of concatenated time steps: {combined.sizes["time"]}')

    # De-duplicate times (np.unique also sorts chronologically)
    _, index = np.unique(combined['time'].values, return_index=True)
    combined = combined.isel(time=index)
    print(f'Number of unique time steps: {combined.sizes["time"]}')

    # Mask unphysical negative relative humidity before interpolation. The combomet
    # record has a handful of negative RH_met spikes (e.g. down to ~-112% on
    # 2019-01-07 14:00-14:45); these are set to NaN so the linear interpolation below
    # fills them from the bracketing valid data. High values (>100%) are left untouched
    # as mild supersaturation can be physical.
    rh_attrs = combined['RH_met'].attrs
    n_bad_rh = int((combined['RH_met'] < 0).sum())
    combined['RH_met'] = combined['RH_met'].where(combined['RH_met'] >= 0)
    combined['RH_met'].attrs = rh_attrs
    print(f'Masked {n_bad_rh} negative RH_met value(s) to NaN before interpolation')

    # Linearly interpolate across NaNs (as in tmp_replace_surface_met.py)
    for v in BASE_VARS:
        attrs = combined[v].attrs
        combined[v] = combined[v].interpolate_na(dim='time', method='linear')
        combined[v].attrs = attrs

    # ------------------------------------------------------------------
    # Compute thermodynamic parameters with MetPy on the interpolated data
    # (same .data*units pattern as tmp_replace_surface_met.py)
    # ------------------------------------------------------------------
    P = combined['P_met'].data * units.hPa       # pressure
    T = combined['T_aosmet'].data * units.degC   # temperature
    RH = combined['RH_met'].data / 100.          # relative humidity [fraction]
    Td = combined['DP_combomet'].data * units.degC  # dewpoint

    mr_q = mixing_ratio_from_relative_humidity(P, T, RH).to('kg/kg')
    theta_q = potential_temperature(P, T).to('K')
    thetae_q = equivalent_potential_temperature(P, T, Td).to('K')
    thetav_q = virtual_potential_temperature(P, T, mr_q).to('K')

    def wrap(q, long_name, unit_str):
        da = xr.DataArray(q.magnitude, coords={'time': combined['time']}, dims=['time'])
        da.attrs = {'long_name': long_name, 'units': unit_str}
        return da

    combined['mixing_ratio'] = wrap(
        mr_q, 'Water vapor mixing ratio from relative humidity', 'kg/kg')
    combined['potential_temperature'] = wrap(
        theta_q, 'Potential temperature', 'K')
    combined['equivalent_potential_temperature'] = wrap(
        thetae_q, 'Equivalent potential temperature', 'K')
    combined['virtual_potential_temperature'] = wrap(
        thetav_q, 'Virtual potential temperature', 'K')

    # ------------------------------------------------------------------
    # Global attributes and output
    # ------------------------------------------------------------------
    startdate = pd.to_datetime(combined['time'].values[0]).strftime('%Y%m%d')
    enddate = pd.to_datetime(combined['time'].values[-1]).strftime('%Y%m%d')
    output_filename = f'{output_dir}COR_combomet_derived_surface_{startdate}_{enddate}_15min.nc'

    combined.attrs = {
        'title': 'Combomet-derived surface thermodynamic parameters',
        'Institution': 'Pacific Northwest National Laboratory',
        'Contact': 'Enoch Jo, enochjo2009@gmail.com',
        'Created_on': time.ctime(time.time()),
        'source_files': ', '.join(os.path.basename(f) for f in filelist),
        'processing_note': (
            'Concatenates the monthly COR_15min_aerosol_meteorology combomet files, keeping '
            'only P_met, T_aosmet, RH_met, DP_combomet, U_met, V_met on a single time '
            'coordinate, de-duplicated and linearly interpolated across NaNs. mixing_ratio, '
            'potential_temperature, equivalent_potential_temperature and '
            'virtual_potential_temperature are computed from these with MetPy '
            '(equivalent_potential_temperature uses DP_combomet as dewpoint; '
            'virtual_potential_temperature uses the RH-derived mixing_ratio).'),
    }

    # Delete file if it already exists
    if os.path.isfile(output_filename):
        os.remove(output_filename)

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in combined.data_vars}

    combined.to_netcdf(path=output_filename, mode='w', format='NETCDF4',
                       unlimited_dims='time', encoding=encoding)
    print(f'Output saved: {output_filename}')
