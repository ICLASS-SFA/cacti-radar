import numpy as np
import os, glob, sys
import time, datetime, calendar
# from pytz import timezone, utc
import pytz
import xarray as xr

if __name__ == '__main__':

    #########################################################
    # Load MCS track stats
    # print('Loading track stats file')
    print((time.ctime()))

    # startdate = '20181110.1800'
    # enddate = '20181112.2359'
    # startdate = '20181101.0000'
    # enddate = '20181130.2359'
    startdate = '20181015.0000'
    enddate = '20190303.0000'

    # Maximum time difference allowed to match the datasets
    time_window = 10  # [second]

    # Input/output file locations
    # stats_path = os.path.expandvars('$ICLASS') + f'/cacti/radar_processing/taranis_corcsapr2cfrppiqcM1_celltracking.c1/stats/'
    stats_path = os.path.expandvars('$ICLASS') + f'/cacti/radar_processing/taranis_corcsapr2cfrppiqcM1_mpgridded_celltracking.c1/stats/'
    sonde_path = os.path.expandvars('$ICLASS') + f'cacti/sounding_stats/'
    output_path = stats_path

    # Input file basenames
    stats_filebase = 'stats_tracknumbersv1.0_'
    # pixel_filebase = 'celltracks_'
    # sat_filebase = 'corvisstpx2drectg16v4minnisX1.regrid2csapr2gridded.c1.'

    # Output statistics filename
    output_filename = f'{output_path}interpsonde_celltrack_{startdate}_{enddate}.nc'

    # Track statistics file dimension names
    trackdimname = 'tracks'
    timedimname = 'times'
    relative_time_dimname = 'reltime'

    # Track statistics file
    trackstats_file = f'{stats_path}{stats_filebase}{startdate}_{enddate}.nc'
    muparcel_file = f'{sonde_path}CACTI_M1_interpsonde_muparcel_stats.nc'
    uvq_file = f'{sonde_path}CACTI_M1_interpsonde_wind_humidity_indices.nc'


    # Read track statistics file
    print(trackstats_file)
    dsstats = xr.open_dataset(trackstats_file, decode_times=False)
    ntracks = dsstats.dims[trackdimname]
    # ntimes = dsstats.dims[timedimname]
    stats_basetime = dsstats['basetime']
    basetime_units = dsstats['basetime'].units
    # cell_area = dsstats['cell_area'].values
    # pixel_radius = dsstats.attrs['pixel_radius_km']
    # Get cell initiation time
    stats_basetime0 = stats_basetime.sel(times=0).data
    dsstats.close()

    print(f'Total Number of Tracks: {ntracks}')

    # Read sonde MU parcel file
    dsmup = xr.open_dataset(muparcel_file, decode_times=False)
    year_sonde = dsmup.year.data
    month_sonde = dsmup.month.data
    day_sonde = dsmup.day.data
    hour_sonde = dsmup.hour.data
    minute_sonde = dsmup.minute.data
    seconds_sonde = dsmup.seconds.data
    # height_sonde = dsmup.height.data
    # cape_sonde = dsmup.cape.data
    # cin_sonde = dsmup.cin.data
    # lcl_p_sonde = dsmup.lcl_p.data
    # lcl_t_sonde = dsmup.lcl_t.data
    # lcl_z_sonde = dsmup.lcl_z.data
    # lnb_p_sonde = dsmup.lnb_p.data
    # lnb_t_sonde = dsmup.lnb_t.data
    # lnb_z_sonde = dsmup.lnb_z.data

    ntimes_mup = dsmup.dims['time']

    # Read sonde wind/humidity file
    dsuvq = xr.open_dataset(uvq_file, decode_times=False)
    ntimes_uvq = dsuvq.dims['time']

    # Double check to make sure the number of times between the sonde files is the same
    # The assumption is that both files have the exact same times
    if (ntimes_uvq != ntimes_mup):
        print('Error: number of times not the same between the two sonde files!')
        sys.exti()
    # import pdb; pdb.set_trace()
    
    # Calculate sonde basetime
    sonde_basetime = np.full(ntimes_mup, dtype=float, fill_value=np.nan)
    for tt in range(0, ntimes_mup):
        sonde_basetime[tt] = calendar.timegm(datetime.datetime(
            year_sonde[tt], month_sonde[tt], day_sonde[tt], hour_sonde[tt], minute_sonde[tt], 0, tzinfo=pytz.UTC).timetuple())

    
    # Create matched sonde variable arrays
    # sonde_matchtime = np.full(ntracks, dtype=float, fill_value=np.nan)
    # height = np.full(ntracks, dtype=float, fill_value=np.nan)
    # cape = np.full(ntracks, dtype=float, fill_value=np.nan)
    # cin = np.full(ntracks, dtype=float, fill_value=np.nan)
    # lcl_p = np.full(ntracks, dtype=float, fill_value=np.nan)
    # lcl_t = np.full(ntracks, dtype=float, fill_value=np.nan)
    # lcl_z = np.full(ntracks, dtype=float, fill_value=np.nan)
    # lnb_p = np.full(ntracks, dtype=float, fill_value=np.nan)
    # lnb_t = np.full(ntracks, dtype=float, fill_value=np.nan)
    # lnb_z = np.full(ntracks, dtype=float, fill_value=np.nan)
    # lfc_p = np.full(ntracks, dtype=float, fill_value=np.nan)
    # lfc_t = np.full(ntracks, dtype=float, fill_value=np.nan)
    # lfc_z = np.full(ntracks, dtype=float, fill_value=np.nan)
    # shear3mag = np.full(ntracks, dtype=float, fill_value=np.nan)
    # shear3dir = np.full(ntracks, dtype=float, fill_value=np.nan)
    # shear6mag = np.full(ntracks, dtype=float, fill_value=np.nan)
    # shear6dir = np.full(ntracks, dtype=float, fill_value=np.nan)
    # u10z = np.full(ntracks, dtype=float, fill_value=np.nan)
    # v10z = np.full(ntracks, dtype=float, fill_value=np.nan)
    # q10z = np.full(ntracks, dtype=float, fill_value=np.nan)
    # rh10z = np.full(ntracks, dtype=float, fill_value=np.nan)
    # u1500z = np.full(ntracks, dtype=float, fill_value=np.nan)
    # v1500z = np.full(ntracks, dtype=float, fill_value=np.nan)
    # q1500z = np.full(ntracks, dtype=float, fill_value=np.nan)
    # rh1500z = np.full(ntracks, dtype=float, fill_value=np.nan)
    # u3000z = np.full(ntracks, dtype=float, fill_value=np.nan)
    # v3000z = np.full(ntracks, dtype=float, fill_value=np.nan)
    # q3000z = np.full(ntracks, dtype=float, fill_value=np.nan)
    # rh3000z = np.full(ntracks, dtype=float, fill_value=np.nan)
    # u6000z = np.full(ntracks, dtype=float, fill_value=np.nan)
    # v6000z = np.full(ntracks, dtype=float, fill_value=np.nan)
    # q6000z = np.full(ntracks, dtype=float, fill_value=np.nan)
    # rh6000z = np.full(ntracks, dtype=float, fill_value=np.nan)
    # u850p = np.full(ntracks, dtype=float, fill_value=np.nan)
    # v850p = np.full(ntracks, dtype=float, fill_value=np.nan)
    # q850p = np.full(ntracks, dtype=float, fill_value=np.nan)
    # rh850p = np.full(ntracks, dtype=float, fill_value=np.nan)
    # u700p = np.full(ntracks, dtype=float, fill_value=np.nan)
    # v700p = np.full(ntracks, dtype=float, fill_value=np.nan)
    # q700p = np.full(ntracks, dtype=float, fill_value=np.nan)
    # rh700p = np.full(ntracks, dtype=float, fill_value=np.nan)
    # u500p = np.full(ntracks, dtype=float, fill_value=np.nan)
    # v500p = np.full(ntracks, dtype=float, fill_value=np.nan)
    # q500p = np.full(ntracks, dtype=float, fill_value=np.nan)
    # rh500p = np.full(ntracks, dtype=float, fill_value=np.nan)
    # u300p = np.full(ntracks, dtype=float, fill_value=np.nan)
    # v300p = np.full(ntracks, dtype=float, fill_value=np.nan)

    # Number of relative sonde time to track initiation (-3, -2, -1, 0 hour)
    nreltime = 4
    sonde_matchtime = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    height = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    cape = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    cin = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    lcl_p = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    lcl_t = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    lcl_z = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    lnb_p = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    lnb_t = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    lnb_z = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    lfc_p = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    lfc_t = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    lfc_z = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    shear3mag = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    shear3dir = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    shear6mag = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    shear6dir = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    u10z = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    v10z = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    q10z = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    rh10z = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    u1500z = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    v1500z = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    q1500z = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    rh1500z = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    u3000z = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    v3000z = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    q3000z = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    rh3000z = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    u6000z = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    v6000z = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    q6000z = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    rh6000z = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    u850p = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    v850p = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    q850p = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    rh850p = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    u700p = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    v700p = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    q700p = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    rh700p = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    u500p = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    v500p = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    q500p = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    rh500p = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    u300p = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)
    v300p = np.full((ntracks,nreltime), dtype=float, fill_value=np.nan)

    # Loop over each cell track to find match sonde time
    for tt in range(0, ntracks):
    # for tt in range(0, 10):
        # Tracks at initiation time
        matchindex = np.where(np.abs(sonde_basetime - stats_basetime0[tt]) < time_window)[0]
        # Initiation time -1, -2, -3 hour 
        matchindex_1h = np.where(np.abs(sonde_basetime - (stats_basetime0[tt] - 3600*1)) < time_window)[0]
        matchindex_2h = np.where(np.abs(sonde_basetime - (stats_basetime0[tt] - 3600*2)) < time_window)[0]
        matchindex_3h = np.where(np.abs(sonde_basetime - (stats_basetime0[tt] - 3600*3)) < time_window)[0]
        # if (len(matchindex) == 1) & (len(matchindex_1h) == 1):
        if (len(matchindex) == 1) & (len(matchindex_1h) == 1) & \
            (len(matchindex_2h) == 1) & (len(matchindex_3h) == 1):
            print(f'Track #: {tt}')
            sonde_matchtime[tt,0] = sonde_basetime[matchindex]
            height[tt,0] = dsmup.height.data[matchindex]
            cape[tt,0] = dsmup.cape.data[matchindex]
            cin[tt,0] = dsmup.cin.data[matchindex]
            lcl_p[tt,0] = dsmup.lcl_p.data[matchindex]
            lcl_t[tt,0] = dsmup.lcl_t.data[matchindex]
            lcl_z[tt,0] = dsmup.lcl_z.data[matchindex]
            lnb_p[tt,0] = dsmup.lnb_p.data[matchindex]
            lnb_t[tt,0] = dsmup.lnb_t.data[matchindex]
            lnb_z[tt,0] = dsmup.lnb_z.data[matchindex]
            lfc_p[tt,0] = dsmup.lfc_p.data[matchindex]
            lfc_t[tt,0] = dsmup.lfc_t.data[matchindex]
            lfc_z[tt,0] = dsmup.lfc_z.data[matchindex]
            shear3mag[tt,0] = dsuvq.shear3mag.data[matchindex]
            shear3dir[tt,0] = dsuvq.shear3dir.data[matchindex]
            shear6mag[tt,0] = dsuvq.shear6mag.data[matchindex]
            shear6dir[tt,0] = dsuvq.shear6dir.data[matchindex]
            u10z[tt,0] = dsuvq.u10z.data[matchindex]
            v10z[tt,0] = dsuvq.v10z.data[matchindex]
            q10z[tt,0] = dsuvq.q10z.data[matchindex]
            rh10z[tt,0] = dsuvq.rh10z.data[matchindex]
            u1500z[tt,0] = dsuvq.u1500z.data[matchindex]
            v1500z[tt,0] = dsuvq.v1500z.data[matchindex]
            q1500z[tt,0] = dsuvq.q1500z.data[matchindex]
            rh1500z[tt,0] = dsuvq.rh1500z.data[matchindex]
            u3000z[tt,0] = dsuvq.u3000z.data[matchindex]
            v3000z[tt,0] = dsuvq.v3000z.data[matchindex]
            q3000z[tt,0] = dsuvq.q3000z.data[matchindex]
            rh3000z[tt,0] = dsuvq.rh3000z.data[matchindex]
            u6000z[tt,0] = dsuvq.u6000z.data[matchindex]
            v6000z[tt,0] = dsuvq.v6000z.data[matchindex]
            q6000z[tt,0] = dsuvq.q6000z.data[matchindex]
            rh6000z[tt,0] = dsuvq.rh6000z.data[matchindex]
            u850p[tt,0] = dsuvq.u850p.data[matchindex]
            v850p[tt,0] = dsuvq.v850p.data[matchindex]
            q850p[tt,0] = dsuvq.q850p.data[matchindex]
            rh850p[tt,0] = dsuvq.rh850p.data[matchindex]
            u700p[tt,0] = dsuvq.u700p.data[matchindex]
            v700p[tt,0] = dsuvq.v700p.data[matchindex]
            q700p[tt,0] = dsuvq.q700p.data[matchindex]
            rh700p[tt,0] = dsuvq.rh700p.data[matchindex]
            u500p[tt,0] = dsuvq.u500p.data[matchindex]
            v500p[tt,0] = dsuvq.v500p.data[matchindex]
            q500p[tt,0] = dsuvq.q500p.data[matchindex]
            rh500p[tt,0] = dsuvq.rh500p.data[matchindex]
            u300p[tt,0] = dsuvq.u300p.data[matchindex]
            v300p[tt,0] = dsuvq.v300p.data[matchindex]

            # -1 hour
            sonde_matchtime[tt,1] = sonde_basetime[matchindex_1h]
            height[tt,1] = dsmup.height.data[matchindex_1h]
            cape[tt,1] = dsmup.cape.data[matchindex_1h]
            cin[tt,1] = dsmup.cin.data[matchindex_1h]
            lcl_p[tt,1] = dsmup.lcl_p.data[matchindex_1h]
            lcl_t[tt,1] = dsmup.lcl_t.data[matchindex_1h]
            lcl_z[tt,1] = dsmup.lcl_z.data[matchindex_1h]
            lnb_p[tt,1] = dsmup.lnb_p.data[matchindex_1h]
            lnb_t[tt,1] = dsmup.lnb_t.data[matchindex_1h]
            lnb_z[tt,1] = dsmup.lnb_z.data[matchindex_1h]
            lfc_p[tt,1] = dsmup.lfc_p.data[matchindex_1h]
            lfc_t[tt,1] = dsmup.lfc_t.data[matchindex_1h]
            lfc_z[tt,1] = dsmup.lfc_z.data[matchindex_1h]
            shear3mag[tt,1] = dsuvq.shear3mag.data[matchindex_1h]
            shear3dir[tt,1] = dsuvq.shear3dir.data[matchindex_1h]
            shear6mag[tt,1] = dsuvq.shear6mag.data[matchindex_1h]
            shear6dir[tt,1] = dsuvq.shear6dir.data[matchindex_1h]
            u10z[tt,1] = dsuvq.u10z.data[matchindex_1h]
            v10z[tt,1] = dsuvq.v10z.data[matchindex_1h]
            q10z[tt,1] = dsuvq.q10z.data[matchindex_1h]
            rh10z[tt,1] = dsuvq.rh10z.data[matchindex_1h]
            u1500z[tt,1] = dsuvq.u1500z.data[matchindex_1h]
            v1500z[tt,1] = dsuvq.v1500z.data[matchindex_1h]
            q1500z[tt,1] = dsuvq.q1500z.data[matchindex_1h]
            rh1500z[tt,1] = dsuvq.rh1500z.data[matchindex_1h]
            u3000z[tt,1] = dsuvq.u3000z.data[matchindex_1h]
            v3000z[tt,1] = dsuvq.v3000z.data[matchindex_1h]
            q3000z[tt,1] = dsuvq.q3000z.data[matchindex_1h]
            rh3000z[tt,1] = dsuvq.rh3000z.data[matchindex_1h]
            u6000z[tt,1] = dsuvq.u6000z.data[matchindex_1h]
            v6000z[tt,1] = dsuvq.v6000z.data[matchindex_1h]
            q6000z[tt,1] = dsuvq.q6000z.data[matchindex_1h]
            rh6000z[tt,1] = dsuvq.rh6000z.data[matchindex_1h]
            u850p[tt,1] = dsuvq.u850p.data[matchindex_1h]
            v850p[tt,1] = dsuvq.v850p.data[matchindex_1h]
            q850p[tt,1] = dsuvq.q850p.data[matchindex_1h]
            rh850p[tt,1] = dsuvq.rh850p.data[matchindex_1h]
            u700p[tt,1] = dsuvq.u700p.data[matchindex_1h]
            v700p[tt,1] = dsuvq.v700p.data[matchindex_1h]
            q700p[tt,1] = dsuvq.q700p.data[matchindex_1h]
            rh700p[tt,1] = dsuvq.rh700p.data[matchindex_1h]
            u500p[tt,1] = dsuvq.u500p.data[matchindex_1h]
            v500p[tt,1] = dsuvq.v500p.data[matchindex_1h]
            q500p[tt,1] = dsuvq.q500p.data[matchindex_1h]
            rh500p[tt,1] = dsuvq.rh500p.data[matchindex_1h]
            u300p[tt,1] = dsuvq.u300p.data[matchindex_1h]
            v300p[tt,1] = dsuvq.v300p.data[matchindex_1h]

            # -2 hour
            sonde_matchtime[tt,2] = sonde_basetime[matchindex_2h]
            height[tt,2] = dsmup.height.data[matchindex_2h]
            cape[tt,2] = dsmup.cape.data[matchindex_2h]
            cin[tt,2] = dsmup.cin.data[matchindex_2h]
            lcl_p[tt,2] = dsmup.lcl_p.data[matchindex_2h]
            lcl_t[tt,2] = dsmup.lcl_t.data[matchindex_2h]
            lcl_z[tt,2] = dsmup.lcl_z.data[matchindex_2h]
            lnb_p[tt,2] = dsmup.lnb_p.data[matchindex_2h]
            lnb_t[tt,2] = dsmup.lnb_t.data[matchindex_2h]
            lnb_z[tt,2] = dsmup.lnb_z.data[matchindex_2h]
            lfc_p[tt,2] = dsmup.lfc_p.data[matchindex_2h]
            lfc_t[tt,2] = dsmup.lfc_t.data[matchindex_2h]
            lfc_z[tt,2] = dsmup.lfc_z.data[matchindex_2h]
            shear3mag[tt,2] = dsuvq.shear3mag.data[matchindex_2h]
            shear3dir[tt,2] = dsuvq.shear3dir.data[matchindex_2h]
            shear6mag[tt,2] = dsuvq.shear6mag.data[matchindex_2h]
            shear6dir[tt,2] = dsuvq.shear6dir.data[matchindex_2h]
            u10z[tt,2] = dsuvq.u10z.data[matchindex_2h]
            v10z[tt,2] = dsuvq.v10z.data[matchindex_2h]
            q10z[tt,2] = dsuvq.q10z.data[matchindex_2h]
            rh10z[tt,2] = dsuvq.rh10z.data[matchindex_2h]
            u1500z[tt,2] = dsuvq.u1500z.data[matchindex_2h]
            v1500z[tt,2] = dsuvq.v1500z.data[matchindex_2h]
            q1500z[tt,2] = dsuvq.q1500z.data[matchindex_2h]
            rh1500z[tt,2] = dsuvq.rh1500z.data[matchindex_2h]
            u3000z[tt,2] = dsuvq.u3000z.data[matchindex_2h]
            v3000z[tt,2] = dsuvq.v3000z.data[matchindex_2h]
            q3000z[tt,2] = dsuvq.q3000z.data[matchindex_2h]
            rh3000z[tt,2] = dsuvq.rh3000z.data[matchindex_2h]
            u6000z[tt,2] = dsuvq.u6000z.data[matchindex_2h]
            v6000z[tt,2] = dsuvq.v6000z.data[matchindex_2h]
            q6000z[tt,2] = dsuvq.q6000z.data[matchindex_2h]
            rh6000z[tt,2] = dsuvq.rh6000z.data[matchindex_2h]
            u850p[tt,2] = dsuvq.u850p.data[matchindex_2h]
            v850p[tt,2] = dsuvq.v850p.data[matchindex_2h]
            q850p[tt,2] = dsuvq.q850p.data[matchindex_2h]
            rh850p[tt,2] = dsuvq.rh850p.data[matchindex_2h]
            u700p[tt,2] = dsuvq.u700p.data[matchindex_2h]
            v700p[tt,2] = dsuvq.v700p.data[matchindex_2h]
            q700p[tt,2] = dsuvq.q700p.data[matchindex_2h]
            rh700p[tt,2] = dsuvq.rh700p.data[matchindex_2h]
            u500p[tt,2] = dsuvq.u500p.data[matchindex_2h]
            v500p[tt,2] = dsuvq.v500p.data[matchindex_2h]
            q500p[tt,2] = dsuvq.q500p.data[matchindex_2h]
            rh500p[tt,2] = dsuvq.rh500p.data[matchindex_2h]
            u300p[tt,2] = dsuvq.u300p.data[matchindex_2h]
            v300p[tt,2] = dsuvq.v300p.data[matchindex_2h]

            # -3 hour
            sonde_matchtime[tt,3] = sonde_basetime[matchindex_3h]
            height[tt,3] = dsmup.height.data[matchindex_3h]
            cape[tt,3] = dsmup.cape.data[matchindex_3h]
            cin[tt,3] = dsmup.cin.data[matchindex_3h]
            lcl_p[tt,3] = dsmup.lcl_p.data[matchindex_3h]
            lcl_t[tt,3] = dsmup.lcl_t.data[matchindex_3h]
            lcl_z[tt,3] = dsmup.lcl_z.data[matchindex_3h]
            lnb_p[tt,3] = dsmup.lnb_p.data[matchindex_3h]
            lnb_t[tt,3] = dsmup.lnb_t.data[matchindex_3h]
            lnb_z[tt,3] = dsmup.lnb_z.data[matchindex_3h]
            lfc_p[tt,3] = dsmup.lfc_p.data[matchindex_3h]
            lfc_t[tt,3] = dsmup.lfc_t.data[matchindex_3h]
            lfc_z[tt,3] = dsmup.lfc_z.data[matchindex_3h]
            shear3mag[tt,3] = dsuvq.shear3mag.data[matchindex_3h]
            shear3dir[tt,3] = dsuvq.shear3dir.data[matchindex_3h]
            shear6mag[tt,3] = dsuvq.shear6mag.data[matchindex_3h]
            shear6dir[tt,3] = dsuvq.shear6dir.data[matchindex_3h]
            u10z[tt,3] = dsuvq.u10z.data[matchindex_3h]
            v10z[tt,3] = dsuvq.v10z.data[matchindex_3h]
            q10z[tt,3] = dsuvq.q10z.data[matchindex_3h]
            rh10z[tt,3] = dsuvq.rh10z.data[matchindex_3h]
            u1500z[tt,3] = dsuvq.u1500z.data[matchindex_3h]
            v1500z[tt,3] = dsuvq.v1500z.data[matchindex_3h]
            q1500z[tt,3] = dsuvq.q1500z.data[matchindex_3h]
            rh1500z[tt,3] = dsuvq.rh1500z.data[matchindex_3h]
            u3000z[tt,3] = dsuvq.u3000z.data[matchindex_3h]
            v3000z[tt,3] = dsuvq.v3000z.data[matchindex_3h]
            q3000z[tt,3] = dsuvq.q3000z.data[matchindex_3h]
            rh3000z[tt,3] = dsuvq.rh3000z.data[matchindex_3h]
            u6000z[tt,3] = dsuvq.u6000z.data[matchindex_3h]
            v6000z[tt,3] = dsuvq.v6000z.data[matchindex_3h]
            q6000z[tt,3] = dsuvq.q6000z.data[matchindex_3h]
            rh6000z[tt,3] = dsuvq.rh6000z.data[matchindex_3h]
            u850p[tt,3] = dsuvq.u850p.data[matchindex_3h]
            v850p[tt,3] = dsuvq.v850p.data[matchindex_3h]
            q850p[tt,3] = dsuvq.q850p.data[matchindex_3h]
            rh850p[tt,3] = dsuvq.rh850p.data[matchindex_3h]
            u700p[tt,3] = dsuvq.u700p.data[matchindex_3h]
            v700p[tt,3] = dsuvq.v700p.data[matchindex_3h]
            q700p[tt,3] = dsuvq.q700p.data[matchindex_3h]
            rh700p[tt,3] = dsuvq.rh700p.data[matchindex_3h]
            u500p[tt,3] = dsuvq.u500p.data[matchindex_3h]
            v500p[tt,3] = dsuvq.v500p.data[matchindex_3h]
            q500p[tt,3] = dsuvq.q500p.data[matchindex_3h]
            rh500p[tt,3] = dsuvq.rh500p.data[matchindex_3h]
            u300p[tt,3] = dsuvq.u300p.data[matchindex_3h]
            v300p[tt,3] = dsuvq.v300p.data[matchindex_3h]

        else:
            print(f'No match sonde time found: {stats_basetime0[tt]}')

        # import pdb; pdb.set_trace()
    

    ##################################
    # Write to netcdf
    print('Writing output netcdf ... ')
    t0_write = time.time()

    # Define variable list
    varlist = {'basetime_cell': ([trackdimname], stats_basetime0), \
                'basetime_sonde': ([trackdimname, relative_time_dimname], sonde_matchtime), \
                'lpl_z': ([trackdimname, relative_time_dimname], height), \
                'cape': ([trackdimname, relative_time_dimname], cape), \
                'cin': ([trackdimname, relative_time_dimname], cin), \
                'lcl_p': ([trackdimname, relative_time_dimname], lcl_p), \
                'lcl_t': ([trackdimname, relative_time_dimname], lcl_t), \
                'lcl_z': ([trackdimname, relative_time_dimname], lcl_z), \
                'lnb_p': ([trackdimname, relative_time_dimname], lnb_p), \
                'lnb_t': ([trackdimname, relative_time_dimname], lnb_t), \
                'lnb_z': ([trackdimname, relative_time_dimname], lnb_z), \
                'lfc_p': ([trackdimname, relative_time_dimname], lfc_p), \
                'lfc_t': ([trackdimname, relative_time_dimname], lfc_t), \
                'lfc_z': ([trackdimname, relative_time_dimname], lfc_z), \
                'shear3mag': ([trackdimname, relative_time_dimname], shear3mag), \
                'shear3dir': ([trackdimname, relative_time_dimname], shear3dir), \
                'shear6mag': ([trackdimname, relative_time_dimname], shear6mag), \
                'shear6dir': ([trackdimname, relative_time_dimname], shear6dir), \
                'u10z': ([trackdimname, relative_time_dimname], u10z), \
                'v10z': ([trackdimname, relative_time_dimname], v10z), \
                'q10z': ([trackdimname, relative_time_dimname], q10z), \
                'rh10z': ([trackdimname, relative_time_dimname], rh10z), \
                'u1500z': ([trackdimname, relative_time_dimname], u1500z), \
                'v1500z': ([trackdimname, relative_time_dimname], v1500z), \
                'q1500z': ([trackdimname, relative_time_dimname], q1500z), \
                'rh1500z': ([trackdimname, relative_time_dimname], rh1500z), \
                'u3000z': ([trackdimname, relative_time_dimname], u3000z), \
                'v3000z': ([trackdimname, relative_time_dimname], v3000z), \
                'q3000z': ([trackdimname, relative_time_dimname], q3000z), \
                'rh3000z': ([trackdimname, relative_time_dimname], rh3000z), \
                'u6000z': ([trackdimname, relative_time_dimname], u6000z), \
                'v6000z': ([trackdimname, relative_time_dimname], v6000z), \
                'q6000z': ([trackdimname, relative_time_dimname], q6000z), \
                'rh6000z': ([trackdimname, relative_time_dimname], rh6000z), \
                'u850p': ([trackdimname, relative_time_dimname], u850p), \
                'v850p': ([trackdimname, relative_time_dimname], v850p), \
                'q850p': ([trackdimname, relative_time_dimname], q850p), \
                'rh850p': ([trackdimname, relative_time_dimname], rh850p), \
                'u700p': ([trackdimname, relative_time_dimname], u700p), \
                'v700p': ([trackdimname, relative_time_dimname], v700p), \
                'q700p': ([trackdimname, relative_time_dimname], q700p), \
                'rh700p': ([trackdimname, relative_time_dimname], rh700p), \
                'u500p': ([trackdimname, relative_time_dimname], u500p), \
                'v500p': ([trackdimname, relative_time_dimname], v500p), \
                'q500p': ([trackdimname, relative_time_dimname], q500p), \
                'rh500p': ([trackdimname, relative_time_dimname], rh500p), \
                'u300p': ([trackdimname, relative_time_dimname], u300p), \
                'v300p': ([trackdimname, relative_time_dimname], v300p), \
              }

    # Define coordinate list
    coordlist = {trackdimname: ([trackdimname], np.arange(0, ntracks)), \
                    relative_time_dimname:([relative_time_dimname], [0,-1,-2,-3])
                }

    # Define global attributes
    gattrlist = {'title':  'Sonde MU parcel statistics matched to tracked cells', \
                 'Institution': 'Pacific Northwest National Laboratoy', \
                 'Contact': 'Zhe Feng, zhe.feng@pnnl.gov', \
                 'Created_on':  time.ctime(time.time()), \
                 'source_trackfile': trackstats_file, \
                 'source_muparcelfile': muparcel_file, \
                 'source_uvqfile': uvq_file, \
                 'startdate': startdate, \
                 'enddate': enddate, \
                }
    # Define xarray dataset
    dsout = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)

    dsout.basetime_cell.attrs['long_name'] = 'Epoch time of each cell in a track'
    dsout.basetime_cell.attrs['units'] = basetime_units

    dsout.basetime_sonde.attrs['long_name'] = 'Epoch time of closest interpsonde'
    dsout.basetime_sonde.attrs['units'] = basetime_units

    dsout[relative_time_dimname].attrs['long_name'] = 'Relative sonde time prior to track initiation'
    dsout[relative_time_dimname].attrs['units'] = 'hour'

    # Copy original variable attributes
    dsout.lpl_z.attrs = dsmup.height.attrs
    dsout.cape.attrs = dsmup.cape.attrs
    dsout.cin.attrs = dsmup.cin.attrs
    dsout.lcl_p.attrs = dsmup.lcl_p.attrs
    dsout.lcl_t.attrs = dsmup.lcl_t.attrs
    dsout.lcl_z.attrs = dsmup.lcl_z.attrs
    dsout.lnb_p.attrs = dsmup.lnb_p.attrs
    dsout.lnb_t.attrs = dsmup.lnb_t.attrs
    dsout.lnb_z.attrs = dsmup.lnb_z.attrs
    dsout.lfc_p.attrs = dsmup.lfc_p.attrs
    dsout.lfc_t.attrs = dsmup.lfc_t.attrs
    dsout.lfc_z.attrs = dsmup.lfc_z.attrs

    dsout.shear3mag.attrs = dsuvq.shear3mag.attrs
    dsout.shear3dir.attrs = dsuvq.shear3dir.attrs
    dsout.shear6mag.attrs = dsuvq.shear6mag.attrs
    dsout.shear6dir.attrs = dsuvq.shear6dir.attrs
    dsout.u10z.attrs = dsuvq.u10z.attrs
    dsout.v10z.attrs = dsuvq.v10z.attrs
    dsout.q10z.attrs = dsuvq.q10z.attrs
    dsout.rh10z.attrs = dsuvq.rh10z.attrs
    dsout.u1500z.attrs = dsuvq.u1500z.attrs
    dsout.v1500z.attrs = dsuvq.v1500z.attrs
    dsout.q1500z.attrs = dsuvq.q1500z.attrs
    dsout.rh1500z.attrs = dsuvq.rh1500z.attrs
    dsout.u3000z.attrs = dsuvq.u3000z.attrs
    dsout.v3000z.attrs = dsuvq.v3000z.attrs
    dsout.q3000z.attrs = dsuvq.q3000z.attrs
    dsout.rh3000z.attrs = dsuvq.rh3000z.attrs
    dsout.u6000z.attrs = dsuvq.u6000z.attrs
    dsout.v6000z.attrs = dsuvq.v6000z.attrs
    dsout.q6000z.attrs = dsuvq.q6000z.attrs
    dsout.rh6000z.attrs = dsuvq.rh6000z.attrs
    dsout.u850p.attrs = dsuvq.u850p.attrs
    dsout.v850p.attrs = dsuvq.v850p.attrs
    dsout.q850p.attrs = dsuvq.q850p.attrs
    dsout.rh850p.attrs = dsuvq.rh850p.attrs
    dsout.u700p.attrs = dsuvq.u700p.attrs
    dsout.v700p.attrs = dsuvq.v700p.attrs
    dsout.q700p.attrs = dsuvq.q700p.attrs
    dsout.rh700p.attrs = dsuvq.rh700p.attrs
    dsout.u500p.attrs = dsuvq.u500p.attrs
    dsout.v500p.attrs = dsuvq.v500p.attrs
    dsout.q500p.attrs = dsuvq.q500p.attrs
    dsout.rh500p.attrs = dsuvq.rh500p.attrs
    dsout.u300p.attrs = dsuvq.u300p.attrs
    dsout.v300p.attrs = dsuvq.v300p.attrs

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encodelist = {var: comp for var in dsout.data_vars}

    # Write netcdf file
    dsout.to_netcdf(path=output_filename, mode='w', format='NETCDF4_CLASSIC', unlimited_dims=trackdimname, encoding=encodelist)
    print(f'Output saved: {output_filename}')

    # import pdb; pdb.set_trace()
