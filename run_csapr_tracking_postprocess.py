import subprocess
import textwrap
import os
import pandas as pd

if __name__ == "__main__":
    start_dates = [
        "20181129", 
        "20181204", 
        "20181205", 
        "20181219", 
        "20190122", 
        "20190123", 
        "20190125", 
        "20190129", 
        "20190208", 
    ]
    start_time = 12

    # Config file
    config_file = './config_csapr_lasso.yaml'
    # Processing code name
    code_name = 'match_interpsonde_timeseries_celltracks.py'

    # Loop over each case date
    for ii, idate in enumerate(start_dates):
        print(ii, idate)
        # Start datetime (stard_date + start_time)
        sdate = pd.to_datetime(idate, format='%Y%m%d') + pd.Timedelta(start_time, 'h')
        # End datetime (at 00:00)
        edate = (sdate + pd.Timedelta(1, 'D')).floor(freq='D')
        # Convert to strings
        sdate_str = sdate.strftime('%Y%m%d.%H%M')
        edate_str = edate.strftime('%Y%m%d.%H%M')

        # Run command
        cmd = f"python {code_name} {config_file} {sdate_str} {edate_str}"
        print(cmd)
        subprocess.run(cmd, shell=True)
        # import pdb; pdb.set_trace()