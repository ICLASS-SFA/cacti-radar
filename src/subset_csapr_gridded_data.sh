#!/bin/bash
module load nco

# Input file directory
indir='/gpfs/wolf/atm131/proj-shared/zfeng/cacti/csapr/taranis_corcsapr2cfrppiqcM1_gridded.c1/'
basename='taranis_corcsapr2cfrppiqcM1.c1.'
search_string=${indir}${basename}'20190125*nc'
# Output directory
outdir='/gpfs/wolf/atm131/proj-shared/zfeng/pyflextrkr_paper/radar/input/'

# Search input files
files=$(ls -1 ${search_string})

# Loop over each file
for ifile in ${files}; do
    # Strip file path to get base filename
    fname="$(basename -- $ifile)"
    # Make output filename
    out_fname=${outdir}${fname}
    # Run subset command
    ncks -O -v time,z,y,x,alt,point_longitude,point_latitude,taranis_attenuation_corrected_reflectivity ${ifile} ${out_fname}
    # Print output filename
    echo ${out_fname}
done
