#!/bin/bash
# Doesn't work because the cdf files have more than the "standard" (time,lat,lon) dimensions.

# Input and output directories
input_dir="/global/cfs/cdirs/m1657/enochjo/goamazon/arm/maovisstgridg13minnisX1.c1/"
output_dir="/global/cfs/cdirs/m1657/enochjo/goamazon/arm/maointerpolatedsondeM1.c1.singletime/"
mkdir -p "$output_dir"

# Loop through all NetCDF files in the input directory
for file in "$input_dir"/*.nc; do
    # Extract the base name of the file (without the directory and extension)
    base_name=$(basename "${file::-10}" .nc)
    

    # Use splitsel to split the file by time
    cdo splitsel,1 "$file" "$output_dir/${base_name}_"

    # Optional: Rename files to include timestamps
    for split_file in "$output_dir/${base_name}"*.nc; do
        # Extract timestamp
        timestamp=$(cdo showtimestamp "$split_file" | tr -d '\n')

        
        
        # Format timestamp into YYYYMMDDHHMMSS (optional; adjust format as needed)
        # formatted_time=$(date -d "$timestamp" +"%Y%m%d%H%M%S")
        formatted_time=$(date -d "$timestamp" +"%H%M%S")

        # echo "${output_dir}/${base_name}.${formatted_time}.cdf"
        # read 

        # Rename the file with the timestamp
        # mv "$split_file" "${output_dir}/${base_name}_${formatted_time}.nc"
        mv "$split_file" "${output_dir}/${base_name}.${formatted_time}.nc"
    done
done

echo "Splitting complete. All files are saved in $output_dir."