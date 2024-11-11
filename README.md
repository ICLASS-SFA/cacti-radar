# **CACTI Radar Cell Tracking Database & Analysis**

---
This repository contains codes and Jupyter Notebooks for the CACTI radar convective cell tracking database creation and analysis.

The cell tracking database and relationships with their environments are documented in [Feng et al. (2022), MWR](https://doi.org/10.1175/MWR-D-21-0237.1).

The convective cell tracking is produced with [PyFLEXTRKR](https://github.com/FlexTRKR/PyFLEXTRKR), documented in [Feng et al. (2023) GMD](https://doi.org/10.5194/gmd-16-2753-2023).

* The /goes16 directory contains codes to regrid GOES16 data onto the radar Cartesian grid.
* The /src directory contains Python scripts for post processing additional convective cell statistics and sounding environments to match the cell track statistics.
* The /notebooks directory contains analyses and plotting for tracked cell statistics.



## Post processing to create cell tracking database
---

The `${}` are command line inputs, examples:

> ${config}: yaml config file specifying input data locations
> 
> Example: /src/config_csapr.yaml

--

* **Regrid GOES16 data to match radar Cartesian grid:**

`python regrid_goes16_to_csapr.py ${config}`

* **Compute radar 3D profile statistics to cell tracks:**

`python calc_3d_radarstats_to_celltracks.py ${config}`

* **Compute GOES16 cloud retrieval statistics to cell tracks:**

`python calc_sat_stats_to_celltracks.py ${config}`

* **Match sounding convective parameters to cell tracks:**

`python match_interpsonde_timeseries_celltracks.py ${config}`

* **Combine all post processed statistics to cell database:**

`python combine_stats_files.py ${config}`


# **References**

---

Feng, Z., Hardin, J., Barnes, H. C., Li, J., Leung, L. R., Varble, A., & Zhang, Z. (2023). PyFLEXTRKR: a flexible feature tracking Python software for convective cloud analysis. *Geosci. Model Dev.*, 16(10), 2753-2776. [https://doi.org/10.5194/gmd-16-2753-2023](https://doi.org/10.5194/gmd-16-2753-2023)

Feng, Z., Varble, A., Hardin, J., Marquis, J., Hunzinger, A., Zhang, Z., & Thieman, M. (2022). Deep Convection Initiation, Growth, and Environments in the Complex Terrain of Central Argentina during CACTI. *Monthly Weather Review*, 150(5), 1135-1155. [https://doi.org/10.1175/MWR-D-21-0237.1](https://doi.org/10.1175/MWR-D-21-0237.1)