# SRTM-based horizon profile computation module
#
# Pipeline:
#   1. convert_srtm_to_xyz.py  - Extract lon/lat/elev from SRTM raster tiles
#   2. calculate_horizon.py    - Compute azimuth/dip/distance (pure NumPy port
#                                of the original C++/OpenMP calculator)
#   3. aggregate_horizon.py    - Aggregate the samples into a horizon profile
