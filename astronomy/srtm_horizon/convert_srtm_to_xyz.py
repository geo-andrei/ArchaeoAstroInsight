"""
Python translation of convert_srtm_raster_to_xyz.r

Converts SRTM raster elevation data to a space-separated text file of
longitude, latitude, elevation for use by the horizon topography C++ calculator.

Uses GDAL (always available inside QGIS) to read SRTM tiles in any format
(.hgt, .asc, .tif, etc.).

Original R script by Sherry Towers (smtowers@asu.edu), Nov 2013.
Python port for the A2i QGIS plugin.
"""
import os
import numpy as np


def find_srtm_tiles(srtm_folder, min_lon, max_lon, min_lat, max_lat):
    """
    Scan *srtm_folder* for raster files whose geographic extent overlaps the
    bounding box (min_lon, max_lon, min_lat, max_lat).

    Supports .hgt, .asc, .tif / .tiff, .dem, .dt2 and any other
    GDAL-readable single-band raster.

    Returns:
        list[str]: Absolute paths to the tiles that overlap.
    """
    from osgeo import gdal
    gdal.UseExceptions()

    supported_ext = {'.hgt', '.asc', '.tif', '.tiff', '.dem', '.dt2'}
    tiles = []

    for fname in os.listdir(srtm_folder):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in supported_ext:
            continue
        filepath = os.path.join(srtm_folder, fname)
        try:
            ds = gdal.Open(filepath, gdal.GA_ReadOnly)
            if ds is None:
                continue
            gt = ds.GetGeoTransform()
            cols = ds.RasterXSize
            rows = ds.RasterYSize

            tile_min_lon = gt[0]
            tile_max_lon = gt[0] + cols * gt[1]
            tile_max_lat = gt[3]
            tile_min_lat = gt[3] + rows * gt[5]  # gt[5] is negative

            ds = None  # close

            # Check bounding-box overlap
            if (tile_max_lon > min_lon and tile_min_lon < max_lon and
                    tile_max_lat > min_lat and tile_min_lat < max_lat):
                tiles.append(filepath)
        except Exception:
            continue

    return tiles


def convert_srtm_to_xyz(latitude, longitude, srtm_folder, output_file=None,
                         del_angle=1.5, max_points=25000):
    """
    Read SRTM raster tiles covering the area around *latitude* / *longitude*,
    crop to a bounding box of ±del_angle degrees, and return the terrain point
    cloud as ``longitude, latitude, elevation`` arrays (optionally also writing
    a space-separated text file).

    Parameters
    ----------
    latitude : float
        Observer latitude (decimal degrees, EPSG:4326).
    longitude : float
        Observer longitude (decimal degrees, EPSG:4326).
    srtm_folder : str
        Path to the folder containing SRTM raster tiles.
    output_file : str or None
        If given, also write the point cloud to this space-separated text file
        (``longitude latitude elevation`` per line). ``None`` (default) skips
        the file and just returns the arrays.
    del_angle : float
        Half-size of the bounding box in degrees (default 1.5 ≈ 167 km).
        Must be large enough to capture distant terrain that defines
        the horizon (HeyWhatsThat uses ~200 km).
    max_points : int
        Maximum number of terrain points to return. If the window contains
        more pixels than this, the data is subsampled evenly to keep the
        horizon calculation running in a reasonable time (default 25000).

    Returns
    -------
    tuple(float, np.ndarray, np.ndarray, np.ndarray)
        ``(closest_elev, lon, lat, elev)`` — the elevation (metres) at the grid
        point closest to the observer, and the point-cloud arrays.
    """
    from osgeo import gdal
    gdal.UseExceptions()

    min_lon = longitude - del_angle
    max_lon = longitude + del_angle
    min_lat = latitude - del_angle
    max_lat = latitude + del_angle

    # --- Find which tiles overlap the bounding box ---
    tiles = find_srtm_tiles(srtm_folder, min_lon, max_lon, min_lat, max_lat)
    if not tiles:
        raise FileNotFoundError(
            "No SRTM tiles found in '{}' covering ({:.4f}, {:.4f}) to "
            "({:.4f}, {:.4f})".format(srtm_folder, min_lat, min_lon,
                                      max_lat, max_lon))

    # --- Count total raw pixels across all tiles to compute a global step ---
    tile_infos = []
    total_raw = 0
    for tile_path in tiles:
        ds = gdal.Open(tile_path, gdal.GA_ReadOnly)
        if ds is None:
            continue
        gt = ds.GetGeoTransform()
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        full_cols = ds.RasterXSize
        full_rows = ds.RasterYSize

        col_start = int(max(0, (min_lon - gt[0]) / gt[1]))
        col_end   = int(min(full_cols, (max_lon - gt[0]) / gt[1] + 1))
        row_start = int(max(0, (max_lat - gt[3]) / gt[5]))
        row_end   = int(min(full_rows, (min_lat - gt[3]) / gt[5] + 1))
        win_cols = col_end - col_start
        win_rows = row_end - row_start
        ds = None

        if win_cols <= 0 or win_rows <= 0:
            continue
        total_raw += win_cols * win_rows
        tile_infos.append({
            'path': tile_path, 'gt': gt, 'nodata': nodata,
            'full_cols': full_cols, 'full_rows': full_rows,
            'col_start': col_start, 'col_end': col_end,
            'row_start': row_start, 'row_end': row_end,
            'win_cols': win_cols, 'win_rows': win_rows,
        })

    if not tile_infos:
        raise FileNotFoundError(
            "No readable SRTM data in bounding box.")

    # Global step size so total points across ALL tiles stays under max_points
    step = 1
    if max_points > 0 and total_raw > max_points:
        step = int(np.ceil(np.sqrt(total_raw / max_points)))

    # --- Read tiles with max-pooling (preserves terrain peaks) ---
    all_lon = []
    all_lat = []
    all_elev = []

    for ti in tile_infos:
        gt = ti['gt']
        nodata = ti['nodata']

        ds = gdal.Open(ti['path'], gdal.GA_ReadOnly)
        data = ds.GetRasterBand(1).ReadAsArray(
            ti['col_start'], ti['row_start'], ti['win_cols'], ti['win_rows'])
        ds = None

        if data is None:
            continue

        if step > 1:
            data = data[::step, ::step].astype(np.float64)
            col_centers = np.arange(ti['col_start'], ti['col_end'], step)
            row_centers = np.arange(ti['row_start'], ti['row_end'], step)
            eff_rows, eff_cols = data.shape
            if nodata is not None:
                data[data == float(nodata)] = np.nan
        else:
            eff_rows = ti['win_rows']
            eff_cols = ti['win_cols']
            col_centers = np.arange(ti['col_start'], ti['col_end'])
            row_centers = np.arange(ti['row_start'], ti['row_end'])
            data = data.astype(np.float64)
            if nodata is not None:
                data[data == float(nodata)] = np.nan

        print("[SRTM] Reading tile {} — {}x{} px, step={} → "
              "{}x{}={} pts (from {}x{})".format(
                  os.path.basename(ti['path']),
                  ti['win_cols'], ti['win_rows'], step,
                  eff_cols, eff_rows, eff_cols * eff_rows,
                  ti['full_cols'], ti['full_rows']))

        lon_arr = gt[0] + (col_centers + 0.5) * gt[1]
        lat_arr = gt[3] + (row_centers + 0.5) * gt[5]

        lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)
        lon_flat = lon_grid.ravel()
        lat_flat = lat_grid.ravel()
        elev_flat = data.ravel()

        # Remove NaN (nodata blocks) and fine-crop to exact bbox
        valid = (np.isfinite(elev_flat) &
                 (lon_flat >= min_lon) & (lon_flat <= max_lon) &
                 (lat_flat >= min_lat) & (lat_flat <= max_lat))

        all_lon.append(lon_flat[valid])
        all_lat.append(lat_flat[valid])
        all_elev.append(elev_flat[valid])

    if not all_lon or all(len(a) == 0 for a in all_lon):
        raise ValueError(
            "No elevation data found in bounding box after reading tiles.")

    x = np.concatenate(all_lon)
    y = np.concatenate(all_lat)
    z = np.concatenate(all_elev)

    # Sort by longitude then latitude (same as R's order(x, y))
    sort_idx = np.lexsort((y, x))
    x = x[sort_idx]
    y = y[sort_idx]
    z = z[sort_idx]

    # Elevation at the point closest to the observer
    dist_sq = (x - longitude) ** 2 + (y - latitude) ** 2
    closest_elev = float(z[np.argmin(dist_sq)])

    # Diagnostic: elevation statistics
    print("[SRTM] Terrain stats: min={:.0f}m, max={:.0f}m, mean={:.0f}m, "
          "observer={:.0f}m".format(
              float(np.min(z)), float(np.max(z)), float(np.mean(z)),
              closest_elev))
    above_observer = np.sum(z > closest_elev)
    print("[SRTM] Points above observer: {}/{} ({:.1f}%)".format(
        above_observer, len(z), 100.0 * above_observer / len(z)))

    # Optionally write the point cloud (space-separated, no header, matching
    # R's write.table); the in-memory arrays are returned regardless.
    if output_file is not None:
        print("[SRTM] Writing {} elevation points to XYZ file".format(len(x)))
        out_arr = np.column_stack((x, y, z))
        np.savetxt(output_file, out_arr, fmt='%s', delimiter=' ')

    return closest_elev, x, y, z
