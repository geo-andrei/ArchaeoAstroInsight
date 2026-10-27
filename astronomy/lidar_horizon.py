"""
LiDAR (.laz / .las) horizon profile for the declination step.

An independent horizon method alongside HeyWhatsThat and SRTM:
    1. read each file's header (point count, extent, CRS) -- no point decoding
    2. choose the DEM cell size from the point density (choose_resolution)
    3. rasterize ground points (class 2) per file with QGIS's PDAL
       ('pdal:exportraster') and mosaic the tiles
    4. dip/azimuth from the observer to EVERY DEM cell (direct: the grid is
       already dense, so no Monte-Carlo triangle sampling as for SRTM)
       -> maximum dip per 0.1 deg azimuth bin = the horizon

The returned dict has the same shape as compute_srtm_horizon(), so the rest of
the declination code handles it the same way. The horizon only contains terrain
inside the LiDAR files; a warning reports how far that coverage reaches.
"""
import logging
import math
import os
import shutil
import struct
import tempfile
import time

import numpy as np
from osgeo import gdal, osr

from .srtm_horizon.calculate_horizon import _dip_azimuth
from .srtm_horizon.aggregate_horizon import aggregate_horizon_from_arrays

_OBSERVER_HEIGHT = 2.0       # metres above ground (same as SRTM)
_MIN_RES_M = 1.0             # never finer: horizon angles gain nothing below 1 m
_MAX_CELLS = 4_000_000       # cell budget for the whole mosaic
_PTS_PER_CELL = 2.0          # target number of points per DEM cell
_MAX_EMPTY_FRACTION = 0.10   # coarsen once if more cells than this stay empty
_CHUNK = 500_000             # cells per vectorised dip batch (bounds memory)

_log = logging.getLogger(__name__)


def _warn(msg):
    print("[LIDAR] " + msg)
    try:
        from qgis.utils import iface
        if iface is not None:
            iface.messageBar().pushWarning("ArchaeoAstroInsight ", msg)
    except Exception as _exc:
        _log.debug("suppressed non-fatal error: %s", _exc)


# ---------------------------------------------------------------------------
# Header / CRS
# ---------------------------------------------------------------------------
def _crs_record(user_id_bytes, rec_id, data, wkt, epsg):
    """Pick up a CRS from a LASF_Projection (E)VLR: OGC WKT (2112) or GeoKeys (34735)."""
    user_id = user_id_bytes.split(b"\0")[0].decode("ascii", "ignore")
    if user_id != "LASF_Projection":
        return wkt, epsg
    if rec_id == 2112 and data:
        wkt = data.split(b"\0")[0].decode("utf-8", "ignore").strip() or wkt
    elif rec_id == 34735 and len(data) >= 8:
        keys = struct.unpack("<%dH" % (len(data) // 2), data[: len(data) // 2 * 2])
        proj = geo = None
        for i in range(min(keys[3], (len(keys) - 4) // 4)):
            kid, loc, _count, val = keys[4 + 4 * i: 8 + 4 * i]
            if loc == 0 and val not in (0, 32767):
                if kid == 3072:      # ProjectedCSTypeGeoKey
                    proj = val
                elif kid == 2048:    # GeographicTypeGeoKey
                    geo = val
        epsg = epsg or proj or geo
    return wkt, epsg


def read_las_header(path):
    """Point count, XY extent and CRS (osr.SpatialReference or None) of a LAS/LAZ
    file. LAZ only compresses the point records, so the header and (E)VLRs are
    read directly."""
    with open(path, "rb") as f:
        hdr = f.read(375)
        if hdr[:4] != b"LASF":
            raise ValueError("Not a LAS/LAZ file: {}".format(path))
        vmaj, vmin = hdr[24], hdr[25]
        header_size = struct.unpack_from("<H", hdr, 94)[0]
        n_vlr = struct.unpack_from("<I", hdr, 100)[0]
        count = struct.unpack_from("<I", hdr, 107)[0]
        maxx, minx, maxy, miny = struct.unpack_from("<4d", hdr, 179)
        evlr_start = n_evlr = 0
        if (vmaj, vmin) >= (1, 4) and len(hdr) >= 255:
            evlr_start, n_evlr = struct.unpack_from("<QI", hdr, 235)
            count = struct.unpack_from("<Q", hdr, 247)[0] or count

        wkt = epsg = None
        f.seek(header_size)
        for _ in range(n_vlr):
            vh = f.read(54)
            if len(vh) < 54:
                break
            rec_id, rec_len = struct.unpack_from("<HH", vh, 18)
            wkt, epsg = _crs_record(vh[2:18], rec_id, f.read(rec_len), wkt, epsg)
        if evlr_start and n_evlr:
            f.seek(evlr_start)
            for _ in range(n_evlr):
                vh = f.read(60)
                if len(vh) < 60:
                    break
                rec_id = struct.unpack_from("<H", vh, 18)[0]
                rec_len = struct.unpack_from("<Q", vh, 20)[0]
                wkt, epsg = _crs_record(vh[2:18], rec_id, f.read(rec_len), wkt, epsg)

    srs = None
    if wkt or epsg:
        try:
            srs = osr.SpatialReference()
            srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            err = srs.ImportFromWkt(wkt) if wkt else srs.ImportFromEPSG(int(epsg))
            if err != 0:
                srs = None
        except Exception as _exc:
            _log.debug("unreadable CRS in %s: %s", path, _exc)
            srs = None
    return {"path": path, "count": int(count), "minx": minx, "maxx": maxx,
            "miny": miny, "maxy": maxy, "srs": srs}


def choose_resolution(infos, unit_m=1.0):
    """DEM cell size in METRES for a set of headers: as fine as the point density
    supports (~2 points per cell), never finer than 1 m, and coarse enough to
    stay within the cell budget. Rounded up to 0.5 m."""
    area = sum(max(0.0, i["maxx"] - i["minx"]) * max(0.0, i["maxy"] - i["miny"])
               for i in infos) * unit_m ** 2
    points = sum(i["count"] for i in infos)
    if area <= 0 or points <= 0:
        raise RuntimeError("The LiDAR headers report no points or an empty extent.")
    res = max(_MIN_RES_M,
              math.sqrt(_PTS_PER_CELL / (points / area)),
              math.sqrt(area / _MAX_CELLS))
    return math.ceil(res * 2.0) / 2.0


def _lidar_files(folder):
    """.laz/.las files in *folder*, skipping QGIS's generated '<name>.copc.laz'
    index when the original '<name>.laz' sits next to it."""
    if not folder or not os.path.isdir(folder):
        raise FileNotFoundError("LiDAR folder not found: {}".format(folder))
    names = [n for n in os.listdir(folder)
             if os.path.splitext(n)[1].lower() in (".laz", ".las")]
    lower = {n.lower() for n in names}
    keep = [n for n in names
            if not (n.lower().endswith(".copc.laz")
                    and n.lower()[: -len(".copc.laz")] + ".laz" in lower)]
    if not keep:
        raise FileNotFoundError("No .laz/.las files found in {}".format(folder))
    return [os.path.join(folder, n) for n in sorted(keep)]


# ---------------------------------------------------------------------------
# Ground DEM
# ---------------------------------------------------------------------------
def _export_ground_dems(paths, res_units, out_dir):
    """Rasterize ground points (class 2) of each file. Returns (tif paths,
    fraction of empty cells across all tiles)."""
    import processing
    tifs, empty, total = [], 0, 0
    for k, path in enumerate(paths):
        out = os.path.join(out_dir, "dem_%03d_%s.tif" % (k, str(res_units).replace(".", "p")))
        processing.run("pdal:exportraster", {
            "INPUT": path,
            "ATTRIBUTE": "Z",
            "RESOLUTION": res_units,
            "TILE_SIZE": 1000,
            "FILTER_EXPRESSION": "Classification = 2",
            "OUTPUT": out,
        })
        if not os.path.isfile(out):
            continue
        ds = gdal.Open(out)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        nodata = band.GetNoDataValue()
        bad = ~np.isfinite(arr) if nodata is None else (arr == nodata) | ~np.isfinite(arr)
        ds = None
        empty += int(bad.sum())
        total += int(arr.size)
        if int(bad.sum()) < arr.size:
            tifs.append(out)
    return tifs, (empty / float(total) if total else 1.0)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def compute_lidar_horizon(latitude, longitude, lidar_folder):
    """
    Horizon profile at (latitude, longitude) from the LiDAR files in
    *lidar_folder*. Returns ``{'data': [{'az', 'alt'}, ...], 'metadata': {...}}``,
    the same structure as compute_srtm_horizon().
    """
    t0 = time.time()
    infos = [read_las_header(p) for p in _lidar_files(lidar_folder)]

    no_crs = [os.path.basename(i["path"]) for i in infos if i["srs"] is None]
    infos = [i for i in infos if i["srs"] is not None]
    if no_crs:
        _warn("LiDAR files without a CRS were skipped: {}".format(", ".join(no_crs)))
    if not infos:
        raise RuntimeError("None of the LiDAR files has a CRS (all skipped).")

    srs = infos[0]["srs"]
    if srs.IsGeographic():
        raise RuntimeError("LiDAR in a geographic CRS (degrees) is not supported; "
                           "use files in a projected CRS.")
    other = [os.path.basename(i["path"]) for i in infos[1:] if not srs.IsSame(i["srs"])]
    if other:
        _warn("LiDAR files in a different CRS than the first file were skipped: {}"
              .format(", ".join(other)))
        infos = [infos[0]] + [i for i in infos[1:] if srs.IsSame(i["srs"])]
    unit_m = srs.GetLinearUnits() or 1.0   # metres per CRS unit (Z assumed same unit)

    res_m = choose_resolution(infos, unit_m)
    tmp = tempfile.mkdtemp(prefix="aai_lidar_")
    try:
        tifs, empty = _export_ground_dems([i["path"] for i in infos], res_m / unit_m, tmp)
        if tifs and empty > _MAX_EMPTY_FRACTION:
            print("[LIDAR] {:.0%} empty cells at {} m, retrying at {} m".format(
                empty, res_m, res_m * 2))
            res_m *= 2
            tifs, empty = _export_ground_dems([i["path"] for i in infos], res_m / unit_m, tmp)
        if not tifs:
            raise RuntimeError("No ground points (class 2) found in the LiDAR files.")

        vrt = gdal.BuildVRT(os.path.join(tmp, "mosaic.vrt"), tifs)
        band = vrt.GetRasterBand(1)
        dem = band.ReadAsArray().astype(np.float64)
        nodata = band.GetNoDataValue()
        gt = vrt.GetGeoTransform()
        band = vrt = None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    valid = np.isfinite(dem) if nodata is None else (dem != nodata) & np.isfinite(dem)

    wgs = osr.SpatialReference()
    wgs.ImportFromEPSG(4326)
    wgs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    to_dem = osr.CoordinateTransformation(wgs, srs)
    to_wgs = osr.CoordinateTransformation(srs, wgs)

    # --- observer must stand on the LiDAR ground surface ---
    ox, oy = to_dem.TransformPoint(float(longitude), float(latitude))[:2]
    col = int(math.floor((ox - gt[0]) / gt[1]))
    row = int(math.floor((oy - gt[3]) / gt[5]))
    if not (0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]) or not valid[row, col]:
        raise RuntimeError(
            "The observer ({:.5f}, {:.5f}) is outside the LiDAR coverage; "
            "the LiDAR horizon cannot be computed there.".format(latitude, longitude))
    ground = float(dem[row, col]) * unit_m
    obs_elev = ground + _OBSERVER_HEIGHT

    # --- every valid cell -> lon/lat/elevation ---
    rows, cols = np.nonzero(valid)
    xs = gt[0] + (cols + 0.5) * gt[1]
    ys = gt[3] + (rows + 0.5) * gt[5]
    zs = dem[rows, cols] * unit_m
    ll = np.asarray(to_wgs.TransformPoints(np.column_stack((xs, ys))), dtype=np.float64)
    lons, lats = ll[:, 0], ll[:, 1]

    az_parts, dip_parts, dist_parts = [], [], []
    for s in range(0, lons.size, _CHUNK):
        az, dip, dist = _dip_azimuth(latitude, longitude, obs_elev,
                                     lats[s:s + _CHUNK], lons[s:s + _CHUNK], zs[s:s + _CHUNK])
        az_parts.append(np.mod(az, 360.0))
        dip_parts.append(dip)
        dist_parts.append(dist)
    az = np.concatenate(az_parts)
    dip = np.concatenate(dip_parts)
    dist = np.concatenate(dist_parts)
    good = np.isfinite(az) & np.isfinite(dip) & (dist > 0)
    az, dip, dist = az[good], dip[good], dist[good]

    # --- coverage: farthest LiDAR terrain in each 1-degree direction ---
    far = np.zeros(360)
    np.maximum.at(far, np.floor(az).astype(int) % 360, dist)
    reached = far[far > 0]
    near_km, far_km = float(reached.min()) / 1000.0, float(reached.max()) / 1000.0
    missing = int((far == 0).sum())

    horizon = aggregate_horizon_from_arrays(az, dip)

    msg = ("LiDAR horizon uses only terrain inside the LiDAR files: {:.2f}-{:.2f} km "
           "from the observer ({} m cells). Terrain beyond is ignored, so the horizon "
           "may be underestimated where distant relief defines it.").format(
               near_km, far_km, res_m)
    if missing:
        msg += (" No LiDAR terrain at all in {} deg of azimuth (observer near the edge "
                "of coverage); the horizon there is interpolated.").format(missing)
    _warn(msg)
    print("[LIDAR] {} file(s), {} cells, horizon in {:.1f}s".format(
        len(infos), int(good.sum()), time.time() - t0))

    return {
        "data": horizon,
        "metadata": {
            "source": "LIDAR",
            "elevation": ground,
            "resolution_m": res_m,
            "coverage_km": (near_km, far_km),
            "files": len(infos),
        },
    }
