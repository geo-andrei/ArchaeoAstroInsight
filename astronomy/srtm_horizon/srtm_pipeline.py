"""
High-level SRTM horizon pipeline.

Orchestrates the three stages:
    1. convert_srtm_to_xyz  – extract lon/lat/elev from SRTM raster tiles
    2. calculate_horizon    – compute azimuth / dip-angle / distance (NumPy)
    3. aggregate_horizon    – bin and aggregate into a horizon profile

The public function ``compute_srtm_horizon()`` returns a dict in the same
format as ``script.download_hwt()``, so the rest of the plugin (declination,
star matching) works unchanged regardless of the data source.

Stage 2 is a pure-Python/NumPy port of the original C++/OpenMP horizon
calculator, so the plugin ships no compiled binaries.
"""
import time

from .convert_srtm_to_xyz import convert_srtm_to_xyz
from .calculate_horizon import calculate_horizon_topography
from .aggregate_horizon import aggregate_horizon_from_arrays, interpolate_altitude


# Observer height above ground (metres) added to SRTM ground elevation
_OBSERVER_HEIGHT = 2.0


def compute_srtm_horizon(latitude, longitude, srtm_folder, plugin_dir=None,
                          target_azimuth=-1, on_waiting=None):
    """
    Run the full SRTM-based horizon pipeline for a given observer location.

    Parameters
    ----------
    latitude, longitude : float
        Observer position in decimal degrees (EPSG:4326).
    srtm_folder : str
        Path to the folder containing SRTM raster tiles.
    plugin_dir : str, optional
        Unused; kept for backward compatibility with the previous
        C++-executable signature.
    target_azimuth : float
        If >= 0, only compute terrain along this azimuth (speed optimisation).
        Use -1 (default) for a full 360-degree profile.
    on_waiting : callable or None
        Optional ``callback(message)`` for progress feedback.

    Returns
    -------
    dict
        ``{'data': [{'az': float, 'alt': float}, ...],
           'metadata': {'source': 'SRTM', 'elevation': float}}``
        Compatible with the HeyWhatsThat ``hor`` dict consumed by the rest
        of the plugin.
    """
    pipeline_start = time.time()
    print("[SRTM] ===== Starting SRTM horizon pipeline =====")
    print("[SRTM] Observer: ({:.5f}, {:.5f})".format(latitude, longitude))
    print("[SRTM] SRTM folder: {}".format(srtm_folder))

    # --- Stage 1: SRTM raster → terrain point cloud ---
    if on_waiting:
        on_waiting("[SRTM] Stage 1/3: Extracting elevation data from SRTM tiles...")
    t1 = time.time()
    print("[SRTM] Stage 1/3: Extracting elevation from SRTM tiles...")

    try:
        observer_ground_elev, lon, lat, elev = convert_srtm_to_xyz(
            latitude, longitude, srtm_folder)
    except Exception as e:
        raise RuntimeError(
            "SRTM tile extraction failed: {}".format(e)) from e

    observer_elev = observer_ground_elev + _OBSERVER_HEIGHT
    t1_elapsed = time.time() - t1
    print("[SRTM] Stage 1 done in {:.1f}s — observer elevation: {:.1f}m "
          "(+{:.0f}m height = {:.1f}m), {} terrain points".format(
              t1_elapsed, observer_ground_elev, _OBSERVER_HEIGHT,
              observer_elev, len(lon)))

    # --- Stage 2: Compute horizon topography (NumPy) ---
    if on_waiting:
        on_waiting("[SRTM] Stage 2/3: Computing horizon topography...")
    t2 = time.time()
    print("[SRTM] Stage 2/3: Computing horizon topography (NumPy)...")

    try:
        azimuths, dips, _dists = calculate_horizon_topography(
            latitude, longitude, observer_elev, lon, lat, elev,
            seed=0, target_azimuth=target_azimuth)
    except Exception as e:
        raise RuntimeError(
            "Horizon topography computation failed: {}".format(e)) from e

    t2_elapsed = time.time() - t2
    print("[SRTM] Stage 2 done in {:.1f}s — {} horizon samples".format(
        t2_elapsed, len(azimuths)))

    # --- Stage 3: Aggregate samples into a horizon profile ---
    if on_waiting:
        on_waiting("[SRTM] Stage 3/3: Aggregating horizon profile...")
    t3 = time.time()
    print("[SRTM] Stage 3/3: Aggregating horizon profile...")

    try:
        horizon = aggregate_horizon_from_arrays(azimuths, dips)
    except Exception as e:
        raise RuntimeError(
            "Horizon aggregation failed: {}".format(e)) from e

    t3_elapsed = time.time() - t3
    total_elapsed = time.time() - pipeline_start
    print("[SRTM] Stage 3 done in {:.1f}s — {} azimuth bins".format(
        t3_elapsed, len(horizon)))
    print("[SRTM] ===== SRTM pipeline complete in {:.1f}s =====".format(
        total_elapsed))

    return {
        'data': horizon,
        'metadata': {
            'source': 'SRTM',
            'elevation': observer_ground_elev,
        },
    }


def srtm_hor2alt(hor, azimuth):
    """
    Interpolate altitude at *azimuth* from an SRTM horizon profile.

    Drop-in replacement for ``script.hor2alt()``.

    Parameters
    ----------
    hor : dict
        Horizon dict returned by ``compute_srtm_horizon()``.
    azimuth : float
        Azimuth in degrees (0-360).

    Returns
    -------
    float
        Altitude (dip angle) in degrees, rounded to 2 decimals.
    """
    return interpolate_altitude(hor['data'], azimuth)
