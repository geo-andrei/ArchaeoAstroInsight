"""
Pure-Python / NumPy port of ``calculate_horizon_topography_new.cpp``.

Computes the local horizon (azimuth, dip angle, distance) seen from an
observer by interpolating an SRTM terrain point cloud: for every point it
builds triangles from that point's nearest neighbours, Monte-Carlo samples the
triangle surfaces, and computes the azimuth and dip angle (elevation angle
above the astronomical horizontal, corrected for Earth curvature and
atmospheric refraction) from the observer to each sampled point.

This replaces the bundled compiled executable ``horizon_calc.exe`` so the
plugin ships no binaries, as required by the QGIS plugin repository. The
nearest-neighbour search uses ``scipy.spatial.cKDTree`` and the geometry is
fully vectorised with NumPy, recovering the speed the C++/OpenMP version got
from threads.

Original algorithm (C++ / R) by Sherry Towers (smtowers@asu.edu), 2013-2014,
extended by Marc Frincu (marc.frincu@ntu.ac.uk), 2020-2025. Ported to
Python/NumPy for the ArchaeoAstroInsight plugin. The original terms permit
free use and sharing provided the author and copyright information remain
intact.
"""
import numpy as np

_EARTH_A_KM = 6378.1370   # WGS84 semi-major axis (km)
_EARTH_B_KM = 6356.7523   # WGS84 semi-minor axis (km)
_REFRACTION_K = 0.13      # atmospheric refraction coefficient (k in the dip term)


def _dip_azimuth(lat1, lon1, alt1, lat2, lon2, alt2):
    """
    Vectorised port of the C++ ``calculate_dip_angle`` (metres, refraction on).

    The observer ``(lat1, lon1, alt1)`` is scalar (decimal degrees / metres);
    the targets ``(lat2, lon2, alt2)`` are NumPy arrays. Returns
    ``(azimuth_deg, dip_deg, dist_m)`` as arrays.
    """
    pi = np.pi
    scale = 1000.0  # metres per km

    th1 = np.radians(lat1)
    ph1 = np.radians(lon1)
    th2 = np.radians(lat2)
    ph2 = np.radians(lon2)

    # Azimuth from initial bearing (great-circle), in degrees.
    a = np.sin(ph2 - ph1) * np.cos(th2)
    b = np.cos(th1) * np.sin(th2) - np.sin(th1) * np.cos(th2) * np.cos(ph2 - ph1)
    azimuth = np.degrees(np.arctan2(a, b))

    # Geocentric radius at each latitude (ellipsoid).
    ae = _EARTH_A_KM * scale
    be = _EARTH_B_KM * scale
    R1 = np.sqrt((ae**4 * np.cos(th1)**2 + be**4 * np.sin(th1)**2) /
                 (ae**2 * np.cos(th1)**2 + be**2 * np.sin(th1)**2))
    R2 = np.sqrt((ae**4 * np.cos(th2)**2 + be**4 * np.sin(th2)**2) /
                 (ae**2 * np.cos(th2)**2 + be**2 * np.sin(th2)**2))

    # Great-circle distance (haversine) and the refraction correction to alt2.
    dlat = th2 - th1
    dlon = ph2 - ph1
    hav = (np.sin(dlat / 2.0)**2 +
           np.sin(dlon / 2.0)**2 * np.cos(th1) * np.cos(th2))
    c = 2.0 * np.arctan2(np.sqrt(hav), np.sqrt(1.0 - hav))
    dist = R2 * c
    adiff = dist**2 * _REFRACTION_K / (R2 * 2.0)

    # Cartesian coordinates of observer (1) and target (2); gamma is the
    # angle between the two position vectors from the Earth centre.
    x1 = (R1 + alt1) * np.cos(ph1) * np.sin(pi / 2.0 - th1)
    y1 = (R1 + alt1) * np.sin(ph1) * np.sin(pi / 2.0 - th1)
    z1 = (R1 + alt1) * np.cos(pi / 2.0 - th1)
    x2 = (R2 + alt2) * np.cos(ph2) * np.sin(pi / 2.0 - th2)
    y2 = (R2 + alt2) * np.sin(ph2) * np.sin(pi / 2.0 - th2)
    z2 = (R2 + alt2) * np.cos(pi / 2.0 - th2)
    d1 = np.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
    d2 = np.sqrt(x2 * x2 + y2 * y2 + z2 * z2)
    cos_gamma = np.clip((x1 * x2 + y1 * y2 + z1 * z2) / (d1 * d2), -1.0, 1.0)
    gamma = np.abs(np.arccos(cos_gamma))

    # Degenerate samples (a point landing on the observer -> w == 0 or
    # gamma == 0) produce non-finite dips; they are dropped by the caller,
    # mirroring the C++/Windows "1.#INF" tokens the aggregator used to skip.
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.sqrt((R2 + alt2 + adiff)**2 + (R1 + alt1)**2 -
                    2.0 * (R1 + alt1) * (R2 + alt2 + adiff) * np.cos(gamma))
        v = (alt1 + R1) / np.cos(gamma) - R2 - alt2 - adiff
        dip = -np.degrees(np.arcsin(v * np.sin(pi / 2.0 - gamma) / w))

    return azimuth, dip, dist


def _nsample_for_distance(dist_m):
    """
    Number of Monte-Carlo samples per triangle as a function of the
    observer-to-point distance (metres), matching the C++ tiers: nearer terrain
    is sampled more densely.
    """
    ns = np.ones(dist_m.shape, dtype=np.int64)
    ns[dist_m < 50000.0] = 10
    ns[dist_m < 30000.0] = 10
    ns[dist_m < 15000.0] = 25
    ns[dist_m < 10000.0] = 50
    ns[dist_m < 2000.0] = 100
    ns[dist_m < 1000.0] = 250
    return ns


def _triangle_pairs(n_neighbours):
    """Unordered neighbour-slot pairs (u, v) forming the interpolation triangles."""
    return [(u, v)
            for u in range(n_neighbours)
            for v in range(u + 1, n_neighbours)]


def calculate_horizon_topography(latitude, longitude, elevation,
                                 lon, lat, elev,
                                 seed=0, target_azimuth=-1.0,
                                 az_bin_half_width=0.005):
    """
    Compute raw horizon samples from an SRTM terrain point cloud.

    Parameters
    ----------
    latitude, longitude, elevation : float
        Observer position (decimal degrees) and elevation (metres, ground +
        observer height).
    lon, lat, elev : array-like
        The terrain point cloud (decimal degrees / metres), as produced by
        ``convert_srtm_to_xyz``.
    seed : int
        Seed for the random sampler (reproducible horizons).
    target_azimuth : float
        If ``>= 0``, only terrain within ``az_bin_half_width`` degrees of this
        bearing is processed (speed optimisation). ``-1`` = full 360 degrees.
    az_bin_half_width : float
        Half-width (degrees) of the azimuth filter when ``target_azimuth >= 0``.

    Returns
    -------
    tuple(np.ndarray, np.ndarray, np.ndarray)
        ``(azimuth_deg, dip_deg, dist_km)`` over every Monte-Carlo sample,
        with non-finite samples removed. Feed to
        ``aggregate_horizon_from_arrays`` to obtain the horizon profile.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)

    lon = np.asarray(lon, dtype=np.float64).ravel()
    lat = np.asarray(lat, dtype=np.float64).ravel()
    elev = np.asarray(elev, dtype=np.float64).ravel()
    # Clamp sub-sea-level terrain to 0 (C++: `if (num3 < 0.0) num3 = 0.0`).
    elev = np.where(elev < 0.0, 0.0, elev)

    # Optional azimuth filter (only keep terrain near a target bearing).
    if target_azimuth is not None and target_azimuth >= 0.0:
        th_o = np.radians(latitude)
        ph_o = np.radians(longitude)
        th_t = np.radians(lat)
        ph_t = np.radians(lon)
        aa = np.sin(ph_t - ph_o) * np.cos(th_t)
        bb = (np.cos(th_o) * np.sin(th_t) -
              np.sin(th_o) * np.cos(th_t) * np.cos(ph_t - ph_o))
        bearing = np.degrees(np.arctan2(aa, bb))
        bearing = np.where(bearing < 0.0, bearing + 360.0, bearing)
        keep = np.abs(target_azimuth - bearing) <= az_bin_half_width
        lon, lat, elev = lon[keep], lat[keep], elev[keep]

    # Prepend the observer as point 0, exactly like the C++.
    plon = np.concatenate(([float(longitude)], lon))
    plat = np.concatenate(([float(latitude)], lat))
    pelev = np.concatenate(([float(elevation)], elev))
    n_points = plon.size
    if n_points < 3:
        return np.array([]), np.array([]), np.array([])

    # k-nearest neighbours in raw (lon, lat) space. The C++ greedily removes the
    # nearest point repeatedly, which yields exactly the k nearest neighbours.
    xy = np.column_stack((plon, plat))
    tree = cKDTree(xy)
    k_query = min(7, n_points)  # self + up to 6 neighbours
    _, nn_idx = tree.query(xy, k=k_query)
    if nn_idx.ndim == 1:  # k_query == 1 guard
        nn_idx = nn_idx[:, None]
    max_neighbours = nn_idx.shape[1] - 1

    az_out, dip_out, dist_out = [], [], []

    def _sample_group(point_indices, n_neighbours):
        n_neighbours = min(n_neighbours, max_neighbours)
        if point_indices.size == 0 or n_neighbours < 2:
            return
        # Neighbours (skip column 0 = self).
        neigh = nn_idx[point_indices, 1:1 + n_neighbours]

        # Observer -> each point distance sets the per-point sample count.
        _, _, dist_i = _dip_azimuth(latitude, longitude, elevation,
                                    plat[point_indices], plon[point_indices],
                                    pelev[point_indices])
        ns_arr = _nsample_for_distance(dist_i)
        pairs = _triangle_pairs(n_neighbours)

        # Vectorise per distinct sample-count tier.
        for ns in np.unique(ns_arr):
            tier = ns_arr == ns
            if not np.any(tier):
                continue
            pi_sel = point_indices[tier]
            neigh_sel = neigh[tier]
            a_lon = plon[pi_sel]
            a_lat = plat[pi_sel]
            a_elev = pelev[pi_sel]
            for u, v in pairs:
                b_idx = neigh_sel[:, u]
                c_idx = neigh_sel[:, v]
                b_lon, b_lat, b_elev = plon[b_idx], plat[b_idx], pelev[b_idx]
                c_lon, c_lat, c_elev = plon[c_idx], plat[c_idx], pelev[c_idx]
                # Skip degenerate triangles whose two neighbours share a
                # latitude or a longitude (C++: `wlat[j]!=wlat[k] && wlong...`).
                valid = (b_lat != c_lat) & (b_lon != c_lon)
                if not np.any(valid):
                    continue
                al, ab, ae_ = a_lon[valid], a_lat[valid], a_elev[valid]
                bl, bb_, be_ = b_lon[valid], b_lat[valid], b_elev[valid]
                cl, cb, ce = c_lon[valid], c_lat[valid], c_elev[valid]

                # Uniform random points on the triangle (barycentric).
                r1 = rng.random((al.size, int(ns)))
                r2 = rng.random((al.size, int(ns)))
                sr1 = np.sqrt(r1)
                wa = 1.0 - sr1
                wb = sr1 * (1.0 - r2)
                wc = sr1 * r2
                s_lon = (wa * al[:, None] + wb * bl[:, None] +
                         wc * cl[:, None]).ravel()
                s_lat = (wa * ab[:, None] + wb * bb_[:, None] +
                         wc * cb[:, None]).ravel()
                s_elev = (wa * ae_[:, None] + wb * be_[:, None] +
                          wc * ce[:, None]).ravel()

                az, dip, dist = _dip_azimuth(latitude, longitude, elevation,
                                             s_lat, s_lon, s_elev)
                az = np.where(az < 0.0, az + 360.0, az)
                good = np.isfinite(az) & np.isfinite(dip) & np.isfinite(dist)
                az_out.append(az[good])
                dip_out.append(dip[good])
                dist_out.append(dist[good] / 1000.0)

    # Point 0 (the observer) uses 6 neighbours; every other point uses 4.
    _sample_group(np.array([0]), 6)
    _sample_group(np.arange(1, n_points), 4)

    if not az_out:
        return np.array([]), np.array([]), np.array([])
    return (np.concatenate(az_out),
            np.concatenate(dip_out),
            np.concatenate(dist_out))
