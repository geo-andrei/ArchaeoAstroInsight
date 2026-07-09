import os
import math
import time
import io
import csv
import re
import requests
import pandas as pd

from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsProject
from .srtm_horizon.srtm_pipeline import compute_srtm_horizon, srtm_hor2alt

QUERY_CGI   = "http://www.heywhatsthat.com/bin/query.cgi"
PAN_CGI     = "http://www.heywhatsthat.com/iphone/pan.cgi"
HORIZON_CSV = "http://www.heywhatsthat.com/api/horizon.csv"




    
def _get_hwt_id(lat: float, lon: float, *, timeout: int = 30) -> str:
    """Get HeyWhatsThat panorama id/code for a lat/lon."""
    r = requests.get(QUERY_CGI, params={"lat": lat, "lon": lon}, timeout=timeout)
    r.raise_for_status()
    text = (r.text or "").strip()

    code = text.split()[0] if text else ""
    if not code or len(code) < 4:
        raise RuntimeError(f"Unexpected HeyWhatsThat query response: {text[:200]}")
    return code


def _download_horizon_csv(
    hwt_id: str,
    *,
    resolution: str = ".125",
    timeout: int = 30,
    max_wait_s: int = 180
) -> list[dict]:
    """
    Download horizon profile from HeyWhatsThat as a list of {"az": float, "alt": float}.
    Retries while panorama is being generated.
    """
    # Trigger panorama generation (best-effort)
    try:
        requests.get(PAN_CGI, params={"id": hwt_id}, timeout=timeout)
    except Exception:
        pass

    deadline = time.time() + max_wait_s
    last_snippet = None

    def _looks_like_html(s: str) -> bool:
        s2 = s.lstrip().lower()
        return s2.startswith("<!doctype") or s2.startswith("<html") or "<body" in s2[:500]

    def _detect_delimiter(header_line: str) -> str:
        return ";" if header_line.count(";") > header_line.count(",") else ","

    def _norm(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (name or "").strip().lower())

    AZ_KEYS  = {"binbottom", "az", "azi", "azimuth", "bearing", "bin", "bincenter"}
    ALT_KEYS = {"alt", "altitude", "elev", "elevation", "horizonalt", "horalt"}

    while time.time() < deadline:
        r = requests.get(HORIZON_CSV, params={"id": hwt_id, "resolution": resolution}, timeout=timeout)
        r.raise_for_status()
        text = (r.text or "").strip()
        if not text:
            time.sleep(5)
            continue

        last_snippet = text[:400]

        if _looks_like_html(text):
            time.sleep(5)
            continue
        if "wait" in text[:80].lower():
            time.sleep(5)
            continue

        buf = io.StringIO(text)
        first_line = buf.readline()
        if not first_line:
            time.sleep(5)
            continue

        delim = _detect_delimiter(first_line)
        buf.seek(0)

        reader = csv.reader(buf, delimiter=delim)
        rows = list(reader)
        if len(rows) < 2:
            time.sleep(5)
            continue

        header = rows[0]
        nheader = [_norm(h) for h in header]

        az_idx = None
        alt_idx = None
        for i, nh in enumerate(nheader):
            if az_idx is None and nh in AZ_KEYS:
                az_idx = i
            if alt_idx is None and nh in ALT_KEYS:
                alt_idx = i

        data = []
        for row in rows[1:]:
            if not row or len(row) < 2:
                continue
            try:
                if az_idx is not None and alt_idx is not None and az_idx < len(row) and alt_idx < len(row):
                    az = float(row[az_idx])
                    alt = float(row[alt_idx])
                else:
                    # fallback: first two numeric columns
                    nums = []
                    for cell in row:
                        try:
                            nums.append(float(cell))
                        except Exception:
                            pass
                    if len(nums) < 2:
                        continue
                    az, alt = nums[0], nums[1]

                data.append({"az": az, "alt": alt})
            except Exception:
                continue

        if data:
            data.sort(key=lambda d: d["az"])
            return data

        time.sleep(5)

    raise RuntimeError(
        "HeyWhatsThat horizon not ready / failed: couldn't parse horizon.csv. "
        f"Last response snippet:\n{last_snippet}"
    )


def _alt_from_profile(profile: list[dict], azimuth: float) -> float:
    """Linear interpolation on circular 0..360 domain for profile entries {'az','alt'}."""
    import bisect

    if not profile:
        raise RuntimeError("Empty horizon profile")

    a = float(azimuth) % 360.0

    azs = [float(p["az"]) for p in profile]
    alts = [float(p["alt"]) for p in profile]

    if len(azs) < 2:
        return float(alts[0])

    i = bisect.bisect_left(azs, a)

    if i == 0:
        a0, h0 = azs[0], alts[0]
        a1, h1 = azs[1], alts[1]
    elif i >= len(azs):
        a0, h0 = azs[-1], alts[-1]
        a1, h1 = azs[0] + 360.0, alts[0]
        if a < a0:
            a += 360.0
    else:
        a0, h0 = azs[i - 1], alts[i - 1]
        a1, h1 = azs[i], alts[i]

    if a1 == a0:
        return float(h0)

    t = (a - a0) / (a1 - a0)
    return float(h0 + t * (h1 - h0))


def _compute_declination(lat_deg: float, az_deg: float, alt_deg: float) -> float:
    """Astronomical declination formula."""
    lat = math.radians(float(lat_deg))
    az = math.radians(float(az_deg))
    alt = math.radians(float(alt_deg))

    dec = math.asin(
        math.sin(alt) * math.sin(lat)
        + math.cos(alt) * math.cos(lat) * math.cos(az)
    )
    return math.degrees(dec)


# ---------------------------------------------------------------------------
# Horizon cache
# ---------------------------------------------------------------------------
# The horizon profile depends ONLY on the observer location and the method
# (SRTM vs HeyWhatsThat), NOT on the azimuth source. Computing it once and
# reusing it across azimuth sources avoids re-running the (expensive) SRTM C++
# horizon calculator -- or re-querying the HeyWhatsThat service -- for every
# (azimuth_source x method) combination.
_HORIZON_CACHE = {}


def _get_horizon_cached(use_srtm, lat, lon, *, srtm_path=None, plugin_dir=None,
                        resolution: str = ".125", max_wait_s: int = 180):
    """Return the horizon profile for (method, lat, lon), computing it only once."""
    key = (
        "SRTM" if use_srtm else "HWT",
        round(float(lat), 6),
        round(float(lon), 6),
        srtm_path if use_srtm else resolution,
    )
    cached = _HORIZON_CACHE.get(key)
    if cached is not None:
        print("[Declination] Reusing cached {} horizon for ({:.5f}, {:.5f})".format(
            key[0], lat, lon))
        return cached

    if use_srtm:
        value = compute_srtm_horizon(
            latitude=lat, longitude=lon, srtm_folder=srtm_path, plugin_dir=plugin_dir
        )
    else:
        hwt_id = _get_hwt_id(lat, lon)
        value = _download_horizon_csv(hwt_id, resolution=resolution, max_wait_s=max_wait_s)

    _HORIZON_CACHE[key] = value
    return value


def run_declination_pipeline(
    features_results_dir: str,
    detections_layer,
    *,
    use_srtm=False,
    srtm_path=None,
    plugin_dir=None,
    resolution: str = ".125",
    max_wait_s: int = 180,
    azimuth_source: str = "feature",
    resnet_csv: str = None   
) -> str:
    """
    Reads azimuth_runs_*.csv in features_results_dir, uses SUMMARY_mean rows,
    computes horizon altitude via HeyWhatsThat (single cached profile),
    outputs declinations.csv.
    """

    # Build the per-detection azimuth table from the chosen source.
    if azimuth_source == "feature":
        # Helga (Monte-Carlo feature detection): read azimuth_runs, keep SUMMARY_mean.
        if not os.path.isdir(features_results_dir):
            raise RuntimeError("No azimuth_runs CSV found (features_results missing).")
        csv_files = [
            f for f in os.listdir(features_results_dir)
            if f.startswith("azimuth_runs") and f.endswith(".csv")
        ]
        if not csv_files:
            raise RuntimeError("No azimuth_runs CSV found.")
        df = pd.read_csv(os.path.join(features_results_dir, csv_files[0]))
        df_mean = df[df["run_index"] == "SUMMARY_mean"].copy()

    elif azimuth_source == "resnet":
        # ResNet: azimuths come straight from the ResNet predictions CSV.
        # No azimuth_runs / features_results needed for the input.
        if not resnet_csv or not os.path.isfile(resnet_csv):
            raise FileNotFoundError("ResNet CSV path is missing or invalid")
        df_mean = pd.read_csv(resnet_csv).copy()
        if "filename" not in df_mean.columns or "azimuth_deg" not in df_mean.columns:
            raise ValueError("ResNet CSV must contain 'filename' and 'azimuth_deg'")
        df_mean["det_id"] = range(len(df_mean))

    else:
        raise ValueError("Unknown azimuth_source: {}".format(azimuth_source))
        
    # 🔍 DEBUG BLOCK
    print("\n==============================")
    print(f"[DEBUG] AZIMUTH SOURCE: {azimuth_source}")
    print("[DEBUG] df_mean columns:", df_mean.columns.tolist())
    print("[DEBUG] df_mean sample:")
    print(df_mean.head())
    print("==============================\n")

    # Transform detection centroid to WGS84 (observer point)
    layer_crs = detections_layer.crs()
    wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
    transform = QgsCoordinateTransform(layer_crs, wgs84, QgsProject.instance())

    feats = list(detections_layer.getFeatures())
    if not feats:
        raise RuntimeError("Detections layer has no features.")

    # Use first feature centroid as observer (cached horizon)
    centroid = feats[0].geometry().centroid().asPoint()
    centroid_wgs = transform.transform(centroid)
    lat = centroid_wgs.y()
    lon = centroid_wgs.x()

    # ---------------------------------------------------------
    # Fetch horizon profile (SRTM or HeyWhatsThat)
    # ---------------------------------------------------------

    if use_srtm:
        hor = _get_horizon_cached(
            True, lat, lon, srtm_path=srtm_path, plugin_dir=plugin_dir
        )
    else:
        horizon_profile = _get_horizon_cached(
            False, lat, lon, resolution=resolution, max_wait_s=max_wait_s
        )

    # Compute declinations per mean azimuth row
    results = []
    for i, (_, row) in enumerate(df_mean.iterrows()):
        az = float(row["azimuth_deg"])

        # 🔍 DEBUG (only first few rows)
        if i < 5:
            print(f"[DEBUG] Row {i} → azimuth_deg = {az}")

        if use_srtm:
            altitude = srtm_hor2alt(hor, az)
        else:
            altitude = _alt_from_profile(horizon_profile, az)

        decl = _compute_declination(lat, az, altitude)

        results.append({
            "filename": row["filename"],
            "det_id": int(row["det_id"]),
            "azimuth_deg": az,
            "observer_lat": lat,
            "observer_lon": lon,
            "horizon_altitude_deg": altitude,
            "declination_deg": decl,
            "horizon_model": "SRTM" if use_srtm else "HWT"
        })

    suffix = "_SRTM" if use_srtm else "_HWT"

    os.makedirs(features_results_dir, exist_ok=True)
    out_csv = os.path.join(
        features_results_dir,
        f"declinations_{azimuth_source}{suffix}.csv"
    )
    
    
    pd.DataFrame(results).to_csv(out_csv, index=False)
    
    print(f"[DEBUG] Running declination with source = {azimuth_source}")
    return out_csv