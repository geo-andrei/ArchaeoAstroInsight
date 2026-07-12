# curvigram_runner.py
from __future__ import annotations
import os
import glob
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
# Use Agg to avoid any GUI needs when running headless
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import your plotting logic from the module that holds the functions you shared
# If you named that file differently, adjust the import accordingly.
from .azimuth_curvigram import plot_az  # <- your calc_az/curvigram_az/distribution_az/plot_az live here


def _find_csv_with_required_columns(root: str, required=("filename", "run_index", "azimuth_deg")) -> str:
    """
    Search `root` for a CSV that contains the required columns; prefer most-recent.
    Raises FileNotFoundError if none found.
    """
    candidates = sorted(
        glob.glob(os.path.join(root, "*.csv")),
        key=os.path.getmtime,
        reverse=True,
    )
    for p in candidates:
        try:
            # Read small sample first for speed; fallback to full on need
            df_head = pd.read_csv(p, nrows=5)
            if all(c in df_head.columns for c in required):
                return p
        except Exception:
            continue
    raise FileNotFoundError(
        f"No CSV with columns {required} found in: {root}"
    )


def _coerce_numeric_run(val: Any) -> Union[int, str]:
    """
    Try to turn a run label into an int (nice sorting); otherwise keep string.
    """
    try:
        # pandas often keeps numeric run_index as str; accept floats like "3.0" too
        iv = int(float(str(val)))
        return iv
    except Exception:
        return str(val)


def _binwidth_from_heff_epa(h_eff_deg: float) -> float:
    """
    plot_az multiplies `binwidth` by sqrt(5) if kernel='epanechnikov'.
    To achieve effective Epanechnikov bandwidth h_eff, pass h_eff / sqrt(5).
    """
    return float(h_eff_deg) / np.sqrt(5.0)


def run_curvigram_pipeline(
    features_results_dir: str,
    *,
    latitude: float,
    h_eff_deg: float = 3.0,
    iterations: int = 100,
    save_dir: Optional[str] = None,
    csv_path: Optional[str] = None,
    progress_cb: Optional[callable] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Build curvigrams for *each* run_index present in the CSV found in `features_results_dir`.

    Parameters
    ----------
    features_results_dir : str
        Directory containing the CSV with columns: filename, run_index, azimuth_deg.
    latitude : float
        Used only for reference lines on the plot (does not change KDE shape).
    h_eff_deg : float
        Effective Epanechnikov bandwidth (smoothing) in degrees. Larger => smoother.
    iterations : int
        Null-model iterations passed to plot_az() -> distribution_az(); balance speed/quality.
    save_dir : str|None
        Destination directory for PNG outputs. Default: `<features_results_dir>/curvigrams`.
    csv_path : str|None
        If provided, use this CSV path directly; otherwise auto-detect in `features_results_dir`.
    progress_cb : callable|None
        Optional progress callback: receives integers 0..100.
    rng : numpy.random.Generator|None
        For reproducibility; forwarded to the underlying functions when applicable.

    Returns
    -------
    dict
        {
          "csv_path": "...",
          "save_dir": "...",
          "runs": {
            <run_label>: {
              "n_samples": int,
              "figure_path": str,
              "threshold_global": float | None,
              "threshold_percentile": float | None,
              "peaks": [ {angle, height, prominence, index}, ... ],
            },
            ...
          },
          "skipped_runs": [<run_label>, ...]
        }
    """
    if progress_cb:
        progress_cb(0)

    if not os.path.isdir(features_results_dir):
        raise NotADirectoryError(f"features_results_dir not found: {features_results_dir}")

    if csv_path is None:
        csv_path = _find_csv_with_required_columns(features_results_dir)

    df = pd.read_csv(csv_path)
    # Filter out SUMMARY rows; keep only numeric run_index rows
    run_str = df["run_index"].astype(str)
    mask_numeric = run_str.str.fullmatch(r"\d+(\.0+)?")
    df_runs = df[mask_numeric].copy()
    if df_runs.empty:
        raise ValueError("No numeric run_index rows found (1,2,3,...) in CSV.")

    # Coerce types
    df_runs["run_index_norm"] = df_runs["run_index"].apply(_coerce_numeric_run)
    df_runs["azimuth_deg"] = pd.to_numeric(df_runs["azimuth_deg"], errors="coerce")
    df_runs = df_runs.dropna(subset=["azimuth_deg"])

    # Decide output folder
    out_dir = save_dir or os.path.join(features_results_dir, "curvigrams")
    os.makedirs(out_dir, exist_ok=True)

    # One figure per run
    run_labels = sorted(df_runs["run_index_norm"].unique(), key=lambda v: (isinstance(v, str), v))
    total = len(run_labels)
    results: Dict[Union[int, str], Any] = {}
    skipped: List[Union[int, str]] = []

    # Prepare binwidth for plot_az (which internally multiplies by sqrt(5))
    binwidth_input = _binwidth_from_heff_epa(h_eff_deg)

    for idx, run_lab in enumerate(run_labels, start=1):
        if progress_cb:
            # spread 5..95 across runs
            progress_cb(5 + int(90 * (idx - 1) / max(1, total)))

        sel = df_runs["run_index_norm"] == run_lab
        az = df_runs.loc[sel, "azimuth_deg"].astype(float).to_numpy()

        if az.size < 3:
            skipped.append(run_lab)
            continue

        title = f"Azimuth Curvigram: Run {run_lab} (all files)"
        fig, ax, new_az, normed, renorm, info = plot_az(
            az,
            latitude=latitude,
            binwidth=binwidth_input,
            title=title,
            kernel="epanechnikov",
            iterations=int(iterations),
            draw_thresholds=True,
            mark_significant_peaks=True,
            min_peak_separation_deg=5.0,
            peak_prominence=1.0,
            xtick_step=50.0,
            show=False,   # headless-safe
        )

        # Save PNG
        safe_run = str(run_lab).replace(os.sep, "_")
        fig_path = os.path.join(out_dir, f"curvigram_helga_run{safe_run}.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        results[run_lab] = {
            "n_samples": int(az.size),
            "figure_path": fig_path,
            "threshold_global": info.get("global_threshold"),
            "threshold_percentile": info.get("global_threshold_percentile"),
            "peaks": info.get("peaks_significant", []),
        }

    if progress_cb:
        progress_cb(100)

    return {
        "csv_path": csv_path,
        "save_dir": out_dir,
        "runs": results,
        "skipped_runs": skipped,
    }
    

def run_curvigram_from_resnet_csv_single(
    *,
    resnet_results_dir: Optional[str] = None,
    csv_path: Optional[str] = None,
    latitude: float,
    h_eff_deg: float = 3.0,
    iterations: int = 100,
    save_dir: Optional[str] = None,
    progress_cb: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Build ONE curvigram from a ResNet predictions CSV with columns:
      filename, prediction, azimuth_deg

    Uses ALL rows' azimuth_deg (no grouping).
    """
    def _progress(p):
        try:
            if progress_cb:
                progress_cb(int(max(0, min(100, p))))
        except Exception:
            pass

    _progress(0)

    # Find CSV
    if csv_path is None:
        if not resnet_results_dir:
            raise ValueError("Provide either csv_path or resnet_results_dir.")
        csv_path = _find_csv_with_required_columns(
            resnet_results_dir,
            required=("filename", "prediction", "azimuth_deg"),
        )

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "azimuth_deg" not in df.columns:
        raise ValueError("CSV must contain column 'azimuth_deg'.")

    # Clean azimuths
    az = pd.to_numeric(df["azimuth_deg"], errors="coerce").dropna().astype(float).to_numpy()
    if az.size < 3:
        raise ValueError("Not enough azimuth values (need at least 3).")

    # Output path
    root_dir = os.path.dirname(os.path.abspath(csv_path))
    out_dir = save_dir or os.path.join(root_dir, "curvigrams_resnet")
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "curvigram_resnet.png")

    # Effective Epanechnikov bandwidth: plot_az multiplies by sqrt(5) internally
    binwidth_input = float(h_eff_deg) / np.sqrt(5.0)

    _progress(25)

    # Build the curvigram
    title = "Azimuth Curvigram: ResNet (all predictions)"
    fig, ax, new_az, normed, renorm, info = plot_az(
        az,
        latitude=latitude,
        binwidth=binwidth_input,
        title=title,
        kernel="epanechnikov",
        iterations=int(iterations),
        draw_thresholds=True,
        mark_significant_peaks=True,
        min_peak_separation_deg=5.0,
        peak_prominence=1.0,
        xtick_step=50.0,
        show=False,
    )

    _progress(85)

    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    _progress(100)

    return {
        "csv_path": csv_path,
        "save_dir": out_dir,
        "n_samples": int(az.size),
        "figure_path": fig_path,
        "threshold_global": info.get("global_threshold"),
        "threshold_percentile": info.get("global_threshold_percentile"),
        "peaks": info.get("peaks_significant", []),
    }


def run_declination_curvigrams(
    results_csv_path: str,
    *,
    save_dir: str,
    latitude: Optional[float] = None,
    h_eff_deg: float = 3.0,
    iterations: int = 20,
    do_hwt: bool = True,
    do_srtm: bool = True,
    progress_cb: Optional[callable] = None,
) -> Dict[str, str]:
    """
    Build declination curvigrams from the centralized Results/results.csv.

    Renders one PNG per checked horizon method, reading the ResNet declination columns:
        do_hwt  -> 'Declination Resnet Model HWT (deg)'  -> curvigram_declination_hwt.png
        do_srtm -> 'Declination Resnet Model SRTM (deg)' -> curvigram_declination_srtm.png

    Returns {method: figure_path, ...} for the curvigrams actually produced.
    """
    def _progress(p):
        try:
            if progress_cb:
                progress_cb(int(max(0, min(100, p))))
        except Exception:
            pass

    if not os.path.isfile(results_csv_path):
        raise FileNotFoundError("results.csv not found: {}".format(results_csv_path))

    df = pd.read_csv(results_csv_path)

    # Observer latitude is required by astrocult's API but unused for the declination
    # curve itself; take it from results.csv when available.
    if latitude is None:
        if "observer_lat" in df.columns and df["observer_lat"].notna().any():
            latitude = float(df["observer_lat"].dropna().iloc[0])
        else:
            latitude = 45.0

    os.makedirs(save_dir, exist_ok=True)

    targets = []
    if do_hwt:
        targets.append(("HWT", "Declination Resnet Model HWT (deg)", "curvigram_declination_hwt.png"))
    if do_srtm:
        targets.append(("SRTM", "Declination Resnet Model SRTM (deg)", "curvigram_declination_srtm.png"))

    out: Dict[str, str] = {}
    _progress(0)
    for idx, (method, col, fname) in enumerate(targets, start=1):
        if col not in df.columns:
            continue
        decl = pd.to_numeric(df[col], errors="coerce").dropna().astype(float).to_numpy()
        if decl.size < 3:
            continue
        fig_path = os.path.join(save_dir, fname)
        _plot_declination_curvigram(
            decl,
            latitude=latitude,
            h_eff_deg=h_eff_deg,
            iterations=iterations,
            save_path=fig_path,
            title="Declination Curvigram: ResNet ({})".format(method),
        )
        out[method] = fig_path
        _progress(int(100 * idx / max(1, len(targets))))

    _progress(100)
    return out


def _plot_declination_curvigram(declinations, *, latitude, h_eff_deg, iterations,
                                save_path, title):
    """Render ONE declination curvigram via astrocult.Declination.plot_decl_adaptive."""
    try:
        from astrocult.Declination import Declination
    except Exception as e:
        raise ImportError(
            "The 'astrocult' package (and its dependency 'skyfield') is required for "
            "declination curvigrams. Install them by running 02_install_global_admin.bat "
            "as Administrator, then restart QGIS. (original import error: {})".format(e)
        )

    decl = np.asarray(declinations, dtype=float).ravel()
    # astrocult's plot_curvigram multiplies binwidth by sqrt(5) internally, so pass
    # h_eff/sqrt(5) to get an effective support of h_eff degrees (same as the azimuth side).
    binwidths = np.full(decl.size, float(h_eff_deg) / np.sqrt(5.0))

    fig, ax, new_dec, normed, renorm, info = Declination().plot_decl_adaptive(
        decl, float(latitude), binwidths,
        title=title, iterations=int(iterations), show=False, xtick_step=30.0,
    )
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path

