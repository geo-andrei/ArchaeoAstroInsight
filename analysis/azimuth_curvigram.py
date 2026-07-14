# -*- coding: utf-8 -*-
"""
azimuth_curvigram.py  --  thin adapter over the `astrocult` library.

Historically this module hard-coded the azimuth curvigram math (calc_az,
curvigram_az, distribution_az, plot_az, estimate_null_max_threshold,
find_circular_peaks).  That code was a copy of the functions now published in
the `astrocult` package by their original author, so we now delegate to the
library instead of maintaining a fork.

Why we call `astrocult.HelperFunctions.plot_curvigram` directly rather than the
public `astrocult.Azimuth.plot_az_adaptive`:
    astrocult 0.0.2's `plot_az_adaptive` has a guard bug
        `if None in [azimuths, binwidths, latitude]:`
    which raises "truth value of an array is ambiguous" for array inputs.
    `plot_curvigram` (what that wrapper ultimately calls) works fine, so we drive
    it ourselves.  Once the upstream guard is fixed this can switch to
    `Azimuth().plot_az_adaptive(...)`.

The public `plot_az(...)` signature below is kept byte-compatible with what
`curvigram_runner.py` already calls, so nothing else in the plugin changes.
It returns the same 6-tuple: (fig, ax, new_az, normed_density, renorm_density, info).
"""
from __future__ import annotations

import logging
import numpy as np


def _astrocult():
    """Lazily import astrocult; raise a clear, actionable error if it is missing."""
    try:
        from astrocult.Azimuth import Azimuth
        from astrocult import HelperFunctions
    except Exception as e:  # ImportError (astrocult/multipledispatch absent) or other
        raise ImportError(
            "The 'astrocult' package (and its dependency 'multipledispatch') is "
            "required for curvigrams. Install them into QGIS's Python by running "
            "02_install_global_admin.bat as Administrator, then restart QGIS. "
            f"(original import error: {e})"
        )
    return Azimuth(), HelperFunctions


def calc_az(altitude_deg: float, latitude_deg: float, declination_deg: float) -> float:
    """Azimuth (deg, [0,360)) from a declination -- thin pass-through to astrocult.

    Kept for backward compatibility / reference-line use.
    """
    azobj, _ = _astrocult()
    return float(azobj.calc_az(float(altitude_deg), float(latitude_deg), float(declination_deg), True))


def plot_az(
    azimuth,
    latitude: float,
    binwidth: float,
    *,
    title: str = "",
    kernel: str = "epanechnikov",          # accepted for compat; astrocult is Epanechnikov
    iterations: int = 100,
    label: str | None = None,
    xlim: tuple[float, float] = (0, 360),
    show: bool = False,
    savepath: str | None = None,
    ax=None,
    line_color: str = "darkcyan",
    fill_color: str = "cadetblue",
    fill_alpha: float = 0.8,
    three_sigma_level: float = 3.0,
    obliquity_deg: float = 23.56,
    draw_reference_lines: bool = True,
    grid_step: float = 0.5,
    wrap_inputs: bool = True,
    normalize: str = "pdf",
    loc: float = 0.0,
    scale: float = 359.0,
    n_uniform: int = 80_000,
    rng=None,
    draw_thresholds: bool = True,
    null_percentile: float = 99.0,
    null_iterations: int | None = None,
    mark_significant_peaks: bool = True,
    min_peak_separation_deg: float = 5.0,
    peak_prominence: float | None = 1.0,
    xtick_step: float = 50.0,
):
    """Build an azimuth curvigram via the astrocult library.

    `binwidth` is the (single) Epanechnikov-equivalent bandwidth in degrees, just
    as before: astrocult's `plot_curvigram` multiplies it by sqrt(5) internally
    (gaussian->Epanechnikov), so passing `h_eff/sqrt(5)` yields an effective
    support of `h_eff` degrees -- identical to the previous implementation.

    Returns (fig, ax, new_az, normed_density, renorm_density, info), where `info`
    contains: global_threshold, global_threshold_percentile, peaks_significant,
    peaks_all, three_sigma_level, mean_null_std.
    """
    azobj, HF = _astrocult()

    az = np.asarray(azimuth, dtype=float).ravel()
    if az.size == 0:
        raise ValueError("`azimuth` must contain at least one value.")

    # One constant per-sample bandwidth (the adaptive API takes an array).
    binwidths = np.full(az.size, float(binwidth), dtype=float)

    # Reference verticals: solstice azimuths (+/- obliquity) and cardinal E/W,
    # matching the previous behaviour. plot_curvigram also mirrors each at 360-x.
    ref_lines = []
    if draw_reference_lines:
        for eps in (obliquity_deg, -obliquity_deg):
            try:
                ref_lines.append(float(azobj.calc_az(0.0, float(latitude), float(eps), True)))
            except Exception as _exc:
                logging.getLogger(__name__).debug("suppressed non-fatal error: %s", _exc)
        ref_lines += [90.0, 270.0]

    return HF.plot_curvigram(
        az,
        float(latitude),
        binwidths,
        azobj.curvigram_az_adaptive,
        azobj.distribution_az_adaptive,
        ref_lines=ref_lines,
        title=title,
        iterations=int(iterations),
        label=label,
        xlim=xlim,
        show=show,
        savepath=savepath,
        ax=ax,
        line_color=line_color,
        fill_color=fill_color,
        fill_alpha=fill_alpha,
        three_sigma_level=three_sigma_level,
        obliquity_deg=obliquity_deg,
        draw_reference_lines=draw_reference_lines,
        grid_step=grid_step,
        wrap_inputs=wrap_inputs,
        normalize=normalize,
        loc=loc,
        scale=scale,
        n_uniform=int(n_uniform),
        rng=rng,
        draw_thresholds=draw_thresholds,
        null_percentile=null_percentile,
        null_iterations=null_iterations,
        mark_significant_peaks=mark_significant_peaks,
        min_peak_separation_deg=min_peak_separation_deg,
        peak_prominence=peak_prominence,
        xtick_step=xtick_step,
    )


__all__ = ["plot_az", "calc_az"]
