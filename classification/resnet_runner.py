# -*- coding: utf-8 -*-
"""
resnet_runner.py: batch inference for grayscale PNG crops using a Keras .keras model.

Features
- CPU-only, quiet TensorFlow (good citizen inside QGIS).
- Auto-detects model input layout (NCHW vs NHWC) and reshapes input accordingly.
- Streams files in batches to keep RAM usage modest.
- Writes CSV with columns: filename, prediction.
- Optional progress callback: progress_cb(int 0..100).

Requirements (already in your environment):
  tensorflow==2.18.1, numpy==1.26.4, opencv-python-headless, pandas

Usage (from plugin):
  from .resnet_runner import run_resnet_inference
  csv_path = run_resnet_inference(
      folders=[<path_to_pngs>],
      keras_model_path=<path_to_model.keras>,
      image_size=100,
      num_classes=72,
      batch_size=128,
      output_dir=<results_dir>,
      progress_cb=lambda p: ...
  )
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import cv2

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras

# Optional QGIS UI message bar (safe import)
try:
    from qgis.utils import iface  # type: ignore
except Exception:
    iface = None  # not running inside QGIS

# --- Keep TF quiet & CPU-only in QGIS ---
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass
# avoid oversubscribing threads inside QGIS
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)


def _list_pngs(folder_list: Iterable[str | Path]) -> List[Tuple[Path, str]]:
    """Return [(folder_path, filename), ...] for all PNGs in the given folders."""
    if isinstance(folder_list, (str, Path)):
        folder_list = [folder_list]
    out: List[Tuple[Path, str]] = []
    for d in folder_list:
        p = Path(d)
        if not p.is_dir():
            continue
        for f in os.listdir(p):
            fp = p / f
            if fp.is_file() and f.lower().endswith(".png"):
                out.append((p, f))
    return out


def _load_image_gray(folder: Path, fname: str, out_size: int) -> np.ndarray:
    """Read PNG as grayscale, resize to out_size, return float32 array HxW in range ~0..255."""
    img = cv2.imread(str(folder / fname), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(folder / fname))
    if img.shape[:2] != (out_size, out_size):
        img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def _infer_layout(model: keras.Model) -> str:
    """Infer 'NCHW' or 'NHWC' from the first input tensor shape."""
    ish = model.inputs[0].shape  # e.g., (None, 1, 100, 100) or (None, 100, 100, 1)
    dims = [None if d is None else int(d) for d in ish]
    if len(dims) == 4 and dims[1] == 1:  # channels in index 1 => NCHW
        return "NCHW"
    return "NHWC"


def _batched(seq, batch_size: int):
    it = iter(seq)
    while True:
        chunk = []
        try:
            for _ in range(batch_size):
                chunk.append(next(it))
        except StopIteration:
            pass
        if not chunk:
            return
        yield chunk


def run_resnet_inference(
    folders: Iterable[str | Path],
    keras_model_path: str | Path,
    *,
    image_size: int = 100,
    num_classes: Optional[int] = None,   # used only in output filename tag
    batch_size: int = 128,
    output_dir: str | Path | None = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> Path:
    """
    Run classification over all PNG files found in 'folders' and write a CSV.

    Parameters
    ----------
    folders : Iterable[str|Path]
        One or more directories containing PNG images.
    keras_model_path : str|Path
        Path to a Keras v3 '.keras' model file (full model).
    image_size : int
        Target H=W resize in pixels (must match training, e.g., 100).
    num_classes : int|None
        For naming the CSV; does not affect inference.
    batch_size : int
        Prediction batch size.
    output_dir : str|Path|None
        Directory where CSV will be written. Defaults to CWD/test_results.
    progress_cb : callable(int)|None
        Receives 0..100 progress updates.

    Returns
    -------
    Path
        Path to the CSV file with 'filename,prediction'.
    """
    def _progress(p: int):
        try:
            if progress_cb:
                progress_cb(max(0, min(100, int(p))))
        except Exception:
            pass

    # Collect files
    pairs = _list_pngs(folders)
    if not pairs:
        raise RuntimeError("No PNG files found in the provided folders.")

    # Load model
    model_path = Path(keras_model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f".keras model not found: {model_path}")
    model = keras.models.load_model(str(model_path), compile=False)
    layout = _infer_layout(model)

    # Prepare output
    if output_dir is None:
        output_dir = Path.cwd() / "test_results"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_tag = model_path.stem
    ncls_tag = f"{num_classes}" if num_classes is not None else "nc"
    out_csv = out_dir / f"classes_{ncls_tag}_prediction_{model_tag}.csv"

    filenames: List[str] = []
    preds_all: List[int] = []

    total = len(pairs)
    done = 0
    _progress(0)

    # Stream in batches
    for chunk in _batched(pairs, max(1, batch_size)):
        # Load batch as HxW float32
        X2D = np.stack([_load_image_gray(d, f, image_size) for d, f in chunk], axis=0)  # (B,H,W)

        # Add channel and order per layout
        if layout == "NCHW":
            X = X2D[:, np.newaxis, :, :]        # (B,1,H,W)
        else:
            X = X2D[:, :, :, np.newaxis]        # (B,H,W,1)

        # Predict
        y = model.predict(X, batch_size=min(len(chunk), batch_size), verbose=0)
        if y.ndim == 1:                          # binary logits/probs
            labels = (y > 0.5).astype(int)
        else:                                     # multi-class
            labels = np.argmax(y, axis=1)

        preds_all.extend([int(v) for v in labels])
        filenames.extend([f for _, f in chunk])

        done += len(chunk)
        _progress(int(100 * done / total))

     # Write CSV (add azimuth_deg = prediction * 5)
    azimuths = [int(p) * 5 for p in preds_all]
    df = pd.DataFrame({
        "filename": filenames,
        "prediction": preds_all,
        "azimuth_deg": azimuths,
    })
    df.to_csv(out_csv, index=False)

    # NICE: notify in QGIS message bar if available
    try:
        if iface is not None:
            iface.messageBar().pushInfo("ArchaeoAstroInsight ", f"ResNet predictions saved: {out_csv}")
    except Exception:
        pass

    _progress(100)
    return out_csv


__all__ = ["run_resnet_inference"]
