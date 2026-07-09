# binarizer.py
# -*- coding: utf-8 -*-
import os
from pathlib import Path
import cv2
import numpy as np
from qgis.utils import iface

# --- single-image helpers (unchanged logic) ---
def binarize_image(image_path, brightness_threshold=150):
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    brightness = np.mean(image_rgb, axis=2)
    binary_image = np.zeros_like(brightness, dtype=np.uint8)
    binary_image[brightness < brightness_threshold] = 1
    return binary_image

def binarize_image_by_color_similarity(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    light_pink   = np.array([254, 186, 247])  # highlights
    dark_purple1 = np.array([128,   0, 128])  # very_dark
    dark_purple2 = np.array([182,  31, 213])  # darkest

    pink_distance    = np.sqrt(np.sum((image_rgb - light_pink)  ** 2, axis=2))
    purple1_distance = np.sqrt(np.sum((image_rgb - dark_purple1)** 2, axis=2))
    purple2_distance = np.sqrt(np.sum((image_rgb - dark_purple2)** 2, axis=2))
    purple_distance  = np.minimum(purple1_distance, purple2_distance)

    binary_image = np.zeros_like(pink_distance, dtype=np.uint8)
    binary_image[purple_distance < pink_distance] = 1
    return binary_image

# --- batch driver ---
def run_binarization_pipeline(stage3_folder, progress_cb=None):
    """
    Reads images from `stage3_folder` (e.g. .../stage3_1) and writes two binary sets
    into sibling folders inside the same parent:
      - .../stage31_binary
      - .../stage31_binary2
    """
    def _progress(p):
        try:
            if progress_cb: progress_cb(int(max(0, min(100, p))))
        except Exception:
            pass

    in_path = Path(stage3_folder)
    if not in_path.is_dir():
        iface.messageBar().pushWarning("ArchaeoAstroInsight ", f"Binarizer: input folder not found: {stage3_folder}")
        return

    parent = in_path.parent
    out1 = parent / "stage31_binary"
    out2 = parent / "stage31_binary2"
    out1.mkdir(parents=True, exist_ok=True)
    out2.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [p for p in in_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    total = len(files)
    if total == 0:
        iface.messageBar().pushWarning("ArchaeoAstroInsight ", f"Binarizer: no images in {stage3_folder}")
        return

    _progress(0)
    for i, fp in enumerate(files, 1):
        stem = fp.stem

        b1 = binarize_image(fp, brightness_threshold=150)
        if b1 is not None:
            cv2.imwrite(str(out1 / f"{stem}.png"), b1 * 255)

        b2 = binarize_image_by_color_similarity(fp)
        if b2 is not None:
            cv2.imwrite(str(out2 / f"{stem}.png"), b2 * 255)

        _progress(int(100 * i / total))

    iface.messageBar().pushInfo(
        "ArchaeoAstroInsight ",
        f"Binarization complete.\nOutput: {out1}\nOutput: {out2}"
    )

    # Return the two output folders in case caller wants to chain more
    return str(out1), str(out2)
