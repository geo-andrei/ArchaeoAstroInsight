# azimuth_processor.py
# -*- coding: utf-8 -*-
import logging
import os, glob
import cv2
import numpy as np
from qgis.utils import iface

def run_azimuth_pipeline(input_folder, progress_cb=None):
    """
    Process all images in 'input_folder' and write:
      input_folder/stage1, input_folder/stage2, input_folder/stage3_1

    progress_cb: optional callable(int 0..100)
    """
    def _progress(p):
        try:
            if progress_cb: progress_cb(int(max(0, min(100, p))))
        except Exception as _exc:
            logging.getLogger(__name__).debug("suppressed non-fatal error: %s", _exc)

    if not input_folder or not os.path.isdir(input_folder):
        iface.messageBar().pushWarning("ArchaeoAstroInsight ", "Azimuth: input folder is missing or invalid.")
        return

    stage1 = os.path.join(input_folder, "stage1")
    stage2 = os.path.join(input_folder, "stage2")
    stage3 = os.path.join(input_folder, "stage3_1")
    os.makedirs(stage1, exist_ok=True)
    os.makedirs(stage2, exist_ok=True)
    os.makedirs(stage3, exist_ok=True)

    # OpenCV BGR colors
    colors = {
        "very_dark": (128, 0, 128),
        "darkest": (182, 31, 213),
        "shadows": (217, 118, 24),
        "midtones": (255, 153, 57),
        "highlights": (254, 186, 247),
    }

    # Gather images
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    image_paths = []
    for e in exts:
        image_paths.extend(glob.glob(os.path.join(input_folder, e)))
    image_paths = sorted(image_paths)

    if not image_paths:
        iface.messageBar().pushWarning("ArchaeoAstroInsight ", f"Azimuth: no images found in {input_folder}.")
        return

    total = len(image_paths)
    done = 0
    _progress(0)

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        img = cv2.imread(image_path)
        if img is None:
            done += 1
            _progress(int(100 * done / total))
            continue

        # Step 1: Contrast stretch (grayscale)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        min_b, max_b = gray.min(), gray.max()
        if max_b == min_b:
            contrast = np.zeros_like(gray, dtype=np.uint8)
        else:
            contrast = ((gray - min_b) / (max_b - min_b) * 255).astype(np.uint8)

        # Thresholds
        t1, t2, t3, t4, _t5 = np.percentile(contrast, [15, 30, 50, 70, 90])

        # Fast vectorized mapping to colors
        output = np.zeros_like(img)
        m1 = contrast <  t1
        m2 = (contrast >= t1) & (contrast <  t2)
        m3 = (contrast >= t2) & (contrast <  t3)
        m4 = (contrast >= t3) & (contrast <  t4)
        m5 = (contrast >= t4)

        output[m1] = colors["very_dark"]
        output[m2] = colors["darkest"]
        output[m3] = colors["shadows"]
        output[m4] = colors["midtones"]
        output[m5] = colors["highlights"]

        cv2.imwrite(os.path.join(stage1, filename), output)

        # Step 2: remove blue-dominant pixels
        B, G, R = output[:,:,0], output[:,:,1], output[:,:,2]
        mask_blue = (B > G) & (B > R)
        output2 = output.copy()
        output2[mask_blue] = colors["highlights"]
        cv2.imwrite(os.path.join(stage2, filename), output2)

        # Step 3: remove exact darkest color
        darkest = np.array(colors["darkest"], dtype=np.uint8)
        mask_darkest = np.all(output2 == darkest, axis=2)
        output3 = output2.copy()
        output3[mask_darkest] = colors["highlights"]
        cv2.imwrite(os.path.join(stage3, filename), output3)

        done += 1
        _progress(int(100 * done / total))

    iface.messageBar().pushInfo("ArchaeoAstroInsight ", f"Azimuth pipeline complete. Output: {stage1}, {stage2}, {stage3}")
