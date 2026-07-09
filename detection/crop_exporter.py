# -*- coding: utf-8 -*-
from qgis.utils import iface
from qgis.core import (
    QgsProject, QgsMapLayer, QgsCoordinateTransform, QgsRectangle, QgsFeatureRequest
)
import numpy as np
import cv2
import os, csv


def export_crops(
    raster_layer,
    vector_layer_name="YOLO_Detections",
    output_dir=r"D:\GIS\Scripts\Python\YOLO\crops_png",
    padding_px=6,
    make_square=False,
    resize_to=None,               # e.g., (224, 224) or None
    progress_cb=None,             # callback(int 0..100)
    vector_layer_obj=None         # pass detections layer directly (optional)
):
    """Export PNG crops for each bbox in the detections layer, memory-safe (tile/block based)."""

    def _progress(p):
        try:
            if progress_cb is not None:
                progress_cb(int(max(0, min(100, p))))
        except Exception:
            pass

    # --- Validate raster ---
    if not raster_layer or raster_layer.type() != QgsMapLayer.RasterLayer:
        iface.messageBar().pushWarning("ArchaeoAstroInsight ", "Please select a valid raster layer (image to crop).")
        return

    provider = raster_layer.dataProvider()
    raster_extent = raster_layer.extent()
    width  = raster_layer.width()
    height = raster_layer.height()
    raster_crs = raster_layer.crs()

    if width == 0 or height == 0 or raster_extent.isEmpty():
        iface.messageBar().pushCritical("ArchaeoAstroInsight ", "Invalid raster extent/dimensions.")
        return

    # Pixel size
    res_x = raster_extent.width() / float(width)
    res_y = raster_extent.height() / float(height)
    if res_x == 0 or res_y == 0:
        iface.messageBar().pushCritical("ArchaeoAstroInsight ", "Zero pixel resolution.")
        return

    _progress(5)

    # --- Find detections vector layer ---
    if vector_layer_obj is not None:
        vlyr = vector_layer_obj
    else:
        v_layers = QgsProject.instance().mapLayersByName(vector_layer_name)
        if not v_layers:
            iface.messageBar().pushWarning("ArchaeoAstroInsight ", f"Vector layer '{vector_layer_name}' not found.")
            return
        vlyr = v_layers[0]
    v_crs = vlyr.crs()

    # --- Transform (vector -> raster) if needed ---
    transform = None
    if v_crs != raster_crs:
        transform = QgsCoordinateTransform(v_crs, raster_crs, QgsProject.instance())

    # --- Prepare output dir + manifest ---
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")
    new_manifest = not os.path.exists(manifest_path)

    feats = list(vlyr.getFeatures(QgsFeatureRequest()))
    total_features = len(feats)
    exported = 0

    # Use up to 3 bands for RGB-like crops; if only 1 band exists, we’ll replicate to 3
    band_count = min(3, provider.bandCount())
    if band_count <= 0:
        iface.messageBar().pushCritical("ArchaeoAstroInsight ", "No bands available in raster.")
        return

    def _geo_rect_for_pixel_window(x1p, y1p, x2p, y2p):
        """
        Convert pixel window (inclusive coords) to QgsRectangle in raster CRS.
        """
        # Note: y is from top in pixel space, raster y decreases downward.
        geo_xmin = raster_extent.xMinimum() + x1p * res_x
        geo_xmax = raster_extent.xMinimum() + (x2p + 1) * res_x
        geo_ymax = raster_extent.yMaximum() - y1p * res_y
        geo_ymin = raster_extent.yMaximum() - (y2p + 1) * res_y
        return QgsRectangle(geo_xmin, geo_ymin, geo_xmax, geo_ymax)

    with open(manifest_path, "a", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        if new_manifest:
            writer.writerow([
                "filename", "fid", "class", "confidence",
                "x1_px", "y1_px", "x2_px", "y2_px",
                "x1_geo", "y1_geo", "x2_geo", "y2_geo",
                "width_px", "height_px"
            ])

        for feat in feats:
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                exported += 1
                _progress(10 + int(90 * exported / max(1, total_features)))
                continue

            fields = vlyr.fields().names()
            cls_val = feat["class"] if "class" in fields else ""
            cls_raw = str(cls_val).strip()
            if cls_raw in {"2.0", "2"}:  # skip class 2
                exported += 1
                _progress(10 + int(90 * exported / max(1, total_features)))
                continue

            # Transform geometry to raster CRS if needed
            g = geom
            if transform is not None:
                try:
                    g = geom.clone()
                    g.transform(transform)
                except Exception:
                    exported += 1
                    _progress(10 + int(90 * exported / max(1, total_features)))
                    continue

            rect = g.boundingBox()
            if rect.isEmpty():
                exported += 1
                _progress(10 + int(90 * exported / max(1, total_features)))
                continue

            # Geo bbox
            x1_geo, y1_geo = rect.xMinimum(), rect.yMinimum()
            x2_geo, y2_geo = rect.xMaximum(), rect.yMaximum()

            # Geo -> pixel (float)
            px1 = (x1_geo - raster_extent.xMinimum()) / res_x
            px2 = (x2_geo - raster_extent.xMinimum()) / res_x
            py1 = (raster_extent.yMaximum() - y1_geo) / res_y
            py2 = (raster_extent.yMaximum() - y2_geo) / res_y

            # sort + pad
            x1p, x2p = sorted([px1, px2])
            y1p, y2p = sorted([py1, py2])

            x1p = int(np.floor(x1p)) - padding_px
            y1p = int(np.floor(y1p)) - padding_px
            x2p = int(np.ceil(x2p))  + padding_px
            y2p = int(np.ceil(y2p))  + padding_px

            # optional square
            if make_square:
                w = x2p - x1p + 1
                h = y2p - y1p + 1
                if w > h:
                    d = w - h
                    y1p -= d // 2
                    y2p = y1p + w - 1
                elif h > w:
                    d = h - w
                    x1p -= d // 2
                    x2p = x1p + h - 1

            # clamp
            x1p = max(0, x1p)
            y1p = max(0, y1p)
            x2p = min(width - 1,  x2p)
            y2p = min(height - 1, y2p)
            if x2p <= x1p or y2p <= y1p:
                exported += 1
                _progress(10 + int(90 * exported / max(1, total_features)))
                continue

            crop_w = int(x2p - x1p + 1)
            crop_h = int(y2p - y1p + 1)

            # Read only this crop window from raster provider (memory-safe)
            crop_extent = _geo_rect_for_pixel_window(x1p, y1p, x2p, y2p)

            chans = []
            ok = True
            for band_idx in range(1, band_count + 1):
                blk = provider.block(band_idx, crop_extent, crop_w, crop_h)
                if blk is None:
                    ok = False
                    break
                arr = np.array(blk.data(), dtype=np.float32).reshape(crop_h, crop_w)
                chans.append(arr)

            if not ok or not chans:
                exported += 1
                _progress(10 + int(90 * exported / max(1, total_features)))
                continue

            crop = np.dstack(chans)

            # Normalize per-crop (keeps appearance consistent; avoids global normalization memory cost)
            crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # If single-band, repeat to 3-channel
            if crop.shape[2] == 1:
                crop = np.repeat(crop, 3, axis=2)

            # Convert RGB->BGR for OpenCV write (only if 3+ channels)
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR) if crop.shape[2] >= 3 else crop

            # optional resize
            if isinstance(resize_to, tuple) and len(resize_to) == 2:
                try:
                    crop_bgr = cv2.resize(crop_bgr, resize_to, interpolation=cv2.INTER_AREA)
                except Exception:
                    pass

            conf_val = feat["confidence"] if "confidence" in fields else None

            # filename
            fname = f"crop_f{feat.id():06d}"
            if cls_val not in [None, ""]:
                fname += f"_cls{str(cls_val)}"
            if conf_val not in [None, ""]:
                try:
                    fname += f"_conf{float(conf_val):.2f}"
                except Exception:
                    fname += f"_conf{str(conf_val)}"
            fname += ".png"

            fpath = os.path.join(output_dir, fname)
            cv2.imwrite(fpath, crop_bgr)

            writer.writerow([
                fname, feat.id(), cls_val, conf_val,
                x1p, y1p, x2p, y2p,
                x1_geo, y1_geo, x2_geo, y2_geo,
                crop_bgr.shape[1], crop_bgr.shape[0]
            ])

            exported += 1
            _progress(10 + int(90 * exported / max(1, total_features)))

    iface.messageBar().pushInfo(
        "ArchaeoAstroInsight ",
        f"PNG export done. Features read: {total_features}, PNGs processed: {exported}\n"
        f"Folder: {output_dir}"
    )
