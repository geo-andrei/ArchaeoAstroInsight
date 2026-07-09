# -*- coding: utf-8 -*-
#BACKUP
from qgis.utils import iface
from qgis.core import (
    QgsMapLayer,
    QgsProject,
    QgsCoordinateTransform,
    QgsRectangle,
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsField,
    QgsFields,
    QgsWkbTypes,
    QgsCoordinateReferenceSystem
)
from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor
from qgis.core import QgsLineSymbolLayer

import numpy as np
import cv2
from ultralytics import YOLO
import math
import os

# === Parameters ===
tile_size = 224
IOU_THRESH = 0.45   # NMS IoU threshold (used both in YOLO and global NMS)

# Cache models by path (fast when switching models)
_models = {}


def _get_model(path: str):
    """
    Load (or reuse cached) YOLO model from a given filesystem path.
    """
    if not path:
        raise ValueError("Empty YOLO model path.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"YOLO model file not found: {path}")
    m = _models.get(path)
    if m is None:
        m = YOLO(path)
        _models[path] = m
    return m


def run_yolo_on_layer(
    layer,
    conf_threshold: float,
    model_path: str,
    tile_overlap_percent: int = 25,   # <-- NEW (0..99)
    progress_cb=None,
    is_canceled=None
):
    """
    Run YOLO detection on the provided raster layer with the given confidence and model.

    Args:
        layer: QgsRasterLayer to analyze.
        conf_threshold (float): YOLO confidence threshold (0..1).
        model_path (str): Filesystem path to a YOLO .pt/.pth model to use.
        tile_overlap_percent (int): tile overlap in percent (0..99). 99 means very dense overlap.
        progress_cb (callable|None): optional function accepting an int 0..100 for progress updates.
        is_canceled (callable|None): optional function returning True to request early cancel.

    Returns:
        QgsVectorLayer or None
    """

    # Clamp overlap percent to 0..99 to avoid stride=0 / infinite tiling
    try:
        tile_overlap_percent = int(tile_overlap_percent)
    except Exception:
        tile_overlap_percent = 25
    tile_overlap_percent = max(0, min(tile_overlap_percent, 99))
    tile_overlap = tile_overlap_percent / 100.0

    # Helper shorthands
    def _progress(p):
        if progress_cb is not None:
            try:
                progress_cb(int(p))
            except Exception:
                pass

    def _canceled():
        try:
            return bool(is_canceled and is_canceled())
        except Exception:
            return False

    _progress(1)  # started

    # --- Validate raster layer ---
    if not layer or layer.type() != QgsMapLayer.RasterLayer:
        iface.messageBar().pushWarning("ArchaeoAstroInsight ", "Please select a valid raster layer.")
        return None

    provider = layer.dataProvider()
    raster_extent = layer.extent()
    width = layer.width()
    height = layer.height()
    raster_crs = layer.crs()

    if width == 0 or height == 0 or raster_extent.isEmpty():
        iface.messageBar().pushCritical("ArchaeoAstroInsight ", "Invalid raster. Check extent or dimensions.")
        return None

    res_x = raster_extent.width() / float(width)
    res_y = raster_extent.height() / float(height)
    if res_x == 0 or res_y == 0:
        iface.messageBar().pushCritical("ArchaeoAstroInsight ", "Zero resolution. Aborting.")
        return None

    # --- Visible view extent (map CRS -> raster CRS) ---
    canvas = iface.mapCanvas()
    map_extent = canvas.extent()
    map_crs = canvas.mapSettings().destinationCrs()
    transform = QgsCoordinateTransform(map_crs, raster_crs, QgsProject.instance())
    view_extent = transform.transformBoundingBox(map_extent)
    view_extent = view_extent.intersect(raster_extent)
    if view_extent.isEmpty():
        iface.messageBar().pushWarning("ArchaeoAstroInsight ", "View does not intersect raster.")
        return None

    # Pixel coordinates of visible window
    x_min = max(0, int((view_extent.xMinimum() - raster_extent.xMinimum()) / res_x))
    x_max = min(width,  int((view_extent.xMaximum() - raster_extent.xMinimum()) / res_x))
    y_min = max(0, int((raster_extent.yMaximum() - view_extent.yMaximum()) / res_y))
    y_max = min(height, int((raster_extent.yMaximum() - view_extent.yMinimum()) / res_y))

    _progress(5)  # extents parsed

    if _canceled():
        return None

    

    _progress(15)  # image ready

    if _canceled():
        return None

    # --- Create memory vector layer for detections ---
    vector_layer = QgsVectorLayer("Polygon?crs=" + raster_crs.authid(), "YOLO_Detections", "memory")
    prov = vector_layer.dataProvider()
    fields = QgsFields()
    fields.append(QgsField(name="confidence", type=QVariant.Double))
    fields.append(QgsField(name="class", type=QVariant.String))
    prov.addAttributes(fields)
    vector_layer.updateFields()

    tile_counter = 0

    # Load/cached model for the requested path
    try:
        model = _get_model(model_path)
    except Exception as e:
        iface.messageBar().pushCritical("ArchaeoAstroInsight ", f"Failed to load YOLO model:\n{e}")
        return None

    _progress(20)  # model ready

    # --- Overlapping tiled inference setup ---
    tile_stride = max(1, int(tile_size * (1.0 - tile_overlap)))
    if tile_stride > tile_size:
        tile_stride = tile_size

    def _read_tile_bgr(x_pix: int, y_pix: int):
        """
        Read a tile from the raster provider at pixel offset (x_pix, y_pix)
        and return it as a uint8 BGR image of shape (tile_size, tile_size, bands).
        Returns None if tile is incomplete or cannot be read.
        """
        # tile bounds in raster CRS
        geo_xmin = raster_extent.xMinimum() + x_pix * res_x
        geo_xmax = raster_extent.xMinimum() + (x_pix + tile_size) * res_x
        geo_ymax = raster_extent.yMaximum() - y_pix * res_y
        geo_ymin = raster_extent.yMaximum() - (y_pix + tile_size) * res_y
        tile_extent = QgsRectangle(geo_xmin, geo_ymin, geo_xmax, geo_ymax)

        # Read up to first 3 bands (YOLO generally expects 3-channel)
        band_count = min(3, provider.bandCount())
        if band_count <= 0:
            return None

        chans = []
        for band_idx in range(1, band_count + 1):
            blk = provider.block(band_idx, tile_extent, tile_size, tile_size)
            if blk is None:
                return None

            # IMPORTANT: avoid dtype=float unless you truly need it (saves RAM)
            arr = np.array(blk.data(), dtype=np.float32).reshape(tile_size, tile_size)
            chans.append(arr)

        tile = np.dstack(chans)

        # Normalize per-tile to 0..255 then uint8
        tile = cv2.normalize(tile, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # If 1-band, expand to 3 bands for YOLO
        if tile.shape[2] == 1:
            tile = np.repeat(tile, 3, axis=2)

        # Convert RGB->BGR if you have true RGB; if your bands are not RGB, keep as-is
        tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        return tile_bgr


    def _tile_starts(min_pix, max_pix, size, stride):
        """
        Generate tile start indices so that:
          - tiles are spaced by 'stride'
          - last tile covers the end (max_pix)
        """
        starts = list(range(min_pix, max_pix - size + 1, stride))
        if not starts:
            # visible window smaller than one tile: single tile from min_pix
            return [min_pix]

        last = starts[-1]
        if last + size < max_pix:
            # add extra tile to cover the edge
            starts.append(max_pix - size)
        return sorted(set(starts))

    xs = _tile_starts(x_min, x_max, tile_size, tile_stride)
    ys = _tile_starts(y_min, y_max, tile_size, tile_stride)

    total_tiles = len(xs) * len(ys)
    processed_tiles = 0

    # Collect all detections (for global NMS)
    # Each entry: (QgsRectangle, conf, cls)
    all_detections = []

    # --- Tiled inference over visible window (with overlap) ---
    for y in ys:
        if _canceled():
            break
        for x in xs:
            if _canceled():
                break

            tile = _read_tile_bgr(x, y)
            if tile is None or tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                processed_tiles += 1
                _progress(20 + int(80 * processed_tiles / max(1, total_tiles)))
                continue


            # skip partial edge tiles (matches original logic)
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                processed_tiles += 1
                _progress(20 + int(80 * processed_tiles / max(1, total_tiles)))
                continue

            tile_counter += 1

            # Run YOLO on this tile
            results = model.predict(tile, conf=conf_threshold, iou=IOU_THRESH, verbose=False)
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = str(box.cls[0].item()) if box.cls is not None else ""

                    # Tile-relative bbox -> global pixel coords
                    x1, y1_box, x2, y2_box = xyxy
                    x1_global = x + x1
                    x2_global = x + x2
                    y1_global = y + y1_box
                    y2_global = y + y2_box

                    # Pixel -> geographic (raster CRS)
                    geo_x1 = raster_extent.xMinimum() + x1_global * res_x
                    geo_x2 = raster_extent.xMinimum() + x2_global * res_x
                    geo_y1 = raster_extent.yMaximum() - y1_global * res_y
                    geo_y2 = raster_extent.yMaximum() - y2_global * res_y

                    rect = QgsRectangle(geo_x1, geo_y2, geo_x2, geo_y1)  # xmin, ymin, xmax, ymax
                    all_detections.append((rect, conf, cls))

            processed_tiles += 1
            _progress(20 + int(80 * processed_tiles / max(1, total_tiles)))

    # --- Global NMS across overlapping tiles ---
    def _rect_iou(r1: QgsRectangle, r2: QgsRectangle) -> float:
        ixmin = max(r1.xMinimum(), r2.xMinimum())
        iymin = max(r1.yMinimum(), r2.yMinimum())
        ixmax = min(r1.xMaximum(), r2.xMaximum())
        iymax = min(r1.yMaximum(), r2.yMaximum())

        iw = max(0.0, ixmax - ixmin)
        ih = max(0.0, iymax - iymin)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0

        area1 = (r1.xMaximum() - r1.xMinimum()) * (r1.yMaximum() - r1.yMinimum())
        area2 = (r2.xMaximum() - r2.xMinimum()) * (r2.yMaximum() - r2.yMinimum())
        return inter / (area1 + area2 - inter)

    # Sort by confidence (descending)
    all_detections.sort(key=lambda d: d[1], reverse=True)

    kept = []
    for rect, conf, cls in all_detections:
        keep = True
        for k_rect, k_conf, k_cls in kept:
            if _rect_iou(rect, k_rect) >= IOU_THRESH:
                keep = False
                break
        if keep:
            kept.append((rect, conf, cls))

    # Create features from NMS result
    feature_counter = 0
    feats = []
    for rect, conf, cls in kept:
        feat = QgsFeature()
        feat.setGeometry(QgsGeometry.fromRect(rect))
        feat.setAttributes([conf, cls])
        feats.append(feat)
        feature_counter += 1

    if feats:
        prov.addFeatures(feats)
        vector_layer.updateExtents()

    # Add layer to project
    QgsProject.instance().addMapLayer(vector_layer)

    # Style polygons: red outline, no fill
    try:
        symbol = vector_layer.renderer().symbol()
        outline = symbol.symbolLayer(0)
        if isinstance(outline, QgsLineSymbolLayer):
            outline.setColor(QColor("red"))
            outline.setWidth(0.8)
        symbol.setColor(QColor(0, 0, 0, 0))
        vector_layer.triggerRepaint()
    except Exception:
        pass

    _progress(100)

    iface.messageBar().pushInfo(
        "ArchaeoAstroInsight ",
        f"Detection complete. Tiles: {tile_counter}, boxes: {feature_counter}, "
        f"conf: {conf_threshold:.2f}, IoU: {IOU_THRESH}, overlap: {tile_overlap_percent:d}%"
    )

    return vector_layer
