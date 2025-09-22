"""Flange hole counting utility using OpenCV.

This module provides a command line interface as well as reusable helper
functions to acquire images from up to three UVC compliant cameras and count
bolt holes on industrial flanges. The implementation follows the specification
provided in the task description and is designed to operate on Windows and
Linux/Raspberry Pi devices.

High level pipeline:

1. Discover and open up to three cameras, acquire single frames sequentially.
2. Optionally undistort each frame using camera calibration files.
3. Optionally stitch all frames to an orthorectified top view using ORB/FLANN
   feature matching and homography estimation.
4. Generate and apply a configurable ring-shaped ROI mask representing the
   expected bolt circle region.
5. Pre-process the ROI (grayscale, blur, illumination compensation, morphology).
6. Detect candidate holes using contour analysis (Pipeline A).
7. Optionally validate counts with a Canny/Hough based pipeline (Pipeline B).
8. Analyse the radial structure of detected holes to find concentric circles
   and check angular uniformity.
9. Export results and diagnostic metrics as JSON/CSV and optionally visualise.

The module exposes helper functions to facilitate testing as well as small unit
tests covering geometric primitives (roundness, px/mm conversion, clustering).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is optional, but required in prod
    yaml = None  # type: ignore

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ROIConfig:
    """Configuration for the ring shaped region of interest."""

    inner_radius_mm: Optional[float] = None
    outer_radius_mm: Optional[float] = None
    inner_radius_px: Optional[int] = None
    outer_radius_px: Optional[int] = None


@dataclass
class ThresholdConfig:
    """Thresholding parameters used during illumination compensation."""

    type: str = "otsu"  # {"otsu", "adaptive_mean", "adaptive_gaussian"}
    block_size: int = 31
    c: int = 2


@dataclass
class MorphologyConfig:
    """Morphology configuration for noise removal."""

    operation: str = "open"  # {"open", "close"}
    kernel_size: int = 3
    iterations: int = 1


@dataclass
class HoughConfig:
    """Parameters for the optional Hough circle cross check."""

    enabled: bool = False
    dp: float = 1.2
    min_dist_factor: float = 1.2
    param1: int = 120  # higher Canny threshold; median heuristic seeds lower
    param2: int = 25  # accumulator threshold; lower catches more circles
    min_radius_mm: Optional[float] = None
    max_radius_mm: Optional[float] = None
    min_radius_px: Optional[int] = None
    max_radius_px: Optional[int] = None


@dataclass
class DetectionConfig:
    """Detection related parameters."""

    min_area_px: float = 50.0
    roundness_threshold: float = 0.85
    min_radius_mm: Optional[float] = None
    max_radius_mm: Optional[float] = None
    min_radius_px: Optional[float] = None
    max_radius_px: Optional[float] = None
    cluster_tolerance_px: float = 5.0
    angle_std_threshold_deg: float = 5.0
    hough: HoughConfig = field(default_factory=HoughConfig)


@dataclass
class StitchingConfig:
    """Image stitching parameters for multi camera setups."""

    enabled: bool = False
    reference_camera: int = 0
    warp_width: int = 1920
    warp_height: int = 1080
    max_features: int = 2000
    good_match_percent: float = 0.15


@dataclass
class CalibrationConfig:
    """Path configuration for loading camera calibration files."""

    camera_matrix: Optional[str] = None
    dist_coeffs: Optional[str] = None


@dataclass
class CaptureConfig:
    """Camera acquisition related configuration."""

    max_cameras: int = 3
    indices: List[int] = field(default_factory=list)
    resolution: Tuple[int, int] = (1280, 720)
    fps: int = 15
    backend_priority: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "Windows": ["msmf", "dshow", "any"],
            "Linux": ["v4l2", "any"],
            "Darwin": ["avfoundation", "any"],
        }
    )
    calibrations: Dict[int, CalibrationConfig] = field(default_factory=dict)
    stitching: StitchingConfig = field(default_factory=StitchingConfig)


@dataclass
class OutputConfig:
    """Result export configuration."""

    output_dir: str = "output"
    json_name: str = "result.json"
    csv_name: str = "result.csv"


@dataclass
class ProcessingConfig:
    """Combination of processing related parameters."""

    roi: ROIConfig = field(default_factory=ROIConfig)
    pixels_per_mm: Optional[float] = None
    blur_kernel: int = 5
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)


@dataclass
class AppConfig:
    """Top level application configuration."""

    capture: CaptureConfig = field(default_factory=CaptureConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def load_yaml_config(path: Optional[str]) -> AppConfig:
    """Load configuration from YAML and merge with defaults.

    Parameters
    ----------
    path:
        Optional path to a YAML configuration file. If ``None`` or loading
        fails, defaults are used.
    """

    config = AppConfig()
    if path is None:
        return config

    if yaml is None:
        raise RuntimeError("PyYAML is not installed but a config file was provided")

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    # Helper to deep update dataclasses
    def update_dataclass(dc_obj, values):
        for key, value in values.items():
            if not hasattr(dc_obj, key):
                logging.warning("Unknown config key '%s'", key)
                continue
            attr = getattr(dc_obj, key)
            if dataclasses.is_dataclass(attr):  # type: ignore
                update_dataclass(attr, value)
            elif isinstance(attr, dict) and isinstance(value, dict):
                attr.update(value)
            else:
                setattr(dc_obj, key, value)

    import dataclasses  # local import to avoid global dependency earlier

    update_dataclass(config, data)

    # Convert calibration dictionary entries into dataclasses
    calib_dict = {}
    for key, entry in config.capture.calibrations.items():
        calib_dict[int(key)] = CalibrationConfig(**entry) if isinstance(entry, dict) else entry
    config.capture.calibrations = calib_dict

    return config


def ensure_output_dir(output: OutputConfig) -> Path:
    """Ensure the output directory exists and return its path."""

    out_dir = Path(output.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def backend_name_to_cv2(backend_name: str) -> int:
    """Map human friendly backend names to OpenCV constants."""

    mapping = {
        "any": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "v4l2": cv2.CAP_V4L2,
        "avfoundation": cv2.CAP_AVFOUNDATION,
    }
    return mapping.get(backend_name.lower(), cv2.CAP_ANY)


def discover_camera_indices(max_cameras: int, backend_priority: Sequence[str]) -> List[int]:
    """Probe camera indices using the provided backend priority order."""

    found: List[int] = []
    max_probe = 10  # probe the first ten indices to cover most setups
    for index in range(max_probe):
        if len(found) >= max_cameras:
            break
        for backend_name in backend_priority:
            backend = backend_name_to_cv2(backend_name)
            cap = cv2.VideoCapture(index, backend)
            if cap is not None and cap.isOpened():
                logging.info("Camera index %d available via backend %s", index, backend_name)
                found.append(index)
                cap.release()
                break
            if cap is not None:
                cap.release()
        else:
            logging.debug("Camera index %d not available", index)
    return found


def resolve_camera_indices(config: CaptureConfig) -> List[int]:
    """Resolve camera indices based on configuration or automatic discovery."""

    if config.indices:
        logging.info("Using configured camera indices: %s", config.indices)
        return config.indices[: config.max_cameras]

    system = platform.system()
    backend_priority = config.backend_priority.get(system, config.backend_priority.get("Linux", ["any"]))
    indices = discover_camera_indices(config.max_cameras, backend_priority)
    if not indices:
        raise RuntimeError("No cameras detected. Please connect a UVC camera or specify indices explicitly.")
    return indices


def load_matrix(path: str) -> np.ndarray:
    """Load calibration matrix from .npy, .npz or OpenCV YAML/XML."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    if file_path.suffix.lower() in {".npy", ".npz"}:
        data = np.load(str(file_path))
        # np.load on npz returns dict-like, ensure we fetch array
        if isinstance(data, np.ndarray):
            return data
        # fallback for npz: use first array
        first_key = list(data.keys())[0]
        return data[first_key]

    fs = cv2.FileStorage(str(file_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Unable to open calibration file via cv2.FileStorage: {path}")
    node = fs.getFirstTopLevelNode()
    matrix = node.mat()
    fs.release()
    return matrix


def undistort_image(image: np.ndarray, calibration: CalibrationConfig) -> np.ndarray:
    """Apply camera calibration if available."""

    if not calibration.camera_matrix or not calibration.dist_coeffs:
        return image

    camera_matrix = load_matrix(calibration.camera_matrix)
    dist_coeffs = load_matrix(calibration.dist_coeffs)
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, w_roi, h_roi = roi
    if w_roi > 0 and h_roi > 0:
        undistorted = undistorted[y : y + h_roi, x : x + w_roi]
    return undistorted


def capture_single_frame(index: int, backend: int, resolution: Tuple[int, int], fps: int) -> Optional[np.ndarray]:
    """Capture a single frame from the specified camera."""

    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        logging.error("Unable to open camera %d using backend %s", index, backend)
        return None

    width, height = resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Small delay allows exposure/gain to settle on some UVC devices
    time.sleep(0.2)

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        logging.error("Camera %d returned no frame", index)
        return None
    logging.info("Captured frame from camera %d with shape %s", index, frame.shape)
    return frame


def capture_frames(config: CaptureConfig) -> Tuple[List[np.ndarray], List[Tuple[int, str]]]:
    """Capture frames sequentially from up to three cameras."""

    indices = resolve_camera_indices(config)
    system = platform.system()
    backend_priority = config.backend_priority.get(system, config.backend_priority.get("Linux", ["any"]))

    frames: List[np.ndarray] = []
    meta: List[Tuple[int, str]] = []
    for index in indices:
        frame = None
        chosen_backend = "any"
        for backend_name in backend_priority:
            backend = backend_name_to_cv2(backend_name)
            frame = capture_single_frame(index, backend, config.resolution, config.fps)
            if frame is not None:
                chosen_backend = backend_name
                break
        if frame is None:
            logging.error("Failed to capture from camera %d", index)
            continue
        frames.append(frame)
        meta.append((index, chosen_backend))

    if not frames:
        raise RuntimeError("Failed to capture images from any camera.")
    return frames, meta


def stitch_frames(frames: Sequence[np.ndarray], stitching: StitchingConfig) -> np.ndarray:
    """Stitch frames to an orthorectified top view using ORB features."""

    if not stitching.enabled or len(frames) == 1:
        return frames[0]

    logging.info("Stitching %d frames", len(frames))
    orb = cv2.ORB_create(nfeatures=stitching.max_features)
    reference_index = min(stitching.reference_camera, len(frames) - 1)
    base = frames[reference_index]
    base_keypoints, base_descriptors = orb.detectAndCompute(base, None)
    if base_descriptors is None:
        raise RuntimeError("ORB failed to compute descriptors for base frame")

    base_h, base_w = base.shape[:2]
    accumulation = np.zeros((stitching.warp_height, stitching.warp_width, 3), dtype=base.dtype)
    homographies = [np.eye(3, dtype=np.float32) for _ in frames]

    index_order = list(range(len(frames)))
    index_order.pop(reference_index)

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1),
        dict(checks=50),
    )

    for idx in index_order:
        frame = frames[idx]
        keypoints, descriptors = orb.detectAndCompute(frame, None)
        if descriptors is None:
            raise RuntimeError(f"ORB failed to compute descriptors for frame {idx}")

        matches = flann.knnMatch(descriptors, base_descriptors, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 8:
            raise RuntimeError("Not enough good matches to compute homography")

        src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([base_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        if H is None:
            raise RuntimeError("Homography computation failed")

        homographies[idx] = H.astype(np.float32)

    for idx, frame in enumerate(frames):
        H = homographies[idx]
        warped = cv2.warpPerspective(frame, H, (stitching.warp_width, stitching.warp_height))
        # Accumulate by simple averaging (naive but effective for static scenes)
        mask = (warped > 0).astype(np.uint8)
        accumulation += warped * mask

    # Normalise by number of contributing frames; avoid division by zero
    accumulation = accumulation / max(len(frames), 1)
    accumulation = accumulation.astype(frames[0].dtype)
    return accumulation


def compute_radius_pixels(value_mm: Optional[float], value_px: Optional[float], pixels_per_mm: Optional[float]) -> Optional[float]:
    """Convert radius configuration to pixels using calibration information."""

    if value_px is not None:
        return float(value_px)
    if value_mm is not None:
        if pixels_per_mm is None:
            raise ValueError("Pixels-per-mm calibration required for radius in mm")
        return value_mm * pixels_per_mm
    return None


def create_roi_mask(shape: Tuple[int, int, int], roi: ROIConfig, pixels_per_mm: Optional[float]) -> np.ndarray:
    """Create a donut shaped ROI mask based on configured radii."""

    height, width = shape[:2]
    center = (width // 2, height // 2)
    inner = compute_radius_pixels(roi.inner_radius_mm, roi.inner_radius_px, pixels_per_mm)
    outer = compute_radius_pixels(roi.outer_radius_mm, roi.outer_radius_px, pixels_per_mm)

    mask = np.zeros((height, width), dtype=np.uint8)
    if outer is None:
        outer = min(height, width) / 2.0
    cv2.circle(mask, center, int(outer), 255, thickness=-1)
    if inner is not None and inner > 0:
        cv2.circle(mask, center, int(inner), 0, thickness=-1)
    return mask


def preprocess_image(image: np.ndarray, proc_cfg: ProcessingConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Convert to grayscale and apply illumination compensation."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if proc_cfg.blur_kernel > 0:
        gray = cv2.GaussianBlur(gray, (proc_cfg.blur_kernel, proc_cfg.blur_kernel), 0)

    thresh_cfg = proc_cfg.threshold
    if thresh_cfg.type == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif thresh_cfg.type == "adaptive_mean":
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            max(3, thresh_cfg.block_size | 1),
            thresh_cfg.c,
        )
    elif thresh_cfg.type == "adaptive_gaussian":
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            max(3, thresh_cfg.block_size | 1),
            thresh_cfg.c,
        )
    else:
        raise ValueError(f"Unsupported threshold type: {thresh_cfg.type}")

    morph_cfg = proc_cfg.morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_cfg.kernel_size, morph_cfg.kernel_size))
    operation = cv2.MORPH_OPEN if morph_cfg.operation == "open" else cv2.MORPH_CLOSE
    binary = cv2.morphologyEx(binary, operation, kernel, iterations=morph_cfg.iterations)
    return gray, binary


def contour_roundness(area: float, perimeter: float) -> float:
    """Compute circularity measure (4πA / P^2)."""

    if perimeter == 0:
        return 0.0
    return (4 * math.pi * area) / (perimeter * perimeter)


def detect_contours(binary: np.ndarray, detection_cfg: DetectionConfig, pixels_per_mm: Optional[float]) -> Tuple[List[Tuple[Tuple[float, float], float]], List[float]]:
    """Detect potential holes using contour analysis."""

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Tuple[Tuple[float, float], float]] = []
    roundness_values: List[float] = []

    min_radius_px = compute_radius_pixels(detection_cfg.min_radius_mm, detection_cfg.min_radius_px, pixels_per_mm)
    max_radius_px = compute_radius_pixels(detection_cfg.max_radius_mm, detection_cfg.max_radius_px, pixels_per_mm)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < detection_cfg.min_area_px:
            continue
        perimeter = cv2.arcLength(contour, True)
        roundness = contour_roundness(area, perimeter)
        if roundness < detection_cfg.roundness_threshold:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if min_radius_px and radius < min_radius_px:
            continue
        if max_radius_px and radius > max_radius_px:
            continue
        candidates.append(((x, y), radius))
        roundness_values.append(roundness)
    logging.info("Contour pipeline found %d candidates", len(candidates))
    return candidates, roundness_values


def estimate_flange_center(binary: np.ndarray) -> Tuple[float, float]:
    """Estimate flange centre using the largest circular contour."""

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = binary.shape[:2]
        return w / 2.0, h / 2.0

    largest = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest)
    logging.debug("Estimated flange centre at (%.2f, %.2f) with radius %.2f", x, y, radius)
    return float(x), float(y)


def cluster_radii(points: Sequence[Tuple[Tuple[float, float], float]], tolerance: float, centre: Tuple[float, float]) -> Dict[int, List[Tuple[Tuple[float, float], float, float]]]:
    """Cluster detections into concentric rings based on radius."""

    cx, cy = centre
    radii_with_points: List[Tuple[float, Tuple[float, float], float]] = []
    for (x, y), radius in points:
        radial_distance = math.hypot(x - cx, y - cy)
        radii_with_points.append((radial_distance, (x, y), radius))

    radii_with_points.sort(key=lambda item: item[0])
    clusters: Dict[int, List[Tuple[Tuple[float, float], float, float]]] = {}
    cluster_idx = 0
    for radial, point, radius in radii_with_points:
        if not clusters:
            clusters[cluster_idx] = [(point, radius, radial)]
            continue
        last_cluster = clusters[cluster_idx]
        last_radial = last_cluster[-1][2]
        if abs(radial - last_radial) <= tolerance:
            last_cluster.append((point, radius, radial))
        else:
            cluster_idx += 1
            clusters[cluster_idx] = [(point, radius, radial)]
    return clusters


def analyse_angular_uniformity(cluster: List[Tuple[Tuple[float, float], float, float]], centre: Tuple[float, float]) -> Tuple[List[float], float]:
    """Compute polar angles and their delta standard deviation."""

    cx, cy = centre
    angles = []
    for (x, y), _, _ in cluster:
        angle = math.degrees(math.atan2(y - cy, x - cx)) % 360.0
        angles.append(angle)
    if len(angles) < 2:
        return angles, 0.0
    angles.sort()
    deltas = []
    for i in range(len(angles)):
        next_angle = angles[(i + 1) % len(angles)]
        delta = (next_angle - angles[i]) % 360.0
        deltas.append(delta)
    std = float(np.std(deltas)) if deltas else 0.0
    return angles, std


def detect_hough_circles(gray: np.ndarray, binary: np.ndarray, detection_cfg: DetectionConfig, pixels_per_mm: Optional[float]) -> List[Tuple[Tuple[float, float], float]]:
    """Optional Pipeline B detection using Canny + HoughCircles."""

    if not detection_cfg.hough.enabled:
        return []

    median_val = float(np.median(gray))
    sigma = 0.33  # heuristic balancing noise vs. detail
    lower = int(max(0, (1.0 - sigma) * median_val))
    upper = int(min(255, (1.0 + sigma) * median_val))
    edges = cv2.Canny(gray, lower, max(lower + 1, upper))

    hough_cfg = detection_cfg.hough
    min_radius = compute_radius_pixels(hough_cfg.min_radius_mm, hough_cfg.min_radius_px, pixels_per_mm)
    max_radius = compute_radius_pixels(hough_cfg.max_radius_mm, hough_cfg.max_radius_px, pixels_per_mm)

    # minDist scales with expected diameter; default factor 1.2 keeps circles separate
    min_dist = 1.0
    if hough_cfg.min_radius_px:
        min_dist = hough_cfg.min_radius_px * 2 * hough_cfg.min_dist_factor
    elif min_radius:
        min_dist = min_radius * 2 * hough_cfg.min_dist_factor

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=hough_cfg.dp,
        minDist=min_dist,
        param1=hough_cfg.param1,
        param2=hough_cfg.param2,
        minRadius=int(min_radius) if min_radius else 0,
        maxRadius=int(max_radius) if max_radius else 0,
    )

    results: List[Tuple[Tuple[float, float], float]] = []
    if circles is not None:
        for circle in np.round(circles[0, :]).astype(int):
            x, y, radius = circle
            if binary[y, x] == 0:  # ensure circle lies within foreground
                continue
            results.append(((float(x), float(y)), float(radius)))
    logging.info("Hough pipeline found %d candidates", len(results))
    return results


def consolidate_detections(contour_detections: List[Tuple[Tuple[float, float], float]], hough_detections: List[Tuple[Tuple[float, float], float]], tolerance: float = 5.0) -> List[Tuple[Tuple[float, float], float]]:
    """Merge detections by proximity; used to compare contour and hough counts."""

    if not contour_detections:
        return hough_detections
    if not hough_detections:
        return contour_detections

    merged = contour_detections.copy()
    for hough_pt, hough_radius in hough_detections:
        hx, hy = hough_pt
        if any(math.hypot(cx - hx, cy - hy) <= tolerance for (cx, cy), _ in contour_detections):
            continue
        merged.append((hough_pt, hough_radius))
    return merged


def generate_visualisation(
    image: np.ndarray,
    detections: Dict[int, List[Tuple[Tuple[float, float], float, float]]],
    centre: Tuple[float, float],
    angle_stats: Dict[int, float],
) -> np.ndarray:
    """Generate overlay visualisation with indices and circles."""

    vis = image.copy()
    cx, cy = int(centre[0]), int(centre[1])
    cv2.circle(vis, (cx, cy), 5, (0, 255, 255), -1)
    for cluster_idx, cluster in detections.items():
        colour = tuple(int(c) for c in np.random.randint(0, 255, size=3))
        for idx, ((x, y), radius, _) in enumerate(cluster):
            cv2.circle(vis, (int(x), int(y)), int(radius), colour, 2)
            cv2.putText(vis, f"{cluster_idx}:{idx}", (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        cv2.putText(
            vis,
            f"Cluster {cluster_idx} std={angle_stats.get(cluster_idx, 0.0):.2f}",
            (10, 30 + 20 * cluster_idx),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
            cv2.LINE_AA,
        )
    return vis


def export_results(
    output_cfg: OutputConfig,
    meta: List[Tuple[int, str]],
    pixels_per_mm: Optional[float],
    counts_per_cluster: Dict[int, int],
    total_count: int,
    metrics: Dict[str, float],
) -> None:
    """Export results to JSON and append to CSV."""

    ensure_output_dir(output_cfg)
    timestamp = datetime.utcnow().isoformat()
    json_path = Path(output_cfg.output_dir) / output_cfg.json_name
    csv_path = Path(output_cfg.output_dir) / output_cfg.csv_name

    payload = {
        "timestamp": timestamp,
        "cameras": [dict(index=index, backend=backend) for index, backend in meta],
        "pixels_per_mm": pixels_per_mm,
        "counts": counts_per_cluster,
        "total_count": total_count,
        "metrics": metrics,
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logging.info("Wrote JSON results to %s", json_path)

    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["timestamp", "camera_indices", "pixels_per_mm", "counts", "total", "metrics"])
        writer.writerow([
            timestamp,
            ";".join(str(index) for index, _ in meta),
            pixels_per_mm or "",
            json.dumps(counts_per_cluster),
            total_count,
            json.dumps(metrics),
        ])
    logging.info("Appended results to %s", csv_path)


def compute_contrast_metric(gray: np.ndarray) -> float:
    """Simple focus/contrast metric using Laplacian variance."""

    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    logging.debug("Contrast metric (Laplacian variance): %.2f", variance)
    return variance


def process_image(
    image: np.ndarray,
    config: AppConfig,
    meta: Optional[List[Tuple[int, str]]] = None,
    save_debug: bool = False,
    debug_dir: Optional[Path] = None,
) -> Tuple[Dict[int, int], int, Dict[str, float], Dict[int, List[Tuple[Tuple[float, float], float, float]]], Tuple[float, float], np.ndarray, np.ndarray]:
    """Process a single image and return detection results and intermediates."""

    pixels_per_mm = config.processing.pixels_per_mm

    mask = create_roi_mask(image.shape, config.processing.roi, pixels_per_mm)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    gray, binary = preprocess_image(masked_image, config.processing)
    if save_debug and debug_dir is not None:
        cv2.imwrite(str(debug_dir / "1_gray.png"), gray)
        cv2.imwrite(str(debug_dir / "2_binary.png"), binary)
        cv2.imwrite(str(debug_dir / "3_mask.png"), mask)

    contour_detections, roundness_values = detect_contours(binary, config.detection, pixels_per_mm)
    centre = estimate_flange_center(binary)
    clusters = cluster_radii(contour_detections, config.detection.cluster_tolerance_px, centre)

    angle_stats: Dict[int, float] = {}
    counts_per_cluster: Dict[int, int] = {}
    delta_phi_std_values = []
    for cluster_idx, cluster in clusters.items():
        angles, std = analyse_angular_uniformity(cluster, centre)
        angle_stats[cluster_idx] = std
        counts_per_cluster[cluster_idx] = len(cluster)
        if std:
            delta_phi_std_values.append(std)

    total_count = sum(counts_per_cluster.values())

    hough_detections = detect_hough_circles(gray, binary, config.detection, pixels_per_mm)
    uncertain = False
    if hough_detections:
        if abs(len(hough_detections) - total_count) > 1:  # tolerance of 1 hole difference
            uncertain = True

    merged_detections = consolidate_detections(contour_detections, hough_detections)

    metrics = {
        "contrast": compute_contrast_metric(gray),
        "roundness_mean": float(np.mean(roundness_values)) if roundness_values else 0.0,
        "delta_phi_std": float(np.mean(delta_phi_std_values)) if delta_phi_std_values else 0.0,
        "uncertain": float(uncertain),
    }

    if save_debug and debug_dir is not None:
        overlay = generate_visualisation(image, clusters, centre, angle_stats)
        cv2.imwrite(str(debug_dir / "4_overlay.png"), overlay)

    return counts_per_cluster, total_count, metrics, clusters, centre, gray, binary


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Flange hole counter")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--save-debug", action="store_true", help="Save debug images")
    parser.add_argument("--single-shot", action="store_true", help="Capture a single set of frames and exit")
    parser.add_argument("--visualize", action="store_true", help="Display visualisation window")
    parser.add_argument("--test-image", type=str, default=None, help="Run pipeline on a static image (skips camera capture)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--run-tests", action="store_true", help="Execute unit tests and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s [%(levelname)s] %(message)s")

    if args.run_tests:
        import unittest

        unittest.main(module=__name__, argv=[sys.argv[0]], exit=False)
        return

    config = load_yaml_config(args.config)
    debug_dir = ensure_output_dir(config.output) if args.save_debug else None

    frame = None
    meta: List[Tuple[int, str]] = []
    if args.test_image:
        frame = cv2.imread(args.test_image)
        if frame is None:
            raise RuntimeError(f"Unable to load test image: {args.test_image}")
        logging.info("Loaded test image %s", args.test_image)
    else:
        frames, meta = capture_frames(config.capture)
        frames = [
            undistort_image(frame, config.capture.calibrations.get(index, CalibrationConfig()))
            for frame, (index, _) in zip(frames, meta)
        ]
        frame = stitch_frames(frames, config.capture.stitching)

    counts, total, metrics, clusters, centre, gray, binary = process_image(
        frame,
        config,
        meta,
        save_debug=args.save_debug,
        debug_dir=debug_dir,
    )

    export_results(config.output, meta, config.processing.pixels_per_mm, counts, total, metrics)

    logging.info("Detected %d holes across %d clusters", total, len(counts))
    for cluster_idx, count in counts.items():
        logging.info("Cluster %d -> %d holes", cluster_idx, count)

    if args.visualize:
        overlay = generate_visualisation(frame, clusters, centre, {idx: 0.0 for idx in clusters})
        cv2.imshow("Flange hole detection", overlay)
        cv2.imshow("Gray", gray)
        cv2.imshow("Binary", binary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.single_shot:
        return


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def run_static_test(image_path: str, config_path: Optional[str] = None) -> Tuple[Dict[int, int], int, Dict[str, float]]:
    """Convenience wrapper to run the pipeline on a static image."""

    config = load_yaml_config(config_path)
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Unable to read image at {image_path}")
    counts, total, metrics, *_ = process_image(image, config)
    return counts, total, metrics


# ---------------------------------------------------------------------------
# Unit tests (bonus)
# ---------------------------------------------------------------------------

import unittest


class GeometryTests(unittest.TestCase):
    """Verify helper computations independent of OpenCV heavy lifting."""

    def test_roundness_of_circle(self) -> None:
        radius = 10
        area = math.pi * radius * radius
        perimeter = 2 * math.pi * radius
        roundness = contour_roundness(area, perimeter)
        self.assertAlmostEqual(roundness, 1.0, places=5)

    def test_pixels_per_mm_conversion(self) -> None:
        value = compute_radius_pixels(5.0, None, 10.0)
        self.assertEqual(value, 50.0)
        with self.assertRaises(ValueError):
            compute_radius_pixels(5.0, None, None)

    def test_cluster_radii(self) -> None:
        detections = [((10.0, 0.0), 1.0), ((20.0, 0.0), 1.0), ((10.0, 10.0), 1.0)]
        centre = (0.0, 0.0)
        clusters = cluster_radii(detections, 5.0, centre)
        self.assertGreaterEqual(len(clusters), 1)
        total = sum(len(items) for items in clusters.values())
        self.assertEqual(total, len(detections))


if __name__ == "__main__":
    main()
