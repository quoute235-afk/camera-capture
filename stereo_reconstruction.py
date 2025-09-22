"""Utilities for calibrating stereo camera rigs and reconstructing 3D views."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class StereoCalibrationData:
    """Container with the intrinsic and extrinsic stereo parameters."""

    image_size: Tuple[int, int]
    camera_matrix_left: np.ndarray
    dist_coeffs_left: np.ndarray
    camera_matrix_right: np.ndarray
    dist_coeffs_right: np.ndarray
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    rectification_matrix_left: np.ndarray
    rectification_matrix_right: np.ndarray
    projection_matrix_left: np.ndarray
    projection_matrix_right: np.ndarray
    disparity_to_depth_map: np.ndarray
    reprojection_error: float

    def to_dict(self) -> Dict[str, object]:
        """Serialise the calibration to a plain dictionary."""

        return {
            "image_size": list(self.image_size),
            "camera_matrix_left": self.camera_matrix_left.tolist(),
            "dist_coeffs_left": self.dist_coeffs_left.tolist(),
            "camera_matrix_right": self.camera_matrix_right.tolist(),
            "dist_coeffs_right": self.dist_coeffs_right.tolist(),
            "rotation_matrix": self.rotation_matrix.tolist(),
            "translation_vector": self.translation_vector.tolist(),
            "rectification_matrix_left": self.rectification_matrix_left.tolist(),
            "rectification_matrix_right": self.rectification_matrix_right.tolist(),
            "projection_matrix_left": self.projection_matrix_left.tolist(),
            "projection_matrix_right": self.projection_matrix_right.tolist(),
            "disparity_to_depth_map": self.disparity_to_depth_map.tolist(),
            "reprojection_error": float(self.reprojection_error),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "StereoCalibrationData":
        """Re-create a :class:`StereoCalibrationData` from a JSON dictionary."""

        def _array(key: str) -> np.ndarray:
            value = data.get(key)
            if value is None:
                raise KeyError(f"Missing calibration field '{key}'.")
            return np.array(value, dtype=np.float64)

        image_size = data.get("image_size")
        if image_size is None:
            raise KeyError("Missing calibration field 'image_size'.")
        width, height = (int(image_size[0]), int(image_size[1]))

        return cls(
            image_size=(width, height),
            camera_matrix_left=_array("camera_matrix_left"),
            dist_coeffs_left=_array("dist_coeffs_left"),
            camera_matrix_right=_array("camera_matrix_right"),
            dist_coeffs_right=_array("dist_coeffs_right"),
            rotation_matrix=_array("rotation_matrix"),
            translation_vector=_array("translation_vector"),
            rectification_matrix_left=_array("rectification_matrix_left"),
            rectification_matrix_right=_array("rectification_matrix_right"),
            projection_matrix_left=_array("projection_matrix_left"),
            projection_matrix_right=_array("projection_matrix_right"),
            disparity_to_depth_map=_array("disparity_to_depth_map"),
            reprojection_error=float(data.get("reprojection_error", 0.0)),
        )


def _prepare_object_points(
    pattern_size: Tuple[int, int],
    square_size: float,
) -> np.ndarray:
    cols, rows = pattern_size
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    object_points = np.zeros((rows * cols, 3), np.float32)
    object_points[:, :2] = grid * square_size
    return object_points


def _find_corners(
    image: np.ndarray,
    pattern_size: Tuple[int, int],
    criteria: Tuple[int, int, float],
) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not found:
        return None
    refined = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria,
    )
    return refined


def compute_stereo_calibration(
    image_pairs: Iterable[Tuple[np.ndarray, np.ndarray]],
    pattern_size: Tuple[int, int] = (6, 6),
    square_size: float = 1.0,
) -> StereoCalibrationData:
    """Compute stereo calibration parameters from chessboard image pairs."""

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-5,
    )

    object_points: List[np.ndarray] = []
    image_points_left: List[np.ndarray] = []
    image_points_right: List[np.ndarray] = []

    image_size: Optional[Tuple[int, int]] = None

    base_object_points = _prepare_object_points(pattern_size, square_size)

    for left, right in image_pairs:
        if left is None or right is None:
            continue
        if left.shape[:2] != right.shape[:2]:
            raise ValueError("Stereo pair images must share the same resolution.")

        if image_size is None:
            image_size = (left.shape[1], left.shape[0])

        corners_left = _find_corners(left, pattern_size, criteria)
        corners_right = _find_corners(right, pattern_size, criteria)

        if corners_left is None or corners_right is None:
            continue

        object_points.append(base_object_points.copy())
        image_points_left.append(corners_left)
        image_points_right.append(corners_right)

    if not object_points:
        raise ValueError("Keine gültigen Schachbrettpaare für die Kalibrierung gefunden.")

    if image_size is None:
        raise ValueError("Die Bildgröße der Kalibrierung konnte nicht bestimmt werden.")

    rms, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, rotation_matrix, translation_vector, _, _ = (
        cv2.stereoCalibrate(
            object_points,
            image_points_left,
            image_points_right,
            None,
            None,
            None,
            None,
            image_size,
            criteria=criteria,
            flags=0,
        )
    )

    (
        rectification_matrix_left,
        rectification_matrix_right,
        projection_matrix_left,
        projection_matrix_right,
        disparity_to_depth_map,
        _,
        _,
    ) = cv2.stereoRectify(
        camera_matrix_left,
        dist_coeffs_left,
        camera_matrix_right,
        dist_coeffs_right,
        image_size,
        rotation_matrix,
        translation_vector,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    return StereoCalibrationData(
        image_size=image_size,
        camera_matrix_left=camera_matrix_left,
        dist_coeffs_left=dist_coeffs_left,
        camera_matrix_right=camera_matrix_right,
        dist_coeffs_right=dist_coeffs_right,
        rotation_matrix=rotation_matrix,
        translation_vector=translation_vector,
        rectification_matrix_left=rectification_matrix_left,
        rectification_matrix_right=rectification_matrix_right,
        projection_matrix_left=projection_matrix_left,
        projection_matrix_right=projection_matrix_right,
        disparity_to_depth_map=disparity_to_depth_map,
        reprojection_error=rms,
    )


def save_calibration(path: Path, calibration: StereoCalibrationData) -> None:
    """Persist calibration data as JSON."""

    path = Path(path)
    path.write_text(json.dumps(calibration.to_dict(), indent=2), encoding="utf-8")


def load_calibration(path: Path) -> StereoCalibrationData:
    """Load calibration parameters from *path*."""

    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return StereoCalibrationData.from_dict(data)


def create_default_matcher(
    min_disparity: int = 0,
    num_disparities: int = 128,
    block_size: int = 5,
) -> cv2.StereoMatcher:
    """Create a default StereoSGBM matcher suitable for medium resolutions."""

    num_disparities = max(16, int(np.ceil(num_disparities / 16)) * 16)
    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    matcher.setUniquenessRatio(10)
    matcher.setSpeckleWindowSize(100)
    matcher.setSpeckleRange(32)
    matcher.setDisp12MaxDiff(1)
    matcher.setPreFilterCap(63)
    matcher.setP1(8 * 3 * block_size ** 2)
    matcher.setP2(32 * 3 * block_size ** 2)
    return matcher


@dataclass
class StereoReconstructionResult:
    """Result of a single stereo reconstruction step."""

    rectified_left: np.ndarray
    rectified_right: np.ndarray
    disparity: np.ndarray
    depth_colormap: np.ndarray
    point_cloud: np.ndarray
    mask: np.ndarray

    def filtered_points(self) -> np.ndarray:
        """Return only valid 3D points."""

        valid = self.mask.reshape(-1)
        points = self.point_cloud.reshape(-1, 3)
        return points[valid]


class StereoReconstructor:
    """Convenience wrapper that caches rectification maps and computes depth."""

    def __init__(
        self,
        calibration: StereoCalibrationData,
        matcher: Optional[cv2.StereoMatcher] = None,
    ) -> None:
        self.calibration = calibration
        self.matcher = matcher or create_default_matcher()
        self._map_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def _get_maps(
        self, image_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if image_size not in self._map_cache:
            map_left = cv2.initUndistortRectifyMap(
                self.calibration.camera_matrix_left,
                self.calibration.dist_coeffs_left,
                self.calibration.rectification_matrix_left,
                self.calibration.projection_matrix_left,
                image_size,
                cv2.CV_32FC1,
            )
            map_right = cv2.initUndistortRectifyMap(
                self.calibration.camera_matrix_right,
                self.calibration.dist_coeffs_right,
                self.calibration.rectification_matrix_right,
                self.calibration.projection_matrix_right,
                image_size,
                cv2.CV_32FC1,
            )
            self._map_cache[image_size] = (*map_left, *map_right)
        return self._map_cache[image_size]

    def _rectify_pair(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        if left.shape[:2] != right.shape[:2]:
            raise ValueError("Linkes und rechtes Bild müssen gleich groß sein.")

        height, width = left.shape[:2]
        if (width, height) != self.calibration.image_size:
            raise ValueError(
                "Die Auflösung der Live-Bilder entspricht nicht der Kalibrierung."
            )

        map1_left, map2_left, map1_right, map2_right = self._get_maps((width, height))
        rect_left = cv2.remap(left, map1_left, map2_left, cv2.INTER_LINEAR)
        rect_right = cv2.remap(right, map1_right, map2_right, cv2.INTER_LINEAR)
        return rect_left, rect_right, (map1_left, map2_left, map1_right, map2_right)

    @staticmethod
    def _remap_mask(mask: np.ndarray, map1: np.ndarray, map2: np.ndarray) -> np.ndarray:
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        rect_mask = cv2.remap(mask, map1, map2, cv2.INTER_NEAREST)
        return rect_mask

    def compute(
        self,
        left: np.ndarray,
        right: np.ndarray,
        masks: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> StereoReconstructionResult:
        """Compute a depth map and point cloud from *left* and *right* frames."""

        rect_left, rect_right, maps = self._rectify_pair(left, right)
        map1_left, map2_left, map1_right, map2_right = maps

        gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

        disparity_raw = self.matcher.compute(gray_left, gray_right).astype(np.float32)
        disparity = disparity_raw / 16.0

        mask = np.isfinite(disparity)
        mask &= disparity > 0

        if masks is not None:
            left_mask, right_mask = masks
            left_mask_rect = self._remap_mask(left_mask, map1_left, map2_left)
            right_mask_rect = self._remap_mask(right_mask, map1_right, map2_right)
            combined_mask = cv2.bitwise_or(left_mask_rect, right_mask_rect)
            mask &= combined_mask > 0

        point_cloud = cv2.reprojectImageTo3D(disparity, self.calibration.disparity_to_depth_map)
        point_cloud = np.where(mask[..., None], point_cloud, np.nan)

        depth_colormap = _colorize_disparity(disparity, mask)

        return StereoReconstructionResult(
            rectified_left=rect_left,
            rectified_right=rect_right,
            disparity=disparity,
            depth_colormap=depth_colormap,
            point_cloud=point_cloud,
            mask=mask,
        )


def _colorize_disparity(disparity: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if disparity.size == 0:
        return np.zeros((*disparity.shape, 3), dtype=np.uint8)

    valid_values = disparity[mask]
    if valid_values.size == 0:
        return np.zeros((*disparity.shape, 3), dtype=np.uint8)

    disp_min = np.nanmin(valid_values)
    disp_max = np.nanmax(valid_values)
    if disp_max - disp_min < 1e-3:
        disp_max = disp_min + 1e-3

    normalized = (disparity - disp_min) / (disp_max - disp_min)
    normalized = np.clip(normalized, 0, 1)
    normalized_uint8 = (normalized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized_uint8, cv2.COLORMAP_TURBO)
    colored[~mask] = 0
    return colored


def load_reconstructor(
    path: Path,
    matcher: Optional[cv2.StereoMatcher] = None,
) -> StereoReconstructor:
    """Load calibration data from *path* and build a :class:`StereoReconstructor`."""

    calibration = load_calibration(path)
    return StereoReconstructor(calibration, matcher=matcher)
