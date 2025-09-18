"""Grayscale inversion and Canny edge filters for camera frames."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def invert_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert *frame* to grayscale, invert it, and return a 3-channel image."""

    if frame is None or frame.ndim != 3:
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    return cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)


def canny_from_inverted(
    frame: np.ndarray, thresholds: Tuple[int, int] = (60, 180)
) -> np.ndarray:
    """Compute a black and white Canny edge map from an inverted image."""

    if frame is None:
        return frame

    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        return frame

    edges = cv2.Canny(gray, thresholds[0], thresholds[1])
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
