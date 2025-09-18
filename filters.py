"""Image processing filters for camera frames."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def invert(frame: np.ndarray) -> np.ndarray:
    """Return a grayscale-inverted frame with edge highlights.

    The input frame is converted to grayscale, inverted and converted back to
    BGR so it can be rendered alongside the colour streams in the GUI. Canny
    edges are detected on the original luminance channel and drawn in red to
    improve the visibility of structural features.
    """

    if frame is None or frame.ndim != 3:
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inverted_gray = cv2.bitwise_not(gray)
    inverted_bgr = cv2.cvtColor(inverted_gray, cv2.COLOR_GRAY2BGR)

    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(inverted_bgr, contours, -1, (0, 0, 255), 1)

    return inverted_bgr


def denoise(frame: np.ndarray) -> np.ndarray:
    """Reduce sensor noise while keeping edges crisp for subsequent filters."""

    if frame is None or frame.ndim != 3:
        return frame

    # fastNlMeansDenoisingColored preserves colours and edges better than a
    # simple blur and helps stabilise later edge detection.
    denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    return denoised


def canny_edges(
    frame: np.ndarray,
    thresholds: Tuple[int, int] = (60, 180),
) -> np.ndarray:
    """Return the Canny edge map of *frame* as a 3-channel image."""

    if frame is None or frame.ndim != 3:
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(blurred, thresholds[0], thresholds[1])
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

