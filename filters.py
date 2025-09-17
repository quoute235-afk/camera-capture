"""Image processing filters for camera frames."""

import cv2
import numpy as np


def invert(frame: np.ndarray) -> np.ndarray:
    """Return the color-inverted version of ``frame``."""

    return cv2.bitwise_not(frame)
