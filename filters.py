"""Image processing filters for camera frames."""
from __future__ import annotations

import cv2
import numpy as np


def invert(frame: np.ndarray) -> np.ndarray:
    """Return a grayscale-inverted frame with red contours highlighting edges.

    The input frame is converted to grayscale, inverted, and then converted
    back to a 3-channel image so it can be rendered alongside color frames in
    the GUI. Canny edge detection is used to extract the dominant geometry and
    the detected contours are traced with a red outline for better visibility.
    """
    # Convert to grayscale to ensure the inversion only affects luminance.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image and convert it back to BGR for display.
    inverted_gray = cv2.bitwise_not(gray)
    inverted_bgr = cv2.cvtColor(inverted_gray, cv2.COLOR_GRAY2BGR)

    # Detect edges and outline their contours in red to highlight geometry.
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(inverted_bgr, contours, -1, (0, 0, 255), 1)

    return inverted_bgr
