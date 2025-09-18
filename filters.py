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


def detect_small_circles(frame: np.ndarray) -> np.ndarray:
    """Detect and visualize circular features with normalized radii.

    The filter highlights the largest circles in the scene and analyses the
    remaining detections for groups of similarly sized circles. When at least
    three circles with comparable radii are found, their mean radius is used to
    render all non-maximal circles with a consistent size. The intention is to
    stabilise the visualisation of the small circular holes in the workpiece
    despite noisy detections.
    """

    if frame is None or frame.ndim != 3:
        return frame

    output = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2.0)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=120,
        param2=35,
        minRadius=5,
        maxRadius=0,
    )

    if circles is None or not len(circles):
        return output

    circles = np.round(circles[0]).astype(int)
    circle_list = [(int(x), int(y), int(r)) for x, y, r in circles]
    circle_list.sort(key=lambda c: c[2], reverse=True)

    largest_radius = circle_list[0][2]
    large_threshold = max(5, int(largest_radius * 0.8))
    large_circles = [c for c in circle_list if c[2] >= large_threshold]
    large_circle_set = {tuple(c) for c in large_circles}

    groups = []
    for circle in circle_list:
        if tuple(circle) in large_circle_set:
            continue
        radius = circle[2]
        placed = False
        for group in groups:
            tolerance = max(2.0, 0.08 * group["mean_radius"])
            if abs(group["mean_radius"] - radius) <= tolerance:
                group["members"].append(circle)
                group["mean_radius"] = float(
                    np.mean([member[2] for member in group["members"]])
                )
                placed = True
                break
        if not placed:
            groups.append({"mean_radius": float(radius), "members": [circle]})

    repeated_group = None
    for group in groups:
        if len(group["members"]) < 3:
            continue
        if repeated_group is None:
            repeated_group = group
            continue
        if len(group["members"]) > len(repeated_group["members"]):
            repeated_group = group
            continue
        if (
            len(group["members"]) == len(repeated_group["members"])
            and group["mean_radius"] < repeated_group["mean_radius"]
        ):
            repeated_group = group

    normalised_radius = None
    if repeated_group is not None:
        normalised_radius = int(round(repeated_group["mean_radius"]))

    small_candidates = [
        circle for circle in circle_list if tuple(circle) not in large_circle_set
    ]

    for x, y, radius in large_circles:
        cv2.circle(output, (x, y), radius, (255, 0, 0), 2)
        cv2.circle(output, (x, y), 2, (255, 0, 0), 3)

    if normalised_radius is not None:
        for x, y, _ in small_candidates:
            cv2.circle(output, (x, y), normalised_radius, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (0, 255, 0), 3)
        cv2.putText(
            output,
            f"Mittelradius: {normalised_radius}px",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    else:
        for x, y, radius in small_candidates:
            cv2.circle(output, (x, y), radius, (0, 255, 255), 1)
            cv2.circle(output, (x, y), 2, (0, 255, 255), 3)
        cv2.putText(
            output,
            "Nicht genug gleich große Kreise gefunden",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return output
