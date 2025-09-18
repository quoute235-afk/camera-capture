"""Image processing filters for camera frames."""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

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


def _fit_circle_least_squares(points: np.ndarray) -> Tuple[float, float, float]:
    """Return the centre and radius of the least-squares circle through *points*."""

    if len(points) < 3:
        raise ValueError("At least three points are required to fit a circle")

    # Solve the linearised circle equation: x^2 + y^2 = 2ax + 2by + c
    a_matrix = np.hstack((2.0 * points, np.ones((points.shape[0], 1))))
    b_vector = np.sum(points ** 2, axis=1)
    solution, *_ = np.linalg.lstsq(a_matrix, b_vector, rcond=None)
    cx, cy, c_term = solution
    radius_sq = c_term + cx * cx + cy * cy
    radius = math.sqrt(max(radius_sq, 0.0))
    return float(cx), float(cy), float(radius)


def _deduplicate_circles(
    circles: Sequence[Tuple[int, int, int]],
    min_center_distance: float,
) -> List[Tuple[int, int, int]]:
    """Merge detections that share (almost) the same centre."""

    merged: List[Tuple[int, int, int]] = []
    for x, y, radius in sorted(circles, key=lambda c: c[2], reverse=True):
        if all(np.hypot(x - mx, y - my) >= min_center_distance for mx, my, _ in merged):
            merged.append((x, y, radius))
    return merged


def _select_dominant_radius(
    circles: Sequence[Tuple[int, int, int]]
) -> Tuple[List[Tuple[int, int, int]], float]:
    """Return circles that share the dominant radius and the average radius."""

    if not circles:
        return [], 0.0

    radii = np.array([c[2] for c in circles], dtype=np.float32)
    median_radius = float(np.median(radii))
    radius_tolerance = max(2.0, 0.25 * median_radius)
    selected = [
        circle for circle in circles if abs(circle[2] - median_radius) <= radius_tolerance
    ]
    if not selected:
        return [], 0.0

    average_radius = float(np.mean([c[2] for c in selected]))
    return selected, average_radius


def detect_small_circles(
    frame: np.ndarray,
    inverted_primary: np.ndarray | None = None,
    inverted_secondary: np.ndarray | None = None,
) -> np.ndarray:
    """Combine inverted feeds and Canny edges to detect flange geometry.

    The filter operates on camera 0 and expects the original frame together with
    the inverted representations of camera 0 and camera 1. Both inverted views
    are converted to grayscale, merged, and processed using a Canny edge
    detector. The resulting edge map is searched for two circle populations:

    * the largest outer circle representing the flange and
    * at least three similarly sized circles representing the bolt holes.

    All confirmed bolt holes are rendered with the averaged radius of the
    dominant cluster. The detected flange outline and the fitted mid-circle are
    annotated together with a small preview of the combined Canny edge map.
    """

    if frame is None or frame.ndim != 3:
        return frame

    output = frame.copy()

    if inverted_primary is None or inverted_secondary is None:
        cv2.putText(
            output,
            "Invertierte Ansichten fehlen",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    height, width = frame.shape[:2]

    if inverted_primary.shape[:2] != (height, width):
        inverted_primary = cv2.resize(inverted_primary, (width, height))
    if inverted_secondary.shape[:2] != (height, width):
        inverted_secondary = cv2.resize(inverted_secondary, (width, height))

    gray_primary = cv2.cvtColor(inverted_primary, cv2.COLOR_BGR2GRAY)
    gray_secondary = cv2.cvtColor(inverted_secondary, cv2.COLOR_BGR2GRAY)

    edges_primary = cv2.Canny(gray_primary, 60, 180)
    edges_secondary = cv2.Canny(gray_secondary, 60, 180)
    combined_edges = cv2.bitwise_or(edges_primary, edges_secondary)

    blurred_edges = cv2.GaussianBlur(combined_edges, (7, 7), 1.5)

    max_small_radius = max(6, int(round(min(height, width) * 0.12)))
    small_circle_candidates = cv2.HoughCircles(
        blurred_edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=12,
        param1=160,
        param2=10,
        minRadius=3,
        maxRadius=max_small_radius,
    )

    if small_circle_candidates is None:
        cv2.putText(
            output,
            "Keine Schraubenlöcher erkannt",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    circles = np.round(small_circle_candidates[0]).astype(int)
    circles_list = [(int(x), int(y), int(r)) for x, y, r in circles]

    if len(circles_list) < 3:
        cv2.putText(
            output,
            "Mindestens 3 Löcher benötigt",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    median_radius = float(np.median([c[2] for c in circles_list]))
    dedup_radius = max(4.0, 0.6 * median_radius)
    merged_circles = _deduplicate_circles(circles_list, dedup_radius)

    selected_circles, avg_small_radius = _select_dominant_radius(merged_circles)
    if len(selected_circles) < 3:
        cv2.putText(
            output,
            "Keine stabile Lochgruppe",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    normalised_radius = max(1, int(round(avg_small_radius)))

    centres = np.array([(float(x), float(y)) for x, y, _ in selected_circles], dtype=np.float32)
    try:
        mid_cx, mid_cy, mid_radius = _fit_circle_least_squares(centres)
    except ValueError:
        mid_cx = float(np.mean(centres[:, 0]))
        mid_cy = float(np.mean(centres[:, 1]))
        radial_distances = np.linalg.norm(centres - np.array([mid_cx, mid_cy]), axis=1)
        mid_radius = float(np.mean(radial_distances)) if radial_distances.size else float(normalised_radius)

    outer_min_radius = max(
        30,
        int(round(mid_radius + avg_small_radius * 1.5)),
        max_small_radius * 2,
    )
    outer_max_radius = int(round(min(math.hypot(width, height) / 2.0, min(height, width) * 0.95)))

    flange_circle = None
    if outer_max_radius > outer_min_radius:
        large_circle_candidates = cv2.HoughCircles(
            blurred_edges,
            cv2.HOUGH_GRADIENT,
            dp=1.1,
            minDist=max(40, int(round(mid_radius))),
            param1=180,
            param2=25,
            minRadius=outer_min_radius,
            maxRadius=outer_max_radius,
        )
        if large_circle_candidates is not None:
            candidate_list = np.round(large_circle_candidates[0]).astype(int)
            sorted_candidates = sorted(
                [(int(x), int(y), int(r)) for x, y, r in candidate_list],
                key=lambda c: c[2],
                reverse=True,
            )
            for candidate in sorted_candidates:
                distance = math.hypot(candidate[0] - mid_cx, candidate[1] - mid_cy)
                if distance <= max(30.0, 0.25 * max(mid_radius, 1.0)):
                    flange_circle = candidate
                    break
            if flange_circle is None and sorted_candidates:
                flange_circle = sorted_candidates[0]

    if flange_circle is None:
        flange_circle = (
            int(round(mid_cx)),
            int(round(mid_cy)),
            int(round(mid_radius + avg_small_radius * 2.2)),
        )

    flange_cx, flange_cy, flange_radius = flange_circle

    cv2.circle(output, (flange_cx, flange_cy), flange_radius, (255, 0, 0), 2)
    cv2.circle(output, (int(round(mid_cx)), int(round(mid_cy))), int(round(mid_radius)), (0, 255, 255), 1)
    cv2.circle(output, (int(round(mid_cx)), int(round(mid_cy))), 3, (0, 0, 255), -1)

    for x, y, _ in selected_circles:
        cv2.circle(output, (int(x), int(y)), normalised_radius, (0, 255, 0), 2)
        cv2.circle(output, (int(x), int(y)), 2, (0, 255, 0), -1)

    cv2.putText(
        output,
        f"Loecher: {len(selected_circles)}  r={normalised_radius}px",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"Flansch: {flange_radius}px  Mitte: {int(round(mid_radius))}px",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "Basis: invertiert + Canny (Cam0+1)",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    preview_height = min(120, max(30, height // 3))
    preview_width = max(30, int(round(preview_height * width / max(height, 1))))
    preview_width = min(preview_width, max(30, width - 20))
    if (
        preview_height > 0
        and preview_width > 0
        and preview_height + 20 < height
        and preview_width + 20 < width
    ):
        preview_edges = cv2.resize(combined_edges, (preview_width, preview_height))
        preview_bgr = cv2.cvtColor(preview_edges, cv2.COLOR_GRAY2BGR)
        y0 = height - preview_height - 10
        x0 = width - preview_width - 10
        roi = output[y0 : y0 + preview_height, x0 : x0 + preview_width]
        blended = cv2.addWeighted(preview_bgr, 0.6, roi, 0.4, 0.0)
        output[y0 : y0 + preview_height, x0 : x0 + preview_width] = blended
        cv2.rectangle(
            output,
            (x0, y0),
            (x0 + preview_width, y0 + preview_height),
            (0, 255, 255),
            1,
        )
        cv2.putText(
            output,
            "Canny Merge",
            (x0 + 5, y0 + preview_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output
