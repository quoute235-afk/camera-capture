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


def _refine_ring_points(
    points: np.ndarray, max_iterations: int = 5
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Iteratively remove outliers until the circle fit stabilises."""

    active_mask = np.ones(len(points), dtype=bool)
    circle_parameters: Tuple[float, float, float] | None = None

    for _ in range(max_iterations):
        active_indices = np.flatnonzero(active_mask)
        if len(active_indices) < 4:
            break

        cx, cy, radius = _fit_circle_least_squares(points[active_indices])
        circle_parameters = (cx, cy, radius)

        distances = np.linalg.norm(points[active_indices] - np.array([cx, cy]), axis=1)
        deviation = np.abs(distances - radius)
        allowed_deviation = max(3.0, 0.08 * radius)
        keep_mask = deviation <= allowed_deviation
        if np.all(keep_mask):
            break

        # Remove outliers from the active set and repeat.
        active_mask[active_indices[~keep_mask]] = False

    if circle_parameters is None:
        raise ValueError("Unable to fit circle to the provided points")

    final_indices = np.flatnonzero(active_mask)
    if len(final_indices) >= 3:
        cx, cy, radius = _fit_circle_least_squares(points[final_indices])
        circle_parameters = (cx, cy, radius)

    return final_indices, circle_parameters


def detect_small_circles(frame: np.ndarray) -> np.ndarray:
    """Detect flange patterns consisting of a large circle and bolt holes.

    The filter searches for 4-18 small circles with a shared radius that lie on a
    common mid-circle. If such a constellation is found, the average radius of the
    small circles is used to render the detections. The outer flange diameter is
    derived from the same centre so that only consistent geometries are
    highlighted.
    """

    if frame is None or frame.ndim != 3:
        return frame

    output = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.equalizeHist(gray)

    # Detect candidate screw holes.
    small_circle_candidates = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=12,
        param1=180,
        param2=18,
        minRadius=3,
        maxRadius=45,
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

    if len(circles_list) < 4:
        cv2.putText(
            output,
            "Nicht genügend Schraubenlöcher",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    dedup_radius = max(4.0, float(np.median([c[2] for c in circles_list])) * 0.6)
    merged_circles = _deduplicate_circles(circles_list, dedup_radius)

    selected_circles, avg_small_radius = _select_dominant_radius(merged_circles)
    if len(selected_circles) < 4 or len(selected_circles) > 18:
        cv2.putText(
            output,
            "Kein konsistenter Schraubenloch-Kreis",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    centers = np.array([(float(x), float(y)) for x, y, _ in selected_circles], dtype=np.float32)
    try:
        inlier_indices, (mid_cx, mid_cy, mid_radius) = _refine_ring_points(centers)
    except ValueError:
        cv2.putText(
            output,
            "Mittlerer Kreis konnte nicht bestimmt werden",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    if len(inlier_indices) < 4 or len(inlier_indices) > 18:
        cv2.putText(
            output,
            "Zu wenige/zu viele Schraubenlöcher",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    inlier_circles = [selected_circles[int(idx)] for idx in inlier_indices]
    inlier_centers = centers[inlier_indices]

    # Validate the angular distribution to suppress false positives.
    vectors = inlier_centers - np.array([mid_cx, mid_cy])
    angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
    angles = np.sort((angles + 360.0) % 360.0)
    diffs = np.diff(np.concatenate([angles, angles[:1] + 360.0]))
    min_sep = float(diffs.min()) if len(diffs) else 360.0
    coverage = 360.0 - float(diffs.max()) if len(diffs) else 0.0

    if min_sep < 10.0 or coverage < 180.0:
        cv2.putText(
            output,
            "Schraubenlöcher nicht rund verteilt",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    avg_small_radius = float(np.mean([c[2] for c in inlier_circles]))
    normalised_radius = int(round(avg_small_radius))

    # Determine the flange diameter (outer circle) close to the fitted centre.
    flange_circle = None
    min_outer_radius = int(round(mid_radius + avg_small_radius * 1.2))
    max_outer_radius = int(round(mid_radius * 1.9 + avg_small_radius))
    if max_outer_radius > min_outer_radius + 2:
        large_circle_candidates = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=max(30, int(mid_radius * 0.5)),
            param1=200,
            param2=60,
            minRadius=min_outer_radius,
            maxRadius=max_outer_radius,
        )
        if large_circle_candidates is not None:
            candidate_list = np.round(large_circle_candidates[0]).astype(int)
            filtered_candidates = []
            for x, y, r in candidate_list:
                if r <= min_outer_radius:
                    continue
                center_distance = math.hypot(x - mid_cx, y - mid_cy)
                if center_distance > max(20.0, 0.2 * mid_radius):
                    continue
                filtered_candidates.append((x, y, r))

            if filtered_candidates:
                flange_circle = max(filtered_candidates, key=lambda c: c[2])

    if flange_circle is None:
        flange_circle = (
            int(round(mid_cx)),
            int(round(mid_cy)),
            int(round(mid_radius + avg_small_radius * 1.6)),
        )

    flange_cx, flange_cy, flange_radius = flange_circle

    # Draw flange and mid-circle.
    cv2.circle(output, (flange_cx, flange_cy), flange_radius, (255, 0, 0), 2)
    cv2.circle(output, (int(round(mid_cx)), int(round(mid_cy))), int(round(mid_radius)), (0, 255, 255), 1)
    cv2.circle(output, (int(round(mid_cx)), int(round(mid_cy))), 3, (0, 0, 255), -1)

    for x, y, _ in inlier_circles:
        cv2.circle(output, (int(x), int(y)), normalised_radius, (0, 255, 0), 2)
        cv2.circle(output, (int(x), int(y)), 2, (0, 255, 0), -1)

    cv2.putText(
        output,
        f"Loecher: {len(inlier_circles)}  r={normalised_radius}px",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"Mittenkreis: {int(round(mid_radius))}px  Flansch: {flange_radius}px",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output
