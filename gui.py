"""Tkinter GUI for displaying multiple camera feeds and filters."""
from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

from camera_capture import MultiCameraCapture
from filters import canny_from_inverted, invert_grayscale
from stereo_reconstruction import (
    StereoReconstructor,
    create_default_matcher,
    load_calibration,
)


class DualInvertApp:
    """GUI application showing original and processed views for cameras."""

    def __init__(
        self,
        root: tk.Tk,
        camera_indices: Iterable[int] = (0, 1),
    ) -> None:
        self.root = root
        self.root.title("Dual Kamera Ansicht mit Invertierung")

        self._camera_indices: List[int] = list(camera_indices)

        self._stereo_renderer: Optional[StereoReconstructor] = None
        self._stereo_status: Optional[str] = None
        self._stereo_pair: Optional[Tuple[int, int]] = None
        self._stereo_calibration_size: Optional[Tuple[int, int]] = None
        self._load_stereo_configuration()

        capture_frame_size = self._stereo_calibration_size or (640, 360)
        self._capture = MultiCameraCapture(
            self._camera_indices,
            frame_size=capture_frame_size,
        )
        self._photo_images: Dict[Tuple[int, str], tk.PhotoImage] = {}
        self._last_frames: Dict[int, Dict[str, np.ndarray]] = {}
        self._live_interval_ms = 33  # ~30 FPS for the live preview
        self._canny_interval_ms = 150  # ~6-7 FPS for the Canny view
        self._last_canny_timestamp: Optional[float] = None

        initial_status = self._stereo_status or "Inaktiv. Bitte 'Start' drücken."
        self.status_var = tk.StringVar(value=initial_status)

        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._update_loop()

    def _build_layout(self) -> None:
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.LabelFrame(main_frame, text="Steuerung", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        start_button = ttk.Button(control_frame, text="Start", command=self.start_cameras)
        start_button.pack(fill=tk.X, pady=(0, 5))

        stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_cameras)
        stop_button.pack(fill=tk.X)

        save_button = ttk.Button(
            control_frame, text="Speichern", command=self.save_frames
        )
        save_button.pack(fill=tk.X, pady=(5, 0))

        autofocus_button = ttk.Button(
            control_frame,
            text="Refresh Auto Fokus",
            command=self.refresh_autofocus,
        )
        autofocus_button.pack(fill=tk.X, pady=(5, 0))

        quit_button = ttk.Button(control_frame, text="Beenden", command=self.on_close)
        quit_button.pack(fill=tk.X, pady=(5, 0))

        status_label = ttk.Label(control_frame, textvariable=self.status_var, wraplength=140)
        status_label.pack(fill=tk.X, pady=(15, 0))

        video_frame = ttk.Frame(main_frame, padding=10)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._labels: Dict[int, Dict[str, ttk.Label]] = {}

        max_column_index = 0

        for row, index in enumerate(self._camera_indices):
            panel_config = [
                ("original", "Original", "Kein Signal"),
                ("canny", "Canny (Invertiert)", "Kein Signal"),
            ]
            if row == 0 and len(self._camera_indices) >= 2:
                default_text = (
                    "Warte auf Stereo-Daten"
                    if self._stereo_renderer is not None
                    else (self._stereo_status or "Stereo deaktiviert")
                )
                panel_config.append(("3d", "3D Ansicht", default_text))
            max_column_index = max(max_column_index, len(panel_config) - 1)

            row_labels: Dict[str, ttk.Label] = {}
            for column, (key, title_suffix, default_text) in enumerate(panel_config):
                row_labels[key] = self._create_image_panel(
                    video_frame,
                    f"Kamera {row + 1} - {title_suffix}",
                    row,
                    column,
                    default_text=default_text,
                )
            self._labels[index] = row_labels

        for row in range(len(self._camera_indices)):
            video_frame.grid_rowconfigure(row, weight=1)
        for column in range(max_column_index + 1):
            video_frame.grid_columnconfigure(column, weight=1)

    def _load_stereo_configuration(self) -> None:
        self._stereo_renderer = None
        self._stereo_calibration_size = None
        self._stereo_pair = None
        self._stereo_status = None

        if len(self._camera_indices) < 2:
            self._stereo_status = "Stereo-Ansicht benötigt zwei Kameras."
            return

        self._stereo_pair = (self._camera_indices[0], self._camera_indices[1])
        calibration_path = Path("stereo_calibration.json")
        if not calibration_path.exists():
            self._stereo_status = (
                f"Stereo-Ansicht deaktiviert: '{calibration_path.name}' fehlt."
            )
            return

        try:
            calibration = load_calibration(calibration_path)
        except Exception as exc:  # pragma: no cover - defensive for GUI usage
            self._stereo_status = f"Stereo-Ansicht deaktiviert: {exc}"
            return

        width, height = calibration.image_size
        self._stereo_calibration_size = (int(width), int(height))
        matcher = create_default_matcher()
        self._stereo_renderer = StereoReconstructor(calibration, matcher=matcher)

    def _create_image_panel(
        self,
        parent: ttk.Frame,
        title: str,
        row: int,
        column: int,
        default_text: str = "Kein Signal",
    ) -> ttk.Label:
        panel = ttk.Frame(parent)
        panel.grid(row=row, column=column, padx=5, pady=5, sticky="nsew")

        label_title = ttk.Label(panel, text=title, anchor="center")
        label_title.pack(fill=tk.X)

        image_label = ttk.Label(panel, text=default_text, anchor="center")
        image_label.pack(fill=tk.BOTH, expand=True)

        return image_label

    def start_cameras(self) -> None:
        if self._capture.is_running():
            return

        if not self._capture.start():
            self.status_var.set("Fehler beim Öffnen der Kameras.")
            self._capture.stop()
            return

        self.status_var.set("Live-Ansicht aktiv.")
        self._last_canny_timestamp = None

    def stop_cameras(self) -> None:
        self._capture.stop()
        self.status_var.set("Aufnahme gestoppt.")
        self._last_canny_timestamp = None

    def on_close(self) -> None:
        self.stop_cameras()
        self.root.destroy()

    def refresh_autofocus(self) -> None:
        if not self._capture.is_running():
            self.status_var.set("Autofokus nur bei aktiver Aufnahme möglich.")
            return

        if self._capture.refresh_autofocus():
            self.status_var.set("Autofokus wurde neu kalibriert.")
        else:
            self.status_var.set("Autofokus wird von den Kameras nicht unterstützt.")

    def save_frames(self) -> None:
        if not self._last_frames:
            self.status_var.set("Keine Bilder zum Speichern verfügbar.")
            return

        output_dir = Path("captures")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved = 0
        for index, views in self._last_frames.items():
            for view_name, image in views.items():
                if image is None:
                    continue
                filename = output_dir / f"kamera{index}_{view_name}_{timestamp}.png"
                if cv2.imwrite(str(filename), image):
                    saved += 1

        if saved:
            self.status_var.set(
                f"{saved} Bilder gespeichert im Ordner '{output_dir}'."
            )
        else:
            self.status_var.set("Speichern fehlgeschlagen.")

    def _update_loop(self) -> None:
        if self._capture.is_running():
            frames: Dict[int, np.ndarray] = {}
            canny_views: Dict[int, np.ndarray] = {}
            canny_refresh: List[int] = []
            missing_indices: List[int] = []

            now = time.perf_counter()
            update_canny = (
                self._last_canny_timestamp is None
                or (now - self._last_canny_timestamp) * 1000 >= self._canny_interval_ms
            )
            if update_canny:
                self._last_canny_timestamp = now

            for index in self._camera_indices:
                frame = self._capture.read(index)
                if frame is None:
                    missing_indices.append(index)
                    continue

                original_frame = frame.copy()
                frames[index] = original_frame

                self._last_frames.setdefault(index, {})
                self._last_frames[index]["original"] = original_frame.copy()

                stored_canny = self._last_frames[index].get("canny")
                refresh_canny = update_canny or stored_canny is None
                if refresh_canny:
                    inverted = invert_grayscale(original_frame)
                    stored_canny = canny_from_inverted(inverted)
                    self._last_frames[index]["canny"] = stored_canny.copy()
                if stored_canny is not None:
                    canny_views[index] = stored_canny
                    if refresh_canny:
                        canny_refresh.append(index)

            for index, original_frame in frames.items():
                self._update_image(
                    self._labels[index]["original"],
                    original_frame,
                    (index, "original"),
                )
            for index in canny_refresh:
                self._update_image(
                    self._labels[index]["canny"],
                    canny_views[index],
                    (index, "canny"),
                )

            if (
                self._stereo_renderer is not None
                and self._stereo_pair is not None
                and update_canny
            ):
                left_index, right_index = self._stereo_pair
                left_frame = frames.get(left_index)
                right_frame = frames.get(right_index)
                left_canny = canny_views.get(left_index)
                right_canny = canny_views.get(right_index)

                if (
                    left_frame is not None
                    and right_frame is not None
                    and left_canny is not None
                    and right_canny is not None
                    and "3d" in self._labels.get(left_index, {})
                ):
                    try:
                        result = self._stereo_renderer.compute(
                            left_frame,
                            right_frame,
                            masks=(left_canny, right_canny),
                        )
                    except Exception as exc:  # pragma: no cover - runtime safety
                        self.status_var.set(f"Stereo-Fehler: {exc}")
                        self._clear_stereo_view(left_index)
                    else:
                        depth_view = result.depth_colormap
                        self._last_frames[left_index]["3d"] = depth_view.copy()
                        self._update_image(
                            self._labels[left_index]["3d"],
                            depth_view,
                            (left_index, "3d"),
                        )
                elif self._stereo_pair[0] in self._labels:
                    self._clear_stereo_view(self._stereo_pair[0])

            for index in missing_indices:
                if index in self._labels:
                    for key in ("original", "canny"):
                        if key in self._labels[index]:
                            self._clear_image(self._labels[index][key], (index, key))
                self._last_frames.pop(index, None)

            if self._stereo_pair is not None:
                left_index, right_index = self._stereo_pair
                if (
                    left_index in missing_indices
                    or right_index in missing_indices
                    or left_index not in frames
                    or right_index not in frames
                ):
                    self._clear_stereo_view(left_index)

        self.root.after(self._live_interval_ms, self._update_loop)

    def _update_image(self, label: ttk.Label, frame, key: Tuple[int, str]) -> None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        success, encoded_image = cv2.imencode(".png", rgb_frame)
        if not success:
            return

        b64_data = base64.b64encode(encoded_image).decode("ascii")
        photo = tk.PhotoImage(data=b64_data)

        label.configure(image=photo, text="")
        self._photo_images[key] = photo

    def _clear_image(self, label: ttk.Label, key: Tuple[int, str]) -> None:
        label.configure(image="", text="Kein Signal")
        self._photo_images.pop(key, None)
        index, view_name = key
        if index in self._last_frames:
            self._last_frames[index].pop(view_name, None)
            if not self._last_frames[index]:
                self._last_frames.pop(index)

    def _clear_stereo_view(self, left_index: int) -> None:
        if left_index not in self._labels:
            return
        label = self._labels[left_index].get("3d")
        if label is None:
            return
        self._clear_image(label, (left_index, "3d"))
        message = (
            "Warte auf Stereo-Daten"
            if self._stereo_renderer is not None
            else (self._stereo_status or "Stereo deaktiviert")
        )
        label.configure(text=message)
