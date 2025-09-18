"""Tkinter GUI for displaying multiple camera feeds and filters."""
from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

from camera_capture import MultiCameraCapture
from filters import canny_from_inverted, invert_grayscale


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
        self._capture = MultiCameraCapture(self._camera_indices)
        self._photo_images: Dict[Tuple[int, str], tk.PhotoImage] = {}
        self._last_frames: Dict[int, Dict[str, np.ndarray]] = {}

        self.status_var = tk.StringVar(value="Inaktiv. Bitte 'Start' drücken.")

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

        panel_config = [
            ("original", "Original"),
            ("inverted", "Invertiert (Graustufen)"),
            ("canny", "Canny (Invertiert)"),
        ]
        max_column_index = len(panel_config) - 1

        for row, index in enumerate(self._camera_indices):
            row_labels: Dict[str, ttk.Label] = {}
            for column, (key, title_suffix) in enumerate(panel_config):
                row_labels[key] = self._create_image_panel(
                    video_frame,
                    f"Kamera {row + 1} - {title_suffix}",
                    row,
                    column,
                )
            self._labels[index] = row_labels

        for row in range(len(self._camera_indices)):
            video_frame.grid_rowconfigure(row, weight=1)
        for column in range(max_column_index + 1):
            video_frame.grid_columnconfigure(column, weight=1)

    def _create_image_panel(
        self,
        parent: ttk.Frame,
        title: str,
        row: int,
        column: int,
    ) -> ttk.Label:
        panel = ttk.Frame(parent)
        panel.grid(row=row, column=column, padx=5, pady=5, sticky="nsew")

        label_title = ttk.Label(panel, text=title, anchor="center")
        label_title.pack(fill=tk.X)

        image_label = ttk.Label(panel, text="Kein Signal", anchor="center")
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

    def stop_cameras(self) -> None:
        self._capture.stop()
        self.status_var.set("Aufnahme gestoppt.")

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
            inverted_views: Dict[int, np.ndarray] = {}
            canny_views: Dict[int, np.ndarray] = {}
            missing_indices: List[int] = []

            for index in self._camera_indices:
                frame = self._capture.read(index)
                if frame is None:
                    missing_indices.append(index)
                    continue

                original_frame = frame.copy()
                inverted = invert_grayscale(original_frame)
                canny_view = canny_from_inverted(inverted)

                frames[index] = original_frame
                inverted_views[index] = inverted
                canny_views[index] = canny_view

                self._last_frames.setdefault(index, {})
                self._last_frames[index]["original"] = original_frame.copy()
                self._last_frames[index]["inverted"] = inverted.copy()
                self._last_frames[index]["canny"] = canny_view.copy()

            for index, original_frame in frames.items():
                self._update_image(
                    self._labels[index]["original"],
                    original_frame,
                    (index, "original"),
                )
                self._update_image(
                    self._labels[index]["inverted"],
                    inverted_views[index],
                    (index, "inverted"),
                )
                self._update_image(
                    self._labels[index]["canny"],
                    canny_views[index],
                    (index, "canny"),
                )

            for index in missing_indices:
                if index in self._labels:
                    for key in ("original", "inverted", "canny"):
                        if key in self._labels[index]:
                            self._clear_image(self._labels[index][key], (index, key))
                self._last_frames.pop(index, None)

        self.root.after(100, self._update_loop)

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
