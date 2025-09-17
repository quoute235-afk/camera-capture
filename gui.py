"""Tkinter GUI for displaying multiple camera feeds and filters."""
from __future__ import annotations

import base64
from typing import Dict, Iterable, List, Tuple

import cv2
import tkinter as tk
from tkinter import ttk

from camera_capture import MultiCameraCapture
from filters import invert


class DualInvertApp:
    """GUI application showing original and inverted views for cameras."""

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

        status_label = ttk.Label(control_frame, textvariable=self.status_var, wraplength=140)
        status_label.pack(fill=tk.X, pady=(15, 0))

        video_frame = ttk.Frame(main_frame, padding=10)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._labels: Dict[int, Dict[str, ttk.Label]] = {}

        for row, index in enumerate(self._camera_indices):
            self._labels[index] = {
                "original": self._create_image_panel(
                    video_frame, f"Kamera {row + 1} - Original", row, 0
                ),
                "inverted": self._create_image_panel(
                    video_frame, f"Kamera {row + 1} - Invertiert", row, 1
                ),
            }

        for row in range(len(self._camera_indices)):
            video_frame.grid_rowconfigure(row, weight=1)
        for column in range(2):
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

    def _update_loop(self) -> None:
        if self._capture.is_running():
            for index in self._camera_indices:
                frame = self._capture.read(index)
                if frame is not None:
                    self._update_image(self._labels[index]["original"], frame, (index, "original"))
                    inverted = invert(frame)
                    self._update_image(self._labels[index]["inverted"], inverted, (index, "inverted"))
                else:
                    self._clear_image(self._labels[index]["original"], (index, "original"))
                    self._clear_image(self._labels[index]["inverted"], (index, "inverted"))

        self.root.after(33, self._update_loop)

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
