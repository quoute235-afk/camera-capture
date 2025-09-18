"""Camera capture management for multi-camera setups."""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import cv2


class MultiCameraCapture:
    """Utility class to manage multiple OpenCV camera capture devices.

    Parameters
    ----------
    camera_indices:
        Iterable of integer indices representing the cameras to open.
    frame_size:
        Optional ``(width, height)`` tuple. When provided, captured frames
        are resized to this resolution before being returned.
    backend:
        Optional OpenCV backend identifier passed to ``cv2.VideoCapture``.
    """

    def __init__(
        self,
        camera_indices: Iterable[int],
        frame_size: Optional[Tuple[int, int]] = (640, 360),
        backend: int = cv2.CAP_ANY,
    ) -> None:
        self._indices = list(camera_indices)
        self._frame_size = frame_size
        self._backend = backend
        self._captures: Dict[int, cv2.VideoCapture] = {}

    def start(self) -> bool:
        """Open all configured camera indices."""

        if self.is_running():
            return True

        opened: Dict[int, cv2.VideoCapture] = {}
        success = True
        for index in self._indices:
            capture = cv2.VideoCapture(index, self._backend)
            if not capture.isOpened():
                capture.release()
                success = False
                break
            opened[index] = capture

        if not success:
            for capture in opened.values():
                capture.release()
            return False

        self._captures = opened
        return True

    def stop(self) -> None:
        """Release all opened camera handles."""

        for capture in self._captures.values():
            capture.release()
        self._captures.clear()

    def is_running(self) -> bool:
        """Return ``True`` when at least one camera is opened."""

        return bool(self._captures)

    def read(self, index: int):
        """Read a frame from the camera identified by ``index``."""

        capture = self._captures.get(index)
        if capture is None or not capture.isOpened():
            return None

        success, frame = capture.read()
        if not success or frame is None:
            return None

        if self._frame_size is not None:
            frame = cv2.resize(frame, self._frame_size)
        return frame

    def iter_indices(self) -> Iterable[int]:
        """Yield the configured camera indices in the initial order."""

        return iter(self._indices)

    def refresh_autofocus(self) -> bool:
        """Attempt to retrigger auto focus on all opened cameras.

        Returns
        -------
        bool
            ``True`` if at least one capture acknowledged the autofocus
            command, otherwise ``False``.
        """

        if not self.is_running():
            return False

        autofocus_prop = getattr(cv2, "CAP_PROP_AUTOFOCUS", None)
        if autofocus_prop is None:
            return False

        refreshed = False
        for capture in self._captures.values():
            if capture is None or not capture.isOpened():
                continue

            # Toggle the autofocus flag to encourage the camera to refocus.
            if capture.set(autofocus_prop, 0) and capture.set(autofocus_prop, 1):
                refreshed = True
                continue

            # Some devices only allow enabling autofocus without toggling.
            if capture.set(autofocus_prop, 1):
                refreshed = True

        return refreshed
