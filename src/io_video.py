"""Video input.

Thin streaming wrapper around cv2.VideoCapture: opens a local path or URL,
iterates frames, exposes stream metadata. No caching - the main loop owns
all timing decisions.
"""

from __future__ import annotations

import contextlib
import os

import cv2


@contextlib.contextmanager
def _silenced_stderr():
    """Mute stderr (fd 2) for the duration of the block.

    OpenCV's capture backends print their own failure lines straight to the
    C-level stderr (bypassing cv2.utils.logging), so opening a bad path
    would show their chatter next to our clean error. Scoped tightly to the
    VideoCapture construction only.
    """
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


class VideoSource:
    """Streaming reader for a video file or URL."""

    def __init__(self, source: str):
        with _silenced_stderr():
            self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise IOError(f"cannot open video source: {source}")
        self.source = source

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self):
        """Return the next frame (BGR ndarray) or None at end of stream."""
        ok, frame = self._cap.read()
        return frame if ok else None

    def frames(self):
        """Yield frames until the stream ends."""
        while (frame := self.read()) is not None:
            yield frame

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
