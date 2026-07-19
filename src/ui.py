"""UI / overlay.

Display-side code only: mouse target selection on frame 0, and the per-frame
overlay (bounding box, motion trajectory, state banner, fps counter).
Nothing here influences tracking - overlays are drawn on a copy of the data
the pipeline already produced.
"""

from __future__ import annotations

import cv2

from .tracker import Box, pixel_to_box

# BGR colors per displayed state.
STATE_COLORS = {
    "TRACKING": (0, 200, 0),
    "LOST": (0, 0, 255),
    "RE-ACQUIRED": (0, 200, 200),
}
TRAIL_COLOR = (0, 220, 220)
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def select_pixel(window: str, frame, box_size: int) -> tuple[int, int] | None:
    """Interactive target selection on frame 0.

    Left-click marks the target pixel (a preview of the init box is drawn),
    another click moves it, ENTER/SPACE confirms, ESC/q aborts -> None.
    """
    picked: dict = {"pt": None}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            picked["pt"] = (x, y)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    h, w = frame.shape[:2]
    try:
        while True:
            disp = frame.copy()
            cv2.putText(disp, "click target | ENTER/SPACE confirm | ESC quit",
                        (20, 40), FONT, 1.0, TEXT_COLOR, 2)
            if picked["pt"] is not None:
                px, py = picked["pt"]
                bx, by, bw, bh = pixel_to_box(px, py, box_size, w, h)
                cv2.rectangle(disp, (bx, by), (bx + bw, by + bh),
                              STATE_COLORS["TRACKING"], 2)
                cv2.drawMarker(disp, (px, py), STATE_COLORS["TRACKING"],
                               cv2.MARKER_CROSS, 20, 2)
            cv2.imshow(window, disp)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 32) and picked["pt"] is not None:  # ENTER / SPACE
                return picked["pt"]
            if key in (27, ord("q")):  # ESC / q
                return None
            # user closed the window with the X button -> exit cleanly
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return None
    finally:
        cv2.setMouseCallback(window, lambda *a: None)


def draw_overlay(frame, box: Box | None, trail: list[tuple[int, int]],
                 state: str, fps: float, tracker_name: str,
                 confidence: float | None = None) -> None:
    """Draw box, trajectory, state banner and fps counter onto `frame` in place.

    `confidence` is the backend's raw score (PSR for our MOSSE, 0/1 for the
    OpenCV backends) - shown so loss events are visible live.
    """
    color = STATE_COLORS.get(state, TEXT_COLOR)
    if box is not None:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    for a, b in zip(trail, trail[1:]):
        cv2.line(frame, a, b, TRAIL_COLOR, 2)
    if trail:
        cv2.circle(frame, trail[-1], 4, TRAIL_COLOR, -1)

    # state banner (top-left) and fps/tracker/confidence info (top-right)
    cv2.rectangle(frame, (10, 10), (330, 60), (0, 0, 0), -1)
    cv2.putText(frame, state, (20, 47), FONT, 1.2, color, 3)
    info = f"{tracker_name}  {fps:5.1f} fps"
    if confidence is not None:
        info += f"  conf {confidence:5.1f}"
    cv2.putText(frame, info, (max(10, frame.shape[1] - 520), 47), FONT, 1.0,
                TEXT_COLOR, 2)
