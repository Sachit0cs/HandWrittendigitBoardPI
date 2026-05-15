"""
Renderer Module
---------------
All OpenCV drawing / UI overlay logic lives here so the main loop stays clean.

Draws on the live webcam frame:
  • Green bounding box around each detected digit ROI
  • Predicted digit + confidence % above the box
  • Color-coded confidence bar (green = high, yellow = medium, red = low)
  • FPS counter (top-left)
  • Status bar (bottom)
"""

import cv2
import numpy as np
import time


# ── colour palette (BGR) ─────────────────────────────────────────────────────
GREEN  = (0, 220, 0)
YELLOW = (0, 220, 220)
RED    = (0, 60, 220)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
BLUE   = (220, 100, 0)
GRAY   = (180, 180, 180)
# ─────────────────────────────────────────────────────────────────────────────

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS  = 2


def _confidence_color(conf: float):
    if conf >= 0.80:
        return GREEN
    if conf >= 0.60:
        return YELLOW
    return RED


def draw_digit_box(frame: np.ndarray, x1, y1, x2, y2,
                   digit: int, confidence: float, is_confident: bool):
    """Draw one detection result on frame (modifies in-place)."""
    color = _confidence_color(confidence)

    # Bounding rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS)

    # Label text
    label = f"{digit}  ({confidence*100:.1f}%)"
    if not is_confident:
        label = f"? ({confidence*100:.1f}%)"

    # Background pill behind text so it is readable on any background
    (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
    cv2.rectangle(frame,
                  (x1, y1 - th - baseline - 6),
                  (x1 + tw + 4, y1),
                  color, -1)
    cv2.putText(frame, label,
                (x1 + 2, y1 - baseline - 2),
                FONT, FONT_SCALE, BLACK, THICKNESS)

    # Thin confidence bar below the bounding box
    bar_w = x2 - x1
    bar_filled = int(bar_w * confidence)
    cv2.rectangle(frame, (x1, y2 + 2), (x1 + bar_w, y2 + 8), GRAY, -1)
    cv2.rectangle(frame, (x1, y2 + 2), (x1 + bar_filled, y2 + 8), color, -1)


def draw_fps(frame: np.ndarray, fps: float):
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (10, 30), FONT, FONT_SCALE, GREEN, THICKNESS)


def draw_status_bar(frame: np.ndarray, num_detections: int, mode: str = "LIVE"):
    h, w = frame.shape[:2]
    bar_h = 28
    cv2.rectangle(frame, (0, h - bar_h), (w, h), BLACK, -1)
    status = f"Mode: {mode}  |  Digits detected: {num_detections}  |  Press Q to quit"
    cv2.putText(frame, status,
                (8, h - 8), FONT, 0.5, WHITE, 1)


def draw_threshold_preview(frame: np.ndarray, thresh: np.ndarray,
                           position: str = "top-right", size: int = 180):
    """
    Embed a small thumbnail of the thresholded image so the user can see
    what the model is actually looking at.
    """
    h, w = frame.shape[:2]
    preview = cv2.resize(thresh, (size, size))
    preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

    if position == "top-right":
        x_off = w - size - 10
        y_off = 10
    else:
        x_off, y_off = 10, 50

    frame[y_off:y_off + size, x_off:x_off + size] = preview_bgr
    cv2.rectangle(frame,
                  (x_off, y_off),
                  (x_off + size, y_off + size),
                  BLUE, 1)
    cv2.putText(frame, "Threshold view",
                (x_off + 2, y_off + size + 15),
                FONT, 0.4, BLUE, 1)


class FPSCounter:
    """Rolling-average FPS counter (avoids jitter from single-frame timing)."""
    def __init__(self, window: int = 15):
        self._times = []
        self._window = window

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0
