"""
Preprocessor Module
-------------------
Converts a raw BGR webcam frame or cropped ROI into the 28x28 grayscale
normalized array that the MNIST-trained model expects.

Pipeline:
  BGR frame  ->  grayscale  ->  blur (noise removal)  ->  adaptive threshold
  ->  find contours  ->  crop each contour  ->  resize to 28x28
  ->  invert (white digit on black)  ->  normalize to [0,1]
  ->  reshape to (1, 28, 28, 1)  for the CNN
"""

import cv2
import numpy as np


# ── tuneable constants ────────────────────────────────────────────────────────
MIN_CONTOUR_AREA   = 500    # ignore tiny specks (noise)
MAX_CONTOUR_AREA   = 80_000 # ignore the whole-frame contour
PADDING            = 10     # pixels of breathing room around each digit
TARGET_SIZE        = (28, 28)
# ─────────────────────────────────────────────────────────────────────────────


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR webcam frame to grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def apply_threshold(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive thresholding works much better than a fixed threshold under
    varying lighting conditions (desk lamps, sunlight, shadows).

    Result is a binary image: digits appear WHITE on a BLACK background
    after the inversion step below.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,   # invert so digit is white
        blockSize=11,
        C=2,
    )
    return thresh


def find_digit_contours(thresh: np.ndarray):
    """
    Return contours that are plausibly handwritten digits.
    Filters out noise (too small) and the whole-page contour (too large).
    """
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA:
            digit_contours.append(cnt)
    # Sort left-to-right so multi-digit numbers are read in order
    digit_contours.sort(key=lambda c: cv2.boundingRect(c)[0])
    return digit_contours


def extract_roi(thresh: np.ndarray, contour, frame_shape) -> np.ndarray | None:
    """
    Crop the thresholded image to the bounding box of one contour,
    add padding, resize to 28×28, then normalize pixel values to [0, 1].
    Returns None if the crop ends up empty.
    """
    h, w = frame_shape[:2]
    x, y, cw, ch = cv2.boundingRect(contour)

    # Add padding without going outside frame boundaries
    x1 = max(0, x - PADDING)
    y1 = max(0, y - PADDING)
    x2 = min(w, x + cw + PADDING)
    y2 = min(h, y + ch + PADDING)

    roi = thresh[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    roi_resized = cv2.resize(roi, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    roi_normalized = roi_resized.astype(np.float32) / 255.0
    # CNN expects shape (1, 28, 28, 1)
    return roi_normalized.reshape(1, 28, 28, 1)


def get_bounding_box(contour, frame_shape, padding: int = PADDING):
    """Return padded bounding box (x1, y1, x2, y2) clamped to frame."""
    h, w = frame_shape[:2]
    x, y, cw, ch = cv2.boundingRect(contour)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + cw + padding)
    y2 = min(h, y + ch + padding)
    return x1, y1, x2, y2
