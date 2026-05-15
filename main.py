"""
main.py — Handwriting Recognition Digit Board
==============================================
Entry point for the software simulation.

Controls
--------
  Q          Quit
  T          Toggle threshold preview panel
  S          Save current frame as screenshot
  L          Toggle CSV prediction logging
  SPACE      Pause / resume

Usage
-----
  python main.py                        # uses default webcam + Keras .h5 model
  python main.py --camera 1             # secondary webcam
  python main.py --model model/mnist_model.tflite   # use TFLite backend
  python main.py --no-preview           # hide threshold panel

Architecture overview
---------------------
  WebcamCapture
      |
      v
  Preprocessor  (grayscale -> threshold -> find contours)
      |
      v  (for each contour)
  DigitClassifier  (CNN / TFLite inference)
      |
      v
  Renderer  (draw boxes, labels, FPS on frame)
      |
      v
  cv2.imshow  (display)  +  optional Flask stream
"""

import argparse
import sys
import os
import cv2

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.dirname(__file__))

from modules.preprocessor import (
    to_grayscale, apply_threshold,
    find_digit_contours, extract_roi, get_bounding_box,
)
from modules.classifier import DigitClassifier
from modules.renderer   import FPSCounter, draw_digit_box, draw_fps, \
                                draw_status_bar, draw_threshold_preview
from modules.logger     import log_prediction


# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL  = os.path.join(os.path.dirname(__file__), "model", "mnist_model.h5")
DEFAULT_CAMERA = 0
WINDOW_NAME    = "Handwriting Digit Board"
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Handwriting Recognition Digit Board")
    parser.add_argument("--camera",     type=int,   default=DEFAULT_CAMERA)
    parser.add_argument("--model",      type=str,   default=DEFAULT_MODEL)
    parser.add_argument("--no-preview", action="store_true",
                        help="Hide threshold preview panel")
    parser.add_argument("--log",        action="store_true",
                        help="Enable CSV prediction logging from start")
    return parser.parse_args()


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)   # CAP_DSHOW = faster on Windows
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)               # fallback (Linux / Mac)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera {index}.\n"
            "• Make sure your webcam is plugged in.\n"
            "• Try --camera 1 if you have multiple cameras.\n"
            "• Close other apps that may be using the camera."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def main():
    args = parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        print("  Run  python train_model.py  first to train and save the model.")
        sys.exit(1)

    print(f"[Init] Loading model from: {args.model}")
    classifier = DigitClassifier(args.model)

    # ── Open webcam ───────────────────────────────────────────────────────────
    print(f"[Init] Opening camera {args.camera}...")
    cap = open_camera(args.camera)

    fps_counter   = FPSCounter(window=20)
    show_preview  = not args.no_preview
    logging_on    = args.log
    paused        = False
    screenshot_n  = 0

    print("[Running] Press Q to quit, T to toggle threshold, SPACE to pause.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 800, 600)

    last_frame = None   # used when paused

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from camera.")
                break
            last_frame = frame.copy()
        else:
            frame = last_frame.copy()

        # ── Preprocessing ─────────────────────────────────────────────────────
        gray   = to_grayscale(frame)
        thresh = apply_threshold(gray)
        contours = find_digit_contours(thresh)

        # ── Inference per detected ROI ────────────────────────────────────────
        num_detections = 0
        for cnt in contours:
            roi = extract_roi(thresh, cnt, frame.shape)
            if roi is None:
                continue

            digit, confidence = classifier.predict(roi)
            is_confident = classifier.is_confident(confidence)

            x1, y1, x2, y2 = get_bounding_box(cnt, frame.shape)
            draw_digit_box(frame, x1, y1, x2, y2, digit, confidence, is_confident)
            num_detections += 1

            if logging_on:
                log_prediction(digit, confidence, is_confident)

        # ── UI overlays ───────────────────────────────────────────────────────
        fps = fps_counter.tick()
        draw_fps(frame, fps)
        if show_preview:
            draw_threshold_preview(frame, thresh)
        mode = "PAUSED" if paused else "LIVE"
        draw_status_bar(frame, num_detections, mode=mode)

        cv2.imshow(WINDOW_NAME, frame)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("t"):
            show_preview = not show_preview
        elif key == ord("l"):
            logging_on = not logging_on
            state = "ON" if logging_on else "OFF"
            print(f"[Logger] CSV logging {state}")
        elif key == ord("s"):
            screenshot_n += 1
            name = f"screenshot_{screenshot_n:03d}.png"
            cv2.imwrite(name, frame)
            print(f"[Screenshot] Saved {name}")
        elif key == 32:   # SPACE
            paused = not paused
            print(f"[{'PAUSED' if paused else 'RESUMED'}]")

    cap.release()
    cv2.destroyAllWindows()
    print("[Done] Bye!")


if __name__ == "__main__":
    main()
