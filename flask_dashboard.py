"""
flask_dashboard.py — Optional Web Dashboard
============================================
Streams the annotated webcam feed via MJPEG so you can view it in any
browser on the same network (useful for demo/presentation).

Run:
    python flask_dashboard.py

Then open:  http://localhost:5000

Architecture
------------
Main thread  →  background thread reads frames, runs inference, encodes JPEG
Flask route  →  serves MJPEG stream  (multipart/x-mixed-replace)
"""

import os
import sys
import threading
import time
import cv2

from flask import Flask, Response, render_template_string

sys.path.insert(0, os.path.dirname(__file__))
from modules.preprocessor import (to_grayscale, apply_threshold,
                                   find_digit_contours, extract_roi, get_bounding_box)
from modules.classifier   import DigitClassifier
from modules.renderer     import (FPSCounter, draw_digit_box, draw_fps,
                                  draw_status_bar, draw_threshold_preview)

MODEL_PATH  = os.path.join("model", "mnist_model.h5")
CAMERA_IDX  = 0

app = Flask(__name__)

# Shared state between inference thread and Flask thread
_lock         = threading.Lock()
_output_frame = None   # latest annotated JPEG bytes


# ── HTML page ─────────────────────────────────────────────────────────────────
INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Handwriting Digit Board</title>
  <style>
    body { background:#111; color:#eee; font-family:monospace; text-align:center; margin:0; }
    h1   { color:#0f0; margin-top:20px; }
    img  { border:3px solid #0f0; border-radius:8px; margin-top:16px; max-width:95vw; }
    p    { color:#888; font-size:0.85em; }
  </style>
</head>
<body>
  <h1>Handwriting Recognition Digit Board</h1>
  <img src="/video_feed" />
  <p>Live MJPEG stream &nbsp;|&nbsp; Refresh page if stream stalls</p>
</body>
</html>
"""


def inference_loop():
    """Background thread: capture -> process -> encode -> store in _output_frame."""
    global _output_frame

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run train_model.py first.")

    classifier  = DigitClassifier(MODEL_PATH)
    cap         = cv2.VideoCapture(CAMERA_IDX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_IDX)
    fps_counter = FPSCounter()

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        gray   = to_grayscale(frame)
        thresh = apply_threshold(gray)
        conts  = find_digit_contours(thresh)

        n = 0
        for cnt in conts:
            roi = extract_roi(thresh, cnt, frame.shape)
            if roi is None:
                continue
            digit, conf = classifier.predict(roi)
            x1, y1, x2, y2 = get_bounding_box(cnt, frame.shape)
            draw_digit_box(frame, x1, y1, x2, y2, digit, conf,
                           classifier.is_confident(conf))
            n += 1

        fps = fps_counter.tick()
        draw_fps(frame, fps)
        draw_threshold_preview(frame, thresh)
        draw_status_bar(frame, n, mode="STREAM")

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with _lock:
            _output_frame = jpeg.tobytes()


def generate():
    """Generator that yields MJPEG frames."""
    while True:
        with _lock:
            if _output_frame is None:
                time.sleep(0.05)
                continue
            frame = _output_frame
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)   # ~30 fps ceiling


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()
    print("Dashboard running at  http://localhost:5000")
    print("Press Ctrl+C to stop.")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
