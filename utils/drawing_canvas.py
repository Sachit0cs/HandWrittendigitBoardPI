"""
drawing_canvas.py — Mouse Drawing Mode (Bonus Feature)
=======================================================
Instead of holding paper in front of the webcam, use your mouse to draw
a digit on a virtual black canvas, then classify it.

Run:
    python utils/drawing_canvas.py

Controls
--------
  Draw     Left mouse button
  Clear    C key
  Predict  SPACE or P key
  Quit     Q key
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from modules.classifier import DigitClassifier
from modules.preprocessor import TARGET_SIZE

CANVAS_SIZE  = 400
BRUSH_RADIUS = 16
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "..", "model", "mnist_model.h5")

canvas      = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
drawing     = False
last_pos    = None
result_text = "Draw a digit, then press SPACE"


def preprocess_canvas(canvas: np.ndarray) -> np.ndarray:
    """Resize the canvas drawing to 28x28 and normalize for the CNN."""
    resized    = cv2.resize(canvas, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, 28, 28, 1)


def mouse_callback(event, x, y, flags, param):
    global drawing, last_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing  = True
        last_pos = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(canvas, last_pos, (x, y), 255, BRUSH_RADIUS * 2)
        last_pos = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing  = False
        last_pos = None


def main():
    global canvas, result_text

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("  Run  python train_model.py  first.")
        sys.exit(1)

    clf = DigitClassifier(MODEL_PATH)

    win = "Drawing Canvas — Press SPACE to predict, C to clear, Q to quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse_callback)

    while True:
        display = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        # Result overlay at the bottom
        cv2.rectangle(display, (0, CANVAS_SIZE - 50), (CANVAS_SIZE, CANVAS_SIZE),
                      (0, 0, 0), -1)
        cv2.putText(display, result_text, (10, CANVAS_SIZE - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        cv2.imshow(win, display)
        key = cv2.waitKey(10) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas[:] = 0
            result_text = "Canvas cleared. Draw again."
        elif key in (32, ord("p")):   # SPACE or P
            if canvas.max() == 0:
                result_text = "Canvas is empty — draw something first!"
            else:
                roi = preprocess_canvas(canvas)
                digit, conf = clf.predict(roi)
                result_text = f"Prediction: {digit}   Confidence: {conf*100:.1f}%"
                print(result_text)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
