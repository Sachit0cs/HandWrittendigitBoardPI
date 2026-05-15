# Handwriting Recognition Digit Board
### Project 5 — Software Simulation (Webcam Only)

A real-time handwritten digit recognition system built with Python, OpenCV, and TensorFlow.
Hold a piece of paper with a hand-drawn digit in front of your webcam — the system detects,
extracts, and classifies it in real time.

---

## Table of Contents
1. [Beginner Concepts Explained](#1-beginner-concepts-explained)
2. [Project Folder Structure](#2-project-folder-structure)
3. [Installation](#3-installation)
4. [How to Run](#4-how-to-run)
5. [How It Works — Image Pipeline](#5-how-it-works--image-pipeline)
6. [Confidence Scores Explained](#6-confidence-scores-explained)
7. [Controls & UI Guide](#7-controls--ui-guide)
8. [Optional Features](#8-optional-features)
9. [Troubleshooting](#9-troubleshooting)
10. [Resume Bullet Points](#10-resume-bullet-points)
11. [Future Raspberry Pi Migration](#11-future-raspberry-pi-migration)

---

## 1. Beginner Concepts Explained

### What is MNIST?
MNIST (Modified National Institute of Standards and Technology) is a free dataset of
**70,000 grayscale images** of handwritten digits (0–9), each 28×28 pixels.
- 60,000 images for training the model
- 10,000 images for testing accuracy
- It is the "Hello World" benchmark of image classification in machine learning.

### What is a CNN (Convolutional Neural Network)?
A CNN learns to recognize patterns in images by applying small sliding filters (kernels):
```
Raw image  →  [Conv Layer: detect edges]  →  [Conv Layer: detect shapes]
           →  [Dense Layer: classify]      →  Digit prediction (0-9)
```
- **Conv layers** scan the image for strokes and curves
- **MaxPooling** shrinks the image to focus on dominant features
- **Dense (Fully Connected) layers** combine features → produce a digit prediction
- Our model reaches ~99% accuracy on the MNIST test set in 5 training epochs

### What is TensorFlow Lite?
TensorFlow Lite (TFLite) is a compressed, optimized version of a full TF model:
- `.h5` model = full size, used on desktop/laptop (this project's default)
- `.tflite` model = small & fast, used on Raspberry Pi / microcontrollers
- Both formats are saved by `train_model.py` — same accuracy, different file sizes

### What is ROI Extraction?
ROI = Region of Interest.  The step where we crop the part of the image that contains
a digit so the model only sees the digit, not the background noise.

### What is Thresholding?
Converts a grayscale image to pure black-and-white:
- Pixels darker than a threshold → 0 (black)
- Pixels brighter → 255 (white)
We use **adaptive thresholding** which adjusts the threshold based on local pixel
neighborhoods — essential for handling uneven lighting.

### Image Preprocessing Pipeline
```
Webcam BGR frame
        ↓
Grayscale conversion  (3 channels → 1 channel)
        ↓
Gaussian blur         (removes camera noise)
        ↓
Adaptive threshold    (creates black/white binary image)
        ↓
Find contours         (locate connected white regions = digit candidates)
        ↓
Filter contours       (by area — remove noise and background)
        ↓
Crop ROI + padding    (isolate each digit)
        ↓
Resize to 28×28       (match MNIST training size)
        ↓
Normalize [0, 1]      (pixel values 0–255 → 0.0–1.0)
        ↓
Reshape (1, 28, 28, 1) → CNN input
        ↓
Softmax probabilities [10 values] → argmax → predicted digit
```

### Real-Time Frame Processing
The main loop runs at ~20-30 FPS:
1. Read one frame from webcam
2. Preprocess the entire frame
3. Find all digit-shaped contours
4. For each contour: extract ROI → classify → draw result
5. Display annotated frame
6. Repeat

---

## 2. Project Folder Structure

```
HandwritingDigitBoard/
│
├── main.py                   ← Main application (webcam + real-time detection)
├── train_model.py            ← Train CNN on MNIST, saves .h5 and .tflite
├── flask_dashboard.py        ← Optional web dashboard (MJPEG stream)
├── install.bat               ← Windows one-click installer
├── requirements.txt          ← Python packages
│
├── model/
│   ├── mnist_model.h5        ← (generated) Keras model
│   └── mnist_model.tflite   ← (generated) TFLite model for Pi
│
├── modules/
│   ├── preprocessor.py       ← Grayscale, threshold, contour detection, ROI
│   ├── classifier.py         ← TF/TFLite model loading and inference
│   ├── renderer.py           ← OpenCV drawing: boxes, labels, FPS, status bar
│   └── logger.py             ← CSV prediction logging
│
├── utils/
│   └── drawing_canvas.py     ← Mouse drawing mode (no webcam needed)
│
├── logs/
│   └── predictions.csv       ← (generated) Prediction history log
│
└── README.md
```

---

## 3. Installation

### Prerequisites
- Python 3.9 or newer
- A working webcam
- ~500 MB free disk space (for TensorFlow)

### Option A — Double-click installer (Windows)
```
Double-click  install.bat
```

### Option B — Manual terminal install
```bash
pip install opencv-python numpy tensorflow Flask Pillow
```

### Verify install
```bash
python -c "import cv2, numpy, tensorflow; print('All OK')"
```

---

## 4. How to Run

### Step 1 — Train the model (run once, ~2-3 minutes)
```bash
python train_model.py
```
This downloads MNIST (~11 MB), trains for 5 epochs, and saves:
- `model/mnist_model.h5`
- `model/mnist_model.tflite`

### Step 2 — Start the live digit board
```bash
python main.py
```

### Optional flags
```bash
python main.py --camera 1           # if your webcam index is 1
python main.py --model model/mnist_model.tflite   # use TFLite backend
python main.py --no-preview         # hide the threshold thumbnail
python main.py --log                # enable CSV logging from startup
```

### Mouse drawing mode (no paper needed)
```bash
python utils/drawing_canvas.py
```

### Web dashboard
```bash
python flask_dashboard.py
# Then open: http://localhost:5000
```

---

## 5. How It Works — Image Pipeline

### Grayscale Conversion
```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```
Reduces 3-channel BGR to 1-channel — simpler and faster for thresholding.

### Gaussian Blur
```python
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
```
Smooths out camera grain/noise before thresholding to reduce false detections.

### Adaptive Thresholding
```python
thresh = cv2.adaptiveThreshold(blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            blockSize=11, C=2)
```
Unlike simple thresholding (one global value), adaptive threshold calculates
a local threshold for each pixel's neighborhood — crucial for uneven lighting.
`THRESH_BINARY_INV` inverts so the digit (dark ink) becomes white.

### Contour Detection
```python
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```
Finds the outlines of all connected white regions (= candidate digits).
Filters by area (500–80000 px²) to remove noise specks and full-frame background.

### CNN Inference
The model outputs a **softmax** vector of 10 probabilities (one per digit class).
```
Input: (1, 28, 28, 1)  →  CNN  →  [0.01, 0.02, 0.95, 0.01, ...]
                                       0     1     2     3  ...
                          Prediction: 2  |  Confidence: 95%
```

---

## 6. Confidence Scores Explained

### What is Softmax?
The CNN's final layer uses **softmax activation** which converts raw numbers into
probabilities that sum to 1.0:
```
Raw:      [2.1,  0.3,  8.4,  0.1,  ...]
Softmax:  [0.01, 0.01, 0.95, 0.00, ...]   ← sums to 1.0
```
The highest value is the prediction; its value IS the confidence.

### Confidence thresholds (configurable in `classifier.py`)
| Confidence | Color  | Meaning                        |
|-----------|--------|--------------------------------|
| ≥ 80%     | Green  | High confidence                |
| 60–79%    | Yellow | Moderate — prediction shown    |
| < 60%     | Red    | Uncertain — shows `?` symbol   |

### Handling uncertain predictions
When confidence < 60%, the label shows `? (XX%)` instead of the digit.
This prevents the system from displaying misleading results for:
- Partial digits at the frame edge
- Non-digit objects
- Overlapping/touching digits

---

## 7. Controls & UI Guide

| Key     | Action                              |
|---------|-------------------------------------|
| Q       | Quit the application                |
| T       | Toggle threshold preview panel      |
| S       | Save current frame as PNG           |
| L       | Toggle CSV prediction logging       |
| SPACE   | Pause / resume video                |

### UI Overlay Explained
```
┌─────────────────────────────────────────┐
│ FPS: 28.4         ┌──────────────────┐  │
│                   │  Threshold view  │  │
│  ┌──────┐         │  (binary image)  │  │
│  │  3   │ 97.2%   └──────────────────┘  │
│  └──────┘                               │
│  ████████░░  ← confidence bar          │
│─────────────────────────────────────────│
│ Mode: LIVE  |  Digits detected: 1  | Q │
└─────────────────────────────────────────┘
```

---

## 8. Optional Features

### Flask Web Dashboard (`flask_dashboard.py`)
- Streams annotated video as MJPEG to `http://localhost:5000`
- Works on the same network — open on your phone or another computer
- Uses a background thread for inference, main thread for Flask

### Mouse Drawing Canvas (`utils/drawing_canvas.py`)
- Black canvas, draw with left mouse button
- Press SPACE to classify your drawing
- Press C to clear and draw again
- Perfect for testing the model without holding paper to the webcam

### CSV Prediction Logging (`modules/logger.py`)
- Press L during live run to toggle on/off
- Saves to `logs/predictions.csv`
- Columns: `timestamp, predicted_digit, confidence, is_confident`

---

## 9. Troubleshooting

### Webcam not opening
```
RuntimeError: Cannot open camera 0
```
- Unplug and replug the webcam
- Try `--camera 1` (if you have multiple cameras, e.g., a laptop + USB webcam)
- Close video conferencing apps (Zoom, Teams) that lock the camera
- On Windows: check Camera privacy settings → allow desktop apps

### TensorFlow installation issues
```bash
# If pip install tensorflow fails, try the CPU-only version (lighter):
pip install tensorflow-cpu

# Python 3.12 may have issues — Python 3.10 or 3.11 recommended
python --version
```

### Poor digit detection (too many / too few boxes)
Adjust these constants in `modules/preprocessor.py`:
```python
MIN_CONTOUR_AREA = 500    # increase if noise is detected as digits
MAX_CONTOUR_AREA = 80000  # decrease if whole hand is detected as a digit
```

### Incorrect predictions
- Write digits **larger** — at least 4-5 cm tall on paper
- Use a **dark pen or marker** (light pencil reflects poorly)
- Keep the paper **flat** (curved paper warps the digit shape)
- Improve lighting — avoid strong shadows on the paper
- The model was trained on MNIST style digits — write them upright, not cursive

### Low FPS (< 10 fps)
- Switch to the TFLite model: `python main.py --model model/mnist_model.tflite`
- Reduce webcam resolution (edit `main.py` CAP_PROP_FRAME_WIDTH to 320)
- Close other heavy applications

### Lighting problems
- Avoid backlighting (don't sit with a bright window behind the paper)
- Use a desk lamp angled from the side
- The threshold preview panel (press T) shows exactly what the model sees —
  if the digits aren't clearly white on black there, adjust your lighting

---

## 10. Resume Bullet Points

> Copy-paste these directly into your resume / LinkedIn.

- **Engineered a real-time handwritten digit recognition system** using Python, OpenCV, and TensorFlow, achieving ~99% classification accuracy on MNIST through a custom 2-block CNN architecture with dropout regularization.

- **Designed a modular image processing pipeline** (grayscale → adaptive thresholding → contour detection → ROI extraction → 28×28 normalization) that robustly detects and isolates handwritten digits from live webcam frames at 20–30 FPS under varying lighting conditions.

- **Deployed dual inference backends** (full Keras `.h5` for development and TensorFlow Lite `.tflite` for embedded deployment), demonstrating cross-platform software architecture skills and awareness of Raspberry Pi hardware constraints.

- **Built supplementary features** including a Flask MJPEG web dashboard for network-accessible streaming, a mouse-driven virtual drawing canvas for model validation, softmax confidence thresholding for uncertain-prediction suppression, and a CSV prediction logger — producing a portfolio-ready, production-structured mini-project.

---

## 11. Future Raspberry Pi Migration

When you get your Raspberry Pi hardware, here are the exact changes needed:

### A. Replace webcam with Pi Camera (picamera2)

**Current (webcam):**
```python
# main.py
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
ret, frame = cap.read()
```

**Pi Camera replacement:**
```python
# Install: pip install picamera2
from picamera2 import Picamera2
import numpy as np

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
))
picam2.start()

# In loop:
frame = picam2.capture_array()          # returns numpy BGR array
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # picamera2 gives RGB
```

### B. Add OLED Display (SSD1306 via I2C)

```bash
pip install adafruit-circuitpython-ssd1306 Pillow
```

```python
import board
import busio
import adafruit_ssd1306
from PIL import Image, ImageDraw, ImageFont

i2c    = busio.I2C(board.SCL, board.SDA)
oled   = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c)

def show_on_oled(digit: int, confidence: float):
    img  = Image.new("1", (128, 64))
    draw = ImageDraw.Draw(img)
    draw.text((5,  5), f"Digit: {digit}",        fill=255)
    draw.text((5, 30), f"Conf:  {confidence*100:.0f}%", fill=255)
    oled.image(img)
    oled.show()
```

Call `show_on_oled(digit, confidence)` inside the main loop after each prediction.

### C. Add TTS Speaker Output (pyttsx3 or espeak)

```bash
# On Raspberry Pi OS:
sudo apt install espeak
pip install pyttsx3
```

```python
import pyttsx3
engine = pyttsx3.init()
engine.setProperty("rate", 150)

def speak_digit(digit: int):
    engine.say(f"Digit {digit}")
    engine.runAndWait()
```

Only speak when the prediction changes to avoid repetitive audio:
```python
last_spoken = -1
if digit != last_spoken and is_confident:
    speak_digit(digit)
    last_spoken = digit
```

### D. Deploy TFLite on Raspberry Pi Zero W

The `.tflite` model already exists — just use the TFLite backend:
```bash
# Raspberry Pi: install the lightweight runtime (not full TF)
pip install tflite-runtime

# Run with TFLite model
python main.py --model model/mnist_model.tflite
```

The `DigitClassifier` class in `modules/classifier.py` already handles both
`tensorflow` and `tflite-runtime` backends — no other code changes needed.

### Pi Zero W Performance Tips
- Use `--no-preview` flag to skip the threshold overlay (saves ~5ms/frame)
- Resize webcam/camera to 320×240: set `CAP_PROP_FRAME_WIDTH = 320`
- Consider running inference every 2nd frame to halve CPU load
- Enable GPU memory split: `gpu_mem=128` in `/boot/config.txt`

---

*Built by Sachit Jain — May 2026*
#   H a n d W r i t t e n d i g i t B o a r d P I  
 