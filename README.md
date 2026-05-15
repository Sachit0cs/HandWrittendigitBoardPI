# Handwriting Recognition Digit Board

> Real-time handwritten digit recognition using Python, OpenCV, and TensorFlow.

A real-time computer vision project that detects handwritten digits from a webcam feed and classifies them using a CNN trained on the MNIST dataset.

Simply hold a paper with a handwritten digit in front of the webcam, and the system detects, extracts, and predicts the digit instantly.

---

# Features

- Real-time handwritten digit recognition
- CNN trained on MNIST dataset
- Live webcam inference
- Confidence score display
- Adaptive thresholding for uneven lighting
- Contour-based digit extraction
- CSV prediction logging
- Optional Flask web dashboard
- Mouse drawing canvas mode
- TensorFlow Lite support for Raspberry Pi deployment
- Modular project architecture

---

# Demo Workflow

```text
Webcam Feed → Detect Digit → Extract ROI → CNN Prediction → Display Result
```

---

# Tech Stack

| Technology | Purpose |
| --- | --- |
| Python | Core programming language |
| OpenCV | Image processing & webcam handling |
| TensorFlow | CNN model training & inference |
| TensorFlow Lite | Embedded deployment |
| NumPy | Numerical computations |
| Flask | Optional dashboard |
| Pillow | Image rendering |

---

# Project Structure

```text
HandwritingDigitBoard/
│
├── main.py
├── train_model.py
├── flask_dashboard.py
├── install.bat
├── requirements.txt
│
├── model/
│   ├── mnist_model.h5
│   └── mnist_model.tflite
│
├── modules/
│   ├── preprocessor.py
│   ├── classifier.py
│   ├── renderer.py
│   └── logger.py
│
├── utils/
│   └── drawing_canvas.py
│
├── logs/
│   └── predictions.csv
│
└── README.md
```

---

# Installation

## Prerequisites

- Python 3.9 or newer
- Webcam
- ~500 MB free storage for TensorFlow

---

## Clone Repository

```bash
git clone <your-repository-url>
cd HandwritingDigitBoard
```

---

## Install Dependencies

```bash
pip install opencv-python numpy tensorflow Flask Pillow
```

---

## Verify Installation

```bash
python -c "import cv2, numpy, tensorflow; print('All OK')"
```

---

# Usage

## Step 1 — Train the Model

Run once:

```bash
python train_model.py
```

This will:
- download the MNIST dataset,
- train the CNN model,
- save `.h5` and `.tflite` models.

Generated files:

```text
model/mnist_model.h5
model/mnist_model.tflite
```

---

## Step 2 — Start Live Digit Recognition

```bash
python main.py
```

The webcam window opens and starts detecting handwritten digits in real time.

---

# Optional Flags

```bash
python main.py --camera 1
python main.py --model model/mnist_model.tflite
python main.py --no-preview
python main.py --log
```

| Flag | Purpose |
| --- | --- |
| `--camera` | Select webcam index |
| `--model` | Use custom model path |
| `--no-preview` | Hide threshold preview |
| `--log` | Enable CSV logging |

---

# Mouse Drawing Mode

No webcam needed.

```bash
python utils/drawing_canvas.py
```

Features:
- Draw digits using mouse
- Press `SPACE` to classify
- Press `C` to clear canvas

---

# Flask Dashboard

```bash
python flask_dashboard.py
```

Open:

```text
http://localhost:5000
```

Dashboard includes:
- MJPEG webcam stream
- Live predictions
- Remote viewing support

---

# How It Works

# Image Processing Pipeline

```text
Webcam Frame
    ↓
Grayscale Conversion
    ↓
Gaussian Blur
    ↓
Adaptive Thresholding
    ↓
Contour Detection
    ↓
ROI Extraction
    ↓
Resize to 28×28
    ↓
CNN Prediction
    ↓
Confidence Display
```

---

# CNN Architecture

The model uses a Convolutional Neural Network trained on the MNIST dataset.

Pipeline:

```text
Conv Layer → MaxPooling → Conv Layer → Dense Layer → Softmax Output
```

The final output layer predicts probabilities for digits `0–9`.

---

# Confidence Scores

The CNN outputs softmax probabilities.

Example:

```text
[0.01, 0.02, 0.95, 0.01, ...]
```

Prediction:

```text
Digit = 2
Confidence = 95%
```

---

# Confidence Thresholds

| Confidence | Status |
| --- | --- |
| ≥ 80% | High confidence |
| 60–79% | Moderate confidence |
| < 60% | Uncertain prediction |

Low-confidence predictions display:

```text
?
```

instead of misleading outputs.

---

# Keyboard Controls

| Key | Action |
| --- | --- |
| `Q` | Quit application |
| `T` | Toggle threshold preview |
| `S` | Save current frame |
| `L` | Toggle CSV logging |
| `SPACE` | Pause / resume video |

---

# CSV Prediction Logging

Predictions can be logged into:

```text
logs/predictions.csv
```

Columns:

```csv
timestamp,predicted_digit,confidence,is_confident
```

---

# Troubleshooting

| Problem | Solution |
| --- | --- |
| Webcam not opening | Try `--camera 1` |
| TensorFlow install failure | Use `tensorflow-cpu` |
| Poor detection | Improve lighting |
| Wrong predictions | Write larger digits |
| Low FPS | Use TFLite model |
| Excess noise | Adjust contour thresholds |

---

# Performance Tips

- Use dark markers instead of pencil
- Keep paper flat
- Avoid shadows on paper
- Use the threshold preview (`T`) to check preprocessing quality
- Use TFLite backend for faster inference on low-end hardware

---

# Future Improvements

- Multi-digit equation recognition
- OCR sentence recognition
- GPU acceleration
- Touchscreen drawing mode
- Voice output for predictions
- Cloud prediction dashboard
- Edge deployment on Raspberry Pi

---

# Raspberry Pi Deployment

The project already supports TensorFlow Lite deployment.

Run:

```bash
pip install tflite-runtime
python main.py --model model/mnist_model.tflite
```

Optimizations:
- Lower camera resolution
- Skip preview rendering
- Run inference every alternate frame

---

# Resume Bullet Points

- Built a real-time handwritten digit recognition system using Python, OpenCV, and TensorFlow with ~99% MNIST accuracy.
- Designed a modular image preprocessing pipeline featuring adaptive thresholding, contour extraction, and ROI normalization.
- Implemented dual inference support using TensorFlow and TensorFlow Lite for desktop and embedded deployment.
- Developed supplementary features including Flask streaming dashboard, CSV prediction logging, and interactive drawing canvas.

---

# License

MIT License

Free for educational and personal use.
