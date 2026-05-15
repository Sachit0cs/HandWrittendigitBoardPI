"""
train_model.py
--------------
Downloads MNIST, trains a small CNN, then saves:
  • model/mnist_model.h5        (full TensorFlow model for desktop use)
  • model/mnist_model.tflite    (TensorFlow Lite model for Raspberry Pi use)

Run once before launching the main app:
    python train_model.py

What is MNIST?
--------------
MNIST (Modified National Institute of Standards and Technology) is a
dataset of 70,000 grayscale images of handwritten digits 0-9, each 28×28
pixels.  It is the "Hello World" of image classification.

What is a CNN?
--------------
A Convolutional Neural Network learns spatial patterns in images by sliding
small filters (kernels) across the image.  Early layers detect edges,
later layers detect digit shapes.  For MNIST a tiny 2-conv-layer network
reaches ~99 % test accuracy.

What is TensorFlow Lite?
------------------------
TFLite is a compressed, optimized version of a TF model.  It runs fast
on devices with little RAM (Raspberry Pi, microcontrollers).
"""

import os
import numpy as np

print("Loading TensorFlow...")
import tensorflow as tf
from tensorflow.keras import layers, models


MODEL_DIR   = os.path.join(os.path.dirname(__file__), "model")
H5_PATH     = os.path.join(MODEL_DIR, "mnist_model.h5")
TFLITE_PATH = os.path.join(MODEL_DIR, "mnist_model.tflite")

os.makedirs(MODEL_DIR, exist_ok=True)


def build_model() -> tf.keras.Model:
    """
    Compact CNN architecture:
      Conv2D  -> MaxPool  -> Conv2D  -> MaxPool  -> Flatten  -> Dense -> Softmax

    Why two conv blocks?
      First block learns edges/strokes, second learns digit shapes.
      More layers = better accuracy but slower inference.
    """
    model = models.Sequential([
        # Input: 28x28 greyscale image
        layers.Input(shape=(28, 28, 1)),

        # First conv block
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second conv block
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classifier head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),   # 10 digits 0-9
    ])
    return model


def convert_to_tflite(keras_model: tf.keras.Model, output_path: str):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"[TFLite] Saved to '{output_path}'  ({size_kb:.1f} KB)")


def main():
    # ── 1. Load & normalise MNIST ─────────────────────────────────────────────
    print("[MNIST] Downloading / loading dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype(np.float32) / 255.0
    x_test  = x_test.astype(np.float32)  / 255.0

    # Add channel dimension: (N, 28, 28) -> (N, 28, 28, 1)
    x_train = x_train[..., np.newaxis]
    x_test  = x_test[..., np.newaxis]

    print(f"  Train: {x_train.shape}  |  Test: {x_test.shape}")

    # ── 2. Build & compile model ──────────────────────────────────────────────
    model = build_model()
    model.summary()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── 3. Train ──────────────────────────────────────────────────────────────
    print("\n[Training] Starting...")
    history = model.fit(
        x_train, y_train,
        epochs=5,           # 5 epochs is plenty for ~99% accuracy on MNIST
        batch_size=128,
        validation_split=0.1,
        verbose=1,
    )

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n[Evaluation] Test accuracy: {acc*100:.2f}%  |  Loss: {loss:.4f}")

    # ── 5. Save .h5 ───────────────────────────────────────────────────────────
    model.save(H5_PATH)
    print(f"[Model] Saved Keras model to '{H5_PATH}'")

    # ── 6. Convert & save .tflite ─────────────────────────────────────────────
    convert_to_tflite(model, TFLITE_PATH)
    print("\nDone!  Run  python main.py  to start the digit board.")


if __name__ == "__main__":
    main()
