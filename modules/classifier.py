"""
Classifier Module
-----------------
Wraps TensorFlow / TensorFlow Lite model loading and inference.

Why two backends?
  • Full TensorFlow model (.h5 / SavedModel) – easiest for development on a
    desktop/laptop with plenty of RAM.
  • TensorFlow Lite (.tflite) – the format used on Raspberry Pi (much smaller,
    faster inference on ARM).  The same .tflite file works on both platforms.

Usage:
    clf = DigitClassifier("model/mnist_model.h5")
    digit, confidence = clf.predict(roi_array)   # roi_array shape (1,28,28,1)
"""

import numpy as np
import os


CONFIDENCE_THRESHOLD = 0.60   # below this we call the prediction "uncertain"


class DigitClassifier:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.backend = self._detect_backend(model_path)
        self.model = self._load_model()
        print(f"[Classifier] Loaded '{model_path}' using backend: {self.backend}")

    # ── backend detection ────────────────────────────────────────────────────
    @staticmethod
    def _detect_backend(path: str) -> str:
        if path.endswith(".tflite"):
            return "tflite"
        return "tensorflow"

    # ── model loading ────────────────────────────────────────────────────────
    def _load_model(self):
        if self.backend == "tflite":
            return self._load_tflite()
        return self._load_tf()

    def _load_tf(self):
        import tensorflow as tf  # lazy import keeps startup fast if TFLite used
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at '{self.model_path}'.\n"
                "Run  python train_model.py  to train and save it first."
            )
        return tf.keras.models.load_model(self.model_path)

    def _load_tflite(self):
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=self.model_path)
        except ImportError:
            # tflite-runtime only (lighter package, available on Pi)
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()
        return interpreter

    # ── inference ────────────────────────────────────────────────────────────
    def predict(self, roi: np.ndarray) -> tuple[int, float]:
        """
        Run inference on a preprocessed ROI.

        Parameters
        ----------
        roi : np.ndarray  shape (1, 28, 28, 1), dtype float32, values in [0,1]

        Returns
        -------
        digit      : int   predicted digit 0-9
        confidence : float softmax probability of the top class  (0.0 – 1.0)
        """
        if self.backend == "tflite":
            return self._predict_tflite(roi)
        return self._predict_tf(roi)

    def _predict_tf(self, roi: np.ndarray) -> tuple[int, float]:
        probabilities = self.model.predict(roi, verbose=0)[0]  # shape (10,)
        digit = int(np.argmax(probabilities))
        confidence = float(probabilities[digit])
        return digit, confidence

    def _predict_tflite(self, roi: np.ndarray) -> tuple[int, float]:
        inp  = self.model.get_input_details()
        out  = self.model.get_output_details()
        self.model.set_tensor(inp[0]["index"], roi)
        self.model.invoke()
        probabilities = self.model.get_tensor(out[0]["index"])[0]
        digit = int(np.argmax(probabilities))
        confidence = float(probabilities[digit])
        return digit, confidence

    def is_confident(self, confidence: float) -> bool:
        return confidence >= CONFIDENCE_THRESHOLD
