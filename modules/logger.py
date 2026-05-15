"""
Logger Module
-------------
Appends each prediction to a CSV file in the logs/ folder.
Useful for reviewing accuracy, building a dataset, or demo purposes.
"""

import csv
import os
from datetime import datetime


LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "predictions.csv")
HEADERS  = ["timestamp", "predicted_digit", "confidence", "is_confident"]


def _ensure_file():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(HEADERS)


def log_prediction(digit: int, confidence: float, is_confident: bool):
    _ensure_file()
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        digit,
        f"{confidence:.4f}",
        is_confident,
    ]
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow(row)
