"""
Convert drowsiness_resnet50v2.h5 -> drowsiness_resnet50v2.tflite

Run once:
    pip install "tensorflow==2.15.*"
    python scripts/convert_model.py

Output: assets/models/drowsiness_resnet50v2.tflite
"""
from pathlib import Path
import sys

import tensorflow as tf
from tensorflow import keras

HERE = Path(__file__).resolve().parent
APP_ROOT = HERE.parent
PROJECT_ROOT = APP_ROOT.parent

H5_PATH = PROJECT_ROOT / "Models" / "drowsiness_resnet50v2.h5"
OUT_PATH = APP_ROOT / "assets" / "models" / "drowsiness_resnet50v2.tflite"


def main() -> None:
    if not H5_PATH.exists():
        sys.exit(f"Model not found: {H5_PATH}")

    print(f"Loading {H5_PATH} ...")
    model = keras.models.load_model(str(H5_PATH), compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]

    print("Converting ...")
    tflite_bytes = converter.convert()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_bytes(tflite_bytes)
    print(f"\nWrote {OUT_PATH}  ({len(tflite_bytes) / (1024 * 1024):.1f} MB)")


if __name__ == "__main__":
    main()
