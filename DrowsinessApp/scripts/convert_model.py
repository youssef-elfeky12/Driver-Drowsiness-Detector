"""
Convert drowsiness_efficientnet_b0.h5 → TensorFlow.js graph model.

Run once:
    pip install tensorflow tensorflowjs
    python scripts/convert_model.py

Output: public/models/efficientnet_b0/{model.json, group1-shard*.bin}
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_ROOT = HERE.parent
PROJECT_ROOT = APP_ROOT.parent

H5_PATH = PROJECT_ROOT / "Models" / "drowsiness_efficientnet_b0.h5"
OUT_DIR = APP_ROOT / "public" / "models" / "efficientnet_b0"


def main() -> None:
    if not H5_PATH.exists():
        sys.exit(f"Model not found: {H5_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "tensorflowjs.converters.converter",
        "--input_format=keras",
        "--output_format=tfjs_graph_model",
        str(H5_PATH),
        str(OUT_DIR),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"\nDone. Model written to {OUT_DIR}")


if __name__ == "__main__":
    main()
