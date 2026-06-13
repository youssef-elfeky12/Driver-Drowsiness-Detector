"""One-time: convert the small 6-class drowsiness CNN to ONNX for the web app.

Source model : Models/drowsiness_eyeyawnnod.h5
  - input  : 145x145x3, BGR, pixels scaled /255.0 (matches drowsiness_detector.py)
  - output : 6-class softmax [yawn, no_yawn, Closed, Open, front, down]

The browser app (onnxruntime-web) feeds an already-normalized [1,145,145,3]
float32 tensor, so we export the bare model graph (no preprocessing baked in).

Output: DrowsinessApp/webapp/public/models/classifier.onnx
Also copies the YuNet ONNX detector into the same folder.

Run:  python scripts/web_convert.py
"""
import os
import shutil
import numpy as np
import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime as ort

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_H5 = os.path.join(BASE, "Models", "drowsiness_eyeyawnnod.h5")
YUNET_SRC = os.path.join(
    BASE, "DrowsinessApp", "assets", "models", "face_detection_yunet_2023mar.onnx"
)
OUT_DIR = os.path.join(BASE, "DrowsinessApp", "webapp", "public", "models")
OUT_ONNX = os.path.join(OUT_DIR, "classifier.onnx")
OUT_YUNET = os.path.join(OUT_DIR, "face_detection_yunet_2023mar.onnx")

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading", SRC_H5)
model = tf.keras.models.load_model(SRC_H5, compile=False)
in_shape = model.input_shape  # (None,145,145,3)
H, W, C = in_shape[1], in_shape[2], in_shape[3]
n_out = model.output_shape[-1]
print(f"  input {H}x{W}x{C}  outputs {n_out}")
assert n_out == 6, "expected a 6-class model"

spec = (tf.TensorSpec((1, H, W, C), tf.float32, name="input"),)

# Keras 3 + tf2onnx: from_keras is flaky, so route through a tf.function.
@tf.function(input_signature=list(spec))
def serve(x):
    return model(x, training=False)

print("Converting to ONNX (opset 13)...")
onnx_model, _ = tf2onnx.convert.from_function(
    serve, input_signature=list(spec), opset=13, output_path=OUT_ONNX
)
print("  wrote", OUT_ONNX)

# ---- Numerical verification: Keras vs ONNX on random inputs ----
print("Verifying ONNX matches Keras...")
sess = ort.InferenceSession(OUT_ONNX, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name

max_err = 0.0
for _ in range(5):
    x = np.random.rand(1, H, W, C).astype(np.float32)
    k = model.predict(x, verbose=0)
    o = sess.run([out_name], {in_name: x})[0]
    max_err = max(max_err, float(np.max(np.abs(k - o))))
print(f"  max abs diff over 5 random inputs: {max_err:.3e}")
assert max_err < 1e-4, "ONNX output diverged from Keras!"

# ---- Copy YuNet detector alongside ----
shutil.copyfile(YUNET_SRC, OUT_YUNET)
print("  copied YuNet ->", OUT_YUNET)

print("\nDONE. Class order: [yawn, no_yawn, Closed, Open, front, down]")
print("Preprocess in browser: BGR, resize 145x145, pixels / 255.0")
