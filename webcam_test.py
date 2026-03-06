import cv2
import numpy as np
import tensorflow as tf
import os

# ── Load model (supports both .keras and .h5) ──────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
for name in ('drowsiness_model.keras', 'drowsiness_model.h5'):
    path = os.path.join(BASE, name)
    if os.path.exists(path):
        model = tf.keras.models.load_model(path)
        print(f"Loaded model: {name}")
        break
else:
    raise FileNotFoundError("No model file found. Run the notebook and save the model first.")

# ── Class mapping (alphabetical order used by flow_from_directory) ─────────
# Drowsy=0, Non Drowsy=1
CLASS_NAMES = {0: 'Drowsy', 1: 'Not Drowsy'}
COLORS      = {0: (0, 0, 255), 1: (0, 200, 0)}   # red / green (BGR)

IMG_SIZE = (224, 224)

def preprocess(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

# ── Open webcam ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check that it is connected.")

print("Webcam running — press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    inp        = preprocess(frame)
    preds      = model.predict(inp, verbose=0)[0]   # shape (2,)
    class_id   = int(np.argmax(preds))
    confidence = float(preds[class_id]) * 100

    label = CLASS_NAMES[class_id]
    color = COLORS[class_id]
    text  = f"{label}  {confidence:.1f}%"

    # Semi-transparent background bar for readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, text, (15, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)

    cv2.imshow('Driver Drowsiness Detector  [Q = quit]', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
