import cv2
import numpy as np
import tensorflow as tf
import os

# ── Load model ─────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE, 'drowsiness_model.keras')
model = tf.keras.models.load_model(model_path, compile=False)

input_shape = model.input_shape
IMG_H, IMG_W = input_shape[1], input_shape[2]
IMG_SIZE = (IMG_W, IMG_H)
print(f"Loaded model — input: {IMG_H}x{IMG_W}, outputs: {model.output_shape[-1]} classes")

# ── Face cascade ───────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ── Classes: 0=Drowsy, 1=Non Drowsy (alphabetical, as set by flow_from_directory) ──
CLASS_NAMES = {0: 'Drowsy', 1: 'Not Drowsy'}

def preprocess(roi):
    img = cv2.resize(roi, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

# ── Open webcam ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Webcam running — press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (15, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 165, 255), 2, cv2.LINE_AA)
    else:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        face_roi   = frame[y:y + h, x:x + w]
        preds      = model.predict(preprocess(face_roi), verbose=0)[0]
        class_id   = int(np.argmax(preds))
        confidence = float(preds[class_id]) * 100

        label     = CLASS_NAMES[class_id]
        is_drowsy = class_id == 0
        color     = (0, 0, 255) if is_drowsy else (0, 200, 0)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, f"{label}  {confidence:.1f}%", (15, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)

    cv2.imshow('Drowsiness Detector (keras model)  [Q = quit]', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
