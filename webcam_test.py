import cv2
import numpy as np
import tensorflow as tf
import os

# ── Load model ─────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE, 'drowiness_new6.h5')
model = tf.keras.models.load_model(model_path, compile=False)

input_shape = model.input_shape
IMG_H, IMG_W = input_shape[1], input_shape[2]
IMG_SIZE = (IMG_W, IMG_H)
print(f"Loaded model — input: {IMG_H}x{IMG_W}, outputs: {model.output_shape[-1]} classes")

# ── Cascades ───────────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# ── Classes: 0=yawn, 1=no_yawn, 2=Closed, 3=Open ──────────────────────────
def preprocess(roi):
    img = cv2.resize(roi, IMG_SIZE)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def predict_roi(roi):
    """Returns (class_id, confidence, full probs array)"""
    preds = model.predict(preprocess(roi), verbose=0)[0]
    class_id = int(np.argmax(preds))
    return class_id, float(preds[class_id]) * 100, preds

# ── Open webcam ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Webcam running — press Q to quit.")

last_eye_count = 0   # tracks how many eyes were visible in previous frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Default status strings
    yawn_text = "Yawn: --"
    eye_text  = "Eyes: --"
    yawn_color = (200, 200, 200)
    eye_color  = (200, 200, 200)
    is_drowsy  = False

    if len(faces) == 0:
        yawn_text = "No face detected"
        yawn_color = (0, 165, 255)
        last_eye_count = 0
    else:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        face_roi = frame[y:y + h, x:x + w]

        # ── Yawn detection: feed face crop, read classes 0 & 1 ────────────
        _, _, probs = predict_roi(face_roi)
        yawn_conf    = float(probs[0]) * 100
        no_yawn_conf = float(probs[1]) * 100
        if yawn_conf > no_yawn_conf:
            yawn_text  = f"Yawn: YES  {yawn_conf:.1f}%"
            yawn_color = (0, 0, 255)
            is_drowsy  = True
        else:
            yawn_text  = f"Yawn: No  {no_yawn_conf:.1f}%"
            yawn_color = (0, 200, 0)

        # ── Eye detection: search only upper 55% of face to avoid mouth/chin ──
        eye_region_h = int(h * 0.55)
        face_gray_upper = gray[y:y + eye_region_h, x:x + w]

        # Min/max eye size relative to face width
        min_eye = int(w * 0.10)
        max_eye = int(w * 0.40)
        eyes = eye_cascade.detectMultiScale(
            face_gray_upper,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(min_eye, min_eye),
            maxSize=(max_eye, max_eye)
        )

        # Keep at most 2 eyes (largest two by area)
        if len(eyes) > 2:
            eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]

        num_eyes_found = len(eyes)

        # Eyes that were previously seen but now missing are likely closed
        # We assess detected eyes + count missing ones as closed votes
        closed_votes = max(0, last_eye_count - num_eyes_found)  # missing eyes = closed
        open_votes   = 0

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame,
                          (x + ex, y + ey),
                          (x + ex + ew, y + ey + eh),
                          (0, 255, 255), 2)
            eye_roi = face_roi[ey:ey + eh, ex:ex + ew]
            _, _, probs = predict_roi(eye_roi)
            closed_conf = float(probs[2]) * 100
            open_conf   = float(probs[3]) * 100
            if closed_conf > open_conf:
                closed_votes += 1
            else:
                open_votes += 1

        # Update last known eye count (only update when eyes are clearly visible)
        if num_eyes_found >= 2:
            last_eye_count = 2
        elif num_eyes_found == 1 and last_eye_count == 0:
            last_eye_count = 1

        total_assessed = closed_votes + open_votes
        if total_assessed == 0:
            eye_text  = "Eyes: not detected"
            eye_color = (0, 165, 255)
        elif closed_votes > open_votes:
            eye_text  = f"Eyes: CLOSED ({closed_votes}/{total_assessed})"
            eye_color = (0, 0, 255)
            is_drowsy = True
        else:
            eye_text  = f"Eyes: Open ({open_votes}/{total_assessed})"
            eye_color = (0, 200, 0)

    # ── HUD ────────────────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 105), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    overall       = "⚠ DROWSY" if is_drowsy else "ALERT"
    overall_color = (0, 0, 255) if is_drowsy else (0, 200, 0)
    cv2.putText(frame, overall,    (15, 38),  cv2.FONT_HERSHEY_SIMPLEX, 1.2, overall_color,    3, cv2.LINE_AA)
    cv2.putText(frame, yawn_text,  (15, 68),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, yawn_color, 2, cv2.LINE_AA)
    cv2.putText(frame, eye_text,   (15, 95),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color,  2, cv2.LINE_AA)

    cv2.imshow('Driver Drowsiness Detector  [Q = quit]', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

