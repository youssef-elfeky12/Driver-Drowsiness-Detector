import os
import threading
import time
import winsound
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

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
EYE_WINDOW_SECONDS = 2.0
EYE_THRESHOLD = 0.50
EYE_STREAK_TO_ALARM = 3
YAWN_WINDOW_SECONDS = 2.0
YAWN_THRESHOLD = 0.5
YAWN_TRIGGER_COUNT = 2
YAWN_LOOKBACK_SECONDS = 20.0
PULL_OVER_SECONDS = 5.0


def preprocess(roi):
    img = cv2.resize(roi, IMG_SIZE)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def predict_roi(roi):
    """Returns (class_id, confidence, full probs array)"""
    preds = model.predict(preprocess(roi), verbose=0)[0]
    class_id = int(np.argmax(preds))
    return class_id, float(preds[class_id]) * 100, preds


def prune_samples(samples, max_age, now):
    while samples and now - samples[0][0] > max_age:
        samples.popleft()


def window_majority_state(samples, window_seconds, threshold, now):
    relevant = [value for ts, value in samples if now - ts <= window_seconds]
    if not relevant:
        return None

    true_ratio = sum(relevant) / len(relevant)
    false_ratio = 1.0 - true_ratio
    if true_ratio >= threshold and true_ratio >= false_ratio:
        return True
    if false_ratio >= threshold and false_ratio > true_ratio:
        return False
    return None


def start_alarm(alarm_state):
    if alarm_state["active"]:
        return

    alarm_state["active"] = True
    alarm_state["stop_event"].clear()

    def alarm_worker():
        while not alarm_state["stop_event"].is_set():
            # Continuous "extreme" sound (2500Hz)
            winsound.Beep(2500, 500)
            if alarm_state["stop_event"].is_set():
                break

    alarm_state["thread"] = threading.Thread(target=alarm_worker, daemon=True)
    alarm_state["thread"].start()


def stop_alarm(alarm_state):
    if not alarm_state["active"]:
        return

    alarm_state["stop_event"].set()
    alarm_state["active"] = False
    alarm_state["thread"] = None


def play_pull_over_sound():
    def worker():
        for _ in range(3):
            winsound.Beep(2600, 180)
            time.sleep(0.12)

    threading.Thread(target=worker, daemon=True).start()

# ── Open webcam ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Webcam running — press Q to quit.")

last_eye_count = 0   # tracks how many eyes were visible in previous frame
eye_samples = deque()
yawn_samples = deque()
yawn_events = deque()
last_eye_window_check = time.monotonic()
last_yawn_window_check = time.monotonic()
eye_closed_streak = 0
eye_window_state = None
last_yawn_window_state = False
pull_over_until = 0.0
pull_over_flash_on = False
alarm_state = {
    "active": False,
    "stop_event": threading.Event(),
    "thread": None,
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.monotonic()
    prune_samples(eye_samples, EYE_WINDOW_SECONDS + 0.5, now)
    prune_samples(yawn_samples, YAWN_WINDOW_SECONDS + 0.5, now)
    while yawn_events and now - yawn_events[0] > YAWN_LOOKBACK_SECONDS:
        yawn_events.popleft()

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Default status strings
    yawn_text = "Yawn (2s): --"
    eye_text  = "Eyes (2s): --"
    yawn_color = (200, 200, 200)
    eye_color  = (200, 200, 200)
    current_yawn_detected = None
    current_eye_closed = None
    is_drowsy = alarm_state["active"]
    current_yawn_debug = "Current Yawn: --"
    current_eye_debug = "Current Eye: --"

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
            yawn_text  = f"Yawn now: YES  {yawn_conf:.1f}%"
            yawn_color = (0, 0, 255)
            current_yawn_detected = True
            current_yawn_debug = f"Current Yawn: YES ({yawn_conf:.1f}%)"
        else:
            yawn_text  = f"Yawn now: No  {no_yawn_conf:.1f}%"
            yawn_color = (0, 200, 0)
            current_yawn_detected = False
            current_yawn_debug = f"Current Yawn: No ({no_yawn_conf:.1f}%)"

        yawn_samples.append((now, current_yawn_detected))

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
            eye_text  = "Eyes now: not detected"
            eye_color = (0, 165, 255)
            current_eye_debug = "Current Eye: not detected"
        elif closed_votes > open_votes:
            eye_text  = f"Eyes now: CLOSED ({closed_votes}/{total_assessed})"
            eye_color = (0, 0, 255)
            current_eye_closed = True
            current_eye_debug = f"Current Eye: CLOSED ({closed_votes}/{total_assessed})"
        else:
            eye_text  = f"Eyes now: Open ({open_votes}/{total_assessed})"
            eye_color = (0, 200, 0)
            current_eye_closed = False
            current_eye_debug = f"Current Eye: Open ({open_votes}/{total_assessed})"

        if current_eye_closed is not None:
            eye_samples.append((now, current_eye_closed))

    if now - last_eye_window_check >= EYE_WINDOW_SECONDS:
        eye_window_state = window_majority_state(eye_samples, EYE_WINDOW_SECONDS, EYE_THRESHOLD, now)
        last_eye_window_check = now

        if eye_window_state is True:
            eye_closed_streak += 1
        elif eye_window_state is False:
            eye_closed_streak = 0
            stop_alarm(alarm_state)

        if eye_closed_streak >= EYE_STREAK_TO_ALARM:
            start_alarm(alarm_state)

    if now - last_yawn_window_check >= YAWN_WINDOW_SECONDS:
        yawn_window_state = window_majority_state(yawn_samples, YAWN_WINDOW_SECONDS, YAWN_THRESHOLD, now)
        last_yawn_window_check = now

        if yawn_window_state is True and not last_yawn_window_state:
            yawn_events.append(now)
            if len(yawn_events) >= YAWN_TRIGGER_COUNT:
                pull_over_until = now + PULL_OVER_SECONDS
                play_pull_over_sound()

        last_yawn_window_state = (yawn_window_state is True)

    if eye_window_state is True:
        eye_text = "Eyes (2s): CLOSED"
        eye_color = (0, 0, 255)
        is_drowsy = True
    elif eye_window_state is False:
        eye_text = "Eyes (2s): Open"
        eye_color = (0, 200, 0)
    elif eye_window_state is None and not eye_text.startswith("Eyes now"):
        eye_text = "Eyes (2s): inconclusive"
        eye_color = (0, 165, 255)

    if last_yawn_window_state:
        yawn_text = f"Yawn (2s): YES  events {len(yawn_events)}/2"
        yawn_color = (0, 0, 255)
        is_drowsy = True
    elif current_yawn_detected is not None:
        yawn_text = f"Yawn (2s): No  events {len(yawn_events)}/2"
        yawn_color = (0, 200, 0)

    if alarm_state["active"]:
        is_drowsy = True

    # ── HUD ────────────────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 170), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    overall = "DROWSY" if is_drowsy else "ALERT"
    overall_color = (0, 0, 255) if is_drowsy else (0, 200, 0)
    cv2.putText(frame, overall,    (15, 38),  cv2.FONT_HERSHEY_SIMPLEX, 1.2, overall_color,    3, cv2.LINE_AA)
    cv2.putText(frame, yawn_text,  (15, 68),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, yawn_color, 2, cv2.LINE_AA)
    cv2.putText(frame, eye_text,   (15, 95),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color,  2, cv2.LINE_AA)
    counter_text = f"2s closed counter: {eye_closed_streak}/3"
    counter_color = (0, 0, 255) if eye_closed_streak > 0 else (180, 180, 180)
    cv2.putText(frame, counter_text, (15, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.7, counter_color, 2, cv2.LINE_AA)
    alarm_text = "Alarm: ON" if alarm_state["active"] else "Alarm: off"
    alarm_color = (0, 0, 255) if alarm_state["active"] else (180, 180, 180)
    cv2.putText(frame, alarm_text, (15, 149), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alarm_color, 2, cv2.LINE_AA)

    # Small live debug status on right side (instant, frame-level)
    right_x = max(15, frame.shape[1] - 320)
    cv2.putText(frame, current_yawn_debug, (right_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, current_eye_debug,  (right_x, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # Show PULL OVER if manual timer is active OR if the continuous eye alarm is ringing
    if now < pull_over_until or alarm_state["active"]:
        pull_over_flash_on = not pull_over_flash_on
        if pull_over_flash_on:
            flash = frame.copy()
            cv2.rectangle(flash, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
            cv2.addWeighted(flash, 0.18, frame, 0.82, 0, frame)
        text_size = cv2.getTextSize("PULL OVER", cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        cv2.putText(frame, "PULL OVER", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5, cv2.LINE_AA)

    cv2.imshow('Driver Drowsiness Detector  [Q = quit]', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stop_alarm(alarm_state)
cap.release()
cv2.destroyAllWindows()

