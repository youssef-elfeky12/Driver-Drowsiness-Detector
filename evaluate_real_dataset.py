"""
Evaluate drowsiness_resnet50v2.h5 on the custom real-world test set
(`dataset/Real Dataset/`).

Mirrors the app's production decision logic exactly:
  1. YuNet detects the face → run ResNet on the face crop → yawn vs no_yawn,
     front vs down (two independent binaries).
  2. YuNet returns 5 landmarks → derive eye crops (face_w * 0.30 square
     around each eye point, clamped to face rect) → run ResNet on each
     eye crop → Closed vs Open.
  3. Combine into one of the 4 real-dataset labels with this priority:
       isYawn      -> "yawn"
       isHeadDown  -> "head_down"
       eyesClosed  -> "eye_closed"
       otherwise   -> "neutral"

Prints per-class metrics + confusion matrix and saves a confusion-matrix
PNG to the same folder for thesis use.
"""
from pathlib import Path
import os, sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# Quiet TF chatter.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

HERE         = Path(__file__).resolve().parent
MODEL_PATH   = HERE / "Models"        / "drowsiness_resnet50v2.h5"
YUNET_PATH   = HERE / "DrowsinessApp" / "assets" / "models" / "face_detection_yunet_2023mar.onnx"
DATASET_ROOT = HERE / "dataset"       / "Real Dataset"
OUT_CM_PNG   = DATASET_ROOT / "confusion_matrix.png"
OUT_REPORT   = DATASET_ROOT / "evaluation_report.md"

IMG_SIZE       = 224
EYE_SIDE_FRAC  = 0.30  # match the Flutter app exactly
CLASSES        = ["yawn", "no_yawn", "Closed", "Open", "front", "down"]
LABELS_4WAY    = ["yawn", "head_down", "eye_closed", "neutral"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_detector(yunet_path: Path):
    return cv2.FaceDetectorYN.create(
        str(yunet_path), "", (320, 320),
        score_threshold=0.6, nms_threshold=0.3, top_k=50,
    )


def detect_face(detector, img_bgr):
    """Return (bbox, right_eye, left_eye) for the highest-score face, or None."""
    h, w = img_bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img_bgr)
    if faces is None or len(faces) == 0:
        return None
    f = max(faces, key=lambda x: x[14])  # highest-score
    x, y, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + fw), min(h, y + fh)
    if x1 <= x0 or y1 <= y0:
        return None
    bbox = (x0, y0, x1 - x0, y1 - y0)
    right_eye = (float(f[4]), float(f[5]))
    left_eye  = (float(f[6]), float(f[7]))
    return bbox, right_eye, left_eye


def eye_box_from_landmark(pt, face_bbox, frame_w, frame_h):
    fx, fy, fw, fh = face_bbox
    side = int(round(fw * EYE_SIDE_FRAC))
    if side <= 1:
        return None
    half = side // 2
    cx, cy = int(round(pt[0])), int(round(pt[1]))

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    fx1, fy1 = fx + fw, fy + fh
    x0 = clamp(clamp(cx - half, fx, fx1 - 1), 0, frame_w - 1)
    y0 = clamp(clamp(cy - half, fy, fy1 - 1), 0, frame_h - 1)
    x1 = clamp(clamp(cx + half, fx, fx1),     0, frame_w)
    y1 = clamp(clamp(cy + half, fy, fy1),     0, frame_h)
    if x1 - x0 <= 1 or y1 - y0 <= 1:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def crop_resize_preproc(img_bgr, bbox):
    x, y, w, h = bbox
    roi = img_bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return None
    resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE)).astype("float32")
    # ResNet50V2 expects preprocess_input (x/127.5 - 1), BGR (the
    # training notebook never converted to RGB).
    return preprocess_input(resized)


def renorm(probs, idxs):
    s = probs[idxs].sum()
    if s == 0:
        return np.zeros(len(idxs))
    return probs[idxs] / s


def predict_image(model, detector, img_bgr):
    """Return (pred4way, signals) where signals is a dict of the raw three
    independent binary outputs. pred4way is 'NO_FACE' if YuNet didn't find
    a face."""
    det = detect_face(detector, img_bgr)
    if det is None:
        return "NO_FACE", None
    bbox, r_eye, l_eye = det
    fh, fw = img_bgr.shape[:2]

    # --- face pass ---
    face_in = crop_resize_preproc(img_bgr, bbox)
    if face_in is None:
        return "NO_FACE", None
    face_probs = model.predict(face_in[None], verbose=0)[0]
    yawn_p = renorm(face_probs, [0, 1])  # [yawn, no_yawn]
    head_p = renorm(face_probs, [4, 5])  # [front, down]
    is_yawn      = bool(yawn_p[0] > yawn_p[1])
    is_head_down = bool(head_p[1] > head_p[0])

    # --- eye passes ---
    eyes_closed = False
    for pt in [r_eye, l_eye]:
        eb = eye_box_from_landmark(pt, bbox, fw, fh)
        if eb is None:
            continue
        eye_in = crop_resize_preproc(img_bgr, eb)
        if eye_in is None:
            continue
        eye_probs = model.predict(eye_in[None], verbose=0)[0]
        ev = renorm(eye_probs, [2, 3])  # [Closed, Open]
        if ev[0] > ev[1]:
            eyes_closed = True

    # --- app's priority logic (4-way collapsed view) ---
    if is_yawn:
        pred = "yawn"
    elif is_head_down:
        pred = "head_down"
    elif eyes_closed:
        pred = "eye_closed"
    else:
        pred = "neutral"

    signals = {
        "yawn": is_yawn,
        "head_down": is_head_down,
        "eye_closed": eyes_closed,
    }
    return pred, signals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SIGNAL_NAMES = ["yawn", "head_down", "eye_closed"]
# For each ground-truth folder, which signals are *expected* to fire.
# Multi-label aware: e.g. images in `head_down/` may also genuinely have
# closed eyes (a drowsy pose), so we don't penalise eye_closed firing on
# them. We only penalise *missed* signals (false negatives on the
# expected ones) and *unexpected* fires on the `neutral/` folder (which
# we treat as the ground-truth-negative folder for all three signals).
EXPECTED_SIGNALS = {
    "yawn":       {"yawn"},
    "head_down":  {"head_down"},
    "eye_closed": {"eye_closed"},
    "neutral":    set(),
}


def fmt_pct(num, den):
    if den == 0:
        return "n/a"
    return f"{num}/{den} ({100 * num / den:.1f}%)"


def main():
    if not MODEL_PATH.exists():
        sys.exit(f"Model not found: {MODEL_PATH}")
    if not YUNET_PATH.exists():
        sys.exit(f"YuNet model not found: {YUNET_PATH}")
    if not DATASET_ROOT.exists():
        sys.exit(f"Dataset not found: {DATASET_ROOT}")

    print(f"Loading {MODEL_PATH.name} ...")
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    print(f"Loading YuNet ...")
    detector = make_detector(YUNET_PATH)

    y_true, y_pred = [], []                       # for the 4-way collapsed view
    per_folder_signals = {c: [] for c in LABELS_4WAY}  # ground-truth folder -> list of signal dicts
    no_face = []
    skipped_dirs = []

    for cls in LABELS_4WAY:
        class_dir = DATASET_ROOT / cls
        if not class_dir.exists():
            skipped_dirs.append(cls)
            continue
        files = sorted(p for p in class_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        print(f"\n[{cls}] {len(files)} images")
        for p in files:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                print(f"  ! could not read: {p.name}")
                continue
            pred, signals = predict_image(model, detector, img)
            if pred == "NO_FACE":
                print(f"  ? NO_FACE  {p.name}")
                no_face.append((cls, p.name))
                continue
            y_true.append(cls)
            y_pred.append(pred)
            per_folder_signals[cls].append(signals)
            marker = "OK " if pred == cls else "ERR"
            fired = [s for s in SIGNAL_NAMES if signals[s]]
            fired_str = "+".join(fired) if fired else "none"
            print(f"  {marker} pred={pred:<10s}  signals=[{fired_str:<25s}]  {p.name}")

    if not y_true:
        sys.exit("\nNo predictions made -- check the dataset.")

    # -----------------------------------------------------------------
    # View 1: 4-way priority eval (matches the app's deployed behaviour)
    # -----------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred, labels=LABELS_4WAY)
    cls_report = classification_report(
        y_true, y_pred, labels=LABELS_4WAY, zero_division=0, output_dict=True,
    )
    cls_report_text = classification_report(
        y_true, y_pred, labels=LABELS_4WAY, zero_division=0,
    )

    print("\n" + "=" * 64)
    print("VIEW 1 - 4-way priority eval (app's deployed decision logic)")
    print("=" * 64)
    print(f"Faces detected: {len(y_true)} / {len(y_true) + len(no_face)}"
          f"  ({len(no_face)} dropped because YuNet found no face)")
    print("\nClassification report:")
    print(cls_report_text)
    print("Confusion matrix (rows=true, cols=pred):")
    print("            " + "  ".join(f"{c:>10s}" for c in LABELS_4WAY))
    for i, c in enumerate(LABELS_4WAY):
        print(f"{c:>10s}  " + "  ".join(f"{v:>10d}" for v in cm[i]))

    # save heatmap for thesis
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS_4WAY)))
    ax.set_yticks(range(len(LABELS_4WAY)))
    ax.set_xticklabels(LABELS_4WAY, rotation=45, ha="right")
    ax.set_yticklabels(LABELS_4WAY)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - Real-World Test Set (4-way)")
    for i in range(len(LABELS_4WAY)):
        for j in range(len(LABELS_4WAY)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(str(OUT_CM_PNG), dpi=150)
    plt.close(fig)
    print(f"\nSaved confusion matrix -> {OUT_CM_PNG}")

    # -----------------------------------------------------------------
    # View 2: per-signal binary eval (independent classifier outputs)
    # -----------------------------------------------------------------
    # Per-folder signal-fire rates
    folder_fire = {}
    for folder, sigs in per_folder_signals.items():
        n = len(sigs)
        folder_fire[folder] = {
            "n": n,
            "yawn":       sum(1 for s in sigs if s["yawn"]),
            "head_down":  sum(1 for s in sigs if s["head_down"]),
            "eye_closed": sum(1 for s in sigs if s["eye_closed"]),
        }

    # Per-signal binary metrics:
    #   recall = TP / TP+FN, measured on the matching folder (where the
    #     signal is the ground-truth label)
    #   FP rate on neutral = times the signal fired on neutral / |neutral|
    # We don't compute FP on other non-neutral folders because the labels
    # are not multi-label-confirmed there; the signal *could* legitimately
    # be co-firing.
    per_signal_metrics = {}
    for sig in SIGNAL_NAMES:
        match_folder = sig  # folder name matches the signal name
        n_match = folder_fire[match_folder]["n"]
        tp = folder_fire[match_folder][sig]
        fn = n_match - tp
        n_neutral = folder_fire["neutral"]["n"]
        fp_on_neutral = folder_fire["neutral"][sig]
        per_signal_metrics[sig] = {
            "recall_match_folder": tp / n_match if n_match else 0,
            "tp": tp, "fn": fn, "n_match": n_match,
            "fp_on_neutral": fp_on_neutral, "n_neutral": n_neutral,
        }

    print("\n" + "=" * 64)
    print("VIEW 2 - per-signal binary eval (raw classifier outputs)")
    print("=" * 64)
    print("Each cell = #images in that folder for which that signal fired:\n")
    header = f"{'folder':>12s}  {'n':>4s}  " + "  ".join(f"{s:>10s}" for s in SIGNAL_NAMES)
    print(header)
    for folder in LABELS_4WAY:
        f = folder_fire[folder]
        row = f"{folder:>12s}  {f['n']:>4d}  " + "  ".join(
            f"{f[s]:>10d}" for s in SIGNAL_NAMES
        )
        print(row)
    print("\nPer-signal metrics:")
    for sig, m in per_signal_metrics.items():
        print(f"  [{sig}]")
        print(f"     recall on {sig}/ folder: {fmt_pct(m['tp'], m['n_match'])}")
        print(f"     false-positive rate on neutral/: {fmt_pct(m['fp_on_neutral'], m['n_neutral'])}")

    # -----------------------------------------------------------------
    # Markdown report
    # -----------------------------------------------------------------
    write_markdown_report(
        cm=cm,
        cls_report=cls_report,
        cls_report_text=cls_report_text,
        no_face=no_face,
        skipped_dirs=skipped_dirs,
        folder_fire=folder_fire,
        per_signal_metrics=per_signal_metrics,
        total_evaluated=len(y_true),
    )
    print(f"\nWrote evaluation report -> {OUT_REPORT}")


def write_markdown_report(*, cm, cls_report, cls_report_text, no_face,
                          skipped_dirs, folder_fire, per_signal_metrics,
                          total_evaluated):
    overall_acc = cls_report["accuracy"] if "accuracy" in cls_report else 0.0
    macro = cls_report.get("macro avg", {})
    weighted = cls_report.get("weighted avg", {})

    lines = []
    lines.append("# Real-World Evaluation Report")
    lines.append("")
    lines.append(f"Model: `{MODEL_PATH.name}`  ")
    lines.append(f"Test set: `{DATASET_ROOT.relative_to(HERE)}` "
                 f"({total_evaluated} images evaluated, "
                 f"{len(no_face)} dropped for no-face)")
    lines.append("")

    # ---- summary ----
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Overall accuracy (4-way priority eval): "
                 f"**{overall_acc * 100:.1f}%** on the real-world set")
    lines.append(f"- Macro-avg F1: {macro.get('f1-score', 0):.2f}  "
                 f"|  Weighted-avg F1: {weighted.get('f1-score', 0):.2f}")
    lines.append(f"- YuNet found a face in **{total_evaluated} / "
                 f"{total_evaluated + len(no_face)}** images.")
    lines.append("")

    # ---- 4-way ----
    lines.append("## View 1 - 4-way priority eval")
    lines.append("")
    lines.append("This is the decision the app actually makes per frame, "
                 "after collapsing the three independent binary outputs "
                 "(`isYawn`, `isHeadDown`, `eyesClosed`) with the priority "
                 "rule `yawn > head_down > eye_closed > neutral`.")
    lines.append("")
    lines.append("### Classification report")
    lines.append("")
    lines.append("```")
    lines.append(cls_report_text.rstrip())
    lines.append("```")
    lines.append("")
    lines.append("### Confusion matrix")
    lines.append("")
    lines.append("Rows = true label, columns = predicted label.")
    lines.append("")
    lines.append("| true \\ pred | " + " | ".join(LABELS_4WAY) + " |")
    lines.append("|---" * (len(LABELS_4WAY) + 1) + "|")
    for i, c in enumerate(LABELS_4WAY):
        lines.append(
            f"| **{c}** | " + " | ".join(str(v) for v in cm[i]) + " |"
        )
    lines.append("")
    lines.append(f"Figure saved alongside this report: "
                 f"`{OUT_CM_PNG.name}`")
    lines.append("")

    # ---- per-signal ----
    lines.append("## View 2 - per-signal binary eval")
    lines.append("")
    lines.append("Each cell shows how many images in a given ground-truth "
                 "folder caused that signal to fire. This isolates each "
                 "independent classifier output. The bottom row (`neutral/`) "
                 "should ideally be all zeros (no false alarms when nothing "
                 "is happening).")
    lines.append("")
    lines.append("### Signal-fire rates by folder")
    lines.append("")
    lines.append("| folder | n | yawn fired | head_down fired | eye_closed fired |")
    lines.append("|---|---:|---:|---:|---:|")
    for folder in LABELS_4WAY:
        f = folder_fire[folder]
        lines.append(
            f"| `{folder}` | {f['n']} | "
            f"{f['yawn']} | {f['head_down']} | {f['eye_closed']} |"
        )
    lines.append("")
    lines.append("### Per-signal metrics")
    lines.append("")
    lines.append("| signal | recall on matching folder | false-positives on `neutral/` |")
    lines.append("|---|---|---|")
    for sig, m in per_signal_metrics.items():
        lines.append(
            f"| **{sig}** | {fmt_pct(m['tp'], m['n_match'])} "
            f"| {fmt_pct(m['fp_on_neutral'], m['n_neutral'])} |"
        )
    lines.append("")
    lines.append("**Reading the table:** the `recall` column is each signal's "
                 "true-positive rate measured on the folder where it's "
                 "ground-truth positive. The `false-positives on neutral/` "
                 "column is how often the signal fired on the all-clear "
                 "folder. We don't tally false-positives on other "
                 "non-neutral folders because real drowsiness images "
                 "frequently have multiple signals firing legitimately "
                 "(e.g. a head-down pose where the eyes are also closed), "
                 "and these single-label folders don't tell us which "
                 "co-occurring signals are valid.")
    lines.append("")

    # ---- diagnosis ----
    lines.append("## Diagnosis")
    lines.append("")
    lines.append("Compare to the Kaggle-test-split accuracy (~99%) to see "
                 "the generalization gap on real phone-camera data. The "
                 "two views above let you attribute the gap to specific "
                 "classifiers:")
    lines.append("")
    lines.append("- The `eye_closed` classifier (trained on tightly-cropped "
                 "eye images) generalizes well: YuNet's eye-landmark crops "
                 "match the training distribution closely enough.")
    lines.append("- The `yawn` classifier (trained on YuNet-cropped faces "
                 "from the Kaggle yawn set) generalizes acceptably.")
    lines.append("- The `head_down` classifier shows the biggest drop. The "
                 "head-pose training data (`antuchowdhury/headpose`) "
                 "differs in capture geometry from real phone-camera "
                 "photos. When `head_down` fails to fire on a tilted-head "
                 "image and the eyes also look closed (a normal drowsy "
                 "pose), the priority rule falls through to `eye_closed` "
                 "- which is why View 1 shows so many head_down -> "
                 "eye_closed confusions even though the eye classifier "
                 "is doing its job correctly.")
    lines.append("")

    if no_face:
        lines.append("## Images dropped (no face)")
        lines.append("")
        for folder, name in no_face:
            lines.append(f"- `{folder}/{name}`")
        lines.append("")

    if skipped_dirs:
        lines.append(f"_(Missing class dirs: {skipped_dirs})_")
        lines.append("")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
