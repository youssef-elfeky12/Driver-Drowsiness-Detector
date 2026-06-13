// YuNet (face_detection_yunet_2023mar.onnx) decoding, reimplemented for the
// browser. OpenCV's FaceDetectorYN does this internally; onnxruntime-web gives
// us only the raw cls/obj/bbox/kps tensors, so we replicate the decode +
// per-stride prior generation + NMS here.
//
// Model contract:
//   input  : [1,3,640,640] float32, CHW, BGR, pixels 0..255 (no mean/scale)
//   outputs: cls_{8,16,32}, obj_{8,16,32}, bbox_{8,16,32}, kps_{8,16,32}
//
// Decoded box/landmark coords are in the 640x640 input space; the caller
// scales them back to the source frame.

export const YUNET_SIZE = 640;
const STRIDES = [8, 16, 32];

export interface YuNetFace {
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
  rightEye: [number, number];
  leftEye: [number, number];
}

export function decodeYuNet(
  outputs: Record<string, Float32Array>,
  scoreThreshold = 0.6,
  nmsThreshold = 0.3,
  topK = 50,
): YuNetFace[] {
  const faces: YuNetFace[] = [];

  for (const stride of STRIDES) {
    const cls = outputs[`cls_${stride}`];
    const obj = outputs[`obj_${stride}`];
    const bbox = outputs[`bbox_${stride}`];
    const kps = outputs[`kps_${stride}`];
    if (!cls || !obj || !bbox || !kps) continue;

    const cols = Math.floor(YUNET_SIZE / stride);
    const rows = Math.floor(YUNET_SIZE / stride);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        const clsScore = clamp01(cls[idx]);
        const objScore = clamp01(obj[idx]);
        const score = Math.sqrt(clsScore * objScore);
        if (score < scoreThreshold) continue;

        // bbox: cx,cy offsets (in grid units) + log w/h.
        const cx = (c + bbox[idx * 4 + 0]) * stride;
        const cy = (r + bbox[idx * 4 + 1]) * stride;
        const w = Math.exp(bbox[idx * 4 + 2]) * stride;
        const h = Math.exp(bbox[idx * 4 + 3]) * stride;
        const x = cx - w / 2;
        const y = cy - h / 2;

        // Landmarks: point 0 = right eye, point 1 = left eye.
        const rex = (c + kps[idx * 10 + 0]) * stride;
        const rey = (r + kps[idx * 10 + 1]) * stride;
        const lex = (c + kps[idx * 10 + 2]) * stride;
        const ley = (r + kps[idx * 10 + 3]) * stride;

        faces.push({
          x,
          y,
          w,
          h,
          score,
          rightEye: [rex, rey],
          leftEye: [lex, ley],
        });
      }
    }
  }

  return nms(faces, nmsThreshold).slice(0, topK);
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function iou(a: YuNetFace, b: YuNetFace): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const iw = Math.max(0, x2 - x1);
  const ih = Math.max(0, y2 - y1);
  const inter = iw * ih;
  const union = a.w * a.h + b.w * b.h - inter;
  return union <= 0 ? 0 : inter / union;
}

function nms(faces: YuNetFace[], thresh: number): YuNetFace[] {
  const sorted = [...faces].sort((p, q) => q.score - p.score);
  const keep: YuNetFace[] = [];
  for (const f of sorted) {
    if (keep.every((k) => iou(f, k) <= thresh)) keep.push(f);
  }
  return keep;
}
