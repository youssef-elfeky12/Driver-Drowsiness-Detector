// Browser port of lib/services/detector.dart.
//
// Pipeline per frame:
//   1. YuNet (onnxruntime-web) → face bbox + right/left eye landmarks
//   2. small 6-class CNN on the face crop  → yawn/no_yawn + front/down binaries
//   3. same CNN on each eye crop           → Closed/Open
//
// Class index map: 0=yawn 1=no_yawn 2=Closed 3=Open 4=front 5=down
// Classifier input: [1,145,145,3] NHWC, BGR, pixels /255.
// YuNet input:      [1,3,640,640] CHW,  BGR, pixels 0..255.

import * as ort from "onnxruntime-web";
import {
  DetectionResult,
  EyePrediction,
  FaceBox,
  FaceClassName,
  FacePrediction,
  makeResult,
} from "./types";
import { decodeYuNet, YUNET_SIZE } from "./yunet";

const IMG_SIZE = 145;
const EYE_SIDE_FRAC = 0.3;
const YAWN_IDX = [0, 1];
const HEAD_IDX = [4, 5];
const EYE_IDX = [2, 3];

// onnxruntime-web loads its wasm runtime from this CDN path (version-pinned so
// it matches the installed package). Avoids bundling the .wasm files.
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

export class Detector {
  private yunet: ort.InferenceSession | null = null;
  private clf: ort.InferenceSession | null = null;

  // Reusable offscreen canvases for resizing crops.
  private yunetCanvas: HTMLCanvasElement;
  private yunetCtx: CanvasRenderingContext2D;
  private cropCanvas: HTMLCanvasElement;
  private cropCtx: CanvasRenderingContext2D;

  constructor() {
    this.yunetCanvas = document.createElement("canvas");
    this.yunetCanvas.width = YUNET_SIZE;
    this.yunetCanvas.height = YUNET_SIZE;
    this.yunetCtx = this.yunetCanvas.getContext("2d", {
      willReadFrequently: true,
    })!;
    this.cropCanvas = document.createElement("canvas");
    this.cropCanvas.width = IMG_SIZE;
    this.cropCanvas.height = IMG_SIZE;
    this.cropCtx = this.cropCanvas.getContext("2d", {
      willReadFrequently: true,
    })!;
  }

  get isReady() {
    return this.yunet !== null;
  }
  get canClassify() {
    return this.clf !== null;
  }

  async init(onProgress?: (msg: string) => void): Promise<void> {
    const opts: ort.InferenceSession.SessionOptions = {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    };
    onProgress?.("Loading face detector…");
    this.yunet = await ort.InferenceSession.create(
      "/models/face_detection_yunet_2023mar.onnx",
      opts,
    );
    onProgress?.("Loading model…");
    this.clf = await ort.InferenceSession.create(
      "/models/classifier.onnx",
      opts,
    );
    onProgress?.("Ready");
  }

  /**
   * Run the full pipeline on a video frame.
   * @param source the <video> (or canvas) element to read the current frame from
   * @param srcW   processing width  (frame is sampled at this resolution)
   * @param srcH   processing height
   * @param confThreshold unused at this layer (the alert engine applies it) but
   *                kept for parity; YuNet uses its own fixed score threshold.
   */
  async detect(
    source: CanvasImageSource,
    srcW: number,
    srcH: number,
  ): Promise<DetectionResult> {
    const ts = Date.now();
    if (!this.yunet || !this.clf) {
      return makeResult([], srcW, srcH, ts);
    }

    const yunetFaces = await this.runYuNet(source, srcW, srcH);
    const out: FacePrediction[] = [];

    for (const yf of yunetFaces) {
      // Clamp the face rect to frame bounds.
      const x0 = clampInt(yf.x, 0, srcW - 1);
      const y0 = clampInt(yf.y, 0, srcH - 1);
      const x1 = clampInt(yf.x + yf.w, 0, srcW);
      const y1 = clampInt(yf.y + yf.h, 0, srcH);
      const fw = x1 - x0;
      const fh = y1 - y0;
      if (fw <= 0 || fh <= 0) continue;
      const faceBox: FaceBox = { x: x0, y: y0, w: fw, h: fh };

      const probs = await this.classify(source, faceBox, srcW, srcH);

      const yawnP = renorm(probs, YAWN_IDX);
      const headP = renorm(probs, HEAD_IDX);
      const isYawn = yawnP[0] > yawnP[1];
      const yawnConf = isYawn ? yawnP[0] : yawnP[1];
      const isHeadDown = headP[1] > headP[0];
      const headConf = isHeadDown ? headP[1] : headP[0];

      let faceClass: FaceClassName;
      let faceConf: number;
      if (isYawn) {
        faceClass = "yawn";
        faceConf = yawnConf;
      } else if (isHeadDown) {
        faceClass = "down";
        faceConf = headConf;
      } else {
        faceClass = yawnConf >= headConf ? "no_yawn" : "front";
        faceConf = (yawnConf + headConf) / 2;
      }

      const eyes: EyePrediction[] = [];
      for (const pt of [yf.rightEye, yf.leftEye]) {
        const eb = eyeBoxFromLandmark(pt, faceBox, srcW, srcH);
        if (!eb) continue;
        const ep = await this.classify(source, eb, srcW, srcH);
        const ev = renorm(ep, EYE_IDX);
        const closed = ev[0] > ev[1];
        eyes.push({
          box: eb,
          eyeClass: closed ? "Closed" : "Open",
          conf: closed ? ev[0] : ev[1],
        });
      }

      out.push({
        box: faceBox,
        faceClass,
        conf: faceConf,
        isYawn,
        yawnConf,
        isHeadDown,
        headPoseConf: headConf,
        eyes,
      });
    }

    return makeResult(out, srcW, srcH, ts);
  }

  private async runYuNet(
    source: CanvasImageSource,
    srcW: number,
    srcH: number,
  ) {
    // Stretch-resize the frame into the 640x640 input (matches OpenCV
    // FaceDetectorYN.setInputSize, which also just resizes, no letterbox).
    this.yunetCtx.drawImage(source, 0, 0, YUNET_SIZE, YUNET_SIZE);
    const data = this.yunetCtx.getImageData(0, 0, YUNET_SIZE, YUNET_SIZE).data;

    // CHW, BGR, 0..255.
    const plane = YUNET_SIZE * YUNET_SIZE;
    const input = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      const r = data[i * 4 + 0];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      input[i] = b; // B plane
      input[plane + i] = g; // G plane
      input[2 * plane + i] = r; // R plane
    }
    const tensor = new ort.Tensor("float32", input, [1, 3, YUNET_SIZE, YUNET_SIZE]);
    const result = await this.yunet!.run({
      [this.yunet!.inputNames[0]]: tensor,
    });

    const outs: Record<string, Float32Array> = {};
    for (const name of this.yunet!.outputNames) {
      outs[name] = result[name].data as Float32Array;
    }
    const faces = decodeYuNet(outs);

    // Scale 640x640 coords back to the source frame (per-axis).
    const sx = srcW / YUNET_SIZE;
    const sy = srcH / YUNET_SIZE;
    return faces.map((f) => ({
      x: f.x * sx,
      y: f.y * sy,
      w: f.w * sx,
      h: f.h * sy,
      score: f.score,
      rightEye: [f.rightEye[0] * sx, f.rightEye[1] * sy] as [number, number],
      leftEye: [f.leftEye[0] * sx, f.leftEye[1] * sy] as [number, number],
    }));
  }

  /** Run the 6-class CNN on a crop, return raw softmax probabilities. */
  private async classify(
    source: CanvasImageSource,
    box: FaceBox,
    _srcW: number,
    _srcH: number,
  ): Promise<Float32Array> {
    // Draw the crop region resized to 145x145.
    this.cropCtx.drawImage(
      source,
      box.x,
      box.y,
      box.w,
      box.h,
      0,
      0,
      IMG_SIZE,
      IMG_SIZE,
    );
    const data = this.cropCtx.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data;

    // NHWC, BGR, /255.
    const input = new Float32Array(IMG_SIZE * IMG_SIZE * 3);
    for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
      const r = data[i * 4 + 0];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      input[i * 3 + 0] = b / 255;
      input[i * 3 + 1] = g / 255;
      input[i * 3 + 2] = r / 255;
    }
    const tensor = new ort.Tensor("float32", input, [1, IMG_SIZE, IMG_SIZE, 3]);
    const result = await this.clf!.run({
      [this.clf!.inputNames[0]]: tensor,
    });
    return result[this.clf!.outputNames[0]].data as Float32Array;
  }

  dispose() {
    this.yunet?.release();
    this.clf?.release();
    this.yunet = null;
    this.clf = null;
  }
}

function renorm(probs: Float32Array, indices: number[]): number[] {
  const subset = indices.map((i) => probs[i]);
  const sum = subset.reduce((a, b) => a + b, 0);
  if (sum === 0) return subset.map(() => 0);
  return subset.map((v) => v / sum);
}

function clampInt(v: number, lo: number, hi: number): number {
  v = Math.round(v);
  return v < lo ? lo : v > hi ? hi : v;
}

function eyeBoxFromLandmark(
  pt: [number, number],
  face: FaceBox,
  frameW: number,
  frameH: number,
): FaceBox | null {
  const side = Math.round(face.w * EYE_SIDE_FRAC);
  if (side <= 1) return null;
  const half = Math.floor(side / 2);
  const cx = Math.round(pt[0]);
  const cy = Math.round(pt[1]);
  const faceX1 = face.x + face.w;
  const faceY1 = face.y + face.h;
  const c = (v: number, lo: number, hi: number) =>
    v < lo ? lo : v > hi ? hi : v;
  const x0 = c(c(cx - half, face.x, faceX1 - 1), 0, frameW - 1);
  const y0 = c(c(cy - half, face.y, faceY1 - 1), 0, frameH - 1);
  const x1 = c(c(cx + half, face.x, faceX1), 0, frameW);
  const y1 = c(c(cy + half, face.y, faceY1), 0, frameH);
  const w = x1 - x0;
  const h = y1 - y0;
  if (w <= 1 || h <= 1) return null;
  return { x: x0, y: y0, w, h };
}
