import * as tf from '@tensorflow/tfjs';
import type { DetectionResult, EyeClass, EyePrediction, FaceBox, FaceClass, FacePrediction } from '../types';

declare const cv: any;

const MODEL_URL = '/models/efficientnet_b0/model.json';
const FACE_CASCADE_URL = '/haarcascades/haarcascade_frontalface_default.xml';
const EYE_CASCADE_URL = '/haarcascades/haarcascade_eye.xml';

const IMG_SIZE = 224;

// Class index map (from notebook):
// 0=yawn, 1=no_yawn, 2=Closed, 3=Open, 4=front, 5=down
const FACE_INDICES = [0, 1, 4, 5] as const; // softmax subset for face crop
const FACE_LABELS: FaceClass[] = ['yawn', 'no_yawn', 'front', 'down'];
const EYE_INDICES = [2, 3] as const;
const EYE_LABELS: EyeClass[] = ['Closed', 'Open'];

let model: tf.GraphModel | null = null;
let faceCascade: any = null;
let eyeCascade: any = null;
let opencvReady = false;

export function isReady(): boolean {
  return model !== null && opencvReady && faceCascade !== null && eyeCascade !== null;
}

async function waitForOpenCV(): Promise<void> {
  if (opencvReady) return;
  await new Promise<void>((resolve) => {
    const check = () => {
      if ((window as any).cv && (window as any).cv.Mat) {
        opencvReady = true;
        resolve();
      } else {
        setTimeout(check, 100);
      }
    };
    check();
  });
}

async function fetchToFS(url: string, name: string): Promise<void> {
  const resp = await fetch(url);
  const buf = await resp.arrayBuffer();
  cv.FS_createDataFile('/', name, new Uint8Array(buf), true, false, false);
}

export async function initDetector(onProgress?: (msg: string) => void): Promise<void> {
  onProgress?.('Loading TensorFlow.js…');
  await tf.ready();
  await tf.setBackend('webgl');

  onProgress?.('Loading model…');
  model = await tf.loadGraphModel(MODEL_URL);
  // warm up
  const dummy = tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]);
  (model.predict(dummy) as tf.Tensor).dispose();
  dummy.dispose();

  onProgress?.('Loading OpenCV…');
  await waitForOpenCV();

  onProgress?.('Loading Haar cascades…');
  await fetchToFS(FACE_CASCADE_URL, 'face.xml');
  await fetchToFS(EYE_CASCADE_URL, 'eye.xml');
  faceCascade = new cv.CascadeClassifier();
  eyeCascade = new cv.CascadeClassifier();
  faceCascade.load('face.xml');
  eyeCascade.load('eye.xml');

  onProgress?.('Ready');
}

function softmaxOver(probs: Float32Array | number[], indices: readonly number[]): number[] {
  const subset = indices.map((i) => probs[i]);
  // already softmaxed by model — just renormalize
  const sum = subset.reduce((a, b) => a + b, 0) || 1;
  return subset.map((v) => v / sum);
}

async function classifyCrop(
  rgbaMat: any,
  box: FaceBox,
  imageData: ImageData,
): Promise<Float32Array> {
  // Crop from imageData (CPU) — faster than going through cv.Mat for tensor build.
  const { x, y, w, h } = box;
  const w2 = imageData.width;
  // Build a small offscreen canvas to crop+resize
  const off = document.createElement('canvas');
  off.width = IMG_SIZE;
  off.height = IMG_SIZE;
  const octx = off.getContext('2d')!;
  // We use a separate canvas to draw the source first, then crop via drawImage
  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = w2;
  srcCanvas.height = imageData.height;
  srcCanvas.getContext('2d')!.putImageData(imageData, 0, 0);
  octx.drawImage(srcCanvas, x, y, w, h, 0, 0, IMG_SIZE, IMG_SIZE);

  return tf.tidy(() => {
    const t = tf.browser.fromPixels(off).toFloat().expandDims(0);
    const out = model!.predict(t) as tf.Tensor;
    return out.dataSync() as Float32Array;
  });
}

function detectFaces(srcMat: any): FaceBox[] {
  const gray = new cv.Mat();
  cv.cvtColor(srcMat, gray, cv.COLOR_RGBA2GRAY);
  const faces = new cv.RectVector();
  faceCascade.detectMultiScale(gray, faces, 1.3, 5);
  const boxes: FaceBox[] = [];
  for (let i = 0; i < faces.size(); i++) {
    const r = faces.get(i);
    boxes.push({ x: r.x, y: r.y, w: r.width, h: r.height });
  }
  faces.delete();
  gray.delete();
  return boxes;
}

function detectEyes(srcMat: any, face: FaceBox): FaceBox[] {
  const roi = srcMat.roi(new cv.Rect(face.x, face.y, face.w, face.h));
  const gray = new cv.Mat();
  cv.cvtColor(roi, gray, cv.COLOR_RGBA2GRAY);
  const eyes = new cv.RectVector();
  // Eyes are typically in upper half of face
  eyeCascade.detectMultiScale(gray, eyes, 1.1, 5);
  const boxes: FaceBox[] = [];
  for (let i = 0; i < eyes.size(); i++) {
    const r = eyes.get(i);
    // upper-half filter to reduce false positives (mouth/nostrils)
    if (r.y + r.height / 2 < face.h * 0.6) {
      boxes.push({
        x: face.x + r.x,
        y: face.y + r.y,
        w: r.width,
        h: r.height,
      });
    }
  }
  eyes.delete();
  gray.delete();
  roi.delete();
  // keep at most 2 largest
  return boxes.sort((a, b) => b.w * b.h - a.w * a.h).slice(0, 2);
}

export async function detectFrame(
  video: HTMLVideoElement,
  confThreshold: number,
): Promise<DetectionResult> {
  if (!isReady()) {
    return { faces: [], ts: performance.now(), faceLost: true };
  }

  const w = video.videoWidth;
  const h = video.videoHeight;
  if (w === 0 || h === 0) {
    return { faces: [], ts: performance.now(), faceLost: true };
  }

  const cap = document.createElement('canvas');
  cap.width = w;
  cap.height = h;
  const cctx = cap.getContext('2d')!;
  cctx.drawImage(video, 0, 0, w, h);
  const imageData = cctx.getImageData(0, 0, w, h);

  const src = cv.matFromImageData(imageData);
  const faceBoxes = detectFaces(src);

  const faces: FacePrediction[] = [];
  for (const fb of faceBoxes) {
    const probs = await classifyCrop(src, fb, imageData);
    const faceProbs = softmaxOver(probs, FACE_INDICES);
    let bestI = 0;
    for (let i = 1; i < faceProbs.length; i++) if (faceProbs[i] > faceProbs[bestI]) bestI = i;
    const faceClass = FACE_LABELS[bestI];
    const faceConf = faceProbs[bestI];

    const eyeBoxes = detectEyes(src, fb);
    const eyes: EyePrediction[] = [];
    for (const eb of eyeBoxes) {
      const eprobs = await classifyCrop(src, eb, imageData);
      const ep = softmaxOver(eprobs, EYE_INDICES);
      const ei = ep[0] > ep[1] ? 0 : 1;
      eyes.push({
        box: eb,
        eyeClass: EYE_LABELS[ei],
        eyeConf: ep[ei],
      });
    }

    faces.push({
      box: fb,
      faceClass,
      faceConf,
      eyes,
    });
  }

  src.delete();

  // confidence gate is applied by alertEngine, not here
  void confThreshold;

  return { faces, ts: performance.now(), faceLost: faces.length === 0 };
}
