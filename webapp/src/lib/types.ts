// Mirrors lib/models/types.dart from the Flutter app.

export type FaceClassName = "yawn" | "no_yawn" | "front" | "down";
export type EyeClassName = "Closed" | "Open";

export enum AlertLevel {
  none = "none",
  eyesClosing = "eyesClosing",
  drowsy = "drowsy",
  warning = "warning",
  critical = "critical",
  emergency = "emergency",
}

export type EventType =
  | "yawn"
  | "headDown"
  | "drowsy"
  | "critical"
  | "emergency";

export interface FaceBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface EyePrediction {
  box: FaceBox;
  eyeClass: EyeClassName;
  conf: number;
}

export interface FacePrediction {
  box: FaceBox;
  // The most "alarming" signal wins for the single display label.
  faceClass: FaceClassName;
  conf: number;
  // Independent binaries (yawn-vs-noYawn, front-vs-down).
  isYawn: boolean;
  yawnConf: number;
  isHeadDown: boolean;
  headPoseConf: number;
  eyes: EyePrediction[];
}

export interface DetectionResult {
  faces: FacePrediction[];
  frameWidth: number;
  frameHeight: number;
  tsMs: number;
  readonly faceLost: boolean;
}

export function makeResult(
  faces: FacePrediction[],
  frameWidth: number,
  frameHeight: number,
  tsMs: number,
): DetectionResult {
  return {
    faces,
    frameWidth,
    frameHeight,
    tsMs,
    get faceLost() {
      return this.faces.length === 0;
    },
  };
}

export interface TripEvent {
  ts: number;
  type: EventType;
}

export interface Trip {
  id: string;
  startedAt: number;
  endedAt: number;
  events: TripEvent[];
  longestClosedMs: number;
}

export interface AppSettings {
  confidenceThreshold: number; // 0..1
  emergencyNumber: string;
  alarmVolume: number; // 0..1
  keepScreenOn: boolean;
}

export const defaultSettings: AppSettings = {
  confidenceThreshold: 0.6,
  emergencyNumber: "112",
  alarmVolume: 1.0,
  keepScreenOn: true,
};
