export type FaceClass = 'yawn' | 'no_yawn' | 'front' | 'down';
export type EyeClass = 'Closed' | 'Open';

export const CLASS_NAMES = ['yawn', 'no_yawn', 'Closed', 'Open', 'front', 'down'] as const;
export type ClassName = (typeof CLASS_NAMES)[number];

export interface FaceBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface FacePrediction {
  box: FaceBox;
  faceClass: FaceClass;
  faceConf: number;
  eyes: EyePrediction[];
}

export interface EyePrediction {
  box: FaceBox;
  eyeClass: EyeClass;
  eyeConf: number;
}

export interface DetectionResult {
  faces: FacePrediction[];
  ts: number;
  faceLost: boolean;
}

export type AlertLevel = 'none' | 'eyes-closing' | 'drowsy' | 'warning' | 'critical' | 'emergency';

export interface Settings {
  confidenceThreshold: number;
  emergencyNumber: string;
  alarmVolume: number;
  keepScreenOn: boolean;
}

export interface TripEvent {
  ts: number;
  type: 'yawn' | 'head-down' | 'drowsy' | 'critical' | 'emergency';
}

export interface Trip {
  id: string;
  startedAt: number;
  endedAt: number;
  events: TripEvent[];
  longestClosedMs: number;
}
