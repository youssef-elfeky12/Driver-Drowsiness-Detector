import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import type { DetectionResult } from '../types';

export interface CameraHandle {
  video: HTMLVideoElement | null;
  drawOverlay: (result: DetectionResult) => void;
}

export const CameraView = forwardRef<CameraHandle, {}>(function CameraView(_props, ref) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useImperativeHandle(ref, () => ({
    get video() {
      return videoRef.current;
    },
    drawOverlay(result) {
      const v = videoRef.current;
      const c = canvasRef.current;
      if (!v || !c || v.videoWidth === 0) return;
      c.width = v.videoWidth;
      c.height = v.videoHeight;
      const ctx = c.getContext('2d');
      if (!ctx) return;
      ctx.clearRect(0, 0, c.width, c.height);

      for (const f of result.faces) {
        // Face box
        ctx.lineWidth = 3;
        ctx.strokeStyle = '#22C55E';
        ctx.strokeRect(f.box.x, f.box.y, f.box.w, f.box.h);
        ctx.fillStyle = 'rgba(34,197,94,0.85)';
        ctx.font = '600 18px Inter, system-ui, sans-serif';
        const label = `face: ${f.faceClass}  ${(f.faceConf * 100).toFixed(0)}%`;
        const tw = ctx.measureText(label).width + 12;
        ctx.fillRect(f.box.x, Math.max(0, f.box.y - 26), tw, 24);
        ctx.fillStyle = '#0B0F14';
        ctx.fillText(label, f.box.x + 6, Math.max(18, f.box.y - 8));

        // Eye boxes
        for (const e of f.eyes) {
          ctx.strokeStyle = e.eyeClass === 'Closed' ? '#EF4444' : '#3B82F6';
          ctx.lineWidth = 2;
          ctx.strokeRect(e.box.x, e.box.y, e.box.w, e.box.h);
          ctx.fillStyle = e.eyeClass === 'Closed' ? 'rgba(239,68,68,0.85)' : 'rgba(59,130,246,0.85)';
          ctx.font = '600 12px Inter, system-ui, sans-serif';
          const elabel = `${e.eyeClass} ${(e.eyeConf * 100).toFixed(0)}%`;
          const ew = ctx.measureText(elabel).width + 8;
          ctx.fillRect(e.box.x, Math.max(0, e.box.y - 18), ew, 16);
          ctx.fillStyle = '#0B0F14';
          ctx.fillText(elabel, e.box.x + 4, Math.max(12, e.box.y - 5));
        }
      }
    },
  }), []);

  useEffect(() => {
    let stream: MediaStream | null = null;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'user',
            width: { ideal: 640 },
            height: { ideal: 480 },
          },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
      } catch (e) {
        console.error('Camera error:', e);
      }
    })();
    return () => {
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return (
    <div className="absolute inset-0 bg-black overflow-hidden">
      <video
        ref={videoRef}
        playsInline
        muted
        className="absolute inset-0 w-full h-full object-cover scale-x-[-1]"
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full object-cover scale-x-[-1] pointer-events-none"
      />
    </div>
  );
});
