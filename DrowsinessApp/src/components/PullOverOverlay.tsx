import { AlertTriangle } from 'lucide-react';

export function PullOverOverlay({ onDismiss }: { onDismiss: () => void }) {
  return (
    <div className="absolute inset-0 z-30 bg-amber/95 text-bg flex flex-col items-center justify-center p-8 animate-slide-up">
      <AlertTriangle size={96} strokeWidth={2.2} className="mb-4" />
      <div className="text-5xl font-extrabold tracking-tight">PULL OVER</div>
      <div className="mt-3 text-lg font-medium opacity-80 text-center max-w-xs">
        Multiple drowsiness signals detected. Find a safe place to stop and rest.
      </div>
      <button
        onClick={onDismiss}
        className="mt-10 px-8 py-4 rounded-2xl bg-bg text-text text-lg font-semibold shadow-lg hover:bg-surface2 transition"
      >
        I'm OK
      </button>
    </div>
  );
}
