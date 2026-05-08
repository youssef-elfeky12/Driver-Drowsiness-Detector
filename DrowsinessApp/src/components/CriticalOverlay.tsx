import { Siren } from 'lucide-react';

export function CriticalOverlay({
  countdown,
  onCancel,
}: {
  countdown: number;
  onCancel: () => void;
}) {
  return (
    <div className="absolute inset-0 z-40 flex flex-col items-center justify-center p-8 animate-pulse-red">
      <div className="absolute inset-0 bg-danger/50" />
      <div className="relative flex flex-col items-center text-text">
        <Siren size={96} strokeWidth={2.4} className="mb-3 text-text" />
        <div className="text-3xl font-extrabold tracking-tight text-center">
          EMERGENCY
        </div>
        <div className="text-base font-semibold mt-1 opacity-90">CALLING 112 IN</div>
        <div className="mt-3 text-[120px] leading-none font-extrabold tabular drop-shadow-lg">
          {countdown}
        </div>
        <button
          onClick={onCancel}
          className="mt-8 px-10 py-4 rounded-2xl bg-bg/80 text-text text-lg font-semibold backdrop-blur"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
