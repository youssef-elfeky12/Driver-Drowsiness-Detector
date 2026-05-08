import { Eye } from 'lucide-react';

export function WarningOverlay({ closedMs }: { closedMs: number }) {
  return (
    <div className="absolute inset-0 z-30 bg-amber/80 text-bg flex flex-col items-center justify-center p-8 animate-slide-up">
      <Eye size={96} strokeWidth={2.2} className="mb-4" />
      <div className="text-4xl font-extrabold tracking-tight">EYES CLOSED</div>
      <div className="text-2xl font-semibold mt-1">WAKE UP</div>
      <div className="mt-6 text-base font-medium opacity-80 tabular">
        {(closedMs / 1000).toFixed(1)}s
      </div>
    </div>
  );
}
