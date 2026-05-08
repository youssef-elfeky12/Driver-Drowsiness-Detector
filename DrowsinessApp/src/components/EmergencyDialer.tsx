import { Phone, PhoneOff } from 'lucide-react';

interface Props {
  digitsTyped: string;
  number: string;
  callingActive: boolean;
  callConnected: boolean;
  onCancel: () => void;
}

export function EmergencyDialer({ digitsTyped, number, callingActive, callConnected, onCancel }: Props) {
  return (
    <div className="absolute right-3 bottom-3 z-50 w-[260px] rounded-2xl overflow-hidden bg-[#0E0E0E] border border-white/10 shadow-2xl animate-slide-in-right">
      <div className="bg-gradient-to-b from-[#1a1a1a] to-[#0E0E0E] px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-ok animate-pulse" />
          <span className="text-[11px] font-mono text-muted">PHONE</span>
        </div>
        <span className="text-[10px] font-mono text-muted">EMERGENCY</span>
      </div>

      <div className="bg-[#0a0a0a] px-4 py-6 text-center">
        <div className="text-[11px] uppercase tracking-widest text-muted mb-1">
          {callConnected ? 'Connected' : callingActive ? 'Calling…' : 'Dialing'}
        </div>
        <div className="text-3xl font-extrabold tabular tracking-wider text-text min-h-[40px]">
          {digitsTyped || ' '}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 p-3 bg-[#0E0E0E]">
        {['1', '2', '3', '4', '5', '6', '7', '8', '9', '*', '0', '#'].map((k) => {
          const isLit = digitsTyped.includes(k) && number.includes(k);
          return (
            <div
              key={k}
              className={`h-11 rounded-xl flex items-center justify-center font-semibold text-lg transition ${
                isLit ? 'bg-primary/30 text-primary border border-primary/60' : 'bg-[#1a1a1a] text-muted'
              }`}
            >
              {k}
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-2 bg-[#0E0E0E] border-t border-white/5">
        <div className="p-3 flex items-center justify-center gap-2 text-ok">
          <Phone size={16} />
          <span className="text-xs font-semibold">{callConnected ? 'On call' : 'Calling'}</span>
        </div>
        <button
          onClick={onCancel}
          className="p-3 flex items-center justify-center gap-2 text-danger hover:bg-danger/10 transition"
        >
          <PhoneOff size={16} />
          <span className="text-xs font-semibold">End</span>
        </button>
      </div>
    </div>
  );
}
