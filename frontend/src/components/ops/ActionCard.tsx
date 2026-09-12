import { useState } from 'react';
import { motion } from 'framer-motion';
import { Check, Loader2, RotateCcw } from 'lucide-react';
import type { Recommendation } from '../../api/client';
import { ACTION, FLAG } from '../../lib/domain';
import { inr, localTime, mw, mwh, window_ } from '../../lib/format';

interface Props {
  action: Recommendation;
  pending: boolean;
  error?: string;
  onToggleAck: () => void;
  onHover: (eventId: string | null) => void;
  onSelect: (t: number) => void;
}

export default function ActionCard({ action: a, pending, error, onToggleAck, onHover, onSelect }: Props) {
  const [open, setOpen] = useState(false);
  const acked = !!a.acknowledged_at;
  const meta = ACTION[a.action] ?? { label: a.action, icon: Check };
  const flag = FLAG[a.flag] ?? { label: a.flag, icon: Check };
  const Icon = meta.icon;

  return (
    <motion.article
      layout
      initial={{ opacity: 0, x: 24 }}
      animate={{ opacity: acked ? 0.6 : 1, x: 0 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
      onMouseEnter={() => onHover(a.linked_event_id ?? null)}
      onMouseLeave={() => onHover(null)}
      onFocus={() => onHover(a.linked_event_id ?? null)}
      onBlur={() => onHover(null)}
      className="relative rounded-2xl bg-white/60 border border-white/70 shadow-sm p-4 hover:shadow-md hover:bg-white/75 transition-[background,box-shadow]"
    >
      <header className="flex items-start gap-3">
        <span className="w-9 h-9 shrink-0 rounded-xl bg-blue-100/70 text-blue-700 flex items-center justify-center">
          <Icon className="w-4 h-4" />
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-[13px] font-bold tracking-wide text-slate-800 uppercase">{meta.label}</p>
          <button onClick={() => onSelect(Date.parse(a.valid_from))} className="text-[11px] text-slate-500 hover:text-blue-700 num" title="Pin this window on the charts">
            {window_(a.valid_from, a.valid_to)} · {localTime(a.valid_from, 'd MMM')}
          </button>
        </div>
        <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-slate-600 bg-white/70 border border-white px-2 py-0.5 rounded-full">
          <flag.icon className="w-3 h-3" />
          {flag.label}
        </span>
      </header>

      <div className="mt-3 flex items-end justify-between gap-3">
        <div>
          <p className="text-2xl font-bold text-slate-800 num leading-none">{mwh(a.mwh)}</p>
          <p className="text-[11px] text-slate-500 num mt-1">at {mw(a.power_mw)}</p>
        </div>
        <div className="text-right">
          <p className="text-lg font-bold text-green-700 num leading-none">{inr(a.value_inr)}</p>
          <p className="text-[11px] text-slate-500 mt-1">value</p>
        </div>
      </div>

      {/* Two-state confidence: an operator needs "safe to act on the forecast alone?", not a probability. */}
      <p className="mt-3 text-[11px] font-semibold flex items-center gap-1.5" title={`model confidence ${(a.confidence * 100).toFixed(0)}%`}>
        <span className="tracking-[0.2em] text-blue-700" aria-hidden>
          {a.decisive ? '●●●' : '●●○'}
        </span>
        <span className={a.decisive ? 'text-blue-800' : 'text-slate-600'}>{a.decisive ? 'Decisive: whole P10–P90 band clears the threshold' : 'Indicative: P50 crosses, band straddles'}</span>
      </p>

      <p className={`mt-2 text-xs text-slate-600 leading-relaxed ${open ? '' : 'line-clamp-2'}`}>{a.rationale}</p>
      <div className="mt-2 flex items-center justify-between gap-2">
        <button onClick={() => setOpen((o) => !o)} className="text-[11px] font-medium text-blue-600 hover:text-blue-800">
          {open ? 'Less' : 'Full rationale'}
        </button>
        <button
          onClick={onToggleAck}
          disabled={pending}
          aria-busy={pending}
          className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-[11px] font-semibold border transition-colors disabled:opacity-60 ${
            acked ? 'bg-white/60 text-slate-600 border-white hover:bg-white' : 'bg-blue-600 text-white border-blue-600 hover:bg-blue-700'
          }`}
        >
          {pending ? <Loader2 className="w-3 h-3 animate-spin" /> : acked ? <RotateCcw className="w-3 h-3" /> : <Check className="w-3 h-3" />}
          {acked ? 'Undo' : 'Acknowledge'}
        </button>
      </div>
      {acked && (
        <p className="mt-2 text-[11px] text-slate-500 num">
          Acknowledged {localTime(a.acknowledged_at!, 'd MMM HH:mm')} · recorded by the API
        </p>
      )}
      {error && (
        <p role="alert" className="mt-2 text-[11px] font-medium text-red-700">
          {error}
        </p>
      )}
    </motion.article>
  );
}
