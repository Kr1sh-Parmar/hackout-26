import { motion } from 'framer-motion';
import type { GridEvent } from '../../api/client';
import { FLAG, sevStatus, STATUS, statusVar } from '../../lib/domain';
import { mw, mwh, window_ } from '../../lib/format';
import { H } from '../charts/common';

interface Props {
  events: GridEvent[];
  start: number; // first forecast hour (ms)
  end: number; // last forecast hour (ms)
  activeId?: string | null;
  onHover: (id: string | null) => void;
  onSelect: (t: number) => void;
}

export default function EventTimeline({ events, start, end, activeId, onHover, onSelect }: Props) {
  const lo = start - H / 2;
  const span = end + H / 2 - lo;
  const pos = (t: number) => Math.min(100, Math.max(0, ((t - lo) / span) * 100));
  const now = Date.now();
  const showNow = now > lo && now < lo + span;

  return (
    <ol className="flex flex-col gap-2">
      {events.map((e, i) => {
        const s = sevStatus(e.severity);
        const flag = FLAG[e.flag] ?? { label: e.flag, icon: FLAG.NORMAL.icon };
        const from = Date.parse(e.valid_from) - H / 2;
        const to = Date.parse(e.valid_to) + H / 2;
        const on = activeId === e.event_id;
        return (
          <motion.li key={e.event_id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 * i }}>
            <button
              onClick={() => onSelect(Date.parse(e.valid_from))}
              onMouseEnter={() => onHover(e.event_id)}
              onMouseLeave={() => onHover(null)}
              onFocus={() => onHover(e.event_id)}
              onBlur={() => onHover(null)}
              className={`w-full text-left rounded-xl px-3 py-2.5 border transition-colors ${on ? 'bg-white/85 border-white shadow' : 'bg-white/45 border-white/60 hover:bg-white/70'}`}
            >
              <div className="flex items-center gap-2 text-xs">
                <flag.icon className="w-3.5 h-3.5 shrink-0" style={{ color: statusVar(s) }} />
                <span className="font-semibold text-slate-800">{flag.label}</span>
                <span className={`px-1.5 py-0.5 rounded-full border text-[10px] font-semibold ${STATUS[s].chip}`}>
                  Sev {e.severity} · {STATUS[s].label}
                </span>
                <span className="ml-auto font-mono text-[11px] text-slate-500">T+{e.lead_hours}h</span>
              </div>
              <div className="mt-1 flex justify-between gap-2 text-[11px] text-slate-500 num">
                <span>{window_(e.valid_from, e.valid_to)}</span>
                <span>
                  {mw(e.magnitude_mw)} · {mwh(e.energy_mwh)}
                </span>
              </div>
              {/* Where in the 72 h horizon this sits. */}
              <div className="relative mt-2 h-1.5 rounded-full bg-slate-200/70">
                {showNow && <span className="absolute -top-1 w-px h-3.5 bg-blue-600" style={{ left: `${pos(now)}%` }} title="now" />}
                <span
                  className="absolute inset-y-0 rounded-full min-w-[4px]"
                  style={{ left: `${pos(from)}%`, width: `${pos(to) - pos(from)}%`, background: statusVar(s) }}
                />
              </div>
            </button>
          </motion.li>
        );
      })}
    </ol>
  );
}
