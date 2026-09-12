import { type ReactNode } from 'react';
import { fmtTime, hourIn } from '../../lib/format';

export const H = 3_600_000;
export const axisTick = { fill: 'var(--ink-muted)', fontSize: 11 };

// Short enough never to wrap in the y-axis gutter, even at India's 70 GW fleet scale.
export const gw = (v: number) => {
  const a = Math.abs(v);
  if (a >= 10_000) return `${Math.round(v / 1000)} GW`;
  if (a >= 1000) return `${(v / 1000).toFixed(1)} GW`;
  return `${Math.round(v)} MW`;
};

/** Region-local ticks every 12 h, with the day name at midnight. */
export function timeAxisProps(ts: number[]) {
  return {
    dataKey: 't',
    type: 'number' as const,
    scale: 'time' as const,
    domain: ['dataMin', 'dataMax'] as [string, string],
    ticks: ts.filter((t) => hourIn(t) % 12 === 0),
    tickFormatter: (t: number) => fmtTime(t, hourIn(t) === 0 ? 'EEE d' : 'HH:mm'),
    tick: axisTick,
    tickLine: false,
    axisLine: { stroke: 'var(--baseline)' },
    minTickGap: 12,
  };
}

/** Contiguous hourly runs where `pred` holds, padded half an hour so a single hour is still visible. */
export function ranges<T extends { t: number }>(rows: T[], pred: (r: T) => boolean): [number, number][] {
  const out: [number, number][] = [];
  let start: number | null = null;
  rows.forEach((r, i) => {
    if (!pred(r)) return;
    start ??= r.t;
    if (i === rows.length - 1 || !pred(rows[i + 1])) {
      out.push([start - H / 2, r.t + H / 2]);
      start = null;
    }
  });
  return out;
}

export const HatchDefs = () => (
  <defs>
    <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
      <line x1="0" y1="0" x2="0" y2="6" stroke="var(--baseline)" strokeWidth="2" />
    </pattern>
  </defs>
);

export function TooltipCard({ title, children }: { title: ReactNode; children: ReactNode }) {
  return (
    <div className="rounded-xl bg-white/85 backdrop-blur-xl border border-white/70 shadow-xl px-3 py-2 text-xs min-w-[200px]">
      <p className="font-semibold text-slate-800 mb-1.5">{title}</p>
      <div className="flex flex-col gap-1">{children}</div>
    </div>
  );
}

export function TipRow({ color, label, value, sub, dashed }: { color: string; label: string; value: ReactNode; sub?: ReactNode; dashed?: boolean }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-3 shrink-0" style={{ borderTop: `2px ${dashed ? 'dashed' : 'solid'} ${color}` }} />
      <span className="text-slate-500">{label}</span>
      <span className="ml-auto font-semibold text-slate-800 num">{value}</span>
      {sub && <span className="text-slate-400 num">{sub}</span>}
    </div>
  );
}

/** Legend: identity is carried by the text label, never by colour alone. */
export function SeriesKey({ items }: { items: { color: string; label: string; kind?: 'line' | 'band' | 'dashed' | 'hatch' | 'shade' }[] }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-600">
      {items.map((it) => (
        <li key={it.label} className="flex items-center gap-1.5">
          {it.kind === 'band' || it.kind === 'shade' ? (
            <span className="w-3.5 h-2.5 rounded-sm" style={{ background: it.color, opacity: it.kind === 'band' ? 0.35 : 0.25 }} />
          ) : it.kind === 'hatch' ? (
            <span className="w-3.5 h-2.5 rounded-sm border border-slate-300" style={{ background: 'repeating-linear-gradient(45deg, var(--baseline) 0 2px, transparent 2px 5px)' }} />
          ) : (
            <span className="w-4" style={{ borderTop: `2px ${it.kind === 'dashed' ? 'dashed' : 'solid'} ${it.color}` }} />
          )}
          {it.label}
        </li>
      ))}
    </ul>
  );
}
