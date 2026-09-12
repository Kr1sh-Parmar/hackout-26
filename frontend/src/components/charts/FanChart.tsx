import { useMemo } from 'react';
import { useReducedMotion } from 'framer-motion';
import { Area, CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { ForecastPoint, GridEvent, Tech } from '../../api/client';
import { sevStatus, statusVar, TECH_COLOR } from '../../lib/domain';
import { localTime, mw } from '../../lib/format';
import { axisTick, gw, H, HatchDefs, ranges, timeAxisProps, TipRow, TooltipCard } from './common';

export interface FanRow {
  t: number;
  lead: number;
  night: boolean;
  solarBand?: [number, number];
  solarP50?: number;
  solarCal?: boolean;
  windBand?: [number, number];
  windP50?: number;
  windCal?: boolean;
}

/** One row per valid hour, both technologies side by side (Recharts wants bands as [lo, hi] tuples). */
export function toFanRows(points: ForecastPoint[]): FanRow[] {
  const byT = new Map<number, FanRow>();
  for (const p of points) {
    const t = Date.parse(p.valid_ts_utc);
    const r = byT.get(t) ?? { t, lead: p.lead_hours, night: false };
    if (p.tech === 'solar') {
      Object.assign(r, { solarBand: [p.p10_mw, p.p90_mw], solarP50: p.p50_mw, solarCal: p.calibrated, night: p.p90_mw <= 0 });
    } else {
      Object.assign(r, { windBand: [p.p10_mw, p.p90_mw], windP50: p.p50_mw, windCal: p.calibrated });
    }
    byT.set(t, r);
  }
  return [...byT.values()].sort((a, b) => a.t - b.t);
}

interface Props {
  rows: FanRow[];
  techs: Tech[];
  events?: GridEvent[];
  activeEventId?: string | null;
  selected?: number | null;
  onSelect?: (t: number) => void;
  height?: number;
}

export default function FanChart({ rows, techs, events = [], activeEventId, selected, onSelect, height = 320 }: Props) {
  const reduce = useReducedMotion();
  const show = (t: Tech) => techs.includes(t);
  const nights = useMemo(() => ranges(rows, (r) => r.night), [rows]);
  const uncalibrated = useMemo(
    () => ranges(rows, (r) => techs.some((t) => (t === 'solar' ? r.solarCal : r.windCal) === false)),
    [rows, techs],
  );
  // Not memoised: ticks depend on the display timezone too, which can arrive after the rows do.
  const axis = timeAxisProps(rows.map((r) => r.t));

  return (
    <div style={{ height }} className={onSelect ? 'cursor-crosshair' : undefined}>
      <ResponsiveContainer>
        <ComposedChart
          data={rows}
          margin={{ top: 8, right: 12, left: 0, bottom: 0 }}
          onClick={(s) => s?.activeLabel != null && onSelect?.(Number(s.activeLabel))}
        >
          <HatchDefs />
          <CartesianGrid stroke="var(--grid-line)" vertical={false} />
          {/* Night shading explains why solar is zero for a third of the chart. */}
          {nights.map(([a, b]) => (
            <ReferenceArea key={`n${a}`} x1={a} x2={b} fill="var(--baseline)" fillOpacity={0.14} stroke="none" ifOverflow="hidden" />
          ))}
          {uncalibrated.map(([a, b]) => (
            <ReferenceArea
              key={`u${a}`}
              x1={a}
              x2={b}
              fill="url(#hatch)"
              fillOpacity={0.45}
              stroke="none"
              ifOverflow="hidden"
              label={{ value: 'uncalibrated', position: 'insideTopLeft', fill: 'var(--ink-muted)', fontSize: 10 }}
            />
          ))}
          {events.map((e) => {
            const on = e.event_id === activeEventId;
            const c = statusVar(sevStatus(e.severity));
            return (
              <ReferenceArea
                key={e.event_id}
                x1={Date.parse(e.valid_from) - H / 2}
                x2={Date.parse(e.valid_to) + H / 2}
                fill={c}
                fillOpacity={on ? 0.3 : 0.1}
                stroke={on ? c : 'none'}
                ifOverflow="hidden"
              />
            );
          })}
          <XAxis {...axis} />
          <YAxis tick={axisTick} tickLine={false} axisLine={false} width={64} tickFormatter={gw} />

          {/* Bands first so the medians draw on top. */}
          {show('wind') && <Area dataKey="windBand" stroke="none" fill={TECH_COLOR.wind} fillOpacity={0.2} isAnimationActive={!reduce} />}
          {show('solar') && <Area dataKey="solarBand" stroke="none" fill={TECH_COLOR.solar} fillOpacity={0.22} isAnimationActive={!reduce} />}
          {show('wind') && <Line dataKey="windP50" stroke={TECH_COLOR.wind} strokeWidth={2.25} dot={false} activeDot={{ r: 4 }} isAnimationActive={!reduce} />}
          {show('solar') && <Line dataKey="solarP50" stroke={TECH_COLOR.solar} strokeWidth={2.25} dot={false} activeDot={{ r: 4 }} isAnimationActive={!reduce} />}

          {selected != null && <ReferenceLine x={selected} stroke="#2563eb" strokeWidth={1.5} strokeDasharray="4 3" ifOverflow="hidden" />}
          <Tooltip cursor={{ stroke: 'var(--baseline)' }} content={({ active, payload }) => active && payload?.length ? <FanTip row={payload[0].payload as FanRow} techs={techs} /> : null} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function FanTip({ row, techs }: { row: FanRow; techs: Tech[] }) {
  return (
    <TooltipCard title={`${localTime(new Date(row.t).toISOString(), 'EEE d MMM, HH:mm')} · T+${row.lead}h`}>
      {techs.includes('solar') && row.solarP50 != null && (
        <TipRow color={TECH_COLOR.solar} label="Solar" value={mw(row.solarP50)} sub={`${mw(row.solarBand![0])}–${mw(row.solarBand![1])}`} />
      )}
      {techs.includes('wind') && row.windP50 != null && (
        <TipRow color={TECH_COLOR.wind} label="Wind" value={mw(row.windP50)} sub={`${mw(row.windBand![0])}–${mw(row.windBand![1])}`} />
      )}
      <p className="text-[10px] text-slate-400 mt-1">
        P50 with P10–P90 band · {techs.every((t) => (t === 'solar' ? row.solarCal : row.windCal) !== false) ? 'conformally calibrated' : 'band NOT calibrated'}
        {row.night && ' · night'}
      </p>
    </TooltipCard>
  );
}
