import { useMemo } from 'react';
import { useReducedMotion } from 'framer-motion';
import { Area, CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { GridEvent, OutlookPoint } from '../../api/client';
import { sevStatus, statusVar, TECH_COLOR } from '../../lib/domain';
import { localTime, mw, mwh } from '../../lib/format';
import { axisTick, gw, H, ranges, timeAxisProps, TipRow, TooltipCard } from './common';

interface Row {
  t: number;
  p: OutlookPoint;
  netBand: [number, number];
  windStack: [number, number];
  solarStack: [number, number];
  demand: number;
  net: number;
  mustRun: number;
  surplus: boolean;
}

interface Props {
  data: OutlookPoint[];
  events?: GridEvent[];
  activeEventId?: string | null;
  selected?: number | null;
  onSelect?: (t: number) => void;
  height?: number;
}

export default function NetLoadChart({ data, events = [], activeEventId, selected, onSelect, height = 280 }: Props) {
  const reduce = useReducedMotion();
  const rows: Row[] = useMemo(
    () =>
      data.map((p) => ({
        t: Date.parse(p.valid_ts_utc),
        p,
        netBand: [p.net_load_p10_mw, p.net_load_p90_mw],
        // Demand envelope: net load at the bottom, then what wind and solar subtract from demand.
        windStack: [p.net_load_mw, p.net_load_mw + p.wind_p50_mw],
        solarStack: [p.net_load_mw + p.wind_p50_mw, p.demand_mw],
        demand: p.demand_mw,
        net: p.net_load_mw,
        mustRun: p.must_run_mw,
        surplus: p.headroom_mw < 0,
      })),
    [data],
  );
  // Negative headroom: renewables push net load under the must-run floor. Label it in MWh --
  // a colour with no number attached is not actionable.
  const surplus = useMemo(
    () =>
      ranges(rows, (r) => r.surplus).map(([a, b]) => ({
        a,
        b,
        mwh: rows.filter((r) => r.t > a && r.t < b).reduce((s, r) => s - r.p.headroom_mw, 0),
      })),
    [rows],
  );
  // Not memoised: ticks depend on the display timezone too, which can arrive after the rows do.
  const axis = timeAxisProps(rows.map((r) => r.t));

  return (
    <div style={{ height }} className={onSelect ? 'cursor-crosshair' : undefined}>
      <ResponsiveContainer>
        <ComposedChart data={rows} margin={{ top: 16, right: 12, left: 0, bottom: 0 }} onClick={(s) => s?.activeLabel != null && onSelect?.(Number(s.activeLabel))}>
          <CartesianGrid stroke="var(--grid-line)" vertical={false} />
          {events.map((e) => {
            const on = e.event_id === activeEventId;
            const c = statusVar(sevStatus(e.severity));
            return (
              <ReferenceArea key={e.event_id} x1={Date.parse(e.valid_from) - H / 2} x2={Date.parse(e.valid_to) + H / 2} fill={c} fillOpacity={on ? 0.3 : 0.1} stroke={on ? c : 'none'} ifOverflow="hidden" />
            );
          })}
          {surplus.map((s) => (
            <ReferenceArea
              key={`s${s.a}`}
              x1={s.a}
              x2={s.b}
              fill="var(--status-critical)"
              fillOpacity={0.16}
              stroke="none"
              ifOverflow="hidden"
              label={{ value: `${mwh(s.mwh)} over-gen`, position: 'insideTop', fill: 'var(--status-critical)', fontSize: 10, fontWeight: 600 }}
            />
          ))}
          <XAxis {...axis} />
          <YAxis tick={axisTick} tickLine={false} axisLine={false} width={64} tickFormatter={gw} domain={[0, 'auto']} />

          <Area dataKey="solarStack" stroke="none" fill={TECH_COLOR.solar} fillOpacity={0.28} isAnimationActive={!reduce} />
          <Area dataKey="windStack" stroke="none" fill={TECH_COLOR.wind} fillOpacity={0.24} isAnimationActive={!reduce} />
          <Area dataKey="netBand" stroke="none" fill="var(--series-net)" fillOpacity={0.22} isAnimationActive={!reduce} />
          <Line dataKey="demand" stroke="var(--ink)" strokeWidth={1.5} dot={false} isAnimationActive={!reduce} />
          <Line dataKey="net" stroke="var(--series-net)" strokeWidth={2.25} dot={false} activeDot={{ r: 4 }} isAnimationActive={!reduce} />
          <Line dataKey="mustRun" stroke="var(--status-critical)" strokeWidth={1.5} strokeDasharray="6 4" dot={false} isAnimationActive={false} />

          {selected != null && <ReferenceLine x={selected} stroke="#2563eb" strokeWidth={1.5} strokeDasharray="4 3" ifOverflow="hidden" />}
          <Tooltip cursor={{ stroke: 'var(--baseline)' }} content={({ active, payload }) => (active && payload?.length ? <NetTip r={payload[0].payload as Row} /> : null)} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function NetTip({ r }: { r: Row }) {
  const p = r.p;
  return (
    <TooltipCard title={`${localTime(p.valid_ts_utc, 'EEE d MMM, HH:mm')} · T+${p.lead_hours}h`}>
      <TipRow color="var(--ink)" label="Demand" value={mw(p.demand_mw)} />
      <TipRow color={TECH_COLOR.wind} label="Wind P50" value={mw(p.wind_p50_mw)} />
      <TipRow color={TECH_COLOR.solar} label="Solar P50" value={mw(p.solar_p50_mw)} />
      <TipRow color="var(--series-net)" label="Net load" value={mw(p.net_load_mw)} sub={`${mw(p.net_load_p10_mw)}–${mw(p.net_load_p90_mw)}`} />
      <TipRow color="var(--status-critical)" dashed label="Must-run" value={mw(p.must_run_mw)} />
      <p className={`text-[10px] mt-1 font-semibold ${p.headroom_mw < 0 ? 'text-red-700' : 'text-slate-500'}`}>
        Headroom {mw(p.headroom_mw)}
        {p.ramp_mw_per_h != null && <span className="font-normal text-slate-400"> · ramp {mw(p.ramp_mw_per_h)}/h</span>}
      </p>
    </TooltipCard>
  );
}
