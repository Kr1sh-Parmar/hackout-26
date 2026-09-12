import { useReducedMotion } from 'framer-motion';
import { Area, AreaChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { SweepPoint } from '../../api/client';
import { axisTick } from './common';

/** Diminishing returns: the last size before the marginal GWh per MWh drops under a quarter of the first step's. */
export function findKnee(d: SweepPoint[]): SweepPoint | null {
  if (d.length < 3) return null;
  const slope = (i: number) => (d[i].curtailment_avoided_gwh_yr - d[i - 1].curtailment_avoided_gwh_yr) / (d[i].energy_capacity_mwh - d[i - 1].energy_capacity_mwh || 1);
  const first = slope(1);
  if (!(first > 0)) return null;
  for (let i = 2; i < d.length; i++) if (slope(i) < first * 0.25) return d[i - 1];
  return null;
}

interface Props {
  data: SweepPoint[];
  knee: SweepPoint | null;
  onHover: (p: SweepPoint | null) => void;
}

// Three measures on three scales become three stacked charts sharing an x-axis. Never a dual axis.
export default function SweepChart({ data, knee, onHover }: Props) {
  const reduce = useReducedMotion();
  const common = {
    data,
    syncId: 'sweep',
    margin: { top: 8, right: 16, left: 0, bottom: 0 },
    onMouseMove: (s: { activeTooltipIndex?: number }) => onHover(s.activeTooltipIndex != null ? data[s.activeTooltipIndex] ?? null : null),
    onMouseLeave: () => onHover(null),
  };
  const x = (show: boolean) => (
    <XAxis
      dataKey="energy_capacity_mwh"
      type="number"
      domain={['dataMin', 'dataMax']}
      tick={show ? axisTick : false}
      height={show ? 24 : 4}
      tickLine={false}
      axisLine={{ stroke: 'var(--baseline)' }}
      tickFormatter={(v: number) => `${v.toLocaleString('en-IN')} MWh`}
    />
  );
  const kneeLine = knee && <ReferenceLine x={knee.energy_capacity_mwh} stroke="#2563eb" strokeDasharray="4 3" label={{ value: 'knee', position: 'insideTopRight', fill: '#2563eb', fontSize: 10, fontWeight: 600 }} />;
  const tip = <Tooltip cursor={{ stroke: 'var(--baseline)' }} content={() => null} />;

  return (
    <div className="flex flex-col gap-1">
      <Label text="Curtailment avoided · GWh / yr" />
      <div style={{ height: 200 }}>
        <ResponsiveContainer>
          <AreaChart {...common}>
            <defs>
              <linearGradient id="sweepFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#2563eb" stopOpacity={0.35} />
                <stop offset="100%" stopColor="#2563eb" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="var(--grid-line)" vertical={false} />
            {x(false)}
            <YAxis tick={axisTick} tickLine={false} axisLine={false} width={52} />
            {kneeLine}
            {tip}
            <Area dataKey="curtailment_avoided_gwh_yr" stroke="#2563eb" strokeWidth={2.5} fill="url(#sweepFill)" isAnimationActive={!reduce} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <Label text="Value · ₹ crore / yr" />
      <div style={{ height: 120 }}>
        <ResponsiveContainer>
          <LineChart {...common}>
            <CartesianGrid stroke="var(--grid-line)" vertical={false} />
            {x(false)}
            <YAxis tick={axisTick} tickLine={false} axisLine={false} width={52} tickFormatter={(v: number) => (v / 1e7).toFixed(1)} />
            {kneeLine}
            {tip}
            <Line dataKey="value_inr_yr" stroke="var(--ink)" strokeWidth={2} dot={false} isAnimationActive={!reduce} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <Label text="Full cycles / yr" />
      <div style={{ height: 130 }}>
        <ResponsiveContainer>
          <LineChart {...common}>
            <CartesianGrid stroke="var(--grid-line)" vertical={false} />
            {x(true)}
            <YAxis tick={axisTick} tickLine={false} axisLine={false} width={52} />
            {kneeLine}
            {tip}
            <Line dataKey="cycles_per_year" stroke="var(--ink-muted)" strokeWidth={2} dot={false} isAnimationActive={!reduce} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

const Label = ({ text }: { text: string }) => <p className="text-[11px] font-medium uppercase tracking-wider text-slate-500 pl-1">{text}</p>;
