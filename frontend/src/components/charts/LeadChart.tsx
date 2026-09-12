import { useReducedMotion } from 'framer-motion';
import { CartesianGrid, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { axisTick, TipRow, TooltipCard } from './common';

export interface LeadSeries {
  key: string;
  label: string;
  color: string;
  dash?: string;
  width?: number;
}

interface Props {
  data: object[]; // rows keyed by `lead_hours`
  series: LeadSeries[];
  yFormat: (v: number) => string;
  band?: { y1: number; y2: number; label: string };
  refs?: { y: number; label: string; color?: string }[];
  height?: number;
}

/** Any metric against lead hour. Accuracy is reported per lead hour, never as one 72 h average. */
export default function LeadChart({ data, series, yFormat, band, refs = [], height = 240 }: Props) {
  const reduce = useReducedMotion();
  return (
    <div style={{ height }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 10, right: 16, left: 0, bottom: 4 }}>
          <CartesianGrid stroke="var(--grid-line)" vertical={false} />
          {band && (
            <ReferenceArea
              y1={band.y1}
              y2={band.y2}
              fill="var(--status-good)"
              fillOpacity={0.13}
              stroke="none"
              ifOverflow="extendDomain"
              label={{ value: band.label, position: 'insideTopRight', fill: 'var(--status-good)', fontSize: 10, fontWeight: 600 }}
            />
          )}
          {refs.map((r) => (
            <ReferenceLine
              key={r.label}
              y={r.y}
              stroke={r.color ?? 'var(--baseline)'}
              strokeDasharray="4 3"
              ifOverflow="extendDomain"
              label={{ value: r.label, position: 'insideBottomLeft', fill: 'var(--ink-muted)', fontSize: 10 }}
            />
          ))}
          <XAxis
            dataKey="lead_hours"
            type="number"
            domain={['dataMin', 'dataMax']}
            tick={axisTick}
            tickLine={false}
            axisLine={{ stroke: 'var(--baseline)' }}
            tickFormatter={(v: number) => `${v}h`}
            allowDecimals={false}
          />
          <YAxis tick={axisTick} tickLine={false} axisLine={false} width={52} tickFormatter={yFormat} />
          {series.map((s) => (
            <Line key={s.key} dataKey={s.key} stroke={s.color} strokeWidth={s.width ?? 2} strokeDasharray={s.dash} dot={false} activeDot={{ r: 3.5 }} isAnimationActive={!reduce} />
          ))}
          <Tooltip
            cursor={{ stroke: 'var(--baseline)' }}
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <TooltipCard title={`Lead ${label} h`}>
                  {series.map((s) => {
                    const v = (payload[0].payload as Record<string, unknown>)[s.key];
                    return typeof v === 'number' ? <TipRow key={s.key} color={s.color} dashed={!!s.dash} label={s.label} value={yFormat(v)} /> : null;
                  })}
                </TooltipCard>
              ) : null
            }
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
