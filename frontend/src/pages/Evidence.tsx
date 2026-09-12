import { Crosshair, Gauge, Info, ShieldCheck, Target } from 'lucide-react';
import type { Tech } from '../api/client';
import { useBacktest, useSite } from '../api/queries';
import { useConsoleParams } from '../lib/url';
import { TECH_COLOR } from '../lib/domain';
import { pct } from '../lib/format';
import { Panel, PanelState, Skeleton } from '../components/ui/Panel';
import { Segmented } from '../components/ui/Segmented';
import { StatTile } from '../components/ui/StatTile';
import LeadChart from '../components/charts/LeadChart';
import { SeriesKey } from '../components/charts/common';

const pctOr = (v: number | null | undefined, digits = 2) => (v == null ? '—' : (v * 100).toFixed(digits));

export default function Evidence() {
  const { region, param, set } = useConsoleParams();
  const tech: Tech = param('tech') === 'wind' ? 'wind' : 'solar';
  const bt = useBacktest(region, tech);
  const { physicsOnly } = useSite(region);

  const rows = bt.data?.data ?? []; // served in lead-hour order
  // The headline comes from the API, computed by the same summarise() that writes the published
  // report. It is never re-derived here: the obvious client-side pooling gives a different number.
  const s = bt.data?.summary;
  const color = TECH_COLOR[tech];
  const name = tech === 'solar' ? 'Solar' : 'Wind';
  const inTarget = (v: number | null | undefined) => s != null && v != null && v >= s.picp_target_low && v <= s.picp_target_high;
  const emptyProps = {
    empty: rows.length === 0,
    emptyTitle: physicsOnly ? 'No labels, so no accuracy claim' : 'No backtest for this region',
    emptyHint: physicsOnly
      ? `${region} publishes no metered generation to score against. A transfer region proves the pipeline runs, never that it is accurate.`
      : 'Run scripts/backtest.py for this region.',
  };

  const errorSeries = [
    { key: 'nrmse_model', label: 'This model', color, width: 2.75 },
    { key: 'nrmse_persistence', label: 'Persistence', color: 'var(--series-persistence)' },
    { key: 'nrmse_physics', label: 'Physics only', color: 'var(--series-physics)' },
    { key: 'nrmse_tso', label: 'Elia day-ahead', color: 'var(--series-tso)', dash: '6 4' },
    { key: 'nrmse_tso_wa', label: 'Elia week-ahead', color: 'var(--series-tso)', dash: '2 3' },
  ];

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-3">
        <div className="mr-auto">
          <h1 className="text-[22px] font-semibold text-slate-800">Evidence</h1>
          <p className="text-xs text-slate-700">
            {s?.folds ? `${s.folds}-fold walk-forward` : 'Walk-forward'} backtest
            {s ? ` over ${s.n_rows.toLocaleString('en-IN')} scored hours` : ''}, per lead hour, normalised by installed capacity.
          </p>
        </div>
        <Segmented id="ev-tech" label="Technology" value={tech} onChange={(v) => set('tech', v === 'solar' ? null : v)} options={[{ value: 'solar', label: 'Solar' }, { value: 'wind', label: 'Wind' }]} />
      </div>

      <div className="grid grid-cols-2 xl:grid-cols-4 gap-3 md:gap-4">
        {bt.isPending ? (
          [0, 1, 2, 3].map((i) => <Skeleton key={i} height={118} />)
        ) : !s ? null : (
          <>
            <StatTile
              label={`${name} nRMSE${tech === 'solar' ? ' · daylight' : ''}`}
              icon={Crosshair}
              value={pctOr(s.nrmse_mean)}
              unit={s.nrmse_mean == null ? undefined : '%'}
              sub={`row-weighted over ${s.leads_scored} lead hours · persistence ${pctOr(s.nrmse_persistence, 1)}%`}
            />
            <StatTile
              label="Skill vs persistence"
              icon={Gauge}
              value={s.skill_mean == null ? '—' : s.skill_mean.toFixed(3)}
              status={s.skill_mean == null ? undefined : s.skill_mean > 0.4 ? 'good' : 'warning'}
              sub="target above 0.40"
              delay={0.06}
            />
            <StatTile
              label="PICP · 80% band"
              icon={ShieldCheck}
              value={pctOr(s.picp_mean, 1)}
              unit={s.picp_mean == null ? undefined : '%'}
              status={s.picp_mean == null ? undefined : inTarget(s.picp_mean) ? 'good' : 'warning'}
              sub={`target ${pct(s.picp_target_low, 0)}–${pct(s.picp_target_high, 0)}`}
              delay={0.12}
            />
            <StatTile label="Lead hours in band" icon={Target} value={String(s.leads_in_band)} unit={`/ ${s.leads_scored}`} sub="calibrated to within ±2 pp" delay={0.18} />
          </>
        )}
      </div>

      <Panel
        title="Error by lead hour"
        subtitle={`nRMSE as a share of installed capacity${tech === 'solar' ? ', daylight hours only (night zeros would halve it and mean nothing)' : ''}.`}
        right={<SeriesKey items={errorSeries.map((x) => ({ color: x.color, label: x.label, kind: x.dash ? 'dashed' : 'line' }))} />}
      >
        <PanelState q={bt} height={300} {...emptyProps}>
          <LeadChart data={rows} series={errorSeries} yFormat={(v) => pct(v, 1)} height={300} />
          <p className="mt-3 flex gap-2 text-xs text-slate-500">
            <Info className="w-3.5 h-3.5 mt-0.5 shrink-0" />
            Elia's day-ahead is issued around 18:00 the day before, so it is effectively a 6–30 h product. Its week-ahead is 144 h and beyond. A 24–72 h
            forecast belongs between the two, and that is where it sits. We do not claim to beat the TSO.
          </p>
        </PanelState>
      </Panel>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <Panel title="Skill vs persistence" subtitle="1 − model RMSE / persistence RMSE. Above 0 beats the naive baseline." delay={0.05}>
          <PanelState q={bt} height={240} {...emptyProps}>
            <LeadChart
              data={rows}
              series={[{ key: 'skill_vs_persistence', label: 'Skill', color, width: 2.5 }]}
              yFormat={(v) => v.toFixed(2)}
              refs={[{ y: 0, label: 'no skill' }, { y: 0.4, label: 'target 0.40', color: 'var(--status-good)' }]}
            />
          </PanelState>
        </Panel>
        <Panel title="Interval coverage" subtitle="How often the truth landed inside the P10–P90 band. It should be 80%: honest bands sit in the green strip." delay={0.1}>
          <PanelState q={bt} height={240} {...emptyProps}>
            <LeadChart
              data={rows}
              series={[{ key: 'picp_80', label: 'Observed coverage', color, width: 2.5 }]}
              yFormat={(v) => pct(v, 0)}
              band={s ? { y1: s.picp_target_low, y2: s.picp_target_high, label: `target ${pct(s.picp_target_low, 0)}–${pct(s.picp_target_high, 0)}` } : undefined}
              refs={[{ y: 0.8, label: 'nominal 80%' }]}
            />
          </PanelState>
        </Panel>
      </div>

      <Panel title="Sharpness" subtitle="Mean P10–P90 width as a share of capacity. Coverage is easy with wide bands; being right and narrow is the hard part." delay={0.15}>
        <PanelState q={bt} height={200} {...emptyProps}>
          <LeadChart data={rows} series={[{ key: 'mean_width_frac', label: 'Mean band width', color, width: 2.5 }]} yFormat={(v) => pct(v, 1)} height={200} />
        </PanelState>
      </Panel>
    </div>
  );
}
