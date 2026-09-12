import { useEffect, useMemo, useState } from 'react';
import { BatteryCharging, IndianRupee, RefreshCw, Zap } from 'lucide-react';
import type { SweepPoint } from '../api/client';
import { useSite, useSweep } from '../api/queries';
import { useConsoleParams } from '../lib/url';
import { gwh, inr, mwh } from '../lib/format';
import { Panel, PanelState } from '../components/ui/Panel';
import SweepChart, { findKnee } from '../components/charts/SweepChart';

function useDebounced<T>(value: T, ms = 300) {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setV(value), ms);
    return () => clearTimeout(id);
  }, [value, ms]);
  return v;
}

export default function Planning() {
  const { region, runTs } = useConsoleParams();
  const [maxMwh, setMaxMwh] = useState(1000);
  const [steps, setSteps] = useState(25);
  // Sized against the same run the rest of the console is showing.
  const sw = useSweep(region, useDebounced(maxMwh), useDebounced(steps), runTs);
  const { physicsOnly } = useSite(region);
  const [hover, setHover] = useState<SweepPoint | null>(null);

  const data = sw.data?.data ?? [];
  const knee = useMemo(() => findKnee(data), [data]);
  const allZero = data.length > 0 && data.every((p) => p.curtailment_avoided_gwh_yr === 0);
  const shown = hover ?? knee ?? data[data.length - 1];

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h1 className="text-[22px] font-semibold text-slate-800">Planning · battery sizing</h1>
        <p className="text-xs text-slate-700">Storage is a declared config parameter, so the platform can re-run the decision layer across every size and show where extra MWh stop paying.</p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[320px_minmax(0,1fr)] gap-4 items-start">
        <div className="flex flex-col gap-4">
          <Panel title="Sweep range">
            <div className="flex flex-col gap-5">
              <Slider label="Largest battery" value={maxMwh} min={100} max={5000} step={100} onChange={setMaxMwh} fmt={mwh} />
              <Slider label="Sizes evaluated" value={steps} min={5} max={60} step={1} onChange={setSteps} fmt={(v) => `${v} steps`} />
              {sw.isFetching && (
                <p className="flex items-center gap-1.5 text-[11px] text-slate-500">
                  <RefreshCw className="w-3 h-3 animate-spin" /> Re-running the sweep…
                </p>
              )}
            </div>
          </Panel>

          {shown && !allZero && (
            <Panel title={hover ? 'At this size' : knee ? 'At the knee' : 'At the largest size'} delay={0.05}>
              <dl className="grid grid-cols-2 gap-3">
                <Readout icon={BatteryCharging} label="Capacity" value={mwh(shown.energy_capacity_mwh)} />
                <Readout icon={Zap} label="Curtailment avoided" value={`${gwh(shown.curtailment_avoided_gwh_yr)}/yr`} />
                <Readout icon={IndianRupee} label="Value" value={`${inr(shown.value_inr_yr)}/yr`} />
                <Readout icon={RefreshCw} label="Cycles" value={`${Math.round(shown.cycles_per_year)}/yr`} />
              </dl>
            </Panel>
          )}
        </div>

        <Panel title="What each extra MWh of storage buys" subtitle="Hover to read any size. The knee marks where returns start diminishing." delay={0.08}>
          <PanelState
            q={sw}
            height={480}
            empty={data.length === 0 || allZero}
            emptyTitle={physicsOnly ? 'No sweep for a transfer region' : 'No surplus to store in this run'}
            emptyHint={
              physicsOnly
                ? `${region} has no demand feed, so there is no net-load outlook for storage to act on.`
                : 'The sweep annualises this run’s 72 h outlook. It has no over-generation hour, so every battery size avoids 0 GWh and is worth ₹0. That is a real result, not a failure: pick another run, or wait for one that forecasts surplus.'
            }
          >
            <SweepChart data={data} knee={knee} onHover={setHover} />
          </PanelState>
        </Panel>
      </div>
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange, fmt }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; fmt: (v: number) => string }) {
  return (
    <label className="flex flex-col gap-2">
      <span className="flex justify-between text-xs">
        <span className="font-medium text-slate-600">{label}</span>
        <span className="font-semibold text-slate-800 num">{fmt(value)}</span>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className="accent-blue-600" />
    </label>
  );
}

function Readout({ icon: Icon, label, value }: { icon: typeof Zap; label: string; value: string }) {
  return (
    <div className="rounded-xl bg-white/55 border border-white/70 p-3">
      <dt className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-slate-500">
        <Icon className="w-3 h-3" />
        {label}
      </dt>
      <dd className="mt-1 text-sm font-bold text-slate-800 num">{value}</dd>
    </div>
  );
}
