import { useMemo, useState } from 'react';
import { ChevronLeft, ChevronRight, Table2, LineChart as ChartIcon } from 'lucide-react';
import type { Tech } from '../api/client';
import { useExplain, useForecast } from '../api/queries';
import { useConsoleParams } from '../lib/url';
import { TECH_COLOR } from '../lib/domain';
import { localTime, mw, pct } from '../lib/format';
import { Panel, PanelState } from '../components/ui/Panel';
import { Segmented } from '../components/ui/Segmented';
import FanChart, { toFanRows, type FanRow } from '../components/charts/FanChart';
import LeadChart from '../components/charts/LeadChart';
import DriversList from '../components/charts/DriversList';
import { SeriesKey } from '../components/charts/common';

type TechSel = 'both' | Tech;

export default function ForecastDetail() {
  const { region, runTs, param, set } = useConsoleParams();
  const techSel = (['solar', 'wind'].includes(param('tech') ?? '') ? param('tech') : 'both') as TechSel;
  const techs: Tech[] = techSel === 'both' ? ['solar', 'wind'] : [techSel];
  const horizon = Number(param('h')) || 72;
  const [table, setTable] = useState(false);

  // Always fetch both technologies and filter in the chart: colour follows the entity, and night
  // shading (derived from solar) survives a wind-only view.
  const fc = useForecast(region, { runTs, horizon });
  const rows = useMemo(() => toFanRows(fc.data?.data ?? []), [fc.data]);
  const capacity = useMemo(() => Object.fromEntries((fc.data?.data ?? []).map((p) => [p.tech, p.capacity_mw])) as Partial<Record<Tech, number>>, [fc.data]);

  const at = param('at');
  const idx = useMemo(() => {
    const i = at ? rows.findIndex((r) => r.t === Date.parse(at)) : -1;
    return i >= 0 ? i : Math.max(0, rows.findIndex((r) => r.lead >= 24 && !r.night));
  }, [rows, at]);
  const row: FanRow | undefined = rows[idx];
  const pick = (i: number) => rows[i] && set('at', new Date(rows[i].t).toISOString());

  const [driverTechPick, setDriverTech] = useState<Tech | null>(null);
  const driverTech: Tech = techs.length === 1 ? techs[0] : (driverTechPick ?? (row && !row.night ? 'solar' : 'wind'));
  const ex = useExplain(region, { tech: driverTech, validTs: row ? new Date(row.t).toISOString() : undefined, runTs });
  const drivers = (ex.data?.data ?? []).filter((d) => d.tech === driverTech && Date.parse(d.valid_ts_utc) === row?.t).sort((a, b) => a.rank - b.rank);
  const correction = drivers[0]?.prediction_cf;

  const widths = useMemo(
    () =>
      rows.map((r) => ({
        lead_hours: r.lead,
        solar: r.solarBand && !r.night && capacity.solar ? (r.solarBand[1] - r.solarBand[0]) / capacity.solar : null,
        wind: r.windBand && capacity.wind ? (r.windBand[1] - r.windBand[0]) / capacity.wind : null,
      })),
    [rows, capacity],
  );

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-[22px] font-semibold text-slate-800 mr-auto">Forecast detail</h1>
        <Segmented
          id="tech"
          label="Technology"
          value={techSel}
          onChange={(v) => set('tech', v === 'both' ? null : v)}
          options={[
            { value: 'both', label: 'Both' },
            { value: 'solar', label: <><Dot c={TECH_COLOR.solar} />Solar</> },
            { value: 'wind', label: <><Dot c={TECH_COLOR.wind} />Wind</> },
          ]}
        />
        <Segmented id="h" label="Horizon" value={horizon} onChange={(v) => set('h', v === 72 ? null : v)} options={[24, 48, 72].map((h) => ({ value: h, label: `${h} h` }))} />
        <Segmented
          id="view"
          label="View"
          value={table ? 'table' : 'chart'}
          onChange={(v) => setTable(v === 'table')}
          options={[
            { value: 'chart', label: <><ChartIcon className="w-3.5 h-3.5" />Chart</> },
            { value: 'table', label: <><Table2 className="w-3.5 h-3.5" />Table</> },
          ]}
        />
      </div>

      <Panel
        title={`Probabilistic forecast · ${horizon} h`}
        subtitle="Drag the scrubber or click the chart to inspect an hour."
        right={<SeriesKey items={[...techs.map((t) => ({ color: TECH_COLOR[t], label: t === 'solar' ? 'Solar P50 · P10–P90' : 'Wind P50 · P10–P90' })), { color: '', label: 'Uncalibrated', kind: 'hatch' as const }]} />}
      >
        <PanelState q={fc} height={380} empty={rows.length === 0} emptyTitle="No forecast for this region yet" emptyHint="Run scripts/run_cycle.py, or start the API in replay mode.">
          {table ? (
            <ForecastTable rows={rows} techs={techs} selected={row?.t} onPick={(t) => pick(rows.findIndex((r) => r.t === t))} />
          ) : (
            <>
              <FanChart rows={rows} techs={techs} selected={row?.t} onSelect={(t) => pick(rows.findIndex((r) => r.t === t))} height={360} />
              <div className="mt-3 flex items-center gap-2">
                <IconBtn label="Previous hour" onClick={() => pick(idx - 1)} disabled={idx <= 0}><ChevronLeft className="w-4 h-4" /></IconBtn>
                <input type="range" min={0} max={rows.length - 1} value={idx} onChange={(e) => pick(Number(e.target.value))} className="flex-1 accent-blue-600" aria-label="Forecast hour" />
                <IconBtn label="Next hour" onClick={() => pick(idx + 1)} disabled={idx >= rows.length - 1}><ChevronRight className="w-4 h-4" /></IconBtn>
              </div>
            </>
          )}
        </PanelState>
      </Panel>

      <div className="grid grid-cols-1 xl:grid-cols-[360px_minmax(0,1fr)] gap-4 items-start">
        <Panel title="Selected hour" delay={0.05}>
          {row ? (
            <div className="flex flex-col gap-4">
              <div>
                <p className="text-2xl font-bold text-slate-800">{localTime(new Date(row.t).toISOString(), 'EEE d MMM, HH:mm')}</p>
                <p className="text-xs text-slate-500 mt-1">
                  Lead T+{row.lead} h · {row.night ? 'night' : 'daylight'}
                </p>
              </div>
              {techs.map((t) => {
                const p50 = t === 'solar' ? row.solarP50 : row.windP50;
                const band = t === 'solar' ? row.solarBand : row.windBand;
                const cal = t === 'solar' ? row.solarCal : row.windCal;
                if (p50 == null || !band) return null;
                return (
                  <div key={t} className="rounded-2xl bg-white/55 border border-white/70 p-3">
                    <div className="flex items-center justify-between text-xs">
                      <span className="flex items-center gap-1.5 font-semibold text-slate-700"><Dot c={TECH_COLOR[t]} />{t === 'solar' ? 'Solar' : 'Wind'}</span>
                      <span className={`px-2 py-0.5 rounded-full border text-[10px] font-semibold ${cal ? 'bg-green-100/70 text-green-800 border-green-200' : 'bg-slate-100/80 text-slate-600 border-slate-200'}`}>
                        {cal ? 'Calibrated band' : 'Band not calibrated'}
                      </span>
                    </div>
                    <p className="mt-2 text-2xl font-bold text-slate-800 num">{mw(p50)}</p>
                    <p className="text-xs text-slate-500 num">P10 {mw(band[0])} · P90 {mw(band[1])}</p>
                    {capacity[t] && <p className="text-[11px] text-slate-400 num mt-0.5">{pct(p50 / capacity[t]!)} of {mw(capacity[t]!)} installed</p>}
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-sm text-slate-500">No hour selected.</p>
          )}
        </Panel>

        <Panel
          title="Why is the forecast here?"
          subtitle="Top drivers of the ML correction on top of physics, in capacity-factor points."
          delay={0.1}
          right={
            techs.length > 1 && (
              <Segmented id="drv" label="Driver technology" value={driverTech} onChange={setDriverTech} options={[{ value: 'solar', label: 'Solar' }, { value: 'wind', label: 'Wind' }]} />
            )
          }
        >
          <PanelState
            q={ex.isPending && !row ? [] : ex}
            height={260}
            empty={drivers.length === 0}
            emptyTitle="No drivers stored for this hour"
            emptyHint="SHAP is precomputed by run_cycle only for hours the residual model serves. Physics-only regions and night-time solar have none."
          >
            {correction != null && (
              <p className="mb-4 text-sm text-slate-600">
                Net ML correction for this hour:{' '}
                <b className="text-slate-800 num">
                  {correction >= 0 ? '+' : '−'}
                  {Math.abs(correction * 100).toFixed(2)} pt CF
                </b>
                {capacity[driverTech] && <span className="num"> ≈ {correction >= 0 ? '+' : '−'}{mw(Math.abs(correction * capacity[driverTech]!))}</span>} versus physics alone.
              </p>
            )}
            <DriversList drivers={drivers} color={TECH_COLOR[driverTech]} />
          </PanelState>
        </Panel>
      </div>

      <Panel title="Uncertainty grows with lead time" subtitle="P10–P90 band width as a share of installed capacity. Solar is shown in daylight only." delay={0.15}>
        <PanelState q={fc} height={220} empty={rows.length === 0}>
          <LeadChart
            height={220}
            data={widths}
            yFormat={(v) => pct(v, 0)}
            series={techs.map((t) => ({ key: t, label: t === 'solar' ? 'Solar band width' : 'Wind band width', color: TECH_COLOR[t] }))}
          />
        </PanelState>
      </Panel>
    </div>
  );
}

const Dot = ({ c }: { c: string }) => <span className="inline-block w-2 h-2 rounded-full" style={{ background: c }} />;

function IconBtn({ label, onClick, disabled, children }: { label: string; onClick: () => void; disabled?: boolean; children: React.ReactNode }) {
  return (
    <button aria-label={label} onClick={onClick} disabled={disabled} className="p-1.5 rounded-full bg-white/70 border border-white text-slate-600 hover:bg-white disabled:opacity-40">
      {children}
    </button>
  );
}

function ForecastTable({ rows, techs, selected, onPick }: { rows: FanRow[]; techs: Tech[]; selected?: number; onPick: (t: number) => void }) {
  const cell = (p50?: number, band?: [number, number]) => (p50 == null || !band ? '—' : `${mw(p50)} (${mw(band[0])}–${mw(band[1])})`);
  return (
    <div className="overflow-auto max-h-[380px] styled-scroll rounded-xl">
      <table className="w-full text-xs num">
        <thead className="sticky top-0 bg-white/90 backdrop-blur text-[11px] uppercase tracking-wider text-slate-500">
          <tr>
            <th className="text-left px-3 py-2 font-medium">Local time</th>
            <th className="text-right px-3 py-2 font-medium">Lead</th>
            {techs.map((t) => (
              <th key={t} className="text-right px-3 py-2 font-medium">{t} P50 (P10–P90)</th>
            ))}
            <th className="text-right px-3 py-2 font-medium">Calibrated</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.t} onClick={() => onPick(r.t)} className={`cursor-pointer border-t border-white/60 ${r.t === selected ? 'bg-blue-50/80' : 'hover:bg-white/50'}`}>
              <td className="px-3 py-1.5 text-slate-700">{localTime(new Date(r.t).toISOString(), 'EEE d MMM HH:mm')}</td>
              <td className="px-3 py-1.5 text-right text-slate-500">{r.lead} h</td>
              {techs.map((t) => (
                <td key={t} className="px-3 py-1.5 text-right text-slate-800">{t === 'solar' ? cell(r.solarP50, r.solarBand) : cell(r.windP50, r.windBand)}</td>
              ))}
              <td className="px-3 py-1.5 text-right text-slate-500">{techs.every((t) => (t === 'solar' ? r.solarCal : r.windCal)) ? 'yes' : 'no'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
