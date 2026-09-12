import { useMemo, useRef, useState, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { motion, useScroll, useSpring } from 'framer-motion';
import { ArrowRight, BrainCircuit, CloudSun, Gauge, History, ListChecks, Sigma, Sun, Wind, Zap, type LucideIcon } from 'lucide-react';
import { useActions, useBacktest, useForecast, useSites } from '../api/queries';
import { ACTION, regionName, roleOf } from '../lib/domain';
import { inr, localTime, mw, mwh, pct, window_ } from '../lib/format';
import FanChart, { toFanRows } from '../components/charts/FanChart';
import { SeriesKey } from '../components/charts/common';
import { PanelState } from '../components/ui/Panel';
import AnimatedCounter from '../components/ui/AnimatedCounter';
import CursorGlow from '../components/ui/CursorGlow';
import LoadingScreen from '../components/ui/LoadingScreen';
import ScrollProgress from '../components/ui/ScrollProgress';
import ScrollReveal from '../components/ui/ScrollReveal';
import TiltCard from '../components/ui/TiltCard';

const PIPELINE: { icon: LucideIcon; title: string; body: string; why: string }[] = [
  {
    icon: CloudSun,
    title: 'Three weather models',
    body: 'Hourly Open-Meteo forecasts from ECMWF IFS, ICON and GFS at every capacity-weighted grid point.',
    why: 'Where the three models disagree, the forecast is genuinely harder, and that disagreement is itself an uncertainty feature.',
  },
  {
    icon: Sun,
    title: 'Physics baseline',
    body: 'pvlib turns irradiance into PV output across fleet archetypes, including single-axis trackers. windpowerlib turns hub-height wind into turbine output.',
    why: 'Physics carries the forecast on day one and transfers to any region, even one with no history.',
  },
  {
    icon: BrainCircuit,
    title: 'ML learns the residual',
    body: 'LightGBM predicts only what physics gets wrong, in capacity factor, from NWP features and lead time. There is no autoregressive lag under 24 h.',
    why: 'This is NWP post-processing, not time-series extrapolation, so the backtest cannot cheat on data that would not exist at forecast time.',
  },
  {
    icon: Sigma,
    title: 'Calibrated uncertainty',
    body: 'Quantile heads plus split-conformal calibration on a recent window, pooled into 12 h lead bands and conditioned on sun elevation.',
    why: 'A P10–P90 band that really covers 80% is a reserve requirement. One that does not is decoration.',
  },
  {
    icon: ListChecks,
    title: 'Decisions, sized and priced',
    body: 'Net load = demand − renewables. Every flagged hour becomes curtail, charge, discharge, commit backup or hold. Each action is clamped to the real asset and priced in ₹.',
    why: 'Operators do not act on curves. They act on "discharge 160 MWh, worth ₹10 L, decisive".',
  },
];

const pctOr = (v: number | null | undefined, digits = 2) => (v == null ? '—' : pct(v, digits));
const fixed2 = (v: number | null | undefined) => (v == null ? '—' : v.toFixed(2));

export default function Landing() {
  const scrollRef = useRef<HTMLDivElement>(null);
  const pipeRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress: pipeProgress } = useScroll({ container: scrollRef, target: pipeRef, offset: ['start 70%', 'end 60%'], layoutEffect: false });
  const lineScale = useSpring(pipeProgress, { stiffness: 120, damping: 30 });

  const [intro] = useState(() => {
    try {
      if (sessionStorage.getItem('zb:intro')) return false;
      sessionStorage.setItem('zb:intro', '1');
    } catch {
      /* storage blocked: show the intro every time */
    }
    return true;
  });

  const sites = useSites();
  // The live showcase is the region where accuracy is actually measured.
  const region = sites.data?.find((s) => !s.physics_only)?.region_id ?? 'BE';
  const fc = useForecast(region);
  const ac = useActions(region);
  const solarBt = useBacktest(region, 'solar');
  const windBt = useBacktest(region, 'wind');

  const rows = useMemo(() => toFanRows(fc.data?.data ?? []), [fc.data]);
  // Headlines are served by the API from the same summary the published report uses.
  const solar = solarBt.data?.summary;
  const wind = windBt.data?.summary;
  const topActions = useMemo(() => [...(ac.data?.data ?? [])].sort((a, b) => b.value_inr - a.value_inr).slice(0, 3), [ac.data]);
  const now = rows[0] ? (rows[0].solarP50 ?? 0) + (rows[0].windP50 ?? 0) : null;
  const atStake = (ac.data?.data ?? []).reduce((s, a) => s + a.value_inr, 0);
  const skillFloor = solar?.skill_mean != null && wind?.skill_mean != null ? `${Math.min(solar.skill_mean, wind.skill_mean).toFixed(2)}+` : '—';

  return (
    <div ref={scrollRef} className="relative h-screen overflow-y-auto overflow-x-hidden styled-scroll scroll-smooth">
      {intro && <LoadingScreen />}
      <CursorGlow />
      <div className="fixed top-0 inset-x-0 z-50">
        <ScrollProgress containerRef={scrollRef as React.RefObject<HTMLElement>} />
      </div>

      <header className="sticky top-0 z-40 px-4 md:px-8 pt-4">
        <nav className="max-w-6xl mx-auto flex items-center gap-4 px-5 py-3 rounded-full bg-white/45 backdrop-blur-xl border border-white/50 shadow-lg shadow-blue-900/5">
          <span className="flex items-center gap-2 font-bold text-slate-800">
            <Zap className="w-5 h-5 text-blue-600 fill-blue-600" /> ZERO <span className="text-blue-600 -ml-1">BIAS</span>
          </span>
          <a href="#pipeline" className="hidden sm:inline ml-auto text-sm text-slate-600 hover:text-slate-900">
            How it works
          </a>
          <a href="#evidence" className="hidden sm:inline text-sm text-slate-600 hover:text-slate-900">
            Evidence
          </a>
          <Link to={`/ops/${region}`} className="ml-auto sm:ml-0 inline-flex items-center gap-1.5 px-4 py-2 rounded-full bg-blue-600 text-white text-sm font-semibold hover:bg-blue-700">
            Open console <ArrowRight className="w-4 h-4" />
          </Link>
        </nav>
      </header>

      {/* HERO */}
      <section className="px-4 md:px-8 pt-10 pb-16 md:pt-16">
        <div className="max-w-6xl mx-auto grid lg:grid-cols-[1.05fr_1fr] gap-8 items-center">
          <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: intro ? 2.6 : 0 }} className="glass-panel !rounded-[2rem] p-6 md:p-10">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-blue-700">AI renewable generation forecasting</p>
            <h1 className="mt-3 text-4xl md:text-5xl font-semibold text-slate-900 leading-[1.08] tracking-tight">
              Know tomorrow's <span className="bg-gradient-to-r from-orange-500 to-amber-400 bg-clip-text text-transparent">sun</span> and{' '}
              <span className="bg-gradient-to-r from-blue-600 to-cyan-500 bg-clip-text text-transparent">wind</span>, and what to do about it.
            </h1>
            <p className="mt-4 text-slate-600 leading-relaxed">
              Regional 24–72 h forecasts with calibrated P10–P90 bands, turned into grid actions that are sized to the asset, priced in rupees, and marked decisive or indicative.
            </p>

            <div className="mt-6 flex flex-wrap gap-3">
              <LiveStat label={`${region} renewables, next hour`} loading={fc.isPending} failed={fc.isError}>
                {now != null && (
                  <>
                    <AnimatedCounter value={Math.round(now).toLocaleString('en-US')} className="num" /> MW
                  </>
                )}
              </LiveStat>
              <LiveStat label="Actions this run" loading={ac.isPending} failed={ac.isError}>
                {ac.data && (
                  <>
                    <AnimatedCounter value={String(ac.data.data.length)} /> · {inr(atStake)}
                  </>
                )}
              </LiveStat>
              {fc.data?.replay_mode && (
                <span className="inline-flex items-center gap-1.5 self-center px-3 py-1.5 rounded-full text-xs font-bold bg-amber-100/80 text-amber-800 border border-amber-200">
                  <History className="w-3.5 h-3.5" /> REPLAY — cached {localTime(fc.data.issued_at, 'd MMM yyyy')}
                </span>
              )}
            </div>

            <div className="mt-8 flex flex-wrap gap-3">
              <Link to={`/ops/${region}`} className="inline-flex items-center gap-2 px-5 py-3 rounded-full bg-blue-600 text-white font-semibold shadow-lg shadow-blue-600/25 hover:bg-blue-700 group">
                Open the operations console <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </Link>
              <a href="#pipeline" className="inline-flex items-center gap-2 px-5 py-3 rounded-full bg-white/70 border border-white text-slate-700 font-semibold hover:bg-white">
                How it works
              </a>
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0, x: 40 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.8, delay: intro ? 2.8 : 0.2 }}>
            <TiltCard className="rounded-[2rem]">
              <div className="glass-panel !rounded-[2rem] p-5">
                <div className="flex items-center justify-between gap-2 mb-3">
                  <p className="text-sm font-semibold text-slate-800">Live · {regionName(region)} · next 72 h</p>
                  <SeriesKey
                    items={[
                      { color: 'var(--series-solar)', label: 'Solar' },
                      { color: 'var(--series-wind)', label: 'Wind' },
                    ]}
                  />
                </div>
                <PanelState q={fc} height={260} empty={rows.length === 0} emptyTitle="No forecast yet">
                  <FanChart rows={rows} techs={['solar', 'wind']} height={260} />
                </PanelState>
                <p className="mt-2 text-[11px] text-slate-500">Line = P50 · band = P10–P90 · hatched = lead under 24 h, not calibrated</p>
              </div>
            </TiltCard>
          </motion.div>
        </div>
      </section>

      {/* PROBLEM */}
      <Section eyebrow="The problem" title="Renewables are weather. Grids run on commitments.">
        <div className="grid md:grid-cols-3 gap-4">
          {[
            { icon: Sun, t: 'Surplus gets curtailed', d: 'When midday solar overshoots demand minus must-run, clean energy is thrown away unless storage was told to be empty.' },
            { icon: Wind, t: 'Ramps get over-reserved', d: 'Without a trustworthy band, operators hold expensive backup for the worst case they cannot quantify.' },
            { icon: Gauge, t: 'A single number hides risk', d: 'A point forecast cannot tell a safe action from a gamble. An honest interval can.' },
          ].map((c, i) => (
            <ScrollReveal key={c.t} delay={i * 0.1}>
              <div className="glass-panel p-6 h-full">
                <c.icon className="w-6 h-6 text-blue-600" />
                <h3 className="mt-3 font-semibold text-slate-800">{c.t}</h3>
                <p className="mt-2 text-sm text-slate-600 leading-relaxed">{c.d}</p>
              </div>
            </ScrollReveal>
          ))}
        </div>
      </Section>

      {/* PIPELINE */}
      <Section id="pipeline" eyebrow="How it works" title="From three weather models to one decision.">
        <div ref={pipeRef} className="relative pl-10 md:pl-14">
          <div className="absolute left-4 md:left-6 top-2 bottom-2 w-0.5 bg-white/50 rounded-full" />
          <motion.div className="absolute left-4 md:left-6 top-2 bottom-2 w-0.5 bg-gradient-to-b from-blue-600 via-cyan-400 to-green-500 rounded-full origin-top" style={{ scaleY: lineScale }} />
          <div className="flex flex-col gap-5">
            {PIPELINE.map((s, i) => (
              <ScrollReveal key={s.title} direction={i % 2 ? 'right' : 'left'}>
                <div className="relative">
                  <span className="absolute -left-10 md:-left-14 top-5 w-8 h-8 md:w-9 md:h-9 -translate-x-1/2 ml-4 md:ml-6 rounded-full bg-white border-2 border-blue-500 text-blue-700 text-xs font-bold flex items-center justify-center shadow">
                    {i + 1}
                  </span>
                  <div className="glass-panel p-5 md:p-6 grid md:grid-cols-[1fr_0.9fr] gap-4">
                    <div>
                      <h3 className="flex items-center gap-2 font-semibold text-slate-800">
                        <s.icon className="w-5 h-5 text-blue-600" /> {s.title}
                      </h3>
                      <p className="mt-2 text-sm text-slate-600 leading-relaxed">{s.body}</p>
                    </div>
                    <p className="text-sm text-slate-700 leading-relaxed rounded-2xl bg-blue-50/70 border border-blue-100 p-4">
                      <span className="block text-[10px] font-semibold uppercase tracking-wider text-blue-700 mb-1">Why it matters</span>
                      {s.why}
                    </p>
                  </div>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </Section>

      {/* EVIDENCE */}
      <Section
        id="evidence"
        eyebrow="Evidence"
        title="Measured, not promised."
        sub={`${solar?.folds ? `${solar.folds}-fold walk-forward` : 'Walk-forward'} backtest on ${regionName(region)}, scored per lead hour. These are the published headline numbers, served live by the API.`}
      >
        <PanelState q={[solarBt, windBt]} height={160} empty={!solar && !wind} emptyTitle="No backtest published" emptyHint="Run scripts/backtest.py for this region.">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <Metric label="Solar nRMSE" note="daylight hours only" value={pctOr(solar?.nrmse_mean)} />
            <Metric label="Wind nRMSE" note="all hours" value={pctOr(wind?.nrmse_mean)} />
            <Metric label="Skill vs persistence" note={`solar ${fixed2(solar?.skill_mean)} · wind ${fixed2(wind?.skill_mean)}`} value={skillFloor} />
            <Metric
              label="80% band coverage"
              note={solar ? `target ${pct(solar.picp_target_low, 0)}–${pct(solar.picp_target_high, 0)}` : 'target band'}
              value={`${pctOr(solar?.picp_mean, 1)} / ${pctOr(wind?.picp_mean, 1)}`}
            />
          </div>
        </PanelState>
        <ScrollReveal>
          <div className="mt-4 glass-panel p-5 grid md:grid-cols-2 gap-4 text-sm text-slate-600 leading-relaxed">
            <p>
              <b className="text-slate-800">Honest benchmark.</b> Elia's day-ahead (effectively 6–30 h) and week-ahead (144 h+) bracket our 24–72 h band, and we land between them
              on both technologies. We do not claim to beat the TSO.
            </p>
            <p>
              <b className="text-slate-800">We tested bigger models, and they lost.</b> A seq2seq stack and a graph variant were well calibrated but less sharp than LightGBM over
              the same folds. The simpler model is what ships.
            </p>
          </div>
        </ScrollReveal>
        <div className="mt-4">
          <Link to={`/evidence/${region}`} className="inline-flex items-center gap-1.5 text-sm font-semibold text-blue-700 hover:text-blue-900">
            Per-lead-hour charts <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </Section>

      {/* DECISIONS */}
      <Section eyebrow="Decisions" title="Every forecast ends in an action." sub={`The highest-value actions from the current ${regionName(region)} run.`}>
        <PanelState q={ac} height={200} empty={topActions.length === 0} emptyTitle="No actions in this run" emptyHint="Every hour of the horizon is inside operating limits.">
          <div className="grid md:grid-cols-3 gap-4">
            {topActions.map((a, i) => {
              const meta = ACTION[a.action];
              return (
                <ScrollReveal key={a.action_id} delay={i * 0.1}>
                  <TiltCard className="rounded-[1.5rem] h-full">
                    <div className="glass-panel p-5 h-full">
                      <p className="flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-slate-800">
                        {meta && <meta.icon className="w-4 h-4 text-blue-600" />} {meta?.label ?? a.action}
                      </p>
                      <p className="text-[11px] text-slate-500 num mt-0.5">
                        {window_(a.valid_from, a.valid_to)} · {localTime(a.valid_from, 'd MMM')}
                      </p>
                      <div className="mt-4 flex items-end justify-between">
                        <span className="text-3xl font-bold text-slate-800 num">{mwh(a.mwh)}</span>
                        <span className="text-lg font-bold text-green-700 num">{inr(a.value_inr)}</span>
                      </div>
                      <p className="mt-3 text-[11px] font-semibold text-blue-800">
                        <span className="tracking-[0.2em]">{a.decisive ? '●●●' : '●●○'}</span> {a.decisive ? 'Decisive' : 'Indicative'} · at {mw(a.power_mw)}
                        {a.acknowledged_at && <span className="ml-1 text-slate-500 font-medium">· acknowledged</span>}
                      </p>
                      <p className="mt-2 text-xs text-slate-600 leading-relaxed line-clamp-3">{a.rationale}</p>
                    </div>
                  </TiltCard>
                </ScrollReveal>
              );
            })}
          </div>
        </PanelState>
      </Section>

      {/* REGIONS */}
      <Section eyebrow="Regions" title="Prove accuracy where the truth is published. Prove transfer where it is not.">
        <PanelState q={sites} height={180} empty={!sites.data?.length} emptyTitle="No regions configured">
          <div className="grid md:grid-cols-2 gap-4">
            {(sites.data ?? []).map((s, i) => {
              const role = roleOf(s);
              return (
                <ScrollReveal key={s.region_id} direction={i % 2 ? 'right' : 'left'}>
                  <Link to={`/ops/${s.region_id}`} className="block glass-panel p-6 h-full hover:bg-white/60 transition-colors group">
                    <role.icon className="w-6 h-6 text-blue-600" />
                    <h3 className="mt-3 font-semibold text-slate-800">
                      {regionName(s.region_id)}: {role.label.toLowerCase()}
                    </h3>
                    <p className="mt-2 text-sm text-slate-600 leading-relaxed">{role.blurb}</p>
                    <p className="mt-3 text-xs text-slate-500 num">
                      {mw(s.capacity_mw.solar ?? 0)} solar · {mw(s.capacity_mw.wind ?? 0)} wind
                    </p>
                    <span className="mt-4 inline-flex items-center gap-1 text-sm font-semibold text-blue-700">
                      Open {s.region_id} <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                    </span>
                  </Link>
                </ScrollReveal>
              );
            })}
          </div>
        </PanelState>
      </Section>

      <footer className="px-4 md:px-8 pb-12">
        <div className="max-w-6xl mx-auto glass-panel !rounded-[2rem] p-8 md:p-10 flex flex-col md:flex-row items-center gap-6 justify-between">
          <div>
            <h2 className="text-2xl font-semibold text-slate-900">See what the grid needs in the next 72 hours.</h2>
            <p className="mt-1 text-sm text-slate-600">Team ZERO BIAS</p>
          </div>
          <Link to={`/ops/${region}`} className="inline-flex items-center gap-2 px-6 py-3 rounded-full bg-blue-600 text-white font-semibold shadow-lg shadow-blue-600/25 hover:bg-blue-700">
            Open console <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </footer>
    </div>
  );
}

function Section({ id, eyebrow, title, sub, children }: { id?: string; eyebrow: string; title: string; sub?: string; children: ReactNode }) {
  return (
    <section id={id} className="px-4 md:px-8 py-12 md:py-16 scroll-mt-20">
      <div className="max-w-6xl mx-auto">
        <ScrollReveal>
          <div className="mb-8 max-w-3xl rounded-3xl bg-white/35 backdrop-blur-md px-6 py-5 border border-white/40">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-blue-700">{eyebrow}</p>
            <h2 className="mt-2 text-2xl md:text-4xl font-semibold text-slate-900 tracking-tight">{title}</h2>
            {sub && <p className="mt-2 text-sm text-slate-600">{sub}</p>}
          </div>
        </ScrollReveal>
        {children}
      </div>
    </section>
  );
}

function LiveStat({ label, loading, failed, children }: { label: string; loading: boolean; failed: boolean; children: ReactNode }) {
  return (
    <div className="rounded-2xl bg-white/60 border border-white/70 px-4 py-2.5 min-w-[160px]">
      <p className="text-[10px] uppercase tracking-wider text-slate-500">{label}</p>
      <p className="text-lg font-bold text-slate-800 num">{failed ? <span className="text-sm text-red-700">API offline</span> : loading ? '…' : children}</p>
    </div>
  );
}

function Metric({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <ScrollReveal>
      <TiltCard className="rounded-[1.5rem] h-full">
        <div className="glass-panel p-5 h-full">
          <p className="text-[11px] uppercase tracking-wider text-slate-500">{label}</p>
          <p className="mt-2 text-3xl font-bold text-slate-900 num">{value}</p>
          <p className="mt-1 text-xs text-slate-500">{note}</p>
        </div>
      </TiltCard>
    </ScrollReveal>
  );
}
