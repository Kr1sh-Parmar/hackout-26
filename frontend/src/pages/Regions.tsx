import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { AlertTriangle, ArrowRight, Clock } from 'lucide-react';
import type { SiteInfo } from '../api/client';
import { useForecast, useHealth, useSites } from '../api/queries';
import { regionName, roleOf, TECH_COLOR } from '../lib/domain';
import { mw } from '../lib/format';
import { Panel, PanelState } from '../components/ui/Panel';
import TiltCard from '../components/ui/TiltCard';

export default function Regions() {
  const sites = useSites();
  return (
    <div className="flex flex-col gap-4">
      <div>
        <h1 className="text-[22px] font-semibold text-slate-800">Regions</h1>
        <p className="text-xs text-slate-700">One pipeline, two roles: prove accuracy where the truth is published, prove transfer where it is not.</p>
      </div>
      <PanelState q={sites} height={360} empty={!sites.data?.length} emptyTitle="No regions configured" emptyHint="Add a YAML file under config/regions/.">
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          {sites.data?.map((s, i) => <RegionCard key={s.region_id} site={s} delay={i * 0.1} />)}
        </div>
      </PanelState>
    </div>
  );
}

function RegionCard({ site, delay }: { site: SiteInfo; delay: number }) {
  const health = useHealth();
  const fc = useForecast(site.region_id);
  const role = roleOf(site);
  const warnings = (health.data?.warnings ?? []).filter((w) => w.startsWith(site.region_id));
  const solar = site.capacity_mw.solar ?? 0;
  const wind = site.capacity_mw.wind ?? 0;

  return (
    <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ delay, duration: 0.5 }}>
      <TiltCard className="rounded-[1.5rem] h-full">
        <Panel className="h-full">
          <div className="flex flex-col sm:flex-row gap-5">
            <Ring solar={solar} wind={wind} />
            <div className="flex-1 min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="text-lg font-bold text-slate-800">{regionName(site.region_id)}</h2>
                <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-[11px] font-semibold ${role.chip}`}>
                  <role.icon className="w-3 h-3" />
                  {role.label}
                </span>
              </div>
              <p className="mt-2 text-xs text-slate-600 leading-relaxed">{role.blurb}</p>
              <dl className="mt-4 grid grid-cols-2 gap-2 text-xs">
                <Fact label="Solar installed" value={mw(solar)} dot={TECH_COLOR.solar} />
                <Fact label="Wind installed" value={mw(wind)} dot={TECH_COLOR.wind} />
                <Fact label="Timezone" value={site.timezone} />
                <Fact label="Model" value={fc.data?.model_version ?? (fc.isError ? 'unavailable' : '…')} mono />
              </dl>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {site.nwp_models.map((m) => (
                  <span key={m} className="px-2 py-0.5 rounded-full bg-white/70 border border-white text-[10px] font-mono text-slate-600">
                    {m}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-4 border-t border-white/60 pt-3">
            {warnings.length ? (
              <details>
                <summary className="cursor-pointer text-xs font-semibold text-orange-800 flex items-center gap-1.5">
                  <AlertTriangle className="w-3.5 h-3.5" /> {warnings.length} health warning{warnings.length > 1 ? 's' : ''}
                </summary>
                <ul className="mt-2 flex flex-col gap-1.5 text-[11px] text-slate-600">
                  {warnings.map((w) => (
                    <li key={w}>{w}</li>
                  ))}
                </ul>
              </details>
            ) : (
              <p className="text-xs text-slate-500 flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5" /> {health.data ? 'No warnings for this region' : 'Health unavailable'}
              </p>
            )}
          </div>

          <Link to={`/ops/${site.region_id}`} className="mt-4 inline-flex items-center gap-1.5 px-4 py-2 rounded-full bg-blue-600 text-white text-xs font-semibold hover:bg-blue-700 group">
            Open operations <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
          </Link>
        </Panel>
      </TiltCard>
    </motion.div>
  );
}

function Ring({ solar, wind }: { solar: number; wind: number }) {
  const total = solar + wind || 1;
  const r = 52;
  const c = 2 * Math.PI * r;
  const solarLen = (solar / total) * c;
  return (
    <div className="relative w-36 h-36 shrink-0 mx-auto">
      <svg viewBox="0 0 128 128" className="w-full h-full -rotate-90">
        <circle cx="64" cy="64" r={r} fill="none" stroke="rgba(148,163,184,.2)" strokeWidth="14" />
        <motion.circle cx="64" cy="64" r={r} fill="none" stroke={TECH_COLOR.wind} strokeWidth="14" strokeDasharray={`${c} ${c}`} initial={{ strokeDashoffset: c }} animate={{ strokeDashoffset: solarLen }} transition={{ duration: 1.2, ease: 'easeOut' }} />
        <motion.circle cx="64" cy="64" r={r} fill="none" stroke={TECH_COLOR.solar} strokeWidth="14" initial={{ strokeDasharray: `0 ${c}` }} animate={{ strokeDasharray: `${solarLen} ${c}` }} transition={{ duration: 1, ease: 'easeOut' }} />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-xl font-bold text-slate-800 num">{(total / 1000).toFixed(1)}</span>
        <span className="text-[10px] text-slate-500">GW installed</span>
      </div>
    </div>
  );
}

function Fact({ label, value, dot, mono }: { label: string; value: string; dot?: string; mono?: boolean }) {
  return (
    <div className="rounded-xl bg-white/50 border border-white/70 px-3 py-2">
      <dt className="text-[10px] uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
        {dot && <span className="w-1.5 h-1.5 rounded-full" style={{ background: dot }} />}
        {label}
      </dt>
      <dd className={`mt-0.5 font-semibold text-slate-800 truncate num ${mono ? 'font-mono text-[11px]' : ''}`}>{value}</dd>
    </div>
  );
}
