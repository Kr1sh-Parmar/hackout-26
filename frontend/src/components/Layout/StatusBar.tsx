import { NavLink, useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle2, History, Undo2, WifiOff } from 'lucide-react';
import { useForecast, useHealth, useRuns, useSites } from '../../api/queries';
import { useConsoleParams } from '../../lib/url';
import { ago, localTime } from '../../lib/format';
import { NAV } from './Sidebar';

export default function StatusBar() {
  const { region, runTs, set } = useConsoleParams();
  const sites = useSites();
  const health = useHealth();
  const runs = useRuns(region);
  const fc = useForecast(region, { runTs });
  const navigate = useNavigate();
  const { pathname, search } = useLocation();

  const regionScoped = pathname !== '/regions';
  const replay = fc.data?.replay_mode ?? health.data?.replay_mode;
  const switchRegion = (r: string) => {
    // A run and an hour belong to one region: carrying them across would 404 or pin a meaningless hour.
    const sp = new URLSearchParams(search);
    sp.delete('run_ts');
    sp.delete('at');
    navigate(pathname.replace(/^(\/[^/]+)\/[^/]+/, `$1/${r}`) + (sp.size ? `?${sp}` : ''));
  };

  return (
    <div className="flex flex-col gap-3 px-4 md:px-8 pt-5 pb-2">
      <div className="flex flex-wrap items-center gap-2 md:gap-3">
        {regionScoped && sites.data && (
          <div role="radiogroup" aria-label="Region" className="inline-flex p-1 rounded-full bg-white/60 border border-white/60 shadow-sm">
            {sites.data.map((s) => {
              const on = s.region_id === region;
              return (
                <button
                  key={s.region_id}
                  role="radio"
                  aria-checked={on}
                  onClick={() => switchRegion(s.region_id)}
                  className={`relative px-3.5 py-1.5 text-xs font-bold tracking-wide rounded-full ${on ? 'text-blue-700' : 'text-slate-500 hover:text-slate-800'}`}
                >
                  {on && <motion.span layoutId="region-pill" className="absolute inset-0 rounded-full bg-white shadow" transition={{ type: 'spring', stiffness: 400, damping: 32 }} />}
                  <span className="relative">{s.region_id}</span>
                </button>
              );
            })}
          </div>
        )}

        {replay && (
          <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold tracking-wide bg-amber-100/80 text-amber-800 border border-amber-200" title="Serving a frozen snapshot, not live data">
            <History className="w-3.5 h-3.5" />
            REPLAY{fc.data ? ` — cached ${localTime(fc.data.issued_at, 'd MMM yyyy')}` : ''}
          </span>
        )}

        {regionScoped && runs.data && runs.data.runs.length > 0 && (
          <label className="inline-flex items-center gap-2 pl-3 pr-2 py-1 rounded-full text-xs bg-white/60 border border-white/60 text-slate-600">
            <span className="text-slate-400">run</span>
            <select
              aria-label="Forecast run"
              value={runTs ?? ''}
              onChange={(e) => set('run_ts', e.target.value || null)}
              className="bg-transparent font-semibold text-slate-800 num outline-none cursor-pointer py-0.5"
            >
              <option value="">Latest · {localTime(runs.data.runs[0], 'd MMM HH:mm')}</option>
              {runs.data.runs.slice(1).map((r) => (
                <option key={r} value={r}>
                  {localTime(r, 'd MMM HH:mm')}
                </option>
              ))}
            </select>
          </label>
        )}

        {regionScoped && runTs && (
          <button
            onClick={() => set('run_ts', null)}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold bg-violet-100/80 text-violet-800 border border-violet-200 hover:bg-violet-100"
          >
            <Undo2 className="w-3.5 h-3.5" /> Past run · back to latest
          </button>
        )}

        {regionScoped && fc.data && (
          <span className="hidden md:inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs bg-white/60 border border-white/60 text-slate-600">
            <span>issued {ago(fc.data.issued_at)}</span>
            <span className="text-slate-400">·</span>
            <span className="font-mono text-[11px]">{fc.data.model_version}</span>
          </span>
        )}

        <div className="ml-auto">
          <HealthPill health={health} />
        </div>
      </div>

      {/* Below lg the sidebar is hidden; the console must stay navigable. */}
      <nav className="lg:hidden flex gap-1 overflow-x-auto [scrollbar-width:none] -mx-1 px-1" aria-label="Console">
        {NAV.map((item) => (
          <NavLink
            key={item.label}
            to={item.to(region)}
            className={({ isActive }) => `flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium whitespace-nowrap ${isActive ? 'bg-blue-100/70 text-blue-700' : 'bg-white/40 text-slate-600'}`}
          >
            <item.icon className="w-3.5 h-3.5" />
            {item.label}
          </NavLink>
        ))}
      </nav>
    </div>
  );
}

function HealthPill({ health }: { health: ReturnType<typeof useHealth> }) {
  if (health.isError)
    return (
      <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold bg-red-100/80 text-red-700 border border-red-200">
        <WifiOff className="w-3.5 h-3.5" /> API offline
      </span>
    );
  if (!health.data) return <span className="inline-block w-24 h-7 rounded-full bg-white/50 animate-pulse" />;

  const { status, warnings } = health.data;
  const ok = status === 'ok';
  return (
    // Native <details> is the popover: keyboard- and screen-reader-friendly for free.
    <details className="relative">
      <summary
        className={`list-none cursor-pointer inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold border ${
          ok ? 'bg-green-100/70 text-green-800 border-green-200' : 'bg-orange-100/80 text-orange-800 border-orange-200'
        }`}
      >
        <span className={`w-2 h-2 rounded-full ${ok ? 'bg-status-good animate-pulse-glow' : 'bg-status-serious'}`} />
        {ok ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertTriangle className="w-3.5 h-3.5" />}
        {ok ? 'Healthy' : `Degraded · ${warnings.length}`}
      </summary>
      <div className="absolute right-0 mt-2 w-[min(22rem,80vw)] z-50 glass-panel p-4 text-xs">
        <p className="font-semibold text-slate-800 mb-2">System health</p>
        {warnings.length === 0 ? (
          <p className="text-slate-500">No warnings. Every region has a fresh run and a current calibration.</p>
        ) : (
          <ul className="flex flex-col gap-2">
            {warnings.map((w) => (
              <li key={w} className="flex gap-2 text-slate-600">
                <AlertTriangle className="w-3.5 h-3.5 shrink-0 text-status-serious mt-0.5" />
                <span>{w}</span>
              </li>
            ))}
          </ul>
        )}
        <p className="text-slate-400 mt-3">Checked {ago(health.data.timestamp)}</p>
      </div>
    </details>
  );
}
