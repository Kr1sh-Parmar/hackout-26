import { motion } from 'framer-motion';
import type { Driver } from '../../api/client';

const NWP: Record<string, string> = { ecmwf_ifs025: 'ECMWF', icon_seamless: 'ICON', gfs_seamless: 'GFS' };

export const featureLabel = (f: string) => {
  const [base, model] = f.split('__');
  const label = base.replace(/_/g, ' ');
  return model ? `${label} (${NWP[model] ?? model})` : label;
};

const fmtValue = (v: number | null | undefined) => (v == null ? '—' : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2));

/**
 * Diverging bars of signed SHAP contributions, in capacity-factor points. Positive means the
 * feature pushed the forecast ABOVE what physics alone predicted.
 */
export default function DriversList({ drivers, color }: { drivers: Driver[]; color: string }) {
  const m = Math.max(...drivers.map((d) => Math.abs(d.contribution)), 1e-9);
  return (
    <ul className="flex flex-col gap-2.5">
      {drivers.map((d, i) => {
        const up = d.contribution >= 0;
        return (
          <li key={d.feature} className="grid grid-cols-[minmax(0,12rem)_1fr_4.5rem] items-center gap-3 text-xs">
            <span className="min-w-0">
              <span className="block truncate font-medium text-slate-700 first-letter:uppercase" title={d.feature}>
                {featureLabel(d.feature)}
              </span>
              <span className="block text-[10px] text-slate-400 num">value {fmtValue(d.value)}</span>
            </span>
            <div className="relative h-3 rounded-full bg-slate-200/50" aria-hidden>
              <span className="absolute inset-y-[-3px] left-1/2 w-px bg-slate-400/70" />
              <motion.span
                className="absolute inset-y-0 rounded-full"
                style={{ background: up ? color : 'var(--ink-muted)', [up ? 'left' : 'right']: '50%' }}
                initial={{ width: 0 }}
                animate={{ width: `${(Math.abs(d.contribution) / m) * 50}%` }}
                transition={{ delay: i * 0.05, duration: 0.5, ease: 'easeOut' }}
              />
            </div>
            <span className={`text-right font-semibold num ${up ? 'text-slate-800' : 'text-slate-500'}`}>
              {up ? '+' : '−'}
              {Math.abs(d.contribution * 100).toFixed(2)} pt
            </span>
          </li>
        );
      })}
    </ul>
  );
}
