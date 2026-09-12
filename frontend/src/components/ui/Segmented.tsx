import { type ReactNode } from 'react';
import { motion } from 'framer-motion';

export function Segmented<T extends string | number>({
  id,
  value,
  options,
  onChange,
  label,
}: {
  id: string; // unique per screen: drives the shared-layout pill animation
  value: T;
  options: { value: T; label: ReactNode }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex p-1 rounded-full bg-white/50 border border-white/60 shadow-sm">
      {options.map((o) => {
        const on = o.value === value;
        return (
          <button
            key={String(o.value)}
            role="radio"
            aria-checked={on}
            onClick={() => onChange(o.value)}
            className={`relative px-3 py-1 text-xs font-semibold rounded-full transition-colors ${on ? 'text-slate-900' : 'text-slate-500 hover:text-slate-800'}`}
          >
            {on && (
              <motion.span layoutId={`seg-${id}`} className="absolute inset-0 rounded-full bg-white shadow" transition={{ type: 'spring', stiffness: 400, damping: 32 }} />
            )}
            <span className="relative inline-flex items-center gap-1.5">{o.label}</span>
          </button>
        );
      })}
    </div>
  );
}
