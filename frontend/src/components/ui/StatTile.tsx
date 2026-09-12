import { motion } from 'framer-motion';
import { type LucideIcon } from 'lucide-react';
import AnimatedCounter from './AnimatedCounter';
import TiltCard from './TiltCard';
import { STATUS, type Status } from '../../lib/domain';

interface Props {
  label: string;
  value: string;
  unit?: string;
  sub?: string;
  icon: LucideIcon;
  status?: Status;
  sparkline?: number[];
  sparkColor?: string;
  delay?: number;
}

export function StatTile({ label, value, unit, sub, icon: Icon, status, sparkline, sparkColor = '#2563eb', delay = 0 }: Props) {
  const numeric = /\d/.test(value);
  return (
    <motion.div initial={{ opacity: 0, y: 24, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: 0.5, delay, ease: [0.21, 0.47, 0.32, 0.98] }}>
      <TiltCard className="rounded-2xl h-full">
        <div className="group relative h-full overflow-hidden bg-white/45 backdrop-blur-xl rounded-2xl p-4 border border-white/50 shadow-lg">
          <div className="absolute inset-0 -translate-x-full group-hover:translate-x-full transition-transform duration-700 bg-gradient-to-r from-transparent via-white/40 to-transparent pointer-events-none" />
          {sparkline && <Spark values={sparkline} color={sparkColor} />}

          <div className="relative flex items-center gap-2 mb-2">
            <span className={`w-8 h-8 rounded-full flex items-center justify-center ${status ? STATUS[status].chip : 'bg-blue-100/60 text-blue-600'}`}>
              <Icon className="w-4 h-4" />
            </span>
            <span className="text-[11px] font-medium uppercase tracking-wider text-slate-500 leading-tight">{label}</span>
          </div>
          <div className="relative flex items-baseline gap-1">
            {numeric ? (
              <AnimatedCounter value={value} className="text-2xl md:text-3xl font-bold text-slate-800 num" />
            ) : (
              <span className="text-2xl md:text-3xl font-bold text-slate-800">{value}</span>
            )}
            {unit && <span className="text-sm font-semibold text-slate-500">{unit}</span>}
          </div>
          <div className="relative mt-1 flex flex-wrap items-center gap-1.5 text-[11px] text-slate-500">
            {status && (
              <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full border text-[10px] font-semibold ${STATUS[status].chip}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${STATUS[status].dot}`} />
                {STATUS[status].label}
              </span>
            )}
            {sub && <span className="truncate">{sub}</span>}
          </div>
        </div>
      </TiltCard>
    </motion.div>
  );
}

function Spark({ values, color }: { values: number[]; color: string }) {
  if (values.length < 2) return null;
  const max = Math.max(...values);
  const min = Math.min(...values);
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * 100},${38 - ((v - min) / (max - min || 1)) * 34}`).join(' ');
  return (
    <svg viewBox="0 0 100 40" preserveAspectRatio="none" className="absolute right-3 bottom-3 w-24 h-10 opacity-50" aria-hidden>
      <motion.polyline
        points={pts}
        fill="none"
        stroke={color}
        strokeWidth={2}
        vectorEffect="non-scaling-stroke"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 1.4, ease: 'easeOut' }}
      />
    </svg>
  );
}
