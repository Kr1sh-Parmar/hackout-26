import { NavLink } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Activity, BatteryCharging, BookOpen, Globe, LineChart, ShieldCheck, Zap, type LucideIcon } from 'lucide-react';
import { useConsoleParams } from '../../lib/url';

export const NAV: { to: (region: string) => string; label: string; icon: LucideIcon; hint: string }[] = [
  { to: (r) => `/ops/${r}`, label: 'Operations', icon: Activity, hint: 'Act now' },
  { to: (r) => `/forecast/${r}`, label: 'Forecast', icon: LineChart, hint: '72 h bands + drivers' },
  { to: (r) => `/evidence/${r}`, label: 'Evidence', icon: ShieldCheck, hint: 'Why believe it' },
  { to: (r) => `/planning/${r}`, label: 'Planning', icon: BatteryCharging, hint: 'Battery sizing' },
  { to: () => '/regions', label: 'Regions', icon: Globe, hint: 'Validation · transfer' },
];

export default function Sidebar() {
  const { region } = useConsoleParams();

  return (
    <div className="flex flex-col h-full p-6 pt-8 gap-4">
      <NavLink to="/" className="flex items-center gap-2 mb-6 pl-2 group">
        <Zap className="w-6 h-6 text-blue-600 fill-blue-600 group-hover:rotate-12 transition-transform" />
        <span className="font-bold text-xl text-slate-800 tracking-tight">
          ZERO <span className="text-blue-600">BIAS</span>
        </span>
      </NavLink>

      <nav className="flex-1 flex flex-col gap-1" aria-label="Console">
        {NAV.map((item) => (
          <NavLink
            key={item.label}
            to={item.to(region)}
            className={({ isActive }) =>
              `relative flex items-center gap-3 px-4 py-2.5 rounded-[1rem] transition-colors text-sm ${
                isActive ? 'text-blue-700' : 'text-slate-700 hover:bg-white/40 hover:text-slate-900'
              }`
            }
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <motion.span
                    layoutId="nav-pill"
                    className="absolute inset-0 rounded-[1rem] bg-blue-100/60 border border-white/60 shadow-sm"
                    transition={{ type: 'spring', stiffness: 380, damping: 32 }}
                  />
                )}
                <item.icon className="relative w-4 h-4 shrink-0" />
                <span className="relative flex flex-col leading-tight">
                  <span className="font-semibold">{item.label}</span>
                  <span className="text-[10px] text-slate-500">{item.hint}</span>
                </span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <NavLink
        to="/"
        className="mt-auto flex items-center gap-3 bg-white/60 rounded-2xl p-4 border border-white/50 hover:bg-white/80 transition-colors"
      >
        <BookOpen className="w-4 h-4 text-blue-600 shrink-0" />
        <span className="flex flex-col leading-tight">
          <span className="text-sm font-semibold text-slate-800">How it works</span>
          <span className="text-[11px] text-slate-500">NWP → physics → ML → decisions</span>
        </span>
      </NavLink>
    </div>
  );
}
