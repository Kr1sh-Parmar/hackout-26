import { Outlet, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { Clock } from 'lucide-react';
import DashboardShell from './DashboardShell';
import Sidebar from './Sidebar';
import StatusBar from './StatusBar';
import ProvenanceFooter from './ProvenanceFooter';
import { useForecast } from '../../api/queries';
import { useConsoleParams } from '../../lib/url';
import { localTime, minutesLabel } from '../../lib/format';

// A run lands every 6 h. Past 7 h one has been missed: say so, while still showing the data.
const STALE_AFTER_MIN = 7 * 60;

export default function ConsoleLayout() {
  const { pathname } = useLocation();
  const { region, runTs } = useConsoleParams();
  const fc = useForecast(region, { runTs });
  const d = fc.data;
  const stale = d && !d.replay_mode && !runTs && (d.age_minutes ?? 0) > STALE_AFTER_MIN;

  return (
    <DashboardShell sidebar={<Sidebar />} header={<StatusBar />}>
      {stale && (
        <div role="alert" className="mb-4 flex items-center gap-2 rounded-2xl px-4 py-3 text-sm bg-amber-50/90 border border-amber-200 text-amber-900">
          <Clock className="w-4 h-4 shrink-0" />
          Showing the {localTime(d.issued_at, 'HH:mm')} run, {minutesLabel(d.age_minutes!)} old. A newer run has not landed.
        </div>
      )}
      <AnimatePresence mode="wait">
        <motion.div
          key={pathname}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.25, ease: 'easeOut' }}
        >
          <Outlet />
        </motion.div>
      </AnimatePresence>
      {pathname !== '/regions' && <ProvenanceFooter prov={d} />}
    </DashboardShell>
  );
}
