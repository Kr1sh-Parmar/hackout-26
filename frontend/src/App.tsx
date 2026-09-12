import { Navigate, Route, Routes, useLocation } from 'react-router-dom';
import { useSites } from './api/queries';
import { setDisplayTz } from './lib/format';
import ConsoleLayout from './components/Layout/ConsoleLayout';
import FloatingOrbs from './components/ui/FloatingOrbs';
import Landing from './pages/Landing';
import Operations from './pages/Operations';
import ForecastDetail from './pages/ForecastDetail';
import Evidence from './pages/Evidence';
import Planning from './pages/Planning';
import Regions from './pages/Regions';

export default function App() {
  // Show times in the viewed region's timezone. Set during render, before any child formats a time.
  const region = useLocation().pathname.split('/')[2] || 'BE';
  const sites = useSites();
  setDisplayTz(sites.data?.find((s) => s.region_id === region)?.timezone);

  return (
    <div className="relative">
      <FloatingOrbs />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route element={<ConsoleLayout />}>
          <Route path="/ops/:region" element={<Operations />} />
          <Route path="/forecast/:region" element={<ForecastDetail />} />
          <Route path="/evidence/:region" element={<Evidence />} />
          <Route path="/planning/:region" element={<Planning />} />
          <Route path="/regions" element={<Regions />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
