import type { Provenance } from '../../api/client';
import { displayTz, localTime, utcTime } from '../../lib/format';

/** Every screen must answer: which model, which calibration, which run, was it replayed? */
export default function ProvenanceFooter({ prov }: { prov?: Provenance }) {
  if (!prov) return null;
  const items: [string, string][] = [
    ['Model', prov.model_version ?? 'unavailable'],
    ['Calibrated', prov.calibration_date ?? 'not calibrated'],
    ['NWP', prov.nwp_models?.length ? prov.nwp_models.join(' · ') : '—'],
    ['Issued', `${utcTime(prov.issued_at)} (${localTime(prov.issued_at, 'd MMM HH:mm')} ${displayTz()})`],
    ['Times shown in', displayTz()],
    ['Source', prov.replay_mode ? 'replay snapshot' : 'live'],
  ];
  return (
    <footer className="mt-8 flex flex-wrap gap-x-6 gap-y-1 text-[11px] rounded-2xl bg-white/55 backdrop-blur-md border border-white/60 px-4 py-3">
      {items.map(([k, v]) => (
        <span key={k}>
          <span className="uppercase tracking-wider text-slate-500 mr-1.5">{k}</span>
          <span className="font-mono text-slate-800">{v}</span>
        </span>
      ))}
    </footer>
  );
}
