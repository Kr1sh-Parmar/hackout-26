import { useMemo, useState } from 'react';
import { AnimatePresence } from 'framer-motion';
import { Activity, AlarmClock, Gauge, Info, ListChecks } from 'lucide-react';
import { ApiError } from '../api/client';
import { useAckAction, useActions, useEvents, useForecast, useHealth, useOutlook, useSite } from '../api/queries';
import { useConsoleParams } from '../lib/url';
import { FLAG, regionName, sevStatus, TECH_COLOR } from '../lib/domain';
import { inr, mw } from '../lib/format';
import { Panel, PanelState, Skeleton } from '../components/ui/Panel';
import { StatTile } from '../components/ui/StatTile';
import FanChart, { toFanRows } from '../components/charts/FanChart';
import NetLoadChart from '../components/charts/NetLoadChart';
import { SeriesKey } from '../components/charts/common';
import ActionCard from '../components/ops/ActionCard';
import EventTimeline from '../components/ops/EventTimeline';

export default function Operations() {
  const { region, runTs, param, set } = useConsoleParams();
  const fc = useForecast(region, { runTs });
  const ol = useOutlook(region, runTs);
  const ev = useEvents(region, runTs);
  const ac = useActions(region, runTs);
  const health = useHealth();
  const ack = useAckAction(region, runTs);
  const { physicsOnly, sites } = useSite(region);
  const validation = sites.data?.find((s) => !s.physics_only);
  const [hoverEvent, setHoverEvent] = useState<string | null>(null);

  const rows = useMemo(() => toFanRows(fc.data?.data ?? []), [fc.data]);
  const events = useMemo(() => [...(ev.data?.data ?? [])].sort((a, b) => a.valid_from.localeCompare(b.valid_from)), [ev.data]);
  const actions = useMemo(
    () =>
      [...(ac.data?.data ?? [])].sort(
        (a, b) => Number(!!a.acknowledged_at) - Number(!!b.acknowledged_at) || a.valid_from.localeCompare(b.valid_from),
      ),
    [ac.data],
  );
  const open = actions.filter((a) => !a.acknowledged_at);

  const at = param('at');
  const selected = at ? Date.parse(at) : null;
  const select = (t: number) => {
    set('at', new Date(t).toISOString());
    document.getElementById('fan')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };
  const ackError = (id: string) => {
    if (!ack.isError || ack.variables?.id !== id) return undefined;
    const e = ack.error;
    return e instanceof ApiError ? `${e.message}${e.hint ? ` ${e.hint}` : ''}` : String(e);
  };

  const noDemand = physicsOnly ? `${region} has no demand feed: net load, events and actions are not computed for a transfer region.` : undefined;
  const first = rows[0];
  const nextEvent = events[0];

  return (
    <div className="flex flex-col gap-4">
      {physicsOnly && (
        <div className="flex items-start gap-3 rounded-2xl px-4 py-3 bg-blue-50/80 border border-blue-100 text-sm text-slate-700">
          <Info className="w-4 h-4 mt-0.5 text-blue-600 shrink-0" />
          <p>
            <b>{regionName(region)} is a transfer region.</b> Physics-only forecast, uncalibrated bands, no demand feed, so no net load, events
            or actions, and no accuracy claim.
            {validation && ` ${regionName(validation.region_id)} is where accuracy is measured.`}
          </p>
        </div>
      )}

      {/* Five-second row: output, next event, open actions, health. */}
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-3 md:gap-4">
        {fc.isPending ? (
          [0, 1, 2, 3].map((i) => <Skeleton key={i} height={118} />)
        ) : (
          <>
            <StatTile
              label="Renewables, next hour"
              icon={Activity}
              value={first ? Math.round((first.solarP50 ?? 0) + (first.windP50 ?? 0)).toLocaleString('en-US') : '—'}
              unit={first ? 'MW' : undefined}
              sub={first ? `solar ${mw(first.solarP50 ?? 0)} · wind ${mw(first.windP50 ?? 0)}` : fc.isError ? 'forecast unavailable' : 'no forecast'}
              sparkline={rows.map((r) => (r.solarP50 ?? 0) + (r.windP50 ?? 0))}
              sparkColor={TECH_COLOR.wind}
            />
            {/* A query that FAILED is unknown, never "all clear": no data must not earn a Good status. */}
            <StatTile
              label="Next grid event"
              icon={AlarmClock}
              value={nextEvent ? String(nextEvent.lead_hours) : !ev.data ? '—' : physicsOnly ? 'n/a' : 'None'}
              unit={nextEvent ? 'h ahead' : undefined}
              status={nextEvent ? sevStatus(nextEvent.severity) : !ev.data || physicsOnly ? undefined : 'good'}
              sub={nextEvent ? FLAG[nextEvent.flag]?.label : ev.isError ? 'events unavailable' : !ev.data ? 'loading…' : physicsOnly ? 'no demand feed' : 'clear 72 h horizon'}
              delay={0.06}
            />
            <StatTile
              label="Open actions"
              icon={ListChecks}
              value={physicsOnly ? 'n/a' : !ac.data ? '—' : String(open.length)}
              status={ac.data && open.length ? 'warning' : undefined}
              sub={physicsOnly ? 'no decision layer' : ac.isError ? 'actions unavailable' : !ac.data ? 'loading…' : `${inr(open.reduce((s, a) => s + a.value_inr, 0))} at stake`}
              delay={0.12}
            />
            <StatTile
              label="Model health"
              icon={Gauge}
              value={
                health.data
                  ? health.data.status === 'ok'
                    ? 'Healthy'
                    : 'Degraded'
                  : health.isError
                    ? (health.error as { status?: number } | null)?.status
                      ? 'API error'
                      : 'Offline'
                    : '…'
              }
              status={health.data ? (health.data.status === 'ok' ? 'good' : 'serious') : health.isError ? 'critical' : undefined}
              sub={health.data ? `${health.data.warnings.length} warning${health.data.warnings.length === 1 ? '' : 's'}` : undefined}
              delay={0.18}
            />
          </>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_380px] gap-4 items-start">
        <div className="flex flex-col gap-4 order-2 lg:order-1 min-w-0">
          <div id="fan">
            <Panel
              title={`Generation forecast · ${rows.length ? `${rows.length} h horizon` : '72 h'}${runTs ? ' · past run' : ''}`}
              subtitle="P50 line, P10–P90 band. Click an hour to pin it."
              right={
                <SeriesKey
                  items={[
                    { color: TECH_COLOR.solar, label: 'Solar' },
                    { color: TECH_COLOR.wind, label: 'Wind' },
                    { color: 'var(--baseline)', label: 'Night', kind: 'shade' },
                    { color: '', label: 'Uncalibrated', kind: 'hatch' },
                    { color: 'var(--status-warning)', label: 'Event', kind: 'shade' },
                  ]}
                />
              }
            >
              <PanelState q={fc} height={320} empty={rows.length === 0} emptyTitle="No forecast for this region yet" emptyHint="Run scripts/run_cycle.py for this region, or start the API in replay mode.">
                <FanChart rows={rows} techs={['solar', 'wind']} events={events} activeEventId={hoverEvent} selected={selected} onSelect={select} />
              </PanelState>
            </Panel>
          </div>

          <Panel
            title="Net load vs must-run"
            subtitle="Demand minus renewables. Below the dashed floor is over-generation."
            delay={0.08}
            right={
              <SeriesKey
                items={[
                  { color: 'var(--ink)', label: 'Demand' },
                  { color: 'var(--series-net)', label: 'Net load' },
                  { color: 'var(--status-critical)', label: 'Must-run', kind: 'dashed' },
                  { color: TECH_COLOR.wind, label: 'Wind', kind: 'band' },
                  { color: TECH_COLOR.solar, label: 'Solar', kind: 'band' },
                ]}
              />
            }
          >
            <PanelState q={ol} height={280} empty={!ol.data?.data.length} emptyTitle="No net-load outlook" emptyHint={noDemand ?? 'The decision layer has not produced an outlook for this run yet.'}>
              <NetLoadChart data={ol.data?.data ?? []} events={events} activeEventId={hoverEvent} selected={selected} onSelect={select} />
            </PanelState>
          </Panel>
        </div>

        <div className="flex flex-col gap-4 order-1 lg:order-2 lg:sticky lg:top-0">
          <Panel
            title="Action queue"
            subtitle="What to do, how much, what it is worth, how sure."
            delay={0.1}
            right={!physicsOnly && ac.data && <span className="text-xs font-semibold text-slate-600 num">{open.length} open</span>}
          >
            <PanelState q={ac} height={220} empty={actions.length === 0} emptyTitle={physicsOnly ? 'No actions for a transfer region' : 'No actions needed'} emptyHint={noDemand ?? 'Every hour of the horizon is inside operating limits. Hold.'}>
              <div className="flex flex-col gap-3 max-h-[620px] overflow-auto styled-scroll pr-1 -mr-1">
                <AnimatePresence initial={false}>
                  {actions.map((a) => (
                    <ActionCard
                      key={a.action_id}
                      action={a}
                      pending={ack.isPending && ack.variables?.id === a.action_id}
                      error={ackError(a.action_id)}
                      onToggleAck={() => ack.mutate({ id: a.action_id, ack: !a.acknowledged_at })}
                      onHover={setHoverEvent}
                      onSelect={select}
                    />
                  ))}
                </AnimatePresence>
              </div>
            </PanelState>
          </Panel>

          <Panel title="Event timeline" subtitle="Position within the 72 h horizon. Click to pin." delay={0.16}>
            <PanelState q={[ev, fc]} height={160} empty={events.length === 0} emptyTitle={physicsOnly ? 'No events for a transfer region' : 'No grid events'} emptyHint={noDemand ?? 'No over-generation, ramp or deficit flagged in this run.'}>
              {first && <EventTimeline events={events} start={first.t} end={rows[rows.length - 1].t} activeId={hoverEvent} onHover={setHoverEvent} onSelect={select} />}
            </PanelState>
          </Panel>
        </div>
      </div>
    </div>
  );
}
