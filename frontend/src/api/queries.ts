import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  get,
  request,
  type AckResponse,
  type ActionsResponse,
  type BacktestResponse,
  type EventsResponse,
  type ExplainResponse,
  type ForecastResponse,
  type Health,
  type OutlookResponse,
  type RunsResponse,
  type SiteInfo,
  type SweepResponse,
  type Tech,
} from './client';

// A new run lands every 6 h; a 5-minute poll is cheap insurance, not a live feed.
const POLL = { staleTime: 5 * 60_000, refetchInterval: 5 * 60_000 };

// Config never changes at runtime, but a fetch that FAILED must heal: region timezone and the region
// switcher both hang off this query, and nothing else would ever retry it after an API outage.
export const useSites = () =>
  useQuery({
    queryKey: ['sites'],
    queryFn: () => get<SiteInfo[]>('/sites'),
    staleTime: Infinity,
    refetchInterval: (q) => (q.state.status === 'error' ? 10_000 : false),
  });

/** The region's site config, and whether it is a physics-only transfer region. */
export function useSite(region: string) {
  const sites = useSites();
  const site = sites.data?.find((s) => s.region_id === region);
  return { site, physicsOnly: !!site?.physics_only, sites };
}

export const useHealth = () =>
  useQuery({ queryKey: ['health'], queryFn: () => get<Health>('/health'), ...POLL, refetchInterval: 60_000 });

export const useRuns = (region: string) =>
  useQuery({ queryKey: ['runs', region], queryFn: () => get<RunsResponse>('/runs', { region_id: region }), ...POLL });

export const useForecast = (region: string, opts: { runTs?: string; horizon?: number; tech?: Tech } = {}) =>
  useQuery({
    queryKey: ['forecast', region, opts],
    queryFn: () =>
      get<ForecastResponse>('/forecast', {
        region_id: region,
        run_ts: opts.runTs,
        horizon_hours: opts.horizon ?? 72,
        tech: opts.tech,
      }),
    ...POLL,
  });

export const useOutlook = (region: string, runTs?: string, horizon = 72) =>
  useQuery({
    queryKey: ['outlook', region, runTs, horizon],
    queryFn: () => get<OutlookResponse>('/outlook', { region_id: region, run_ts: runTs, horizon_hours: horizon }),
    ...POLL,
  });

export const useEvents = (region: string, runTs?: string, minSeverity = 1) =>
  useQuery({
    queryKey: ['events', region, runTs, minSeverity],
    queryFn: () => get<EventsResponse>('/events', { region_id: region, run_ts: runTs, min_severity: minSeverity }),
    ...POLL,
  });

export const actionsKey = (region: string, runTs?: string) => ['actions', region, runTs ?? null];

export const useActions = (region: string, runTs?: string) =>
  useQuery({
    queryKey: actionsKey(region, runTs),
    queryFn: () => get<ActionsResponse>('/actions', { region_id: region, run_ts: runTs }),
    ...POLL,
  });

/** Acknowledge or withdraw, persisted by the API (not the browser), then reflected in the cached actions. */
export function useAckAction(region: string, runTs?: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, ack }: { id: string; ack: boolean }) =>
      request<AckResponse>(ack ? 'POST' : 'DELETE', `/actions/${encodeURIComponent(id)}/ack`, { region_id: region, run_ts: runTs }),
    onSuccess: (res) =>
      qc.setQueryData<ActionsResponse>(actionsKey(region, runTs), (old) =>
        old && { ...old, data: old.data.map((a) => (a.action_id === res.action_id ? { ...a, acknowledged_at: res.acknowledged_at ?? null } : a)) },
      ),
  });
}

export const useSweep = (region: string, maxMwh: number, steps: number, runTs?: string) =>
  useQuery({
    queryKey: ['sweep', region, maxMwh, steps, runTs],
    queryFn: () => get<SweepResponse>('/storage/sweep', { region_id: region, max_mwh: maxMwh, steps, run_ts: runTs }),
    placeholderData: keepPreviousData, // slider drags must not blank the chart
    staleTime: 10 * 60_000,
  });

export const useBacktest = (region: string, tech: Tech) =>
  useQuery({
    queryKey: ['backtest', region, tech],
    queryFn: () => get<BacktestResponse>('/backtest', { region_id: region, tech }),
    staleTime: Infinity, // aggregated across the whole walk-forward; changes only on retrain
  });

export const useExplain = (region: string, opts: { tech?: Tech; validTs?: string; runTs?: string; topN?: number }) =>
  useQuery({
    queryKey: ['explain', region, opts],
    queryFn: () =>
      get<ExplainResponse>('/explain', {
        region_id: region,
        tech: opts.tech,
        valid_ts: opts.validTs,
        run_ts: opts.runTs,
        top_n: opts.topN ?? 8,
      }),
    enabled: !!opts.validTs,
    placeholderData: keepPreviousData,
    staleTime: 10 * 60_000,
  });
