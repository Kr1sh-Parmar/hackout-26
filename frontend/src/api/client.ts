import type { components } from './schema';

type S = components['schemas'];
export type Tech = S['Tech'];
// Provenance is only a Pydantic base class, so OpenAPI inlines it into each response.
export type Provenance = Omit<S['ForecastResponse'], 'data'>;
export type ForecastResponse = S['ForecastResponse'];
export type ForecastPoint = S['ForecastPoint'];
export type OutlookResponse = S['OutlookResponse'];
export type OutlookPoint = S['OutlookPoint'];
export type EventsResponse = S['EventsResponse'];
export type GridEvent = S['GridEvent'];
export type ActionsResponse = S['ActionsResponse'];
export type Recommendation = S['Recommendation'];
export type AckResponse = S['AckResponse'];
export type RunsResponse = S['RunsResponse'];
export type SweepResponse = S['SweepResponse'];
export type SweepPoint = S['SweepPoint'];
export type BacktestResponse = S['BacktestResponse'];
export type BacktestSummary = S['BacktestSummary'];
export type LeadHourMetric = S['LeadHourMetric'];
export type ExplainResponse = S['ExplainResponse'];
export type Driver = S['Driver'];
export type Health = S['Health'];
export type SiteInfo = S['SiteInfo'];

// Same-origin by default: Vite (dev) and nginx (docker) proxy /api to FastAPI, so the
// browser needs no CORS and no hardcoded host. VITE_API_BASE points it anywhere else.
export const API_BASE: string = import.meta.env.VITE_API_BASE ?? '/api';

/** Every failure the UI can render: the backend's {error, message, hint}, or our own for a dead backend. */
export class ApiError extends Error {
  constructor(
    public status: number,
    public error: string,
    message: string,
    public hint?: string,
  ) {
    super(message);
  }
}

type Params = Record<string, string | number | undefined | null>;

const unreachable = () =>
  new ApiError(
    0,
    'ApiUnreachable',
    `Cannot reach the forecast API (${API_BASE}).`,
    'Start the backend: make demo  (or REPLAY_MODE=true uvicorn src.api.main:app --port 8000)',
  );

export async function request<T>(method: 'GET' | 'POST' | 'DELETE', path: string, params: Params = {}): Promise<T> {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v != null && v !== '') q.set(k, String(v));

  let r: Response;
  try {
    r = await fetch(`${API_BASE}${path}${q.size ? `?${q}` : ''}`, { method });
  } catch {
    throw unreachable();
  }
  if (!r.ok) {
    const text = await r.text();
    let body: { error?: string; message?: string; hint?: string; detail?: unknown } = {};
    try {
      body = JSON.parse(text);
    } catch {
      /* not JSON: a proxy answering, not FastAPI */
    }
    // With the backend down, the proxy answers instead: Vite with an empty 500, nginx with
    // 502/504. FastAPI's own errors always carry `error` (domain) or `detail` (validation).
    if (!body.error && !body.detail && ([502, 503, 504].includes(r.status) || (r.status === 500 && !text))) {
      throw unreachable();
    }
    const message = body.message ?? (body.detail ? JSON.stringify(body.detail) : `${r.status} ${r.statusText}`);
    throw new ApiError(r.status, body.error ?? 'HttpError', message, body.hint);
  }
  return r.json() as Promise<T>;
}

export const get = <T,>(path: string, params: Params = {}) => request<T>('GET', path, params);
