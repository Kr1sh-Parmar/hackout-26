import { formatDistanceToNowStrict } from 'date-fns';

// One place for number formatting, so no screen shows `1234.5678 MW` while another shows `1,235 MW`.
const n0 = (v: number) => v.toLocaleString('en-IN', { maximumFractionDigits: 0 });

export const mw = (v: number) => `${n0(v)} MW`;
export const mwh = (v: number) => `${n0(v)} MWh`;
export const gwh = (v: number) => `${v.toLocaleString('en-IN', { maximumFractionDigits: 1 })} GWh`;
export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;

// Lakh and crore, not millions: the audience is Indian grid operators.
export const inr = (v: number) => {
  const a = Math.abs(v);
  const s = v < 0 ? '−' : '';
  if (a >= 1e7) return `${s}₹${(a / 1e7).toFixed(1)} Cr`;
  if (a >= 1e5) return `${s}₹${(a / 1e5).toFixed(1)} L`;
  return `${s}₹${n0(a)}`;
};

/*
 * Timestamps arrive as UTC and are converted only here, into the REGION's timezone from /sites,
 * not the browser's. A Belgian solar peak must read 13:30 Brussels, not 17:00 IST.
 * ponytail: module-level setting written by App on navigation; pass tz explicitly if two regions
 * ever share one screen.
 */
let tz: string | undefined;
let dtf = makeFormatter();

function makeFormatter() {
  const opts: Intl.DateTimeFormatOptions = { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' };
  try {
    return new Intl.DateTimeFormat('en-GB', { ...opts, timeZone: tz });
  } catch {
    return new Intl.DateTimeFormat('en-GB', opts); // unknown zone name: fall back to the browser's
  }
}

export function setDisplayTz(next?: string) {
  if (next === tz) return;
  tz = next;
  dtf = makeFormatter();
}
export const displayTz = () => dtf.resolvedOptions().timeZone;

function parts(t: number) {
  const o: Record<string, string> = {};
  for (const p of dtf.formatToParts(t)) o[p.type] = p.value;
  return o;
}

export const hourIn = (t: number) => Number(parts(t).hour);

/** Tokens: EEE weekday, d day, MMM month, yyyy year, HH hour, mm minute. */
export function fmtTime(t: number | string, pattern: string) {
  const p = parts(typeof t === 'string' ? Date.parse(t) : t);
  const map: Record<string, string> = { EEE: p.weekday, MMM: p.month, yyyy: p.year, HH: p.hour, mm: p.minute, d: p.day };
  return pattern.replace(/EEE|MMM|yyyy|HH|mm|d/g, (k) => map[k]);
}

export const localTime = (iso: string, pattern = 'EEE HH:mm') => fmtTime(iso, pattern);
export const utcTime = (iso: string) => `${new Date(iso).toISOString().slice(0, 16).replace('T', ' ')} UTC`;
export const ago = (iso: string) => formatDistanceToNowStrict(new Date(iso), { addSuffix: true });
// Rows are hourly intervals and `valid_to` is the START of the last hour, so the window ends an hour later.
export const window_ = (from: string, to: string) => `${fmtTime(from, 'EEE HH:mm')}–${fmtTime(Date.parse(to) + 3_600_000, 'HH:mm')}`;

export const minutesLabel = (m: number) =>
  m < 60 ? `${Math.round(m)} min` : m < 48 * 60 ? `${(m / 60).toFixed(1)} h` : `${(m / 1440).toFixed(1)} d`;

export const humanise = (s: string) => s.replace(/_/g, ' ').toLowerCase().replace(/^\w/, (c) => c.toUpperCase());
