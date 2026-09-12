# dev-02 — Frontend and Design System

**Stack:** React 18 + TypeScript · Vite · Recharts (+ D3 scales) · Tailwind CSS · TanStack Query
**Audience:** a grid operator scanning a screen, not a data scientist reading a notebook.

> **The design rule that governs everything below:** the top of the screen must be readable in **five
> seconds without interaction**. An operator glances at this between other tasks. If they have to hover,
> filter or scroll to learn whether anything needs attention, the interface has failed regardless of how
> good the forecast is.

---

## 1. Design tokens

These are a **validated** palette — the categorical hues were run through a colour-vision-deficiency
checker for adjacent-pair separation and normal-vision distance, in both light and dark mode. Do not
substitute hues by eye; if you need a new series colour, take the next slot in order.

```css
/* ui/src/styles/tokens.css */
:root {
  color-scheme: light;

  /* surfaces */
  --surface-1:   #fcfcfb;   /* cards, chart backgrounds */
  --surface-2:   #f0efec;   /* table headers, insets */
  --page:        #f9f9f7;   /* app background */

  /* ink */
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #898781;

  /* structure */
  --grid-line: #e1e0d9;
  --baseline:  #c3c2b7;
  --border:    rgba(11, 11, 11, 0.10);

  /* categorical series — fixed order, never cycled */
  --series-1: #2a78d6;  /* blue   → wind          */
  --series-2: #eb6834;  /* orange → solar         */
  --series-3: #1baf7a;  /* aqua   → net load      */
  --series-4: #eda100;  /* yellow → persistence   */

  /* status — reserved, never reused as a series colour */
  --status-good:     #0ca30c;
  --status-warning:  #fab219;
  --status-serious:  #ec835a;
  --status-critical: #d03b3b;

  /* type */
  --font-ui:   system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", monospace;

  /* spacing — 4px base */
  --sp-1: 4px;  --sp-2: 8px;  --sp-3: 12px; --sp-4: 16px;
  --sp-5: 24px; --sp-6: 32px; --sp-7: 48px; --sp-8: 64px;

  --radius-sm: 3px; --radius-md: 6px;
  --shadow-card: 0 1px 2px rgba(11,11,11,.05), 0 8px 24px -18px rgba(11,11,11,.25);
}

@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-1: #1a1a19;  --surface-2: #232322;  --page: #0d0d0d;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
    --grid-line: #2c2c2a;  --baseline: #383835;  --border: rgba(255,255,255,.10);
    --series-1: #3987e5;  --series-2: #d95926;  --series-3: #199e70;  --series-4: #c98500;
  }
}
:root[data-theme="dark"] { /* same overrides as the media block — the toggle must win both ways */ }
```

**Series assignment is fixed.** Wind is always `--series-1`, solar always `--series-2`, net load always
`--series-3`. Colour follows the entity, never its position in a filtered list — if a filter removes wind,
solar must not repaint blue.

### Type scale

| Role | Size / weight | Use |
|---|---|---|
| Display | 32px / 600 | Hero figure on a stat tile |
| H1 | 22px / 600 | Page title |
| H2 | 17px / 600 | Panel title |
| Body | 14px / 400 | Everything |
| Label | 12px / 500, `0.06em` tracking, uppercase | Axis labels, table headers, eyebrows |
| Mono | 13px / 400, `tabular-nums` | All numbers in tables and axes |

Any column of digits gets `font-variant-numeric: tabular-nums`. Numbers that shift horizontally as they
update are the fastest way to make a dashboard feel unreliable.

---

## 2. Screen inventory

| Screen | Route | Purpose |
|---|---|---|
| **Operations** | `/` | The default. Everything an operator needs to act on now. |
| **Forecast detail** | `/forecast/:regionId` | Full 72 h fan chart, per-technology, per-lead-hour |
| **Evidence** | `/evidence` | Skill curve, coverage plot, SHAP — the "why believe this" screen |
| **Planning** | `/planning` | Storage sizing sweep, curtailment analytics |
| **Regions** | `/regions` | Configured regions, archetypes, fleet parameters |

Build **Operations** first and completely. The others are one panel each, lifted from it.

---

## 3. Operations screen layout

```
┌──────────────────────────────────────────────────────────────────────┐
│ STATUS BAR      [Region ▾]  Next event: 6h  ·  3 actions  ·  ● healthy│
├──────────────────────────────────┬───────────────────────────────────┤
│                                  │  ACTION QUEUE                     │
│   FAN CHART                      │  ┌─────────────────────────────┐  │
│   solar + wind, P10–P90          │  │ CHARGE  42 MWh  ₹1.8L  ●●●  │  │
│   actuals overlay, 72 h          │  ├─────────────────────────────┤  │
│                                  │  │ CURTAIL 18 MWh  ₹0.5L  ●●○  │  │
├──────────────────────────────────┤  └─────────────────────────────┘  │
│                                  │                                   │
│   NET LOAD STACK                 │  EVENT TIMELINE                   │
│   demand · renewables · must-run │  ▓▓ over-gen  11:00–14:00  in 35h │
│   headroom shaded, events marked │  ░░ ramp      18:00–20:00  in 42h │
└──────────────────────────────────┴───────────────────────────────────┘
```

Desktop: 2-column grid, main panel `1fr`, side rail `380px`.
Below 1024px: single column, action queue moves above the charts — on a small screen the *decisions*
matter more than the curves.

---

## 4. Component inventory

Build in this order. Each is independently testable with mock data.

| # | Component | Props (abridged) |
|---|---|---|
| 1 | `<StatusBar>` | `region, onRegionChange, health, nextEvent, openActions` |
| 2 | `<StatTile>` | `label, value, unit, delta?, status?, sparkline?` |
| 3 | `<FanChart>` | `data: ForecastPoint[], techs, showActuals, highlight?` |
| 4 | `<NetLoadChart>` | `data: OutlookPoint[], mustRun, events` |
| 5 | `<ActionCard>` | `action: Recommendation, onAcknowledge` |
| 6 | `<EventTimeline>` | `events: GridEvent[], now, horizonHours` |
| 7 | `<SkillCurve>` | `data: LeadHourMetric[], series[]` |
| 8 | `<CoveragePlot>` | `nominal[], observed[]` |
| 9 | `<SweepChart>` | `data: SweepPoint[], knee?` |
| 10 | `<ProvenanceFooter>` | `modelVersion, issuedAt, calibrationDate, replayMode` |

### `<StatTile>`

Use sparingly. A row of five tiles where only two carry meaning trains people to ignore all five.
Reserve them for: current output, next event lead time, open actions, model health.

```tsx
interface StatTileProps {
  label: string;
  value: number | string;
  unit?: string;
  delta?: { value: number; goodDirection: 'up' | 'down' };
  status?: 'good' | 'warning' | 'serious' | 'critical';
  sparkline?: number[];
}
```

Status colour never travels alone — pair it with an icon and a text label. A colourblind operator, a
printed screenshot and a forced-colours display all lose hue.

### `<ActionCard>`

The most important component on the screen. It must communicate four things at a glance:
**what to do, how much, what it is worth, and how sure we are.**

```tsx
interface ActionCardProps {
  action: Recommendation;
  onAcknowledge?: (id: string) => void;
}
```

```
┌───────────────────────────────────────────────┐
│ ● CHARGE STORAGE            Tue 11:00–14:00   │  ← status dot + verb + window
│                                               │
│   42 MWh              ₹1.8 lakh avoided       │  ← size and value, large
│                                               │
│   ●●● decisive                                │  ← confidence, see below
│   Whole P10–P90 band clears the must-run      │
│   floor. Battery has 210 MWh of headroom.     │  ← plain-language rationale
│                                     [ Ack ]   │
└───────────────────────────────────────────────┘
```

**Confidence is two-state, not a percentage.**

- **Decisive** (`●●●`) — the *entire* P10–P90 band clears the threshold. This is a recommendation.
- **Indicative** (`●●○`) — the P50 crosses but the band straddles it. This is a suggestion.

An operator does not need a probability; they need to know whether the action is safe to take on the
forecast alone. Reducing it to two states is the whole point.

---

## 5. Chart specifications

Charts here are read for decisions, so they follow stricter rules than a report chart.

### Universal rules

- **One y-axis. Never two.** Two measures at different scales become two stacked charts sharing an x-axis.
  A dual-axis chart lets the reader infer any correlation you like by changing the scales.
- **Direct labels on ≤ 4 series**, plus a legend. Identity must never be carried by colour alone.
- **Recessive chrome.** Gridlines `--grid-line` at 1px, axis `--baseline`, tick labels `--text-muted`.
- **Chart text uses ink tokens, never the series colour** — a coloured mark beside the label carries
  identity.
- **Every axis label names a value the chart actually reaches.** No axis ending at 8,000 when the data
  peaks at 5,200.
- **Crosshair + tooltip on every time-series chart.** These are interactive by nature.

### `<FanChart>`

```tsx
<ComposedChart data={rows}>
  <CartesianGrid stroke="var(--grid-line)" vertical={false} />
  <XAxis dataKey="leadHours" tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
         tickLine={false} axisLine={{ stroke: 'var(--baseline)' }} />
  <YAxis unit=" MW" width={62} tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
         tickLine={false} axisLine={false} />

  {/* band first, so the median line draws on top */}
  <Area dataKey="solarBand" stroke="none" fill="var(--series-2)" fillOpacity={0.18} />
  <Area dataKey="windBand"  stroke="none" fill="var(--series-1)" fillOpacity={0.18} />

  <Line dataKey="solarP50" stroke="var(--series-2)" strokeWidth={2} dot={false} />
  <Line dataKey="windP50"  stroke="var(--series-1)" strokeWidth={2} dot={false} />
  <Line dataKey="solarActual" stroke="var(--text-primary)" strokeWidth={1}
        strokeDasharray="4 3" dot={false} />

  <ReferenceArea x1={ev.start} x2={ev.end} fill="var(--status-critical)" fillOpacity={0.10} />
  <Tooltip content={<FanTooltip />} cursor={{ stroke: 'var(--baseline)' }} />
</ComposedChart>
```

Recharts wants band data as `[lower, upper]` tuples per point:

```ts
const rows = points.map(p => ({
  leadHours: p.lead_hours,
  solarBand: [p.p10_mw, p.p90_mw] as [number, number],
  solarP50: p.p50_mw,
}));
```

**Night shading.** Shade `is_day === 0` columns with `--grid-line` at 40 % opacity. It costs nothing and
instantly explains why solar is zero for a third of the chart — without it, first-time viewers ask.

### `<NetLoadChart>`

Stacked area: demand as the envelope, solar and wind as what is subtracted, must-run as a dashed
`--status-critical` reference line. Shade where `headroom < 0` and label the window with its **MWh
surplus** — a colour with no number attached is not actionable.

### `<CoveragePlot>`

The differentiating chart, and it is small. Nominal coverage on x, observed on y, a 45° reference line,
and your points on it. If the points sit on the line, the intervals are honest. Fifteen seconds to read,
and almost nobody else will show one.

---

## 6. State management

```
Server state  → TanStack Query   (forecasts, outlook, events, actions)
URL state     → React Router     (region, run_ts, selected event)
UI state      → useState/Context (theme, panel collapse, acknowledged actions)
```

Put the region and selected run in the **URL**, not component state. An operator sharing a link to a
specific event is the single most valuable collaboration feature this UI can have, and it is free.

```ts
// ui/src/hooks/useForecast.ts
export function useForecast(regionId: string, runTs?: string) {
  return useQuery({
    queryKey: ['forecast', regionId, runTs ?? 'latest'],
    queryFn: () => api.getForecast({ regionId, runTs, horizonHours: 72 }),
    staleTime: 5 * 60_000,
    refetchInterval: 5 * 60_000,       // new run lands every 6 h; poll is cheap insurance
    retry: 2,
  });
}
```

---

## 7. API client — generate it, do not write it

The backend publishes an OpenAPI schema. Generating the client means the frontend can never drift from the
backend's types, and it removes an entire category of integration bug at zero cost.

```bash
npm i -D openapi-typescript
npx openapi-typescript http://localhost:8000/openapi.json -o src/api/schema.d.ts
```

```ts
// ui/src/api/client.ts
import type { paths } from './schema';

type ForecastResponse =
  paths['/forecast']['get']['responses']['200']['content']['application/json'];

const BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

export const api = {
  async getForecast(p: { regionId: string; runTs?: string; horizonHours?: number }) {
    const q = new URLSearchParams({
      region_id: p.regionId,
      ...(p.runTs && { run_ts: p.runTs }),
      horizon_hours: String(p.horizonHours ?? 72),
    });
    const r = await fetch(`${BASE}/forecast?${q}`);
    if (!r.ok) throw new ApiError(r.status, await r.text());
    return (await r.json()) as ForecastResponse;
  },
};
```

Add `"gen:api": "openapi-typescript http://localhost:8000/openapi.json -o src/api/schema.d.ts"` to
`package.json` scripts and re-run it whenever the backend changes.

---

## 8. Mock-first development

The frontend must not wait for real forecasts. Build against fixtures from hour one.

```ts
// ui/src/mocks/forecast.ts — realistic shapes, not lorem
export const mockForecast: ForecastResponse = {
  region_id: 'BE',
  issued_at: '2026-03-14T00:00:00Z',
  model_version: 'residual-gbdt-v3.2',
  calibration_date: '2026-03-13',
  nwp_models: ['ecmwf_ifs025', 'icon_seamless', 'gfs_seamless'],
  replay_mode: true,
  data: generateDiurnalProfile(72),   // sinusoidal solar, Weibull-ish wind, widening bands
};
```

Make the mock **physically plausible** — solar zero at night, peaking near local noon; bands widening with
lead time. A mock with flat random noise hides every layout bug that real data will expose.

---

## 9. Empty, loading and error states

Every panel needs all four states designed, not just the happy path.

| State | Treatment |
|---|---|
| Loading | Skeleton with the chart's real dimensions — never a spinner that collapses the layout |
| Empty | "No forecast for this region yet." + what to do about it |
| Error | What failed and the retry action. Never a bare "Something went wrong." |
| Stale | Render the data **and** a banner: "Showing the 09:00 run — the 15:00 run has not landed." |

**Stale is the one people forget**, and it is the most operationally dangerous. Silently showing an old
forecast as if it were current is worse than showing nothing.

---

## 10. Replay mode is visible

When `replay_mode: true` comes back from the API, show a persistent chip in the status bar:
**`REPLAY — cached 14 Mar 2026`**.

This is an integrity feature, not a debug flag. During a demo you will be serving from cache; saying so on
screen is what makes it honest rather than a thing someone discovers.

---

## 11. Accessibility

- Contrast ≥ 4.5:1 for body text, ≥ 3:1 for large text and UI borders.
- Status is never colour-alone — always icon + text label.
- Every chart has a table view toggle. This serves screen readers *and* the operator who wants the exact
  number, so it earns its place twice.
- Full keyboard navigation; visible `:focus-visible` ring on every interactive element.
- Respect `prefers-reduced-motion` — no chart entrance animations when it is set.

---

## 12. Project setup

```bash
npm create vite@latest ui -- --template react-ts
cd ui
npm i recharts @tanstack/react-query react-router-dom date-fns clsx
npm i -D tailwindcss @tailwindcss/vite openapi-typescript vitest @testing-library/react
```

```
ui/src/
├── api/          client.ts · schema.d.ts (generated)
├── components/   charts/ · cards/ · layout/ · primitives/
├── hooks/        useForecast · useOutlook · useActions · useEvents
├── mocks/        fixture data
├── pages/        Operations · ForecastDetail · Evidence · Planning · Regions
├── styles/       tokens.css · index.css
└── lib/          format.ts · time.ts
```

`lib/format.ts` centralises number formatting — do it once or you will get `1234.5678 MW` on some screens
and `1,235 MW` on others:

```ts
export const mw   = (v: number) => `${v.toLocaleString('en-IN', { maximumFractionDigits: 0 })} MW`;
export const mwh  = (v: number) => `${v.toLocaleString('en-IN', { maximumFractionDigits: 0 })} MWh`;
export const inr  = (v: number) =>
  v >= 1e7 ? `₹${(v / 1e7).toFixed(1)} Cr` :
  v >= 1e5 ? `₹${(v / 1e5).toFixed(1)} L`  :
             `₹${v.toLocaleString('en-IN')}`;
```

Lakh and crore, not millions — the audience is Indian grid operators.

---

## 13. Frontend checklist

- [ ] Tokens defined in `tokens.css`; no hardcoded hex anywhere in components
- [ ] Both light and dark palettes defined at token level, toggle wins over OS in both directions
- [ ] Series colours fixed per entity, never repainted by a filter
- [ ] API client generated from `/openapi.json`, not hand-written
- [ ] Every panel has loading, empty, error and stale states
- [ ] No dual-axis chart anywhere
- [ ] Direct labels plus legend on every multi-series chart
- [ ] Numbers use `tabular-nums` and the shared formatters
- [ ] Region and run in the URL, so a view is shareable
- [ ] Replay mode visible in the status bar
- [ ] Readable at 400px width
- [ ] Top of Operations screen readable in five seconds without interaction
