# Premium SaaS Analytics Dashboard - Implementation Plan

You've requested a complete architectural pivot to a modern, light-themed, glassmorphic SaaS dashboard. This plan outlines the technical and structural approach to building this specific design system.

## Proposed Tech Stack
- **Framework:** React 18 with TypeScript, bundled by Vite.
- **Styling:** Tailwind CSS. We will extend the Tailwind config to include the specific "Light Blue/White" theme, bright blue accents, and soft diffuse shadows.
- **Charts:** Recharts. Styled with incredibly thin accent lines, subtle area fills, and minimal grid chrome.
- **Icons:** Lucide-react (clean, scaleable, and matches premium SaaS vibes).

## Design System Translation
- **Base Shell:** A full-screen environmental image overlaid with a `backdrop-blur-2xl bg-white/70` glassmorphic application window, massive rounded corners (`rounded-[2rem]`), and subtle white structural borders.
- **Color Palette:**
  - *Accent:* Bright Blue (`#2563eb` or `blue-600`)
  - *Text:* Midnight (`#0f172a`), Muted (`#64748b` - gray-blue)
  - *Status:* Green (Positive), Amber (Warning), Red (Negative)
- **Formatting:** Generous padding (`p-6` to `p-8`), large border radius (`rounded-2xl`), and soft, diffuse drop shadows (`shadow-xl shadow-blue-500/5`).

## Proposed Component Architecture

### `frontend/src/`

#### 1. Layout Shell (`components/Layout/`)
- **`DashboardShell.tsx`**: The main glassmorphic wrapper that houses the Sidebar and Content area.
- **`Sidebar.tsx`**: Persistent left navigation. Contains Brand logo, icon+label nav items (active state: solid bright blue pill). Collapsible on mobile.
- **`Header.tsx`**: Minimal top bar inside the content area. Contains a rounded search field and circular utility icons.

#### 2. Dashboard Grid (`components/Dashboard/`)
- **`HeroBanner.tsx`**: Large horizontal card with strong typography on the left, an atmospheric image section on the right, and highlighted primary blue words.
- **`KPIGrid.tsx`**: Responsive grid (2x2 or 4x1) of equal-sized cards featuring small icons, robust numeric values, and mini sparkline visualizations.
- **`AnalyticsSection.tsx`**: The asymmetric split container.
  - *Left (70%)*: **`PrimaryChart.tsx`** (Clean Recharts area chart with minimal lines).
  - *Right (30%)*: **`SecondaryMetrics.tsx`** (Compact cards with ring indicators).
- **`DataTable.tsx`**: Reusable table component housed inside a rounded card, featuring subtle row separators and colored status pills.
- **`SystemVisualization.tsx`**: A visual summary card mapping relationships (e.g., Solar → Grid → Battery) creatively centered with surrounding metrics.

## The core idea

Build a premium, light, rounded SaaS monitoring dashboard with a persistent sidebar, minimal header, large hero banner, KPI grid, asymmetric analytics layout, card-based information architecture, generous whitespace, subtle glassmorphism, and a strong blue/green semantic color system.
