# EnergiSaaS - Design System & Layout Logic

This document outlines the complete architectural layout, visual language, and specific Tailwind CSS parameters used to build the responsive, premium glassmorphic frontend.

## 1. Global Background & Environment Layer
Instead of using flat colors, the application sits on top of a dynamic atmospheric background to provide a sense of depth and modernity.
- **Component:** `index.html` structure.
- **Logic:** We use a full-screen fixed image with `object-cover`. This image sits at `z-[-1]`, underneath the entire React application mounting point.
- **Why:** This provides the vibrant environmental context (e.g., the solar farm photo) that refracts through the glassmorphic layers stacked above it.

## 2. The Dashboard Shell (The Master Pane)
The entire application (Sidebar + Header + Dashboard Content) is encased in a single, massive floating glass window.
- **Component:** `DashboardShell.tsx`
- **Classes:** `bg-white/30 backdrop-blur-xl rounded-[2rem] border border-white/40 shadow-2xl shadow-blue-900/10`
- **Breakdown of parameters:**
  - `bg-white/30`: Creates an ultra-light 30% opaque white tint.
  - `backdrop-blur-xl`: The core glass filter. It heavily blurs the background image directly behind the pane, mimicking frosted glass.
  - `rounded-[2rem]`: Massive, sweeping 32px rounded corners, standard in premium iOS/SaaS design.
  - `border-white/40`: A semi-transparent 1px stroke that simulates the edge-lighting of physical glass.
  - `shadow-blue-900/10`: A highly diffuse, soft blue shadow that lifts the pane off the screen.

## 3. Nested Glass Elements (The Card System)
Every individual data card inside the dashboard (KPIs, Charts, Tables) is built as a distinct, smaller pane of glass floating *inside* the wider Dashboard Shell.
- **Base Classes:** `bg-white/40 backdrop-blur-xl rounded-[1.5rem] border border-white/50 shadow-lg`
- **Design Logic:** 
  - Having a blur inside a blur creates incredible visual depth. 
  - We use `bg-white/40` (slightly more opaque than the master pane) to ensure the text printed on these cards remains perfectly readable, without losing the premium frosted aesthetic.
  - Generous internal padding (usually `p-5` or `p-6`) ensures whitespace allows the data to breathe.

## 4. Color Palette & Typography
- **Primary Accent:** Bright Blue (`text-blue-500`, `text-blue-600`). Used for active navigation states, primary chart lines, and emphasis text.
- **Typography:** The `Inter` sans-serif font family.
- **Text Hierarchy:**
  - **Headers/Titles:** `text-slate-800 font-bold`.
  - **Subtitles/Labels:** `text-slate-500 font-medium text-xs` or `text-sm`.
  - **Numeric Data:** We rely heavily on size contrast (e.g., swapping from `text-[10px]` to `text-3xl`) to draw the eye immediately to the KPI.

## 5. Structural Layout Grid
The dashboard utilizes an asymmetric, highly responsive grid system:

### Row 1: The Hero Banner
- A 50/50 horizontal split.
- **Left:** Primary status, large messaging, and localized data pills.
- **Right:** Contextual photo overlaid with a tightly nested **Weather Widget** (`absolute top-4 right-6 z-20`). The widget uses `bg-white/50 backdrop-blur-xl` to stand out aggressively against the photo.

### Row 2: KPI Grid
- A uniform 4-column layout (`grid-cols-1 md:grid-cols-2 lg:grid-cols-4`).
- Each card incorporates an abstract SVG sparkline behind the text (`absolute right-0 bottom-2 opacity-50 z-0`) to provide non-intrusive data context without cluttering the screen.

### Row 3: The Asymmetric Core (70/30 Split)
- **Left Column (approx. 60-70% width):** Focuses on heavy data visualization. Contains the `AnalyticsSection` (Recharts Area chart with sweeping blue gradients) and `SitePerformance` (Data tables).
- **Right Column (approx. 30-40% width):** Focuses on secondary dense summaries. Contains `SecondaryCards` which holds Environmental Impact lists, SVG performance rings, and the mocked Isometric Energy Flow diagram.

## 6. Iconography & Badges
- **Icons:** We use `lucide-react` for crisp vector iconography.
- **Icon Backgrounds:** Instead of raw icons, we place them inside tinted circles matching their intent. For example, a success icon will use `text-green-500` inside a `bg-green-100/50` circle. This softens the design and adds a pop of color to the monochromatic glass.
