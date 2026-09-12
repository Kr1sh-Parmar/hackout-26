import {
  BatteryCharging,
  BatteryMedium,
  CheckCircle2,
  Factory,
  FlaskConical,
  HelpCircle,
  PauseCircle,
  Scissors,
  ShieldCheck,
  Sun,
  TrendingDown,
  TrendingUp,
  Wind,
  type LucideIcon,
} from 'lucide-react';

export type Status = 'good' | 'warning' | 'serious' | 'critical';

// Status colour never travels alone: every use pairs it with an icon and this text label.
export const STATUS: Record<Status, { label: string; chip: string; dot: string }> = {
  good: { label: 'Good', chip: 'bg-green-100/70 text-green-800 border-green-200', dot: 'bg-status-good' },
  warning: { label: 'Watch', chip: 'bg-amber-100/70 text-amber-800 border-amber-200', dot: 'bg-status-warning' },
  serious: { label: 'Serious', chip: 'bg-orange-100/80 text-orange-800 border-orange-200', dot: 'bg-status-serious' },
  critical: { label: 'Critical', chip: 'bg-red-100/80 text-red-800 border-red-200', dot: 'bg-status-critical' },
};

export const sevStatus = (severity: number): Status => (severity >= 4 ? 'critical' : severity === 3 ? 'serious' : 'warning');
export const statusVar = (s: Status) => `var(--status-${s})`;

export const FLAG: Record<string, { label: string; icon: LucideIcon }> = {
  NORMAL: { label: 'Normal', icon: CheckCircle2 },
  OVER_GENERATION: { label: 'Over-generation', icon: Sun },
  STEEP_RAMP: { label: 'Steep ramp', icon: TrendingUp },
  DEFICIT_RISK: { label: 'Deficit risk', icon: TrendingDown },
  STORM_SHUTDOWN: { label: 'Storm shutdown', icon: Wind },
  LOW_CONFIDENCE: { label: 'Low confidence', icon: HelpCircle },
};

export const ACTION: Record<string, { label: string; icon: LucideIcon }> = {
  HOLD: { label: 'Hold', icon: PauseCircle },
  CURTAIL: { label: 'Curtail', icon: Scissors },
  CHARGE_BESS: { label: 'Charge storage', icon: BatteryCharging },
  DISCHARGE_BESS: { label: 'Discharge storage', icon: BatteryMedium },
  COMMIT_BACKUP: { label: 'Commit backup', icon: Factory },
};

export const TECH_COLOR = { solar: 'var(--series-solar)', wind: 'var(--series-wind)' } as const;

// Display names only. A region missing here still works everywhere under its id.
export const REGION_NAME: Record<string, string> = { BE: 'Belgium · Elia', IN: 'India' };
export const regionName = (id: string) => REGION_NAME[id] ?? id;

// A region's role follows from `/sites` physics_only, never from its id.
export const REGION_ROLE = {
  validation: {
    label: 'Validation region',
    icon: ShieldCheck,
    chip: 'bg-green-100/70 text-green-800 border-green-200',
    blurb: 'Metered generation, the TSO forecast and load on one timebase. The model is trained, calibrated and scored here.',
  },
  transfer: {
    label: 'Transfer region · physics-only',
    icon: FlaskConical,
    chip: 'bg-blue-100/70 text-blue-800 border-blue-200',
    blurb: 'No metered labels to train or calibrate against: a physics-only forecast from live weather, every band marked uncalibrated, and no accuracy claim.',
  },
};
export const roleOf = (site?: { physics_only?: boolean }) => REGION_ROLE[site?.physics_only ? 'transfer' : 'validation'];
