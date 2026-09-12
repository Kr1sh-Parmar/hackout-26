import { type ReactNode } from 'react';
import { motion } from 'framer-motion';
import { AlertOctagon, Inbox, RotateCw } from 'lucide-react';
import { ApiError } from '../../api/client';

export function Panel({
  title,
  subtitle,
  right,
  className = '',
  children,
  delay = 0,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  right?: ReactNode;
  className?: string;
  children: ReactNode;
  delay?: number;
}) {
  return (
    <motion.section
      className={`glass-panel p-4 md:p-5 ${className}`}
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay, ease: [0.21, 0.47, 0.32, 0.98] }}
    >
      {(title || right) && (
        <header className="flex flex-wrap items-start justify-between gap-3 mb-4">
          <div>
            {title && <h2 className="text-[15px] font-semibold text-slate-800">{title}</h2>}
            {subtitle && <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      {children}
    </motion.section>
  );
}

type Q = { isPending: boolean; isError: boolean; error: unknown; refetch: () => unknown };

/**
 * Loading, error and empty are designed states, not afterthoughts. Stale is handled once in
 * ConsoleLayout. Loading keeps the panel's real height so the layout never jumps.
 */
export function PanelState({
  q,
  height,
  empty,
  emptyTitle = 'Nothing to show',
  emptyHint,
  children,
}: {
  q: Q | Q[];
  height: number | string;
  empty?: boolean;
  emptyTitle?: ReactNode;
  emptyHint?: ReactNode;
  children: ReactNode;
}) {
  const qs = Array.isArray(q) ? q : [q];
  const failed = qs.find((x) => x.isError);
  if (failed) return <ErrorState error={failed.error} onRetry={() => qs.forEach((x) => x.refetch())} height={height} />;
  if (qs.some((x) => x.isPending)) return <Skeleton height={height} />;
  if (empty)
    return (
      <div style={{ minHeight: height }} className="flex flex-col items-center justify-center text-center gap-2 px-6">
        <Inbox className="w-8 h-8 text-slate-300" />
        <p className="text-sm font-semibold text-slate-700">{emptyTitle}</p>
        {emptyHint && <p className="text-xs text-slate-500 max-w-sm">{emptyHint}</p>}
      </div>
    );
  return <>{children}</>;
}

export function Skeleton({ height, className = '' }: { height: number | string; className?: string }) {
  return (
    <div style={{ height }} className={`relative overflow-hidden rounded-2xl bg-white/40 ${className}`} aria-busy="true" aria-label="Loading">
      <div className="absolute inset-0 -translate-x-full animate-[shimmer-sweep_1.4s_infinite] bg-gradient-to-r from-transparent via-white/60 to-transparent" />
    </div>
  );
}

export function ErrorState({ error, onRetry, height }: { error: unknown; onRetry: () => void; height: number | string }) {
  const e = error instanceof ApiError ? error : null;
  return (
    <div role="alert" style={{ minHeight: height }} className="flex flex-col items-center justify-center text-center gap-2 px-6">
      <AlertOctagon className="w-8 h-8 text-status-critical" />
      <p className="text-sm font-semibold text-slate-800">
        {e ? e.message : String(error)}
        {e && e.status > 0 && <span className="ml-2 font-mono text-[11px] text-slate-400">{e.status} {e.error}</span>}
      </p>
      {e?.hint && <p className="text-xs text-slate-500 max-w-md">{e.hint}</p>}
      <button
        onClick={onRetry}
        className="mt-2 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold bg-white/70 border border-white/60 text-slate-700 hover:bg-white"
      >
        <RotateCw className="w-3.5 h-3.5" /> Retry
      </button>
    </div>
  );
}
