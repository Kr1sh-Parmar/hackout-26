import { useParams, useSearchParams } from 'react-router-dom';

/**
 * Region, run, hour and tech live in the URL, not component state, so an operator can share
 * a link to exactly what they are looking at.
 */
export function useConsoleParams() {
  const { region = 'BE' } = useParams();
  const [sp, setSp] = useSearchParams();

  const set = (key: string, value: string | number | null | undefined) =>
    setSp(
      (prev) => {
        const next = new URLSearchParams(prev);
        if (value == null || value === '') next.delete(key);
        else next.set(key, String(value));
        return next;
      },
      { replace: true },
    );

  return { region, runTs: sp.get('run_ts') ?? undefined, param: (k: string) => sp.get(k), set };
}
