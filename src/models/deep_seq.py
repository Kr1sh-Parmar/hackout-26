"""Rungs 6 and 7: sequence-over-the-horizon deep models (benchmark only).

THE SEQUENCE AXIS IS THE FORECAST HORIZON, NOT A HISTORY OF PAST OUTPUT.
One 00:00 UTC run produces 49 lead hours (24..72). This module predicts all 49
residuals in a single forward pass. That is the only "sequence" this problem
has: at lead hour 48 `power_lag_1h` does not exist, so a recurrent model over
past generation would be scoring a backtest on data the live system never sees.
The permitted lags are the existing 24 h / 168 h ones, and they arrive as plain
features like everything else.

Rung 6 is the trunk alone. Rung 7 is the trunk plus a spatial branch over the
NWP grid points, and the honest claim for it is "learned spatial aggregation vs.
the fixed capacity weights" -- there is no per-turbine data in this project, so
nothing here models wakes.

Neither rung is served. `scripts/train_deep.py` measures them against the
LightGBM incumbent and writes the comparison; `registry.py` never sees them.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..ingest.regional import WX_VARS
from .residual_gbdt import QUANTILES

# The lead axis is a CONSTANT, never inferred per run: 4 of the 922 runs are
# ragged (1, 22, 24, 46, 47 rows) and inferring the axis from them yields ragged
# tensors that silently misalign lead hour 24 of one run with 48 of another.
LEADS: tuple[int, ...] = tuple(range(24, 73))

RUN_KEY = ["region_id", "run_ts_utc"]
GEO = ["latitude", "longitude", "weight"]


# --------------------------------------------------------------------------- #
# reshape: flat rows <-> (run, lead) sequences
# --------------------------------------------------------------------------- #
def _scatter_index(frame: pd.DataFrame, leads: tuple[int, ...]):
    """(run row, lead slot) coordinates for every row, plus the run ordering."""
    region = (
        frame["region_id"].to_numpy()
        if "region_id" in frame.columns
        else np.zeros(len(frame), dtype=int)
    )
    run = pd.to_datetime(frame["run_ts_utc"], utc=True).to_numpy()
    keys = pd.MultiIndex.from_arrays([region, run])
    # run_ts ascending; region only breaks ties, so a single-region fold is
    # ordered purely by issue time.
    uniq = pd.MultiIndex.from_tuples(sorted(set(keys), key=lambda k: (k[1], k[0])))
    r = np.asarray(uniq.get_indexer(keys))
    slot = frame["lead_hours"].to_numpy(dtype=float).astype(int) - leads[0]
    ok = (r >= 0) & (slot >= 0) & (slot < len(leads))
    return r, slot, ok, uniq


def run_timestamps(frame: pd.DataFrame, leads: tuple[int, ...] = LEADS) -> np.ndarray:
    """Issue times of the runs, in the order `to_sequences` stacks them."""
    _, _, _, uniq = _scatter_index(frame, leads)
    ts = pd.DatetimeIndex(pd.to_datetime([k[1] for k in uniq], utc=True))
    return ts.tz_localize(None).to_numpy()


def to_sequences(
    frame: pd.DataFrame,
    x,
    weight=None,
    leads: tuple[int, ...] = LEADS,
):
    """Flat rows -> `(X, mask, pos)` on a fixed (n_runs, len(leads)) grid.

    `X` is zero-filled at gaps, `mask` is the sample weight (0 at gaps, and 0 for
    the censored/curtailed rows the gold layer already marks), and `pos` carries
    the row's positional index in `frame` so `from_sequences` can put the
    predictions back exactly where they came from. `pos == -1` is a pad.
    """
    xv = np.asarray(x.to_numpy() if isinstance(x, pd.DataFrame) else x, dtype=np.float32)
    if xv.ndim == 1:
        xv = xv[:, None]
    r, slot, ok, uniq = _scatter_index(frame, leads)
    n_runs, n_lead = len(uniq), len(leads)

    w = np.ones(len(frame)) if weight is None else np.asarray(weight, dtype=float)

    shape = (n_runs, n_lead, *xv.shape[1:])
    seq = np.zeros(shape, dtype=np.float32)
    mask = np.zeros((n_runs, n_lead), dtype=np.float32)
    pos = np.full((n_runs, n_lead), -1, dtype=np.int64)

    seq[r[ok], slot[ok]] = xv[ok]
    mask[r[ok], slot[ok]] = np.nan_to_num(w[ok]).astype(np.float32)
    pos[r[ok], slot[ok]] = np.arange(len(frame))[ok]
    return seq, mask, pos


def from_sequences(yhat, pos: np.ndarray, n_rows: int) -> np.ndarray:
    """`(n_runs, n_lead, K)` -> `(n_rows, K)` in the original row order.

    Padded slots are DROPPED, never emitted: a row that does not exist must not
    reach the report, where it would be scored against a zero it invented.
    """
    y = np.asarray(yhat, dtype=float)
    out = np.full((n_rows, y.shape[-1]), np.nan, dtype=float)
    sel = pos >= 0
    out[pos[sel]] = y[sel]
    return out


def node_table(wx: pd.DataFrame, leads: tuple[int, ...] = LEADS) -> pd.DataFrame:
    """Per-grid-point weather with `nwp_model` pivoted into columns.

    Indexed by `(run_ts_utc, valid_ts_utc, grid_point_id)`. Missing model/point
    combinations (liege has no gfs_seamless) come out NaN and are handled by the
    standardiser's indicator-plus-median policy, not by a fill of zero.
    """
    d = wx[wx["lead_hours"].isin(leads)]
    vars_ = [v for v in WX_VARS if v in d.columns]
    piv = d.set_index(["run_ts_utc", "valid_ts_utc", "grid_point_id", "nwp_model"])[vars_].unstack(
        "nwp_model"
    )
    piv.columns = [f"{v}__{m}" for v, m in piv.columns]
    return piv.sort_index(axis=1)


def node_meta(wx: pd.DataFrame) -> pd.DataFrame:
    """One lat/lon/weight row per grid point, weights renormalised over the nodes.

    The four usable nodes carry capacity weights summing to 0.88 (namur never
    appears beyond lead hour 23, and is absent here). Renormalising mirrors
    `ingest/regional.capacity_weighted`: a plain weighted sum over the points
    that did report returns a smaller regional value, which reads as calm
    weather rather than as a missing point.
    """
    m = wx.groupby("grid_point_id", observed=True)[GEO].mean().sort_index()
    m["weight"] = m["weight"] / m["weight"].sum()
    return m


def node_matrix(frame: pd.DataFrame, table: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
    """`(n_rows, n_nodes, n_node_features)` aligned to `frame`'s row order."""
    key = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(frame["run_ts_utc"], utc=True),
            pd.to_datetime(frame["valid_ts_utc"], utc=True),
        ]
    )
    blocks = []
    for node in meta.index:
        sub = table.xs(node, level="grid_point_id").reindex(key)
        geo = np.tile(meta.loc[node, GEO].to_numpy(dtype=float), (len(frame), 1))
        blocks.append(np.hstack([sub.to_numpy(dtype=float), geo]))
    return np.stack(blocks, axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# NaN policy + standardisation, fitted on TRAIN ROWS ONLY
# --------------------------------------------------------------------------- #
class Standardiser:
    """Drop-dead columns out, missingness indicators in, then z-score.

    LightGBM eats NaN natively and learns a "missing" branch; torch cannot see a
    NaN without producing one. Imputing to the train median WITHOUT an indicator
    would tell the model "average airmass" at 03:00 -- which is the model
    silently learning that night is average daylight. Hence the `_isna`
    companion column, which is mandatory for `airmass` (47.8% NaN: night).

    Everything here is fitted on the TRAIN slice alone. Fitting on the whole
    fold inflates only the deep model, which is exactly the shape of a fake win.
    """

    def __init__(self, max_nan: float = 0.7) -> None:
        # ponytail: 0.7, not 0.5. `airmass` is NaN every night, so on a training
        # window that spans a Belgian winter it measures 0.48-0.62 missing and a
        # 0.5 gate would reject a column whose missingness IS the signal (and is
        # carried explicitly by `airmass_isna`). Tighten it if a feature ever
        # arrives whose missingness is not itself informative.
        self.max_nan = float(max_nan)

    def fit(self, x: np.ndarray, columns: list[str]) -> Standardiser:
        v = np.asarray(x, dtype=np.float64).reshape(-1, np.shape(x)[-1])
        nan_frac = np.isnan(v).mean(axis=0)
        self.keep = np.flatnonzero(nan_frac < 1.0)
        kept = nan_frac[self.keep]
        bad = [columns[i] for i, f in zip(self.keep, kept) if f > self.max_nan]
        if bad:
            raise ValueError(
                f"columns {bad} are >{self.max_nan:.0%} NaN on train; a median impute there "
                "is a fabricated value, not a fill. Drop them or fix the feature builder."
            )
        self.indicator = self.keep[kept > 0]
        v = v[:, self.keep]
        self.median = np.nanmedian(v, axis=0)
        filled = np.where(np.isnan(v), self.median, v)
        extra = np.isnan(np.asarray(x, dtype=np.float64).reshape(-1, np.shape(x)[-1]))[
            :, self.indicator
        ].astype(np.float64)
        both = np.hstack([filled, extra])
        self.mean = both.mean(axis=0)
        self.std = np.maximum(both.std(axis=0), 1e-6)
        self.columns = [columns[i] for i in self.keep] + [
            f"{columns[i]}_isna" for i in self.indicator
        ]
        return self

    def transform(self, x) -> np.ndarray:
        v = np.asarray(x, dtype=np.float64)
        lead = v.shape[:-1]
        v = v.reshape(-1, v.shape[-1])
        extra = np.isnan(v[:, self.indicator]).astype(np.float64)
        kept = v[:, self.keep]
        kept = np.where(np.isnan(kept), self.median, kept)
        both = np.hstack([kept, extra])
        return (((both - self.mean) / self.std).astype(np.float32)).reshape(*lead, both.shape[1])


# --------------------------------------------------------------------------- #
# architecture
# --------------------------------------------------------------------------- #
class NHiTSBlock(nn.Module):
    """One rate of an N-HiTS-style multi-rate stack.

    Pool the horizon by `r`, push it through an MLP, project straight back to
    the full horizon. Not N-HiTS: there is no doubly-residual backcast.
    """

    def __init__(self, length: int, channels: int, pool_r: int, hidden: int, n_out: int) -> None:
        super().__init__()
        self.r, self.length, self.n_out = pool_r, length, n_out
        pooled = math.ceil(length / pool_r)
        self.g = nn.Sequential(
            nn.Linear(pooled * channels, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.fc = nn.Linear(hidden, length * n_out)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = nn.functional.avg_pool1d(h.transpose(1, 2), self.r, self.r, ceil_mode=True)
        return self.fc(self.g(z.transpose(1, 2).flatten(1))).view(-1, self.length, self.n_out)


def _haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    p = np.radians(np.stack([lat, lon], axis=1))
    dlat = p[:, None, 0] - p[None, :, 0]
    dlon = p[:, None, 1] - p[None, :, 1]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(p[:, 0])[:, None] * np.cos(p[:, 0])[None, :] * np.sin(dlon / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class GraphBranch(nn.Module):
    """Two dense message-passing layers over the NWP grid points, then a pool.

    Four nodes. `torch-geometric` for this would be a dependency for one
    `einsum`; propagation over a 4x4 dense adjacency IS one einsum.

    `theta` is initialised at the renormalised capacity weights, so rung 7
    starts as a strict superset of the fixed aggregation the rest of the
    platform uses: it begins at the incumbent and can only move off it if the
    data pays for the move.
    """

    def __init__(self, n_features: int, meta: pd.DataFrame, dim: int = 16, ablate: bool = False):
        super().__init__()
        self.dim = dim
        lat = meta["latitude"].to_numpy(dtype=float)
        lon = meta["longitude"].to_numpy(dtype=float)
        d = _haversine_km(lat, lon)
        if ablate:
            a = np.eye(len(meta))
        else:
            tau = float(np.median(d[~np.eye(len(meta), dtype=bool)])) or 1.0
            e = np.exp(-d / tau - (-d / tau).max(axis=1, keepdims=True))
            a = e / e.sum(axis=1, keepdims=True)
        self.register_buffer("A", torch.tensor(a, dtype=torch.float32))

        self.enc = nn.Linear(n_features, dim)
        self.nb = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(2))
        self.self_ = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(2))
        w = torch.tensor(np.log(meta["weight"].to_numpy(dtype=float)), dtype=torch.float32)
        self.theta = nn.Parameter(w, requires_grad=not ablate)

    def forward(self, xn: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.enc(xn))
        for nb, sf in zip(self.nb, self.self_, strict=True):
            h = torch.relu(nb(torch.einsum("ij,btjf->btif", self.A, h)) + sf(h))
        return torch.einsum("n,btnf->btf", torch.softmax(self.theta, dim=0), h)


class DeepSeq(nn.Module):
    """Rung 6 (`graph=None`) / rung 7 (`graph=GraphBranch(...)`).

    One model, three outputs: the quantiles share a trunk instead of being three
    independently fitted heads, so they cannot drift apart the way the GBDT's
    do. They can still cross, and `predict_deep` still sorts them.
    """

    def __init__(
        self,
        n_features: int,
        graph: GraphBranch | None = None,
        length: int = len(LEADS),
        hidden: int = 128,
        channels: int = 32,
    ) -> None:
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(n_features, channels), nn.ReLU())
        self.graph = graph
        c = channels + (graph.dim if graph is not None else 0)
        # ponytail: forecast-only stack; add the backcast if run count passes ~5k.
        # The doubly-residual head is ~200k params per block on its own and 900
        # training runs cannot carry 1.3M.
        self.blocks = nn.ModuleList(
            NHiTSBlock(length, c, r, hidden, len(QUANTILES)) for r in (1, 4, 12)
        )
        self.y_std = 1.0

    def forward(self, x: torch.Tensor, xn: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encode(x)
        if self.graph is not None:
            h = torch.cat([h, self.graph(xn)], dim=-1)
        return sum(b(h) for b in self.blocks)


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------------------- #
# fit / predict
# --------------------------------------------------------------------------- #
def pinball_loss(yhat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked pinball over the three quantiles.

    The mask carries both structural pads and the gold layer's `sample_weight`,
    so curtailed hours -- censored labels, not observations -- contribute
    nothing to the gradient.
    """
    a = torch.tensor(QUANTILES, dtype=yhat.dtype).view(1, 1, -1)
    d = y[:, :, None] - yhat
    pl = torch.maximum(a * d, (a - 1) * d)
    return (pl.sum(-1) * mask).sum() / mask.sum().clamp(min=1)


def _val_split(n_runs: int, run_ts: np.ndarray | None, val_frac: float, gap_runs: int):
    """Last `val_frac` of runs by issue time, with a gap.

    NOT a random run split: a run issued on day d and one on d+1 share 25 of
    their 49 valid hours, so a random split over runs still leaks on the VALID
    axis and early stopping then decides on hours it has already seen.
    """
    n_val = max(int(round(n_runs * val_frac)), 1)
    val = np.arange(n_runs - n_val, n_runs)
    tr = np.arange(max(n_runs - n_val - gap_runs, 0))
    if run_ts is not None and len(tr) and len(val):
        gap = run_ts[val[0]] - run_ts[tr[-1]]
        assert gap > np.timedelta64(gap_runs, "D"), f"validation gap collapsed to {gap}"
    return tr, val


def fit(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    xn: np.ndarray | None = None,
    *,
    meta: pd.DataFrame | None = None,
    ablate: bool = False,
    seed: int = 0,
    run_ts: np.ndarray | None = None,
    epochs: int = 300,
    patience: int = 30,
    batch: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    threads: int = 4,
) -> DeepSeq:
    """Fit one seed on `(n_runs, n_lead, F)` inputs and residual targets.

    `y` is the RAW residual `y_cf - physics_cf`; it is scaled by its own
    masked train std internally and the scale is carried on the model so
    `predict_deep` can undo it.
    """
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    tr, val = _val_split(len(x), run_ts, 0.15, 3)
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(np.nan_to_num(y), dtype=torch.float32)
    mt = torch.tensor(mask * np.isfinite(y), dtype=torch.float32)
    nt = torch.tensor(xn, dtype=torch.float32) if xn is not None else None

    obs = np.nan_to_num(y)[tr][mask[tr] > 0]
    y_std = float(obs.std()) if obs.size and obs.std() > 1e-6 else 1.0
    yt = yt / y_std

    graph = GraphBranch(xn.shape[-1], meta, ablate=ablate) if xn is not None else None
    model = DeepSeq(x.shape[-1], graph)
    model.y_std = y_std
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best, best_state, bad = float("inf"), copy.deepcopy(model.state_dict()), 0
    for _ in range(epochs):
        model.train()
        for b in np.array_split(rng.permutation(tr), max(len(tr) // batch, 1)):
            if not len(b):
                continue
            opt.zero_grad()
            out = model(xt[b], None if nt is None else nt[b])
            loss = pinball_loss(out, yt[b], mt[b])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            v = pinball_loss(
                model(xt[val], None if nt is None else nt[val]), yt[val], mt[val]
            ).item()
        if v < best - 1e-9:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def predict_sequences(model: DeepSeq, x: np.ndarray, xn: np.ndarray | None = None) -> np.ndarray:
    """Residual quantiles on the run grid, back in capacity-factor units."""
    model.eval()
    with torch.no_grad():
        out = model(
            torch.tensor(x, dtype=torch.float32),
            None if xn is None else torch.tensor(xn, dtype=torch.float32),
        )
    return out.numpy() * model.y_std


def predict_deep(
    models: list[DeepSeq],
    x: np.ndarray,
    xn: np.ndarray | None,
    pos: np.ndarray,
    physics_cf,
    capacity_mw,
    index=None,
) -> pd.DataFrame:
    """Mirror of `residual_gbdt.predict_residual`: physics + residual -> MW, sorted.

    Seeds are averaged BEFORE sorting. On ~900 runs a single seed moves nRMSE by
    tenths of a point, so one seed is not a measurement.
    """
    n_rows = len(np.asarray(physics_cf))
    resid = np.mean(
        [from_sequences(predict_sequences(m, x, xn), pos, n_rows) for m in models], axis=0
    )
    cf = np.asarray(physics_cf, dtype=float)[:, None] + resid
    cf = np.clip(np.sort(cf, axis=1), 0.0, 1.0)

    cap = np.asarray(capacity_mw, dtype=float)
    qs = sorted(QUANTILES)
    out = pd.DataFrame(
        {f"p{int(round(q * 100))}_mw": cf[:, i] * cap for i, q in enumerate(qs)},
        index=index,
    )
    for i, q in enumerate(qs):
        out[f"p{int(round(q * 100))}_cf"] = cf[:, i]
    return out
