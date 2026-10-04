"""Baseline models for the harness: naive0, naive_drift, v0 (the current engine).

Interface (see research/harness.py): predict(ctx, ticker, positions, horizons) -> {h: (Q, p_up)}
with Q (n, 19) end-of-horizon log-return quantiles at QS19 = 0.05..0.95 and p_up (n,).

SortedPool + batch_dist2 are reusable by analog-search lanes: they reproduce the
engine's float32 distances bit-for-bit, several as-of dates at a time.
"""
import math
import threading

import numpy as np

import engine
from research.harness import QS19, Z19, norm_cdf


class Naive0:
    """Zero-drift random walk: r ~ N(0, (sig63 * sqrt h)^2), sig63 = trailing 63-day
    std of daily log returns at the as-of bar (engine's td.sig, bars <= p only)."""
    name = "naive0"

    def predict(self, ctx, t, positions, horizons):
        td = ctx.td(t)
        s = td.sig[positions]
        n = len(positions)
        return {h: (Z19[None, :] * (s * math.sqrt(h))[:, None], np.full(n, 0.5)) for h in horizons}


class NaiveDrift:
    """naive0 plus the stock's own walk-forward mean daily log return x h, where the
    mean is over every bar up to the as-of bar: (logc[p] - logc[0]) / p."""
    name = "naive_drift"

    def predict(self, ctx, t, positions, horizons):
        td = ctx.td(t)
        p = np.asarray(positions)
        s = td.sig[p]
        mu = (td.logc[p] - td.logc[0]) / np.maximum(p, 1)
        out = {}
        for h in horizons:
            sh = s * math.sqrt(h)
            out[h] = (mu[:, None] * h + Z19[None, :] * sh[:, None], norm_cdf(mu * h / sh))
        return out


class SortedPool:
    """An engine.Pool with its rows permuted by their 5-bar end date, exposing the
    attributes Engine._select / Engine._paths read. Rows that can be eligible at as-of
    day D (end date <= D for every horizon >= 5) are then the prefix [0, prefix(D)), so
    distances need only be computed there. Selection results do not depend on row
    order (Engine._select sorts candidates by (distance, unique key))."""

    def __init__(self, pool):
        perm = np.argsort(pool.end[5], kind="stable")
        self.perm = perm
        self.XT = np.ascontiguousarray(pool.XT[:, perm])
        for a in ("gpos", "tid", "local", "day", "sig", "block", "key"):
            setattr(self, a, getattr(pool, a)[perm])
        self.end = {h: v[perm] for h, v in pool.end.items()}
        self.logc = pool.logc            # indexed by gpos: unchanged
        self.tickers = pool.tickers
        self.n = pool.n

    def prefix(self, D):
        return int(np.searchsorted(self.end[5], D, side="right"))


def batch_dist2(sp, Qb, M, R=16384):
    """Squared distances (len(Qb), M) from setups Qb to rows [0, M) of a SortedPool.
    Same float32 operations in the same feature order as engine.Pool.dist2, so every
    value is bit-identical to the engine's; chunked over rows for cache efficiency."""
    Qb = np.asarray(Qb, dtype=np.float32)
    B, F = Qb.shape
    D2 = np.zeros((B, M), np.float32)
    tmp = np.empty((B, R), np.float32)
    for c0 in range(0, M, R):
        c1 = min(M, c0 + R)
        t = tmp[:, : c1 - c0]
        out = D2[:, c0:c1]
        for f in range(F):
            np.subtract(sp.XT[f, c0:c1][None, :], Qb[:, f:f + 1], out=t)
            np.multiply(t, t, out=t)
            np.add(out, t, out=out)
    return D2


class V0:
    """The current engine (commit 4d9b581) via its own internals, unchanged: the time
    rule (analog dated A usable only if A + h <= as-of), K=250 de-clustered analogs,
    Gaussian distance weights (Engine._select with the PRE shortcut, as in
    Engine.walk_forward), vol-scaled analog end returns (Engine._paths). Distances come
    from batch_dist2 on a SortedPool (bit-identical values, faster). The 19 quantiles
    are the engine's weighted quantile (engine.wquantile) of the weighted scaled end
    returns, so q10/q25/q50/q75/q90 equal the engine's p10..p90 exactly (no
    interpolation). p_up = weighted share of scaled end returns > 0 (engine's odds.p_up).
    Abstains (NaN row) when fewer than K_MIN analogs are eligible, like the engine.
    use_fast=False runs Engine.walk_forward's own code path (for the equality proof)."""
    name = "v0"
    use_fast = True
    batch = 16

    def __init__(self):
        self._sorted = {}
        self._lock = threading.Lock()

    def sorted_pool(self, eng, pool):
        """The shared basket pool's sorted copy is cached; a per-target pool is not
        (predict is called once per ticker, so there is nothing to reuse)."""
        if eng.pools.get("__basket__") is not pool:
            return SortedPool(pool)
        with self._lock:
            sp = self._sorted.get("basket")
            if sp is None or sp[0] is not pool:
                sp = (pool, SortedPool(pool))
                self._sorted["basket"] = sp
            return sp[1]

    def _emit(self, out, i, eng, pool, sel, w, h, sig_now):
        if len(sel) < engine.K_MIN:
            return
        _, scaled, _ = eng._paths(pool, sel, h, sig_now, end_only=True)
        e = scaled[:, 0]
        out[h][0][i] = engine.wquantile(e, w, QS19)
        out[h][1][i] = float(np.sum(w * (e > 0)) / np.sum(w))

    @staticmethod
    def _pre(d2, M, target=8000, stride=16):
        """A cheap stand-in for the engine's PRE shortcut: ALL prefix rows within a
        distance threshold (a ball, so it is a top-|pre| set; ties included). The
        threshold is the (target/stride)-th smallest of every stride-th row. Engine._select
        uses it only if it holds >= K*CANDIDATE_MULT eligible rows (then the nearest
        eligible rows are provably inside it); otherwise it does its own full scan.
        Rows >= M are never eligible at this as-of date, so they are never needed."""
        if M <= 4 * target:
            return None
        sub = d2[:M:stride]
        k = target // stride
        tau = np.partition(sub, k)[k]
        return np.flatnonzero(d2[:M] <= tau)

    def predict(self, ctx, t, positions, horizons):
        eng = ctx.eng
        td = ctx.td(t)
        pool = eng.pool_for(t)
        positions = np.asarray(positions, dtype=np.int64)
        n = len(positions)
        out = {h: (np.full((n, 19), np.nan), np.full(n, np.nan)) for h in horizons}
        if not self.use_fast:
            for i, p in enumerate(positions):          # Engine.walk_forward, verbatim
                q = td.X[p]
                d2 = pool.dist2(q)
                pre = np.argpartition(d2, engine.PRE - 1)[:engine.PRE] if pool.n > engine.PRE else None
                for h in horizons:
                    sel, _, w = eng._select(pool, q, d2, td.day[p], h, pre=pre)
                    self._emit(out, i, eng, pool, sel, w, h, td.sig[p])
            return out
        sp = self.sorted_pool(eng, pool)
        for b0 in range(0, n, self.batch):
            bp = positions[b0: b0 + self.batch]
            Ms = [sp.prefix(td.day[p]) for p in bp]
            D2 = batch_dist2(sp, td.X[bp], max(max(Ms), 1))
            for j, p in enumerate(bp):
                i, M, d2 = b0 + j, Ms[j], D2[j]
                pre = self._pre(d2, M)
                for h in horizons:
                    sel, _, w = eng._select(sp, td.X[p], d2, td.day[p], h, pre=pre)
                    self._emit(out, i, eng, sp, sel, w, h, td.sig[p])
        return out


MODELS = {m.name: m for m in (Naive0(), NaiveDrift(), V0())}
