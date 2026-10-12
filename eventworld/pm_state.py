"""Polymarket block-strict state, action and target computation on the per-market order store.

Conventions:
  * orders of a market are sorted by (block, log_index); `idx` is the row position.
  * history at an order k is H = orders[:n_hist] with n_hist = number of orders in blocks < block_k  (block-strict);
    for a clock time t with no anchor order, n_hist = number of orders with ts < t.
  * state phi(H): p_buy / p_sell = price of the last taker buy / sell in H; mid = mean of the available ones;
    spread = p_buy - p_sell; D = volume in [t-24h, t) within H; N/F/V = count / signed flow / volume in [t-1h, t) within H;
    age = days since first trade; ttl = days to end_date (None if missing / already passed / > 5 years);
    actor state u^w = (position in YES tokens, prior order count, signed usd flow) of wallet w within H.
  * window for horizon delta: orders[n_hist:n_end] with n_end = number of orders with ts < t + delta.
    own = orders of the actor wallet, others = the rest (same-block orders of others are part of the window, not the history).
"""
import os
import numpy as np
import polars as pl

ROOT = "data/processed/polymarket"
STORE = f"{ROOT}/orders_by_market"
DELTAS = {"5m": 300, "1h": 3600, "24h": 86400}
DAY = 86400
HOUR = 3600
STATE_COLS = ["p_buy", "p_sell", "mid", "spread", "D", "N1h", "F1h", "V1h", "age_d", "ttl_d", "since_last_s"]
ACTOR_COLS = ["w_pos", "w_n_prev", "w_flow", "w_since_last_s"]


def _ffill(a):
    idx = np.where(~np.isnan(a), np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    return a[idx]


def _mid(pb, ps):
    both = ~np.isnan(pb) & ~np.isnan(ps)
    out = np.where(both, (pb + ps) / 2, np.where(np.isnan(pb), ps, pb))
    return out


class MarketOrders:
    """All block-level orders of one market plus prefix structures for O(log n) state queries."""

    def __init__(self, market_id, df=None, meta=None):
        self.market_id = str(market_id)
        if df is None:
            df = pl.read_parquet(f"{STORE}/market_id={market_id}/*.parquet")
        self.df = df
        self.n = df.height
        self.block = df["block"].to_numpy().astype(np.int64)
        self.ts = df["ts"].to_numpy().astype(np.int64)
        self.log_index = df["log_index"].to_numpy().astype(np.int64)
        self.wallet = df["wallet"].to_numpy().astype(np.int64)
        self.sign = df["sign"].to_numpy().astype(np.int64)
        self.usd = df["usd"].to_numpy().astype(np.float64)
        self.tokens = df["tokens"].to_numpy().astype(np.float64)
        self.p = df["p"].to_numpy().astype(np.float64)
        assert (np.diff(self.block) >= 0).all() and (np.diff(self.ts) >= 0).all(), "store not sorted"
        # state after order i; sub-cent dust orders (usd == 0 -> p is exactly 0 or 1) and token-less orders (p nan)
        # carry no price information: they are excluded from the last-buy / last-sell price state but still counted in N.
        self.p_ok = (self.usd > 0) & ~np.isnan(self.p)
        self.lb = _ffill(np.where((self.sign > 0) & self.p_ok, self.p, np.nan))
        self.ls = _ffill(np.where((self.sign < 0) & self.p_ok, self.p, np.nan))
        self.cum_usd = np.concatenate([[0.0], np.cumsum(self.usd)])
        self.cum_flow = np.concatenate([[0.0], np.cumsum(self.sign * self.usd)])
        # number of orders in blocks strictly before block_i
        self.block_start = np.searchsorted(self.block, self.block, side="left")
        # wallet segments: orders sorted by (wallet, idx)
        order = np.lexsort((np.arange(self.n), self.wallet))
        self.w_order = order
        ws = self.wallet[order]
        uniq, starts = np.unique(ws, return_index=True)
        ends = np.append(starts[1:], self.n)
        self.w_seg = dict(zip(uniq.tolist(), zip(starts.tolist(), ends.tolist())))
        self.w_pos_pref = np.concatenate([[0.0], np.cumsum((self.sign * self.tokens)[order])])
        self.w_flow_pref = np.concatenate([[0.0], np.cumsum((self.sign * self.usd)[order])])
        self.w_usd_pref = np.concatenate([[0.0], np.cumsum(self.usd[order])])
        self.t_first = int(self.ts[0]) if self.n else None
        self.t_last = int(self.ts[-1]) if self.n else None
        self.meta = meta or {}
        ed = self.meta.get("end_ts")
        self.end_ts = int(ed) if ed is not None and not (isinstance(ed, float) and np.isnan(ed)) else None

    # ---- history cuts -------------------------------------------------------------------------
    def n_hist_at_time(self, t):
        return np.searchsorted(self.ts, np.asarray(t), side="left")

    def n_hist_at_order(self, i):
        return self.block_start[np.asarray(i)]

    # ---- wallet helpers -----------------------------------------------------------------------
    def wallet_rows(self, w, lo, hi):
        """idx of wallet w's orders with lo <= idx < hi (ascending)."""
        seg = self.w_seg.get(int(w))
        if seg is None:
            return np.empty(0, dtype=np.int64)
        s, e = seg
        rows = self.w_order[s:e]
        a = np.searchsorted(rows, lo, side="left")
        b = np.searchsorted(rows, hi, side="left")
        return rows[a:b]

    def wallet_prefix(self, w, n_hist):
        """(position, n_prev, flow, last_ts) of wallet w over orders idx < n_hist."""
        seg = self.w_seg.get(int(w))
        if seg is None:
            return 0.0, 0, 0.0, None
        s, e = seg
        rows = self.w_order[s:e]
        r = int(np.searchsorted(rows, n_hist, side="left"))
        pos = self.w_pos_pref[s + r] - self.w_pos_pref[s]
        flow = self.w_flow_pref[s + r] - self.w_flow_pref[s]
        last = int(self.ts[rows[r - 1]]) if r > 0 else None
        return float(pos), r, float(flow), last

    # ---- state ---------------------------------------------------------------------------------
    def state(self, t, n_hist, w=None):
        """Summary state phi(H) at clock time t with history = orders[:n_hist]; w = actor wallet id or None."""
        t = int(t); n_hist = int(n_hist)
        if n_hist > 0:
            pb, ps = self.lb[n_hist - 1], self.ls[n_hist - 1]
            since = t - int(self.ts[n_hist - 1])
        else:
            pb, ps, since = np.nan, np.nan, None
        mid = float(_mid(np.array([pb]), np.array([ps]))[0])
        a24 = min(int(np.searchsorted(self.ts, t - DAY, side="left")), n_hist)
        a1 = min(int(np.searchsorted(self.ts, t - HOUR, side="left")), n_hist)
        st = {"t": t, "n_hist": n_hist,
              "p_buy": None if np.isnan(pb) else float(pb), "p_sell": None if np.isnan(ps) else float(ps),
              "mid": None if np.isnan(mid) else mid,
              "spread": None if (np.isnan(pb) or np.isnan(ps)) else float(pb - ps),
              "D": float(self.cum_usd[n_hist] - self.cum_usd[a24]),
              "N1h": int(n_hist - a1), "F1h": float(self.cum_flow[n_hist] - self.cum_flow[a1]),
              "V1h": float(self.cum_usd[n_hist] - self.cum_usd[a1]),
              "age_d": (t - self.t_first) / DAY if self.t_first is not None else None,
              "ttl_d": self.ttl_days(t), "since_last_s": since}
        if w is not None and w > 0:
            pos, nprev, flow, last = self.wallet_prefix(w, n_hist)
            st.update({"w_pos": pos, "w_n_prev": nprev, "w_flow": flow, "w_since_last_s": None if last is None else t - last})
        else:
            st.update({"w_pos": None, "w_n_prev": None, "w_flow": None, "w_since_last_s": None})
        return st

    def ttl_days(self, t):
        if self.end_ts is None:
            return None
        d = (self.end_ts - t) / DAY
        if d < 0 or d > 5 * 365:
            return None
        return d

    # ---- windows / targets ---------------------------------------------------------------------
    def window(self, t, n_hist, delta, w=None):
        """Orders in the action window [block of t, t+delta): own list, others' aggregates, end state."""
        t = int(t); n_hist = int(n_hist)
        n_end = int(np.searchsorted(self.ts, t + delta, side="left"))
        n_end = max(n_end, n_hist)
        N = n_end - n_hist
        V = float(self.cum_usd[n_end] - self.cum_usd[n_hist])
        F = float(self.cum_flow[n_end] - self.cum_flow[n_hist])
        own_rows = self.wallet_rows(w, n_hist, n_end) if (w is not None and w > 0) else np.empty(0, dtype=np.int64)
        own = [(int(self.ts[i] - t), int(self.sign[i]), float(self.usd[i]), float(self.p[i]), float(self.tokens[i])) for i in own_rows]
        own_V = float(self.usd[own_rows].sum()); own_F = float((self.sign[own_rows] * self.usd[own_rows]).sum())
        own_pos = float((self.sign[own_rows] * self.tokens[own_rows]).sum())
        end = self.state(t + delta, n_end, w)
        return {"n_end": n_end, "N_all": N, "F_all": F, "V_all": V,
                "own": own, "own_N": len(own), "own_F": own_F, "own_V": own_V, "own_dpos": own_pos,
                "N_o": N - len(own), "F_o": F - own_F, "V_o": V - own_V, "end": end}

    def recent(self, n_hist, k):
        """Last k history orders (oldest first) as dicts."""
        lo = max(0, n_hist - k)
        return [{"idx": int(i), "ts": int(self.ts[i]), "wallet": int(self.wallet[i]), "sign": int(self.sign[i]),
                 "usd": float(self.usd[i]), "p": float(self.p[i])} for i in range(lo, n_hist)]

    def orders_in(self, lo, hi, k=None):
        rows = range(lo, hi) if k is None or hi - lo <= k else range(lo, lo + k)
        return [{"idx": int(i), "ts": int(self.ts[i]), "wallet": int(self.wallet[i]), "sign": int(self.sign[i]),
                 "usd": float(self.usd[i]), "p": float(self.p[i])} for i in rows]


_MARKETS = None


def market_meta(market_id=None):
    """Row(s) of markets.parquet as dicts; end_ts in epoch seconds."""
    global _MARKETS
    if _MARKETS is None:
        m = pl.read_parquet(f"{ROOT}/markets.parquet")
        m = m.with_columns(pl.col("end_date").dt.epoch("s").alias("end_ts"))
        _MARKETS = {r["market_id"]: r for r in m.to_dicts()}
    return _MARKETS if market_id is None else _MARKETS.get(str(market_id))


def load_market(market_id):
    return MarketOrders(market_id, meta=market_meta(market_id))


def delta_bin_edges():
    """21-bin signed grid for Delta-mid scoring (bin 10 = exactly zero)."""
    pos = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25])
    return pos


def delta_bin(x):
    """Map Delta-mid to bin index 0..20 (10 = no change, >10 up, <10 down)."""
    pos = delta_bin_edges()
    x = np.asarray(x, dtype=float)
    mag = np.searchsorted(pos, np.abs(x), side="left") + 1  # 1..10 for |x|>0
    b = np.where(x == 0, 10, np.where(x > 0, 10 + mag, 10 - mag))
    return b


def hour_bars(m, t, n_hist, L=24):
    """Hourly bars of the block-strict history ending at t: L bars [t-(k+1)h, t-kh), oldest first,
    each (close_mid, volume, net_flow, n); close = mid after the last history order before the bar end (carried forward)."""
    t = int(t); n_hist = int(n_hist)
    edges = t - HOUR * np.arange(L, -1, -1)  # L+1 edges ascending
    cuts = np.minimum(np.searchsorted(m.ts, edges, side="left"), n_hist)
    out = []
    for k in range(L):
        a, b = int(cuts[k]), int(cuts[k + 1])
        if b > 0:
            pb, ps = m.lb[b - 1], m.ls[b - 1]
            close = float(_mid(np.array([pb]), np.array([ps]))[0])
            if np.isnan(close):
                close = None
        else:
            close = None
        out.append([close, float(m.cum_usd[b] - m.cum_usd[a]), float(m.cum_flow[b] - m.cum_flow[a]), b - a])
    return out


MarketOrders.hour_bars = hour_bars
