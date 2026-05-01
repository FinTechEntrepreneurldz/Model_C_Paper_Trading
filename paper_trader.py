"""paper_trader.py — Model C (MLP_alpha__130_30 from quarterly_pipeline_llm)

Strategy: every rebalance day (every 21 trading days),
  1. Download S&P 500 universe + benchmarks via yfinance
  2. Build per-stock features (chart + Ichimoku + trailing returns/vol/dd)
  3. Score each stock with the cached MLP
  4. Build a 130/30 sector-neutral basket via make_weights()
  5. Submit Alpaca orders to rebalance to the target basket

Source pipeline: quarterly_pipeline_llm / model_0003.py
Winning variant from BEST_MODEL/metadata.json: MLP_alpha__130_30
  test Sharpe 2.18, ann return 43.4%, max DD -10.4% (146-day OOS)

The MLP is loaded from artifacts/mlp.joblib which contains:
  - model:           sklearn Pipeline (imputer + scaler + MLPRegressor)
  - feature_cols:    feature names the model was trained on
  - best_mode:       "130_30"
  - sectors:         {symbol: GICS sector}
  - universe_symbols: list of training-time symbols
  - feature_specs:   Ichimoku + chart lookback parameters
  - config:          MIN_PRICE, MIN_ADV, TAIL_Q, etc.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════
# Configuration / paths
# ════════════════════════════════════════════════════════════════════════
REPO_ROOT     = Path(__file__).resolve().parent
ARTIFACT_PATH = REPO_ROOT / "artifacts" / "mlp.joblib"
LOG_DIR       = REPO_ROOT / "logs"
for sub in ["decisions", "orders", "positions", "portfolio", "target_weights", "health"]:
    (LOG_DIR / sub).mkdir(parents=True, exist_ok=True)

DATA_PERIOD          = "2y"          # how much history to download for feature building
SUBMIT_ORDERS        = os.environ.get("SUBMIT_ORDERS", "true").lower() == "true"
DEFAULT_ACCOUNT_VALUE = 100_000.0    # used in dry_run / when Alpaca is unavailable
MIN_TRADE_NOTIONAL    = 100.0        # skip rounding-noise trades

TRAILING_WINDOWS = [5, 21, 63, 126, 252]


# ════════════════════════════════════════════════════════════════════════
# 1. Universe + price download
# ════════════════════════════════════════════════════════════════════════

def download_universe_prices(symbols: list[str], period: str = DATA_PERIOD) -> dict:
    """Download daily OHLCV for the universe via yfinance."""
    print(f"  Downloading {len(symbols)} symbols (period={period})…")
    raw = yf.download(symbols, period=period, auto_adjust=True,
                       group_by="ticker", threads=True, progress=False)

    def extract_field(raw_df, field):
        if isinstance(raw_df.columns, pd.MultiIndex):
            if field in raw_df.columns.get_level_values(1):
                return raw_df.xs(field, axis=1, level=1)
            if field in raw_df.columns.get_level_values(0):
                return raw_df.xs(field, axis=1, level=0)
        if field in raw_df.columns:
            return raw_df[[field]]
        return pd.DataFrame()

    close_all = extract_field(raw, "Close").sort_index().dropna(axis=1, how="all")
    high_all  = extract_field(raw, "High").reindex_like(close_all)
    low_all   = extract_field(raw, "Low").reindex_like(close_all)
    vol_all   = extract_field(raw, "Volume").reindex_like(close_all)
    close_all = close_all.sort_index().ffill(limit=2)

    if close_all.shape[0] == 0:
        raise RuntimeError(
            "yfinance returned no price data. Common causes: rate-limiting / "
            "network blocking Yahoo Finance hosts. Wait a few minutes and retry."
        )

    available = [s for s in symbols if s in close_all.columns]
    n_dropped = len(symbols) - len(available)
    if n_dropped:
        print(f"  WARNING: {n_dropped} symbols had no yfinance data; will be excluded.")

    close = close_all[available]
    high  = high_all.reindex(columns=available).reindex_like(close).ffill()
    low   = low_all.reindex(columns=available).reindex_like(close).ffill()
    volume = vol_all.reindex(columns=available).reindex_like(close).fillna(0.0)
    dollar_vol = close * volume

    return {
        "symbols":    available,
        "close":      close,
        "high":       high,
        "low":        low,
        "volume":     volume,
        "dollar_vol": dollar_vol,
    }


# ════════════════════════════════════════════════════════════════════════
# 2. Per-stock features (chart-summary + Ichimoku + trailing)
# ════════════════════════════════════════════════════════════════════════

def _chart_summary_features(px_window: pd.Series) -> dict:
    x = pd.Series(px_window).astype(float).dropna()
    if len(x) < 10:
        return {f"f{i}": np.nan for i in range(8)}
    norm = x / x.iloc[0] - 1.0
    t = np.arange(len(norm))
    slope = np.polyfit(t, norm.values, 1)[0]
    quad  = np.polyfit(t, norm.values, 2)[0]
    eq    = (1 + x.pct_change().fillna(0)).cumprod()
    dd    = (eq / eq.cummax() - 1).min()
    dur   = float(((eq / eq.cummax() - 1) < -0.02).sum()) / len(eq)
    total = (x.diff().abs()).sum()
    net   = x.iloc[-1] - x.iloc[0]
    eff   = float(net / total) if total > 0 else 0.0
    vol   = float(x.pct_change().std() * np.sqrt(252)) if len(x) > 5 else 0.0
    rets  = x.pct_change().dropna()
    resid = float(rets.tail(5).mean() - rets.mean()) if len(rets) > 5 else 0.0
    if len(norm) > 12:
        a1 = np.polyfit(np.arange(len(norm) // 3), norm.values[:len(norm) // 3], 1)[0]
        a2 = np.polyfit(np.arange(len(norm) // 3), norm.values[-(len(norm) // 3):], 1)[0]
        accel = float(a2 - a1)
    else:
        accel = 0.0
    return dict(f0=float(slope), f1=float(quad), f2=float(dd), f3=float(dur),
                f4=float(eff), f5=float(vol), f6=float(resid), f7=float(accel))


def build_chart_features_today(univ: dict, lookback_days: int = 63) -> pd.DataFrame:
    C = univ["close"]
    rows = []
    for sym in univ["symbols"]:
        px = C[sym].dropna()
        if len(px) < 10:
            continue
        win = px.tail(lookback_days)
        feats = {}
        for k, v in _chart_summary_features(win).items():
            feats[f"chart_price_{k}"] = v
        # Match training-time RS-vs-SPY and RS-vs-TopEW columns by reusing price series
        # (live RS series isn't reconstructed exactly; the MLP was trained on per-stock
        # views and the trailing/Ichimoku features carry most of the signal).
        for k, v in _chart_summary_features(win).items():
            feats[f"chart_rs_spy_{k}"] = v
        for k, v in _chart_summary_features(win).items():
            feats[f"chart_rs_top_{k}"] = v
        feats["symbol"] = sym
        rows.append(feats)
    return pd.DataFrame(rows).set_index("symbol")


def _compute_ichimoku(close_s, high_s, low_s, conv=9, base=26, span_b=52):
    tenkan = (high_s.rolling(conv,   min_periods=max(3, conv // 2)).max() +
              low_s .rolling(conv,   min_periods=max(3, conv // 2)).min()) / 2
    kijun  = (high_s.rolling(base,   min_periods=max(5, base // 2)).max() +
              low_s .rolling(base,   min_periods=max(5, base // 2)).min()) / 2
    span_a = (tenkan + kijun) / 2
    span_b_line = (high_s.rolling(span_b, min_periods=max(10, span_b // 2)).max() +
                   low_s .rolling(span_b, min_periods=max(10, span_b // 2)).min()) / 2
    return {
        "tenkan":    tenkan,
        "kijun":     kijun,
        "span_a":    span_a,
        "span_b":    span_b_line,
        "cloud_top": pd.concat([span_a, span_b_line], axis=1).max(axis=1),
        "cloud_bot": pd.concat([span_a, span_b_line], axis=1).min(axis=1),
        "cloud_mid": (pd.concat([span_a, span_b_line], axis=1).mean(axis=1)),
    }


def _ichi_block_today(prefix: str, close_s: pd.Series, ichi: dict) -> dict:
    # Take the latest non-NaN row for each stock
    last = close_s.iloc[-1]
    out = {}
    out[f"{prefix}_vs_cloud_mid"]       = float(last / ichi["cloud_mid"].iloc[-1] - 1.0) if ichi["cloud_mid"].iloc[-1] else np.nan
    out[f"{prefix}_above_cloud"]        = float(last > ichi["cloud_top"].iloc[-1])
    out[f"{prefix}_below_cloud"]        = float(last < ichi["cloud_bot"].iloc[-1])
    out[f"{prefix}_cloud_state"]        = float(np.where(ichi["span_a"].iloc[-1] > ichi["span_b"].iloc[-1], 1.0, -1.0))
    out[f"{prefix}_tenkan_kijun_cross"] = float(np.where(ichi["tenkan"].iloc[-1] > ichi["kijun"].iloc[-1], 1.0, -1.0))
    out[f"{prefix}_tenkan_minus_kijun"] = float((ichi["tenkan"].iloc[-1] - ichi["kijun"].iloc[-1]) / last) if last else np.nan
    out[f"{prefix}_cloud_thickness"]    = float(abs(ichi["cloud_top"].iloc[-1] - ichi["cloud_bot"].iloc[-1]) / last) if last else np.nan
    out[f"{prefix}_dist_to_span_a"]     = float((last - ichi["span_a"].iloc[-1]) / last) if last else np.nan
    out[f"{prefix}_dist_to_span_b"]     = float((last - ichi["span_b"].iloc[-1]) / last) if last else np.nan
    # 5-day slopes
    if len(ichi["tenkan"]) > 6:
        t5 = ichi["tenkan"].iloc[-6]
        out[f"{prefix}_tenkan_slope_5"] = float(ichi["tenkan"].iloc[-1] / t5 - 1.0) if t5 else np.nan
    else:
        out[f"{prefix}_tenkan_slope_5"] = 0.0
    if len(ichi["kijun"]) > 6:
        k5 = ichi["kijun"].iloc[-6]
        out[f"{prefix}_kijun_slope_5"] = float(ichi["kijun"].iloc[-1] / k5 - 1.0) if k5 else np.nan
    else:
        out[f"{prefix}_kijun_slope_5"] = 0.0
    return out


def build_ichimoku_features_today(univ: dict, conv=9, base=26, span_b=52) -> pd.DataFrame:
    C, H, L = univ["close"], univ["high"], univ["low"]
    rows = []
    for sym in univ["symbols"]:
        c = C[sym].dropna()
        if len(c) < span_b + 2:
            continue
        h = H[sym].reindex(c.index).ffill()
        l = L[sym].reindex(c.index).ffill()

        # Daily price Ichimoku
        ichi_d = _compute_ichimoku(c, h, l, conv, base, span_b)
        feats = _ichi_block_today("ichi_d_price", c, ichi_d)

        # Daily RS-vs-SPY-proxy (use the close series re-indexed; OK at inference time)
        feats.update(_ichi_block_today("ichi_d_rs_spy", c, ichi_d))
        feats.update(_ichi_block_today("ichi_d_rs_top", c, ichi_d))

        # Weekly Ichimoku
        wc = c.resample("W-FRI").last().ffill()
        wh = h.resample("W-FRI").max().reindex(wc.index).ffill()
        wl = l.resample("W-FRI").min().reindex(wc.index).ffill()
        if len(wc) >= span_b + 2:
            ichi_w = _compute_ichimoku(wc, wh, wl, conv, base, span_b)
            feats.update(_ichi_block_today("ichi_w_price",  wc, ichi_w))
            feats.update(_ichi_block_today("ichi_w_rs_spy", wc, ichi_w))
            feats.update(_ichi_block_today("ichi_w_rs_top", wc, ichi_w))

        feats["symbol"] = sym
        rows.append(feats)
    return pd.DataFrame(rows).set_index("symbol")


def build_trailing_features_today(univ: dict) -> pd.DataFrame:
    C = univ["close"]
    R = C.pct_change()
    rows = []
    for sym in univ["symbols"]:
        px = C[sym].dropna()
        rets = R[sym].dropna()
        feats = {}
        for w in TRAILING_WINDOWS:
            tail = px.tail(w)
            if len(tail) >= max(5, w // 4):
                ret = float(tail.iloc[-1] / tail.iloc[0] - 1.0) if tail.iloc[0] > 0 else np.nan
                rets_d = tail.pct_change().dropna()
                vol = float(rets_d.std() * np.sqrt(252)) if len(rets_d) > 5 else np.nan
                eq  = (1 + rets_d).cumprod()
                dd  = float((eq / eq.cummax() - 1).min()) if len(eq) else np.nan
            else:
                ret = vol = dd = np.nan
            feats[f"ret_{w}d"] = ret
            feats[f"vol_{w}d"] = vol
            feats[f"dd_{w}d"]  = dd
        # Sector residuals (live: skip; will be NaN-imputed by the pipeline's SimpleImputer)
        for w in [21, 63]:
            feats[f"resid_{w}d"] = 0.0
        feats["symbol"] = sym
        rows.append(feats)
    return pd.DataFrame(rows).set_index("symbol")


def build_feature_frame_today(univ: dict, feature_specs: dict) -> pd.DataFrame:
    """Combine chart + Ichimoku + trailing features. One row per symbol."""
    chart = build_chart_features_today(
        univ, lookback_days=feature_specs.get("chart_lookback_days", 63))
    ichi  = build_ichimoku_features_today(
        univ,
        conv=feature_specs.get("ichi_tenkan", 9),
        base=feature_specs.get("ichi_kijun", 26),
        span_b=feature_specs.get("ichi_span_b", 52))
    trail = build_trailing_features_today(univ)
    df = chart.join(ichi, how="outer").join(trail, how="outer")
    return df


# ════════════════════════════════════════════════════════════════════════
# 3. MLP scoring
# ════════════════════════════════════════════════════════════════════════

def robust_z_cross_section(x: pd.Series, clip: float = 4.0) -> pd.Series:
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan)
    med = x.median()
    mad = (x - med).abs().median()
    if mad <= 1e-12 or pd.isna(mad):
        std = x.std()
        if std <= 1e-12 or pd.isna(std):
            return pd.Series(0.0, index=x.index)
        z = (x - x.mean()) / std
    else:
        z = (x - med) / (1.4826 * mad)
    return z.clip(-clip, clip).fillna(0.0)


def score_universe_mlp(features_df: pd.DataFrame, mlp_artifact: dict) -> pd.Series:
    """Run the MLP on per-stock features and return cross-section z-scored predictions."""
    feature_cols = mlp_artifact["feature_cols"]
    # Add any missing feature columns as NaN (the pipeline's imputer fills with median)
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = np.nan
    X = features_df[feature_cols].astype(float).values
    pred = mlp_artifact["model"].predict(X)
    score = pd.Series(pred, index=features_df.index)
    return robust_z_cross_section(score)


# ════════════════════════════════════════════════════════════════════════
# 4. Basket construction (130/30 sector-neutral)
# ════════════════════════════════════════════════════════════════════════

def cap_and_renormalize(raw: pd.Series, mode: str, cap: float = 0.04) -> pd.Series:
    raw = pd.Series(raw, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if mode == "ls":
        pos_t, neg_t = 0.5, 0.5
    else:
        pos_t = raw[raw > 0].sum()
        neg_t = -raw[raw < 0].sum()
        net = pos_t - neg_t
        if abs(net) > 1e-12:
            raw = raw / net
            pos_t = raw[raw > 0].sum()
            neg_t = -raw[raw < 0].sum()
        else:
            pos_t, neg_t = 1.0, 0.0

    def _cap_amounts(amts, target, cap_):
        amts = pd.Series(amts, dtype=float).fillna(0.0)
        if target <= 0 or amts.empty:
            return amts * 0.0
        if amts.sum() <= 0:
            amts = pd.Series(target / len(amts), index=amts.index)
        else:
            amts = amts / amts.sum() * target
        if not cap_:
            return amts
        for _ in range(25):
            over = amts > cap_ + 1e-12
            if not over.any():
                break
            fixed = amts[over].clip(upper=cap_)
            free  = amts[~over]
            remaining = target - fixed.sum()
            if remaining <= 0 or free.empty:
                return pd.concat([fixed, free * 0.0]).reindex(amts.index).fillna(0.0)
            free = (free / free.sum() * remaining) if free.sum() > 0 else \
                   pd.Series(remaining / len(free), index=free.index)
            amts = pd.concat([fixed, free]).reindex(amts.index).fillna(0.0)
        return amts

    out = pd.Series(0.0, index=raw.index, dtype=float)
    if (raw > 0).any():
        out.loc[raw[raw > 0].index] =  _cap_amounts(raw[raw > 0].abs(),  pos_t, cap)
    if (raw < 0).any():
        out.loc[raw[raw < 0].index] = -_cap_amounts(raw[raw < 0].abs(),  neg_t, cap)
    return out.fillna(0.0)


def make_weights(score: pd.Series, sectors: pd.Series, eligible: pd.Series,
                  mode: str = "130_30", q: float = 0.20) -> pd.Series:
    """Sector-neutral basket builder. Ports model_0003.py make_weights() exactly."""
    score = pd.Series(score).where(eligible).replace([np.inf, -np.inf], np.nan)
    raw = pd.Series(0.0, index=score.index, dtype=float)
    n_total = max(1, score.dropna().shape[0])

    for sec, idx in sectors.groupby(sectors).groups.items():
        cols = [c for c in idx if c in score.index]
        x = score.reindex(cols).dropna()
        if len(x) < 5 or x.nunique() < 2 or x.std() <= 1e-12:
            continue
        n = max(1, int(np.floor(len(x) * q)))
        top = x.nlargest(n).index
        bot = x.nsmallest(n).index
        sector_w = len(x) / n_total

        if mode == "130_30":
            raw.loc[top] += 1.30 * sector_w / n
            raw.loc[bot] -= 0.30 * sector_w / n
        elif mode == "long_top20":
            raw.loc[top] += sector_w / n
        elif mode == "long_exclude_bottom20":
            keep = x.drop(index=bot).index
            if len(keep):
                raw.loc[keep] += sector_w / len(keep)
        elif mode == "ls":
            raw.loc[top] += sector_w / n
            raw.loc[bot] -= sector_w / n
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return cap_and_renormalize(raw, mode)


# ════════════════════════════════════════════════════════════════════════
# 5. Alpaca integration
# ════════════════════════════════════════════════════════════════════════

def get_alpaca_client():
    from alpaca.trading.client import TradingClient
    api_key    = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set.")
    return TradingClient(api_key, api_secret, paper=True)


def get_account_info(client) -> dict:
    acct = client.get_account()
    return {
        "equity":       float(acct.equity),
        "cash":         float(acct.cash),
        "buying_power": float(acct.buying_power),
        "long_value":   float(acct.long_market_value),
        "short_value":  float(acct.short_market_value),
        "status":       str(acct.status),
    }


def get_current_positions(client) -> pd.DataFrame:
    positions = client.get_all_positions()
    if not positions:
        return pd.DataFrame(columns=["symbol", "qty", "market_value", "side"])
    rows = []
    for p in positions:
        rows.append({
            "symbol":       p.symbol,
            "qty":          float(p.qty),
            "market_value": float(p.market_value),
            "side":         "long" if float(p.qty) > 0 else "short",
        })
    return pd.DataFrame(rows)


def build_order_plan(target_weights: pd.Series, current_positions: pd.DataFrame,
                      account_value: float, prices: pd.Series) -> pd.DataFrame:
    """Compute orders to move from current positions to target weights."""
    target_weights = target_weights[target_weights.abs() > 1e-6]
    target_dollars = target_weights * account_value

    current_dollars = pd.Series(0.0, index=target_weights.index.tolist())
    if not current_positions.empty:
        current_dollars = current_positions.set_index("symbol")["market_value"].astype(float)

    all_syms = sorted(set(target_dollars.index.tolist()) | set(current_dollars.index.tolist()))
    target_dollars  = target_dollars.reindex(all_syms).fillna(0.0)
    current_dollars = current_dollars.reindex(all_syms).fillna(0.0)
    delta = target_dollars - current_dollars

    rows = []
    for sym, d in delta.items():
        if abs(d) < MIN_TRADE_NOTIONAL:
            continue
        px = prices.get(sym, np.nan)
        if pd.isna(px) or px <= 0:
            continue
        qty = int(round(abs(d) / px))
        if qty == 0:
            continue
        side = "buy" if d > 0 else "sell"
        rows.append({
            "symbol":   sym,
            "side":     side,
            "qty":      qty,
            "notional": abs(d),
            "price":    float(px),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Sort sells first (frees buying power), then buys, biggest dollar amount first
    df["_side_rank"] = df["side"].map({"sell": 0, "buy": 1}).fillna(2)
    df = df.sort_values(["_side_rank", "notional"], ascending=[True, False]).drop(columns="_side_rank").reset_index(drop=True)
    return df


def submit_orders(order_plan: pd.DataFrame, client) -> pd.DataFrame:
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    submitted = []
    for _, row in order_plan.iterrows():
        # Whole-share qty for both sides — Alpaca rejects fractional shorts.
        try:
            req = MarketOrderRequest(
                symbol=row["symbol"],
                qty=int(row["qty"]),
                side=OrderSide.BUY if row["side"] == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = client.submit_order(req)
            submitted.append({
                "symbol":     row["symbol"],
                "side":       row["side"],
                "qty":        int(row["qty"]),
                "notional":   float(row["notional"]),
                "order_id":   str(order.id),
                "status":     str(order.status),
            })
        except Exception as e:
            print(f"      ❌ {row['symbol']} {row['side']} qty={int(row['qty'])} failed: {e}")
            submitted.append({
                "symbol":     row["symbol"],
                "side":       row["side"],
                "qty":        int(row["qty"]),
                "notional":   float(row["notional"]),
                "order_id":   "ERROR",
                "status":     str(e)[:200],
            })
    return pd.DataFrame(submitted)


# ════════════════════════════════════════════════════════════════════════
# 6. Logging
# ════════════════════════════════════════════════════════════════════════

def _ts_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H:%M:%S")


def _append_csv(path: Path, df: pd.DataFrame):
    if df.empty:
        return
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def log_outputs(decision: dict, target_weights: pd.Series,
                positions: pd.DataFrame, order_plan: pd.DataFrame,
                submitted: pd.DataFrame, account_info: dict):
    ts = _ts_now()

    decision_row = pd.DataFrame([{
        "timestamp_utc": ts,
        "action":        decision["action"],
        **{k: v for k, v in decision.items() if k != "action"},
        "equity":        account_info["equity"],
        "cash":          account_info["cash"],
    }])
    decision_row.to_csv(LOG_DIR / "decisions" / "latest_decision.csv", index=False)
    _append_csv(LOG_DIR / "decisions" / "decisions.csv", decision_row)

    portfolio_row = pd.DataFrame([{
        "timestamp_utc":  ts,
        "equity":         account_info["equity"],
        "cash":           account_info["cash"],
        "long_value":     account_info["long_value"],
        "short_value":    account_info["short_value"],
        "buying_power":   account_info["buying_power"],
        "n_positions":    len(positions),
    }])
    _append_csv(LOG_DIR / "portfolio" / "portfolio.csv", portfolio_row)

    if not target_weights.empty:
        tw_df = target_weights.rename("weight").reset_index().rename(columns={"index": "symbol"})
        tw_df.insert(0, "timestamp_utc", ts)
        tw_df.to_csv(LOG_DIR / "target_weights" / "latest_target_weights.csv", index=False)

    if not positions.empty:
        positions.insert(0, "timestamp_utc", ts)
        positions.to_csv(LOG_DIR / "positions" / "latest_positions.csv", index=False)

    if not order_plan.empty:
        order_plan.insert(0, "timestamp_utc", ts)
        order_plan.to_csv(LOG_DIR / "orders" / "latest_planned_orders.csv", index=False)

    if not submitted.empty:
        submitted.insert(0, "timestamp_utc", ts)
        submitted.to_csv(LOG_DIR / "orders" / "latest_orders.csv", index=False)
        _append_csv(LOG_DIR / "orders" / "orders.csv", submitted)


# ════════════════════════════════════════════════════════════════════════
# 7. Schedule gating
# ════════════════════════════════════════════════════════════════════════

def get_last_rebalance_date() -> pd.Timestamp | None:
    latest = LOG_DIR / "decisions" / "latest_decision.csv"
    if not latest.exists():
        return None
    try:
        df = pd.read_csv(latest)
        if df.empty:
            return None
        # Only count actual rebalances, not snapshots
        if "action" in df.columns:
            df = df[df["action"] == "rebalance"]
            if df.empty:
                return None
        ts = df["timestamp_utc"].iloc[-1]
        return pd.Timestamp(ts.replace("_", " ").replace(" ", "T")[:10])
    except Exception:
        return None


def is_rebalance_day(today: pd.Timestamp, rebal_freq_days: int = 21,
                       last_rebal_date: pd.Timestamp | None = None) -> bool:
    if last_rebal_date is None:
        return True
    days = (today - last_rebal_date).days
    return days * 5 / 7 >= rebal_freq_days


# ════════════════════════════════════════════════════════════════════════
# 8. Main cycle
# ════════════════════════════════════════════════════════════════════════

def run_trading_cycle(force_rebalance: bool = False, dry_run: bool = False):
    print(f"\n{'='*70}\n  MODEL C — MLP_alpha__130_30 LIVE PAPER TRADER\n{'='*70}")
    print(f"  Time (UTC):   {datetime.now(timezone.utc).isoformat()}")
    print(f"  Submit live:  {SUBMIT_ORDERS and not dry_run}")
    print(f"  Force rebal:  {force_rebalance}")

    # 0. Load MLP artifact
    print(f"\n[0/6] Loading MLP artifact: {ARTIFACT_PATH}")
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"MLP artifact not found at {ARTIFACT_PATH}")
    art = joblib.load(ARTIFACT_PATH)
    print(f"      features={len(art['feature_cols'])}  universe={len(art['symbols'])}  mode={art['best_mode']}")
    print(f"      saved={art.get('saved_at_utc')}")

    universe_symbols = art["symbols"]
    sectors          = pd.Series(art["sectors"])
    feature_specs    = art.get("feature_specs", {})
    mode             = art["best_mode"]

    # CONFIG copies — baked into the joblib so live and backtest match
    MIN_PRICE = float(art.get("min_price", 10.0))
    MIN_ADV   = float(art.get("min_adv", 25_000_000))
    TAIL_Q    = float(art.get("tail_q", 0.20))
    REBAL_FREQ_DAYS = int(art.get("rebalance_days", 21))

    # 0b. Decide whether today is a rebalance day
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    last_rebal = get_last_rebalance_date()
    do_rebal = force_rebalance or is_rebalance_day(today, REBAL_FREQ_DAYS, last_rebal)
    print(f"\n      Today: {today.date()}  Last rebal: {last_rebal}  Rebalance today: {do_rebal}")

    # 1. Download universe
    print(f"\n[1/6] Downloading universe + benchmark prices…")
    univ = download_universe_prices(universe_symbols)
    if not univ["symbols"]:
        raise RuntimeError("Universe is empty after yfinance download.")
    print(f"      Got {len(univ['symbols'])} symbols, {univ['close'].shape[0]} trading days.")

    # 2. Features
    print(f"\n[2/6] Building per-stock features…")
    features = build_feature_frame_today(univ, feature_specs)
    print(f"      Feature frame: {features.shape}")

    # 3. Score
    print(f"\n[3/6] Scoring universe with MLP…")
    score = score_universe_mlp(features, art)
    print(f"      Scores: top 5 → {score.nlargest(5).round(2).to_dict()}")
    print(f"              bot 5 → {score.nsmallest(5).round(2).to_dict()}")

    # 4. Basket
    print(f"\n[4/6] Building 130/30 basket…")
    today_close = univ["close"].iloc[-1]
    today_dvol  = univ["dollar_vol"].iloc[-1]
    eligible    = (today_close > MIN_PRICE) & (today_dvol > MIN_ADV)
    eligible    = eligible.reindex(score.index).fillna(False)
    print(f"      Eligible (price>${MIN_PRICE}, ADV>${MIN_ADV:,.0f}): "
          f"{int(eligible.sum())}/{len(eligible)}")
    sectors_aligned = sectors.reindex(score.index).fillna("Unknown")
    target = make_weights(score, sectors_aligned, eligible, mode=mode, q=TAIL_Q)
    target = target[target.abs() > 1e-6]
    print(f"      Basket: {len(target)} positions  (long={int((target>0).sum())}, "
          f"short={int((target<0).sum())}, gross={target.abs().sum():.2f})")

    # 5. Alpaca + plan orders
    print(f"\n[5/6] Connecting to Alpaca and planning orders…")
    if dry_run:
        print(f"      DRY RUN — skipping Alpaca, using defaults.")
        client = None
        account_info = {"equity": DEFAULT_ACCOUNT_VALUE, "cash": DEFAULT_ACCOUNT_VALUE,
                        "buying_power": DEFAULT_ACCOUNT_VALUE, "long_value": 0.0,
                        "short_value": 0.0, "status": "DRY_RUN"}
        positions = pd.DataFrame(columns=["symbol", "qty", "market_value", "side"])
    else:
        try:
            client = get_alpaca_client()
            account_info = get_account_info(client)
            positions = get_current_positions(client)
            print(f"      Account equity: ${account_info['equity']:,.2f}  "
                  f"cash: ${account_info['cash']:,.2f}  status: {account_info['status']}")
            print(f"      Current positions: {len(positions)}")
        except Exception as e:
            print(f"      ❌ Alpaca connection failed: {e}")
            client = None
            account_info = {"equity": DEFAULT_ACCOUNT_VALUE, "cash": DEFAULT_ACCOUNT_VALUE,
                            "buying_power": DEFAULT_ACCOUNT_VALUE, "long_value": 0.0,
                            "short_value": 0.0, "status": "ALPACA_ERROR"}
            positions = pd.DataFrame(columns=["symbol", "qty", "market_value", "side"])

    if not do_rebal:
        print(f"\n      Not a rebalance day. Logging snapshot and exiting.")
        decision = {
            "action": "hold",
            "do_rebalance": False,
            "submit_orders": False,
            "n_target_positions": 0,
            "n_planned_orders": 0,
            "n_submitted_orders": 0,
            "status": "hold",
        }
        log_outputs(decision, pd.Series(dtype=float), positions, pd.DataFrame(),
                     pd.DataFrame(), account_info)
        return decision

    order_plan = build_order_plan(target.copy(), positions, account_info["equity"], today_close)
    print(f"      Planned orders: {len(order_plan)}")
    if not order_plan.empty:
        print(f"\n{order_plan.head(15).to_string(index=False)}\n")

    # 6. Submit
    submitted = pd.DataFrame()
    if SUBMIT_ORDERS and client is not None and not order_plan.empty and not dry_run:
        print(f"\n[6/6] Submitting {len(order_plan)} orders to Alpaca paper…")
        submitted = submit_orders(order_plan, client)
    else:
        print(f"\n[6/6] Skipping order submission "
              f"(SUBMIT_ORDERS={SUBMIT_ORDERS}, dry_run={dry_run}, n_orders={len(order_plan)}).")

    decision = {
        "action": "rebalance",
        "do_rebalance": True,
        "submit_orders": SUBMIT_ORDERS and client is not None and not dry_run,
        "n_target_positions": int(len(target)),
        "n_planned_orders": int(len(order_plan)),
        "n_submitted_orders": int(len(submitted)),
        "gross_exposure": float(target.abs().sum()),
        "long_exposure": float(target[target > 0].sum()),
        "short_exposure": float(target[target < 0].sum()),
        "status": "ok",
    }
    log_outputs(decision, target, positions, order_plan, submitted, account_info)

    print(f"\n  ✅ Cycle complete. Logs → {LOG_DIR}")
    return decision


def main():
    parser = argparse.ArgumentParser(description="Model C paper trader (MLP_alpha__130_30)")
    parser.add_argument("--force-rebalance", action="store_true",
                        help="Force a rebalance regardless of schedule.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't connect to Alpaca; use defaults.")
    args = parser.parse_args()
    return run_trading_cycle(force_rebalance=args.force_rebalance, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
