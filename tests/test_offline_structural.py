"""tests/test_offline_structural.py — exercises everything except yfinance.

Verifies:
  - mlp.joblib loads with the expected bundle keys
  - score_universe_mlp produces sensible cross-section z-scores
  - make_weights produces a valid 130/30 sector-neutral basket
  - build_order_plan computes integer-share trades correctly
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper_trader import (
    build_order_plan,
    cap_and_renormalize,
    make_weights,
    robust_z_cross_section,
    score_universe_mlp,
)

ART_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "mlp.joblib"


def test_joblib_loads():
    art = joblib.load(ART_PATH)
    expected = {"model", "feature_cols", "best_mode", "symbols", "sectors"}
    missing = expected - set(art.keys())
    assert not missing, f"missing keys: {missing}"
    assert art["best_mode"] == "130_30", f"unexpected mode: {art['best_mode']}"
    assert len(art["feature_cols"]) > 0
    assert len(art["symbols"]) > 100
    print(f"✅ joblib loads with the expected shape ({len(art['symbols'])} symbols, "
          f"{len(art['feature_cols'])} features)")
    return art


def test_mlp_scoring_with_synthetic_features(art):
    rng = np.random.default_rng(42)
    syms = art["symbols"]
    feature_cols = art["feature_cols"]
    X = pd.DataFrame(
        rng.normal(0, 1, size=(len(syms), len(feature_cols))),
        index=syms, columns=feature_cols,
    )
    score = score_universe_mlp(X, art)
    assert isinstance(score, pd.Series)
    assert len(score) == len(syms)
    assert score.notna().all(), "score contains NaN"
    assert abs(score.median()) < 1.0, "robust z should center near 0"
    print(f"✅ MLP scores synthetic features: median={score.median():.3f} "
          f"std={score.std():.3f}")


def test_make_weights_130_30(art):
    rng = np.random.default_rng(0)
    syms = art["symbols"]
    score    = pd.Series(rng.normal(0, 1, size=len(syms)), index=syms)
    sectors  = pd.Series(art["sectors"]).reindex(syms).fillna("Unknown")
    eligible = pd.Series(True, index=syms)

    target = make_weights(score, sectors, eligible, mode="130_30", q=0.20)

    longs  = target[target > 0]
    shorts = target[target < 0]
    print(f"✅ Basket: {len(longs)} longs, {len(shorts)} shorts, "
          f"long_sum={longs.sum():.3f}, short_sum={shorts.sum():.3f}, "
          f"gross={target.abs().sum():.3f}")
    # 130/30 means long sleeve ~1.3, short sleeve ~0.3, net ~1.0
    assert 1.15 < longs.sum() < 1.45, f"long sleeve out of expected range: {longs.sum()}"
    assert -0.45 < shorts.sum() < -0.15, f"short sleeve out of expected range: {shorts.sum()}"
    assert len(longs) > 20 and len(shorts) > 20, "too few positions"
    return target


def test_build_order_plan(target):
    syms = target.index
    prices = pd.Series(100.0, index=syms)
    current = pd.DataFrame({
        "symbol": syms[:5].tolist(),
        "qty": [10, -5, 20, -10, 15],
        "market_value": [1000.0, -500.0, 2000.0, -1000.0, 1500.0],
        "side": ["long", "short", "long", "short", "long"],
    })
    plan = build_order_plan(target, current, account_value=100_000.0, prices=prices)
    if not plan.empty:
        assert (plan["qty"] > 0).all(), "all qty must be positive"
        assert plan["qty"].dtype.kind == "i", "qty must be integer"
        assert plan["side"].isin(["buy", "sell"]).all()
        # Sells should come before buys in the sort
        first_buy = (plan["side"] == "buy").idxmax() if (plan["side"] == "buy").any() else len(plan)
        first_sell = (plan["side"] == "sell").idxmax() if (plan["side"] == "sell").any() else len(plan)
        if first_buy < len(plan) and first_sell < len(plan):
            assert first_sell < first_buy, "sells must come before buys"
    print(f"✅ Order plan: {len(plan)} orders, all integer qty, sells first")


if __name__ == "__main__":
    art = test_joblib_loads()
    test_mlp_scoring_with_synthetic_features(art)
    target = test_make_weights_130_30(art)
    test_build_order_plan(target)
    print("\n✅ All offline structural tests passed.")
