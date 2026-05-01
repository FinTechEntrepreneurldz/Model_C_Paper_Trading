# Model C — MLP_alpha 130/30 Paper Trader

Sector-neutral 130/30 long-short equity strategy using an MLPRegressor trained
on Ichimoku, chart-summary, and trailing-momentum features over the S&P 500.

**Source pipeline:** [`quarterly_pipeline_llm`](../quarterly_pipeline_llm) (model_0003.py)

## Backtest (test period, 146 trading days OOS)

| Metric        | Value      |
| ------------- | ---------- |
| Sharpe        | **2.18**   |
| Sortino       | 3.33       |
| Annual return | 43.4%      |
| Max drawdown  | -10.4%     |
| Calmar        | 4.18       |
| Hit rate      | 58.2%      |
| t-stat        | 1.66       |

⚠️ The t-stat is below the conventional 1.96 significance threshold — 146 days
is a small sample. Live performance is the only ground truth.

## How it runs

- **Daily cron at 14:00 UTC weekdays** (~30 min after market open ET)
- Loads `artifacts/mlp.joblib`, downloads the universe, builds features, scores stocks
- Builds a 130/30 sector-neutral basket: 1.30× long sleeve / 0.30× short sleeve
- Submits market orders to Alpaca paper account
- **Rebalance gating**: only rebalances every ~21 trading days (gates on `latest_decision.csv`)
- Commits log CSVs back to this repo every run

## Setup checklist

- [ ] Alpaca paper account created (separate from Model A and Model B accounts)
- [ ] `ALPACA_API_KEY` added as a repo secret
- [ ] `ALPACA_SECRET_KEY` added as a repo secret
- [ ] Test run with `force_rebalance: true` succeeds
- [ ] Dashboard `models.yaml` updated with `model_c` entry

## Manual run

```bash
# Dry run, no Alpaca
python paper_trader.py --dry-run --force-rebalance

# Force a rebalance against the live paper account (needs env vars)
ALPACA_API_KEY=PK... ALPACA_SECRET_KEY=... python paper_trader.py --force-rebalance
```

## Repo layout

```
artifacts/mlp.joblib            # sklearn pipeline + feature/sector/config bundle
paper_trader.py                 # the strategy + Alpaca integration
requirements.txt
logs/
  decisions/                    # latest_decision.csv + decisions.csv (history)
  orders/                       # latest_orders.csv + latest_planned_orders.csv
  positions/                    # latest_positions.csv
  portfolio/                    # portfolio.csv (equity curve)
  target_weights/               # latest_target_weights.csv
.github/workflows/
  daily_paper_trader.yml        # cron + manual dispatch
tests/
  test_offline_structural.py    # exercises ML + basket logic w/o yfinance
```
