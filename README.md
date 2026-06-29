# APS1052 Project — FLEX Daily Log-Return Forecasting & Quantile Trading

**Team:** Yicheng Yao, Jarvis Wang, JiangChuan Yu

A machine-learning / deep-learning pipeline that predicts the **next-day log return of
FLEX (Flex Ltd., mid-cap, Information Technology)** and turns the predictions into a
quantile-based trading strategy, evaluated with proper anti–data-snooping tests.

## What the notebook does

`FLEX_LogReturn_Trading.ipynb` runs end-to-end:

1. **Data acquisition** — downloads the target and all inputs (Yahoo Finance + FRED) and
   caches every series to `data/*.csv` (so the project is reproducible offline).
2. **Feature engineering** — builds 18 indicators; **12 of 18 (67%) are NOT from the
   target's own OHLCV bar** (sector ETF, SPY, VIX, MOVE, yield-curve & credit spreads,
   risk-appetite ratio, calendar/holiday, De Prado fractional differencing).
   The **custom indicator** (programmed by us) is *relative strength vs. sector*.
3. **Scaling (hybrid)** — rolling scaling for price/return/macro series, independent-row
   scaling for Ta-Lib features, encoding for calendar features.
4. **Split** — chronological ~70/15/15 train/validation/test, `TimeSeriesSplit` inside.
5. **Feature selection** — `SelectKBest` (mutual information); **discards the worst 4**.
6. **Model selection** — manual grid-search CV loop over **5 models**
   (Ridge, Random Forest, LightGBM, SVR, Keras MLP); ≥1 requires scaling (SVR & MLP).
7. **Fine-tuning / regularization** of the best model.
8. **Evaluation** — MAE, Profit Factor, Spearman RHO, directional accuracy by quantile
   (validation + test), plus **SHAP** feature importance on the test set.
9. **Trading** — long the top predicted-move quantile (Q5), short the bottom (Q1),
   flat otherwise; **test equity curve vs. buy-and-hold**.
10. **Diagnostics** — White reality check p-value, Monte Carlo permutation p-value
    (Profit Factor metric), Sharpe ratio, Profit Factor, CAGR.

## How to run

### Option A — course environment (recommended)
```bash
conda activate py3.11_AI
pip install lightgbm shap      # the only two packages the course env lacks
jupyter notebook FLEX_LogReturn_Trading.ipynb
```

### Option B — fresh environment
```bash
conda env create -f environment.yml
conda activate flex_aps1052
jupyter notebook FLEX_LogReturn_Trading.ipynb
```

Then run the cells top to bottom. First run downloads data to `data/`; later runs reuse it.

## Data sources

| Series | Symbol / ID | Source |
| --- | --- | --- |
| Target: FLEX | `FLEX` | Yahoo Finance |
| Sector ETF (Info Tech) | `XLK` | Yahoo Finance |
| Broad market | `SPY` | Yahoo Finance |
| Equity volatility | `^VIX` | Yahoo Finance |
| Bond volatility | `^MOVE` | Yahoo Finance |
| Risk-appetite ratio | `HYG`, `IEF` | Yahoo Finance |
| Yield-curve spread | `T10Y2Y` | FRED |
| High-yield credit spread | `BAMLH0A0HYM2` | FRED |

## Notes

- **Target:** next-day **log return** (regression). Equity curve uses log-return math
  (`cumsum` → `exp`); converts to simple return `exp(r) − 1` for P&L / Profit Factor.
- **No PyTorch** is used (course rule). The neural net is a Keras MLP (TensorFlow backend).
- `lightgbm` and `shap` fall back gracefully (sklearn GradientBoosting / permutation
  importance) if not installed, so the notebook still runs.
- See `../project_decisions.txt` for the full list of design decisions.
