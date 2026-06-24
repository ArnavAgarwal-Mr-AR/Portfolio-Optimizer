# Portfolio Optimizer — Input & Output Explainer Reference

This document serves as an integration reference detailing the data types, dimensions, and financial metrics utilized across the data ingestion, estimation, optimization, and backtesting layers.

---

## 1. Data Pipeline Layer (`data_pipeline.py`)

This layer downloads, caches, aligns, and cleans price datasets.

### `fetch_prices`
Downloads historical daily closing prices.

- **Inputs**:
  - `tickers` (list of str): Stock/ETF symbols to download (e.g. `["SPY", "TLT", "GLD"]`).
  - `start_date` (str): Start date as `"YYYY-MM-DD"`.
  - `end_date` (str): End date as `"YYYY-MM-DD"`.
  - `use_cache` (bool, default `True`): If True, checks for and merges with local `data/raw_prices.csv` before executing API downloads.
- **Outputs**:
  - `prices` (pd.DataFrame): Daily closing prices.
    - **Index**: DatetimeIndex (trading days).
    - **Columns**: Tickers (e.g. `["SPY", "TLT", "GLD"]`).
    - **Shape**: `(N_days, N_assets)`.

### `fetch_risk_free_rate`
Downloads interest rates from FRED or falls back to Yahoo Finance `^IRX`.

- **Inputs**:
  - `start_date` (str): Start date as `"YYYY-MM-DD"`.
  - `end_date` (str): End date as `"YYYY-MM-DD"`.
  - `use_cache` (bool, default `True`): If True, loads local `data/raw_rf_rate.csv`.
- **Outputs**:
  - `rf_rate` (pd.Series): Annualized risk-free rate represented as a decimal fraction.
    - **Index**: DatetimeIndex (daily).
    - **Value**: float (e.g. `0.045` for 4.5% annualized).

### `align_data`
Synchronizes price quotes and risk-free rates.

- **Inputs**:
  - `prices` (pd.DataFrame): Daily asset prices.
  - `rf_rate` (pd.Series): Daily risk-free interest rates.
- **Outputs**:
  - `tuple` of `(cleaned_prices, aligned_rf)`:
    - `cleaned_prices` (pd.DataFrame): Cleaned asset prices with holiday NaNs filled. Shape: `(N_trading_days, N_assets)`.
    - `aligned_rf` (pd.Series): Risk-free rates reindexed and forward-filled to match the asset trading calendar. Shape: `(N_trading_days,)`.

---

## 2. Feature Engineering Layer (`feature_engineering.py`)

Computes return metrics and covariance matrices.

### `compute_returns`
Transforms prices into returns.

- **Inputs**:
  - `prices_df` (pd.DataFrame): Cleaned daily prices. Shape: `(N_days, N_assets)`.
  - `method` (str, default `"log"`): Returns formula: `"log"` for $\ln(P_t/P_{t-1})$ or `"pct"` for $(P_t - P_{t-1})/P_{t-1}$.
- **Outputs**:
  - `returns` (pd.DataFrame): Daily asset returns. Shape: `(N_days - 1, N_assets)`.

### `get_annualized_returns`
Calculates mean annualized returns.

- **Inputs**:
  - `returns` (pd.DataFrame): Daily returns. Shape: `(N_days, N_assets)`.
  - `periods_per_year` (int, default `252`): Annualizing scaling factor.
- **Outputs**:
  - `mu` (pd.Series): Annualized expected returns. Index is asset tickers. Shape: `(N_assets,)`.

### `get_covariance_matrix`
Calculates the annualized asset return covariance matrix.

- **Inputs**:
  - `returns` (pd.DataFrame): Daily returns. Shape: `(N_days, N_assets)`.
  - `method` (str, default `"shrinkage"`): Covariance model. `"sample"` for standard sample covariance; `"shrinkage"` for **Ledoit-Wolf shrinkage** estimator.
  - `periods_per_year` (int, default `252`): Scaling factor.
- **Outputs**:
  - `covariance` (pd.DataFrame): Annualized covariance matrix. Index and columns are tickers. Shape: `(N_assets, N_assets)`.

---

## 3. Optimization Engines (`optimizers.py`)

All optimization functions enforce **long-only** constraints: $\sum w_i = 1.0$, $w_i \ge 0.0$.

### `mean_variance_optimize`
- **Inputs**:
  - `mu` (pd.Series): Expected annualized returns. Shape: `(N_assets,)`.
  - `cov` (pd.DataFrame): Covariance matrix. Shape: `(N_assets, N_assets)`.
  - `target_return` (float, optional): Constraint for minimum portfolio return.
  - `risk_aversion` (float, default `1.0`): Coefficient trade-off value $\lambda$.
- **Outputs**:
  - `weights` (pd.Series): Weights mapping to tickers. Shape: `(N_assets,)`.

### `min_variance_optimize`
- **Inputs**:
  - `cov` (pd.DataFrame): Covariance matrix. Shape: `(N_assets, N_assets)`.
- **Outputs**:
  - `weights` (pd.Series): Minimum variance weights. Shape: `(N_assets,)`.

### `max_sharpe_optimize`
- **Inputs**:
  - `mu` (pd.Series): Expected returns. Shape: `(N_assets,)`.
  - `cov` (pd.DataFrame): Covariance matrix. Shape: `(N_assets, N_assets)`.
  - `rf` (float): Annualized risk-free rate decimal.
- **Outputs**:
  - `weights` (pd.Series): Tangency portfolio weights maximizing Sharpe. Shape: `(N_assets,)`.
  - *Fallback*: If high risk-free rate results in a negative risk premium ($\max(\mu - r_f) \le 0$), returns `min_variance_optimize(cov)`.

### `risk_parity_optimize`
- **Inputs**:
  - `cov` (pd.DataFrame): Covariance matrix. Shape: `(N_assets, N_assets)`.
- **Outputs**:
  - `weights` (pd.Series): Weights that equate risk contributions. Shape: `(N_assets,)`.

---

## 4. Evaluation and Backtesting Layer (`evaluation.py`)

Computes risk metrics and backtests allocations out-of-sample.

### `calculate_portfolio_returns`
- **Inputs**:
  - `returns_df` (pd.DataFrame): Daily returns for each asset. Shape: `(N_days, N_assets)`.
  - `weights` (pd.Series): Portfolio weights. Shape: `(N_assets,)`.
- **Outputs**:
  - `portfolio_returns` (pd.Series): Daily portfolio returns series. Shape: `(N_days,)`.

### `calculate_cumulative_returns`
- **Inputs**:
  - `portfolio_returns` (pd.Series): Daily portfolio returns.
  - `is_log_returns` (bool, default `True`): If True, converts log returns to simple returns before compounding.
- **Outputs**:
  - `cum_returns` (pd.Series): Cumulative growth index starting at `1.0`.

### `calculate_risk_contribution`
- **Inputs**:
  - `weights` (pd.Series): Portfolio weights. Shape: `(N_assets,)`.
  - `cov` (pd.DataFrame): Covariance matrix. Shape: `(N_assets, N_assets)`.
- **Outputs**:
  - `risk_df` (pd.DataFrame): Columns:
    - `Weight`: Capital allocation weight.
    - `Absolute Contribution`: Annualized volatility units contributed by the asset.
    - `Percentage Contribution`: Relative risk share (sums to $1.00$).

### `evaluate_performance`
Calculates comprehensive metrics for a given portfolio return series.

- **Inputs**:
  - `portfolio_returns` (pd.Series): Daily portfolio returns.
  - `rf_rate` (float, default `0.04`): Risk-free rate baseline.
  - `periods_per_year` (int, default `252`): Annualizing scaling.
  - `is_log_returns` (bool, default `True`): Log-to-simple conversion flag.
- **Outputs**:
  - `metrics` (dict): Dictionary with keys:
    - `"Annualized Return"` (float): Geometric average return.
    - `"Annualized Volatility"` (float): Annualized standard deviation.
    - `"Sharpe Ratio"` (float): Return premium per unit of risk.
    - `"Sortino Ratio"` (float): Return premium per unit of downside risk.
    - `"Max Drawdown"` (float): Peak-to-trough drop value (negative).
    - `"Daily VaR (95%)"` (float): Value at Risk (5th percentile, daily return).
    - `"Daily CVaR (95%)"` (float): Conditional Value at Risk (expected loss beyond VaR).

### `run_backtest`
Backtests all strategies by training weights on training returns and testing out-of-sample.

- **Inputs**:
  - `train_returns` (pd.DataFrame): In-sample returns. Shape: `(N_train_days, N_assets)`.
  - `test_returns` (pd.DataFrame): Out-of-sample returns. Shape: `(N_test_days, N_assets)`.
  - `cov_method` (str, default `"shrinkage"`): Covariance method.
  - `rf_rate` (float): Backtest risk-free rate.
- **Outputs**:
  - `results` (dict): Dictionary keys:
    - `"weights"` (pd.DataFrame): Columns = strategies, Index = asset tickers. Shape: `(N_assets, N_strategies)`.
    - `"portfolio_returns"` (pd.DataFrame): Columns = strategies, Index = test dates. Daily returns. Shape: `(N_test_days, N_strategies)`.
    - `"cumulative_returns"` (pd.DataFrame): Columns = strategies, Index = test dates. Cumulative portfolio values. Shape: `(N_test_days, N_strategies)`.
    - `"metrics"` (pd.DataFrame): Columns = metrics, Index = strategies. Performance summary dataframe. Shape: `(N_strategies, N_metrics)`.
