import os
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the root folder to the system path to allow importing library packages
sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimizer.config import config
from optimizer.data_pipeline import fetch_prices, fetch_risk_free_rate, align_data
from optimizer.feature_engineering import compute_returns
from optimizer.evaluation import run_backtest, calculate_risk_contribution

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("portfolio_optimizer_cli")

def save_static_plots(results, train_returns, cov_train, rf_val):
    """
    Saves static Matplotlib/Seaborn-like charts to the reports directory.
    """
    reports_dir = config.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style for clean reports
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # 1. Backtest Cumulative Returns Line Chart
    plt.figure(figsize=(10, 6))
    for col in results["cumulative_returns"].columns:
        plt.plot(results["cumulative_returns"].index, results["cumulative_returns"][col], label=col, linewidth=2)
    plt.title("Out-of-Sample Backtest: Cumulative Returns", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Portfolio Growth (Starting at 1.0)", fontsize=12)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig(reports_dir / "backtest_cumulative_returns.png", dpi=300)
    plt.close()
    
    # 2. Portfolio Weights Bar Chart
    plt.figure(figsize=(10, 6))
    weights_df = results["weights"]
    x = np.arange(len(weights_df.index))
    width = 0.15
    
    for i, col in enumerate(weights_df.columns):
        plt.bar(x + (i - len(weights_df.columns)/2) * width + width/2, weights_df[col], width, label=col)
        
    plt.title("Asset Allocation Weights by Strategy", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(x, weights_df.index)
    plt.xlabel("Assets", fontsize=12)
    plt.ylabel("Portfolio Weight", fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig(reports_dir / "asset_allocations.png", dpi=300)
    plt.close()
    
    # 3. Risk Contribution Bar Chart
    plt.figure(figsize=(10, 6))
    # Collect risk contributions per asset
    rc_pct_dict = {}
    for col in weights_df.columns:
        rc_df = calculate_risk_contribution(weights_df[col], cov_train)
        rc_pct_dict[col] = rc_df["Percentage Contribution"]
    rc_pct_df = pd.DataFrame(rc_pct_dict)
    
    for i, col in enumerate(rc_pct_df.columns):
        plt.bar(x + (i - len(rc_pct_df.columns)/2) * width + width/2, rc_pct_df[col], width, label=col)
        
    plt.title("Risk Contribution (% of Volatility) by Asset", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(x, rc_pct_df.index)
    plt.xlabel("Assets", fontsize=12)
    plt.ylabel("Risk Contribution", fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig(reports_dir / "risk_contributions.png", dpi=300)
    plt.close()
    
    # 4. Efficient Frontier Plot with Simulated Portfolios
    plt.figure(figsize=(10, 6))
    n_sims = 2000
    n_assets = len(train_returns.columns)
    sim_weights = np.random.dirichlet(np.ones(n_assets), n_sims)
    
    mu_train = train_returns.mean() * 252
    sim_returns = sim_weights @ mu_train.values
    sim_vols = np.zeros(n_sims)
    for idx in range(n_sims):
        sim_vols[idx] = np.sqrt(sim_weights[idx] @ cov_train.values @ sim_weights[idx])
    
    sharpes = (sim_returns - rf_val) / sim_vols
    
    sc = plt.scatter(sim_vols, sim_returns, c=sharpes, cmap="viridis", s=5, alpha=0.5, label="Simulated Portfolios")
    plt.colorbar(sc, label="Sharpe Ratio")
    
    # Plot optimized points
    for col in weights_df.columns:
        w = weights_df[col].values
        ret = w @ mu_train.values
        vol = np.sqrt(w @ cov_train.values @ w)
        plt.scatter(vol, ret, marker="*", s=150, edgecolors="black", label=f"Optimal: {col}")
        
    plt.title("Efficient Frontier (In-Sample Training Data)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Annualized Volatility (Risk)", fontsize=12)
    plt.ylabel("Annualized Expected Return", fontsize=12)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", loc="best")
    plt.tight_layout()
    plt.savefig(reports_dir / "efficient_frontier.png", dpi=300)
    plt.close()
    
    logger.info(f"Saved static performance charts to {reports_dir.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Smart Beta Portfolio Optimizer CLI Runner")
    parser.add_argument("--tickers", type=str, help="Comma-separated list of tickers (e.g., SPY,TLT,GLD)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--backtest-start", type=str, help="Backtest (out-of-sample) start date (YYYY-MM-DD)")
    parser.add_argument("--cov", type=str, choices=["sample", "shrinkage"], help="Covariance estimation method")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching of prices")
    args = parser.parse_args()

    # Apply CLI overrides to config variables
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] if args.tickers else config.tickers
    start_date = args.start if args.start else config.start_date
    end_date = args.end if args.end else config.end_date
    backtest_start_date = args.backtest_start if args.backtest_start else config.backtest_start_date
    cov_method = args.cov if args.cov else config.covariance_method
    use_cache = not args.no_cache

    logger.info("Initializing Portfolio Optimization Engine...")
    logger.info(f"Assets: {tickers}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Out-of-Sample Split Date: {backtest_start_date}")
    logger.info(f"Covariance Method: {cov_method}")
    
    # 1. Fetch data
    try:
        raw_prices = fetch_prices(tickers, start_date, end_date, use_cache=use_cache)
        raw_rf = fetch_risk_free_rate(start_date, end_date, use_cache=use_cache)
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        sys.exit(1)

    # 2. Align and clean
    prices, rf_rate = align_data(raw_prices, raw_rf)
    
    # 3. Compute returns
    returns = compute_returns(prices, method="log")
    
    # 4. Split into train (in-sample) and test (out-of-sample)
    train_returns = returns.loc[:backtest_start_date]
    # Remove the split date if it double-counts, or slice cleanly
    test_returns = returns.loc[backtest_start_date:]
    
    if train_returns.empty or test_returns.empty:
        logger.error("Training or testing returns slice is empty. Verify dates.")
        sys.exit(1)
        
    # Get mean risk-free rate during backtest period to use as a baseline metric
    rf_backtest_val = rf_rate.loc[backtest_start_date:].mean()
    if np.isnan(rf_backtest_val):
        rf_backtest_val = config.default_risk_free_rate
        
    logger.info(f"Average risk-free rate in backtest period: {rf_backtest_val:.2%}")

    # 5. Run backtest
    logger.info("Running optimization on train set and backtesting out-of-sample...")
    results = run_backtest(
        train_returns,
        test_returns,
        cov_method=cov_method,
        rf_rate=rf_backtest_val,
        is_log_returns=True
    )
    
    # 6. Print results exactly as required
    print("\n" + "=" * 32)
    print("=== Optimized Portfolio Weights ===")
    print("=" * 32)
    # Format weights to 2 decimal places
    weights_fmt = results["weights"].round(2)
    print(weights_fmt.to_string())
    print("\n" + "=" * 32)
    print("=== Performance Summary ===")
    print("=" * 32)
    
    # Create structured terminal output
    metrics = results["metrics"].copy()
    metrics_display = pd.DataFrame(index=metrics.index)
    metrics_display["Ann.Return"] = metrics["Annualized Return"].apply(lambda val: f"{val:.1%}")
    metrics_display["Ann.Vol"] = metrics["Annualized Volatility"].apply(lambda val: f"{val:.1%}")
    metrics_display["Sharpe"] = metrics["Sharpe Ratio"].apply(lambda val: f"{val:.2f}")
    metrics_display["Max Drawdown"] = metrics["Max Drawdown"].apply(lambda val: f"{val:.1%}")
    
    # Adjust column header to match the prompt's layout exactly
    print(metrics_display.to_string())
    print("=" * 32)
    
    # 7. Export outputs to disk
    reports_dir = config.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    results["weights"].to_csv(reports_dir / "portfolio_weights.csv")
    results["metrics"].to_csv(reports_dir / "performance_metrics.csv")
    results["cumulative_returns"].to_csv(reports_dir / "cumulative_returns.csv")
    
    # Generate charts
    # Calculate cov_train for efficient frontier plotting
    from optimizer.feature_engineering import get_covariance_matrix
    cov_train = get_covariance_matrix(train_returns, method=cov_method)
    
    try:
        save_static_plots(results, train_returns, cov_train, rf_backtest_val)
    except Exception as e:
        logger.warning(f"Could not generate or save static plots: {e}")
        
    logger.info("Portfolio optimization task completed successfully.")

if __name__ == "__main__":
    main()
