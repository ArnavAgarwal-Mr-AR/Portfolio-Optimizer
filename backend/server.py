import os
import sys
import io
import logging
from typing import List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure local library is discoverable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from optimizer.config import config
from optimizer.data_pipeline import fetch_prices, fetch_risk_free_rate, align_data
from optimizer.feature_engineering import compute_returns, get_covariance_matrix, get_annualized_returns
from optimizer.evaluation import run_backtest, calculate_risk_contribution
from optimizer.report_generator import build_pdf_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("portfolio_optimizer_api")

app = FastAPI(
    title="AlphaOptima Portfolio Optimizer API",
    description="REST backend service for quantitative smart beta portfolio optimization.",
    version="1.0.0"
)

# Enable CORS for frontend Vite dev server (and vercel domains if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local/Vercel serverless accessibility
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizeRequest(BaseModel):
    tickers: Optional[List[str]] = Field(default=None, description="Asset symbols list")
    start_date: Optional[str] = Field(default=None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")
    backtest_start_date: Optional[str] = Field(default=None, description="Backtest split date (YYYY-MM-DD)")
    cov_method: Optional[str] = Field(default="shrinkage", description="sample or shrinkage covariance")
    rf_value: Optional[float] = Field(default=None, description="Manual risk free rate (decimal)")
    sim_count: Optional[int] = Field(default=3000, description="Efficient frontier simulation count")

@app.get("/api/tickers")
def get_default_tickers():
    """Returns the default asset universe tickers config."""
    return {"tickers": config.tickers}

def get_optimization_payload(req: OptimizeRequest) -> dict:
    """
    Runs the data pipeline and solver backtests, and returns the unified result payload.
    """
    # 1. Resolve parameters (use defaults if empty)
    tickers_to_use = req.tickers if req.tickers and len(req.tickers) > 0 else config.tickers
    start_dt = req.start_date if req.start_date else config.start_date
    end_dt = req.end_date if req.end_date else config.end_date
    split_dt = req.backtest_start_date if req.backtest_start_date else config.backtest_start_date
    method_cov = req.cov_method if req.cov_method else config.covariance_method
    sim_size = req.sim_count if req.sim_count else 3000
    
    logger.info(f"Optimize request received. Assets: {tickers_to_use}, Horizon: {start_dt} to {end_dt}, Split: {split_dt}")

    # 2. Fetch and align price series
    try:
        raw_prices = fetch_prices(tickers_to_use, start_dt, end_dt, use_cache=True)
        raw_rf = fetch_risk_free_rate(start_dt, end_dt, use_cache=True)
    except Exception as e:
        logger.error(f"Data pipeline fetch error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch market data: {str(e)}")
        
    prices, rf_rate = align_data(raw_prices, raw_rf)
    
    # 3. Compute daily log returns
    returns = compute_returns(prices, method="log")
    
    # 4. Split In-Sample and Out-of-Sample slices
    train_returns = returns.loc[:split_dt]
    test_returns = returns.loc[split_dt:]
    
    if train_returns.empty or test_returns.empty:
        raise HTTPException(
            status_code=400, 
            detail="Training or testing returns slice is empty. Verify that your date split is correct."
        )

    # 5. Resolve risk-free rate
    if req.rf_value is not None:
        rf_val = req.rf_value
    else:
        rf_val = rf_rate.loc[split_dt:].mean()
        if np.isnan(rf_val):
            rf_val = config.default_risk_free_rate

    # 6. Run backtest engine
    try:
        backtest_results = run_backtest(
            train_returns,
            test_returns,
            cov_method=method_cov,
            rf_rate=rf_val,
            is_log_returns=True
        )
    except Exception as e:
        logger.error(f"Solver engine failed: {e}")
        raise HTTPException(status_code=500, detail=f"Solvers failed to converge: {str(e)}")

    # 7. Format output payload for Recharts and React
    
    # Weights Object
    weights_out = {}
    for col in backtest_results["weights"].columns:
        weights_out[col] = backtest_results["weights"][col].to_dict()
        
    # Metrics Object
    metrics_out = {}
    for strategy in backtest_results["metrics"].index:
        metrics_out[strategy] = backtest_results["metrics"].loc[strategy].to_dict()
        
    # Cumulative Returns Array formatted for Recharts (e.g. [{"date": "2023-01-01", "Max Sharpe": 1.01, ...}])
    cum_returns_df = backtest_results["cumulative_returns"]
    cum_returns_out = []
    for idx, row in cum_returns_df.iterrows():
        item = {"date": idx.strftime("%Y-%m-%d")}
        for col in cum_returns_df.columns:
            item[col] = float(row[col])
        cum_returns_out.append(item)

    # Risk Contributions
    cov_train = get_covariance_matrix(train_returns, method=method_cov)
    risk_contrib_out = {}
    for strategy in backtest_results["weights"].columns:
        w_series = backtest_results["weights"][strategy]
        rc_df = calculate_risk_contribution(w_series, cov_train)
        rc_list = []
        for asset, row in rc_df.iterrows():
            rc_list.append({
                "asset": asset,
                "weight": float(row["Weight"]),
                "absolute": float(row["Absolute Contribution"]),
                "pct": float(row["Percentage Contribution"])
            })
        risk_contrib_out[strategy] = rc_list

    # Efficient Frontier simulated points
    # We generate a subset of simulated portfolios to keep network payload fast
    n_sims = min(sim_size, 1500)  # cap in-API to preserve speed
    n_assets = len(tickers_to_use)
    
    np.random.seed(42)
    sim_weights = np.random.dirichlet(np.ones(n_assets), n_sims)
    
    mu_train = get_annualized_returns(train_returns)
    sim_returns = sim_weights @ mu_train.values
    sim_vols = np.zeros(n_sims)
    for i in range(n_sims):
        sim_vols[i] = np.sqrt(sim_weights[i] @ cov_train.values @ sim_weights[i])
        
    sim_sharpe = (sim_returns - rf_val) / sim_vols
    
    simulated_frontier = []
    for i in range(n_sims):
        simulated_frontier.append({
            "vol": float(sim_vols[i]),
            "return": float(sim_returns[i]),
            "sharpe": float(sim_sharpe[i])
        })
        
    # Optimal frontier stars
    optimal_frontier = []
    for strategy in backtest_results["weights"].columns:
        w = backtest_results["weights"][strategy].values
        ret = w @ mu_train.values
        vol = np.sqrt(w @ cov_train.values @ w)
        optimal_frontier.append({
            "name": strategy,
            "vol": float(vol),
            "return": float(ret),
            "sharpe": float((ret - rf_val) / vol)
        })

    # Individual asset annualized return & volatility (from train_returns)
    individual_stats = {}
    for ticker in train_returns.columns:
        expected_ret = float(mu_train[ticker])
        volatility = float(np.sqrt(cov_train.loc[ticker, ticker]))
        individual_stats[ticker] = {
            "expected_return": expected_ret,
            "volatility": volatility,
            "sharpe": (expected_ret - rf_val) / volatility if volatility > 0 else 0.0
        }

    return {
        "tickers": list(train_returns.columns),
        "date_range": {
            "start": start_dt,
            "end": end_dt,
            "split": split_dt
        },
        "rf_rate": float(rf_val),
        "cov_method": method_cov,
        "weights": weights_out,
        "metrics": metrics_out,
        "cumulative_returns": cum_returns_out,
        "risk_contributions": risk_contrib_out,
        "individual_stats": individual_stats,
        "efficient_frontier": {
            "simulated": simulated_frontier,
            "optimal": optimal_frontier
        }
    }

@app.post("/api/optimize")
def run_optimization(req: OptimizeRequest):
    """
    Solves optimization portfolios (MVO, Min Variance, Max Sharpe, Risk Parity)
    and backtests their performance out-of-sample.
    """
    return get_optimization_payload(req)

@app.post("/api/export-pdf")
def export_pdf_report(req: OptimizeRequest):
    """
    Solves allocations and generates a ReportLab PDF compiling all results.
    Returns the PDF as a streamable file download.
    """
    try:
        results_payload = get_optimization_payload(req)
        pdf_bytes = build_pdf_report(results_payload)
        
        # Return as downloadable binary stream
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=alphaoptima_portfolio_report.pdf",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF report: {str(e)}")

# Serve production static files if they are built
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.isdir(frontend_dist):
    logger.info(f"Mounting production frontend static files from: {frontend_dist}")
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    # Allow running directly from command line
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)

