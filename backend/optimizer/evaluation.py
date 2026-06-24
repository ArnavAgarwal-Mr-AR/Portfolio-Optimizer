import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def calculate_portfolio_returns(returns_df, weights):
    """
    Calculate daily portfolio returns given asset returns and weights.
    Assumes a constant-mix rebalanced portfolio (daily rebalanced to target weights).
    
    Parameters:
    - returns_df: pd.DataFrame, daily returns for each asset
    - weights: pd.Series, asset weights
    
    Returns:
    - pd.Series: daily portfolio returns
    """
    # Align columns
    w = weights.reindex(returns_df.columns).fillna(0.0)
    return returns_df @ w

def calculate_cumulative_returns(portfolio_returns, is_log_returns=True):
    """
    Calculate cumulative returns from daily returns series.
    
    Parameters:
    - portfolio_returns: pd.Series
    - is_log_returns: bool, if True, converts log returns to simple returns first
    
    Returns:
    - pd.Series: cumulative returns starting from 1.0
    """
    if is_log_returns:
        # For log returns: cum_ret_t = exp(sum(r_i))
        return np.exp(portfolio_returns.cumsum())
    else:
        # For simple returns: cum_ret_t = prod(1 + r_i)
        return (1.0 + portfolio_returns).cumprod()

def calculate_risk_contribution(weights, cov):
    """
    Calculate the absolute and percentage risk contribution of each asset.
    
    Parameters:
    - weights: pd.Series, asset weights
    - cov: pd.DataFrame, annualized covariance matrix
    
    Returns:
    - pd.DataFrame: columns=['Weight', 'Absolute Contribution', 'Percentage Contribution']
    """
    w = weights.values
    sigma = cov.values
    
    # Portfolio volatility (annualized)
    port_vol = np.sqrt(w @ sigma @ w)
    
    if port_vol == 0:
        marginal_contrib = np.zeros(len(w))
        abs_contrib = np.zeros(len(w))
        pct_contrib = np.zeros(len(w))
    else:
        # Marginal contribution to risk (d_vol / d_w)
        marginal_contrib = (sigma @ w) / port_vol
        # Absolute risk contribution of each asset (w_i * d_vol / d_w_i)
        abs_contrib = w * marginal_contrib
        # Percentage risk contribution
        pct_contrib = abs_contrib / port_vol
        
    return pd.DataFrame({
        "Weight": weights,
        "Absolute Contribution": pd.Series(abs_contrib, index=weights.index),
        "Percentage Contribution": pd.Series(pct_contrib, index=weights.index)
    })

def evaluate_performance(portfolio_returns, rf_rate=0.04, periods_per_year=252, is_log_returns=True):
    """
    Evaluate key historical performance metrics for a portfolio.
    
    Parameters:
    - portfolio_returns: pd.Series, daily portfolio returns
    - rf_rate: float, annualized risk-free rate
    - periods_per_year: int, trading periods per year (default 252)
    - is_log_returns: bool, whether returns are log returns
    
    Returns:
    - dict: dictionary of performance metrics
    """
    if portfolio_returns.empty:
        return {}
        
    # Convert daily returns to simple returns for performance calculations if they are log returns
    if is_log_returns:
        simple_returns = np.exp(portfolio_returns) - 1.0
    else:
        simple_returns = portfolio_returns
        
    # Annualized return (arithmetic mean annualized)
    ann_return = simple_returns.mean() * periods_per_year
    
    # Annualized volatility
    ann_vol = simple_returns.std() * np.sqrt(periods_per_year)
    
    # Sharpe Ratio
    excess_return = ann_return - rf_rate
    sharpe = excess_return / ann_vol if ann_vol > 0 else 0.0
    
    # Downside Volatility & Sortino Ratio
    # Downside volatility considers only negative returns
    downside_returns = simple_returns[simple_returns < 0.0]
    if downside_returns.empty:
        downside_vol = 0.0
        sortino = 0.0
    else:
        # Sortino downside deviation formula: sqrt(mean(min(r, 0)^2)) * sqrt(252)
        downside_vol = np.sqrt(np.mean(simple_returns.clip(upper=0.0) ** 2)) * np.sqrt(periods_per_year)
        sortino = excess_return / downside_vol if downside_vol > 0 else 0.0
        
    # Max Drawdown
    cum_returns = calculate_cumulative_returns(portfolio_returns, is_log_returns=is_log_returns)
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()
    
    # Value at Risk (VaR) 95% (daily)
    var_95_daily = np.percentile(simple_returns, 5)
    
    # Conditional Value at Risk (CVaR) 95% (daily)
    cvar_95_daily = simple_returns[simple_returns <= var_95_daily].mean()
    if np.isnan(cvar_95_daily):
        cvar_95_daily = var_95_daily
        
    return {
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Daily VaR (95%)": var_95_daily,
        "Daily CVaR (95%)": cvar_95_daily
    }

def run_backtest(train_returns, test_returns, cov_method="shrinkage", rf_rate=0.04, is_log_returns=True):
    """
    Run backtest for all four strategies (MVO, Min Var, Max Sharpe, Risk Parity)
    by training weights on train_returns and backtesting them on test_returns.
    
    Parameters:
    - train_returns: pd.DataFrame, daily returns for training (optimization)
    - test_returns: pd.DataFrame, daily returns for testing (backtest)
    - cov_method: str, "shrinkage" or "sample"
    - rf_rate: float, annualized risk-free rate
    - is_log_returns: bool, whether returns are log returns
    
    Returns:
    - dict: {
        'weights': DataFrame of strategy weights,
        'portfolio_returns': DataFrame of daily portfolio returns,
        'cumulative_returns': DataFrame of daily cumulative returns,
        'metrics': DataFrame of strategy performance metrics
      }
    """
    from optimizer.feature_engineering import get_annualized_returns, get_covariance_matrix
    from optimizer.optimizers import (
        mean_variance_optimize,
        min_variance_optimize,
        max_sharpe_optimize,
        risk_parity_optimize
    )
    
    # 1. Estimate parameters on training data
    mu_train = get_annualized_returns(train_returns)
    cov_train = get_covariance_matrix(train_returns, method=cov_method)
    
    # 2. Optimize portfolios
    weights = {}
    
    # Equal Weight (Benchmark)
    n = train_returns.shape[1]
    weights["Equal Weight"] = pd.Series(1.0 / n, index=train_returns.columns)
    
    # Mean-Variance Optimization
    try:
        weights["Mean-Variance"] = mean_variance_optimize(mu_train, cov_train)
    except Exception as e:
        logger.warning(f"MVO failed in backtest: {e}. Using equal weight.")
        weights["Mean-Variance"] = weights["Equal Weight"]
        
    # Minimum Variance
    try:
        weights["Min Variance"] = min_variance_optimize(cov_train)
    except Exception as e:
        logger.warning(f"Min Variance failed in backtest: {e}. Using equal weight.")
        weights["Min Variance"] = weights["Equal Weight"]
        
    # Maximum Sharpe Ratio
    try:
        weights["Max Sharpe"] = max_sharpe_optimize(mu_train, cov_train, rf_rate)
    except Exception as e:
        logger.warning(f"Max Sharpe failed in backtest: {e}. Using equal weight.")
        weights["Max Sharpe"] = weights["Equal Weight"]
        
    # Risk Parity
    try:
        weights["Risk Parity"] = risk_parity_optimize(cov_train)
    except Exception as e:
        logger.warning(f"Risk Parity failed in backtest: {e}. Using equal weight.")
        weights["Risk Parity"] = weights["Equal Weight"]
        
    weights_df = pd.DataFrame(weights)
    
    # 3. Apply weights to test data (out-of-sample)
    port_returns = {}
    cum_returns = {}
    metrics = {}
    
    for strategy, w in weights.items():
        # Daily portfolio returns
        ret_series = calculate_portfolio_returns(test_returns, w)
        port_returns[strategy] = ret_series
        
        # Cumulative returns
        cum_returns[strategy] = calculate_cumulative_returns(ret_series, is_log_returns=is_log_returns)
        
        # Performance metrics
        metrics[strategy] = evaluate_performance(
            ret_series, rf_rate=rf_rate, is_log_returns=is_log_returns
        )
        
    port_returns_df = pd.DataFrame(port_returns)
    cum_returns_df = pd.DataFrame(cum_returns)
    metrics_df = pd.DataFrame(metrics).T
    
    return {
        "weights": weights_df,
        "portfolio_returns": port_returns_df,
        "cumulative_returns": cum_returns_df,
        "metrics": metrics_df
    }
