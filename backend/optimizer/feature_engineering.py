import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def _ledoit_wolf_shrinkage(X, assume_centered=False):
    """Estimate the asymptotically optimal Ledoit-Wolf shrinkage intensity."""
    n_samples, n_features = X.shape
    if n_features == 1:
        return 0.0

    if not assume_centered:
        X = X - X.mean(axis=0)

    X2 = X**2
    emp_cov_trace = np.sum(X2, axis=0) / n_samples
    mu = np.sum(emp_cov_trace) / n_features
    
    # Calculate sums of cross-products
    beta_ = np.sum(X2.T @ X2)
    delta_ = np.sum((X.T @ X) ** 2)
    
    delta_ /= n_samples**2
    
    # Calculate beta and delta coefficients
    beta = 1.0 / (n_features * n_samples) * (beta_ / n_samples - delta_)
    delta = delta_ - 2.0 * mu * emp_cov_trace.sum() + n_features * mu**2
    delta /= n_features
    
    beta = min(beta, delta)
    shrinkage = 0.0 if beta == 0 else beta / delta
    return shrinkage

def _ledoit_wolf_numpy(X, assume_centered=False):
    """Estimate the shrunk Ledoit-Wolf covariance matrix using pure NumPy."""
    n_samples, n_features = X.shape
    if not assume_centered:
        X = X - X.mean(axis=0)
    
    # Empirical covariance matrix
    emp_cov = (X.T @ X) / n_samples
    shrinkage = _ledoit_wolf_shrinkage(X, assume_centered=True)
    
    mu = np.trace(emp_cov) / n_features
    shrunk_cov = (1.0 - shrinkage) * emp_cov + shrinkage * mu * np.identity(n_features)
    return shrunk_cov, shrinkage

def compute_returns(prices_df, method="log"):
    """
    Compute asset returns from prices.
    
    Parameters:
    - prices_df: pd.DataFrame
    - method: str, "log" (logarithmic returns) or "pct" (percentage returns)
    
    Returns:
    - pd.DataFrame: asset returns
    """
    if prices_df.empty:
        raise ValueError("Prices DataFrame is empty.")
        
    if method == "log":
        # Log returns: ln(P_t / P_t-1)
        returns = np.log(prices_df / prices_df.shift(1))
    elif method == "pct":
        # Percentage returns: (P_t - P_t-1) / P_t-1
        returns = prices_df.pct_change()
    else:
        raise ValueError(f"Unknown return calculation method: {method}. Use 'log' or 'pct'.")
        
    # Drop first row because it will always be NaN due to shifting
    return returns.dropna(how="all")

def get_annualized_returns(returns, periods_per_year=252):
    """
    Annualize daily returns.
    
    Parameters:
    - returns: pd.DataFrame
    - periods_per_year: int, defaults to 252 (trading days in a year)
    
    Returns:
    - pd.Series: annualized expected returns per asset
    """
    return returns.mean() * periods_per_year

def get_covariance_matrix(returns, method="shrinkage", periods_per_year=252):
    """
    Calculate annualized covariance matrix.
    Uses Ledoit-Wolf shrinkage by default to improve stability.
    
    Parameters:
    - returns: pd.DataFrame
    - method: str, "shrinkage" (Ledoit-Wolf) or "sample" (standard sample covariance)
    - periods_per_year: int, defaults to 252
    
    Returns:
    - pd.DataFrame: annualized covariance matrix
    """
    if returns.empty:
        raise ValueError("Returns DataFrame is empty.")
        
    if method == "shrinkage":
        try:
            # Fit local Ledoit-Wolf estimator
            cov_array, _ = _ledoit_wolf_numpy(returns.values)
            cov_array = cov_array * periods_per_year
            cov_df = pd.DataFrame(cov_array, index=returns.columns, columns=returns.columns)
            return cov_df
        except Exception as e:
            logger.warning(f"Ledoit-Wolf covariance shrinkage failed: {e}. Falling back to sample covariance.")
            method = "sample"
            
    if method == "sample":
        cov_df = returns.cov() * periods_per_year
        return cov_df
    else:
        raise ValueError(f"Unknown covariance method: {method}. Use 'shrinkage' or 'sample'.")
