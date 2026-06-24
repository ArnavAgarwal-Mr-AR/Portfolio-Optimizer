import logging
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

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
            # Fit Ledoit-Wolf estimator
            lw = LedoitWolf().fit(returns.values)
            cov_array = lw.covariance_ * periods_per_year
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
