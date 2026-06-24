import sys
import numpy as np
import pandas as pd
import pytest

import os

# Add the root folder to the system path to allow importing library packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optimizer.optimizers import (
    mean_variance_optimize,
    min_variance_optimize,
    max_sharpe_optimize,
    risk_parity_optimize
)
from optimizer.feature_engineering import get_covariance_matrix, compute_returns

@pytest.fixture
def sample_data():
    """Generates synthetic stock returns for 4 assets."""
    np.random.seed(42)
    n_days = 252
    assets = ["AssetA", "AssetB", "AssetC", "AssetD"]
    
    # Generate returns with different means and volatilities
    means = [0.12, 0.08, 0.05, 0.15]
    vols = [0.18, 0.12, 0.10, 0.25]
    
    returns = np.random.normal(size=(n_days, len(assets)))
    for idx in range(len(assets)):
        returns[:, idx] = returns[:, idx] * (vols[idx] / np.sqrt(252)) + (means[idx] / 252)
        
    df_returns = pd.DataFrame(returns, columns=assets)
    return df_returns

def test_optimizers_constraints(sample_data):
    """Verify that all optimizers return weights that sum to 1.0 and are long-only (>= 0)."""
    mu = sample_data.mean() * 252
    cov = sample_data.cov() * 252
    rf = 0.03
    
    # Test MVO
    w_mvo = mean_variance_optimize(mu, cov)
    assert isinstance(w_mvo, pd.Series)
    assert np.isclose(w_mvo.sum(), 1.0, atol=1e-5)
    assert (w_mvo >= 0.0).all()
    assert list(w_mvo.index) == list(sample_data.columns)
    
    # Test Min Variance
    w_min = min_variance_optimize(cov)
    assert isinstance(w_min, pd.Series)
    assert np.isclose(w_min.sum(), 1.0, atol=1e-5)
    assert (w_min >= 0.0).all()
    assert list(w_min.index) == list(sample_data.columns)
    
    # Test Max Sharpe
    w_sharpe = max_sharpe_optimize(mu, cov, rf)
    assert isinstance(w_sharpe, pd.Series)
    assert np.isclose(w_sharpe.sum(), 1.0, atol=1e-5)
    assert (w_sharpe >= 0.0).all()
    assert list(w_sharpe.index) == list(sample_data.columns)
    
    # Test Risk Parity
    w_rp = risk_parity_optimize(cov)
    assert isinstance(w_rp, pd.Series)
    assert np.isclose(w_rp.sum(), 1.0, atol=1e-5)
    assert (w_rp >= 0.0).all()
    assert list(w_rp.index) == list(sample_data.columns)
    
    # Mathematical check: Verify that Risk Parity risk contributions are equal (1/N = 25% each)
    from optimizer.evaluation import calculate_risk_contribution
    rc_df = calculate_risk_contribution(w_rp, cov)
    rc_pct = rc_df["Percentage Contribution"]
    n_assets = len(sample_data.columns)
    expected_rc = np.repeat(1.0 / n_assets, n_assets)
    np.testing.assert_allclose(rc_pct.values, expected_rc, atol=1e-2)

def test_max_sharpe_edge_case(sample_data):
    """Verify that Max Sharpe optimizer handles cases where asset returns are lower than the risk-free rate."""
    mu = sample_data.mean() * 252
    cov = sample_data.cov() * 252
    
    # Set risk-free rate higher than all asset expected returns
    high_rf = mu.max() + 0.05
    
    # This should trigger a fallback to Minimum Variance
    w_sharpe = max_sharpe_optimize(mu, cov, high_rf)
    w_min = min_variance_optimize(cov)
    
    assert isinstance(w_sharpe, pd.Series)
    assert np.isclose(w_sharpe.sum(), 1.0, atol=1e-5)
    assert (w_sharpe >= 0.0).all()
    
    # Assert that the weights are mathematically identical to Minimum Variance
    pd.testing.assert_series_equal(w_sharpe, w_min)

def test_covariance_shrinkage(sample_data):
    """Verify that covariance estimators return correct dimensions and handle inputs correctly."""
    # Test sample covariance
    cov_sample = get_covariance_matrix(sample_data, method="sample")
    assert cov_sample.shape == (4, 4)
    assert (cov_sample.index == sample_data.columns).all()
    
    # Test shrinkage covariance
    cov_shrink = get_covariance_matrix(sample_data, method="shrinkage")
    assert cov_shrink.shape == (4, 4)
    assert (cov_shrink.index == sample_data.columns).all()
    
    # Test empty dataframe
    with pytest.raises(ValueError):
        compute_returns(pd.DataFrame())
        
    with pytest.raises(ValueError):
        get_covariance_matrix(pd.DataFrame())
