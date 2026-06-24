import logging
import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from optimizer.exceptions import OptimizationError

logger = logging.getLogger(__name__)

def mean_variance_optimize(mu, cov, target_return=None, risk_aversion=1.0):
    """
    Mean-Variance Optimization (Markowitz).
    Finds the optimal weights that minimize risk (or minimize risk minus expected return).
    
    Parameters:
    - mu: pd.Series, annualized expected returns per asset
    - cov: pd.DataFrame, annualized covariance matrix
    - target_return: float, optional target portfolio return (constraint)
    - risk_aversion: float, risk aversion coefficient lambda (objective: w^T * cov * w - lambda * w^T * mu)
    
    Returns:
    - pd.Series: optimal portfolio weights
    """
    n = len(mu)
    w = cp.Variable(n)
    portfolio_return = mu.values @ w
    portfolio_risk = cp.quad_form(w, cov.values)
    
    # Constraints: weights sum to 1, long-only (weights >= 0)
    constraints = [cp.sum(w) == 1, w >= 0]
    
    if target_return is not None:
        # Prevent infeasibility by adjusting target return if it exceeds the maximum asset return
        max_mu = mu.max()
        if target_return > max_mu:
            logger.warning(f"Target return {target_return:.4f} exceeds maximum asset return {max_mu:.4f}. Clipping to {max_mu:.4f}.")
            target_return = max_mu - 1e-5
        constraints.append(portfolio_return >= target_return)
        objective = cp.Minimize(portfolio_risk)
    else:
        # Standard Markowitz quadratic objective
        objective = cp.Minimize(portfolio_risk - risk_aversion * portfolio_return)
        
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            weights = np.array(w.value)
            # Clip small negative values due to solver precision and re-normalize
            weights[weights < 0] = 0.0
            weights /= np.sum(weights)
            return pd.Series(weights, index=mu.index)
        else:
            raise OptimizationError(f"CVXPY Mean-Variance optimization failed with status: {prob.status}")
    except Exception as e:
        logger.warning(f"CVXPY Mean-Variance optimization failed: {e}. Falling back to scipy SLSQP solver...")
        return _mean_variance_scipy(mu, cov, target_return, risk_aversion)

def _mean_variance_scipy(mu, cov, target_return=None, risk_aversion=1.0):
    n = len(mu)
    
    def objective(w):
        ret = w @ mu.values
        vol2 = w @ cov.values @ w
        if target_return is not None:
            return vol2
        return vol2 - risk_aversion * ret
        
    init_guess = np.repeat(1.0 / n, n)
    bounds = [(0.0, 1.0) for _ in range(n)]
    
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_return is not None:
        constraints.append({"type": "ineq", "fun": lambda w: w @ mu.values - target_return})
        
    res = minimize(objective, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)
    if res.success:
        weights = res.x
        weights[weights < 0] = 0.0
        weights /= np.sum(weights)
        return pd.Series(weights, index=mu.index)
    else:
        logger.error(f"Mean-Variance scipy SLSQP failed: {res.message}. Falling back to equal weight.")
        return pd.Series(np.repeat(1.0 / n, n), index=mu.index)

def min_variance_optimize(cov):
    """
    Minimum Variance Optimization.
    Finds the portfolio weights that minimize the total portfolio risk (volatility).
    
    Parameters:
    - cov: pd.DataFrame, annualized covariance matrix
    
    Returns:
    - pd.Series: optimal portfolio weights
    """
    n = cov.shape[0]
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cov.values))
    constraints = [cp.sum(w) == 1, w >= 0]
    
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            weights = np.array(w.value)
            weights[weights < 0] = 0.0
            weights /= np.sum(weights)
            return pd.Series(weights, index=cov.index)
        else:
            raise OptimizationError(f"CVXPY Minimum Variance failed with status: {prob.status}")
    except Exception as e:
        logger.warning(f"CVXPY Minimum Variance failed: {e}. Falling back to scipy SLSQP solver...")
        return _min_variance_scipy(cov)

def _min_variance_scipy(cov):
    n = cov.shape[0]
    
    def objective(w):
        return w @ cov.values @ w
        
    init_guess = np.repeat(1.0 / n, n)
    bounds = [(0.0, 1.0) for _ in range(n)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    
    res = minimize(objective, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)
    if res.success:
        weights = res.x
        weights[weights < 0] = 0.0
        weights /= np.sum(weights)
        return pd.Series(weights, index=cov.index)
    else:
        logger.error(f"Minimum Variance scipy SLSQP failed: {res.message}. Falling back to equal weight.")
        return pd.Series(np.repeat(1.0 / n, n), index=cov.index)

def max_sharpe_optimize(mu, cov, rf):
    """
    Maximum Sharpe Ratio (Tangency) Portfolio.
    Finds the portfolio that maximizes excess return per unit of volatility.
    Solved using the convex formulation (Sharpe-Lintner transformation).
    
    Parameters:
    - mu: pd.Series, annualized expected returns per asset
    - cov: pd.DataFrame, annualized covariance matrix
    - rf: float, risk-free rate
    
    Returns:
    - pd.Series: optimal portfolio weights
    """
    # Check if any asset return is higher than the risk-free rate
    if (mu - rf).max() <= 0:
        logger.warning(f"No asset has annualized return above the risk-free rate ({rf:.4f}). "
                       "Falling back to Minimum Variance.")
        return min_variance_optimize(cov)
        
    n = len(mu)
    y = cp.Variable(n)
    
    # Minimize y^T * cov * y subject to (mu - rf)^T * y == 1, y >= 0
    obj = cp.Minimize(cp.quad_form(y, cov.values))
    constraints = [(mu.values - rf) @ y == 1, y >= 0]
    
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve()
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            weights = np.array(y.value) / np.sum(y.value)
            weights[weights < 0] = 0.0
            weights /= np.sum(weights)
            return pd.Series(weights, index=mu.index)
        else:
            raise OptimizationError(f"CVXPY Max Sharpe failed with status: {prob.status}")
    except Exception as e:
        logger.warning(f"CVXPY Max Sharpe failed: {e}. Falling back to scipy SLSQP solver...")
        return _max_sharpe_scipy(mu, cov, rf)

def _max_sharpe_scipy(mu, cov, rf):
    n = len(mu)
    
    def neg_sharpe(w):
        ret = w @ mu.values
        vol = np.sqrt(w @ cov.values @ w)
        if vol == 0:
            return 0.0
        return -(ret - rf) / vol
        
    init_guess = np.repeat(1.0 / n, n)
    bounds = [(0.0, 1.0) for _ in range(n)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    
    res = minimize(neg_sharpe, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)
    if res.success:
        weights = res.x
        weights[weights < 0] = 0.0
        weights /= np.sum(weights)
        return pd.Series(weights, index=mu.index)
    else:
        logger.error(f"Max Sharpe scipy SLSQP failed: {res.message}. Falling back to equal weight.")
        return pd.Series(np.repeat(1.0 / n, n), index=mu.index)

def risk_parity_optimize(cov):
    """
    Risk Parity (Equal Risk Contribution) Portfolio.
    Solves the convex formulation: min 0.5 * x^T * cov * x - sum(ln(x_i))
    where weight w_i = x_i / sum(x_j).
    
    Parameters:
    - cov: pd.DataFrame, annualized covariance matrix
    
    Returns:
    - pd.Series: optimal portfolio weights
    """
    n = cov.shape[0]
    
    # Spinu (2013) convex formulation
    def objective(x):
        return 0.5 * (x @ cov.values @ x) - np.sum(np.log(x))
        
    def jacobian(x):
        return cov.values @ x - 1.0 / x
        
    # Initial guess: inverse volatility of each asset
    vols = np.sqrt(np.diag(cov.values))
    init_guess = 1.0 / vols
    init_guess /= np.sum(init_guess)
    
    # x_i must be strictly positive.
    bounds = [(1e-8, None) for _ in range(n)]
    
    try:
        res = minimize(objective, init_guess, jac=jacobian, method="L-BFGS-B", bounds=bounds)
        if res.success:
            weights = res.x / np.sum(res.x)
            weights[weights < 0] = 0.0
            weights /= np.sum(weights)
            return pd.Series(weights, index=cov.index)
        else:
            raise OptimizationError(f"L-BFGS-B convex optimization failed: {res.message}")
    except Exception as e:
        logger.warning(f"Convex Risk Parity solver failed: {e}. Falling back to scipy SLSQP risk-difference minimization...")
        return _risk_parity_slsqp(cov)

def _risk_parity_slsqp(cov):
    n = cov.shape[0]
    
    def risk_contribution(w):
        vol = np.sqrt(w @ cov.values @ w)
        if vol == 0:
            return np.zeros(n)
        return w * (cov.values @ w) / vol
        
    def objective(w):
        rc = risk_contribution(w)
        # Sum of squared differences between all pairs of risk contributions
        diff = rc[:, None] - rc[None, :]
        return np.sum(diff ** 2)
        
    init_guess = np.repeat(1.0 / n, n)
    bounds = [(1e-5, 1.0) for _ in range(n)]  # prevent weight from going to 0
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    
    res = minimize(objective, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)
    if res.success:
        weights = res.x
        weights[weights < 0] = 0.0
        weights /= np.sum(weights)
        return pd.Series(weights, index=cov.index)
    else:
        logger.error(f"Risk Parity SLSQP failed: {res.message}. Falling back to equal weight.")
        return pd.Series(np.repeat(1.0 / n, n), index=cov.index)
