# Quantitative Finance & Mathematical Glossary

This document provides detailed explanations of the financial metrics, statistical concepts, and mathematical optimization models utilized in the Portfolio Optimizer suite.

---

## 1. Core Returns & Statistical Concepts

### Simple Returns vs. Logarithmic Returns
Asset returns measure the change in price over a given period.
- **Simple Returns ($R_t$)**:
  $$R_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1$$
  *Simple returns are asset-additive*: the return of a portfolio is the weighted average of individual asset returns ($R_p = \sum w_i R_i$). However, they are not time-additive.
- **Logarithmic (Log) Returns ($r_t$)**:
  $$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln(P_t) - \ln(P_{t-1})$$
  *Log returns are time-additive*: the return over multiple days is the sum of daily returns ($r_{0,T} = \sum_{t=1}^T r_t$). They are widely used in quantitative finance because they are normally distributed under Black-Scholes assumptions and model continuous compounding.

*Conversion*: $R_t = e^{r_t} - 1$.

### Annualization
Since daily returns are small and noisy, they are annualized to provide a standard comparison scale:
- **Annualized Return ($\mu$)**:
  $$\mu = \text{Mean}(r_{daily}) \times 252$$
  *Assumes 252 trading days per year.*
- **Annualized Volatility ($\sigma$)**:
  $$\sigma = \text{StdDev}(r_{daily}) \times \sqrt{252}$$
  *Volatilities scale with the square root of time under the random walk hypothesis.*

### Covariance Matrix ($\Sigma$)
Covariance measures the joint variability of two assets. The annualized covariance matrix $\Sigma$ is an $N \times N$ symmetric matrix where the diagonal represents variances and off-diagonals represent covariance between asset pairs:
$$\Sigma_{i,j} = \text{Covariance}(r_i, r_j) \times 252$$
- If $\Sigma_{i,j} > 0$, assets tend to move together.
- If $\Sigma_{i,j} < 0$, assets move in opposite directions, providing diversification benefits.

---

## 2. Advanced Covariance Estimators

### Sample Covariance Matrix
Calculated directly from historical returns. While unbiased, it suffers from **estimation error** (known as "Markowitz's Curse"). With $N$ assets, we must estimate $N(N+1)/2$ parameters. If history is short, the matrix becomes noisy and ill-conditioned, leading the optimizer to allocate extreme weights to spurious relationships.

### Ledoit-Wolf Covariance Shrinkage
To stabilize the covariance estimate, Ledoit-Wolf shrinkage linearly interpolates ("shrinks") the noisy sample covariance matrix $S$ toward a highly structured target matrix $F$ (usually a constant correlation matrix):
$$\Sigma_{shrunk} = \delta F + (1 - \delta) S$$
Where $\delta \in [0, 1]$ is the optimal shrinkage intensity calculated analytically.
- **Why it matters**: Shrinkage reduces extreme values in the covariance matrix, guarantees the matrix is positive-definite (invertible), and improves the stability and out-of-sample performance of quadratic optimization models.

---

## 3. Allocation Paradigms & Optimization Models

All portfolio models in this project enforce the **long-only** constraint:
$$\sum_{i=1}^n w_i = 1.0, \quad w_i \ge 0.0$$

### Modern Portfolio Theory (MPT) & The Efficient Frontier
Introduced by Harry Markowitz (1952), MPT states that an investor can construct an optimal portfolio that maximizes return for a given level of risk (volatility). The **Efficient Frontier** is the boundary line representing portfolios that have the lowest risk for each level of expected return.

### Mean-Variance Optimization (MVO)
Maximizes expected portfolio return while minimizing risk:
$$\min_w \quad w^T \Sigma w - \lambda w^T \mu$$
Where:
- $w^T \Sigma w$ is the portfolio variance (risk).
- $w^T \mu$ is the expected portfolio return.
- $\lambda$ is the investor's risk-aversion coefficient. High $\lambda$ values place more weight on maximizing returns, while low values place weight on minimizing risk.

### Minimum Variance Portfolio
Locates the point on the Efficient Frontier with the absolute lowest volatility:
$$\min_w \quad w^T \Sigma w$$
This strategy is entirely independent of expected return estimates ($\mu$), making it highly robust to estimation errors.

### Maximum Sharpe Ratio (Tangency Portfolio)
Finds the portfolio that maximizes risk-adjusted return (excess return per unit of volatility):
$$\max_w \quad \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}}$$
This is a fractional programming problem (non-convex). We solve it by transforming it into a convex quadratic program using the **Sharpe-Lintner transformation**:
$$\min_y \quad y^T \Sigma y \quad \text{s.t.} \quad (\mu - r_f \mathbf{1})^T y = 1.0, \quad y \ge 0.0$$
The final weights are recovered by normalizing the solution: $w = \frac{y}{\sum y_i}$.

### Risk Parity (Equal Risk Contribution)
Unlike traditional portfolios (e.g. 60/40) where equity volatility dominates total risk, Risk Parity allocates capital so that **each asset contributes equally to the portfolio's total volatility**.
- **Marginal Risk Contribution (MRC)**:
  $$\text{MRC}_i = \frac{\partial \sigma_p}{\partial w_i} = \frac{(\Sigma w)_i}{\sigma_p}$$
- **Asset Risk Contribution (RC)**:
  $$\text{RC}_i = w_i \times \text{MRC}_i = w_i \frac{(\Sigma w)_i}{\sigma_p}$$
  Note that $\sum \text{RC}_i = \sigma_p$. Risk Parity requires $\text{RC}_i = \frac{1}{N} \sigma_p$ for all assets.
- **Spinu (2013) Convex Formulation**:
  Solving equal risk contributions directly is non-convex. Spinu formulated the equivalent strictly convex optimization problem:
  $$\min_x \quad \frac{1}{2} x^T \Sigma x - \sum_{i=1}^n \ln(x_i) \quad \text{s.t.} \quad x \ge 0.0$$
  Normalized weights are retrieved as: $w_i = \frac{x_i}{\sum x_j}$.

---

## 4. Performance & Risk Analytics

### Sharpe Ratio
Measures return per unit of total risk:
$$\text{Sharpe} = \frac{R_p - r_f}{\sigma_p}$$
Where $R_p$ is annualized portfolio return, $r_f$ is the risk-free rate, and $\sigma_p$ is annualized portfolio volatility.

### Sortino Ratio
Measures return per unit of downside risk. It replaces total volatility with **downside deviation**, which ignores "good volatility" (positive returns):
$$\text{Sortino} = \frac{R_p - r_f}{\sigma_{down}}$$
Where $\sigma_{down}$ is the annualized downside deviation:
$$\sigma_{down} = \sqrt{\text{Mean}\left( \min(R_{p,t}, 0.0)^2 \right)} \times \sqrt{252}$$

### Maximum Drawdown (Max DD)
Measures the largest peak-to-trough decline in the value of the portfolio:
$$\text{Drawdown}_t = \frac{V_t - \max_{s \le t}(V_s)}{\max_{s \le t}(V_s)}$$
$$\text{Max DD} = \min_{t} (\text{Drawdown}_t)$$
Where $V_t$ is the cumulative value index of the portfolio at time $t$.

### Value at Risk (VaR) 95%
The maximum loss that the portfolio is expected to experience over a one-day horizon with $95\%$ confidence.
- *Calculation*: The 5th percentile of daily returns. For example, a 95% Daily VaR of $-1.5\%$ means there is only a 5% chance the portfolio loses more than $1.5\%$ in a single day.

### Conditional Value at Risk (CVaR) 95% (Expected Shortfall)
Measures the expected loss on days when the loss exceeds the VaR threshold.
- *Calculation*: The average of all returns that fall below the 95% VaR threshold. CVaR provides a better measure of tail risk than VaR because it accounts for the severity of extreme tail losses.

---

## 5. Backtesting Methodology

### Out-of-Sample Backtesting (Walk-Forward validation)
Optimizers are prone to overfitting historical data. To evaluate true performance, we split the data:
- **In-Sample Period (Training)**: Historical data (e.g. 2015–2022) used to estimate mean returns ($\mu$) and covariance ($\Sigma$) and compute optimal weights.
- **Out-of-Sample Period (Testing)**: Subsequent historical data (e.g. 2023–2025). The computed static weights are applied to these returns to evaluate backtest performance. This prevents **look-ahead bias** and tests how the model behaves in unseen market regimes.
