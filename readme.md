# Smart Beta Portfolio Optimizer & Analytics Suite

An institutional-grade quantitative portfolio construction and historical backtesting engine. This application allows users to model, solve, and analyze asset allocations under various risk-budgeting paradigms.

---

## 📖 Table of Contents
1. [Documentation Links](#-documentation-links)
2. [Supported Allocation Methodologies](#-supported-allocation-methodologies)
3. [Systems Architecture Overview](#-systems-architecture-overview)
4. [File & Package Tree](#-file--package-tree)
5. [Installation & Requirements](#-installation--requirements)
6. [Command Line (CLI) Usage](#-command-line-cli-usage)
7. [Streamlit Web Application](#-streamlit-web-application)
8. [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 🔗 Documentation Links

To help you integrate, study, or deploy this project, we have created dedicated guides:
- **[Systems Architecture & Flows](file:///g:/Portfolio-Optimizer/docs/architecture.md)**: Details structural layout, sequence diagrams for historical data alignment, and flowcharts of solver fallback pipelines.
- **[Inputs & Outputs Explainer](file:///g:/Portfolio-Optimizer/docs/input_output_explainer.md)**: Documents variable types, dimensions, array shapes, and parameters of the internal library APIs.
- **[Financial & Mathematical Glossary](file:///g:/Portfolio-Optimizer/docs/financial_math_glossary.md)**: Explains the statistical mechanics and formulas (returns, covariance shrinkage, tail risks) backing the portfolio strategies.

---

## ⚖️ Supported Allocation Methodologies

All solvers enforce a **long-only** constraint ($w_i \ge 0$, $\sum w_i = 1$) to mirror index tilts found in smart beta exchange-traded funds (ETFs) and retail robo-advisors:

1. **Equal Weight (Benchmark)**
   - Allocates capital uniformly: $w_i = \frac{1}{N}$.
   - Acts as a naive benchmark to compare active risk-prevention models.

2. **Mean-Variance Optimization (Markowitz)**
   - Solves: $\min_w \quad w^T \Sigma w - \lambda w^T \mu$
   - Combines expected returns ($\mu$) and covariance ($\Sigma$), trading off risk aversion ($\lambda$).

3. **Minimum Variance Portfolio**
   - Solves: $\min_w \quad w^T \Sigma w$
   - Minimizes total portfolio standard deviation, ignoring expected returns. Useful for defensive, low-volatility tilts.

4. **Maximum Sharpe Ratio (Tangency Portfolio)**
   - Solves: $\max_w \quad \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}}$
   - Solved as a convex program using the Sharpe-Lintner variable transformation to locate the tangency point on the Capital Market Line. Falls back to Minimum Variance if all asset risk premiums are negative ($\mu - r_f \le 0$).

5. **Risk Parity (Equal Risk Contribution)**
   - Solves: $\min_x \quad \frac{1}{2} x^T \Sigma x - \sum \ln(x_i)$ (Spinu 2013)
   - Balances marginal risk contributions ($RC_i = w_i \frac{(\Sigma w)_i}{\sigma_p}$) instead of capital weight, neutralizing equity-concentration risk.

---

## 🏗️ Systems Architecture Overview

The system downloads stock prices from Yahoo Finance and interest rates from FRED (with fallback to 13-Week Treasury Bill `^IRX` if credentials are empty). The returns are computed and processed via a Ledoit-Wolf shrinkage estimator before entering the CVXPY/Scipy optimization engines. Results are evaluated out-of-sample and rendered via Plotly interactive dashboards.

For detailed sequence maps, please refer to the **[Systems Architecture Document](file:///g:/Portfolio-Optimizer/docs/architecture.md)**.

---

## 📂 File & Package Tree

```
g:\Portfolio-Optimizer/
├── app.py                       # Interactive Streamlit Web Dashboard
├── main.py                      # CLI Command Runner
├── config.yaml                  # Universal defaults & ticker lists
├── requirements.txt             # Project library requirements
├── .env.example                 # Credentials template
├── .env                         # Local credentials (FRED Key)
├── docs/                        # Specifications and guides
│   ├── architecture.md          # Systems architecture, flows, and solvers
│   └── input_output_explainer.md # Variable type, shape, and metric indexes
├── reports/                     # Output directory for CSV datasets and charts
│   ├── portfolio_weights.csv
│   ├── performance_metrics.csv
│   ├── cumulative_returns.csv
│   ├── backtest_cumulative_returns.png
│   ├── asset_allocations.png
│   ├── risk_contributions.png
│   └── efficient_frontier.png
├── src/                         
│   └── portfolio_optimizer/     # Library package
│       ├── __init__.py          
│       ├── config.py            # Parameters & env variables loader
│       ├── data_pipeline.py     # Resilient data fetching, cleaning, and cache
│       ├── feature_engineering.py # Daily returns & Ledoit-Wolf covariance
│       ├── optimizers.py        # Convex & scipy non-linear solvers
│       ├── evaluation.py        # Metrics calculations & backtesting
│       ├── exceptions.py        # Package custom exceptions
│       └── visualization.py     # Plotly interactive graphs
└── tests/                       
    └── test_optimizers.py       # Automated testing suite (pytest)
```

---

## Installation & Setup

Ensure **Python 3.8+** is installed on your system.

1. **Clone or download the project** to your local drive.
2. **Install core packages**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Establish Local Settings**:
   Copy `.env.example` to `.env` to override configuration defaults:
   ```bash
   copy .env.example .env
   ```
   You can populate `FRED_API_KEY` with a free key from FRED. If left empty, the pipeline triggers its fallback sequence, pulling the 13-Week Treasury Bill yield (`^IRX`) directly from Yahoo Finance.

---

## 🖥️ Command Line (CLI) Usage

Execute the default backtest pipeline (training weights on 2015–2022 data, backtesting out-of-sample on 2023–2025 data):
```bash
python main.py
```

### CLI Parameters & Customization
```
options:
  -h, --help                 show this help message and exit
  --tickers TICKERS          Comma-separated list of symbols (e.g. SPY,TLT,GLD)
  --start START              Historical data start date (YYYY-MM-DD)
  --end END                  Historical data end date (YYYY-MM-DD)
  --backtest-start BACKTEST  Out-of-sample backtest split date (YYYY-MM-DD)
  --cov {sample,shrinkage}   Covariance matrix estimation model
  --no-cache                 Force download data (disables raw CSV cache)
```

**Example Custom Execution**:
```bash
python main.py --tickers QQQ,TLT,GLD,SPY --start 2018-01-01 --end 2025-12-31 --backtest-start 2024-01-01 --cov shrinkage
```

---

## 🌐 React Web Application & API Server

The project is structured with a decoupled React frontend and a FastAPI backend.

### 1. Running Locally (FastAPI + React)
- **Option A: Automated Launcher (Recommended for Windows)**:
  Double-click **`run_dev.bat`** at the project root. This starts the FastAPI backend server (`port 8000`) and the Vite React server (`port 5173`) in concurrent console logs.
- **Option B: Manual Terminals**:
  1. **Start the API Server**:
     ```bash
     python server.py
     ```
     *Runs on `http://127.0.0.1:8000`.*
  2. **Start the React UI client**:
     ```bash
     cd frontend
     npm run dev
     ```
     *Runs on `http://localhost:5173`.*

### 2. Deploying to Vercel
The workspace is pre-configured for **Vercel** serverless hosting using `vercel.json`:
- **Serverless Ingestion**: Vercel routes `/api/*` REST endpoints to the Python handler in `api/index.py`. It installs dependencies from `requirements.txt` during the serverless container initialization.
- **Static Assets Compilation**: Vercel executes the React Vite build tool inside the `frontend/` subfolder and serves the static files.

To deploy using Vercel CLI, simply run from the repository root:
```bash
vercel
```

---

## Mathematical Formulation Reference

### 1. Mean-Variance Optimization
$$\min_w \quad w^T \Sigma w - \lambda w^T \mu$$
$$\text{subject to} \quad \sum w_i = 1, \quad w_i \ge 0$$
*Where $\Sigma$ is the annualized covariance matrix, $\mu$ is the annualized return, and $\lambda$ is risk aversion.*

### 2. Risk Parity (Equal Risk Contribution)
$$\min_x \quad \frac{1}{2} x^T \Sigma x - \sum_{i=1}^n \ln(x_i)$$
$$\text{subject to} \quad x \ge 0$$
*Once solved, optimal weights are normalized as $\w_i = \frac{x_i}{\sum x_j}$. This guarantees that each asset's risk contribution $RC_i = w_i \frac{(\Sigma w)_i}{\sigma_p}$ is equal.*

### 3. Maximum Sharpe Ratio (Tangency Portfolio)
$$\min_y \quad y^T \Sigma y$$
$$\text{subject to} \quad (\mu - r_f \mathbf{1})^T y = 1, \quad y \ge 0$$
*Solved via the Sharpe-Lintner convex transformation. Final weights are computed as $\w_i = \frac{y_i}{\sum y_j}$.*

### 4. Minimum Variance
$$\min_w \quad w^T \Sigma w$$
$$\text{subject to} \quad \sum w_i = 1, \quad w_i \ge 0$$