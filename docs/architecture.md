# Systems Architecture & Data Flow

This document details the modular software engineering design, data flow patterns, and optimization fallback mechanisms implemented in the Smart Beta Portfolio Optimizer.

---

## 1. High-Level System Architecture

The application is structured into decoupled layers, ensuring separate concerns for configuration, data ingestion, feature estimation, convex solving, evaluation, and interface renderers.

```mermaid
graph TD
    classDef layer fill:#1e293b,stroke:#334155,stroke-width:2px,color:#f8fafc;
    classDef sub fill:#0f172a,stroke:#475569,stroke-width:1px,color:#94a3b8;
    
    subgraph Interfaces ["User Interface Layer"]
        Streamlit["Streamlit Dashboard (app.py)"]
        CLI["CLI Utility (main.py)"]
    end
    
    subgraph Library ["Core Optimization Library (src/portfolio_optimizer)"]
        Config["Config Module (config.py)"]
        Pipeline["Data Pipeline (data_pipeline.py)"]
        Features["Feature Engineering (feature_engineering.py)"]
        Solvers["Optimization Engine (optimizers.py)"]
        Eval["Evaluation & Backtesting (evaluation.py)"]
        Plotting["Visualization Module (visualization.py)"]
    end

    subgraph Data ["External Data Source Layer"]
        YF["Yahoo Finance API (yfinance)"]
        FRED["FRED API (fredapi)"]
    end

    %% Styles
    class Interfaces,Library,Data layer;
    class Streamlit,CLI,Config,Pipeline,Features,Solvers,Eval,Plotting,YF,FRED sub;

    %% Data Requests & Feeds
    FRED -->|10Yr/3Mo Treasury Yields| Pipeline
    YF -->|Asset Adj. Closing Prices & Fallback Yields| Pipeline
    Pipeline -->|Raw CSV Caching| Cache[("Local Data Cache (data/)")]
    
    Config -.->|Loads Settings & Keys| Pipeline
    Config -.->|Loads Parameters| Solvers
    
    Pipeline -->|Historical Arrays| Features
    Features -->|Returns & Covariance Shrinkage| Solvers
    Solvers -->|Optimal Weights| Eval
    Eval -->|Cumulative Returns & Performance Metrics| Plotting
    Plotting -->|Interactive Plotly / Static Matplotlib| Streamlit
    Plotting -->|Interactive Plotly / Static Matplotlib| CLI
```

---

## 2. Ingestion Sequence & Date Alignment Flow

The data pipeline fetches asset prices and interest rates, aligns trading day calendars, cleans missing quotes, and caches raw results to disk to prevent redundant web calls.

```mermaid
sequenceDiagram
    autonumber
    actor CLI/App as UI / Runner
    participant DP as Data Pipeline
    participant Cache as Local Storage (data/)
    participant YF as yfinance API
    participant FRED as Fred API

    CLI/App->>DP: fetch_prices(tickers, start, end)
    DP->>Cache: Check raw_prices.csv
    alt Cache Hit (Tickers & Range exist)
        Cache-->>DP: Return cached DataFrame
    else Cache Miss / No Cache
        DP->>YF: download(tickers, start, end, auto_adjust=True)
        YF-->>DP: Return Raw Prices DataFrame
        DP->>DP: Validate columns & check for full NaN columns
        DP->>Cache: Save/Merge cache to raw_prices.csv
    end
    DP-->>CLI/App: Return prices DataFrame

    CLI/App->>DP: fetch_risk_free_rate(start, end)
    DP->>Cache: Check raw_rf_rate.csv
    alt Cache Hit (Range exists)
        Cache-->>DP: Return cached Series
    else Cache Miss
        alt FRED_API_KEY is configured
            DP->>FRED: get_series(series_id, start, end)
            FRED-->>DP: Return yield percentage series
        else FRED_API_KEY is empty / invalid
            DP->>YF: download(^IRX, start, end)
            YF-->>DP: Return Close DataFrame
            DP->>DP: Squeeze Close into 1D Series
        end
        DP->>DP: Convert yield from percentage to decimal (/ 100)
        DP->>Cache: Save/Merge cache to raw_rf_rate.csv
    end
    DP-->>CLI/App: Return rf_rate Series

    CLI/App->>DP: align_data(prices, rf_rate)
    DP->>DP: Forward-fill and backward-fill asset holidays
    DP->>DP: Reindex and forward-fill rf_rate to match trading days
    DP-->>CLI/App: Return aligned (prices_df, rf_series)
```

---

## 3. Mathematical Optimization & Fallback Pathway

When optimization tasks are triggered, the engine primarily attempts to solve quadratic programs via CVXPY. If solvers experience convergence errors or negative risk premiums, they transition through fallback steps.

### Strategy Selection Overview
```mermaid
flowchart TD
    classDef start fill:#0369a1,stroke:#0284c7,stroke-width:2px,color:#fff;
    classDef step fill:#1e293b,stroke:#334155,stroke-width:1px,color:#e2e8f0;
    classDef check fill:#15803d,stroke:#166534,stroke-width:1px,color:#fff;
    classDef strategy fill:#1d4ed8,stroke:#1e40af,stroke-width:1px,color:#fff;
    classDef done fill:#6b21a8,stroke:#581c87,stroke-width:2px,color:#fff;

    StartNode(["Start Portfolio Optimization"]) --> GetParams["Retrieve Expected Returns mu & Covariance cov"]
    GetParams --> CovType{"Covariance Method?"}

    CovType -->|Shrinkage| FitLW["Fit Ledoit-Wolf Shrinkage covariance"]
    CovType -->|Sample| FitSample["Calculate standard sample covariance"]
    FitLW -->|Fails, falls back to| FitSample
    FitLW --> SolverSelection{"Strategy Selected?"}
    FitSample --> SolverSelection

    SolverSelection -->|Mean Variance| MVO["Mean-Variance Optimization<br/>(CVXPY, SLSQP fallback)<br/>Maximize: mu^T w - lambda * w^T cov w"]
    SolverSelection -->|Min Variance| MinVar["Minimum Variance<br/>(CVXPY, SLSQP fallback)<br/>Minimize: w^T cov w"]
    SolverSelection -->|Max Sharpe| MaxSharpe["Maximum Sharpe<br/>(Risk-premium check → CVXPY/SLSQP)<br/>Maximize: (mu - rf)^T w / sigma_p"]
    SolverSelection -->|Risk Parity| RiskParity["Risk Parity<br/>(L-BFGS-B, SLSQP fallback)<br/>Equalize marginal risk contributions"]

    MVO --> Finish(["Return Series w"])
    MinVar --> Finish
    MaxSharpe --> Finish
    RiskParity --> Finish

    class StartNode,Finish done;
    class GetParams,FitLW,FitSample step;
    class CovType,SolverSelection check;
    class MVO,MinVar,MaxSharpe,RiskParity strategy;
```

### Generic Solver Fallback Chain
```mermaid
flowchart TD
    classDef step fill:#1e293b,stroke:#334155,stroke-width:1px,color:#e2e8f0;
    classDef check fill:#15803d,stroke:#166534,stroke-width:1px,color:#fff;
    classDef err fill:#b91c1c,stroke:#991b1b,stroke-width:1px,color:#fff;
    classDef done fill:#6b21a8,stroke:#581c87,stroke-width:2px,color:#fff;

    Note["NOTE: This fallback pattern is identical across all four strategies.<br/>Only the objective function and primary solver differ.<br/>Risk Parity uses L-BFGS-B as primary instead of CVXPY."]

    Entry(["Strategy Objective Defined"]) --> Primary["Solve via Primary Method<br/>(CVXPY convex solver, or<br/>L-BFGS-B for Risk Parity)"]
    Primary --> PrimaryOK{"Solver Converged /<br/>Status Optimal?"}

    PrimaryOK -->|Yes| Normalize["Clip negative weights<br/>(if any) & Normalize to sum=1"]
    Normalize --> Finish(["Return Series w"])

    PrimaryOK -->|No / Error| LogWarn["Log warning:<br/>Fallback to scipy SLSQP"]
    LogWarn --> Secondary["Solve via scipy.optimize SLSQP<br/>(objective-specific:<br/>MVO / MinVar / neg-Sharpe / risk-diff)"]
    Secondary --> SecondaryOK{"SLSQP Success?"}

    SecondaryOK -->|Yes| Normalize
    SecondaryOK -->|No| LogErr["Log error:<br/>Primary and fallback both failed"]
    LogErr --> EqualWeight["Generate w = 1/N<br/>(Equal Weight, last resort)"]
    EqualWeight --> Finish

    class Entry,Finish done;
    class Primary,Secondary,Normalize,EqualWeight step;
    class PrimaryOK,SecondaryOK check;
    class LogWarn,LogErr err;
```

```mermaid
flowchart TD
    classDef start fill:#0369a1,stroke:#0284c7,stroke-width:2px,color:#fff;
    classDef step fill:#1e293b,stroke:#334155,stroke-width:1px,color:#e2e8f0;
    classDef check fill:#15803d,stroke:#166534,stroke-width:1px,color:#fff;
    classDef err fill:#b91c1c,stroke:#991b1b,stroke-width:1px,color:#fff;
    classDef done fill:#6b21a8,stroke:#581c87,stroke-width:2px,color:#fff;

    StartNode(["Start Portfolio Optimization"]) --> GetParams["Retrieve Expected Returns mu & Covariance cov"]
    GetParams --> CovType{"Covariance Method?"}

    CovType -->|Shrinkage| FitLW["Fit Ledoit-Wolf Shrinkage covariance"]
    CovType -->|Sample| FitSample["Calculate standard sample covariance"]

    FitLW -->|Successful Fit| SolverSelection{"Method Selected?"}
    FitLW -->|Contains NaN / Fails| WarnLW["Log warning"] --> FitSample
    FitSample --> SolverSelection

    %% Mean Variance
    SolverSelection -->|Mean Variance| SolMVO["CVXPY: Minimize quad_form - lambda * w^T * mu"]
    SolMVO --> MVOOK{"CVXPY Optimal?"}
    MVOOK -->|Yes| NormMVO["Clip negative weights & Normalize"] --> Finish(["Return Series w"])
    MVOOK -->|No / Error| FailMVO["Log warning: Fallback to scipy SLSQP"] --> ScipyMVO["Minimize via scipy.optimize SLSQP"]
    ScipyMVO --> ScipyMVOOK{"SLSQP Success?"}
    ScipyMVOOK -->|Yes| NormMVO
    ScipyMVOOK -->|No| ErrMVO["Log error: Fallback to Equal Weight"] --> EqualWeight["Generate w = 1/N"] --> Finish

    %% Min Variance
    SolverSelection -->|Min Variance| SolMin["CVXPY: Minimize quad_form w^T * cov * w"]
    SolMin --> MinOK{"CVXPY Optimal?"}
    MinOK -->|Yes| NormMin["Clip & Normalize"] --> Finish
    MinOK -->|No / Error| FailMin["Log warning: Fallback to scipy SLSQP"] --> ScipyMin["Minimize w^T * cov * w via scipy SLSQP"]
    ScipyMin --> ScipyMinOK{"SLSQP Success?"}
    ScipyMinOK -->|Yes| NormMin
    ScipyMinOK -->|No| ErrMin["Log error: Fallback to Equal Weight"] --> EqualWeight

    %% Max Sharpe
    SolverSelection -->|Max Sharpe| PremCheck{"Expected returns above rf rate?"}
    PremCheck -->|No| WarnSharpe["Log Warning: Risk Premium <= 0. Fallback to Min Var"] --> SolMin
    PremCheck -->|Yes| SolSharpe["CVXPY: Sharpe-Lintner Variable Transformation"]
    SolSharpe --> SharpeOK{"CVXPY Optimal?"}
    SharpeOK -->|Yes| NormSharpe["Calculate w_i = y_i / sum y"] --> Finish
    SharpeOK -->|No / Error| FailSharpe["Log warning: Fallback to scipy SLSQP"] --> ScipySharpe["Minimize negative Sharpe ratio via scipy SLSQP"]
    ScipySharpe --> ScipySharpeOK{"SLSQP Success?"}
    ScipySharpeOK -->|Yes| NormSharpe
    ScipySharpeOK -->|No| ErrSharpe["Log error: Fallback to Equal Weight"] --> EqualWeight

    %% Risk Parity
    SolverSelection -->|Risk Parity| SolRP["Scipy: Solve Spinu 2013 Convex Objective via L-BFGS-B"]
    SolRP --> RPOK{"L-BFGS-B Success?"}
    RPOK -->|Yes| NormRP["Calculate w_i = x_i / sum x"] --> Finish
    RPOK -->|No / Error| FailRP["Log warning: Fallback to SLSQP Risk Difference Minimizer"] --> ScipyRP["Minimize squared risk-contribution differences via scipy SLSQP"]
    ScipyRP --> ScipyRPOK{"SLSQP Success?"}
    ScipyRPOK -->|Yes| NormRP
    ScipyRPOK -->|No| ErrRP["Log error: Fallback to Equal Weight"] --> EqualWeight

    class StartNode,Finish done;
    class SolMVO,SolMin,SolSharpe,SolRP,ScipyMVO,ScipyMin,ScipySharpe,ScipyRP step;
    class MVOOK,MinOK,SharpeOK,RPOK,ScipyMVOOK,ScipyMinOK,ScipySharpeOK,ScipyRPOK,PremCheck,CovType check;
    class FailMVO,FailMin,FailSharpe,FailRP,ErrMVO,ErrMin,ErrSharpe,ErrRP,WarnLW,WarnSharpe err;
```