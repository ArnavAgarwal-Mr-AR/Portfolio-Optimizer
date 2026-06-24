import React, { useState, useEffect } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  ZAxis,
  Cell
} from 'recharts';
import './App.css';
import { Analytics } from '@vercel/analytics/react';

const strategyColors = {
  "Equal Weight": "#64748b",
  "Mean-Variance": "#f43f5e",
  "Min Variance": "#10b981",
  "Max Sharpe": "#f59e0b",
  "Risk Parity": "#6366f1"
};

export default function App() {
  // Input State Configs
  const [tickers, setTickers] = useState("SPY, TLT, GLD, QQQ, EFA, VNQ");
  const [startDate, setStartDate] = useState("2015-01-01");
  const [endDate, setEndDate] = useState("2025-12-31");
  const [backtestStartDate, setBacktestStartDate] = useState("2023-01-01");
  const [covMethod, setCovMethod] = useState("shrinkage");
  const [rfSource, setRfSource] = useState("auto");
  const [rfCustomValue, setRfCustomValue] = useState("4.0");
  const [simCount, setSimCount] = useState(1500);

  // App UI State
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [results, setResults] = useState(null);
  const [activeTab, setActiveTab] = useState("performance");
  const [strategyFocus, setStrategyFocus] = useState("Max Sharpe");
  const [pdfLoading, setPdfLoading] = useState(false);
  const [backendReady, setBackendReady] = useState(false);

  // Fetch defaults and ping backend on load
  useEffect(() => {
    let active = true;
    const checkBackendReady = async () => {
      try {
        const res = await fetch('/api/tickers');
        if (!res.ok) throw new Error("Backend not responding with 200 OK");
        const data = await res.json();
        if (active) {
          if (data.tickers) {
            setTickers(data.tickers.join(', '));
          }
          setBackendReady(true);
        }
      } catch (err) {
        if (active) {
          console.warn("Backend not ready yet, retrying in 2 seconds...", err);
          setTimeout(checkBackendReady, 2000);
        }
      }
    };
    checkBackendReady();
    return () => {
      active = false;
    };
  }, []);

  const handleRunOptimization = async (e) => {
    e.preventDefault();
    setLoading(true);
    setErrorMsg(null);
    setResults(null);

    const tickerList = tickers
      .split(',')
      .map(t => t.trim().toUpperCase())
      .filter(t => t !== '');

    if (tickerList.length === 0) {
      setErrorMsg("Tickers universe list cannot be empty.");
      setLoading(false);
      return;
    }

    try {
      const response = await fetch('/api/optimize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          tickers: tickerList,
          start_date: startDate,
          end_date: endDate,
          backtest_start_date: backtestStartDate,
          cov_method: covMethod,
          rf_value: rfSource === 'custom' ? parseFloat(rfCustomValue) / 100.0 : null,
          sim_count: parseInt(simCount)
        })
      });

      if (!response.ok) {
        let errMsg = "Optimization failed.";
        try {
          const data = await response.json();
          errMsg = data.detail || errMsg;
        } catch (_) {
          errMsg = `${response.status} ${response.statusText} (API Server may be offline)`;
        }
        throw new Error(errMsg);
      }

      const data = await response.json();
      setResults(data);
      // Default focus strategy selection
      if (data.metrics && !data.metrics[strategyFocus]) {
        setStrategyFocus(Object.keys(data.metrics)[0]);
      }
    } catch (err) {
      console.error(err);
      setErrorMsg(err.message);
    } finally {
      setLoading(false);
    }
  };

  // PDF Exporter utility
  const handleExportPDF = async () => {
    setPdfLoading(true);
    const tickerList = tickers
      .split(',')
      .map(t => t.trim().toUpperCase())
      .filter(t => t !== '');

    try {
      const response = await fetch('/api/export-pdf', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          tickers: tickerList,
          start_date: startDate,
          end_date: endDate,
          backtest_start_date: backtestStartDate,
          cov_method: covMethod,
          rf_value: rfSource === 'custom' ? parseFloat(rfCustomValue) / 100.0 : null,
          sim_count: parseInt(simCount)
        })
      });

      if (!response.ok) {
        throw new Error("Failed to generate PDF report from server.");
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `alphaoptima_portfolio_report.pdf`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (err) {
      console.error(err);
      alert(`PDF Export failed: ${err.message}`);
    } finally {
      setPdfLoading(false);
    }
  };

  // CSV Exporter utilities
  const exportWeightsCSV = () => {
    if (!results || !results.weights) return;
    const strategies = Object.keys(results.weights);
    const assetList = Object.keys(results.weights[strategies[0]]);
    const headers = ["Asset", ...strategies];
    
    let rows = [headers.join(",")];
    assetList.forEach(asset => {
      let row = [asset];
      strategies.forEach(strat => {
        row.push(results.weights[strat][asset]);
      });
      rows.push(row.join(","));
    });

    triggerCSVDownload(rows.join("\n"), "optimized_weights.csv");
  };

  const exportMetricsCSV = () => {
    if (!results || !results.metrics) return;
    const strategies = Object.keys(results.metrics);
    const metricNames = Object.keys(results.metrics[strategies[0]]);
    const headers = ["Strategy", ...metricNames];

    let rows = [headers.join(",")];
    strategies.forEach(strat => {
      let row = [strat];
      metricNames.forEach(m => {
        row.push(results.metrics[strat][m]);
      });
      rows.push(row.join(","));
    });

    triggerCSVDownload(rows.join("\n"), "performance_metrics.csv");
  };

  const triggerCSVDownload = (csvContent, fileName) => {
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", fileName);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Convert weights object into recharts bar array format
  const getWeightsChartData = () => {
    if (!results || !results.weights) return [];
    const strategies = Object.keys(results.weights);
    const assets = Object.keys(results.weights[strategies[0]]);
    return assets.map(asset => {
      const item = { name: asset };
      strategies.forEach(strat => {
        item[strat] = parseFloat((results.weights[strat][asset] * 100).toFixed(2));
      });
      return item;
    });
  };

  // Convert risk contributions into recharts bar array format
  const getRiskContributionsData = () => {
    if (!results || !results.risk_contributions) return [];
    const strategies = Object.keys(results.risk_contributions);
    const assets = results.risk_contributions[strategies[0]].map(d => d.asset);
    return assets.map(asset => {
      const item = { name: asset };
      strategies.forEach(strat => {
        const rcItem = results.risk_contributions[strat].find(d => d.asset === asset);
        item[strat] = rcItem ? parseFloat((rcItem.pct * 100).toFixed(2)) : 0;
      });
      return item;
    });
  };

  // Get color based on Sharpe ratio for Scatter points
  const getSharpeColor = (sharpe) => {
    const simulatedFrontier = results?.efficient_frontier?.simulated || [];
    if (!simulatedFrontier.length) return '#4f46e5';
    if (!window._sharpeBounds || window._sharpeBounds.resultsId !== results) {
      const sharpes = simulatedFrontier.map(p => p.sharpe);
      window._sharpeBounds = {
        min: Math.min(...sharpes),
        max: Math.max(...sharpes),
        resultsId: results
      };
    }
    const { min, max } = window._sharpeBounds;
    const range = max - min || 1;
    const pct = Math.max(0, Math.min(1, (sharpe - min) / range));
    const hue = 250 - (pct * 205);
    return `hsl(${hue}, 85%, 55%)`;
  };

  // Custom glowing shape for optimal portfolio nodes on the efficient frontier
  const renderOptimalNode = (props) => {
    const { cx, cy, fill } = props;
    return (
      <g transform={`translate(${cx},${cy})`}>
        {/* Outer glowing ring */}
        <circle r={7} fill="#ffffff" stroke={fill} strokeWidth={2} style={{ filter: 'drop-shadow(0 0 3px rgba(255, 255, 255, 0.7))' }} />
        {/* Inner colored core */}
        <circle r={3} fill={fill} />
      </g>
    );
  };

  return (
    <div className="app-container">
      {/* ----------------- TOP NAVIGATION BAR ----------------- */}
      <nav className="top-nav">
        <div className="brand-wrapper">
          <div className="brand-logo-icon">Ω</div>
          <h1 className="brand-text">AlphaOptima</h1>
          <span className="brand-tag">QUANT ENGINE</span>
        </div>
        <div className="nav-status">
          <div className={`status-dot ${backendReady ? 'operational' : 'connecting'}`}></div>
          <span>System Status: {backendReady ? "Operational" : "Connecting..."}</span>
        </div>
      </nav>

      {/* ----------------- DASHBOARD GRID ----------------- */}
      <div className="dashboard-grid">
        {/* Left Column: Strategy Configurator */}
        <aside className="glass-card" style={{ height: 'fit-content' }}>
          <div className="config-header">
            <h3 className="config-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--accent-indigo)' }}>
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
              </svg>
              Strategy Configurator
            </h3>
          </div>

          <form onSubmit={handleRunOptimization} style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
            <div>
              <span className="form-section-title" style={{ marginTop: 0 }}>Asset Space</span>
              <div className="form-group">
                <label className="form-label">Asset Tickers</label>
                <input
                  type="text"
                  className="form-input"
                  value={tickers}
                  onChange={(e) => setTickers(e.target.value)}
                  placeholder="SPY, TLT, GLD, QQQ..."
                />
              </div>
            </div>

            <div>
              <span className="form-section-title">Backtest Horizon</span>
              <div className="date-presets">
                <button
                  type="button"
                  className="preset-btn"
                  onClick={() => {
                    setStartDate("2020-01-01");
                    setEndDate("2025-12-31");
                    setBacktestStartDate("2024-01-01");
                  }}
                >
                  5Y Preset
                </button>
                <button
                  type="button"
                  className="preset-btn"
                  onClick={() => {
                    setStartDate("2015-01-01");
                    setEndDate("2025-12-31");
                    setBacktestStartDate("2023-01-01");
                  }}
                >
                  10Y Preset
                </button>
                <button
                  type="button"
                  className="preset-btn"
                  onClick={() => {
                    setStartDate("2010-01-01");
                    setEndDate("2025-12-31");
                    setBacktestStartDate("2020-01-01");
                  }}
                >
                  Max Horizon
                </button>
              </div>
              <div className="form-group">
                <label className="form-label">Start Date</label>
                <input
                  type="date"
                  className="form-input"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                />
              </div>
              <div className="form-group">
                <label className="form-label">End Date</label>
                <input
                  type="date"
                  className="form-input"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Backtest Split Date</label>
                <input
                  type="date"
                  className="form-input"
                  value={backtestStartDate}
                  onChange={(e) => setBacktestStartDate(e.target.value)}
                />
              </div>
            </div>

            <div>
              <span className="form-section-title">Model Adjustments</span>
              
              <div className="form-group">
                <label className="form-label">Covariance Estimation</label>
                <div className="segmented-control">
                  <button
                    type="button"
                    className={`segment-btn ${covMethod === 'shrinkage' ? 'active' : ''}`}
                    onClick={() => setCovMethod('shrinkage')}
                  >
                    Shrinkage
                  </button>
                  <button
                    type="button"
                    className={`segment-btn ${covMethod === 'sample' ? 'active' : ''}`}
                    onClick={() => setCovMethod('sample')}
                  >
                    Sample
                  </button>
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Risk-Free Rate Source</label>
                <div className="segmented-control">
                  <button
                    type="button"
                    className={`segment-btn ${rfSource === 'auto' ? 'active' : ''}`}
                    onClick={() => setRfSource('auto')}
                  >
                    Auto (FRED)
                  </button>
                  <button
                    type="button"
                    className={`segment-btn ${rfSource === 'custom' ? 'active' : ''}`}
                    onClick={() => setRfSource('custom')}
                  >
                    Custom
                  </button>
                </div>
              </div>

              {rfSource === 'custom' && (
                <div className="form-group">
                  <label className="form-label">Custom Rate (%)</label>
                  <input
                    type="number"
                    step="0.1"
                    min="0"
                    max="20"
                    className="form-input"
                    value={rfCustomValue}
                    onChange={(e) => setRfCustomValue(e.target.value)}
                  />
                </div>
              )}

              <div className="form-group">
                <label className="form-label">Frontier Simulations</label>
                <input
                  type="number"
                  min="200"
                  max="5000"
                  step="100"
                  className="form-input"
                  value={simCount}
                  onChange={(e) => setSimCount(e.target.value)}
                />
              </div>
            </div>

            <button type="submit" className="btn-run" disabled={loading || !backendReady}>
              {!backendReady ? "CONNECTING TO ENGINE..." : (loading ? "SOLVING CONVEX MODELS..." : "SOLVE ALLOCATIONS")}
            </button>
          </form>
        </aside>

        {/* Right Column: Main Content Area */}
        <main className="workspace-panel">
          {/* Welcome Intro Guide (No Results Yet) */}
          {!results && !loading && (
            <div className="welcome-container">
              <div className="welcome-header">
                <h2 className="welcome-title">Smart Beta Portfolio Optimizer</h2>
                <p className="welcome-subtitle">
                  An institutional-grade quantitative construction and risk modeling engine applying Modern Portfolio Theory (MPT).
                  Configure parameters in the left control deck and click <b>SOLVE ALLOCATIONS</b> to run solvers.
                </p>
              </div>

              <div className="manual-grid">
                <div className="manual-card">
                  <div className="manual-icon blue">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="12" cy="12" r="10"></circle>
                      <line x1="2" y1="12" x2="22" y2="12"></line>
                      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
                    </svg>
                  </div>
                  <div className="manual-text">
                    <h4>Asset Universe</h4>
                    <p>Input ETF/Stock symbols. Standard portfolios combine equities (SPY), bonds (TLT), real estate (VNQ), and commodities (GLD).</p>
                  </div>
                </div>

                <div className="manual-card">
                  <div className="manual-icon emerald">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
                      <line x1="16" y1="2" x2="16" y2="6"></line>
                      <line x1="8" y1="2" x2="8" y2="6"></line>
                      <line x1="3" y1="10" x2="21" y2="10"></line>
                    </svg>
                  </div>
                  <div className="manual-text">
                    <h4>Backtest Horizon & Split</h4>
                    <p>Solvers extract covariance and expected returns in the training range (up to split date), then backtest them out-of-sample.</p>
                  </div>
                </div>

                <div className="manual-card">
                  <div className="manual-icon purple">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21.21 15.89A10 10 0 1 1 8 2.83"></path>
                      <path d="M22 12A10 10 0 0 0 12 2v10z"></path>
                    </svg>
                  </div>
                  <div className="manual-text">
                    <h4>Optimization Strategies</h4>
                    <p>Solve 4 targets: Mean-Variance Optimization, Minimum Variance (risk reduction), Max Sharpe (optimal excess return), and Risk Parity (equal risk budget).</p>
                  </div>
                </div>

                <div className="manual-card">
                  <div className="manual-icon orange">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="18" y1="20" x2="18" y2="10"></line>
                      <line x1="12" y1="20" x2="12" y2="4"></line>
                      <line x1="6" y1="20" x2="6" y2="14"></line>
                    </svg>
                  </div>
                  <div className="manual-text">
                    <h4>Expected Output Analytics</h4>
                    <p>Visualize out-of-sample backtests, target asset allocation weights, risk-budgeting share, and the risk/return boundary frontier space.</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Loading Radar Spinner */}
          {loading && (
            <div className="spinner-container">
              <div className="radar-spinner"></div>
              <p className="spinner-text">GATHERING HISTORICAL DATA & SOLVING CONVEX MODELS...</p>
            </div>
          )}

          {/* Error Banner */}
          {errorMsg && (
            <div className="error-banner">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
                <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"></polygon>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
              </svg>
              <div>
                <h4 className="error-title">Optimization Pipeline Failed</h4>
                <p className="error-desc">{errorMsg}</p>
              </div>
            </div>
          )}

          {/* Results Workspace Grid */}
          {results && !loading && (
            <div className="results-workspace">
              <div className="results-header">
                <h3 className="workspace-title">Analytics Dashboard</h3>
              </div>

              {/* Active Strategy Focus custom selection deck */}
              <div className="strategy-focus-deck">
                <span className="strategy-focus-lbl">Active Strategy Focus:</span>
                <div className="strategy-focus-chips">
                  {Object.keys(results.metrics).map(strat => {
                    const isActive = strategyFocus === strat;
                    const activeColor = strategyColors[strat] || '#fff';
                    return (
                      <button
                        key={strat}
                        type="button"
                        className={`focus-chip ${isActive ? 'active' : ''}`}
                        style={{
                          '--chip-color': activeColor,
                        }}
                        onClick={() => setStrategyFocus(strat)}
                      >
                        <span className="chip-dot" style={{ backgroundColor: activeColor }}></span>
                        {strat}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* KPI Cards Grid */}
              <div className="kpi-grid">
                <div className="kpi-card sharpe">
                  <div className="kpi-icon cyan">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="12" y1="1" x2="12" y2="23"></line>
                      <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
                    </svg>
                  </div>
                  <div className="kpi-content">
                    <span className="kpi-lbl">Sharpe Ratio</span>
                    <span className="kpi-val" style={{ color: 'var(--accent-cyan)' }}>
                      {results.metrics[strategyFocus]["Sharpe Ratio"].toFixed(4)}
                    </span>
                  </div>
                </div>

                <div className="kpi-card return">
                  <div className="kpi-icon emerald">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>
                      <polyline points="17 6 23 6 23 12"></polyline>
                    </svg>
                  </div>
                  <div className="kpi-content">
                    <span className="kpi-lbl">Ann. Return</span>
                    <span className="kpi-val" style={{ color: 'var(--accent-emerald)' }}>
                      {(results.metrics[strategyFocus]["Annualized Return"] * 100).toFixed(4)}%
                    </span>
                  </div>
                </div>

                <div className="kpi-card vol">
                  <div className="kpi-icon orange">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M2 10s3-3 3-3 2.5 6 4 6 3-9 5-9 3.5 12 5 12 3-6 3-6"></path>
                    </svg>
                  </div>
                  <div className="kpi-content">
                    <span className="kpi-lbl">Ann. Volatility</span>
                    <span className="kpi-val" style={{ color: 'var(--accent-orange)' }}>
                      {(results.metrics[strategyFocus]["Annualized Volatility"] * 100).toFixed(4)}%
                    </span>
                  </div>
                </div>

                <div className="kpi-card drawdown">
                  <div className="kpi-icon red">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                      <line x1="12" y1="8" x2="12" y2="12"></line>
                      <line x1="12" y1="16" x2="12.01" y2="16"></line>
                    </svg>
                  </div>
                  <div className="kpi-content">
                    <span className="kpi-lbl">Max Drawdown</span>
                    <span className="kpi-val" style={{ color: 'var(--accent-red)' }}>
                      {(results.metrics[strategyFocus]["Max Drawdown"] * 100).toFixed(4)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Tabbed Interactive Visuals */}
              <div className="tabs-container">
                <nav className="tabs-nav">
                  <button
                    type="button"
                    className={`tab-trigger ${activeTab === 'performance' ? 'active' : ''}`}
                    onClick={() => setActiveTab('performance')}
                  >
                    📈 Performance Analytics
                  </button>
                  <button
                    type="button"
                    className={`tab-trigger ${activeTab === 'allocations' ? 'active' : ''}`}
                    onClick={() => setActiveTab('allocations')}
                  >
                    💼 Portfolio Allocations
                  </button>
                  <button
                    type="button"
                    className={`tab-trigger ${activeTab === 'risk' ? 'active' : ''}`}
                    onClick={() => setActiveTab('risk')}
                  >
                    ⚖️ Risk Budgeting
                  </button>
                  <button
                    type="button"
                    className={`tab-trigger ${activeTab === 'frontier' ? 'active' : ''}`}
                    onClick={() => setActiveTab('frontier')}
                  >
                    🌌 Efficient Frontier
                  </button>
                  <button
                    type="button"
                    className={`tab-trigger ${activeTab === 'diagnostics' ? 'active' : ''}`}
                    onClick={() => setActiveTab('diagnostics')}
                  >
                    🛡️ Diagnostics & Parameters
                  </button>
                </nav>

                <div className="chart-viewport">
                  {activeTab === 'performance' && (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={results.cumulative_returns} margin={{ top: 10, right: 10, left: 10, bottom: 5 }}>
                        <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" opacity={0.4} />
                        <XAxis dataKey="date" stroke="#9ca3af" fontSize={11} tickLine={false} />
                        <YAxis stroke="#9ca3af" fontSize={11} domain={['auto', 'auto']} tickLine={false} tickFormatter={(v) => v.toFixed(2)} />
                        <Tooltip
                          contentStyle={{ 
                            background: 'rgba(10, 10, 10, 0.95)', 
                            border: '1px solid rgba(255, 255, 255, 0.12)', 
                            borderRadius: '6px',
                            padding: '4px 8px',
                            fontSize: '11px'
                          }}
                          itemStyle={{ padding: '0px', margin: '2px 0' }}
                          labelStyle={{ color: '#fff', fontWeight: 'bold', margin: '0 0 2px 0' }}
                          formatter={(value) => value.toFixed(2)}
                        />
                        <Legend verticalAlign="top" height={36} formatter={(value) => <span style={{ color: '#e2e8f0', fontWeight: 500, marginRight: '10px' }}>{value}</span>} wrapperStyle={{ fontSize: '12px' }} />
                        {Object.keys(strategyColors).map(strat => (
                          <Line
                            key={strat}
                            type="monotone"
                            dataKey={strat}
                            stroke={strategyColors[strat]}
                            dot={false}
                            strokeWidth={strat === 'Equal Weight' ? 1.5 : 2.5}
                            strokeDasharray={strat === 'Equal Weight' ? "4 4" : undefined}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  )}

                  {activeTab === 'allocations' && (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={getWeightsChartData()} margin={{ top: 10, right: 10, left: 10, bottom: 5 }}>
                        <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" opacity={0.4} />
                        <XAxis dataKey="name" stroke="#9ca3af" fontSize={11} tickLine={false} />
                        <YAxis stroke="#9ca3af" fontSize={11} tickLine={false} unit="%" />
                        <Tooltip 
                          contentStyle={{ 
                            background: 'rgba(10, 10, 10, 0.95)', 
                            border: '1px solid rgba(255, 255, 255, 0.12)', 
                            borderRadius: '6px',
                            padding: '4px 8px',
                            fontSize: '11px'
                          }} 
                          itemStyle={{ padding: '0px', margin: '2px 0' }}
                          labelStyle={{ color: '#fff', fontWeight: 'bold', margin: '0 0 2px 0' }}
                          formatter={(value) => `${value}%`} 
                        />
                        <Legend verticalAlign="top" height={36} formatter={(value) => <span style={{ color: '#e2e8f0', fontWeight: 500, marginRight: '10px' }}>{value}</span>} wrapperStyle={{ fontSize: '12px' }} />
                        {Object.keys(strategyColors).map(strat => (
                          <Bar key={strat} dataKey={strat} fill={strategyColors[strat]} radius={[4, 4, 0, 0]} maxBarSize={40} />
                        ))}
                      </BarChart>
                    </ResponsiveContainer>
                  )}

                  {activeTab === 'risk' && (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={getRiskContributionsData()} margin={{ top: 10, right: 10, left: 10, bottom: 5 }}>
                        <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" opacity={0.4} />
                        <XAxis dataKey="name" stroke="#9ca3af" fontSize={11} tickLine={false} />
                        <YAxis stroke="#9ca3af" fontSize={11} tickLine={false} unit="%" />
                        <Tooltip 
                          contentStyle={{ 
                            background: 'rgba(10, 10, 10, 0.95)', 
                            border: '1px solid rgba(255, 255, 255, 0.12)', 
                            borderRadius: '6px',
                            padding: '4px 8px',
                            fontSize: '11px'
                          }} 
                          itemStyle={{ padding: '0px', margin: '2px 0' }}
                          labelStyle={{ color: '#fff', fontWeight: 'bold', margin: '0 0 2px 0' }}
                          formatter={(value) => `${value}%`} 
                        />
                        <Legend verticalAlign="top" height={36} formatter={(value) => <span style={{ color: '#e2e8f0', fontWeight: 500, marginRight: '10px' }}>{value}</span>} wrapperStyle={{ fontSize: '12px' }} />
                        {Object.keys(strategyColors).map(strat => (
                          <Bar key={strat} dataKey={strat} fill={strategyColors[strat]} radius={[4, 4, 0, 0]} maxBarSize={40} />
                        ))}
                      </BarChart>
                    </ResponsiveContainer>
                  )}

                  {activeTab === 'frontier' && (
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={{ top: 15, right: 15, bottom: 15, left: 10 }}>
                        <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" opacity={0.4} />
                        <XAxis
                          type="number"
                          dataKey="vol"
                          name="Volatility"
                          stroke="#9ca3af"
                          fontSize={11}
                          tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                          tickLine={false}
                        />
                        <YAxis
                          type="number"
                          dataKey="return"
                          name="Expected Return"
                          stroke="#9ca3af"
                          fontSize={11}
                          tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                          tickLine={false}
                        />
                        <ZAxis type="number" dataKey="sharpe" name="Sharpe Ratio" range={[20, 20]} />
                        <Tooltip
                          cursor={{ strokeDasharray: '3 3' }}
                          contentStyle={{ 
                            background: 'rgba(10, 10, 10, 0.95)', 
                            border: '1px solid rgba(255, 255, 255, 0.12)', 
                            borderRadius: '6px',
                            padding: '4px 8px',
                            fontSize: '11px'
                          }}
                          itemStyle={{ padding: '0px', margin: '2px 0' }}
                          labelStyle={{ color: '#fff', fontWeight: 'bold', margin: '0 0 2px 0' }}
                          formatter={(value, name) => {
                            if (name === "Volatility" || name === "Expected Return") {
                              return `${(value * 100).toFixed(2)}%`;
                            }
                            return value.toFixed(2);
                          }}
                        />
                        <Legend verticalAlign="top" height={36} formatter={(value) => <span style={{ color: '#e2e8f0', fontWeight: 500, marginRight: '10px' }}>{value}</span>} wrapperStyle={{ fontSize: '12px' }} />
                        
                        {/* Simulated scatter with heat map coloring */}
                        <Scatter
                          name="Simulated Portfolios"
                          data={results.efficient_frontier.simulated}
                        >
                          {results.efficient_frontier.simulated.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={getSharpeColor(entry.sharpe)} opacity={0.6} />
                          ))}
                        </Scatter>
                        
                        {/* Optimal points */}
                        {results.efficient_frontier.optimal.map((point) => (
                          <Scatter
                            key={point.name}
                            name={`Optimal: ${point.name}`}
                            data={[point]}
                            fill={strategyColors[point.name] || '#fff'}
                            shape={renderOptimalNode}
                          />
                        ))}
                      </ScatterChart>
                    </ResponsiveContainer>
                  )}

                  {activeTab === 'diagnostics' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', overflowY: 'auto', height: '100%', paddingRight: '0.5rem' }}>
                      <div>
                        <h4 style={{ margin: '0 0 0.5rem 0', color: '#fff', fontSize: '0.95rem' }}>Quantitative Solver & Parameter Diagnostics</h4>
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', margin: 0, lineHeight: 1.5 }}>
                          Detailed inspection of parameters solved from historical asset relationships and portfolio-level risk bounds.
                        </p>
                      </div>

                      <div className="data-details-grid" style={{ gap: '1.5rem' }}>
                        {/* Column 1: Solver Configuration & Horizons */}
                        <div className="section-card" style={{ background: 'rgba(255, 255, 255, 0.01)', border: '1px solid rgba(255, 255, 255, 0.05)', padding: '1rem', height: 'fit-content' }}>
                          <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent-cyan)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Solver & Horizon Context</span>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '0.75rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                            <div>
                              <span style={{ color: '#fff', fontWeight: 600 }}>Estimation Method:</span> <code style={{ color: 'var(--accent-cyan)' }}>{covMethod.toUpperCase()}</code>
                            </div>
                            <div>
                              <span style={{ color: '#fff', fontWeight: 600 }}>Asset Dimension:</span> <code>{results.tickers.length} x {results.tickers.length} covariance</code>
                            </div>
                            <div>
                              <span style={{ color: '#fff', fontWeight: 600 }}>Risk-Free Yield:</span> <span style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>{(results.rf_rate * 100).toFixed(4)}%</span>
                            </div>
                            <div>
                              <span style={{ color: '#fff', fontWeight: 600 }}>In-Sample Period:</span><br />
                              <span style={{ fontSize: '0.75rem' }}>{results.date_range.start} to {results.date_range.split}</span>
                            </div>
                            <div>
                              <span style={{ color: '#fff', fontWeight: 600 }}>Out-of-Sample Period:</span><br />
                              <span style={{ fontSize: '0.75rem' }}>{results.date_range.split} to {results.date_range.end}</span>
                            </div>
                            <div>
                              <span style={{ color: '#fff', fontWeight: 600 }}>Leverage Constraint:</span> <span>Long-Only (100% Invested)</span>
                            </div>
                            <div>
                              <span style={{ color: '#fff', fontWeight: 600 }}>Numerical Solver:</span> <code>CVXPY (OSQP/ECOS Engine)</code>
                            </div>
                          </div>
                        </div>

                        {/* Column 2: Portfolio Risk & Tail Boundaries */}
                        <div className="section-card" style={{ background: 'rgba(255, 255, 255, 0.01)', border: '1px solid rgba(255, 255, 255, 0.05)', padding: '1rem', height: 'fit-content' }}>
                          <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent-emerald)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Tail Risk & VaR Profiles</span>
                          <div className="table-container" style={{ marginTop: '0.75rem' }}>
                            <table className="custom-table" style={{ fontSize: '0.75rem' }}>
                              <thead>
                                <tr>
                                  <th>Strategy</th>
                                  <th style={{ textAlign: 'right' }}>Daily VaR (95%)</th>
                                  <th style={{ textAlign: 'right' }}>Daily CVaR (95%)</th>
                                </tr>
                              </thead>
                              <tbody>
                                {Object.keys(results.metrics).map(strat => {
                                  const varVal = results.metrics[strat]["Daily VaR (95%)"];
                                  const cvarVal = results.metrics[strat]["Daily CVaR (95%)"];
                                  return (
                                    <tr key={strat}>
                                      <td style={{ color: strategyColors[strat], fontWeight: 700 }}>{strat}</td>
                                      <td style={{ textAlign: 'right', color: 'var(--text-muted)' }}>
                                        {varVal !== undefined ? `${(varVal * 100).toFixed(4)}%` : 'N/A'}
                                      </td>
                                      <td style={{ textAlign: 'right', color: 'var(--accent-red)' }}>
                                        {cvarVal !== undefined ? `${(cvarVal * 100).toFixed(4)}%` : 'N/A'}
                                      </td>
                                    </tr>
                                  );
                                })}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      </div>

                      {/* Full Width Table: Individual Asset Solved Parameters */}
                      {results.individual_stats && (
                        <div className="section-card" style={{ background: 'rgba(255, 255, 255, 0.01)', border: '1px solid rgba(255, 255, 255, 0.05)', padding: '1.25rem' }}>
                          <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#f3f4f6', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Solved Asset Input Parameters (Annualized)</span>
                          <div className="table-container" style={{ marginTop: '0.75rem' }}>
                            <table className="custom-table">
                              <thead>
                                <tr>
                                  <th>Ticker Symbol</th>
                                  <th style={{ textAlign: 'right' }}>Expected Return (In-Sample)</th>
                                  <th style={{ textAlign: 'right' }}>Volatility (In-Sample)</th>
                                  <th style={{ textAlign: 'right' }}>Implied Sharpe Ratio</th>
                                  <th style={{ textAlign: 'center' }}>Quote Integrity</th>
                                </tr>
                              </thead>
                              <tbody>
                                {results.tickers.map(ticker => {
                                  const stats = results.individual_stats[ticker];
                                  if (!stats) return null;
                                  return (
                                    <tr key={ticker}>
                                      <td><code style={{ fontSize: '0.85rem', color: 'var(--accent-cyan)', fontWeight: 'bold' }}>{ticker}</code></td>
                                      <td style={{ textAlign: 'right', color: stats.expected_return >= 0 ? 'var(--accent-emerald)' : 'var(--accent-red)' }}>
                                        {(stats.expected_return * 100).toFixed(4)}%
                                      </td>
                                      <td style={{ textAlign: 'right', color: '#f3f4f6' }}>
                                        {(stats.volatility * 100).toFixed(4)}%
                                      </td>
                                      <td style={{ textAlign: 'right', fontWeight: 'bold', color: stats.sharpe >= 0 ? 'var(--accent-cyan)' : 'var(--accent-red)' }}>
                                        {stats.sharpe.toFixed(4)}
                                      </td>
                                      <td style={{ textAlign: 'center' }}>
                                        <span style={{ display: 'inline-block', padding: '2px 6px', borderRadius: '4px', background: 'rgba(16, 185, 129, 0.1)', color: 'var(--accent-emerald)', fontSize: '0.7rem', fontWeight: 700 }}>
                                          ACTIVE
                                        </span>
                                      </td>
                                    </tr>
                                  );
                                })}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Data Grid & Exporters */}
              <div className="data-details-grid">
                <div className="section-card">
                  <div className="section-card-header">
                    <h4 className="section-card-title">Allocation Profiles (%)</h4>
                    <button type="button" className="btn-export" onClick={exportWeightsCSV}>
                      📥 Export Weights
                    </button>
                  </div>
                  <div className="table-container">
                    <table className="custom-table">
                      <thead>
                        <tr>
                          <th>Ticker</th>
                          {Object.keys(results.weights).map(strat => (
                            <th key={strat} style={{ color: strategyColors[strat] }}>{strat}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {results.tickers.map(ticker => (
                          <tr key={ticker}>
                            <td><b>{ticker}</b></td>
                            {Object.keys(results.weights).map(strat => (
                              <td key={strat}>
                                {(results.weights[strat][ticker] * 100).toFixed(4)}%
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="section-card">
                  <div className="section-card-header">
                    <h4 className="section-card-title">Comparative Statistics</h4>
                    <button type="button" className="btn-export" onClick={exportMetricsCSV}>
                      📥 Export Metrics
                    </button>
                  </div>
                  <div className="table-container">
                    <table className="custom-table">
                      <thead>
                        <tr>
                          <th>Strategy</th>
                          <th>Return</th>
                          <th>Vol</th>
                          <th>Sharpe</th>
                          <th>Max DD</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.keys(results.metrics).map(strat => (
                          <tr key={strat}>
                            <td style={{ color: strategyColors[strat], fontWeight: 700 }}>{strat}</td>
                            <td>{(results.metrics[strat]["Annualized Return"] * 100).toFixed(4)}%</td>
                            <td>{(results.metrics[strat]["Annualized Volatility"] * 100).toFixed(4)}%</td>
                            <td style={{ fontWeight: 700, color: '#fff' }}>{results.metrics[strat]["Sharpe Ratio"].toFixed(4)}</td>
                            <td style={{ color: 'var(--accent-red)' }}>
                              {(results.metrics[strat]["Max Drawdown"] * 100).toFixed(4)}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
              
              <div className="pdf-export-footer">
                <button
                  type="button"
                  className="btn-export-pdf-large"
                  disabled={pdfLoading}
                  onClick={handleExportPDF}
                >
                  {pdfLoading ? "GENERATING EXECUTIVE PDF REPORT..." : "📥 Generate & Export Complete PDF Report"}
                </button>
              </div>
            </div>
          )}
        </main>
      </div>

      <footer className="footer">
        AlphaOptima Portfolio Suite — Powered by React, FastAPI & CVXPY
      </footer>
      <Analytics />
    </div>
  );
}
