import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from optimizer.evaluation import calculate_risk_contribution

# Beautiful color palettes
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
STRATEGY_COLORS = {
    "Equal Weight": "#64748b",
    "Mean-Variance": "#f43f5e",
    "Min Variance": "#10b981",
    "Max Sharpe": "#f59e0b",
    "Risk Parity": "#6366f1"
}

def plot_efficient_frontier_plotly(mu, cov, rf, optimal_weights=None, n_portfolios=3000):
    """
    Simulates random portfolios and plots them along with optimal portfolios on the Efficient Frontier.
    
    Parameters:
    - mu: pd.Series, annualized expected returns
    - cov: pd.DataFrame, annualized covariance matrix
    - rf: float, risk-free rate
    - optimal_weights: dict, dictionary of strategy weights (e.g. {'Max Sharpe': Series})
    
    Returns:
    - go.Figure: Plotly figure
    """
    n_assets = len(mu)
    
    # 1. Generate random weights using Dirichlet distribution
    np.random.seed(42)
    random_weights = np.random.dirichlet(np.ones(n_assets), n_portfolios)
    
    # Calculate returns and volatilities for random portfolios
    port_returns = random_weights @ mu.values
    port_vols = np.zeros(n_portfolios)
    for i in range(n_portfolios):
        port_vols[i] = np.sqrt(random_weights[i] @ cov.values @ random_weights[i])
        
    sharpe_ratios = (port_returns - rf) / port_vols
    
    # Create hover text
    hover_texts = []
    for i in range(n_portfolios):
        txt = "<br>".join([f"{ticker}: {random_weights[i, idx]:.1%}" for idx, ticker in enumerate(mu.index)])
        hover_texts.append(f"Return: {port_returns[i]:.2%}<br>Vol: {port_vols[i]:.2%}<br>Sharpe: {sharpe_ratios[i]:.2f}<br><br><b>Weights:</b><br>{txt}")
        
    # Scatter plot of simulated portfolios
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=port_vols,
        y=port_returns,
        mode="markers",
        marker=dict(
            size=4,
            color=sharpe_ratios,
            colorscale="Viridis",
            colorbar=dict(title="Sharpe Ratio", thickness=15),
            showscale=True,
            opacity=0.6
        ),
        text=hover_texts,
        hoverinfo="text",
        name="Simulated Portfolios"
    ))
    
    # 2. Add optimal portfolios as special highlighted dots
    if optimal_weights:
        for name, weights in optimal_weights.items():
            w = weights.values
            p_return = w @ mu.values
            p_vol = np.sqrt(w @ cov.values @ w)
            p_sharpe = (p_return - rf) / p_vol
            
            color = STRATEGY_COLORS.get(name, "#e377c2")
            
            # Hover text for optimal portfolio
            w_txt = "<br>".join([f"{ticker}: {weights[ticker]:.1%}" for ticker in weights.index])
            opt_hover = f"<b>{name} Portfolio</b><br>Return: {p_return:.2%}<br>Vol: {p_vol:.2%}<br>Sharpe: {p_sharpe:.2f}<br><br><b>Weights:</b><br>{w_txt}"
            
            fig.add_trace(go.Scatter(
                x=[p_vol],
                y=[p_return],
                mode="markers+text",
                marker=dict(
                    color=color,
                    size=14,
                    symbol="star",
                    line=dict(color="white", width=1.5)
                ),
                text=[name],
                textposition="top center",
                hovertext=[opt_hover],
                hoverinfo="text",
                name=name
            ))
            
    # Figure styling for premium UI look
    fig.update_layout(
        title=dict(
            text="Efficient Frontier & Optimal Portfolios",
            font=dict(size=18, family="Outfit, Inter, sans-serif")
        ),
        xaxis=dict(
            title="Annualized Volatility (Risk)",
            tickformat=".1%",
            gridcolor="#2A2A2A",
            zerolinecolor="#2A2A2A"
        ),
        yaxis=dict(
            title="Annualized Return",
            tickformat=".1%",
            gridcolor="#2A2A2A",
            zerolinecolor="#2A2A2A"
        ),
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="#E0E0E0", family="Outfit, Inter, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        height=600
    )
    
    return fig

def plot_allocation_weights_plotly(weights_df):
    """
    Creates an interactive grouped bar chart comparing weights across strategies.
    
    Parameters:
    - weights_df: pd.DataFrame, rows are tickers, columns are strategies
    
    Returns:
    - go.Figure: Plotly figure
    """
    # Transform DataFrame for plotting
    df_melted = weights_df.reset_index().melt(
        id_vars="index", var_name="Strategy", value_name="Weight"
    ).rename(columns={"index": "Asset"})
    
    fig = px.bar(
        df_melted,
        x="Strategy",
        y="Weight",
        color="Asset",
        barmode="group",
        text="Weight",
        color_discrete_sequence=COLORS,
        labels={"Weight": "Allocation Weight"}
    )
    
    fig.update_traces(
        texttemplate="%{y:.1%}",
        textposition="outside",
        cliponaxis=False,
        marker=dict(line=dict(width=1, color="#1E1E1E"))
    )
    
    fig.update_layout(
        title=dict(
            text="Asset Allocation Comparison Across Strategies",
            font=dict(size=18, family="Outfit, Inter, sans-serif")
        ),
        xaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#2A2A2A"),
        yaxis=dict(
            gridcolor="#2A2A2A",
            zerolinecolor="#2A2A2A",
            tickformat=".0%",
            range=[0, weights_df.max().max() * 1.15]
        ),
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="#E0E0E0", family="Outfit, Inter, sans-serif"),
        margin=dict(l=60, r=40, t=80, b=60),
        height=500
    )
    
    return fig

def plot_risk_contributions_plotly(weights_df, cov):
    """
    Plots the percentage risk contributions of each asset for all strategies.
    
    Parameters:
    - weights_df: pd.DataFrame, rows are tickers, columns are strategies
    - cov: pd.DataFrame, annualized covariance matrix
    
    Returns:
    - go.Figure: Plotly figure
    """
    contributions = []
    
    # Calculate contributions for each strategy
    for strategy in weights_df.columns:
        w = weights_df[strategy]
        rc_df = calculate_risk_contribution(w, cov)
        rc_pct = rc_df["Percentage Contribution"]
        
        for asset, val in rc_pct.items():
            contributions.append({
                "Strategy": strategy,
                "Asset": asset,
                "Risk Contribution": val
            })
            
    df_contrib = pd.DataFrame(contributions)
    
    fig = px.bar(
        df_contrib,
        x="Strategy",
        y="Risk Contribution",
        color="Asset",
        barmode="group",
        text="Risk Contribution",
        color_discrete_sequence=COLORS,
        labels={"Risk Contribution": "% of Portfolio Risk"}
    )
    
    fig.update_traces(
        texttemplate="%{y:.1%}",
        textposition="outside",
        cliponaxis=False,
        marker=dict(line=dict(width=1, color="#1E1E1E"))
    )
    
    fig.update_layout(
        title=dict(
            text="Risk Contribution Comparison (Volatility Breakdown)",
            font=dict(size=18, family="Outfit, Inter, sans-serif")
        ),
        xaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#2A2A2A"),
        yaxis=dict(
            gridcolor="#2A2A2A",
            zerolinecolor="#2A2A2A",
            tickformat=".0%",
            range=[0, df_contrib["Risk Contribution"].max() * 1.15]
        ),
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="#E0E0E0", family="Outfit, Inter, sans-serif"),
        margin=dict(l=60, r=40, t=80, b=60),
        height=500
    )
    
    return fig

def plot_cumulative_returns_plotly(cum_returns_df):
    """
    Plots historical cumulative returns for all optimized strategies.
    
    Parameters:
    - cum_returns_df: pd.DataFrame, columns are strategies, index is Date
    
    Returns:
    - go.Figure: Plotly figure
    """
    fig = go.Figure()
    
    for strategy in cum_returns_df.columns:
        color = STRATEGY_COLORS.get(strategy, "#e377c2")
        fig.add_trace(go.Scatter(
            x=cum_returns_df.index,
            y=cum_returns_df[strategy],
            mode="lines",
            name=strategy,
            line=dict(color=color, width=2.5),
            hovertemplate="<b>" + strategy + "</b><br>Date: %{x}<br>Value: %{y:.3f}<extra></extra>"
        ))
        
    fig.update_layout(
        title=dict(
            text="Out-of-Sample Backtest: Cumulative Returns",
            font=dict(size=18, family="Outfit, Inter, sans-serif")
        ),
        xaxis=dict(
            title="Date",
            gridcolor="#2A2A2A",
            zerolinecolor="#2A2A2A"
        ),
        yaxis=dict(
            title="Portfolio Growth (Starting at 1.0)",
            gridcolor="#2A2A2A",
            zerolinecolor="#2A2A2A",
            tickformat=".2f"
        ),
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="#E0E0E0", family="Outfit, Inter, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        height=500
    )
    
    return fig
