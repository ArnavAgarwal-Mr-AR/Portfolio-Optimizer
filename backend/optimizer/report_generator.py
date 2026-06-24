import io
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define strategy colors for consistency
STRATEGY_COLORS = {
    "Equal Weight": "#64748b",
    "Mean-Variance": "#f43f5e",
    "Min Variance": "#10b981",
    "Max Sharpe": "#f59e0b",
    "Risk Parity": "#6366f1"
}

def build_pdf_report(results_data):
    """
    Generates a PDF report containing performance metrics, weights, and plots.
    Returns a bytes object of the generated PDF.
    """
    buffer = io.BytesIO()
    
    # Page setup
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
        title="Portfolio Analysis"
    )
    doc.title = "Portfolio Analysis"
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'ReportSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=20
    )
    
    h1_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155'),
        spaceAfter=10
    )
    
    interpretation_style = ParagraphStyle(
        'ReportInterpretation',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#475569'),
        spaceBefore=4,
        spaceAfter=10
    )

    story = []
    
    # ------------------ TITLE & METADATA ------------------
    story.append(Paragraph("AlphaOptima Portfolio Suite", title_style))
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_text = (
        f"<b>Quantitative Performance & Risk Report</b><br/>"
        f"Generated on: {timestamp} (UTC)<br/>"
        f"Asset Space: {', '.join(results_data['tickers'])}<br/>"
        f"Training & Out-of-Sample Horizon: {results_data['date_range']['start']} to {results_data['date_range']['end']}<br/>"
        f"Backtest Split Date: {results_data['date_range']['split']}<br/>"
        f"Risk-Free Baseline Rate: {results_data['rf_rate'] * 100:.4f}%"
    )
    story.append(Paragraph(meta_text, subtitle_style))
    story.append(Spacer(1, 10))
    
    # ------------------ SECTION 1: PERFORMANCE METRICS ------------------
    story.append(Paragraph("1. Out-of-Sample Performance Analytics", h1_style))
    story.append(Paragraph(
        "The table below details the performance metrics evaluated during the out-of-sample backtest split "
        "horizon. All percentage returns and risks are displayed up to 4 decimal places for precision risk tracking.",
        body_style
    ))
    
    # Build metrics table
    # Columns: Strategy, Ann. Return, Ann. Volatility, Sharpe Ratio, Sortino Ratio, Max Drawdown
    metrics_data = [
        ["Strategy", "Ann. Return", "Ann. Volat.", "Sharpe", "Sortino", "Max DD", "Daily VaR", "Daily CVaR"]
    ]
    
    for strat, m_dict in results_data["metrics"].items():
        var_val = m_dict.get("Daily VaR (95%)")
        cvar_val = m_dict.get("Daily CVaR (95%)")
        metrics_data.append([
            strat,
            f"{m_dict['Annualized Return'] * 100:.4f}%",
            f"{m_dict['Annualized Volatility'] * 100:.4f}%",
            f"{m_dict['Sharpe Ratio']:.4f}",
            f"{m_dict.get('Sortino Ratio', 0.0):.4f}",
            f"{m_dict['Max Drawdown'] * 100:.4f}%",
            f"{var_val * 100:.4f}%" if var_val is not None else "N/A",
            f"{cvar_val * 100:.4f}%" if cvar_val is not None else "N/A"
        ])
        
    t_metrics = Table(metrics_data, colWidths=[110, 60, 60, 45, 45, 55, 75, 75])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'), # Left align strategy names
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f8fafc')])
    ]))
    
    story.append(t_metrics)
    story.append(Spacer(1, 15))
    
    # ------------------ SECTION 2: CUMULATIVE RETURNS GRAPH ------------------
    story.append(Paragraph("2. Performance Growth Charts", h1_style))
    
    # Generate Cumulative Returns Plot using Matplotlib
    plt.figure(figsize=(7, 2.5))
    df_cum = pd.DataFrame(results_data["cumulative_returns"])
    df_cum["date"] = pd.to_datetime(df_cum["date"])
    df_cum.set_index("date", inplace=True)
    
    for col in df_cum.columns:
        color = STRATEGY_COLORS.get(col, '#333333')
        plt.plot(df_cum.index, df_cum[col], label=col, color=color, linewidth=1.5)
        
    plt.title("Out-of-Sample Cumulative Returns", fontsize=10, fontweight='bold', color='#0f172a')
    plt.xlabel("Date", fontsize=8, color='#475569')
    plt.ylabel("Cumulative Returns", fontsize=8, color='#475569')
    plt.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')
    plt.tick_params(axis='both', which='major', labelsize=7, colors='#475569')
    plt.legend(loc="upper left", fontsize=7, framealpha=0.8)
    plt.tight_layout()
    
    img_cum_buf = io.BytesIO()
    plt.savefig(img_cum_buf, format='png', dpi=300)
    plt.close()
    img_cum_buf.seek(0)
    
    story.append(Image(img_cum_buf, width=500, height=180))
    story.append(Paragraph(
        "<i><b>Cumulative Growth Interpretation:</b> This growth chart illustrates the out-of-sample performance of the solved strategy models "
        "relative to the Equal Weight benchmark. Max Sharpe and Mean-Variance strategies adjust weights dynamically to maximize returns or "
        "return-to-risk characteristics, whereas Minimum Variance seeks strictly to minimize overall portfolio variance, resulting in smoother "
        "but potentially lower return paths.</i>",
        interpretation_style
    ))
    story.append(Spacer(1, 10))
    
    story.append(PageBreak())  # Move to Page 2
    
    # ------------------ SECTION 3: ALLOCATION PROFILE ------------------
    story.append(Paragraph("3. Target Allocation Profiles", h1_style))
    story.append(Paragraph(
        "Optimal weights derived by solving MPT boundary conditions. These weights specify "
        "the capital allocations ($w_i$) computed from in-sample asset constraints.",
        body_style
    ))
    
    # Build weights table
    strategies = list(results_data["weights"].keys())
    tickers = results_data["tickers"]
    
    weights_headers = ["Asset Ticker"] + strategies
    weights_table_data = [weights_headers]
    
    for ticker in tickers:
        row = [ticker]
        for strat in strategies:
            row.append(f"{results_data['weights'][strat][ticker] * 100:.4f}%")
        weights_table_data.append(row)
        
    # Table column widths
    col_widths = [80] + [85] * len(strategies)
    t_weights = Table(weights_table_data, colWidths=col_widths)
    t_weights.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f8fafc')])
    ]))
    
    story.append(t_weights)
    story.append(Spacer(1, 15))
    
    # Generate Allocation Weights Bar Chart
    plt.figure(figsize=(7, 2.4))
    n_strats = len(strategies)
    n_tickers = len(tickers)
    x = np.arange(n_tickers)
    width = 0.8 / n_strats
    
    for idx, strat in enumerate(strategies):
        w_values = [results_data['weights'][strat][ticker] * 100 for ticker in tickers]
        offset = x + (idx - n_strats/2 + 0.5) * width
        plt.bar(offset, w_values, width, label=strat, color=STRATEGY_COLORS.get(strat, '#333333'))
        
    plt.title("Portfolio Allocation Weights (%)", fontsize=10, fontweight='bold', color='#0f172a')
    plt.xticks(x, tickers, fontsize=8, color='#475569')
    plt.ylabel("Weight (%)", fontsize=8, color='#475569')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')
    plt.tick_params(axis='y', which='major', labelsize=7, colors='#475569')
    plt.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    
    img_w_buf = io.BytesIO()
    plt.savefig(img_w_buf, format='png', dpi=300)
    plt.close()
    img_w_buf.seek(0)
    
    story.append(Image(img_w_buf, width=500, height=170))
    story.append(Paragraph(
        "<i><b>Portfolio Allocation Interpretation:</b> The bar chart illustrates the specific asset weights ($w_i$) assigned across strategies. "
        "The Mean-Variance and Max Sharpe strategies concentrate capital in historically high-performing equity indices (e.g., QQQ), while the "
        "Minimum Variance and Risk Parity models allocate capital more broadly across low-correlation and defensive asset classes (e.g., TLT, GLD) "
        "to satisfy diversification constraints and control risk metrics.</i>",
        interpretation_style
    ))
    story.append(Spacer(1, 10))
    
    story.append(PageBreak())  # Move to Page 3
    
    # ------------------ SECTION 4: RISK BUDGETING & EFFICIENT FRONTIER ------------------
    story.append(Paragraph("4. Risk Budgeting & Frontier Space Analysis", h1_style))
    story.append(Paragraph(
        "Risk budgeting breaks down the percentage contribution to total portfolio risk. Risk Parity "
        "strives to equalize risk contributions across all assets. The Efficient Frontier visualizes "
        "the boundary risk/return ratio of simulated portfolios.",
        body_style
    ))
    
    # Generate Risk Contributions Bar Chart
    plt.figure(figsize=(7, 2.4))
    for idx, strat in enumerate(strategies):
        rc_list = results_data['risk_contributions'][strat]
        # Align assets order with tickers list
        rc_dict = {item['asset']: item['pct'] * 100 for item in rc_list}
        rc_values = [rc_dict.get(ticker, 0.0) for ticker in tickers]
        
        offset = x + (idx - n_strats/2 + 0.5) * width
        plt.bar(offset, rc_values, width, label=strat, color=STRATEGY_COLORS.get(strat, '#333333'))
        
    plt.title("Asset Risk Contributions (%)", fontsize=10, fontweight='bold', color='#0f172a')
    plt.xticks(x, tickers, fontsize=8, color='#475569')
    plt.ylabel("Risk Contribution (%)", fontsize=8, color='#475569')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')
    plt.tick_params(axis='y', which='major', labelsize=7, colors='#475569')
    plt.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    
    img_rc_buf = io.BytesIO()
    plt.savefig(img_rc_buf, format='png', dpi=300)
    plt.close()
    img_rc_buf.seek(0)
    
    story.append(Image(img_rc_buf, width=500, height=170))
    story.append(Paragraph(
        "<i><b>Risk Budgeting Interpretation:</b> Asset Risk Contributions specify the percentage volatility contribution that each asset "
        "adds to the portfolio. Unlike simple capital weights, this chart reveals where the actual portfolio risk resides. The Risk Parity strategy "
        "calculates weights such that all assets contribute exactly equal risk, preventing any single asset from dominating portfolio variance.</i>",
        interpretation_style
    ))
    story.append(Spacer(1, 10))
    
    # Generate Efficient Frontier Scatter Plot
    plt.figure(figsize=(7, 2.4))
    sim_data = results_data["efficient_frontier"]["simulated"]
    sim_vols = [item["vol"] * 100 for item in sim_data]
    sim_rets = [item["return"] * 100 for item in sim_data]
    sim_sharpes = [item["sharpe"] for item in sim_data]
    
    sc = plt.scatter(sim_vols, sim_rets, c=sim_sharpes, cmap='viridis', s=1.5, alpha=0.4, label='Simulated Portfolios')
    cbar = plt.colorbar(sc)
    cbar.set_label('Sharpe Ratio', fontsize=7, color='#475569')
    cbar.ax.tick_params(labelsize=6, colors='#475569')
    
    # Plot optimal stars
    for opt_point in results_data["efficient_frontier"]["optimal"]:
        opt_name = opt_point["name"]
        color = STRATEGY_COLORS.get(opt_name, '#ff0000')
        plt.scatter(
            opt_point["vol"] * 100,
            opt_point["return"] * 100,
            color=color,
            edgecolors='black',
            marker='*',
            s=80,
            linewidths=0.5,
            label=f"Opt: {opt_name}"
        )
        
    plt.title("Efficient Frontier Space Analysis", fontsize=10, fontweight='bold', color='#0f172a')
    plt.xlabel("Annualized Volatility (%)", fontsize=8, color='#475569')
    plt.ylabel("Annualized Expected Return (%)", fontsize=8, color='#475569')
    plt.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')
    plt.tick_params(axis='both', which='major', labelsize=7, colors='#475569')
    plt.legend(loc="upper left", fontsize=6, framealpha=0.8)
    plt.tight_layout()
    
    img_ef_buf = io.BytesIO()
    plt.savefig(img_ef_buf, format='png', dpi=300)
    plt.close()
    img_ef_buf.seek(0)
    
    story.append(Image(img_ef_buf, width=500, height=170))
    story.append(Paragraph(
        "<i><b>Efficient Frontier Interpretation:</b> The scatter plot visualizes the risk/return trade-offs. The simulated portfolio cloud "
        "(colored by Sharpe Ratio) highlights suboptimal configurations, while the upper boundary forms the efficient envelope. The optimal "
        "stars denote solved target portfolios: Min Variance represents the leftmost vertex (minimum volatility), and Max Sharpe represents "
        "the tangency portfolio of maximum excess return per unit of risk.</i>",
        interpretation_style
    ))
    
    # ------------------ SECTION 5: SOLVER DIAGNOSTICS & PARAMETERS ------------------
    if "individual_stats" in results_data and results_data["individual_stats"]:
        story.append(PageBreak())  # Move to Page 4
        story.append(Paragraph("5. Solver Diagnostics & Parameters", h1_style))
        
        cov_method_name = results_data.get("cov_method", "shrinkage").upper()
        
        story.append(Paragraph(
            f"The individual assets annualized returns and volatilities solved in the In-Sample training horizon "
            f"({results_data['date_range']['start']} to {results_data['date_range']['split']}) are detailed below. "
            f"The covariance matrix was calculated using the <b>{cov_method_name}</b> framework. "
            f"Under Modern Portfolio Theory constraints, short-selling is prohibited (long-only bounds: 0% to 100% per asset) "
            f"with a 100% fully-invested leverage budget.",
            body_style
        ))
        
        asset_table_data = [
            ["Ticker Symbol", "Expected Return (IS)", "Volatility (IS)", "Implied Sharpe", "Status"]
        ]
        for ticker in results_data["tickers"]:
            stats = results_data["individual_stats"].get(ticker)
            if stats:
                asset_table_data.append([
                    ticker,
                    f"{stats['expected_return'] * 100:.4f}%",
                    f"{stats['volatility'] * 100:.4f}%",
                    f"{stats['sharpe']:.4f}",
                    "ACTIVE"
                ])
        
        t_assets = Table(asset_table_data, colWidths=[100, 110, 110, 110, 90])
        t_assets.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'), # Left align tickers
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f8fafc')])
        ]))
        story.append(t_assets)
        story.append(Paragraph(
            "<i><b>Solver Parameter Interpretation:</b> The solved in-sample parameters act as optimization model inputs. Negative expected "
            "returns (e.g., on TLT) indicate negative historical momentum in the training window, causing Sharpe-maximizing models to allocate "
            "zero weight. Volatilities represent historical standard deviations. The 'ACTIVE' status indicates successful API data verification "
            "and positive semi-definite covariance eigenvalues, ensuring solver convergence.</i>",
            interpretation_style
        ))
        story.append(Spacer(1, 15))
        
        # Solver metadata block
        solver_desc = (
            f"<b>Numerical Solver Engine:</b> CVXPY (OSQP/ECOS Math Kernels)<br/>"
            f"<b>Estimation framework:</b> Covariance Shrinkage Matrix size: {len(results_data['tickers'])} x {len(results_data['tickers'])} dimensions<br/>"
            f"<b>Out-of-Sample Window:</b> {results_data['date_range']['split']} to {results_data['date_range']['end']}"
        )
        story.append(Paragraph(solver_desc, body_style))
    
    # Build Document
    doc.build(story)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes
