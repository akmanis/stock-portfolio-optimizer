import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import seaborn as sns

# Plotly
import plotly.graph_objects as go

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------- FONT ----------------
pdfmetrics.registerFont(TTFont('DejaVu', 'DejaVuSans.ttf'))

st.set_page_config(
    page_title="Stock Portfolio Optimizer", 
    layout="wide"
)

# ---------------- UI ----------------
st.title("Stock Portfolio Optimizer")

stocks_input = st.text_input(
    "Enter stock symbols (comma-separated)",
    "RELIANCE.NS,TCS.NS,INFY.NS"
)

investment = st.number_input("Investment Amount (₹)", value=10000)

risk_level = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"])

time_range = st.selectbox("Time Period", ["1mo", "6mo", "1y", "5y", "max"])

show_full_data = st.checkbox("Show Full Data")

# ---------------- STATE ----------------
if "data" not in st.session_state:
    st.session_state.data = None

# ---------------- FETCH ----------------
if st.button("Fetch Data"):
    stocks = [s.strip().upper() for s in stocks_input.split(",")]

    data = yf.download(stocks, period=time_range)

    if hasattr(data.columns, "levels"):
        data = data["Close"] if "Close" in data.columns.levels[0] else data["Adj Close"]
    else:
        data = data[["Close"]] if "Close" in data.columns else data[["Adj Close"]]

    data = data.dropna(axis=1, how='all')
    data = data.ffill().dropna()

    st.session_state.data = data

# ---------------- MAIN ----------------
data = st.session_state.data

if data is not None:

    stocks = list(data.columns)

    if len(stocks) < 2:
        st.error("Need at least 2 valid stocks")
        st.stop()

    st.subheader("Stock Price Data")

    if show_full_data:
        st.write(data)
    else:
        st.write(data.tail())

    # ================= PLOTLY CHART =================
    st.subheader("Price Chart")

    selected_stock = st.selectbox("Select Stock", stocks)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[selected_stock],
        mode='lines',
        name=selected_stock
    ))

    fig.update_layout(
        title=f"{selected_stock} Price",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- RETURNS ----------------
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    def portfolio_performance(weights):
        ret = np.sum(mean_returns * weights) * 252
        risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return ret, risk

    def objective(weights):
        ret, risk = portfolio_performance(weights)
        if risk_level == "Low":
            return risk
        elif risk_level == "High":
            return -ret
        else:
            return -(ret / risk)

    num_assets = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]

    optimal = minimize(objective, init_guess,
                       method='SLSQP', bounds=bounds, constraints=constraints)

    weights = optimal.x
    ret, risk = portfolio_performance(weights)

    # ---------------- TABLE ----------------
    df_weights = pd.DataFrame({
        "Stock": stocks,
        "Weight (%)": np.round(weights * 100, 2),
        "Investment (₹)": np.round(weights * investment, 2)
    })

    df_weights = df_weights[df_weights["Weight (%)"] > 0.5]

    # ---------------- PIE ----------------
    st.subheader("Portfolio Allocation")

    fig_pie = go.Figure(data=[go.Pie(
        labels=df_weights["Stock"],
        values=df_weights["Weight (%)"],
        hole=0.3
    )])

    st.plotly_chart(fig_pie, use_container_width=True)

    # ---------------- TABLE ----------------
    st.subheader("Investment Allocation")
    st.dataframe(df_weights, use_container_width=True)

    # ---------------- METRICS ----------------
    st.subheader("Portfolio Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Return", round(ret, 4))
    col2.metric("Risk", round(risk, 4))
    col3.metric("Sharpe", round(ret / risk, 4))

    # ---------------- MONTE CARLO ----------------
    st.subheader("Monte Carlo Simulation")

    num_simulations = 3000
    results = []

    for _ in range(num_simulations):
        w = np.random.random(len(mean_returns))
        w /= np.sum(w)

        r, v = portfolio_performance(w)
        results.append([r, v, r/v])

    results = np.array(results)

    fig_mc = go.Figure()

    fig_mc.add_trace(go.Scatter(
        x=results[:,1],
        y=results[:,0],
        mode='markers',
        marker=dict(color=results[:,2], colorscale='Viridis'),
        name="Portfolios"
    ))

    fig_mc.add_trace(go.Scatter(
        x=[risk],
        y=[ret],
        mode='markers',
        marker=dict(color='red', size=10),
        name="Optimal"
    ))

    fig_mc.update_layout(
        xaxis_title="Risk",
        yaxis_title="Return",
        title="Monte Carlo Simulation"
    )

    st.plotly_chart(fig_mc, use_container_width=True)

    # ---------------- HEATMAP ----------------
    st.subheader("Risk-Return Heatmap")

    fig_hm = go.Figure(data=go.Histogram2d(
        x=results[:,1],
        y=results[:,0],
        colorscale='Viridis'
    ))

    fig_hm.update_layout(
        xaxis_title="Risk",
        yaxis_title="Return"
    )

    st.plotly_chart(fig_hm, use_container_width=True)

    # ---------------- PDF ----------------
    pdf_file = "portfolio_report.pdf"
    doc = SimpleDocTemplate(pdf_file)

    styles = getSampleStyleSheet()
    styles['Normal'].fontName = 'DejaVu'
    styles['Title'].fontName = 'DejaVu'

    elements = []

    elements.append(Paragraph("Portfolio Report", styles['Title']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Time Period: {time_range}", styles['Normal']))
    elements.append(Paragraph(f"Risk Level: {risk_level}", styles['Normal']))
    elements.append(Paragraph(f"Expected Return: {round(ret,4)}", styles['Normal']))
    elements.append(Paragraph(f"Risk: {round(risk,4)}", styles['Normal']))
    elements.append(Spacer(1, 20))

    table_data = [["Stock", "Weight (%)", "Investment (₹)"]]

    for _, row in df_weights.iterrows():
        table_data.append([
            row["Stock"],
            f"{row['Weight (%)']}%",
            f"₹ {row['Investment (₹)']}"
        ])

    table = Table(table_data)

    table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'DejaVu'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    elements.append(table)
    doc.build(elements)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="Download Report",
            data=f,
            file_name="portfolio_report.pdf",
            mime="application/pdf"
        )
