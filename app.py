import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import seaborn as sns

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------- FONT FIX ----------------
pdfmetrics.registerFont(TTFont('DejaVu', 'DejaVuSans.ttf'))

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

    # ---------------- CHART ----------------
    st.subheader("Interactive Price Chart")

    selected_stock = st.selectbox("Select Stock", stocks)

    fig, ax = plt.subplots()
    ax.plot(data.index, data[selected_stock])
    ax.set_title(selected_stock)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)

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
    fig, ax = plt.subplots()
    ax.pie(df_weights["Weight (%)"], labels=df_weights["Stock"],
           autopct='%1.1f%%', startangle=90)
    ax.axis('equal')

    st.subheader("Portfolio Allocation")
    st.pyplot(fig)

    # ---------------- TABLE DISPLAY ----------------
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

    fig, ax = plt.subplots()
    scatter = ax.scatter(results[:,1], results[:,0], c=results[:,2])
    ax.scatter(risk, ret, color='red', s=100)
    ax.set_xlabel("Risk")
    ax.set_ylabel("Return")
    st.pyplot(fig)

    # ---------------- HEATMAP ----------------
    st.subheader("Risk-Return Heatmap")

    fig, ax = plt.subplots()
    sns.kdeplot(x=results[:,1], y=results[:,0], fill=True, cmap="viridis", ax=ax)
    ax.set_xlabel("Risk")
    ax.set_ylabel("Return")
    st.pyplot(fig)

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
    elements.append(Paragraph(f"Risk (Volatility): {round(risk,4)}", styles['Normal']))
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

    # ---------------- DOWNLOAD ----------------
    with open(pdf_file, "rb") as f:
        st.download_button(
            label="Download Report",
            data=f,
            file_name="portfolio_report.pdf",
            mime="application/pdf"
        )