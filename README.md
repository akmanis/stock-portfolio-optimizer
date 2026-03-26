# Stock Portfolio Optimizer

An interactive fintech web application that helps users build and analyze optimal stock portfolios using Modern Portfolio Theory. The app fetches real-time market data, computes risk-return metrics, and provides optimized investment allocations with interactive visualizations.

## Overview

This project demonstrates practical applications of financial economics, data analysis, and optimization techniques. Users can input stock symbols, choose a time horizon and risk preference, and receive an optimized portfolio based on quantitative methods.

## Features

* Real-time stock data using Yahoo Finance
* Portfolio optimization based on risk tolerance (Low, Medium, High)
* Expected return, risk (volatility), and Sharpe ratio calculation
* Interactive price charts using Plotly (zoom, hover, dynamic view)
* Portfolio allocation visualization (pie chart)
* Monte Carlo simulation of random portfolios
* Risk-return heatmap visualization
* Investment allocation breakdown in currency terms
* Downloadable PDF portfolio report
* Adjustable time horizon (1 month to maximum available data)

## Tech Stack

* Python
* Streamlit
* Plotly
* NumPy
* Pandas
* SciPy
* yFinance
* ReportLab
* Seaborn

## Project Structure

```
stock-portfolio-optimizer/
│
├── app.py
├── requirements.txt
├── README.md
├── DejaVuSans.ttf
```

## Installation

1. Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/stock-portfolio-optimizer.git
cd stock-portfolio-optimizer
```

2. Create a virtual environment:

```
python -m venv venv
source venv/bin/activate      (Mac/Linux)
venv\Scripts\activate         (Windows)
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the application:

```
streamlit run app.py
```

## Usage

* Enter stock symbols (comma-separated), e.g. RELIANCE.NS, TCS.NS, INFY.NS
* Select investment amount
* Choose risk tolerance level
* Select time period for analysis
* Click "Fetch Data"
* Explore charts, optimization results, and simulations
* Download the portfolio report as a PDF

## Financial Model

The application is based on Modern Portfolio Theory (Markowitz Optimization):

* Calculates expected returns from historical data
* Computes covariance matrix of asset returns
* Optimizes portfolio weights based on:

  * Minimum risk (Low)
  * Maximum return (High)
  * Maximum Sharpe ratio (Medium)
* Uses constrained optimization (weights sum to 1, no short selling)

## Deployment

The app can be deployed using Streamlit Cloud:

1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect your repository
4. Select app.py and deploy

## Limitations

* Relies on historical data (no future prediction)
* No transaction costs or taxes included
* Assumes normally distributed returns
* Limited to long-only portfolios (no short selling)

## Future Improvements

* Add LSTM or ML-based return prediction
* Include risk-free rate for accurate Sharpe ratio
* Add portfolio comparison feature
* Improve UI with advanced dashboard layout
* Integrate live news or sentiment analysis

## Author

Manish (Economic Sciences,Indian Institute Of Science Education & Research(IISER), Bhopal)

## License

This project is for educational and demonstration purposes.
