"""
Phase 1.1: Download historical stock data
ONE JOB: Get raw data from Yahoo Finance and save it
"""
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

print("="*60)
print("DOWNLOADING STOCK DATA")
print("="*60)

# Define stocks and date range
tickers = ['AAPL', 'TSLA', 'GOOGL']
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

print(f"\nTickers: {', '.join(tickers)}")
print(f"Period: {start_date.date()} to {end_date.date()}\n")

# Download each ticker
for ticker in tickers:
    print(f"Downloading {ticker}...")

    # Download data - don't use yf.download's multi-ticker format
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, auto_adjust=True)

    # Reset index so Date becomes a column
    df.reset_index(inplace=True)

    # Keep only the columns we need
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Save raw data
    filename = f'data/raw/{ticker}_historical.csv'
    df.to_csv(filename, index=False)

    print(f"  ✓ {len(df)} days saved to {filename}")

print("\n✓ Download complete! All data saved to data/raw/")
