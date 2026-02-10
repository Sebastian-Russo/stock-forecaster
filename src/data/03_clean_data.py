"""
Phase 1.3: Clean stock data
ONE JOB: Fix data quality issues
Handle missing values, remove outliers if needed, normalize dates
"""
import pandas as pd

print("="*60)
print("CLEANING STOCK DATA")
print("="*60)

tickers = ['AAPL', 'TSLA', 'GOOGL']

for ticker in tickers:
    print(f"\n{'='*60}")
    print(f"Cleaning {ticker}")
    print(f"{'='*60}")

    # Load raw data
    df = pd.read_csv(f'data/raw/{ticker}_historical.csv')
    print(f"Loaded: {len(df)} rows")

    # Handle Date column - convert to datetime and remove timezone
    print("Normalizing dates...")
    # First parse the datetime string
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    # Remove timezone by converting to timezone-naive
    df['Date'] = df['Date'].dt.tz_localize(None)
    print(f"  ✓ Dates normalized (timezone removed)")

    # Set Date as index
    df.set_index('Date', inplace=True)

    # Check for missing values
    missing_before = df.isnull().sum().sum()
    print(f"Missing values: {missing_before}")

    if missing_before > 0:
        print("Handling missing values...")
        # Forward fill (use previous day's value)
        df = df.fillna(method='ffill')
        # Backward fill for any remaining
        df = df.fillna(method='bfill')
        missing_after = df.isnull().sum().sum()
        print(f"  After cleaning: {missing_after} missing")
    else:
        print("  ✓ No missing values to handle")

    # Check for duplicates
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate dates, removing...")
        df = df[~df.index.duplicated(keep='first')]
    else:
        print("  ✓ No duplicate dates")

    # Sort by date (should already be sorted, but ensure)
    df = df.sort_index()

    # Add daily return column (useful for later)
    df['Daily_Return'] = df['Close'].pct_change()

    # Save cleaned data
    filename = f'data/processed/{ticker}_clean.csv'
    df.to_csv(filename)
    print(f"\n✓ Cleaned data saved: {filename}")
    print(f"  Final shape: {df.shape}")

    # Verify the date format in saved file
    print(f"  Sample date from saved file: {df.index[0]}")

print("\n✓ Phase 1.3 complete! All data cleaned and saved to data/processed/")
