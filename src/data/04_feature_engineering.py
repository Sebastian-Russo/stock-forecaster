"""
Phase 2: Feature Engineering for Time Series
Create features that help predict future stock prices

KEY CONCEPT: In time series, we use PAST values to predict FUTURE
- Lag features: yesterday's price, last week's price
- Moving averages: smooth out noise, show trends
- Technical indicators: patterns traders use
"""
import pandas as pd
import numpy as np

def add_lag_features(df, columns=['Close'], lags=[1, 2, 3, 5, 10, 30]):
    """
    Create lag features: previous day's values

    Example: If today is day 100, lag_1 = day 99's value
    """
    for col in columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

def add_rolling_features(df, column='Close', windows=[7, 14, 30, 60]):
    """
    Create rolling (moving) averages

    MA7 = average of last 7 days
    Helps smooth out daily noise and see trends
    """
    for window in windows:
        # Rolling mean
        df[f'{column}_MA{window}'] = df[column].rolling(window=window).mean()

        # Rolling std (volatility)
        df[f'{column}_std{window}'] = df[column].rolling(window=window).std()

    return df

def add_technical_indicators(df):
    """
    Create technical analysis indicators
    These are patterns traders look for
    """
    # 1. RSI (Relative Strength Index) - measures overbought/oversold
    # Range: 0-100, >70 = overbought, <30 = oversold
    window = 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 2. MACD (Moving Average Convergence Divergence) - momentum indicator
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 3. Bollinger Bands - volatility bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)

    return df

def add_time_features(df):
    """
    Extract features from date
    Market patterns: Monday effect, end-of-month, etc.
    """
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)

    return df

def add_target(df, target_col='Close', horizon=1):
    """
    Create target: tomorrow's price

    horizon=1 means predict 1 day ahead
    """
    df['target'] = df[target_col].shift(-horizon)
    return df

# Process all stocks
print("="*60)
print("FEATURE ENGINEERING")
print("="*60)

tickers = ['AAPL', 'TSLA', 'GOOGL']

for ticker in tickers:
    print(f"\n{'='*60}")
    print(f"Processing {ticker}")
    print(f"{'='*60}")

    # Load clean data
    df = pd.read_csv(f'data/processed/{ticker}_clean.csv', index_col='Date', parse_dates=['Date'])

    print(f"Loaded: {df.shape}")

    # Add all features
    print("\n1. Adding lag features...")
    df = add_lag_features(df, columns=['Close', 'Volume'], lags=[1, 2, 3, 5, 10, 30])
    print(f"   Added lag features, shape: {df.shape}")

    print("2. Adding rolling features...")
    df = add_rolling_features(df, column='Close', windows=[7, 14, 30, 60])
    print(f"   Added rolling features, shape: {df.shape}")

    print("3. Adding technical indicators...")
    df = add_technical_indicators(df)
    print(f"   Added technical indicators, shape: {df.shape}")

    print("4. Adding time features...")
    df = add_time_features(df)
    print(f"   Added time features, shape: {df.shape}")

    print("5. Adding target (tomorrow's price)...")
    df = add_target(df, target_col='Close', horizon=1)
    print(f"   Added target, shape: {df.shape}")

    # Show sample
    print(f"\nFeature columns ({len(df.columns)} total):")
    print(df.columns.tolist())

    print(f"\nSample data (last 5 rows):")
    print(df.tail())

    # Check for NaN (will have some due to lag/rolling calculations)
    nan_count = df.isnull().sum().sum()
    print(f"\nNaN values: {nan_count}")
    print(f"NaN rows will be dropped before training")

    # Save
    filename = f'data/processed/{ticker}_featured.csv'
    df.to_csv(filename)
    print(f"\n✓ Saved: {filename}")

print("\n" + "="*60)
print("FEATURE ANALYSIS")
print("="*60)

# Analyze AAPL features
aapl = pd.read_csv('data/processed/AAPL_featured.csv', index_col='Date', parse_dates=['Date'])

print(f"\nTotal features: {len(aapl.columns)}")
print(f"Total rows: {len(aapl)}")
print(f"Rows with NaN: {aapl.isnull().any(axis=1).sum()}")
print(f"Clean rows (usable for training): {aapl.dropna().shape[0]}")

# Show feature categories
lag_features = [col for col in aapl.columns if 'lag' in col]
ma_features = [col for col in aapl.columns if 'MA' in col or 'std' in col]
tech_features = [col for col in aapl.columns if col in ['RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'BB_middle']]
time_features = [col for col in aapl.columns if col in ['day_of_week', 'day_of_month', 'month', 'quarter', 'is_month_start', 'is_month_end']]

print(f"\nFeature categories:")
print(f"  Lag features: {len(lag_features)}")
print(f"  Moving average features: {len(ma_features)}")
print(f"  Technical indicators: {len(tech_features)}")
print(f"  Time features: {len(time_features)}")
print(f"  Original features: 7 (Open, High, Low, Close, Volume, Daily_Return, target)")

# Visualize some features
print("\n" + "="*60)
print("VISUALIZING KEY FEATURES")
print("="*60)

import matplotlib.pyplot as plt
import seaborn as sns

# Plot 1: Price with moving averages
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

# Clean data for plotting (drop NaN)
plot_data = aapl.dropna()

axes[0].plot(plot_data.index, plot_data['Close'], label='Close', linewidth=2, alpha=0.8)
axes[0].plot(plot_data.index, plot_data['Close_MA7'], label='MA7', linewidth=1.5)
axes[0].plot(plot_data.index, plot_data['Close_MA30'], label='MA30', linewidth=1.5)
axes[0].set_title('AAPL: Price with Moving Averages', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Price ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: RSI
axes[1].plot(plot_data.index, plot_data['RSI'], linewidth=1.5, color='purple')
axes[1].axhline(y=70, color='red', linestyle='--', linewidth=1, label='Overbought')
axes[1].axhline(y=30, color='green', linestyle='--', linewidth=1, label='Oversold')
axes[1].set_title('RSI (Relative Strength Index)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('RSI')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: MACD
axes[2].plot(plot_data.index, plot_data['MACD'], label='MACD', linewidth=1.5)
axes[2].plot(plot_data.index, plot_data['MACD_signal'], label='Signal', linewidth=1.5)
axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[2].set_title('MACD (Moving Average Convergence Divergence)', fontsize=14, fontweight='bold')
axes[2].set_ylabel('MACD')
axes[2].set_xlabel('Date')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/feature_engineering.png', dpi=150)
print("✓ Saved: results/plots/feature_engineering.png")

print("\n✓ Phase 2 complete! Features engineered for all stocks.")
