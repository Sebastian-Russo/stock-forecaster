"""
Phase 1.2: Explore stock data
ONE JOB: Understand the data structure, patterns, statistics
NO modifications to data, just observation
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (15, 6)

print("="*60)
print("EXPLORING STOCK DATA")
print("="*60)

# Load AAPL as example
print("\nLoading AAPL data...")
aapl = pd.read_csv('data/raw/AAPL_historical.csv', index_col='Date', parse_dates=['Date'])

# Basic info
print("\n" + "="*60)
print("DATA STRUCTURE")
print("="*60)
print(f"\nShape: {aapl.shape}")
print(f"Columns: {list(aapl.columns)}")
print(f"Date range: {aapl.index.min()} to {aapl.index.max()}")
print(f"Total trading days: {len(aapl)}")

print("\nFirst 5 rows:")
print(aapl.head())

print("\nLast 5 rows:")
print(aapl.tail())

print("\nData types:")
print(aapl.dtypes)

# Check for missing values
print("\n" + "="*60)
print("DATA QUALITY")
print("="*60)
missing = aapl.isnull().sum()
print("\nMissing values per column:")
print(missing)

if missing.sum() > 0:
    print(f"\n⚠️  Total missing: {missing.sum()} values")
else:
    print("\n✓ No missing values")

# Statistical summary
print("\n" + "="*60)
print("STATISTICAL SUMMARY")
print("="*60)
print(aapl.describe())

# Price range
print("\n" + "="*60)
print("PRICE ANALYSIS")
print("="*60)
print(f"Lowest Close: ${aapl['Close'].min():.2f}")
print(f"Highest Close: ${aapl['Close'].max():.2f}")
print(f"Average Close: ${aapl['Close'].mean():.2f}")
print(f"Current Close: ${aapl['Close'].iloc[-1]:.2f}")

# Calculate daily returns for analysis
daily_returns = aapl['Close'].pct_change()

print("\n" + "="*60)
print("VOLATILITY ANALYSIS")
print("="*60)
print(f"Mean daily return: {daily_returns.mean()*100:.3f}%")
print(f"Std deviation: {daily_returns.std()*100:.3f}%")
print(f"Best day: +{daily_returns.max()*100:.2f}%")
print(f"Worst day: {daily_returns.min()*100:.2f}%")

# Visualizations
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# 1. Price over time
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

axes[0].plot(aapl.index, aapl['Close'], linewidth=2, color='blue')
axes[0].set_title('AAPL Closing Price Over Time', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Price ($)')
axes[0].grid(True, alpha=0.3)

# 2. Volume
axes[1].bar(aapl.index, aapl['Volume'], color='gray', alpha=0.5)
axes[1].set_title('Trading Volume', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Volume')
axes[1].grid(True, alpha=0.3)

# 3. Daily returns
axes[2].plot(aapl.index, daily_returns*100, linewidth=1, color='green', alpha=0.7)
axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[2].set_title('Daily Returns (%)', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Return (%)')
axes[2].set_xlabel('Date')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/aapl_exploration.png', dpi=150)
print("✓ Saved: results/plots/aapl_exploration.png")

# Recent price detail (last 90 days)
recent = aapl.tail(90)
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(recent.index, recent['Close'], linewidth=2, color='blue', label='Close')
ax.plot(recent.index, recent['High'], linewidth=1, color='green', alpha=0.5, label='High')
ax.plot(recent.index, recent['Low'], linewidth=1, color='red', alpha=0.5, label='Low')
ax.fill_between(recent.index, recent['Low'], recent['High'], alpha=0.1, color='gray')
ax.set_title('AAPL - Last 90 Days Detail', fontsize=14, fontweight='bold')
ax.set_ylabel('Price ($)')
ax.set_xlabel('Date')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/aapl_recent.png', dpi=150)
print("✓ Saved: results/plots/aapl_recent.png")

print("\n✓ Phase 1.2 complete! Data explored (no changes made).")