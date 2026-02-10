### Phase 2: Feature Engineering for Time Series
    - What We're Building
        - Transform raw prices into predictive features:
        - Input: Daily stock prices
        - Output: Features that capture patterns and trends
        - Goal: Give model more signal to predict tomorrow's price

```bash
$ python3 src/data/04_feature_engineering.py
```
============================================================
FEATURE ENGINEERING
============================================================

============================================================
Processing AAPL
============================================================
Loaded: (1254, 6)

1. Adding lag features...
   Added lag features, shape: (1254, 18)
2. Adding rolling features...
   Added rolling features, shape: (1254, 26)
3. Adding technical indicators...
   Added technical indicators, shape: (1254, 33)
4. Adding time features...
   Added time features, shape: (1254, 39)
5. Adding target (tomorrow's price)...
   Added target, shape: (1254, 40)

Feature columns (40 total):
['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_5', 'Close_lag_10', 'Close_lag_30', 'Volume_lag_1', 'Volume_lag_2', 'Volume_lag_3', 'Volume_lag_5', 'Volume_lag_10', 'Volume_lag_30', 'Close_MA7', 'Close_std7', 'Close_MA14', 'Close_std14', 'Close_MA30', 'Close_std30', 'Close_MA60', 'Close_std60', 'RSI', 'MACD', 'MACD_signal', 'BB_middle', 'BB_std', 'BB_upper', 'BB_lower', 'day_of_week', 'day_of_month', 'month', 'quarter', 'is_month_start', 'is_month_end', 'target']

Sample data (last 5 rows):
                           Open        High         Low  ...  is_month_start  is_month_end      target
Date                                                     ...
2026-02-02 05:00:00  260.029999  270.489990  259.209991  ...               0             0  269.480011
2026-02-03 05:00:00  269.200012  271.880005  267.609985  ...               0             0  276.489990
2026-02-04 05:00:00  272.290009  278.950012  272.290009  ...               0             0  275.910004
2026-02-05 05:00:00  278.130005  279.500000  273.230011  ...               0             0  278.119995
2026-02-06 05:00:00  277.119995  280.910004  276.929993  ...               0             0         NaN

[5 rows x 40 columns]

NaN values: 407
NaN rows will be dropped before training

✓ Saved: data/processed/AAPL_featured.csv

============================================================
Processing TSLA
============================================================
Loaded: (1254, 6)

1. Adding lag features...
   Added lag features, shape: (1254, 18)
2. Adding rolling features...
   Added rolling features, shape: (1254, 26)
3. Adding technical indicators...
   Added technical indicators, shape: (1254, 33)
4. Adding time features...
   Added time features, shape: (1254, 39)
5. Adding target (tomorrow's price)...
   Added target, shape: (1254, 40)

Feature columns (40 total):
['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_5', 'Close_lag_10', 'Close_lag_30', 'Volume_lag_1', 'Volume_lag_2', 'Volume_lag_3', 'Volume_lag_5', 'Volume_lag_10', 'Volume_lag_30', 'Close_MA7', 'Close_std7', 'Close_MA14', 'Close_std14', 'Close_MA30', 'Close_std30', 'Close_MA60', 'Close_std60', 'RSI', 'MACD', 'MACD_signal', 'BB_middle', 'BB_std', 'BB_upper', 'BB_lower', 'day_of_week', 'day_of_month', 'month', 'quarter', 'is_month_start', 'is_month_end', 'target']

Sample data (last 5 rows):
                           Open        High         Low  ...  is_month_start  is_month_end      target
Date                                                     ...
2026-02-02 05:00:00  421.290009  427.149994  414.500000  ...               0             0  421.959991
2026-02-03 05:00:00  424.269989  428.559998  413.690002  ...               0             0  406.010010
2026-02-04 05:00:00  420.459991  423.899994  399.179993  ...               0             0  397.209991
2026-02-05 05:00:00  397.019989  402.100006  387.529999  ...               0             0  411.109985
2026-02-06 05:00:00  400.869995  414.549988  397.750000  ...               0             0         NaN

[5 rows x 40 columns]

NaN values: 407
NaN rows will be dropped before training

✓ Saved: data/processed/TSLA_featured.csv

============================================================
Processing GOOGL
============================================================
Loaded: (1254, 6)

1. Adding lag features...
   Added lag features, shape: (1254, 18)
2. Adding rolling features...
   Added rolling features, shape: (1254, 26)
3. Adding technical indicators...
   Added technical indicators, shape: (1254, 33)
4. Adding time features...
   Added time features, shape: (1254, 39)
5. Adding target (tomorrow's price)...
   Added target, shape: (1254, 40)

Feature columns (40 total):
['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_5', 'Close_lag_10', 'Close_lag_30', 'Volume_lag_1', 'Volume_lag_2', 'Volume_lag_3', 'Volume_lag_5', 'Volume_lag_10', 'Volume_lag_30', 'Close_MA7', 'Close_std7', 'Close_MA14', 'Close_std14', 'Close_MA30', 'Close_std30', 'Close_MA60', 'Close_std60', 'RSI', 'MACD', 'MACD_signal', 'BB_middle', 'BB_std', 'BB_upper', 'BB_lower', 'day_of_week', 'day_of_month', 'month', 'quarter', 'is_month_start', 'is_month_end', 'target']

Sample data (last 5 rows):
                           Open        High         Low  ...  is_month_start  is_month_end      target
Date                                                     ...
2026-02-02 05:00:00  336.220001  344.829987  335.630005  ...               0             0  339.709991
2026-02-03 05:00:00  347.339996  349.000000  337.470001  ...               0             0  333.040009
2026-02-04 05:00:00  342.959991  343.309998  328.519989  ...               0             0  331.250000
2026-02-05 05:00:00  312.220001  332.690002  306.459991  ...               0             0  322.859985
2026-02-06 05:00:00  327.179993  330.380005  319.920013  ...               0             0         NaN

[5 rows x 40 columns]

NaN values: 407
NaN rows will be dropped before training

✓ Saved: data/processed/GOOGL_featured.csv

============================================================
FEATURE ANALYSIS
============================================================

Total features: 40
Total rows: 1254
Rows with NaN: 60
Clean rows (usable for training): 1194

Feature categories:
  Lag features: 12
  Moving average features: 11
  Technical indicators: 6
  Time features: 6
  Original features: 7 (Open, High, Low, Close, Volume, Daily_Return, target)

============================================================
VISUALIZING KEY FEATURES
============================================================
✓ Saved: results/plots/feature_engineering.png

✓ Phase 2 complete! Features engineered for all stocks.


###
```bash
python3 src/data/04_feature_engineering.py
```
============================================================
FEATURE ENGINEERING
============================================================
============================================================
Processing AAPL
============================================================

Phase 2 Complete!

Feature Engineering Summary
✅ Features Created (40 total per stock)
1. Lag Features (12): Yesterday's values, last week's values

Close_lag_1, Close_lag_2, Close_lag_3, Close_lag_5, Close_lag_10, Close_lag_30
Volume_lag_1, Volume_lag_2, Volume_lag_3, Volume_lag_5, Volume_lag_10, Volume_lag_30
2. Moving Averages (8): Smooth trends

Close_MA7, Close_MA14, Close_MA30, Close_MA60 (7, 14, 30, 60-day averages)
Close_std7, Close_std14, Close_std30, Close_std60 (volatility measures)
3. Technical Indicators (6): Trading patterns

RSI - Overbought/oversold (0-100)
MACD, MACD_signal - Momentum
BB_upper, BB_middle, BB_lower - Bollinger Bands (volatility)
4. Time Features (6): Calendar patterns

day_of_week, day_of_month, month, quarter
is_month_start, is_month_end
5. Target (1): Tomorrow's price

target - What we're trying to predict
Data Ready for Modeling
Total rows: 1,254
Clean rows: 1,194 (60 dropped due to NaN from lag/rolling calculations)
Features: 40
All 3 stocks processed: AAPL, TSLA, GOOGL
What We Learned
✅ Proper workflow: Download → Explore → Clean → Feature Engineering
✅ Fixed data issues at the source (timezone handling in cleaning, not feature engineering)
✅ Created predictive features from raw price data
✅ Visualizations saved showing moving averages, RSI, MACD

### Understanding Time Series Features

Category 1: Lag Features (Historical Values)
What they are: Previous day's values (looking backward in time)
Close Price Lags:
pythonClose_lag_1  # Yesterday's closing price
Close_lag_2  # 2 days ago
Close_lag_3  # 3 days ago
Close_lag_5  # Last week (5 trading days)
Close_lag_10 # Two weeks ago
Close_lag_30 # A month ago
Why they work:

Stock prices have momentum - if it went up yesterday, might go up today
Mean reversion - if it's been going up for days, might correct
Recent history predicts near future

Analogy:
Like weather - if it rained yesterday, higher chance of rain today.
Volume Lags:
pythonVolume_lag_1, Volume_lag_2, etc.
Why volume matters:

High volume + price increase = strong trend (many buyers)
High volume + price decrease = panic selling
Low volume = weak signal


Category 2: Moving Averages (Trend Smoothing)
What they are: Average price over N days - smooths out daily noise
Simple Moving Averages (SMA):
pythonClose_MA7   # Average of last 7 days
Close_MA14  # Average of last 14 days
Close_MA30  # Average of last 30 days
Close_MA60  # Average of last 60 days
```

**Calculation example (MA7):**
```
Day 100 price: $150
Day 101 price: $152
Day 102 price: $151
Day 103 price: $153
Day 104 price: $154
Day 105 price: $152
Day 106 price: $155

MA7 on Day 106 = (150+152+151+153+154+152+155) / 7 = $152.43
Why they're useful:

Trend identification: Price above MA = uptrend, below = downtrend
Support/Resistance: Price often bounces off moving averages
Crossovers: When MA7 crosses above MA30 = bullish signal

Common patterns traders watch:

Golden Cross: MA50 crosses above MA200 → BUY signal
Death Cross: MA50 crosses below MA200 → SELL signal

Rolling Standard Deviation (Volatility):
pythonClose_std7   # How much price varied in last 7 days
Close_std14  # Volatility over 14 days
Close_std30, Close_std60
```

**What it measures:**
- High std = Wild swings (risky, unpredictable)
- Low std = Stable (boring, predictable)

**Example:**
```
Stock A: $100, $101, $99, $100, $101 → Low volatility (±1%)
Stock B: $100, $110, $90, $105, $95  → High volatility (±10%)
```

---

## **Category 3: Technical Indicators (Trading Signals)**

These are formulas traders have used for decades:

### **1. RSI (Relative Strength Index)**

**What it is:** Momentum indicator showing overbought/oversold conditions

**Range:** 0 to 100

**Formula (simplified):**
```
RSI = 100 - (100 / (1 + RS))
RS = Average Gain / Average Loss (over 14 days)
```

**Interpretation:**
- **RSI > 70:** Overbought (might drop soon) 🔴
- **RSI < 30:** Oversold (might rise soon) 🟢
- **RSI = 50:** Neutral

**Example:**
```
Stock goes up 10 days in a row → RSI = 95 → "Too high, expect correction"
Stock drops 10 days in a row → RSI = 15 → "Too low, expect bounce"
Real-world:
Like a rubber band - stretched too far in one direction, it snaps back.

2. MACD (Moving Average Convergence Divergence)
What it is: Shows relationship between two moving averages (momentum)
Components:
pythonMACD = EMA(12) - EMA(26)           # Fast - Slow
MACD_signal = EMA(MACD, 9)         # Signal line
```
*(EMA = Exponential Moving Average - gives more weight to recent prices)*

**Signals:**
- **MACD crosses above signal:** Bullish (BUY) ✓
- **MACD crosses below signal:** Bearish (SELL) ✗
- **MACD above 0:** Uptrend
- **MACD below 0:** Downtrend

**Visualization:**
```
Price going up:
MACD ↗️ (positive and rising)
Signal ↗️ (following)

Price reversing:
MACD ↘️ (crosses below signal) ← SELL SIGNAL

3. Bollinger Bands
What they are: Volatility bands around price (like a tunnel)
Components:
pythonBB_middle = MA(20)                    # 20-day moving average
BB_std = StdDev(20)                   # 20-day standard deviation
BB_upper = BB_middle + (2 × BB_std)   # Upper band
BB_lower = BB_middle - (2 × BB_std)   # Lower band
```

**Interpretation:**
- **Price hits upper band:** Expensive, might pull back
- **Price hits lower band:** Cheap, might bounce
- **Bands narrow:** Low volatility (calm before storm)
- **Bands widen:** High volatility (big moves happening)

**Visual example:**
```
$160 ← BB_upper (expensive)
$150 ← BB_middle (fair value)
$140 ← BB_lower (cheap)

Price at $159? → Near upper band → Consider selling
Price at $141? → Near lower band → Consider buying
Statistical note:
95% of price action stays within 2 standard deviations (the bands)

Category 4: Time Features (Calendar Patterns)
What they are: Extract date information - markets have calendar patterns
Day of Week:
pythonday_of_week  # 0=Monday, 1=Tuesday, ..., 4=Friday
Known patterns:

Monday Effect: Stocks often drop on Mondays (weekend news)
Friday Effect: Often rises (traders close short positions)

Day of Month:
pythonday_of_month  # 1-31
Known patterns:

End of month: Mutual funds rebalance → higher volume
Start of month: People invest paychecks → buying pressure

Month:
pythonmonth  # 1=Jan, 2=Feb, ..., 12=Dec
Known patterns:

January Effect: Small caps rise in January (tax-loss harvesting reversal)
September: Historically worst month for stocks
December: Santa Rally (year-end optimism)

Quarter:
pythonquarter  # 1, 2, 3, or 4
Why it matters:

Earnings season: Companies report quarterly
Q4 = holiday shopping (retail stocks rise)

Month Start/End Flags:
pythonis_month_start  # 1 if first trading day, else 0
is_month_end    # 1 if last trading day, else 0
Why it matters:

Window dressing (funds buy winners to show in reports)
Increased trading volume


Category 5: Target (What We Predict)
pythontarget  # Tomorrow's closing price
Created by:
pythondf['target'] = df['Close'].shift(-1)  # Shift backward by 1 day
```

**Example:**
```
Date        Close    target
2024-01-01  $150     $152  ← Tomorrow's price
2024-01-02  $152     $151
2024-01-03  $151     $155

Common Technical Indicators (Industry Standard)
What we used:
✅ RSI
✅ MACD
✅ Bollinger Bands
✅ Moving Averages (SMA)
Other popular ones (we didn't add but could):
1. Stochastic Oscillator

Similar to RSI, shows momentum
Range: 0-100

2. ADX (Average Directional Index)

Measures trend strength (not direction)
Strong trend = ADX > 25

3. ATR (Average True Range)

Measures volatility (how much stock moves)
High ATR = volatile

4. OBV (On-Balance Volume)

Volume-based momentum
Rising OBV + rising price = strong trend

5. Fibonacci Retracements

Support/resistance levels
Based on 23.6%, 38.2%, 61.8% ratios

6. Ichimoku Cloud

Japanese indicator showing support/resistance
Complex but comprehensive


Why These Features Work (The Theory)
1. Technical Analysis Assumption:
"History repeats itself - patterns recur"
Traders believe:

Past price movements predict future
Charts show human psychology (fear/greed)
Self-fulfilling prophecy (if everyone watches MA crossovers, they matter)

2. Market Efficiency Debate:
Efficient Market Hypothesis (EMH):

All info is already in the price
Technical analysis doesn't work
Random walk (can't predict)

Technical Analysts' View:

Markets aren't perfectly efficient
Behavioral patterns exist
Short-term patterns are exploitable

Reality:

Simple patterns mostly don't work (too many people know them)
But can provide small edge when combined with other factors
Works better for short-term (days) than long-term (years)


Feature Engineering Best Practices
What makes a good feature:
✅ Lag features: Recent history predicts future
✅ Domain knowledge: RSI, MACD used by real traders
✅ Multiple timeframes: Short (7d), medium (30d), long (60d) trends
✅ Interactions: Price + volume together = stronger signal
What to avoid:
❌ Look-ahead bias: Using future information (target leakage)
❌ Too many features: Overfitting, slow training
❌ Correlated features: MA7 and MA14 tell similar story

Our Feature Set Summary
CategoryCountPurposeOriginal6Raw OHLCV dataLag12Recent historyMoving Avg8Trend smoothingTechnical6Trading signalsTime6Calendar patternsTarget1Tomorrow's priceTOTAL40Comprehensive feature set

Key Takeaway
Time series features are fundamentally different from tabular:
Tabular (Churn):

Each row independent
Features describe current state

Time Series (Stocks):

Each row connected to past
Features capture temporal patterns
Order matters critically!