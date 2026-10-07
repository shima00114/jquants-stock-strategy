# Japanese Equity Quantitative Trading Strategy

This repository contains the source code and documentation for a quantitative stock trading strategy targeting Japanese equities using J-Quants market data.

## 📈 Strategy Overview

Starting from an initial negative Sharpe ratio, this model was iteratively refined into a market-neutral mean-reversion strategy. 
By focusing on **transaction cost reduction** and **systematic risk removal**, the strategy achieved a positive Sharpe ratio.

### Key Strategy Mechanics

1. **Short-Term Residual Return Reversal**:
   - Captures short-term overreaction (mean-reversion) using a 10-day cumulative stock price return.
2. **Market-Neutral Alpha (Beta-Adjusted Excess Return)**:
   - Strips away systematic market risk by subtracting beta-adjusted TOPIX returns from individual stock returns, isolating pure stock-specific alpha.
3. **Turnover Reduction via Smoothing**:
   - Applies a 40-day Simple Moving Average (SMA) filter to smooth raw signals and drastically cut down daily portfolio turnover. This effectively neutralizes the performance drag caused by 0.1% round-trip transaction costs.
4. **Cross-Sectional Rank Normalization**:
   - Normalizes raw predictions daily using percentile ranks (`rank(pct=True) - 0.5`) to equalize long/short market exposures and stabilize volatility.

---

## 🛠️ Repository Structure

```text
.
├── README.md        # Project documentation
└── submission.py    # Main script containing the predict() entry point
