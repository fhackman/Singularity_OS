# ⚡ SINGULARITY CAPITAL OS

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MetaTrader5](https://img.shields.io/badge/MetaTrader5-supported-green.svg)](https://www.metatrader5.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Self-Evolving Capital Intelligence System** — A professional-grade algorithmic trading framework with Bayesian edge estimation, adaptive risk management, and real-time MT5 integration.

![Singularity Capital OS](https://img.shields.io/badge/Version-1.0.0-brightgreen?style=for-the-badge)

---

## 🎯 Overview

Singularity Capital OS is an advanced trading intelligence system that combines:

- **Bayesian Edge Estimation** — Confidence-weighted win rate and expectancy calculations
- **Monte Carlo Simulation** — Risk-of-ruin and drawdown probability analysis
- **Regime Detection** — Adaptive market state identification (Trend/Chop/Panic)
- **Dynamic Risk Management** — Kelly Criterion with regime and drawdown adjustments
- **Kill Switch Protection** — Automatic trading halt on adverse conditions
- **Multi-Agent Portfolio** — Genetic algorithm-optimized agent allocation
- **MT5 Real-Time Data** — Live market prices and trade execution

---

## 🖥️ Screenshots

### Hacker-Themed GUI

```
┌────────────────────────────────────────────────────────────────────┐
│  ⚡ SINGULARITY CAPITAL OS           14:35:00           ● MT5 OK   │
├────────────┬─────────────────────────────────┬─────────────────────┤
│  MT5 CONN  │  XAUUSD    EURUSD    GBPUSD    │  [TRADING SIGNALS]  │
│  Balance:  │  2650.15   1.0852    1.2701    │  ✅ CLEAR TO TRADE  │
│  $10,000   │  ▲ +15     ▼ -2      = 0       │  Risk: 0.45%        │
│            │                                 │                     │
│  [ANALYZE] │  Win Rate: 52.3%               │  [SYSTEM TERMINAL]  │
│            │  Edge: 0.32R                    │  > Analysis done    │
└────────────┴─────────────────────────────────┴─────────────────────┘
│  01アイウエオカキクケコサシス...          v1.0.0 HACKER EDITION   │
└────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/singularity-capital-os.git
cd singularity-capital-os

# Install dependencies automatically
python launcher.py --install
```

### Manual Install

```bash
pip install -r requirements.txt
```

### Using setup.py

```bash
pip install .
# Then run from anywhere:
singularity
```

---

## 🚀 Quick Start

### Launch the Hacker GUI (Recommended)

```bash
python singularity_gui.py
```

### Or use the Interactive Launcher

```bash
python launcher.py
```

### Command Line Options

```bash
python launcher.py --gui        # Launch Hacker GUI directly
python launcher.py --demo       # Run demo mode
python launcher.py --backtest   # Run backtest mode
python launcher.py --install    # Install dependencies only
```

---

## 🏗️ Project Structure

```
Singularity Capital OS/
├── 🖥️ singularity_gui.py      # Hacker-themed GUI with MT5 integration
├── 🔌 mt5_connector.py         # MetaTrader 5 real-time data connector
├── 🧠 singularity_core.py      # Core analysis engine
├── 🧬 multi_agent_system.py    # Multi-agent portfolio optimizer
├── 📊 Complete_Trading_System.py  # CLI trading system
├── 💎 xauusd_agent.py          # XAUUSD specialized agent
├── 🚀 launcher.py              # Unified launcher with menu
├── 📋 requirements.txt         # Python dependencies
├── ⚙️ setup.py                 # Package installer
└── 📈 singularity_multi.pine   # Pine Script (TradingView)
```

---

## 🧠 Core Components

### 1. Bayesian Edge Estimation

Calculates win rate and expectancy with confidence bounds using Beta distribution:

```python
from singularity_core import BayesianEdge

edge = BayesianEdge()
results = edge.update(returns_array)
print(f"Win Rate: {results['win_rate_mean']:.1%}")
print(f"Edge (Lower Bound): {results['expectancy_lb']:.4f}R")
```

### 2. Monte Carlo Risk Analysis

Simulates 10,000 equity curves to estimate:

- Maximum drawdown (95th percentile)
- Longest losing streak
- Risk of ruin probability

### 3. Regime Detection

Identifies market states based on rolling volatility:

- **Regime 0** — Low volatility (Trend)
- **Regime 1** — Normal volatility (Chop)
- **Regime 2** — High volatility (Panic)

### 4. Dynamic Risk Calculator

Adjusts position sizing using:

- Half-Kelly criterion
- Regime-based multipliers
- Drawdown protection factor

### 5. Kill Switch

Automatically halts trading when:

- 20%+ drawdown detected
- Losing streak exceeds threshold
- Negative edge detected

---

## 🔌 MT5 Integration

### Features

| Feature          | Description                       |
| ---------------- | --------------------------------- |
| Real-time Prices | Bid/Ask with spread               |
| Historical Data  | OHLCV for any timeframe           |
| Account Info     | Balance, equity, margin, leverage |
| Trade Execution  | Buy/Sell with SL/TP               |
| Symbol Info      | Contract specs, spreads           |
| Simulation Mode  | Works without MT5 installed       |

### Usage

```python
from mt5_connector import get_connector

connector = get_connector()
success, message = connector.connect()

# Get real-time price
price = connector.get_current_price("XAUUSD")
print(f"XAUUSD: {price['bid']}")

# Get historical data
df = connector.get_historical_data("EURUSD", "M15", 500)
```

---

## 📊 Supported Timeframes

| Code | Timeframe  |
| ---- | ---------- |
| M1   | 1 Minute   |
| M5   | 5 Minutes  |
| M15  | 15 Minutes |
| M30  | 30 Minutes |
| H1   | 1 Hour     |
| H4   | 4 Hours    |
| D1   | Daily      |
| W1   | Weekly     |

---

## 🧬 Multi-Agent System

The genetic algorithm optimizes agent allocation across markets:

```python
from multi_agent_system import EvolutionEngine, PortfolioAllocator

# Initialize agents for multiple symbols
symbols = ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY']
engine = EvolutionEngine(symbols)

# Evolve for 50 generations
best_agents = engine.evolve(generations=50)

# Get optimized allocations
allocator = PortfolioAllocator(best_agents)
allocations = allocator.get_allocations()
```

---

## ⚙️ Configuration

### Risk Parameters

```python
# In singularity_core.py
core = SingularityCore(
    base_risk=0.01,      # 1% base risk per trade
    max_drawdown=0.20,   # 20% max drawdown trigger
    kelly_fraction=0.5,  # Half-Kelly sizing
)
```

### GUI Settings

- **Auto Refresh** — Automatically update prices
- **Sound Alerts** — Audio notifications for signals

---

## 📋 Requirements

- Python 3.8+
- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- customtkinter >= 5.0.0
- MetaTrader5 (optional, for live data)

---

## 🔧 Troubleshooting

### MT5 Not Connecting

1. Ensure MT5 terminal is running
2. Enable "Allow DLL imports" in MT5 settings
3. Try closing and reopening MT5

### Missing Dependencies

```bash
python launcher.py --install
```

### GUI Not Launching

```bash
pip install customtkinter --upgrade
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📧 Contact

- **Author**: Singularity Trading Systems
- **Email**: support@singularity-capital.com
- **GitHub**: [github.com/singularity-capital](https://github.com/singularity-capital)

---

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading financial instruments involves substantial risk of loss and is not suitable for every investor. Past performance is not indicative of future results. The developers are not responsible for any financial losses incurred from using this software.

---

<p align="center">
  <b>⚡ SINGULARITY CAPITAL OS ⚡</b><br>
  <i>Self-Evolving Capital Intelligence</i>
</p>
