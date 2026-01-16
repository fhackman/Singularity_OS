"""
SINGULARITY CAPITAL OS - Complete Integration
Ready-to-use trading system with all components

Usage:
    python complete_trading_system.py --mode backtest --symbol XAUUSD
    python complete_trading_system.py --mode live --symbols XAUUSD,BTCUSD,USOIL,EURUSD
"""

# ============================================================================
# AUTO-INSTALL DEPENDENCIES
# ============================================================================
import subprocess
import sys

def _ensure_dependencies():
    """Automatically install missing dependencies."""
    required = {
        'numpy': 'numpy>=1.21.0',
        'pandas': 'pandas>=1.3.0',
        'scipy': 'scipy>=1.7.0',
    }
    
    missing = []
    for package, pip_spec in required.items():
        try:
            __import__(package)
        except ImportError:
            missing.append((package, pip_spec))
    
    if missing:
        print("📦 Installing missing dependencies...")
        for package, pip_spec in missing:
            print(f"   ⚙️  Installing {package}...")
            try:
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', pip_spec, '--quiet'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                print(f"   ✅ {package} installed successfully!")
            except subprocess.CalledProcessError:
                print(f"   ❌ Failed to install {package}. Please run: pip install {pip_spec}")
                sys.exit(1)
        print("✅ All dependencies installed!\n")

_ensure_dependencies()

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional
import argparse


# ============================================================================
# CUSTOM JSON ENCODER FOR NUMPY TYPES
# ============================================================================
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ============================================================================
# CORE ENGINE (Embedded)
# ============================================================================

class BayesianEdge:
    def __init__(self):
        self.alpha = 1.0
        self.beta = 1.0
    
    def update(self, returns: np.ndarray) -> Dict:
        wins = np.sum(returns > 0)
        losses = np.sum(returns < 0)
        self.alpha += wins
        self.beta += losses
        
        from scipy import stats
        win_rate_mean = self.alpha / (self.alpha + self.beta)
        win_rate_lb = stats.beta.ppf(0.10, self.alpha, self.beta)
        
        avg_win = np.mean(returns[returns > 0]) if wins > 0 else 0
        avg_loss = abs(np.mean(returns[returns < 0])) if losses > 0 else 0
        
        expectancy_lb = win_rate_lb * avg_win - (1 - win_rate_lb) * avg_loss
        
        return {
            'win_rate_mean': win_rate_mean,
            'win_rate_lb': win_rate_lb,
            'expectancy_lb': expectancy_lb,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }


class RegimeDetector:
    def detect(self, returns: np.ndarray) -> int:
        if len(returns) < 20:
            return 1
        
        vol = pd.Series(returns).rolling(20).std().dropna()
        if len(vol) == 0:
            return 1
        
        q33, q66 = np.percentile(vol, [33, 66])
        current_vol = vol.iloc[-1]
        
        if current_vol < q33:
            return 0  # Low variance
        elif current_vol < q66:
            return 1  # Mid
        else:
            return 2  # High variance


class DynamicRisk:
    def calculate(self, edge_lb: float, regime: int, current_dd: float, worst_dd: float) -> float:
        base_risk = 0.01
        
        if edge_lb <= 0:
            return 0.0
        
        kelly = edge_lb / 1.0
        regime_factor = {0: 1.2, 1: 1.0, 2: 0.3}[regime]
        dd_factor = 0.5 if abs(current_dd) > abs(worst_dd) else 1.0
        
        return base_risk * kelly * 0.3 * regime_factor * dd_factor


# ============================================================================
# MARKET AGENT
# ============================================================================

class MarketAgent:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.returns = []
        self.bayesian = BayesianEdge()
        self.regime_detector = RegimeDetector()
        self.risk_calc = DynamicRisk()
        self.fitness = 0.0
        self.alive = True
    
    def update(self, new_returns: List[float]):
        self.returns.extend(new_returns)
        self._calculate_fitness()
    
    def _calculate_fitness(self):
        if len(self.returns) < 10:
            return
        
        r = np.array(self.returns)
        log_growth = np.mean(np.log1p(r))
        
        cum = np.cumsum(r)
        running_max = np.maximum.accumulate(cum)
        dd = cum - running_max
        max_dd = abs(np.min(dd))
        
        tail = abs(np.percentile(r, 5))
        
        self.fitness = log_growth - 0.5 * max_dd - 0.3 * tail
    
    def analyze(self) -> Dict:
        if len(self.returns) < 10:
            return {'error': 'Insufficient data'}
        
        r = np.array(self.returns)
        edge = self.bayesian.update(r)
        regime = self.regime_detector.detect(r)
        
        cum = np.cumsum(r)
        running_max = np.maximum.accumulate(cum)
        dd = cum - running_max
        current_dd = dd[-1]
        worst_dd = np.min(dd)
        
        risk = self.risk_calc.calculate(edge['expectancy_lb'], regime, current_dd, worst_dd)
        
        killed = edge['expectancy_lb'] <= 0 or regime == 2
        
        return {
            'symbol': self.symbol,
            'fitness': self.fitness,
            'edge': edge,
            'regime': regime,
            'risk_pct': risk,
            'current_dd': current_dd,
            'killed': killed
        }


# ============================================================================
# PORTFOLIO SYSTEM
# ============================================================================

class PortfolioSystem:
    def __init__(self, symbols: List[str]):
        self.agents = {s: MarketAgent(s) for s in symbols}
    
    def update_all(self, returns_dict: Dict[str, List[float]]):
        for symbol, returns in returns_dict.items():
            if symbol in self.agents:
                self.agents[symbol].update(returns)
    
    def get_allocations(self) -> Dict[str, float]:
        fitnesses = {s: max(a.fitness, 0) for s, a in self.agents.items() if a.alive}
        
        if not fitnesses or sum(fitnesses.values()) == 0:
            return {s: 1.0/len(self.agents) for s in self.agents}
        
        total = sum(fitnesses.values())
        return {s: f/total for s, f in fitnesses.items()}
    
    def get_dashboard(self) -> Dict:
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'agents': {},
            'allocations': self.get_allocations()
        }
        
        for symbol, agent in self.agents.items():
            analysis = agent.analyze()
            dashboard['agents'][symbol] = analysis
        
        return dashboard


# ============================================================================
# CSV DATA LOADER
# ============================================================================

def load_csv_returns(filepath: str) -> pd.DataFrame:
    """Load trading returns from CSV (date, symbol, R format)"""
    try:
        df = pd.read_csv(filepath)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return pd.DataFrame()


def generate_sample_data(symbols: List[str], n_trades: int = 100) -> pd.DataFrame:
    """Generate sample trading data for testing"""
    data = []
    
    # Characteristics by symbol
    params = {
        'XAUUSD': {'mean': 0.30, 'std': 1.42},
        'BTCUSD': {'mean': 0.25, 'std': 2.10},
        'USOIL': {'mean': 0.15, 'std': 1.85},
        'EURUSD': {'mean': 0.18, 'std': 1.25}
    }
    
    for symbol in symbols:
        p = params.get(symbol, {'mean': 0.2, 'std': 1.5})
        returns = np.random.normal(p['mean'], p['std'], n_trades)
        
        for i, r in enumerate(returns):
            data.append({
                'date': pd.Timestamp('2025-01-01') + pd.Timedelta(days=i),
                'symbol': symbol,
                'R': r
            })
    
    return pd.DataFrame(data)


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def run_backtest(df: pd.DataFrame, symbols: List[str]):
    """Run backtest on historical data"""
    print("\n" + "="*80)
    print("🔄 RUNNING BACKTEST")
    print("="*80)
    
    portfolio = PortfolioSystem(symbols)
    
    # Group by symbol
    results = []
    
    for symbol in symbols:
        symbol_data = df[df['symbol'] == symbol]['R'].values
        
        if len(symbol_data) < 10:
            print(f"⚠️ {symbol}: Insufficient data ({len(symbol_data)} trades)")
            continue
        
        # Update agent
        portfolio.agents[symbol].update(list(symbol_data))
        
        # Get analysis
        analysis = portfolio.agents[symbol].analyze()
        results.append(analysis)
    
    # Final dashboard
    dashboard = portfolio.get_dashboard()
    
    print("\n📊 BACKTEST RESULTS")
    print("-"*80)
    
    for symbol in symbols:
        if symbol in dashboard['agents']:
            info = dashboard['agents'][symbol]
            print(f"\n{symbol}:")
            print(f"  Fitness: {info.get('fitness', 0):.4f}")
            if 'edge' in info:
                print(f"  Edge (LB): {info['edge']['expectancy_lb']:.4f}R")
                print(f"  Win Rate: {info['edge']['win_rate_mean']*100:.1f}%")
            print(f"  Risk: {info.get('risk_pct', 0)*100:.4f}%")
            print(f"  Killed: {info.get('killed', False)}")
    
    print("\n💰 ALLOCATIONS:")
    for symbol, weight in dashboard['allocations'].items():
        print(f"  {symbol}: {weight*100:.2f}%")
    
    # Export results
    with open('backtest_results.json', 'w') as f:
        json.dump(dashboard, f, indent=2, cls=NumpyEncoder)
    print("\n✅ Results saved to: backtest_results.json")


# ============================================================================
# LIVE MONITORING
# ============================================================================

def run_live_monitor(symbols: List[str]):
    """Monitor live trading (simulated)"""
    print("\n" + "="*80)
    print("🔴 LIVE MONITORING MODE")
    print("="*80)
    
    portfolio = PortfolioSystem(symbols)
    iteration = 0
    
    print("\nGenerating simulated live data...")
    print("(In production, connect to your broker API)\n")
    
    while iteration < 5:  # Demo: 5 iterations
        iteration += 1
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}")
        print(f"{'='*80}")
        
        # Simulate new returns
        new_returns = {}
        for symbol in symbols:
            # Random 5-10 new trades
            n = np.random.randint(5, 11)
            params = {
                'XAUUSD': {'mean': 0.30, 'std': 1.42},
                'BTCUSD': {'mean': 0.25, 'std': 2.10},
                'USOIL': {'mean': 0.15, 'std': 1.85},
                'EURUSD': {'mean': 0.18, 'std': 1.25}
            }
            p = params.get(symbol, {'mean': 0.2, 'std': 1.5})
            new_returns[symbol] = list(np.random.normal(p['mean'], p['std'], n))
        
        # Update portfolio
        portfolio.update_all(new_returns)
        
        # Get dashboard
        dashboard = portfolio.get_dashboard()
        
        # Display
        print("\n📊 CURRENT STATUS:")
        for symbol in symbols:
            if symbol in dashboard['agents']:
                info = dashboard['agents'][symbol]
                status = "🔴 KILLED" if info.get('killed', False) else "✅ ACTIVE"
                print(f"{symbol}: {status} | Fitness={info.get('fitness', 0):.4f} | Risk={info.get('risk_pct', 0)*100:.4f}%")
        
        print(f"\n💰 Allocations: {dashboard['allocations']}")
        
        import time
        time.sleep(1)
    
    print("\n✅ Live monitoring demo complete")


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Singularity Capital OS')
    parser.add_argument('--mode', choices=['backtest', 'live', 'demo'], default='demo',
                       help='Operating mode')
    parser.add_argument('--symbols', type=str, default='XAUUSD,BTCUSD,USOIL,EURUSD',
                       help='Comma-separated symbols')
    parser.add_argument('--csv', type=str, help='Path to CSV file (for backtest)')
    
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    print("="*80)
    print("🧠 SINGULARITY CAPITAL OS")
    print("Self-Evolving Capital Intelligence")
    print("="*80)
    print(f"\nMode: {args.mode.upper()}")
    print(f"Symbols: {', '.join(symbols)}")
    
    if args.mode == 'backtest':
        if args.csv:
            df = load_csv_returns(args.csv)
        else:
            print("\n⚠️ No CSV provided, generating sample data...")
            df = generate_sample_data(symbols, n_trades=200)
        
        run_backtest(df, symbols)
    
    elif args.mode == 'live':
        run_live_monitor(symbols)
    
    else:  # demo
        print("\n🎯 DEMO MODE - Generating sample data...\n")
        df = generate_sample_data(symbols, n_trades=100)
        run_backtest(df, symbols)


if __name__ == "__main__":
    main()