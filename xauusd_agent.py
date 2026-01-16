"""
XAUUSD Specialized Agent
Optimized for Gold trading characteristics

Characteristics:
- High noise, stop hunting common
- Mean-reverting in certain regimes
- Sensitive to USD news (NFP, CPI, FOMC)
- Best with tight stops and quick exits
"""

# ============================================================================
# AUTO-INSTALL DEPENDENCIES
# ============================================================================
import subprocess
import sys as _sys

def _ensure_dependencies():
    """Automatically install missing dependencies."""
    required = {
        'numpy': 'numpy>=1.21.0',
        'pandas': 'pandas>=1.3.0',
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
                    [_sys.executable, '-m', 'pip', 'install', pip_spec, '--quiet'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                print(f"   ✅ {package} installed successfully!")
            except subprocess.CalledProcessError:
                print(f"   ❌ Failed to install {package}. Please run: pip install {pip_spec}")
                _sys.exit(1)
        print("✅ All dependencies installed!\n")

_ensure_dependencies()

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple
import sys
sys.path.append('.')

# Import core components (assuming singularity_core.py exists)
try:
    from singularity_core import SingularityCore
except:
    print("⚠️ singularity_core.py not found - using standalone mode")


class XAUUSDAgent:
    """Specialized agent for XAUUSD trading"""
    
    def __init__(self, base_risk: float = 0.01):
        self.symbol = 'XAUUSD'
        self.base_risk = base_risk
        
        # XAUUSD-specific parameters
        self.optimal_timeframe = 15  # 15-min optimal for gold
        self.stop_hunt_protection = True
        self.news_avoidance = True
        
        # Core engine
        try:
            self.core = SingularityCore(base_risk=base_risk)
        except:
            self.core = None
            print("⚠️ Running without SingularityCore")
        
        # Trading rules specific to XAUUSD
        self.rules = {
            'max_position_time': 240,  # Max 4 hours
            'stop_loss_atr_multiple': 1.5,  # Tight stops
            'take_profit_rr': 1.8,  # Slightly better RR
            'avoid_news_hours': [8, 9, 10, 13, 14],  # EST hours to avoid
            'regime_preference': 'mean_revert'  # Gold loves mean reversion
        }
        
        # Performance tracking
        self.returns = []
        self.equity_curve = [0]
        self.trades_log = []
    
    def analyze_current_state(self, returns: np.ndarray) -> Dict:
        """Analyze XAUUSD with specialized logic"""
        
        if self.core:
            # Use full Singularity Core
            analysis = self.core.analyze(returns)
        else:
            # Fallback basic analysis
            analysis = self._basic_analysis(returns)
        
        # Add XAUUSD-specific insights
        analysis['xauusd_insights'] = self._xauusd_specific_analysis(returns)
        
        return analysis
    
    def _basic_analysis(self, returns: np.ndarray) -> Dict:
        """Basic analysis if core not available"""
        if len(returns) < 10:
            return {'error': 'Insufficient data'}
        
        wins = np.sum(returns > 0)
        total = len(returns)
        
        return {
            'statistics': {
                'total_trades': total,
                'win_rate': wins / total if total > 0 else 0,
                'avg_win': np.mean(returns[returns > 0]) if wins > 0 else 0,
                'avg_loss': abs(np.mean(returns[returns < 0])) if (total - wins) > 0 else 0,
                'expectancy': np.mean(returns),
                'volatility': np.std(returns),
                'current_equity': np.sum(returns)
            }
        }
    
    def _xauusd_specific_analysis(self, returns: np.ndarray) -> Dict:
        """XAUUSD-specific market analysis"""
        
        insights = {}
        
        # 1. Stop Hunt Detection (consecutive small losses)
        if len(returns) >= 5:
            recent = returns[-5:]
            small_losses = np.sum((recent < 0) & (recent > -0.5))
            insights['stop_hunt_suspected'] = small_losses >= 3
        
        # 2. Mean Reversion Potential
        if len(returns) >= 20:
            # Check if recent moves are extreme
            z_score = (returns[-1] - np.mean(returns[-20:])) / np.std(returns[-20:])
            insights['mean_revert_signal'] = abs(z_score) > 1.5
            insights['z_score'] = z_score
        
        # 3. Noise Level (XAUUSD specific)
        if len(returns) >= 10:
            # High noise = many small moves
            small_moves = np.sum(np.abs(returns[-10:]) < 0.5) / 10
            insights['noise_level'] = 'HIGH' if small_moves > 0.6 else 'NORMAL'
        
        # 4. Trend Exhaustion (for mean reversion entry)
        if len(returns) >= 15:
            recent_trend = np.sum(returns[-15:] > 0) / 15
            insights['trend_exhaustion'] = recent_trend > 0.75 or recent_trend < 0.25
        
        return insights
    
    def get_position_size(self, analysis: Dict, account_balance: float) -> float:
        """Calculate XAUUSD position size in lots"""
        
        if 'risk' in analysis:
            risk_pct = analysis['risk']['risk_pct']
        else:
            risk_pct = self.base_risk
        
        # XAUUSD-specific adjustments
        xau_insights = analysis.get('xauusd_insights', {})
        
        # Reduce risk if stop hunt suspected
        if xau_insights.get('stop_hunt_suspected', False):
            risk_pct *= 0.5
        
        # Reduce risk in high noise
        if xau_insights.get('noise_level') == 'HIGH':
            risk_pct *= 0.7
        
        # Calculate position size
        risk_amount = account_balance * risk_pct
        
        # XAUUSD: 1 lot = $100 per point, typical stop = 15-20 points
        typical_stop_points = 17
        stop_value = typical_stop_points * 100  # $1700 per lot
        
        position_lots = risk_amount / stop_value
        
        return round(position_lots, 2)
    
    def should_trade_now(self, analysis: Dict, current_hour: int) -> Tuple[bool, str]:
        """Decide if should trade XAUUSD now"""
        
        # Kill switch check
        if 'kill_switch' in analysis:
            if analysis['kill_switch']['triggered']:
                return False, f"Kill Switch: {analysis['kill_switch']['reason']}"
        
        # News avoidance
        if self.news_avoidance and current_hour in self.rules['avoid_news_hours']:
            return False, "News hour - avoiding trades"
        
        # Edge check
        if 'edge' in analysis:
            if analysis['edge'].get('expectancy_lb', 0) <= 0:
                return False, "No statistical edge"
        
        # Regime check (XAUUSD prefers NOT to trade in panic)
        if 'regime' in analysis:
            if analysis['regime']['id'] == 2:
                return False, "Panic regime - too volatile"
        
        # XAUUSD insights
        xau_insights = analysis.get('xauusd_insights', {})
        
        # Don't trade if stop hunt suspected and we're in drawdown
        if xau_insights.get('stop_hunt_suspected', False):
            if 'statistics' in analysis:
                if analysis['statistics'].get('current_dd', 0) < -3:
                    return False, "Stop hunt + drawdown - pause"
        
        return True, "All clear"
    
    def get_entry_signals(self, analysis: Dict) -> Dict:
        """Generate XAUUSD entry signals"""
        
        xau_insights = analysis.get('xauusd_insights', {})
        
        signals = {
            'mean_revert_long': False,
            'mean_revert_short': False,
            'breakout_long': False,
            'breakout_short': False
        }
        
        # Mean reversion signals (primary for XAUUSD)
        if xau_insights.get('mean_revert_signal', False):
            z = xau_insights.get('z_score', 0)
            if z < -1.5:
                signals['mean_revert_long'] = True
            elif z > 1.5:
                signals['mean_revert_short'] = True
        
        # Trend exhaustion reversals
        if xau_insights.get('trend_exhaustion', False):
            # Would need price data to determine direction
            # This is a placeholder
            pass
        
        return signals
    
    def log_trade(self, entry_price: float, exit_price: float, 
                  direction: str, r_multiple: float):
        """Log trade for analysis"""
        
        self.returns.append(r_multiple)
        self.equity_curve.append(self.equity_curve[-1] + r_multiple)
        
        trade_log = {
            'entry': entry_price,
            'exit': exit_price,
            'direction': direction,
            'r_multiple': r_multiple,
            'equity': self.equity_curve[-1]
        }
        
        self.trades_log.append(trade_log)
    
    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        
        if len(self.returns) == 0:
            return {'status': 'No trades yet'}
        
        returns_arr = np.array(self.returns)
        
        report = {
            'symbol': self.symbol,
            'total_trades': len(returns_arr),
            'win_rate': np.sum(returns_arr > 0) / len(returns_arr),
            'avg_r': np.mean(returns_arr),
            'total_r': np.sum(returns_arr),
            'max_r': np.max(returns_arr),
            'min_r': np.min(returns_arr),
            'std_r': np.std(returns_arr),
            'sharpe_r': np.mean(returns_arr) / np.std(returns_arr) if np.std(returns_arr) > 0 else 0,
            'max_dd_r': self._calculate_max_dd(returns_arr),
            'current_equity_r': self.equity_curve[-1]
        }
        
        return report
    
    def _calculate_max_dd(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cum_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = cum_returns - running_max
        return abs(np.min(drawdown))


# Example Usage
if __name__ == "__main__":
    print("="*80)
    print("XAUUSD SPECIALIZED AGENT - Demo")
    print("="*80)
    
    # Create agent
    agent = XAUUSDAgent(base_risk=0.01)
    
    # Simulate some XAUUSD trades (R-multiples)
    np.random.seed(42)
    simulated_returns = []
    
    # XAUUSD characteristics: mean=0.3R, std=1.4R, slight positive skew
    base_returns = np.random.normal(0.3, 1.4, 100)
    
    # Add some stop hunts (small losses)
    stop_hunts = np.random.choice([0, -0.3, -0.4], size=20)
    simulated_returns = np.concatenate([base_returns, stop_hunts])
    np.random.shuffle(simulated_returns)
    
    # Analyze
    analysis = agent.analyze_current_state(simulated_returns)
    
    print("\n📊 ANALYSIS RESULTS:")
    print(json.dumps(analysis.get('statistics', {}), indent=2))
    
    if 'xauusd_insights' in analysis:
        print("\n💎 XAUUSD INSIGHTS:")
        print(json.dumps(analysis['xauusd_insights'], indent=2))
    
    # Position sizing
    account_balance = 10000
    position_lots = agent.get_position_size(analysis, account_balance)
    print(f"\n💰 Position Size: {position_lots} lots")
    
    # Trading decision
    current_hour = 11  # 11 AM EST
    can_trade, reason = agent.should_trade_now(analysis, current_hour)
    print(f"\n✅ Can Trade: {can_trade}")
    print(f"   Reason: {reason}")
    
    # Entry signals
    signals = agent.get_entry_signals(analysis)
    print(f"\n🎯 Entry Signals:")
    for signal, active in signals.items():
        if active:
            print(f"   {signal}: ✓")
    
    # Log some trades
    print("\n📝 Logging sample trades...")
    for r in simulated_returns[:10]:
        agent.log_trade(2650.0, 2650.0 + r*10, 'LONG', r)
    
    # Performance report
    report = agent.get_performance_report()
    print("\n📈 PERFORMANCE REPORT:")
    print(json.dumps(report, indent=2))
    
    print("\n" + "="*80)
    print("🎯 XAUUSD Agent ready for live trading!")
    print("="*80)