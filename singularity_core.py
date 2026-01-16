"""
Singularity Capital OS - Core Engine
Self-Evolving Capital Intelligence System

Core Components:
- Bayesian Edge Estimation
- Monte Carlo Simulation
- Regime Detection (Manual HMM Approximation)
- Dynamic Risk Calculator
- CUSUM Drift Detector
- Kill Switch Logic
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
from scipy import stats
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class BayesianEdge:
    """Bayesian Edge Estimator using Beta Distribution"""
    
    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        self.alpha = alpha_prior
        self.beta = beta_prior
        
    def update(self, returns: np.ndarray) -> Dict[str, float]:
        """Update posterior distribution and calculate edge"""
        wins = np.sum(returns > 0)
        losses = np.sum(returns < 0)
        
        # Update posterior
        self.alpha += wins
        self.beta += losses
        
        # Calculate statistics
        win_rate_mean = self.alpha / (self.alpha + self.beta)
        win_rate_lb = stats.beta.ppf(0.10, self.alpha, self.beta)  # Conservative 10th percentile
        
        # Expectancy calculation
        avg_win = np.mean(returns[returns > 0]) if wins > 0 else 0
        avg_loss = abs(np.mean(returns[returns < 0])) if losses > 0 else 0
        
        expectancy_mean = win_rate_mean * avg_win - (1 - win_rate_mean) * avg_loss
        expectancy_lb = win_rate_lb * avg_win - (1 - win_rate_lb) * avg_loss
        
        return {
            'win_rate_mean': win_rate_mean,
            'win_rate_lb': win_rate_lb,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy_mean': expectancy_mean,
            'expectancy_lb': expectancy_lb,
            'n_trades': len(returns)
        }


class MonteCarloSimulator:
    """Monte Carlo Worst Case Scenario Generator"""
    
    @staticmethod
    def simulate(returns: np.ndarray, n_sims: int = 10000, n_trades: int = None) -> Dict[str, float]:
        """Run Monte Carlo simulation"""
        if n_trades is None:
            n_trades = len(returns)
            
        if len(returns) < 10:
            return {'max_dd_95': 0, 'losing_streak_95': 0, 'final_equity_5': 0}
        
        max_dds = []
        losing_streaks = []
        final_equities = []
        
        for _ in range(n_sims):
            # Random sample with replacement
            sim_returns = np.random.choice(returns, size=n_trades, replace=True)
            
            # Calculate metrics
            cum_returns = np.cumsum(sim_returns)
            running_max = np.maximum.accumulate(cum_returns)
            drawdown = cum_returns - running_max
            max_dd = abs(np.min(drawdown))
            
            # Losing streak
            streak = 0
            max_streak = 0
            for r in sim_returns:
                if r < 0:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 0
            
            max_dds.append(max_dd)
            losing_streaks.append(max_streak)
            final_equities.append(cum_returns[-1])
        
        return {
            'max_dd_95': np.percentile(max_dds, 95),
            'losing_streak_95': np.percentile(losing_streaks, 95),
            'final_equity_5': np.percentile(final_equities, 5)
        }


class RegimeDetector:
    """Regime Detection using Rolling Volatility (HMM Approximation)"""
    
    def __init__(self, window: int = 20):
        self.window = window
        self.regimes = {0: 'Low Variance (Trend)', 1: 'Mid (Chop)', 2: 'High Variance (Panic)'}
        
    def detect(self, returns: np.ndarray) -> Tuple[int, str, float]:
        """Detect current regime"""
        if len(returns) < self.window:
            return 1, self.regimes[1], 0.0
        
        # Calculate rolling volatility
        vol_series = pd.Series(returns).rolling(self.window).std().dropna()
        
        if len(vol_series) == 0:
            return 1, self.regimes[1], 0.0
        
        # Quantile thresholds
        q33, q66 = np.percentile(vol_series, [33, 66])
        current_vol = vol_series.iloc[-1]
        
        # Classify regime
        if current_vol < q33:
            regime_id = 0
        elif current_vol < q66:
            regime_id = 1
        else:
            regime_id = 2
        
        return regime_id, self.regimes[regime_id], current_vol


class CUSUMDetector:
    """CUSUM Algorithm for Edge Drift Detection"""
    
    def __init__(self, threshold: float = 5.0, drift: float = 0.5):
        self.threshold = threshold
        self.drift = drift
        self.cusum_pos = 0
        self.cusum_neg = 0
        
    def update(self, value: float, target: float = 0) -> bool:
        """Update CUSUM and check for drift"""
        deviation = value - target
        
        self.cusum_pos = max(0, self.cusum_pos + deviation - self.drift)
        self.cusum_neg = max(0, self.cusum_neg - deviation - self.drift)
        
        return self.cusum_pos > self.threshold or self.cusum_neg > self.threshold
    
    def reset(self):
        """Reset CUSUM values"""
        self.cusum_pos = 0
        self.cusum_neg = 0


class DynamicRiskCalculator:
    """Dynamic Kelly Fraction with Regime & Drawdown Adjustments"""
    
    def __init__(self, base_risk: float = 0.01, kelly_fraction: float = 0.3):
        self.base_risk = base_risk
        self.kelly_fraction = kelly_fraction
        
    def calculate(self, edge_info: Dict, regime_id: int, current_dd: float, 
                  worst_dd: float) -> Dict[str, float]:
        """Calculate dynamic risk per trade"""
        
        # Kelly Criterion
        expectancy = edge_info['expectancy_lb']
        avg_win = edge_info['avg_win']
        
        if avg_win == 0:
            kelly = 0
        else:
            kelly = expectancy / avg_win
        
        # Regime factor
        regime_factors = {0: 1.2, 1: 1.0, 2: 0.3}
        regime_factor = regime_factors.get(regime_id, 1.0)
        
        # Drawdown factor
        if worst_dd == 0:
            dd_factor = 1.0
        else:
            dd_ratio = abs(current_dd) / abs(worst_dd)
            dd_factor = 0.5 if dd_ratio > 1.0 else 1.0
        
        # Final risk
        risk = self.base_risk * kelly * self.kelly_fraction * regime_factor * dd_factor
        
        return {
            'kelly': kelly,
            'regime_factor': regime_factor,
            'dd_factor': dd_factor,
            'risk_pct': max(0, risk),
            'kelly_fraction': self.kelly_fraction
        }


class KillSwitch:
    """God Kill Switch - Ultimate Risk Control"""
    
    @staticmethod
    def check(edge_lb: float, drift_detected: bool, regime_id: int, 
              correlation: float = None, max_dd_pct: float = 0.20) -> Tuple[bool, str]:
        """Check if kill switch should be triggered"""
        
        reasons = []
        
        # Edge check
        if edge_lb <= 0:
            reasons.append("Edge <= 0")
        
        # Drift check
        if drift_detected:
            reasons.append("Edge Drift Detected")
        
        # Regime check
        if regime_id == 2:
            reasons.append("Panic Regime")
        
        # Correlation check (systemic risk)
        if correlation is not None and correlation > 0.85:
            reasons.append("High Correlation (Systemic Risk)")
        
        killed = len(reasons) > 0
        reason_str = " | ".join(reasons) if killed else "All Clear"
        
        return killed, reason_str


class SingularityCore:
    """Main Core Engine - Integrates All Components"""
    
    def __init__(self, base_risk: float = 0.01):
        self.bayesian = BayesianEdge()
        self.monte_carlo = MonteCarloSimulator()
        self.regime_detector = RegimeDetector()
        self.cusum = CUSUMDetector()
        self.risk_calc = DynamicRiskCalculator(base_risk=base_risk)
        self.kill_switch = KillSwitch()
        
    def analyze(self, returns: np.ndarray) -> Dict:
        """Complete analysis of trading system"""
        
        if len(returns) < 10:
            return {'error': 'Insufficient data (need at least 10 trades)'}
        
        # 1. Bayesian Edge
        edge_info = self.bayesian.update(returns)
        
        # 2. Monte Carlo
        mc_results = self.monte_carlo.simulate(returns)
        
        # 3. Regime Detection
        regime_id, regime_name, current_vol = self.regime_detector.detect(returns)
        
        # 4. CUSUM Drift Detection
        drift_detected = False
        if len(returns) >= 20:
            recent_expectancy = np.mean(returns[-20:])
            drift_detected = self.cusum.update(recent_expectancy, edge_info['expectancy_mean'])
        
        # 5. Current Drawdown
        cum_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = cum_returns - running_max
        current_dd = drawdown[-1]
        
        # 6. Dynamic Risk
        risk_info = self.risk_calc.calculate(
            edge_info, regime_id, current_dd, mc_results['max_dd_95']
        )
        
        # 7. Kill Switch
        killed, kill_reason = self.kill_switch.check(
            edge_info['expectancy_lb'], drift_detected, regime_id
        )
        
        # 8. Statistics
        stats_info = {
            'total_trades': len(returns),
            'win_rate': edge_info['win_rate_mean'],
            'avg_win': edge_info['avg_win'],
            'avg_loss': edge_info['avg_loss'],
            'expectancy': edge_info['expectancy_mean'],
            'volatility': np.std(returns),
            'cvar_5': np.percentile(returns, 5),
            'current_dd': current_dd,
            'current_equity': cum_returns[-1]
        }
        
        return {
            'statistics': stats_info,
            'edge': edge_info,
            'monte_carlo': mc_results,
            'regime': {'id': regime_id, 'name': regime_name, 'volatility': current_vol},
            'drift_detected': drift_detected,
            'risk': risk_info,
            'kill_switch': {'triggered': killed, 'reason': kill_reason}
        }


# Example Usage
if __name__ == "__main__":
    # Example XAUUSD returns in R-multiples
    np.random.seed(42)
    example_returns = np.random.normal(0.32, 1.42, 300)  # Mean=0.32R, Std=1.42R
    
    core = SingularityCore(base_risk=0.01)
    results = core.analyze(example_returns)
    
    print("=" * 80)
    print("SINGULARITY CAPITAL OS - ANALYSIS REPORT")
    print("=" * 80)
    
    print("\n📊 STATISTICS")
    for key, value in results['statistics'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\n🎯 BAYESIAN EDGE")
    for key, value in results['edge'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\n⚠️ MONTE CARLO WORST CASE")
    for key, value in results['monte_carlo'].items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\n🌐 REGIME: {results['regime']['name']} (Volatility: {results['regime']['volatility']:.4f})")
    print(f"🔍 DRIFT DETECTED: {results['drift_detected']}")
    
    print("\n💰 DYNAMIC RISK")
    for key, value in results['risk'].items():
        print(f"  {key}: {value:.6f}")
    
    print(f"\n🛑 KILL SWITCH: {'TRIGGERED' if results['kill_switch']['triggered'] else 'OK'}")
    print(f"   Reason: {results['kill_switch']['reason']}")