"""
Multi-Agent Singularity Capital OS
Self-Evolving Portfolio Intelligence

Components:
- MarketAgent: Individual trading organism
- Fitness Function: Survival-based evaluation
- Evolution Engine: Genetic algorithm for agent evolution
- Portfolio Allocator: Dynamic capital distribution
- Meta-Learner: Contextual bandit for strategy selection
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
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from copy import deepcopy
import json


@dataclass
class AgentGenome:
    """DNA of a trading agent"""
    entry_type: str  # 'breakout', 'mean_revert', 'trend'
    exit_type: str   # 'tp_sl', 'trailing', 'time_exit'
    timeframe: int   # Minutes (5, 15, 60, 240)
    risk_profile: float  # 0.1 to 1.0 (multiplier on base risk)
    regime_sensitivity: float  # 0.3 to 1.0 (how much to react to regime)
    
    def mutate(self, mutation_rate: float = 0.2) -> 'AgentGenome':
        """Create mutated copy of genome"""
        new_genome = deepcopy(self)
        
        if np.random.random() < mutation_rate:
            new_genome.entry_type = np.random.choice(['breakout', 'mean_revert', 'trend'])
        
        if np.random.random() < mutation_rate:
            new_genome.exit_type = np.random.choice(['tp_sl', 'trailing', 'time_exit'])
        
        if np.random.random() < mutation_rate:
            new_genome.timeframe = np.random.choice([5, 15, 60, 240])
        
        if np.random.random() < mutation_rate:
            new_genome.risk_profile = np.clip(
                new_genome.risk_profile + np.random.normal(0, 0.1), 0.1, 1.0
            )
        
        if np.random.random() < mutation_rate:
            new_genome.regime_sensitivity = np.clip(
                new_genome.regime_sensitivity + np.random.normal(0, 0.1), 0.3, 1.0
            )
        
        return new_genome
    
    @staticmethod
    def crossover(genome1: 'AgentGenome', genome2: 'AgentGenome') -> 'AgentGenome':
        """Combine two genomes"""
        return AgentGenome(
            entry_type=np.random.choice([genome1.entry_type, genome2.entry_type]),
            exit_type=np.random.choice([genome1.exit_type, genome2.exit_type]),
            timeframe=np.random.choice([genome1.timeframe, genome2.timeframe]),
            risk_profile=(genome1.risk_profile + genome2.risk_profile) / 2,
            regime_sensitivity=(genome1.regime_sensitivity + genome2.regime_sensitivity) / 2
        )


class MarketAgent:
    """Individual Trading Agent for a specific market"""
    
    def __init__(self, symbol: str, genome: Optional[AgentGenome] = None):
        self.symbol = symbol
        self.genome = genome or self._random_genome()
        self.returns = []
        self.edge = 0.0
        self.fitness = 0.0
        self.alive = True
        self.trades_count = 0
        self.generation = 0
        
    def _random_genome(self) -> AgentGenome:
        """Generate random genome"""
        return AgentGenome(
            entry_type=np.random.choice(['breakout', 'mean_revert', 'trend']),
            exit_type=np.random.choice(['tp_sl', 'trailing', 'time_exit']),
            timeframe=np.random.choice([5, 15, 60, 240]),
            risk_profile=np.random.uniform(0.1, 1.0),
            regime_sensitivity=np.random.uniform(0.3, 1.0)
        )
    
    def update_returns(self, new_returns: List[float]):
        """Add new trading results"""
        self.returns.extend(new_returns)
        self.trades_count += len(new_returns)
        self._calculate_fitness()
    
    def _calculate_fitness(self):
        """Calculate survival-based fitness score"""
        if len(self.returns) < 10:
            self.fitness = 0.0
            return
        
        returns_arr = np.array(self.returns)
        
        # Log growth (expected geometric return)
        log_growth = np.mean(np.log1p(returns_arr))
        
        # Max Drawdown
        cum_returns = np.cumsum(returns_arr)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = cum_returns - running_max
        max_dd = abs(np.min(drawdown))
        
        # Tail Risk (CVaR at 5%)
        tail_risk = abs(np.percentile(returns_arr, 5))
        
        # Consistency (lower variance = better)
        consistency_penalty = np.std(returns_arr) * 0.1
        
        # Fitness = Survival Utility
        self.fitness = (
            log_growth 
            - 0.5 * max_dd 
            - 0.3 * tail_risk 
            - consistency_penalty
        )
        
        # Calculate edge
        wins = np.sum(returns_arr > 0)
        total = len(returns_arr)
        if total > 0:
            win_rate = wins / total
            avg_win = np.mean(returns_arr[returns_arr > 0]) if wins > 0 else 0
            avg_loss = abs(np.mean(returns_arr[returns_arr < 0])) if (total - wins) > 0 else 0
            self.edge = win_rate * avg_win - (1 - win_rate) * avg_loss
        else:
            self.edge = 0
    
    def kill(self, reason: str = "Low fitness"):
        """Terminate agent"""
        self.alive = False
        self.fitness = 0
        print(f"🔴 {self.symbol} Agent KILLED - {reason}")
    
    def to_dict(self) -> Dict:
        """Export agent state"""
        return {
            'symbol': self.symbol,
            'alive': self.alive,
            'fitness': self.fitness,
            'edge': self.edge,
            'trades': self.trades_count,
            'generation': self.generation,
            'genome': {
                'entry': self.genome.entry_type,
                'exit': self.genome.exit_type,
                'timeframe': self.genome.timeframe,
                'risk_profile': self.genome.risk_profile,
                'regime_sensitivity': self.genome.regime_sensitivity
            }
        }


class EvolutionEngine:
    """Genetic Algorithm for Agent Evolution"""
    
    def __init__(self, mutation_rate: float = 0.2, elitism: int = 1):
        self.mutation_rate = mutation_rate
        self.elitism = elitism
    
    def evolve_population(self, agents: List[MarketAgent]) -> List[MarketAgent]:
        """Evolve agent population"""
        if len(agents) < 2:
            return agents
        
        # Sort by fitness
        agents.sort(key=lambda x: x.fitness, reverse=True)
        
        # Keep elite agents
        new_population = agents[:self.elitism]
        
        # Kill worst performers
        kill_count = len(agents) // 4
        for agent in agents[-kill_count:]:
            agent.kill("Bottom 25% fitness")
        
        # Create new agents through mutation and crossover
        while len(new_population) < len(agents):
            # Selection (tournament)
            parent1 = self._tournament_select(agents)
            parent2 = self._tournament_select(agents)
            
            # Crossover
            if np.random.random() < 0.7:
                child_genome = AgentGenome.crossover(parent1.genome, parent2.genome)
            else:
                child_genome = deepcopy(parent1.genome)
            
            # Mutation
            child_genome = child_genome.mutate(self.mutation_rate)
            
            # Create new agent
            child = MarketAgent(parent1.symbol, child_genome)
            child.generation = max(parent1.generation, parent2.generation) + 1
            new_population.append(child)
        
        return new_population
    
    def _tournament_select(self, agents: List[MarketAgent], k: int = 3) -> MarketAgent:
        """Tournament selection"""
        tournament = np.random.choice(agents, size=min(k, len(agents)), replace=False)
        return max(tournament, key=lambda x: x.fitness)


class PortfolioAllocator:
    """Dynamic Capital Allocation across Agents"""
    
    def __init__(self, min_allocation: float = 0.05):
        self.min_allocation = min_allocation
    
    def allocate(self, agents: List[MarketAgent]) -> Dict[str, float]:
        """Calculate capital allocation weights"""
        alive_agents = [a for a in agents if a.alive and a.fitness > 0]
        
        if not alive_agents:
            # Equal allocation if no fitness data
            return {a.symbol: 1.0 / len(agents) for a in agents}
        
        # Fitness-based allocation
        fitnesses = np.array([a.fitness for a in alive_agents])
        fitnesses = np.maximum(fitnesses, 0)  # No negative allocations
        
        if fitnesses.sum() == 0:
            weights = np.ones(len(alive_agents)) / len(alive_agents)
        else:
            weights = fitnesses / fitnesses.sum()
        
        # Create allocation dict
        allocation = {}
        for i, agent in enumerate(alive_agents):
            allocation[agent.symbol] = max(weights[i], self.min_allocation)
        
        # Normalize to 1.0
        total = sum(allocation.values())
        allocation = {k: v / total for k, v in allocation.items()}
        
        # Dead agents get 0
        for agent in agents:
            if not agent.alive:
                allocation[agent.symbol] = 0.0
        
        return allocation


class MetaLearner:
    """Contextual Bandit for Strategy Selection"""
    
    def __init__(self, exploration_rate: float = 0.1):
        self.exploration_rate = exploration_rate
        self.history = []
    
    def select_agent(self, agents: List[MarketAgent], context: Dict) -> MarketAgent:
        """Select best agent based on context (regime, volatility, etc.)"""
        alive_agents = [a for a in agents if a.alive]
        
        if not alive_agents:
            return agents[0]  # Fallback
        
        # Exploration vs Exploitation
        if np.random.random() < self.exploration_rate:
            return np.random.choice(alive_agents)
        
        # Exploitation: choose best fitness
        return max(alive_agents, key=lambda x: x.fitness)
    
    def update(self, agent: MarketAgent, context: Dict, reward: float):
        """Update learning history"""
        self.history.append({
            'symbol': agent.symbol,
            'context': context,
            'reward': reward,
            'fitness': agent.fitness
        })


class MultiAgentSystem:
    """Complete Multi-Agent Portfolio System"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.agents = [MarketAgent(symbol) for symbol in symbols]
        self.evolution = EvolutionEngine()
        self.allocator = PortfolioAllocator()
        self.meta_learner = MetaLearner()
        self.iteration = 0
        
    def update(self, returns_dict: Dict[str, List[float]]):
        """Update all agents with new returns"""
        for agent in self.agents:
            if agent.symbol in returns_dict:
                agent.update_returns(returns_dict[agent.symbol])
        
        self.iteration += 1
    
    def get_allocations(self) -> Dict[str, float]:
        """Get current capital allocations"""
        return self.allocator.allocate(self.agents)
    
    def evolve(self):
        """Trigger evolution process"""
        agents_by_symbol = {}
        for agent in self.agents:
            if agent.symbol not in agents_by_symbol:
                agents_by_symbol[agent.symbol] = []
            agents_by_symbol[agent.symbol].append(agent)
        
        # Evolve each symbol's agents
        new_agents = []
        for symbol, symbol_agents in agents_by_symbol.items():
            evolved = self.evolution.evolve_population(symbol_agents)
            new_agents.extend(evolved)
        
        self.agents = new_agents
    
    def get_status(self) -> Dict:
        """Get system status report"""
        allocations = self.get_allocations()
        
        status = {
            'iteration': self.iteration,
            'total_agents': len(self.agents),
            'alive_agents': sum(1 for a in self.agents if a.alive),
            'allocations': allocations,
            'agents': [a.to_dict() for a in self.agents]
        }
        
        return status
    
    def export_state(self, filename: str):
        """Export system state to JSON"""
        state = self.get_status()
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"✅ State exported to {filename}")


# Example Usage
if __name__ == "__main__":
    # Initialize multi-agent system
    symbols = ['XAUUSD', 'BTCUSD', 'USOIL', 'EURUSD']
    mas = MultiAgentSystem(symbols)
    
    # Simulate some trading results
    np.random.seed(42)
    for iteration in range(5):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration + 1}")
        print(f"{'='*80}")
        
        # Generate fake returns for each symbol
        returns = {
            'XAUUSD': list(np.random.normal(0.3, 1.4, 20)),
            'BTCUSD': list(np.random.normal(0.2, 2.0, 20)),
            'USOIL': list(np.random.normal(0.1, 1.8, 20)),
            'EURUSD': list(np.random.normal(0.15, 1.2, 20))
        }
        
        # Update agents
        mas.update(returns)
        
        # Get allocations
        allocations = mas.get_allocations()
        print("\n💰 CAPITAL ALLOCATIONS:")
        for symbol, weight in allocations.items():
            print(f"  {symbol}: {weight*100:.2f}%")
        
        # Show agent status
        print("\n🧬 AGENT STATUS:")
        for agent in mas.agents:
            if agent.alive:
                print(f"  {agent.symbol}: Fitness={agent.fitness:.4f}, Edge={agent.edge:.4f}, Gen={agent.generation}")
        
        # Evolve every 2 iterations
        if (iteration + 1) % 2 == 0:
            print("\n🔄 EVOLVING POPULATION...")
            mas.evolve()
    
    # Export final state
    mas.export_state('mas_state.json')