#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║           SINGULARITY CAPITAL OS - MT5 DATA CONNECTOR                         ║
║                    Real-Time MetaTrader 5 Integration                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Provides real-time and historical data from MetaTrader 5.
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
        'MetaTrader5': 'MetaTrader5',
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

_ensure_dependencies()

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import time

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ MetaTrader5 package not available. Using simulation mode.")


# ============================================================================
# MT5 TIMEFRAMES MAPPING
# ============================================================================
TIMEFRAMES = {
    'M1': mt5.TIMEFRAME_M1 if MT5_AVAILABLE else 1,
    'M5': mt5.TIMEFRAME_M5 if MT5_AVAILABLE else 5,
    'M15': mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15,
    'M30': mt5.TIMEFRAME_M30 if MT5_AVAILABLE else 30,
    'H1': mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 60,
    'H4': mt5.TIMEFRAME_H4 if MT5_AVAILABLE else 240,
    'D1': mt5.TIMEFRAME_D1 if MT5_AVAILABLE else 1440,
    'W1': mt5.TIMEFRAME_W1 if MT5_AVAILABLE else 10080,
}


# ============================================================================
# MT5 CONNECTOR CLASS
# ============================================================================
class MT5Connector:
    """
    MetaTrader 5 Data Connector
    
    Provides real-time and historical data from MT5 terminal.
    Falls back to simulation mode if MT5 is not available.
    """
    
    def __init__(self):
        self.connected = False
        self.account_info = {}
        self.symbols_info = {}
        self.simulation_mode = not MT5_AVAILABLE
        self._price_callbacks = []
        self._price_thread = None
        self._running = False
        
    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================
    
    def connect(self, path: str = None, login: int = None, 
                password: str = None, server: str = None) -> Tuple[bool, str]:
        """
        Connect to MT5 terminal.
        
        Args:
            path: Path to MT5 terminal (optional)
            login: Account login (optional)
            password: Account password (optional)
            server: Server name (optional)
            
        Returns:
            Tuple of (success, message)
        """
        if self.simulation_mode:
            self.connected = True
            self.account_info = self._get_simulated_account()
            return True, "Connected in SIMULATION mode (MT5 not available)"
        
        try:
            # Initialize MT5
            if path:
                init_result = mt5.initialize(path)
            else:
                init_result = mt5.initialize()
            
            if not init_result:
                error = mt5.last_error()
                return False, f"MT5 initialization failed: {error}"
            
            # Login if credentials provided
            if login and password and server:
                auth_result = mt5.login(login, password=password, server=server)
                if not auth_result:
                    error = mt5.last_error()
                    return False, f"MT5 login failed: {error}"
            
            self.connected = True
            self.account_info = self._get_account_info()
            return True, f"Connected to MT5 - Account: {self.account_info.get('login', 'N/A')}"
            
        except Exception as e:
            return False, f"Connection error: {str(e)}"
    
    def disconnect(self) -> Tuple[bool, str]:
        """Disconnect from MT5 terminal."""
        self._stop_price_stream()
        
        if self.simulation_mode:
            self.connected = False
            return True, "Disconnected from simulation mode"
        
        try:
            mt5.shutdown()
            self.connected = False
            return True, "Disconnected from MT5"
        except Exception as e:
            return False, f"Disconnect error: {str(e)}"
    
    def is_connected(self) -> bool:
        """Check if connected to MT5."""
        if self.simulation_mode:
            return self.connected
        
        if not self.connected:
            return False
        
        try:
            info = mt5.terminal_info()
            return info is not None
        except:
            return False
    
    # ========================================================================
    # ACCOUNT INFORMATION
    # ========================================================================
    
    def _get_account_info(self) -> Dict:
        """Get account information from MT5."""
        if self.simulation_mode:
            return self._get_simulated_account()
        
        try:
            info = mt5.account_info()
            if info is None:
                return {}
            
            # Convert trade mode to readable name
            trade_mode_names = {0: 'Demo', 1: 'Contest', 2: 'Real'}
            trade_mode = trade_mode_names.get(info.trade_mode, f'Unknown ({info.trade_mode})')
            
            return {
                'login': info.login,
                'name': info.name,
                'server': info.server,
                'balance': info.balance,
                'equity': info.equity,
                'margin': info.margin,
                'free_margin': info.margin_free,
                'leverage': info.leverage,
                'profit': info.profit,
                'currency': info.currency,
                'trade_mode': trade_mode,
            }
        except Exception as e:
            print(f"Error getting account info: {e}")
            return {}
    
    def _get_simulated_account(self) -> Dict:
        """Get simulated account info."""
        return {
            'login': 'SIMULATION',
            'name': 'Singularity Trader',
            'server': 'Simulated Server',
            'balance': 10000.00,
            'equity': 10000.00,
            'margin': 0.0,
            'free_margin': 10000.00,
            'leverage': 100,
            'profit': 0.0,
            'currency': 'USD',
            'trade_mode': 'Simulation',
        }
    
    def get_account_info(self) -> Dict:
        """Get current account information."""
        if self.connected:
            self.account_info = self._get_account_info()
        return self.account_info
    
    # ========================================================================
    # SYMBOL INFORMATION
    # ========================================================================
    
    def get_symbols(self) -> List[str]:
        """Get list of available symbols."""
        if self.simulation_mode:
            return ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSD', 
                    'USOIL', 'US30', 'US500', 'NAS100', 'GER40']
        
        try:
            symbols = mt5.symbols_get()
            if symbols is None:
                return []
            return [s.name for s in symbols if s.visible]
        except:
            return []
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get detailed symbol information."""
        if self.simulation_mode:
            return self._get_simulated_symbol_info(symbol)
        
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                return {}
            
            return {
                'name': info.name,
                'description': info.description,
                'point': info.point,
                'digits': info.digits,
                'spread': info.spread,
                'contract_size': info.trade_contract_size,
                'volume_min': info.volume_min,
                'volume_max': info.volume_max,
                'volume_step': info.volume_step,
                'swap_long': info.swap_long,
                'swap_short': info.swap_short,
                'bid': info.bid,
                'ask': info.ask,
            }
        except Exception as e:
            print(f"Error getting symbol info: {e}")
            return {}
    
    def _get_simulated_symbol_info(self, symbol: str) -> Dict:
        """Get simulated symbol info."""
        symbol_configs = {
            'XAUUSD': {'digits': 2, 'point': 0.01, 'contract_size': 100, 'spread': 25},
            'EURUSD': {'digits': 5, 'point': 0.00001, 'contract_size': 100000, 'spread': 10},
            'GBPUSD': {'digits': 5, 'point': 0.00001, 'contract_size': 100000, 'spread': 15},
            'USDJPY': {'digits': 3, 'point': 0.001, 'contract_size': 100000, 'spread': 12},
            'BTCUSD': {'digits': 2, 'point': 0.01, 'contract_size': 1, 'spread': 500},
            'USOIL': {'digits': 2, 'point': 0.01, 'contract_size': 1000, 'spread': 40},
        }
        
        config = symbol_configs.get(symbol, {'digits': 5, 'point': 0.00001, 'contract_size': 100000, 'spread': 20})
        
        base_prices = {
            'XAUUSD': 2650.00, 'EURUSD': 1.0850, 'GBPUSD': 1.2700,
            'USDJPY': 148.50, 'BTCUSD': 95000.00, 'USOIL': 72.50,
        }
        
        price = base_prices.get(symbol, 1.0000)
        
        return {
            'name': symbol,
            'description': f'{symbol} Simulated',
            'point': config['point'],
            'digits': config['digits'],
            'spread': config['spread'],
            'contract_size': config['contract_size'],
            'volume_min': 0.01,
            'volume_max': 100.0,
            'volume_step': 0.01,
            'swap_long': -5.0,
            'swap_short': -5.0,
            'bid': price,
            'ask': price + config['spread'] * config['point'],
        }
    
    # ========================================================================
    # PRICE DATA
    # ========================================================================
    
    def get_current_price(self, symbol: str) -> Dict:
        """Get current bid/ask price for symbol."""
        if self.simulation_mode:
            return self._get_simulated_price(symbol)
        
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {}
            
            return {
                'symbol': symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'time': datetime.fromtimestamp(tick.time),
                'spread': round((tick.ask - tick.bid) / mt5.symbol_info(symbol).point)
            }
        except Exception as e:
            print(f"Error getting price: {e}")
            return {}
    
    def _get_simulated_price(self, symbol: str) -> Dict:
        """Get simulated price with random fluctuation."""
        info = self._get_simulated_symbol_info(symbol)
        bid = info['bid']
        
        # Add small random fluctuation
        fluctuation = np.random.normal(0, info['point'] * 10)
        bid += fluctuation
        ask = bid + info['spread'] * info['point']
        
        return {
            'symbol': symbol,
            'bid': round(bid, info['digits']),
            'ask': round(ask, info['digits']),
            'last': round(bid, info['digits']),
            'volume': np.random.randint(100, 1000),
            'time': datetime.now(),
            'spread': info['spread']
        }
    
    def get_historical_data(self, symbol: str, timeframe: str = 'M15',
                           count: int = 500) -> pd.DataFrame:
        """
        Get historical OHLCV data.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe string (M1, M5, M15, M30, H1, H4, D1, W1)
            count: Number of bars to retrieve
            
        Returns:
            DataFrame with columns: time, open, high, low, close, volume
        """
        if self.simulation_mode:
            return self._get_simulated_historical(symbol, timeframe, count)
        
        try:
            tf = TIMEFRAMES.get(timeframe, mt5.TIMEFRAME_M15)
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
            
            if rates is None or len(rates) == 0:
                return pd.DataFrame()
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
            df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            
            return df
            
        except Exception as e:
            print(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    def _get_simulated_historical(self, symbol: str, timeframe: str, 
                                   count: int) -> pd.DataFrame:
        """Generate simulated historical data."""
        info = self._get_simulated_symbol_info(symbol)
        price = info['bid']
        
        # Timeframe to minutes mapping
        tf_minutes = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080
        }
        minutes = tf_minutes.get(timeframe, 15)
        
        # Generate price series
        np.random.seed(42)  # Reproducible for same symbol
        returns = np.random.normal(0, 0.001, count)
        
        prices = [price]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        prices = prices[1:]
        
        # Generate OHLCV
        data = []
        now = datetime.now()
        
        for i, close in enumerate(prices):
            bar_time = now - timedelta(minutes=minutes * (count - i - 1))
            
            # Generate OHLC from close
            volatility = info['point'] * np.random.randint(5, 30)
            open_price = close + np.random.uniform(-volatility, volatility)
            high = max(open_price, close) + np.random.uniform(0, volatility)
            low = min(open_price, close) - np.random.uniform(0, volatility)
            volume = np.random.randint(100, 5000)
            
            data.append({
                'time': bar_time,
                'open': round(open_price, info['digits']),
                'high': round(high, info['digits']),
                'low': round(low, info['digits']),
                'close': round(close, info['digits']),
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    # ========================================================================
    # RETURNS CALCULATION (For Singularity Core)
    # ========================================================================
    
    def calculate_returns(self, symbol: str, timeframe: str = 'M15',
                         count: int = 200, return_type: str = 'pct') -> np.ndarray:
        """
        Calculate returns from historical data.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe string
            count: Number of bars
            return_type: 'pct' for percentage, 'r_multiple' for R-multiples
            
        Returns:
            Array of returns
        """
        df = self.get_historical_data(symbol, timeframe, count + 1)
        
        if df.empty or len(df) < 10:
            return np.array([])
        
        if return_type == 'pct':
            returns = df['close'].pct_change().dropna().values
        else:
            # Simulate R-multiples (for demo purposes)
            pct_returns = df['close'].pct_change().dropna().values
            # Scale to R-multiple range (assuming 1% = 0.5R typical)
            returns = pct_returns * 50  # Rough conversion
        
        return returns
    
    # ========================================================================
    # REAL-TIME STREAMING
    # ========================================================================
    
    def start_price_stream(self, symbols: List[str], callback, interval: float = 1.0):
        """
        Start streaming price updates.
        
        Args:
            symbols: List of symbols to stream
            callback: Function to call with price updates
            interval: Update interval in seconds
        """
        self._price_callbacks.append(callback)
        
        if self._price_thread is None or not self._price_thread.is_alive():
            self._running = True
            self._stream_symbols = symbols
            self._stream_interval = interval
            self._price_thread = threading.Thread(target=self._price_stream_loop, daemon=True)
            self._price_thread.start()
    
    def _price_stream_loop(self):
        """Background thread for price streaming."""
        while self._running:
            prices = {}
            for symbol in self._stream_symbols:
                price_data = self.get_current_price(symbol)
                if price_data:
                    prices[symbol] = price_data
            
            for callback in self._price_callbacks:
                try:
                    callback(prices)
                except Exception as e:
                    print(f"Price callback error: {e}")
            
            time.sleep(self._stream_interval)
    
    def _stop_price_stream(self):
        """Stop price streaming."""
        self._running = False
        if self._price_thread:
            self._price_thread.join(timeout=2.0)
    
    # ========================================================================
    # TRADE EXECUTION
    # ========================================================================
    
    def place_order(self, symbol: str, order_type: str, volume: float,
                   sl: float = None, tp: float = None, 
                   comment: str = "Singularity") -> Dict:
        """
        Place a trade order.
        
        Args:
            symbol: Symbol name
            order_type: 'buy' or 'sell'
            volume: Lot size
            sl: Stop loss price (optional)
            tp: Take profit price (optional)
            comment: Order comment
            
        Returns:
            Order result dictionary
        """
        if self.simulation_mode:
            return self._simulate_order(symbol, order_type, volume, sl, tp, comment)
        
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {'success': False, 'error': 'Symbol not found'}
            
            if not symbol_info.visible:
                mt5.symbol_select(symbol, True)
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {'success': False, 'error': 'Cannot get price'}
            
            price = tick.ask if order_type == 'buy' else tick.bid
            
            # Prepare request
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': symbol,
                'volume': volume,
                'type': mt5.ORDER_TYPE_BUY if order_type == 'buy' else mt5.ORDER_TYPE_SELL,
                'price': price,
                'deviation': 20,
                'magic': 234000,
                'comment': comment,
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }
            
            if sl:
                request['sl'] = sl
            if tp:
                request['tp'] = tp
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {'success': False, 'error': f'Order failed: {result.comment}'}
            
            return {
                'success': True,
                'ticket': result.order,
                'price': result.price,
                'volume': volume,
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _simulate_order(self, symbol: str, order_type: str, volume: float,
                        sl: float, tp: float, comment: str) -> Dict:
        """Simulate order execution."""
        price_data = self.get_current_price(symbol)
        price = price_data['ask'] if order_type == 'buy' else price_data['bid']
        
        return {
            'success': True,
            'ticket': np.random.randint(100000, 999999),
            'price': price,
            'volume': volume,
            'simulated': True,
        }
    
    def get_positions(self) -> List[Dict]:
        """Get open positions."""
        if self.simulation_mode:
            return []
        
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            result = []
            for pos in positions:
                result.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'buy' if pos.type == 0 else 'sell',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'profit': pos.profit,
                    'sl': pos.sl,
                    'tp': pos.tp,
                })
            
            return result
            
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []
    
    def close_position(self, ticket: int) -> Dict:
        """Close a position by ticket."""
        if self.simulation_mode:
            return {'success': True, 'message': 'Simulated close'}
        
        try:
            position = mt5.positions_get(ticket=ticket)
            if position is None or len(position) == 0:
                return {'success': False, 'error': 'Position not found'}
            
            pos = position[0]
            
            # Prepare close request
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': pos.symbol,
                'volume': pos.volume,
                'type': mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                'position': ticket,
                'price': mt5.symbol_info_tick(pos.symbol).bid if pos.type == 0 else mt5.symbol_info_tick(pos.symbol).ask,
                'deviation': 20,
                'magic': 234000,
                'comment': 'Close by Singularity',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {'success': False, 'error': result.comment}
            
            return {'success': True, 'message': 'Position closed'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================
_connector_instance = None

def get_connector() -> MT5Connector:
    """Get singleton MT5 connector instance."""
    global _connector_instance
    if _connector_instance is None:
        _connector_instance = MT5Connector()
    return _connector_instance


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("MT5 CONNECTOR TEST")
    print("=" * 80)
    
    connector = MT5Connector()
    
    # Connect
    success, message = connector.connect()
    print(f"\n{message}")
    
    if success:
        # Account info
        print("\n📊 ACCOUNT INFO:")
        info = connector.get_account_info()
        for k, v in info.items():
            print(f"   {k}: {v}")
        
        # Symbols
        symbols = connector.get_symbols()[:5]
        print(f"\n📈 SYMBOLS: {symbols}")
        
        # Price data
        print("\n💰 CURRENT PRICES:")
        for symbol in ['XAUUSD', 'EURUSD']:
            price = connector.get_current_price(symbol)
            print(f"   {symbol}: Bid={price.get('bid')}, Ask={price.get('ask')}")
        
        # Historical data
        print("\n📉 HISTORICAL DATA (XAUUSD M15, last 10 bars):")
        df = connector.get_historical_data('XAUUSD', 'M15', 10)
        print(df.to_string())
        
        # Returns for Singularity Core
        print("\n📊 RETURNS (for Singularity Core):")
        returns = connector.calculate_returns('XAUUSD', 'M15', 100)
        print(f"   Shape: {returns.shape}")
        print(f"   Mean: {np.mean(returns):.6f}")
        print(f"   Std: {np.std(returns):.6f}")
        
        # Disconnect
        connector.disconnect()
        print("\n✅ Disconnected")
