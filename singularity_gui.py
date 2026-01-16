#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║           SINGULARITY CAPITAL OS - HACKER THEME GUI                           ║
║                    Self-Evolving Capital Intelligence                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Professional GUI with hacker/cyberpunk theme and MT5 real-time data.
"""

# ============================================================================
# AUTO-INSTALL DEPENDENCIES
# ============================================================================
import subprocess
import sys
import os

def _ensure_dependencies():
    """Automatically install missing dependencies."""
    required = {
        'numpy': 'numpy>=1.21.0',
        'pandas': 'pandas>=1.3.0',
        'scipy': 'scipy>=1.7.0',
        'customtkinter': 'customtkinter',
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
                if package not in ['MetaTrader5']:  # MT5 optional
                    print(f"   ⚠️  Could not install {package}")

_ensure_dependencies()

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
import pandas as pd
import customtkinter as ctk
from datetime import datetime
import threading
import time
import random
import json
from typing import Dict, List, Optional

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


# ============================================================================
# HACKER THEME COLORS
# ============================================================================
class HackerTheme:
    """Cyberpunk/Hacker color scheme."""
    
    # Main colors
    BG_DARK = "#0a0a0a"
    BG_MEDIUM = "#121212"
    BG_LIGHT = "#1a1a1a"
    BG_PANEL = "#0d1117"
    
    # Accent colors (neon)
    NEON_GREEN = "#00ff41"
    NEON_CYAN = "#00d4ff"
    NEON_MAGENTA = "#ff00ff"
    NEON_YELLOW = "#ffff00"
    NEON_RED = "#ff0040"
    NEON_ORANGE = "#ff6600"
    
    # Status colors
    SUCCESS = "#00ff41"
    WARNING = "#ffcc00"
    ERROR = "#ff0040"
    INFO = "#00d4ff"
    
    # Text colors
    TEXT_PRIMARY = "#e0e0e0"
    TEXT_SECONDARY = "#888888"
    TEXT_DIM = "#505050"
    
    # Terminal colors
    TERMINAL_BG = "#0a0a0a"
    TERMINAL_TEXT = "#00ff41"
    
    # Chart colors
    CANDLE_UP = "#00ff41"
    CANDLE_DOWN = "#ff0040"
    
    # Gradient definitions
    GRADIENT_START = "#00ff41"
    GRADIENT_END = "#00d4ff"


# ============================================================================
# CUSTOM WIDGETS
# ============================================================================
class NeonLabel(ctk.CTkLabel):
    """Label with neon glow effect simulation."""
    
    def __init__(self, master, text="", color=HackerTheme.NEON_GREEN, **kwargs):
        super().__init__(
            master, 
            text=text,
            text_color=color,
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            **kwargs
        )


class TerminalText(ctk.CTkTextbox):
    """Terminal-style text display."""
    
    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            fg_color=HackerTheme.TERMINAL_BG,
            text_color=HackerTheme.TERMINAL_TEXT,
            font=ctk.CTkFont(family="Consolas", size=11),
            border_width=1,
            border_color=HackerTheme.NEON_GREEN,
            corner_radius=0,
            **kwargs
        )
    
    def append(self, text: str, color: str = None):
        """Append text to terminal."""
        self.configure(state="normal")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.insert("end", f"[{timestamp}] {text}\n")
        self.see("end")
        self.configure(state="disabled")
    
    def clear(self):
        """Clear terminal."""
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.configure(state="disabled")


class StatusIndicator(ctk.CTkFrame):
    """Status indicator with pulsing effect."""
    
    def __init__(self, master, label: str, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        
        self.indicator = ctk.CTkLabel(
            self, text="●", font=ctk.CTkFont(size=16),
            text_color=HackerTheme.TEXT_DIM
        )
        self.indicator.pack(side="left", padx=(0, 5))
        
        self.label = ctk.CTkLabel(
            self, text=label, text_color=HackerTheme.TEXT_SECONDARY,
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.label.pack(side="left")
        
        self._status = "inactive"
    
    def set_status(self, status: str):
        """Set status: 'active', 'warning', 'error', 'inactive'."""
        colors = {
            'active': HackerTheme.SUCCESS,
            'warning': HackerTheme.WARNING,
            'error': HackerTheme.ERROR,
            'inactive': HackerTheme.TEXT_DIM,
        }
        self.indicator.configure(text_color=colors.get(status, HackerTheme.TEXT_DIM))
        self._status = status


class DataPanel(ctk.CTkFrame):
    """Panel for displaying key-value data."""
    
    def __init__(self, master, title: str, **kwargs):
        super().__init__(
            master, 
            fg_color=HackerTheme.BG_PANEL,
            border_width=1,
            border_color=HackerTheme.NEON_CYAN,
            corner_radius=5,
            **kwargs
        )
        
        # Title
        self.title_label = ctk.CTkLabel(
            self, text=f"[ {title} ]",
            text_color=HackerTheme.NEON_CYAN,
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold")
        )
        self.title_label.pack(pady=(10, 5), padx=10, anchor="w")
        
        # Content frame
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        self.rows = {}
    
    def add_row(self, key: str, value: str = "-", highlight: bool = False):
        """Add a data row."""
        row = ctk.CTkFrame(self.content, fg_color="transparent")
        row.pack(fill="x", pady=2)
        
        key_label = ctk.CTkLabel(
            row, text=key, text_color=HackerTheme.TEXT_SECONDARY,
            font=ctk.CTkFont(family="Consolas", size=11),
            anchor="w", width=120
        )
        key_label.pack(side="left")
        
        value_color = HackerTheme.NEON_GREEN if highlight else HackerTheme.TEXT_PRIMARY
        value_label = ctk.CTkLabel(
            row, text=value, text_color=value_color,
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
            anchor="e"
        )
        value_label.pack(side="right")
        
        self.rows[key] = value_label
    
    def update_row(self, key: str, value: str, color: str = None):
        """Update a row value."""
        if key in self.rows:
            self.rows[key].configure(text=value)
            if color:
                self.rows[key].configure(text_color=color)


class PriceDisplay(ctk.CTkFrame):
    """Real-time price display widget."""
    
    def __init__(self, master, symbol: str, **kwargs):
        super().__init__(
            master,
            fg_color=HackerTheme.BG_PANEL,
            border_width=1,
            border_color=HackerTheme.NEON_GREEN,
            corner_radius=5,
            **kwargs
        )
        
        self.symbol = symbol
        self._last_price = 0
        
        # Symbol name
        self.symbol_label = ctk.CTkLabel(
            self, text=symbol,
            text_color=HackerTheme.NEON_CYAN,
            font=ctk.CTkFont(family="Consolas", size=14, weight="bold")
        )
        self.symbol_label.pack(pady=(10, 0))
        
        # Price
        self.price_label = ctk.CTkLabel(
            self, text="----.--",
            text_color=HackerTheme.NEON_GREEN,
            font=ctk.CTkFont(family="Consolas", size=24, weight="bold")
        )
        self.price_label.pack(pady=5)
        
        # Spread
        self.spread_label = ctk.CTkLabel(
            self, text="Spread: --",
            text_color=HackerTheme.TEXT_SECONDARY,
            font=ctk.CTkFont(family="Consolas", size=10)
        )
        self.spread_label.pack(pady=(0, 5))
        
        # Change indicator
        self.change_label = ctk.CTkLabel(
            self, text="",
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.change_label.pack(pady=(0, 10))
    
    def update_price(self, bid: float, ask: float, spread: int):
        """Update price display."""
        # Determine color based on price movement
        if bid > self._last_price:
            color = HackerTheme.CANDLE_UP
            arrow = "▲"
        elif bid < self._last_price:
            color = HackerTheme.CANDLE_DOWN
            arrow = "▼"
        else:
            color = HackerTheme.NEON_GREEN
            arrow = ""
        
        self.price_label.configure(text=f"{bid:.5g}", text_color=color)
        self.spread_label.configure(text=f"Spread: {spread}")
        self.change_label.configure(text=arrow, text_color=color)
        
        self._last_price = bid


# ============================================================================
# MAIN APPLICATION
# ============================================================================
class SingularityGUI(ctk.CTk):
    """Main Singularity Capital OS GUI Application."""
    
    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.title("⚡ SINGULARITY CAPITAL OS v1.0 - HACKER EDITION ⚡")
        self.geometry("1400x900")
        self.minsize(1200, 700)
        self.configure(fg_color=HackerTheme.BG_DARK)
        
        # State
        self.mt5_connected = False
        self.running = True
        self.update_interval = 1000  # ms
        self.watched_symbols = []  # Will be populated from MT5
        self.default_symbols = ['XAUUSD', 'EURUSD', 'GBPUSD', 'BTCUSD']  # Fallback
        self.price_widgets = {}
        self.prices_frame = None  # Will hold reference to prices frame
        
        # Import modules (after GUI init)
        self._import_modules()
        
        # Build UI
        self._create_header()
        self._create_main_content()
        self._create_footer()
        
        # Initialize
        self._init_terminal()
        
        # Start matrix animation
        self._start_matrix_animation()
        
        # Bind close event
        self.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _import_modules(self):
        """Import trading modules."""
        try:
            from mt5_connector import MT5Connector, get_connector
            self.connector = get_connector()
            self._log("MT5 Connector loaded", "success")
        except Exception as e:
            self.connector = None
            self._log(f"MT5 Connector unavailable: {e}", "warning")
        
        try:
            from singularity_core import SingularityCore
            self.core = SingularityCore()
            self._log("Singularity Core loaded", "success")
        except Exception as e:
            self.core = None
            self._log(f"Core unavailable: {e}", "warning")
    
    # ========================================================================
    # UI CREATION
    # ========================================================================
    
    def _create_header(self):
        """Create header section."""
        self.header = ctk.CTkFrame(self, fg_color=HackerTheme.BG_MEDIUM, height=80)
        self.header.pack(fill="x", padx=10, pady=(10, 5))
        self.header.pack_propagate(False)
        
        # Left side - Logo and title
        left_frame = ctk.CTkFrame(self.header, fg_color="transparent")
        left_frame.pack(side="left", padx=20, pady=10)
        
        self.ascii_title = ctk.CTkLabel(
            left_frame,
            text="⚡ SINGULARITY",
            font=ctk.CTkFont(family="Consolas", size=28, weight="bold"),
            text_color=HackerTheme.NEON_GREEN
        )
        self.ascii_title.pack(anchor="w")
        
        self.subtitle = ctk.CTkLabel(
            left_frame,
            text="CAPITAL OS // SELF-EVOLVING INTELLIGENCE",
            font=ctk.CTkFont(family="Consolas", size=10),
            text_color=HackerTheme.NEON_CYAN
        )
        self.subtitle.pack(anchor="w")
        
        # Right side - Status indicators
        right_frame = ctk.CTkFrame(self.header, fg_color="transparent")
        right_frame.pack(side="right", padx=20, pady=10)
        
        self.mt5_status = StatusIndicator(right_frame, "MT5 CONNECTION")
        self.mt5_status.pack(anchor="e", pady=2)
        
        self.core_status = StatusIndicator(right_frame, "CORE ENGINE")
        self.core_status.pack(anchor="e", pady=2)
        self.core_status.set_status("active" if self.core else "inactive")
        
        self.data_status = StatusIndicator(right_frame, "DATA STREAM")
        self.data_status.pack(anchor="e", pady=2)
        
        # Center - Time display
        center_frame = ctk.CTkFrame(self.header, fg_color="transparent")
        center_frame.pack(side="left", expand=True, pady=10)
        
        self.time_label = ctk.CTkLabel(
            center_frame,
            text="00:00:00",
            font=ctk.CTkFont(family="Consolas", size=32, weight="bold"),
            text_color=HackerTheme.NEON_MAGENTA
        )
        self.time_label.pack()
        
        self.date_label = ctk.CTkLabel(
            center_frame,
            text="YYYY-MM-DD",
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color=HackerTheme.TEXT_SECONDARY
        )
        self.date_label.pack()
        
        # Start time update
        self._update_time()
    
    def _create_main_content(self):
        """Create main content area."""
        self.main = ctk.CTkFrame(self, fg_color="transparent")
        self.main.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Configure grid
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_columnconfigure(1, weight=2)
        self.main.grid_columnconfigure(2, weight=1)
        self.main.grid_rowconfigure(0, weight=1)
        
        # Left panel - Controls & Connection
        self._create_left_panel()
        
        # Center panel - Analysis & Charts
        self._create_center_panel()
        
        # Right panel - Signals & Terminal
        self._create_right_panel()
    
    def _create_left_panel(self):
        """Create left control panel."""
        left = ctk.CTkFrame(self.main, fg_color=HackerTheme.BG_MEDIUM, corner_radius=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # Connection section
        conn_panel = DataPanel(left, "MT5 CONNECTION")
        conn_panel.pack(fill="x", padx=10, pady=10)
        
        # Connect button
        self.connect_btn = ctk.CTkButton(
            conn_panel.content,
            text="⚡ CONNECT MT5",
            fg_color=HackerTheme.NEON_GREEN,
            text_color=HackerTheme.BG_DARK,
            hover_color=HackerTheme.NEON_CYAN,
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            command=self._toggle_connection
        )
        self.connect_btn.pack(fill="x", pady=10)
        
        # Account panel
        self.account_panel = DataPanel(left, "ACCOUNT INFO")
        self.account_panel.pack(fill="x", padx=10, pady=5)
        self.account_panel.add_row("Login:", "-")
        self.account_panel.add_row("Balance:", "-", highlight=True)
        self.account_panel.add_row("Equity:", "-", highlight=True)
        self.account_panel.add_row("Margin:", "-")
        self.account_panel.add_row("Free Margin:", "-")
        self.account_panel.add_row("Leverage:", "-")
        self.account_panel.add_row("Mode:", "-")
        
        # Symbol selector
        symbol_panel = DataPanel(left, "SYMBOL SELECTOR")
        symbol_panel.pack(fill="x", padx=10, pady=5)
        
        self.symbol_var = ctk.StringVar(value="")
        self.symbol_combo = ctk.CTkComboBox(
            symbol_panel.content,
            values=["Connect MT5 to load symbols..."],
            variable=self.symbol_var,
            fg_color=HackerTheme.BG_DARK,
            button_color=HackerTheme.NEON_GREEN,
            border_color=HackerTheme.NEON_GREEN,
            dropdown_fg_color=HackerTheme.BG_DARK,
            font=ctk.CTkFont(family="Consolas", size=12),
            command=self._on_symbol_change,
            state="disabled"
        )
        self.symbol_combo.pack(fill="x", pady=5)
        
        # Refresh symbols button
        self.refresh_symbols_btn = ctk.CTkButton(
            symbol_panel.content,
            text="🔄 REFRESH SYMBOLS",
            fg_color=HackerTheme.BG_DARK,
            text_color=HackerTheme.NEON_CYAN,
            border_width=1,
            border_color=HackerTheme.NEON_CYAN,
            hover_color=HackerTheme.NEON_CYAN,
            font=ctk.CTkFont(family="Consolas", size=10),
            height=25,
            command=self._refresh_symbols,
            state="disabled"
        )
        self.refresh_symbols_btn.pack(fill="x", pady=2)
        
        # Timeframe selector
        self.tf_var = ctk.StringVar(value="M15")
        self.tf_combo = ctk.CTkComboBox(
            symbol_panel.content,
            values=["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
            variable=self.tf_var,
            fg_color=HackerTheme.BG_DARK,
            button_color=HackerTheme.NEON_CYAN,
            border_color=HackerTheme.NEON_CYAN,
            dropdown_fg_color=HackerTheme.BG_DARK,
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.tf_combo.pack(fill="x", pady=5)
        
        # Analyze button
        self.analyze_btn = ctk.CTkButton(
            symbol_panel.content,
            text="🔍 ANALYZE",
            fg_color=HackerTheme.NEON_CYAN,
            text_color=HackerTheme.BG_DARK,
            hover_color=HackerTheme.NEON_MAGENTA,
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            command=self._run_analysis
        )
        self.analyze_btn.pack(fill="x", pady=10)
        
        # Settings panel
        settings_panel = DataPanel(left, "SETTINGS")
        settings_panel.pack(fill="x", padx=10, pady=5)
        
        self.auto_refresh_var = ctk.BooleanVar(value=True)
        self.auto_refresh = ctk.CTkCheckBox(
            settings_panel.content,
            text="Auto Refresh",
            variable=self.auto_refresh_var,
            fg_color=HackerTheme.NEON_GREEN,
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.auto_refresh.pack(anchor="w", pady=2)
        
        self.sound_var = ctk.BooleanVar(value=False)
        self.sound_check = ctk.CTkCheckBox(
            settings_panel.content,
            text="Sound Alerts",
            variable=self.sound_var,
            fg_color=HackerTheme.NEON_GREEN,
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.sound_check.pack(anchor="w", pady=2)
    
    def _create_center_panel(self):
        """Create center analysis panel."""
        center = ctk.CTkFrame(self.main, fg_color=HackerTheme.BG_MEDIUM, corner_radius=10)
        center.grid(row=0, column=1, sticky="nsew", padx=5)
        
        # Price displays row
        self.prices_frame = ctk.CTkFrame(center, fg_color="transparent")
        self.prices_frame.pack(fill="x", padx=10, pady=10)
        
        # Empty placeholder - will be populated after MT5 connection
        self.prices_placeholder = ctk.CTkLabel(
            self.prices_frame,
            text="Connect to MT5 to view real-time prices...",
            text_color=HackerTheme.TEXT_SECONDARY,
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.prices_placeholder.pack(pady=30)
        
        # Analysis results
        self.analysis_panel = DataPanel(center, "ANALYSIS RESULTS")
        self.analysis_panel.pack(fill="x", padx=10, pady=5)
        
        # Edge info
        self.analysis_panel.add_row("Win Rate:", "-")
        self.analysis_panel.add_row("Expectancy:", "-", highlight=True)
        self.analysis_panel.add_row("Edge (LB):", "-", highlight=True)
        self.analysis_panel.add_row("Avg Win:", "-")
        self.analysis_panel.add_row("Avg Loss:", "-")
        
        # Regime info
        self.regime_panel = DataPanel(center, "MARKET REGIME")
        self.regime_panel.pack(fill="x", padx=10, pady=5)
        self.regime_panel.add_row("Regime:", "-")
        self.regime_panel.add_row("Volatility:", "-")
        self.regime_panel.add_row("Drift Detected:", "-")
        
        # Risk info
        self.risk_panel = DataPanel(center, "DYNAMIC RISK")
        self.risk_panel.pack(fill="x", padx=10, pady=5)
        self.risk_panel.add_row("Kelly:", "-")
        self.risk_panel.add_row("Risk %:", "-", highlight=True)
        self.risk_panel.add_row("Regime Factor:", "-")
        self.risk_panel.add_row("DD Factor:", "-")
        
        # Kill switch
        self.kill_panel = DataPanel(center, "KILL SWITCH")
        self.kill_panel.pack(fill="x", padx=10, pady=5)
        self.kill_panel.add_row("Status:", "ARMED", highlight=True)
        self.kill_panel.add_row("Reason:", "All Clear")
        
        # Monte Carlo
        self.mc_panel = DataPanel(center, "MONTE CARLO (95%)")
        self.mc_panel.pack(fill="x", padx=10, pady=5)
        self.mc_panel.add_row("Max DD:", "-")
        self.mc_panel.add_row("Lose Streak:", "-")
        self.mc_panel.add_row("Final Equity 5%:", "-")
    
    def _create_right_panel(self):
        """Create right signals and terminal panel."""
        right = ctk.CTkFrame(self.main, fg_color=HackerTheme.BG_MEDIUM, corner_radius=10)
        right.grid(row=0, column=2, sticky="nsew", padx=(5, 0))
        
        # Signals panel
        signals_title = ctk.CTkLabel(
            right, text="[ TRADING SIGNALS ]",
            text_color=HackerTheme.NEON_MAGENTA,
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold")
        )
        signals_title.pack(pady=(10, 5), padx=10, anchor="w")
        
        self.signals_frame = ctk.CTkFrame(right, fg_color=HackerTheme.BG_PANEL, corner_radius=5)
        self.signals_frame.pack(fill="x", padx=10, pady=5)
        
        self.signal_label = ctk.CTkLabel(
            self.signals_frame,
            text="⏳ Waiting for analysis...",
            text_color=HackerTheme.TEXT_SECONDARY,
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.signal_label.pack(pady=20)
        
        # Terminal
        terminal_title = ctk.CTkLabel(
            right, text="[ SYSTEM TERMINAL ]",
            text_color=HackerTheme.NEON_GREEN,
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold")
        )
        terminal_title.pack(pady=(15, 5), padx=10, anchor="w")
        
        self.terminal = TerminalText(right, height=300)
        self.terminal.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Clear button
        clear_btn = ctk.CTkButton(
            right,
            text="CLEAR LOG",
            fg_color=HackerTheme.BG_DARK,
            text_color=HackerTheme.NEON_RED,
            border_width=1,
            border_color=HackerTheme.NEON_RED,
            hover_color=HackerTheme.NEON_RED,
            font=ctk.CTkFont(family="Consolas", size=10),
            height=25,
            command=self.terminal.clear
        )
        clear_btn.pack(fill="x", padx=10, pady=(0, 10))
    
    def _create_footer(self):
        """Create footer section."""
        self.footer = ctk.CTkFrame(self, fg_color=HackerTheme.BG_MEDIUM, height=30)
        self.footer.pack(fill="x", padx=10, pady=(5, 10))
        self.footer.pack_propagate(False)
        
        # Matrix animation label
        self.matrix_label = ctk.CTkLabel(
            self.footer,
            text="",
            font=ctk.CTkFont(family="Consolas", size=10),
            text_color=HackerTheme.NEON_GREEN
        )
        self.matrix_label.pack(side="left", padx=10)
        
        # Version
        version_label = ctk.CTkLabel(
            self.footer,
            text="v1.0.0 // HACKER EDITION",
            font=ctk.CTkFont(family="Consolas", size=10),
            text_color=HackerTheme.TEXT_DIM
        )
        version_label.pack(side="right", padx=10)
    
    # ========================================================================
    # FUNCTIONALITY
    # ========================================================================
    
    def _init_terminal(self):
        """Initialize terminal with startup messages."""
        self._log("SINGULARITY CAPITAL OS INITIALIZED", "success")
        self._log("System ready. Connect to MT5 to begin.", "info")
    
    def _log(self, message: str, level: str = "info"):
        """Log message to terminal."""
        prefix = {
            'success': '[✓]',
            'error': '[✗]',
            'warning': '[!]',
            'info': '[i]',
        }
        if hasattr(self, 'terminal'):
            self.terminal.append(f"{prefix.get(level, '[>]')} {message}")
    
    def _update_time(self):
        """Update time display."""
        now = datetime.now()
        self.time_label.configure(text=now.strftime("%H:%M:%S"))
        self.date_label.configure(text=now.strftime("%Y-%m-%d // %A").upper())
        self.after(1000, self._update_time)
    
    def _start_matrix_animation(self):
        """Start matrix-style animation in footer."""
        chars = "01アイウエオカキクケコサシスセソタチツテト"
        
        def animate():
            if self.running:
                text = ''.join(random.choice(chars) for _ in range(60))
                self.matrix_label.configure(text=text)
                self.after(100, animate)
        
        animate()
    
    def _toggle_connection(self):
        """Toggle MT5 connection."""
        if not self.connector:
            self._log("MT5 Connector not available", "error")
            return
        
        if self.mt5_connected:
            # Disconnect
            success, message = self.connector.disconnect()
            if success:
                self.mt5_connected = False
                self.connect_btn.configure(text="⚡ CONNECT MT5", fg_color=HackerTheme.NEON_GREEN)
                self.mt5_status.set_status("inactive")
                self.data_status.set_status("inactive")
                self._log(message, "info")
        else:
            # Connect
            self._log("Connecting to MT5...", "info")
            success, message = self.connector.connect()
            if success:
                self.mt5_connected = True
                self.connect_btn.configure(text="⛔ DISCONNECT", fg_color=HackerTheme.NEON_RED)
                self.mt5_status.set_status("active")
                self._log(message, "success")
                
                # Load real data from MT5
                self._update_account_info()
                self._load_mt5_symbols()
                self._start_data_stream()
            else:
                self._log(message, "error")
    
    def _update_account_info(self):
        """Update account information display."""
        if not self.connector:
            return
        
        info = self.connector.get_account_info()
        
        self.account_panel.update_row("Login:", str(info.get('login', '-')))
        self.account_panel.update_row("Balance:", f"${info.get('balance', 0):,.2f}")
        self.account_panel.update_row("Equity:", f"${info.get('equity', 0):,.2f}")
        self.account_panel.update_row("Margin:", f"${info.get('margin', 0):,.2f}")
        self.account_panel.update_row("Free Margin:", f"${info.get('free_margin', 0):,.2f}")
        self.account_panel.update_row("Leverage:", f"1:{info.get('leverage', 0)}")
        self.account_panel.update_row("Mode:", info.get('trade_mode', '-'))
    
    def _load_mt5_symbols(self):
        """Load available symbols from MT5."""
        if not self.connector:
            return
        
        self._log("Loading symbols from MT5...", "info")
        
        # Get all symbols from MT5
        all_symbols = self.connector.get_symbols()
        
        if not all_symbols:
            self._log("No symbols found, using defaults", "warning")
            all_symbols = self.default_symbols
        
        # Filter to common trading symbols (prioritize popular ones)
        priority_symbols = ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSD', 
                           'USOIL', 'US30', 'NAS100', 'US500', 'GER40',
                           'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURGBP']
        
        # Build ordered list with priority symbols first
        ordered_symbols = []
        for sym in priority_symbols:
            if sym in all_symbols:
                ordered_symbols.append(sym)
        
        # Add remaining symbols
        for sym in all_symbols:
            if sym not in ordered_symbols:
                ordered_symbols.append(sym)
        
        self.watched_symbols = ordered_symbols
        
        # Update symbol dropdown
        self.symbol_combo.configure(values=self.watched_symbols, state="normal")
        self.refresh_symbols_btn.configure(state="normal")
        
        if self.watched_symbols:
            self.symbol_var.set(self.watched_symbols[0])
        
        self._log(f"Loaded {len(self.watched_symbols)} symbols from MT5", "success")
        
        # Create price widgets for first 4 symbols
        self._create_price_widgets()
    
    def _create_price_widgets(self):
        """Create or update price display widgets."""
        # Clear existing widgets
        for widget in self.prices_frame.winfo_children():
            widget.destroy()
        self.price_widgets.clear()
        
        # Get first 4 watched symbols
        display_symbols = self.watched_symbols[:4] if self.watched_symbols else self.default_symbols[:4]
        
        for symbol in display_symbols:
            price_widget = PriceDisplay(self.prices_frame, symbol)
            price_widget.pack(side="left", expand=True, fill="both", padx=5)
            self.price_widgets[symbol] = price_widget
    
    def _refresh_symbols(self):
        """Refresh symbol list from MT5."""
        if self.mt5_connected:
            self._load_mt5_symbols()
            self._log("Symbols refreshed", "success")
    
    def _start_data_stream(self):
        """Start real-time data streaming."""
        if not self.connector:
            return
        
        self.data_status.set_status("active")
        self._log("Data stream started", "success")
        
        def update_prices():
            while self.running and self.mt5_connected:
                # Update prices for displayed symbols
                for symbol, widget in self.price_widgets.items():
                    price = self.connector.get_current_price(symbol)
                    if price:
                        # Use after() to safely update from background thread
                        self.after(0, lambda w=widget, p=price: w.update_price(
                            p.get('bid', 0),
                            p.get('ask', 0),
                            p.get('spread', 0)
                        ))
                time.sleep(1)
        
        threading.Thread(target=update_prices, daemon=True).start()
    
    def _on_symbol_change(self, choice):
        """Handle symbol selection change."""
        self._log(f"Symbol changed to {choice}", "info")
    
    def _run_analysis(self):
        """Run Singularity Core analysis."""
        symbol = self.symbol_var.get()
        timeframe = self.tf_var.get()
        
        self._log(f"Analyzing {symbol} on {timeframe}...", "info")
        
        if not self.connector or not self.core:
            self._log("Connector or Core not available", "error")
            return
        
        # Get returns data
        returns = self.connector.calculate_returns(symbol, timeframe, 200, 'r_multiple')
        
        if len(returns) < 10:
            self._log("Insufficient data for analysis", "warning")
            return
        
        # Run analysis
        try:
            results = self.core.analyze(returns)
            self._display_results(results)
            self._log("Analysis complete!", "success")
        except Exception as e:
            self._log(f"Analysis error: {e}", "error")
    
    def _display_results(self, results: Dict):
        """Display analysis results in panels."""
        # Edge info
        edge = results.get('edge', {})
        self.analysis_panel.update_row("Win Rate:", f"{edge.get('win_rate_mean', 0)*100:.1f}%")
        self.analysis_panel.update_row("Expectancy:", f"{edge.get('expectancy_mean', 0):.4f}R")
        
        edge_lb = edge.get('expectancy_lb', 0)
        edge_color = HackerTheme.SUCCESS if edge_lb > 0 else HackerTheme.ERROR
        self.analysis_panel.update_row("Edge (LB):", f"{edge_lb:.4f}R", edge_color)
        
        self.analysis_panel.update_row("Avg Win:", f"{edge.get('avg_win', 0):.4f}R")
        self.analysis_panel.update_row("Avg Loss:", f"{edge.get('avg_loss', 0):.4f}R")
        
        # Regime info
        regime = results.get('regime', {})
        regime_names = {0: "TREND", 1: "CHOP", 2: "PANIC"}
        regime_colors = {0: HackerTheme.SUCCESS, 1: HackerTheme.WARNING, 2: HackerTheme.ERROR}
        regime_id = regime.get('id', 1)
        self.regime_panel.update_row("Regime:", regime_names.get(regime_id, "?"), regime_colors.get(regime_id))
        self.regime_panel.update_row("Volatility:", f"{regime.get('volatility', 0):.4f}")
        
        drift = results.get('drift_detected', False)
        self.regime_panel.update_row("Drift Detected:", "YES" if drift else "NO",
                                     HackerTheme.ERROR if drift else HackerTheme.SUCCESS)
        
        # Risk info
        risk = results.get('risk', {})
        self.risk_panel.update_row("Kelly:", f"{risk.get('kelly', 0):.4f}")
        self.risk_panel.update_row("Risk %:", f"{risk.get('risk_pct', 0)*100:.4f}%")
        self.risk_panel.update_row("Regime Factor:", f"{risk.get('regime_factor', 0):.2f}")
        self.risk_panel.update_row("DD Factor:", f"{risk.get('dd_factor', 0):.2f}")
        
        # Kill switch
        kill = results.get('kill_switch', {})
        triggered = kill.get('triggered', False)
        self.kill_panel.update_row("Status:", 
                                   "🔴 TRIGGERED" if triggered else "🟢 ARMED",
                                   HackerTheme.ERROR if triggered else HackerTheme.SUCCESS)
        self.kill_panel.update_row("Reason:", kill.get('reason', 'N/A'))
        
        # Monte Carlo
        mc = results.get('monte_carlo', {})
        self.mc_panel.update_row("Max DD:", f"{mc.get('max_dd_95', 0):.2f}R")
        self.mc_panel.update_row("Lose Streak:", f"{mc.get('losing_streak_95', 0):.0f}")
        self.mc_panel.update_row("Final Equity 5%:", f"{mc.get('final_equity_5', 0):.2f}R")
        
        # Update signal display
        self._update_signal_display(results)
    
    def _update_signal_display(self, results: Dict):
        """Update trading signal display."""
        kill = results.get('kill_switch', {})
        regime = results.get('regime', {})
        edge = results.get('edge', {})
        
        # Clear existing
        for widget in self.signals_frame.winfo_children():
            widget.destroy()
        
        if kill.get('triggered', False):
            signal_text = "⛔ DO NOT TRADE"
            signal_color = HackerTheme.ERROR
            reason = kill.get('reason', '')
        elif regime.get('id') == 2:
            signal_text = "⚠️ HIGH RISK MODE"
            signal_color = HackerTheme.WARNING
            reason = "Panic regime detected"
        elif edge.get('expectancy_lb', 0) <= 0:
            signal_text = "⚠️ NO EDGE"
            signal_color = HackerTheme.WARNING
            reason = "No statistical edge"
        else:
            signal_text = "✅ CLEAR TO TRADE"
            signal_color = HackerTheme.SUCCESS
            reason = f"Risk: {results.get('risk', {}).get('risk_pct', 0)*100:.2f}%"
        
        signal = ctk.CTkLabel(
            self.signals_frame,
            text=signal_text,
            text_color=signal_color,
            font=ctk.CTkFont(family="Consolas", size=18, weight="bold")
        )
        signal.pack(pady=(15, 5))
        
        reason_label = ctk.CTkLabel(
            self.signals_frame,
            text=reason,
            text_color=HackerTheme.TEXT_SECONDARY,
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        reason_label.pack(pady=(0, 15))
    
    def _on_close(self):
        """Handle window close."""
        self.running = False
        if self.connector and self.mt5_connected:
            self.connector.disconnect()
        self.destroy()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    """Main entry point."""
    app = SingularityGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
