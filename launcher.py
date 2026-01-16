#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║           SINGULARITY CAPITAL OS - UNIFIED LAUNCHER                          ║
║                    Self-Evolving Capital Intelligence                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Professional launcher with auto-dependency installation and interactive menu.

Usage:
    python launcher.py              # Interactive menu
    python launcher.py --demo       # Quick demo mode
    python launcher.py --install    # Just install dependencies
"""

import subprocess
import sys
import os
import importlib
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

REQUIRED_PACKAGES = {
    'numpy': {'import_name': 'numpy', 'pip_name': 'numpy', 'min_version': '1.21.0'},
    'pandas': {'import_name': 'pandas', 'pip_name': 'pandas', 'min_version': '1.3.0'},
    'scipy': {'import_name': 'scipy', 'pip_name': 'scipy', 'min_version': '1.7.0'},
    'customtkinter': {'import_name': 'customtkinter', 'pip_name': 'customtkinter', 'min_version': '5.0.0'},
}

# Terminal colors (ANSI codes work on most terminals)
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Enable colors on Windows
if sys.platform == 'win32':
    os.system('color')

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_banner():
    """Print the Singularity Capital OS banner"""
    banner = f"""
{Colors.OKCYAN}╔═══════════════════════════════════════════════════════════════════════════════╗
║{Colors.HEADER}           ███████╗██╗███╗   ██╗ ██████╗ ██╗   ██╗██╗      █████╗ ██████╗     {Colors.OKCYAN}║
║{Colors.HEADER}           ██╔════╝██║████╗  ██║██╔════╝ ██║   ██║██║     ██╔══██╗██╔══██╗    {Colors.OKCYAN}║
║{Colors.HEADER}           ███████╗██║██╔██╗ ██║██║  ███╗██║   ██║██║     ███████║██████╔╝    {Colors.OKCYAN}║
║{Colors.HEADER}           ╚════██║██║██║╚██╗██║██║   ██║██║   ██║██║     ██╔══██║██╔══██╗    {Colors.OKCYAN}║
║{Colors.HEADER}           ███████║██║██║ ╚████║╚██████╔╝╚██████╔╝███████╗██║  ██║██║  ██║    {Colors.OKCYAN}║
║{Colors.HEADER}           ╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝    {Colors.OKCYAN}║
║                                                                                   ║
║{Colors.OKGREEN}                    CAPITAL OS v1.0 - Self-Evolving Intelligence                 {Colors.OKCYAN}║
╚═══════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
    print(banner)


def print_status(message, status='info'):
    """Print formatted status message"""
    icons = {
        'info': f'{Colors.OKBLUE}ℹ️ ',
        'success': f'{Colors.OKGREEN}✅',
        'warning': f'{Colors.WARNING}⚠️ ',
        'error': f'{Colors.FAIL}❌',
        'progress': f'{Colors.OKCYAN}🔄',
        'rocket': f'{Colors.OKGREEN}🚀',
    }
    icon = icons.get(status, icons['info'])
    print(f"{icon} {message}{Colors.ENDC}")


def print_progress(iterable, prefix='', length=50):
    """Print progress bar"""
    total = len(iterable)
    for i, item in enumerate(iterable):
        percent = (i + 1) / total
        filled = int(length * percent)
        bar = '█' * filled + '░' * (length - filled)
        print(f'\r{Colors.OKBLUE}{prefix} |{bar}| {percent*100:.1f}%{Colors.ENDC}', end='')
        yield item
    print()


# ============================================================================
# DEPENDENCY MANAGEMENT
# ============================================================================

def check_package(package_info):
    """Check if a package is installed and meets version requirements"""
    try:
        module = importlib.import_module(package_info['import_name'])
        version = getattr(module, '__version__', '0.0.0')
        return True, version
    except ImportError:
        return False, None


def install_package(pip_name, quiet=False):
    """Install a package using pip"""
    cmd = [sys.executable, '-m', 'pip', 'install', pip_name]
    if quiet:
        cmd.extend(['--quiet', '--disable-pip-version-check'])
    
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL if quiet else None,
                             stderr=subprocess.DEVNULL if quiet else None)
        return True
    except subprocess.CalledProcessError:
        return False


def ensure_dependencies(show_progress=True):
    """Check and install all required dependencies"""
    if show_progress:
        print(f"\n{Colors.BOLD}📦 Checking Dependencies...{Colors.ENDC}\n")
    
    missing = []
    installed = []
    
    for name, info in REQUIRED_PACKAGES.items():
        is_installed, version = check_package(info)
        if is_installed:
            if show_progress:
                print_status(f"{name} v{version} - OK", 'success')
            installed.append(name)
        else:
            missing.append((name, info))
    
    if missing:
        print(f"\n{Colors.WARNING}📥 Installing missing packages...{Colors.ENDC}\n")
        
        for name, info in missing:
            print_status(f"Installing {name}...", 'progress')
            pip_spec = f"{info['pip_name']}>={info['min_version']}"
            
            if install_package(pip_spec, quiet=True):
                print_status(f"{name} installed successfully!", 'success')
                installed.append(name)
            else:
                print_status(f"Failed to install {name}!", 'error')
                print(f"   Try: pip install {pip_spec}")
    
    all_ok = len(installed) == len(REQUIRED_PACKAGES)
    
    if show_progress:
        if all_ok:
            print(f"\n{Colors.OKGREEN}✅ All dependencies satisfied!{Colors.ENDC}\n")
        else:
            print(f"\n{Colors.FAIL}⚠️  Some dependencies could not be installed.{Colors.ENDC}\n")
    
    return all_ok


# ============================================================================
# MODULE RUNNERS
# ============================================================================

def run_backtest():
    """Run backtest mode"""
    print_status("Starting Backtest Mode...", 'rocket')
    try:
        from Complete_Trading_System import main
        sys.argv = ['Complete_Trading_System.py', '--mode', 'backtest']
        main()
    except Exception as e:
        print_status(f"Backtest error: {e}", 'error')


def run_live():
    """Run live monitoring mode"""
    print_status("Starting Live Monitor Mode...", 'rocket')
    try:
        from Complete_Trading_System import main
        sys.argv = ['Complete_Trading_System.py', '--mode', 'live']
        main()
    except Exception as e:
        print_status(f"Live monitor error: {e}", 'error')


def run_demo():
    """Run demo mode"""
    print_status("Starting Demo Mode...", 'rocket')
    try:
        from Complete_Trading_System import main
        sys.argv = ['Complete_Trading_System.py', '--mode', 'demo']
        main()
    except Exception as e:
        print_status(f"Demo error: {e}", 'error')


def run_xauusd_agent():
    """Run XAUUSD specialized agent"""
    print_status("Starting XAUUSD Agent...", 'rocket')
    try:
        # Change to script directory to allow local imports
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        import xauusd_agent
        import importlib
        importlib.reload(xauusd_agent)
    except Exception as e:
        print_status(f"XAUUSD Agent error: {e}", 'error')


def run_multi_agent():
    """Run multi-agent system"""
    print_status("Starting Multi-Agent System...", 'rocket')
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        import multi_agent_system
        import importlib
        importlib.reload(multi_agent_system)
    except Exception as e:
        print_status(f"Multi-Agent error: {e}", 'error')


def run_core_test():
    """Test core engine"""
    print_status("Testing Singularity Core Engine...", 'rocket')
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        import singularity_core
        import importlib
        importlib.reload(singularity_core)
    except Exception as e:
        print_status(f"Core test error: {e}", 'error')


def run_gui():
    """Launch the Hacker-themed GUI"""
    print_status("Launching Singularity GUI (Hacker Edition)...", 'rocket')
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        from singularity_gui import main
        main()
    except Exception as e:
        print_status(f"GUI error: {e}", 'error')


def run_mt5_test():
    """Test MT5 connection"""
    print_status("Testing MT5 Connector...", 'rocket')
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        import mt5_connector
        import importlib
        importlib.reload(mt5_connector)
    except Exception as e:
        print_status(f"MT5 test error: {e}", 'error')


# ============================================================================
# MENU SYSTEM
# ============================================================================

def show_menu():
    """Display interactive menu"""
    menu = f"""
{Colors.BOLD}╔════════════════════════════════════════════════════════════╗
║                    🎯 MAIN MENU                              ║
╠════════════════════════════════════════════════════════════╣
║                                                              ║
║   {Colors.HEADER}[G]{Colors.ENDC}{Colors.BOLD}  🖥️  LAUNCH HACKER GUI (MT5 + Real Data)            ║
║                                                              ║
║   {Colors.OKGREEN}[1]{Colors.ENDC}{Colors.BOLD}  📊 Run Backtest Mode                                ║
║   {Colors.OKGREEN}[2]{Colors.ENDC}{Colors.BOLD}  🔴 Run Live Monitor Mode                             ║
║   {Colors.OKGREEN}[3]{Colors.ENDC}{Colors.BOLD}  🎮 Run Demo Mode                                     ║
║   {Colors.OKGREEN}[4]{Colors.ENDC}{Colors.BOLD}  💎 Run XAUUSD Agent                                  ║
║   {Colors.OKGREEN}[5]{Colors.ENDC}{Colors.BOLD}  🧬 Run Multi-Agent System                            ║
║   {Colors.OKGREEN}[6]{Colors.ENDC}{Colors.BOLD}  🧪 Test Core Engine                                  ║
║   {Colors.OKGREEN}[7]{Colors.ENDC}{Colors.BOLD}  🔌 Test MT5 Connection                               ║
║   {Colors.OKGREEN}[8]{Colors.ENDC}{Colors.BOLD}  📦 Check/Install Dependencies                        ║
║   {Colors.FAIL}[0]{Colors.ENDC}{Colors.BOLD}  🚪 Exit                                              ║
║                                                              ║
╚════════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
    print(menu)


def interactive_menu():
    """Run interactive menu loop"""
    while True:
        show_menu()
        
        try:
            choice = input(f"{Colors.OKCYAN}Enter your choice [G, 0-8]: {Colors.ENDC}").strip().upper()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Colors.WARNING}Goodbye! 👋{Colors.ENDC}")
            break
        
        if choice == '0':
            print(f"\n{Colors.OKGREEN}Thank you for using Singularity Capital OS! 🚀{Colors.ENDC}\n")
            break
        elif choice == 'G':
            run_gui()
            continue  # GUI handles its own loop
        elif choice == '1':
            run_backtest()
        elif choice == '2':
            run_live()
        elif choice == '3':
            run_demo()
        elif choice == '4':
            run_xauusd_agent()
        elif choice == '5':
            run_multi_agent()
        elif choice == '6':
            run_core_test()
        elif choice == '7':
            run_mt5_test()
        elif choice == '8':
            ensure_dependencies(show_progress=True)
        else:
            print_status(f"Invalid choice: {choice}", 'warning')
        
        input(f"\n{Colors.OKCYAN}Press Enter to continue...{Colors.ENDC}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Singularity Capital OS Launcher')
    parser.add_argument('--gui', action='store_true', help='Launch Hacker GUI directly')
    parser.add_argument('--demo', action='store_true', help='Run demo mode directly')
    parser.add_argument('--backtest', action='store_true', help='Run backtest mode directly')
    parser.add_argument('--live', action='store_true', help='Run live mode directly')
    parser.add_argument('--install', action='store_true', help='Just install dependencies')
    parser.add_argument('--no-banner', action='store_true', help='Skip banner display')
    
    args = parser.parse_args()
    
    # Print banner
    if not args.no_banner:
        print_banner()
    
    # Ensure dependencies
    if args.install:
        ensure_dependencies(show_progress=True)
        return
    
    if not ensure_dependencies(show_progress=True):
        print_status("Please install missing dependencies and try again.", 'error')
        return
    
    # Direct mode execution
    if args.gui:
        run_gui()
    elif args.demo:
        run_demo()
    elif args.backtest:
        run_backtest()
    elif args.live:
        run_live()
    else:
        # Interactive menu
        interactive_menu()


if __name__ == "__main__":
    main()
