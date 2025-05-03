"""
Setup environment script for stock prediction model - Python 3.12 compatible

This script sets up the project directories and installs required dependencies
with proper versioning to avoid compatibility issues with Python 3.12.
"""
import os
import subprocess
import sys
import platform

# List of directories to create
folders = [
    'ml/data/raw',
    'ml/data/processed',
    'ml/models/checkpoints',
    'ml/models/final',
    'ml/models/plots',
    'ml/preprocessing/scalers',
    'ml/prediction/signals',
    'ml/prediction/results',
    'ml/prediction/visualizations',
    'ml/evaluation',
    'ml/logs',
    'db',
    'config',
    'notebooks'
]

print("📁 Creating project folders...")
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✔️  Created: {folder}")

# Core dependencies with Python 3.12 compatibility
core_packages = [
    "numpy>=1.26.0",           # Use the latest numpy for Python 3.12
    "pandas>=2.1.0",           # Compatible with numpy 1.26+
    "matplotlib>=3.8.0",
    "scikit-learn>=1.3.2",
    "tensorflow>=2.15.0",      # Compatible with Python 3.12
    "seaborn>=0.13.0",
    "yfinance>=0.2.28",
    "tqdm>=4.66.1"
]

# Additional packages for enhanced features
additional_packages = [
    "xgboost>=2.0.0",
    "lightgbm>=4.1.0",
    "scipy>=1.12.0",
    "python-dotenv>=1.0.0",
    "imbalanced-learn>=0.11.0",  # For balancing datasets
    "joblib>=1.3.2",
    "statsmodels>=0.14.0"
]

# Optional packages
optional_packages = [
    "torch>=2.1.0",            # Deep learning alternative
    "nltk>=3.8.1",             # For NLP features
    "spacy>=3.7.0",            # For NLP features
    "sqlalchemy>=2.0.20",      # For database management
    "pymysql>=1.1.0",          # For MySQL connection
    "plotly>=5.18.0"           # For interactive visualizations
]

# TA-Lib installation is special and platform-dependent
def install_talib():
    """Install TA-Lib based on platform"""
    system = platform.system()
    
    try:
        if system == "Windows":
            print("For Windows, please install TA-Lib manually:")
            print("1. Download the appropriate wheel file from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
            print("2. Install with: pip install <downloaded_file.whl>")
        elif system == "Darwin":  # macOS
            # Try using conda first
            try:
                subprocess.check_call(["conda", "install", "-c", "conda-forge", "ta-lib", "-y"])
                print("✅ TA-Lib installed successfully via conda")
                return True
            except:
                # If conda fails, try homebrew
                print("Installing TA-Lib dependencies via homebrew...")
                subprocess.check_call(["brew", "install", "ta-lib"])
                subprocess.check_call([sys.executable, "-m", "pip", "install", "ta-lib"])
                print("✅ TA-Lib installed successfully via homebrew")
        elif system == "Linux":
            print("Installing TA-Lib dependencies...")
            subprocess.check_call(["apt-get", "update"])
            subprocess.check_call(["apt-get", "install", "-y", "build-essential", "ta-lib"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ta-lib"])
            print("✅ TA-Lib installed successfully")
        return True
    except Exception as e:
        print(f"⚠️ TA-Lib installation error: {e}")
        print("You can continue without TA-Lib, but some technical indicators won't be available.")
        print("See https://github.com/mrjbq7/ta-lib for manual installation instructions.")
        return False

# Function to install packages with progress reporting
def install_packages(packages, package_type=""):
    print(f"\n📦 Installing {package_type} packages...")
    
    success_count = 0
    failed_packages = []
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            success_count += 1
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            failed_packages.append(package)
    
    print(f"\n✅ {success_count}/{len(packages)} {package_type} packages installed successfully")
    
    if failed_packages:
        print(f"❌ Failed to install {len(failed_packages)} packages: {', '.join(failed_packages)}")
    
    return failed_packages

# Alternative to TA-Lib for Python 3.12
def install_alternative_ta():
    """Install alternative technical analysis libraries"""
    print("\n📊 Installing alternative technical analysis libraries...")
    alt_packages = [
        "pandas-ta>=0.3.14b0",  # Alternative to TA-Lib with pandas integration
        "ta>=0.10.2",          # Another alternative technical analysis library
        "finta>=1.3"           # Financial Technical Analysis library
    ]
    
    failed = install_packages(alt_packages, "alternative TA")
    return len(failed) < len(alt_packages)

# Main function to run the setup process
def setup():
    print("\n🚀 Starting Stock Predictor setup for Python 3.12...\n")
    
    # Check Python version
    py_version = platform.python_version()
    print(f"Python version: {py_version}")
    
    # Install core packages
    core_failed = install_packages(core_packages, "core")
    
    if core_failed:
        print("\n⚠️ Some core packages failed to install. This may affect functionality.")
        proceed = input("Would you like to continue with the setup? (y/n): ").lower()
        if proceed != 'y':
            print("Setup aborted.")
            return
    
    # Install additional packages
    install_additional = input("\n📦 Do you want to install additional packages for enhanced features? (y/n): ").lower()
    if install_additional == 'y':
        install_packages(additional_packages, "additional")
    
    # Install optional packages
    install_optional = input("\n📦 Do you want to install optional packages? (y/n): ").lower()
    if install_optional == 'y':
        install_packages(optional_packages, "optional")
    
    # TA-Lib alternatives for Python 3.12
    print("\n📊 For Python 3.12, TA-Lib requires special handling.")
    print("Would you like to:")
    print("1. Try installing TA-Lib (may require compilation)")
    print("2. Install alternative TA libraries that are Python 3.12 compatible")
    print("3. Skip technical analysis libraries")
    
    ta_choice = input("Enter your choice (1/2/3): ")
    
    if ta_choice == '1':
        install_talib()
    elif ta_choice == '2':
        install_alternative_ta()
    
    # Create a basic configuration file
    create_config()
    
    # Create a modified feature engineering file for TA alternatives if needed
    if ta_choice == '2':
        create_alt_feature_engineering()
    
    print("\n🎉 Setup complete! You can now run the stock prediction model with:")
    print("python stock_predictor_main.py all")
    print("\nFor specific commands, use:")
    print("python stock_predictor_main.py [download|process|train|predict|test]")
    print("\nFor more options, see:")
    print("python stock_predictor_main.py --help")

def create_config():
    """Create a basic configuration file"""
    config_path = "config/config.json"
    
    # Skip if config already exists
    if os.path.exists(config_path):
        print(f"\n✔️ Configuration file already exists at {config_path}")
        return
    
    # Basic configuration
    config = {
        "model": {
            "model_type": "simple_hybrid",
            "model_complexity": "simple",
            "task": "classification",
            "target_col": "Target_Direction_5d",
            "confidence_threshold": 0.7
        },
        "training": {
            "epochs": 30,
            "batch_size": 32,
            "sequence_length": 20,
            "test_size": 0.2,
            "use_augmentation": True,
            "use_ensemble": False
        },
        "data": {
            "start_date": "2010-01-01",
            "days": 100
        },
        "system": {
            "version": "1.0.0",
            "use_gpu": True,
            "python_version": "3.12"
        }
    }
    
    # Create the config directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Write the configuration file
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n✔️ Created default configuration file at {config_path}")

def create_alt_feature_engineering():
    """Create a small adapter file for alternative TA libraries"""
    adapter_path = "ml/preprocessing/ta_adapter.py"
    
    os.makedirs(os.path.dirname(adapter_path), exist_ok=True)
    
    adapter_content = '''"""
Technical Analysis Adapter for Python 3.12

This module provides an adapter to use alternative technical analysis libraries
when TA-Lib is not available. It mimics the TA-Lib API for common functions.
"""
import numpy as np
import pandas as pd

# Try to import TA-Lib first
try:
    import talib
    USING_TALIB = True
except ImportError:
    USING_TALIB = False
    # Import alternatives
    try:
        import pandas_ta as pta
    except ImportError:
        pass
    
    try:
        import ta
    except ImportError:
        pass
    
    try:
        from finta import TA as finta_ta
    except ImportError:
        pass

def SMA(close, timeperiod=20):
    """Simple Moving Average"""
    if USING_TALIB:
        return talib.SMA(close, timeperiod)
    
    # Fallback implementations
    try:
        return pta.sma(close, length=timeperiod)
    except:
        try:
            # Convert to pandas series if numpy array
            if isinstance(close, np.ndarray):
                close = pd.Series(close)
            return ta.trend.sma_indicator(close, window=timeperiod)
        except:
            try:
                # Convert to pandas series if numpy array
                if isinstance(close, np.ndarray):
                    close = pd.Series(close)
                return close.rolling(window=timeperiod).mean()
            except:
                return np.full_like(close, np.nan)

def EMA(close, timeperiod=20):
    """Exponential Moving Average"""
    if USING_TALIB:
        return talib.EMA(close, timeperiod)
    
    # Fallback implementations
    try:
        return pta.ema(close, length=timeperiod)
    except:
        try:
            # Convert to pandas series if numpy array
            if isinstance(close, np.ndarray):
                close = pd.Series(close)
            return ta.trend.ema_indicator(close, window=timeperiod)
        except:
            try:
                # Convert to pandas series if numpy array
                if isinstance(close, np.ndarray):
                    close = pd.Series(close)
                return close.ewm(span=timeperiod, adjust=False).mean()
            except:
                return np.full_like(close, np.nan)

def MACD(close, fastperiod=12, slowperiod=26, signalperiod=9):
    """Moving Average Convergence Divergence"""
    if USING_TALIB:
        return talib.MACD(close, fastperiod, slowperiod, signalperiod)
    
    # Fallback implementations
    try:
        macd = pta.macd(close, fast=fastperiod, slow=slowperiod, signal=signalperiod)
        return macd['MACD_12_26_9'], macd['MACDs_12_26_9'], macd['MACDh_12_26_9']
    except:
        try:
            # Convert to pandas series if numpy array
            if isinstance(close, np.ndarray):
                close = pd.Series(close)
            macd = ta.trend.MACD(close, window_slow=slowperiod, window_fast=fastperiod, window_sign=signalperiod)
            return macd.macd(), macd.macd_signal(), macd.macd_diff()
        except:
            try:
                # Convert to pandas series if numpy array
                if isinstance(close, np.ndarray):
                    close = pd.Series(close)
                macd_data = finta_ta.MACD(pd.DataFrame({'close': close}), period_fast=fastperiod, period_slow=slowperiod, signal=signalperiod)
                return macd_data['MACD'], macd_data['SIGNAL'], macd_data['MACD'] - macd_data['SIGNAL']
            except:
                return np.full_like(close, np.nan), np.full_like(close, np.nan), np.full_like(close, np.nan)

def RSI(close, timeperiod=14):
    """Relative Strength Index"""
    if USING_TALIB:
        return talib.RSI(close, timeperiod)
    
    # Fallback implementations
    try:
        return pta.rsi(close, length=timeperiod)
    except:
        try:
            # Convert to pandas series if numpy array
            if isinstance(close, np.ndarray):
                close = pd.Series(close)
            return ta.momentum.RSIIndicator(close, window=timeperiod).rsi()
        except:
            try:
                # Convert to pandas series if numpy array
                if isinstance(close, np.ndarray):
                    close = pd.Series(close)
                return finta_ta.RSI(pd.DataFrame({'close': close}), period=timeperiod)
            except:
                return np.full_like(close, np.nan)

def BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
    """Bollinger Bands"""
    if USING_TALIB:
        return talib.BBANDS(close, timeperiod, nbdevup, nbdevdn, matype)
    
    # Fallback implementations
    try:
        bbands = pta.bbands(close, length=timeperiod, std=nbdevup)
        return bbands['BBU_20_2.0'], bbands['BBM_20_2.0'], bbands['BBL_20_2.0']
    except:
        try:
            # Convert to pandas series if numpy array
            if isinstance(close, np.ndarray):
                close = pd.Series(close)
            indicator_bb = ta.volatility.BollingerBands(close=close, window=timeperiod, window_dev=nbdevup)
            return indicator_bb.bollinger_hband(), indicator_bb.bollinger_mavg(), indicator_bb.bollinger_lband()
        except:
            try:
                # Convert to pandas series if numpy array
                if isinstance(close, np.ndarray):
                    close = pd.Series(close)
                bb = finta_ta.BBANDS(pd.DataFrame({'close': close}), period=timeperiod, std_multiplier=nbdevup)
                return bb['BB_UPPER'], bb['BB_MIDDLE'], bb['BB_LOWER']
            except:
                # Fallback to manual calculation
                try:
                    # Convert to pandas series if numpy array
                    if isinstance(close, np.ndarray):
                        close = pd.Series(close)
                    
                    middle = close.rolling(window=timeperiod).mean()
                    std = close.rolling(window=timeperiod).std()
                    upper = middle + nbdevup * std
                    lower = middle - nbdevdn * std
                    return upper, middle, lower
                except:
                    return np.full_like(close, np.nan), np.full_like(close, np.nan), np.full_like(close, np.nan)

# Add more TA-Lib function equivalents as needed
'''
    
    with open(adapter_path, 'w') as f:
        f.write(adapter_content)
    
    print(f"\n✔️ Created TA adapter file at {adapter_path}")
    print("This adapter provides alternative implementations for technical indicators")
    print("Edit ml/preprocessing/feature_engineering.py to use 'from ml.preprocessing.ta_adapter import *' instead of 'import talib'")

if __name__ == "__main__":
    setup()