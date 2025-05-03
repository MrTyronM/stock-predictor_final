"""
Improved setup environment script for stock prediction model

This script sets up the project directories and installs required dependencies
with proper versioning to avoid compatibility issues.
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

# Core dependencies with strict versioning for compatibility
core_packages = [
    "numpy==1.24.3",           # Use older version to avoid TensorFlow issues
    "pandas==2.0.3",
    "matplotlib==3.7.2",
    "scikit-learn==1.3.0",
    "tensorflow==2.12.0",      # Compatible with numpy 1.24.3
    "seaborn==0.12.2",
    "yfinance==0.2.28",
    "tqdm==4.66.1"
]

# Additional packages for enhanced features
additional_packages = [
    "xgboost==1.7.6",
    "lightgbm==3.3.5",
    "scipy==1.10.1",
    "python-dotenv==1.0.0",
    "imbalanced-learn==0.10.1",  # For balancing datasets
    "joblib==1.3.2",
    "statsmodels==0.14.0"
]

# Optional pytorch for users who want to experiment with it
optional_packages = [
    "torch==2.0.1",            # Deep learning alternative
    "nltk==3.8.1",             # For NLP features
    "spacy==3.6.1",            # For NLP features
    "sqlalchemy==2.0.20",      # For database management
    "pymysql==1.1.0",          # For MySQL connection
    "plotly==5.16.1"           # For interactive visualizations
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

# Main function to run the setup process
def setup():
    print("\n🚀 Starting Stock Predictor setup...\n")
    
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
    
    # Install TA-Lib
    install_talib_option = input("\n📊 Do you want to install TA-Lib for technical indicators? (y/n): ").lower()
    if install_talib_option == 'y':
        install_talib()
    
    # Create a basic configuration file
    create_config()
    
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
            "use_gpu": True
        }
    }
    
    # Create the config directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Write the configuration file
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n✔️ Created default configuration file at {config_path}")

if __name__ == "__main__":
    setup()