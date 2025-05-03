import os
import subprocess
import sys

# List of directories to create
folders = [
    'data/raw',
    'data/processed',
    'models',
    'logs',
    'outputs',
    'notebooks'
]

print("📁 Creating project folders...")
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✔️  Created: {folder}")

# Optional: install dependencies
packages = [
    "pandas", "numpy", "matplotlib", "scikit-learn",
    "tensorflow", "torch", "ta-lib", "yfinance", "tqdm"
]

install = input("\n📦 Do you want to install core dependencies now? (y/n): ").lower()
if install == 'y':
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print("\n✅ All dependencies installed.")
    except Exception as e:
        print(f"\n❌ Failed to install some packages: {e}")
else:
    print("⏭️ Skipped dependency installation.")

print("\n🚀 Setup complete!")
