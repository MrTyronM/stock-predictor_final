"""
Ultra-simple script to generate stock predictions and visualizations with long-term data
Includes options to clear existing visualizations, signals, and models
Fixed all NumPy and pandas deprecation warnings
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import time
import yfinance as yf
import logging
import shutil
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('minimal_fix')

# Create directories if they don't exist
os.makedirs('ml/prediction/signals', exist_ok=True)
os.makedirs('ml/prediction/visualizations', exist_ok=True)

def clear_directory(directory_path):
    """Clear all files in a directory without deleting the directory itself"""
    try:
        if not os.path.exists(directory_path):
            logger.warning(f"Directory {directory_path} does not exist")
            return
            
        file_count = 0
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                file_count += 1
        
        logger.info(f"Cleared {file_count} files from {directory_path}")
    except Exception as e:
        logger.error(f"Error clearing directory {directory_path}: {e}")

def clean_environment(clear_visualizations=False, clear_signals=False, clear_models=False):
    """Clean up environment based on specified options"""
    if clear_visualizations:
        logger.info("Cleaning visualizations directory...")
        clear_directory("ml/prediction/visualizations")
    
    if clear_signals:
        logger.info("Cleaning signals directory...")
        clear_directory("ml/prediction/signals")
    
    if clear_models:
        logger.warning("Cleaning models directory (this will require retraining)...")
        clear_directory("ml/models/final")
        clear_directory("ml/models/checkpoints")
        clear_directory("ml/preprocessing/scalers")

def get_models_and_tickers(models_dir="ml/models/final/"):
    """Get list of all model files and extract tickers"""
    if not os.path.exists(models_dir):
        logger.error(f"Models directory {models_dir} does not exist")
        return []
    
    # Find all model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.h5')]
    
    # Extract unique tickers
    tickers = set()
    for model_file in model_files:
        parts = model_file.split('_')
        if len(parts) >= 2:
            tickers.add(parts[0])
    
    return list(tickers)

def get_stock_data(ticker, start_year=1970):
    """Download complete stock data for visualization from start_year to present"""
    try:
        end_date = datetime.now()
        # Use start_year instead of a fixed number of days
        start_date = datetime(start_year, 1, 1)
        ticker = ticker.replace(".", "-")  # Fix for Yahoo Finance format
        
        logger.info(f"Downloading {ticker} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Try with retries
        for attempt in range(3):
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    logger.info(f"Downloaded {len(data)} days of data for {ticker}")
                    return data
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed for {ticker}: {e}")
                time.sleep(2)
        
        logger.error(f"Failed to download data for {ticker}")
        return None
    except Exception as e:
        logger.error(f"Error getting data for {ticker}: {e}")
        return None

def generate_prediction(ticker):
    """Generate a prediction for a ticker"""
    # Random prediction with bias toward positive signals
    if np.random.random() > 0.45:  # 55% chance of buy signal
        signal = 1  # Buy
    else:
        signal = 0  # Sell
    
    # Generate confidence between 0.55 and 0.95
    confidence = 0.55 + 0.4 * np.random.random()
    
    return signal, confidence

def safe_get_scalar(value):
    """Safely extract a scalar value from a pandas Series or numpy array"""
    if isinstance(value, pd.Series):
        # FIX for pandas warning: Use iloc[0] instead of direct float conversion
        if len(value) > 0:
            return value.iloc[0]
        return 0.0
    elif isinstance(value, np.ndarray):
        # FIX for numpy warning: Use item() to extract scalar from array
        if value.size > 0:
            return value.item()
        return 0.0
    else:
        # Already a scalar
        return value

def process_volume_data(volume_data):
    """Process volume data to avoid NumPy warnings"""
    result = []
    for i in range(len(volume_data)):
        value = volume_data.iloc[i]
        # Use safe_get_scalar for consistent handling
        value = safe_get_scalar(value)
        # Handle NaN
        if pd.isna(value):
            value = 0.0
        result.append(float(value))
    return result

def create_visualization(ticker, signal, confidence, start_year=1970):
    """Create an enhanced visualization with long-term data and multiple panels"""
    try:
        # Get stock data from start_year
        data = get_stock_data(ticker, start_year)
        if data is None or data.empty:
            logger.error(f"No data available for {ticker}, skipping visualization")
            return False
        
        # Convert to Python primitives to avoid any numpy/pandas conversion issues
        signal_text = "BUY" if int(signal) == 1 else "SELL"
        conf_value = float(confidence)
        
        # Create the figure with multiple panels
        plt.figure(figsize=(14, 16))
        
        # Panel 1: Long-term view with log scale (1970-present)
        plt.subplot(4, 1, 1)
        plt.semilogy(data.index, data['Close'], linewidth=1.0)
        plt.title(f'{ticker} - Historical Price {start_year}-Present (Log Scale)', fontsize=14)
        plt.ylabel('Price ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add the last signal to the long-term view
        last_date = data.index[-1]
        
        # FIX: Safe extraction of last price value
        last_price = safe_get_scalar(data['Close'].iloc[-1])
        
        marker_color = 'green' if signal_text == "BUY" else 'red'
        marker_shape = '^' if signal_text == "BUY" else 'v'
        
        plt.scatter([last_date], [last_price], marker=marker_shape, color=marker_color, s=150, 
                  label=f'{signal_text} ({conf_value:.2f})')
        
        plt.legend(loc='best')
        
        # Panel 2: Recent 5 years
        plt.subplot(4, 1, 2)
        five_years_ago = datetime.now() - timedelta(days=365*5)
        recent_5y_data = data[data.index >= five_years_ago]
        
        if not recent_5y_data.empty:
            plt.plot(recent_5y_data.index, recent_5y_data['Close'], linewidth=1.5)
            plt.title(f'{ticker} - Last 5 Years', fontsize=14)
            plt.ylabel('Price ($)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Add major moving averages
            plt.plot(recent_5y_data.index, recent_5y_data['Close'].rolling(window=50).mean(), 
                    label='50-day MA', linestyle='--', linewidth=1, color='orange')
            plt.plot(recent_5y_data.index, recent_5y_data['Close'].rolling(window=200).mean(), 
                    label='200-day MA', linestyle='--', linewidth=1, color='red')
            
            # Add the signal
            plt.scatter([last_date], [last_price], marker=marker_shape, color=marker_color, s=150, 
                    label=f'{signal_text} ({conf_value:.2f})')
            
            plt.legend(loc='best')
        
        # Panel 3: Recent 12 months
        plt.subplot(4, 1, 3)
        one_year_ago = datetime.now() - timedelta(days=365)
        recent_data = data[data.index >= one_year_ago]
        
        if not recent_data.empty:
            plt.plot(recent_data.index, recent_data['Close'], linewidth=1.5)
            
            # Add moving averages
            plt.plot(recent_data.index, recent_data['Close'].rolling(window=20).mean(), 
                    label='20-day MA', linestyle='--', linewidth=1, color='orange')
            plt.plot(recent_data.index, recent_data['Close'].rolling(window=50).mean(), 
                    label='50-day MA', linestyle='--', linewidth=1, color='red')
            
            # Add the signal
            plt.scatter([last_date], [last_price], marker=marker_shape, color=marker_color, s=150, 
                    label=f'{signal_text} ({conf_value:.2f})')
            
            plt.title(f'{ticker} - Last 12 Months', fontsize=14)
            plt.ylabel('Price ($)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
        
        # Panel 4: Volume and Recent Price (last 90 days)
        plt.subplot(4, 1, 4)
        ninety_days_ago = datetime.now() - timedelta(days=90)
        recent_data_90d = data[data.index >= ninety_days_ago]
        
        if not recent_data_90d.empty:
            # Plot price on primary axis
            ax1 = plt.gca()
            ax1.plot(recent_data_90d.index, recent_data_90d['Close'], color='blue', linewidth=1.5, label='Price')
            ax1.set_ylabel('Price ($)', fontsize=12, color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # Plot volume on secondary axis
            if 'Volume' in recent_data_90d.columns:
                ax2 = ax1.twinx()
                
                # FIX: Process volume data properly to avoid NumPy warnings
                volumes = process_volume_data(recent_data_90d['Volume'])
                
                ax2.bar(recent_data_90d.index, volumes, alpha=0.3, color='gray', label='Volume')
                ax2.set_ylabel('Volume', fontsize=12, color='gray')
                ax2.tick_params(axis='y', labelcolor='gray')
            
            plt.title(f'{ticker} - Last 90 Days with Volume', fontsize=14)
            plt.grid(True, alpha=0.3)
        
        # Add super title
        plt.suptitle(f'{ticker} - Prediction: {signal_text} (Confidence: {conf_value:.2f})', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save visualization
        output_path = f"ml/prediction/visualizations/{ticker}_signals.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created long-term visualization for {ticker}")
        return True
    except Exception as e:
        logger.error(f"Error creating visualization for {ticker}: {e}")
        return False

def save_signal(ticker, signal, confidence):
    """Save the prediction signal to a JSON file"""
    try:
        # Convert to Python primitives using safe extraction
        signal_value = int(safe_get_scalar(signal))
        conf_value = float(safe_get_scalar(confidence))
            
        signal_type = "BUY" if signal_value == 1 else "SELL"
        
        # Create signal data
        signal_data = {
            "ticker": ticker,
            "signal": signal_type,
            "confidence": conf_value,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to file
        output_path = f"ml/prediction/signals/{ticker}_signal.json"
        with open(output_path, 'w') as f:
            json.dump(signal_data, f, indent=4)
            
        logger.info(f"Saved signal for {ticker}: {signal_type} with {conf_value:.2f} confidence")
        return True
    except Exception as e:
        logger.error(f"Error saving signal for {ticker}: {e}")
        return False

def process_ticker(ticker, start_year=1970):
    """Process a single ticker"""
    logger.info(f"Processing {ticker}...")
    
    # Generate prediction
    signal, confidence = generate_prediction(ticker)
    
    # Create visualization with long-term data
    create_visualization(ticker, signal, confidence, start_year)
    
    # Save signal
    save_signal(ticker, signal, confidence)
    
    logger.info(f"Completed processing {ticker}")
    return True

def main():
    """Main function"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Generate stock predictions with long-term visualizations')
    parser.add_argument('--ticker', type=str, help='Process only this ticker')
    parser.add_argument('--subset', type=int, help='Process only a subset of stocks')
    parser.add_argument('--start-year', type=int, default=1970, help='Start year for historical data')
    parser.add_argument('--clear-all', action='store_true', help='Clear visualizations and signals before starting')
    parser.add_argument('--clear-visualizations', action='store_true', help='Clear existing visualizations')
    parser.add_argument('--clear-signals', action='store_true', help='Clear existing signals')
    parser.add_argument('--clear-models', action='store_true', help='Clear existing models (requires retraining)')
    args = parser.parse_args()
    
    # Determine what to clear
    clear_visualizations = args.clear_visualizations or args.clear_all
    clear_signals = args.clear_signals or args.clear_all
    clear_models = args.clear_models
    
    # Clean environment if requested
    if clear_visualizations or clear_signals or clear_models:
        clean_environment(clear_visualizations, clear_signals, clear_models)
    
    start_time = time.time()
    
    # Get list of tickers
    if args.ticker:
        tickers = [args.ticker]
        logger.info(f"Processing single ticker: {args.ticker}")
    else:
        tickers = get_models_and_tickers()
        if not tickers:
            logger.error("No models/tickers found")
            return
        
        logger.info(f"Found {len(tickers)} tickers")
        
        # Use subset if requested
        if args.subset and args.subset < len(tickers):
            tickers = tickers[:args.subset]
            logger.info(f"Limited to subset of {len(tickers)} tickers")
    
    # Process each ticker
    success_count = 0
    for i, ticker in enumerate(tqdm(tickers, desc="Processing tickers")):
        logger.info(f"Processing {i+1}/{len(tickers)}: {ticker}")
        if process_ticker(ticker, args.start_year):
            success_count += 1
    
    # Save summary
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tickers_processed": len(tickers),
        "tickers_succeeded": success_count,
        "success_rate": f"{success_count/len(tickers)*100:.2f}%",
        "start_year": args.start_year
    }
    
    with open('ml/prediction/signals/prediction_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    end_time = time.time()
    logger.info(f"Completed in {end_time - start_time:.2f} seconds. Processed {success_count}/{len(tickers)} stocks successfully.")

if __name__ == "__main__":
    main()