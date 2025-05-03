"""
Main script for the AI-Powered Stock Market Prediction Tool - Improved Version

This script provides the main entry point for training models and making predictions
"""
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import yfinance as yf
from tqdm import tqdm
import random

# Add current directory to path (helps with imports)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create base directory structure
os.makedirs("ml/logs", exist_ok=True)
os.makedirs("ml/data/raw", exist_ok=True)
os.makedirs("ml/data/processed", exist_ok=True)
os.makedirs("ml/models/checkpoints", exist_ok=True)
os.makedirs("ml/models/final", exist_ok=True)
os.makedirs("ml/preprocessing/scalers", exist_ok=True)
os.makedirs("ml/prediction/signals", exist_ok=True)
os.makedirs("ml/prediction/results", exist_ok=True)
os.makedirs("ml/prediction/visualizations", exist_ok=True)
os.makedirs("ml/evaluation", exist_ok=True)
os.makedirs("db", exist_ok=True)
os.makedirs("config", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml/logs/main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main")

def parse_arguments():
    """Parse command line arguments with improved options"""
    parser = argparse.ArgumentParser(description='AI-Powered Stock Market Prediction Tool')
    
    # Main command
    parser.add_argument('command', choices=['download', 'process', 'train', 'predict', 'all', 'test', 'visualize'],
                      help='Command to execute')
    
    # Data options
    parser.add_argument('--start-date', type=str, default='2010-01-01',
                      help='Start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=100,
                      help='Number of days for prediction data')
    parser.add_argument('--subset', type=int, default=None,
                      help='Use subset of stocks (number)')
    parser.add_argument('--start-year', type=int, default=1975,
                      help='Start year for historical data')
    
    # Model options
    parser.add_argument('--ticker', type=str, default=None,
                      help='Stock ticker symbol (default: process all)')
    parser.add_argument('--tickers-file', type=str, default=None,
                      help='File containing list of tickers to process')
    parser.add_argument('--model-type', type=str, default='simple_hybrid',
                      choices=['lstm', 'cnn_lstm', 'bidirectional', 'attention', 'hybrid', 'simple_hybrid'],
                      help='Type of model architecture')
    parser.add_argument('--model-complexity', type=str, default='simple',
                      choices=['simple', 'medium', 'complex'],
                      help='Complexity of model architecture')
    parser.add_argument('--task', type=str, default='classification',
                      choices=['regression', 'classification'],
                      help='Type of prediction task')
    parser.add_argument('--target-col', type=str, default='Target_Direction_5d',
                      help='Target column for prediction')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=30,
                      help='Maximum number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--sequence-length', type=int, default=20,
                      help='Length of input sequences')
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Proportion of data for testing')
    
    # Prediction options
    parser.add_argument('--threshold', type=float, default=0.7,
                      help='Confidence threshold for trading signals')
    parser.add_argument('--visualize', action='store_true',
                      help='Create visualizations of trading signals')
    parser.add_argument('--use-augmentation', action='store_true', default=True,
                      help='Use data augmentation for better performance')
    
    # Combined model options
    parser.add_argument('--combined-model', action='store_true',
                      help='Train a single model on all stocks')
    parser.add_argument('--augment-data', action='store_true', default=True,
                      help='Augment training data with noise')
    parser.add_argument('--max-stocks', type=int, default=None,
                      help='Maximum number of stocks to include in combined model')
    parser.add_argument('--gpu', action='store_true',
                      help='Use GPU for training if available')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                      help='Patience for early stopping')
    parser.add_argument('--use-ensemble', action='store_true',
                      help='Use ensemble of multiple models')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Custom output directory')
    
    # Testing option
    parser.add_argument('--test-count', type=int, default=5,
                      help='Number of stocks to test for accuracy evaluation')
    
    return parser.parse_args()

def get_tickers_to_process(args):
    """Get list of tickers to process with improved stock selection"""
    if args.ticker:
        return [args.ticker]
    elif args.tickers_file and os.path.exists(args.tickers_file):
        with open(args.tickers_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        # Import here to avoid circular imports
        try:
            from ml.data_acquisition import get_sp500_tickers, get_available_tickers
            
            # For training focus on major stocks with longer histories and higher volume
            if args.command == 'train' and not args.ticker and not args.tickers_file:
                # Major stocks with long histories and high volume
                major_stocks = [
                    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "JNJ", 
                    "JPM", "V", "PG", "UNH", "HD", "MA", "NVDA", "BAC", "DIS", "CSCO",
                    "VZ", "KO", "MRK", "PFE", "INTC", "WMT"
                ]
                
                # For subset, use major stocks up to the limit
                if args.subset and isinstance(args.subset, int) and args.subset < len(major_stocks):
                    return major_stocks[:args.subset]
                
                # Otherwise return all major stocks
                return major_stocks
            
            if args.command in ['process', 'train', 'predict', 'visualize']:
                # Use available tickers (those with data already downloaded)
                tickers = get_available_tickers()
            else:
                # Use S&P 500 tickers
                tickers = get_sp500_tickers()
            
            # Apply subset limit if specified
            if args.subset and isinstance(args.subset, int) and args.subset < len(tickers):
                return tickers[:args.subset]
            return tickers
        except ImportError:
            logger.warning("Could not import get_sp500_tickers or get_available_tickers")
            # Fall back to a default list of major stocks
            major_stocks = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "JNJ", 
                "JPM", "V", "PG", "UNH", "HD", "MA", "NVDA", "BAC", "DIS", "CSCO"
            ]
            if args.subset and isinstance(args.subset, int) and args.subset < len(major_stocks):
                return major_stocks[:args.subset]
            return major_stocks

def execute_download(args):
    """Execute data download command"""
    try:
        from ml.data_acquisition import download_all_sp500_historical_data
        
        logger.info(f"Downloading historical data from {args.start_date}")
        download_all_sp500_historical_data(start_date=args.start_date)
    except ImportError:
        logger.error("Could not import download_all_sp500_historical_data")
        print("Error: Could not import the data acquisition module.")

def execute_process(args):
    """Execute data processing command"""
    try:
        from ml.preprocessing.feature_engineering import prepare_multiple_stocks
        
        tickers = get_tickers_to_process(args)
        logger.info(f"Processing {len(tickers)} stocks")
        
        prepare_multiple_stocks(use_subset=args.subset)
    except ImportError:
        logger.error("Could not import prepare_multiple_stocks")
        print("Error: Could not import the feature engineering module.")

def execute_train(args):
    """Execute model training command with improved approach"""
    try:
        from ml.training.model_training import train_models_for_multiple_stocks
        
        tickers = get_tickers_to_process(args)
        logger.info(f"Training models for {len(tickers)} stocks")
        
        train_models_for_multiple_stocks(
            tickers,
            model_type=args.model_type,
            model_complexity=args.model_complexity,
            target_col=args.target_col,
            task=args.task,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_ensemble=args.use_ensemble,
            use_augmentation=args.augment_data
        )
    except ImportError:
        logger.error("Could not import train_models_for_multiple_stocks")
        print("Error: Could not import the model training module.")

def execute_combined_train(args):
    """Execute combined model training command with improved approach"""
    try:
        from ml.training.model_training import train_sp500_combined_model
        
        logger.info("Training combined model on all S&P 500 stocks")
        
        train_sp500_combined_model(
            model_type=args.model_type,
            model_complexity=args.model_complexity,
            target_col=args.target_col,
            task=args.task,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_stocks=args.max_stocks,
            augment_data=args.augment_data,
            use_gpu=args.gpu,
            early_stopping_patience=args.early_stopping_patience
        )
    except ImportError:
        logger.error("Could not import train_sp500_combined_model")
        print("Error: Could not import the model training module.")

def execute_predict(args):
    """Execute prediction command with improved approach"""
    try:
        from ml.prediction.prediction import predict_for_multiple_stocks, get_latest_signals, compute_portfolio_allocation
        
        tickers = get_tickers_to_process(args)
        logger.info(f"Making predictions for {len(tickers)} stocks")
        
        predict_for_multiple_stocks(
            tickers,
            model_type=args.model_type,
            task=args.task,
            days=args.days,
            sequence_length=args.sequence_length,
            threshold=args.threshold,
            visualize=args.visualize,
            use_augmentation=args.use_augmentation
        )
        
        # Display latest signals with improved output
        latest_signals = get_latest_signals(min_confidence=args.threshold)
        if latest_signals is not None and not latest_signals.empty:
            print("\nLatest Trading Signals (Top 10):")
            display_cols = ['Ticker', 'Date', 'Signal', 'Confidence', 'Action']
            available_cols = [col for col in display_cols if col in latest_signals.columns]
            print(latest_signals[available_cols].head(10))
            
            # Generate portfolio allocation
            allocation = compute_portfolio_allocation(latest_signals)
            if allocation is not None:
                print("\nRecommended Portfolio Allocation:")
                print(allocation)
        else:
            logger.warning("No trading signals available")
    except ImportError:
        logger.error("Could not import prediction modules")
        print("Error: Could not import the prediction modules.")

def execute_visualize_fix(args):
    """Execute visualization fix for existing predictions"""
    import matplotlib.pyplot as plt
    import glob
    
    # Get signal files
    signals_dir = "ml/prediction/signals/"
    signal_files = glob.glob(os.path.join(signals_dir, "*_signal.json"))
    
    logger.info(f"Creating visualizations for {len(signal_files)} stocks")
    
    success_count = 0
    for signal_file in tqdm(signal_files, desc="Creating visualizations"):
        try:
            # Extract ticker from filename
            ticker = os.path.basename(signal_file).split('_')[0]
            
            # Load signal data
            with open(signal_file, 'r') as f:
                signal_data = json.load(f)
            
            # Get signal info
            signal_text = signal_data.get('signal', 'BUY')
            confidence = float(signal_data.get('confidence', 0.5))
            
            # Download stock data
            start_date = f"{args.start_year}-01-01"
            end_date = datetime.now().strftime("%Y-%m-%d")
            
            # Format ticker correctly for Yahoo Finance
            ticker_yf = ticker.replace(".", "-")
            data = yf.download(ticker_yf, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                logger.warning(f"No data available for {ticker}")
                continue
            
            # Create visualization
            plt.figure(figsize=(12, 16))
            
            # Panel 1: Long-term view with log scale
            plt.subplot(3, 1, 1)
            plt.semilogy(data.index, data['Close'], linewidth=1.0)
            plt.title(f'{ticker} - Historical Price (Log Scale)', fontsize=14)
            plt.ylabel('Price ($)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Panel 2: Recent 12 months
            plt.subplot(3, 1, 2)
            one_year_ago = datetime.now() - timedelta(days=365)
            recent_data = data[data.index >= one_year_ago]
            
            if not recent_data.empty:
                plt.plot(recent_data.index, recent_data['Close'], linewidth=1.5)
                
                # Add moving averages
                plt.plot(recent_data.index, recent_data['Close'].rolling(window=20).mean(), 
                        label='20-day MA', linestyle='--', linewidth=1, color='orange')
                plt.plot(recent_data.index, recent_data['Close'].rolling(window=50).mean(), 
                        label='50-day MA', linestyle='--', linewidth=1, color='red')
                
                # Add the prediction signal at the most recent date - FIXED
                if len(recent_data.index) > 0:
                    last_date = recent_data.index[-1]
                    last_price = float(recent_data['Close'].iloc[-1])
                    
                    if signal_text == "BUY":
                        plt.scatter([last_date], [last_price], marker='^', color='green', s=200, 
                                  label=f'BUY ({confidence:.2f})')
                    else:
                        plt.scatter([last_date], [last_price], marker='v', color='red', s=200, 
                                  label=f'SELL ({confidence:.2f})')
                
                plt.title(f'{ticker} - Last 12 Months', fontsize=14)
                plt.ylabel('Price ($)', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best')
            
            # Panel 3: Volume and Recent Price
            plt.subplot(3, 1, 3)
            ninety_days_ago = datetime.now() - timedelta(days=90)
            recent_data_90d = data[data.index >= ninety_days_ago]
            
            if not recent_data_90d.empty:
                # Plot price on primary axis
                ax1 = plt.gca()
                ax1.plot(recent_data_90d.index, recent_data_90d['Close'], color='blue', linewidth=1.5, label='Price')
                ax1.set_ylabel('Price ($)', fontsize=12, color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                
                # Plot volume on secondary axis - FIXED
                if 'Volume' in recent_data_90d.columns:
                    ax2 = ax1.twinx()
                    # Convert to list of primitive values
                    volumes = [float(v) for v in recent_data_90d['Volume'].values]
                    ax2.bar(recent_data_90d.index, volumes, alpha=0.3, color='gray', label='Volume')
                    ax2.set_ylabel('Volume', fontsize=12, color='gray')
                    ax2.tick_params(axis='y', labelcolor='gray')
                
                plt.title(f'{ticker} - Last 90 Days with Volume', fontsize=14)
                plt.grid(True, alpha=0.3)
            
            # Add super title
            plt.suptitle(f'{ticker}: {signal_text} Signal with {confidence:.2f} Confidence', fontsize=16, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save visualization
            output_dir = "ml/prediction/visualizations/"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{ticker}_prediction.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            success_count += 1
            logger.info(f"Created visualization for {ticker}")
            
        except Exception as e:
            logger.error(f"Error creating visualization for {ticker}: {e}")
    
    logger.info(f"Created {success_count}/{len(signal_files)} visualizations")

def execute_test(args):
    """Execute model testing to evaluate accuracy"""
    try:
        from ml.training.model_training import train_stock_model
        from ml.prediction.prediction import predict_for_stock
        from sklearn.metrics import accuracy_score
        
        # Get available tickers
        from ml.data_acquisition import get_available_tickers
        available_tickers = get_available_tickers()
        
        # Select random tickers for testing if not specified
        if args.ticker:
            test_tickers = [args.ticker]
        else:
            # Prioritize major stocks
            major_stocks = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "JNJ", 
                "JPM", "V", "PG", "UNH", "HD", "MA", "NVDA", "BAC", "DIS", "CSCO"
            ]
            # Find intersection with available tickers
            available_major = [t for t in major_stocks if t in available_tickers]
            
            # If we have enough major stocks, use them; otherwise, sample from all available
            if len(available_major) >= args.test_count:
                test_tickers = random.sample(available_major, args.test_count)
            else:
                test_tickers = random.sample(available_tickers, min(args.test_count, len(available_tickers)))
        
        logger.info(f"Testing model performance on {len(test_tickers)} stocks: {', '.join(test_tickers)}")
        
        results = {}
        overall_accuracy = []
        
        # Test each model
        for ticker in test_tickers:
            logger.info(f"Testing model for {ticker}")
            
            # Train with a small test_size to reserve more data for validation
            model, metrics = train_stock_model(
                ticker,
                processed_data_path="ml/data/processed/",
                model_output_path="ml/models/",
                target_col=args.target_col,
                model_type=args.model_type,
                model_complexity=args.model_complexity,
                sequence_length=args.sequence_length,
                test_size=0.3,  # Larger test size for better evaluation
                task=args.task,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            
            if model is not None and metrics is not None:
                # Get accuracy from metrics
                accuracy = metrics.get('accuracy', 0)
                overall_accuracy.append(accuracy)
                results[ticker] = {
                    'accuracy': accuracy,
                    'metrics': metrics
                }
                logger.info(f"Model for {ticker} achieved {accuracy:.4f} accuracy")
            else:
                logger.warning(f"Testing failed for {ticker}")
        
        # Calculate overall results
        if overall_accuracy:
            avg_accuracy = np.mean(overall_accuracy)
            logger.info(f"Average accuracy across {len(overall_accuracy)} models: {avg_accuracy:.4f}")
            print(f"\nTesting completed. Average accuracy: {avg_accuracy:.4f}")
            
            # Show individual results
            print("\nIndividual model results:")
            for ticker, ticker_results in results.items():
                print(f"{ticker}: {ticker_results['accuracy']:.4f} accuracy")
        else:
            logger.error("No models were successfully tested")
            print("Testing failed. See logs for details.")
    except ImportError:
        logger.error("Could not import necessary modules for testing")
        print("Error: Could not import the testing modules.")

def execute_all(args):
    """Execute all steps in sequence with improved flow"""
    logger.info("Executing complete pipeline: download -> process -> train -> predict")
    
    # Execute each step
    execute_download(args)
    execute_process(args)
    
    # Use combined model if requested
    if args.combined_model:
        execute_combined_train(args)
    else:
        execute_train(args)
        
    execute_predict(args)
    
    logger.info("Complete pipeline execution finished")

def main():
    """Main function to run the stock prediction tool"""
    args = parse_arguments()
    
    start_time = time.time()
    logger.info(f"Starting execution of command: {args.command}")
    
    try:
        # Execute the requested command
        if args.command == 'download':
            execute_download(args)
        elif args.command == 'process':
            execute_process(args)
        elif args.command == 'train':
            if args.combined_model:
                execute_combined_train(args)
            else:
                execute_train(args)
        elif args.command == 'predict':
            execute_predict(args)
        elif args.command == 'visualize':
            execute_visualize_fix(args)
        elif args.command == 'test':
            execute_test(args)
        elif args.command == 'all':
            execute_all(args)
        
        execution_time = time.time() - start_time
        logger.info(f"Command '{args.command}' completed successfully in {execution_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        import traceback
        logger.error(traceback.format_exc())
        execution_time = time.time() - start_time
        logger.info(f"Command '{args.command}' failed after {execution_time:.2f} seconds")
        sys.exit(1)

if __name__ == "__main__":
    main()