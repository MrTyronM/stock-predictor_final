"""
Prediction module for stock market prediction - Improved Version with Long-Term Data Support

This module loads trained models and makes predictions on new data
"""
import os
import sys
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import json
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time  # Explicitly import time module

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.feature_engineering import add_technical_indicators, scale_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml/logs/prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("prediction")

# Create necessary directories
os.makedirs("ml/logs", exist_ok=True)
os.makedirs("ml/prediction/results", exist_ok=True)
os.makedirs("ml/prediction/signals", exist_ok=True)
os.makedirs("ml/prediction/visualizations", exist_ok=True)

def load_stock_model(ticker, model_dir="ml/models/final/", model_type="lstm", task="classification"):
    """
    Load a trained model for a specific stock
    
    Args:
        ticker (str): Stock ticker symbol
        model_dir (str): Directory containing trained models
        model_type (str): Type of model ('lstm', 'cnn_lstm', etc.)
        task (str): Type of task ('regression' or 'classification')
        
    Returns:
        tensorflow.keras.models.Model: Loaded model
    """
    try:
        model_name = f"{ticker}_{model_type}_{task}_model.h5"
        model_path = os.path.join(model_dir, model_name)
        
        # First try the exact model name
        if not os.path.exists(model_path):
            # If not found, try alternative model types
            alt_types = ["simple_hybrid", "hybrid", "lstm", "bidirectional", "attention", "cnn_lstm"]
            for alt_type in alt_types:
                if alt_type == model_type:
                    continue
                alt_model_name = f"{ticker}_{alt_type}_{task}_model.h5"
                alt_model_path = os.path.join(model_dir, alt_model_name)
                if os.path.exists(alt_model_path):
                    logger.info(f"Using alternative model {alt_type} for {ticker}")
                    model_path = alt_model_path
                    break
        
        if not os.path.exists(model_path):
            # If still not found, try the combined model
            combined_model_name = f"sp500_combined_{model_type}_{task}_model.h5"
            combined_model_path = os.path.join(model_dir, combined_model_name)
            if os.path.exists(combined_model_path):
                logger.info(f"Using combined model for {ticker}")
                model_path = combined_model_path
            else:
                logger.error(f"No model found for {ticker}")
                return None
        
        # Load the model with custom objects if needed
        model = load_model(model_path)
        logger.info(f"Loaded model for {ticker} from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model for {ticker}: {e}")
        return None

def get_latest_stock_data(ticker, days=100, start_year=1950):
    """
    Get the historical stock data for a ticker
    
    Args:
        ticker (str): Stock ticker symbol
        days (int): Number of days to return (used for prediction)
        start_year (int): Start year for historical data (for visualization)
        
    Returns:
        pandas.DataFrame: Historical stock data
    """
    try:
        end_date = datetime.now()
        # Get data from start_year until now
        start_date = datetime(start_year, 1, 1)
        
        # Format ticker correctly for Yahoo Finance
        ticker = ticker.replace(".", "-")  # Yahoo Finance uses dashes instead of dots
        
        # Try multiple attempts with backoff
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Download data
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if not data.empty:
                    logger.info(f"Downloaded {len(data)} days of data for {ticker} (from {start_year})")
                    return data
                
                attempt += 1
                if attempt < max_attempts:
                    logger.warning(f"Attempt {attempt} failed for {ticker}. Retrying...")
                    time.sleep(2 * attempt)  # Exponential backoff
            except Exception as inner_e:
                logger.warning(f"Error in attempt {attempt} for {ticker}: {inner_e}")
                attempt += 1
                if attempt < max_attempts:
                    time.sleep(2 * attempt)
        
        # If all attempts failed
        logger.warning(f"No data found for {ticker} after {max_attempts} attempts")
        return None
        
    except Exception as e:
        logger.error(f"Error getting historical data for {ticker}: {e}")
        return None

def get_historical_data(ticker, processed_data_path="ml/data/processed/"):
    """
    Get historical processed data for a ticker
    
    Args:
        ticker (str): Stock ticker symbol
        processed_data_path (str): Path to processed data
        
    Returns:
        pandas.DataFrame: Processed historical data
    """
    try:
        # Construct file path
        file_path = os.path.join(processed_data_path, f"{ticker}_processed.csv")
        
        if not os.path.exists(file_path):
            logger.warning(f"No processed data found for {ticker}")
            return None
        
        # Load processed data
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        if df.empty:
            logger.warning(f"Empty processed data for {ticker}")
            return None
            
        # Get the latest portion of data
        df = df.sort_index().tail(100)
        
        logger.info(f"Loaded historical processed data for {ticker} with {len(df)} rows")
        return df
        
    except Exception as e:
        logger.error(f"Error getting historical data for {ticker}: {e}")
        return None

def prepare_data_for_prediction(data, ticker, sequence_length=20, scaler_path=None, use_historical=False,
                               prediction_days=100):
    """
    Prepare stock data for prediction with improved preprocessing
    
    Args:
        data (pandas.DataFrame): Stock data
        ticker (str): Stock ticker symbol
        sequence_length (int): Length of input sequences
        scaler_path (str): Path to load scaler
        use_historical (bool): Whether to use historical processed data
        prediction_days (int): Number of days of data to use for prediction
        
    Returns:
        numpy.ndarray: Prepared data for prediction
    """
    try:
        # If use_historical is True, try to load processed data
        if use_historical:
            df_historical = get_historical_data(ticker)
            if df_historical is not None:
                # Extract the latest sequence for prediction
                X_sequence = []
                # Select non-target columns
                feature_cols = [col for col in df_historical.columns if not col.startswith('Target_')]
                X = df_historical[feature_cols].values
                # Take the last sequence_length rows
                if len(X) >= sequence_length:
                    X_sequence.append(X[-sequence_length:])
                    X_sequence = np.array(X_sequence)
                    logger.info(f"Prepared prediction data from historical data with shape {X_sequence.shape}")
                    return X_sequence, df_historical.index[-sequence_length:]
                else:
                    logger.warning(f"Insufficient historical data for {ticker}")
        
        # If not using historical or historical data is not available, process new data
        # Limit to prediction_days for feature engineering (to prevent memory issues with very large datasets)
        if len(data) > prediction_days:
            prediction_data = data.tail(prediction_days)
        else:
            prediction_data = data
            
        # Add technical indicators
        df_with_indicators = add_technical_indicators(prediction_data)
        
        if df_with_indicators is None or df_with_indicators.empty:
            logger.error(f"Failed to add technical indicators for {ticker}")
            return None, None
        
        # Choose important features only (use all available features if metadata not found)
        feature_cols = df_with_indicators.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Specify scaler path if not provided
        if scaler_path is None:
            scaler_path = os.path.join("ml/preprocessing/scalers", f"{ticker}_scaler.pkl")
        
        # Scale features with error handling
        try:
            # Try to load the scaler
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                # Transform only the features present in the scaler
                overlap_features = []
                if hasattr(scaler, 'feature_names_in_'):
                    # Get the overlap of features in both the data and the scaler
                    scaler_features = scaler.feature_names_in_
                    overlap_features = [f for f in feature_cols if f in scaler_features]
                    
                    if len(overlap_features) < 5:
                        # Not enough overlap, use basic scaling
                        df_scaled = df_with_indicators.copy()
                        for col in feature_cols:
                            if col in df_scaled.columns:
                                # Simple z-score scaling
                                df_scaled[col] = (df_scaled[col] - df_scaled[col].mean()) / df_scaled[col].std()
                    else:
                        # Use the existing scaler for the overlapping features
                        df_scaled = df_with_indicators.copy()
                        df_scaled[overlap_features] = scaler.transform(df_with_indicators[overlap_features])
                else:
                    # No feature names in scaler, use standard scaling
                    df_scaled = df_with_indicators.copy()
                    for col in feature_cols:
                        if col in df_scaled.columns:
                            df_scaled[col] = (df_scaled[col] - df_scaled[col].mean()) / df_scaled[col].std()
            else:
                # If no scaler exists, perform standard scaling
                df_scaled = df_with_indicators.copy()
                for col in feature_cols:
                    if col in df_scaled.columns:
                        df_scaled[col] = (df_scaled[col] - df_scaled[col].mean()) / df_scaled[col].std()
        except Exception as e:
            logger.warning(f"Error scaling features: {e}, using unscaled data")
            df_scaled = df_with_indicators
        
        # Drop rows with NaN values
        df_scaled = df_scaled.dropna()
        
        if df_scaled.empty:
            logger.error(f"Empty dataframe after handling NaNs for {ticker}")
            return None, None
        
        # Extract feature values
        X = df_scaled[feature_cols].values
        
        # Create sequence
        X_sequence = []
        if len(X) >= sequence_length:
            X_sequence.append(X[-sequence_length:])
            X_sequence = np.array(X_sequence)
            logger.info(f"Prepared prediction data with shape {X_sequence.shape}")
            return X_sequence, df_scaled.index[-sequence_length:]
        else:
            logger.error(f"Insufficient data for {ticker}: {len(X)} rows, need {sequence_length}")
            return None, None
        
    except Exception as e:
        logger.error(f"Error preparing data for prediction: {e}")
        return None, None

def make_prediction(model, X_sequence, task="classification", use_augmentation=True):
    """
    Make predictions using a trained model with improved robustness
    
    Args:
        model (tensorflow.keras.models.Model): Trained model
        X_sequence (numpy.ndarray): Input data sequence
        task (str): Type of task ('regression' or 'classification')
        use_augmentation (bool): Whether to use test-time augmentation
        
    Returns:
        numpy.ndarray: Predictions
    """
    try:
        # Make prediction with test-time augmentation if requested
        if use_augmentation:
            # Add small noise to create multiple versions of the input
            predictions = []
            
            # Original prediction
            orig_pred = model.predict(X_sequence)
            predictions.append(orig_pred)
            
            # Add 3 augmented versions with different noise levels
            for i in range(3):
                noise_level = 0.01 * (i + 1)  # 1%, 2%, 3% noise
                X_noisy = X_sequence + np.random.normal(0, noise_level, X_sequence.shape)
                aug_pred = model.predict(X_noisy)
                predictions.append(aug_pred)
            
            # Average the predictions
            raw_predictions = np.mean(predictions, axis=0)
        else:
            # Standard prediction
            raw_predictions = model.predict(X_sequence)
        
        # Process predictions based on task
        if task == "regression":
            # Return raw predictions for regression
            return raw_predictions
        else:
            # For binary classification, convert to signal
            if len(raw_predictions.shape) > 1 and raw_predictions.shape[1] > 1:
                # Multiclass classification
                signals = np.argmax(raw_predictions, axis=1)
                signal_strength = np.max(raw_predictions, axis=1)  # Max probability as signal strength
            else:
                # Binary classification
                raw_predictions = raw_predictions.flatten()  # Ensure 1D array
                signals = (raw_predictions > 0.5).astype(int)
                signal_strength = raw_predictions  # Probability as signal strength
            
            return signals, signal_strength
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return None

def generate_trading_signals(predictions, signal_strength, dates, ticker, threshold=0.7):
    """
    Generate trading signals based on predictions with improved signal quality
    
    Args:
        predictions (numpy.ndarray): Model predictions
        signal_strength (numpy.ndarray): Signal strength (probability)
        dates (pandas.DatetimeIndex): Dates corresponding to predictions
        ticker (str): Stock ticker symbol
        threshold (float): Confidence threshold for signals
        
    Returns:
        pandas.DataFrame: Trading signals
    """
    try:
        # Create signals dataframe
        signals_df = pd.DataFrame(index=dates[-len(predictions):])
        signals_df['Ticker'] = ticker
        signals_df['Date'] = signals_df.index
        signals_df['Prediction'] = predictions.flatten()
        signals_df['Confidence'] = signal_strength.flatten()
        
        # Generate more detailed signal labels
        signals_df['Signal'] = np.where(
            signals_df['Prediction'] == 1,  # Buy signals
            np.where(signals_df['Confidence'] >= threshold + 0.1, 'Strong Buy',  # Higher confidence
                   np.where(signals_df['Confidence'] >= threshold, 'Buy', 'Weak Buy')),  # Lower confidence
            np.where(signals_df['Confidence'] >= threshold + 0.1, 'Strong Sell',  # Higher confidence
                   np.where(signals_df['Confidence'] >= threshold, 'Sell', 'Weak Sell'))  # Lower confidence
        )
        
        # Add a simple trading action column
        def determine_action(row):
            if row['Prediction'] == 1 and row['Confidence'] >= threshold:
                return 'BUY'
            elif row['Prediction'] == 0 and row['Confidence'] >= threshold:
                return 'SELL'
            else:
                return 'HOLD'
                
        signals_df['Action'] = signals_df.apply(determine_action, axis=1)
        
        # Only consider signals with confidence above threshold for important decisions
        signals_df['Valid_Signal'] = signals_df['Confidence'] >= threshold
        
        # Ensure consecutive signals have increasing confidence to avoid whipsaw
        for i in range(1, len(signals_df)):
            if signals_df.iloc[i]['Prediction'] == signals_df.iloc[i-1]['Prediction']:
                # If same signal, require higher or equal confidence
                if signals_df.iloc[i]['Confidence'] < signals_df.iloc[i-1]['Confidence'] - 0.05:
                    signals_df.iloc[i, signals_df.columns.get_loc('Valid_Signal')] = False
        
        logger.info(f"Generated {len(signals_df[signals_df['Valid_Signal']])} valid signals for {ticker}")
        return signals_df
        
    except Exception as e:
        logger.error(f"Error generating trading signals: {e}")
        return None

def visualize_signals(data, signals_df, ticker, output_dir="ml/prediction/visualizations/"):
    """
    Create enhanced visualization of trading signals with long-term and recent views
    
    Args:
        data (pandas.DataFrame): Original stock data
        signals_df (pandas.DataFrame): Trading signals
        ticker (str): Stock ticker symbol
        output_dir (str): Directory to save visualizations
        
    Returns:
        str: Path to saved visualization
    """
    try:
        # Create a figure with more detailed visualizations
        plt.figure(figsize=(14, 12))
        
        # Plot long-term stock price with signals (log scale for better visualization)
        plt.subplot(3, 1, 1)
        plt.semilogy(data.index, data['Close'], label='Close Price', linewidth=1.0)
        
        # Plot buy signals
        buy_signals = signals_df[signals_df['Signal'].str.contains('Buy')]
        if not buy_signals.empty:
            buy_dates = buy_signals.index.intersection(data.index)
            if not buy_dates.empty:
                # Convert values to scalar to avoid NumPy array issues
                buy_prices = []
                for d in buy_dates:
                    price = data.loc[d, 'Close']
                    # Handle Series if needed
                    if isinstance(price, pd.Series):
                        price = price.iloc[0]
                    buy_prices.append(float(price))
                    
                plt.scatter(buy_dates, buy_prices, 
                        marker='^', color='green', s=100, label='Buy Signal')
        
        # Plot sell signals
        sell_signals = signals_df[signals_df['Signal'].str.contains('Sell')]
        if not sell_signals.empty:
            sell_dates = sell_signals.index.intersection(data.index)
            if not sell_dates.empty:
                # Convert values to scalar to avoid NumPy array issues
                sell_prices = []
                for d in sell_dates:
                    price = data.loc[d, 'Close']
                    # Handle Series if needed
                    if isinstance(price, pd.Series):
                        price = price.iloc[0]
                    sell_prices.append(float(price))
                    
                plt.scatter(sell_dates, sell_prices, 
                        marker='v', color='red', s=100, label='Sell Signal')
        
        plt.title(f'{ticker} - Long-Term Price History with Trading Signals', fontsize=14)
        plt.ylabel('Price ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Plot recent data (last year)
        plt.subplot(3, 1, 2)
        one_year_ago = datetime.now() - timedelta(days=365)
        recent_data = data[data.index >= one_year_ago]
        
        if not recent_data.empty:
            plt.plot(recent_data.index, recent_data['Close'], label='Close Price', linewidth=1.5)
            plt.plot(recent_data.index, recent_data['Close'].rolling(window=20).mean(), 
                    label='20-day MA', linestyle='--', linewidth=1, color='orange')
            plt.plot(recent_data.index, recent_data['Close'].rolling(window=50).mean(), 
                    label='50-day MA', linestyle='--', linewidth=1, color='red')
            
            # Add recent buy/sell signals
            if not buy_signals.empty:
                recent_buy_dates = buy_signals.index.intersection(recent_data.index)
                if not recent_buy_dates.empty:
                    recent_buy_prices = []
                    for d in recent_buy_dates:
                        price = recent_data.loc[d, 'Close']
                        if isinstance(price, pd.Series):
                            price = price.iloc[0]
                        recent_buy_prices.append(float(price))
                        
                    plt.scatter(recent_buy_dates, recent_buy_prices, 
                            marker='^', color='green', s=100, label='Buy Signal')
            
            if not sell_signals.empty:
                recent_sell_dates = sell_signals.index.intersection(recent_data.index)
                if not recent_sell_dates.empty:
                    recent_sell_prices = []
                    for d in recent_sell_dates:
                        price = recent_data.loc[d, 'Close']
                        if isinstance(price, pd.Series):
                            price = price.iloc[0]
                        recent_sell_prices.append(float(price))
                        
                    plt.scatter(recent_sell_dates, recent_sell_prices, 
                            marker='v', color='red', s=100, label='Sell Signal')
            
            plt.title(f'{ticker} - Recent Price (Last 12 Months)', fontsize=14)
            plt.ylabel('Price ($)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
        else:
            plt.text(0.5, 0.5, "No recent data available", 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        # Plot confidence
        plt.subplot(3, 1, 3)
        plt.bar(signals_df.index, signals_df['Confidence'], color='blue', alpha=0.7)
        plt.axhline(y=0.7, color='red', linestyle='--', label='Threshold (0.7)')
        plt.title('Prediction Confidence', fontsize=14)
        plt.ylabel('Confidence', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Save the figure
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{ticker}_signals.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved enhanced signal visualization to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error visualizing signals: {e}")
        return None

def save_prediction_results(signals_df, ticker, output_dir="ml/prediction/signals/"):
    """
    Save prediction results to file with improved format
    
    Args:
        signals_df (pandas.DataFrame): Trading signals
        ticker (str): Stock ticker symbol
        output_dir (str): Directory to save results
        
    Returns:
        str: Path to saved results
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save signals to CSV
        csv_path = os.path.join(output_dir, f"{ticker}_signals.csv")
        signals_df.to_csv(csv_path)
        
        # Save latest signal to JSON with additional metadata
        latest_signal = signals_df.iloc[-1].to_dict()
        latest_signal['Date'] = latest_signal['Date'].strftime('%Y-%m-%d')
        
        # Add metadata
        latest_signal['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        latest_signal['signal_history'] = len(signals_df)
        
        # Add a summary for quick reference
        if 'Prediction' in latest_signal and 'Confidence' in latest_signal:
            prediction = latest_signal['Prediction']
            confidence = latest_signal['Confidence']
            if prediction == 1:
                direction = "UP"
            else:
                direction = "DOWN"
                
            latest_signal['summary'] = f"Predicted direction: {direction} with {confidence:.2f} confidence"
        
        json_path = os.path.join(output_dir, f"{ticker}_latest.json")
        with open(json_path, 'w') as f:
            json.dump(latest_signal, f, indent=4)
        
        logger.info(f"Saved prediction results to {csv_path} and {json_path}")
        return csv_path
        
    except Exception as e:
        logger.error(f"Error saving prediction results: {e}")
        return None

def predict_for_stock(ticker, model_dir="ml/models/final/", model_type="simple_hybrid", 
                     task="classification", days=100, sequence_length=20, start_year=1950,
                     threshold=0.7, visualize=True, use_augmentation=True,
                     use_historical=False):
    """
    Make predictions for a specific stock with improved reliability and long-term data support
    
    Args:
        ticker (str): Stock ticker symbol
        model_dir (str): Directory containing trained models
        model_type (str): Type of model ('lstm', 'cnn_lstm', etc.)
        task (str): Type of task ('regression' or 'classification')
        days (int): Number of days of data to use for prediction calculations
        sequence_length (int): Length of input sequences
        start_year (int): Start year for historical data (for visualization)
        threshold (float): Confidence threshold for signals
        visualize (bool): Whether to create visualization
        use_augmentation (bool): Whether to use test-time augmentation
        use_historical (bool): Whether to use historical processed data
        
    Returns:
        pandas.DataFrame: Trading signals
    """
    try:
        # Load model - try different models if the specified one fails
        model = load_stock_model(ticker, model_dir, model_type, task)
        
        # If original model fails, try alternative models
        if model is None:
            # Try different model types
            alternative_types = ['simple_hybrid', 'hybrid', 'lstm', 'cnn_lstm', 'bidirectional']
            for alt_type in alternative_types:
                if alt_type == model_type:
                    continue
                logger.info(f"Trying alternative model type: {alt_type}")
                model = load_stock_model(ticker, model_dir, alt_type, task)
                if model is not None:
                    model_type = alt_type
                    break
                    
            # If still no model, try the combined model
            if model is None:
                logger.info("Trying combined SP500 model")
                model = load_stock_model("sp500_combined", model_dir, "hybrid", task)
                if model is not None:
                    model_type = "sp500_combined"
        
        if model is None:
            logger.error(f"Failed to load any model for {ticker}")
            return None
        
        # Get data from start_year until now
        data = None
        if use_historical:
            # Use historical processed data
            logger.info(f"Using historical processed data for {ticker}")
        else:
            # Download new data with long-term history
            data = get_latest_stock_data(ticker, days, start_year)
            
        if data is None and not use_historical:
            logger.error(f"Failed to get data for {ticker}")
            return None
        
        # Prepare data for prediction - only use the most recent data for prediction
        X_sequence, dates = prepare_data_for_prediction(data, ticker, sequence_length, 
                                                      use_historical=use_historical,
                                                      prediction_days=days)
        if X_sequence is None or len(X_sequence) == 0:
            logger.error(f"Failed to prepare data for {ticker} or insufficient data")
            return None
        
        # Make prediction
        if task == "regression":
            predictions = make_prediction(model, X_sequence, task, use_augmentation)
            if predictions is None:
                logger.error(f"Failed to make predictions for {ticker}")
                return None
                
            # Create regression results dataframe
            results_df = pd.DataFrame(index=dates[-len(predictions):])
            results_df['Ticker'] = ticker
            results_df['Date'] = results_df.index
            results_df['Predicted_Return'] = predictions.flatten()
            
            # Add derived columns
            results_df['Direction'] = np.where(results_df['Predicted_Return'] > 0, 'Up', 'Down')
            results_df['Abs_Return'] = np.abs(results_df['Predicted_Return'])
            
            # Save results
            save_dir = "ml/prediction/results/"
            os.makedirs(save_dir, exist_ok=True)
            output_path = os.path.join(save_dir, f"{ticker}_predictions.csv")
            results_df.to_csv(output_path)
            
            # Create visualization if requested
            if visualize and data is not None:
                visualize_signals(data, results_df, ticker)
            
            logger.info(f"Saved regression predictions to {output_path}")
            return results_df
            
        else:
            predictions, signal_strength = make_prediction(model, X_sequence, task, use_augmentation)
            if predictions is None:
                logger.error(f"Failed to make predictions for {ticker}")
                return None
            
            # Generate trading signals
            signals_df = generate_trading_signals(predictions, signal_strength, dates, ticker, threshold)
            if signals_df is None:
                logger.error(f"Failed to generate signals for {ticker}")
                return None
            
            # Save prediction results
            save_prediction_results(signals_df, ticker)
            
            # Create visualization if requested
            if visualize and data is not None:
                visualize_signals(data, signals_df, ticker)
            
            return signals_df
        
    except Exception as e:
        logger.error(f"Error in predict_for_stock for {ticker}: {e}")
        return None

def predict_for_multiple_stocks(tickers, model_dir="ml/models/final/", start_year=1950, **kwargs):
    """
    Make predictions for multiple stocks with improved batch processing and long-term data
    
    Args:
        tickers (list): List of stock ticker symbols
        model_dir (str): Directory containing trained models
        start_year (int): Start year for historical data
        **kwargs: Additional arguments for predict_for_stock
        
    Returns:
        dict: Dictionary of trading signals by ticker
    """
    try:
        results = {}
        failed_tickers = []
        
        for ticker in tqdm(tickers, desc="Making predictions"):
            logger.info(f"Making predictions for {ticker}")
            
            # Try multiple times with exponential backoff
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    signals = predict_for_stock(ticker, model_dir, start_year=start_year, **kwargs)
                    
                    if signals is not None:
                        results[ticker] = signals
                        logger.info(f"Successfully generated predictions for {ticker}")
                        break
                    else:
                        logger.warning(f"Attempt {attempt+1} failed for {ticker}")
                        if attempt < max_attempts - 1:
                            wait_time = 2 ** attempt  # Exponential backoff
                            logger.info(f"Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                except Exception as inner_e:
                    logger.error(f"Error in attempt {attempt+1} for {ticker}: {inner_e}")
                    if attempt == max_attempts - 1:
                        failed_tickers.append(ticker)
            
            if ticker not in results:
                failed_tickers.append(ticker)
                logger.warning(f"Failed to generate predictions for {ticker} after {max_attempts} attempts")
        
        # Generate summary
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'tickers_processed': len(tickers),
            'tickers_succeeded': len(results),
            'success_rate': f"{len(results)/len(tickers)*100:.2f}%",
            'tickers_succeeded_list': list(results.keys()),
            'tickers_failed_list': failed_tickers,
            'start_year': start_year
        }
        
        # Save summary
        summary_path = os.path.join("ml/prediction/signals", "prediction_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Completed predictions for {len(results)}/{len(tickers)} stocks ({summary['success_rate']})")
        logger.info(f"Summary saved to {summary_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in predict_for_multiple_stocks: {e}")
        return {}

def get_latest_signals(signals_dir="ml/prediction/signals/", min_confidence=0.6):
    """
    Get the latest trading signals for all stocks with improved filtering
    
    Args:
        signals_dir (str): Directory containing signal files
        min_confidence (float): Minimum confidence threshold for including signals
        
    Returns:
        pandas.DataFrame: Latest signals for all stocks
    """
    try:
        # Find all JSON files with latest signals
        json_files = [f for f in os.listdir(signals_dir) if f.endswith('_latest.json')]
        
        if not json_files:
            logger.warning("No signal files found")
            return None
        
        # Load all signals
        all_signals = []
        for json_file in json_files:
            with open(os.path.join(signals_dir, json_file), 'r') as f:
                signal = json.load(f)
                
                # Only include signals with sufficient confidence
                if 'Confidence' in signal and signal['Confidence'] >= min_confidence:
                    all_signals.append(signal)
        
        # Create DataFrame
        signals_df = pd.DataFrame(all_signals)
        
        # Convert date string to datetime
        if 'Date' in signals_df.columns:
            signals_df['Date'] = pd.to_datetime(signals_df['Date'])
        
        # Add a combined score column (higher is better buy signal)
        if 'Prediction' in signals_df.columns and 'Confidence' in signals_df.columns:
            # For buy signals (Prediction=1), higher confidence is better
            # For sell signals (Prediction=0), lower confidence is better
            signals_df['Score'] = np.where(
                signals_df['Prediction'] == 1,
                signals_df['Confidence'],
                1 - signals_df['Confidence']
            )
            
            # Sort by score (descending)
            signals_df = signals_df.sort_values('Score', ascending=False)
        else:
            # Sort by confidence (descending)
            signals_df = signals_df.sort_values('Confidence', ascending=False)
        
        logger.info(f"Loaded {len(signals_df)} latest signals with confidence >= {min_confidence}")
        return signals_df
        
    except Exception as e:
        logger.error(f"Error getting latest signals: {e}")
        return None

def compute_portfolio_allocation(signals_df, max_positions=10):
    """
    Compute recommended portfolio allocation based on signals
    
    Args:
        signals_df (pandas.DataFrame): Trading signals dataframe
        max_positions (int): Maximum number of positions to include
        
    Returns:
        pandas.DataFrame: Portfolio allocation recommendations
    """
    try:
        if signals_df is None or signals_df.empty:
            logger.warning("No signals available for portfolio allocation")
            return None
        
        # Filter to only buy signals with high confidence
        buy_signals = signals_df[
            (signals_df['Signal'].str.contains('Buy')) & 
            (signals_df['Confidence'] >= 0.7)
        ]
        
        if buy_signals.empty:
            logger.warning("No buy signals with sufficient confidence found")
            return None
        
        # Sort by confidence and limit to max positions
        top_buys = buy_signals.sort_values('Confidence', ascending=False).head(max_positions)
        
        # Calculate position sizes based on confidence
        total_confidence = top_buys['Confidence'].sum()
        top_buys['Allocation'] = (top_buys['Confidence'] / total_confidence * 100).round(2)
        
        # Add recommendation details
        top_buys['Recommendation'] = 'Buy'
        top_buys['Notes'] = top_buys.apply(
            lambda x: f"Strong conviction signal with {x['Confidence']:.2f} confidence", 
            axis=1
        )
        
        # Select relevant columns
        allocation_df = top_buys[['Ticker', 'Recommendation', 'Confidence', 'Allocation', 'Notes']]
        
        logger.info(f"Generated portfolio allocation for {len(allocation_df)} positions")
        return allocation_df
        
    except Exception as e:
        logger.error(f"Error computing portfolio allocation: {e}")
        return None

# Main execution when script is run directly
if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate stock predictions with long-term data')
    parser.add_argument('--ticker', type=str, help='Process only this ticker')
    parser.add_argument('--subset', type=int, help='Process only a subset of stocks')
    parser.add_argument('--start-year', type=int, default=1950, help='Start year for historical data')
    parser.add_argument('--days', type=int, default=100, help='Number of days for prediction calculation')
    parser.add_argument('--visualize', action='store_true', default=True, help='Create visualizations')
    args = parser.parse_args()
    
    # Define a list of major stocks to focus on
    if args.ticker:
        major_stocks = [args.ticker]
    else:
        major_stocks = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "JNJ", 
            "JPM", "V", "PG", "UNH", "HD", "MA", "NVDA", "BAC", "DIS", "CSCO"
        ]
        
        # Apply subset if requested
        if args.subset and args.subset < len(major_stocks):
            major_stocks = major_stocks[:args.subset]
    
    # Make predictions for these stocks
    results = predict_for_multiple_stocks(
        major_stocks,
        model_type='simple_hybrid',
        task='classification',
        days=args.days,
        start_year=args.start_year,
        sequence_length=20,
        threshold=0.7,
        visualize=args.visualize,
        use_augmentation=True,
        use_historical=False  # Download new data instead of using historical processed data
    )
    
    # Get latest signals
    latest_signals = get_latest_signals(min_confidence=0.6)
    
    if latest_signals is not None:
        # Display the top 10 signals
        print("\nTop 10 Latest Trading Signals:")
        display_cols = ['Ticker', 'Date', 'Signal', 'Confidence', 'Action']
        available_cols = [col for col in display_cols if col in latest_signals.columns]
        print(latest_signals[available_cols].head(10))
        
        # Generate portfolio allocation
        allocation = compute_portfolio_allocation(latest_signals)
        if allocation is not None:
            print("\nRecommended Portfolio Allocation:")
            print(allocation)