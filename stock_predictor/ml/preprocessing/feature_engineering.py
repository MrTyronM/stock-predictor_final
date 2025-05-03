"""
Feature engineering module for stock prediction - Python 3.12 Compatible Version
This module creates technical indicators and prepares data for ML models
"""
import os
import pandas as pd
import numpy as np
import logging
import time  # Add missing time module
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import joblib
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml/logs/feature_engineering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("feature_engineering")

# Create necessary directories
os.makedirs("ml/logs", exist_ok=True)
os.makedirs("ml/data/processed", exist_ok=True)
os.makedirs("ml/preprocessing/scalers", exist_ok=True)

# Define talib as global variable
talib = None

# Try to import TA-Lib, falling back to our adapter if not available
try:
    import talib
    logger.info("Using TA-Lib for technical indicators")
except ImportError:
    try:
        # Try to use our adapter
        from ml.preprocessing.ta_adapter import *
        logger.info("Using TA adapter (alternative libraries) for technical indicators")
    except ImportError:
        # If neither is available, create dummy functions to avoid errors
        logger.warning("TA-Lib and alternatives not available. Some indicators will be missing.")

# Define these functions at module level to ensure they're always available
def SMA(prices, timeperiod=14):
    # Basic implementation of Simple Moving Average
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    return prices.rolling(window=timeperiod).mean().values

def EMA(prices, timeperiod=14):
    # Basic implementation of Exponential Moving Average
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    return prices.ewm(span=timeperiod, adjust=False).mean().values

def RSI(prices, timeperiod=14):
    # Basic implementation of RSI
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))).values

def MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9):
    # Basic implementation of MACD
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    ema_fast = prices.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = prices.ewm(span=slowperiod, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signalperiod, adjust=False).mean()
    hist = macd - signal
    return macd.values, signal.values, hist.values

def BBANDS(prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
    # Basic implementation of Bollinger Bands
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    middle = prices.rolling(window=timeperiod).mean()
    std = prices.rolling(window=timeperiod).std()
    upper = middle + nbdevup * std
    lower = middle - nbdevdn * std
    return upper.values, middle.values, lower.values

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe with robust error handling
    
    Args:
        df (pandas.DataFrame): Stock data with OHLCV columns
        
    Returns:
        pandas.DataFrame: DataFrame with added technical indicators
    """
    try:
        # Make sure the dataframe has the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column {col} not found in dataframe")
                return df
        
        # Make a copy to avoid modifying the original
        df_result = df.copy()
        
        # Handle NaN and infinite values
        df_result = df_result.replace([np.inf, -np.inf], np.nan)
        
        # Price derived indicators - Add with robust error handling
        try:
            # Moving Averages
            df_result['MA5'] = SMA(df_result['Close'], timeperiod=5)
            df_result['MA10'] = SMA(df_result['Close'], timeperiod=10)
            df_result['MA20'] = SMA(df_result['Close'], timeperiod=20)
            df_result['MA50'] = SMA(df_result['Close'], timeperiod=50)
            df_result['MA200'] = SMA(df_result['Close'], timeperiod=200)
        except Exception as e:
            logger.warning(f"Error calculating Moving Averages: {e}")
            # Fall back to pandas implementation
            df_result['MA5'] = df_result['Close'].rolling(window=5).mean()
            df_result['MA10'] = df_result['Close'].rolling(window=10).mean()
            df_result['MA20'] = df_result['Close'].rolling(window=20).mean()
            df_result['MA50'] = df_result['Close'].rolling(window=50).mean()
            df_result['MA200'] = df_result['Close'].rolling(window=200).mean()

        try:
            # Exponential Moving Averages
            df_result['EMA5'] = EMA(df_result['Close'], timeperiod=5)
            df_result['EMA10'] = EMA(df_result['Close'], timeperiod=10)
            df_result['EMA20'] = EMA(df_result['Close'], timeperiod=20)
            df_result['EMA50'] = EMA(df_result['Close'], timeperiod=50)
            df_result['EMA200'] = EMA(df_result['Close'], timeperiod=200)
        except Exception as e:
            logger.warning(f"Error calculating EMAs: {e}")
            # Fall back to pandas implementation
            df_result['EMA5'] = df_result['Close'].ewm(span=5, adjust=False).mean()
            df_result['EMA10'] = df_result['Close'].ewm(span=10, adjust=False).mean()
            df_result['EMA20'] = df_result['Close'].ewm(span=20, adjust=False).mean()
            df_result['EMA50'] = df_result['Close'].ewm(span=50, adjust=False).mean()
            df_result['EMA200'] = df_result['Close'].ewm(span=200, adjust=False).mean()

        try:
            # Moving Average Convergence Divergence (MACD)
            macd, macd_signal, macd_hist = MACD(df_result['Close'])
            df_result['MACD'] = macd
            df_result['MACD_signal'] = macd_signal
            df_result['MACD_hist'] = macd_hist
        except Exception as e:
            logger.warning(f"Error calculating MACD: {e}")
            # Create basic MACD using EMA differences if MACD function fails
            try:
                df_result['MACD'] = df_result['EMA20'] - df_result['EMA50']
                df_result['MACD_signal'] = df_result['MACD'].ewm(span=9, adjust=False).mean()
                df_result['MACD_hist'] = df_result['MACD'] - df_result['MACD_signal']
            except Exception as e2:
                logger.warning(f"Error calculating fallback MACD: {e2}")

        # Bollinger Bands - FIX FOR THE ERROR
        try:
            # Direct calculation instead of using the function
            middle = df_result['Close'].rolling(window=20).mean()
            std = df_result['Close'].rolling(window=20).std()
            upper = middle + 2 * std
            lower = middle - 2 * std
            
            # Assign individually
            df_result['BB_middle'] = middle
            df_result['BB_upper'] = upper
            df_result['BB_lower'] = lower
            
            # Calculate derived metrics individually
            df_result['BB_width'] = (upper - lower) / middle
            
            # FIX: Calculate %B carefully to avoid error
            bb_range = upper - lower
            # Avoid division by zero
            bb_range = bb_range.replace(0, np.nan)
            df_result['BB_%B'] = (df_result['Close'] - lower) / bb_range
        except Exception as e:
            logger.warning(f"Error calculating Bollinger Bands: {e}")
            try:
                # Minimal fallback
                df_result['BB_middle'] = df_result['MA20']
                std = df_result['Close'].rolling(window=20).std()
                df_result['BB_upper'] = df_result['BB_middle'] + 2 * std
                df_result['BB_lower'] = df_result['BB_middle'] - 2 * std
                # Skip the problematic calculations
            except Exception as e2:
                logger.warning(f"Could not calculate fallback Bollinger Bands: {e2}")

        try:
            # Relative Strength Index (RSI)
            df_result['RSI'] = RSI(df_result['Close'], timeperiod=14)
        except Exception as e:
            logger.warning(f"Error calculating RSI: {e}")
            # Implement a basic RSI calculation as fallback
            try:
                delta = df_result['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                df_result['RSI'] = 100 - (100 / (1 + rs))
            except Exception as e2:
                logger.warning(f"Error calculating fallback RSI: {e2}")

        # Custom Indicators - with error handling for each
        try:
            # Daily Returns
            df_result['Daily_Return'] = df_result['Close'].pct_change()
        except Exception as e:
            logger.warning(f"Error calculating Daily Returns: {e}")
        
        try:
            # Log Returns
            df_result['Log_Return'] = np.log(df_result['Close'] / df_result['Close'].shift(1))
        except Exception as e:
            logger.warning(f"Error calculating Log Returns: {e}")
        
        try:
            # Volatility (20-day standard deviation of returns)
            df_result['Volatility_20d'] = df_result['Daily_Return'].rolling(window=20).std()
        except Exception as e:
            logger.warning(f"Error calculating Volatility: {e}")
        
        try:
            # Daily High-Low Range
            df_result['Daily_Range'] = (df_result['High'] - df_result['Low']) / df_result['Close']
        except Exception as e:
            logger.warning(f"Error calculating Daily Range: {e}")
        
        try:
            # Price Momentum
            df_result['Momentum_5d'] = df_result['Close'] / df_result['Close'].shift(5) - 1
            df_result['Momentum_10d'] = df_result['Close'] / df_result['Close'].shift(10) - 1
            df_result['Momentum_20d'] = df_result['Close'] / df_result['Close'].shift(20) - 1
        except Exception as e:
            logger.warning(f"Error calculating Price Momentum: {e}")
        
        try:
            # Volume Momentum
            df_result['Volume_1d_change'] = df_result['Volume'] / df_result['Volume'].shift(1)
            df_result['Volume_5d_avg'] = df_result['Volume'].rolling(window=5).mean()
            df_result['Volume_10d_avg'] = df_result['Volume'].rolling(window=10).mean()
            df_result['Volume_to_MA20'] = df_result['Volume'] / df_result['Volume'].rolling(window=20).mean()
        except Exception as e:
            logger.warning(f"Error calculating Volume Momentum: {e}")
        
        try:
            # Price Acceleration
            df_result['Price_Accel'] = df_result['Momentum_5d'] - df_result['Momentum_5d'].shift(5)
        except Exception as e:
            logger.warning(f"Error calculating Price Acceleration: {e}")
        
        try:
            # Moving Average Crossovers (binary indicators)
            df_result['MA_5_10_cross'] = np.where(df_result['MA5'] > df_result['MA10'], 1, 0)
            df_result['MA_10_20_cross'] = np.where(df_result['MA10'] > df_result['MA20'], 1, 0)
            df_result['MA_20_50_cross'] = np.where(df_result['MA20'] > df_result['MA50'], 1, 0)
            df_result['MA_50_200_cross'] = np.where(df_result['MA50'] > df_result['MA200'], 1, 0)
        except Exception as e:
            logger.warning(f"Error calculating MA crossovers: {e}")
        
        try:    
            # Distance from Moving Averages (percentage)
            df_result['Dist_from_MA50_pct'] = (df_result['Close'] - df_result['MA50']) / df_result['MA50'] * 100
            df_result['Dist_from_MA200_pct'] = (df_result['Close'] - df_result['MA200']) / df_result['MA200'] * 100
        except Exception as e:
            logger.warning(f"Error calculating MA distances: {e}")
        
        # Added these basic indicators that are more likely to succeed
        df_result['Close_normalized'] = df_result['Close'] / df_result['Close'].iloc[0] if len(df_result) > 0 else df_result['Close']
        df_result['Open_Close_Diff'] = df_result['Close'] - df_result['Open']
        df_result['High_Low_Diff'] = df_result['High'] - df_result['Low']
        
        # Replace NaN values with 0
        df_result = df_result.replace([np.inf, -np.inf], np.nan)
        
        # Ensure we have at least some features
        if len(df_result.columns) <= len(df.columns):
            logger.warning("No indicators were successfully added. Adding basic ones.")
            # Add some very basic indicators that are guaranteed to work
            df_result['Close_Shift_1'] = df_result['Close'].shift(1)
            df_result['Close_Shift_2'] = df_result['Close'].shift(2)
            df_result['Volume_Shift_1'] = df_result['Volume'].shift(1)
        
        logger.info(f"Added {len(df_result.columns) - len(df.columns)} technical indicators")
        return df_result
        
    except Exception as e:
        logger.error(f"Error adding technical indicators: {e}")
        return df

def create_prediction_features(df, window_sizes=[5, 10, 20], target_horizons=[1, 3, 5, 10], threshold=0.01):
    """
    Create lagged features and target variables for prediction
    
    Args:
        df (pandas.DataFrame): DataFrame with technical indicators
        window_sizes (list): List of window sizes for lagged features
        target_horizons (list): List of horizons for target variables
        threshold (float): Threshold for determining significant price movement
        
    Returns:
        pandas.DataFrame: DataFrame with features and targets
    """
    try:
        df_result = df.copy()
        
        # Create lagged features for each selected indicator
        key_indicators = [
            'Close', 'Volume', 'MA5', 'MA20', 'MA50', 'EMA20', 
            'MACD', 'RSI', 'BB_%B', 'Daily_Return', 'Volatility_20d'
        ]
        
        # Check which indicators are available
        available_indicators = [ind for ind in key_indicators if ind in df_result.columns]
        
        for indicator in available_indicators:
            for window in window_sizes:
                # Create lagged values
                for lag in range(1, window + 1):
                    df_result[f'{indicator}_lag_{lag}'] = df_result[indicator].shift(lag)
                
                # Create rolling statistics
                df_result[f'{indicator}_mean_{window}'] = df_result[indicator].rolling(window=window).mean()
                df_result[f'{indicator}_std_{window}'] = df_result[indicator].rolling(window=window).std()
                df_result[f'{indicator}_min_{window}'] = df_result[indicator].rolling(window=window).min()
                df_result[f'{indicator}_max_{window}'] = df_result[indicator].rolling(window=window).max()
        
        # Trend strength indicators
        if 'RSI' in df_result.columns:
            df_result['RSI_Trend_Strength'] = np.where(df_result['RSI'] > 50, 1, 0)
        
        # Volatility regime indicators
        if 'Volatility_20d' in df_result.columns:
            df_result['High_Volatility'] = np.where(df_result['Volatility_20d'] > df_result['Volatility_20d'].rolling(window=50).mean(), 1, 0)
        
        # Create target variables for different horizons - IMPROVED FORMAT
        for horizon in target_horizons:
            # Future returns
            df_result[f'Target_Return_{horizon}d'] = df_result['Close'].pct_change(horizon).shift(-horizon)
            
            # Binary target (1 if price goes up, 0 if it goes down)
            # Ensure it's always an integer (0 or 1) for classification
            df_result[f'Target_Direction_{horizon}d'] = np.where(df_result[f'Target_Return_{horizon}d'] > 0, 1, 0).astype(int)
            
            # Three-class target (1 for significant rise, -1 for significant drop, 0 for flat)
            # Using threshold for significance (e.g., 1%)
            df_result[f'Target_Signal_{horizon}d'] = np.where(df_result[f'Target_Return_{horizon}d'] > threshold, 1, 
                                                np.where(df_result[f'Target_Return_{horizon}d'] < -threshold, -1, 0)).astype(int)
            
            # Significant move indicator (binary classification for significant movement in either direction)
            df_result[f'Target_Significant_{horizon}d'] = np.where(
                abs(df_result[f'Target_Return_{horizon}d']) > threshold, 1, 0
            ).astype(int)
        
        logger.info(f"Created features for {len(window_sizes)} window sizes and targets for {len(target_horizons)} horizons")
        return df_result
        
    except Exception as e:
        logger.error(f"Error creating prediction features: {e}")
        return df

def select_important_features(df, target_col, n_features=40):
    """
    Select the most important features using feature selection techniques
    
    Args:
        df (pandas.DataFrame): DataFrame with features and targets
        target_col (str): Target column name
        n_features (int): Number of features to select
        
    Returns:
        list: List of selected feature names
    """
    try:
        # Drop rows with NaN values
        df_clean = df.dropna()
        
        # Get feature columns (excluding target columns)
        feature_cols = [col for col in df_clean.columns if not col.startswith('Target_')]
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        # Choose appropriate score function
        if target_col.startswith('Target_Return'):
            # For regression
            score_func = f_regression
        else:
            # For classification
            score_func = mutual_info_regression
        
        # Select features
        selector = SelectKBest(score_func=score_func, k=min(n_features, len(feature_cols)))
        selector.fit(X, y)
        
        # Get indices of selected features
        selected_indices = selector.get_support(indices=True)
        
        # Get feature names
        selected_features = [feature_cols[i] for i in selected_indices]
        
        logger.info(f"Selected {len(selected_features)} important features for {target_col}")
        return selected_features
        
    except Exception as e:
        logger.error(f"Error selecting important features: {e}")
        # Return all features if selection fails
        return [col for col in df.columns if not col.startswith('Target_')]

def prepare_train_test_data(df, target_col, test_size=0.2, sequence_length=20, prediction_horizon=5):
    """
    Prepare training and testing data for time series prediction with improved handling
    
    Args:
        df (pandas.DataFrame): DataFrame with features and targets
        target_col (str): Name of the target column
        test_size (float): Proportion of data to use for testing
        sequence_length (int): Length of input sequences
        prediction_horizon (int): How many steps ahead to predict
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    try:
        # Drop rows with NaN values
        df = df.dropna()
        
        # Check if target column exists
        if target_col not in df.columns:
            logger.error(f"Target column {target_col} not found in dataframe")
            return None, None, None, None
        
        # Select only important features to reduce dimensionality
        important_features = select_important_features(df, target_col)
        
        # Add the target column to the feature set
        feature_cols = important_features
        
        # Split into features and target
        X = df[feature_cols].values
        
        # Properly format target based on column type
        if target_col.startswith('Target_Direction') or target_col.startswith('Target_Significant'):
            # Binary classification - ensure target is int
            y = df[target_col].astype(int).values
        elif target_col.startswith('Target_Signal'):
            # Multiclass - could use one-hot encoding if needed
            y = df[target_col].astype(int).values
        else:
            # Regression
            y = df[target_col].values
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(len(df) - sequence_length - prediction_horizon + 1):
            X_sequences.append(X[i:i+sequence_length])
            y_sequences.append(y[i+sequence_length+prediction_horizon-1])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        # Split into training and testing sets
        split_idx = int(len(X_sequences) * (1 - test_size))
        X_train = X_sequences[:split_idx]
        X_test = X_sequences[split_idx:]
        y_train = y_sequences[:split_idx]
        y_test = y_sequences[split_idx:]
        
        logger.info(f"Prepared data with sequence length {sequence_length} and prediction horizon {prediction_horizon}")
        logger.info(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error preparing train/test data: {e}")
        return None, None, None, None

def balance_classification_dataset(X, y):
    """
    Balance binary classification dataset using simplified techniques
    for Python 3.12 compatibility
    
    Args:
        X (numpy.ndarray): Features array
        y (numpy.ndarray): Target array
        
    Returns:
        tuple: Balanced X and y arrays
    """
    try:
        # Try using SMOTE if available
        try:
            from imblearn.over_sampling import SMOTE
            
            original_shape = X.shape
            
            # Reshape for SMOTE (SMOTE doesn't work directly with 3D data)
            X_reshaped = X.reshape(X.shape[0], -1)
            
            # Apply SMOTE to balance classes
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)
            
            # Reshape back to original 3D shape
            X_balanced = X_resampled.reshape(X_resampled.shape[0], original_shape[1], original_shape[2])
            
            logger.info(f"Balanced dataset from {X.shape[0]} to {X_balanced.shape[0]} samples using SMOTE")
            return X_balanced, y_resampled
        
        except ImportError:
            # If SMOTE not available, use a simple random oversampling approach
            classes, counts = np.unique(y, return_counts=True)
            max_count = np.max(counts)
            
            X_balanced = []
            y_balanced = []
            
            for cls in classes:
                # Get indices of this class
                indices = np.where(y == cls)[0]
                
                # Add all existing samples
                X_balanced.append(X[indices])
                y_balanced.append(y[indices])
                
                # Oversample if needed
                if len(indices) < max_count:
                    # Randomly sample with replacement
                    n_oversample = max_count - len(indices)
                    oversample_indices = np.random.choice(indices, size=n_oversample, replace=True)
                    X_balanced.append(X[oversample_indices])
                    y_balanced.append(y[oversample_indices])
            
            # Combine and shuffle
            X_balanced = np.vstack(X_balanced)
            y_balanced = np.concatenate(y_balanced)
            
            # Shuffle
            indices = np.random.permutation(len(X_balanced))
            X_balanced = X_balanced[indices]
            y_balanced = y_balanced[indices]
            
            logger.info(f"Balanced dataset from {X.shape[0]} to {X_balanced.shape[0]} samples using random oversampling")
            return X_balanced, y_balanced
            
    except Exception as e:
        logger.error(f"Error balancing dataset: {e}. Using original data.")
        return X, y

def augment_financial_data(X_train, y_train, augmentation_factor=2):
    """
    Augment financial time series data with domain-specific techniques
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training targets
        augmentation_factor (int): How many augmented copies to create
        
    Returns:
        tuple: Augmented X and y arrays
    """
    try:
        X_augmented = [X_train]
        y_augmented = [y_train]
        
        # 1. Magnitude warping (small scaling of values)
        for i in range(1):
            scale_factor = np.random.uniform(0.95, 1.05, size=(1, X_train.shape[2]))
            X_warped = X_train * scale_factor
            X_augmented.append(X_warped)
            y_augmented.append(y_train)
            
        # 2. Jittering (add small noise)
        for i in range(1):
            noise_level = 0.01  # 1% noise
            noise = np.random.normal(0, noise_level, X_train.shape)
            X_noisy = X_train + noise
            X_augmented.append(X_noisy)
            y_augmented.append(y_train)
        
        # Combine and shuffle
        X_combined = np.vstack(X_augmented)
        y_combined = np.concatenate(y_augmented)
        
        # Create random permutation of indices
        indices = np.random.permutation(len(X_combined))
        
        logger.info(f"Augmented dataset from {len(X_train)} to {len(X_combined)} samples")
        return X_combined[indices], y_combined[indices]
        
    except Exception as e:
        logger.error(f"Error augmenting data: {e}. Using original data.")
        return X_train, y_train

def scale_features(df, scaler_path=None, feature_cols=None, scaler_type='standard'):
    """
    Scale features using StandardScaler or MinMaxScaler with improved error handling
    
    Args:
        df (pandas.DataFrame): DataFrame with features
        scaler_path (str): Path to save/load scaler
        feature_cols (list): List of columns to scale
        scaler_type (str): Type of scaler ('standard' or 'minmax')
        
    Returns:
        pandas.DataFrame, sklearn.preprocessing._data.Scaler: Scaled DataFrame and scaler
    """
    try:
        # Copy dataframe to avoid modifying the original
        df_scaled = df.copy()
        
        # If feature columns not specified, scale all numeric columns
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            # Exclude target columns from scaling
            feature_cols = [col for col in feature_cols if isinstance(col, str) and not col.startswith('Target_')]
        
        # Create scaler
        if scaler_type.lower() == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        
        # If scaler path is provided and file exists, load the scaler
        if scaler_path and os.path.exists(scaler_path):
            # Don't use existing scaler if features don't match
            try:
                scaler = joblib.load(scaler_path)
                # Check if scaler is compatible with current data
                if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != len(feature_cols):
                    logger.warning(f"Loaded scaler expects {scaler.n_features_in_} features but got {len(feature_cols)}. Creating new scaler.")
                    # Create new scaler instead
                    if scaler_type.lower() == 'minmax':
                        scaler = MinMaxScaler()
                    else:
                        scaler = StandardScaler()
                    scaler.fit(df[feature_cols])
                    # Save the new scaler
                    joblib.dump(scaler, scaler_path)
                    logger.info(f"Created and saved new scaler to {scaler_path}")
                else:
                    logger.info(f"Loaded scaler from {scaler_path}")
            except Exception as e:
                logger.warning(f"Error loading scaler: {e}. Creating new scaler.")
                # Create new scaler
                if scaler_type.lower() == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    scaler = StandardScaler()
                scaler.fit(df[feature_cols])
                # Save the new scaler
                joblib.dump(scaler, scaler_path)
                logger.info(f"Created and saved new scaler to {scaler_path}")
        else:
            # Fit the scaler on the features
            scaler.fit(df[feature_cols])
            
            # Save scaler if path is provided
            if scaler_path:
                os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
                joblib.dump(scaler, scaler_path)
                logger.info(f"Saved scaler to {scaler_path}")
        
        # Transform the features
        df_scaled[feature_cols] = scaler.transform(df[feature_cols])
        
        logger.info(f"Scaled {len(feature_cols)} features")
        return df_scaled, scaler
        
    except Exception as e:
        logger.error(f"Error scaling features: {e}")
        # Fall back to original data if scaling fails
        return df, None

def prepare_multiple_stocks(raw_data_path="ml/data/raw/", processed_data_path="ml/data/processed/", 
                          window_sizes=[5, 10, 20], target_horizons=[1, 3, 5, 10],
                          use_subset=None):
    """
    Process multiple stocks and prepare them for machine learning
    
    Args:
        raw_data_path (str): Directory with raw stock data
        processed_data_path (str): Directory to save processed data
        window_sizes (list): Window sizes for feature creation
        target_horizons (list): Target prediction horizons
        use_subset (int): Number of stocks to process (None for all)
        
    Returns:
        dict: Dictionary of processed DataFrames by ticker
    """
    try:
        # Create output directory
        os.makedirs(processed_data_path, exist_ok=True)
        
        # Get list of available tickers
        csv_files = [f for f in os.listdir(raw_data_path) if f.endswith('.csv') and f != 'metadata.csv']
        tickers = [os.path.splitext(f)[0] for f in csv_files]
        
        # Use subset if specified
        if use_subset and isinstance(use_subset, int) and use_subset < len(tickers):
            tickers = tickers[:use_subset]
        
        logger.info(f"Processing {len(tickers)} stocks")
        
        processed_dfs = {}
        
        # Process each ticker
        for ticker in tqdm(tickers, desc="Processing stocks"):
            try:
                # Load raw data
                file_path = os.path.join(raw_data_path, f"{ticker}.csv")
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Add technical indicators
                df_with_indicators = add_technical_indicators(df)
                
                # Create prediction features
                df_features = create_prediction_features(df_with_indicators, window_sizes, target_horizons)
                
                # Scale features
                scaler_path = os.path.join("ml/preprocessing/scalers", f"{ticker}_scaler.pkl")
                df_scaled, _ = scale_features(df_features, scaler_path)
                
                # Save processed data
                output_path = os.path.join(processed_data_path, f"{ticker}_processed.csv")
                df_scaled.to_csv(output_path)
                
                processed_dfs[ticker] = df_scaled
                logger.info(f"Processed {ticker} and saved to {output_path}")
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue
        
        logger.info(f"Completed processing {len(processed_dfs)} stocks")
        return processed_dfs
        
    except Exception as e:
        logger.error(f"Error in prepare_multiple_stocks: {e}")
        return {}

def create_combined_dataset(processed_data_path="ml/data/processed/", target_col="Target_Direction_5d", 
                           sequence_length=20, test_size=0.2, max_stocks=None):
    """
    Create a combined dataset from all processed stock data
    
    Args:
        processed_data_path (str): Path to processed data files
        target_col (str): Target column name
        sequence_length (int): Length of input sequences
        test_size (float): Proportion of data to use for testing
        max_stocks (int): Maximum number of stocks to include (None for all)
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    try:
        # Get list of processed data files
        import glob
        data_files = glob.glob(os.path.join(processed_data_path, "*_processed.csv"))
        
        if max_stocks is not None and max_stocks < len(data_files):
            # Randomly select max_stocks files if limited
            import random
            random.shuffle(data_files)
            data_files = data_files[:max_stocks]
        
        logger.info(f"Creating combined dataset from {len(data_files)} stocks")
        
        all_X_train = []
        all_X_test = []
        all_y_train = []
        all_y_test = []
        
        # Process each stock's data
        for file_path in tqdm(data_files, desc="Loading stock data"):
            try:
                # Extract ticker from filename
                ticker = os.path.basename(file_path).split('_')[0]
                
                # Load data
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Check if target column exists
                if target_col not in df.columns:
                    logger.warning(f"Target column {target_col} not found for {ticker}, skipping")
                    continue
                
                # Prepare train/test data
                X_train, X_test, y_train, y_test = prepare_train_test_data(
                    df, target_col, test_size, sequence_length
                )
                
                if X_train is None or len(X_train) == 0:
                    logger.warning(f"No valid data for {ticker}, skipping")
                    continue
                
                # Add ticker-specific identifier feature
                # This helps the model learn stock-specific patterns
                ticker_id = np.ones((X_train.shape[0], X_train.shape[1], 1)) * hash(ticker) % 100 / 100.0
                X_train = np.concatenate([X_train, ticker_id], axis=2)
                
                ticker_id_test = np.ones((X_test.shape[0], X_test.shape[1], 1)) * hash(ticker) % 100 / 100.0
                X_test = np.concatenate([X_test, ticker_id_test], axis=2)
                
                # Append to combined dataset
                all_X_train.append(X_train)
                all_X_test.append(X_test)
                all_y_train.append(y_train)
                all_y_test.append(y_test)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        if not all_X_train:
            logger.error("No valid data found for any stock")
            return None, None, None, None
        
        # Combine all data
        X_train_combined = np.vstack(all_X_train)
        X_test_combined = np.vstack(all_X_test)
        y_train_combined = np.concatenate(all_y_train)
        y_test_combined = np.concatenate(all_y_test)
        
        # Shuffle training data
        train_indices = np.arange(len(X_train_combined))
        np.random.shuffle(train_indices)
        X_train_combined = X_train_combined[train_indices]
        y_train_combined = y_train_combined[train_indices]
        
        # For classification, balance the dataset
        if target_col.startswith('Target_Direction') or target_col.startswith('Target_Significant'):
            X_train_combined, y_train_combined = balance_classification_dataset(X_train_combined, y_train_combined)
        
        # Augment training data
        X_train_combined, y_train_combined = augment_financial_data(X_train_combined, y_train_combined)
        
        logger.info(f"Combined dataset created: X_train shape: {X_train_combined.shape}, "
                  f"X_test shape: {X_test_combined.shape}")
        
        return X_train_combined, X_test_combined, y_train_combined, y_test_combined
        
    except Exception as e:
        logger.error(f"Error creating combined dataset: {e}")
        return None, None, None, None