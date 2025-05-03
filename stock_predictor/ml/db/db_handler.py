"""
Database handler module for stock market prediction

This module handles database operations for storing predictions and results
"""
import os
import sys
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
import sqlite3
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import pymysql

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml/logs/db_handler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("db_handler")

# Create necessary directories
os.makedirs("ml/logs", exist_ok=True)
os.makedirs("db", exist_ok=True)

# Create base class for SQLAlchemy models
Base = declarative_base()

# Define SQLAlchemy models
class StockData(Base):
    """Model for storing stock price data"""
    __tablename__ = 'stock_data'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    adjusted_close = Column(Float)
    
    def __repr__(self):
        return f"<StockData(ticker='{self.ticker}', date='{self.date}', close='{self.close}')>"

class PredictionResult(Base):
    """Model for storing prediction results"""
    __tablename__ = 'prediction_results'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    prediction_date = Column(DateTime, nullable=False)
    model_type = Column(String(50))
    predicted_value = Column(Float)
    confidence = Column(Float)
    signal = Column(String(20))
    actual_value = Column(Float, nullable=True)
    error = Column(Float, nullable=True)
    
    def __repr__(self):
        return f"<PredictionResult(ticker='{self.ticker}', date='{self.date}', signal='{self.signal}')>"

class ModelMetadata(Base):
    """Model for storing trained model metadata"""
    __tablename__ = 'model_metadata'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    model_type = Column(String(50))
    training_date = Column(DateTime, nullable=False)
    parameters = Column(Text)  # JSON string of parameters
    performance_metrics = Column(Text)  # JSON string of metrics
    model_path = Column(String(255))
    
    def __repr__(self):
        return f"<ModelMetadata(ticker='{self.ticker}', model_type='{self.model_type}')>"

class UserFeedback(Base):
    """Model for storing user feedback on predictions"""
    __tablename__ = 'user_feedback'
    
    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer, nullable=False)
    user_id = Column(String(50), nullable=True)
    feedback_type = Column(String(20))  # 'accurate', 'inaccurate', 'helpful', etc.
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<UserFeedback(prediction_id='{self.prediction_id}', feedback_type='{self.feedback_type}')>"

def create_database_connection(db_type='sqlite', user=None, password=None, host=None, database=None):
    """
    Create database connection
    
    Args:
        db_type (str): Type of database ('sqlite' or 'mysql')
        user (str): Database user (for MySQL)
        password (str): Database password (for MySQL)
        host (str): Database host (for MySQL)
        database (str): Database name
        
    Returns:
        sqlalchemy.engine.Engine: Database connection engine
    """
    try:
        if db_type.lower() == 'sqlite':
            # SQLite connection
            db_path = os.path.join('db', 'stock_prediction.db')
            engine = create_engine(f'sqlite:///{db_path}', 
                                 connect_args={'check_same_thread': False},
                                 poolclass=StaticPool)
            logger.info(f"Created SQLite database connection to {db_path}")
        else:
            # MySQL connection
            if not all([user, password, host, database]):
                raise ValueError("For MySQL connection, user, password, host, and database must be provided")
            
            engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')
            logger.info(f"Created MySQL database connection to {host}/{database}")
        
        # Create tables if they don't exist
        Base.metadata.create_all(engine)
        
        return engine
        
    except Exception as e:
        logger.error(f"Error creating database connection: {e}")
        return None

def store_stock_data(engine, df, ticker):
    """
    Store stock price data in the database
    
    Args:
        engine (sqlalchemy.engine.Engine): Database connection engine
        df (pandas.DataFrame): DataFrame with stock data
        ticker (str): Stock ticker symbol
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create a copy of the dataframe
        df_copy = df.copy()
        
        # Reset index to make date a column
        if df_copy.index.name == 'Date':
            df_copy = df_copy.reset_index()
        
        # Ensure date column is named consistently
        if 'Date' in df_copy.columns:
            df_copy.rename(columns={'Date': 'date'}, inplace=True)
        
        # Add ticker column
        df_copy['ticker'] = ticker
        
        # Standardize column names
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adjusted_close'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df_copy.columns:
                df_copy.rename(columns={old_col: new_col}, inplace=True)
        
        # Select only the columns that match the model
        table_columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adjusted_close']
        df_to_insert = df_copy[table_columns].copy()
        
        # Store data in database
        df_to_insert.to_sql('stock_data', engine, if_exists='append', index=False)
        
        logger.info(f"Stored {len(df_to_insert)} rows of stock data for {ticker}")
        return True
        
    except Exception as e:
        logger.error(f"Error storing stock data for {ticker}: {e}")
        return False

def store_prediction_results(engine, predictions_df):
    """
    Store prediction results in the database
    
    Args:
        engine (sqlalchemy.engine.Engine): Database connection engine
        predictions_df (pandas.DataFrame): DataFrame with prediction results
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create a copy of the dataframe
        df_copy = predictions_df.copy()
        
        # Add prediction date
        df_copy['prediction_date'] = datetime.now()
        
        # Standardize column names
        if 'Date' in df_copy.columns:
            df_copy.rename(columns={'Date': 'date'}, inplace=True)
        
        if 'Ticker' in df_copy.columns:
            df_copy.rename(columns={'Ticker': 'ticker'}, inplace=True)
        
        if 'Prediction' in df_copy.columns:
            df_copy.rename(columns={'Prediction': 'predicted_value'}, inplace=True)
        
        if 'Confidence' in df_copy.columns:
            df_copy.rename(columns={'Confidence': 'confidence'}, inplace=True)
        
        if 'Signal' in df_copy.columns:
            df_copy.rename(columns={'Signal': 'signal'}, inplace=True)
        
        # Store data in database
        df_copy.to_sql('prediction_results', engine, if_exists='append', index=False)
        
        logger.info(f"Stored {len(df_copy)} prediction results")
        return True
        
    except Exception as e:
        logger.error(f"Error storing prediction results: {e}")
        return False

def store_model_metadata(engine, ticker, model_type, parameters, metrics, model_path):
    """
    Store model metadata in the database
    
    Args:
        engine (sqlalchemy.engine.Engine): Database connection engine
        ticker (str): Stock ticker symbol
        model_type (str): Type of model
        parameters (dict): Model parameters
        metrics (dict): Performance metrics
        model_path (str): Path to the saved model
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Convert dictionaries to JSON strings
        parameters_json = json.dumps(parameters)
        metrics_json = json.dumps(metrics)
        
        # Create model metadata object
        model_metadata = ModelMetadata(
            ticker=ticker,
            model_type=model_type,
            training_date=datetime.now(),
            parameters=parameters_json,
            performance_metrics=metrics_json,
            model_path=model_path
        )
        
        # Add to session and commit
        session.add(model_metadata)
        session.commit()
        
        logger.info(f"Stored metadata for {ticker} {model_type} model")
        return True
        
    except Exception as e:
        logger.error(f"Error storing model metadata: {e}")
        if 'session' in locals():
            session.rollback()
        return False
    finally:
        if 'session' in locals():
            session.close()

def store_user_feedback(engine, prediction_id, feedback_type, comment=None, user_id=None):
    """
    Store user feedback on predictions
    
    Args:
        engine (sqlalchemy.engine.Engine): Database connection engine
        prediction_id (int): ID of the prediction
        feedback_type (str): Type of feedback
        comment (str): User comment
        user_id (str): User identifier
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Create feedback object
        feedback = UserFeedback(
            prediction_id=prediction_id,
            user_id=user_id,
            feedback_type=feedback_type,
            comment=comment,
            created_at=datetime.now()
        )
        
        # Add to session and commit
        session.add(feedback)
        session.commit()
        
        logger.info(f"Stored user feedback for prediction {prediction_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error storing user feedback: {e}")
        if 'session' in locals():
            session.rollback()
        return False
    finally:
        if 'session' in locals():
            session.close()

def get_latest_predictions(engine, ticker=None, limit=10):
    """
    Get latest prediction results from the database
    
    Args:
        engine (sqlalchemy.engine.Engine): Database connection engine
        ticker (str): Stock ticker symbol (optional)
        limit (int): Maximum number of results to return
        
    Returns:
        pandas.DataFrame: DataFrame with prediction results
    """
    try:
        query = "SELECT * FROM prediction_results"
        
        if ticker:
            query += f" WHERE ticker = '{ticker}'"
        
        query += " ORDER BY prediction_date DESC, date DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        # Execute query
        results = pd.read_sql(query, engine)
        
        logger.info(f"Retrieved {len(results)} prediction results")
        return results
        
    except Exception as e:
        logger.error(f"Error getting prediction results: {e}")
        return pd.DataFrame()

def get_performance_metrics(engine, ticker=None):
    """
    Get model performance metrics from the database
    
    Args:
        engine (sqlalchemy.engine.Engine): Database connection engine
        ticker (str): Stock ticker symbol (optional)
        
    Returns:
        pandas.DataFrame: DataFrame with model performance metrics
    """
    try:
        query = "SELECT * FROM model_metadata"
        
        if ticker:
            query += f" WHERE ticker = '{ticker}'"
        
        # Execute query
        results = pd.read_sql(query, engine)
        
        # Parse JSON strings
        if not results.empty:
            results['parameters'] = results['parameters'].apply(json.loads)
            results['performance_metrics'] = results['performance_metrics'].apply(json.loads)
        
        logger.info(f"Retrieved {len(results)} model metadata records")
        return results
        
    except Exception as e:
        logger.error(f"Error getting model performance metrics: {e}")
        return pd.DataFrame()

def calculate_prediction_accuracy(engine, ticker=None, days=30):
    """
    Calculate prediction accuracy based on historical predictions
    
    Args:
        engine (sqlalchemy.engine.Engine): Database connection engine
        ticker (str): Stock ticker symbol (optional)
        days (int): Number of days to include
        
    Returns:
        dict: Dictionary with accuracy metrics
    """
    try:
        # Get predictions with actual values
        query = f"""
        SELECT ticker, date, prediction_date, predicted_value, actual_value, signal, confidence
        FROM prediction_results
        WHERE actual_value IS NOT NULL
        """
        
        if ticker:
            query += f" AND ticker = '{ticker}'"
        
        if days:
            query += f" AND prediction_date >= DATE('now', '-{days} day')"
        
        # Execute query
        results = pd.read_sql(query, engine)
        
        if results.empty:
            logger.warning("No prediction results with actual values found")
            return {}
        
        # Calculate accuracy metrics
        metrics = {}
        
        # Overall accuracy (correct direction)
        results['correct_direction'] = ((results['predicted_value'] > 0) & (results['actual_value'] > 0)) | \
                                     ((results['predicted_value'] <= 0) & (results['actual_value'] <= 0))
        
        metrics['direction_accuracy'] = results['correct_direction'].mean()
        
        # Mean absolute error
        results['absolute_error'] = abs(results['predicted_value'] - results['actual_value'])
        metrics['mae'] = results['absolute_error'].mean()
        
        # Signal-specific accuracy
        if 'signal' in results.columns:
            signal_accuracy = {}
            for signal in results['signal'].unique():
                signal_df = results[results['signal'] == signal]
                if not signal_df.empty:
                    signal_accuracy[signal] = signal_df['correct_direction'].mean()
            
            metrics['signal_accuracy'] = signal_accuracy
        
        # Confidence correlation
        if 'confidence' in results.columns:
            metrics['confidence_correlation'] = results[['confidence', 'correct_direction']].corr().iloc[0, 1]
        
        logger.info(f"Calculated prediction accuracy metrics based on {len(results)} predictions")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating prediction accuracy: {e}")
        return {}

def update_actual_values(engine):
    """
    Update prediction results with actual values
    
    Args:
        engine (sqlalchemy.engine.Engine): Database connection engine
        
    Returns:
        int: Number of records updated
    """
    try:
        # Create session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Get predictions without actual values
        query = """
        SELECT pr.id, pr.ticker, pr.date, sd.close
        FROM prediction_results pr
        JOIN stock_data sd ON pr.ticker = sd.ticker AND pr.date = sd.date
        WHERE pr.actual_value IS NULL
        """
        
        # Execute query
        results = pd.read_sql(query, engine)
        
        if results.empty:
            logger.info("No prediction results to update")
            return 0
        
        # Update records
        update_count = 0
        for _, row in results.iterrows():
            session.execute(
                "UPDATE prediction_results SET actual_value = :actual, error = :error WHERE id = :id",
                {
                    'actual': float(row['close']),
                    'error': float(row['close']) - float(row['predicted_value']) if 'predicted_value' in row else None,
                    'id': int(row['id'])
                }
            )
            update_count += 1
        
        session.commit()
        
        logger.info(f"Updated {update_count} prediction results with actual values")
        return update_count
        
    except Exception as e:
        logger.error(f"Error updating actual values: {e}")
        if 'session' in locals():
            session.rollback()
        return 0
    finally:
        if 'session' in locals():
            session.close()

if __name__ == "__main__":
    # Example usage
    engine = create_database_connection()
    
    if engine is not None:
        # Create tables
        Base.metadata.create_all(engine)
        
        # Print tables
        from sqlalchemy import inspect
        inspector = inspect(engine)
        print("Database tables:")
        for table_name in inspector.get_table_names():
            print(f" - {table_name}")