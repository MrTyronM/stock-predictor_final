"""
API Server for AI Stock Market Prediction Tool

This module provides a web API for the frontend to interact with the backend
"""
import os
import sys
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, abort
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from ml.data_acquisition import download_stock_data, get_sp500_tickers, get_available_tickers
from ml.preprocessing.feature_engineering import add_technical_indicators
from ml.prediction.prediction import predict_for_stock, predict_for_multiple_stocks, get_latest_signals
from ml.evaluation.evaluation import backtest_trading_strategy
from config.config import config, create_backup, restore_from_backup
from ml.db.db_handler import create_database_connection, get_latest_predictions, get_performance_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml/logs/api_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api_server")

# Create necessary directories
os.makedirs("ml/logs", exist_ok=True)

# Initialize Flask app
app = Flask(__name__, static_folder='frontend')

# Create database connection
db_type = config.get("database", "type", "sqlite")
db_host = config.get("database", "host", "")
db_user = config.get("database", "user", "")
db_password = config.get("database", "password", "")
db_name = config.get("database", "database", "")

db_engine = create_database_connection(
    db_type=db_type,
    user=db_user,
    password=db_password,
    host=db_host,
    database=db_name
)

# Serve frontend
@app.route('/')
def index():
    """Serve the index.html file"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# API endpoints
@app.route('/api/tickers', methods=['GET'])
def get_tickers():
    """Get list of available tickers"""
    try:
        source = request.args.get('source', 'available')
        
        if source == 'sp500':
            tickers = get_sp500_tickers()
        else:
            tickers = get_available_tickers()
        
        return jsonify({
            'success': True,
            'tickers': tickers,
            'count': len(tickers)
        })
    except Exception as e:
        logger.error(f"Error getting tickers: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stock/<ticker>', methods=['GET'])
def get_stock_data(ticker):
    """Get stock data for a specific ticker"""
    try:
        days = int(request.args.get('days', 100))
        include_indicators = request.args.get('indicators', 'false').lower() == 'true'
        
        # Download data
        stock_data = download_stock_data(ticker, start_date=None, end_date=None)
        
        if stock_data is None:
            return jsonify({
                'success': False,
                'error': f"No data found for {ticker}"
            }), 404
        
        # Add technical indicators if requested
        if include_indicators:
            stock_data = add_technical_indicators(stock_data)
        
        # Limit to requested number of days
        if days > 0:
            stock_data = stock_data.tail(days)
        
        # Convert to dict for JSON serialization
        stock_dict = {
            'dates': stock_data.index.strftime('%Y-%m-%d').tolist(),
            'open': stock_data['Open'].tolist(),
            'high': stock_data['High'].tolist(),
            'low': stock_data['Low'].tolist(),
            'close': stock_data['Close'].tolist(),
            'volume': stock_data['Volume'].tolist()
        }
        
        # Add indicators if included
        if include_indicators:
            for col in stock_data.columns:
                if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']:
                    stock_dict[col] = stock_data[col].tolist()
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'data': stock_dict
        })
    except Exception as e:
        logger.error(f"Error getting stock data for {ticker}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict/<ticker>', methods=['GET'])
def predict_stock(ticker):
    """Get prediction for a specific ticker"""
    try:
        model_type = request.args.get('model_type', config.get("model", "model_type"))
        task = request.args.get('task', 'classification')
        days = int(request.args.get('days', 100))
        threshold = float(request.args.get('threshold', config.get("model", "confidence_threshold")))
        visualize = request.args.get('visualize', 'true').lower() == 'true'
        
        # Make prediction
        signals = predict_for_stock(
            ticker, 
            model_type=model_type,
            task=task,
            days=days,
            threshold=threshold,
            visualize=visualize
        )
        
        if signals is None:
            return jsonify({
                'success': False,
                'error': f"Failed to generate prediction for {ticker}"
            }), 404
        
        # Convert DataFrame to dict
        signals_dict = signals.to_dict(orient='records')
        
        # Get visualization path if available
        viz_path = f"ml/prediction/visualizations/{ticker}_signals.png"
        has_visualization = os.path.exists(viz_path)
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'predictions': signals_dict,
            'has_visualization': has_visualization,
            'visualization_path': viz_path if has_visualization else None
        })
    except Exception as e:
        logger.error(f"Error predicting for {ticker}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Get latest predictions for all stocks"""
    try:
        limit = int(request.args.get('limit', 10))
        filter_signal = request.args.get('signal', None)
        filter_confidence = float(request.args.get('confidence', 0))
        
        # Get predictions from database
        if db_engine is not None:
            predictions = get_latest_predictions(db_engine, limit=limit)
            predictions_dict = predictions.to_dict(orient='records')
        else:
            # Fallback to file-based predictions
            signals = get_latest_signals()
            predictions_dict = signals.to_dict(orient='records') if signals is not None else []
        
        # Apply filters
        if filter_signal:
            predictions_dict = [p for p in predictions_dict if p.get('signal', '').lower() == filter_signal.lower()]
        
        if filter_confidence > 0:
            predictions_dict = [p for p in predictions_dict if p.get('confidence', 0) >= filter_confidence]
        
        return jsonify({
            'success': True,
            'predictions': predictions_dict,
            'count': len(predictions_dict)
        })
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/backtest', methods=['POST'])
def backtest():
    """Run backtest for a trading strategy"""
    try:
        data = request.json
        ticker = data.get('ticker')
        signals = data.get('signals', [])
        days = int(data.get('days', 100))
        initial_capital = float(data.get('initial_capital', 10000))
        commission = float(data.get('commission', 0.001))
        
        if not ticker:
            return jsonify({
                'success': False,
                'error': "Ticker is required"
            }), 400
        
        # Get historical prices
        stock_data = download_stock_data(ticker, start_date=None, end_date=None)
        
        if stock_data is None:
            return jsonify({
                'success': False,
                'error': f"No data found for {ticker}"
            }), 404
        
        # Limit to requested number of days
        if days > 0:
            stock_data = stock_data.tail(days)
        
        # Use provided signals or generate random ones for testing
        if not signals:
            signals = np.random.choice([-1, 0, 1], size=len(stock_data))
        
        # Run backtest
        backtest_results = backtest_trading_strategy(
            signals,
            stock_data['Close'].values,
            initial_capital,
            commission
        )
        
        if not backtest_results:
            return jsonify({
                'success': False,
                'error': f"Failed to run backtest for {ticker}"
            }), 500
        
        # Return results
        return jsonify({
            'success': True,
            'ticker': ticker,
            'backtest_results': {
                'initial_capital': backtest_results['initial_capital'],
                'final_value': backtest_results['final_value'],
                'total_return': backtest_results['total_return'],
                'buy_hold_return': backtest_results['buy_hold_return'],
                'outperformance': backtest_results['outperformance'],
                'annualized_return': backtest_results['annualized_return'],
                'volatility': backtest_results['volatility'],
                'sharpe_ratio': backtest_results['sharpe_ratio'],
                'num_trades': backtest_results['num_trades'],
                'win_rate': backtest_results['win_rate'],
                'portfolio_value': backtest_results['portfolio_value'],
                # Trades can be too large for JSON response
                'trades_count': len(backtest_results['trades'])
            }
        })
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/config', methods=['GET', 'PUT'])
def handle_config():
    """Get or update configuration"""
    try:
        if request.method == 'GET':
            # Get configuration
            section = request.args.get('section')
            key = request.args.get('key')
            
            if section and key:
                value = config.get(section, key)
                return jsonify({
                    'success': True,
                    section: {key: value}
                })
            elif section:
                values = config.get(section)
                return jsonify({
                    'success': True,
                    section: values
                })
            else:
                return jsonify({
                    'success': True,
                    'config': config.get_all()
                })
        else:  # PUT
            # Update configuration
            data = request.json
            
            if not data:
                return jsonify({
                    'success': False,
                    'error': "No data provided"
                }), 400
            
            section = request.args.get('section')
            
            if section:
                # Update specific section
                if section not in data:
                    return jsonify({
                        'success': False,
                        'error': f"Section {section} not found in request data"
                    }), 400
                
                config.update_section(section, data[section])
            else:
                # Update all sections
                for section, values in data.items():
                    config.update_section(section, values)
            
            # Save configuration
            config.save_config()
            
            return jsonify({
                'success': True,
                'message': "Configuration updated successfully"
            })
    except Exception as e:
        logger.error(f"Error handling configuration: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/backup', methods=['POST'])
def backup():
    """Create backup of configuration and data"""
    try:
        data = request.json or {}
        include_data = data.get('include_data', True)
        include_models = data.get('include_models', True)
        
        # Create backup
        backup_file = create_backup(config, include_data, include_models)
        
        if not backup_file:
            return jsonify({
                'success': False,
                'error': "Failed to create backup"
            }), 500
        
        return jsonify({
            'success': True,
            'backup_file': backup_file,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/restore', methods=['POST'])
def restore():
    """Restore configuration and data from backup"""
    try:
        data = request.json
        backup_file = data.get('backup_file')
        restore_data = data.get('restore_data', True)
        restore_models = data.get('restore_models', True)
        
        if not backup_file:
            return jsonify({
                'success': False,
                'error': "Backup file path is required"
            }), 400
        
        # Restore from backup
        success = restore_from_backup(backup_file, config, restore_data, restore_models)
        
        if not success:
            return jsonify({
                'success': False,
                'error': f"Failed to restore from backup {backup_file}"
            }), 500
        
        return jsonify({
            'success': True,
            'message': f"Successfully restored from backup {backup_file}"
        })
    except Exception as e:
        logger.error(f"Error restoring from backup: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/visualizations/<path:filename>')
def get_visualization(filename):
    """Get visualization image"""
    try:
        # Check if file exists
        file_path = f"ml/prediction/visualizations/{filename}"
        if not os.path.exists(file_path):
            abort(404)
        
        return send_from_directory('ml/prediction/visualizations', filename)
    except Exception as e:
        logger.error(f"Error getting visualization {filename}: {e}")
        abort(500)

@app.route('/api/training', methods=['POST'])
def start_training():
    """Start model training"""
    try:
        data = request.json
        tickers = data.get('tickers', [])
        model_type = data.get('model_type', config.get("model", "model_type"))
        model_complexity = data.get('model_complexity', config.get("model", "model_complexity"))
        epochs = int(data.get('epochs', config.get("model", "epochs")))
        
        # This should be run asynchronously in a real application
        # For simplicity, we'll just return a success response
        
        return jsonify({
            'success': True,
            'message': f"Started training for {len(tickers)} tickers (not implemented)",
            'job_id': datetime.now().strftime("%Y%m%d%H%M%S")
        })
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get model performance metrics"""
    try:
        ticker = request.args.get('ticker')
        
        # Get metrics from database
        if db_engine is not None:
            metrics = get_performance_metrics(db_engine, ticker)
            metrics_dict = metrics.to_dict(orient='records') if not metrics.empty else []
        else:
            # Fallback to placeholder metrics
            metrics_dict = [{
                'ticker': ticker or 'AAPL',
                'model_type': 'hybrid',
                'training_date': datetime.now().strftime("%Y-%m-%d"),
                'performance_metrics': {
                    'accuracy': 0.725,
                    'precision': 0.68,
                    'recall': 0.71,
                    'f1': 0.695
                }
            }]
        
        return jsonify({
            'success': True,
            'metrics': metrics_dict
        })
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    try:
        # Get counts
        available_tickers = len(get_available_tickers())
        prediction_files = len([f for f in os.listdir("ml/prediction/signals") if f.endswith("_latest.json")])
        model_files = len([f for f in os.listdir("ml/models/final") if f.endswith(".h5")])
        
        # Get storage usage
        data_size = get_directory_size("ml/data")
        models_size = get_directory_size("ml/models")
        
        return jsonify({
            'success': True,
            'status': {
                'available_tickers': available_tickers,
                'trained_models': model_files,
                'prediction_count': prediction_files,
                'data_size_mb': round(data_size / (1024 * 1024), 2),
                'models_size_mb': round(models_size / (1024 * 1024), 2),
                'version': config.get("system", "version"),
                'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        })
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def get_directory_size(directory):
    """Calculate directory size in bytes"""
    total_size = 0
    if os.path.exists(directory):
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)
    return total_size

if __name__ == "__main__":
    # Get port from configuration
    port = int(os.environ.get("PORT", 5000))
    
    # Run app
    app.run(host="0.0.0.0", port=port, debug=True)