"""
Evaluation module for stock market prediction models

This module evaluates model performance, compares different models,
and generates performance reports
"""
import os
import sys
import numpy as np
import pandas as pd
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_curve, auc
)
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from tabulate import tabulate

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.feature_engineering import prepare_train_test_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml/logs/evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("evaluation")

# Create necessary directories
os.makedirs("ml/logs", exist_ok=True)
os.makedirs("ml/evaluation/reports", exist_ok=True)
os.makedirs("ml/evaluation/visualizations", exist_ok=True)
os.makedirs("ml/evaluation/comparisons", exist_ok=True)

def evaluate_regression_model(y_true, y_pred, ticker=None, model_name=None):
    """
    Evaluate regression model performance
    
    Args:
        y_true (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
        ticker (str): Stock ticker symbol
        model_name (str): Name of the model
        
    Returns:
        dict: Dictionary with performance metrics
    """
    try:
        # Calculate metrics
        metrics = {}
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Calculate additional metrics
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
        metrics['median_ae'] = np.median(np.abs(y_true - y_pred))  # Median Absolute Error
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))  # Maximum Error
        
        # Calculate directional accuracy (correctly predicting up/down)
        y_true_dir = np.sign(np.diff(y_true, prepend=y_true[0]))
        y_pred_dir = np.sign(np.diff(y_pred, prepend=y_pred[0]))
        metrics['directional_accuracy'] = np.mean(y_true_dir == y_pred_dir)
        
        # Log metrics
        logger.info(f"Regression metrics for {ticker or 'unknown'} {model_name or 'model'}:")
        logger.info(f"  MSE: {metrics['mse']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        logger.info(f"  Directional Accuracy: {metrics['directional_accuracy']:.2f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating regression model: {e}")
        return {}

def evaluate_classification_model(y_true, y_pred, y_proba=None, ticker=None, model_name=None, labels=None):
    """
    Evaluate classification model performance
    
    Args:
        y_true (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
        y_proba (numpy.ndarray): Prediction probabilities
        ticker (str): Stock ticker symbol
        model_name (str): Name of the model
        labels (list): Class labels
        
    Returns:
        dict: Dictionary with performance metrics
    """
    try:
        # Calculate metrics
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # For multi-class, use 'weighted' average
        multi_class = len(np.unique(y_true)) > 2
        avg_type = 'weighted' if multi_class else 'binary'
        
        metrics['precision'] = precision_score(y_true, y_pred, average=avg_type, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=avg_type, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average=avg_type, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Detailed classification report
        if labels is None:
            labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['class_report'] = class_report
        
        # ROC AUC for binary classification with probability estimates
        if not multi_class and y_proba is not None and y_proba.shape[1] == 2:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                metrics['roc_auc'] = auc(fpr, tpr)
                metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
            except Exception as e:
                logger.warning(f"Could not calculate ROC curve: {e}")
        
        # Log metrics
        logger.info(f"Classification metrics for {ticker or 'unknown'} {model_name or 'model'}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1: {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating classification model: {e}")
        return {}

def evaluate_model_on_test_data(model_path, X_test, y_test, task='classification', ticker=None):
    """
    Evaluate a trained model on test data
    
    Args:
        model_path (str): Path to the trained model
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test targets
        task (str): Type of task ('regression' or 'classification')
        ticker (str): Stock ticker symbol
        
    Returns:
        dict: Dictionary with performance metrics
    """
    try:
        # Load model
        model = load_model(model_path)
        model_name = os.path.basename(model_path)
        
        # Make predictions
        y_pred_raw = model.predict(X_test)
        
        # Process predictions based on task
        if task == 'regression':
            y_pred = y_pred_raw.flatten()
            metrics = evaluate_regression_model(y_test, y_pred, ticker, model_name)
            
        else:  # classification
            # Extract probabilities
            if len(y_pred_raw.shape) > 1 and y_pred_raw.shape[1] > 1:
                # Multi-class
                y_proba = y_pred_raw
                y_pred = np.argmax(y_pred_raw, axis=1)
            else:
                # Binary
                y_proba = np.column_stack((1 - y_pred_raw.flatten(), y_pred_raw.flatten()))
                y_pred = (y_pred_raw > 0.5).astype(int).flatten()
            
            metrics = evaluate_classification_model(y_test, y_pred, y_proba, ticker, model_name)
        
        logger.info(f"Evaluated model {model_name} for {ticker or 'unknown'}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating model {model_path}: {e}")
        return {}

def evaluate_model_on_new_data(model_path, data_path, ticker, task='classification',
                             target_col='Target_Direction_5d', sequence_length=20):
    """
    Evaluate a trained model on new data
    
    Args:
        model_path (str): Path to the trained model
        data_path (str): Path to processed data
        ticker (str): Stock ticker symbol
        task (str): Type of task ('regression' or 'classification')
        target_col (str): Target column name
        sequence_length (int): Length of input sequences
        
    Returns:
        dict: Dictionary with performance metrics
    """
    try:
        # Load data
        data_file = os.path.join(data_path, f"{ticker}_processed.csv")
        if not os.path.exists(data_file):
            logger.error(f"Data file not found: {data_file}")
            return {}
        
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        
        # Prepare data
        _, X_test, _, y_test = prepare_train_test_data(
            df, target_col, test_size=0.2, sequence_length=sequence_length
        )
        
        if X_test is None or y_test is None:
            logger.error(f"Failed to prepare test data for {ticker}")
            return {}
        
        # Evaluate model
        metrics = evaluate_model_on_test_data(model_path, X_test, y_test, task, ticker)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating model on new data: {e}")
        return {}

def visualize_regression_predictions(y_true, y_pred, ticker=None, model_name=None, output_dir="ml/evaluation/visualizations/"):
    """
    Visualize regression model predictions
    
    Args:
        y_true (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
        ticker (str): Stock ticker symbol
        model_name (str): Name of the model
        output_dir (str): Directory to save visualizations
        
    Returns:
        str: Path to the saved visualization
    """
    try:
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot true vs predicted values
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('True vs Predicted Values')
        
        # Add correlation line
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        plt.annotate(f'Correlation: {corr:.4f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
        
        # Plot prediction errors
        plt.subplot(1, 2, 2)
        errors = y_true - y_pred
        plt.hist(errors, bins=30, alpha=0.7)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors')
        
        # Add error statistics
        plt.annotate(f'Mean Error: {np.mean(errors):.4f}\nStd Dev: {np.std(errors):.4f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{ticker or 'unknown'}_{model_name or 'model'}_regression.png")
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved regression visualization to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error visualizing regression predictions: {e}")
        return None

def visualize_classification_results(y_true, y_pred, y_proba=None, ticker=None, model_name=None, 
                                  output_dir="ml/evaluation/visualizations/"):
    """
    Visualize classification model results
    
    Args:
        y_true (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
        y_proba (numpy.ndarray): Prediction probabilities
        ticker (str): Stock ticker symbol
        model_name (str): Name of the model
        output_dir (str): Directory to save visualizations
        
    Returns:
        str: Path to the saved visualization
    """
    try:
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot confusion matrix
        plt.subplot(2, 2, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Plot class distribution
        plt.subplot(2, 2, 2)
        class_counts = np.bincount(y_true)
        class_names = [f'Class {i}' for i in range(len(class_counts))]
        plt.bar(class_names, class_counts)
        plt.xlabel('Class')
        plt.ylabel('Frequency')
        plt.title('Class Distribution')
        
        # Plot prediction accuracy by class
        plt.subplot(2, 2, 3)
        accuracies = []
        for cls in range(len(class_counts)):
            mask = (y_true == cls)
            if np.sum(mask) > 0:
                acc = np.mean(y_pred[mask] == cls)
                accuracies.append(acc)
            else:
                accuracies.append(0)
        
        plt.bar(class_names, accuracies)
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.title('Prediction Accuracy by Class')
        
        # Plot ROC curve for binary classification
        if y_proba is not None and len(np.unique(y_true)) == 2:
            plt.subplot(2, 2, 4)
            try:
                if y_proba.shape[1] == 2:
                    proba = y_proba[:, 1]
                else:
                    proba = y_proba.flatten()
                
                fpr, tpr, _ = roc_curve(y_true, proba)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc='lower right')
            except Exception as e:
                logger.warning(f"Could not plot ROC curve: {e}")
                plt.text(0.5, 0.5, "ROC curve unavailable", ha='center', va='center')
        else:
            plt.subplot(2, 2, 4)
            plt.text(0.5, 0.5, "ROC curve only available for binary classification", ha='center', va='center')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{ticker or 'unknown'}_{model_name or 'model'}_classification.png")
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved classification visualization to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error visualizing classification results: {e}")
        return None

def generate_performance_report(metrics, ticker=None, model_name=None, task='classification',
                             output_dir="ml/evaluation/reports/"):
    """
    Generate a performance report for a model
    
    Args:
        metrics (dict): Dictionary with performance metrics
        ticker (str): Stock ticker symbol
        model_name (str): Name of the model
        task (str): Type of task ('regression' or 'classification')
        output_dir (str): Directory to save report
        
    Returns:
        str: Path to the saved report
    """
    try:
        # Create report content
        report = []
        
        # Report header
        report.append(f"# Performance Report")
        report.append(f"- Stock: {ticker or 'Unknown'}")
        report.append(f"- Model: {model_name or 'Unknown'}")
        report.append(f"- Task: {task}")
        report.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Main metrics
        report.append("## Key Performance Metrics")
        
        if task == 'regression':
            # Regression metrics
            metrics_table = [
                ["MSE", f"{metrics.get('mse', 'N/A'):.4f}"],
                ["RMSE", f"{metrics.get('rmse', 'N/A'):.4f}"],
                ["MAE", f"{metrics.get('mae', 'N/A'):.4f}"],
                ["R²", f"{metrics.get('r2', 'N/A'):.4f}"],
                ["MAPE", f"{metrics.get('mape', 'N/A'):.2f}%"],
                ["Median AE", f"{metrics.get('median_ae', 'N/A'):.4f}"],
                ["Max Error", f"{metrics.get('max_error', 'N/A'):.4f}"],
                ["Directional Accuracy", f"{metrics.get('directional_accuracy', 'N/A'):.2f}"]
            ]
            
            report.append(f"```")
            report.append(tabulate(metrics_table, headers=["Metric", "Value"]))
            report.append(f"```")
            
        else:
            # Classification metrics
            metrics_table = [
                ["Accuracy", f"{metrics.get('accuracy', 'N/A'):.4f}"],
                ["Precision", f"{metrics.get('precision', 'N/A'):.4f}"],
                ["Recall", f"{metrics.get('recall', 'N/A'):.4f}"],
                ["F1 Score", f"{metrics.get('f1', 'N/A'):.4f}"]
            ]
            
            if 'roc_auc' in metrics:
                metrics_table.append(["ROC AUC", f"{metrics.get('roc_auc', 'N/A'):.4f}"])
            
            report.append(f"```")
            report.append(tabulate(metrics_table, headers=["Metric", "Value"]))
            report.append(f"```")
            
            # Confusion matrix
            if 'confusion_matrix' in metrics:
                report.append("")
                report.append("## Confusion Matrix")
                report.append(f"```")
                cm = np.array(metrics['confusion_matrix'])
                
                # Generate labels
                labels = [f"Class {i}" for i in range(cm.shape[0])]
                
                # Create table with row and column headers
                cm_table = tabulate(cm, headers=labels, showindex=labels)
                report.append(cm_table)
                report.append(f"```")
            
            # Classification report
            if 'class_report' in metrics:
                report.append("")
                report.append("## Detailed Classification Report")
                report.append(f"```")
                
                # Extract metrics from classification report
                cr = metrics['class_report']
                classes = sorted([k for k in cr.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']])
                
                # Build table rows
                rows = []
                for cls in classes:
                    cls_metrics = cr[cls]
                    rows.append([
                        cls,
                        f"{cls_metrics['precision']:.4f}",
                        f"{cls_metrics['recall']:.4f}",
                        f"{cls_metrics['f1-score']:.4f}",
                        f"{cls_metrics['support']}"
                    ])
                
                # Add averages
                for avg_type in ['macro avg', 'weighted avg']:
                    if avg_type in cr:
                        rows.append([
                            avg_type,
                            f"{cr[avg_type]['precision']:.4f}",
                            f"{cr[avg_type]['recall']:.4f}",
                            f"{cr[avg_type]['f1-score']:.4f}",
                            f"{cr[avg_type]['support']}"
                        ])
                
                # Generate table
                table = tabulate(rows, headers=["Class", "Precision", "Recall", "F1", "Support"])
                report.append(table)
                report.append(f"```")
        
        # Save report
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"{ticker or 'unknown'}_{model_name or 'model'}_{timestamp}.md")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Saved performance report to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        return None

def compare_models(models_metrics, tickers=None, task='classification', output_dir="ml/evaluation/comparisons/"):
    """
    Compare performance of multiple models
    
    Args:
        models_metrics (dict): Dictionary with model metrics
        tickers (list): List of stock ticker symbols
        task (str): Type of task ('regression' or 'classification')
        output_dir (str): Directory to save comparison report
        
    Returns:
        str: Path to the saved comparison report
    """
    try:
        # Create report content
        report = []
        
        # Report header
        report.append(f"# Model Comparison Report")
        report.append(f"- Stocks: {', '.join(tickers) if tickers else 'All'}")
        report.append(f"- Task: {task}")
        report.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Comparison table
        report.append("## Performance Comparison")
        
        if task == 'regression':
            # Regression metrics to compare
            headers = ["Model", "Stock", "RMSE", "MAE", "R²", "Dir. Acc."]
            rows = []
            
            for model_name, stocks_metrics in models_metrics.items():
                for ticker, metrics in stocks_metrics.items():
                    if tickers is None or ticker in tickers:
                        rows.append([
                            model_name,
                            ticker,
                            f"{metrics.get('rmse', 'N/A'):.4f}",
                            f"{metrics.get('mae', 'N/A'):.4f}",
                            f"{metrics.get('r2', 'N/A'):.4f}",
                            f"{metrics.get('directional_accuracy', 'N/A'):.2f}"
                        ])
        else:
            # Classification metrics to compare
            headers = ["Model", "Stock", "Accuracy", "Precision", "Recall", "F1"]
            rows = []
            
            for model_name, stocks_metrics in models_metrics.items():
                for ticker, metrics in stocks_metrics.items():
                    if tickers is None or ticker in tickers:
                        rows.append([
                            model_name,
                            ticker,
                            f"{metrics.get('accuracy', 'N/A'):.4f}",
                            f"{metrics.get('precision', 'N/A'):.4f}",
                            f"{metrics.get('recall', 'N/A'):.4f}",
                            f"{metrics.get('f1', 'N/A'):.4f}"
                        ])
        
        # Sort rows by stock and then by model
        rows.sort(key=lambda x: (x[1], x[0]))
        
        # Generate table
        report.append(f"```")
        report.append(tabulate(rows, headers=headers))
        report.append(f"```")
        
        # Best model analysis
        report.append("")
        report.append("## Best Model Analysis")
        
        # Group by stock and find best model for each
        stocks = {}
        for row in rows:
            model_name, ticker = row[0], row[1]
            if ticker not in stocks:
                stocks[ticker] = []
            stocks[ticker].append(row)
        
        # For each stock, find the best model
        best_models = []
        for ticker, stock_rows in stocks.items():
            # For regression, sort by RMSE (lower is better)
            # For classification, sort by F1 score (higher is better)
            if task == 'regression':
                sort_idx = 2  # RMSE index
                is_lower_better = True
            else:
                sort_idx = 5  # F1 index
                is_lower_better = False
            
            # Convert metric to float for sorting
            for row in stock_rows:
                try:
                    row[sort_idx] = float(row[sort_idx])
                except ValueError:
                    row[sort_idx] = float('inf') if is_lower_better else float('-inf')
            
            # Sort rows
            if is_lower_better:
                stock_rows.sort(key=lambda x: x[sort_idx])
            else:
                stock_rows.sort(key=lambda x: x[sort_idx], reverse=True)
            
            # Add best model to list
            best_models.append([
                ticker,
                stock_rows[0][0],  # Best model name
                stock_rows[0][2],  # RMSE or Accuracy
                stock_rows[0][3],  # MAE or Precision
                stock_rows[0][4],  # R² or Recall
                stock_rows[0][5]   # Dir. Acc. or F1
            ])
        
        # Generate best models table
        if task == 'regression':
            best_headers = ["Stock", "Best Model", "RMSE", "MAE", "R²", "Dir. Acc."]
        else:
            best_headers = ["Stock", "Best Model", "Accuracy", "Precision", "Recall", "F1"]
        
        report.append(f"```")
        report.append(tabulate(best_models, headers=best_headers))
        report.append(f"```")
        
        # Visualize comparison
        report.append("")
        report.append("## Performance Visualization")
        report.append("")
        report.append("Please refer to the accompanying visualization file for a graphical comparison.")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Extract model names and stocks
        model_names = sorted(list(models_metrics.keys()))
        all_tickers = sorted(list(set([ticker for model_metrics in models_metrics.values() for ticker in model_metrics.keys()])))
        
        if tickers:
            plot_tickers = [t for t in all_tickers if t in tickers]
        else:
            plot_tickers = all_tickers
        
        # Plot performance metric
        if task == 'regression':
            metric_name = 'rmse'
            metric_label = 'RMSE (lower is better)'
            lower_is_better = True
        else:
            metric_name = 'f1'
            metric_label = 'F1 Score (higher is better)'
            lower_is_better = False
        
        # Set up bar positions
        n_models = len(model_names)
        bar_width = 0.8 / n_models
        positions = np.arange(len(plot_tickers))
        
        # Plot bars for each model
        for i, model_name in enumerate(model_names):
            # Get metrics for this model
            values = []
            for ticker in plot_tickers:
                if ticker in models_metrics.get(model_name, {}):
                    metric_value = models_metrics[model_name][ticker].get(metric_name, None)
                    values.append(float(metric_value) if metric_value is not None else float('nan'))
                else:
                    values.append(float('nan'))
            
            # Plot bars
            plt.bar(
                positions + i * bar_width - 0.4 + bar_width/2, 
                values, 
                width=bar_width, 
                label=model_name,
                alpha=0.7
            )
        
        # Add labels and legend
        plt.xlabel('Stock')
        plt.ylabel(metric_label)
        plt.title(f'Model Performance Comparison - {metric_label}')
        plt.xticks(positions, plot_tickers, rotation=45)
        plt.legend()
        
        # Adjust y-limit for better visualization
        if not lower_is_better:
            plt.ylim(0, 1.1)
        
        plt.tight_layout()
        
        # Save visualization
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = os.path.join(output_dir, f"model_comparison_{timestamp}.png")
        plt.savefig(viz_path)
        plt.close()
        
        # Save report
        output_path = os.path.join(output_dir, f"model_comparison_{timestamp}.md")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Saved model comparison report to {output_path}")
        logger.info(f"Saved model comparison visualization to {viz_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        return None

def evaluate_all_models(models_dir="ml/models/final/", data_dir="ml/data/processed/", 
                       output_dir="ml/evaluation/reports/", tickers=None):
    """
    Evaluate all trained models
    
    Args:
        models_dir (str): Directory with trained models
        data_dir (str): Directory with processed data
        output_dir (str): Directory to save reports
        tickers (list): List of stock ticker symbols to evaluate
        
    Returns:
        dict: Dictionary with evaluation results
    """
    try:
        # Get all model files
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
        
        if not model_files:
            logger.error(f"No model files found in {models_dir}")
            return {}
        
        # Parse model information
        models_info = []
        for model_file in model_files:
            # Extract information from filename
            # Expected format: TICKER_MODELTYPE_TASK_model.h5
            try:
                parts = model_file.split('_')
                ticker = parts[0]
                model_type = parts[1]
                task = parts[2]
                
                # Skip if ticker not in list
                if tickers and ticker not in tickers:
                    continue
                
                models_info.append({
                    'file': model_file,
                    'path': os.path.join(models_dir, model_file),
                    'ticker': ticker,
                    'model_type': model_type,
                    'task': task
                })
            except IndexError:
                logger.warning(f"Could not parse model filename: {model_file}")
                continue
        
        # Evaluate each model
        results = {}
        for model_info in models_info:
            logger.info(f"Evaluating model: {model_info['file']}")
            
            # Evaluate on test data
            metrics = evaluate_model_on_new_data(
                model_info['path'],
                data_dir,
                model_info['ticker'],
                model_info['task']
            )
            
            if not metrics:
                logger.warning(f"Could not evaluate model: {model_info['file']}")
                continue
            
            # Generate report
            generate_performance_report(
                metrics,
                model_info['ticker'],
                model_info['model_type'],
                model_info['task'],
                output_dir
            )
            
            # Store results
            model_key = f"{model_info['model_type']}_{model_info['task']}"
            if model_key not in results:
                results[model_key] = {}
            
            results[model_key][model_info['ticker']] = metrics
        
        # Compare models
        if results:
            compare_models(results, tickers)
        
        logger.info(f"Completed evaluation of {len(models_info)} models")
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating models: {e}")
        return {}

def backtest_trading_strategy(predictions, actual_prices, initial_capital=10000, commission=0.001):
    """
    Backtest a simple trading strategy based on model predictions
    
    Args:
        predictions (list): List of trading signals (1 for buy, -1 for sell, 0 for hold)
        actual_prices (list): List of actual prices
        initial_capital (float): Initial capital
        commission (float): Commission rate per trade
        
    Returns:
        dict: Dictionary with backtest results
    """
    try:
        if len(predictions) != len(actual_prices):
            logger.error(f"Length mismatch: predictions ({len(predictions)}) vs prices ({len(actual_prices)})")
            return {}
        
        # Initialize variables
        capital = initial_capital
        holdings = 0
        trades = []
        portfolio_value = []
        
        # Run backtest
        for i in range(len(predictions)):
            # Current price
            price = actual_prices[i]
            signal = predictions[i]
            
            # Calculate portfolio value
            current_value = capital + holdings * price
            portfolio_value.append(current_value)
            
            # Execute trade based on signal
            if signal == 1 and holdings == 0:  # Buy signal
                # Calculate max shares to buy
                max_shares = capital / (price * (1 + commission))
                holdings = max_shares
                trade_cost = holdings * price * (1 + commission)
                capital -= trade_cost
                
                trades.append({
                    'day': i,
                    'type': 'buy',
                    'price': price,
                    'shares': holdings,
                    'cost': trade_cost,
                    'capital_after': capital
                })
                
            elif signal == -1 and holdings > 0:  # Sell signal
                # Sell all holdings
                trade_value = holdings * price * (1 - commission)
                capital += trade_value
                
                trades.append({
                    'day': i,
                    'type': 'sell',
                    'price': price,
                    'shares': holdings,
                    'value': trade_value,
                    'capital_after': capital
                })
                
                holdings = 0
        
        # Calculate final portfolio value
        final_value = capital
        if holdings > 0:
            final_value += holdings * actual_prices[-1]
        
        # Calculate performance metrics
        total_return = (final_value / initial_capital - 1) * 100
        buy_hold_return = (actual_prices[-1] / actual_prices[0] - 1) * 100
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(portfolio_value)):
            daily_return = (portfolio_value[i] / portfolio_value[i-1]) - 1
            daily_returns.append(daily_return)
        
        # Calculate metrics
        annualized_return = (1 + total_return/100) ** (252/len(predictions)) - 1
        volatility = np.std(daily_returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Count trades
        num_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['type'] == 'sell' and t['value'] > t['price'] * t['shares'])
        
        # Calculate win rate
        win_rate = winning_trades / (num_trades / 2) if num_trades > 0 else 0
        
        # Results
        results = {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'outperformance': total_return - buy_hold_return,
            'annualized_return': annualized_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'portfolio_value': portfolio_value,
            'trades': trades
        }
        
        logger.info(f"Backtest results: Return={total_return:.2f}%, B&H={buy_hold_return:.2f}%, Trades={num_trades}, Win={win_rate:.2f}")
        return results
        
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        return {}

def visualize_backtest_results(results, ticker=None, model_name=None, output_dir="ml/evaluation/visualizations/"):
    """
    Visualize backtest results
    
    Args:
        results (dict): Backtest results
        ticker (str): Stock ticker symbol
        model_name (str): Name of the model
        output_dir (str): Directory to save visualization
        
    Returns:
        str: Path to the saved visualization
    """
    try:
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot portfolio value
        plt.subplot(2, 2, 1)
        plt.plot(results['portfolio_value'], label='Portfolio Value')
        
        # Mark trades on the chart
        buy_days = [t['day'] for t in results['trades'] if t['type'] == 'buy']
        buy_values = [results['portfolio_value'][day] for day in buy_days]
        
        sell_days = [t['day'] for t in results['trades'] if t['type'] == 'sell']
        sell_values = [results['portfolio_value'][day] for day in sell_days]
        
        plt.scatter(buy_days, buy_values, marker='^', color='green', label='Buy')
        plt.scatter(sell_days, sell_values, marker='v', color='red', label='Sell')
        
        plt.xlabel('Day')
        plt.ylabel('Portfolio Value ($)')
        plt.title('Portfolio Value Over Time')
        plt.legend()
        plt.grid(True)
        
        # Plot returns comparison
        plt.subplot(2, 2, 2)
        returns = [
            results['total_return'],
            results['buy_hold_return']
        ]
        labels = ['Trading Strategy', 'Buy & Hold']
        colors = ['blue', 'gray']
        
        plt.bar(labels, returns, color=colors)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.ylabel('Return (%)')
        plt.title('Performance Comparison')
        
        for i, v in enumerate(returns):
            plt.text(i, v + (1 if v >= 0 else -3), f"{v:.1f}%", ha='center')
        
        # Plot trade outcomes
        plt.subplot(2, 2, 3)
        
        # Calculate profit/loss for each completed trade
        profits = []
        for i in range(0, len(results['trades']), 2):
            if i + 1 < len(results['trades']):
                buy = results['trades'][i]
                sell = results['trades'][i + 1]
                profit = sell['value'] - buy['cost']
                profits.append(profit)
        
        if profits:
            plt.hist(profits, bins=10, color='green' if np.mean(profits) >= 0 else 'red', alpha=0.7)
            plt.axvline(x=0, color='k', linestyle='--')
            plt.axvline(x=np.mean(profits), color='r', linestyle='-', label=f'Mean: ${np.mean(profits):.2f}')
            
            plt.xlabel('Profit/Loss per Trade ($)')
            plt.ylabel('Frequency')
            plt.title(f'Trade Outcomes (Win Rate: {results["win_rate"]*100:.1f}%)')
            plt.legend()
        else:
            plt.text(0.5, 0.5, "No completed trades", ha='center', va='center', transform=plt.gca().transAxes)
        
        # Plot key metrics
        plt.subplot(2, 2, 4)
        metrics = [
            results['annualized_return'],
            results['volatility'],
            results['sharpe_ratio'] * 100
        ]
        metric_labels = ['Ann. Return (%)', 'Volatility (%)', 'Sharpe Ratio (×100)']
        
        plt.bar(metric_labels, metrics, color='blue', alpha=0.7)
        plt.ylabel('Value')
        plt.title('Risk-Return Metrics')
        
        for i, v in enumerate(metrics):
            plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
        
        plt.tight_layout()
        
        # Save visualization
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{ticker or 'unknown'}_{model_name or 'model'}_backtest.png")
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved backtest visualization to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error visualizing backtest results: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    from tensorflow.keras.models import load_model
    import numpy as np
    
    # Create test data
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 1])
    y_proba = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
        [0.9, 0.1],
        [0.6, 0.4],
        [0.2, 0.8],
        [0.7, 0.3],
        [0.1, 0.9],
        [0.8, 0.2],
        [0.4, 0.6],
        [0.3, 0.7]
    ])
    
    # Evaluate classification model
    metrics = evaluate_classification_model(y_true, y_pred, y_proba, "AAPL", "lstm_model")
    
    # Visualize results
    visualize_classification_results(y_true, y_pred, y_proba, "AAPL", "lstm_model")
    
    # Generate report
    generate_performance_report(metrics, "AAPL", "lstm_model")
    
    # Backtest example
    prices = np.linspace(100, 120, 20) + np.random.randn(20) * 2
    signals = np.array([0, 1, 0, 0, -1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1, 0, 0, -1, 0, 0])
    
    backtest_results = backtest_trading_strategy(signals, prices)
    
    if backtest_results:
        visualize_backtest_results(backtest_results, "AAPL", "lstm_model")