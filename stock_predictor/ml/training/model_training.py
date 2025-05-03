"""
Model training module for stock market prediction - Improved Version

This module implements LSTM and other deep learning models for stock prediction
"""
import os
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Bidirectional, GRU, Conv1D, MaxPooling1D, Flatten, Attention, MultiHeadAttention, Concatenate, Add
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import json
from tqdm import tqdm
import sys
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.feature_engineering import prepare_train_test_data, balance_classification_dataset, augment_financial_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml/logs/model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_training")

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Create necessary directories
os.makedirs("ml/logs", exist_ok=True)
os.makedirs("ml/models", exist_ok=True)
os.makedirs("ml/models/checkpoints", exist_ok=True)
os.makedirs("ml/models/final", exist_ok=True)
os.makedirs("ml/evaluation", exist_ok=True)

def build_lstm_model(input_shape, output_dim=1, model_complexity='medium', dropout_rate=0.3, 
                   learning_rate=0.001, loss=None, metrics=None, task='regression'):
    """
    Build an LSTM model for stock prediction with improved architecture
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, features)
        output_dim (int): Number of output dimensions
        model_complexity (str): Model complexity ('simple', 'medium', 'complex')
        dropout_rate (float): Dropout rate for regularization
        learning_rate (float): Learning rate for Adam optimizer
        loss (str): Loss function
        metrics (list): List of metrics
        task (str): Type of model ('regression' or 'classification')
        
    Returns:
        tensorflow.keras.models.Sequential: Compiled LSTM model
    """
    try:
        # Set appropriate metrics based on task
        if metrics is None:
            if task == 'regression':
                metrics = ['mae', 'mse']
            else:
                metrics = ['accuracy']
        
        # Set appropriate loss function based on task
        if loss is None:
            if task == 'regression':
                loss = 'mse'
            elif output_dim == 1:  # Binary classification
                loss = 'binary_crossentropy'
            else:  # Multi-class classification
                loss = 'categorical_crossentropy'
            
        model = Sequential()
        
        # First LSTM layer with input shape
        if model_complexity == 'simple':
            # Simple model - better for limited data
            model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
            model.add(Dropout(dropout_rate))
            model.add(BatchNormalization())
            model.add(Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-4, l2=1e-3)))
            model.add(Dropout(dropout_rate))
            
        elif model_complexity == 'medium':
            # Medium complexity
            model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
            model.add(Dropout(dropout_rate))
            model.add(BatchNormalization())
            
            model.add(LSTM(64, return_sequences=False))
            model.add(Dropout(dropout_rate))
            model.add(BatchNormalization())
            
            model.add(Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-4, l2=1e-3)))
            model.add(Dropout(dropout_rate))
            
        else:  # complex
            # Complex model
            model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
            model.add(Dropout(dropout_rate))
            model.add(BatchNormalization())
            
            model.add(Bidirectional(LSTM(64, return_sequences=True)))
            model.add(Dropout(dropout_rate))
            model.add(BatchNormalization())
            
            model.add(Bidirectional(LSTM(32, return_sequences=False)))
            model.add(Dropout(dropout_rate))
            model.add(BatchNormalization())
            
            model.add(Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-4, l2=1e-3)))
            model.add(Dropout(dropout_rate))
        
        # Output layer
        if task == 'regression':
            model.add(Dense(output_dim, activation='linear'))
        elif output_dim == 1:
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(output_dim, activation='softmax'))
        
        # Compile model with appropriate optimizer
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        logger.info(f"Built {model_complexity} LSTM model for {task} with loss={loss}")
        return model
        
    except Exception as e:
        logger.error(f"Error building LSTM model: {e}")
        return None

def build_advanced_model(input_shape, output_dim=1, model_type='hybrid', dropout_rate=0.3, 
                       learning_rate=0.001, loss=None, metrics=None, task='regression'):
    """
    Build an advanced model architecture for stock prediction with improved design
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, features)
        output_dim (int): Number of output dimensions
        model_type (str): Type of advanced model ('cnn_lstm', 'bidirectional', 'attention', 'hybrid')
        dropout_rate (float): Dropout rate for regularization
        learning_rate (float): Learning rate for Adam optimizer
        loss (str): Loss function
        metrics (list): List of metrics
        task (str): Type of task ('regression' or 'classification')
        
    Returns:
        tensorflow.keras.models.Model: Compiled advanced model
    """
    try:
        # Set appropriate metrics based on task
        if metrics is None:
            if task == 'regression':
                metrics = ['mae', 'mse']
            else:
                metrics = ['accuracy']
        
        # Set appropriate loss function based on task
        if loss is None:
            if task == 'regression':
                loss = 'mse'
            elif output_dim == 1:  # Binary classification
                loss = 'binary_crossentropy'
            else:  # Multi-class classification
                loss = 'categorical_crossentropy'
        
        inputs = Input(shape=input_shape)
        
        if model_type == 'cnn_lstm':
            # CNN-LSTM architecture - good for extracting temporal patterns
            x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
            x = MaxPooling1D(pool_size=2)(x)
            x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(dropout_rate)(x)
            
            x = LSTM(64, return_sequences=True)(x)
            x = Dropout(dropout_rate)(x)
            x = BatchNormalization()(x)
            
            x = LSTM(32, return_sequences=False)(x)
            x = Dropout(dropout_rate)(x)
            x = BatchNormalization()(x)
            
        elif model_type == 'bidirectional':
            # Bidirectional GRU architecture - better for learning patterns in both directions
            x = Bidirectional(GRU(64, return_sequences=True))(inputs)
            x = Dropout(dropout_rate)(x)
            x = BatchNormalization()(x)
            
            x = Bidirectional(GRU(32, return_sequences=False))(x)
            x = Dropout(dropout_rate)(x)
            x = BatchNormalization()(x)
            
        elif model_type == 'attention':
            # Attention-based architecture - focuses on important parts of the sequence
            x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
            
            # Add attention mechanism
            attn_layer = MultiHeadAttention(num_heads=4, key_dim=8)(x, x)
            x = Add()([x, attn_layer])
            x = BatchNormalization()(x)
            
            x = LSTM(64, return_sequences=False)(x)
            x = Dropout(dropout_rate)(x)
            x = BatchNormalization()(x)
            
        else:  # hybrid - IMPROVED for better convergence
            # Simplified hybrid architecture
            # CNN path for feature extraction
            conv_path = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
            conv_path = MaxPooling1D(pool_size=2)(conv_path)
            conv_path = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(conv_path)
            conv_path = Dropout(dropout_rate/2)(conv_path)  # Less dropout in parallel paths
            
            # LSTM path for sequence learning
            lstm_path = LSTM(64, return_sequences=True)(inputs)
            lstm_path = Dropout(dropout_rate/2)(lstm_path)
            
            # Merge paths
            merged = Concatenate()([conv_path, lstm_path])
            
            # Process merged features
            x = LSTM(32, return_sequences=False)(merged)
            x = Dropout(dropout_rate)(x)
            x = BatchNormalization()(x)
        
        # Common dense layers with stronger regularization
        x = Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=1e-4, l2=1e-3))(x)
        x = Dropout(dropout_rate)(x)
        
        # Output layer
        if task == 'regression':
            outputs = Dense(output_dim, activation='linear')(x)
        elif output_dim == 1:
            outputs = Dense(1, activation='sigmoid')(x)
        else:
            outputs = Dense(output_dim, activation='softmax')(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        logger.info(f"Built advanced {model_type} model for {task} with loss={loss}")
        return model
        
    except Exception as e:
        logger.error(f"Error building advanced model: {e}")
        return None

def build_simple_hybrid_model(input_shape, output_dim=1, dropout_rate=0.3, learning_rate=0.001, task='classification'):
    """
    Build a simpler hybrid model specifically for limited stock data
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, features)
        output_dim (int): Number of output dimensions
        dropout_rate (float): Dropout rate for regularization
        learning_rate (float): Learning rate for Adam optimizer
        task (str): Type of task ('regression' or 'classification')
        
    Returns:
        tensorflow.keras.models.Model: Compiled simple hybrid model
    """
    try:
        inputs = Input(shape=input_shape)
        
        # CNN path for extracting patterns
        conv = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
        
        # LSTM path for sequence learning
        lstm = LSTM(32, return_sequences=False)(inputs)
        lstm = Dropout(dropout_rate)(lstm)
        lstm = BatchNormalization()(lstm)
        
        # Combine paths
        combined = Concatenate()([Flatten()(conv), lstm])
        
        # Dense layers with strong regularization
        x = Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=1e-4, l2=1e-3))(combined)
        x = Dropout(dropout_rate)(x)
        
        # Output layer
        if task == 'regression':
            outputs = Dense(output_dim, activation='linear')(x)
            loss = 'mse'
            metrics = ['mae']
        elif output_dim == 1:
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            outputs = Dense(output_dim, activation='softmax')(x)
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        logger.info(f"Built simple hybrid model for {task} with loss={loss}")
        return model
        
    except Exception as e:
        logger.error(f"Error building simple hybrid model: {e}")
        return None

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, 
              checkpoint_path=None, tensorboard_path=None, patience=10, initial_lr=0.001):
    """
    Train a deep learning model with improved training strategy
    
    Args:
        model (tensorflow.keras.models.Model): Model to train
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training targets
        X_val (numpy.ndarray): Validation features
        y_val (numpy.ndarray): Validation targets
        epochs (int): Maximum number of epochs
        batch_size (int): Batch size for training
        checkpoint_path (str): Path to save model checkpoints
        tensorboard_path (str): Path for TensorBoard logs
        patience (int): Patience for early stopping
        initial_lr (float): Initial learning rate
        
    Returns:
        tensorflow.keras.models.Model, dict: Trained model and training history
    """
    try:
        callbacks = []
        
        # Early stopping - more patience for complex models
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Better learning rate schedule - more aggressive at start, gentler reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # More gentle reduction (was 0.2)
            patience=5,
            min_lr=1e-5,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            model_checkpoint = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(model_checkpoint)
        
        # TensorBoard logging
        if tensorboard_path:
            os.makedirs(tensorboard_path, exist_ok=True)
            tensorboard = TensorBoard(
                log_dir=tensorboard_path,
                histogram_freq=1,
                write_graph=True
            )
            callbacks.append(tensorboard)
        
        # Train the model with class weights for imbalanced data
        class_weights = None
        if len(np.unique(y_train)) <= 10:  # Only for classification with few classes
            # Calculate class weights inversely proportional to frequency
            class_counts = np.bincount(y_train.astype(int).flatten())
            total = len(y_train)
            class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
            logger.info(f"Using class weights: {class_weights}")
        
        # Train the model
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        training_time = time.time() - start_time
        
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        
        return model, history.history
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None, None

def evaluate_model(model, X_test, y_test, task='regression'):
    """
    Evaluate model performance on test data with improved metrics
    
    Args:
        model (tensorflow.keras.models.Model): Trained model
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test targets
        task (str): Type of task ('regression' or 'classification')
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Metrics dictionary
        metrics = {}
        
        if task == 'regression':
            # Regression metrics
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['r2'] = r2_score(y_test, y_pred)
            
            # Additional financial metrics
            # Direction accuracy (correct sign prediction)
            direction_actual = np.sign(y_test)
            direction_pred = np.sign(y_pred)
            metrics['direction_accuracy'] = np.mean(direction_actual == direction_pred)
            
            logger.info(f"Regression metrics: MSE={metrics['mse']:.4f}, RMSE={metrics['rmse']:.4f}, "
                     f"MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}, "
                     f"Direction Accuracy={metrics['direction_accuracy']:.4f}")
            
        else:
            # Classification metrics - FIXED FOR CORRECT HANDLING
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                # Multi-class classification
                y_pred_class = np.argmax(y_pred, axis=1)
                signal_strength = np.max(y_pred, axis=1)
            else:
                # Binary classification
                y_pred = y_pred.flatten()  # Ensure 1D array
                y_pred_class = (y_pred > 0.5).astype(int)
                signal_strength = y_pred
            
            # Convert y_test to 1D array if needed
            if len(y_test.shape) > 1:
                y_test = y_test.flatten()
            
            # Make sure both are same type
            y_test = y_test.astype(int)
            y_pred_class = y_pred_class.astype(int)
            
            # Calculate metrics
            metrics['accuracy'] = accuracy_score(y_test, y_pred_class)
            
            # Only calculate these if there are multiple classes
            unique_classes = np.unique(np.concatenate([y_test, y_pred_class]))
            if len(unique_classes) > 1:
                metrics['precision'] = precision_score(y_test, y_pred_class, average='weighted')
                metrics['recall'] = recall_score(y_test, y_pred_class, average='weighted')
                metrics['f1'] = f1_score(y_test, y_pred_class, average='weighted')
                
                # Add classification report and confusion matrix
                class_report = classification_report(y_test, y_pred_class, output_dict=True)
                conf_matrix = confusion_matrix(y_test, y_pred_class)
                
                metrics['classification_report'] = class_report
                metrics['confusion_matrix'] = conf_matrix.tolist()
                
                logger.info(f"Classification metrics: Accuracy={metrics['accuracy']:.4f}, "
                         f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
                         f"F1={metrics['f1']:.4f}")
            else:
                logger.warning("Only one class present in predictions - skipping precision/recall/f1")
                metrics['precision'] = 0.0
                metrics['recall'] = 0.0
                metrics['f1'] = 0.0
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return {'error': str(e)}

def save_model_results(model, history, metrics, model_path, metadata_path, plotting_path=None):
    """
    Save model, history, and evaluation results
    
    Args:
        model (tensorflow.keras.models.Model): Trained model
        history (dict): Training history
        metrics (dict): Evaluation metrics
        model_path (str): Path to save the model
        metadata_path (str): Path to save metadata
        plotting_path (str): Path to save plots
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save model metadata
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        metadata = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_history': history,
            'evaluation_metrics': metrics
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)
        logger.info(f"Model metadata saved to {metadata_path}")
        
        # Create and save plots if path is provided
        if plotting_path and history:
            os.makedirs(plotting_path, exist_ok=True)
            
            # Plot training & validation loss
            plt.figure(figsize=(12, 8))
            
            # Loss plot
            plt.subplot(2, 2, 1)
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Accuracy plot (if available)
            if 'accuracy' in history:
                plt.subplot(2, 2, 2)
                plt.plot(history['accuracy'], label='Training Accuracy')
                plt.plot(history['val_accuracy'], label='Validation Accuracy')
                plt.title('Model Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
            
            # Other metrics plots
            subplot_idx = 3
            for metric in history.keys():
                if metric not in ['loss', 'val_loss', 'accuracy', 'val_accuracy']:
                    if subplot_idx <= 4:  # Maximum 4 subplots
                        plt.subplot(2, 2, subplot_idx)
                        plt.plot(history[metric], label=f'Training {metric}')
                        val_metric = f'val_{metric}'
                        if val_metric in history:
                            plt.plot(history[val_metric], label=f'Validation {metric}')
                        plt.title(f'Model {metric.capitalize()}')
                        plt.xlabel('Epoch')
                        plt.ylabel(metric.capitalize())
                        plt.legend()
                        subplot_idx += 1
            
            plt.tight_layout()
            plt.savefig(os.path.join(plotting_path, 'training_history.png'))
            plt.close()
            
            # If it's a classification task, plot confusion matrix
            if 'confusion_matrix' in metrics:
                plt.figure(figsize=(8, 6))
                cm = np.array(metrics['confusion_matrix'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.tight_layout()
                plt.savefig(os.path.join(plotting_path, 'confusion_matrix.png'))
                plt.close()
            
            logger.info(f"Model plots saved to {plotting_path}")
            
            return True
        
    except Exception as e:
        logger.error(f"Error saving model results: {e}")
        return False

def create_stock_prediction_ensemble(X_train, y_train, X_val, y_val, task='classification'):
    """
    Create an ensemble of different models for better prediction accuracy
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training targets
        X_val (numpy.ndarray): Validation features
        y_val (numpy.ndarray): Validation targets
        task (str): Type of task ('regression' or 'classification')
        
    Returns:
        callable: Function to make ensemble predictions
    """
    try:
        models = []
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # 1. Simple LSTM model
        logger.info("Building model 1/3 for ensemble: Simple LSTM")
        model1 = build_lstm_model(
            input_shape=input_shape,
            output_dim=1,
            model_complexity='simple',
            task=task
        )
        
        # 2. CNN-LSTM model
        logger.info("Building model 2/3 for ensemble: CNN-LSTM")
        model2 = build_advanced_model(
            input_shape=input_shape,
            output_dim=1,
            model_type='cnn_lstm',
            task=task
        )
        
        # 3. Simple Hybrid model
        logger.info("Building model 3/3 for ensemble: Simple Hybrid")
        model3 = build_simple_hybrid_model(
            input_shape=input_shape,
            output_dim=1,
            task=task
        )
        
        # Train each model
        for i, model in enumerate([model1, model2, model3]):
            logger.info(f"Training ensemble model {i+1}/3...")
            model.fit(
                X_train, y_train,
                epochs=30,  # Fewer epochs for ensemble models
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
                ],
                verbose=1
            )
            models.append(model)
        
        # Function to make ensemble predictions
        def ensemble_predict(X):
            predictions = []
            for model in models:
                pred = model.predict(X)
                predictions.append(pred)
            
            # Average predictions for final output
            ensemble_pred = np.mean(predictions, axis=0)
            return ensemble_pred
        
        logger.info("Ensemble model creation complete")
        return ensemble_predict
        
    except Exception as e:
        logger.error(f"Error creating ensemble model: {e}")
        return None

def predict_with_augmentation(model, X, n_augmentations=5):
    """
    Make predictions with test-time augmentation for more robustness
    
    Args:
        model (tensorflow.keras.models.Model): Trained model
        X (numpy.ndarray): Input data
        n_augmentations (int): Number of augmentations to perform
        
    Returns:
        numpy.ndarray: Augmented predictions
    """
    try:
        predictions = []
        
        # Original prediction
        pred_original = model.predict(X)
        predictions.append(pred_original)
        
        # Augmented predictions
        for i in range(n_augmentations):
            # Add small noise to input sequence
            noise_level = 0.01 * (i + 1)  # Gradually increase noise
            X_noisy = X + np.random.normal(0, noise_level, X.shape)
            
            pred_aug = model.predict(X_noisy)
            predictions.append(pred_aug)
        
        # Average all predictions
        final_pred = np.mean(predictions, axis=0)
        logger.info(f"Made predictions with {n_augmentations} augmentations")
        return final_pred
        
    except Exception as e:
        logger.error(f"Error in test-time augmentation: {e}")
        return model.predict(X)  # Fall back to original prediction

def train_stock_model(ticker, processed_data_path, model_output_path, target_col='Target_Direction_5d',
                    model_type='hybrid', model_complexity='medium', sequence_length=20,
                    test_size=0.2, task='classification', epochs=50, batch_size=32,
                    use_ensemble=False, use_augmentation=True):
    """
    Train a model for a specific stock with improved training approach
    
    Args:
        ticker (str): Stock ticker symbol
        processed_data_path (str): Path to processed data
        model_output_path (str): Directory to save model outputs
        target_col (str): Target column name
        model_type (str): Type of model to build
        model_complexity (str): Complexity of model architecture
        sequence_length (int): Length of input sequences
        test_size (float): Proportion of data for testing
        task (str): Type of task ('regression' or 'classification')
        epochs (int): Maximum number of epochs
        batch_size (int): Batch size for training
        use_ensemble (bool): Whether to use ensemble modeling
        use_augmentation (bool): Whether to use data augmentation
        
    Returns:
        tuple: Model, evaluation metrics
    """
    try:
        logger.info(f"Training model for {ticker} using {model_type} architecture")
        
        # Load processed data
        file_path = os.path.join(processed_data_path, f"{ticker}_processed.csv")
        if not os.path.exists(file_path):
            logger.error(f"Processed data file not found: {file_path}")
            return None, None
        
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Prepare data for training
        X_train, X_test, y_train, y_test = prepare_train_test_data(
            df, target_col, test_size, sequence_length
        )
        
        if X_train is None:
            logger.error(f"Failed to prepare training data for {ticker}")
            return None, None
        
        # Split test data into validation and test
        val_size = int(len(X_test) * 0.5)
        X_val, X_test = X_test[:val_size], X_test[val_size:]
        y_val, y_test = y_test[:val_size], y_test[val_size:]
        
        # For classification, balance the dataset
        if task == 'classification' and len(np.unique(y_train)) <= 10:
            X_train, y_train = balance_classification_dataset(X_train, y_train)
        
        # Apply data augmentation if requested
        if use_augmentation:
            X_train, y_train = augment_financial_data(X_train, y_train)
        
        # Determine output dimension
        if task == 'regression':
            output_dim = 1
        elif target_col.startswith('Target_Direction') or target_col.startswith('Target_Significant'):
            output_dim = 1  # Binary classification
        elif target_col.startswith('Target_Signal'):
            output_dim = 3  # Three-class classification (assumes -1, 0, 1)
        else:
            output_dim = 1
        
        # Create directories for model artifacts
        os.makedirs(model_output_path, exist_ok=True)
        model_name = f"{ticker}_{model_type}_{task}"
        checkpoint_path = os.path.join(model_output_path, "checkpoints", f"{model_name}_checkpoint.h5")
        final_model_path = os.path.join(model_output_path, "final", f"{model_name}_model.h5")
        metadata_path = os.path.join(model_output_path, "final", f"{model_name}_metadata.json")
        tensorboard_path = os.path.join(model_output_path, "logs", model_name)
        plotting_path = os.path.join(model_output_path, "plots", model_name)
        
        # Train the model
        if use_ensemble:
            # Create and train ensemble
            logger.info(f"Creating ensemble model for {ticker}")
            ensemble_predictor = create_stock_prediction_ensemble(
                X_train, y_train, X_val, y_val, task=task
            )
            
            # Evaluate ensemble
            if ensemble_predictor:
                y_pred = ensemble_predictor(X_test)
                
                # Convert ensemble predictions for classification
                if task == 'classification':
                    if output_dim == 1:
                        y_pred_class = (y_pred > 0.5).astype(int)
                    else:
                        y_pred_class = np.argmax(y_pred, axis=1)
                    
                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred_class),
                        'ensemble': True
                    }
                else:
                    metrics = {
                        'mse': mean_squared_error(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'mae': mean_absolute_error(y_test, y_pred),
                        'ensemble': True
                    }
                
                logger.info(f"Ensemble model metrics: {metrics}")
                return ensemble_predictor, metrics
            else:
                logger.error("Ensemble creation failed, falling back to single model")
        
        # Build single model (fallback if ensemble fails or not requested)
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        if model_type == 'simple_hybrid':
            # Special case for simple hybrid model
            model = build_simple_hybrid_model(
                input_shape, output_dim, 
                task=task
            )
        elif model_type == 'lstm':
            model = build_lstm_model(
                input_shape, output_dim, model_complexity, 
                task=task
            )
        else:
            model = build_advanced_model(
                input_shape, output_dim, model_type, 
                task=task
            )
        
        if model is None:
            logger.error(f"Failed to build model for {ticker}")
            return None, None
        
        # Train model
        trained_model, history = train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size,
            checkpoint_path=checkpoint_path,
            tensorboard_path=tensorboard_path
        )
        
        if trained_model is None:
            logger.error(f"Failed to train model for {ticker}")
            return None, None
        
        # Evaluate model
        metrics = evaluate_model(trained_model, X_test, y_test, task=task)
        
        # Save model results
        save_model_results(
            trained_model, history, metrics,
            final_model_path, metadata_path, plotting_path
        )
        
        logger.info(f"Successfully trained and evaluated model for {ticker}")
        return trained_model, metrics
        
    except Exception as e:
        logger.error(f"Error training stock model for {ticker}: {e}")
        return None, None

def train_models_for_multiple_stocks(tickers, processed_data_path="ml/data/processed/", 
                                   model_output_path="ml/models/", **kwargs):
    """
    Train models for multiple stocks with improved training approach
    
    Args:
        tickers (list): List of stock ticker symbols
        processed_data_path (str): Path to processed data
        model_output_path (str): Directory to save model outputs
        **kwargs: Additional arguments for train_stock_model
        
    Returns:
        dict: Dictionary of model metrics by ticker
    """
    try:
        results = {}
        
        for ticker in tqdm(tickers, desc="Training models"):
            logger.info(f"Starting training for {ticker}")
            
            model, metrics = train_stock_model(
                ticker, 
                processed_data_path, 
                model_output_path, 
                **kwargs
            )
            
            if model is not None and metrics is not None:
                results[ticker] = metrics
                logger.info(f"Completed model training for {ticker}")
            else:
                logger.warning(f"Model training failed for {ticker}")
        
        # Save summary of all results
        summary_path = os.path.join(model_output_path, "model_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        logger.info(f"Completed training for {len(results)} stocks")
        logger.info(f"Results summary saved to {summary_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error training models for multiple stocks: {e}")
        return {}

def train_sp500_combined_model(processed_data_path="ml/data/processed/", model_output_path="ml/models/",
                              model_type="hybrid", model_complexity="medium", target_col="Target_Direction_5d",
                              task="classification", sequence_length=20, test_size=0.2,
                              epochs=50, batch_size=32, max_stocks=None, augment_data=True,
                              use_gpu=True, early_stopping_patience=15, use_cyclical_lr=False):
    """
    Train a single model on data from all S&P 500 stocks with improved approach
    
    Args:
        processed_data_path (str): Path to processed data
        model_output_path (str): Directory to save model outputs
        model_type (str): Type of model to build
        model_complexity (str): Complexity of model architecture
        target_col (str): Target column name
        task (str): Type of task ('regression' or 'classification')
        sequence_length (int): Length of input sequences
        test_size (float): Proportion of data for testing
        epochs (int): Maximum number of epochs
        batch_size (int): Batch size for training
        max_stocks (int): Maximum number of stocks to include (None for all)
        augment_data (bool): Whether to augment training data
        use_gpu (bool): Whether to use GPU for training
        early_stopping_patience (int): Patience for early stopping
        use_cyclical_lr (bool): Whether to use cyclical learning rate
        
    Returns:
        tuple: Model, evaluation metrics
    """
    try:
        logger.info(f"Training consolidated S&P 500 model using {model_type} architecture")
        
        # Configure GPU usage if requested
        if use_gpu:
            logger.info("Using GPU for training if available")
            # Allow memory growth to prevent TensorFlow from allocating all GPU memory
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    # Currently, memory growth needs to be the same across GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Found {len(gpus)} GPU(s), enabled memory growth")
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    logger.warning(f"Error setting GPU memory growth: {e}")
        else:
            logger.info("Using CPU for training")
            # Force CPU usage
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
        # Import combined dataset function
        from preprocessing.feature_engineering import create_combined_dataset
        
        # Create combined dataset from all stocks
        X_train, X_test, y_train, y_test = create_combined_dataset(
            processed_data_path, 
            target_col,
            sequence_length,
            test_size,
            max_stocks
        )
        
        if X_train is None:
            logger.error("Failed to create combined dataset")
            return None, None
        
        # For classification, balance the dataset to prevent bias
        if task == 'classification' and target_col.startswith('Target_Direction'):
            X_train, y_train = balance_classification_dataset(X_train, y_train)
        
        # Augment training data if requested
        if augment_data:
            X_train, y_train = augment_financial_data(X_train, y_train, augmentation_factor=3)
        
        # Split test data into validation and test sets
        val_size = int(len(X_test) * 0.5)
        X_val, X_test = X_test[:val_size], X_test[val_size:]
        y_val, y_test = y_test[:val_size], y_test[val_size:]
        
        # Determine output dimension
        if task == 'regression':
            output_dim = 1
        elif target_col.startswith('Target_Direction') or target_col.startswith('Target_Significant'):
            output_dim = 1  # Binary classification
        elif target_col.startswith('Target_Signal'):
            output_dim = 3  # Three-class classification
        else:
            output_dim = 1
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Select simpler architecture for combined model
        if model_type == 'hybrid' and model_complexity == 'complex':
            logger.info("Downgrading to medium complexity for combined model")
            model_complexity = 'medium'
        
        if model_type == 'simple_hybrid':
            model = build_simple_hybrid_model(
                input_shape, output_dim,
                task=task
            )
        elif model_type == 'lstm':
            model = build_lstm_model(
                input_shape, output_dim, model_complexity,
                task=task
            )
        else:
            model = build_advanced_model(
                input_shape, output_dim, model_type,
                task=task
            )
        
        if model is None:
            logger.error("Failed to build model")
            return None, None
        
        # Create directories for model artifacts
        os.makedirs(model_output_path, exist_ok=True)
        model_name = f"sp500_combined_{model_type}_{task}"
        checkpoint_path = os.path.join(model_output_path, "checkpoints", f"{model_name}_checkpoint.h5")
        final_model_path = os.path.join(model_output_path, "final", f"{model_name}_model.h5")
        metadata_path = os.path.join(model_output_path, "final", f"{model_name}_metadata.json")
        tensorboard_path = os.path.join(model_output_path, "logs", model_name)
        plotting_path = os.path.join(model_output_path, "plots", model_name)
        
        # Set up custom learning rate if requested
        initial_lr = 0.001
        callbacks = []
        
        # Early stopping with patience
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Set up cyclical learning rate if requested
        if use_cyclical_lr:
            from tensorflow.keras.callbacks import LearningRateScheduler
            
            def cyclical_lr(epoch, lr):
                # Simple cyclical learning rate
                cycle_length = 10
                min_lr = 1e-5
                max_lr = 5e-3
                
                cycle = np.floor(1 + epoch / cycle_length)
                x = np.abs(epoch / cycle_length - (2 * cycle - 1) / 2)
                return min_lr + (max_lr - min_lr) * np.maximum(0, (1 - x))
            
            lr_scheduler = LearningRateScheduler(cyclical_lr)
            callbacks.append(lr_scheduler)
            logger.info("Using cyclical learning rate")
        
        # Checkpoint callback
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(model_checkpoint)
        
        # TensorBoard callback
        tensorboard = TensorBoard(
            log_dir=tensorboard_path,
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard)
        
        # Train model with appropriate callbacks
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        training_time = time.time() - start_time
        
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        # Evaluate model with test-time augmentation for more robust results
        y_pred = predict_with_augmentation(model, X_test)
        
        # Evaluate based on task type
        if task == 'regression':
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'direction_accuracy': np.mean(np.sign(y_test) == np.sign(y_pred))
            }
        else:
            if output_dim == 1:
                y_pred_class = (y_pred > 0.5).astype(int)
            else:
                y_pred_class = np.argmax(y_pred, axis=1)
                
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred_class),
                'precision': precision_score(y_test, y_pred_class, average='weighted'),
                'recall': recall_score(y_test, y_pred_class, average='weighted'),
                'f1': f1_score(y_test, y_pred_class, average='weighted')
            }
        
        # Save model results
        save_model_results(
            model, history.history, metrics,
            final_model_path, metadata_path, plotting_path
        )
        
        logger.info(f"Successfully trained and evaluated consolidated S&P 500 model")
        logger.info(f"Metrics: {metrics}")
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"Error training consolidated S&P 500 model: {e}")
        return None, None

# Main execution
if __name__ == "__main__":
    # Set up focused list of stocks with long histories and high liquidity
    major_stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "JNJ", 
        "JPM", "V", "PG", "UNH", "HD", "MA", "NVDA", "BAC", "DIS", "CSCO"
    ]
    
    # Example: Train models for major stocks only
    train_models_for_multiple_stocks(
        major_stocks,
        model_type='simple_hybrid',  # Use simpler model for better convergence
        task='classification',
        target_col='Target_Direction_5d',
        epochs=30,
        batch_size=32,
        use_ensemble=False,
        use_augmentation=True
    )