"""
Script to retrain stock prediction models from scratch
"""
import os
import sys
import argparse
import subprocess
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml/logs/retrain.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("retrain")

def clean_directories():
    """Clear previous model artifacts for a clean start"""
    logger.info("Cleaning directories for fresh start...")
    dirs_to_clean = [
        "ml/models/checkpoints",
        "ml/models/final",
        "ml/models/plots",
        "ml/preprocessing/scalers",
        "ml/prediction/signals",
        "ml/prediction/results",
        "ml/prediction/visualizations",
    ]
    
    for directory in dirs_to_clean:
        try:
            if os.path.exists(directory):
                # Remove files in the directory but keep the directory itself
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                logger.info(f"Cleaned {directory}")
            else:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created {directory}")
        except Exception as e:
            logger.error(f"Error cleaning {directory}: {e}")

def run_command(command, description):
    """Run a command and log its output"""
    logger.info(f"Starting {description}...")
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            shell=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            line = line.strip()
            if line:
                logger.info(line)
        
        # Wait for process to complete
        process.wait()
        
        # Get any errors
        stderr = process.stderr.read()
        if stderr:
            logger.error(f"Errors during {description}:")
            for line in stderr.strip().split('\n'):
                logger.error(line)
        
        duration = time.time() - start_time
        if process.returncode == 0:
            logger.info(f"Completed {description} successfully in {duration:.2f} seconds")
            return True
        else:
            logger.error(f"Failed {description} with return code {process.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"Error during {description}: {e}")
        return False

def download_data(subset=None):
    """Download stock data"""
    cmd = "python stock_predictor_main.py download"
    if subset:
        cmd += f" --subset {subset}"
    return run_command(cmd, "data download")

def process_data(subset=None):
    """Process downloaded data"""
    cmd = "python stock_predictor_main.py process"
    if subset:
        cmd += f" --subset {subset}"
    return run_command(cmd, "data processing")

def train_models(subset=None, model_type="simple_hybrid", epochs=30):
    """Train models on processed data"""
    cmd = f"python stock_predictor_main.py train --model-type {model_type} --epochs {epochs}"
    if subset:
        cmd += f" --subset {subset}"
    return run_command(cmd, "model training")

def generate_predictions(subset=None, use_historical=True):
    """Generate predictions using trained models"""
    cmd = "python stock_predictor_main.py predict --visualize"
    if subset:
        cmd += f" --subset {subset}"
    if use_historical:
        cmd += " --use-historical"
    return run_command(cmd, "prediction generation")

def full_retraining_pipeline(subset=None, model_type="simple_hybrid", epochs=30, clean=True, use_historical=True):
    """Run the complete retraining pipeline"""
    start_time = time.time()
    logger.info("Starting full retraining pipeline")
    
    if clean:
        clean_directories()
    
    # Step 1: Download data
    if not download_data(subset):
        logger.error("Data download failed. Stopping pipeline.")
        return False
    
    # Step 2: Process data
    if not process_data(subset):
        logger.error("Data processing failed. Stopping pipeline.")
        return False
    
    # Step 3: Train models
    if not train_models(subset, model_type, epochs):
        logger.error("Model training failed. Stopping pipeline.")
        return False
    
    # Step 4: Generate predictions
    if not generate_predictions(subset, use_historical):
        logger.error("Prediction generation failed.")
        # Continue anyway since we have trained models
    
    total_time = time.time() - start_time
    logger.info(f"Full retraining pipeline completed in {total_time:.2f} seconds")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain stock prediction models from scratch")
    parser.add_argument("--subset", type=int, help="Number of stocks to use (default: all)")
    parser.add_argument("--model-type", type=str, default="simple_hybrid", 
                        choices=["simple_hybrid", "lstm", "cnn_lstm", "bidirectional", "attention", "hybrid"],
                        help="Type of model to train")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--no-clean", action="store_true", help="Skip cleaning directories")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    parser.add_argument("--skip-process", action="store_true", help="Skip data processing")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--skip-predict", action="store_true", help="Skip prediction generation")
    parser.add_argument("--no-historical", action="store_true", help="Don't use historical data for predictions")
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("ml/logs", exist_ok=True)
    
    start_time = time.time()
    logger.info(f"Starting retraining pipeline with arguments: {args}")
    
    try:
        if not args.no_clean:
            clean_directories()
        
        if not args.skip_download:
            download_data(args.subset)
        else:
            logger.info("Skipping data download as requested")
        
        if not args.skip_process:
            process_data(args.subset)
        else:
            logger.info("Skipping data processing as requested")
        
        if not args.skip_train:
            train_models(args.subset, args.model_type, args.epochs)
        else:
            logger.info("Skipping model training as requested")
        
        if not args.skip_predict:
            generate_predictions(args.subset, not args.no_historical)
        else:
            logger.info("Skipping prediction generation as requested")
        
        total_time = time.time() - start_time
        logger.info(f"Retraining pipeline completed in {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during retraining pipeline: {e}")
        sys.exit(1)
    
    sys.exit(0)