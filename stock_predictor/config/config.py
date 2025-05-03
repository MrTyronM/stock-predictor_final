"""
Configuration module for the AI Stock Market Prediction Tool

This module loads and manages configurations for the entire system
"""
import os
import json
import yaml
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml/logs/config.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("config")

# Create necessary directories
os.makedirs("ml/logs", exist_ok=True)
os.makedirs("config", exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    "system": {
        "version": "1.0.0",
        "log_level": "INFO",
        "max_threads": 4,
        "use_gpu": True,
        "data_storage_path": "ml/data/",
        "cache_predictions": True,
        "auto_cleanup": False
    },
    "data": {
        "historical_range": "max",  # "1y", "2y", "5y", "10y", "max"
        "data_sources": ["yahoo"],
        "indicators_set": "standard",  # "basic", "standard", "advanced", "comprehensive"
        "update_frequency": "daily",  # "daily", "weekly", "monthly", "manual"
        "update_time": "01:00"
    },
    "model": {
        "model_type": "hybrid",  # "lstm", "cnn_lstm", "bidirectional", "attention", "hybrid"
        "model_complexity": "medium",  # "simple", "medium", "complex"
        "prediction_horizon": 5,  # days
        "confidence_threshold": 0.7,
        "sequence_length": 20,
        "batch_size": 32,
        "epochs": 50,
        "early_stopping_patience": 10,
        "test_size": 0.2
    },
    "stocks": {
        "process_sp500": True,
        "process_watchlist": False,
        "process_custom": False,
        "custom_tickers": []
    },
    "database": {
        "type": "sqlite",  # "sqlite", "mysql"
        "host": "",
        "user": "",
        "password": "",
        "database": "",
        "auto_update_actual_values": True
    },
    "ui": {
        "theme": "light",  # "light", "dark"
        "default_view": "dashboard",
        "chart_style": "standard",
        "show_predictions_count": 10,
        "notification_duration": 5000
    }
}

class Config:
    """
    Configuration class for managing system settings
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Initialize configuration
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        self.config = DEFAULT_CONFIG.copy()
        self.load_config()
    
    def load_config(self):
        """
        Load configuration from file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.json'):
                        loaded_config = json.load(f)
                    else:
                        loaded_config = yaml.safe_load(f)
                
                # Update configuration with loaded values
                self._update_dict(self.config, loaded_config)
                logger.info(f"Loaded configuration from {self.config_path}")
                return True
            else:
                logger.info(f"Configuration file {self.config_path} not found, using defaults")
                self.save_config()  # Save default configuration
                return False
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
    
    def save_config(self):
        """
        Save configuration to file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                if self.config_path.endswith('.json'):
                    json.dump(self.config, f, indent=4)
                else:
                    yaml.dump(self.config, f, default_flow_style=False)
            
            logger.info(f"Saved configuration to {self.config_path}")
            return True
                
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get(self, section, key=None, default=None):
        """
        Get configuration value
        
        Args:
            section (str): Configuration section
            key (str): Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        try:
            if section not in self.config:
                return default
            
            if key is None:
                return self.config[section]
            
            if key not in self.config[section]:
                return default
            
            return self.config[section][key]
                
        except Exception as e:
            logger.error(f"Error getting configuration value: {e}")
            return default
    
    def set(self, section, key, value):
        """
        Set configuration value
        
        Args:
            section (str): Configuration section
            key (str): Configuration key
            value: Configuration value
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if section not in self.config:
                self.config[section] = {}
            
            self.config[section][key] = value
            return True
                
        except Exception as e:
            logger.error(f"Error setting configuration value: {e}")
            return False
    
    def update_section(self, section, values):
        """
        Update configuration section
        
        Args:
            section (str): Configuration section
            values (dict): Configuration values
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if section not in self.config:
                self.config[section] = {}
            
            self.config[section].update(values)
            return True
                
        except Exception as e:
            logger.error(f"Error updating configuration section: {e}")
            return False
    
    def _update_dict(self, target, source):
        """
        Recursively update dictionary
        
        Args:
            target (dict): Target dictionary
            source (dict): Source dictionary
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_dict(target[key], value)
            else:
                target[key] = value
    
    def reset_to_defaults(self):
        """
        Reset configuration to defaults
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.config = DEFAULT_CONFIG.copy()
            return self.save_config()
                
        except Exception as e:
            logger.error(f"Error resetting configuration: {e}")
            return False
    
    def export_config(self, file_path):
        """
        Export configuration to file
        
        Args:
            file_path (str): Path to export file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                if file_path.endswith('.json'):
                    json.dump(self.config, f, indent=4)
                else:
                    yaml.dump(self.config, f, default_flow_style=False)
            
            logger.info(f"Exported configuration to {file_path}")
            return True
                
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False
    
    def import_config(self, file_path):
        """
        Import configuration from file
        
        Args:
            file_path (str): Path to import file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"Import file {file_path} not found")
                return False
            
            with open(file_path, 'r') as f:
                if file_path.endswith('.json'):
                    imported_config = json.load(f)
                else:
                    imported_config = yaml.safe_load(f)
            
            # Update configuration with imported values
            self._update_dict(self.config, imported_config)
            
            # Save the updated configuration
            self.save_config()
            
            logger.info(f"Imported configuration from {file_path}")
            return True
                
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False
    
    def get_all(self):
        """
        Get complete configuration
        
        Returns:
            dict: Complete configuration
        """
        return self.config.copy()

def load_model_config(model_name, models_dir="ml/models/final"):
    """
    Load model-specific configuration
    
    Args:
        model_name (str): Model name
        models_dir (str): Models directory
        
    Returns:
        dict: Model configuration
    """
    try:
        # Check for model metadata file
        metadata_file = os.path.join(models_dir, f"{model_name}_metadata.json")
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Extract configuration
            if 'model_architecture' in metadata:
                model_config = metadata['model_architecture']
                logger.info(f"Loaded model configuration for {model_name}")
                return model_config
        
        logger.warning(f"No model configuration found for {model_name}")
        return {}
            
    except Exception as e:
        logger.error(f"Error loading model configuration: {e}")
        return {}

def get_data_directory(config, data_type="raw"):
    """
    Get data directory path
    
    Args:
        config (Config): Configuration object
        data_type (str): Data type ('raw' or 'processed')
        
    Returns:
        str: Data directory path
    """
    base_path = config.get("system", "data_storage_path", "ml/data/")
    return os.path.join(base_path, data_type)

def get_models_directory(config, model_type="final"):
    """
    Get models directory path
    
    Args:
        config (Config): Configuration object
        model_type (str): Model type ('final' or 'checkpoints')
        
    Returns:
        str: Models directory path
    """
    return os.path.join("ml/models/", model_type)

def create_backup(config, include_data=True, include_models=True):
    """
    Create backup of configuration and optionally data and models
    
    Args:
        config (Config): Configuration object
        include_data (bool): Include data in backup
        include_models (bool): Include models in backup
        
    Returns:
        str: Path to backup file
    """
    try:
        # Create backup directory
        backup_dir = "backups"
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create backup file name
        backup_file = os.path.join(backup_dir, f"backup_{timestamp}.json")
        
        # Create backup data
        backup_data = {
            "timestamp": timestamp,
            "config": config.get_all(),
            "included": {
                "data": include_data,
                "models": include_models
            }
        }
        
        # Add data file list if included
        if include_data:
            data_dir = get_data_directory(config)
            if os.path.exists(data_dir):
                data_files = os.listdir(data_dir)
                backup_data["data_files"] = data_files
        
        # Add model file list if included
        if include_models:
            models_dir = get_models_directory(config)
            if os.path.exists(models_dir):
                model_files = os.listdir(models_dir)
                backup_data["model_files"] = model_files
        
        # Save backup file
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, indent=4)
        
        logger.info(f"Created backup: {backup_file}")
        return backup_file
            
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return None

def restore_from_backup(backup_file, config, restore_data=True, restore_models=True):
    """
    Restore configuration and optionally data and models from backup
    
    Args:
        backup_file (str): Path to backup file
        config (Config): Configuration object
        restore_data (bool): Restore data
        restore_models (bool): Restore models
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not os.path.exists(backup_file):
            logger.error(f"Backup file {backup_file} not found")
            return False
        
        # Load backup file
        with open(backup_file, 'r') as f:
            backup_data = json.load(f)
        
        # Restore configuration
        if "config" in backup_data:
            config._update_dict(config.config, backup_data["config"])
            config.save_config()
            logger.info("Restored configuration from backup")
        
        # TODO: Implement data and model restoration if needed
        # This would involve copying files from backup locations
        
        return True
            
    except Exception as e:
        logger.error(f"Error restoring from backup: {e}")
        return False

# Create global configuration instance
config = Config()

if __name__ == "__main__":
    # Example usage
    cfg = Config()
    
    # Get configuration values
    print("Model type:", cfg.get("model", "model_type"))
    print("Prediction horizon:", cfg.get("model", "prediction_horizon"))
    
    # Set configuration values
    cfg.set("model", "batch_size", 64)
    print("Updated batch size:", cfg.get("model", "batch_size"))
    
    # Save configuration
    cfg.save_config()
    
    # Create backup
    backup_file = create_backup(cfg)
    
    # Reset to defaults
    cfg.reset_to_defaults()
    print("Reset batch size:", cfg.get("model", "batch_size"))
    
    # Restore from backup
    if backup_file:
        restore_from_backup(backup_file, cfg)
        print("Restored batch size:", cfg.get("model", "batch_size"))