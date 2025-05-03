# AI-Powered Stock Market Prediction Tool

This comprehensive tool uses deep learning to analyze stock market data, identify patterns, and generate trading signals. It's designed for financial analysts, investors, and stock market enthusiasts who want to leverage AI for stock market predictions.

## Features

- **Data Collection**: Automatically downloads historical stock data for S&P 500 companies from Yahoo Finance (1950-present)
- **Advanced Feature Engineering**: Calculates 50+ technical indicators including moving averages, oscillators, and momentum indicators
- **Deep Learning Models**: Uses LSTM and other advanced neural network architectures to predict stock movements
- **Trading Signals**: Generates Buy/Sell/Hold signals based on model predictions
- **Performance Evaluation**: Comprehensive metrics to evaluate prediction accuracy
- **Visualization**: Plots of price movements with signals and confidence levels
- **Local Execution**: Runs completely on your local machine

## System Requirements

- Windows, macOS, or Linux operating system
- Python 3.7 or higher (Python 3.9 recommended)
- Anaconda Distribution (for easy environment management)
- Administrator privileges (for installation)
- 8GB+ RAM recommended for model training
- GPU support optional but recommended for faster training

## Installation Guide

### Step 1: Install Anaconda

If you don't have Anaconda installed:

1. Download Anaconda from [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
2. Follow the installation instructions for your operating system
3. Verify installation by opening a terminal/command prompt and running: `conda --version`

### Step 2: Set Up the Project

1. Download or clone this repository:
   ```
   git clone https://github.com/yourusername/stock-predictor.git
   cd stock-predictor
   ```

2. Run the setup script to create the environment and directory structure:
   ```
   python setup_environment.py
   ```

3. Activate the Conda environment:
   ```
   conda activate stock_predictor
   ```

## Usage Instructions

The tool provides a command-line interface for all operations. Here's how to use it:

### Basic Operations

1. **Download Historical Data**:
   ```
   python stock_predictor_main.py download
   ```

2. **Process Data and Calculate Technical Indicators**:
   ```
   python stock_predictor_main.py process
   ```

3. **Train Prediction Models**:
   ```
   python stock_predictor_main.py train
   ```

4. **Generate Trading Signals**:
   ```
   python stock_predictor_main.py predict
   ```

5. **Run the Complete Pipeline**:
   ```
   python stock_predictor_main.py all
   ```

### Advanced Options

- **Process Specific Stocks**:
  ```
  python stock_predictor_main.py process --ticker AAPL
  ```

- **Use Advanced Model Architecture**:
  ```
  python stock_predictor_main.py train --model-type hybrid --model-complexity complex
  ```

- **Custom Training Parameters**:
  ```
  python stock_predictor_main.py train --epochs 100 --batch-size 64 --sequence-length 30
  ```

- **Adjust Prediction Confidence Threshold**:
  ```
  python stock_predictor_main.py predict --threshold 0.8
  ```

- **Get Help on Available Options**:
  ```
  python stock_predictor_main.py --help
  ```

## Project Structure

```
stock_predictor/
├── ml/                      # Machine learning components
│   ├── data/                # Data storage
│   │   ├── raw/             # Raw data from Yahoo Finance
│   │   └── processed/       # Processed data with features
│   ├── models/              # Trained models
│   │   ├── checkpoints/     # Training checkpoints
│   │   └── final/           # Final trained models
│   ├── preprocessing/       # Data preprocessing modules
│   ├── training/            # Model training code
│   ├── prediction/          # Prediction generation
│   │   ├── results/         # Prediction results
│   │   ├── signals/         # Trading signals
│   │   └── visualizations/  # Signal visualizations
│   └── logs/                # Log files
├── db/                      # Database files
├── notebooks/               # Jupyter notebooks for examples
└── config/                  # Configuration files
```

## Training Process

The model training process involves:

1. **Data Preparation**: Processes historical price data and calculates technical indicators
2. **Feature Engineering**: Creates lagged features and different prediction targets
3. **Model Architecture**: Uses LSTM (Long Short-Term Memory) neural networks
4. **Training**: Fits the model on historical data with early stopping
5. **Evaluation**: Calculates performance metrics on test data

Model training on the full S&P 500 can take several hours depending on your hardware. For testing, you can use the `--subset` option to process fewer stocks.

## Prediction Interpretation

The tool generates three types of trading signals:

- **Buy Signal**: Model predicts the stock price will increase
- **Sell Signal**: Model predicts the stock price will decrease
- **Hold/No Signal**: Model prediction confidence is below the threshold

Each signal includes a confidence score. Higher confidence scores indicate stronger signals.

## Example Notebook

For interactive exploration, check out the example Jupyter notebook:

```
jupyter notebook notebooks/example_usage.ipynb
```

## Troubleshooting

- **Package Installation Errors**: Try installing problematic packages manually with `pip install <package_name>`
- **Memory Errors During Training**: Reduce batch size or use a smaller subset of stocks
- **TA-Lib Installation Issues**: Follow platform-specific instructions at [https://github.com/mrjbq7/ta-lib](https://github.com/mrjbq7/ta-lib)
- **Long Training Times**: Consider using GPU acceleration if available or reducing model complexity