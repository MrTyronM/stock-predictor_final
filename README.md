# AI-Powered Stock Market Prediction Tool

A comprehensive tool that uses machine learning to predict stock market trends and provide trading signals.

## Features

- Data acquisition from Yahoo Finance for all S&P 500 stocks
- Technical indicator calculation
- LSTM and advanced deep learning model training on combined stock data
- Trading signal generation
- Visualization of predictions

## Installation

1. Make sure you have Anaconda installed
2. Run `setup_environment.py` to set up the environment
3. Activate the conda environment: `conda activate stock_predictor`

## Usage

- Download data: `python stock_predictor_main.py download`
- Process data: `python stock_predictor_main.py process`
- Train a consolidated model on all S&P 500 stocks: `python stock_predictor_main.py train --combined-model`
- Train individual models: `python stock_predictor_main.py train`
- Make predictions: `python stock_predictor_main.py predict`
- Run the complete pipeline: `python stock_predictor_main.py all --combined-model`

## Advanced Options

Run `python stock_predictor_main.py --help` to see all available options.

For the consolidated model training, useful options include:
- `--augment-data`: Apply data augmentation techniques
- `--gpu`: Use GPU acceleration if available
- `--cyclical-lr`: Use cyclical learning rate
- `--early-stopping-patience 15`: Adjust patience for early stopping
- `--max-stocks 100`: Limit the number of stocks to use (for testing)
