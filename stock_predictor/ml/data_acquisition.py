"""
Data acquisition module for stock market prediction

This module downloads historical stock data from Yahoo Finance for S&P 500 companies
"""
import os
import pandas as pd
import numpy as np
import logging
import yfinance as yf
import requests
import bs4 as bs
import pickle
import time
from datetime import datetime
from tqdm import tqdm
import random
import io
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml/logs/data_acquisition.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_acquisition")

# Create necessary directories
os.makedirs("ml/logs", exist_ok=True)
os.makedirs("ml/data/raw", exist_ok=True)

def get_sp500_tickers():
    """
    Get list of S&P 500 tickers directly from Yahoo Finance
    
    Returns:
        list: List of S&P 500 ticker symbols
    """
    try:
        # Try to load from cache first
        cache_path = os.path.join("ml/data", "sp500_tickers.pkl")
        if os.path.exists(cache_path):
            # Check if cache is recent (less than 7 days old)
            cache_time = os.path.getmtime(cache_path)
            if (time.time() - cache_time) < 7 * 24 * 60 * 60:  # 7 days in seconds
                with open(cache_path, "rb") as f:
                    tickers = pickle.load(f)
                logger.info(f"Loaded {len(tickers)} S&P 500 tickers from cache")
                return tickers
        
        # If no recent cache, fetch from reliable sources
        tickers = []
        
        # Try GitHub maintained list
        try:
            url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
            response = requests.get(url)
            if response.status_code == 200:
                # Parse CSV directly from response content
                csv_data = io.StringIO(response.text)
                reader = csv.reader(csv_data)
                next(reader)  # Skip header
                tickers = [row[0] for row in reader]
                logger.info(f"Retrieved {len(tickers)} tickers from GitHub dataset")
            else:
                logger.warning(f"Failed to download S&P 500 constituents list: {response.status_code}")
        except Exception as e:
            logger.warning(f"Error retrieving tickers from GitHub: {e}")
            
        # If GitHub failed, try Wikipedia
        if not tickers:
            try:
                url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
                response = requests.get(url)
                soup = bs.BeautifulSoup(response.text, 'html.parser')
                tables = soup.find_all('table', {'class': 'wikitable'})
                
                if tables and len(tables) > 0:
                    table = tables[0]  # First table contains current S&P 500 companies
                    rows = table.find_all('tr')
                    
                    for row in rows[1:]:  # Skip header row
                        cells = row.find_all('td')
                        if cells and len(cells) > 0:
                            ticker = cells[0].text.strip()
                            tickers.append(ticker)
                    
                    logger.info(f"Retrieved {len(tickers)} tickers from Wikipedia")
            except Exception as e:
                logger.warning(f"Error retrieving tickers from Wikipedia: {e}")
        
        # If we still don't have tickers, use a hardcoded list
        if not tickers or len(tickers) < 400:
            logger.warning("Could not get complete S&P 500 component list from external sources")
            
            # Use a manually curated list of the largest S&P 500 companies
            fallback_tickers = [
                "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "TSLA", "UNH", "XOM", 
                "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "LLY", "ABBV", "AVGO", "PEP", "KO", 
                "COST", "TMO", "MCD", "ACN", "ABT", "WMT", "BAC", "CRM", "LIN", "PFE", "ADBE", 
                "CSCO", "NKE", "TXN", "DHR", "NEE", "VZ", "PM", "RTX", "CMCSA", "INTC", "WFC", 
                "IBM", "AMD", "QCOM", "T", "COP", "HON", "LOW", "ORCL", "CAT", "AMAT", "GE",
                "SPGI", "INTU", "DE", "AXP", "LMT", "SBUX", "BKNG", "MMC", "GS", "PLD", "BLK",
                "MDLZ", "GILD", "ADI", "REGN", "AMT", "MS", "TJX", "ISRG", "MO", "VRTX", "SYK",
                "C", "ADP", "TMUS", "ZTS", "ETN", "SO", "CB", "CL", "SCHW", "DUK", "BDX", "ELV",
                "EOG", "CME", "APD", "LRCX", "NSC", "ITW", "MPC", "AON", "ICE", "SLB", "CVS", "BSX"
            ]
            tickers = fallback_tickers
            logger.info(f"Using fallback list of {len(tickers)} tickers")
        
        # Clean up tickers for Yahoo Finance format (replace dots with hyphens)
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        # Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(tickers, f)
        
        logger.info(f"Saved {len(tickers)} S&P 500 tickers to cache")
        return tickers
        
    except Exception as e:
        logger.error(f"Error getting S&P 500 tickers: {e}")
        
        # Ultimate fallback - just use top 30 companies
        fallback_tickers = [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "TSLA", "UNH", "XOM", 
            "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "LLY", "ABBV", "AVGO", "PEP", "KO", 
            "COST", "TMO", "MCD", "ACN", "ABT", "WMT", "BAC", "CRM"
        ]
        logger.info(f"Using fallback list of {len(fallback_tickers)} tickers")
        return fallback_tickers

def get_available_tickers():
    """
    Get list of tickers that have already been downloaded
    
    Returns:
        list: List of ticker symbols with data available
    """
    try:
        # Get list of CSV files in the raw data directory
        data_dir = "ml/data/raw"
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and f != 'metadata.csv']
        
        # Extract ticker symbols from filenames
        tickers = [os.path.splitext(f)[0] for f in csv_files]
        
        logger.info(f"Found {len(tickers)} tickers with available data")
        return tickers
        
    except Exception as e:
        logger.error(f"Error getting available tickers: {e}")
        return []

def download_stock_data(ticker, start_date='1950-01-01', end_date=None, interval='1d'):
    """
    Download historical stock data from Yahoo Finance
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date for historical data (YYYY-MM-DD)
        end_date (str): End date for historical data (YYYY-MM-DD), defaults to today
        interval (str): Data interval ('1d', '1wk', '1mo')
        
    Returns:
        pandas.DataFrame: Historical stock data
    """
    try:
        # Default end date to today if not specified
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Download data from Yahoo Finance
        ticker_yf = yf.Ticker(ticker)
        df = ticker_yf.history(start=start_date, end=end_date, interval=interval)
        
        if df.empty:
            logger.warning(f"No data found for {ticker}")
            return None
        
        # Drop unnecessary columns and rename for consistency
        if 'Dividends' in df.columns:
            df = df.drop(['Dividends', 'Stock Splits'], axis=1, errors='ignore')
        
        # Ensure all column names are capitalized
        df.columns = [col.capitalize() for col in df.columns]
        
        logger.info(f"Downloaded {len(df)} rows of historical data for {ticker}")
        return df
        
    except Exception as e:
        logger.error(f"Error downloading data for {ticker}: {e}")
        return None

def save_stock_data(df, ticker, output_dir="ml/data/raw/"):
    """
    Save stock data to CSV file
    
    Args:
        df (pandas.DataFrame): Stock data
        ticker (str): Stock ticker symbol
        output_dir (str): Directory to save data
        
    Returns:
        str: Path to saved CSV file
    """
    try:
        if df is None or df.empty:
            logger.warning(f"No data to save for {ticker}")
            return None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV
        output_path = os.path.join(output_dir, f"{ticker}.csv")
        df.to_csv(output_path)
        
        logger.info(f"Saved data for {ticker} to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving data for {ticker}: {e}")
        return None

def download_sp500_with_retry(tickers, start_date='1950-01-01', max_retries=3, delay=1):
    """
    Download data for S&P 500 stocks with retry mechanism
    
    Args:
        tickers (list): List of ticker symbols
        start_date (str): Start date for historical data
        max_retries (int): Maximum number of retry attempts
        delay (int): Delay between retries in seconds
        
    Returns:
        dict: Dictionary of successful downloads by ticker
    """
    results = {}
    failed_tickers = []
    
    for ticker in tqdm(tickers, desc="Downloading stock data"):
        retries = 0
        success = False
        
        while retries < max_retries and not success:
            try:
                df = download_stock_data(ticker, start_date)
                
                if df is not None and not df.empty:
                    save_path = save_stock_data(df, ticker)
                    if save_path:
                        results[ticker] = save_path
                        success = True
                    else:
                        retries += 1
                else:
                    retries += 1
                
                # Add delay between requests to avoid rate limiting
                time.sleep(delay + random.random())
                
            except Exception as e:
                logger.error(f"Attempt {retries+1} failed for {ticker}: {e}")
                retries += 1
                time.sleep(delay * retries)  # Increasing delay with each retry
        
        if not success:
            failed_tickers.append(ticker)
    
    # Log results
    logger.info(f"Successfully downloaded data for {len(results)} stocks")
    if failed_tickers:
        logger.warning(f"Failed to download data for {len(failed_tickers)} stocks: {', '.join(failed_tickers)}")
    
    return results

def download_all_sp500_historical_data(start_date='1950-01-01'):
    """
    Download historical data for all S&P 500 companies
    
    Args:
        start_date (str): Start date for historical data
        
    Returns:
        dict: Dictionary of downloaded data files by ticker
    """
    # Get S&P 500 tickers
    tickers = get_sp500_tickers()
    
    if not tickers:
        logger.error("Failed to get S&P 500 tickers")
        return {}
    
    # Download data for all tickers
    logger.info(f"Downloading historical data for {len(tickers)} S&P 500 stocks")
    results = download_sp500_with_retry(tickers, start_date)
    
    # Save metadata
    metadata = {
        'download_date': datetime.now().strftime('%Y-%m-%d'),
        'start_date': start_date,
        'tickers_count': len(results),
        'tickers': list(results.keys())
    }
    
    metadata_path = os.path.join("ml/data/raw", "metadata.csv")
    pd.DataFrame([metadata]).to_csv(metadata_path, index=False)
    
    logger.info(f"Download complete. Downloaded data for {len(results)} stocks.")
    return results

if __name__ == "__main__":
    # Example usage
    logger.info("Starting data acquisition")
    download_all_sp500_historical_data()