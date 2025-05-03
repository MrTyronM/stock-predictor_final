/**
 * API Connection module for the AI Stock Market Prediction Tool
 * This module handles all communication with the backend API
 */

class StockPredictorAPI {
    constructor(baseUrl = '/api') {
        this.baseUrl = baseUrl;
        this.headers = {
            'Content-Type': 'application/json'
        };
    }

    /**
     * Get available tickers
     * @param {string} source - Source of tickers ('available' or 'sp500')
     * @returns {Promise} - Promise resolving to API response
     */
    async getTickers(source = 'available') {
        try {
            const response = await fetch(`${this.baseUrl}/tickers?source=${source}`);
            
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            
            return response.json();
        } catch (error) {
            console.error('Error fetching tickers:', error);
            throw error;
        }
    }

    /**
     * Get stock data for a specific ticker
     * @param {string} ticker - Stock ticker symbol
     * @param {number} days - Number of days of data to retrieve
     * @param {boolean} includeIndicators - Whether to include technical indicators
     * @returns {Promise} - Promise resolving to API response
     */
    async getStockData(ticker, days = 100, includeIndicators = false) {
        try {
            const response = await fetch(
                `${this.baseUrl}/stock/${ticker}?days=${days}&indicators=${includeIndicators}`
            );
            
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            
            return response.json();
        } catch (error) {
            console.error(`Error fetching data for ${ticker}:`, error);
            throw error;
        }
    }

    /**
     * Get prediction for a specific ticker
     * @param {string} ticker - Stock ticker symbol
     * @param {Object} options - Prediction options
     * @returns {Promise} - Promise resolving to API response
     */
    async getPrediction(ticker, options = {}) {
        try {
            const { modelType, task, days, threshold, visualize } = options;
            
            let url = `${this.baseUrl}/predict/${ticker}?`;
            
            if (modelType) url += `&model_type=${modelType}`;
            if (task) url += `&task=${task}`;
            if (days) url += `&days=${days}`;
            if (threshold) url += `&threshold=${threshold}`;
            if (visualize !== undefined) url += `&visualize=${visualize}`;
            
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            
            return response.json();
        } catch (error) {
            console.error(`Error getting prediction for ${ticker}:`, error);
            throw error;
        }
    }

    /**
     * Get latest predictions for all stocks
     * @param {Object} filters - Filter options
     * @returns {Promise} - Promise resolving to API response
     */
    async getPredictions(filters = {}) {
        try {
            const { limit, signal, confidence } = filters;
            
            let url = `${this.baseUrl}/predictions?`;
            
            if (limit) url += `&limit=${limit}`;
            if (signal) url += `&signal=${signal}`;
            if (confidence) url += `&confidence=${confidence}`;
            
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            
            return response.json();
        } catch (error) {
            console.error('Error fetching predictions:', error);
            throw error;
        }
    }

    /**
     * Run backtest for a strategy
     * @param {Object} backtestParams - Backtest parameters
     * @returns {Promise} - Promise resolving to API response
     */
    async runBacktest(backtestParams) {
        try {
            const response = await fetch(`${this.baseUrl}/backtest`, {
                method: 'POST',
                headers: this.headers,
                body: JSON.stringify(backtestParams)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            
            return response.json();
        } catch (error) {
            console.error('Error running backtest:', error);
            throw error;
        }
    }

    /**
     * Get configuration
     * @param {string} section - Configuration section
     * @param {string} key - Configuration key
     * @returns {Promise} - Promise resolving to API response
     */
    async getConfig(section, key) {
        try {
            let url = `${this.baseUrl}/config`;
            
            if (section) url += `?section=${section}`;
            if (key) url += `&key=${key}`;
            
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            
            return response.json();
        } catch (error) {
            console.error('Error fetching configuration:', error);
            throw error;
        }
    }

    /**
     * Update configuration
     * @param {Object} configData - Configuration data
     * @param {string} section - Configuration section
     * @returns {Promise} - Promise resolving to API response
     */
    async updateConfig(configData, section) {
        try {
            let url = `${this.baseUrl}/config`;
            
            if (section) url += `?section=${section}`;
            
            const response = await fetch(url, {
                method: 'PUT',
                headers: this.headers,
                body: JSON.stringify(configData)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            
            return response.json();
        } catch (error) {
            console.error('Error updating configuration:', error);
            throw error;
        }
    }

    /**
     * Create backup
     * @param {Object} options - Backup options
     * @returns {Promise} - Promise resolving to API response
     */
    async createBackup(options = {}) {
        try {
            const response = await fetch(`${this.baseUrl}/backup`, {
                method: 'POST',
                headers: this.headers,
                body: JSON.stringify(options)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            
            return response.json();
        } catch (error) {
            console.error('Error creating backup:', error);
            throw error;
        }
    }

    /**
     * Restore from backup
     * @param {Object} options - Restore options
     * @returns {Promise} - Promise resolving to API response
     */
    async restoreFromBackup(options) {
        try {
            const response = await fetch(`${this.baseUrl}/restore`, {
                method: 'POST',
                headers: this.headers,
                body: JSON.stringify(options)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            
            return response.json();
        } catch (error) {
            console.error('Error restoring from backup:', error);
            throw error;
        }
    }

    /**
     * Get system status
     * @returns {Promise} - Promise resolving to API response
     */
    async getStatus() {
        try {
            const response = await fetch(`${this.baseUrl}/status`);
            
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            
            return response.json();
        } catch (error) {
            console.error('Error fetching system status:', error);
            throw error;
        }
    }

    /**
     * Start model training
     * @param {Object} trainingParams - Training parameters
     * @returns {Promise} - Promise resolving to API response
     */
    async startTraining(trainingParams) {
        try {
            const response = await fetch(`${this.baseUrl}/training`, {
                method: 'POST',
                headers: this.headers,
                body: JSON.stringify(trainingParams)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            
            return response.json();
        } catch (error) {
            console.error('Error starting training:', error);
            throw error;
        }
    }

    /**
     * Get model performance metrics
     * @param {string} ticker - Stock ticker symbol
     * @returns {Promise} - Promise resolving to API response
     */
    async getMetrics(ticker) {
        try {
            let url = `${this.baseUrl}/metrics`;
            
            if (ticker) url += `?ticker=${ticker}`;
            
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            
            return response.json();
        } catch (error) {
            console.error('Error fetching metrics:', error);
            throw error;
        }
    }
}

// Create global API instance
const api = new StockPredictorAPI();

// Export for use in other scripts
window.StockPredictorAPI = StockPredictorAPI;
window.api = api;