/**
 * Main JavaScript file for the AI Stock Market Prediction Tool
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initApp();
});

/**
 * Initialize the application
 */
function initApp() {
    // Set up navigation
    setupNavigation();

    // Initialize charts
    initCharts();

    // Set up event listeners
    setupEventListeners();

    // Update dynamic content
    updateDynamicContent();
}

/**
 * Set up navigation between pages
 */
function setupNavigation() {
    const navLinks = document.querySelectorAll('.main-nav a');
    const pages = document.querySelectorAll('.page');
    const viewAllLinks = document.querySelectorAll('.view-all');

    // Main navigation
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all links
            navLinks.forEach(l => l.classList.remove('active'));
            
            // Add active class to clicked link
            this.classList.add('active');
            
            // Get the page ID from data attribute
            const pageId = this.getAttribute('data-page');
            
            // Hide all pages
            pages.forEach(page => page.classList.remove('active'));
            
            // Show the selected page
            document.getElementById(pageId).classList.add('active');
        });
    });

    // "View All" links
    viewAllLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Get the target page from data attribute
            const pageId = this.getAttribute('data-page');
            
            // Remove active class from all links
            navLinks.forEach(l => l.classList.remove('active'));
            
            // Add active class to corresponding nav link
            document.querySelector(`.main-nav a[data-page="${pageId}"]`).classList.add('active');
            
            // Hide all pages
            pages.forEach(page => page.classList.remove('active'));
            
            // Show the target page
            document.getElementById(pageId).classList.add('active');
        });
    });
}

/**
 * Initialize charts using Chart.js
 */
function initCharts() {
    // Dashboard Buy Signals Chart
    const buySignalsCtx = document.getElementById('buy-signals-chart').getContext('2d');
    const buySignalsChart = new Chart(buySignalsCtx, {
        type: 'bar',
        data: {
            labels: ['AAPL', 'MSFT', 'NVDA', 'META', 'AMZN', 'AMD', 'PYPL', 'V'],
            datasets: [{
                label: 'Confidence (%)',
                data: [85, 78, 88, 70, 65, 62, 75, 68],
                backgroundColor: '#4CAF50',
                borderColor: '#388E3C',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Confidence: ${context.raw}%`;
                        }
                    }
                }
            }
        }
    });

    // Dashboard Sell Signals Chart
    const sellSignalsCtx = document.getElementById('sell-signals-chart').getContext('2d');
    const sellSignalsChart = new Chart(sellSignalsCtx, {
        type: 'bar',
        data: {
            labels: ['TSLA', 'AMD', 'NFLX', 'DIS', 'BA', 'GE', 'F', 'XOM'],
            datasets: [{
                label: 'Confidence (%)',
                data: [92, 75, 82, 68, 73, 65, 60, 58],
                backgroundColor: '#F44336',
                borderColor: '#D32F2F',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Confidence: ${context.raw}%`;
                        }
                    }
                }
            }
        }
    });

    // Stock Price Chart in Analysis Page
    if (document.getElementById('stock-chart')) {
        const stockChartCtx = document.getElementById('stock-chart').getContext('2d');
        const stockPriceChart = new Chart(stockChartCtx, {
            type: 'line',
            data: {
                labels: [
                    '2025-03-24', '2025-03-25', '2025-03-26', '2025-03-27', '2025-03-28',
                    '2025-03-31', '2025-04-01', '2025-04-02', '2025-04-03', '2025-04-04',
                    '2025-04-07', '2025-04-08', '2025-04-09', '2025-04-10', '2025-04-11',
                    '2025-04-14', '2025-04-15', '2025-04-16', '2025-04-17', '2025-04-18',
                    '2025-04-21', '2025-04-22'
                ],
                datasets: [
                    {
                        label: 'Price',
                        data: [
                            181.20, 182.50, 185.20, 184.80, 183.70,
                            186.30, 190.10, 187.40, 185.90, 184.20,
                            182.30, 185.70, 187.90, 188.10, 187.50,
                            188.75, 189.20, 190.10, 189.70, 188.90,
                            190.10, 192.45
                        ],
                        borderColor: '#2196F3',
                        backgroundColor: 'rgba(33, 150, 243, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1
                    },
                    {
                        label: 'Predicted Trend',
                        data: [
                            null, null, null, null, null,
                            null, null, null, null, null,
                            null, null, null, null, null,
                            null, null, null, null, null,
                            190.10, 192.45, 196.30, 198.70, 203.40, 205.80
                        ],
                        borderColor: '#9C27B0',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.1
                    }
                ]
            },
            options: {
                scales: {
                    y: {
                        ticks: {
                            callback: function(value) {
                                return '$' + value;
                            }
                        }
                    },
                    x: {
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: $${context.raw}`;
                            }
                        }
                    }
                }
            }
        });
    }

    // Modal Chart
    if (document.getElementById('modal-chart')) {
        const modalChartCtx = document.getElementById('modal-chart').getContext('2d');
        const modalChart = new Chart(modalChartCtx, {
            type: 'line',
            data: {
                labels: ['2025-04-15', '2025-04-16', '2025-04-17', '2025-04-18', '2025-04-21', '2025-04-22'],
                datasets: [
                    {
                        label: 'Price',
                        data: [189.20, 190.10, 189.70, 188.90, 190.10, 192.45],
                        borderColor: '#2196F3',
                        backgroundColor: 'rgba(33, 150, 243, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1
                    },
                    {
                        label: 'Prediction',
                        data: [null, null, null, null, 190.10, 192.45, 196.30, 201.70, 205.80],
                        borderColor: '#9C27B0',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.1
                    }
                ]
            },
            options: {
                scales: {
                    y: {
                        ticks: {
                            callback: function(value) {
                                return '$' + value;
                            }
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: $${context.raw}`;
                            }
                        }
                    }
                }
            }
        });
    }
}

/**
 * Set up event listeners for interactive elements
 */
function setupEventListeners() {
    // Chart period buttons
    const periodButtons = document.querySelectorAll('.period-btn');
    periodButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all period buttons
            periodButtons.forEach(btn => btn.classList.remove('active'));
            
            // Add active class to clicked button
            this.classList.add('active');
            
            // Update chart data (would fetch real data based on period)
            // For demo, we'll just log the selected period
            console.log('Selected period:', this.textContent);
        });
    });

    // Chart tabs
    const chartTabs = document.querySelectorAll('.chart-tabs .tab-btn');
    chartTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            // Remove active class from all tabs
            chartTabs.forEach(t => t.classList.remove('active'));
            
            // Add active class to clicked tab
            this.classList.add('active');
            
            // Get the tab ID from data attribute
            const tabId = this.getAttribute('data-tab');
            
            // Update chart content (would show different chart based on tab)
            console.log('Selected tab:', tabId);
        });
    });

    // View buttons in prediction list
    const viewButtons = document.querySelectorAll('.btn-view');
    const stockModal = document.getElementById('stock-modal');
    const closeModalButtons = document.querySelectorAll('.close-modal, .btn-close');
    
    viewButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Show modal
            stockModal.style.display = 'block';
            
            // Get ticker from row (for real app, would fetch details based on ticker)
            const ticker = this.closest('tr').querySelector('td:first-child').textContent;
            console.log('Viewing details for:', ticker);
        });
    });
    
    // Close modal buttons
    closeModalButtons.forEach(button => {
        button.addEventListener('click', function() {
            stockModal.style.display = 'none';
        });
    });
    
    // Close modal when clicking outside
    window.addEventListener('click', function(e) {
        if (e.target === stockModal) {
            stockModal.style.display = 'none';
        }
    });

    // Analyze button in Stock Analysis page
    const analyzeButton = document.getElementById('analyze-btn');
    if (analyzeButton) {
        analyzeButton.addEventListener('click', function() {
            const ticker = document.getElementById('analysis-ticker').value.toUpperCase();
            if (ticker) {
                // Would fetch and display data for the specified ticker
                console.log('Analyzing ticker:', ticker);
                
                // For demo, update the title
                const stockTitle = document.querySelector('.stock-title h3');
                if (stockTitle) {
                    stockTitle.textContent = `${ticker} - Sample Company`;
                }
            }
        });
    }

    // Range slider in Settings
    const confidenceThreshold = document.getElementById('confidence-threshold');
    if (confidenceThreshold) {
        confidenceThreshold.addEventListener('input', function() {
            const rangeValue = this.parentElement.querySelector('.range-value');
            rangeValue.textContent = this.value + '%';
        });
    }

    // Save buttons in Settings forms
    const saveForms = document.querySelectorAll('.settings-form');
    saveForms.forEach(form => {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Would save settings to backend
            console.log('Saving settings from form:', this);
            
            // Show success message
            showNotification('Settings saved successfully!', 'success');
        });
    });

    // Timeframe selectors for dashboard charts
    const timeframeSelectors = document.querySelectorAll('#buy-signals-timeframe, #sell-signals-timeframe');
    timeframeSelectors.forEach(selector => {
        selector.addEventListener('change', function() {
            // Would update chart based on selected timeframe
            console.log('Changed timeframe to:', this.value, 'for', this.id);
        });
    });

    // Filter controls in Predictions page
    const filterControls = document.querySelectorAll('#signal-filter, #confidence-filter, #date-filter');
    filterControls.forEach(control => {
        control.addEventListener('change', function() {
            // Would filter prediction table based on selections
            console.log('Filter updated:', this.id, 'to', this.value);
        });
    });

    // Search box in Predictions page
    const searchBox = document.getElementById('ticker-search');
    if (searchBox) {
        searchBox.addEventListener('input', function() {
            // Would filter table based on search input
            console.log('Searching for:', this.value);
        });
    }

    // Pagination buttons
    const prevButton = document.querySelector('.btn-prev');
    const nextButton = document.querySelector('.btn-next');
    
    if (prevButton && nextButton) {
        prevButton.addEventListener('click', function() {
            if (!this.disabled) {
                // Would load previous page of results
                console.log('Loading previous page');
            }
        });
        
        nextButton.addEventListener('click', function() {
            // Would load next page of results
            console.log('Loading next page');
        });
    }
}

/**
 * Update dynamic content (date, stats, etc.)
 */
function updateDynamicContent() {
    // Update current date
    const currentDateElements = document.querySelectorAll('.current-date');
    const currentDate = new Date();
    const dateOptions = { year: 'numeric', month: 'long', day: 'numeric' };
    const formattedDate = currentDate.toLocaleDateString('en-US', dateOptions);
    
    currentDateElements.forEach(element => {
        element.textContent = formattedDate;
    });

    // Set up refresh buttons
    const refreshButtons = document.querySelectorAll('.refresh-btn, .btn-refresh');
    refreshButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Show loading state
            this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...';
            
            // Simulate refresh (would fetch new data)
            setTimeout(() => {
                this.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh';
                showNotification('Data refreshed successfully!', 'success');
            }, 1500);
        });
    });
}

/**
 * Show a notification message
 * 
 * @param {string} message - The message to display
 * @param {string} type - The type of notification ('success', 'error', 'warning')
 */
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    
    // Add icon based on type
    let icon = 'info-circle';
    if (type === 'success') icon = 'check-circle';
    if (type === 'error') icon = 'exclamation-circle';
    if (type === 'warning') icon = 'exclamation-triangle';
    
    notification.innerHTML = `
        <i class="fas fa-${icon}"></i>
        <span>${message}</span>
        <button class="close-notification"><i class="fas fa-times"></i></button>
    `;
    
    // Add to document
    if (!document.querySelector('.notifications-container')) {
        const container = document.createElement('div');
        container.className = 'notifications-container';
        document.body.appendChild(container);
    }
    
    document.querySelector('.notifications-container').appendChild(notification);
    
    // Add close button functionality
    notification.querySelector('.close-notification').addEventListener('click', function() {
        notification.remove();
    });
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        notification.classList.add('fade-out');
        setTimeout(() => {
            notification.remove();
        }, 500);
    }, 5000);
}

/**
 * Fetch data from backend API
 * 
 * @param {string} endpoint - API endpoint
 * @param {Object} params - Query parameters
 * @returns {Promise} - Promise resolving to API response
 */
function fetchData(endpoint, params = {}) {
    // Build query string
    const queryString = Object.keys(params)
        .map(key => `${encodeURIComponent(key)}=${encodeURIComponent(params[key])}`)
        .join('&');
    
    const url = `/api/${endpoint}${queryString ? '?' + queryString : ''}`;
    
    return fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            return response.json();
        })
        .catch(error => {
            console.error('API request failed:', error);
            showNotification('Failed to fetch data from server.', 'error');
            throw error;
        });
}

/**
 * Send data to backend API
 * 
 * @param {string} endpoint - API endpoint
 * @param {Object} data - Data to send
 * @returns {Promise} - Promise resolving to API response
 */
function sendData(endpoint, data) {
    return fetch(`/api/${endpoint}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        return response.json();
    })
    .catch(error => {
        console.error('API request failed:', error);
        showNotification('Failed to send data to server.', 'error');
        throw error;
    });
}