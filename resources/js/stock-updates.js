document.addEventListener('DOMContentLoaded', function() {
    // Initialize WebSocket connection
    const socket = new WebSocket(`ws://${window.location.hostname}:6001/app/stock-updates`);
    
    socket.onopen = function(e) {
        console.log('WebSocket connection established');
    };
    
    socket.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            
            if (data.event === 'price-updated') {
                updateStockPrice(data.data);
            }
        } catch (error) {
            console.error('Error processing WebSocket message:', error);
        }
    };
    
    socket.onerror = function(error) {
        console.error('WebSocket error:', error);
    };
    
    socket.onclose = function(event) {
        console.log('WebSocket connection closed');
    };
    
    // Update stock price elements in the DOM
    function updateStockPrice(data) {
        // Find all elements with the stock symbol class
        const elements = document.querySelectorAll(`.stock-price-${data.symbol}`);
        
        elements.forEach(element => {
            // Update price
            element.textContent = `$${parseFloat(data.price).toFixed(2)}`;
            
            // Update change elements if they exist
            const changeElement = element.parentNode.querySelector(`.stock-change-${data.symbol}`);
            if (changeElement) {
                const changeText = data.change >= 0 
                    ? `+$${parseFloat(data.change).toFixed(2)} (+${parseFloat(data.change_percent).toFixed(2)}%)`
                    : `-$${Math.abs(parseFloat(data.change)).toFixed(2)} (${parseFloat(data.change_percent).toFixed(2)}%)`;
                
                changeElement.textContent = changeText;
                
                // Update classes for styling
                changeElement.classList.remove('text-success', 'text-danger');
                changeElement.classList.add(data.change >= 0 ? 'text-success' : 'text-danger');
            }
        });
    }
});