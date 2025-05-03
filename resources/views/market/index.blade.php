<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Market Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <!-- Your main CSS file should include the market dashboard styles -->
</head>
<body>
    <div class="container">
        <div id="market-dashboard-root"></div>
    </div>

    <!-- React and Babel -->
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/babel-standalone@6/babel.min.js"></script>

    <!-- Pass the data from Laravel to JavaScript -->
    <script>
        window.dashboardData = {!! $dashboardData !!};
    </script>

    <!-- React Component -->
    <script type="text/babel">
        // Market Overview Component
        function MarketOverview({ data }) {
            const { marketIndex, marketStatus, lastUpdated, tradingVolume, advancers, decliners } = data;
            const isPositive = marketIndex.change >= 0;
            
            return (
                <div className="card mb-3">
                    <div className="card-header">
                        <h5 className="card-title">Market Overview</h5>
                    </div>
                    <div className="card-body">
                        <div className="row">
                            <div className="col-md-6">
                                <div className="market-index-container">
                                    <div className="market-index-value">
                                        {marketIndex.value.toLocaleString()}
                                    </div>
                                    <div className={`market-index-change ${isPositive ? "market-up" : "market-down"}`}>
                                        {isPositive ? '▲' : '▼'} {marketIndex.change.toLocaleString()} ({marketIndex.percentChange.toFixed(2)}%)
                                    </div>
                                    <div className="mb-2">
                                        Status: <span className={`market-status ${marketStatus === 'open' ? 'market-status-open' : 'market-status-closed'}`}>
                                            {marketStatus.toUpperCase()}
                                        </span>
                                    </div>
                                    <div className="text-secondary">Last Updated: {lastUpdated}</div>
                                </div>
                            </div>
                            <div className="col-md-6">
                                <div className="dashboard-metrics">
                                    <div className="dashboard-metric">
                                        <div className="metric-value">{tradingVolume.toLocaleString()}</div>
                                        <div className="metric-label">Trading Volume</div>
                                    </div>
                                    <div className="dashboard-metric">
                                        <div className="metric-value text-success">{advancers}</div>
                                        <div className="metric-label">Advancers</div>
                                    </div>
                                    <div className="dashboard-metric">
                                        <div className="metric-value text-danger">{decliners}</div>
                                        <div className="metric-label">Decliners</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            );
        }

        // Most Active Stocks Component
        function MostActiveStocks({ stocks }) {
            return (
                <div className="card mb-3">
                    <div className="card-header">
                        <h5 className="card-title">Most Active Stocks</h5>
                    </div>
                    <div className="card-body p-0">
                        <div className="table-container">
                            <table className="table">
                                <thead>
                                    <tr>
                                        <th>Symbol</th>
                                        <th>Name</th>
                                        <th>Last</th>
                                        <th>Change</th>
                                        <th>Volume</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {stocks.map((stock, index) => (
                                        <tr key={index} className="stock-row">
                                            <td><strong>{stock.symbol}</strong></td>
                                            <td>{stock.name}</td>
                                            <td>${stock.lastPrice.toFixed(2)}</td>
                                            <td className={stock.change >= 0 ? "market-up" : "market-down"}>
                                                {stock.change >= 0 ? '▲' : '▼'} ${Math.abs(stock.change).toFixed(2)} ({stock.percentChange.toFixed(2)}%)
                                            </td>
                                            <td>{stock.volume.toLocaleString()}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            );
        }

        // Latest Predictions Component
        function LatestPredictions({ predictions }) {
            return (
                <div className="card mb-3">
                    <div className="card-header">
                        <h5 className="card-title">Latest Market Predictions</h5>
                    </div>
                    <div className="card-body p-0">
                        <div className="table-container">
                            <table className="table">
                                <thead>
                                    <tr>
                                        <th>Analyst</th>
                                        <th>Target</th>
                                        <th>Prediction</th>
                                        <th>Target Price</th>
                                        <th>Date</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {predictions.map((prediction, index) => {
                                        const predictionClass = 
                                            prediction.prediction === 'Bullish' ? 'prediction-bullish' :
                                            prediction.prediction === 'Bearish' ? 'prediction-bearish' : 
                                            'prediction-neutral';
                                        
                                        return (
                                            <tr key={index} className="stock-row">
                                                <td><strong>{prediction.analyst}</strong></td>
                                                <td>{prediction.target}</td>
                                                <td>
                                                    <span className={`prediction-badge ${predictionClass}`}>
                                                        {prediction.prediction}
                                                    </span>
                                                </td>
                                                <td>{prediction.targetPrice.toLocaleString()}</td>
                                                <td>{prediction.date}</td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            );
        }

        // Main Dashboard Component
        function MarketDashboard({ data }) {
            return (
                <div>
                    <h1 className="mb-4">Market Dashboard</h1>
                    
                    <MarketOverview data={data.marketData} />
                    
                    <div className="row">
                        <div className="col-md-8">
                            <MostActiveStocks stocks={data.mostActiveStocks} />
                        </div>
                        <div className="col-md-4">
                            <LatestPredictions predictions={data.latestPredictions} />
                        </div>
                    </div>
                </div>
            );
        }

        // Render the main component
        const root = ReactDOM.createRoot(document.getElementById('market-dashboard-root'));
        root.render(<MarketDashboard data={window.dashboardData} />);
    </script>
</body>
</html>