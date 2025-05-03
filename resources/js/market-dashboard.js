import React, { useState } from 'react';
import ReactDOM from 'react-dom';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import 'react-tabs/style/react-tabs.css';

const MarketDashboard = ({ data }) => {
    const marketData = data.marketData;
    const mostActiveStocks = data.mostActiveStocks;
    const latestPredictions = data.latestPredictions;
    
    // Prepare trend data for chart
    const trendData = marketData.trend.dates.map((date, index) => ({
        date: date,
        value: marketData.trend.values[index]
    }));
    
    // Prepare sector data for chart
    const sectorData = Object.entries(marketData.sectors).map(([sector, data]) => ({
        name: sector,
        value: data.average
    }));
    
    // Colors for charts
    const COLORS = ['#00e676', '#00bcd4', '#651fff', '#ff9100', '#ff3d00', '#76ff03'];
    
    return (
        <div className="market-dashboard">
            <Tabs>
                <TabList>
                    <Tab>Market Overview</Tab>
                    <Tab>Sectors</Tab>
                    <Tab>Top Movers</Tab>
                    <Tab>Trading Volume</Tab>
                    <Tab>Predictions</Tab>
                </TabList>
                
                <TabPanel>
                    <div className="dashboard-panel">
                        <h3>Market Trend</h3>
                        <div className="chart-container" style={{ height: '400px' }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart
                                    data={trendData}
                                    margin={{ top: 20, right: 30, left: 20, bottom: 50 }}
                                >
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis 
                                        dataKey="date" 
                                        angle={-45}
                                        textAnchor="end"
                                        tick={{ fontSize: 12 }}
                                        height={60}
                                    />
                                    <YAxis />
                                    <Tooltip />
                                    <Legend />
                                    <Line 
                                        type="monotone" 
                                        dataKey="value" 
                                        stroke="#00e676" 
                                        name="Market Index"
                                        activeDot={{ r: 8 }} 
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </TabPanel>
                
                <TabPanel>
                    <div className="dashboard-panel">
                        <h3>Sector Performance</h3>
                        <div className="chart-container" style={{ height: '400px' }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart
                                    data={sectorData}
                                    margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                                    layout="vertical"
                                >
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis type="number" />
                                    <YAxis dataKey="name" type="category" />
                                    <Tooltip />
                                    <Legend />
                                    <Bar 
                                        dataKey="value" 
                                        name="Performance (%)" 
                                        fill="#00e676" 
                                    />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </TabPanel>
                
                <TabPanel>
                    <div className="dashboard-panel">
                        <div className="row">
                            <div className="col-md-6">
                                <h3>Top Gainers</h3>
                                <table className="table table-dark">
                                    <thead>
                                        <tr>
                                            <th>Symbol</th>
                                            <th>Price</th>
                                            <th>Change</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {marketData.gainers.map((stock, index) => (
                                            <tr key={index}>
                                                <td>{stock.symbol}</td>
                                                <td>${stock.price.toFixed(2)}</td>
                                                <td className="text-success">
                                                    +{stock.change.toFixed(2)} (+{stock.change_percent.toFixed(2)}%)
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                            <div className="col-md-6">
                                <h3>Top Losers</h3>
                                <table className="table table-dark">
                                    <thead>
                                        <tr>
                                            <th>Symbol</th>
                                            <th>Price</th>
                                            <th>Change</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {marketData.losers.map((stock, index) => (
                                            <tr key={index}>
                                                <td>{stock.symbol}</td>
                                                <td>${stock.price.toFixed(2)}</td>
                                                <td className="text-danger">
                                                    {stock.change.toFixed(2)} ({stock.change_percent.toFixed(2)}%)
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </TabPanel>
                
                <TabPanel>
                    <div className="dashboard-panel">
                        <h3>Most Active Stocks</h3>
                        <table className="table table-dark">
                            <thead>
                                <tr>
                                    <th>Symbol</th>
                                    <th>Price</th>
                                    <th>Volume</th>
                                </tr>
                            </thead>
                            <tbody>
                                {mostActiveStocks.map((stock, index) => (
                                    <tr key={index}>
                                        <td>{stock.symbol}</td>
                                        <td>${stock.price.toFixed(2)}</td>
                                        <td>{stock.volume.toLocaleString()}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </TabPanel>
                
                <TabPanel>
                    <div className="dashboard-panel">
                        <h3>Top Predictions</h3>
                        <table className="table table-dark">
                            <thead>
                                <tr>
                                    <th>Symbol</th>
                                    <th>Current Price</th>
                                    <th>Predicted Price</th>
                                    <th>Change</th>
                                    <th>Recommendation</th>
                                    <th>Confidence</th>
                                </tr>
                            </thead>
                            <tbody>
                                {latestPredictions.map((prediction, index) => (
                                    <tr key={index}>
                                        <td>{prediction.symbol}</td>
                                        <td>${prediction.current_price.toFixed(2)}</td>
                                        <td>${prediction.predicted_price.toFixed(2)}</td>
                                        <td className={prediction.percent_change >= 0 ? "text-success" : "text-danger"}>
                                            {prediction.percent_change >= 0 ? "+" : ""}
                                            {prediction.percent_change.toFixed(2)}%
                                        </td>
                                        <td>
                                            <span className={`prediction-recommendation recommendation-${prediction.recommendation}`}>
                                                {prediction.recommendation.charAt(0).toUpperCase() + prediction.recommendation.slice(1)}
                                            </span>
                                        </td>
                                        <td>{prediction.confidence.toFixed(1)}%</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </TabPanel>
            </Tabs>
        </div>
    );
};

// Mount component
if (document.getElementById('market-dashboard')) {
    const element = document.getElementById('market-dashboard');
    const data = JSON.parse(element.getAttribute('data-market'));
    
    ReactDOM.render(<MarketDashboard data={data} />, element);
}