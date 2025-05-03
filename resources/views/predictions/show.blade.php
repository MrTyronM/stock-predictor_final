@extends('layouts.app')

@section('content')
<div class="prediction-container">
    <h1 class="page-title">Predictions for {{ $stock->name }} ({{ $stock->symbol }})</h1>

    <div class="card mb-4">
        <div class="card-header">
            <h2 class="card-title">Price Predictions</h2>
        </div>
        <div class="card-body">
            <!-- Chart container -->
            <div class="chart-container mb-4">
                <canvas id="predictionChart"></canvas>
            </div>
            
            <!-- Prediction cards -->
            <div class="prediction-cards">
                @forelse($predictions as $prediction)
                    @php
                        $recommendation = strtolower($prediction->recommendation);
                    @endphp
                    <div class="prediction-card recommendation-{{ $recommendation }}">
                        <div class="prediction-date">{{ $prediction->prediction_date->format('Y-m-d') }}</div>
                        <div class="prediction-price">${{ number_format($prediction->predicted_price, 2) }}</div>
                        <div class="prediction-recommendation">{{ ucfirst($recommendation) }}</div>
                    </div>
                @empty
                    <div class="alert alert-info">No predictions available for this stock.</div>
                @endforelse
            </div>
        </div>
    </div>
    
    <div class="card">
        <div class="card-header">
            <h2 class="card-title">Analysis</h2>
        </div>
        <div class="card-body">
            <div class="row">
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="analysis-card">
                        <div class="analysis-title">Confidence</div>
                        <div class="analysis-value">{{ number_format($latestPrediction->confidence ?? 0, 1) }}%</div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="analysis-card">
                        <div class="analysis-title">Volatility</div>
                        <div class="analysis-value">{{ number_format($volatility ?? 0, 1) }}%</div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="analysis-card">
                        <div class="analysis-title">Accuracy</div>
                        <div class="analysis-value">{{ number_format($accuracy ?? 0, 1) }}%</div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="analysis-card">
                        <div class="analysis-title">Risk Level</div>
                        <div class="analysis-value">{{ $riskLevel ?? 'Medium' }}</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<style>
    .prediction-container {
        margin-bottom: 2rem;
    }
    
    .page-title {
        margin-bottom: 1.5rem;
        color: var(--accent-color);
    }
    
    .chart-container {
        height: 350px;
        width: 100%;
    }
    
    .prediction-cards {
        display: flex;
        gap: 1rem;
        overflow-x: auto;
        padding-bottom: 0.5rem;
    }
    
    .prediction-card {
        flex: 0 0 160px;
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border-top: 3px solid #ffab40; /* Default is hold - amber/orange */
    }
    
    .prediction-card.recommendation-buy {
        border-top-color: #00e676; /* Green for buy */
    }
    
    .prediction-card.recommendation-sell {
        border-top-color: #ff5252; /* Red for sell */
    }
    
    .prediction-date {
        font-size: 0.9rem;
        color: #aaa;
        margin-bottom: 0.5rem;
    }
    
    .prediction-price {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .prediction-recommendation {
        font-size: 0.9rem;
        font-weight: 500;
        color: #ffab40; /* Default is hold - amber/orange */
    }
    
    .recommendation-buy {
        color: #00e676; /* Green for buy */
    }
    
    .recommendation-sell {
        color: #ff5252; /* Red for sell */
    }
    
    .analysis-card {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 1rem;
        height: 100%;
        text-align: center;
    }
    
    .analysis-title {
        font-size: 0.9rem;
        color: #aaa;
        margin-bottom: 0.5rem;
    }
    
    .analysis-value {
        font-size: 1.5rem;
        font-weight: 700;
    }
</style>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Extract data from prediction cards
    const predictionCards = document.querySelectorAll('.prediction-card');
    const dates = [];
    const prices = [];
    const recommendations = [];
    
    predictionCards.forEach(card => {
        const dateText = card.querySelector('.prediction-date').textContent;
        const priceText = card.querySelector('.prediction-price').textContent;
        const recommendationText = card.querySelector('.prediction-recommendation').textContent;
        
        dates.push(dateText);
        prices.push(parseFloat(priceText.replace(/[$,]/g, '')));
        recommendations.push(recommendationText);
    });
    
    // If no data found, show message in chart area
    if (dates.length === 0) {
        const chartContainer = document.querySelector('.chart-container');
        chartContainer.innerHTML = '<div class="no-data-message">No prediction data available</div>';
        return;
    }
    
    // Create point colors based on recommendations
    const pointColors = recommendations.map(r => {
        if (r.toLowerCase().includes('buy')) return '#00e676';  // Green
        if (r.toLowerCase().includes('sell')) return '#ff5252'; // Red
        return '#ffab40';  // Amber/Orange for Hold
    });
    
    // Set up chart
    const ctx = document.getElementById('predictionChart').getContext('2d');
    const priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Price Prediction ($)',
                data: prices,
                backgroundColor: 'rgba(0, 230, 118, 0.1)',
                borderColor: '#00e676',
                borderWidth: 2,
                tension: 0.3,
                pointBackgroundColor: pointColors,
                pointBorderColor: '#333',
                pointRadius: 6,
                pointHoverRadius: 8,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#aaa'
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#aaa',
                        callback: function(value) {
                            return '$' + value;
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: '#1e1e1e',
                    titleColor: '#fff',
                    bodyColor: '#aaa',
                    borderColor: '#333',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            const index = context.dataIndex;
                            return [
                                'Price: $' + context.raw,
                                'Action: ' + recommendations[index]
                            ];
                        }
                    }
                }
            }
        }
    });
});
</script>
@endsection