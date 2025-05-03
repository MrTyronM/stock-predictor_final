@extends('layouts.app')

@section('content')
<h1>Market Dashboard</h1>

<!-- Market Overview -->
<div class="card">
    <div class="card-header">
        <h5>Market Overview</h5>
    </div>
    <div class="card-body">
        <div class="market-index-value">
            {{ number_format($dashboardData['marketData']['marketIndex']['value'], 2) }}
            <span class="{{ $dashboardData['marketData']['marketIndex']['change'] >= 0 ? 'market-up' : 'market-down' }}">
                {{ $dashboardData['marketData']['marketIndex']['change'] >= 0 ? '▲' : '▼' }} 
                {{ number_format($dashboardData['marketData']['marketIndex']['change'], 2) }} 
                ({{ number_format($dashboardData['marketData']['marketIndex']['percentChange'], 2) }}%)
            </span>
        </div>
        
        <div class="dashboard-metrics">
            <div class="dashboard-metric">
                <div class="metric-value">
                    <span class="market-status {{ $dashboardData['marketData']['marketStatus'] == 'open' ? 'market-status-open' : 'market-status-closed' }}">
                        {{ strtoupper($dashboardData['marketData']['marketStatus']) }}
                    </span>
                </div>
                <div class="metric-label">Market Status</div>
            </div>
            
            <div class="dashboard-metric">
                <div class="metric-value">{{ number_format($dashboardData['marketData']['tradingVolume']) }}</div>
                <div class="metric-label">Trading Volume</div>
            </div>
            
            <div class="dashboard-metric">
                <div class="metric-value"><span class="market-up">{{ $dashboardData['marketData']['advancers'] }}</span></div>
                <div class="metric-label">Advancers</div>
            </div>
            
            <div class="dashboard-metric">
                <div class="metric-value"><span class="market-down">{{ $dashboardData['marketData']['decliners'] }}</span></div>
                <div class="metric-label">Decliners</div>
            </div>
            
            <div class="dashboard-metric">
                <div class="metric-value">{{ $dashboardData['marketData']['lastUpdated'] }}</div>
                <div class="metric-label">Last Updated</div>
            </div>
        </div>
    </div>
</div>

<div class="row">
    <div class="col-8">
        <!-- Most Active Stocks -->
        <div class="card">
            <div class="card-header">
                <h5>Most Active Stocks</h5>
            </div>
            <div class="card-body" style="padding: 0;">
                <div class="table-container">
                    <table>
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
                            @foreach($dashboardData['mostActiveStocks'] as $stock)
                            <tr class="stock-row">
                                <td><strong>{{ $stock['symbol'] }}</strong></td>
                                <td>{{ $stock['name'] }}</td>
                                <td>${{ number_format($stock['lastPrice'], 2) }}</td>
                                <td class="{{ $stock['change'] >= 0 ? 'market-up' : 'market-down' }}">
                                    {{ $stock['change'] >= 0 ? '▲' : '▼' }} 
                                    ${{ number_format(abs($stock['change']), 2) }} 
                                    ({{ number_format($stock['percentChange'], 2) }}%)
                                </td>
                                <td>{{ number_format($stock['volume']) }}</td>
                            </tr>
                            @endforeach
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    
    <div class="col-4">
        <!-- Latest Predictions -->
        <div class="card">
            <div class="card-header">
                <h5>Latest Market Predictions</h5>
            </div>
            <div class="card-body">
                @foreach($dashboardData['latestPredictions'] as $prediction)
                <div class="prediction-card">
                    <div class="prediction-date">{{ $prediction['date'] }}</div>
                    <div><strong>{{ $prediction['analyst'] }}</strong> on {{ $prediction['target'] }}</div>
                    <div class="prediction-price">${{ number_format($prediction['targetPrice']) }}</div>
                    <div>
                        <span class="prediction-badge {{ $prediction['prediction'] == 'Bullish' ? 'prediction-bullish' : ($prediction['prediction'] == 'Bearish' ? 'prediction-bearish' : 'prediction-neutral') }}">
                            {{ $prediction['prediction'] }}
                        </span>
                    </div>
                </div>
                @endforeach
            </div>
        </div>
    </div>
</div>
@endsection

@section('scripts')
<script>
    // Add any market dashboard specific JavaScript here
    document.addEventListener('DOMContentLoaded', function() {
        // Stock row clickable effect
        const stockRows = document.querySelectorAll('.stock-row');
        stockRows.forEach(row => {
            row.addEventListener('click', function() {
                const symbol = this.querySelector('td:first-child').textContent.trim();
                window.location.href = `/stocks/${symbol}`;
            });
        });
    });
</script>
@endsection