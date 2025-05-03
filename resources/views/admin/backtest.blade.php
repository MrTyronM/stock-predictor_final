@extends('layouts.app')

@section('content')
    <div class="mb-3">
        <a href="{{ route('admin.dashboard') }}" class="btn btn-secondary">&larr; Back to Dashboard</a>
    </div>

    <h2 class="mb-3">Model Backtesting Tool</h2>
    
    <div class="card mb-4">
        <div class="card-header">
            <h3 class="card-title">Select Stock and Backtest Period</h3>
        </div>
        <div class="card-body">
            <form action="{{ route('admin.backtest.run') }}" method="POST">
                @csrf
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="form-group">
                            <label for="stock_id" class="form-label">Select Stock</label>
                            <select id="stock_id" name="stock_id" class="form-control" required>
                                <option value="">-- Select a stock --</option>
                                @foreach($stocks as $stock)
                                    <option value="{{ $stock->id }}" {{ $selectedStockId == $stock->id ? 'selected' : '' }}>
                                        {{ $stock->symbol }} - {{ $stock->name }}
                                    </option>
                                @endforeach
                            </select>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="form-group">
                            <label for="period" class="form-label">Backtest Period (Days)</label>
                            <select id="period" name="period" class="form-control">
                                <option value="30" {{ $period == 30 ? 'selected' : '' }}>30 Days</option>
                                <option value="60" {{ $period == 60 ? 'selected' : '' }}>60 Days</option>
                                <option value="90" {{ $period == 90 ? 'selected' : '' }}>90 Days</option>
                                <option value="180" {{ $period == 180 ? 'selected' : '' }}>180 Days</option>
                                <option value="365" {{ $period == 365 ? 'selected' : '' }}>365 Days</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div class="mt-3">
                    <button type="submit" class="btn btn-primary">Run Backtest</button>
                </div>
                
                <div class="mt-3">
                    <p class="text-secondary"><small>Note: Backtesting runs a simulation of how the model would have performed historically. This helps evaluate the accuracy and trading performance of the prediction model.</small></p>
                </div>
            </form>
        </div>
    </div>
    
    @if($backtestResults)
        @if(isset($backtestResults['error']))
            <div class="alert alert-danger">
                {{ $backtestResults['error'] }}
            </div>
        @else
            <div class="card mb-4">
                <div class="card-header">
                    <h3 class="card-title">Backtest Results for {{ $backtestResults['stock_symbol'] }}</h3>
                </div>
                <div class="card-body">
                    <div class="row mb-4">
                        <div class="col-md-3">
                            <div class="card" style="background-color: var(--bg-tertiary);">
                                <div class="card-body text-center">
                                    <h4>Direction Accuracy</h4>
                                    <h2 class="text-accent">{{ number_format($backtestResults['test_metrics']['direction_accuracy'], 1) }}%</h2>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card" style="background-color: var(--bg-tertiary);">
                                <div class="card-body text-center">
                                    <h4>Mean Error</h4>
                                    <h2 class="text-accent">${{ number_format($backtestResults['test_metrics']['mae'], 2) }}</h2>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card" style="background-color: var(--bg-tertiary);">
                                <div class="card-body text-center">
                                    <h4>Trading Return</h4>
                                    <h2 class="{{ $backtestResults['trading_performance']['total_return'] >= 0 ? 'text-success' : 'text-danger' }}">
                                        {{ number_format($backtestResults['trading_performance']['total_return'], 1) }}%
                                    </h2>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card" style="background-color: var(--bg-tertiary);">
                                <div class="card-body text-center">
                                    <h4>Number of Trades</h4>
                                    <h2 class="text-accent">{{ $backtestResults['trading_performance']['num_trades'] }}</h2>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div id="backtest-chart" style="height: 400px;"></div>
                    
                    <h4 class="mt-4 mb-3">Trading Signals</h4>
                    <div class="table-container">
                        <table class="table">
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>Actual Price</th>
                                    <th>Predicted Price</th>
                                    <th>Signal</th>
                                </tr>
                            </thead>
                            <tbody>
                                @foreach($backtestResults['signals'] as $signal)
                                    <tr>
                                        <td>{{ $signal['date'] }}</td>
                                        <td>${{ number_format($signal['actual_price'], 2) }}</td>
                                        <td>${{ number_format($signal['predicted_price'], 2) }}</td>
                                        <td>
                                            <span class="prediction-recommendation recommendation-{{ $signal['signal'] }}">
                                                {{ ucfirst($signal['signal']) }}
                                            </span>
                                        </td>
                                    </tr>
                                @endforeach
                            </tbody>
                        </table>
                    </div>
                    
                    @if(count($backtestResults['trades']) > 0)
                        <h4 class="mt-4 mb-3">Trades</h4>
                        <div class="table-container">
                            <table class="table">
                                <thead>
                                    <tr>
                                        <th>Buy Date</th>
                                        <th>Buy Price</th>
                                        <th>Sell Date</th>
                                        <th>Sell Price</th>
                                        <th>Profit/Loss</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    @foreach($backtestResults['trades'] as $trade)
                                        <tr>
                                            <td>{{ $trade['buy_date'] }}</td>
                                            <td>${{ number_format($trade['buy_price'], 2) }}</td>
                                            <td>{{ $trade['sell_date'] }}</td>
                                            <td>${{ number_format($trade['sell_price'], 2) }}</td>
                                            <td class="{{ $trade['profit'] >= 0 ? 'text-success' : 'text-danger' }}">
                                                ${{ number_format($trade['profit'], 2) }} ({{ number_format($trade['profit_percent'], 1) }}%)
                                            </td>
                                        </tr>
                                    @endforeach
                                </tbody>
                            </table>
                        </div>
                    @endif
                </div>
            </div>
            
            <div class="card mb-4">
                <div class="card-header">
                    <h3 class="card-title">Model Performance Metrics</h3>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <h4>Training Set</h4>
                            <div class="table-container">
                                <table class="table">
                                    <tbody>
                                        <tr>
                                            <td>Mean Absolute Error (MAE)</td>
                                            <td>${{ number_format($backtestResults['train_metrics']['mae'], 4) }}</td>
                                        </tr>
                                        <tr>
                                            <td>Root Mean Squared Error (RMSE)</td>
                                            <td>${{ number_format($backtestResults['train_metrics']['rmse'], 4) }}</td>
                                        </tr>
                                        <tr>
                                            <td>Mean Absolute Percentage Error (MAPE)</td>
                                            <td>{{ number_format($backtestResults['train_metrics']['mape'], 2) }}%</td>
                                        </tr>
                                        <tr>
                                            <td>Direction Accuracy</td>
                                            <td>{{ number_format($backtestResults['train_metrics']['direction_accuracy'], 2) }}%</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <h4>Test Set</h4>
                            <div class="table-container">
                                <table class="table">
                                    <tbody>
                                        <tr>
                                            <td>Mean Absolute Error (MAE)</td>
                                            <td>${{ number_format($backtestResults['test_metrics']['mae'], 4) }}</td>
                                        </tr>
                                        <tr>
                                            <td>Root Mean Squared Error (RMSE)</td>
                                            <td>${{ number_format($backtestResults['test_metrics']['rmse'], 4) }}</td>
                                        </tr>
                                        <tr>
                                            <td>Mean Absolute Percentage Error (MAPE)</td>
                                            <td>{{ number_format($backtestResults['test_metrics']['mape'], 2) }}%</td>
                                        </tr>
                                        <tr>
                                            <td>Direction Accuracy</td>
                                            <td>{{ number_format($backtestResults['test_metrics']['direction_accuracy'], 2) }}%</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        @endif
    @endif

    @if($backtestResults && !isset($backtestResults['error']))
        <script>
            document.addEventListener("DOMContentLoaded", function() {
                const testDates = @json($backtestResults['test_data']['dates']);
                const actualPrices = @json($backtestResults['test_data']['actual_prices']);
                const predictedPrices = @json($backtestResults['test_data']['predicted_prices']);
                const signals = @json($backtestResults['signals']);
                
                const seriesData = [
                    {
                        name: 'Actual Price',
                        type: 'line',
                        data: actualPrices.map((price, i) => ({
                            x: new Date(testDates[i]).getTime(),
                            y: price
                        }))
                    },
                    {
                        name: 'Predicted Price',
                        type: 'line',
                        data: predictedPrices.map((price, i) => ({
                            x: new Date(testDates[i]).getTime(),
                            y: price
                        }))
                    }
                ];
                
                // Add buy and sell signals
                const buySignals = signals.filter(s => s.signal === 'buy').map(s => ({
                    x: new Date(s.date).getTime(),
                    y: s.actual_price,
                    marker: {
                        size: 8,
                        fillColor: '#00e676',
                        strokeColor: '#fff',
                        radius: 4
                    }
                }));
                
                const sellSignals = signals.filter(s => s.signal === 'sell').map(s => ({
                    x: new Date(s.date).getTime(),
                    y: s.actual_price,
                    marker: {
                        size: 8,
                        fillColor: '#ff5252',
                        strokeColor: '#fff',
                        radius: 4
                    }
                }));
                
                if (buySignals.length > 0) {
                    seriesData.push({
                        name: 'Buy Signal',
                        type: 'scatter',
                        data: buySignals
                    });
                }
                
                if (sellSignals.length > 0) {
                    seriesData.push({
                        name: 'Sell Signal',
                        type: 'scatter',
                        data: sellSignals
                    });
                }
                
                const options = {
                    series: seriesData,
                    chart: {
                        height: 400,
                        type: 'line',
                        background: '#1e1e1e',
                        foreColor: '#e0e0e0',
                        toolbar: {
                            show: true
                        },
                        zoom: {
                            enabled: true
                        }
                    },
                    colors: ['#00e676', '#40c4ff', '#00c853', '#ff5252'],
                    stroke: {
                        curve: 'smooth',
                        width: [3, 2, 0, 0]
                    },
                    markers: {
                        size: [0, 0, 8, 8]
                    },
                    title: {
                        text: 'Backtest Results',
                        align: 'center',
                        style: {
                            color: '#e0e0e0'
                        }
                    },
                    xaxis: {
                        type: 'datetime',
                        labels: {
                            style: {
                                colors: '#a0a0a0'
                            }
                        }
                    },
                    yaxis: {
                        labels: {
                            style: {
                                colors: '#a0a0a0'
                            },
                            formatter: function(val) {
                                return '$' + val.toFixed(2);
                            }
                        }
                    },
                    tooltip: {
                        theme: 'dark',
                        shared: true,
                        y: {
                            formatter: function(val) {
                                return '$' + val.toFixed(2);
                            }
                        }
                    },
                    grid: {
                        borderColor: '#333333'
                    },
                    legend: {
                        position: 'top',
                        horizontalAlign: 'center',
                        labels: {
                            colors: '#e0e0e0'
                        }
                    }
                };
                
                const chart = new ApexCharts(document.querySelector("#backtest-chart"), options);
                chart.render();
            });
        </script>
    @endif
@endsection