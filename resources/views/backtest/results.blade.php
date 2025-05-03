@extends('layouts.app')

@section('content')
<div class="container">
    <div class="d-flex justify-content-between align-items-center mb-3">
        <h2>Backtest Results: {{ $stock->symbol }}</h2>
        <a href="{{ route('backtest.index') }}" class="btn btn-primary">New Backtest</a>
    </div>
    
    <div class="card mb-4">
        <div class="card-header">
            <h3 class="card-title">Strategy Performance Summary</h3>
        </div>
        <div class="card-body">
            <div class="row">
                <div class="col-md-6">
                    <h4>{{ $results['summary']['strategy_name'] }}</h4>
                    <p>
                        <strong>Parameters:</strong>
                        @foreach($results['summary']['strategy_params'] as $param => $value)
                            {{ str_replace('_', ' ', ucfirst($param)) }}: {{ $value }}{{ !$loop->last ? ', ' : '' }}
                        @endforeach
                    </p>
                    <table class="table table-bordered">
                        <tr>
                            <th>Initial Capital</th>
                            <td>${{ number_format($results['summary']['initial_capital'], 2) }}</td>
                        </tr>
                        <tr>
                            <th>Final Portfolio Value</th>
                            <td>${{ number_format($results['summary']['final_value'], 2) }}</td>
                        </tr>
                        <tr>
                            <th>Total Return</th>
                            <td class="{{ $results['summary']['total_return'] >= 0 ? 'text-success' : 'text-danger' }}">
                                {{ number_format($results['summary']['total_return'], 2) }}%
                            </td>
                        </tr>
                        <tr>
                            <th>Buy & Hold Return</th>
                            <td class="{{ $results['summary']['buy_hold_return'] >= 0 ? 'text-success' : 'text-danger' }}">
                                {{ number_format($results['summary']['buy_hold_return'], 2) }}%
                            </td>
                        </tr>
                        <tr>
                            <th>Outperformance</th>
                            <td class="{{ $results['summary']['outperformance'] >= 0 ? 'text-success' : 'text-danger' }}">
                                {{ number_format($results['summary']['outperformance'], 2) }}%
                            </td>
                        </tr>
                        <tr>
                            <th>Total Trades</th>
                            <td>{{ $results['summary']['total_trades'] }}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="col-md-6">
                    <div id="performance-chart" style="height: 300px;"></div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="row mb-4">
        <div class="col-md-12">
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Strategy Visualization</h3>
                </div>
                <div class="card-body">
                    <div id="strategy-chart" style="height: 400px;"></div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <div class="card-header">
            <h3 class="card-title">Trade History</h3>
        </div>
        <div class="card-body">
            @if(count($results['trades']) > 0)
                <div class="table-container">
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Type</th>
                                <th>Price</th>
                                <th>Shares</th>
                                <th>Amount</th>
                                <th>Cash After</th>
                                <th>Portfolio Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            @foreach($results['trades'] as $trade)
                                <tr>
                                    <td>{{ $trade['date'] }}</td>
                                    <td class="{{ $trade['type'] == 'buy' ? 'text-success' : 'text-danger' }}">
                                        {{ ucfirst($trade['type']) }}
                                    </td>
                                    <td>${{ number_format($trade['price'], 2) }}</td>
                                    <td>{{ $trade['shares'] }}</td>
                                    <td>
                                        @if($trade['type'] == 'buy')
                                            ${{ number_format($trade['cost'], 2) }}
                                        @else
                                            ${{ number_format($trade['revenue'], 2) }}
                                        @endif
                                    </td>
                                    <td>${{ number_format($trade['cash_after'], 2) }}</td>
                                    <td>${{ number_format($trade['portfolio_after'], 2) }}</td>
                                </tr>
                            @endforeach
                        </tbody>
                    </table>
                </div>
            @else
                <p>No trades executed during the backtest period.</p>
            @endif
        </div>
    </div>
</div>

@endsection

@push('scripts')
<script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
<script>
    document.addEventListener('DOMContentLoaded', function() {
        // Performance comparison chart
        const performanceData = [
            {
                name: 'Strategy',
                data: @json($results['portfolio_values'])
            },
            {
                name: 'Buy & Hold',
                data: Array({{ count($results['portfolio_values']) }}).fill(0).map((_, i) => {
                    const initialShares = Math.floor({{ $results['summary']['initial_capital'] }} / {{ $results['prices'][0] }});
                    return initialShares * {{ json_encode($results['prices']) }}[i];
                })
            }
        ];
        
        const performanceOptions = {
            series: performanceData,
            chart: {
                type: 'line',
                height: 300,
                background: '#1e1e1e',
                foreColor: '#e0e0e0',
                toolbar: {
                    show: false
                }
            },
            stroke: {
                curve: 'smooth',
                width: 2
            },
            colors: ['#00e676', '#ff9800'],
            xaxis: {
                categories: Array.from({ length: {{ count($results['portfolio_values']) }} }, (_, i) => i + 1),
                labels: {
                    show: false
                }
            },
            yaxis: {
                labels: {
                    formatter: function(val) {
                        return '$' + val.toFixed(2);
                    }
                }
            },
            legend: {
                position: 'top'
            }
        };
        
        const performanceChart = new ApexCharts(document.querySelector("#performance-chart"), performanceOptions);
        performanceChart.render();
        
        // Strategy visualization
        const strategyData = [];
        
        // Add price series
        strategyData.push({
            name: 'Price',
            type: 'line',
            data: @json($results['prices'])
        });
        
        // Add strategy-specific indicators
        @if($strategy == 'sma_crossover')
            strategyData.push({
                name: 'Short MA',
                type: 'line',
                data: @json($results['short_ma'])
            });
            
            strategyData.push({
                name: 'Long MA',
                type: 'line',
                data: @json($results['long_ma'])
            });
        @elseif($strategy == 'rsi')
            strategyData.push({
                name: 'RSI',
                type: 'line',
                data: @json($results['rsi'])
            });
        @elseif($strategy == 'momentum')
            strategyData.push({
                name: 'Momentum',
                type: 'line',
                data: @json($results['momentum'])
            });
        @endif
        
        const strategyOptions = {
            series: strategyData,
            chart: {
                height: 400,
                type: 'line',
                background: '#1e1e1e',
                foreColor: '#e0e0e0',
                toolbar: {
                    show: true
                }
            },
            stroke: {
                curve: 'smooth',
                width: 2
            },
            colors: ['#00e676', '#00bcd4', '#ff9800'],
            xaxis: {
                categories: @json($results['dates']),
                labels: {
                    rotate: -45,
                    style: {
                        fontSize: '10px'
                    }
                }
            },
            yaxis: [
                {
                    title: {
                        text: 'Price'
                    },
                    labels: {
                        formatter: function(val) {
                            return '$' + val.toFixed(2);
                        }
                    }
                },
                @if($strategy == 'rsi')
                {
                    opposite: true,
                    min: 0,
                    max: 100,
                    title: {
                        text: 'RSI'
                    }
                }
                @endif
            ],
            annotations: {
                points: [
                    @foreach($results['trades'] as $trade)
                        {
                            x: '{{ $trade['date'] }}',
                            y: {{ $trade['price'] }},
                            marker: {
                                size: 6,
                                fillColor: '{{ $trade['type'] == 'buy' ? '#00e676' : '#ff5252' }}',
                                strokeColor: '#fff',
                                radius: 2
                            },
                            label: {
                                borderColor: '{{ $trade['type'] == 'buy' ? '#00e676' : '#ff5252' }}',
                                style: {
                                    color: '#fff',
                                    background: '{{ $trade['type'] == 'buy' ? '#00e676' : '#ff5252' }}'
                                },
                                text: '{{ ucfirst($trade['type']) }}'
                            }
                        },
                    @endforeach
                ]
            },
            legend: {
                position: 'top'
            }
        };
        
        const strategyChart = new ApexCharts(document.querySelector("#strategy-chart"), strategyOptions);
        strategyChart.render();
    });
</script>
@endpush