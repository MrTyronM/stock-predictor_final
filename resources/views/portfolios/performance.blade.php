@extends('layouts.app')

@section('content')
    <div class="mb-3">
        <a href="{{ route('portfolios.show', $portfolio) }}" class="btn btn-secondary">&larr; Back to Portfolio</a>
    </div>

    <h2 class="mb-3">{{ $portfolio->name }} - Performance Analysis</h2>
    
    <div class="row mb-4">
        <div class="col-md-4">
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Portfolio Summary</h3>
                </div>
                <div class="card-body">
                    <div class="d-flex justify-content-between mb-3">
                        <div>Total Value:</div>
                        <div class="text-accent">${{ number_format($metrics['total_value'], 2) }}</div>
                    </div>
                    <div class="d-flex justify-content-between mb-3">
                        <div>Total Cost:</div>
                        <div>${{ number_format($metrics['total_cost'], 2) }}</div>
                    </div>
                    <div class="d-flex justify-content-between mb-3">
                        <div>Total Gain/Loss:</div>
                        <div class="{{ $metrics['total_gain_loss'] >= 0 ? 'text-success' : 'text-danger' }}">
                            ${{ number_format($metrics['total_gain_loss'], 2) }} 
                            ({{ number_format($metrics['total_gain_loss_percent'], 2) }}%)
                        </div>
                    </div>
                    <div class="d-flex justify-content-between mb-3">
                        <div>Number of Assets:</div>
                        <div>{{ count($metrics['items']) }}</div>
                    </div>
                </div>
            </div>
        </div>
        <div class="col-md-8">
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Portfolio Performance</h3>
                </div>
                <div class="card-body">
                    <div id="performance-chart" style="height: 250px;"></div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="row mb-4">
        <div class="col-md-6">
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Asset Allocation</h3>
                </div>
                <div class="card-body">
                    <div id="allocation-chart" style="height: 300px;"></div>
                </div>
            </div>
        </div>
        <div class="col-md-6">
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Prediction Summary</h3>
                </div>
                <div class="card-body">
                    <div id="prediction-chart" style="height: 300px;"></div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <div class="card-header">
            <h3 class="card-title">Asset Performance</h3>
        </div>
        <div class="card-body">
            <div class="table-container">
                <table class="table">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Name</th>
                            <th>Shares</th>
                            <th>Purchase Price</th>
                            <th>Current Price</th>
                            <th>Current Value</th>
                            <th>Gain/Loss</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        @foreach($metrics['items'] as $itemId => $item)
                            <tr>
                                <td>{{ $item['symbol'] }}</td>
                                <td>{{ $item['name'] }}</td>
                                <td>{{ $item['shares'] }}</td>
                                <td>${{ number_format($item['purchase_price'], 2) }}</td>
                                <td>${{ number_format($item['current_price'], 2) }}</td>
                                <td>${{ number_format($item['current_value'], 2) }}</td>
                                <td class="{{ $item['gain_loss'] >= 0 ? 'text-success' : 'text-danger' }}">
                                    ${{ number_format($item['gain_loss'], 2) }}
                                    ({{ number_format($item['gain_loss_percent'], 2) }}%)
                                </td>
                                <td>
                                    <a href="{{ route('stocks.show', $item['stock_id']) }}" class="btn btn-primary btn-sm">View</a>
                                </td>
                            </tr>
                        @endforeach
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Performance chart
            const performanceDates = @json($metrics['historical_performance']['dates']);
            const performanceValues = @json($metrics['historical_performance']['values']);
            
            const performanceData = performanceDates.map((date, index) => ({
                x: new Date(date).getTime(),
                y: performanceValues[index]
            }));
            
            const performanceOptions = {
                series: [{
                    name: 'Portfolio Value',
                    data: performanceData
                }],
                chart: {
                    type: 'area',
                    height: 250,
                    background: '#1e1e1e',
                    foreColor: '#e0e0e0',
                    toolbar: {
                        show: false
                    }
                },
                stroke: {
                    curve: 'smooth',
                    width: 2,
                    colors: ['#00e676']
                },
                fill: {
                    type: 'gradient',
                    gradient: {
                        shadeIntensity: 1,
                        opacityFrom: 0.7,
                        opacityTo: 0.3,
                        stops: [0, 90, 100],
                        colorStops: [
                            {
                                offset: 0,
                                color: '#00e676',
                                opacity: 0.4
                            },
                            {
                                offset: 100,
                                color: '#00e676',
                                opacity: 0
                            }
                        ]
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
                    x: {
                        format: 'dd MMM yyyy'
                    },
                    y: {
                        formatter: function(val) {
                            return '$' + val.toFixed(2);
                        }
                    }
                },
                grid: {
                    borderColor: '#333333'
                }
            };
            
            const performanceChart = new ApexCharts(document.querySelector("#performance-chart"), performanceOptions);
            performanceChart.render();
            
            // Allocation chart
            const allocationLabels = [];
            const allocationValues = [];
            
            @foreach($metrics['allocation'] as $stockId => $data)
                allocationLabels.push('{{ $data["symbol"] }}');
                allocationValues.push({{ $data["percent"] }});
            @endforeach
            
            const allocationOptions = {
                series: allocationValues,
                chart: {
                    type: 'donut',
                    height: 300,
                    background: '#1e1e1e',
                    foreColor: '#e0e0e0'
                },
                labels: allocationLabels,
                colors: ['#00e676', '#40c4ff', '#ffab40', '#ff5252', '#8e24aa', '#1e88e5', '#43a047'],
                plotOptions: {
                    pie: {
                        donut: {
                            size: '60%',
                            labels: {
                                show: true,
                                total: {
                                    show: true,
                                    label: 'Total Value',
                                    formatter: function() {
                                        return '${{ number_format($metrics["total_value"], 2) }}';
                                    }
                                }
                            }
                        }
                    }
                },
                legend: {
                    position: 'bottom',
                    labels: {
                        colors: '#e0e0e0'
                    }
                },
                tooltip: {
                    theme: 'dark',
                    y: {
                        formatter: function(val) {
                            return val.toFixed(2) + '%';
                        }
                    }
                },
                responsive: [{
                    breakpoint: 480,
                    options: {
                        chart: {
                            height: 250
                        },
                        legend: {
                            position: 'bottom'
                        }
                    }
                }]
            };
            
            const allocationChart = new ApexCharts(document.querySelector("#allocation-chart"), allocationOptions);
            allocationChart.render();
            
            // Prediction chart
            const predictionOptions = {
                series: [
                    {{ $metrics['prediction_summary']['buy'] }},
                    {{ $metrics['prediction_summary']['hold'] }},
                    {{ $metrics['prediction_summary']['sell'] }}
                ],
                chart: {
                    height: 300,
                    type: 'pie',
                    background: '#1e1e1e',
                    foreColor: '#e0e0e0'
                },
                labels: ['Buy', 'Hold', 'Sell'],
                colors: ['#00e676', '#ffab40', '#ff5252'],
                legend: {
                    position: 'bottom',
                    labels: {
                        colors: '#e0e0e0'
                    }
                },
                tooltip: {
                    theme: 'dark'
                },
                responsive: [{
                    breakpoint: 480,
                    options: {
                        chart: {
                            height: 250
                        },
                        legend: {
                            position: 'bottom'
                        }
                    }
                }]
            };
            
            const predictionChart = new ApexCharts(document.querySelector("#prediction-chart"), predictionOptions);
            predictionChart.render();
        });
    </script>
@endsection