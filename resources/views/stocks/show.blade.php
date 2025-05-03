@extends('layouts.app')

@section('content')
    <div class="mb-3">
        <a href="{{ route('stocks.index') }}" class="btn btn-secondary">&larr; Back to Stocks</a>
    </div>

    <h2 class="mb-3">{{ $stock->name }} ({{ $stock->symbol }})</h2>
    
    @if($stock->description)
        <div class="card mb-4">
            <div class="card-header">
                <h3 class="card-title">Description</h3>
            </div>
            <div class="card-body">
                <p>{{ $stock->description }}</p>
            </div>
        </div>
    @endif
    
    <div class="card mb-4">
        <div class="card-header">
            <h3 class="card-title">Historical Data (Last 30 Days)</h3>
        </div>
        <div class="card-body">
            @if(count($stockData) > 0)
                <div id="stock-chart" style="width: 100%; height: 400px;"></div>
            @else
                <p>No historical data available for this stock.</p>
            @endif
        </div>
    </div>

    <!-- Export Buttons Below Chart -->
    <div class="mb-3">
        <div class="btn-group">
            <a href="{{ route('stocks.export', ['stock' => $stock, 'type' => 'prices', 'format' => 'csv']) }}" class="btn btn-secondary btn-sm">Export Price Data (CSV)</a>
            <a href="{{ route('stocks.export', ['stock' => $stock, 'type' => 'prices', 'format' => 'json']) }}" class="btn btn-secondary btn-sm">Export Price Data (JSON)</a>
        </div>
    </div>
    
    <div class="card">
        <div class="card-header">
            <h3 class="card-title">Price History</h3>
        </div>
        <div class="card-body">
            @if(count($stockData) > 0)
                <div class="table-container">
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Open</th>
                                <th>High</th>
                                <th>Low</th>
                                <th>Close</th>
                                <th>Volume</th>
                            </tr>
                        </thead>
                        <tbody>
                            @foreach($stockData as $data)
                                <tr>
                                    <td>{{ $data->date }}</td>
                                    <td>${{ number_format($data->open, 2) }}</td>
                                    <td>${{ number_format($data->high, 2) }}</td>
                                    <td>${{ number_format($data->low, 2) }}</td>
                                    <td>${{ number_format($data->close, 2) }}</td>
                                    <td>{{ number_format($data->volume) }}</td>
                                </tr>
                            @endforeach
                        </tbody>
                    </table>
                </div>
            @else
                <p>No historical data available for this stock.</p>
            @endif
        </div>
    </div>

    <script>
        document.addEventListener("DOMContentLoaded", function() {
            @if(count($stockData) > 0)
                const stockData = @json($stockData);
                
                const seriesData = stockData.map(item => ({
                    x: new Date(item.date).getTime(),
                    y: [
                        parseFloat(item.open),
                        parseFloat(item.high),
                        parseFloat(item.low),
                        parseFloat(item.close)
                    ]
                }));
                
                const volumeData = stockData.map(item => ({
                    x: new Date(item.date).getTime(),
                    y: parseInt(item.volume)
                }));
                
                const options = {
                    series: [{
                        name: 'Stock Price',
                        type: 'candlestick',
                        data: seriesData
                    }, {
                        name: 'Volume',
                        type: 'bar',
                        data: volumeData
                    }],
                    chart: {
                        type: 'candlestick',
                        height: 400,
                        background: '#1e1e1e',
                        foreColor: '#e0e0e0',
                        toolbar: {
                            show: true,
                            tools: {
                                download: true,
                                selection: true,
                                zoom: true,
                                zoomin: true,
                                zoomout: true,
                                pan: true,
                                reset: true
                            }
                        }
                    },
                    plotOptions: {
                        candlestick: {
                            colors: {
                                upward: '#00e676',
                                downward: '#ff5252'
                            }
                        },
                        bar: {
                            colors: {
                                ranges: [{
                                    from: 0,
                                    to: Infinity,
                                    color: '#40c4ff'
                                }]
                            }
                        }
                    },
                    title: {
                        text: '{{ $stock->symbol }} Stock Price',
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
                    yaxis: [{
                        tooltip: {
                            enabled: true
                        },
                        labels: {
                            style: {
                                colors: '#a0a0a0'
                            },
                            formatter: function(val) {
                                return '$' + val.toFixed(2);
                            }
                        }
                    }, {
                        opposite: true,
                        labels: {
                            style: {
                                colors: '#a0a0a0'
                            },
                            formatter: function(val) {
                                if (val >= 1000000) {
                                    return (val / 1000000).toFixed(1) + 'M';
                                } else if (val >= 1000) {
                                    return (val / 1000).toFixed(1) + 'K';
                                } else {
                                    return val;
                                }
                            }
                        }
                    }],
                    grid: {
                        borderColor: '#333333'
                    },
                    tooltip: {
                        theme: 'dark'
                    }
                };
                
                const chart = new ApexCharts(document.querySelector("#stock-chart"), options);
                chart.render();
            @endif
        });
    </script>
@endsection
