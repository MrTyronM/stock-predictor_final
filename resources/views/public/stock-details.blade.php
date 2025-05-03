@extends('layouts.public')

@section('content')
<div class="container">
    <div class="card mb-4">
        <div class="card-header">
            <div class="d-flex justify-content-between align-items-center">
                <h2>{{ $stock->symbol }} - {{ $stock->name }}</h2>
                <div>
                    <a href="{{ route('public.index') }}" class="btn btn-secondary">Back to Home</a>
                    @auth
                        <a href="{{ route('stocks.show', $stock->id) }}" class="btn btn-primary">Full Analysis</a>
                    @else
                        <a href="{{ route('login') }}" class="btn btn-primary">Login for Full Analysis</a>
                    @endauth
                </div>
            </div>
        </div>
        <div class="card-body">
            <div class="row">
                <div class="col-md-6">
                    <h3>Price: ${{ number_format($stockData->last()->close, 2) }}</h3>
                    
                    @php
                        $change = $stockData->last()->close - $stockData[count($stockData) - 2]->close;
                        $changePercent = ($change / $stockData[count($stockData) - 2]->close) * 100;
                    @endphp
                    
                    <h4 class="{{ $change >= 0 ? 'text-success' : 'text-danger' }}">
                        {{ $change >= 0 ? '+' : '' }}{{ number_format($change, 2) }} 
                        ({{ $change >= 0 ? '+' : '' }}{{ number_format($changePercent, 2) }}%)
                    </h4>
                    
                    <div class="mt-4">
                        <p><strong>Open:</strong> ${{ number_format($stockData->last()->open, 2) }}</p>
                        <p><strong>High:</strong> ${{ number_format($stockData->last()->high, 2) }}</p>
                        <p><strong>Low:</strong> ${{ number_format($stockData->last()->low, 2) }}</p>
                        <p><strong>Volume:</strong> {{ number_format($stockData->last()->volume) }}</p>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <div id="price-chart" style="height: 300px;"></div>
                </div>
            </div>
        </div>
    </div>
    
    @if($news->count() > 0)
        <div class="card mb-4">
            <div class="card-header">
                <h3 class="card-title">Related News</h3>
            </div>
            <div class="card-body">
                @foreach($news as $item)
                    <div class="mb-3 pb-3 border-bottom">
                        <h4>{{ $item->title }}</h4>
                        <p class="text-muted">{{ $item->source }} • {{ $item->published_date->format('M d, Y') }}</p>
                        <p>{{ Str::limit($item->content, 200) }}</p>
                        @if($item->url)
                            <a href="{{ $item->url }}" target="_blank" class="btn btn-sm btn-outline-primary">Read More</a>
                        @endif
                    </div>
                @endforeach
            </div>
        </div>
    @endif
    
    <div class="card">
        <div class="card-header">
            <h3 class="card-title">Historical Data</h3>
        </div>
        <div class="card-body">
            <div class="table-responsive">
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
                        @foreach($stockData->reverse() as $data)
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
        </div>
    </div>
    
    <div class="text-center mt-4">
        <p class="mb-2">Want access to AI-powered predictions and advanced analysis for {{ $stock->symbol }}?</p>
        @auth
            <a href="{{ route('stocks.show', $stock->id) }}" class="btn btn-lg btn-primary">View Full Analysis</a>
        @else
            <a href="{{ route('register') }}" class="btn btn-lg btn-primary">Sign Up Now</a>
            <a href="{{ route('login') }}" class="btn btn-lg btn-outline-primary ms-2">Login</a>
        @endauth
    </div>
</div>

@endsection

@push('scripts')
<script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
<script>
    document.addEventListener('DOMContentLoaded', function() {
        const priceData = @json($stockData->map(function($data) {
            return [
                new Date($data->date).getTime(),
                $data->close
            ];
        }));
        
        const options = {
            series: [{
                name: 'Price',
                data: priceData
            }],
            chart: {
                type: 'area',
                height: 300,
                toolbar: {
                    show: false
                }
            },
            dataLabels: {
                enabled: false
            },
            stroke: {
                curve: 'smooth',
                width: 2
            },
            fill: {
                type: 'gradient',
                gradient: {
                    shadeIntensity: 1,
                    opacityFrom: 0.7,
                    opacityTo: 0.3,
                    stops: [0, 90, 100]
                }
            },
            xaxis: {
                type: 'datetime'
            },
            yaxis: {
                labels: {
                    formatter: function (val) {
                        return '$' + val.toFixed(2);
                    }
                }
            },
            tooltip: {
                x: {
                    format: 'dd MMM yyyy'
                }
            }
        };
        
        const chart = new ApexCharts(document.querySelector("#price-chart"), options);
        chart.render();
    });
</script>
@endpush