@extends('layouts.app')

@section('content')
    <h2 class="mb-3">Stock Screener</h2>
    
    <div class="card mb-4">
        <div class="card-header">
            <h3 class="card-title">Filter Stocks</h3>
        </div>
        <div class="card-body">
            <form action="{{ route('stocks.screener') }}" method="GET">
                <div class="row">
                    <div class="col-md-4">
                        <div class="form-group">
                            <label for="recommendation" class="form-label">Recommendation</label>
                            <select id="recommendation" name="recommendation" class="form-control">
                                <option value="">Any</option>
                                <option value="buy" {{ $recommendation == 'buy' ? 'selected' : '' }}>Buy</option>
                                <option value="sell" {{ $recommendation == 'sell' ? 'selected' : '' }}>Sell</option>
                                <option value="hold" {{ $recommendation == 'hold' ? 'selected' : '' }}>Hold</option>
                            </select>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="form-group">
                            <label for="confidence_min" class="form-label">Minimum Confidence (%)</label>
                            <input type="number" id="confidence_min" name="confidence_min" class="form-control" min="0" max="100" value="{{ $confidenceMin }}">
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="form-group">
                            <label for="price_min" class="form-label">Minimum Price ($)</label>
                            <input type="number" id="price_min" name="price_min" class="form-control" min="0" step="0.01" value="{{ $priceMin }}">
                        </div>
                    </div>
                </div>
                <div class="row mt-3">
                    <div class="col-md-4">
                        <div class="form-group">
                            <label for="price_max" class="form-label">Maximum Price ($)</label>
                            <input type="number" id="price_max" name="price_max" class="form-control" min="0" step="0.01" value="{{ $priceMax }}">
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="form-group">
                            <label for="change_min" class="form-label">Minimum Predicted Change (%)</label>
                            <input type="number" id="change_min" name="change_min" class="form-control" step="0.1" value="{{ $changeMin }}">
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="form-group">
                            <label for="change_max" class="form-label">Maximum Predicted Change (%)</label>
                            <input type="number" id="change_max" name="change_max" class="form-control" step="0.1" value="{{ $changeMax }}">
                        </div>
                    </div>
                </div>
                <div class="mt-3">
                    <button type="submit" class="btn btn-primary">Apply Filters</button>
                    <a href="{{ route('stocks.screener') }}" class="btn btn-secondary">Reset</a>
                </div>
            </form>
        </div>
    </div>
    
    <div class="card">
        <div class="card-header">
            <h3 class="card-title">Results ({{ count($filteredStocks) }} stocks)</h3>
        </div>
        <div class="card-body">
            @if(count($filteredStocks) > 0)
                <div class="table-container">
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Name</th>
                                <th>Current Price</th>
                                <th>Predicted Price</th>
                                <th>Change</th>
                                <th>Recommendation</th>
                                <th>Confidence</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            @foreach($filteredStocks as $stock)
                                @php
                                    $latestPrice = $stock->stockData->first()->close ?? 0;
                                    $prediction = $stock->predictions->first();
                                    $predictedPrice = $prediction->predicted_price ?? 0;
                                    $change = $latestPrice > 0 ? (($predictedPrice - $latestPrice) / $latestPrice) * 100 : 0;
                                @endphp
                                <tr>
                                    <td>{{ $stock->symbol }}</td>
                                    <td>{{ $stock->name }}</td>
                                    <td>${{ number_format($latestPrice, 2) }}</td>
                                    <td>${{ number_format($predictedPrice, 2) }}</td>
                                    <td class="{{ $change >= 0 ? 'text-success' : 'text-danger' }}">
                                        {{ $change >= 0 ? '+' : '' }}{{ number_format($change, 2) }}%
                                    </td>
                                    <td>
                                        <span class="prediction-recommendation recommendation-{{ $prediction->recommendation }}">
                                            {{ ucfirst($prediction->recommendation) }}
                                        </span>
                                    </td>
                                    <td>{{ number_format($prediction->confidence, 1) }}%</td>
                                    <td>
                                        <div class="btn-group">
                                            <a href="{{ route('stocks.show', $stock) }}" class="btn btn-primary btn-sm">View</a>
                                            <a href="{{ route('predictions.show', $stock) }}" class="btn btn-secondary btn-sm">Predictions</a>
                                        </div>
                                    </td>
                                </tr>
                            @endforeach
                        </tbody>
                    </table>
                </div>
            @else
                <p>No stocks match your filter criteria. Try adjusting your filters.</p>
            @endif
        </div>
    </div>
@endsection