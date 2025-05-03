@extends('layouts.app')

@section('content')
    <h2 class="mb-3">Dashboard</h2>
    
    <div class="card mb-4">
        <div class="card-header">
            <h3 class="card-title">Welcome, {{ auth()->user()->name }}</h3>
        </div>
        <div class="card-body">
            <p>Welcome to StockPredictor, your AI-powered stock market prediction platform.</p>
            <p>Browse stocks, view predictions, and make informed investment decisions.</p>
        </div>
    </div>
    
    <h3 class="mb-3">Recent Predictions</h3>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px;">
        @forelse($stocks as $stock)
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">{{ $stock->name }} ({{ $stock->symbol }})</h3>
                </div>
                <div class="card-body">
                    @if($stock->predictions->count() > 0)
                        <div class="prediction-card">
                            <span class="prediction-date">{{ $stock->predictions->first()->prediction_date }}</span>
                            <span class="prediction-price">${{ number_format($stock->predictions->first()->predicted_price, 2) }}</span>
                            <span class="prediction-recommendation recommendation-{{ $stock->predictions->first()->recommendation }}">
                                {{ ucfirst($stock->predictions->first()->recommendation) }}
                            </span>
                            <span class="prediction-confidence">Confidence: {{ $stock->predictions->first()->confidence }}%</span>
                        </div>
                    @else
                        <p>No predictions available yet.</p>
                    @endif
                </div>
                <div class="card-footer">
                    <a href="{{ route('predictions.show', $stock) }}" class="btn btn-primary">View Details</a>
                </div>
            </div>
        @empty
            <div class="card">
                <div class="card-body">
                    <p>No stocks available. Please check back later or contact an administrator.</p>
                </div>
            </div>
        @endforelse
    </div>
@endsection