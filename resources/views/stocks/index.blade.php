@extends('layouts.app')

@section('content')
    <h2 class="mb-3">Available Stocks</h2>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px;">
        @forelse($stocks as $stock)
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">{{ $stock->name }} ({{ $stock->symbol }})</h3>
                </div>
                <div class="card-body">
                    @if($stock->description)
                        <p>{{ $stock->description }}</p>
                    @else
                        <p>{{ $stock->symbol }} - {{ $stock->name }}</p>
                    @endif
                </div>
                <div class="card-footer">
                    <a href="{{ route('stocks.show', $stock) }}" class="btn btn-primary">View Details</a>
                    <a href="{{ route('predictions.show', $stock) }}" class="btn btn-secondary">View Predictions</a>
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