@extends('layouts.app')

@section('content')
    <div class="d-flex justify-content-between align-items-center mb-4">
        <h2>My Portfolios</h2>
        <a href="{{ route('portfolios.create') }}" class="btn btn-primary">Create New Portfolio</a>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px;">
        @forelse($portfolios as $portfolio)
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">{{ $portfolio->name }}</h3>
                </div>
                <div class="card-body">
                    @if($portfolio->description)
                        <p>{{ $portfolio->description }}</p>
                    @endif
                    <p>{{ $portfolio->items->count() }} stocks</p>
                </div>
                <div class="card-footer">
                    <a href="{{ route('portfolios.show', $portfolio) }}" class="btn btn-primary">View Details</a>
                </div>
            </div>
        @empty
            <div class="card">
                <div class="card-body">
                    <p>You don't have any portfolios yet. Create your first portfolio to start tracking stocks!</p>
                </div>
            </div>
        @endforelse
    </div>
@endsection