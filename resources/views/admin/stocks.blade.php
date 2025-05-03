@extends('layouts.app')

@section('content')
    <div class="mb-3">
        <a href="{{ route('admin.dashboard') }}" class="btn btn-secondary">&larr; Back to Dashboard</a>
    </div>

    <h2 class="mb-3">Manage Stocks</h2>
    
    <div class="card mb-4">
        <div class="card-header">
            <h3 class="card-title">Add New Stock</h3>
        </div>
        <div class="card-body">
            <form action="{{ route('admin.stocks.store') }}" method="POST">
                @csrf
                
                <div class="form-group">
                    <label for="symbol" class="form-label">Stock Symbol</label>
                    <input type="text" id="symbol" name="symbol" class="form-control" required placeholder="e.g., AAPL, MSFT, GOOGL">
                    <small class="text-secondary">Enter the stock symbol exactly as it appears on Yahoo Finance</small>
                </div>
                
                <div class="form-group">
                    <label for="name" class="form-label">Company Name</label>
                    <input type="text" id="name" name="name" class="form-control" required placeholder="e.g., Apple Inc., Microsoft Corporation">
                </div>
                
                <div class="form-group">
                    <label for="description" class="form-label">Description (Optional)</label>
                    <textarea id="description" name="description" class="form-control" rows="3" placeholder="Brief description of the company..."></textarea>
                </div>
                
                <button type="submit" class="btn btn-primary">Add Stock</button>
            </form>
        </div>
    </div>
    
    <div class="card">
        <div class="card-header">
            <h3 class="card-title">Current Stocks</h3>
        </div>
        <div class="card-body">
            @if(count($stocks) > 0)
                <div class="table-container">
                    <table class="table">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Symbol</th>
                                <th>Name</th>
                                <th>Description</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            @foreach($stocks as $stock)
                                <tr>
                                    <td>{{ $stock->id }}</td>
                                    <td>{{ $stock->symbol }}</td>
                                    <td>{{ $stock->name }}</td>
                                    <td>{{ Str::limit($stock->description, 50) }}</td>
                                    <td>
                                        <div style="display: flex; gap: 10px;">
                                            <a href="{{ route('stocks.show', $stock) }}" class="btn btn-secondary" style="padding: 0.25rem 0.5rem; font-size: 0.875rem;">View</a>
                                            
                                            <form action="{{ route('admin.stocks.destroy', $stock) }}" method="POST" onsubmit="return confirm('Are you sure you want to delete this stock? All associated data will be permanently removed.');">
                                                @csrf
                                                @method('DELETE')
                                                <button type="submit" class="btn btn-danger" style="padding: 0.25rem 0.5rem; font-size: 0.875rem;">Delete</button>
                                            </form>
                                        </div>
                                    </td>
                                </tr>
                            @endforeach
                        </tbody>
                    </table>
                </div>
            @else
                <p>No stocks have been added yet.</p>
            @endif
        </div>
    </div>
@endsection