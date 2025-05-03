@extends('layouts.app')

@section('content')
    <div class="mb-3">
        <a href="{{ route('portfolios.show', $portfolio) }}" class="btn btn-secondary">&larr; Back to Portfolio</a>
    </div>

    <h2 class="mb-3">Edit Portfolio</h2>
    
    <div class="card">
        <div class="card-header">
            <h3 class="card-title">Portfolio Details</h3>
        </div>
        <div class="card-body">
            <form action="{{ route('portfolios.update', $portfolio) }}" method="POST">
                @csrf
                @method('PUT')
                
                <div class="form-group">
                    <label for="name" class="form-label">Portfolio Name</label>
                    <input type="text" id="name" name="name" class="form-control" value="{{ old('name', $portfolio->name) }}" required>
                </div>
                
                <div class="form-group">
                    <label for="description" class="form-label">Description (Optional)</label>
                    <textarea id="description" name="description" class="form-control" rows="3">{{ old('description', $portfolio->description) }}</textarea>
                </div>
                
                <button type="submit" class="btn btn-primary">Update Portfolio</button>
            </form>
        </div>
    </div>
@endsection