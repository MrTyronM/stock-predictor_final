@extends('layouts.app')

@section('content')
    <div class="mb-3">
        <a href="{{ route('portfolios.index') }}" class="btn btn-secondary">&larr; Back to Portfolios</a>
    </div>

    <h2 class="mb-3">Create New Portfolio</h2>
    
    <div class="card">
        <div class="card-header">
            <h3 class="card-title">Portfolio Details</h3>
        </div>
        <div class="card-body">
            <form action="{{ route('portfolios.store') }}" method="POST">
                @csrf
                
                <div class="form-group">
                    <label for="name" class="form-label">Portfolio Name</label>
                    <input type="text" id="name" name="name" class="form-control" value="{{ old('name') }}" required>
                </div>
                
                <div class="form-group">
                    <label for="description" class="form-label">Description (Optional)</label>
                    <textarea id="description" name="description" class="form-control" rows="3">{{ old('description') }}</textarea>
                </div>
                
                <button type="submit" class="btn btn-primary">Create Portfolio</button>
            </form>
        </div>
    </div>
@endsection