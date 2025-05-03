@extends('layouts.app')

@section('content')
<div class="container">
    <div class="d-flex justify-content-between align-items-center mb-3">
        <h2>Training Logs for {{ $stock->symbol }}</h2>
        <a href="{{ route('admin.model-training.index') }}" class="btn btn-primary">Back to Model Training</a>
    </div>
    
    <div class="card">
        <div class="card-header">
            <h3 class="card-title">Training Process Logs</h3>
        </div>
        <div class="card-body">
            <div class="log-container" style="background-color: #1e1e1e; color: #ddd; padding: 15px; border-radius: 4px; height: 500px; overflow-y: auto; font-family: monospace;">
                <pre>{{ $logs }}</pre>
            </div>
        </div>
    </div>
</div>
@endsection