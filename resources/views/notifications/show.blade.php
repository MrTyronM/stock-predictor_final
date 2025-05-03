@extends('layouts.app')

@section('content')
<div class="container">
    <div class="d-flex justify-content-between align-items-center mb-3">
        <h2>{{ $notification->title }}</h2>
        <div>
            @if(!$notification->read)
                <form action="{{ route('notifications.read', $notification->id) }}" method="POST" class="d-inline">
                    @csrf
                    <button type="submit" class="btn btn-primary">Mark as Read</button>
                </form>
            @endif
            <form action="{{ route('notifications.destroy', $notification->id) }}" method="POST" class="d-inline">
                @csrf
                @method('DELETE')
                <button type="submit" class="btn btn-danger" onclick="return confirm('Are you sure you want to delete this notification?')">Delete</button>
            </form>
            <a href="{{ route('notifications.index') }}" class="btn btn-secondary">Back to Notifications</a>
        </div>
    </div>
    
    <div class="card">
        <div class="card-header">
            <div class="d-flex justify-content-between">
                <span>
                    <span class="badge bg-{{ $notification->type }}">
                        {{ ucfirst($notification->type) }}
                    </span>
                    @if($notification->stock)
                        <span class="ms-2">Stock: <a href="{{ route('stocks.show', $notification->stock->id) }}">{{ $notification->stock->symbol }}</a></span>
                    @endif
                </span>
                <span>{{ $notification->created_at->format('Y-m-d H:i') }}</span>
            </div>
        </div>
        <div class="card-body">
            <div class="notification-message">
                {{ $notification->message }}
            </div>
        </div>
    </div>
</div>
@endsection