@extends('layouts.app')

@section('content')
    <div class="mb-3">
        <a href="{{ route('admin.dashboard') }}" class="btn btn-secondary">&larr; Back to Dashboard</a>
    </div>

    <h2 class="mb-3">User Feedback</h2>
    
    <div class="card">
        <div class="card-header">
            <h3 class="card-title">Feedback Submissions</h3>
        </div>
        <div class="card-body">
            @if(count($feedback) > 0)
                <div class="table-container">
                    <table class="table">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>User</th>
                                <th>Stock</th>
                                <th>Rating</th>
                                <th>Comment</th>
                                <th>Submitted</th>
                            </tr>
                        </thead>
                        <tbody>
                            @foreach($feedback as $item)
                                <tr>
                                    <td>{{ $item->id }}</td>
                                    <td>{{ $item->user->name }}</td>
                                    <td>
                                        @if($item->prediction && $item->prediction->stock)
                                            {{ $item->prediction->stock->symbol }}
                                        @else
                                            N/A
                                        @endif
                                    </td>
                                    <td>
                                        @if($item->rating)
                                            <div style="display: flex;">
                                                @for($i = 1; $i <= 5; $i++)
                                                    @if($i <= $item->rating)
                                                        <span style="color: var(--accent-primary);">★</span>
                                                    @else
                                                        <span style="color: var(--text-secondary);">★</span>
                                                    @endif
                                                @endfor
                                            </div>
                                        @else
                                            No rating
                                        @endif
                                    </td>
                                    <td>{{ $item->comment }}</td>
                                    <td>{{ $item->created_at->format('Y-m-d H:i') }}</td>
                                </tr>
                            @endforeach
                        </tbody>
                    </table>
                </div>
            @else
                <p>No feedback submissions found.</p>
            @endif
        </div>
    </div>
@endsection