@extends('layouts.app')

@section('content')
    <div class="mb-3">
        <a href="{{ route('admin.dashboard') }}" class="btn btn-secondary">&larr; Back to Dashboard</a>
    </div>

    <h2 class="mb-3">User Management</h2>
    
    <div class="card">
        <div class="card-header">
            <h3 class="card-title">Registered Users</h3>
        </div>
        <div class="card-body">
            @if(count($users) > 0)
                <div class="table-container">
                    <table class="table">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Name</th>
                                <th>Email</th>
                                <th>Role</th>
                                <th>Registered On</th>
                            </tr>
                        </thead>
                        <tbody>
                            @foreach($users as $user)
                                <tr>
                                    <td>{{ $user->id }}</td>
                                    <td>{{ $user->name }}</td>
                                    <td>{{ $user->email }}</td>
                                    <td>
                                        <span class="prediction-recommendation {{ $user->role === 'admin' ? 'recommendation-buy' : 'recommendation-hold' }}">
                                            {{ ucfirst($user->role) }}
                                        </span>
                                    </td>
                                    <td>{{ $user->created_at->format('Y-m-d H:i') }}</td>
                                </tr>
                            @endforeach
                        </tbody>
                    </table>
                </div>
            @else
                <p>No users found.</p>
            @endif
        </div>
    </div>
@endsection