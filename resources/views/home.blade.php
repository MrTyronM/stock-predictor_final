{{-- This file redirects to dashboard, it's using layout to maintain consistency --}}
@extends('layouts.app')

@section('content')
<div class="container">
    <div class="text-center py-5">
        <h2>Redirecting to Dashboard...</h2>
        <p>If you are not redirected automatically, <a href="{{ route('dashboard') }}">click here</a>.</p>
        
        <script>
            window.location.href = "{{ route('dashboard') }}";
        </script>
    </div>
</div>
@endsection