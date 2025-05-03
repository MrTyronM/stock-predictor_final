<!DOCTYPE html>
<htm lang="{{ str_replace('_', '-', app()->getLocale()) }}">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="csrf-token" content="{{ csrf_token() }}">

    <title>{{ config('app.name', 'Stock Predictor') }}</title>

    <!-- Fonts & Icons -->
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">

    <!-- Theme CSS -->
    @if(session('theme', 'dark') === 'dark')
        <link rel="stylesheet" href="{{ asset('css/custom.css') }}">
    @else
        <link rel="stylesheet" href="{{ asset('css/theme-light.css') }}">
    @endif

</head>
<body>
    <!-- Sidebar -->
    <aside class="sidebar">
        <div class="sidebar-header">SP</div>
        <ul class="sidebar-menu">
            <li class="sidebar-item"><a href="{{ route('dashboard') }}" class="sidebar-link {{ request()->routeIs('dashboard') ? 'active' : '' }}">
                <i class="fas fa-home sidebar-icon"></i><span class="sidebar-label">Dashboard</span></a></li>
            <li class="sidebar-item"><a href="{{ route('stocks.index') }}" class="sidebar-link {{ request()->routeIs('stocks.index') ? 'active' : '' }}">
                <i class="fas fa-chart-line sidebar-icon"></i><span class="sidebar-label">Stocks</span></a></li>
            <li class="sidebar-item"><a href="{{ route('stocks.screener') }}" class="sidebar-link {{ request()->routeIs('stocks.screener') ? 'active' : '' }}">
                <i class="fas fa-filter sidebar-icon"></i><span class="sidebar-label">Screener</span></a></li>
            <li class="sidebar-item"><a href="{{ route('portfolios.index') }}" class="sidebar-link {{ request()->routeIs('portfolios.*') ? 'active' : '' }}">
                <i class="fas fa-briefcase sidebar-icon"></i><span class="sidebar-label">Portfolios</span></a></li>
            <li class="sidebar-item"><a href="{{ route('news.index') }}" class="sidebar-link {{ request()->routeIs('news.*') ? 'active' : '' }}">
                <i class="fas fa-newspaper sidebar-icon"></i><span class="sidebar-label">News</span></a></li>
            <li class="sidebar-item"><a href="{{ route('market.dashboard') }}" class="sidebar-link {{ request()->routeIs('market.dashboard') ? 'active' : '' }}">
                <i class="fas fa-globe sidebar-icon"></i><span class="sidebar-label">Market</span></a></li>
            <li class="sidebar-item"><a href="{{ route('notifications.index') }}" class="sidebar-link {{ request()->routeIs('notifications.*') ? 'active' : '' }}">
                <i class="fas fa-bell sidebar-icon"></i><span class="sidebar-label">Notifications</span></a></li>
            @if(auth()->check() && auth()->user()->isAdmin())
            <li class="sidebar-item"><a href="{{ route('admin.dashboard') }}" class="sidebar-link {{ request()->routeIs('admin.*') ? 'active' : '' }}">
                <i class="fas fa-user-shield sidebar-icon"></i><span class="sidebar-label">Admin</span></a></li>
            @endif
            <li class="sidebar-item">
                <form method="POST" action="{{ route('logout') }}">
                    @csrf
                    <button type="submit" class="sidebar-link" style="background: none; border: none; width: 100%; text-align: left;">
                        <i class="fas fa-sign-out-alt sidebar-icon"></i><span class="sidebar-label">Logout</span>
                    </button>
                </form>
            </li>
        </ul>
    </aside>

    <!-- Main Content -->
    <main class="main-content">
        @if(session('success'))
            <div class="alert alert-success">{{ session('success') }}</div>
        @endif
        @if(session('error'))
            <div class="alert alert-danger">{{ session('error') }}</div>
        @endif
        @if(session('warning'))
            <div class="alert alert-warning">{{ session('warning') }}</div>
        @endif
        @if($errors->any())
            <div class="alert alert-danger">
                <ul style="margin: 0; padding-left: 20px;">
                    @foreach($errors->all() as $error)<li>{{ $error }}</li>@endforeach
                </ul>
            </div>
        @endif

        @yield('content')

        <footer>
            © {{ date('Y') }} Stock Predictor | AI-Powered Financial Insights
        </footer>
    </main>
</body>
</htm