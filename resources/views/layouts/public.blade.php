<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="csrf-token" content="{{ csrf_token() }}">

    <title>{{ config('app.name', 'Laravel') }} @yield('title')</title>

    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">

    <!-- Styles -->
    <link href="{{ asset('css/custom.css') }}" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.3.0/css/all.min.css">
    
    @yield('styles')
</head>
<body>
    <div id="app">
        <nav class="navbar">
            <div class="container navbar-container">
                <a class="navbar-logo" href="{{ url('/') }}">
                    {{ config('app.name', 'Laravel') }}
                </a>
                
                <button class="navbar-toggle" id="navbarToggle">
                    <i class="navbar-toggle-icon">≡</i>
                </button>
                
                <ul class="navbar-menu" id="navbarMenu">
                    <li class="navbar-item">
                        <a class="navbar-link {{ request()->routeIs('public.index') ? 'active' : '' }}" href="{{ url('/') }}">Home</a>
                    </li>
                    <li class="navbar-item">
                        <a class="navbar-link {{ request()->routeIs('market.dashboard') ? 'active' : '' }}" href="{{ route('market.dashboard') }}">Market Dashboard</a>
                    </li>
                    
                    <!-- Authentication Links -->
                    @guest
                        @if (Route::has('login'))
                            <li class="navbar-item">
                                <a class="navbar-link {{ request()->routeIs('login') ? 'active' : '' }}" href="{{ route('login') }}">{{ __('Login') }}</a>
                            </li>
                        @endif

                        @if (Route::has('register'))
                            <li class="navbar-item">
                                <a class="navbar-link {{ request()->routeIs('register') ? 'active' : '' }}" href="{{ route('register') }}">{{ __('Register') }}</a>
                            </li>
                        @endif
                    @else
                        <li class="navbar-item">
                            <a class="navbar-link" href="{{ route('logout') }}"
                               onclick="event.preventDefault(); document.getElementById('logout-form').submit();">
                                {{ __('Logout') }}
                            </a>
                            <form id="logout-form" action="{{ route('logout') }}" method="POST" class="d-none">
                                @csrf
                            </form>
                        </li>
                    @endguest
                </ul>
            </div>
        </nav>

        <main class="py-4">
            @yield('content')
        </main>

        <footer class="mt-4 py-3">
            <div class="container">
                <div class="row">
                    <div class="col-md-6">
                        <p class="mb-0">&copy; {{ date('Y') }} {{ config('app.name', 'Laravel') }}. All rights reserved.</p>
                    </div>
                    <div class="col-md-6 text-end">
                        <p class="mb-0">Powered by Laravel {{ app()->version() }}</p>
                    </div>
                </div>
            </div>
        </footer>
    </div>

    <!-- Scripts -->
    <script>
        // Simple mobile menu toggle
        document.addEventListener('DOMContentLoaded', function() {
            const navbarToggle = document.getElementById('navbarToggle');
            const navbarMenu = document.getElementById('navbarMenu');
            
            if (navbarToggle && navbarMenu) {
                navbarToggle.addEventListener('click', function() {
                    navbarMenu.classList.toggle('active');
                });
            }
        });
    </script>
    
    @yield('scripts')
</body>
</html>