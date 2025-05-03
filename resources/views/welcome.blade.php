<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Stock Predictor - AI-Powered Financial Analysis</title>
    
    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    
    <!-- Custom CSS -->
    <link rel="stylesheet" href="{{ asset('css/custom.css') }}">
    

</head>
<body>
    <nav class="navbar">
        <div class="container navbar-container">
            <a href="{{ url('/') }}" class="navbar-logo">StockPredictor</a>
            
            <button class="navbar-toggle" id="navbar-toggle">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="3" y1="12" x2="21" y2="12"></line>
                    <line x1="3" y1="6" x2="21" y2="6"></line>
                    <line x1="3" y1="18" x2="21" y2="18"></line>
                </svg>
            </button>
            
            <ul class="navbar-menu" id="navbar-menu">
                <li class="navbar-item">
                    <a href="{{ route('about') }}" class="navbar-link">About</a>
                </li>
                @if (Route::has('login'))
                    @auth
                        <li class="navbar-item">
                            <a href="{{ route('dashboard') }}" class="navbar-link">Dashboard</a>
                        </li>
                    @else
                        <li class="navbar-item">
                            <a href="{{ route('login') }}" class="navbar-link">Login</a>
                        </li>
                        @if (Route::has('register'))
                            <li class="navbar-item">
                                <a href="{{ route('register') }}" class="navbar-link">Register</a>
                            </li>
                        @endif
                    @endauth
                @endif
            </ul>
        </div>
    </nav>

    <main>
        <section class="hero">
            <div class="container">
                <h1>AI-Powered Stock Market Predictions</h1>
                <p>Make smarter investment decisions with our advanced machine learning algorithms. Our platform analyzes historical data to predict stock price movements and generate buy, hold, or sell recommendations.</p>
                <div class="hero-buttons">
                    @auth
                        <a href="{{ route('dashboard') }}" class="btn btn-primary">Go to Dashboard</a>
                    @else
                        <a href="{{ route('login') }}" class="btn btn-primary">Get Started</a>
                        <a href="{{ route('about') }}" class="btn btn-secondary">Learn More</a>
                    @endauth
                </div>
            </div>
        </section>
        
        <section class="features">
            <div class="container">
                <div class="features-grid">
                    <div class="feature-card">
                        <div class="feature-icon">📊</div>
                        <h3 class="feature-title">Accurate Predictions</h3>
                        <p class="feature-description">Our LSTM neural networks analyze historical stock data to predict future price movements with high accuracy.</p>
                    </div>
                    
                    <div class="feature-card">
                        <div class="feature-icon">💡</div>
                        <h3 class="feature-title">Smart Recommendations</h3>
                        <p class="feature-description">Get clear buy, hold, or sell signals for each stock based on predicted price trends and market analysis.</p>
                    </div>
                    
                    <div class="feature-card">
                        <div class="feature-icon">📱</div>
                        <h3 class="feature-title">User-Friendly Interface</h3>
                        <p class="feature-description">Access intuitive charts, detailed analytics, and easy-to-understand reports to guide your investment decisions.</p>
                    </div>
                </div>
            </div>
        </section>
    </main>

    <footer class="container mt-4" style="padding: 20px 0; border-top: 1px solid var(--border-color); margin-top: 40px;">
        <p class="text-secondary" style="text-align: center;">© {{ date('Y') }} Stock Predictor | AI-Powered Financial Analysis Tool</p>
    </footer>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const navbarToggle = document.getElementById('navbar-toggle');
            const navbarMenu = document.getElementById('navbar-menu');
            
            if (navbarToggle && navbarMenu) {
                navbarToggle.addEventListener('click', function() {
                    navbarMenu.classList.toggle('active');
                });
            }
        });
    </script>
</body>
</html>