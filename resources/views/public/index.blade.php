@extends('layouts.public')

@section('title', ' - Home')

@section('content')
<div class="container">
    <div class="card mb-4">
        <div class="card-body">
            <h1 class="mb-3">Welcome to Stock Prediction Platform</h1>
            <p class="lead">Track stocks, analyze market trends, and get AI-powered predictions.</p>
            <hr>
            <p>Join thousands of users making smarter investment decisions with our machine learning algorithms.</p>
            <a class="btn btn-primary" href="{{ route('register') }}" role="button">Sign Up Now</a>
            <a class="btn btn-secondary ms-2" href="{{ route('login') }}" role="button">Login</a>
        </div>
    </div>
    
    <div class="row mb-4">
        <div class="col-md-6">
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Today's Top Gainers</h3>
                </div>
                <div class="card-body">
                    @if(isset($gainers) && count($gainers) > 0)
                        <div class="table-container">
                            <table class="table">
                                <thead>
                                    <tr>
                                        <th>Symbol</th>
                                        <th>Price</th>
                                        <th>Change</th>
                                        <th></th>
                                    </tr>
                                </thead>
                                <tbody>
                                    @foreach($gainers as $stock)
                                        <tr>
                                            <td>{{ $stock['symbol'] }}</td>
                                            <td>${{ number_format($stock['price'], 2) }}</td>
                                            <td class="market-up">
                                                +{{ number_format($stock['change'], 2) }} (+{{ number_format($stock['change_percent'], 2) }}%)
                                            </td>
                                            <td>
                                                <a href="{{ route('public.stock.details', $stock['symbol']) }}" class="btn btn-primary btn-sm">Details</a>
                                            </td>
                                        </tr>
                                    @endforeach
                                </tbody>
                            </table>
                        </div>
                    @else
                        <p>No gainers data available.</p>
                    @endif
                </div>
            </div>
        </div>
        
        <div class="col-md-6">
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Today's Top Losers</h3>
                </div>
                <div class="card-body">
                    @if(isset($losers) && count($losers) > 0)
                        <div class="table-container">
                            <table class="table">
                                <thead>
                                    <tr>
                                        <th>Symbol</th>
                                        <th>Price</th>
                                        <th>Change</th>
                                        <th></th>
                                    </tr>
                                </thead>
                                <tbody>
                                    @foreach($losers as $stock)
                                        <tr>
                                            <td>{{ $stock['symbol'] }}</td>
                                            <td>${{ number_format($stock['price'], 2) }}</td>
                                            <td class="market-down">
                                                {{ number_format($stock['change'], 2) }} ({{ number_format($stock['change_percent'], 2) }}%)
                                            </td>
                                            <td>
                                                <a href="{{ route('public.stock.details', $stock['symbol']) }}" class="btn btn-primary btn-sm">Details</a>
                                            </td>
                                        </tr>
                                    @endforeach
                                </tbody>
                            </table>
                        </div>
                    @else
                        <p>No losers data available.</p>
                    @endif
                </div>
            </div>
        </div>
    </div>
    
    <div class="card mb-4">
        <div class="card-header">
            <h3 class="card-title">Latest Market News</h3>
        </div>
        <div class="card-body">
            @if(isset($latestNews) && $latestNews->count() > 0)
                <div class="row">
                    @foreach($latestNews as $news)
                        <div class="col-md-6 mb-3">
                            <div class="card">
                                <div class="card-body">
                                    <h5 class="card-title">{{ $news->title }}</h5>
                                    <h6 class="mb-2 text-secondary">{{ $news->source }} • {{ $news->published_date->format('M d, Y') }}</h6>
                                    <p>{{ Str::limit($news->content, 150) }}</p>
                                    @if($news->url)
                                        <a href="{{ $news->url }}" target="_blank" class="text-accent">Read More</a>
                                    @endif
                                </div>
                                @if($news->stocks->count() > 0)
                                    <div class="card-footer">
                                        <small>Related stocks: 
                                            @foreach($news->stocks as $stock)
                                                <a href="{{ route('public.stock.details', $stock->symbol) }}">{{ $stock->symbol }}</a>{{ !$loop->last ? ', ' : '' }}
                                            @endforeach
                                        </small>
                                    </div>
                                @endif
                            </div>
                        </div>
                    @endforeach
                </div>
                
                <div class="text-center mt-3">
                    <a href="{{ route('public.market.overview') }}" class="btn btn-primary">View Market Overview</a>
                </div>
            @else
                <p>No news available.</p>
            @endif
        </div>
    </div>
    
    <div class="card">
        <div class="card-header">
            <h3 class="card-title">Platform Features</h3>
        </div>
        <div class="card-body">
            <div class="row">
                <div class="col-md-4 mb-4">
                    <div class="text-center">
                        <i class="fas fa-chart-line fa-3x mb-3 text-accent"></i>
                        <h4>Advanced Analytics</h4>
                        <p>Track and analyze stock performance with technical indicators.</p>
                    </div>
                </div>
                
                <div class="col-md-4 mb-4">
                    <div class="text-center">
                        <i class="fas fa-brain fa-3x mb-3 text-accent"></i>
                        <h4>AI Predictions</h4>
                        <p>Get price predictions powered by machine learning algorithms.</p>
                    </div>
                </div>
                
                <div class="col-md-4 mb-4">
                    <div class="text-center">
                        <i class="fas fa-wallet fa-3x mb-3 text-accent"></i>
                        <h4>Portfolio Management</h4>
                        <p>Track your investments and analyze portfolio performance.</p>
                    </div>
                </div>
                
                <div class="col-md-4 mb-4">
                    <div class="text-center">
                        <i class="fas fa-bell fa-3x mb-3 text-accent"></i>
                        <h4>Alerts & Notifications</h4>
                        <p>Get notified about important price movements and predictions.</p>
                    </div>
                </div>
                
                <div class="col-md-4 mb-4">
                    <div class="text-center">
                        <i class="fas fa-newspaper fa-3x mb-3 text-accent"></i>
                        <h4>Market News</h4>
                        <p>Stay informed with the latest financial news and updates.</p>
                    </div>
                </div>
                
                <div class="col-md-4 mb-4">
                    <div class="text-center">
                        <i class="fas fa-robot fa-3x mb-3 text-accent"></i>
                        <h4>Strategy Backtesting</h4>
                        <p>Test trading strategies on historical data before investing.</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
@endsection