@extends('layouts.app')

@section('content')
<div class="container">
    <h2 class="mb-4">Strategy Backtesting</h2>
    
    @if(session('error'))
        <div class="alert alert-danger">
            {{ session('error') }}
        </div>
    @endif
    
    <div class="card">
        <div class="card-header">
            <h3 class="card-title">Backtest Parameters</h3>
        </div>
        <div class="card-body">
            <form action="{{ route('backtest.run') }}" method="POST">
                @csrf
                
                <div class="row mb-3">
                    <div class="col-md-6">
                        <div class="form-group">
                            <label for="stock_id">Select Stock</label>
                            <select name="stock_id" id="stock_id" class="form-control" required>
                                <option value="">-- Select Stock --</option>
                                @foreach($stocks as $stock)
                                    <option value="{{ $stock->id }}">{{ $stock->symbol }} - {{ $stock->name }}</option>
                                @endforeach
                            </select>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="form-group">
                            <label for="strategy">Trading Strategy</label>
                            <select name="strategy" id="strategy" class="form-control" required>
                                <option value="">-- Select Strategy --</option>
                                <option value="sma_crossover">Moving Average Crossover</option>
                                <option value="momentum">Momentum</option>
                                <option value="rsi">RSI (Relative Strength Index)</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div class="row mb-3">
                    <div class="col-md-4">
                        <div class="form-group">
                            <label for="start_date">Start Date</label>
                            <input type="date" name="start_date" id="start_date" class="form-control" 
                                   value="{{ \Carbon\Carbon::now()->subYear()->format('Y-m-d') }}" required>
                        </div>
                    </div>
                    
                    <div class="col-md-4">
                        <div class="form-group">
                            <label for="end_date">End Date</label>
                            <input type="date" name="end_date" id="end_date" class="form-control" 
                                   value="{{ \Carbon\Carbon::now()->format('Y-m-d') }}" required>
                        </div>
                    </div>
                    
                    <div class="col-md-4">
                        <div class="form-group">
                            <label for="initial_capital">Initial Capital ($)</label>
                            <input type="number" name="initial_capital" id="initial_capital" class="form-control" 
                                   value="10000" min="100" step="100" required>
                        </div>
                    </div>
                </div>
                
                <div class="strategy-info mb-4">
                    <div id="sma-info" class="d-none">
                        <h4>Moving Average Crossover Strategy</h4>
                        <p>This strategy generates buy signals when a short-term moving average crosses above a long-term moving average, and sell signals when it crosses below.</p>
                        <ul>
                            <li>Short-term MA: 20 days</li>
                            <li>Long-term MA: 50 days</li>
                        </ul>
                    </div>
                    
                    <div id="momentum-info" class="d-none">
                        <h4>Momentum Strategy</h4>
                        <p>This strategy buys stocks that have shown strong positive performance over a recent period, based on the idea that stocks that have performed well will continue to do so.</p>
                        <ul>
                            <li>Lookback period: 14 days</li>
                            <li>Momentum threshold: 3%</li>
                            <li>Hold period: 5 days</li>
                        </ul>
                    </div>
                    
                    <div id="rsi-info" class="d-none">
                        <h4>RSI (Relative Strength Index) Strategy</h4>
                        <p>This strategy uses the RSI indicator to identify overbought and oversold conditions. It buys when the RSI crosses above the oversold threshold and sells when it crosses above the overbought threshold.</p>
                        <ul>
                            <li>RSI Period: 14 days</li>
                            <li>Oversold threshold: 30</li>
                            <li>Overbought threshold: 70</li>
                        </ul>
                    </div>
                </div>
                
                <button type="submit" class="btn btn-primary">Run Backtest</button>
            </form>
        </div>
    </div>
</div>

@endsection

@push('scripts')
<script>
    document.addEventListener('DOMContentLoaded', function() {
        const strategySelect = document.getElementById('strategy');
        const infoElements = {
            'sma_crossover': document.getElementById('sma-info'),
            'momentum': document.getElementById('momentum-info'),
            'rsi': document.getElementById('rsi-info')
        };
        
        // Show/hide strategy info based on selection
        strategySelect.addEventListener('change', function() {
            // Hide all
            Object.values(infoElements).forEach(el => el.classList.add('d-none'));
            
            // Show selected
            const selected = strategySelect.value;
            if (selected && infoElements[selected]) {
                infoElements[selected].classList.remove('d-none');
            }
        });
    });
</script>
@endpush