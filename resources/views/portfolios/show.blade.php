@extends('layouts.app')

@section('content')
    <div class="mb-3">
        <a href="{{ route('portfolios.index') }}" class="btn btn-secondary">&larr; Back to Portfolios</a>
    </div>

    <div class="d-flex justify-content-between align-items-center mb-4">
        <h2>{{ $portfolio->name }}</h2>
        <div class="d-flex gap-2">
            <a href="{{ route('portfolios.performance', $portfolio) }}" class="btn btn-primary">View Performance Analysis</a>
            <a href="{{ route('portfolios.edit', $portfolio) }}" class="btn btn-secondary">Edit Portfolio</a>
            <form action="{{ route('portfolios.destroy', $portfolio) }}" method="POST" class="d-inline">
                @csrf
                @method('DELETE')
                <button type="submit" class="btn btn-danger" onclick="return confirm('Are you sure you want to delete this portfolio?')">Delete</button>
            </form>
        </div>
    </div>

    @if($portfolio->description)
        <div class="card mb-4">
            <div class="card-body">
                <p>{{ $portfolio->description }}</p>
            </div>
        </div>
    @endif

    {{-- Add Stock Section --}}
    <div class="card mb-4">
        <div class="card-header">
            <h3 class="card-title">Add Stock</h3>
        </div>
        <div class="card-body">
            <form action="{{ route('portfolios.add-stock', $portfolio) }}" method="POST">
                @csrf
                <div class="form-group">
                    <label for="stock_id" class="form-label">Select Stock</label>
                    <select id="stock_id" name="stock_id" class="form-control" required>
                        <option value="">-- Select a stock --</option>
                        @foreach(\App\Models\Stock::orderBy('symbol')->get() as $stock)
                            <option value="{{ $stock->id }}">{{ $stock->symbol }} - {{ $stock->name }}</option>
                        @endforeach
                    </select>
                </div>

                <div class="form-check mb-3">
                    <input class="form-check-input" type="checkbox" id="is_watchlist" name="is_watchlist" value="1" onchange="toggleWatchlistMode(this.checked)">
                    <label class="form-check-label" for="is_watchlist">Add to watchlist only (no shares)</label>
                </div>

                <div id="investment-details">
                    <div class="form-group">
                        <label for="shares" class="form-label">Number of Shares</label>
                        <input type="number" id="shares" name="shares" class="form-control" min="0" step="1" value="0">
                    </div>

                    <div class="form-group">
                        <label for="purchase_price" class="form-label">Purchase Price (Optional)</label>
                        <input type="number" id="purchase_price" name="purchase_price" class="form-control" min="0" step="0.01">
                    </div>

                    <div class="form-group">
                        <label for="purchase_date" class="form-label">Purchase Date (Optional)</label>
                        <input type="date" id="purchase_date" name="purchase_date" class="form-control">
                    </div>
                </div>

                <button type="submit" class="btn btn-primary">Add Stock</button>
            </form>
        </div>
    </div>

    {{-- Investments Table --}}
    <h3 class="mb-3">Portfolio Stocks</h3>
    <div class="card mb-4">
        <div class="card-header">
            <h3 class="card-title">Investments</h3>
        </div>
        <div class="card-body">
            @if($portfolio->items->where('is_watchlist', false)->count() > 0)
                <div class="table-container">
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Name</th>
                                <th>Shares</th>
                                <th>Purchase Price</th>
                                <th>Current Price</th>
                                <th>Current Value</th>
                                <th>Gain/Loss</th>
                                <th>Prediction</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            @foreach($portfolio->items->where('is_watchlist', false) as $item)
                                @php
                                    $currentPrice = $item->stock->stockData->sortByDesc('date')->first()->close ?? 0;
                                    $purchasePrice = $item->purchase_price ?? $currentPrice;
                                    $currentValue = $currentPrice * $item->shares;
                                    $purchaseValue = $purchasePrice * $item->shares;
                                    $gainLoss = $currentValue - $purchaseValue;
                                    $gainLossPercent = $purchaseValue > 0 ? ($gainLoss / $purchaseValue) * 100 : 0;
                                    $prediction = $item->stock->predictions->first();
                                @endphp
                                <tr>
                                    <td>{{ $item->stock->symbol }}</td>
                                    <td>{{ $item->stock->name }}</td>
                                    <td>{{ $item->shares }}</td>
                                    <td>${{ number_format($purchasePrice, 2) }}</td>
                                    <td>${{ number_format($currentPrice, 2) }}</td>
                                    <td>${{ number_format($currentValue, 2) }}</td>
                                    <td class="{{ $gainLoss >= 0 ? 'text-success' : 'text-danger' }}">
                                        ${{ number_format($gainLoss, 2) }} ({{ number_format($gainLossPercent, 2) }}%)
                                    </td>
                                    <td>
                                        @if($prediction)
                                            <span class="prediction-recommendation recommendation-{{ $prediction->recommendation }}">
                                                {{ ucfirst($prediction->recommendation) }}
                                            </span>
                                        @else
                                            N/A
                                        @endif
                                    </td>
                                    <td>
                                        <form action="{{ route('portfolios.remove-stock', [$portfolio, $item]) }}" method="POST">
                                            @csrf
                                            @method('DELETE')
                                            <button type="submit" class="btn btn-danger btn-sm" onclick="return confirm('Are you sure you want to remove this stock?')">Remove</button>
                                        </form>
                                    </td>
                                </tr>
                            @endforeach
                        </tbody>
                    </table>
                </div>
            @else
                <p>No investments added to this portfolio yet.</p>
            @endif
        </div>
    </div>

    {{-- Watchlist Table --}}
    <div class="card">
        <div class="card-header">
            <h3 class="card-title">Watchlist</h3>
        </div>
        <div class="card-body">
            @if($portfolio->items->where('is_watchlist', true)->count() > 0)
                <div class="table-container">
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Name</th>
                                <th>Current Price</th>
                                <th>Prediction</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            @foreach($portfolio->items->where('is_watchlist', true) as $item)
                                @php
                                    $currentPrice = $item->stock->stockData->sortByDesc('date')->first()->close ?? 0;
                                    $prediction = $item->stock->predictions->first();
                                @endphp
                                <tr>
                                    <td>{{ $item->stock->symbol }}</td>
                                    <td>{{ $item->stock->name }}</td>
                                    <td>${{ number_format($currentPrice, 2) }}</td>
                                    <td>
                                        @if($prediction)
                                            <span class="prediction-recommendation recommendation-{{ $prediction->recommendation }}">
                                                {{ ucfirst($prediction->recommendation) }}
                                            </span>
                                        @else
                                            N/A
                                        @endif
                                    </td>
                                    <td>
                                        <form action="{{ route('portfolios.remove-stock', [$portfolio, $item]) }}" method="POST">
                                            @csrf
                                            @method('DELETE')
                                            <button type="submit" class="btn btn-danger btn-sm" onclick="return confirm('Are you sure you want to remove this stock?')">Remove</button>
                                        </form>
                                    </td>
                                </tr>
                            @endforeach
                        </tbody>
                    </table>
                </div>
            @else
                <p>No stocks added to watchlist yet.</p>
            @endif
        </div>
    </div>

    <script>
        function toggleWatchlistMode(isWatchlist) {
            const investmentDetails = document.getElementById('investment-details');
            investmentDetails.style.display = isWatchlist ? 'none' : 'block';
        }
    </script>
@endsection
