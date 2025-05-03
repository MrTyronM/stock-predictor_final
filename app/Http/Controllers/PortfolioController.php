<?php

namespace App\Http\Controllers;

use App\Models\Portfolio;
use App\Models\PortfolioItem;
use App\Models\Stock;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;

use Illuminate\Foundation\Auth\Access\AuthorizesRequests;

class PortfolioController extends Controller
{
    use AuthorizesRequests;
    public function index()
    {
        $portfolios = Auth::user()->portfolios;
        return view('portfolios.index', compact('portfolios'));
    }

    public function show(Portfolio $portfolio)
    {
        $this->authorize('view', $portfolio);
        
        $portfolio->load(['items.stock' => function($query) {
            $query->with(['predictions' => function($q) {
                $q->latest('prediction_date')->limit(1);
            }]);
        }]);
        
        return view('portfolios.show', compact('portfolio'));
    }

    public function create()
    {
        return view('portfolios.create');
    }

    public function store(Request $request)
    {
        $validated = $request->validate([
            'name' => 'required|string|max:255',
            'description' => 'nullable|string',
        ]);
        
        $portfolio = new Portfolio();
        $portfolio->user_id = Auth::id();
        $portfolio->name = $validated['name'];
        $portfolio->description = $validated['description'];
        $portfolio->save();
        
        return redirect()->route('portfolios.show', $portfolio)
            ->with('success', 'Portfolio created successfully!');
    }

    public function edit(Portfolio $portfolio)
    {
        $this->authorize('update', $portfolio);
        return view('portfolios.edit', compact('portfolio'));
    }

    public function update(Request $request, Portfolio $portfolio)
    {
        $this->authorize('update', $portfolio);
        
        $validated = $request->validate([
            'name' => 'required|string|max:255',
            'description' => 'nullable|string',
        ]);
        
        $portfolio->update($validated);
        
        return redirect()->route('portfolios.show', $portfolio)
            ->with('success', 'Portfolio updated successfully!');
    }

    public function destroy(Portfolio $portfolio)
    {
        $this->authorize('delete', $portfolio);
        
        $portfolio->delete();
        
        return redirect()->route('portfolios.index')
            ->with('success', 'Portfolio deleted successfully!');
    }
    
    public function addStock(Request $request, Portfolio $portfolio)
    {
        $this->authorize('update', $portfolio);
        
        $validated = $request->validate([
            'stock_id' => 'required|exists:stocks,id',
            'shares' => 'required_without:is_watchlist|integer|min:0',
            'purchase_price' => 'nullable|numeric|min:0',
            'purchase_date' => 'nullable|date',
            'is_watchlist' => 'sometimes|boolean',
        ]);
        
        $is_watchlist = isset($validated['is_watchlist']) && $validated['is_watchlist'];
        
        // Check if stock already exists in portfolio
        $existingItem = $portfolio->items()
            ->where('stock_id', $validated['stock_id'])
            ->first();
            
        if ($existingItem) {
            if ($is_watchlist) {
                $existingItem->update([
                    'is_watchlist' => true,
                    'shares' => 0,
                    'purchase_price' => null,
                    'purchase_date' => null,
                ]);
            } else {
                $existingItem->update([
                    'shares' => $validated['shares'],
                    'purchase_price' => $validated['purchase_price'] ?? null,
                    'purchase_date' => $validated['purchase_date'] ?? null,
                    'is_watchlist' => false,
                ]);
            }
        } else {
            $portfolio->items()->create([
                'stock_id' => $validated['stock_id'],
                'shares' => $is_watchlist ? 0 : ($validated['shares'] ?? 0),
                'purchase_price' => $is_watchlist ? null : ($validated['purchase_price'] ?? null),
                'purchase_date' => $is_watchlist ? null : ($validated['purchase_date'] ?? null),
                'is_watchlist' => $is_watchlist,
            ]);
        }
        
        $stockName = Stock::find($validated['stock_id'])->symbol;
        $action = $is_watchlist ? 'added to watchlist' : 'added to portfolio';
        
        return back()->with('success', "$stockName $action successfully!");
    }
    
    public function removeStock(Portfolio $portfolio, PortfolioItem $item)
    {
        $this->authorize('update', $portfolio);
        
        if ($item->portfolio_id != $portfolio->id) {
            abort(403, 'Unauthorized action.');
        }
        
        $stockName = $item->stock->symbol;
        $item->delete();
        
        return back()->with('success', "$stockName removed successfully!");
    }

    public function performance(Portfolio $portfolio)
    {
        // Verify ownership
        $this->authorize('view', $portfolio);
        
        // Load portfolio data
        $portfolio->load(['items.stock.stockData' => function($query) {
            $query->orderBy('date', 'desc');
        }, 'items.stock.predictions']);
        
        // Calculate performance metrics
        $metrics = $this->calculatePerformanceMetrics($portfolio);
        
        return view('portfolios.performance', compact('portfolio', 'metrics'));
    }

    private function calculatePerformanceMetrics($portfolio)
    {
        $metrics = [
            'total_value' => 0,
            'total_cost' => 0,
            'total_gain_loss' => 0,
            'total_gain_loss_percent' => 0,
            'items' => [],
            'historical_performance' => [],
            'allocation' => [],
            'prediction_summary' => [
                'buy' => 0,
                'hold' => 0,
                'sell' => 0
            ]
        ];
        
        // Calculate current performance
        foreach ($portfolio->items as $item) {
            if ($item->is_watchlist) {
                continue; // Skip watchlist items
            }
            
            $stockData = $item->stock->stockData;
            
            if ($stockData->isEmpty()) {
                continue;
            }
            
            $latestPrice = $stockData->first()->close;
            $purchasePrice = $item->purchase_price ?? $latestPrice;
            
            $currentValue = $latestPrice * $item->shares;
            $cost = $purchasePrice * $item->shares;
            $gainLoss = $currentValue - $cost;
            $gainLossPercent = $cost > 0 ? ($gainLoss / $cost) * 100 : 0;
            
            $metrics['total_value'] += $currentValue;
            $metrics['total_cost'] += $cost;
            
            // Track item-specific metrics
            $metrics['items'][$item->id] = [
                'stock_id' => $item->stock_id,
                'symbol' => $item->stock->symbol,
                'name' => $item->stock->name,
                'shares' => $item->shares,
                'purchase_price' => $purchasePrice,
                'current_price' => $latestPrice,
                'current_value' => $currentValue,
                'cost' => $cost,
                'gain_loss' => $gainLoss,
                'gain_loss_percent' => $gainLossPercent
            ];
            
            // Track allocation by stock
            $metrics['allocation'][$item->stock_id] = [
                'symbol' => $item->stock->symbol,
                'value' => $currentValue,
                'percent' => 0 // Will calculate after total is known
            ];
            
            // Track prediction summary
            if (!$item->stock->predictions->isEmpty()) {
                $prediction = $item->stock->predictions->first();
                $metrics['prediction_summary'][$prediction->recommendation]++;
            }
        }
        
        // Calculate total gain/loss
        $metrics['total_gain_loss'] = $metrics['total_value'] - $metrics['total_cost'];
        $metrics['total_gain_loss_percent'] = $metrics['total_cost'] > 0 ? 
            ($metrics['total_gain_loss'] / $metrics['total_cost']) * 100 : 0;
        
        // Calculate allocation percentages
        if ($metrics['total_value'] > 0) {
            foreach ($metrics['allocation'] as $stockId => $data) {
                $metrics['allocation'][$stockId]['percent'] = 
                    ($data['value'] / $metrics['total_value']) * 100;
            }
        }
        
        // Generate historical performance data (simulated for demonstration)
        // In a real app, you'd use actual historical purchase/sale records
        $startDate = now()->subMonths(6);
        $dates = [];
        $values = [];
        
        for ($date = $startDate; $date->lte(now()); $date->addDays(7)) {
            $dates[] = $date->format('Y-m-d');
            
            // Simulate historical value - replace with actual historical data in production
            $randomFactor = 0.95 + (mt_rand(0, 20) / 100); // Random factor between 0.95 and 1.15
            $value = $metrics['total_cost'] * $randomFactor;
            
            // Gradually approach current value
            $daysTotal = now()->diffInDays($startDate);
            $daysCurrent = $date->diffInDays($startDate);
            $progressFactor = $daysCurrent / $daysTotal;
            
            $value = $metrics['total_cost'] + (($metrics['total_value'] - $metrics['total_cost']) * $progressFactor * $randomFactor);
            
            $values[] = $value;
        }
        
        $metrics['historical_performance'] = [
            'dates' => $dates,
            'values' => $values
        ];
        
        return $metrics;
    }

    

    
}