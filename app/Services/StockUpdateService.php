<?php

namespace App\Services;

use App\Models\Stock;
use App\Models\StockData;
use Illuminate\Support\Facades\Http;
use App\Events\StockPriceUpdated;
use Illuminate\Support\Facades\Log;

class StockUpdateService
{
    public function updateLiveStockData()
    {
        $stocks = Stock::all();
        
        foreach ($stocks as $stock) {
            try {
                // In a real app, you'd connect to a real-time market data API
                // For this example, we'll simulate by adding small random changes to the last price
                $latestData = StockData::where('stock_id', $stock->id)
                    ->orderBy('date', 'desc')
                    ->first();
                
                if (!$latestData) {
                    continue;
                }
                
                $lastPrice = $latestData->close;
                
                // Generate a small random change (-0.5% to +0.5%)
                $changePercent = (mt_rand(-50, 50) / 10000);
                $newPrice = $lastPrice * (1 + $changePercent);
                
                // For simulation purposes, create a "live" update
                $liveUpdate = [
                    'symbol' => $stock->symbol,
                    'price' => $newPrice,
                    'change' => $newPrice - $lastPrice,
                    'change_percent' => $changePercent * 100,
                    'volume' => mt_rand(1000, 10000),
                    'timestamp' => now()->timestamp
                ];
                
                // Broadcast the update via WebSockets
                event(new StockPriceUpdated($liveUpdate));
            } catch (\Exception $e) {
                Log::error("Error updating stock {$stock->symbol}: " . $e->getMessage());
                continue;
            }
        }
        
        return true;
    }
}