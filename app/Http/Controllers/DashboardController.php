<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Stock;
use App\Models\Prediction;
use App\Models\StockData;

class DashboardController extends Controller
{
    public function index()
    {
        $stocks = Stock::with([
            'predictions' => function($query) {
                $query->latest('prediction_date')->limit(1);
            },
            'stockData' => function($query) {
                $query->latest('date')->limit(2);
            }
        ])->get();
        
        // Get top performers
        $topPerformers = $this->getTopPerformers();
        
        return view('dashboard', compact('stocks', 'topPerformers'));
    }
    
    private function getTopPerformers()
    {
        $stocks = Stock::with(['stockData' => function($query) {
            $query->latest('date')->limit(2);
        }])->get();
        
        $performanceData = [];
        
        foreach ($stocks as $stock) {
            $latestData = $stock->stockData->sortByDesc('date')->take(2);
            
            if ($latestData->count() >= 2) {
                $current = $latestData->first()->close;
                $previous = $latestData->last()->close;
                $change = $current - $previous;
                $changePercent = ($change / $previous) * 100;
                
                $performanceData[] = [
                    'stock' => $stock,
                    'current' => $current,
                    'change' => $change,
                    'changePercent' => $changePercent
                ];
            }
        }
        
        // Sort by percent change (highest first)
        usort($performanceData, function($a, $b) {
            return $b['changePercent'] <=> $a['changePercent'];
        });
        
        return array_slice($performanceData, 0, 5);
    }
}