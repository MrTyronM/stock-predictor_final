<?php

namespace App\Http\Controllers;

use App\Http\Controllers\MarketController;
use Illuminate\Http\Request;

class ReactController extends Controller
{
    public function marketDashboard()
    {
        // Get market data from MarketController methods
        $marketController = new MarketController();
        $marketData = $marketController->getMarketOverview();
        $mostActiveStocks = $marketController->getMostActiveStocks();
        $latestPredictions = $marketController->getLatestPredictions();
        
        // Prepare data for the view
        $dashboardData = [
            'marketData' => $marketData,
            'mostActiveStocks' => $mostActiveStocks,
            'latestPredictions' => $latestPredictions
        ];
        
        // Return directly styled HTML
        return response()->view('direct.market-dashboard', [
            'dashboardData' => $dashboardData
        ]);
    }
}