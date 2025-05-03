<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Stock;
use App\Models\StockData;
use Carbon\Carbon;

class MarketController extends Controller
{
    /**
     * Display the market index page
     */
    public function index()
    {
        return view('market.index');
    }

    /**
     * Get market overview data
     *
     * @return array
     */
    public function getMarketOverview()
    {
        // In a real application, this would fetch data from a market API or database
        return [
            'marketIndex' => [
                'value' => 4832.75,
                'change' => 23.45,
                'percentChange' => 0.49
            ],
            'marketStatus' => 'open',
            'lastUpdated' => now()->format('Y-m-d H:i:s'),
            'tradingVolume' => 2538900000,
            'advancers' => 285,
            'decliners' => 218
        ];
    }

    /**
     * Get most active stocks
     *
     * @return array
     */
    public function getMostActiveStocks()
    {
        // Try to get real data first
        $stocks = Stock::take(5)->get();
        
        if ($stocks->count() > 0) {
            $activeStocks = [];
            
            foreach ($stocks as $stock) {
                $latestData = StockData::where('stock_id', $stock->id)
                    ->orderBy('date', 'desc')
                    ->first();
                
                if ($latestData) {
                    $previousData = StockData::where('stock_id', $stock->id)
                        ->where('date', '<', $latestData->date)
                        ->orderBy('date', 'desc')
                        ->first();
                    
                    $change = $previousData ? $latestData->close - $previousData->close : 0;
                    $percentChange = $previousData ? ($change / $previousData->close) * 100 : 0;
                    
                    $activeStocks[] = [
                        'symbol' => $stock->symbol,
                        'name' => $stock->name,
                        'lastPrice' => $latestData->close,
                        'change' => $change,
                        'percentChange' => $percentChange,
                        'volume' => $latestData->volume ?? rand(10000000, 100000000)
                    ];
                }
            }
            
            if (!empty($activeStocks)) {
                return $activeStocks;
            }
        }
        
        // Fallback to sample data if no real data found
        return [
            [
                'symbol' => 'AAPL',
                'name' => 'Apple Inc.',
                'lastPrice' => 175.42,
                'change' => 2.37,
                'percentChange' => 1.37,
                'volume' => 82345600
            ],
            [
                'symbol' => 'MSFT',
                'name' => 'Microsoft Corporation',
                'lastPrice' => 326.89,
                'change' => 4.12,
                'percentChange' => 1.28,
                'volume' => 32457800
            ],
            [
                'symbol' => 'AMZN',
                'name' => 'Amazon.com Inc.',
                'lastPrice' => 142.36,
                'change' => -1.24,
                'percentChange' => -0.86,
                'volume' => 45678900
            ],
            [
                'symbol' => 'GOOGL',
                'name' => 'Alphabet Inc.',
                'lastPrice' => 138.20,
                'change' => 1.75,
                'percentChange' => 1.28,
                'volume' => 28764500
            ],
            [
                'symbol' => 'META',
                'name' => 'Meta Platforms Inc.',
                'lastPrice' => 327.56,
                'change' => 5.32,
                'percentChange' => 1.65,
                'volume' => 18945600
            ]
        ];
    }

    /**
     * Get latest market predictions
     *
     * @return array
     */
    public function getLatestPredictions()
    {
        // Sample market predictions data
        // In a real application, this would fetch data from a service or database
        return [
            [
                'analyst' => 'Morgan Stanley',
                'target' => 'S&P 500',
                'prediction' => 'Bullish',
                'targetPrice' => 5100,
                'date' => Carbon::now()->subDays(2)->format('Y-m-d')
            ],
            [
                'analyst' => 'Goldman Sachs',
                'target' => 'NASDAQ',
                'prediction' => 'Neutral',
                'targetPrice' => 16250,
                'date' => Carbon::now()->subDays(3)->format('Y-m-d')
            ],
            [
                'analyst' => 'JP Morgan',
                'target' => 'Dow Jones',
                'prediction' => 'Bullish',
                'targetPrice' => 38500,
                'date' => Carbon::now()->subDays(1)->format('Y-m-d')
            ]
        ];
    }
}