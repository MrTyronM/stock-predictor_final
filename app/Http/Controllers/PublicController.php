<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Stock;
use App\Models\StockData;
use App\Models\News;
use Carbon\Carbon;

class PublicController extends Controller
{
    public function index()
    {
        // Get top gainers and losers
        $gainers = [];
        $losers = [];
        
        $stocks = Stock::with(['stockData' => function($query) {
            $query->where('date', '>=', Carbon::now()->subDays(2)->format('Y-m-d'))
                ->orderBy('date', 'desc');
            }])->get();
        
            foreach ($stocks as $stock) {
                if ($stock->stockData->count() < 2) {
                    continue;
                }
                
                $latestData = $stock->stockData->first();
                $previousData = $stock->stockData->skip(1)->first();
                
                // Calculate daily change
                $change = $latestData->close - $previousData->close;
                $changePercent = ($change / $previousData->close) * 100;
                
                $stockInfo = [
                    'id' => $stock->id,
                    'symbol' => $stock->symbol,
                    'name' => $stock->name,
                    'price' => $latestData->close,
                    'change' => $change,
                    'change_percent' => $changePercent
                ];
                
                // Add to gainers or losers
                if ($changePercent > 0) {
                    $gainers[] = $stockInfo;
                } else {
                    $losers[] = $stockInfo;
                }
            }
            
            // Sort gainers and losers
            usort($gainers, function($a, $b) {
                return $b['change_percent'] <=> $a['change_percent'];
            });
            
            usort($losers, function($a, $b) {
                return $a['change_percent'] <=> $b['change_percent'];
            });
            
            // Take top 5 of each
            $gainers = array_slice($gainers, 0, 5);
            $losers = array_slice($losers, 0, 5);
            
            // Get latest news
            $latestNews = News::orderBy('published_date', 'desc')
                ->limit(5)
                ->get();
                
            return view('public.index', compact('gainers', 'losers', 'latestNews'));
        }
        
        public function stockDetails($symbol)
        {
            $stock = Stock::where('symbol', $symbol)->firstOrFail();
            
            // Get historical data
            $stockData = StockData::where('stock_id', $stock->id)
                ->orderBy('date', 'desc')
                ->limit(30)
                ->get()
                ->reverse();
                
            // Get related news
            $news = $stock->news()
                ->orderBy('published_date', 'desc')
                ->limit(5)
                ->get();
                
            return view('public.stock-details', compact('stock', 'stockData', 'news'));
        }
        
        public function marketOverview()
        {
            $marketController = new MarketController();
            $marketData = $marketController->getMarketOverview();
            $mostActiveStocks = $marketController->getMostActiveStocks();
            
            return view('public.market-overview', compact('marketData', 'mostActiveStocks'));
        }
    }
