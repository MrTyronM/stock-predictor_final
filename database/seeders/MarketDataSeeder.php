<?php

namespace Database\Seeders;

use Illuminate\Database\Seeder;
use App\Models\Stock;
use App\Models\StockData;
use App\Models\Prediction;
use Carbon\Carbon;

class MarketDataSeeder extends Seeder
{
    public function run()
    {
        // Sample stocks
        $stocks = [
            ['symbol' => 'AAPL', 'name' => 'Apple Inc.'],
            ['symbol' => 'MSFT', 'name' => 'Microsoft Corporation'],
            ['symbol' => 'GOOGL', 'name' => 'Alphabet Inc.'],
            ['symbol' => 'AMZN', 'name' => 'Amazon.com, Inc.'],
            ['symbol' => 'TSLA', 'name' => 'Tesla, Inc.'],
            ['symbol' => 'META', 'name' => 'Meta Platforms, Inc.'],
            ['symbol' => 'NFLX', 'name' => 'Netflix, Inc.'],
            ['symbol' => 'NVDA', 'name' => 'NVIDIA Corporation'],
            ['symbol' => 'DIS', 'name' => 'The Walt Disney Company'],
            ['symbol' => 'INTC', 'name' => 'Intel Corporation']
        ];
        
        foreach ($stocks as $stockData) {
            $stock = Stock::firstOrCreate(
                ['symbol' => $stockData['symbol']],
                ['name' => $stockData['name']]
            );
            
            // Generate 90 days of historical data
            $this->generateHistoricalData($stock);
            
            // Generate predictions
            $this->generatePredictions($stock);
        }
    }
    
    private function generateHistoricalData($stock)
    {
        // Starting price between $50 and $500
        $price = mt_rand(5000, 50000) / 100;
        
        for ($i = 90; $i >= 0; $i--) {
            $date = Carbon::now()->subDays($i)->format('Y-m-d');
            
            // Random daily change between -3% and +3%
            $changePercent = (mt_rand(-300, 300) / 10000);
            $price = max(1, $price * (1 + $changePercent));
            
            $open = $price * (1 + (mt_rand(-100, 100) / 10000));
            $high = max($open, $price) * (1 + (mt_rand(0, 200) / 10000));
            $low = min($open, $price) * (1 - (mt_rand(0, 200) / 10000));
            
            StockData::updateOrCreate(
                ['stock_id' => $stock->id, 'date' => $date],
                [
                    'open' => $open,
                    'high' => $high,
                    'low' => $low,
                    'close' => $price,
                    'volume' => mt_rand(1000000, 50000000)
                ]
            );
        }
    }
    
    private function generatePredictions($stock)
    {
        // Get latest stock price
        $latestData = StockData::where('stock_id', $stock->id)
            ->orderBy('date', 'desc')
            ->first();
            
        if (!$latestData) {
            return;
        }
        
        $currentPrice = $latestData->close;
        
        // Generate predictions for next 5 days
        for ($i = 1; $i <= 5; $i++) {
            $date = Carbon::now()->addDays($i)->format('Y-m-d');
            
            // Random prediction between -5% and +5%
            $changePercent = (mt_rand(-500, 500) / 10000);
            $predictedPrice = $currentPrice * (1 + $changePercent);
            
            // Determine recommendation
            $recommendation = 'hold';
            if ($changePercent > 0.01) {
                $recommendation = 'buy';
            } elseif ($changePercent < -0.01) {
                $recommendation = 'sell';
            }
            
            // Random confidence between 60% and 90%
            $confidence = mt_rand(60, 90);
            
            Prediction::updateOrCreate(
                ['stock_id' => $stock->id, 'prediction_date' => $date],
                [
                    'predicted_price' => $predictedPrice,
                    'recommendation' => $recommendation,
                    'confidence' => $confidence,
                    'created_at' => Carbon::now(),
                    'updated_at' => Carbon::now()
                ]
            );
        }
    }
}