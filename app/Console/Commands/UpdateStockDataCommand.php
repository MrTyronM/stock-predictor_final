<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use App\Models\Stock;
use App\Models\StockData;
use Carbon\Carbon;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

class UpdateStockDataCommand extends Command
{
    protected $signature = 'app:update-stock-data';
    protected $description = 'Update stock data for all stocks';

    public function handle()
    {
        $this->info('Starting stock data update...');
        
        $stocks = Stock::all();
        $updated = 0;
        $errors = 0;
        
        $bar = $this->output->createProgressBar(count($stocks));
        $bar->start();
        
        foreach ($stocks as $stock) {
            try {
                // In a real app, this would connect to a market data API
                // For this example, we'll generate simulated data
                $this->updateStockWithSimulatedData($stock);
                $updated++;
            } catch (\Exception $e) {
                Log::error("Error updating data for {$stock->symbol}: " . $e->getMessage());
                $errors++;
            }
            
            $bar->advance();
        }
        
        $bar->finish();
        $this->newLine(2);
        
        $this->info("Stock data update completed!");
        $this->info("Updated: {$updated} stocks");
        $this->info("Errors: {$errors} stocks");
        
        return Command::SUCCESS;
    }
    
    private function updateStockWithSimulatedData($stock)
    {
        // Get the latest data
        $latestData = StockData::where('stock_id', $stock->id)
            ->orderBy('date', 'desc')
            ->first();
            
        if (!$latestData) {
            $this->error("No historical data found for {$stock->symbol}");
            return;
        }
        
        // Generate a new date (next trading day)
        $lastDate = Carbon::parse($latestData->date);
        $newDate = $this->getNextTradingDay($lastDate);
        
        // Skip if we already have data for this date
        if (StockData::where('stock_id', $stock->id)->where('date', $newDate->format('Y-m-d'))->exists()) {
            return;
        }
        
        // Generate realistic price movement (between -3% and +3%)
        $changePercent = (mt_rand(-300, 300) / 10000);
        $lastClose = $latestData->close;
        
        $open = $lastClose * (1 + (mt_rand(-100, 100) / 10000));
        $close = $lastClose * (1 + $changePercent);
        $high = max($open, $close) * (1 + (mt_rand(0, 200) / 10000));
        $low = min($open, $close) * (1 - (mt_rand(0, 200) / 10000));
        $volume = mt_rand(1000000, 20000000);
        
        // Create new stock data record
        StockData::create([
            'stock_id' => $stock->id,
            'date' => $newDate->format('Y-m-d'),
            'open' => $open,
            'high' => $high,
            'low' => $low,
            'close' => $close,
            'volume' => $volume
        ]);
    }
    
    private function getNextTradingDay($date)
    {
        $nextDay = clone $date;
        $nextDay->addDay();
        
        // Skip weekends
        while ($nextDay->isWeekend()) {
            $nextDay->addDay();
        }
        
        return $nextDay;
    }
}