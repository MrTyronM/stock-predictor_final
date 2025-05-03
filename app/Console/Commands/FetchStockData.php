<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use App\Models\Stock;
use Illuminate\Support\Facades\Process;

class FetchStockData extends Command
{
    protected $signature = 'app:fetch-stock-data';
    protected $description = 'Fetch historical stock data for all stocks';

    public function handle()
    {
        $stocks = Stock::all();
        
        if ($stocks->isEmpty()) {
            $this->error('No stocks found in the database');
            return Command::FAILURE;
        }
        
        $this->info('Starting to fetch historical data for ' . $stocks->count() . ' stocks...');
        
        $bar = $this->output->createProgressBar($stocks->count());
        $bar->start();
        
        foreach ($stocks as $stock) {
            $this->info("\nFetching data for {$stock->symbol}...");
            
            $scriptPath = base_path('ml/fetch_stock_data.py');
            $command = "python $scriptPath --stock_id {$stock->id} --symbol {$stock->symbol}";
            
            $process = Process::timeout(300)->run($command);
            
            if ($process->successful()) {
                $this->info("Successfully fetched data for {$stock->symbol}");
            } else {
                $this->error("Failed to fetch data for {$stock->symbol}: " . $process->errorOutput());
            }
            
            $bar->advance();
        }
        
        $bar->finish();
        $this->newLine(2);
        $this->info('Stock data fetch completed!');
        
        return Command::SUCCESS;
    }
}
