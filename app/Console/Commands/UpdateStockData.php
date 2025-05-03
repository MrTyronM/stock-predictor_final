<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use App\Models\Stock;
use Illuminate\Support\Facades\Process;

class UpdateStockData extends Command
{
    protected $signature = 'app:update-stock-data';
    protected $description = 'Update stock data from Yahoo Finance';

    public function handle()
    {
        $stocks = Stock::all();
        
        if ($stocks->isEmpty()) {
            $this->error('No stocks found in the database');
            return Command::FAILURE;
        }
        
        $this->info('Starting to fetch updated data for ' . $stocks->count() . ' stocks...');
        
        $bar = $this->output->createProgressBar($stocks->count());
        $bar->start();
        
        foreach ($stocks as $stock) {
            $this->info("\nUpdating data for {$stock->symbol}...");
            
            // Use yesterday's date as the start date
            $startDate = now()->subDay()->format('Y-m-d');
            
            $scriptPath = base_path('ml/fetch_stock_data.py');
            $command = "python $scriptPath --stock_id {$stock->id} --symbol {$stock->symbol} --start_date $startDate";
            
            $process = Process::timeout(300)->run($command);
            
            if ($process->successful()) {
                $this->info("Successfully updated data for {$stock->symbol}");
            } else {
                $this->error("Failed to update data for {$stock->symbol}: " . $process->errorOutput());
            }
            
            $bar->advance();
        }
        
        $bar->finish();
        $this->newLine(2);
        $this->info('Stock data update completed!');
        
        return Command::SUCCESS;
    }
}