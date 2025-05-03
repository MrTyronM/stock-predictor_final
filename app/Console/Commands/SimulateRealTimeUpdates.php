<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use App\Services\StockUpdateService;

class SimulateRealTimeUpdates extends Command
{
    protected $signature = 'app:simulate-realtime';
    protected $description = 'Simulate real-time stock updates';

    protected $updateService;

    public function __construct(StockUpdateService $updateService)
    {
        parent::__construct();
        $this->updateService = $updateService;
    }

    public function handle()
    {
        $this->info('Starting real-time updates simulation...');
        
        // Loop for a specified time (e.g., 1 hour)
        $endTime = now()->addHour();
        
        while (now()->lt($endTime)) {
            $this->updateService->updateLiveStockData();
            
            // Wait a few seconds between updates
            sleep(5);
            
            // Show progress
            $this->output->write('.');
        }
        
        $this->newLine();
        $this->info('Real-time simulation completed!');
        
        return Command::SUCCESS;
    }
}