<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use App\Services\NewsService;

class FetchNewsCommand extends Command
{
    protected $signature = 'app:fetch-news';
    protected $description = 'Fetch financial news from external API';

    protected $newsService;

    public function __construct(NewsService $newsService)
    {
        parent::__construct();
        $this->newsService = $newsService;
    }

    public function handle()
    {
        $this->info('Fetching financial news...');
        
        $result = $this->newsService->fetchFinancialNews();
        
        if ($result) {
            $this->info('Successfully fetched financial news.');
        } else {
            $this->error('Failed to fetch financial news.');
        }
        
        $this->info('Fetching stock-specific news...');
        
        $stocks = \App\Models\Stock::all();
        $bar = $this->output->createProgressBar(count($stocks));
        $bar->start();
        
        foreach ($stocks as $stock) {
            $this->newsService->fetchStockSpecificNews($stock);
            $bar->advance();
        }
        
        $bar->finish();
        $this->newLine(2);
        
        $this->info('News fetching completed!');
        
        return Command::SUCCESS;
    }
}