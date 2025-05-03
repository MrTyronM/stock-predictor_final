<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use App\Models\Stock;
use App\Models\User;
use App\Models\Notification;
use Carbon\Carbon;

class GenerateDailySummary extends Command
{
    protected $signature = 'app:generate-daily-summary';
    protected $description = 'Generate daily stock summary and notify users';

    public function handle()
    {
        $this->info('Generating daily stock summary...');
        
        $stocks = Stock::with(['stockData' => function($query) {
            $query->orderBy('date', 'desc')->limit(2);
        }, 'predictions' => function($query) {
            $query->latest('prediction_date')->limit(1);
        }])->get();
        
        // Create notifications for significant movements
        $this->createNotifications($stocks);
        
        $this->info('Daily summary generated successfully!');
        
        return Command::SUCCESS;
    }
    
    private function createNotifications($stocks)
    {
        $this->info('Creating user notifications...');
        
        $usersWithPortfolios = User::whereHas('portfolios.items')->get();
        $strongBuyStocks = [];
        $strongSellStocks = [];
        
        foreach ($stocks as $stock) {
            if ($stock->stockData->count() < 2 || !$stock->predictions->count()) {
                continue;
            }
            
            $latestData = $stock->stockData->first();
            $previousData = $stock->stockData->last();
            $prediction = $stock->predictions->first();
            
            // Calculate daily change
            $dailyChange = (($latestData->close - $previousData->close) / $previousData->close) * 100;
            
            // Calculate predicted change
            $predictedChange = (($prediction->predicted_price - $latestData->close) / $latestData->close) * 100;
            
            // Identify significant movements: >3% daily change or >5% predicted change
            if (abs($dailyChange) > 3 || $predictedChange > 5) {
                // Find users who have this stock in their portfolio/watchlist
                foreach ($usersWithPortfolios as $user) {
                    $userHasStock = false;
                    
                    foreach ($user->portfolios as $portfolio) {
                        if ($portfolio->items->where('stock_id', $stock->id)->count() > 0) {
                            $userHasStock = true;
                            break;
                        }
                    }
                    
                    if ($userHasStock) {
                        $title = "";
                        $message = "";
                        $type = "info";
                        
                        if (abs($dailyChange) > 3) {
                            // Significant daily movement
                            $direction = $dailyChange > 0 ? "up" : "down";
                            $title = "{$stock->symbol} moved {$direction} by " . number_format(abs($dailyChange), 1) . "%";
                            $message = "{$stock->name} closed at \${$latestData->close} today, which is a " . number_format(abs($dailyChange), 1) . "% " . ($dailyChange > 0 ? "increase" : "decrease") . " from yesterday.";
                            $type = $dailyChange > 0 ? "success" : "danger";
                        } else if ($predictedChange > 5) {
                            // Strong buy signal
                            $title = "Strong {$prediction->recommendation} signal for {$stock->symbol}";
                            $message = "Our model predicts a " . number_format($predictedChange, 1) . "% increase for {$stock->name} with {$prediction->confidence}% confidence.";
                            $type = "success";
                        } else if ($predictedChange < -5) {
                            // Strong sell signal
                            $title = "Strong {$prediction->recommendation} signal for {$stock->symbol}";
                            $message = "Our model predicts a " . number_format(abs($predictedChange), 1) . "% decrease for {$stock->name} with {$prediction->confidence}% confidence.";
                            $type = "danger";
                        }
                        
                        // Create notification
                        Notification::create([
                            'user_id' => $user->id,
                            'stock_id' => $stock->id,
                            'title' => $title,
                            'message' => $message,
                            'type' => $type
                        ]);
                        
                        $this->info("Created notification for user {$user->id} about {$stock->symbol}");
                    }
                }
                
                // Track strong buy/sell stocks for daily summary
                if ($prediction->recommendation === 'buy' && $predictedChange > 7) {
                    $strongBuyStocks[] = $stock;
                } else if ($prediction->recommendation === 'sell' && $predictedChange < -7) {
                    $strongSellStocks[] = $stock;
                }
            }
        }
        
        // Create daily summary notification for all users
        if (count($strongBuyStocks) > 0 || count($strongSellStocks) > 0) {
            $users = User::all();
            
            foreach ($users as $user) {
                $title = "Daily Stock Market Summary";
                $message = "Today's market summary: ";
                
                if (count($strongBuyStocks) > 0) {
                    $message .= count($strongBuyStocks) . " strong buy recommendations including ";
                    $message .= implode(", ", array_slice(array_map(function($stock) {
                        return $stock->symbol;
                    }, $strongBuyStocks), 0, 3));
                    
                    if (count($strongBuyStocks) > 3) {
                        $message .= " and more";
                    }
                    
                    if (count($strongSellStocks) > 0) {
                        $message .= ". ";
                    }
                }
                
                if (count($strongSellStocks) > 0) {
                    $message .= count($strongSellStocks) . " strong sell signals including ";
                    $message .= implode(", ", array_slice(array_map(function($stock) {
                        return $stock->symbol;
                    }, $strongSellStocks), 0, 3));
                    
                    if (count($strongSellStocks) > 3) {
                        $message .= " and more";
                    }
                }
                
                $message .= ". Check your portfolio for details.";
                
                Notification::create([
                    'user_id' => $user->id,
                    'title' => $title,
                    'message' => $message,
                    'type' => 'info'
                ]);
                
                $this->info("Created daily summary notification for user {$user->id}");
            }
        }
    }
}