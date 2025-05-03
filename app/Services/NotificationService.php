<?php

namespace App\Services;

use App\Models\Notification;
use App\Models\Stock;
use App\Models\User;
use App\Models\Portfolio;
use App\Models\Watchlist;

class NotificationService
{
    /**
     * Create a stock price alert notification for users who have this stock
     * in their portfolios or watchlists
     */
    public function createPriceAlertNotification(Stock $stock, $priceChange, $percentChange)
    {
        // Determine notification type based on percentage change
        $type = 'info';
        if (abs($percentChange) > 5) {
            $type = $percentChange > 0 ? 'success' : 'danger';
        } else if (abs($percentChange) > 2) {
            $type = $percentChange > 0 ? 'info' : 'warning';
        }
        
        $title = "{$stock->symbol} Price Alert";
        $changeDirection = $percentChange > 0 ? 'up' : 'down';
        $message = "{$stock->symbol} ({$stock->name}) has gone {$changeDirection} by " . 
                   abs($percentChange) . "% ($" . number_format(abs($priceChange), 2) . ").";
        
        // Find users who have this stock in their portfolios or watchlists
        $portfolioUsers = Portfolio::whereHas('items', function($query) use ($stock) {
            $query->where('stock_id', $stock->id);
        })->pluck('user_id')->toArray();
        
        $watchlistUsers = Watchlist::whereHas('stocks', function($query) use ($stock) {
            $query->where('id', $stock->id);
        })->pluck('user_id')->toArray();
        
        $userIds = array_unique(array_merge($portfolioUsers, $watchlistUsers));
        
        // Create notification for each user
        foreach ($userIds as $userId) {
            Notification::create([
                'user_id' => $userId,
                'stock_id' => $stock->id,
                'title' => $title,
                'message' => $message,
                'type' => $type
            ]);
        }
    }
    
    /**
     * Create a prediction notification for users following this stock
     */
    public function createPredictionNotification(Stock $stock, $predictedPrice, $currentPrice, $recommendation, $confidence)
    {
        $percentChange = (($predictedPrice - $currentPrice) / $currentPrice) * 100;
        
        // Determine notification type based on recommendation
        $type = 'info';
        if ($recommendation == 'buy') {
            $type = 'success';
        } else if ($recommendation == 'sell') {
            $type = 'danger';
        }
        
        $title = "New {$stock->symbol} Prediction";
        $changeDirection = $percentChange > 0 ? 'increase' : 'decrease';
        $message = "Our AI model predicts {$stock->symbol} will {$changeDirection} by " . 
                   abs(number_format($percentChange, 2)) . "% to $" . number_format($predictedPrice, 2) . 
                   " within the next trading period. Recommendation: " . strtoupper($recommendation) . 
                   " (Confidence: {$confidence}%)";
        
        // Find users who have this stock in their watchlists
        $watchlistUsers = Watchlist::whereHas('stocks', function($query) use ($stock) {
            $query->where('id', $stock->id);
        })->pluck('user_id')->toArray();
        
        // Create notification for each user
        foreach ($watchlistUsers as $userId) {
            Notification::create([
                'user_id' => $userId,
                'stock_id' => $stock->id,
                'title' => $title,
                'message' => $message,
                'type' => $type
            ]);
        }
    }
    
    /**
     * Create a general system notification for all users or specific users
     */
    public function createSystemNotification($title, $message, $type = 'info', $userIds = null)
    {
        if ($userIds === null) {
            // Send to all users
            $users = User::all();
            foreach ($users as $user) {
                Notification::create([
                    'user_id' => $user->id,
                    'title' => $title,
                    'message' => $message,
                    'type' => $type
                ]);
            }
        } else {
            // Send to specific users
            foreach ($userIds as $userId) {
                Notification::create([
                    'user_id' => $userId,
                    'title' => $title,
                    'message' => $message,
                    'type' => $type
                ]);
            }
        }
    }
}