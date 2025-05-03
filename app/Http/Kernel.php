<?php

namespace App\Http;

use Illuminate\Foundation\Http\Kernel as HttpKernel;
use Illuminate\Console\Scheduling\Schedule;

class Kernel extends HttpKernel
{
    // Other code and arrays...

    
    
    protected $routeMiddleware = [
        'admin' => \App\Http\Middleware\AdminMiddleware::class, // Ensures only admin users can access certain routes
    ];
    

    protected function schedule(Schedule $schedule)
    {
        // Update stock data daily at 6 PM
        $schedule->command('app:update-stock-data')->dailyAt('18:00');
        
        // Generate daily summary at 6:30 PM (after stock data is updated)
        $schedule->command('app:generate-daily-summary')->dailyAt('18:30');
        
        // Run at 8 PM daily after market close
        $schedule->command('app:update-stock-data')->dailyAt('20:00');

        // Fetch news twice daily
        $schedule->command('app:fetch-news')->twiceDaily(9, 16);


         // Update stock data daily at 6 AM
        $schedule->command('app:update-stock-data')->dailyAt('06:00');
        
        // Fetch news twice daily
    
        $schedule->command('app:fetch-news')->twiceDaily(9, 16);
    }
}