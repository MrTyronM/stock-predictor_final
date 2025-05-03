<?php

namespace App\Http\Controllers;

use App\Models\Stock;
use Illuminate\Http\Request;

class PredictionController extends Controller
{
    /**
     * Display prediction details for a specific stock.
     *
     * @param  Stock  $stock
     * @return \Illuminate\Http\Response
     */
    public function show(Stock $stock)
    {
        // Get predictions for this stock
        $predictions = $stock->predictions()
            ->orderBy('prediction_date', 'asc')
            ->take(5)
            ->get();
            
        // Latest prediction for analysis data
        $latestPrediction = $predictions->last();
        
        // Calculate volatility (standard deviation of predicted prices)
        $volatility = 0;
        if ($predictions->count() > 1) {
            $prices = $predictions->pluck('predicted_price')->toArray();
            $mean = array_sum($prices) / count($prices);
            $squaredDifferences = array_map(function($price) use ($mean) {
                return pow($price - $mean, 2);
            }, $prices);
            $variance = array_sum($squaredDifferences) / count($squaredDifferences);
            $volatility = sqrt($variance);
            
            // Convert to percentage based on mean price
            $volatility = ($volatility / $mean) * 100;
        }
        
        // Set a default accuracy or calculate based on historical predictions
        $accuracy = 75.5; // Placeholder value
        
        // Risk level based on volatility
        $riskLevel = 'Medium';
        if ($volatility > 5) {
            $riskLevel = 'High';
        } elseif ($volatility < 2) {
            $riskLevel = 'Low';
        }
        
        return view('predictions.show', compact(
            'stock', 
            'predictions', 
            'latestPrediction',
            'volatility',
            'accuracy',
            'riskLevel'
        ));
    }
}