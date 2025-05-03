<?php

namespace App\Http\Controllers;

use App\Models\Stock;
use App\Models\StockData;
use Illuminate\Http\Request;

class StockController extends Controller
{
    public function index()
    {
        $stocks = Stock::all();
        return view('stocks.index', compact('stocks'));
    }
    
    public function show(Stock $stock)
    {
        $stockData = StockData::where('stock_id', $stock->id)
            ->orderBy('date', 'desc')
            ->take(100)  // Get more data for indicators
            ->get()
            ->reverse();
            
        // Calculate technical indicators
        $technicalIndicators = $this->calculateTechnicalIndicators($stockData);
            
        return view('stocks.show', compact('stock', 'stockData', 'technicalIndicators'));
    }
    
    public function compare(Request $request)
    {
        $stockIds = $request->input('stocks', []);
        
        if (empty($stockIds)) {
            return redirect()->route('stocks.index')
                ->with('info', 'Please select stocks to compare.');
        }
        
        $stocks = Stock::whereIn('id', $stockIds)
            ->with([
                'stockData' => function($query) {
                    $query->orderBy('date', 'desc')->limit(30);
                },
                'predictions' => function($query) {
                    $query->orderBy('prediction_date', 'desc')->limit(7);
                }
            ])
            ->get();
        
        return view('stocks.compare', compact('stocks'));
    }

    public function screener(Request $request)
    {
        $stocks = Stock::with(['stockData' => function($query) {
            $query->orderBy('date', 'desc')->limit(1);
        }, 'predictions' => function($query) {
            $query->orderBy('prediction_date', 'desc')->limit(1);
        }])->get();
        
        // Process filters
        $recommendation = $request->input('recommendation');
        $priceMin = $request->input('price_min');
        $priceMax = $request->input('price_max');
        $changeMin = $request->input('change_min');
        $changeMax = $request->input('change_max');
        $confidenceMin = $request->input('confidence_min');
        
        $filteredStocks = $stocks->filter(function($stock) use ($recommendation, $priceMin, $priceMax, $changeMin, $changeMax, $confidenceMin) {
            $latestPrice = optional($stock->stockData->first())->close ?? 0;
            $latestPrediction = $stock->predictions->first();
            
            if (!$latestPrediction || !$latestPrice) {
                return false;
            }
            
            $predictedChange = (($latestPrediction->predicted_price - $latestPrice) / $latestPrice) * 100;
            
            // Apply filters
            if ($recommendation && $latestPrediction->recommendation != $recommendation) {
                return false;
            }
            
            if ($priceMin !== null && $latestPrice < $priceMin) {
                return false;
            }
            
            if ($priceMax !== null && $latestPrice > $priceMax) {
                return false;
            }
            
            if ($changeMin !== null && $predictedChange < $changeMin) {
                return false;
            }
            
            if ($changeMax !== null && $predictedChange > $changeMax) {
                return false;
            }
            
            if ($confidenceMin !== null && $latestPrediction->confidence < $confidenceMin) {
                return false;
            }
            
            return true;
        });
        
        return view('stocks.screener', compact('filteredStocks', 'recommendation', 'priceMin', 'priceMax', 'changeMin', 'changeMax', 'confidenceMin'));
    }

    /**
     * Display prediction details and chart for a specific stock.
     *
     * @param  \App\Models\Stock  $stock
     * @return \Illuminate\Http\Response
     */
    public function predictions(Stock $stock)
    {
        // Get the latest stock price
        $latestPrice = $stock->stockData()
            ->orderBy('date', 'desc')
            ->first()
            ->close ?? 0;
            
        // Get predictions for this stock (last 5 days)
        $predictions = $stock->predictions()
            ->orderBy('prediction_date', 'asc')
            ->take(5)
            ->get();
            
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
            
            // Convert to percentage of mean
            $volatility = ($volatility / $mean) * 100;
        }
        
        // Get latest prediction for confidence score
        $latestPrediction = $predictions->last();
        
        // Calculate average accuracy based on historical performance (placeholder)
        // In a real implementation, you'd compare past predictions to actual prices
        $accuracy = 75.0;
        
        // Determine risk level based on volatility
        $riskLevel = 'Medium';
        if ($volatility > 5) {
            $riskLevel = 'High';
        } elseif ($volatility < 2) {
            $riskLevel = 'Low';
        }
            
        return view('stocks.predictions', compact('stock', 'predictions', 'latestPrice', 'latestPrediction', 'volatility', 'accuracy', 'riskLevel'));
    }

    public function export(Request $request, Stock $stock)
    {
        $type = $request->input('type', 'prices');
        $format = $request->input('format', 'csv');
        
        if ($type === 'prices') {
            $data = $stock->stockData()->orderBy('date', 'desc')->get();
            $filename = "{$stock->symbol}_historical_prices";
        } else {
            $data = $stock->predictions()->orderBy('prediction_date', 'desc')->get();
            $filename = "{$stock->symbol}_predictions";
        }
        
        if ($format === 'json') {
            return response()->json($data);
        } else {
            // Default to CSV
            $headers = [
                'Content-Type' => 'text/csv',
                'Content-Disposition' => "attachment; filename=$filename.csv",
                'Pragma' => 'no-cache',
                'Cache-Control' => 'must-revalidate, post-check=0, pre-check=0',
                'Expires' => '0'
            ];
            
            $callback = function() use ($data, $type) {
                $file = fopen('php://output', 'w');
                
                if ($type === 'prices') {
                    fputcsv($file, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']);
                    
                    foreach ($data as $row) {
                        fputcsv($file, [
                            $row->date,
                            $row->open,
                            $row->high,
                            $row->low,
                            $row->close,
                            $row->volume
                        ]);
                    }
                } else {
                    fputcsv($file, ['Date', 'Predicted Price', 'Recommendation', 'Confidence']);
                    
                    foreach ($data as $row) {
                        fputcsv($file, [
                            $row->prediction_date,
                            $row->predicted_price,
                            $row->recommendation,
                            $row->confidence
                        ]);
                    }
                }
                
                fclose($file);
            };
            
            return response()->stream($callback, 200, $headers);
        }
    }
    
    private function calculateTechnicalIndicators($stockData)
    {
        if ($stockData->isEmpty()) {
            return [];
        }
        
        // Get close prices
        $closePrices = $stockData->pluck('close')->toArray();
        
        $indicators = [];
        
        // Simple Moving Averages (SMA)
        $indicators['sma20'] = $this->calculateSMA($closePrices, 20);
        $indicators['sma50'] = $this->calculateSMA($closePrices, 50);
        
        // Relative Strength Index (RSI)
        $indicators['rsi14'] = $this->calculateRSI($closePrices, 14);
        
        // MACD (Moving Average Convergence Divergence)
        $indicators['macd'] = $this->calculateMACD($closePrices);
        
        return $indicators;
    }

    private function calculateSMA($prices, $period)
    {
        $sma = [];
        
        for ($i = 0; $i < count($prices); $i++) {
            if ($i < $period - 1) {
                $sma[] = null;
            } else {
                $sum = 0;
                for ($j = $i - $period + 1; $j <= $i; $j++) {
                    $sum += $prices[$j];
                }
                $sma[] = $sum / $period;
            }
        }
        
        return $sma;
    }

    private function calculateRSI($prices, $period)
    {
        $rsi = [];
        
        // Need at least period + 1 data points to calculate first RSI
        for ($i = 0; $i < count($prices); $i++) {
            if ($i <= $period) {
                $rsi[] = null;
            } else {
                $gains = 0;
                $losses = 0;
                
                for ($j = $i - $period; $j < $i; $j++) {
                    $change = $prices[$j + 1] - $prices[$j];
                    if ($change >= 0) {
                        $gains += $change;
                    } else {
                        $losses -= $change; // Make losses positive
                    }
                }
                
                $averageGain = $gains / $period;
                $averageLoss = $losses / $period;
                
                if ($averageLoss == 0) {
                    $rsi[] = 100;
                } else {
                    $rs = $averageGain / $averageLoss;
                    $rsi[] = 100 - (100 / (1 + $rs));
                }
            }
        }
        
        return $rsi;
    }

    private function calculateMACD($prices)
    {
        $ema12 = $this->calculateEMA($prices, 12);
        $ema26 = $this->calculateEMA($prices, 26);
        
        $macdLine = [];
        $signalLine = [];
        $histogram = [];
        
        // Calculate MACD Line = EMA12 - EMA26
        for ($i = 0; $i < count($prices); $i++) {
            if ($i < 25) { // Need at least 26 data points for first EMA26
                $macdLine[] = null;
            } else {
                $macdLine[] = $ema12[$i] - $ema26[$i];
            }
        }
        
        // Calculate Signal Line = 9-day EMA of MACD Line
        $signalLine = $this->calculateEMA($macdLine, 9);
        
        // Calculate Histogram = MACD Line - Signal Line
        for ($i = 0; $i < count($macdLine); $i++) {
            if ($macdLine[$i] === null || $signalLine[$i] === null) {
                $histogram[] = null;
            } else {
                $histogram[] = $macdLine[$i] - $signalLine[$i];
            }
        }
        
        return [
            'macd_line' => $macdLine,
            'signal_line' => $signalLine,
            'histogram' => $histogram
        ];
    }

    private function calculateEMA($prices, $period)
    {
        $ema = [];
        $multiplier = 2 / ($period + 1);
        
        for ($i = 0; $i < count($prices); $i++) {
            if ($i < $period - 1) {
                $ema[] = null;
            } else if ($i == $period - 1) {
                // First EMA is SMA
                $sum = 0;
                for ($j = 0; $j < $period; $j++) {
                    $sum += $prices[$j];
                }
                $ema[] = $sum / $period;
            } else {
                if ($prices[$i] !== null && $ema[$i-1] !== null) {
                    $ema[] = ($prices[$i] - $ema[$i-1]) * $multiplier + $ema[$i-1];
                } else {
                    $ema[] = null;
                }
            }
        }
        
        return $ema;
    }
}