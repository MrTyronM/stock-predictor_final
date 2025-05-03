<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Stock;
use App\Models\StockData;
use Carbon\Carbon;

class BacktestController extends Controller
{
    public function index()
    {
        $stocks = Stock::all();
        return view('backtest.index', compact('stocks'));
    }
    
    public function runBacktest(Request $request)
    {
        $request->validate([
            'stock_id' => 'required|exists:stocks,id',
            'start_date' => 'required|date',
            'end_date' => 'required|date|after:start_date',
            'initial_capital' => 'required|numeric|min:100',
            'strategy' => 'required|in:sma_crossover,momentum,rsi'
        ]);
        
        $stock = Stock::findOrFail($request->stock_id);
        
        // Get historical data for the selected period
        $stockData = StockData::where('stock_id', $stock->id)
            ->whereBetween('date', [$request->start_date, $request->end_date])
            ->orderBy('date', 'asc')
            ->get();
            
        if ($stockData->count() < 50) {
            return back()->with('error', 'Insufficient historical data for backtesting. Need at least 50 data points.');
        }
        
        // Run the selected strategy
        $strategy = $request->strategy;
        $initialCapital = $request->initial_capital;
        
        switch ($strategy) {
            case 'sma_crossover':
                $results = $this->backtest_sma_crossover($stockData, $initialCapital);
                break;
            case 'momentum':
                $results = $this->backtest_momentum($stockData, $initialCapital);
                break;
            case 'rsi':
                $results = $this->backtest_rsi($stockData, $initialCapital);
                break;
            default:
                return back()->with('error', 'Invalid strategy selected.');
        }
        
        return view('backtest.results', compact('results', 'stock', 'strategy'));
    }
    
    private function backtest_sma_crossover($stockData, $initialCapital)
    {
        // Simple Moving Average crossover strategy (short-term vs long-term)
        $short_window = 20; // 20-day SMA
        $long_window = 50; // 50-day SMA
        
        $results = [
            'dates' => [],
            'prices' => [],
            'short_ma' => [],
            'long_ma' => [],
            'positions' => [],
            'signals' => [],
            'portfolio_values' => [],
            'cash' => [],
            'holdings' => [],
            'trades' => []
        ];
        
        // Calculate SMAs
        $prices = $stockData->pluck('close')->toArray();
        
        for ($i = 0; $i < count($prices); $i++) {
            $short_sum = 0;
            $long_sum = 0;
            $short_count = 0;
            $long_count = 0;
            
            // Calculate short SMA
            for ($j = max(0, $i - $short_window + 1); $j <= $i; $j++) {
                $short_sum += $prices[$j];
                $short_count++;
            }
            
            // Calculate long SMA
            for ($j = max(0, $i - $long_window + 1); $j <= $i; $j++) {
                $long_sum += $prices[$j];
                $long_count++;
            }
            
            $short_ma = $short_count > 0 ? $short_sum / $short_count : null;
            $long_ma = $long_count > 0 ? $long_sum / $long_count : null;
            
            $results['short_ma'][] = $short_ma;
            $results['long_ma'][] = $long_ma;
        }
        
        // Generate trading signals (1 = buy, -1 = sell, 0 = hold)
        $position = 0; // 0 = no position, 1 = long position
        $cash = $initialCapital;
        $holdings = 0;
        $shares = 0;
        
        foreach ($stockData as $i => $data) {
            $results['dates'][] = $data->date;
            $results['prices'][] = $data->close;
            
            $signal = 0;
            
            // Generate signals only after we have both SMAs
            if ($i >= $long_window - 1) {
                $shortMA = $results['short_ma'][$i];
                $longMA = $results['long_ma'][$i];
                
                // Buy signal: short SMA crosses above long SMA
                if ($shortMA > $longMA && $results['short_ma'][$i-1] <= $results['long_ma'][$i-1]) {
                    $signal = 1;
                }
                // Sell signal: short SMA crosses below long SMA
                else if ($shortMA < $longMA && $results['short_ma'][$i-1] >= $results['long_ma'][$i-1]) {
                    $signal = -1;
                }
            }
            
            // Execute trades
            if ($signal == 1 && $position == 0) {
                // Buy all possible shares
                $shares = floor($cash / $data->close);
                $cost = $shares * $data->close;
                $cash -= $cost;
                $holdings = $shares * $data->close;
                $position = 1;
                
                $results['trades'][] = [
                    'date' => $data->date,
                    'type' => 'buy',
                    'price' => $data->close,
                    'shares' => $shares,
                    'cost' => $cost,
                    'cash_after' => $cash,
                    'portfolio_after' => $cash + $holdings
                ];
            }
            else if ($signal == -1 && $position == 1) {
                // Sell all shares
                $revenue = $shares * $data->close;
                $cash += $revenue;
                $shares = 0;
                $holdings = 0;
                $position = 0;
                
                $results['trades'][] = [
                    'date' => $data->date,
                    'type' => 'sell',
                    'price' => $data->close,
                    'shares' => $shares,
                    'revenue' => $revenue,
                    'cash_after' => $cash,
                    'portfolio_after' => $cash + $holdings
                ];
            }
            
            $holdings = $shares * $data->close;
            $portfolio_value = $cash + $holdings;
            
            $results['signals'][] = $signal;
            $results['positions'][] = $position;
            $results['cash'][] = $cash;
            $results['holdings'][] = $holdings;
            $results['portfolio_values'][] = $portfolio_value;
        }
        
        // Calculate performance metrics
        $start_value = $initialCapital;
        $end_value = end($results['portfolio_values']);
        $total_return = (($end_value - $start_value) / $start_value) * 100;
        
        // Calculate buy & hold return for comparison
        $buy_hold_start = $stockData->first()->close;
        $buy_hold_end = $stockData->last()->close;
        $buy_hold_shares = floor($initialCapital / $buy_hold_start);
        $buy_hold_value = $buy_hold_shares * $buy_hold_end;
        $buy_hold_return = (($buy_hold_value - $initialCapital) / $initialCapital) * 100;
        
        $results['summary'] = [
            'initial_capital' => $initialCapital,
            'final_value' => $end_value,
            'total_return' => $total_return,
            'buy_hold_return' => $buy_hold_return,
            'outperformance' => $total_return - $buy_hold_return,
            'total_trades' => count($results['trades']),
            'strategy_name' => 'Simple Moving Average Crossover',
            'strategy_params' => [
                'short_window' => $short_window,
                'long_window' => $long_window
            ]
        ];
        
        return $results;
    }
    
    private function backtest_momentum($stockData, $initialCapital)
    {
        // Momentum strategy (buy stocks that have risen over past N days)
        $lookback_window = 14; // 14-day momentum
        $hold_period = 5; // Hold for 5 days
        
        $results = [
            'dates' => [],
            'prices' => [],
            'momentum' => [],
            'positions' => [],
            'signals' => [],
            'portfolio_values' => [],
            'cash' => [],
            'holdings' => [],
            'trades' => []
        ];
        
        // Calculate momentum
        $prices = $stockData->pluck('close')->toArray();
        
        for ($i = 0; $i < count($prices); $i++) {
            if ($i >= $lookback_window) {
                $momentum = ($prices[$i] / $prices[$i - $lookback_window]) - 1;
            } else {
                $momentum = null;
            }
            
            $results['momentum'][] = $momentum;
        }
        
        // Generate trading signals
        $position = 0; // 0 = no position, 1 = long position
        $cash = $initialCapital;
        $holdings = 0;
        $shares = 0;
        $days_held = 0;
        
        foreach ($stockData as $i => $data) {
            $results['dates'][] = $data->date;
            $results['prices'][] = $data->close;
            
            $signal = 0;
            
            // Generate signals only after we have momentum data
            if ($i >= $lookback_window) {
                $current_momentum = $results['momentum'][$i];
                
                // If in a position, count down hold period
                if ($position == 1) {
                    $days_held++;
                    
                    // Sell after hold period
                    if ($days_held >= $hold_period) {
                        $signal = -1;
                    }
                }
                // If not in a position, check for buy signal
                else if ($position == 0) {
                    // Buy signal: positive momentum
                    if ($current_momentum > 0.03) { // 3% threshold
                        $signal = 1;
                    }
                }
            }
            
            // Execute trades
            if ($signal == 1 && $position == 0) {
                // Buy all possible shares
                $shares = floor($cash / $data->close);
                $cost = $shares * $data->close;
                $cash -= $cost;
                $holdings = $shares * $data->close;
                $position = 1;
                $days_held = 0;
                
                $results['trades'][] = [
                    'date' => $data->date,
                    'type' => 'buy',
                    'price' => $data->close,
                    'shares' => $shares,
                    'cost' => $cost,
                    'cash_after' => $cash,
                    'portfolio_after' => $cash + $holdings
                ];
            }
            else if ($signal == -1 && $position == 1) {
                // Sell all shares
                $revenue = $shares * $data->close;
                $cash += $revenue;
                $shares = 0;
                $holdings = 0;
                $position = 0;
                
                $results['trades'][] = [
                    'date' => $data->date,
                    'type' => 'sell',
                    'price' => $data->close,
                    'shares' => $shares,
                    'revenue' => $revenue,
                    'cash_after' => $cash,
                    'portfolio_after' => $cash + $holdings
                ];
            }
            
            $holdings = $shares * $data->close;
            $portfolio_value = $cash + $holdings;
            
            $results['signals'][] = $signal;
            $results['positions'][] = $position;
            $results['cash'][] = $cash;
            $results['holdings'][] = $holdings;
            $results['portfolio_values'][] = $portfolio_value;
        }
        
        // Calculate performance metrics
        $start_value = $initialCapital;
        $end_value = end($results['portfolio_values']);
        $total_return = (($end_value - $start_value) / $start_value) * 100;
        
        // Calculate buy & hold return for comparison
        $buy_hold_start = $stockData->first()->close;
        $buy_hold_end = $stockData->last()->close;
        $buy_hold_shares = floor($initialCapital / $buy_hold_start);
        $buy_hold_value = $buy_hold_shares * $buy_hold_end;
        $buy_hold_return = (($buy_hold_value - $initialCapital) / $initialCapital) * 100;
        
        $results['summary'] = [
            'initial_capital' => $initialCapital,
            'final_value' => $end_value,
            'total_return' => $total_return,
            'buy_hold_return' => $buy_hold_return,
            'outperformance' => $total_return - $buy_hold_return,
            'total_trades' => count($results['trades']),
            'strategy_name' => 'Momentum',
            'strategy_params' => [
                'lookback_window' => $lookback_window,
                'hold_period' => $hold_period
            ]
        ];
        
        return $results;
    }
    
    private function backtest_rsi($stockData, $initialCapital)
    {
        // RSI strategy (buy oversold, sell overbought)
        $rsi_period = 14;
        $oversold_threshold = 30;
        $overbought_threshold = 70;
        
        $results = [
            'dates' => [],
            'prices' => [],
            'rsi' => [],
            'positions' => [],
            'signals' => [],
            'portfolio_values' => [],
            'cash' => [],
            'holdings' => [],
            'trades' => []
        ];
        
        // Calculate RSI
        $prices = $stockData->pluck('close')->toArray();
        $gains = [];
        $losses = [];
        
        // First calculate price changes
        $changes = [];
        for ($i = 1; $i < count($prices); $i++) {
            $changes[] = $prices[$i] - $prices[$i-1];
        }
        
        // Calculate initial average gain and loss
        $avg_gain = 0;
        $avg_loss = 0;
        
        for ($i = 0; $i < min($rsi_period, count($changes)); $i++) {
            if ($changes[$i] >= 0) {
                $avg_gain += $changes[$i];
            } else {
                $avg_loss += abs($changes[$i]);
            }
        }
        
        $avg_gain /= $rsi_period;
        $avg_loss /= $rsi_period;
        
        // Calculate RSI
        $rsi = [];
        
        for ($i = 0; $i < count($prices); $i++) {
            if ($i == 0) {
                $rsi[] = null; // RSI not defined for first price
                continue;
            }
            
            if ($i <= $rsi_period) {
                // Use SMA for initial RSI values
                $sum_gain = 0;
                $sum_loss = 0;
                
                for ($j = max(0, $i - $rsi_period); $j < $i; $j++) {
                    $change = $prices[$j+1] - $prices[$j];
                    if ($change >= 0) {
                        $sum_gain += $change;
                    } else {
                        $sum_loss += abs($change);
                    }
                }
                
                $avg_gain = $sum_gain / min($i, $rsi_period);
                $avg_loss = $sum_loss / min($i, $rsi_period);
            } else {
                // Use SMMA (smoothed/modified moving average)
                $current_change = $prices[$i] - $prices[$i-1];
                
                if ($current_change >= 0) {
                    $avg_gain = (($avg_gain * ($rsi_period - 1)) + $current_change) / $rsi_period;
                    $avg_loss = (($avg_loss * ($rsi_period - 1)) + 0) / $rsi_period;
                } else {
                    $avg_gain = (($avg_gain * ($rsi_period - 1)) + 0) / $rsi_period;
                    $avg_loss = (($avg_loss * ($rsi_period - 1)) + abs($current_change)) / $rsi_period;
                }
            }
            
            if ($avg_loss == 0) {
                $rs = 100;
            } else {
                $rs = $avg_gain / $avg_loss;
            }
            
            $rsi_value = 100 - (100 / (1 + $rs));
            $rsi[] = $rsi_value;
        }
        
        $results['rsi'] = $rsi;
        
        // Generate trading signals
        $position = 0; // 0 = no position, 1 = long position
        $cash = $initialCapital;
        $holdings = 0;
        $shares = 0;
        
        foreach ($stockData as $i => $data) {
            $results['dates'][] = $data->date;
            $results['prices'][] = $data->close;
            
            $signal = 0;
            
            // Generate signals only after we have RSI data
            if ($i > $rsi_period) {
                $current_rsi = $results['rsi'][$i];
                $prev_rsi = $results['rsi'][$i-1];
                
                // Buy signal: RSI crosses above oversold threshold
                if ($position == 0 && $prev_rsi < $oversold_threshold && $current_rsi >= $oversold_threshold) {
                    $signal = 1;
                }
                // Sell signal: RSI crosses above overbought threshold
                else if ($position == 1 && $prev_rsi < $overbought_threshold && $current_rsi >= $overbought_threshold) {
                    $signal = -1;
                }
            }
            
            // Execute trades
            if ($signal == 1 && $position == 0) {
                // Buy all possible shares
                $shares = floor($cash / $data->close);
                $cost = $shares * $data->close;
                $cash -= $cost;
                $holdings = $shares * $data->close;
                $position = 1;
                
                $results['trades'][] = [
                    'date' => $data->date,
                    'type' => 'buy',
                    'price' => $data->close,
                    'shares' => $shares,
                    'cost' => $cost,
                    'cash_after' => $cash,
                    'portfolio_after' => $cash + $holdings
                ];
            }
            else if ($signal == -1 && $position == 1) {
                // Sell all shares
                $revenue = $shares * $data->close;
                $cash += $revenue;
                $shares = 0;
                $holdings = 0;
                $position = 0;
                
                $results['trades'][] = [
                    'date' => $data->date,
                    'type' => 'sell',
                    'price' => $data->close,
                    'shares' => $shares,
                    'revenue' => $revenue,
                    'cash_after' => $cash,
                    'portfolio_after' => $cash + $holdings
                ];
            }
            
            $holdings = $shares * $data->close;
            $portfolio_value = $cash + $holdings;
            
            $results['signals'][] = $signal;
            $results['positions'][] = $position;
            $results['cash'][] = $cash;
            $results['holdings'][] = $holdings;
            $results['portfolio_values'][] = $portfolio_value;
        }
        
        // Calculate performance metrics
        $start_value = $initialCapital;
        $end_value = end($results['portfolio_values']);
        $total_return = (($end_value - $start_value) / $start_value) * 100;
        
        // Calculate buy & hold return for comparison
        $buy_hold_start = $stockData->first()->close;
        $buy_hold_end = $stockData->last()->close;
        $buy_hold_shares = floor($initialCapital / $buy_hold_start);
        $buy_hold_value = $buy_hold_shares * $buy_hold_end;
        $buy_hold_return = (($buy_hold_value - $initialCapital) / $initialCapital) * 100;
        
        $results['summary'] = [
            'initial_capital' => $initialCapital,
            'final_value' => $end_value,
            'total_return' => $total_return,
            'buy_hold_return' => $buy_hold_return,
            'outperformance' => $total_return - $buy_hold_return,
            'total_trades' => count($results['trades']),
            'strategy_name' => 'Relative Strength Index (RSI)',
            'strategy_params' => [
                'rsi_period' => $rsi_period,
                'oversold_threshold' => $oversold_threshold,
                'overbought_threshold' => $overbought_threshold
            ]
        ];
        
        return $results;
    }
}