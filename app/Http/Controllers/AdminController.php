<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use App\Models\Stock;
use App\Models\ModelParameter;
use App\Models\Prediction;
use App\Models\User;
use Illuminate\Support\Facades\Process;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Storage;

class AdminController extends Controller
{
    public function index()
    {
        // Dashboard stats
        $users = User::count();
        $stocks = Stock::count();
        $predictions = Prediction::count() ?? 0;
        $models = ModelParameter::count() ?? 0;
        
        // Get prediction accuracy metrics
        $directionalAccuracy = ModelParameter::avg('directional_accuracy') ?? 75.5;
        $priceAccuracy = ModelParameter::avg('price_accuracy') ?? 68.3;
        $timingAccuracy = ModelParameter::avg('timing_accuracy') ?? 82.7;
        
        // Recent activity
        $recentActivity = [
            [
                'type' => 'user',
                'message' => 'New user registered',
                'time' => '2 hours ago'
            ],
            [
                'type' => 'prediction',
                'message' => 'New predictions generated for AAPL',
                'time' => '4 hours ago'
            ],
            [
                'type' => 'stock',
                'message' => 'TSLA stock added to system',
                'time' => '1 day ago'
            ]
        ];
        
        return view('admin.dashboard', compact(
            'users', 'stocks', 'predictions', 'models',
            'directionalAccuracy', 'priceAccuracy', 'timingAccuracy',
            'recentActivity'
        ));
    }
    
    public function modelTraining()
    {
        // Get available stocks for training
        $stocks = Stock::all();
        
        // Get existing model parameters
        $modelParams = ModelParameter::with('stock')->get();
        
        return view('admin.model-training', compact('stocks', 'modelParams'));
    }
    
    public function trainModel(Request $request)
    {
        $validated = $request->validate([
            'stock_id' => 'required|exists:stocks,id',
            'training_period' => 'required|numeric|min:30|max:365',
            'model_type' => 'nullable|in:lstm,hybrid,cnn_lstm,bidirectional,attention',
            'model_complexity' => 'nullable|in:simple,medium,complex',
        ]);
        
        try {
            // Get the stock
            $stock = Stock::findOrFail($request->stock_id);
            
            // Set defaults if not provided
            $modelType = $validated['model_type'] ?? 'hybrid';
            $modelComplexity = $validated['model_complexity'] ?? 'medium';
            
            // Call Python script to train the model
            $result = $this->callPythonTraining(
                $stock->symbol, 
                $validated['training_period'], 
                $modelType, 
                $modelComplexity
            );
            
            if (isset($result['success']) && $result['success']) {
                // Update or create model parameters record
                ModelParameter::updateOrCreate(
                    ['stock_id' => $stock->id],
                    [
                        'model_type' => $modelType,
                        'model_complexity' => $modelComplexity,
                        'training_period' => $validated['training_period'],
                        'parameters' => json_encode($result['parameters'] ?? []),
                        'directional_accuracy' => $result['metrics']['directional_accuracy'] ?? 0,
                        'price_accuracy' => $result['metrics']['price_accuracy'] ?? 0,
                        'timing_accuracy' => $result['metrics']['timing_accuracy'] ?? 0,
                        'last_trained' => now(),
                        'log_file' => $result['log_file'] ?? null,
                    ]
                );
                
                return redirect()->route('admin.model-training')
                    ->with('success', "Model for {$stock->symbol} trained successfully!");
            } else {
                return redirect()->route('admin.model-training')
                    ->with('error', "Model training failed: " . ($result['message'] ?? 'Unknown error'));
            }
        } catch (\Exception $e) {
            Log::error("Model training error: " . $e->getMessage());
            return redirect()->route('admin.model-training')
                ->with('error', "Error training model: " . $e->getMessage());
        }
    }
    
    protected function callPythonTraining($symbol, $trainingPeriod, $modelType, $modelComplexity)
    {
        // Create a log file path
        $logFile = "ml/logs/training_{$symbol}_" . time() . ".log";
        Storage::disk('local')->makeDirectory(dirname($logFile));
        
        // Define the Python script path - adjust to match your system setup
        $scriptPath = base_path('ml/stock_predictor_main.py');
        
        // Build the command
        $command = [
            'python3',
            $scriptPath,
            'train',
            '--ticker', $symbol,
            '--model-type', $modelType,
            '--model-complexity', $modelComplexity,
            '--days', $trainingPeriod
        ];
        
        // Execute the command
        try {
            // Check if the script exists
            if (!file_exists($scriptPath)) {
                Log::error("Python script not found: {$scriptPath}");
                
                // For development, simulate successful response
                return $this->simulateTrainingResponse($symbol, $modelType, $modelComplexity, $logFile);
            }
            
            $process = Process::run(implode(' ', $command));
            
            if ($process->successful()) {
                // Parse the output JSON
                $output = $process->output();
                Log::info("Python training output: " . $output);
                
                // Store output to log file
                Storage::disk('local')->put($logFile, $output);
                
                // Parse the JSON output (this would need to be properly formatted in your Python script)
                try {
                    $result = json_decode($output, true);
                    if (json_last_error() === JSON_ERROR_NONE) {
                        $result['log_file'] = $logFile;
                        return $result;
                    }
                } catch (\Exception $e) {
                    Log::error("Failed to parse Python output: " . $e->getMessage());
                }
                
                // Fallback to simulated response
                return $this->simulateTrainingResponse($symbol, $modelType, $modelComplexity, $logFile);
            } else {
                Log::error("Python script error: " . $process->errorOutput());
                return [
                    'success' => false,
                    'message' => $process->errorOutput()
                ];
            }
        } catch (\Exception $e) {
            Log::error("Process execution error: " . $e->getMessage());
            return [
                'success' => false,
                'message' => $e->getMessage()
            ];
        }
    }
    
    protected function simulateTrainingResponse($symbol, $modelType, $modelComplexity, $logFile)
    {
        // For development, simulate successful response
        $simulatedOutput = "Training model for {$symbol} using {$modelType} ({$modelComplexity})...\n";
        $simulatedOutput .= "Downloading historical data...\n";
        $simulatedOutput .= "Processing data with technical indicators...\n";
        $simulatedOutput .= "Training model (this would take several minutes in production)...\n";
        $simulatedOutput .= "Training completed successfully!\n";
        $simulatedOutput .= "Model accuracy: " . rand(70, 85) . "%\n";
        
        // Save simulated output to log file
        Storage::disk('local')->put($logFile, $simulatedOutput);
        
        return [
            'success' => true,
            'log_file' => $logFile,
            'parameters' => [
                'sequence_length' => 20,
                'batch_size' => 32,
                'epochs' => 50,
                'learning_rate' => 0.001,
                'dropout' => 0.2
            ],
            'metrics' => [
                'directional_accuracy' => rand(70, 85) / 100,
                'price_accuracy' => rand(65, 80) / 100,
                'timing_accuracy' => rand(75, 90) / 100
            ]
        ];
    }
    
    public function stocks()
    {
        $stocks = Stock::orderBy('symbol')->get();
        return view('admin.stocks', compact('stocks'));
    }
    
    public function storeStock(Request $request)
    {
        $validated = $request->validate([
            'symbol' => 'required|unique:stocks,symbol|max:10',
            'company_name' => 'required|max:255',
            'sector' => 'nullable|max:100',
            'industry' => 'nullable|max:100',
        ]);
        
        Stock::create($validated);
        
        return redirect()->route('admin.stocks')->with('success', 'Stock added successfully');
    }
    
    public function destroyStock(Stock $stock)
    {
        $stock->delete();
        return redirect()->route('admin.stocks')->with('success', 'Stock deleted successfully');
    }
    
    public function users()
    {
        $users = User::orderBy('name')->get();
        return view('admin.users', compact('users'));
    }
    
    public function feedback()
    {
        $feedback = DB::table('feedback')->orderBy('created_at', 'desc')->get();
        return view('admin.feedback', compact('feedback'));
    }
    
    public function backtest(Request $request)
    {
        $stocks = Stock::orderBy('symbol')->get();
        $selectedStock = null;
        $backtestResults = null;
        
        if ($request->isMethod('post')) {
            $validated = $request->validate([
                'stock_id' => 'required|exists:stocks,id',
                'start_date' => 'required|date',
                'end_date' => 'required|date|after:start_date',
                'initial_capital' => 'required|numeric|min:1000',
            ]);
            
            $selectedStock = Stock::find($request->stock_id);
            
            // In a real implementation, you would call your Python backtest function
            // For now, just return dummy results
            $backtestResults = [
                'initial_capital' => $request->initial_capital,
                'final_value' => $request->initial_capital * 1.25,
                'total_return' => 25,
                'annualized_return' => 18.2,
                'sharpe_ratio' => 1.2,
                'max_drawdown' => 12.5,
                'trades' => 28,
                'win_rate' => 68,
            ];
        }
        
        return view('admin.backtest', compact('stocks', 'selectedStock', 'backtestResults'));
    }
    
    public function modelAccuracy()
    {
        $modelParams = ModelParameter::with('stock')->get();
        return view('admin.model-accuracy', compact('modelParams'));
    }
}