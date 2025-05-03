<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\DashboardController;
use App\Http\Controllers\StockController;
use App\Http\Controllers\PredictionController;
use App\Http\Controllers\AdminController;
use App\Http\Controllers\FeedbackController;
use App\Http\Controllers\NewsController;
use App\Http\Controllers\PortfolioController;
use App\Http\Controllers\NotificationController;
use App\Http\Controllers\ThemeController;
use App\Http\Controllers\ModelTrainingController;
use App\Http\Controllers\ReactController;
use App\Http\Controllers\ImportExportController;
use App\Http\Controllers\MarketController;
use App\Http\Controllers\BacktestController;
use App\Http\Controllers\PublicController;
use App\Http\Controllers\WatchlistController;
use App\Http\Controllers\AnalyticsController;

/*
|--------------------------------------------------------------------------
| Public Routes (No Authentication Required)
|--------------------------------------------------------------------------
*/

Route::get('/', [PublicController::class, 'index'])->name('public.index');
Route::get('/stock/{symbol}', [PublicController::class, 'stockDetails'])->name('public.stock.details');
Route::get('/market-overview', [PublicController::class, 'marketOverview'])->name('public.market.overview');
Route::get('/about', function () {
    return view('about');
})->name('about');

// Direct market dashboard route (no auth required for testing)
Route::get('/direct/dashboard', [ReactController::class, 'marketDashboard'])->name('direct.dashboard');

// Alias 'home' route to 'dashboard' for compatibility
Route::redirect('/home', '/dashboard')->name('home');

/*
|--------------------------------------------------------------------------
| Authentication Routes
|--------------------------------------------------------------------------
*/

require __DIR__.'/auth.php';

/*
|--------------------------------------------------------------------------
| User Routes (Authentication Required)
|--------------------------------------------------------------------------
*/

Route::middleware(['auth'])->group(function () {
    // Dashboard
    Route::get('/dashboard', [DashboardController::class, 'index'])->name('dashboard');
    
    // Stocks & Predictions
    Route::get('/stocks', [StockController::class, 'index'])->name('stocks.index');
    
    // FIXED: Specific route before parameterized route
    Route::get('/stocks/screener', [StockController::class, 'screener'])->name('stocks.screener');
    Route::get('/stocks/{stock}', [StockController::class, 'show'])->name('stocks.show');
    
    Route::get('/stocks/{stock}/export', [StockController::class, 'export'])->name('stocks.export');
    
    // Stock Predictions Route
    Route::get('/stocks/{stock}/predictions', [StockController::class, 'predictions'])->name('predictions.show');
    
    // Feedback
    Route::post('/feedback', [FeedbackController::class, 'store'])->name('feedback.store');
    
    // Portfolios
    Route::resource('portfolios', PortfolioController::class);
    
    // Watchlists
    Route::resource('watchlists', WatchlistController::class);
    Route::post('watchlists/{id}/stocks', [WatchlistController::class, 'addStock'])->name('watchlists.add_stock');
    Route::delete('watchlists/{id}/stocks/{stockId}', [WatchlistController::class, 'removeStock'])->name('watchlists.remove_stock');
    
    // Notifications
    Route::get('/notifications', [NotificationController::class, 'index'])->name('notifications.index');
    Route::post('/notifications/{notification}/mark-read', [NotificationController::class, 'markRead'])->name('notifications.mark-read');
    Route::post('/notifications/mark-all-read', [NotificationController::class, 'markAllRead'])->name('notifications.mark-all-read');
    
    // News
    Route::get('/news', [NewsController::class, 'index'])->name('news.index');
    Route::get('/news/{news}', [NewsController::class, 'show'])->name('news.show');
    
    // Market
    Route::get('/market', [MarketController::class, 'index'])->name('market.index');
    
    // Updated market dashboard route to use direct approach
    Route::get('/market/dashboard', [ReactController::class, 'marketDashboard'])->name('market.dashboard');
    
    // Analytics
    Route::get('/analytics', [AnalyticsController::class, 'index'])->name('analytics.index');
    
    // Backtesting
    Route::get('/backtest', [BacktestController::class, 'index'])->name('backtest.index');
    Route::post('/backtest/run', [BacktestController::class, 'runBacktest'])->name('backtest.run');
    
    // Theme
    Route::post('/theme/toggle', [ThemeController::class, 'toggle'])->name('theme.toggle');
});

/*
|--------------------------------------------------------------------------
| Admin Routes (Authentication Required)
|--------------------------------------------------------------------------
*/

Route::middleware(['auth'])->prefix('admin')->name('admin.')->group(function () {
    // Dashboard
    Route::get('/', [AdminController::class, 'index'])->name('dashboard');
    
    // Model Training
    Route::get('/model-training', [AdminController::class, 'modelTraining'])->name('model-training');
    
    // Add the route for training models via AdminController to match the form action in the view
    Route::post('/train-model', [AdminController::class, 'trainModel'])->name('train-model');
    
    // Stock Management
    Route::get('/stocks', [AdminController::class, 'stocks'])->name('stocks');
    Route::post('/stocks', [AdminController::class, 'storeStock'])->name('stocks.store');
    Route::delete('/stocks/{stock}', [AdminController::class, 'destroyStock'])->name('stocks.destroy');
    
    // News Management
    Route::resource('news', NewsController::class)->except(['index', 'show']);
    
    // User Management
    Route::get('/users', [AdminController::class, 'users'])->name('users');
    
    // Feedback Management
    Route::get('/feedback', [AdminController::class, 'feedback'])->name('feedback');
    
    // Import/Export
    Route::get('/import-export', [ImportExportController::class, 'index'])->name('import-export');
    Route::get('/export-stocks', [ImportExportController::class, 'exportStocks'])->name('export-stocks');
    Route::post('/import-stocks', [ImportExportController::class, 'importStocks'])->name('import-stocks');
    Route::get('/export-stock-data', [ImportExportController::class, 'exportStockData'])->name('export-stock-data');
    Route::post('/import-stock-data', [ImportExportController::class, 'importStockData'])->name('import-stock-data');
    
    // Backtesting & Accuracy
    Route::get('/backtest', [AdminController::class, 'backtest'])->name('backtest');
    Route::post('/backtest', [AdminController::class, 'backtest'])->name('backtest.run');
    Route::get('/model-accuracy', [AdminController::class, 'modelAccuracy'])->name('model-accuracy');
    
    // Model Training Operations - keep these for ModelTrainingController if needed
    Route::post('/model-training/train', [ModelTrainingController::class, 'trainModel'])->name('model-training.train');
    Route::get('/model-training/logs/{stockId}', [ModelTrainingController::class, 'viewLogs'])->name('model-training.logs');
});