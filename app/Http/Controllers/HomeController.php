<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Portfolio;
use App\Models\Watchlist;
use App\Models\Notification;
use App\Models\Prediction;
use Illuminate\Support\Facades\Auth;

class HomeController extends Controller
{
    /**
     * Create a new controller instance.
     *
     * @return void
     */
    public function __construct()
    {
        $this->middleware('auth');
    }

    /**
     * Show the application dashboard.
     *
     * @return \Illuminate\Contracts\Support\Renderable
     */
    public function index()
    {
        $user = Auth::user();
        
        // Get user's portfolios
        $portfolios = Portfolio::where('user_id', $user->id)
            ->withCount('items')
            ->get();
            
        // Get user's watchlists
        $watchlists = Watchlist::where('user_id', $user->id)
            ->withCount('stocks')
            ->get();
            
        // Get recent notifications
        $notifications = Notification::where('user_id', $user->id)
            ->where('read', false)
            ->orderBy('created_at', 'desc')
            ->limit(5)
            ->get();
            
        // Get top predictions
        $topPredictions = Prediction::with('stock')
            ->where('confidence', '>=', 75)
            ->where('prediction_date', '>=', now()->format('Y-m-d'))
            ->orderBy('confidence', 'desc')
            ->limit(5)
            ->get();
            
        return view('home', compact('portfolios', 'watchlists', 'notifications', 'topPredictions'));
    }
}