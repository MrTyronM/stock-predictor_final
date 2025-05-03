<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class PortfolioItem extends Model
{
    use HasFactory;

    protected $fillable = [
        'portfolio_id', 
        'stock_id', 
        'shares', 
        'purchase_price', 
        'purchase_date',
        'is_watchlist'
    ];

    protected $casts = [
        'purchase_date' => 'date',
        'is_watchlist' => 'boolean',
    ];

    public function portfolio()
    {
        return $this->belongsTo(Portfolio::class);
    }

    public function stock()
    {
        return $this->belongsTo(Stock::class);
    }
}