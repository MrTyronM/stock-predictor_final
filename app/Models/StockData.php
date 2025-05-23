<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class StockData extends Model
{
    use HasFactory;

    protected $fillable = ['stock_id', 'date', 'open', 'high', 'low', 'close', 'volume'];

    public function stock()
    {
        return $this->belongsTo(Stock::class);
    }
}