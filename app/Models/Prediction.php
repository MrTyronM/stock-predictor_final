<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Prediction extends Model
{
    use HasFactory;

    protected $fillable = [
        'stock_id',
        'prediction_date',
        'target_date',
        'signal',
        'confidence',
        'predicted_price',
        'current_price',
        'actual_price',
        'status'
    ];

    protected $casts = [
        'prediction_date' => 'datetime',
        'target_date' => 'datetime',
        'confidence' => 'float',
        'predicted_price' => 'decimal:2',
        'current_price' => 'decimal:2',
        'actual_price' => 'decimal:2'
    ];

    public function stock()
    {
        return $this->belongsTo(Stock::class);
    }
}