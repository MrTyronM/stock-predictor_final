<?php
namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class ModelParameter extends Model
{
    use HasFactory;

    protected $fillable = ['stock_id', 'parameters', 'accuracy'];
    
    protected $casts = [
        'parameters' => 'array',
    ];
    
    public function stock()
    {
        return $this->belongsTo(Stock::class);
    }
}