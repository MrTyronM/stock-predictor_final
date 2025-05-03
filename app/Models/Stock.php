<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Stock extends Model
{
    use HasFactory;

    protected $fillable = [
        'symbol',
        'company_name',
        'last_price',
        'change_percent',
        'market_cap',
        'sector',
        'industry'
    ];

    public function modelParameters()
    {
        return $this->hasOne(ModelParameter::class);
    }

    public function predictions()
    {
        return $this->hasMany(Prediction::class);
    }
}