<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class News extends Model
{
    use HasFactory;

    protected $fillable = ['title', 'content', 'source', 'url', 'published_date'];

    protected $casts = [
        'published_date' => 'date',
    ];

    public function stocks()
    {
        return $this->belongsToMany(Stock::class, 'news_stock');
    }
}