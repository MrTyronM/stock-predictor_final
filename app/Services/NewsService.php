<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;
use App\Models\News;
use App\Models\Stock;
use Illuminate\Support\Str;
use Carbon\Carbon;

class NewsService
{
    protected $apiKey;
    protected $baseUrl;
    
    public function __construct()
    {
        $this->apiKey = env('NEWS_API_KEY');
        $this->baseUrl = 'https://newsapi.org/v2';
    }
    
    public function fetchFinancialNews()
    {
        $response = Http::get($this->baseUrl . '/everything', [
            'q' => 'finance OR stock market OR investing',
            'language' => 'en',
            'sortBy' => 'publishedAt',
            'pageSize' => 100,
            'apiKey' => $this->apiKey
        ]);
        
        if ($response->successful()) {
            $data = $response->json();
            
            if (isset($data['articles']) && is_array($data['articles'])) {
                $this->processArticles($data['articles']);
                return true;
            }
        }
        
        return false;
    }
    
    public function fetchStockSpecificNews(Stock $stock)
    {
        $response = Http::get($this->baseUrl . '/everything', [
            'q' => $stock->name . ' OR ' . $stock->symbol,
            'language' => 'en',
            'sortBy' => 'publishedAt',
            'pageSize' => 20,
            'apiKey' => $this->apiKey
        ]);
        
        if ($response->successful()) {
            $data = $response->json();
            
            if (isset($data['articles']) && is_array($data['articles'])) {
                $this->processArticles($data['articles'], $stock);
                return true;
            }
        }
        
        return false;
    }
    
    protected function processArticles($articles, $stock = null)
    {
        foreach ($articles as $article) {
            // Skip if no title or content
            if (!isset($article['title']) || !isset($article['description'])) {
                continue;
            }
            
            // Check if news already exists by title
            $existingNews = News::where('title', $article['title'])->first();
            
            if (!$existingNews) {
                $news = new News();
                $news->title = $article['title'];
                $news->content = $article['description'] . (isset($article['content']) ? "\n\n" . $article['content'] : '');
                $news->source = $article['source']['name'] ?? null;
                $news->url = $article['url'] ?? null;
                $news->published_date = isset($article['publishedAt']) ? Carbon::parse($article['publishedAt']) : Carbon::now();
                $news->save();
                
                // If stock is provided, attach it to the news
                if ($stock) {
                    $news->stocks()->attach($stock->id);
                } else {
                    // Try to find related stocks based on article content
                    $this->findRelatedStocks($news);
                }
            } else if ($stock && !$existingNews->stocks->contains($stock->id)) {
                // If news exists but isn't linked to this stock yet
                $existingNews->stocks()->attach($stock->id);
            }
        }
    }
    
    protected function findRelatedStocks($news)
    {
        // Get all stocks
        $stocks = Stock::all();
        
        $relatedStockIds = [];
        
        foreach ($stocks as $stock) {
            // Check if stock name or symbol is mentioned in the article
            if (Str::contains(strtolower($news->title . ' ' . $news->content), 
                              strtolower($stock->name)) || 
                Str::contains(strtolower($news->title . ' ' . $news->content), 
                              strtolower($stock->symbol))) {
                $relatedStockIds[] = $stock->id;
            }
        }
        
        // Attach related stocks
        if (!empty($relatedStockIds)) {
            $news->stocks()->attach($relatedStockIds);
        }
    }
}