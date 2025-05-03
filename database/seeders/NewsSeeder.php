<?php

namespace Database\Seeders;

use Illuminate\Database\Seeder;
use App\Models\News;
use App\Models\Stock;
use Illuminate\Support\Facades\Log;

class NewsSeeder extends Seeder
{
    public function run()
    {
        // Get stocks - create a lookup by symbol for faster access
        $stocks = Stock::all();
        $stocksBySymbol = [];
        foreach ($stocks as $stock) {
            $stocksBySymbol[$stock->symbol] = $stock;
        }

        // Sample news articles
        $newsArticles = [
            [
                'title' => 'Federal Reserve Holds Interest Rates Steady',
                'content' => "<p>The Federal Reserve announced today that it will maintain current interest rates, citing continued economic growth and stable inflation metrics. This decision comes after months of speculation about potential rate hikes.</p><p>\"Our current monetary policy stance is appropriate given the economic outlook,\" said the Fed Chair in a statement following the meeting. \"We remain committed to our dual mandate of promoting maximum employment and stable prices.\"</p><p>Market analysts widely expected this decision, though some had anticipated a more hawkish tone in the accompanying statement. The Fed's decision suggests confidence in the current economic trajectory while acknowledging ongoing challenges in certain sectors.</p>",
                'source' => 'Financial Times',
                'url' => 'https://www.ft.com/content/sample-article',
                'published_date' => now()->subDays(2),
                'related_stocks' => ['AAPL', 'MSFT', 'GOOGL']
            ],
            [
                'title' => 'Tech Stocks Rally on Strong Earnings Reports',
                'content' => "<p>Technology stocks surged today following better-than-expected earnings reports from several major companies in the sector. Investors responded positively to strong revenue growth and optimistic forward guidance.</p><p>Leading the gains were companies involved in cloud computing and artificial intelligence, which reported significant increases in both revenue and profit margins. Analysts point to continued enterprise adoption of cloud services and increased spending on AI capabilities as key drivers of growth.</p><p>\"We're seeing a fundamental shift in how businesses operate,\" noted a market strategist. \"Companies that are positioned well in the digital transformation space continue to outperform expectations.\"</p>",
                'source' => 'Wall Street Journal',
                'url' => 'https://www.wsj.com/sample-article',
                'published_date' => now()->subDays(1),
                'related_stocks' => ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
            ],
            [
                'title' => 'Oil Prices Stabilize After Recent Volatility',
                'content' => "<p>Global oil prices have stabilized following weeks of volatility triggered by geopolitical tensions and supply concerns. Brent crude settled at \$85 per barrel, representing a modest increase over the previous week.</p><p>The stabilization comes as major producers signaled a commitment to maintaining current production levels, alleviating fears of supply disruptions. Additionally, demand forecasts from major economies have remained consistent with previous projections.</p><p>Energy analysts suggest this period of stability could continue in the near term, though they caution that geopolitical risks remain a potential disruptor to global energy markets. \"We're in a more balanced position now, but the market remains sensitive to external shocks,\" stated an industry expert.</p>",
                'source' => 'Bloomberg',
                'url' => 'https://www.bloomberg.com/sample-article',
                'published_date' => now()->subDays(3),
                'related_stocks' => ['TSLA']
            ],
            [
                'title' => 'Retail Sales Exceed Expectations in Q1',
                'content' => "<p>Retail sales figures for the first quarter have surpassed analyst expectations, growing by 4.2% compared to the same period last year. The stronger-than-anticipated numbers suggest resilient consumer spending despite inflationary pressures.</p><p>Department stores and e-commerce platforms both reported healthy growth, with online sales continuing to gain market share. Consumer electronics and home improvement categories were particularly strong performers during the quarter.</p><p>\"These numbers indicate that consumers remain confident and are continuing to spend across multiple categories,\" commented a retail industry analyst. \"The ability of retailers to pass on some cost increases without significantly impacting volume is also noteworthy.\"</p>",
                'source' => 'CNBC',
                'url' => 'https://www.cnbc.com/sample-article',
                'published_date' => now()->subDays(5),
                'related_stocks' => ['AMZN']
            ],
            [
                'title' => 'New Regulations Proposed for AI Development',
                'content' => "<p>Regulators have proposed a new framework for the development and deployment of artificial intelligence technologies. The guidelines aim to ensure ethical AI development while encouraging innovation in the rapidly evolving field.</p><p>The proposed regulations would require companies developing advanced AI systems to implement rigorous testing protocols and transparency measures. Additionally, high-risk AI applications would be subject to third-party audits before public release.</p><p>Tech industry representatives have expressed cautious optimism about the balanced approach, though some have raised concerns about potential impacts on development timelines. \"We support thoughtful regulation that promotes responsible AI,\" said a spokesperson for a leading tech company. \"The key will be finding the right balance between oversight and innovation.\"</p>",
                'source' => 'Tech Report',
                'url' => 'https://www.techreport.com/sample-article',
                'published_date' => now()->subDays(7),
                'related_stocks' => ['GOOGL', 'MSFT']
            ]
        ];

        $this->command->info('Creating news articles...');

        foreach ($newsArticles as $article) {
            // Check if article already exists (prevent duplicates)
            $existingNews = News::where('title', $article['title'])->first();

            if (!$existingNews) {
                $news = News::create([
                    'title' => $article['title'],
                    'content' => $article['content'],
                    'source' => $article['source'],
                    'url' => $article['url'] ?? null,
                    'published_date' => $article['published_date'],
                ]);

                $this->command->info("Created news: {$article['title']}");

                // Attach related stocks
                $stockIds = [];
                foreach ($article['related_stocks'] as $symbol) {
                    if (isset($stocksBySymbol[$symbol])) {
                        $stockIds[] = $stocksBySymbol[$symbol]->id;
                    } else {
                        $this->command->warn("Stock with symbol {$symbol} not found in database");
                    }
                }

                if (!empty($stockIds)) {
                    $news->stocks()->attach($stockIds);
                    $this->command->info("Attached " . count($stockIds) . " stocks to news article");
                }
            } else {
                $this->command->info("News article '{$article['title']}' already exists, skipping...");
            }
        }

        $this->command->info('News seeding completed!');
    }
}
