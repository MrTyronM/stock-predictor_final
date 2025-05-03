<?php

namespace Database\Seeders;

use Illuminate\Database\Seeder;
use App\Models\Stock;

class StockSeeder extends Seeder
{
    public function run()
    {
        $stocks = [
            [
                'symbol' => 'AAPL',
                'name' => 'Apple Inc.',
                'description' => 'Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.'
            ],
            [
                'symbol' => 'MSFT',
                'name' => 'Microsoft Corporation',
                'description' => 'Microsoft Corporation develops, licenses, and supports software, services, devices, and solutions worldwide.'
            ],
            [
                'symbol' => 'GOOGL',
                'name' => 'Alphabet Inc.',
                'description' => 'Alphabet Inc. provides various products and platforms in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America.'
            ],
            [
                'symbol' => 'AMZN',
                'name' => 'Amazon.com, Inc.',
                'description' => 'Amazon.com, Inc. engages in the retail sale of consumer products and subscriptions in North America and internationally.'
            ],
            [
                'symbol' => 'TSLA',
                'name' => 'Tesla, Inc.',
                'description' => 'Tesla, Inc. designs, develops, manufactures, leases, and sells electric vehicles, and energy generation and storage systems.'
            ],
        ];

        foreach ($stocks as $stock) {
            Stock::create($stock);
        }
    }
}