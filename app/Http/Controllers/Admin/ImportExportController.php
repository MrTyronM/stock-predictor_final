<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use Illuminate\Http\Request;
use App\Models\Stock;
use App\Models\StockData;
use League\Csv\Reader;
use League\Csv\Writer;
use SplTempFileObject;

class ImportExportController extends Controller
{
    public function index()
    {
        return view('admin.import-export');
    }
    
    public function exportStocks()
    {
        $stocks = Stock::all();
        
        $csv = Writer::createFromFileObject(new SplTempFileObject());
        
        // Add headers
        $csv->insertOne(['ID', 'Symbol', 'Name', 'Description', 'Created At', 'Updated At']);
        
        // Add data
        foreach ($stocks as $stock) {
            $csv->insertOne([
                $stock->id,
                $stock->symbol,
                $stock->name,
                $stock->description,
                $stock->created_at,
                $stock->updated_at
            ]);
        }
        
        $headers = [
            'Content-Type' => 'text/csv',
            'Content-Disposition' => 'attachment; filename="stocks.csv"',
            'Pragma' => 'no-cache',
            'Cache-Control' => 'must-revalidate, post-check=0, pre-check=0',
            'Expires' => '0'
        ];
        
        return response($csv->getContent(), 200, $headers);
    }
    
    public function importStocks(Request $request)
    {
        $request->validate([
            'csv_file' => 'required|file|mimes:csv,txt'
        ]);
        
        $file = $request->file('csv_file');
        $csv = Reader::createFromPath($file->getPathname(), 'r');
        $csv->setHeaderOffset(0);
        
        $count = 0;
        foreach ($csv as $record) {
            // Check if stock already exists
            $stock = Stock::where('symbol', $record['Symbol'])->first();
            
            if (!$stock) {
                Stock::create([
                    'symbol' => $record['Symbol'],
                    'name' => $record['Name'],
                    'description' => $record['Description'] ?? null
                ]);
                $count++;
            }
        }
        
        return back()->with('success', "$count stocks imported successfully!");
    }
    
    public function exportStockData(Request $request)
    {
        $request->validate([
            'stock_id' => 'required|exists:stocks,id'
        ]);
        
        $stock = Stock::find($request->stock_id);
        $stockData = $stock->stockData()->orderBy('date', 'desc')->get();
        
        $csv = Writer::createFromFileObject(new SplTempFileObject());
        
        // Add headers
        $csv->insertOne(['Date', 'Open', 'High', 'Low', 'Close', 'Volume']);
        
        // Add data
        foreach ($stockData as $data) {
            $csv->insertOne([
                $data->date,
                $data->open,
                $data->high,
                $data->low,
                $data->close,
                $data->volume
            ]);
        }
        
        $headers = [
            'Content-Type' => 'text/csv',
            'Content-Disposition' => "attachment; filename=\"{$stock->symbol}_data.csv\"",
            'Pragma' => 'no-cache',
            'Cache-Control' => 'must-revalidate, post-check=0, pre-check=0',
            'Expires' => '0'
        ];
        
        return response($csv->getContent(), 200, $headers);
    }
    
    public function importStockData(Request $request)
    {
        $request->validate([
            'stock_id' => 'required|exists:stocks,id',
            'csv_file' => 'required|file|mimes:csv,txt'
        ]);
        
        $stock = Stock::find($request->stock_id);
        
        $file = $request->file('csv_file');
        $csv = Reader::createFromPath($file->getPathname(), 'r');
        $csv->setHeaderOffset(0);
        
        $count = 0;
        foreach ($csv as $record) {
            // Check if data for this date already exists
            $exists = StockData::where('stock_id', $stock->id)
                ->where('date', $record['Date'])
                ->exists();
            
            if (!$exists) {
                StockData::create([
                    'stock_id' => $stock->id,
                    'date' => $record['Date'],
                    'open' => $record['Open'],
                    'high' => $record['High'],
                    'low' => $record['Low'],
                    'close' => $record['Close'],
                    'volume' => $record['Volume']
                ]);
                $count++;
            }
        }
        
        return back()->with('success', "$count data points imported for {$stock->symbol}!");
    }
}