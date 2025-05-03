<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Stock;
use App\Models\StockData;
use App\Models\Prediction;
use App\Models\Portfolio;
use App\Models\PortfolioStock;
use PhpOffice\PhpSpreadsheet\Spreadsheet;
use PhpOffice\PhpSpreadsheet\Writer\Xlsx;
use Carbon\Carbon;
use Illuminate\Support\Facades\Auth;
use Barryvdh\DomPDF\Facade\Pdf;

class ExportController extends Controller
{
    /**
     * Create a new controller instance.
     *
     * @return void
     */
    public function __construct()
    {
        $this->middleware('auth');
    }

    public function index()
    {
        $stocks = Stock::all();
        $portfolios = Auth::user()->portfolios;

        return view('exports.index', compact('stocks', 'portfolios'));
    }

    public function exportStockData(Request $request)
    {
        $request->validate([
            'stock_id' => 'required|exists:stocks,id',
            'start_date' => 'required|date',
            'end_date' => 'required|date|after_or_equal:start_date',
            'format' => 'required|in:csv,xlsx,pdf'
        ]);

        $stock = Stock::findOrFail($request->stock_id);

        $data = StockData::where('stock_id', $stock->id)
            ->whereBetween('date', [$request->start_date, $request->end_date])
            ->orderBy('date', 'asc')
            ->get();

        $format = $request->format;
        $filename = "{$stock->symbol}_data_{$request->start_date}_to_{$request->end_date}";

        return $this->generateExport($data, $format, $filename, 'stock_data', $stock);
    }

    public function exportPredictions(Request $request)
    {
        $request->validate([
            'stock_id' => 'required|exists:stocks,id',
            'start_date' => 'required|date',
            'end_date' => 'required|date|after_or_equal:start_date',
            'format' => 'required|in:csv,xlsx,pdf'
        ]);

        $stock = Stock::findOrFail($request->stock_id);

        $data = Prediction::where('stock_id', $stock->id)
            ->whereBetween('prediction_date', [$request->start_date, $request->end_date])
            ->orderBy('prediction_date', 'asc')
            ->get();

        $format = $request->format;
        $filename = "{$stock->symbol}_predictions_{$request->start_date}_to_{$request->end_date}";

        return $this->generateExport($data, $format, $filename, 'predictions', $stock);
    }

    public function exportPortfolio(Request $request)
    {
        $request->validate([
            'portfolio_id' => 'required|exists:portfolios,id',
            'format' => 'required|in:csv,xlsx,pdf'
        ]);

        $portfolio = Portfolio::findOrFail($request->portfolio_id);

        if ($portfolio->user_id != Auth::id()) {
            return back()->with('error', 'You do not have permission to export this portfolio.');
        }

        $data = $portfolio->items()->with('stock')->get();
        $format = $request->format;
        $filename = "portfolio_{$portfolio->name}_" . now()->format('Y-m-d');

        return $this->generateExport($data, $format, $filename, 'portfolio', $portfolio);
    }

    protected function generateExport($data, $format, $filename, $type, $context = null)
    {
        switch ($format) {
            case 'csv':
                $headers = [
                    'Content-Type' => 'text/csv',
                    'Content-Disposition' => "attachment; filename={$filename}.csv",
                ];

                $callback = function () use ($data, $type) {
                    $handle = fopen('php://output', 'w');

                    switch ($type) {
                        case 'stock_data':
                            fputcsv($handle, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']);
                            foreach ($data as $row) {
                                fputcsv($handle, [$row->date, $row->open, $row->high, $row->low, $row->close, $row->volume]);
                            }
                            break;
                        case 'predictions':
                            fputcsv($handle, ['Prediction Date', 'Predicted Price', 'Recommendation', 'Confidence']);
                            foreach ($data as $row) {
                                fputcsv($handle, [$row->prediction_date, $row->predicted_price, ucfirst($row->recommendation), $row->confidence]);
                            }
                            break;
                        case 'portfolio':
                            fputcsv($handle, ['Stock Symbol', 'Stock Name', 'Shares', 'Purchase Price']);
                            foreach ($data as $item) {
                                fputcsv($handle, [$item->stock->symbol, $item->stock->name, $item->shares, $item->purchase_price]);
                            }
                            break;
                    }

                    fclose($handle);
                };

                return response()->stream($callback, 200, $headers);

            case 'xlsx':
                $spreadsheet = new Spreadsheet();
                $sheet = $spreadsheet->getActiveSheet();

                switch ($type) {
                    case 'stock_data':
                        $sheet->fromArray(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], null, 'A1');
                        $row = 2;
                        foreach ($data as $item) {
                            $sheet->fromArray([$item->date, $item->open, $item->high, $item->low, $item->close, $item->volume], null, 'A' . $row);
                            $row++;
                        }
                        break;
                    case 'predictions':
                        $sheet->fromArray(['Prediction Date', 'Predicted Price', 'Recommendation', 'Confidence'], null, 'A1');
                        $row = 2;
                        foreach ($data as $item) {
                            $sheet->fromArray([$item->prediction_date, $item->predicted_price, ucfirst($item->recommendation), $item->confidence], null, 'A' . $row);
                            $row++;
                        }
                        break;
                    case 'portfolio':
                        $sheet->fromArray(['Stock Symbol', 'Stock Name', 'Shares', 'Purchase Price'], null, 'A1');
                        $row = 2;
                        foreach ($data as $item) {
                            $sheet->fromArray([$item->stock->symbol, $item->stock->name, $item->shares, $item->purchase_price], null, 'A' . $row);
                            $row++;
                        }
                        break;
                }

                $writer = new Xlsx($spreadsheet);
                $tempFile = tempnam(sys_get_temp_dir(), 'xlsx');
                $writer->save($tempFile);

                return response()->download($tempFile, $filename . '.xlsx')->deleteFileAfterSend(true);

            case 'pdf':
                $view = '';
                $viewData = [];

                switch ($type) {
                    case 'stock_data':
                        $view = 'exports.pdf.stock_data';
                        $viewData = ['stock' => $context, 'data' => $data];
                        break;
                    case 'predictions':
                        $view = 'exports.pdf.predictions';
                        $viewData = ['stock' => $context, 'data' => $data];
                        break;
                    case 'portfolio':
                        $view = 'exports.pdf.portfolio';
                        $viewData = ['portfolio' => $context, 'items' => $data];
                        break;
                }

                $pdf = PDF::loadView($view, $viewData);
                return $pdf->download($filename . '.pdf');
        }

        return back()->with('error', 'Unsupported export format selected.');
    }
}