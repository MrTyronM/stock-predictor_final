@extends('layouts.app')

@section('content')
    <div class="mb-3">
        <a href="{{ route('admin.dashboard') }}" class="btn btn-secondary">&larr; Back to Dashboard</a>
    </div>

    <h2 class="mb-3">Bulk Import/Export</h2>
    
    <div class="row">
        <div class="col-md-6">
            <div class="card mb-4">
                <div class="card-header">
                    <h3 class="card-title">Stocks Import/Export</h3>
                </div>
                <div class="card-body">
                    <h4>Export Stocks</h4>
                    <p>Download all stocks as a CSV file.</p>
                    <a href="{{ route('admin.export-stocks') }}" class="btn btn-primary">Export Stocks</a>
                    
                    <hr>
                    
                    <h4>Import Stocks</h4>
                    <p>Upload a CSV file to import stocks. The file should have columns: Symbol, Name, Description (optional).</p>
                    <form action="{{ route('admin.import-stocks') }}" method="POST" enctype="multipart/form-data">
                        @csrf
                        <div class="form-group">
                            <label for="csv_file" class="form-label">CSV File</label>
                            <input type="file" id="csv_file" name="csv_file" class="form-control" required>
                        </div>
                        <button type="submit" class="btn btn-primary">Import Stocks</button>
                    </form>
                </div>
            </div>
        </div>
        
        <div class="col-md-6">
            <div class="card mb-4">
                <div class="card-header">
                    <h3 class="card-title">Stock Data Import/Export</h3>
                </div>
                <div class="card-body">
                    <h4>Export Stock Data</h4>
                    <p>Download historical data for a specific stock as a CSV file.</p>
                    <form action="{{ route('admin.export-stock-data') }}" method="GET">
                        <div class="form-group">
                            <label for="stock_id_export" class="form-label">Select Stock</label>
                            <select id="stock_id_export" name="stock_id" class="form-control" required>
                                <option value="">-- Select a stock --</option>
                                @foreach(App\Models\Stock::orderBy('symbol')->get() as $stock)
                                    <option value="{{ $stock->id }}">{{ $stock->symbol }} - {{ $stock->name }}</option>
                                @endforeach
                            </select>
                        </div>
                        <button type="submit" class="btn btn-primary">Export Stock Data</button>
                    </form>
                    
                    <hr>
                    
                    <h4>Import Stock Data</h4>
                    <p>Upload a CSV file to import historical data for a stock. The file should have columns: Date, Open, High, Low, Close, Volume.</p>
                    <form action="{{ route('admin.import-stock-data') }}" method="POST" enctype="multipart/form-data">
                        @csrf
                        <div class="form-group">
                            <label for="stock_id_import" class="form-label">Select Stock</label>
                            <select id="stock_id_import" name="stock_id" class="form-control" required>
                                <option value="">-- Select a stock --</option>
                                @foreach(App\Models\Stock::orderBy('symbol')->get() as $stock)
                                    <option value="{{ $stock->id }}">{{ $stock->symbol }} - {{ $stock->name }}</option>
                                @endforeach
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="csv_file_data" class="form-label">CSV File</label>
                            <input type="file" id="csv_file_data" name="csv_file" class="form-control" required>
                        </div>
                        <button type="submit" class="btn btn-primary">Import Stock Data</button>
                    </form>
                </div>
            </div>
        </div>
    </div>
@endsection