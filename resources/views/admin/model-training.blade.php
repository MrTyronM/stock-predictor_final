<!-- resources/views/admin/model-training.blade.php -->
@extends('layouts.admin')

@section('content')
<div class="container-fluid">
    <h1 class="h3 mb-4 text-gray-800">Model Training</h1>

    @if(session('success'))
    <div class="alert alert-success">
        {{ session('success') }}
    </div>
    @endif

    @if(session('error'))
    <div class="alert alert-danger">
        {{ session('error') }}
    </div>
    @endif

    <div class="row">
        <div class="col-lg-6">
            <div class="card shadow mb-4">
                <div class="card-header py-3">
                    <h6 class="m-0 font-weight-bold text-primary">Train New Model</h6>
                </div>
                <div class="card-body">
                    <form action="{{ route('admin.train-model') }}" method="POST">
                        @csrf
                        <div class="form-group">
                            <label for="stock_id">Select Stock</label>
                            <select name="stock_id" id="stock_id" class="form-control @error('stock_id') is-invalid @enderror" required>
                                <option value="">-- Select Stock --</option>
                                @foreach($stocks as $stock)
                                    <option value="{{ $stock->id }}">{{ $stock->symbol }} - {{ $stock->company_name }}</option>
                                @endforeach
                            </select>
                            @error('stock_id')
                                <div class="invalid-feedback">{{ $message }}</div>
                            @enderror
                        </div>
                        
                        <div class="form-group">
                            <label for="model_type">Model Type</label>
                            <select name="model_type" id="model_type" class="form-control @error('model_type') is-invalid @enderror" required>
                                <option value="lstm">LSTM</option>
                                <option value="hybrid" selected>Hybrid</option>
                                <option value="cnn_lstm">CNN-LSTM</option>
                                <option value="bidirectional">Bidirectional</option>
                                <option value="attention">Attention</option>
                            </select>
                            @error('model_type')
                                <div class="invalid-feedback">{{ $message }}</div>
                            @enderror
                        </div>
                        
                        <div class="form-group">
                            <label for="model_complexity">Model Complexity</label>
                            <select name="model_complexity" id="model_complexity" class="form-control @error('model_complexity') is-invalid @enderror" required>
                                <option value="simple">Simple</option>
                                <option value="medium" selected>Medium</option>
                                <option value="complex">Complex</option>
                            </select>
                            @error('model_complexity')
                                <div class="invalid-feedback">{{ $message }}</div>
                            @enderror
                        </div>
                        
                        <div class="form-group">
                            <label for="training_period">Training Period (days)</label>
                            <input type="number" name="training_period" id="training_period" class="form-control @error('training_period') is-invalid @enderror" value="180" min="30" max="365" required>
                            @error('training_period')
                                <div class="invalid-feedback">{{ $message }}</div>
                            @enderror
                        </div>
                        
                        <div class="form-group">
                            <button type="submit" class="btn btn-primary">
                                <i class="fas fa-brain mr-1"></i> Train Model
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
        
        <div class="col-lg-6">
            <div class="card shadow mb-4">
                <div class="card-header py-3">
                    <h6 class="m-0 font-weight-bold text-primary">Existing Model Parameters</h6>
                </div>
                <div class="card-body">
                    @if(isset($modelParams) && $modelParams->count() > 0)
                        <div class="table-responsive">
                            <table class="table table-bordered" width="100%" cellspacing="0">
                                <thead>
                                    <tr>
                                        <th>Stock</th>
                                        <th>Model Type</th>
                                        <th>Accuracy</th>
                                        <th>Last Trained</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    @foreach($modelParams as $param)
                                    <tr>
                                        <td>{{ $param->stock->symbol }}</td>
                                        <td>{{ ucfirst($param->model_type) }} ({{ ucfirst($param->model_complexity) }})</td>
                                        <td>{{ number_format($param->directional_accuracy * 100, 1) }}%</td>
                                        <td>
                                            {{ $param->last_trained ? $param->last_trained->format('M d, Y') : 'N/A' }}
                                        </td>
                                        <td>
                                            <button class="btn btn-sm btn-info view-model" data-id="{{ $param->id }}">
                                                <i class="fas fa-eye"></i>
                                            </button>
                                        </td>
                                    </tr>
                                    @endforeach
                                </tbody>
                            </table>
                        </div>
                    @else
                        <p>No model parameters found. Train a model to see parameters here.</p>
                    @endif
                </div>
            </div>
        </div>
    </div>
    
    <div class="row">
        <div class="col-lg-12">
            <div class="card shadow mb-4">
                <div class="card-header py-3">
                    <h6 class="m-0 font-weight-bold text-primary">Training Process</h6>
                </div>
                <div class="card-body">
                    <h5>How the ML Model Training Works</h5>
                    <p>
                        The model training process involves these steps:
                    </p>
                    
                    <div class="row">
                        <div class="col-md-4 mb-4">
                            <div class="card border-left-primary shadow h-100 py-2">
                                <div class="card-body">
                                    <div class="row no-gutters align-items-center">
                                        <div class="col mr-2">
                                            <div class="text-xs font-weight-bold text-primary text-uppercase mb-1">
                                                Step 1</div>
                                            <div class="h5 mb-0 font-weight-bold text-gray-800">Data Collection & Preprocessing</div>
                                            <p class="mt-2">Historical stock data is downloaded and enriched with technical indicators.</p>
                                        </div>
                                        <div class="col-auto">
                                            <i class="fas fa-database fa-2x text-gray-300"></i>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="col-md-4 mb-4">
                            <div class="card border-left-success shadow h-100 py-2">
                                <div class="card-body">
                                    <div class="row no-gutters align-items-center">
                                        <div class="col mr-2">
                                            <div class="text-xs font-weight-bold text-success text-uppercase mb-1">
                                                Step 2</div>
                                            <div class="h5 mb-0 font-weight-bold text-gray-800">Model Training</div>
                                            <p class="mt-2">Deep learning model is trained to identify patterns in the data.</p>
                                        </div>
                                        <div class="col-auto">
                                            <i class="fas fa-brain fa-2x text-gray-300"></i>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="col-md-4 mb-4">
                            <div class="card border-left-info shadow h-100 py-2">
                                <div class="card-body">
                                    <div class="row no-gutters align-items-center">
                                        <div class="col mr-2">
                                            <div class="text-xs font-weight-bold text-info text-uppercase mb-1">
                                                Step 3</div>
                                            <div class="h5 mb-0 font-weight-bold text-gray-800">Evaluation & Storage</div>
                                            <p class="mt-2">Model is evaluated for accuracy and saved for future predictions.</p>
                                        </div>
                                        <div class="col-auto">
                                            <i class="fas fa-check-circle fa-2x text-gray-300"></i>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <h5 class="mt-4">Model Types Explained</h5>
                    <div class="table-responsive">
                        <table class="table table-bordered">
                            <thead>
                                <tr>
                                    <th>Model Type</th>
                                    <th>Description</th>
                                    <th>Best For</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>LSTM</td>
                                    <td>Long Short-Term Memory networks, specialized for sequential data.</td>
                                    <td>Capturing long-term patterns in price movements</td>
                                </tr>
                                <tr>
                                    <td>Hybrid</td>
                                    <td>Combination of CNN, LSTM, and attention mechanisms.</td>
                                    <td>General purpose with best overall performance</td>
                                </tr>
                                <tr>
                                    <td>CNN-LSTM</td>
                                    <td>Uses CNN to extract features before passing to LSTM layers.</td>
                                    <td>Technical analysis patterns recognition</td>
                                </tr>
                                <tr>
                                    <td>Bidirectional</td>
                                    <td>Processes data in both forward and backward directions.</td>
                                    <td>Discovering complex relationships in data</td>
                                </tr>
                                <tr>
                                    <td>Attention</td>
                                    <td>Uses attention mechanism to focus on important parts of data.</td>
                                    <td>Identifying key market events and indicators</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Model Details Modal -->
<div class="modal fade" id="modelDetailsModal" tabindex="-1" role="dialog" aria-labelledby="modelDetailsModalLabel" aria-hidden="true">
    <div class="modal-dialog modal-lg" role="document">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title" id="modelDetailsModalLabel">Model Details</h5>
                <button class="close" type="button" data-dismiss="modal" aria-label="Close">
                    <span aria-hidden="true">×</span>
                </button>
            </div>
            <div class="modal-body" id="modelDetailsBody">
                <!-- Content will be loaded dynamically -->
                <div class="d-flex justify-content-center">
                    <div class="spinner-border text-primary" role="status">
                        <span class="sr-only">Loading...</span>
                    </div>
                </div>
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" type="button" data-dismiss="modal">Close</button>
            </div>
        </div>
    </div>
</div>
@endsection

@section('scripts')
<script>
$(document).ready(function() {
    // View model details
    $('.view-model').on('click', function() {
        var modelId = $(this).data('id');
        $('#modelDetailsModal').modal('show');
        
        // In a real implementation, you would load the model details via AJAX
        // For now, just show a placeholder
        setTimeout(function() {
            $('#modelDetailsBody').html(`
                <div class="row">
                    <div class="col-md-6">
                        <h5>Model Parameters</h5>
                        <table class="table table-sm">
                            <tr>
                                <th>Sequence Length:</th>
                                <td>20</td>
                            </tr>
                            <tr>
                                <th>Batch Size:</th>
                                <td>32</td>
                            </tr>
                            <tr>
                                <th>Epochs:</th>
                                <td>50</td>
                            </tr>
                            <tr>
                                <th>Learning Rate:</th>
                                <td>0.001</td>
                            </tr>
                            <tr>
                                <th>Dropout:</th>
                                <td>0.2</td>
                            </tr>
                        </table>
                    </div>
                    <div class="col-md-6">
                        <h5>Performance Metrics</h5>
                        <div class="chart-container">
                            <canvas id="accuracyChart"></canvas>
                        </div>
                    </div>
                </div>
                <div class="row mt-4">
                    <div class="col-12">
                        <h5>Training History</h5>
                        <div class="chart-container">
                            <canvas id="trainingHistoryChart"></canvas>
                        </div>
                    </div>
                </div>
            `);
            
            // Create accuracy chart
            var accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
            var accuracyChart = new Chart(accuracyCtx, {
                type: 'bar',
                data: {
                    labels: ['Directional', 'Price', 'Timing'],
                    datasets: [{
                        label: 'Accuracy (%)',
                        data: [75.5, 68.3, 82.7],
                        backgroundColor: [
                            'rgba(75, 192, 192, 0.2)',
                            'rgba(54, 162, 235, 0.2)',
                            'rgba(153, 102, 255, 0.2)'
                        ],
                        borderColor: [
                            'rgba(75, 192, 192, 1)',
                            'rgba(54, 162, 235, 1)',
                            'rgba(153, 102, 255, 1)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
            
            // Create training history chart
            var historyCtx = document.getElementById('trainingHistoryChart').getContext('2d');
            var historyChart = new Chart(historyCtx, {
                type: 'line',
                data: {
                    labels: Array.from({length: 50}, (_, i) => i + 1),
                    datasets: [{
                        label: 'Training Loss',
                        data: Array.from({length: 50}, (_, i) => 0.5 - 0.4 * (1 - Math.exp(-i/10))),
                        borderColor: 'rgba(255, 99, 132, 1)',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        fill: false,
                        tension: 0.4
                    },
                    {
                        label: 'Validation Loss',
                        data: Array.from({length: 50}, (_, i) => 0.6 - 0.45 * (1 - Math.exp(-i/15)) + 0.05 * Math.sin(i)),
                        borderColor: 'rgba(54, 162, 235, 1)',
                        backgroundColor: 'rgba(54, 162, 235, 0.1)',
                        fill: false,
                        tension: 0.4
                    }]
                },
                options: {
                    scales: {
                        y: {
                            title: {
                                display: true,
                                text: 'Loss'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Epoch'
                            }
                        }
                    }
                }
            });
        }, 500);
    });
});
</script>
@endsection