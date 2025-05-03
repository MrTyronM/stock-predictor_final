@extends('layouts.app')

<link rel="stylesheet" href="{{ asset('css/styles.css') }}">
@section('content')
<div class="admin-dashboard">
    <div class="dashboard-header">
        <h1 class="dashboard-title">Admin Dashboard</h1>
        <div class="dashboard-actions">
            <a href="#" class="btn btn-primary">
                <i class="fas fa-download mr-2"></i> Generate Report
            </a>
        </div>
    </div>

    <!-- Stats Cards Row -->
    <div class="stats-cards">
        <!-- Users Card -->
        <div class="stats-card stats-primary">
            <div class="stats-card-content">
                <div class="stats-info">
                    <h3 class="stats-title">Users</h3>
                    <div class="stats-value">{{ $totalUsers ?? 2 }}</div>
                    <div class="stats-trend">
                        <i class="fas fa-arrow-up stats-trend-icon"></i>
                        <span>{{ $newUsersToday ?? 0 }} new today</span>
                    </div>
                </div>
                <div class="stats-icon">
                    <i class="fas fa-users"></i>
                </div>
            </div>
            <div class="stats-footer">
                <a href="{{ route('admin.users') ?? '#' }}" class="stats-link">View all users</a>
            </div>
        </div>

        <!-- Stocks Card -->
        <div class="stats-card stats-success">
            <div class="stats-card-content">
                <div class="stats-info">
                    <h3 class="stats-title">Stocks</h3>
                    <div class="stats-value">{{ $totalStocks ?? 10 }}</div>
                    <div class="stats-detail">Tracked assets</div>
                </div>
                <div class="stats-icon">
                    <i class="fas fa-chart-line"></i>
                </div>
            </div>
            <div class="stats-footer">
                <a href="{{ route('admin.stocks') ?? '#' }}" class="stats-link">Manage stocks</a>
            </div>
        </div>

        <!-- Predictions Card -->
        <div class="stats-card stats-info">
            <div class="stats-card-content">
                <div class="stats-info">
                    <h3 class="stats-title">Predictions</h3>
                    <div class="stats-value">{{ $totalPredictions ?? 50 }}</div>
                    <div class="stats-detail">Total generated</div>
                </div>
                <div class="stats-icon">
                    <i class="fas fa-brain"></i>
                </div>
            </div>
            <div class="stats-footer">
                <a href="#" class="stats-link">View predictions</a>
            </div>
        </div>

        <!-- Models Card -->
        <div class="stats-card stats-warning">
            <div class="stats-card-content">
                <div class="stats-info">
                    <h3 class="stats-title">Models</h3>
                    <div class="stats-value">{{ $trainedModels ?? 0 }}</div>
                    <div class="stats-detail">Trained & active</div>
                </div>
                <div class="stats-icon">
                    <i class="fas fa-robot"></i>
                </div>
            </div>
            <div class="stats-footer">
                <a href="{{ route('admin.model-training') ?? '#' }}" class="stats-link">Manage models</a>
            </div>
        </div>
    </div>

    <!-- Second Row: Metrics and Activities -->
    <div class="dashboard-metrics-row">
        <!-- Prediction Accuracy Chart -->
        <div class="dashboard-card">
            <div class="card-header">
                <h2 class="card-title">Prediction Accuracy</h2>
                <div class="card-actions">
                    <button class="btn-icon" title="Refresh data">
                        <i class="fas fa-sync-alt"></i>
                    </button>
                </div>
            </div>
            <div class="card-body">
                <div class="accuracy-chart">
                    <div class="accuracy-metric">
                        <div class="metric-title">Directional Accuracy</div>
                        <div class="metric-value">{{ $directionalAccuracy ?? 75.5 }}%</div>
                        <div class="progress">
                            <div class="progress-bar" style="width: {{ $directionalAccuracy ?? 75.5 }}%"></div>
                        </div>
                    </div>
                    <div class="accuracy-metric">
                        <div class="metric-title">Price Accuracy</div>
                        <div class="metric-value">{{ $priceAccuracy ?? 68.3 }}%</div>
                        <div class="progress">
                            <div class="progress-bar" style="width: {{ $priceAccuracy ?? 68.3 }}%"></div>
                        </div>
                    </div>
                    <div class="accuracy-metric">
                        <div class="metric-title">Timing Accuracy</div>
                        <div class="metric-value">{{ $timingAccuracy ?? 82.7 }}%</div>
                        <div class="progress">
                            <div class="progress-bar" style="width: {{ $timingAccuracy ?? 82.7 }}%"></div>
                        </div>
                    </div>
                </div>
                <div class="accuracy-info">
                    <i class="fas fa-info-circle"></i>
                    <span>Based on predictions made in the last 7 days</span>
                </div>
            </div>
        </div>

        <!-- Recent Activity -->
        <div class="dashboard-card">
            <div class="card-header">
                <h2 class="card-title">Recent Activity</h2>
                <div class="card-actions">
                    <a href="#" class="btn-text">View all</a>
                </div>
            </div>
            <div class="card-body">
                <div class="activity-list">
                    @if(isset($recentActivities) && count($recentActivities) > 0)
                        @foreach($recentActivities as $activity)
                            <div class="activity-item">
                                <div class="activity-icon">
                                    <i class="fas {{ $activity->icon }}"></i>
                                </div>
                                <div class="activity-content">
                                    <div class="activity-title">{{ $activity->title }}</div>
                                    <div class="activity-time">{{ $activity->time }}</div>
                                </div>
                            </div>
                        @endforeach
                    @else
                        <div class="activity-item">
                            <div class="activity-icon">
                                <i class="fas fa-user-plus"></i>
                            </div>
                            <div class="activity-content">
                                <div class="activity-title">New user registered</div>
                                <div class="activity-time">2 hours ago</div>
                            </div>
                        </div>
                        <div class="activity-item">
                            <div class="activity-icon">
                                <i class="fas fa-brain"></i>
                            </div>
                            <div class="activity-content">
                                <div class="activity-title">New predictions generated for AAPL</div>
                                <div class="activity-time">4 hours ago</div>
                            </div>
                        </div>
                        <div class="activity-item">
                            <div class="activity-icon">
                                <i class="fas fa-chart-bar"></i>
                            </div>
                            <div class="activity-content">
                                <div class="activity-title">TSLA stock added to system</div>
                                <div class="activity-time">1 day ago</div>
                            </div>
                        </div>
                    @endif
                </div>
            </div>
        </div>
    </div>

    <!-- Quick Actions -->
    <div class="dashboard-card full-width">
        <div class="card-header">
            <h2 class="card-title">Quick Actions</h2>
        </div>
        <div class="card-body">
            <div class="actions-grid">
                <a href="{{ route('admin.model-training') ?? '#' }}" class="action-card">
                    <div class="action-icon">
                        <i class="fas fa-brain"></i>
                    </div>
                    <div class="action-content">
                        <h3 class="action-title">Train Models</h3>
                        <p class="action-desc">Train and manage AI prediction models</p>
                    </div>
                </a>
                <a href="{{ route('admin.stocks') ?? '#' }}" class="action-card">
                    <div class="action-icon">
                        <i class="fas fa-plus-circle"></i>
                    </div>
                    <div class="action-content">
                        <h3 class="action-title">Add Stocks</h3>
                        <p class="action-desc">Add new stocks to the system</p>
                    </div>
                </a>
                <a href="#" class="action-card">
                    <div class="action-icon">
                        <i class="fas fa-cogs"></i>
                    </div>
                    <div class="action-content">
                        <h3 class="action-title">System Settings</h3>
                        <p class="action-desc">Configure system parameters</p>
                    </div>
                </a>
                <a href="#" class="action-card">
                    <div class="action-icon">
                        <i class="fas fa-file-import"></i>
                    </div>
                    <div class="action-content">
                        <h3 class="action-title">Import/Export</h3>
                        <p class="action-desc">Import or export system data</p>
                    </div>
                </a>
            </div>
        </div>
    </div>
</div>

@endsection