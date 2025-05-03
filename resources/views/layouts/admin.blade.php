<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>{{ config('app.name') }} - Admin</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.1/css/all.min.css">
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <style>
        /* Dark theme matching your main site */
        :root {
            --primary-color: #00c853;
            --sidebar-bg: #212121;
            --main-bg: #121212;
            --card-bg: #262626;
            --text-color: #ffffff;
            --text-muted: #bdbdbd;
            --border-color: #424242;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            background-color: var(--main-bg);
            color: var(--text-color);
            margin: 0;
            padding: 0;
        }
        
        .wrapper {
            display: flex;
            width: 100%;
            min-height: 100vh;
        }
        
        /* Sidebar styles to match your main site */
        #sidebar {
            background-color: var(--sidebar-bg);
            width: 260px;
            min-width: 260px;
            min-height: 100vh;
            transition: all 0.3s;
            z-index: 100;
            color: var(--text-color);
        }
        
        #sidebar .logo {
            padding: 15px 20px;
            font-size: 24px;
            color: var(--primary-color);
            border-bottom: 1px solid var(--border-color);
            font-weight: bold;
        }
        
        #sidebar .sidebar-heading {
            padding: 10px 20px;
            font-size: 14px;
            color: var(--text-muted);
            text-transform: uppercase;
        }
        
        #sidebar .nav-item {
            margin-bottom: 5px;
        }
        
        #sidebar .nav-link {
            color: var(--text-color);
            padding: 15px 20px;
            font-size: 16px;
            transition: all 0.3s;
            display: flex;
            align-items: center;
        }
        
        #sidebar .nav-link i {
            margin-right: 10px;
            width: 25px;
            text-align: center;
        }
        
        #sidebar .nav-link:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
        
        #sidebar .nav-link.active {
            background-color: var(--primary-color);
            color: #212121;
        }
        
        /* Content area */
        #content {
            flex: 1;
            padding: 20px;
            transition: all 0.3s;
        }
        
        /* Navbar */
        .navbar {
            background-color: var(--sidebar-bg);
            border-bottom: 1px solid var(--border-color);
            padding: 15px 20px;
        }
        
        .navbar .navbar-toggler {
            border-color: var(--text-color);
            color: var(--text-color);
        }
        
        .navbar-brand {
            color: var(--primary-color);
            font-weight: bold;
        }
        
        /* Card styling */
        .card {
            background-color: var(--card-bg);
            border: none;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .card-header {
            background-color: rgba(255, 255, 255, 0.05);
            border-bottom: 1px solid var(--border-color);
            padding: 15px 20px;
            font-weight: 600;
        }
        
        .card-body {
            padding: 20px;
        }
        
        /* Form styling */
        .form-control, .form-select {
            background-color: var(--main-bg);
            border: 1px solid var(--border-color);
            color: var(--text-color);
        }
        
        .form-control:focus, .form-select:focus {
            background-color: var(--main-bg);
            color: var(--text-color);
            border-color: var(--primary-color);
            box-shadow: 0 0 0 0.25rem rgba(0, 200, 83, 0.25);
        }
        
        /* Buttons */
        .btn-primary {
            background-color: var(--primary-color);
            border-color: var(--primary-color);
            color: #212121;
        }
        
        .btn-primary:hover {
            background-color: #00b34a;
            border-color: #00b34a;
            color: #212121;
        }
        
        /* Tables */
        .table {
            color: var(--text-color);
        }
        
        .table thead th {
            background-color: rgba(255, 255, 255, 0.05);
            border-color: var(--border-color);
        }
        
        .table td {
            border-color: var(--border-color);
        }
        
        /* Border card styles */
        .border-left-primary {
            border-left: 4px solid var(--primary-color);
        }
        
        .border-left-success {
            border-left: 4px solid #00e676;
        }
        
        .border-left-info {
            border-left: 4px solid #00b0ff;
        }
        
        /* Dropdown */
        .dropdown-menu {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
        }
        
        .dropdown-item {
            color: var(--text-color);
        }
        
        .dropdown-item:hover {
            background-color: rgba(255, 255, 255, 0.1);
            color: var(--text-color);
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            #sidebar {
                margin-left: -260px;
            }
            
            #sidebar.active {
                margin-left: 0;
            }
            
            #content {
                width: 100%;
            }
        }
    </style>
    
    @yield('styles')
</head>
<body>
    <div class="wrapper">
        <!-- Sidebar -->
        <nav id="sidebar">
            <div class="logo">
                <span>SP</span>
            </div>
            
            <ul class="nav flex-column mt-3">
                <li class="nav-item">
                    <a href="{{ route('admin.dashboard') }}" class="nav-link {{ request()->routeIs('admin.dashboard') ? 'active' : '' }}">
                        <i class="fas fa-home"></i> Dashboard
                    </a>
                </li>
                
                <li class="nav-item">
                    <a href="{{ route('admin.model-training') }}" class="nav-link {{ request()->routeIs('admin.model-training') ? 'active' : '' }}">
                        <i class="fas fa-brain"></i> Model Training
                    </a>
                </li>
                
                <li class="nav-item">
                    <a href="{{ route('admin.stocks') }}" class="nav-link {{ request()->routeIs('admin.stocks*') ? 'active' : '' }}">
                        <i class="fas fa-chart-line"></i> Stocks
                    </a>
                </li>
                
                <li class="nav-item">
                    <a href="{{ route('admin.users') }}" class="nav-link {{ request()->routeIs('admin.users') ? 'active' : '' }}">
                        <i class="fas fa-users"></i> Users
                    </a>
                </li>
                
                <li class="nav-item">
                    <a href="{{ route('admin.feedback') }}" class="nav-link {{ request()->routeIs('admin.feedback') ? 'active' : '' }}">
                        <i class="fas fa-comments"></i> Feedback
                    </a>
                </li>
                
                <li class="nav-item">
                    <a href="{{ route('admin.model-accuracy') }}" class="nav-link {{ request()->routeIs('admin.model-accuracy') ? 'active' : '' }}">
                        <i class="fas fa-bullseye"></i> Model Accuracy
                    </a>
                </li>
                
                <li class="nav-item mt-3">
                    <a href="{{ route('dashboard') }}" class="nav-link">
                        <i class="fas fa-arrow-left"></i> Back to Site
                    </a>
                </li>
                
                <li class="nav-item">
                    <a href="#" onclick="event.preventDefault(); document.getElementById('logout-form').submit();" class="nav-link">
                        <i class="fas fa-sign-out-alt"></i> Logout
                    </a>
                </li>
            </ul>
            
            <form id="logout-form" action="{{ route('logout') }}" method="POST" class="d-none">
                @csrf
            </form>
        </nav>

        <!-- Page Content -->
        <div id="content">
            <nav class="navbar navbar-expand-lg">
                <div class="container-fluid">
                    <button type="button" id="sidebarCollapse" class="btn btn-outline-light">
                        <i class="fas fa-bars"></i>
                    </button>
                    
                    <div class="ms-auto">
                        <div class="dropdown">
                            <button class="btn btn-outline-light dropdown-toggle" type="button" id="userDropdown" data-bs-toggle="dropdown" aria-expanded="false">
                                Admin User
                            </button>
                            <ul class="dropdown-menu dropdown-menu-end" aria-labelledby="userDropdown">
                                <li>
                                    <a class="dropdown-item" href="#" onclick="event.preventDefault(); document.getElementById('logout-form').submit();">
                                        <i class="fas fa-sign-out-alt"></i> Logout
                                    </a>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
            </nav>

            <main>
                @yield('content')
            </main>
        </div>
    </div>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    
    <!-- jQuery -->
    <script src="https://code.jquery.com/jquery-3.6.3.min.js"></script>
    
    <script>
        $(document).ready(function () {
            $('#sidebarCollapse').on('click', function () {
                $('#sidebar').toggleClass('active');
            });
        });
    </script>
    
    @yield('scripts')
</body>
</html>