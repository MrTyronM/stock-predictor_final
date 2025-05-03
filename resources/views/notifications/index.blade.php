@extends('layouts.app')

@section('content')
<div class="notifications-container">
    <div class="header-with-actions">
        <h1 class="page-title">Notifications</h1>
        
        @if($unreadCount > 0)
            <form action="{{ route('notifications.mark-all-read') }}" method="POST" class="d-inline">
                @csrf
                <button type="submit" class="btn btn-sm btn-primary">
                    <i class="fas fa-check-double"></i> Mark All as Read
                </button>
            </form>
        @endif
    </div>

    <div class="notifications-list">
        @forelse($notifications as $notification)
            @php
                $isRead = false;
                if (isset($hasReadAt) && $hasReadAt && isset($notification->read_at) && $notification->read_at) {
                    $isRead = true;
                } elseif (isset($hasStatus) && $hasStatus && isset($notification->status) && $notification->status === 'read') {
                    $isRead = true;
                }
                
                // Get notification data
                $data = json_decode($notification->data ?? '{}', true);
                $notificationType = isset($notification->type) ? class_basename($notification->type) : '';
            @endphp
            
            <div class="notification-item {{ $isRead ? 'read' : 'unread' }}">
                <div class="notification-icon">
                    @if(str_contains($notificationType, 'StockAlert') || str_contains($data['type'] ?? '', 'stock'))
                        <i class="fas fa-chart-line"></i>
                    @elseif(str_contains($notificationType, 'PortfolioUpdate') || str_contains($data['type'] ?? '', 'portfolio'))
                        <i class="fas fa-briefcase"></i>
                    @elseif(str_contains($notificationType, 'NewsAlert') || str_contains($data['type'] ?? '', 'news'))
                        <i class="fas fa-newspaper"></i>
                    @else
                        <i class="fas fa-bell"></i>
                    @endif
                </div>
                <div class="notification-content">
                    <div class="notification-title">
                        {{ $data['title'] ?? ($notification->title ?? 'Notification') }}
                    </div>
                    <div class="notification-message">
                        {{ $data['message'] ?? ($notification->message ?? '') }}
                    </div>
                    <div class="notification-meta">
                        <span class="notification-time">{{ \Carbon\Carbon::parse($notification->created_at)->diffForHumans() }}</span>
                        @if(!$isRead)
                            <form action="{{ route('notifications.mark-read', $notification->id) }}" method="POST" class="d-inline">
                                @csrf
                                <button type="submit" class="btn btn-sm btn-link">Mark as read</button>
                            </form>
                        @endif
                    </div>
                </div>
            </div>
        @empty
            <div class="no-notifications">
                <i class="fas fa-bell-slash"></i>
                <p>You have no notifications</p>
            </div>
        @endforelse
    </div>

    {{ $notifications->links() }}
</div>

<style>
    .notifications-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .header-with-actions {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .page-title {
        font-size: 1.8rem;
        margin: 0;
        color: var(--accent-color);
    }
    
    .notifications-list {
        background-color: #1e1e1e;
        border-radius: 8px;
        overflow: hidden;
    }
    
    .notification-item {
        display: flex;
        padding: 16px;
        border-bottom: 1px solid #333;
        transition: background-color 0.2s;
    }
    
    .notification-item:last-child {
        border-bottom: none;
    }
    
    .notification-item:hover {
        background-color: #2a2a2a;
    }
    
    .notification-item.unread {
        background-color: rgba(0, 230, 118, 0.05);
    }
    
    .notification-icon {
        flex: 0 0 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #333;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 15px;
        color: var(--accent-color);
    }
    
    .notification-content {
        flex: 1;
    }
    
    .notification-title {
        font-weight: 500;
        margin-bottom: 5px;
    }
    
    .notification-item.unread .notification-title {
        font-weight: 700;
        color: var(--accent-color);
    }
    
    .notification-message {
        font-size: 0.9rem;
        color: #aaa;
        margin-bottom: 10px;
    }
    
    .notification-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.8rem;
        color: #777;
    }
    
    .btn-link {
        padding: 0;
        color: var(--accent-color);
        background: none;
        border: none;
        text-decoration: underline;
        cursor: pointer;
        font-size: 0.8rem;
    }
    
    .no-notifications {
        padding: 40px;
        text-align: center;
        color: #666;
    }
    
    .no-notifications i {
        font-size: 3rem;
        margin-bottom: 15px;
    }
</style>
@endsection