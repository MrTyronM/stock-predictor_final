<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\DB;

class NotificationController extends Controller
{
    /**
     * Display the user's notifications.
     *
     * @return \Illuminate\Http\Response
     */
    public function index()
    {
        $user = Auth::user();
        
        // Get paginated notifications
        $notifications = $user->notifications()->paginate(10);
        
        // Get the column names from the notifications table
        $columns = DB::getSchemaBuilder()->getColumnListing('notifications');
        
        // Check if read_at or status column exists
        $hasReadAt = in_array('read_at', $columns);
        $hasStatus = in_array('status', $columns);
        
        // Count unread notifications based on available columns
        if ($hasReadAt) {
            $unreadCount = $user->notifications()->whereNull('read_at')->count();
        } elseif ($hasStatus) {
            $unreadCount = $user->notifications()->where('status', 'unread')->count();
        } else {
            // If no read tracking columns exist, assume all are unread
            $unreadCount = $user->notifications()->count();
        }
        
        return view('notifications.index', compact('notifications', 'unreadCount', 'hasReadAt', 'hasStatus'));
    }
    
    /**
     * Mark a single notification as read.
     *
     * @param  string  $id
     * @return \Illuminate\Http\Response
     */
    public function markRead($id)
    {
        $notification = DB::table('notifications')->where('id', $id)->first();
        
        if (!$notification) {
            return redirect()->route('notifications.index')
                ->with('error', 'Notification not found.');
        }
        
        // Check if the notification belongs to the authenticated user
        if ($notification->user_id != Auth::id()) {
            return redirect()->route('notifications.index')
                ->with('error', 'You are not authorized to perform this action.');
        }
        
        // Get the column names from the notifications table
        $columns = DB::getSchemaBuilder()->getColumnListing('notifications');
        
        // Update notification based on available columns
        if (in_array('read_at', $columns)) {
            DB::table('notifications')
                ->where('id', $id)
                ->update(['read_at' => now()]);
        } elseif (in_array('status', $columns)) {
            DB::table('notifications')
                ->where('id', $id)
                ->update(['status' => 'read']);
        }
        
        return redirect()->route('notifications.index')
            ->with('success', 'Notification marked as read.');
    }
    
    /**
     * Mark all notifications as read.
     *
     * @return \Illuminate\Http\Response
     */
    public function markAllRead()
    {
        $user = Auth::user();
        
        // Get the column names from the notifications table
        $columns = DB::getSchemaBuilder()->getColumnListing('notifications');
        
        // Update notifications based on available columns
        if (in_array('read_at', $columns)) {
            $user->notifications()->whereNull('read_at')->update(['read_at' => now()]);
        } elseif (in_array('status', $columns)) {
            $user->notifications()->where('status', 'unread')->update(['status' => 'read']);
        }
        
        return redirect()->route('notifications.index')
            ->with('success', 'All notifications marked as read.');
    }
}