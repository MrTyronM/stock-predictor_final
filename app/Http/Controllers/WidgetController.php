<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Widget;
use App\Models\UserWidget;
use Illuminate\Support\Facades\Auth;

class WidgetController extends Controller
{
    public function index()
    {
        $availableWidgets = Widget::all();
        $userWidgets = UserWidget::where('user_id', Auth::id())
            ->orderBy('position', 'asc')
            ->get();
            
        return view('widgets.index', compact('availableWidgets', 'userWidgets'));
    }
    
    public function add(Request $request)
    {
        $request->validate([
            'widget_id' => 'required|exists:widgets,id'
        ]);
        
        // Check if widget already exists for user
        $exists = UserWidget::where('user_id', Auth::id())
            ->where('widget_id', $request->widget_id)
            ->exists();
            
        if ($exists) {
            return back()->with('error', 'Widget already added to your dashboard.');
        }
        
        // Get max position
        $maxPosition = UserWidget::where('user_id', Auth::id())
            ->max('position') ?? 0;
            
        // Add widget
        UserWidget::create([
            'user_id' => Auth::id(),
            'widget_id' => $request->widget_id,
            'position' => $maxPosition + 1,
            'settings' => json_encode([])
        ]);
        
        return back()->with('success', 'Widget added to your dashboard.');
    }
    
    public function remove($id)
    {
        $widget = UserWidget::where('id', $id)
            ->where('user_id', Auth::id())
            ->firstOrFail();
            
        $widget->delete();
        
        // Reorder remaining widgets
        $remainingWidgets = UserWidget::where('user_id', Auth::id())
            ->orderBy('position', 'asc')
            ->get();
            
        $position = 1;
        foreach ($remainingWidgets as $widget) {
            $widget->position = $position++;
            $widget->save();
        }
        
        return back()->with('success', 'Widget removed from your dashboard.');
    }
    
    public function updatePositions(Request $request)
    {
        $request->validate([
            'positions' => 'required|array',
            'positions.*' => 'required|integer|exists:user_widgets,id'
        ]);
        
        $positions = $request->positions;
        
        foreach ($positions as $index => $widgetId) {
            $widget = UserWidget::where('id', $widgetId)
                ->where('user_id', Auth::id())
                ->firstOrFail();
                
            $widget->position = $index + 1;
            $widget->save();
        }
        
        return response()->json(['success' => true]);
    }
    
    public function updateSettings(Request $request, $id)
    {
        $widget = UserWidget::where('id', $id)
            ->where('user_id', Auth::id())
            ->firstOrFail();
            
        $widget->settings = json_encode($request->settings);
        $widget->save();
        
        return response()->json(['success' => true]);
    }
}