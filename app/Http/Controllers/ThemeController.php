<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class ThemeController extends Controller
{
    public function toggle(Request $request)
    {
        $theme = $request->session()->get('theme', 'dark');
        $newTheme = $theme === 'dark' ? 'light' : 'dark';
        
        $request->session()->put('theme', $newTheme);
        
        return redirect()->back();
    }
}