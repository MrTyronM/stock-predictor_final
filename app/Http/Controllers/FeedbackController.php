<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Feedback;
use Illuminate\Support\Facades\Auth;

class FeedbackController extends Controller
{
    /**
     * Store a newly created feedback in storage.
     *
     * @param  \Illuminate\Http\Request  $request
     * @return \Illuminate\Http\Response
     */
    public function store(Request $request)
    {
        $validated = $request->validate([
            'prediction_id' => 'nullable|exists:predictions,id',
            'comment' => 'required|string|max:500',
            'rating' => 'nullable|integer|min:1|max:5',
        ]);
        
        $feedback = new Feedback();
        $feedback->user_id = Auth::id();
        $feedback->prediction_id = $validated['prediction_id'];
        $feedback->comment = $validated['comment'];
        $feedback->rating = $validated['rating'];
        $feedback->save();
        
        return back()->with('success', 'Feedback submitted successfully!');
    }

    /**
     * Display a listing of the feedback.
     *
     * @return \Illuminate\Http\Response
     */
    public function index()
    {
        // This would be for admin viewing all feedback
        $feedback = Feedback::with(['user', 'prediction.stock'])->latest()->paginate(15);
        return view('feedback.index', compact('feedback'));
    }

    /**
     * Show the form for creating a new feedback.
     *
     * @return \Illuminate\Http\Response
     */
    public function create()
    {
        return view('feedback.create');
    }

    /**
     * Display the specified feedback.
     *
     * @param  \App\Models\Feedback  $feedback
     * @return \Illuminate\Http\Response
     */
    public function show(Feedback $feedback)
    {
        return view('feedback.show', compact('feedback'));
    }

    /**
     * Remove the specified feedback from storage.
     *
     * @param  \App\Models\Feedback  $feedback
     * @return \Illuminate\Http\Response
     */
    public function destroy(Feedback $feedback)
    {
        // Add authorization check to ensure only admins or the feedback owner can delete
        if (Auth::check() && (Auth::user()->isAdmin() || Auth::id() === $feedback->user_id)) {
            $feedback->delete();
            return redirect()->route('feedback.index')->with('success', 'Feedback deleted successfully');
        }
        
        return abort(403, 'Unauthorized action');
    }
}