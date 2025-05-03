@extends('layouts.app')

@section('content')
    <div class="mb-3">
        <a href="{{ route('news.index') }}" class="btn btn-secondary">&larr; Back to News</a>
    </div>

    <div class="card">
        <div class="card-header">
            <h2 class="card-title">{{ $news->title }}</h2>
            <p class="mb-0 text-secondary" style="font-size: 0.9rem;">
                {{ $news->published_date->format('F j, Y') }} | Source: {{ $news->source ?? 'Unknown' }}
                @if($news->url)
                    | <a href="{{ $news->url }}" target="_blank" rel="noopener noreferrer" style="color: var(--accent-primary);">Original article</a>
                @endif
            </p>
        </div>
        <div class="card-body">
            <div class="news-content">
                {!! $news->content !!}
            </div>
            
            @if($news->stocks->count() > 0)
                <div class="mt-4">
                    <h4>Related Stocks</h4>
                    <div class="mt-2">
                        @foreach($news->stocks as $stock)
                            <div class="card mb-2" style="background-color: var(--bg-tertiary);">
                                <div class="card-body" style="padding: 1rem;">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <div>
                                            <h5 class="mb-0">{{ $stock->symbol }} - {{ $stock->name }}</h5>
                                            @if($stock->predictions->count() > 0)
                                                <span class="prediction-recommendation recommendation-{{ $stock->predictions->first()->recommendation }}">
                                                    {{ ucfirst($stock->predictions->first()->recommendation) }}
                                                </span>
                                            @endif
                                        </div>
                                        <a href="{{ route('stocks.show', $stock) }}" class="btn btn-primary">View Stock</a>
                                    </div>
                                </div>
                            </div>
                        @endforeach
                    </div>
                </div>
            @endif
        </div>
    </div>
@endsection