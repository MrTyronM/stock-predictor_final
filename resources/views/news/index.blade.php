    @extends('layouts.app')

    @section('content')
        <h2 class="mb-3">Financial News</h2>
        
        <div class="card">
            <div class="card-body">
                @forelse($news as $item)
                    <div class="news-item mb-4" style="border-bottom: 1px solid var(--border-color); padding-bottom: 1rem;">
                        <h3><a href="{{ route('news.show', $item) }}" style="color: var(--accent-primary);">{{ $item->title }}</a></h3>
                        <p class="text-secondary" style="font-size: 0.9rem;">
                            {{ $item->published_date->format('F j, Y') }} | Source: {{ $item->source ?? 'Unknown' }}
                        </p>
                        <p>{{ Str::limit(strip_tags($item->content), 200) }}</p>
                        
                        @if($item->stocks->count() > 0)
                            <div class="mt-2">
                                <span class="text-secondary">Related stocks:</span>
                                @foreach($item->stocks as $stock)
                                    <a href="{{ route('stocks.show', $stock) }}" class="badge" style="background-color: var(--bg-tertiary); color: var(--text-primary); text-decoration: none; margin-right: 0.5rem; padding: 0.25rem 0.5rem; border-radius: 4px;">
                                        {{ $stock->symbol }}
                                    </a>
                                @endforeach
                            </div>
                        @endif
                    </div>
                @empty
                    <p>No news articles available at this time.</p>
                @endforelse
                
                <div class="mt-4">
                    {{ $news->links() }}
                </div>
            </div>
        </div>
    @endsection