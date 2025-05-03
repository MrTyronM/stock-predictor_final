@extends('layouts.app')

@section('content')
    <div class="card">
        <div class="card-header">
            <h2 class="card-title">About Stock Predictor</h2>
        </div>
        <div class="card-body">
            <h3 class="mb-3 text-accent">Project Background</h3>
            <p>The stock market is complex, and making stock market predictions is very hard for investors. Investors and traders are constantly trying to find tools to help make decisions while investing. These tools could help them to make predictions on a particular stock and look at past data. The traditional stock market has always contained errors in stock market predictions as humans are not enough to analyse such large amounts of data.</p>
            
            <p>In today's era, advancements in machine learning and artificial intelligence (AI) have enabled the creation of more advanced tools capable of analysing large volumes of data, recognizing patterns, and generating data-based forecasts. Our tool uses different deep learning techniques such as LSTM (Long Short-Term Memory) networks. These have recently shown the potential to analyse data and predict stock market movements and allow the investor to make more informed stock market predictions.</p>
            
            <h3 class="mb-3 mt-4 text-accent">Project Aims</h3>
            <ul style="list-style-type: disc; padding-left: 20px;">
                <li>Develop an AI-powered stock market prediction tool using quantitative data (e.g., historical prices, technical indicators) to predict stock prices and identify investment opportunities.</li>
                <li>Create a fully functional application that is user-friendly and accessible to both novice and experienced investors.</li>
                <li>Implement user feedback functionality to improve the application and identify issues within both the UI and predictive model.</li>
                <li>Provide real-time buy, hold, and sell recommendations on a selected range of stocks.</li>
                <li>Continuously improve the tool's accuracy through ongoing model training and refinement.</li>
            </ul>
            
            <h3 class="mb-3 mt-4 text-accent">How It Works</h3>
            <p>Our stock prediction tool uses advanced machine learning algorithms to analyze historical stock data from Yahoo Finance. The core of our prediction system is built on:</p>
            
            <div style="padding-left: 20px;">
                <p><strong class="text-accent">LSTM Neural Networks:</strong> Long Short-Term Memory networks are especially well-suited for financial time series forecasting as they can remember patterns over long periods while adapting to new information.</p>
                
                <p><strong class="text-accent">Technical Indicators:</strong> We incorporate various technical indicators such as moving averages, RSI (Relative Strength Index), and volatility measures to enhance prediction accuracy.</p>
                
                <p><strong class="text-accent">Ensemble Learning:</strong> Using XGBoost, we combine multiple prediction models to produce more accurate and robust forecasts than any single model could provide.</p>
            </div>
            
            <h3 class="mb-3 mt-4 text-accent">Technology Stack</h3>
            <p>Our platform is built using cutting-edge technologies:</p>
            
            <div style="padding-left: 20px;">
                <p><strong class="text-accent">Backend:</strong> Laravel (PHP) for server-side processing, authentication, and database interactions.</p>
                
                <p><strong class="text-accent">Machine Learning:</strong> Python with TensorFlow, Keras, and Scikit-learn for building and training prediction models.</p>
                
                <p><strong class="text-accent">Data Visualization:</strong> JavaScript charting libraries for interactive stock price and prediction visualizations.</p>
                
                <p><strong class="text-accent">Database:</strong> MySQL for storing historical stock data, user interactions, and prediction results.</p>
            </div>
            
            <h3 class="mb-3 mt-4 text-accent">Important Note</h3>
            <p>While we strive for the highest accuracy possible, stock market prediction is inherently uncertain. Our tool provides recommendations based on historical patterns and technical analysis, but all investment decisions should be made with careful consideration and possibly in consultation with a financial advisor. Past performance is not indicative of future results.</p>
        </div>
    </div>
@endsection