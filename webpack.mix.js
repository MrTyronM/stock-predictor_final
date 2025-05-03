const mix = require('laravel-mix');

mix.js('resources/js/app.js', 'public/js')
   .js('resources/js/market-dashboard.js', 'public/js')
   .react()
   .sass('resources/sass/app.scss', 'public/css');