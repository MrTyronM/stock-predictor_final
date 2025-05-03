<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('model_parameters', function (Blueprint $table) {
            $table->id();
            $table->foreignId('stock_id')->constrained()->onDelete('cascade');
            $table->string('model_type')->default('hybrid');
            $table->string('model_complexity')->default('medium');
            $table->integer('training_period')->default(180);
            $table->json('parameters')->nullable();
            $table->float('directional_accuracy')->default(0);
            $table->float('price_accuracy')->default(0);
            $table->float('timing_accuracy')->default(0);
            $table->string('log_file')->nullable();
            $table->timestamp('last_trained')->nullable();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('model_parameters');
    }
};