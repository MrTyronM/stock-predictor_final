<?php

namespace Database\Seeders;

use Illuminate\Database\Seeder;
use App\Models\User;
use Illuminate\Support\Facades\Hash;

class AdminUserSeeder extends Seeder
{
    public function run()
    {
        User::firstOrCreate(
            ['email' => 'admin@stockprediction.com'],
            [
                'name' => 'Admin User',
                'password' => Hash::make('Password123!'),
                'is_admin' => true,
            ]
        );

        $this->command->info('Admin user created successfully.');
        $this->command->info('Email: admin@stockprediction.com');
        $this->command->info('Password: Password123!');
    }
}
