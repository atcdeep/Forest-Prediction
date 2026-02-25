-- Drop existing database if needed
DROP DATABASE IF EXISTS forest_prediction_db;

-- Create database
CREATE DATABASE forest_prediction_db;
USE forest_prediction_db;

-- Create users table
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    uid VARCHAR(255) UNIQUE,
    email VARCHAR(255) UNIQUE,
    name VARCHAR(255),
    photo_url TEXT,
    provider VARCHAR(50),
    role VARCHAR(50) DEFAULT 'user',
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create datasets table with user isolation
CREATE TABLE datasets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500),
    row_count INT DEFAULT 0,
    column_count INT DEFAULT 0,
    is_primary BOOLEAN DEFAULT FALSE,
    is_default BOOLEAN DEFAULT FALSE,
    uploaded_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    UNIQUE KEY unique_user_dataset (user_id, name),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create predictions table with user isolation
CREATE TABLE predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    dataset_name VARCHAR(255),
    predicted_metric VARCHAR(255),
    year INT,
    predicted_value FLOAT,
    accuracy FLOAT,
    prediction_type VARCHAR(50) DEFAULT 'standard',
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_used VARCHAR(100) DEFAULT 'linear_regression',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create dataset_columns table with user isolation
CREATE TABLE dataset_columns (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    dataset_name VARCHAR(255),
    column_name VARCHAR(255),
    data_type VARCHAR(50),
    is_numeric BOOLEAN DEFAULT FALSE,
    min_value FLOAT,
    max_value FLOAT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create user activity table
CREATE TABLE user_activity (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    activity_type VARCHAR(100),
    description TEXT,
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create indexes for better performance
CREATE INDEX idx_predictions_user ON predictions(user_id);
CREATE INDEX idx_datasets_user ON datasets(user_id);
CREATE INDEX idx_predictions_date ON predictions(prediction_date);
CREATE INDEX idx_datasets_primary ON datasets(is_primary);
CREATE INDEX idx_datasets_name ON datasets(name);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_activity_user ON user_activity(user_id);
CREATE INDEX idx_activity_date ON user_activity(created_at);