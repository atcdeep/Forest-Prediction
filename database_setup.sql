-- ============================================================
-- Forest Prediction System - SQLite Database Schema
-- Database file: forest_prediction.db
-- Usage: This schema is auto-applied on first app start via
--        init_database() in app.py. Run manually with:
--          sqlite3 forest_prediction.db < database_setup.sql
-- ============================================================

PRAGMA foreign_keys = ON;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uid TEXT UNIQUE,
    email TEXT UNIQUE,
    name TEXT,
    photo_url TEXT,
    provider TEXT,
    role TEXT DEFAULT 'user',
    is_active INTEGER DEFAULT 1,        -- 1=active, 0=inactive
    global_library_enabled INTEGER DEFAULT 1,
    last_login DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Datasets table (per-user)
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    file_path TEXT,
    row_count INTEGER DEFAULT 0,
    column_count INTEGER DEFAULT 0,
    is_primary INTEGER DEFAULT 0,  -- 1=primary dataset
    is_default INTEGER DEFAULT 0,  -- 1=default forest_data
    uploaded_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    UNIQUE (user_id, name),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    dataset_name TEXT,
    predicted_metric TEXT,
    year INTEGER,
    predicted_value REAL,
    accuracy REAL,
    prediction_type TEXT DEFAULT 'standard',
    prediction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    model_used TEXT DEFAULT 'linear_regression',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Dataset column metadata
CREATE TABLE IF NOT EXISTS dataset_columns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    dataset_name TEXT,
    column_name TEXT,
    data_type TEXT,
    is_numeric INTEGER DEFAULT 0,
    min_value REAL,
    max_value REAL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- User activity log
CREATE TABLE IF NOT EXISTS user_activity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    activity_type TEXT,
    description TEXT,
    ip_address TEXT,
    user_agent TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Global datasets (admin-managed, visible to all users)
CREATE TABLE IF NOT EXISTS global_datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    file_path TEXT,
    row_count INTEGER DEFAULT 0,
    column_count INTEGER DEFAULT 0,
    description TEXT,
    tags TEXT,
    uploaded_by INTEGER,
    is_active INTEGER DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (uploaded_by) REFERENCES users(id) ON DELETE SET NULL
);

-- Dataset approval workflow
CREATE TABLE IF NOT EXISTS dataset_approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_user_id INTEGER NOT NULL,
    dataset_name TEXT NOT NULL,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending','approved','rejected')),
    admin_note TEXT,
    reviewed_by INTEGER,
    reviewed_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (reviewed_by) REFERENCES users(id) ON DELETE SET NULL
);

-- Favorite / starred datasets
CREATE TABLE IF NOT EXISTS favorites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    dataset_name TEXT NOT NULL,
    label TEXT DEFAULT 'favorite' CHECK(label IN ('favorite','primary','archived')),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, dataset_name),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- AI chat history
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
    message TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Dataset cleaning audit log
CREATE TABLE IF NOT EXISTS dataset_clean_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    dataset_name TEXT NOT NULL,
    operation TEXT,
    rows_before INTEGER,
    rows_after INTEGER,
    details TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Notifications
CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    type TEXT DEFAULT 'info',
    title TEXT,
    message TEXT,
    is_read INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Support queries
CREATE TABLE IF NOT EXISTS support_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    issue_type TEXT,
    description TEXT,
    screenshot_path TEXT,
    status TEXT DEFAULT 'open' CHECK(status IN ('open','resolved')),
    admin_response TEXT,
    resolved_by INTEGER,
    resolved_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (resolved_by) REFERENCES users(id) ON DELETE SET NULL
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_predictions_user    ON predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_datasets_user       ON datasets(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_date    ON predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_datasets_primary    ON datasets(is_primary);
CREATE INDEX IF NOT EXISTS idx_datasets_name       ON datasets(name);
CREATE INDEX IF NOT EXISTS idx_users_email         ON users(email);
CREATE INDEX IF NOT EXISTS idx_activity_user       ON user_activity(user_id);
CREATE INDEX IF NOT EXISTS idx_activity_date       ON user_activity(created_at);
CREATE INDEX IF NOT EXISTS idx_favorites_user      ON favorites(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_user           ON chat_history(user_id);
CREATE INDEX IF NOT EXISTS idx_approvals_status    ON dataset_approvals(status);
CREATE INDEX IF NOT EXISTS idx_notifications_user  ON notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_support_status      ON support_queries(status);

-- Default admin user (INSERT OR IGNORE = skip if already exists)
INSERT OR IGNORE INTO users (uid, email, name, role, last_login)
VALUES ('admin_local', 'admin@forestpredict.com', 'Admin User', 'admin', CURRENT_TIMESTAMP);