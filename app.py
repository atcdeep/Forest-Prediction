from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sqlite3
import os
import json
from datetime import datetime, timedelta
import traceback
import csv
import io
import re
import secrets
from functools import wraps
from contextlib import contextmanager

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)  # For session management
CORS(app)

# SQLite database configuration
DB_PATH = 'forest_prediction.db'

# Context manager for database connections
@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        yield conn
    except sqlite3.Error as e:
        print(f"❌ Database connection error: {e}")
        yield None
    finally:
        if conn:
            conn.close()

def execute_query(query, params=(), fetch_one=False, fetch_all=False, commit=False):
    """Helper function to execute SQL queries"""
    with get_db_connection() as conn:
        if not conn:
            return None if not fetch_all else []
        
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            if commit:
                conn.commit()
                return cursor.lastrowid
            
            if fetch_one:
                result = cursor.fetchone()
                return dict(result) if result else None
            
            if fetch_all:
                results = cursor.fetchall()
                return [dict(row) for row in results]
            
            return cursor
        except sqlite3.Error as e:
            print(f"❌ Query error: {e}")
            traceback.print_exc()
            return None if not fetch_all else []

# Initialize database with proper error handling
def init_database():
    try:
        # Connect to SQLite (creates file if not exists)
        with get_db_connection() as conn:
            if not conn:
                print("❌ Failed to connect to SQLite")
                return
            
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uid TEXT UNIQUE,
                    email TEXT UNIQUE,
                    name TEXT,
                    photo_url TEXT,
                    provider TEXT,
                    role TEXT DEFAULT 'user',
                    is_active INTEGER DEFAULT 1,
                    last_login TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create datasets table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    file_path TEXT,
                    row_count INTEGER DEFAULT 0,
                    column_count INTEGER DEFAULT 0,
                    is_primary INTEGER DEFAULT 0,
                    is_default INTEGER DEFAULT 0,
                    uploaded_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    UNIQUE(user_id, name)
                )
            ''')
            
            # Create predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    dataset_name TEXT,
                    predicted_metric TEXT,
                    year INTEGER,
                    predicted_value REAL,
                    accuracy REAL,
                    prediction_type TEXT DEFAULT 'standard',
                    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_used TEXT DEFAULT 'linear_regression',
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            ''')
            
            # Create dataset_columns table
            cursor.execute('''
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
                )
            ''')
            
            # Create user_activity table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    activity_type TEXT,
                    description TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_datasets_user ON datasets(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_datasets_primary ON datasets(is_primary)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_activity_user ON user_activity(user_id)')
            
            # Create default admin user if not exists
            cursor.execute('''
                INSERT OR IGNORE INTO users (uid, email, name, role, last_login) 
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', ('admin_local', 'admin@forestpredict.com', 'Admin User', 'admin'))
            
            conn.commit()
            
            # Get admin ID
            admin_id = get_admin_id()
            
            # Load default forest_data.csv for admin if it exists
            if admin_id and os.path.exists('forest_data.csv'):
                load_forest_data_for_admin(admin_id)
            
            print("✅ Database initialized successfully")
            
    except sqlite3.Error as e:
        print(f"❌ Error initializing database: {e}")
        traceback.print_exc()

def get_admin_id():
    """Get admin user ID"""
    result = execute_query("SELECT id FROM users WHERE email = 'admin@forestpredict.com'", fetch_one=True)
    return result['id'] if result else None

def load_forest_data_for_admin(admin_id):
    """Load the forest_data.csv for admin user"""
    try:
        df = pd.read_csv('forest_data.csv')
        print(f"📊 Loading forest_data.csv with {len(df)} rows and {len(df.columns)} columns for admin")
        
        # Insert dataset for admin
        execute_query(
            """INSERT OR IGNORE INTO datasets (user_id, name, row_count, column_count, is_primary, is_default, description) 
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (admin_id, 'forest_data', len(df), len(df.columns), 1, 1, 'Forest dataset with environmental metrics'),
            commit=True
        )
        
        # Store column metadata for admin
        for column in df.columns:
            is_numeric = 1 if pd.api.types.is_numeric_dtype(df[column]) else 0
            min_val = float(df[column].min()) if is_numeric and len(df[column].dropna()) > 0 else None
            max_val = float(df[column].max()) if is_numeric and len(df[column].dropna()) > 0 else None
            
            execute_query(
                """INSERT INTO dataset_columns (user_id, dataset_name, column_name, data_type, is_numeric, min_value, max_value) 
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (admin_id, 'forest_data', column, str(df[column].dtype), is_numeric, min_val, max_val),
                commit=True
            )
        
        print("✅ forest_data.csv loaded successfully for admin")
            
    except Exception as e:
        print(f"❌ Error loading forest_data.csv: {e}")
        traceback.print_exc()

# Initialize database
init_database()

def sanitize_filename(filename):
    """Remove invalid characters from filename"""
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    if len(filename) > 200:
        name, ext = os.path.splitext(filename)
        filename = name[:195] + ext
    return filename

# Authentication decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'admin':
            return jsonify({'success': False, 'message': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function

def get_current_user_id():
    """Get current logged-in user ID from session"""
    return session.get('user_id')

def log_user_activity(user_id, activity_type, description, ip_address=None, user_agent=None):
    """Log user activity to database"""
    try:
        execute_query('''
            INSERT INTO user_activity (user_id, activity_type, description, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, activity_type, description, ip_address, user_agent), commit=True)
    except Exception as e:
        print(f"Error logging activity: {e}")

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/user-auth')
def user_auth():
    return render_template('auth/user_auth.html')

@app.route('/admin-auth')
def admin_auth():
    return render_template('auth/admin_auth.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('index.html')

@app.route('/admin-dashboard')
@admin_required
def admin_dashboard():
    return render_template('auth/admin_dashboard.html')

# Authentication API Routes
@app.route('/api/auth/firebase-login', methods=['POST'])
def firebase_login():
    """Handle Firebase authentication"""
    try:
        data = request.get_json()
        uid = data.get('uid')
        email = data.get('email')
        name = data.get('name')
        photo_url = data.get('photoUrl')
        provider = data.get('provider', 'google')
        
        if not uid or not email:
            return jsonify({'success': False, 'message': 'Missing user information'})
        
        # Check if user exists
        user = execute_query("SELECT * FROM users WHERE uid = ? OR email = ?", (uid, email), fetch_one=True)
        
        if user:
            # Update existing user
            execute_query('''
                UPDATE users 
                SET name = ?, photo_url = ?, provider = ?, last_login = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (name, photo_url, provider, user['id']), commit=True)
            user_id = user['id']
            role = user['role']
            
            # Check if user has any datasets
            dataset_count = execute_query(
                "SELECT COUNT(*) as count FROM datasets WHERE user_id = ?", 
                (user_id,), 
                fetch_one=True
            )['count']
            
            # If user has no datasets, create forest_data for them
            if dataset_count == 0 and os.path.exists('forest_data.csv'):
                create_forest_data_for_user(user_id)
        else:
            # Create new user
            user_id = execute_query('''
                INSERT INTO users (uid, email, name, photo_url, provider, role, last_login)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (uid, email, name, photo_url, provider, 'user'), commit=True)
            role = 'user'
            
            # Create forest_data dataset for new user if file exists
            if os.path.exists('forest_data.csv'):
                create_forest_data_for_user(user_id)
        
        # Log activity
        log_user_activity(
            user_id, 
            'login', 
            f'User logged in via {provider}',
            request.remote_addr,
            request.user_agent.string
        )
        
        # Set session
        session['user_id'] = user_id
        session['email'] = email
        session['name'] = name
        session['role'] = role
        session['photo_url'] = photo_url
        
        return jsonify({
            'success': True,
            'user': {
                'id': user_id,
                'email': email,
                'name': name,
                'role': role,
                'photo_url': photo_url
            }
        })
        
    except Exception as e:
        print(f"❌ Error in firebase login: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

def create_forest_data_for_user(user_id):
    """Create forest_data dataset for a specific user from forest_data.csv"""
    try:
        df = pd.read_csv('forest_data.csv')
        print(f"📊 Creating forest_data for user {user_id} with {len(df)} rows")
        
        # Insert dataset for user
        execute_query(
            """INSERT OR IGNORE INTO datasets (user_id, name, row_count, column_count, is_primary, is_default, description) 
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, 'forest_data', len(df), len(df.columns), 1, 1, 'Forest dataset with environmental metrics'),
            commit=True
        )
        
        # Store column metadata for user
        for column in df.columns:
            is_numeric = 1 if pd.api.types.is_numeric_dtype(df[column]) else 0
            min_val = float(df[column].min()) if is_numeric and len(df[column].dropna()) > 0 else None
            max_val = float(df[column].max()) if is_numeric and len(df[column].dropna()) > 0 else None
            
            execute_query(
                """INSERT INTO dataset_columns (user_id, dataset_name, column_name, data_type, is_numeric, min_value, max_value) 
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_id, 'forest_data', column, str(df[column].dtype), is_numeric, min_val, max_val),
                commit=True
            )
        
        print(f"✅ forest_data.csv created successfully for user {user_id}")
        
    except Exception as e:
        print(f"❌ Error creating forest_data for user {user_id}: {e}")
        traceback.print_exc()

@app.route('/api/auth/admin-login', methods=['POST'])
def admin_login():
    """Handle admin login"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        # Hardcoded admin credentials
        if username == 'admin' and password == 'Harshdeep*123':
            
            # Get or create admin user
            admin_user = execute_query(
                "SELECT * FROM users WHERE email = 'admin@forestpredict.com'", 
                fetch_one=True
            )
            
            if admin_user:
                # Update last login
                execute_query(
                    "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", 
                    (admin_user['id'],), 
                    commit=True
                )
                user_id = admin_user['id']
            else:
                # Create admin user
                user_id = execute_query('''
                    INSERT INTO users (uid, email, name, role, last_login)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', ('admin_local', 'admin@forestpredict.com', 'Admin User', 'admin'), commit=True)
            
            # Log activity
            log_user_activity(
                user_id,
                'admin_login',
                'Admin logged in',
                request.remote_addr,
                request.user_agent.string
            )
            
            # Set session
            session['user_id'] = user_id
            session['email'] = 'admin@forestpredict.com'
            session['name'] = 'Admin User'
            session['role'] = 'admin'
            
            return jsonify({
                'success': True,
                'user': {
                    'id': user_id,
                    'email': 'admin@forestpredict.com',
                    'name': 'Admin User',
                    'role': 'admin'
                }
            })
        else:
            return jsonify({'success': False, 'message': 'Invalid username or password'})
        
    except Exception as e:
        print(f"❌ Error in admin login: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Handle logout"""
    try:
        if 'user_id' in session:
            log_user_activity(
                session['user_id'],
                'logout',
                'User logged out',
                request.remote_addr,
                request.user_agent.string
            )
        
        session.clear()
        return jsonify({'success': True, 'message': 'Logged out successfully'})
    except Exception as e:
        print(f"❌ Error in logout: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/auth/check-session', methods=['GET'])
def check_session():
    """Check if user is logged in"""
    if 'user_id' in session:
        return jsonify({
            'success': True,
            'authenticated': True,
            'user': {
                'id': session.get('user_id'),
                'email': session.get('email'),
                'name': session.get('name'),
                'role': session.get('role'),
                'photo_url': session.get('photo_url')
            }
        })
    else:
        return jsonify({'success': True, 'authenticated': False})

# User-specific Stats - Each user sees their own stats only
@app.route('/api/stats', methods=['GET'])
@login_required
def get_stats():
    """Get user-specific system statistics - each user sees their own data only"""
    try:
        user_id = get_current_user_id()
        
        # Get total records for THIS USER ONLY
        total_records_result = execute_query(
            "SELECT COALESCE(SUM(row_count), 0) as total_records FROM datasets WHERE user_id = ?", 
            (user_id,), 
            fetch_one=True
        )
        total_records = total_records_result['total_records'] if total_records_result else 0
        
        # Get number of datasets for THIS USER ONLY
        total_datasets_result = execute_query(
            "SELECT COUNT(*) as total_datasets FROM datasets WHERE user_id = ?", 
            (user_id,), 
            fetch_one=True
        )
        total_datasets = total_datasets_result['total_datasets'] if total_datasets_result else 0
        
        # Get number of predictions for THIS USER ONLY
        total_predictions_result = execute_query(
            "SELECT COUNT(*) as total_predictions FROM predictions WHERE user_id = ?", 
            (user_id,), 
            fetch_one=True
        )
        total_predictions = total_predictions_result['total_predictions'] if total_predictions_result else 0
        
        # Check if user has uploaded custom data (excluding default forest_data)
        user_datasets_result = execute_query(
            "SELECT COUNT(*) as user_datasets FROM datasets WHERE user_id = ? AND is_default = 0", 
            (user_id,), 
            fetch_one=True
        )
        user_datasets = user_datasets_result['user_datasets'] if user_datasets_result else 0
        has_user_data = user_datasets > 0
        
        # Get year range from user's primary dataset
        start_year = 2000
        end_year = 2023
        
        primary_dataset = execute_query(
            "SELECT name FROM datasets WHERE user_id = ? AND is_primary = 1", 
            (user_id,), 
            fetch_one=True
        )
        
        if primary_dataset:
            dataset_name = primary_dataset['name']
            year_data = execute_query("""
                SELECT MIN(min_value) as min_year, MAX(max_value) as max_year 
                FROM dataset_columns 
                WHERE user_id = ? AND dataset_name = ? AND column_name = 'Year'
            """, (user_id, dataset_name), fetch_one=True)
            
            if year_data and year_data['min_year']:
                start_year = int(year_data['min_year'])
                end_year = int(year_data['max_year'])
        
        # Get last update time for THIS USER
        last_updated_result = execute_query(
            "SELECT MAX(uploaded_date) as last_updated FROM datasets WHERE user_id = ?", 
            (user_id,), 
            fetch_one=True
        )
        last_updated = last_updated_result['last_updated'] if last_updated_result else None
        
        stats = {
            'success': True,
            'total_records': int(total_records),
            'total_datasets': total_datasets,
            'total_predictions': total_predictions,
            'start_year': start_year,
            'end_year': end_year,
            'has_user_data': has_user_data,
            'last_updated': last_updated
        }
        
        return jsonify(stats)
        
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'total_records': 0,
            'total_datasets': 0,
            'total_predictions': 0,
            'start_year': 2000,
            'end_year': 2023,
            'has_user_data': False
        })

@app.route('/api/load-default-data', methods=['POST'])
@login_required
def load_default_data():
    """Load forest_data.csv for current user"""
    try:
        user_id = get_current_user_id()
        if not os.path.exists('forest_data.csv'):
            return jsonify({'success': False, 'message': 'forest_data.csv not found'})
        
        df = pd.read_csv('forest_data.csv')
        
        # Check if user already has forest_data
        existing = execute_query(
            "SELECT COUNT(*) as count FROM datasets WHERE user_id = ? AND name = 'forest_data'", 
            (user_id,), 
            fetch_one=True
        )
        
        if existing and existing['count'] > 0:
            # Update existing forest_data
            execute_query(
                "UPDATE datasets SET row_count = ?, column_count = ? WHERE user_id = ? AND name = 'forest_data'", 
                (len(df), len(df.columns), user_id), 
                commit=True
            )
            
            # Delete old columns
            execute_query(
                "DELETE FROM dataset_columns WHERE user_id = ? AND dataset_name = 'forest_data'", 
                (user_id,), 
                commit=True
            )
        else:
            # Insert new forest_data
            execute_query(
                """INSERT INTO datasets (user_id, name, row_count, column_count, is_primary, is_default, description) 
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_id, 'forest_data', len(df), len(df.columns), 1, 1, 'Forest dataset with environmental metrics'),
                commit=True
            )
        
        # Insert column metadata
        for column in df.columns:
            is_numeric = 1 if pd.api.types.is_numeric_dtype(df[column]) else 0
            min_val = float(df[column].min()) if is_numeric and len(df[column].dropna()) > 0 else None
            max_val = float(df[column].max()) if is_numeric and len(df[column].dropna()) > 0 else None
            
            execute_query(
                """INSERT INTO dataset_columns (user_id, dataset_name, column_name, data_type, is_numeric, min_value, max_value) 
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_id, 'forest_data', column, str(df[column].dtype), is_numeric, min_val, max_val),
                commit=True
            )
        
        return jsonify({
            'success': True,
            'message': f'✅ Loaded {len(df)} records from forest_data.csv',
            'records': len(df),
            'columns': len(df.columns)
        })
        
    except Exception as e:
        print(f"❌ Error loading default data: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/datasets', methods=['GET'])
@login_required
def get_datasets():
    """Get list of all datasets for current user"""
    try:
        user_id = get_current_user_id()
        
        datasets = execute_query("""
            SELECT 
                d.name,
                d.row_count,
                d.column_count,
                d.is_primary,
                d.is_default,
                d.uploaded_date,
                d.description,
                COUNT(DISTINCT dc.column_name) as actual_columns
            FROM datasets d
            LEFT JOIN dataset_columns dc ON d.name = dc.dataset_name AND dc.user_id = d.user_id
            WHERE d.user_id = ?
            GROUP BY d.id, d.name, d.row_count, d.column_count, d.is_primary, d.is_default, d.uploaded_date, d.description
            ORDER BY d.is_primary DESC, d.uploaded_date DESC
        """, (user_id,), fetch_all=True)
        
        if not datasets:
            datasets = []
        
        return jsonify(datasets)
        
    except Exception as e:
        print(f"❌ Error getting datasets: {e}")
        return jsonify([])

@app.route('/api/dataset/<dataset_name>', methods=['GET'])
@login_required
def get_dataset_data(dataset_name):
    """Get data from specific dataset for current user or specified user (admin)"""
    try:
        # Check if user_id is provided in query params (for admin view)
        target_user_id = request.args.get('user_id')
        current_user_id = get_current_user_id()
        
        # If user_id is provided and current user is admin, use that user_id
        if target_user_id and session.get('role') == 'admin':
            user_id = int(target_user_id)
            print(f"Admin viewing dataset {dataset_name} for user {user_id}")
        else:
            user_id = current_user_id
        
        if not dataset_name:
            return jsonify({'success': False, 'message': 'Dataset name required'})
        
        # Check if user has access to this dataset
        result = execute_query(
            "SELECT file_path, is_default FROM datasets WHERE user_id = ? AND name = ?", 
            (user_id, dataset_name), 
            fetch_one=True
        )
        
        if not result:
            return jsonify({'success': False, 'message': f'Dataset "{dataset_name}" not found or access denied'})
        
        file_path = result['file_path']
        is_default = result['is_default']
        
        # For default dataset or if file_path is not set, read from system forest_data.csv
        if is_default or not file_path or not os.path.exists(file_path):
            if os.path.exists('forest_data.csv'):
                df = pd.read_csv('forest_data.csv')
                data = df.replace({np.nan: None}).to_dict('records')
                return jsonify({'success': True, 'data': data})
            else:
                return jsonify({'success': False, 'message': 'forest_data.csv file not found'})
        
        if file_path and os.path.exists(file_path):
            try:
                file_path = os.path.normpath(file_path)
                
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, encoding='utf-8')
                elif file_path.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                else:
                    return jsonify({'success': False, 'message': 'Unsupported file format'})
                
                data = df.replace({np.nan: None}).to_dict('records')
                return jsonify({'success': True, 'data': data})
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='latin1')
                    data = df.replace({np.nan: None}).to_dict('records')
                    return jsonify({'success': True, 'data': data})
                except Exception as e:
                    return jsonify({'success': False, 'message': f'Error reading file with different encoding: {str(e)}'})
            except Exception as e:
                print(f"❌ Error reading file: {e}")
                return jsonify({'success': False, 'message': f'Error reading file: {str(e)}'})
        else:
            return jsonify({'success': False, 'message': 'Dataset file not found'})
        
    except Exception as e:
        print(f"❌ Error getting dataset data: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/dataset/<dataset_name>/columns', methods=['GET'])
@login_required
def get_dataset_columns(dataset_name):
    """Get columns for a specific dataset for current user"""
    try:
        user_id = get_current_user_id()
        if not dataset_name:
            return jsonify({'success': False, 'columns': []})
        
        # First check if dataset exists for this user
        count_result = execute_query(
            "SELECT COUNT(*) as count FROM datasets WHERE user_id = ? AND name = ?", 
            (user_id, dataset_name), 
            fetch_one=True
        )
        
        if not count_result or count_result['count'] == 0:
            return jsonify({'success': False, 'columns': []})
        
        # Get columns from metadata for this user
        columns_result = execute_query(
            "SELECT column_name FROM dataset_columns WHERE user_id = ? AND dataset_name = ?", 
            (user_id, dataset_name), 
            fetch_all=True
        )
        
        columns = [row['column_name'] for row in columns_result] if columns_result else []
        
        if columns:
            return jsonify({'success': True, 'columns': columns})
        
        # Fallback: try to read dataset directly
        dataset_response = get_dataset_data(dataset_name)
        if dataset_response.json and dataset_response.json.get('success'):
            data = dataset_response.json.get('data', [])
            if data and len(data) > 0:
                columns = list(data[0].keys())
                return jsonify({'success': True, 'columns': columns})
        
        return jsonify({'success': False, 'columns': []})
        
    except Exception as e:
        print(f"❌ Error getting columns: {e}")
        return jsonify({'success': False, 'columns': []})

@app.route('/api/upload', methods=['POST'])
@login_required
def upload_file():
    """Handle file upload for current user - visible to admin"""
    try:
        user_id = get_current_user_id()
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            return jsonify({'success': False, 'message': 'Only CSV and Excel files are supported'})
        
        # Read file
        try:
            if file.filename.endswith('.csv'):
                try:
                    df = pd.read_csv(file)
                except UnicodeDecodeError:
                    file.seek(0)
                    df = pd.read_csv(file, encoding='latin1')
            else:
                df = pd.read_excel(file)
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error reading file: {str(e)}'})
        
        if len(df) == 0:
            return jsonify({'success': False, 'message': 'File is empty'})
        
        # Generate unique dataset name for this user
        base_name = sanitize_filename(os.path.splitext(file.filename)[0])
        dataset_name = base_name
        
        # Check if dataset name exists for this user
        existing = execute_query(
            "SELECT COUNT(*) as count FROM datasets WHERE user_id = ? AND name = ?", 
            (user_id, dataset_name), 
            fetch_one=True
        )
        
        if existing and existing['count'] > 0:
            # Append timestamp to make unique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = f"{base_name}_{timestamp}"
        
        # Save file
        upload_folder = f'uploads/user_{user_id}'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        safe_filename = sanitize_filename(f"{dataset_name}{os.path.splitext(file.filename)[1]}")
        file_path = os.path.normpath(os.path.join(upload_folder, safe_filename))
        
        file.seek(0)
        file.save(file_path)
        
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'message': 'Failed to save file'})
        
        # Insert into database for this user
        execute_query(
            """INSERT INTO datasets (user_id, name, file_path, row_count, column_count, description) 
            VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, dataset_name, file_path, len(df), len(df.columns), f"Uploaded: {file.filename}"),
            commit=True
        )
        
        # Store column metadata for this user
        for column in df.columns:
            is_numeric = 1 if pd.api.types.is_numeric_dtype(df[column]) else 0
            min_val = float(df[column].min()) if is_numeric and len(df[column].dropna()) > 0 else None
            max_val = float(df[column].max()) if is_numeric and len(df[column].dropna()) > 0 else None
            
            execute_query(
                """INSERT INTO dataset_columns (user_id, dataset_name, column_name, data_type, is_numeric, min_value, max_value) 
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_id, dataset_name, column, str(df[column].dtype), is_numeric, min_val, max_val),
                commit=True
            )
        
        # Log the upload activity
        log_user_activity(
            user_id,
            'dataset_upload',
            f'Uploaded dataset: {dataset_name} with {len(df)} rows',
            request.remote_addr,
            request.user_agent.string
        )
        
        print(f"✅ User {user_id} uploaded dataset: {dataset_name} with {len(df)} rows")
        
        return jsonify({
            'success': True,
            'message': f'✅ File uploaded successfully as "{dataset_name}"',
            'dataset_name': dataset_name,
            'rows': len(df),
            'columns': len(df.columns)
        })
        
    except Exception as e:
        print(f"❌ Error uploading file: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})
        
@app.route('/api/manage-datasets', methods=['POST'])
@login_required
def manage_datasets():
    """Manage datasets for current user (set primary, delete, etc.)"""
    try:
        user_id = get_current_user_id()
        data = request.get_json()
        action = data.get('action')
        dataset_name = data.get('dataset_name')
        
        if not action or not dataset_name:
            return jsonify({'success': False, 'message': 'Missing parameters'})
        
        if action == 'set_primary':
            # Reset all primary flags for this user
            execute_query("UPDATE datasets SET is_primary = 0 WHERE user_id = ?", (user_id,), commit=True)
            
            # Set the selected dataset as primary for this user
            execute_query(
                "UPDATE datasets SET is_primary = 1 WHERE user_id = ? AND name = ?", 
                (user_id, dataset_name), 
                commit=True
            )
            
            return jsonify({'success': True, 'message': f'✅ Dataset "{dataset_name}" set as primary'})
        
        elif action == 'delete':
            # Don't allow deletion of default forest_data
            if dataset_name == 'forest_data':
                return jsonify({'success': False, 'message': 'Cannot delete default forest_data dataset'})
            
            # Get file path to delete physical file
            result = execute_query(
                "SELECT file_path FROM datasets WHERE user_id = ? AND name = ?", 
                (user_id, dataset_name), 
                fetch_one=True
            )
            
            # Delete from database
            execute_query(
                "DELETE FROM datasets WHERE user_id = ? AND name = ?", 
                (user_id, dataset_name), 
                commit=True
            )
            
            # Delete physical file
            if result and result['file_path'] and os.path.exists(result['file_path']):
                try:
                    os.remove(result['file_path'])
                except Exception as e:
                    print(f"Warning: Could not delete file: {e}")
            
            return jsonify({'success': True, 'message': f'✅ Dataset "{dataset_name}" deleted successfully'})
        
        else:
            return jsonify({'success': False, 'message': f'Unknown action: {action}'})
        
    except Exception as e:
        print(f"❌ Error managing dataset: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/metrics', methods=['GET'])
@login_required
def get_metrics():
    """Get available metrics for prediction for current user"""
    try:
        user_id = get_current_user_id()
        
        # Get primary dataset name for this user
        primary_result = execute_query(
            "SELECT name FROM datasets WHERE user_id = ? AND is_primary = 1", 
            (user_id,), 
            fetch_one=True
        )
        
        if not primary_result:
            # If no primary, get forest_data for this user
            forest_result = execute_query(
                "SELECT name FROM datasets WHERE user_id = ? AND name = 'forest_data' LIMIT 1", 
                (user_id,), 
                fetch_one=True
            )
            dataset_name = forest_result['name'] if forest_result else None
        else:
            dataset_name = primary_result['name']
        
        if not dataset_name:
            return jsonify([])
        
        # Get numeric columns for this user's dataset
        columns_result = execute_query("""
            SELECT column_name 
            FROM dataset_columns 
            WHERE user_id = ? AND dataset_name = ? AND is_numeric = 1
            ORDER BY column_name
        """, (user_id, dataset_name), fetch_all=True)
        
        numeric_columns = [row['column_name'] for row in columns_result] if columns_result else []
        
        # Create metric list with labels
        metrics = []
        for column in numeric_columns:
            if column.lower() in ['year', 'id', 'index']:
                continue
            
            metric_label = column.replace('_', ' ').title()
            if 'CO2' in metric_label:
                metric_label = metric_label.replace('Co2', 'CO2')
            
            metrics.append({
                'value': column,
                'label': metric_label
            })
        
        return jsonify(metrics)
        
    except Exception as e:
        print(f"❌ Error getting metrics: {e}")
        traceback.print_exc()
        return jsonify([])

@app.route('/api/predict', methods=['POST'])
@login_required
def make_prediction():
    """Make a prediction for current user"""
    try:
        user_id = get_current_user_id()
        data = request.get_json()
        dataset_name = data.get('dataset')
        metric = data.get('metric')
        target_year = int(data.get('target_year'))
        prediction_type = data.get('prediction_type', 'standard')
        
        print(f"📊 Prediction request: user={user_id}, dataset={dataset_name}, metric={metric}, year={target_year}, type={prediction_type}")
        
        if not metric or not target_year:
            return jsonify({'success': False, 'message': 'Missing required parameters'})
        
        # Use primary dataset for this user if none specified
        if not dataset_name:
            primary_result = execute_query(
                "SELECT name FROM datasets WHERE user_id = ? AND is_primary = 1", 
                (user_id,), 
                fetch_one=True
            )
            dataset_name = primary_result['name'] if primary_result else 'forest_data'
        
        # Get dataset data
        dataset_response = get_dataset_data(dataset_name)
        if not dataset_response.json or not dataset_response.json.get('success'):
            return jsonify({'success': False, 'message': 'Failed to load dataset'})
        
        df = pd.DataFrame(dataset_response.json.get('data', []))
        
        if df.empty:
            return jsonify({'success': False, 'message': 'Dataset is empty'})
        
        # Check if metric exists in dataset
        if metric not in df.columns:
            return jsonify({'success': False, 'message': f'Metric "{metric}" not found in dataset'})
        
        # Check for Year column
        if 'Year' not in df.columns:
            year_cols = [col for col in df.columns if 'year' in col.lower()]
            if year_cols:
                df['Year'] = df[year_cols[0]]
            else:
                return jsonify({'success': False, 'message': 'Dataset must contain "Year" column for prediction'})
        
        # Prepare data
        df_clean = df[['Year', metric]].dropna()
        
        if len(df_clean) < 3:
            return jsonify({'success': False, 'message': 'Insufficient data for prediction'})
        
        # Train model
        X = df_clean['Year'].values.reshape(-1, 1)
        y = df_clean[metric].values
        
        model = LinearRegression()
        
        # Calculate accuracy
        if len(df_clean) >= 4:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test) if len(X_test) > 0 else train_score
            accuracy = (train_score + test_score) / 2 if len(X_test) > 0 else train_score
        else:
            model.fit(X, y)
            accuracy = model.score(X, y)
        
        # Make prediction
        prediction = model.predict([[target_year]])[0]
        
        # Save to database for this user
        execute_query(
            """INSERT INTO predictions (user_id, dataset_name, predicted_metric, year, predicted_value, accuracy, prediction_type) 
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, dataset_name, metric, target_year, float(prediction), float(accuracy), prediction_type),
            commit=True
        )
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'accuracy': float(accuracy),
            'data_points': len(df_clean),
            'metric': metric,
            'year': target_year,
            'dataset': dataset_name
        })
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/predictions', methods=['GET'])
@login_required
def get_predictions():
    """Get all predictions for current user"""
    try:
        user_id = get_current_user_id()
        
        predictions = execute_query(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY prediction_date DESC", 
            (user_id,), 
            fetch_all=True
        )
        
        return jsonify(predictions if predictions else [])
        
    except Exception as e:
        print(f"❌ Error getting predictions: {e}")
        return jsonify([])

@app.route('/api/prediction/<int:prediction_id>', methods=['DELETE'])
@login_required
def delete_prediction(prediction_id):
    """Delete a specific prediction for current user"""
    try:
        user_id = get_current_user_id()
        
        execute_query(
            "DELETE FROM predictions WHERE id = ? AND user_id = ?", 
            (prediction_id, user_id), 
            commit=True
        )
        
        return jsonify({'success': True, 'message': 'Prediction deleted successfully'})
        
    except Exception as e:
        print(f"❌ Error deleting prediction: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/clear-predictions', methods=['POST'])
@login_required
def clear_predictions():
    """Clear all predictions for current user"""
    try:
        user_id = get_current_user_id()
        
        execute_query("DELETE FROM predictions WHERE user_id = ?", (user_id,), commit=True)
        
        return jsonify({'success': True, 'message': 'All predictions cleared successfully'})
        
    except Exception as e:
        print(f"❌ Error clearing predictions: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/chart-data', methods=['GET'])
@login_required
def get_chart_data():
    """Get data for charts for current user"""
    user_id = get_current_user_id()
    chart_type = request.args.get('type', 'temperature_trend')
    
    try:
        # Get primary dataset for this user
        primary_result = execute_query(
            "SELECT name FROM datasets WHERE user_id = ? AND is_primary = 1", 
            (user_id,), 
            fetch_one=True
        )
        dataset_name = primary_result['name'] if primary_result else 'forest_data'
        
        # Get dataset data
        dataset_response = get_dataset_data(dataset_name)
        if not dataset_response.json or not dataset_response.json.get('success'):
            return jsonify({'success': False, 'data': []})
        
        df = pd.DataFrame(dataset_response.json.get('data', []))
        
        if df.empty:
            return jsonify({'success': True, 'data': []})
        
        # Prepare chart data based on chart type
        if chart_type == 'temperature_trend' and 'Year' in df.columns and 'Avg Temperature (°C)' in df.columns:
            temp_data = df[['Year', 'Avg Temperature (°C)']].dropna()
            if len(temp_data) > 0:
                temp_by_year = temp_data.groupby('Year')['Avg Temperature (°C)'].mean().reset_index()
                data = [{'year': int(row['Year']), 'value': float(row['Avg Temperature (°C)'])} 
                       for _, row in temp_by_year.iterrows()]
                return jsonify({'success': True, 'data': data})
        
        elif chart_type == 'co2_vs_forest' and 'CO2 Emissions (Tons/Capita)' in df.columns and 'Forest Area (%)' in df.columns:
            co2_forest_data = df[['CO2 Emissions (Tons/Capita)', 'Forest Area (%)']].dropna()
            if len(co2_forest_data) > 0:
                data = [{'co2': float(row['CO2 Emissions (Tons/Capita)']), 
                        'forest': float(row['Forest Area (%)'])} 
                       for _, row in co2_forest_data.iterrows()]
                return jsonify({'success': True, 'data': data})
        
        elif chart_type == 'population_growth' and 'Year' in df.columns and 'Population' in df.columns:
            pop_data = df[['Year', 'Population']].dropna()
            if len(pop_data) > 0:
                pop_by_year = pop_data.groupby('Year')['Population'].sum().reset_index()
                data = [{'year': int(row['Year']), 'value': float(row['Population'])} 
                       for _, row in pop_by_year.iterrows()]
                return jsonify({'success': True, 'data': data})
        
        elif chart_type == 'data_distribution' and 'Country' in df.columns:
            country_counts = df['Country'].value_counts().head(10).reset_index()
            country_counts.columns = ['Country', 'count']
            data = [{'label': row['Country'], 'value': int(row['count'])} 
                   for _, row in country_counts.iterrows()]
            return jsonify({'success': True, 'data': data})
        
        return jsonify({'success': True, 'data': []})
        
    except Exception as e:
        print(f"❌ Error getting chart data: {e}")
        return jsonify({'success': False, 'data': []})

@app.route('/api/visualization/<dataset_name>', methods=['GET'])
@login_required
def get_visualization_data(dataset_name):
    """Get data for custom visualization for current user"""
    user_id = get_current_user_id()
    x_axis = request.args.get('x')
    y_axis = request.args.get('y')
    
    if not x_axis or not y_axis:
        return jsonify({'success': False, 'message': 'Missing axis parameters'})
    
    try:
        # Check if user has access to this dataset
        count_result = execute_query(
            "SELECT COUNT(*) as count FROM datasets WHERE user_id = ? AND name = ?", 
            (user_id, dataset_name), 
            fetch_one=True
        )
        
        if not count_result or count_result['count'] == 0:
            return jsonify({'success': False, 'message': 'Dataset not found or access denied'})
        
        # Get dataset data
        dataset_response = get_dataset_data(dataset_name)
        if not dataset_response.json or not dataset_response.json.get('success'):
            return jsonify({'success': False, 'message': 'Failed to load dataset'})
        
        df = pd.DataFrame(dataset_response.json.get('data', []))
        
        if df.empty:
            return jsonify({'success': False, 'message': 'Dataset is empty'})
        
        # Check if columns exist
        if x_axis not in df.columns or y_axis not in df.columns:
            return jsonify({'success': False, 'message': 'Selected columns not found in dataset'})
        
        # Prepare data
        data = []
        for _, row in df.dropna(subset=[x_axis, y_axis]).head(100).iterrows():
            try:
                x_val = row[x_axis]
                y_val = row[y_axis]
                
                if pd.isna(x_val) or pd.isna(y_val):
                    continue
                
                try:
                    x_val = float(x_val)
                except (ValueError, TypeError):
                    x_val = str(x_val)
                
                try:
                    y_val = float(y_val)
                except (ValueError, TypeError):
                    y_val = str(y_val)
                
                data.append({
                    x_axis: x_val,
                    y_axis: y_val
                })
            except Exception as e:
                print(f"Error processing row: {e}")
                continue
        
        return jsonify({'success': True, 'data': data})
        
    except Exception as e:
        print(f"❌ Error getting visualization data: {e}")
        return jsonify({'success': False, 'message': str(e)})

# Admin routes
@app.route('/api/admin/users', methods=['GET'])
@admin_required
def get_users():
    """Get all users for admin"""
    try:
        users = execute_query('''
            SELECT 
                u.*,
                COUNT(DISTINCT a.id) as activity_count,
                MAX(a.created_at) as last_activity
            FROM users u
            LEFT JOIN user_activity a ON u.id = a.user_id
            GROUP BY u.id
            ORDER BY u.created_at DESC
        ''', fetch_all=True)
        
        return jsonify({'success': True, 'users': users if users else []})
        
    except Exception as e:
        print(f"❌ Error getting users: {e}")
        return jsonify({'success': False, 'users': []})

@app.route('/api/admin/user/<int:user_id>/activity', methods=['GET'])
@admin_required
def get_user_activity(user_id):
    """Get activity for a specific user"""
    try:
        activities = execute_query('''
            SELECT * FROM user_activity 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT 100
        ''', (user_id,), fetch_all=True)
        
        return jsonify({'success': True, 'activities': activities if activities else []})
        
    except Exception as e:
        print(f"❌ Error getting user activity: {e}")
        return jsonify({'success': False, 'activities': []})

@app.route('/api/admin/user/<int:user_id>/toggle-status', methods=['POST'])
@admin_required
def toggle_user_status(user_id):
    """Toggle user active status"""
    try:
        result = execute_query(
            "SELECT is_active FROM users WHERE id = ?", 
            (user_id,), 
            fetch_one=True
        )
        
        if not result:
            return jsonify({'success': False, 'message': 'User not found'})
        
        new_status = 0 if result['is_active'] == 1 else 1
        execute_query(
            "UPDATE users SET is_active = ? WHERE id = ?", 
            (new_status, user_id), 
            commit=True
        )
        
        return jsonify({
            'success': True, 
            'message': f'User {"activated" if new_status == 1 else "deactivated"} successfully',
            'is_active': new_status == 1
        })
        
    except Exception as e:
        print(f"❌ Error toggling user status: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/admin/user/<int:user_id>', methods=['DELETE'])
@admin_required
def delete_user(user_id):
    """Delete a user"""
    try:
        result = execute_query(
            "SELECT role FROM users WHERE id = ?", 
            (user_id,), 
            fetch_one=True
        )
        
        if not result:
            return jsonify({'success': False, 'message': 'User not found'})
        
        if result['role'] == 'admin':
            return jsonify({'success': False, 'message': 'Cannot delete admin user'})
        
        execute_query("DELETE FROM users WHERE id = ?", (user_id,), commit=True)
        
        return jsonify({'success': True, 'message': 'User deleted successfully'})
        
    except Exception as e:
        print(f"❌ Error deleting user: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/admin/dashboard-stats', methods=['GET'])
@admin_required
def get_admin_dashboard_stats():
    """Get admin dashboard statistics"""
    try:
        # Get user counts
        total_users_result = execute_query("SELECT COUNT(*) as total_users FROM users", fetch_one=True)
        total_users = total_users_result['total_users'] if total_users_result else 0
        
        active_users_result = execute_query("SELECT COUNT(*) as active_users FROM users WHERE is_active = 1", fetch_one=True)
        active_users = active_users_result['active_users'] if active_users_result else 0
        
        admin_users_result = execute_query("SELECT COUNT(*) as admin_users FROM users WHERE role = 'admin'", fetch_one=True)
        admin_users = admin_users_result['admin_users'] if admin_users_result else 0
        
        # Get activity stats for last 7 days
        activity_by_day = execute_query('''
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM user_activity
            WHERE created_at >= DATE('now', '-7 days')
            GROUP BY DATE(created_at)
            ORDER BY date
        ''', fetch_all=True) or []
        
        # Get activity types distribution
        activity_types = execute_query('''
            SELECT activity_type, COUNT(*) as count
            FROM user_activity
            GROUP BY activity_type
            ORDER BY count DESC
        ''', fetch_all=True) or []
        
        # Get prediction stats
        total_predictions_result = execute_query("SELECT COUNT(*) as total_predictions FROM predictions", fetch_one=True)
        total_predictions = total_predictions_result['total_predictions'] if total_predictions_result else 0
        
        # Get dataset stats
        total_datasets_result = execute_query("SELECT COUNT(*) as total_datasets FROM datasets", fetch_one=True)
        total_datasets = total_datasets_result['total_datasets'] if total_datasets_result else 0
        
        # Get quick predictions count
        quick_count_result = execute_query(
            "SELECT COUNT(*) as quick_count FROM predictions WHERE prediction_type = 'quick'", 
            fetch_one=True
        )
        quick_count = quick_count_result['quick_count'] if quick_count_result else 0
        
        # Get standard predictions count
        standard_count_result = execute_query(
            "SELECT COUNT(*) as standard_count FROM predictions WHERE prediction_type = 'standard'", 
            fetch_one=True
        )
        standard_count = standard_count_result['standard_count'] if standard_count_result else 0
        
        print(f"📊 Admin Stats - Users: {total_users}, Predictions: {total_predictions}, Datasets: {total_datasets}")
        
        stats = {
            'users': {
                'total': total_users,
                'active': active_users,
                'admins': admin_users
            },
            'activity': {
                'by_day': activity_by_day,
                'by_type': activity_types
            },
            'system': {
                'total_predictions': total_predictions,
                'total_datasets': total_datasets,
                'quick_predictions': quick_count,
                'standard_predictions': standard_count
            }
        }
        
        return jsonify({'success': True, 'stats': stats})
        
    except Exception as e:
        print(f"❌ Error getting admin stats: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'stats': {}})

@app.route('/api/admin/all-predictions', methods=['GET'])
@admin_required
def get_all_predictions():
    """Get all predictions from all users for admin"""
    try:
        predictions = execute_query('''
            SELECT 
                p.id,
                p.user_id,
                p.dataset_name,
                p.predicted_metric,
                p.year,
                p.predicted_value,
                p.accuracy,
                p.prediction_type,
                p.prediction_date,
                p.model_used,
                u.name as user_name,
                u.email as user_email,
                u.photo_url as user_photo
            FROM predictions p
            LEFT JOIN users u ON p.user_id = u.id
            ORDER BY p.prediction_date DESC
        ''', fetch_all=True) or []
        
        print(f"✅ Found {len(predictions)} predictions across all users")
        
        return jsonify({'success': True, 'predictions': predictions, 'count': len(predictions)})
        
    except Exception as e:
        print(f"❌ Error getting all predictions: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'predictions': [], 'message': str(e)})

@app.route('/api/admin/user-predictions/<int:user_id>', methods=['GET'])
@admin_required
def get_user_predictions(user_id):
    """Get predictions for a specific user"""
    try:
        predictions = execute_query('''
            SELECT 
                p.id,
                p.user_id,
                p.dataset_name,
                p.predicted_metric,
                p.year,
                p.predicted_value,
                p.accuracy,
                p.prediction_type,
                p.prediction_date,
                p.model_used,
                u.name as user_name,
                u.email as user_email
            FROM predictions p
            LEFT JOIN users u ON p.user_id = u.id
            WHERE p.user_id = ?
            ORDER BY p.prediction_date DESC
        ''', (user_id,), fetch_all=True) or []
        
        print(f"✅ Found {len(predictions)} predictions for user {user_id}")
        
        return jsonify({'success': True, 'predictions': predictions})
        
    except Exception as e:
        print(f"❌ Error getting user predictions: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'predictions': [], 'message': str(e)})

@app.route('/api/admin/all-datasets', methods=['GET'])
@admin_required
def get_all_datasets():
    """Get all datasets from all users for admin"""
    try:
        datasets = execute_query('''
            SELECT 
                d.id,
                d.user_id,
                d.name,
                d.file_path,
                d.row_count,
                d.column_count,
                d.is_primary,
                d.is_default,
                d.uploaded_date,
                d.description,
                u.name as user_name,
                u.email as user_email,
                u.photo_url as user_photo,
                (SELECT COUNT(*) FROM dataset_columns dc WHERE dc.user_id = d.user_id AND dc.dataset_name = d.name) as actual_columns
            FROM datasets d
            LEFT JOIN users u ON d.user_id = u.id
            ORDER BY d.uploaded_date DESC
        ''', fetch_all=True) or []
        
        print(f"✅ Found {len(datasets)} datasets across all users")
        
        return jsonify({'success': True, 'datasets': datasets, 'count': len(datasets)})
        
    except Exception as e:
        print(f"❌ Error getting all datasets: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'datasets': [], 'message': str(e)})

if __name__ == '__main__':
    # Create necessary directories
    for folder in ['uploads', 'templates', 'templates/auth']:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    print("🚀 Starting Forest Data Analysis & Prediction System...")
    print(f"📁 Forest data file: {'✅ Found' if os.path.exists('forest_data.csv') else '❌ Not found'}")
    print(f"📁 SQLite database: {DB_PATH}")
    print("🌐 Server running at: http://localhost:5000")
    app.run(debug=True, port=5000)
