import sys
import io
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    elif hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sqlite3
import os
import json
from datetime import datetime, timedelta
import traceback
import csv
import io
import re
import secrets
import hashlib
import math
from functools import wraps
try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)  # For session management
CORS(app)

# Database configuration - SQLite
DB_PATH = 'forest_prediction.db'


# ── SQLite / mysql.connector compatibility layer ─────────────────────────────
class DictRow(dict):
    """Dict that also supports integer-index access (like tuple rows)."""
    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


class _SQLiteCursor:
    """Cursor wrapper: converts %s→?, NOW()→CURRENT_TIMESTAMP, returns DictRows."""
    def __init__(self, cur):
        self._cur = cur

    def execute(self, sql, params=None):
        sql = sql.replace('%s', '?').replace('NOW()', 'CURRENT_TIMESTAMP')
        if params is not None:
            self._cur.execute(sql, params)
        else:
            self._cur.execute(sql)

    def _row(self, row):
        if row is None or not self._cur.description:
            return row
        return DictRow({self._cur.description[i][0]: row[i]
                        for i in range(len(self._cur.description))})

    def fetchone(self):
        return self._row(self._cur.fetchone())

    def fetchall(self):
        desc = self._cur.description
        if not desc:
            return self._cur.fetchall()
        rows = self._cur.fetchall()
        return [DictRow({desc[i][0]: row[i] for i in range(len(desc))}) for row in rows]

    @property
    def lastrowid(self):
        return self._cur.lastrowid

    @property
    def rowcount(self):
        return self._cur.rowcount

    def close(self):
        self._cur.close()


class _SQLiteConnection:
    """Connection wrapper compatible with mysql.connector API."""
    def __init__(self, conn):
        self._conn = conn

    def cursor(self, dictionary=False):
        return _SQLiteCursor(self._conn.cursor())

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()

# Initialize database with proper error handling
def init_database():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()

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
                global_library_enabled INTEGER DEFAULT 1,
                last_login DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

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
                uploaded_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                description TEXT,
                UNIQUE (user_id, name),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')

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
                prediction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_used TEXT DEFAULT 'linear_regression',
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')

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

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                activity_type TEXT,
                description TEXT,
                ip_address TEXT,
                user_agent TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')

        cursor.execute('''
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
            )
        ''')

        cursor.execute('''
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
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS favorites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                dataset_name TEXT NOT NULL,
                label TEXT DEFAULT 'favorite' CHECK(label IN ('favorite','primary','archived')),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (user_id, dataset_name),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                message TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')

        cursor.execute('''
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
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                type TEXT DEFAULT 'info',
                title TEXT,
                message TEXT,
                is_read INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')

        cursor.execute('''
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
            )
        ''')

        # Indexes
        for idx in [
            'CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id)',
            'CREATE INDEX IF NOT EXISTS idx_datasets_user ON datasets(user_id)',
            'CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date)',
            'CREATE INDEX IF NOT EXISTS idx_datasets_primary ON datasets(is_primary)',
            'CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name)',
            'CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)',
            'CREATE INDEX IF NOT EXISTS idx_activity_user ON user_activity(user_id)',
            'CREATE INDEX IF NOT EXISTS idx_activity_date ON user_activity(created_at)',
            'CREATE INDEX IF NOT EXISTS idx_favorites_user ON favorites(user_id)',
            'CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_history(user_id)',
            'CREATE INDEX IF NOT EXISTS idx_approvals_status ON dataset_approvals(status)',
            'CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_id)',
            'CREATE INDEX IF NOT EXISTS idx_support_status ON support_queries(status)',
        ]:
            cursor.execute(idx)

        # Default admin user
        cursor.execute('''
            INSERT OR IGNORE INTO users (uid, email, name, role, last_login)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', ('admin_local', 'admin@forestpredict.com', 'Admin User', 'admin'))

        conn.commit()
        print("✅ Database initialized successfully")

        # Get admin ID and load default data
        cursor.execute("SELECT id FROM users WHERE email = 'admin@forestpredict.com'")
        row = cursor.fetchone()
        admin_id = row[0] if row else None

        if admin_id and os.path.exists('forest_data.csv'):
            load_forest_data_for_admin(cursor, conn, admin_id)

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        traceback.print_exc()

def get_admin_id(cursor):
    """Get admin user ID"""
    try:
        cursor.execute("SELECT id FROM users WHERE email = 'admin@forestpredict.com'")
        result = cursor.fetchone()
        return result[0] if result else None
    except:
        return None

def load_forest_data_for_admin(cursor, connection, admin_id):
    """Load the forest_data.csv for admin user"""
    try:
        df = pd.read_csv('forest_data.csv')
        print(f"📊 Loading forest_data.csv with {len(df)} rows and {len(df.columns)} columns for admin")

        cursor.execute(
            """INSERT OR IGNORE INTO datasets (user_id, name, row_count, column_count, is_primary, is_default, description)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (admin_id, 'forest_data', len(df), len(df.columns), 1, 1, 'Forest dataset with environmental metrics')
        )

        for column in df.columns:
            is_numeric = int(pd.api.types.is_numeric_dtype(df[column]))
            min_val = float(df[column].min()) if is_numeric and len(df[column].dropna()) > 0 else None
            max_val = float(df[column].max()) if is_numeric and len(df[column].dropna()) > 0 else None

            cursor.execute(
                """INSERT INTO dataset_columns (user_id, dataset_name, column_name, data_type, is_numeric, min_value, max_value)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (admin_id, 'forest_data', column, str(df[column].dtype), is_numeric, min_val, max_val)
            )

        connection.commit()
        print("✅ forest_data.csv loaded successfully for admin")

    except Exception as e:
        print(f"❌ Error loading forest_data.csv: {e}")
        traceback.print_exc()

# Initialize database
init_database()

def get_db_connection():
    """Create SQLite database connection wrapped in mysql.connector-compatible API"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON")
        return _SQLiteConnection(conn)
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None

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
        connection = get_db_connection()
        if not connection:
            return
        
        cursor = connection.cursor()
        cursor.execute('''
            INSERT INTO user_activity (user_id, activity_type, description, ip_address, user_agent)
            VALUES (%s, %s, %s, %s, %s)
        ''', (user_id, activity_type, description, ip_address, user_agent))
        connection.commit()
        cursor.close()
        connection.close()
    except Exception as e:
        print(f"Error logging activity: {e}")

@app.route('/favicon.ico')
def favicon():
    return send_file(os.path.join(app.static_folder, 'forest-predict-icon.png'), mimetype='image/png')

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
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'Database connection failed'})
        
        cursor = connection.cursor(dictionary=True)
        
        # Check if user exists
        cursor.execute("SELECT * FROM users WHERE uid = %s OR email = %s", (uid, email))
        user = cursor.fetchone()
        
        if user:
            # Update existing user
            cursor.execute('''
                UPDATE users 
                SET name = %s, photo_url = %s, provider = %s, last_login = NOW()
                WHERE id = %s
            ''', (name, photo_url, provider, user['id']))
            user_id = user['id']
            role = user['role']
            
            # Check if user has any datasets
            cursor.execute("SELECT COUNT(*) as count FROM datasets WHERE user_id = %s", (user_id,))
            dataset_count = cursor.fetchone()['count']
            
            # If user has no datasets, create forest_data for them
            if dataset_count == 0 and os.path.exists('forest_data.csv'):
                create_forest_data_for_user(cursor, connection, user_id)
        else:
            # Create new user
            cursor.execute('''
                INSERT INTO users (uid, email, name, photo_url, provider, role, last_login)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ''', (uid, email, name, photo_url, provider, 'user'))
            user_id = cursor.lastrowid
            role = 'user'
            
            # Create forest_data dataset for new user if file exists
            if os.path.exists('forest_data.csv'):
                create_forest_data_for_user(cursor, connection, user_id)
        
        connection.commit()
        
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
        
        cursor.close()
        connection.close()
        
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

def create_forest_data_for_user(cursor, connection, user_id):
    """Create forest_data dataset for a specific user from forest_data.csv"""
    try:
        df = pd.read_csv('forest_data.csv')
        print(f"📊 Creating forest_data for user {user_id} with {len(df)} rows")
        
        # Insert dataset for user
        cursor.execute(
            """INSERT INTO datasets (user_id, name, row_count, column_count, is_primary, is_default, description) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (user_id, 'forest_data', len(df), len(df.columns), True, True, 'Forest dataset with environmental metrics')
        )
        
        # Store column metadata for user
        for column in df.columns:
            is_numeric = pd.api.types.is_numeric_dtype(df[column])
            min_val = float(df[column].min()) if is_numeric and len(df[column].dropna()) > 0 else None
            max_val = float(df[column].max()) if is_numeric and len(df[column].dropna()) > 0 else None
            
            cursor.execute(
                """INSERT INTO dataset_columns (user_id, dataset_name, column_name, data_type, is_numeric, min_value, max_value) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (user_id, 'forest_data', column, str(df[column].dtype), is_numeric, min_val, max_val)
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
            
            connection = get_db_connection()
            if connection:
                cursor = connection.cursor(dictionary=True)
                
                # Get or create admin user
                cursor.execute("SELECT * FROM users WHERE email = 'admin@forestpredict.com'")
                admin_user = cursor.fetchone()
                
                if admin_user:
                    # Update last login
                    cursor.execute("UPDATE users SET last_login = NOW() WHERE id = %s", (admin_user['id'],))
                    user_id = admin_user['id']
                else:
                    # Create admin user
                    cursor.execute('''
                        INSERT INTO users (uid, email, name, role, last_login)
                        VALUES (%s, %s, %s, %s, NOW())
                    ''', ('admin_local', 'admin@forestpredict.com', 'Admin User', 'admin'))
                    user_id = cursor.lastrowid
                
                connection.commit()
                
                # Log activity
                log_user_activity(
                    user_id,
                    'admin_login',
                    'Admin logged in',
                    request.remote_addr,
                    request.user_agent.string
                )
                
                cursor.close()
                connection.close()
            
            # Set session
            session['user_id'] = user_id if 'user_id' in locals() else 1
            session['email'] = 'admin@forestpredict.com'
            session['name'] = 'Admin User'
            session['role'] = 'admin'
            
            return jsonify({
                'success': True,
                'user': {
                    'id': user_id if 'user_id' in locals() else 1,
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
        connection = get_db_connection()
        if not connection:
            return jsonify({
                'success': False, 
                'message': 'Database connection failed',
                'total_records': 0,
                'total_datasets': 0,
                'total_predictions': 0,
                'start_year': 2000,
                'end_year': 2023,
                'has_user_data': False
            })
        
        cursor = connection.cursor(dictionary=True)
        
        # Get total records for THIS USER ONLY
        cursor.execute("SELECT COALESCE(SUM(row_count), 0) as total_records FROM datasets WHERE user_id = %s", (user_id,))
        total_records = cursor.fetchone()['total_records']
        
        # Get number of datasets for THIS USER ONLY
        cursor.execute("SELECT COUNT(*) as total_datasets FROM datasets WHERE user_id = %s", (user_id,))
        total_datasets = cursor.fetchone()['total_datasets']
        
        # Get number of predictions for THIS USER ONLY
        cursor.execute("SELECT COUNT(*) as total_predictions FROM predictions WHERE user_id = %s", (user_id,))
        total_predictions = cursor.fetchone()['total_predictions']
        
        # Check if user has uploaded custom data (excluding default forest_data)
        cursor.execute("SELECT COUNT(*) as user_datasets FROM datasets WHERE user_id = %s AND is_default = FALSE", (user_id,))
        user_datasets = cursor.fetchone()['user_datasets']
        has_user_data = user_datasets > 0
        
        # Get year range from user's primary dataset
        start_year = 2000
        end_year = 2023
        
        cursor.execute("SELECT name FROM datasets WHERE user_id = %s AND is_primary = TRUE", (user_id,))
        primary_dataset = cursor.fetchone()
        if primary_dataset:
            dataset_name = primary_dataset['name']
            cursor.execute("""
                SELECT MIN(min_value) as min_year, MAX(max_value) as max_year 
                FROM dataset_columns 
                WHERE user_id = %s AND dataset_name = %s AND column_name = 'Year'
            """, (user_id, dataset_name))
            year_data = cursor.fetchone()
            if year_data and year_data['min_year']:
                start_year = int(year_data['min_year'])
                end_year = int(year_data['max_year'])
        
        # Get last update time for THIS USER
        cursor.execute("SELECT MAX(uploaded_date) as last_updated FROM datasets WHERE user_id = %s", (user_id,))
        last_updated_result = cursor.fetchone()
        last_updated = last_updated_result['last_updated'] if last_updated_result else None
        
        cursor.close()
        connection.close()
        
        stats = {
            'success': True,
            'total_records': int(total_records),
            'total_datasets': total_datasets,
            'total_predictions': total_predictions,
            'start_year': start_year,
            'end_year': end_year,
            'has_user_data': has_user_data,
            'last_updated': str(last_updated) if last_updated else None
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
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'Database connection failed'})
        
        cursor = connection.cursor()
        
        # Check if user already has forest_data
        cursor.execute("SELECT COUNT(*) FROM datasets WHERE user_id = %s AND name = 'forest_data'", (user_id,))
        if cursor.fetchone()[0] > 0:
            # Update existing forest_data
            cursor.execute("UPDATE datasets SET row_count = %s, column_count = %s WHERE user_id = %s AND name = 'forest_data'", 
                          (len(df), len(df.columns), user_id))
            
            # Delete old columns
            cursor.execute("DELETE FROM dataset_columns WHERE user_id = %s AND dataset_name = 'forest_data'", (user_id,))
        else:
            # Insert new forest_data
            cursor.execute(
                """INSERT INTO datasets (user_id, name, row_count, column_count, is_primary, is_default, description) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (user_id, 'forest_data', len(df), len(df.columns), True, True, 'Forest dataset with environmental metrics')
            )
        
        # Insert column metadata
        for column in df.columns:
            is_numeric = pd.api.types.is_numeric_dtype(df[column])
            min_val = float(df[column].min()) if is_numeric and len(df[column].dropna()) > 0 else None
            max_val = float(df[column].max()) if is_numeric and len(df[column].dropna()) > 0 else None
            
            cursor.execute(
                """INSERT INTO dataset_columns (user_id, dataset_name, column_name, data_type, is_numeric, min_value, max_value) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (user_id, 'forest_data', column, str(df[column].dtype), is_numeric, min_val, max_val)
            )
        
        connection.commit()
        cursor.close()
        connection.close()
        
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
        connection = get_db_connection()
        if not connection:
            return jsonify([])
        
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
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
            WHERE d.user_id = %s
            GROUP BY d.id, d.name, d.row_count, d.column_count, d.is_primary, d.is_default, d.uploaded_date, d.description
            ORDER BY d.is_primary DESC, d.uploaded_date DESC
        """, (user_id,))
        
        datasets = cursor.fetchall()
        cursor.close()
        connection.close()
        
        # Format datasets
        formatted_datasets = []
        for ds in datasets:
            formatted_datasets.append({
                'name': ds['name'],
                'row_count': ds['row_count'],
                'column_count': ds['actual_columns'] or ds['column_count'],
                'is_primary': bool(ds['is_primary']),
                'is_default': bool(ds['is_default']),
                'uploaded_date': str(ds['uploaded_date']) if ds.get('uploaded_date') else None,
                'description': ds['description']
            })
        
        return jsonify({'success': True, 'datasets': formatted_datasets})
        
    except Exception as e:
        print(f"❌ Error getting datasets: {e}")
        return jsonify({'success': True, 'datasets': []})

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
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'Database connection failed'})
        
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT file_path, is_default FROM datasets WHERE user_id = %s AND name = %s", (user_id, dataset_name))
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        
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
    """Get columns for a specific dataset for current user, with is_numeric flag"""
    try:
        user_id = get_current_user_id()
        if not dataset_name:
            return jsonify({'success': False, 'columns': []})

        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'columns': []})

        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT COUNT(*) as count FROM datasets WHERE user_id = %s AND name = %s", (user_id, dataset_name))
        if cursor.fetchone()['count'] == 0:
            cursor.close(); connection.close()
            return jsonify({'success': False, 'columns': []})

        # Try DB metadata first
        cursor.execute("""
            SELECT column_name, data_type FROM dataset_columns
            WHERE user_id = %s AND dataset_name = %s ORDER BY id
        """, (user_id, dataset_name))
        db_cols = cursor.fetchall()
        cursor.close(); connection.close()

        if db_cols:
            NUMERIC_TYPES = {'int','float','double','decimal','numeric','bigint','smallint','tinyint','real','number'}
            result = []
            for c in db_cols:
                dtype = (c.get('data_type') or '').lower()
                is_num = any(t in dtype for t in NUMERIC_TYPES)
                result.append({'column_name': c['column_name'], 'is_numeric': is_num, 'data_type': dtype})
            return jsonify({'success': True, 'columns': result})

        # Fallback: load actual data and infer types
        dataset_response = get_dataset_data(dataset_name)
        resp = dataset_response.get_json()
        if resp and resp.get('success'):
            data = resp.get('data', [])
            if data:
                df_sample = pd.DataFrame(data[:50])
                result = []
                for col in df_sample.columns:
                    is_num = pd.api.types.is_numeric_dtype(df_sample[col])
                    result.append({'column_name': col, 'is_numeric': bool(is_num), 'data_type': 'float' if is_num else 'varchar'})
                return jsonify({'success': True, 'columns': result})

        return jsonify({'success': False, 'columns': []})

    except Exception as e:
        print(f"❌ Error getting columns: {e}")
        return jsonify({'success': False, 'columns': []})

@app.route('/api/dataset/<dataset_name>/row-identifiers', methods=['GET'])
@login_required
def get_row_identifiers(dataset_name):
    """Get the first categorical/non-numeric column and its unique values for row-wise prediction"""
    try:
        user_id = get_current_user_id()
        if not dataset_name:
            return jsonify({'success': False, 'message': 'Dataset name required'})

        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'Database connection failed'})

        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT COUNT(*) as count FROM datasets WHERE user_id = %s AND name = %s", (user_id, dataset_name))
        if cursor.fetchone()['count'] == 0:
            cursor.close()
            connection.close()
            return jsonify({'success': False, 'message': 'Dataset not found or access denied'})
        cursor.close()
        connection.close()

        dataset_response = get_dataset_data(dataset_name)
        resp_json = dataset_response.get_json()
        if not resp_json or not resp_json.get('success'):
            return jsonify({'success': False, 'message': 'Failed to load dataset'})

        df = pd.DataFrame(resp_json.get('data', []))
        if df.empty:
            return jsonify({'success': False, 'message': 'Dataset is empty'})

        # Find first non-numeric column (excluding Year/id/index)
        identifier_col = None
        for col in df.columns:
            if col.lower() in ['year', 'id', 'index']:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                identifier_col = col
                break

        if not identifier_col:
            return jsonify({'success': False, 'message': 'No categorical identifier column found in this dataset'})

        unique_vals = sorted(df[identifier_col].dropna().unique().tolist())
        # Also build identifiers list for all non-numeric columns
        identifiers = []
        for col in df.columns:
            if col.lower() in ['year', 'id', 'index']:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                vals = sorted(df[col].dropna().unique().tolist())
                identifiers.append({'column': col, 'values': vals})
        return jsonify({'success': True, 'identifier_column': identifier_col,
                        'values': unique_vals, 'identifiers': identifiers})

    except Exception as e:
        print(f'❌ Error getting row identifiers: {e}')
        return jsonify({'success': False, 'message': str(e)})


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
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'Database connection failed'})
        
        cursor = connection.cursor()
        
        # Check if dataset name exists for this user
        cursor.execute("SELECT COUNT(*) FROM datasets WHERE user_id = %s AND name = %s", (user_id, dataset_name))
        if cursor.fetchone()[0] > 0:
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
        cursor.execute(
            """INSERT INTO datasets (user_id, name, file_path, row_count, column_count, description) 
            VALUES (%s, %s, %s, %s, %s, %s)""",
            (user_id, dataset_name, file_path, len(df), len(df.columns), f"Uploaded: {file.filename}")
        )
        
        # Store column metadata for this user
        for column in df.columns:
            is_numeric = pd.api.types.is_numeric_dtype(df[column])
            min_val = float(df[column].min()) if is_numeric and len(df[column].dropna()) > 0 else None
            max_val = float(df[column].max()) if is_numeric and len(df[column].dropna()) > 0 else None
            
            cursor.execute(
                """INSERT INTO dataset_columns (user_id, dataset_name, column_name, data_type, is_numeric, min_value, max_value) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (user_id, dataset_name, column, str(df[column].dtype), is_numeric, min_val, max_val)
            )
        
        connection.commit()
        
        # Log the upload activity
        log_user_activity(
            user_id,
            'dataset_upload',
            f'Uploaded dataset: {dataset_name} with {len(df)} rows',
            request.remote_addr,
            request.user_agent.string
        )
        
        cursor.close()
        connection.close()
        
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
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'Database connection failed'})
        
        cursor = connection.cursor()
        
        if action == 'set_primary':
            # Reset all primary flags for this user
            cursor.execute("UPDATE datasets SET is_primary = FALSE WHERE user_id = %s", (user_id,))
            
            # Set the selected dataset as primary for this user
            cursor.execute("UPDATE datasets SET is_primary = TRUE WHERE user_id = %s AND name = %s", (user_id, dataset_name))
            
            connection.commit()
            cursor.close()
            connection.close()
            
            return jsonify({'success': True, 'message': f'✅ Dataset "{dataset_name}" set as primary'})
        
        elif action == 'delete':
            # Don't allow deletion of default forest_data
            if dataset_name == 'forest_data':
                cursor.close()
                connection.close()
                return jsonify({'success': False, 'message': 'Cannot delete default forest_data dataset'})
            
            # Get file path to delete physical file
            cursor.execute("SELECT file_path FROM datasets WHERE user_id = %s AND name = %s", (user_id, dataset_name))
            result = cursor.fetchone()
            
            # Delete from database
            cursor.execute("DELETE FROM datasets WHERE user_id = %s AND name = %s", (user_id, dataset_name))
            connection.commit()
            
            # Delete physical file
            if result and result[0] and os.path.exists(result[0]):
                try:
                    os.remove(result[0])
                except Exception as e:
                    print(f"Warning: Could not delete file: {e}")
            
            cursor.close()
            connection.close()
            
            return jsonify({'success': True, 'message': f'✅ Dataset "{dataset_name}" deleted successfully'})
        
        else:
            cursor.close()
            connection.close()
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
        connection = get_db_connection()
        if not connection:
            return jsonify([])
        
        cursor = connection.cursor(dictionary=True)
        
        # Get primary dataset name for this user
        cursor.execute("SELECT name FROM datasets WHERE user_id = %s AND is_primary = TRUE", (user_id,))
        primary_result = cursor.fetchone()
        
        if not primary_result:
            # If no primary, get forest_data for this user
            cursor.execute("SELECT name FROM datasets WHERE user_id = %s AND name = 'forest_data' LIMIT 1", (user_id,))
            forest_result = cursor.fetchone()
            dataset_name = forest_result['name'] if forest_result else None
        else:
            dataset_name = primary_result['name']
        
        if not dataset_name:
            cursor.close()
            connection.close()
            return jsonify([])
        
        # Get numeric columns for this user's dataset
        cursor.execute("""
            SELECT column_name 
            FROM dataset_columns 
            WHERE user_id = %s AND dataset_name = %s AND is_numeric = TRUE
            ORDER BY column_name
        """, (user_id, dataset_name))
        
        numeric_columns = [row['column_name'] for row in cursor.fetchall()]
        cursor.close()
        connection.close()
        
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

# Add this new endpoint for debugging dataset structure
@app.route('/api/debug-dataset/<dataset_name>', methods=['GET'])
@login_required
def debug_dataset(dataset_name):
    """Debug endpoint to check dataset structure"""
    try:
        user_id = get_current_user_id()
        
        # Get dataset data
        dataset_response = get_dataset_data(dataset_name)
        if not dataset_response.json or not dataset_response.json.get('success'):
            return jsonify({'success': False, 'message': 'Failed to load dataset'})
        
        data = dataset_response.json.get('data', [])
        if not data:
            return jsonify({'success': False, 'message': 'Dataset is empty'})
        
        df = pd.DataFrame(data)
        
        # Analyze dataset structure
        debug_info = {
            'shape': list(df.shape),
            'columns': df.columns.tolist(),
            'dtypes': {},
            'sample_rows': df.head(3).to_dict('records'),
            'numeric_columns': [],
            'non_numeric_columns': [],
            'year_column': None,
            'country_column': None
        }
        
        # Check each column
        for col in df.columns:
            # Try to convert to numeric
            try:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.notna().any():
                    debug_info['dtypes'][col] = str(df[col].dtype)
                    debug_info['numeric_columns'].append({
                        'name': col,
                        'sample': df[col].iloc[0] if len(df) > 0 else None,
                        'non_null_count': df[col].notna().sum(),
                        'unique_values': df[col].nunique()
                    })
                else:
                    debug_info['non_numeric_columns'].append(col)
            except:
                debug_info['non_numeric_columns'].append(col)
            
            # Check for year column
            if 'year' in col.lower():
                debug_info['year_column'] = col
            
            # Check for country/region column
            if any(x in col.lower() for x in ['country', 'nation', 'region', 'name']):
                debug_info['country_column'] = col
        
        return jsonify({'success': True, 'debug_info': debug_info})
        
    except Exception as e:
        print(f"❌ Error debugging dataset: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

# Updated prediction endpoint with better data handling
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
        filter_column = data.get('filter_column')
        filter_value = data.get('filter_value')
        model_name = data.get('model', 'linear_regression')
        poly_degree = int(data.get('degree', 2))
        VALID_MODELS = {'linear_regression','random_forest','decision_tree','gradient_boosting','polynomial_regression'}
        if model_name not in VALID_MODELS:
            model_name = 'linear_regression'

        print(f"📊 Prediction request: user={user_id}, dataset={dataset_name}, metric={metric}, year={target_year}, type={prediction_type}, filter={filter_column}={filter_value}")

        if not target_year:
            return jsonify({'success': False, 'message': 'Missing target year'})

        # Use primary dataset for this user if none specified
        if not dataset_name:
            connection = get_db_connection()
            if connection:
                cursor = connection.cursor()
                cursor.execute("SELECT name FROM datasets WHERE user_id = %s AND is_primary = TRUE", (user_id,))
                result = cursor.fetchone()
                dataset_name = result[0] if result else 'forest_data'
                cursor.close()
                connection.close()

        # Get dataset data
        dataset_response = get_dataset_data(dataset_name)
        if not dataset_response.json or not dataset_response.json.get('success'):
            return jsonify({'success': False, 'message': 'Failed to load dataset'})

        df = pd.DataFrame(dataset_response.json.get('data', []))
        
        print(f"📊 DataFrame shape: {df.shape}")
        print(f"📊 DataFrame columns: {df.columns.tolist()}")
        
        if df.empty:
            return jsonify({'success': False, 'message': 'Dataset is empty'})

        # Special handling for forest_data.csv (wide format: Country Name + year columns as headers)
        # Only melt if the dataframe does NOT already have a proper 'Year' column
        # AND the non-identifier columns look like year numbers (e.g. '2000', '2001', ...)
        if 'Country Name' in df.columns and 'Year' not in df.columns:
            year_like_cols = [col for col in df.columns
                              if col != 'Country Name' and str(col).strip().replace('.', '').isdigit()]
            if year_like_cols:
                print("📊 Detected wide-format forest_data: melting to long format")
                df_melted = pd.melt(df,
                                    id_vars=['Country Name'],
                                    var_name='Year',
                                    value_name='Forest Area (%)')
                df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce')
                df_melted['Forest Area (%)'] = pd.to_numeric(df_melted['Forest Area (%)'], errors='coerce')
                df_melted = df_melted.dropna(subset=['Year', 'Forest Area (%)'])
                print(f"📊 After melting: {len(df_melted)} rows")
                df = df_melted

        # Convert columns to numeric where possible
        for col in df.columns:
            if col != filter_column and col != 'Country Name':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass

        # Find Year column (case-insensitive search)
        year_column = None
        for col in df.columns:
            if 'year' in col.lower():
                year_column = col
                break
        
        if year_column:
            if year_column != 'Year':
                df = df.rename(columns={year_column: 'Year'})
        else:
            # If no year column found, create one from index
            print("⚠️ No year column found, using row index as year")
            df['Year'] = range(len(df))

        # Ensure Year is numeric
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

        # Find Country/Region column for filtering
        country_column = None
        if filter_column:
            country_column = filter_column
        else:
            for col in df.columns:
                if any(x in col.lower() for x in ['country', 'nation', 'region', 'name']):
                    if col != 'Year':
                        country_column = col
                        break

        # Apply filter if specified
        if filter_column and filter_value:
            if filter_column not in df.columns:
                # Try to find a matching column
                for col in df.columns:
                    if filter_column.lower() in col.lower():
                        filter_column = col
                        break
                else:
                    return jsonify({'success': False, 'message': f'Filter column "{filter_column}" not found. Available columns: {df.columns.tolist()}'})
            
            # Convert to string for comparison
            df[filter_column] = df[filter_column].astype(str)
            df = df[df[filter_column] == str(filter_value)]
            
            if df.empty:
                # Try partial matching
                df = pd.DataFrame(dataset_response.json.get('data', []))
                if 'Country Name' in df.columns:
                    df = df[df['Country Name'].str.contains(str(filter_value), case=False, na=False)]
                    if not df.empty:
                        # Melt again
                        df = pd.melt(df, id_vars=['Country Name'], var_name='Year', value_name='Forest Area (%)')
                        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                        df['Forest Area (%)'] = pd.to_numeric(df['Forest Area (%)'], errors='coerce')
                        df = df.dropna()
                    else:
                        return jsonify({'success': False, 'message': f'No data found for {filter_column} = "{filter_value}"'})
                else:
                    return jsonify({'success': False, 'message': f'No data found for {filter_column} = "{filter_value}"'})
            
            print(f"📊 After filtering: {len(df)} rows for {filter_value}")

        # ── ROW-WISE: predict ALL numeric columns for the selected row ──
        if not metric and (filter_column or country_column):
            # Get all numeric columns
            numeric_cols = []
            for col in df.columns:
                if col != 'Year' and col != filter_column and col != country_column:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        # Check if column has enough non-null values
                        if df[col].notna().sum() >= 3:
                            numeric_cols.append(col)
                        else:
                            print(f"⚠️ Column '{col}' has insufficient data: {df[col].notna().sum()} valid points")
            
            print(f"📊 Numeric columns with sufficient data: {numeric_cols}")
            
            if not numeric_cols:
                # If no numeric columns found with sufficient data, try to use Forest Area (%) if it exists
                if 'Forest Area (%)' in df.columns and df['Forest Area (%)'].notna().sum() >= 3:
                    numeric_cols = ['Forest Area (%)']
                else:
                    # Show all columns and their status
                    debug_info = []
                    for col in df.columns:
                        if col != 'Year':
                            is_num = pd.api.types.is_numeric_dtype(df[col])
                            valid_count = df[col].notna().sum() if is_num else 0
                            debug_info.append(f"{col}: numeric={is_num}, valid_points={valid_count}")
                    
                    return jsonify({
                        'success': False, 
                        'message': f'No numeric columns with sufficient data (need at least 3 points). Column status: {", ".join(debug_info)}'
                    })

            all_predictions = []
            for col in numeric_cols:
                # Clean data for this column
                df_clean = df[['Year', col]].copy()
                df_clean = df_clean.dropna()
                
                # Ensure proper types
                df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean = df_clean.dropna()

                print(f"📊 Column '{col}': {len(df_clean)} valid data points after cleaning")

                if len(df_clean) < 3:
                    print(f"⚠️ Insufficient data for column '{col}': {len(df_clean)} points (need at least 3)")
                    continue

                X = df_clean['Year'].values.reshape(-1, 1)
                y = df_clean[col].values

                fitted_model, accuracy, _, _ = _fit_and_score(X, y, model_name, poly_degree)
                prediction = fitted_model.predict([[target_year]])[0]
                
                all_predictions.append({
                    'metric': col,
                    'prediction': float(prediction),
                    'accuracy': float(accuracy),
                    'data_points': len(df_clean),
                    'model_used': model_name
                })

                # Save each prediction to DB
                try:
                    conn = get_db_connection()
                    if conn:
                        cur = conn.cursor()
                        cur.execute(
                            """INSERT INTO predictions (user_id, dataset_name, predicted_metric, year, predicted_value, accuracy, prediction_type, model_used)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                            (user_id, dataset_name, col, target_year, float(prediction), float(accuracy), 'row_wise', model_name)
                        )
                        conn.commit()
                        cur.close()
                        conn.close()
                except Exception as e:
                    print(f"⚠️ Error saving prediction to DB: {e}")

            if not all_predictions:
                return jsonify({
                    'success': False, 
                    'message': 'No valid predictions could be made. Each column needs at least 3 data points. Try using a country with more historical data.'
                })

            return jsonify({
                'success': True,
                'row_wise': True,
                'filter_column': filter_column or country_column,
                'filter_value': filter_value,
                'year': target_year,
                'dataset': dataset_name,
                'predictions': all_predictions
            })

        # ── COLUMN-WISE: predict a specific metric ──
        if metric:
            # Find the metric column (case-insensitive)
            metric_column = None
            for col in df.columns:
                if metric.lower() in col.lower():
                    metric_column = col
                    break
            
            if not metric_column:
                return jsonify({'success': False, 'message': f'Metric "{metric}" not found. Available columns: {df.columns.tolist()}'})
            
            # Convert metric column to numeric
            df[metric_column] = pd.to_numeric(df[metric_column], errors='coerce')
            
            df_clean = df[['Year', metric_column]].copy()
            df_clean = df_clean.dropna()

            print(f"📊 Column '{metric_column}': {len(df_clean)} valid data points after cleaning")

            if len(df_clean) < 3:
                return jsonify({'success': False, 'message': f'Insufficient data for prediction (only {len(df_clean)} row(s) after filtering). Need at least 3 data points.'})

            X = df_clean['Year'].values.reshape(-1, 1)
            y = df_clean[metric_column].values

            fitted_model, accuracy, train_score, test_score = _fit_and_score(X, y, model_name, poly_degree)
            prediction = fitted_model.predict([[target_year]])[0]

            # Save to database
            try:
                connection = get_db_connection()
                if connection:
                    cursor = connection.cursor()
                    cursor.execute(
                        """INSERT INTO predictions (user_id, dataset_name, predicted_metric, year, predicted_value, accuracy, prediction_type, model_used)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                        (user_id, dataset_name, metric_column, target_year, float(prediction), float(accuracy), prediction_type, model_name)
                    )
                    connection.commit()
                    cursor.close()
                    connection.close()
            except Exception as e:
                print(f"⚠️ Error saving prediction to DB: {e}")

            return jsonify({
                'success': True,
                'prediction': float(prediction),
                'accuracy': float(accuracy),
                'train_score': float(train_score),
                'test_score': float(test_score),
                'data_points': len(df_clean),
                'metric': metric_column,
                'year': target_year,
                'dataset': dataset_name,
                'model_used': model_name
            })

        return jsonify({'success': False, 'message': 'Invalid prediction request'})

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
        connection = get_db_connection()
        if not connection:
            return jsonify([])
        
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM predictions WHERE user_id = %s ORDER BY prediction_date DESC", (user_id,))
        
        predictions = cursor.fetchall()
        
        # Format dates
        for pred in predictions:
            if pred['prediction_date']:
                pred['prediction_date'] = str(pred['prediction_date'])
        
        cursor.close()
        connection.close()
        
        return jsonify(predictions)
        
    except Exception as e:
        print(f"❌ Error getting predictions: {e}")
        return jsonify([])

@app.route('/api/prediction/<int:prediction_id>', methods=['DELETE'])
@login_required
def delete_prediction(prediction_id):
    """Delete a specific prediction for current user"""
    try:
        user_id = get_current_user_id()
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'Database connection failed'})
        
        cursor = connection.cursor()
        cursor.execute("DELETE FROM predictions WHERE id = %s AND user_id = %s", (prediction_id, user_id))
        connection.commit()
        
        rows_deleted = cursor.rowcount
        cursor.close()
        connection.close()
        
        if rows_deleted > 0:
            return jsonify({'success': True, 'message': 'Prediction deleted successfully'})
        else:
            return jsonify({'success': False, 'message': 'Prediction not found or access denied'})
        
    except Exception as e:
        print(f"❌ Error deleting prediction: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/clear-predictions', methods=['POST'])
@login_required
def clear_predictions():
    """Clear all predictions for current user"""
    try:
        user_id = get_current_user_id()
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'Database connection failed'})
        
        cursor = connection.cursor()
        cursor.execute("DELETE FROM predictions WHERE user_id = %s", (user_id,))
        connection.commit()
        
        cursor.close()
        connection.close()
        
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
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'data': []})
        
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM datasets WHERE user_id = %s AND is_primary = TRUE", (user_id,))
        result = cursor.fetchone()
        dataset_name = result[0] if result else 'forest_data'
        cursor.close()
        connection.close()
        
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
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'Database connection failed'})
        
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM datasets WHERE user_id = %s AND name = %s", (user_id, dataset_name))
        count = cursor.fetchone()[0]
        cursor.close()
        connection.close()
        
        if count == 0:
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

# Admin routes remain unchanged
@app.route('/api/admin/users', methods=['GET'])
@admin_required
def get_users():
    """Get all users for admin"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'users': []})
        
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute('''
            SELECT 
                u.*,
                COUNT(DISTINCT a.id) as activity_count,
                MAX(a.created_at) as last_activity
            FROM users u
            LEFT JOIN user_activity a ON u.id = a.user_id
            GROUP BY u.id
            ORDER BY u.created_at DESC
        ''')
        
        users = cursor.fetchall()
        
        for user in users:
            user['last_login'] = str(user['last_login']) if user.get('last_login') else None
            user['created_at'] = str(user['created_at']) if user.get('created_at') else None
            user['updated_at'] = str(user['updated_at']) if user.get('updated_at') else None
            user['last_activity'] = str(user['last_activity']) if user.get('last_activity') else None
        
        cursor.close()
        connection.close()
        
        return jsonify({'success': True, 'users': users})
        
    except Exception as e:
        print(f"❌ Error getting users: {e}")
        return jsonify({'success': False, 'users': []})

@app.route('/api/admin/user/<int:user_id>/activity', methods=['GET'])
@admin_required
def get_user_activity(user_id):
    """Get activity for a specific user"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'activities': []})
        
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute('''
            SELECT * FROM user_activity 
            WHERE user_id = %s 
            ORDER BY created_at DESC 
            LIMIT 100
        ''', (user_id,))
        
        activities = cursor.fetchall()
        
        for activity in activities:
            if activity.get('created_at'):
                activity['created_at'] = str(activity['created_at'])
        
        cursor.close()
        connection.close()
        
        return jsonify({'success': True, 'activities': activities})
        
    except Exception as e:
        print(f"❌ Error getting user activity: {e}")
        return jsonify({'success': False, 'activities': []})

@app.route('/api/admin/user/<int:user_id>/toggle-status', methods=['POST'])
@admin_required
def toggle_user_status(user_id):
    """Toggle user active status"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'Database connection failed'})
        
        cursor = connection.cursor()
        
        cursor.execute("SELECT is_active FROM users WHERE id = %s", (user_id,))
        result = cursor.fetchone()
        
        if not result:
            cursor.close()
            connection.close()
            return jsonify({'success': False, 'message': 'User not found'})
        
        new_status = not result[0]
        cursor.execute("UPDATE users SET is_active = %s WHERE id = %s", (new_status, user_id))
        connection.commit()
        
        cursor.close()
        connection.close()
        
        return jsonify({
            'success': True, 
            'message': f'User {"activated" if new_status else "deactivated"} successfully',
            'is_active': new_status
        })
        
    except Exception as e:
        print(f"❌ Error toggling user status: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/admin/user/<int:user_id>', methods=['DELETE'])
@admin_required
def delete_user(user_id):
    """Delete a user"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'Database connection failed'})
        
        cursor = connection.cursor()
        
        cursor.execute("SELECT role FROM users WHERE id = %s", (user_id,))
        result = cursor.fetchone()
        
        if not result:
            cursor.close()
            connection.close()
            return jsonify({'success': False, 'message': 'User not found'})
        
        if result[0] == 'admin':
            cursor.close()
            connection.close()
            return jsonify({'success': False, 'message': 'Cannot delete admin user'})
        
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        connection.commit()
        
        cursor.close()
        connection.close()
        
        return jsonify({'success': True, 'message': 'User deleted successfully'})
        
    except Exception as e:
        print(f"❌ Error deleting user: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/admin/dashboard-stats', methods=['GET'])
@admin_required
def get_admin_dashboard_stats():
    """Get admin dashboard statistics"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'stats': {}})
        
        cursor = connection.cursor(dictionary=True)
        
        # Get user counts
        cursor.execute("SELECT COUNT(*) as total_users FROM users")
        total_users = cursor.fetchone()['total_users']
        
        cursor.execute("SELECT COUNT(*) as active_users FROM users WHERE is_active = 1")
        active_users = cursor.fetchone()['active_users']
        
        cursor.execute("SELECT COUNT(*) as admin_users FROM users WHERE role = 'admin'")
        admin_users = cursor.fetchone()['admin_users']
        
        # Get activity stats for last 7 days
        cursor.execute('''
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM user_activity
            WHERE created_at >= datetime('now', '-7 days')
            GROUP BY DATE(created_at)
            ORDER BY date
        ''')
        activity_by_day = cursor.fetchall()
        
        # Get activity types distribution
        cursor.execute('''
            SELECT activity_type, COUNT(*) as count
            FROM user_activity
            GROUP BY activity_type
            ORDER BY count DESC
        ''')
        activity_types = cursor.fetchall()
        
        # Get prediction stats - FIXED
        cursor.execute("SELECT COUNT(*) as total_predictions FROM predictions")
        total_predictions_result = cursor.fetchone()
        total_predictions = total_predictions_result['total_predictions'] if total_predictions_result else 0
        
        # Get dataset stats - FIXED
        cursor.execute("SELECT COUNT(*) as total_datasets FROM datasets")
        total_datasets_result = cursor.fetchone()
        total_datasets = total_datasets_result['total_datasets'] if total_datasets_result else 0
        
        # Get quick predictions count
        cursor.execute("SELECT COUNT(*) as quick_count FROM predictions WHERE prediction_type = 'quick'")
        quick_count_result = cursor.fetchone()
        quick_count = quick_count_result['quick_count'] if quick_count_result else 0
        
        # Get standard predictions count
        cursor.execute("SELECT COUNT(*) as standard_count FROM predictions WHERE prediction_type = 'standard'")
        standard_count_result = cursor.fetchone()
        standard_count = standard_count_result['standard_count'] if standard_count_result else 0
        
        cursor.close()
        connection.close()
        
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

# Add these routes after your existing admin routes in app.py

@app.route('/api/admin/all-predictions', methods=['GET'])
@admin_required
def get_all_predictions():
    """Get all predictions from all users for admin"""
    try:
        connection = get_db_connection()
        if not connection:
            print("❌ Database connection failed in get_all_predictions")
            return jsonify({'success': False, 'predictions': [], 'message': 'Database connection failed'})
        
        cursor = connection.cursor(dictionary=True)
        
        # Get all predictions with user information - FIXED QUERY
        cursor.execute('''
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
        ''')
        
        predictions = cursor.fetchall()
        print(f"✅ Found {len(predictions)} predictions across all users")
        
        # Format dates and values
        for pred in predictions:
            if pred['prediction_date']:
                pred['prediction_date'] = pred['prediction_date'].isoformat() if hasattr(pred['prediction_date'], 'isoformat') else str(pred['prediction_date'])
            # Ensure numeric values are properly formatted
            if pred['predicted_value'] is not None:
                pred['predicted_value'] = float(pred['predicted_value'])
            if pred['accuracy'] is not None:
                pred['accuracy'] = float(pred['accuracy'])
        
        cursor.close()
        connection.close()
        
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
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'predictions': [], 'message': 'Database connection failed'})
        
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute('''
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
            WHERE p.user_id = %s
            ORDER BY p.prediction_date DESC
        ''', (user_id,))
        
        predictions = cursor.fetchall()
        print(f"✅ Found {len(predictions)} predictions for user {user_id}")
        
        # Format dates and values
        for pred in predictions:
            if pred['prediction_date']:
                pred['prediction_date'] = pred['prediction_date'].isoformat() if hasattr(pred['prediction_date'], 'isoformat') else str(pred['prediction_date'])
            if pred['predicted_value'] is not None:
                pred['predicted_value'] = float(pred['predicted_value'])
            if pred['accuracy'] is not None:
                pred['accuracy'] = float(pred['accuracy'])
        
        cursor.close()
        connection.close()
        
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
        connection = get_db_connection()
        if not connection:
            print("❌ Database connection failed in get_all_datasets")
            return jsonify({'success': False, 'datasets': [], 'message': 'Database connection failed'})
        
        cursor = connection.cursor(dictionary=True)
        
        # Get all datasets with user information - FIXED QUERY
        cursor.execute('''
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
        ''')
        
        datasets = cursor.fetchall()
        print(f"✅ Found {len(datasets)} datasets across all users")
        
        # Format dates
        for ds in datasets:
            if ds['uploaded_date']:
                ds['uploaded_date'] = ds['uploaded_date'].isoformat() if hasattr(ds['uploaded_date'], 'isoformat') else str(ds['uploaded_date'])
        
        cursor.close()
        connection.close()
        
        return jsonify({'success': True, 'datasets': datasets, 'count': len(datasets)})
        
    except Exception as e:
        print(f"❌ Error getting all datasets: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'datasets': [], 'message': str(e)})




# =============================================================================
# HELPER: Build ML model by name
# =============================================================================
def build_model(model_name: str, degree: int = 2):
    """Return a configured sklearn estimator based on model_name."""
    if model_name == 'random_forest':
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == 'decision_tree':
        return DecisionTreeRegressor(max_depth=6, random_state=42)
    elif model_name == 'gradient_boosting':
        return GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_name == 'polynomial_regression':
        return make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    else:  # linear_regression (default)
        return LinearRegression()


def _fit_and_score(X, y, model_name='linear_regression', degree=2):
    """Fit model and return (model, accuracy, train_score, test_score)."""
    model = build_model(model_name, degree)
    if len(X) >= 6:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        train_score = max(0.0, float(model.score(X_train, y_train)))
        test_score  = max(0.0, float(model.score(X_test, y_test)))
        accuracy = round(train_score * 0.4 + test_score * 0.6, 4)
    else:
        model.fit(X, y)
        accuracy = max(0.0, round(float(model.score(X, y)), 4))
        train_score = test_score = accuracy
    return model, accuracy, train_score, test_score


# =============================================================================
# USER: Multi-model comparison
# =============================================================================
@app.route('/api/predict/compare-models', methods=['POST'])
@login_required
def compare_models():
    """Run all 5 models on the same data and return accuracy scores."""
    try:
        data = request.get_json()
        dataset_name  = data.get('dataset')
        metric        = data.get('metric')
        target_year   = int(data.get('target_year', 2030))
        filter_column = data.get('filter_column')
        filter_value  = data.get('filter_value')

        if not metric:
            return jsonify({'success': False, 'message': 'metric required'})

        rj = get_dataset_data(dataset_name).get_json()
        if not rj or not rj.get('success'):
            return jsonify({'success': False, 'message': 'Failed to load dataset'})

        df = pd.DataFrame(rj.get('data', []))
        if df.empty:
            return jsonify({'success': False, 'message': 'Dataset empty'})

        # Melt wide format
        if 'Country Name' in df.columns and 'Year' not in df.columns:
            year_cols = [c for c in df.columns if c != 'Country Name' and str(c).strip().replace('.','').isdigit()]
            if year_cols:
                df = pd.melt(df, id_vars=['Country Name'], var_name='Year', value_name='Forest Area (%)')
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                df['Forest Area (%)'] = pd.to_numeric(df['Forest Area (%)'], errors='coerce')
                df = df.dropna()

        for col in df.columns:
            if 'year' in col.lower():
                df = df.rename(columns={col: 'Year'})
                break
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

        if filter_column and filter_value and filter_column in df.columns:
            df[filter_column] = df[filter_column].astype(str)
            df = df[df[filter_column] == str(filter_value)]

        metric_col = next((c for c in df.columns if metric.lower() in c.lower()), None)
        if not metric_col:
            return jsonify({'success': False, 'message': f'Metric "{metric}" not found'})

        df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')
        df_clean = df[['Year', metric_col]].dropna()
        if len(df_clean) < 4:
            return jsonify({'success': False, 'message': 'Need at least 4 data points'})

        X = df_clean['Year'].values.reshape(-1, 1)
        y = df_clean[metric_col].values

        MODEL_NAMES = ['linear_regression', 'random_forest', 'decision_tree',
                       'gradient_boosting', 'polynomial_regression']
        results = []
        best_accuracy = -1
        best_model_name = 'linear_regression'

        for mn in MODEL_NAMES:
            try:
                model, accuracy, train_score, test_score = _fit_and_score(X, y, mn)
                pred_val = float(model.predict([[target_year]])[0])
                results.append({
                    'model': mn,
                    'label': mn.replace('_', ' ').title(),
                    'accuracy': round(accuracy * 100, 2),
                    'train_score': round(train_score * 100, 2),
                    'test_score': round(test_score * 100, 2),
                    'prediction': round(pred_val, 4)
                })
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = mn
            except Exception as e:
                results.append({'model': mn, 'label': mn.replace('_', ' ').title(),
                                 'accuracy': 0, 'prediction': None, 'error': str(e)})

        return jsonify({
            'success': True,
            'metric': metric_col,
            'year': target_year,
            'dataset': dataset_name,
            'results': results,
            'best_model': best_model_name
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


# =============================================================================
# USER: Dataset Cleaning Tools
# =============================================================================
@app.route('/api/dataset/<dataset_name>/clean', methods=['POST'])
@login_required
def clean_dataset(dataset_name):
    """Apply one or more cleaning operations to a user-owned dataset file."""
    try:
        user_id = get_current_user_id()
        data = request.get_json()
        operations = data.get('operations', [])

        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'DB error'})
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT file_path, is_default FROM datasets WHERE user_id=%s AND name=%s",
                       (user_id, dataset_name))
        ds = cursor.fetchone()
        cursor.close(); connection.close()

        if not ds:
            return jsonify({'success': False, 'message': 'Dataset not found'})
        if ds.get('is_default'):
            return jsonify({'success': False, 'message': 'Cannot clean default dataset'})

        file_path = ds['file_path']
        if not file_path or not os.path.exists(file_path):
            return jsonify({'success': False, 'message': 'File not found'})

        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        rows_before = len(df)
        logs = []

        for op in operations:
            op_type = op.get('type')
            if op_type == 'remove_nulls':
                df = df.dropna()
                logs.append(f'Removed null rows -> {len(df)} rows remain')
            elif op_type == 'remove_duplicates':
                df = df.drop_duplicates()
                logs.append(f'Removed duplicates -> {len(df)} rows remain')
            elif op_type == 'normalize':
                cols = op.get('columns', [])
                num_cols = [c for c in (cols or df.columns) if pd.api.types.is_numeric_dtype(df.get(c, pd.Series(dtype=float)))]
                for c in num_cols:
                    mn, mx = df[c].min(), df[c].max()
                    if mx - mn > 0:
                        df[c] = (df[c] - mn) / (mx - mn)
                logs.append(f'Normalized: {num_cols}')
            elif op_type == 'rename_column':
                old_n = op.get('old_name')
                new_n = op.get('new_name')
                if old_n and new_n and old_n in df.columns:
                    df = df.rename(columns={old_n: new_n})
                    logs.append(f'Renamed "{old_n}" -> "{new_n}"')
            elif op_type == 'remove_column':
                col = op.get('column')
                if col and col in df.columns:
                    df = df.drop(columns=[col])
                    logs.append(f'Removed column "{col}"')
            elif op_type == 'fill_nulls':
                strategy = op.get('strategy', 'mean')
                for c in df.select_dtypes(include=[np.number]).columns:
                    if strategy == 'mean':
                        df[c] = df[c].fillna(df[c].mean())
                    elif strategy == 'median':
                        df[c] = df[c].fillna(df[c].median())
                    else:
                        df[c] = df[c].fillna(0)
                logs.append(f'Filled nulls with {strategy}')

        rows_after = len(df)
        if file_path.endswith('.csv'):
            df.to_csv(file_path, index=False)
        else:
            df.to_excel(file_path, index=False)

        connection = get_db_connection()
        if connection:
            cursor = connection.cursor()
            cursor.execute("UPDATE datasets SET row_count=%s, column_count=%s WHERE user_id=%s AND name=%s",
                           (rows_after, len(df.columns), user_id, dataset_name))
            cursor.execute("DELETE FROM dataset_columns WHERE user_id=%s AND dataset_name=%s", (user_id, dataset_name))
            for col in df.columns:
                is_num = pd.api.types.is_numeric_dtype(df[col])
                mn = float(df[col].min()) if is_num and df[col].notna().any() else None
                mx = float(df[col].max()) if is_num and df[col].notna().any() else None
                cursor.execute(
                    "INSERT INTO dataset_columns (user_id,dataset_name,column_name,data_type,is_numeric,min_value,max_value)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (user_id, dataset_name, col, str(df[col].dtype), is_num, mn, mx))
            cursor.execute(
                "INSERT INTO dataset_clean_log (user_id,dataset_name,operation,rows_before,rows_after,details)"
                " VALUES (%s,%s,%s,%s,%s,%s)",
                (user_id, dataset_name, json.dumps(operations), rows_before, rows_after, '; '.join(logs)))
            connection.commit()
            cursor.close(); connection.close()

        log_user_activity(user_id, 'dataset_clean', f'Cleaned {dataset_name}',
                          request.remote_addr, request.user_agent.string)

        return jsonify({
            'success': True,
            'rows_before': rows_before,
            'rows_after': rows_after,
            'removed': rows_before - rows_after,
            'columns': list(df.columns),
            'logs': logs
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


# =============================================================================
# USER: Prediction History Analytics
# =============================================================================
@app.route('/api/prediction-analytics', methods=['GET'])
@login_required
def prediction_analytics():
    """Return prediction analytics for the current user."""
    try:
        user_id = get_current_user_id()
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'DB error'})
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT DATE(prediction_date) as date,
                   AVG(accuracy)*100 as avg_accuracy, COUNT(*) as count
            FROM predictions WHERE user_id=%s AND accuracy IS NOT NULL
            GROUP BY DATE(prediction_date) ORDER BY date ASC
        """, (user_id,))
        accuracy_trend = cursor.fetchall()

        cursor.execute("""
            SELECT predicted_metric as metric, COUNT(*) as count,
                   AVG(accuracy)*100 as avg_accuracy
            FROM predictions WHERE user_id=%s
            GROUP BY predicted_metric ORDER BY count DESC LIMIT 10
        """, (user_id,))
        top_metrics = cursor.fetchall()

        cursor.execute("""
            SELECT COALESCE(model_used,'linear_regression') as model, COUNT(*) as count
            FROM predictions WHERE user_id=%s GROUP BY model_used
        """, (user_id,))
        model_dist = cursor.fetchall()

        cursor.execute("""
            SELECT
                SUM(CASE WHEN accuracy >= 0.9  THEN 1 ELSE 0 END) as excellent,
                SUM(CASE WHEN accuracy >= 0.7 AND accuracy < 0.9  THEN 1 ELSE 0 END) as good,
                SUM(CASE WHEN accuracy >= 0.5 AND accuracy < 0.7  THEN 1 ELSE 0 END) as fair,
                SUM(CASE WHEN accuracy < 0.5  THEN 1 ELSE 0 END) as poor
            FROM predictions WHERE user_id=%s AND accuracy IS NOT NULL
        """, (user_id,))
        acc_dist = cursor.fetchone()

        cursor.execute("""
            SELECT strftime('%Y-%m', prediction_date) as month, COUNT(*) as count
            FROM predictions WHERE user_id=? GROUP BY month ORDER BY month ASC
        """, (user_id,))
        monthly = cursor.fetchall()

        cursor.execute("SELECT COUNT(*) as total FROM predictions WHERE user_id=%s", (user_id,))
        total = cursor.fetchone()['total']
        cursor.execute("SELECT AVG(accuracy)*100 as avg FROM predictions WHERE user_id=%s AND accuracy IS NOT NULL", (user_id,))
        avg_acc = cursor.fetchone()['avg'] or 0

        cursor.close(); connection.close()

        for r in accuracy_trend:
            if r.get('date'):
                r['date'] = str(r['date'])
            if r.get('avg_accuracy') is not None:
                r['avg_accuracy'] = round(float(r['avg_accuracy']), 2)
        for r in top_metrics:
            if r.get('avg_accuracy') is not None:
                r['avg_accuracy'] = round(float(r['avg_accuracy']), 2)

        return jsonify({
            'success': True,
            'total_predictions': total,
            'avg_accuracy': round(float(avg_acc), 2),
            'accuracy_trend': accuracy_trend,
            'top_metrics': top_metrics,
            'model_distribution': model_dist,
            'accuracy_distribution': acc_dist,
            'monthly_predictions': monthly
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


# =============================================================================
# USER: Export predictions (CSV / Excel)
# =============================================================================
@app.route('/api/export-predictions', methods=['GET'])
@login_required
def export_predictions_file():
    """Export user's predictions as CSV or Excel file download."""
    try:
        user_id = get_current_user_id()
        fmt = request.args.get('format', 'csv').lower()

        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'DB error'})
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, dataset_name, predicted_metric, year, predicted_value,
                   accuracy, prediction_type,
                   COALESCE(model_used,'linear_regression') as model_used,
                   prediction_date
            FROM predictions WHERE user_id=%s ORDER BY prediction_date DESC
        """, (user_id,))
        preds = cursor.fetchall()
        cursor.close(); connection.close()

        for p in preds:
            if p.get('prediction_date'):
                p['prediction_date'] = str(p['prediction_date'])
            if p.get('accuracy') is not None:
                p['accuracy'] = round(float(p['accuracy']) * 100, 2)

        df = pd.DataFrame(preds) if preds else pd.DataFrame(
            columns=['id','dataset_name','predicted_metric','year',
                     'predicted_value','accuracy','prediction_type','model_used','prediction_date'])

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        if fmt == 'excel' and EXCEL_AVAILABLE:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Predictions')
            output.seek(0)
            return send_file(output,
                             mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                             as_attachment=True, download_name=f'predictions_{ts}.xlsx')

        output = io.StringIO()
        df.to_csv(output, index=False)
        return send_file(io.BytesIO(output.getvalue().encode()),
                         mimetype='text/csv', as_attachment=True,
                         download_name=f'predictions_{ts}.csv')

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


# =============================================================================
# USER: Favorites
# =============================================================================
@app.route('/api/favorites', methods=['GET'])
@login_required
def get_favorites():
    """Get all favorites for current user."""
    try:
        user_id = get_current_user_id()
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'favorites': []})
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT f.id, f.dataset_name, f.label, f.created_at,
                   d.row_count, d.column_count, d.description
            FROM favorites f
            LEFT JOIN datasets d ON d.user_id=%s AND d.name=f.dataset_name
            WHERE f.user_id=%s ORDER BY f.created_at DESC
        """, (user_id, user_id))
        favs = cursor.fetchall()
        cursor.close(); connection.close()
        for f in favs:
            if f.get('created_at'):
                f['created_at'] = str(f['created_at'])
        return jsonify({'success': True, 'favorites': favs})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'favorites': [], 'message': str(e)})


@app.route('/api/favorites', methods=['POST'])
@login_required
def upsert_favorite():
    """Add or update a favorite label for a dataset."""
    try:
        user_id = get_current_user_id()
        data = request.get_json()
        dataset_name = data.get('dataset_name')
        label = data.get('label', 'favorite')

        if label not in {'favorite', 'primary', 'archived'}:
            return jsonify({'success': False, 'message': 'Invalid label'})
        if not dataset_name:
            return jsonify({'success': False, 'message': 'dataset_name required'})

        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'DB error'})
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM datasets WHERE user_id=%s AND name=%s", (user_id, dataset_name))
        if cursor.fetchone()[0] == 0:
            cursor.close(); connection.close()
            return jsonify({'success': False, 'message': 'Dataset not found'})
        cursor.execute("""
            INSERT INTO favorites (user_id, dataset_name, label)
            VALUES (?,?,?)
            ON CONFLICT(user_id, dataset_name) DO UPDATE SET label=excluded.label
        """, (user_id, dataset_name, label))
        connection.commit()
        cursor.close(); connection.close()
        return jsonify({'success': True, 'message': f'Dataset marked as {label}'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/favorites/<path:dataset_name>', methods=['DELETE'])
@login_required
def remove_favorite(dataset_name):
    """Remove a favorite label."""
    try:
        user_id = get_current_user_id()
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'DB error'})
        cursor = connection.cursor()
        cursor.execute("DELETE FROM favorites WHERE user_id=%s AND dataset_name=%s", (user_id, dataset_name))
        connection.commit()
        cursor.close(); connection.close()
        return jsonify({'success': True, 'message': 'Favorite removed'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


# =============================================================================
# USER: Country / Region Comparison
# =============================================================================
@app.route('/api/compare-regions', methods=['POST'])
@login_required
def compare_regions():
    """Compare two countries/regions on multiple metrics."""
    try:
        data = request.get_json()
        dataset_name = data.get('dataset')
        region_col   = data.get('region_column')
        region_a     = data.get('region_a')
        region_b     = data.get('region_b')
        metrics      = data.get('metrics', [])

        if not all([dataset_name, region_col, region_a, region_b]):
            return jsonify({'success': False, 'message': 'dataset, region_column, region_a, region_b required'})

        rj = get_dataset_data(dataset_name).get_json()
        if not rj or not rj.get('success'):
            return jsonify({'success': False, 'message': 'Failed to load dataset'})

        df = pd.DataFrame(rj.get('data', []))
        if df.empty:
            return jsonify({'success': False, 'message': 'Dataset empty'})

        if 'Country Name' in df.columns and 'Year' not in df.columns:
            year_cols = [c for c in df.columns if c != 'Country Name' and str(c).strip().replace('.','').isdigit()]
            if year_cols:
                df = pd.melt(df, id_vars=['Country Name'], var_name='Year', value_name='Forest Area (%)')
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                df['Forest Area (%)'] = pd.to_numeric(df['Forest Area (%)'], errors='coerce')

        if region_col not in df.columns:
            return jsonify({'success': False, 'message': f'Column "{region_col}" not found'})

        df[region_col] = df[region_col].astype(str)
        df_a = df[df[region_col].str.lower() == region_a.lower()]
        df_b = df[df[region_col].str.lower() == region_b.lower()]

        if df_a.empty:
            return jsonify({'success': False, 'message': f'No data for "{region_a}"'})
        if df_b.empty:
            return jsonify({'success': False, 'message': f'No data for "{region_b}"'})

        for col in df.columns:
            if 'year' in col.lower():
                df_a = df_a.rename(columns={col: 'Year'})
                df_b = df_b.rename(columns={col: 'Year'})
                break

        if not metrics:
            metrics = [c for c in df.columns if c not in [region_col, 'Year']
                       and pd.api.types.is_numeric_dtype(df[c])][:5]

        comparison = {}
        for m in metrics:
            if m not in df_a.columns or m not in df_b.columns:
                continue
            if 'Year' in df_a.columns:
                series_a = pd.to_numeric(df_a[m], errors='coerce')
                series_a.index = pd.to_numeric(df_a['Year'], errors='coerce')
                series_a = series_a.dropna().sort_index()
                series_b = pd.to_numeric(df_b[m], errors='coerce')
                series_b.index = pd.to_numeric(df_b['Year'], errors='coerce')
                series_b = series_b.dropna().sort_index()
            else:
                series_a = pd.to_numeric(df_a[m], errors='coerce').dropna()
                series_b = pd.to_numeric(df_b[m], errors='coerce').dropna()
            comparison[m] = {
                'region_a': {'label': region_a, 'data': [{'year': int(k), 'value': float(v)} for k, v in series_a.items()]},
                'region_b': {'label': region_b, 'data': [{'year': int(k), 'value': float(v)} for k, v in series_b.items()]},
                'summary': {
                    'a_latest': float(series_a.iloc[-1]) if len(series_a) else None,
                    'b_latest': float(series_b.iloc[-1]) if len(series_b) else None,
                    'a_avg': round(float(series_a.mean()), 4) if len(series_a) else None,
                    'b_avg': round(float(series_b.mean()), 4) if len(series_b) else None,
                }
            }

        return jsonify({
            'success': True,
            'region_a': region_a,
            'region_b': region_b,
            'comparison': comparison,
            'metrics': list(comparison.keys())
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


# =============================================================================
# USER: AI Insights Generator
# =============================================================================
@app.route('/api/insights', methods=['POST'])
@login_required
def generate_insights():
    """Generate rule-based AI insights for a prediction result."""
    try:
        data = request.get_json()
        metric   = data.get('metric', 'Unknown metric')
        year     = data.get('year', 2030)
        value    = data.get('value')
        accuracy = data.get('accuracy', 0)
        dataset  = data.get('dataset', 'the dataset')
        model    = data.get('model', 'linear_regression')
        baseline = data.get('baseline_value')

        if value is None:
            return jsonify({'success': False, 'message': 'value required'})

        metric_lower = metric.lower()
        insights = []
        recommendations = []
        risk_level = 'moderate'

        if baseline is not None:
            change = value - baseline
            pct = (change / abs(baseline) * 100) if baseline != 0 else 0
            trend = 'increase' if change > 0 else 'decrease'
            insights.append(f"The predicted value of {value:.4f} for {metric} in {year} represents a "
                            f"{abs(pct):.1f}% {trend} vs the most recent known value ({baseline:.4f}).")
        else:
            insights.append(f"Predicted {metric}: {value:.4f} for {year} based on {dataset}.")

        acc_pct = accuracy * 100 if accuracy <= 1 else accuracy
        if acc_pct >= 85:
            insights.append(f"Model confidence is HIGH ({acc_pct:.1f}%). Prediction is statistically reliable.")
            risk_level = 'low'
        elif acc_pct >= 65:
            insights.append(f"Model confidence is MODERATE ({acc_pct:.1f}%). Use as a general indicator.")
            risk_level = 'moderate'
        else:
            insights.append(f"Model confidence is LOW ({acc_pct:.1f}%). More data or a different model may improve accuracy.")
            risk_level = 'high'

        if 'forest' in metric_lower:
            if value < 10:
                insights.append("Forest coverage below 10% signals critical deforestation. Immediate intervention required.")
                recommendations += ["Implement large-scale afforestation programs.", "Enforce anti-deforestation policies."]
                risk_level = 'high'
            elif value < 25:
                insights.append("Moderate forest coverage. Sustainable practices are essential.")
                recommendations.append("Promote sustainable forestry regulations.")
            else:
                insights.append("Healthy forest coverage. Maintain conservation efforts.")
                recommendations.append("Continue existing conservation and reforestation programs.")
        elif 'co2' in metric_lower or 'emission' in metric_lower:
            if value > 10:
                insights.append("High CO2 emissions predicted. Urgent decarbonization needed.")
                recommendations += ["Transition to renewable energy.", "Implement carbon tax policies."]
                risk_level = 'high'
            elif value > 5:
                insights.append("Moderate CO2 emissions. Reduction measures should be accelerated.")
                recommendations.append("Improve energy efficiency in industry and transport.")
            else:
                insights.append("Relatively low CO2 emissions. Maintain green policies.")
        elif 'temperature' in metric_lower or 'temp' in metric_lower:
            if value > 30:
                insights.append("High temperature predicted — climate stress for ecosystems.")
                recommendations.append("Expand urban green spaces to mitigate heat islands.")
                risk_level = 'high'
        elif 'population' in metric_lower:
            insights.append(f"Population projection of {value:,.0f} for {year}. Infrastructure planning needed.")
            recommendations.append("Ensure sustainable resource management for growing population.")
        elif 'rainfall' in metric_lower or 'precipitation' in metric_lower:
            if value < 500:
                insights.append("Low rainfall predicted — drought risk. Water conservation critical.")
                risk_level = 'high'
            elif value > 2000:
                insights.append("High rainfall predicted — monitor flood risks and soil erosion.")

        model_notes = {
            'random_forest': "Random Forest provides robust predictions by averaging many decision trees.",
            'decision_tree': "Decision Tree is interpretable but may overfit on small datasets.",
            'polynomial_regression': "Polynomial Regression captures non-linear trends.",
            'gradient_boosting': "Gradient Boosting is among the most accurate models for tabular data.",
            'linear_regression': "Linear Regression assumes a linear relationship between year and metric."
        }
        if model in model_notes:
            insights.append(model_notes[model])

        if not recommendations:
            recommendations += ["Continue monitoring and update predictions with new data.",
                                 "Combine multiple ML models for better accuracy."]

        return jsonify({
            'success': True,
            'insights': insights,
            'recommendations': recommendations,
            'risk_level': risk_level,
            'metric': metric,
            'year': year,
            'predicted_value': value,
            'model_used': model
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


# =============================================================================
# USER: AI Chat Assistant
# =============================================================================
@app.route('/api/chat', methods=['POST'])
@login_required
def ai_chat():
    """Rule-based AI chat assistant."""
    try:
        user_id = get_current_user_id()
        data = request.get_json()
        message = (data.get('message') or '').strip()

        if not message:
            return jsonify({'success': False, 'message': 'Message required'})
        if len(message) > 1000:
            return jsonify({'success': False, 'message': 'Message too long (max 1000 chars)'})

        msg_lower = message.lower()

        # Save user message to history
        try:
            connection = get_db_connection()
            if connection:
                cursor = connection.cursor()
                cursor.execute("INSERT INTO chat_history (user_id, role, message) VALUES (%s,'user',%s)",
                               (user_id, message))
                connection.commit()
                cursor.close(); connection.close()
        except Exception:
            pass

        # ── Rich knowledge base ──────────────────────────────────────────
        def _contains(words):
            return any(w in msg_lower for w in words)

        response = None

        # Greetings
        if _contains(['hello','hi','hey','greetings','namaste','hola']):
            response = (
                "👋 Hello! I am your Forest AI Assistant — trained on global deforestation, "
                "climate science, and ML prediction models.\n\n"
                "Ask me anything:\n"
                "• 🌳 Forest area trends, deforestation rates\n"
                "• 🌡️ Climate risk and CO2 impact\n"
                "• 🤖 Which ML model to use\n"
                "• 📊 How to interpret prediction results\n"
                "• 🗺️ Country-specific forest data\n"
                "• 🔮 How to make predictions\n\n"
                "Example: 'What is the deforestation rate in Brazil?' or 'Which model is most accurate?'"
            )

        # Deforestation specific
        elif _contains(['deforest','forest loss','forest area','tree cover','tree loss']):
            if _contains(['brazil','amazon']):
                response = (
                    "🌿 Brazil — Amazon Deforestation:\n"
                    "• Amazon covers ~5.5 million km² — world's largest tropical rainforest\n"
                    "• Brazil lost ~11,000–15,000 km² of forest per year in peak years\n"
                    "• Amazon absorbs ~2 billion tonnes of CO2 annually\n"
                    "• PRODES (Brazil's monitoring) tracks deforestation annually\n"
                    "• Recent years saw 30–40% reduction after policy tightening\n\n"
                    "📈 In this app: Select Brazil data, choose 'Forest Area' metric, run prediction to see future projections!"
                )
            elif _contains(['indonesia','borneo','sumatra']):
                response = (
                    "🌴 Indonesia — Deforestation Hotspot:\n"
                    "• Indonesia has the 3rd largest tropical forest in the world\n"
                    "• Losing ~600,000 hectares/year mainly to palm oil & logging\n"
                    "• Peat forest destruction releases massive CO2 stores\n"
                    "• Kalimantan (Borneo) and Sumatra most affected\n\n"
                    "📊 Use Compare Regions to compare Indonesia vs Brazil forest data!"
                )
            elif _contains(['india']):
                response = (
                    "🌲 India — Forest Cover:\n"
                    "• India has ~24% forest and tree cover of total land\n"
                    "• Net forest area: ~712,000 km² (ISFR 2021)\n"
                    "• India is one of few countries with increasing forest cover (+1,540 km² in 2021)\n"
                    "• Major forest states: Madhya Pradesh, Arunachal Pradesh, Chhattisgarh\n"
                    "• Mangrove area increased by 17 km² in recent assessment\n\n"
                    "📈 Use the prediction tab to forecast India's forest area trends!"
                )
            elif _contains(['africa','congo','drc']):
                response = (
                    "🌍 Congo Basin — Africa's Lungs:\n"
                    "• Congo Basin is the world's 2nd largest tropical forest (~3.3 million km²)\n"
                    "• DRC losing ~500,000 hectares/year — accelerating trend\n"
                    "• Driven by subsistence agriculture, charcoal production, armed conflict\n"
                    "• Also home to unique species like forest elephants and gorillas\n\n"
                    "📊 Compare African countries using the Compare tab with your forest dataset!"
                )
            else:
                response = (
                    "🌍 Global Deforestation Facts:\n"
                    "• World loses ~10 million hectares of forest per year (FAO 2020)\n"
                    "• 420 million hectares lost since 1990\n"
                    "• Top deforestation countries: Brazil, DRC, Indonesia, Bolivia, Angola\n"
                    "• Agriculture drives 73% of deforestation (cattle, soy, palm oil)\n"
                    "• Tropics have highest rates; temperate zones seeing recovery\n"
                    "• Global Forest Watch tracks real-time satellite data\n\n"
                    "📈 Load your dataset and run predictions to model future forest areas!"
                )

        # CO2 / Emissions / Carbon
        elif _contains(['co2','carbon','emission','greenhouse','ghg']):
            response = (
                "🏭 CO2 & Forest Connection:\n"
                "• Forests absorb ~2.6 billion tonnes of CO2/year globally\n"
                "• Deforestation causes ~10–15% of global GHG emissions\n"
                "• 1 hectare of tropical forest stores ~200–500 tonnes of carbon\n"
                "• REDD+ is a UN program paying countries to keep forests standing\n"
                "• Reforestation can offset ~1.7 billion tonnes CO2/year (Nature, 2019)\n"
                "• Mangroves store 4x more carbon per hectare than tropical forests\n\n"
                "📊 In this app: Predict CO2 emission trends using your dataset and Climate Risk tab!"
            )

        # Climate risk / temperature / warming
        elif _contains(['risk','climate risk','warming','temperature','heat','drought']):
            response = (
                "🌡️ Climate Risk & Forests:\n"
                "• 1.5°C warming: ~4% of forests face climate stress\n"
                "• 2°C warming: ~16% of forests in high-risk zones\n"
                "• 4°C warming: ~60% of forests face severe climate mismatch\n"
                "• Forest fires increasing with temperature (2x area burned since 1980s)\n"
                "• Rainfall changes disrupt forest regeneration cycles\n"
                "• Feedback loop: less forest → more CO2 → more warming → more deforestation\n\n"
                "🌡️ Use the Climate Risk tab to calculate your dataset's risk score (0-100)!\n"
                "• Score 0–30: Low Risk  |  31–60: Moderate  |  61–100: High Risk"
            )

        # ML Models
        elif _contains(['model','algorithm','random forest','gradient','decision tree','polynomial','linear regression','xgboost','which model']):
            response = (
                "🤖 ML Models Available in This App:\n\n"
                "1️⃣ Linear Regression\n"
                "   → Best for: Simple, clearly linear trends\n"
                "   → Speed: ⚡⚡⚡ | Accuracy: ⭐⭐\n\n"
                "2️⃣ Polynomial Regression (degree 2-4)\n"
                "   → Best for: Curved, non-linear patterns\n"
                "   → Speed: ⚡⚡⚡ | Accuracy: ⭐⭐⭐\n\n"
                "3️⃣ Random Forest\n"
                "   → Best for: Noisy data, complex patterns\n"
                "   → Speed: ⚡⚡ | Accuracy: ⭐⭐⭐⭐\n\n"
                "4️⃣ Decision Tree\n"
                "   → Best for: Interpretable decisions, small data\n"
                "   → Speed: ⚡⚡⚡ | Accuracy: ⭐⭐⭐ (can overfit)\n\n"
                "5️⃣ Gradient Boosting\n"
                "   → Best for: Highest accuracy, structured data\n"
                "   → Speed: ⚡ | Accuracy: ⭐⭐⭐⭐⭐\n\n"
                "💡 TIP: Use '⚖️ Compare All Models' button to test all 5 automatically and pick the best!"
            )

        # Accuracy / R2 / interpretation
        elif _contains(['accuracy','r2','r-squared','mean squared','mse','mae','interpret','result']):
            response = (
                "📊 Understanding Prediction Accuracy:\n\n"
                "• Accuracy is shown as R² (coefficient of determination)\n"
                "• R² = 1.0 = perfect prediction\n"
                "• R² = 0.0 = no better than mean\n\n"
                "Rating Guide:\n"
                "🟢 ≥ 90% — Excellent (reliable predictions)\n"
                "🔵 70–90% — Good (use with confidence)\n"
                "🟡 50–70% — Fair (indicative only)\n"
                "🔴 < 50% — Poor (try different model or more data)\n\n"
                "💡 Tips to improve accuracy:\n"
                "• Use more years of data (10+ data points ideal)\n"
                "• Try Gradient Boosting for best results\n"
                "• Remove outliers using Dataset Clean tools\n"
                "• Use Polynomial if your data has curves"
            )

        # Prediction steps
        elif _contains(['predict','forecast','estimate','project','how to predict']):
            year_match = re.search(r'\b(20\d{2})\b', message)
            yr = year_match.group(1) if year_match else '2040'
            response = (
                f"🔮 How to Make a Prediction for {yr}:\n\n"
                "Step 1️⃣: Go to the Predictions tab\n"
                "Step 2️⃣: Select your dataset from the dropdown\n"
                "Step 3️⃣: Choose prediction type:\n"
                "   • Column-wise → predict a single metric (e.g. Forest Area)\n"
                "   • Row-wise → predict all metrics for one country/region\n"
                "Step 4️⃣: Select ML model (or use Compare All Models)\n"
                f"Step 5️⃣: Enter target year ({yr})\n"
                "Step 6️⃣: Click Predict 🎯\n\n"
                "After prediction:\n"
                "• 💡 AI Insights popup shows risk level + recommendations\n"
                "• Use Analytics tab to track accuracy trends over time\n"
                "• Export results as CSV/Excel from Analytics tab"
            )

        # Dataset / upload / data management
        elif _contains(['dataset','upload','csv','excel','data','file','clean','normalize']):
            response = (
                "📂 Dataset Management Guide:\n\n"
                "📤 Upload:\n"
                "• Go to Data tab → Upload CSV or Excel\n"
                "• Supported formats: .csv, .xlsx\n"
                "• CSV should have a Year column + numeric metrics\n\n"
                "🧹 Clean Tools (Climate Risk tab):\n"
                "• Remove Nulls — delete rows with missing values\n"
                "• Remove Duplicates — remove duplicate rows\n"
                "• Fill Nulls (Mean/Median) — fill missing with stats\n"
                "• Normalize Columns — scale values to 0–1 range\n\n"
                "⭐ Favorites — Mark datasets as Favorite/Primary/Archived\n"
                "📤 Submit for Approval — Share your dataset publicly via admin review\n"
                "📥 Export — Download predictions as CSV or Excel"
            )

        # Reforestation / solutions
        elif _contains(['reforest','solution','plant','tree plant','restore','restoration','protect']):
            response = (
                "🌱 Reforestation & Forest Protection:\n\n"
                "• Bonn Challenge: restore 350 million hectares by 2030\n"
                "• Ethiopia planted 350 million trees in one day (2019)\n"
                "• India's National Afforestation Programme adds ~5,000 km²/year\n"
                "• Natural regeneration is 40x cheaper than planting\n"
                "• Protected area coverage: 17.5% of land (Aichi Target 11)\n\n"
                "Economic Value:\n"
                "• 1 billion hectares of reforested land = 200 billion tonnes CO2 offset\n"
                "• Forest ecosystem services worth $2.5 trillion/year globally\n"
                "• Ecotourism from forests: $600 billion/year\n\n"
                "📈 Model the impact of reforestation using the Prediction tab!"
            )

        # Biodiversity / species
        elif _contains(['biodiversity','species','wildlife','animal','ecosystem','habitat']):
            response = (
                "🦜 Forest Biodiversity:\n\n"
                "• Forests contain 3/4 of Earth's terrestrial biodiversity\n"
                "• Amazon alone has 40,000+ plant species, 1,300 bird species\n"
                "• Every 1% of forest lost = ~0.5% species extinction risk\n"
                "• Critical biodiversity zones: Amazon, Congo Basin, SE Asia\n"
                "• Half-Earth Project aims to protect 50% of land for nature\n\n"
                "Forest Loss Effects:\n"
                "• Habitat fragmentation is worse than total area loss\n"
                "• Edge effects reduce effective forest quality by 30–50%\n"
                "• Corridor protection essential for species migration\n\n"
                "📊 Use your dataset metrics to model habitat availability changes!"
            )

        # Country comparison
        elif _contains(['compare','comparison','vs','versus','differ','which country']):
            response = (
                "⚖️ How to Use the Compare Feature:\n\n"
                "Step 1️⃣: Click the '⚖️ Compare' tab\n"
                "Step 2️⃣: Select your dataset\n"
                "Step 3️⃣: Choose the Region Column (e.g. Country)\n"
                "Step 4️⃣: Pick Region A and Region B\n"
                "Step 5️⃣: Choose metrics to compare (multi-select)\n"
                "Step 6️⃣: Click Compare!\n\n"
                "You'll get:\n"
                "• Side-by-side latest values and averages\n"
                "• A multi-line chart showing historical trends\n"
                "• Percentage difference analysis\n\n"
                "💡 Best for: Brazil vs Indonesia, India vs China, etc."
            )

        # Global forest facts
        elif _contains(['global','world','worldwide','total forest','earth','planet']):
            response = (
                "🌍 Global Forest Statistics (FAO 2020):\n\n"
                "• Total forest area: 4.06 billion hectares\n"
                "• 31% of Earth's land surface is covered by forests\n"
                "• Russia has largest forest area: 815 million hectares\n"
                "• 5 countries hold 54% of world's forests:\n"
                "  🇷🇺 Russia → 815M ha  |  🇧🇷 Brazil → 497M ha\n"
                "  🇨🇦 Canada → 347M ha  |  🇺🇸 USA → 310M ha\n"
                "  🇨🇳 China → 220M ha\n\n"
                "• Forest loss 2010–2020: 4.7 million ha/year (net)\n"
                "• Tropical forests: most loss  |  Temperate: slight gain\n"
                "• 93% of forests are naturally regenerated"
            )

        # Analytics / visualization
        elif _contains(['analytics','graph','chart','visualiz','trend','history','statistic']):
            response = (
                "📈 Analytics & Visualizations:\n\n"
                "Analytics Tab shows:\n"
                "• Accuracy trend over time (line chart)\n"
                "• Model usage distribution (doughnut chart)\n"
                "• Your top predicted metrics (bar chart)\n"
                "• Accuracy distribution — how good your predictions are\n"
                "• Export all predictions as CSV or Excel\n\n"
                "Visualization Tab shows:\n"
                "• Time-series charts for your dataset\n"
                "• Compare multiple metrics side by side\n"
                "• Scatter plots and trend lines\n"
                "• Map view with Leaflet.js (select 'Map' view)\n\n"
                "💡 Tip: Analytics updates automatically when you switch to the tab!"
            )

        # Favorites
        elif _contains(['favorite','favourite','star','mark','label','archive','primary']):
            response = (
                "⭐ Favorites System:\n\n"
                "Label your datasets for quick access:\n"
                "• ⭐ Favorite — datasets you use regularly\n"
                "• 🎯 Primary — your main working dataset\n"
                "• 📦 Archived — datasets to keep but not actively use\n\n"
                "How to use:\n"
                "1. Go to the Favorites tab\n"
                "2. Select a dataset from the dropdown\n"
                "3. Choose a label\n"
                "4. Click Save Favorite\n\n"
                "Your favorites are saved to your account and persist across sessions."
            )

        # Export
        elif _contains(['export','download','csv','excel','xlsx','save result']):
            response = (
                "📥 Exporting Predictions:\n\n"
                "From the Analytics tab:\n"
                "• 📄 Export CSV — lightweight, opens in Excel/Google Sheets\n"
                "• 📊 Export Excel — formatted .xlsx with all columns\n\n"
                "Exported columns include:\n"
                "• Dataset name, Predicted metric, Year\n"
                "• Predicted value, Accuracy %, Model used\n"
                "• Prediction type (column-wise / row-wise)\n"
                "• Prediction date/time\n\n"
                "📌 Tip: Run multiple predictions first, then export all at once!"
            )

        # About the app
        elif _contains(['about','app','system','what is','what can','feature','capability']):
            response = (
                "🌲 About Forest Prediction System:\n\n"
                "This app predicts forest-related metrics using Machine Learning.\n\n"
                "Key Features:\n"
                "🔮 Multi-model Predictions (5 ML algorithms)\n"
                "⚖️ Country/Region Comparison Tool\n"
                "📈 Prediction Analytics Dashboard\n"
                "⭐ Dataset Favorites System\n"
                "🌡️ Climate Risk Assessment\n"
                "🧹 Dataset Cleaning Tools\n"
                "🗺️ World Map Visualization\n"
                "💬 AI Chat Assistant (you're using it now!)\n"
                "📁 Dataset Approval System\n"
                "🌐 Global Dataset Library\n\n"
                "Data Requirements:\n"
                "• CSV/Excel with Year column + numeric metrics\n"
                "• Min 3 data points for prediction\n"
                "• Wide or long format both supported"
            )

        else:
            # Smarter fallback with keyword hints
            keywords = []
            if any(w in msg_lower for w in ['forest','tree','wood','timber']): keywords.append('🌳 Forest areas')
            if any(w in msg_lower for w in ['water','rain','flood','river']): keywords.append('💧 Watershed forests')
            if any(w in msg_lower for w in ['fire','wildfire','burn']): keywords.append('🔥 Forest fires & climate')
            if any(w in msg_lower for w in ['policy','law','government','regulation']): keywords.append('📜 Forest conservation policy')

            hint = ('\n\nBased on your question, you might want to ask about:\n• ' + '\n• '.join(keywords)) if keywords else ''
            response = (
                f"🤔 I couldn't find a direct answer for: \"{message}\"\n\n"
                "I'm trained on forest & climate topics. Try asking:\n"
                "• 'What is the deforestation rate in Brazil?'\n"
                "• 'Which ML model is most accurate?'\n"
                "• 'How do I predict forest area in 2040?'\n"
                "• 'What are the global forest statistics?'\n"
                "• 'How do I compare two countries?'\n"
                "• 'What is climate risk score?'\n"
                "• 'How do I clean my dataset?'"
                + hint
            )

        connection = get_db_connection()
        if connection:
            cursor = connection.cursor()
            cursor.execute("INSERT INTO chat_history (user_id, role, message) VALUES (%s,'assistant',%s)",
                           (user_id, response))
            connection.commit()
            cursor.close(); connection.close()

        return jsonify({'success': True, 'response': response})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/chat/history', methods=['GET'])
@login_required
def get_chat_history():
    """Get last 50 chat messages for current user."""
    try:
        user_id = get_current_user_id()
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'history': []})
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT role, message, created_at FROM chat_history
            WHERE user_id=%s ORDER BY created_at DESC LIMIT 50
        """, (user_id,))
        rows = cursor.fetchall()
        cursor.close(); connection.close()
        for r in rows:
            if r.get('created_at'):
                r['created_at'] = str(r['created_at'])
        rows.reverse()
        return jsonify({'success': True, 'history': rows})
    except Exception as e:
        return jsonify({'success': False, 'history': [], 'message': str(e)})


@app.route('/api/chat/clear', methods=['POST'])
@login_required
def clear_chat():
    """Clear chat history for current user."""
    try:
        user_id = get_current_user_id()
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'DB error'})
        cursor = connection.cursor()
        cursor.execute("DELETE FROM chat_history WHERE user_id=%s", (user_id,))
        connection.commit()
        cursor.close(); connection.close()
        return jsonify({'success': True, 'message': 'Chat history cleared'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


# =============================================================================
# USER: Climate Risk Prediction
# =============================================================================
@app.route('/api/climate-risk', methods=['POST'])
@login_required
def climate_risk():
    """Assess climate/deforestation risk based on dataset trends."""
    try:
        data = request.get_json()
        dataset_name  = data.get('dataset')
        filter_column = data.get('filter_column')
        filter_value  = data.get('filter_value')
        horizon_year  = int(data.get('horizon_year', 2050))

        rj = get_dataset_data(dataset_name).get_json()
        if not rj or not rj.get('success'):
            return jsonify({'success': False, 'message': 'Failed to load dataset'})

        df = pd.DataFrame(rj.get('data', []))
        if df.empty:
            return jsonify({'success': False, 'message': 'Dataset empty'})

        if 'Country Name' in df.columns and 'Year' not in df.columns:
            year_cols = [c for c in df.columns if c != 'Country Name' and str(c).strip().replace('.','').isdigit()]
            if year_cols:
                df = pd.melt(df, id_vars=['Country Name'], var_name='Year', value_name='Forest Area (%)')
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                df['Forest Area (%)'] = pd.to_numeric(df['Forest Area (%)'], errors='coerce')

        for col in df.columns:
            if 'year' in col.lower():
                df = df.rename(columns={col: 'Year'})
                break
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

        if filter_column and filter_value and filter_column in df.columns:
            df[filter_column] = df[filter_column].astype(str)
            df = df[df[filter_column].str.lower() == filter_value.lower()]

        risk_score = 50
        risk_metrics = {}

        for col in [c for c in df.columns if c not in ['Year', filter_column]]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            clean = df[['Year', col]].dropna()
            if len(clean) < 4:
                continue
            X = clean['Year'].values.reshape(-1, 1)
            y = clean[col].values
            model, acc, _, _ = _fit_and_score(X, y, 'linear_regression')
            current_val = float(clean[col].iloc[-1])
            future_val  = float(model.predict([[horizon_year]])[0])
            pct_change  = ((future_val - current_val) / abs(current_val) * 100) if current_val != 0 else 0
            risk_metrics[col] = {
                'current': round(current_val, 4),
                'predicted': round(future_val, 4),
                'pct_change': round(pct_change, 2),
                'accuracy': round(acc * 100, 2)
            }
            col_l = col.lower()
            if 'forest' in col_l and pct_change < -20:
                risk_score = min(risk_score + 25, 100)
            elif ('co2' in col_l or 'emission' in col_l) and pct_change > 20:
                risk_score = min(risk_score + 20, 100)
            elif 'temp' in col_l and pct_change > 5:
                risk_score = min(risk_score + 15, 100)

        risk_level = 'Low' if risk_score < 35 else ('Moderate' if risk_score < 65 else 'High')
        risk_color = ('#06d6a0' if risk_level == 'Low' else
                      '#ff9e00' if risk_level == 'Moderate' else '#ef476f')

        return jsonify({
            'success': True,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'horizon_year': horizon_year,
            'metrics': risk_metrics
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


# =============================================================================
# USER: World Map Data
# =============================================================================
@app.route('/api/map-data', methods=['GET'])
@login_required
def get_map_data():
    """Return country-level data for world map visualization."""
    try:
        user_id = get_current_user_id()
        year = int(request.args.get('year', 2020))

        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'data': []})
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM datasets WHERE user_id=%s AND is_primary=TRUE", (user_id,))
        row = cursor.fetchone()
        cursor.close(); connection.close()

        dataset_name = row[0] if row else 'forest_data'
        rj = get_dataset_data(dataset_name).get_json()
        if not rj or not rj.get('success'):
            return jsonify({'success': False, 'data': []})

        df = pd.DataFrame(rj.get('data', []))
        if df.empty:
            return jsonify({'success': False, 'data': []})

        if 'Country Name' in df.columns and 'Year' not in df.columns:
            year_cols = [c for c in df.columns if c != 'Country Name' and str(c).strip().replace('.','').isdigit()]
            year_str = next((c for c in year_cols if str(c).strip() == str(year)), None)
            if year_str:
                result = df[['Country Name', year_str]].dropna()
                result = result.rename(columns={year_str: 'value', 'Country Name': 'country'})
                result['value'] = pd.to_numeric(result['value'], errors='coerce')
                result = result.dropna()
                return jsonify({'success': True, 'data': result.to_dict('records'), 'year': year})

        for col in df.columns:
            if 'year' in col.lower():
                df = df.rename(columns={col: 'Year'})
                break
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        country_col = next((c for c in df.columns if any(x in c.lower() for x in ['country','nation','region'])), None)
        value_col   = next((c for c in df.columns if 'forest' in c.lower()), None)
        if not country_col or not value_col:
            return jsonify({'success': True, 'data': []})
        year_df = df[df['Year'] == year][[country_col, value_col]].dropna()
        year_df = year_df.rename(columns={country_col: 'country', value_col: 'value'})
        return jsonify({'success': True, 'data': year_df.to_dict('records'), 'year': year})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'data': [], 'message': str(e)})


# =============================================================================
# USER: Global Dataset Library
# =============================================================================
@app.route('/api/global-datasets', methods=['GET'])
@login_required
def get_global_datasets_for_user():
    """Return list of global datasets visible to users."""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'datasets': []})
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, name, row_count, column_count, description, tags, created_at
            FROM global_datasets WHERE is_active=TRUE ORDER BY created_at DESC
        """)
        datasets = cursor.fetchall()
        cursor.close(); connection.close()
        for d in datasets:
            if d.get('created_at'):
                d['created_at'] = str(d['created_at'])
        return jsonify({'success': True, 'datasets': datasets})
    except Exception as e:
        return jsonify({'success': False, 'datasets': [], 'message': str(e)})


# =============================================================================
# USER: Submit Dataset for Approval
# =============================================================================
@app.route('/api/submit-for-approval', methods=['POST'])
@login_required
def submit_for_approval():
    """User submits a dataset for admin approval."""
    try:
        user_id = get_current_user_id()
        data = request.get_json()
        dataset_name = data.get('dataset_name')
        if not dataset_name:
            return jsonify({'success': False, 'message': 'dataset_name required'})

        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'DB error'})
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM datasets WHERE user_id=%s AND name=%s", (user_id, dataset_name))
        if cursor.fetchone()[0] == 0:
            cursor.close(); connection.close()
            return jsonify({'success': False, 'message': 'Dataset not found'})
        cursor.execute(
            "SELECT id FROM dataset_approvals WHERE dataset_user_id=%s AND dataset_name=%s AND status='pending'",
            (user_id, dataset_name))
        if cursor.fetchone():
            cursor.close(); connection.close()
            return jsonify({'success': False, 'message': 'Already submitted for approval'})
        cursor.execute("INSERT INTO dataset_approvals (dataset_user_id, dataset_name) VALUES (%s,%s)",
                       (user_id, dataset_name))
        connection.commit()
        cursor.close(); connection.close()
        return jsonify({'success': True, 'message': f'Dataset "{dataset_name}" submitted for approval'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


# =============================================================================
# ADMIN: Global Dataset Library
# =============================================================================
@app.route('/api/admin/global-datasets', methods=['GET'])
@admin_required
def list_global_datasets():
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'datasets': []})
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT g.*, u.name as uploader_name
            FROM global_datasets g LEFT JOIN users u ON g.uploaded_by=u.id
            ORDER BY g.created_at DESC
        """)
        datasets = cursor.fetchall()
        cursor.close(); connection.close()
        for d in datasets:
            if d.get('created_at'):
                d['created_at'] = str(d['created_at'])
        return jsonify({'success': True, 'datasets': datasets})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'datasets': [], 'message': str(e)})


@app.route('/api/admin/global-datasets', methods=['POST'])
@admin_required
def upload_global_dataset():
    try:
        admin_id = get_current_user_id()
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file provided'})
        file = request.files['file']
        description = request.form.get('description', '')
        tags = request.form.get('tags', '')

        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            return jsonify({'success': False, 'message': 'Only CSV/Excel supported'})

        file_bytes = file.read()
        try:
            df = pd.read_csv(io.BytesIO(file_bytes)) if file.filename.endswith('.csv') else pd.read_excel(io.BytesIO(file_bytes))
        except Exception as e:
            return jsonify({'success': False, 'message': f'Cannot read file: {e}'})

        base_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', os.path.splitext(file.filename)[0])[:50]
        upload_folder = 'uploads/global'
        os.makedirs(upload_folder, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        ext = os.path.splitext(file.filename)[1]
        file_path = os.path.join(upload_folder, f"{base_name}_{ts}{ext}")
        with open(file_path, 'wb') as f:
            f.write(file_bytes)

        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'DB error'})
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO global_datasets (name, file_path, row_count, column_count, description, tags, uploaded_by)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
        """, (base_name, file_path, len(df), len(df.columns), description, tags, admin_id))
        connection.commit()
        cursor.close(); connection.close()
        return jsonify({'success': True, 'message': f'Global dataset "{base_name}" uploaded', 'rows': len(df)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/admin/global-datasets/<int:ds_id>', methods=['DELETE'])
@admin_required
def delete_global_dataset(ds_id):
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'DB error'})
        cursor = connection.cursor()
        cursor.execute("SELECT file_path FROM global_datasets WHERE id=%s", (ds_id,))
        row = cursor.fetchone()
        if row and row[0] and os.path.exists(row[0]):
            try:
                os.remove(row[0])
            except Exception:
                pass
        cursor.execute("DELETE FROM global_datasets WHERE id=%s", (ds_id,))
        connection.commit()
        cursor.close(); connection.close()
        return jsonify({'success': True, 'message': 'Global dataset deleted'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


# =============================================================================
# ADMIN: Dataset Approval System
# =============================================================================
@app.route('/api/admin/dataset-approvals', methods=['GET'])
@admin_required
def list_dataset_approvals():
    try:
        status_filter = request.args.get('status', 'pending')
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'items': []})
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT da.*, u.name as user_name, u.email as user_email,
                   d.row_count, d.column_count, d.description
            FROM dataset_approvals da
            LEFT JOIN users u ON da.dataset_user_id=u.id
            LEFT JOIN datasets d ON d.user_id=da.dataset_user_id AND d.name=da.dataset_name
            WHERE da.status=%s ORDER BY da.created_at DESC
        """, (status_filter,))
        items = cursor.fetchall()
        cursor.close(); connection.close()
        for i in items:
            for k in ['reviewed_at','created_at']:
                if i.get(k):
                    i[k] = str(i[k])
        return jsonify({'success': True, 'items': items})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'items': [], 'message': str(e)})


@app.route('/api/admin/dataset-approvals/<int:approval_id>/preview', methods=['GET'])
@admin_required
def preview_approval_dataset(approval_id):
    """Return first 100 rows of the pending dataset so admin can review data."""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'DB error'})
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT dataset_user_id, dataset_name FROM dataset_approvals WHERE id=%s",
            (approval_id,)
        )
        row = cursor.fetchone()
        cursor.close(); connection.close()
        if not row:
            return jsonify({'success': False, 'message': 'Approval not found'})

        file_path = os.path.join(f'uploads/user_{row["dataset_user_id"]}',
                                 f'{row["dataset_name"]}.csv')
        # Try common extensions
        for ext in ['.csv', '.xlsx', '.xls', '']:
            candidate = os.path.normpath(
                os.path.join(f'uploads/user_{row["dataset_user_id"]}',
                             f'{row["dataset_name"]}{ext}'))
            if os.path.exists(candidate):
                file_path = candidate
                break

        if not os.path.exists(file_path):
            # Try finding by DB file_path column
            connection2 = get_db_connection()
            if connection2:
                c2 = connection2.cursor(dictionary=True)
                c2.execute("SELECT file_path FROM datasets WHERE user_id=%s AND name=%s",
                           (row['dataset_user_id'], row['dataset_name']))
                db_row = c2.fetchone()
                c2.close(); connection2.close()
                if db_row and db_row.get('file_path') and os.path.exists(db_row['file_path']):
                    file_path = db_row['file_path']
                else:
                    return jsonify({'success': False, 'message': 'Dataset file not found on server'})

        df = pd.read_csv(file_path) if str(file_path).endswith('.csv') else pd.read_excel(file_path)
        full_len = len(df)
        df_preview = df.head(100).copy()

        # Replace ALL NaN/Inf values with None so JSON serialisation never fails
        import math
        def safe_val(v):
            if v is None:
                return None
            try:
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    return None
            except Exception:
                pass
            if hasattr(v, 'item'):          # numpy scalar → python native
                v = v.item()
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    return None
            return v

        records = [
            {col: safe_val(row[col]) for col in df_preview.columns}
            for row in df_preview.to_dict(orient='records')
        ]

        # Null counts — convert numpy int64 → int
        null_counts = {col: int(df_preview[col].isnull().sum()) for col in df_preview.columns}

        stats = {
            'rows': int(full_len),
            'cols': int(len(df_preview.columns)),
            'columns': list(df_preview.columns),
            'null_counts': null_counts,
            'dtypes': {c: str(df_preview[c].dtype) for c in df_preview.columns},
        }
        return jsonify({'success': True,
                        'dataset_name': row['dataset_name'],
                        'preview': records,
                        'stats': stats})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/admin/dataset-approvals/<int:approval_id>/analyze', methods=['POST'])
@admin_required
def analyze_approval_dataset(approval_id):
    """
    Analyze dataset for originality:
    - Duplicate row detection
    - Column pattern analysis
    - Checks against known public dataset signatures (column name patterns)
    - Returns AI-style originality score
    """
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'DB error'})
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT dataset_user_id, dataset_name FROM dataset_approvals WHERE id=%s",
            (approval_id,)
        )
        row = cursor.fetchone()
        cursor.close(); connection.close()
        if not row:
            return jsonify({'success': False, 'message': 'Approval not found'})

        # Locate file
        file_path = None
        connection2 = get_db_connection()
        if connection2:
            c2 = connection2.cursor(dictionary=True)
            c2.execute("SELECT file_path FROM datasets WHERE user_id=%s AND name=%s",
                       (row['dataset_user_id'], row['dataset_name']))
            db_row = c2.fetchone()
            c2.close(); connection2.close()
            if db_row and db_row.get('file_path') and os.path.exists(db_row['file_path']):
                file_path = db_row['file_path']

        if not file_path:
            for ext in ['.csv', '.xlsx', '.xls']:
                candidate = os.path.normpath(
                    os.path.join(f'uploads/user_{row["dataset_user_id"]}',
                                 f'{row["dataset_name"]}{ext}'))
                if os.path.exists(candidate):
                    file_path = candidate
                    break

        if not file_path:
            return jsonify({'success': False, 'message': 'Dataset file not found'})

        df = pd.read_csv(file_path) if str(file_path).endswith('.csv') else pd.read_excel(file_path)
        total_rows = len(df)

        # ── 1. Duplicate row analysis ──
        dup_rows = int(df.duplicated().sum())
        dup_pct  = round(dup_rows / total_rows * 100, 1) if total_rows else 0

        # ── 2. Null/missing data ──
        null_total = int(df.isnull().sum().sum())
        null_pct   = round(null_total / (total_rows * len(df.columns)) * 100, 1) if total_rows else 0

        # ── 3. Known public dataset column fingerprints ──
        # Compare column name sets against well-known open datasets
        known_fingerprints = {
            'Kaggle - Titanic':       {'survived','pclass','name','sex','age','sibsp','parch','ticket','fare','cabin','embarked'},
            'Kaggle - Iris':          {'sepal_length','sepal_width','petal_length','petal_width','species'},
            'Kaggle - House Prices':  {'mssubclass','lotfrontage','lotarea','street','alley','lotshape','saletype','salecondition','saleprice'},
            'World Bank - Forest':    {'country_name','country_code','indicator_name','indicator_code'},
            'Kaggle - CO2 Emissions': {'country','year','co2','methane','nitrous_oxide','total_ghg'},
            'UN FAO Forestry':        {'area','item','element','unit','year','value'},
            'Global Forest Watch':    {'iso','country','threshold','area_ha','extent_2000_ha','gain_2000_2020_ha'},
            'NASA GISS Climate':      {'year','jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec','j_d','d_n','djf','mam','jja','son'},
        }
        user_cols = set(c.lower().strip().replace(' ','_') for c in df.columns)
        matched_sources = []
        for source, fingerprint in known_fingerprints.items():
            overlap = len(user_cols & fingerprint)
            if overlap >= max(2, len(fingerprint) * 0.5):
                match_pct = round(overlap / len(fingerprint) * 100, 0)
                matched_sources.append({'source': source, 'match_pct': int(match_pct), 'matched_cols': list(user_cols & fingerprint)})

        # ── 4. Internal duplicate check vs other datasets in DB ──
        similar_internal = []
        try:
            conn3 = get_db_connection()
            if conn3:
                c3 = conn3.cursor(dictionary=True)
                c3.execute("""
                    SELECT dc.dataset_name, GROUP_CONCAT(dc.column_name) as cols, d.user_id
                    FROM dataset_columns dc
                    LEFT JOIN datasets d ON d.name=dc.dataset_name AND d.user_id=dc.user_id
                    WHERE dc.user_id != %s
                    GROUP BY dc.dataset_name, d.user_id
                """, (row['dataset_user_id'],))
                all_other = c3.fetchall()
                c3.close(); conn3.close()
                for other in all_other:
                    other_cols = set(c.lower().strip() for c in (other['cols'] or '').split(','))
                    overlap = len(user_cols & other_cols)
                    if overlap > 0 and len(other_cols) > 0:
                        sim_pct = round(overlap / max(len(user_cols), len(other_cols)) * 100, 0)
                        if sim_pct >= 60:
                            similar_internal.append({
                                'dataset': other['dataset_name'],
                                'similarity_pct': int(sim_pct)
                            })
        except Exception:
            pass

        # ── 5. Compute originality score ──
        score = 100
        score -= min(40, dup_pct * 0.8)          # penalise duplicates
        score -= min(20, null_pct * 0.5)          # penalise heavy nulls
        if matched_sources:
            best_match = max(s['match_pct'] for s in matched_sources)
            score -= min(30, best_match * 0.3)    # penalise column fingerprint hits
        if similar_internal:
            score -= min(15, len(similar_internal) * 5)
        score = max(0, round(score, 1))

        # ── 6. Verdict ──
        if score >= 80:
            verdict = 'Likely Original'
            verdict_color = '#06d6a0'
        elif score >= 55:
            verdict = 'Possibly Modified / Derived'
            verdict_color = '#ff9e00'
        else:
            verdict = 'Potentially Duplicated / Copied'
            verdict_color = '#ef476f'

        return jsonify({
            'success': True,
            'dataset_name': row['dataset_name'],
            'total_rows': total_rows,
            'total_cols': len(df.columns),
            'duplicate_rows': dup_rows,
            'duplicate_pct': dup_pct,
            'null_pct': null_pct,
            'originality_score': score,
            'verdict': verdict,
            'verdict_color': verdict_color,
            'matched_sources': matched_sources,
            'similar_internal': similar_internal,
            'columns': list(df.columns),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/admin/dataset-approvals/<int:approval_id>/review', methods=['POST'])
@admin_required
def review_dataset(approval_id):
    try:
        admin_id = get_current_user_id()
        data = request.get_json()
        action = data.get('action')
        note   = data.get('note', '')

        if action not in ('approve', 'reject'):
            return jsonify({'success': False, 'message': 'action must be approve or reject'})

        status = 'approved' if action == 'approve' else 'rejected'
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'DB error'})
        cursor = connection.cursor(dictionary=True)
        # Get dataset owner info before updating
        cursor.execute("SELECT dataset_user_id, dataset_name FROM dataset_approvals WHERE id=%s", (approval_id,))
        row = cursor.fetchone()
        cursor.execute("""
            UPDATE dataset_approvals
            SET status=%s, admin_note=%s, reviewed_by=%s, reviewed_at=NOW()
            WHERE id=%s
        """, (status, note, admin_id, approval_id))
        # Send notification to dataset owner
        if row:
            uid = row['dataset_user_id']
            dname = row['dataset_name']
            if status == 'approved':
                ntitle = '✅ Dataset Approved!'
                nmsg = f'Your dataset "{dname}" has been approved by admin and is now publicly available!'
            else:
                ntitle = '❌ Dataset Rejected'
                nmsg = f'Your dataset "{dname}" was rejected.' + (f' Reason: {note}' if note else '')
            cursor.execute("INSERT INTO notifications (user_id, type, title, message) VALUES (%s,%s,%s,%s)",
                           (uid, status, ntitle, nmsg))
        connection.commit()
        cursor.close(); connection.close()
        log_user_activity(admin_id, f'dataset_{status}',
                          f'Approval id={approval_id} {status}',
                          request.remote_addr, request.user_agent.string)
        return jsonify({'success': True, 'message': f'Dataset {status}'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


# =============================================================================
# ADMIN: User Role Management
# =============================================================================
@app.route('/api/admin/user/<int:user_id>/role', methods=['POST'])
@admin_required
def update_user_role(user_id):
    try:
        data = request.get_json()
        new_role = data.get('role')
        if new_role not in {'user', 'researcher', 'moderator', 'admin'}:
            return jsonify({'success': False, 'message': 'Invalid role'})
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'DB error'})
        cursor = connection.cursor()
        cursor.execute("UPDATE users SET role=%s WHERE id=%s", (new_role, user_id))
        connection.commit()
        cursor.close(); connection.close()
        log_user_activity(get_current_user_id(), 'role_change',
                          f'Changed user {user_id} role to {new_role}',
                          request.remote_addr, request.user_agent.string)
        return jsonify({'success': True, 'message': f'User role updated to {new_role}'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


# =============================================================================
# ADMIN: Advanced Analytics (system-wide)
# =============================================================================
@app.route('/api/admin/advanced-analytics', methods=['GET'])
@admin_required
def admin_advanced_analytics():
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'DB error'})
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT predicted_metric as metric, COUNT(*) as count,
                   AVG(accuracy)*100 as avg_accuracy
            FROM predictions GROUP BY predicted_metric
            ORDER BY count DESC LIMIT 10
        """)
        top_metrics = cursor.fetchall()

        cursor.execute("""
            SELECT COALESCE(model_used,'linear_regression') as model, COUNT(*) as count
            FROM predictions GROUP BY model_used ORDER BY count DESC
        """)
        model_usage = cursor.fetchall()

        cursor.execute("""
            SELECT DATE(created_at) as date, COUNT(*) as count, activity_type
            FROM user_activity
            WHERE created_at >= datetime('now', '-30 days')
            GROUP BY DATE(created_at), activity_type ORDER BY date ASC
        """)
        daily_activity = cursor.fetchall()

        cursor.execute("""
            SELECT strftime('%Y-%W', created_at) as week, COUNT(*) as new_users
            FROM users WHERE role != 'admin'
            GROUP BY week ORDER BY week DESC LIMIT 12
        """)
        new_users_weekly = cursor.fetchall()

        cursor.execute("""
            SELECT
                SUM(CASE WHEN accuracy >= 0.9  THEN 1 ELSE 0 END) as excellent,
                SUM(CASE WHEN accuracy >= 0.7 AND accuracy < 0.9 THEN 1 ELSE 0 END) as good,
                SUM(CASE WHEN accuracy >= 0.5 AND accuracy < 0.7 THEN 1 ELSE 0 END) as fair,
                SUM(CASE WHEN accuracy < 0.5  OR accuracy IS NULL  THEN 1 ELSE 0 END) as poor
            FROM predictions
        """)
        acc_dist = cursor.fetchone()

        cursor.execute("""
            SELECT u.name, u.email, COUNT(d.id) as dataset_count
            FROM users u LEFT JOIN datasets d ON u.id=d.user_id
            WHERE u.role != 'admin' GROUP BY u.id
            ORDER BY dataset_count DESC LIMIT 5
        """)
        top_uploaders = cursor.fetchall()

        cursor.close(); connection.close()

        for r in top_metrics:
            if r.get('avg_accuracy') is not None:
                r['avg_accuracy'] = round(float(r['avg_accuracy']), 2)
        for r in daily_activity:
            if r.get('date'):
                r['date'] = str(r['date'])

        return jsonify({
            'success': True,
            'top_metrics': top_metrics,
            'model_usage': model_usage,
            'daily_activity': daily_activity,
            'new_users_weekly': new_users_weekly,
            'accuracy_distribution': acc_dist,
            'top_uploaders': top_uploaders
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


# =============================================================================
# NOTIFICATIONS
# =============================================================================
@app.route('/api/notifications', methods=['GET'])
@login_required
def get_notifications():
    try:
        user_id = get_current_user_id()
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'notifications': [], 'unread': 0})
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, type, title, message, is_read, created_at
            FROM notifications WHERE user_id=%s
            ORDER BY created_at DESC LIMIT 30
        """, (user_id,))
        notifs = cursor.fetchall()
        cursor.close(); connection.close()
        for n in notifs:
            if n.get('created_at'):
                n['created_at'] = str(n['created_at'])
        unread = sum(1 for n in notifs if not n['is_read'])
        return jsonify({'success': True, 'notifications': notifs, 'unread': unread})
    except Exception as e:
        return jsonify({'success': False, 'notifications': [], 'unread': 0})


@app.route('/api/notifications/mark-read', methods=['POST'])
@login_required
def mark_notifications_read():
    try:
        user_id = get_current_user_id()
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False})
        cursor = connection.cursor()
        cursor.execute("UPDATE notifications SET is_read=TRUE WHERE user_id=%s", (user_id,))
        connection.commit()
        cursor.close(); connection.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False})


# =============================================================================
# SUPPORT QUERIES
# =============================================================================
@app.route('/api/support/submit', methods=['POST'])
@login_required
def submit_support():
    try:
        user_id = get_current_user_id()
        issue_type = request.form.get('issue_type', 'General')
        description = request.form.get('description', '')
        screenshot_path = None
        if 'screenshot' in request.files:
            f = request.files['screenshot']
            if f and f.filename:
                ext = os.path.splitext(f.filename)[1].lower()
                if ext not in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                    return jsonify({'success': False, 'message': 'Invalid image format'})
                os.makedirs('uploads/support', exist_ok=True)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                fname = f"support_{user_id}_{ts}{ext}"
                fpath = os.path.join('uploads/support', fname)
                f.save(fpath)
                screenshot_path = fpath
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'DB error'})
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO support_queries (user_id, issue_type, description, screenshot_path)
            VALUES (%s,%s,%s,%s)
        """, (user_id, issue_type, description, screenshot_path))
        connection.commit()
        cursor.close(); connection.close()
        return jsonify({'success': True, 'message': 'Support query submitted successfully!'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/support/screenshot/<int:query_id>', methods=['GET'])
@login_required
def get_support_screenshot(query_id):
    try:
        user_id = get_current_user_id()
        role = session.get('role', 'user')
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False}), 404
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT user_id, screenshot_path FROM support_queries WHERE id=%s", (query_id,))
        row = cursor.fetchone()
        cursor.close(); connection.close()
        if not row:
            return jsonify({'success': False, 'message': 'Not found'}), 404
        if role != 'admin' and row['user_id'] != user_id:
            return jsonify({'success': False, 'message': 'Forbidden'}), 403
        if not row['screenshot_path'] or not os.path.exists(row['screenshot_path']):
            return jsonify({'success': False, 'message': 'No screenshot'}), 404
        return send_file(row['screenshot_path'])
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/admin/support/queries', methods=['GET'])
@admin_required
def admin_get_support_queries():
    try:
        status_filter = request.args.get('status', 'open')
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'queries': []})
        cursor = connection.cursor(dictionary=True)
        if status_filter == 'all':
            cursor.execute("""
                SELECT sq.*, u.name as user_name, u.email as user_email
                FROM support_queries sq LEFT JOIN users u ON sq.user_id=u.id
                ORDER BY sq.created_at DESC
            """)
        else:
            cursor.execute("""
                SELECT sq.*, u.name as user_name, u.email as user_email
                FROM support_queries sq LEFT JOIN users u ON sq.user_id=u.id
                WHERE sq.status=%s ORDER BY sq.created_at DESC
            """, (status_filter,))
        queries = cursor.fetchall()
        cursor.close(); connection.close()
        for q in queries:
            for k in ['resolved_at', 'created_at']:
                if q.get(k):
                    q[k] = str(q[k])
            q['has_screenshot'] = bool(
                q.get('screenshot_path') and os.path.exists(q.get('screenshot_path', '')))
        return jsonify({'success': True, 'queries': queries})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'queries': [], 'message': str(e)})


@app.route('/api/admin/support/queries/<int:query_id>/resolve', methods=['POST'])
@admin_required
def resolve_support_query(query_id):
    try:
        admin_id = get_current_user_id()
        data = request.get_json()
        response_msg = data.get('response', 'Your issue has been resolved.')
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'DB error'})
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT user_id, issue_type FROM support_queries WHERE id=%s", (query_id,))
        row = cursor.fetchone()
        if not row:
            cursor.close(); connection.close()
            return jsonify({'success': False, 'message': 'Query not found'})
        cursor.execute("""
            UPDATE support_queries SET status='resolved', admin_response=%s, resolved_by=%s, resolved_at=NOW()
            WHERE id=%s
        """, (response_msg, admin_id, query_id))
        cursor.execute("""
            INSERT INTO notifications (user_id, type, title, message)
            VALUES (%s,'support','✅ Support Query Resolved',%s)
        """, (row['user_id'],
               f'Your support query ({row["issue_type"]}) has been resolved. Admin response: {response_msg}'))
        connection.commit()
        cursor.close(); connection.close()
        return jsonify({'success': True, 'message': 'Query resolved and user notified'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/user/my-support-queries', methods=['GET'])
@login_required
def user_my_support_queries():
    try:
        user_id = get_current_user_id()
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': True, 'queries': []})
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, issue_type, description, status, admin_response, created_at
            FROM support_queries WHERE user_id=%s ORDER BY created_at DESC
        """, (user_id,))
        queries = cursor.fetchall()
        cursor.close(); connection.close()
        for q in queries:
            if q.get('created_at'):
                q['created_at'] = str(q['created_at'])
        return jsonify({'success': True, 'queries': queries})
    except Exception as e:
        return jsonify({'success': True, 'queries': []})


# =============================================================================
# GLOBAL LIBRARY USER TOGGLE
# =============================================================================
@app.route('/api/user/global-library/status', methods=['GET'])
@login_required
def get_global_library_status():
    try:
        user_id = get_current_user_id()
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': True, 'enabled': True})
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT global_library_enabled FROM users WHERE id=%s", (user_id,))
        row = cursor.fetchone()
        cursor.close(); connection.close()
        enabled = bool(row['global_library_enabled']) if row else True
        return jsonify({'success': True, 'enabled': enabled})
    except Exception as e:
        return jsonify({'success': True, 'enabled': True})


@app.route('/api/user/global-library/toggle', methods=['POST'])
@login_required
def toggle_global_library():
    try:
        user_id = get_current_user_id()
        data = request.get_json()
        enabled = bool(data.get('enabled', True))
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'DB error'})
        cursor = connection.cursor()
        cursor.execute("UPDATE users SET global_library_enabled=%s WHERE id=%s", (enabled, user_id))
        connection.commit()
        cursor.close(); connection.close()
        return jsonify({'success': True, 'enabled': enabled,
                        'message': f'Global library {"enabled" if enabled else "disabled"}'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/user/global-datasets', methods=['GET'])
@login_required
def user_global_datasets():
    try:
        user_id = get_current_user_id()
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': True, 'datasets': [], 'enabled': False})
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT global_library_enabled FROM users WHERE id=%s", (user_id,))
        row = cursor.fetchone()
        enabled = bool(row['global_library_enabled']) if row else True
        if not enabled:
            cursor.close(); connection.close()
            return jsonify({'success': True, 'datasets': [], 'enabled': False})
        cursor.execute("""
            SELECT id, name, row_count, column_count, description, tags, created_at
            FROM global_datasets WHERE is_active=TRUE ORDER BY created_at DESC
        """)
        datasets = cursor.fetchall()
        cursor.close(); connection.close()
        for d in datasets:
            if d.get('created_at'):
                d['created_at'] = str(d['created_at'])
        return jsonify({'success': True, 'datasets': datasets, 'enabled': True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': True, 'datasets': [], 'enabled': False})


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    # Create necessary directories
    for folder in ['uploads', 'templates', 'templates/auth']:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    print("🚀 Starting Forest Data Analysis & Prediction System...")
    print(f"📁 Forest data file: {'✅ Found' if os.path.exists('forest_data.csv') else '❌ Not found'}")
    print("🌐 Server running at: http://localhost:5000")
    app.run(debug=True, port=5000)
