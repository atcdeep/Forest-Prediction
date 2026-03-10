from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mysql.connector
from mysql.connector import Error
import os
import json
from datetime import datetime, timedelta
import traceback
import csv
import io
import re
import secrets
from functools import wraps

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)  # For session management
CORS(app)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Harshdeep*123',
    'database': 'forest_prediction_db'
}

# Initialize database with proper error handling
def init_database():
    try:
        # First connect without database
        connection = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        
        cursor = connection.cursor()
        
        # Drop and create database
        cursor.execute(f"DROP DATABASE IF EXISTS {DB_CONFIG['database']}")
        cursor.execute(f"CREATE DATABASE {DB_CONFIG['database']}")
        cursor.execute(f"USE {DB_CONFIG['database']}")
        
        # Create tables
        cursor.execute('''
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
            )
        ''')
        
        cursor.execute('''
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
            )
        ''')
        
        cursor.execute('''
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
            )
        ''')
        
        cursor.execute('''
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
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE user_activity (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                activity_type VARCHAR(100),
                description TEXT,
                ip_address VARCHAR(45),
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX idx_predictions_user ON predictions(user_id)')
        cursor.execute('CREATE INDEX idx_datasets_user ON datasets(user_id)')
        cursor.execute('CREATE INDEX idx_predictions_date ON predictions(prediction_date)')
        cursor.execute('CREATE INDEX idx_datasets_primary ON datasets(is_primary)')
        cursor.execute('CREATE INDEX idx_datasets_name ON datasets(name)')
        cursor.execute('CREATE INDEX idx_users_email ON users(email)')
        cursor.execute('CREATE INDEX idx_activity_user ON user_activity(user_id)')
        
        # Create default admin user
        cursor.execute('''
            INSERT IGNORE INTO users (uid, email, name, role, last_login) 
            VALUES (%s, %s, %s, %s, NOW())
        ''', ('admin_local', 'admin@forestpredict.com', 'Admin User', 'admin'))
        
        connection.commit()
        print("✅ Database initialized successfully")
        
        # Get admin ID
        admin_id = get_admin_id(cursor)
        
        # Load default forest_data.csv for admin if it exists
        if admin_id and os.path.exists('forest_data.csv'):
            load_forest_data_for_admin(cursor, connection, admin_id)
        
        cursor.close()
        connection.close()
        
    except Error as e:
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
        
        # Insert dataset for admin
        cursor.execute(
            """INSERT INTO datasets (user_id, name, row_count, column_count, is_primary, is_default, description) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (admin_id, 'forest_data', len(df), len(df.columns), True, True, 'Forest dataset with environmental metrics')
        )
        
        # Store column metadata for admin
        for column in df.columns:
            is_numeric = pd.api.types.is_numeric_dtype(df[column])
            min_val = float(df[column].min()) if is_numeric and len(df[column].dropna()) > 0 else None
            max_val = float(df[column].max()) if is_numeric and len(df[column].dropna()) > 0 else None
            
            cursor.execute(
                """INSERT INTO dataset_columns (user_id, dataset_name, column_name, data_type, is_numeric, min_value, max_value) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)""",
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
    """Create database connection with retry logic"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"❌ Database connection error: {e}")
        try:
            init_database()
            connection = mysql.connector.connect(**DB_CONFIG)
            return connection
        except:
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
            'last_updated': last_updated.isoformat() if last_updated else None
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
                'uploaded_date': ds['uploaded_date'].isoformat() if ds['uploaded_date'] else None,
                'description': ds['description']
            })
        
        return jsonify(formatted_datasets)
        
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
    """Get columns for a specific dataset for current user"""
    try:
        user_id = get_current_user_id()
        if not dataset_name:
            return jsonify({'success': False, 'columns': []})
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'columns': []})
        
        cursor = connection.cursor(dictionary=True)
        
        # First check if dataset exists for this user
        cursor.execute("SELECT COUNT(*) as count FROM datasets WHERE user_id = %s AND name = %s", (user_id, dataset_name))
        if cursor.fetchone()['count'] == 0:
            cursor.close()
            connection.close()
            return jsonify({'success': False, 'columns': []})
        
        # Get columns from metadata for this user
        cursor.execute("SELECT column_name FROM dataset_columns WHERE user_id = %s AND dataset_name = %s", (user_id, dataset_name))
        columns = [row['column_name'] for row in cursor.fetchall()]
        
        cursor.close()
        connection.close()
        
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
        return jsonify({'success': True, 'identifier_column': identifier_col, 'values': unique_vals})

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
                model = LinearRegression()

                if len(df_clean) >= 4:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model.fit(X_train, y_train)
                    train_score = model.score(X_train, y_train)
                    test_score = model.score(X_test, y_test) if len(X_test) > 0 else train_score
                    accuracy = (train_score + test_score) / 2 if len(X_test) > 0 else train_score
                else:
                    model.fit(X, y)
                    accuracy = model.score(X, y)

                prediction = model.predict([[target_year]])[0]
                
                all_predictions.append({
                    'metric': col,
                    'prediction': float(prediction),
                    'accuracy': float(accuracy),
                    'data_points': len(df_clean)
                })

                # Save each prediction to DB
                try:
                    conn = get_db_connection()
                    if conn:
                        cur = conn.cursor()
                        cur.execute(
                            """INSERT INTO predictions (user_id, dataset_name, predicted_metric, year, predicted_value, accuracy, prediction_type)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                            (user_id, dataset_name, col, target_year, float(prediction), float(accuracy), 'row_wise')
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
            model = LinearRegression()

            if len(df_clean) >= 4:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test) if len(X_test) > 0 else train_score
                accuracy = (train_score + test_score) / 2 if len(X_test) > 0 else train_score
            else:
                model.fit(X, y)
                accuracy = model.score(X, y)

            prediction = model.predict([[target_year]])[0]

            # Save to database
            try:
                connection = get_db_connection()
                if connection:
                    cursor = connection.cursor()
                    cursor.execute(
                        """INSERT INTO predictions (user_id, dataset_name, predicted_metric, year, predicted_value, accuracy, prediction_type)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                        (user_id, dataset_name, metric_column, target_year, float(prediction), float(accuracy), prediction_type)
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
                'data_points': len(df_clean),
                'metric': metric_column,
                'year': target_year,
                'dataset': dataset_name
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
                pred['prediction_date'] = pred['prediction_date'].isoformat()
        
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
            if user['last_login']:
                user['last_login'] = user['last_login'].isoformat() if user['last_login'] else None
            if user['created_at']:
                user['created_at'] = user['created_at'].isoformat() if user['created_at'] else None
            if user['updated_at']:
                user['updated_at'] = user['updated_at'].isoformat() if user['updated_at'] else None
            if user['last_activity']:
                user['last_activity'] = user['last_activity'].isoformat() if user['last_activity'] else None
        
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
            if activity['created_at']:
                activity['created_at'] = activity['created_at'].isoformat()
        
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
        
        cursor.execute("SELECT COUNT(*) as active_users FROM users WHERE is_active = TRUE")
        active_users = cursor.fetchone()['active_users']
        
        cursor.execute("SELECT COUNT(*) as admin_users FROM users WHERE role = 'admin'")
        admin_users = cursor.fetchone()['admin_users']
        
        # Get activity stats for last 7 days
        cursor.execute('''
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM user_activity
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
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




if __name__ == '__main__':
    # Create necessary directories
    for folder in ['uploads', 'templates', 'templates/auth']:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    print("🚀 Starting Forest Data Analysis & Prediction System...")
    print(f"📁 Forest data file: {'✅ Found' if os.path.exists('forest_data.csv') else '❌ Not found'}")
    print("🌐 Server running at: http://localhost:5000")
    app.run(debug=True, port=5000)
