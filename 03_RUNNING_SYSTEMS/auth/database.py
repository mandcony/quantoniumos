"""
QuantoniumOS Database Configuration

Database initialization and configuration for QuantoniumOS.
"""

import sqlite3
import threading
from contextlib import contextmanager
from typing import Optional, Dict, Any

# Thread-local storage for database connections
_local = threading.local()

class Database:
    """Thread-safe database wrapper for QuantoniumOS"""
    
    def __init__(self, db_path: str = "quantonium.db"):
        self.db_path = db_path
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # API keys table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get a database connection"""
        if not hasattr(_local, 'conn'):
            _local.conn = sqlite3.connect(self.db_path)
            _local.conn.row_factory = sqlite3.Row
        
        try:
            yield _local.conn
        except Exception:
            _local.conn.rollback()
            raise
        else:
            _local.conn.commit()
    
    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a query"""
        with self.get_connection() as conn:
            return conn.execute(query, params)
    
    def fetchone(self, query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """Fetch one row"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchone()
    
    def fetchall(self, query: str, params: tuple = ()) -> list:
        """Fetch all rows"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchall()
    
    def close(self):
        """Close database connection"""
        if hasattr(_local, 'conn'):
            _local.conn.close()
            del _local.conn

# Global database instance
db = Database()

def initialize_auth():
    """Initialize authentication system"""
    # Database is already initialized in the Database constructor
    print("✅ QuantoniumOS Auth System Initialized")
    return True
