"""
Auth package initialization
"""

from .routes import auth_api, require_auth, require_admin

def initialize_auth(app):
    """Initialize authentication for the Flask app"""
    # Register the auth blueprint
    app.register_blueprint(auth_api)
    
    # Configure session settings
    app.config['SECRET_KEY'] = app.config.get('SECRET_KEY', 'dev-key-change-in-production')
    app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
    
    print("✅ Authentication system initialized")
    return app

# Placeholder database object for compatibility
class MockDB:
    """Mock database object for development"""
    def __init__(self):
        self.connected = False
    
    def init_app(self, app):
        """Initialize with Flask app"""
        self.connected = True
        print("✅ Mock database initialized")
    
    def create_all(self):
        """Create all tables (mock)"""
        print("✅ Database tables created (mock)")

db = MockDB()

__all__ = ['auth_api', 'require_auth', 'require_admin', 'initialize_auth', 'db']
