"""
Quantonium OS - Flask App Entrypoint

Initializes the Flask app and registers symbolic API routes with security middleware.
Includes protected quantum computing API routes with 150-qubit support.
"""

import os
import time
import logging
import platform
import json
import secrets
from datetime import datetime
from flask import Flask, send_from_directory, redirect, jsonify, g, request, render_template_string
from routes import api
from routes import encrypt, decrypt
from auth.routes import auth_api
from security import configure_security
from routes_quantum import quantum_api
from utils.json_logger import setup_json_logger
from utils.security_logger import setup_security_logger, log_security_event, SecurityEventType, SecurityOutcome
from auth import initialize_auth, db, APIKey, APIKeyAuditLog

try:
    from __init__ import __version__ as app_version
except ImportError:
    app_version = "0.3.0-rc1"

# Configure global logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("quantonium_app")

# Track application start time for uptime reporting
APP_START_TIME = time.time()
APP_VERSION = app_version  # Version tracking for API

# Set development environment by default for local development
if 'FLASK_ENV' not in os.environ:
    os.environ['FLASK_ENV'] = 'development'

def create_app():
    app = Flask(__name__, static_folder='static')
    
    # Global middleware to intercept ALL routes, prevent caching, and enforce security
    @app.after_request
    def add_no_cache_headers(response):
        """Add headers to prevent caching for all responses"""
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response
    
    # Global middleware to intercept routes for security and redirection
    @app.before_request
    def redirect_to_os():
        app.logger.info(f"Incoming request to: {request.path} with args: {request.args}")
        
        # CRITICAL: If accessing the root URL, force serve the OS interface directly
        if request.path == '/':
            app.logger.warning("ROOT URL accessed - force serving quantum-os.html directly")
            return send_from_directory('static', 'quantum-os.html')
            
        # Always allow API routes to function as normal
        if request.path.startswith('/api/'):
            app.logger.info("Allowing API route")
            return None
            
        # Always allow static files to be served
        if request.path.startswith('/static/'):
            app.logger.info("Allowing static file")
            return None
            
        # Always allow access to favicon.ico
        if request.path == '/favicon.ico':
            app.logger.info("Allowing favicon")
            return None
            
        # Allow direct access to OS page
        if request.path == '/os':
            app.logger.info("Allowing direct OS access")
            return None
            
        # If accessing any component directly, check if it has embedded parameter
        if (request.path in ['/resonance-encrypt', '/resonance-transform', 
                            '/container-operations', '/quantum-entropy',
                            '/quantum-grid', '/quantum-benchmark', '/64-benchmark']):
            app.logger.warning(f"Component page accessed directly: {request.path}")
            embedded = request.args.get('embedded', 'false').lower() == 'true'
            if not embedded:
                # NOT embedded - this is a direct access attempt - force redirect to OS
                app.logger.warning(f"BLOCKED direct access to {request.path} - redirecting to OS")
                return redirect('/os')
            else:
                app.logger.info(f"Embedded access allowed for {request.path}")
    
    # Configure PostgreSQL database with fallback for SQLite
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.warning("DATABASE_URL not found. Falling back to SQLite in-memory database for development.")
        database_url = "sqlite:///quantonium.db"
    
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
    }
    
    # Initialize authentication module with database and encryption
    initialize_auth(app)
    
    # Generate a master key for JWT secret encryption if not present
    if 'QUANTONIUM_MASTER_KEY' not in os.environ:
        # Set a secure random key for development
        os.environ['QUANTONIUM_MASTER_KEY'] = secrets.token_urlsafe(32)
        logger.warning("QUANTONIUM_MASTER_KEY not found. Using a temporary key for this session.")
    
    # Configure security middleware (replaces permissive CORS)
    talisman, limiter = configure_security(app)
    
    # Add rate limiter middleware
    from middleware.auth import RateLimiter
    app.wsgi_app = RateLimiter(calls=30, period=60)(app.wsgi_app)
    
    # Add NIST 800-53 compliant security audit middleware
    try:
        from middleware.security_audit import initialize_audit_middleware
        initialize_audit_middleware(app)
        logger.info("NIST 800-53 compliant audit middleware initialized")
    except ImportError as e:
        logger.warning(f"Security audit middleware not available: {str(e)}")
    
    app.config["JSON_SORT_KEYS"] = False  # Maintain insertion order
    
    # Enable debug mode for development, disable for production
    is_development = os.environ.get('FLASK_ENV') == 'development'
    app.config["DEBUG"] = is_development
    
    # Disable strict trailing slashes to prevent redirect loops
    app.url_map.strict_slashes = False
    
    # Set up structured JSON logging
    setup_json_logger(app, log_dir="logs", log_level=logging.INFO)
    
    # Set up specialized security logging
    setup_security_logger(app, log_dir="/tmp/logs", log_level=logging.INFO)
    
    # Create tables on startup
    with app.app_context():
        db.create_all()
        
        # Check for encryption migration
        migrate_secrets = os.environ.get('QUANTONIUM_ENCRYPT_SECRETS', 'auto').lower()
        if migrate_secrets in ('auto', 'true', 'yes', '1'):
            try:
                from scripts.encrypt_jwt_secrets import encrypt_api_key_secrets
                logger.info("Running JWT secret encryption migration...")
                results = encrypt_api_key_secrets()
                logger.info(f"Migration results: {results}")
            except Exception as e:
                logger.error(f"Error running JWT secret encryption migration: {str(e)}")
    
    # Register the API blueprints with URL prefix
    # This separates API routes from static content routes
    app.register_blueprint(api, url_prefix='/api')
    app.register_blueprint(auth_api, url_prefix='/api/auth')
    # quantum_api blueprint already has url_prefix='/api/quantum' defined internally
    app.register_blueprint(quantum_api)
    
    # Direct API routes for compatibility with original app
    @app.route('/api/encrypt', methods=['POST'])
    def direct_encrypt():
        from routes import encrypt
        return encrypt()
        
    @app.route('/api/decrypt', methods=['POST'])
    def direct_decrypt():
        from routes import decrypt
        return decrypt()
        
    @app.route('/api/entropy', methods=['POST'])
    def direct_entropy():
        from routes import sample_entropy
        return sample_entropy()
        
    # Additional compatibility routes to ensure both old and new routes work
    @app.route('/api/quantum/encrypt', methods=['POST'])
    def quantum_encrypt_route():
        from routes import encrypt
        return encrypt()
        
    @app.route('/api/quantum/decrypt', methods=['POST'])
    def quantum_decrypt_route():
        from routes import decrypt
        return decrypt()
        
    @app.route('/api/quantum/entropy', methods=['POST'])
    def quantum_entropy_route():
        from routes import sample_entropy
        return sample_entropy()
        
    @app.route('/api/quantum/entropy/sample', methods=['POST'])
    def quantum_entropy_sample_route():
        from routes import sample_entropy
        return sample_entropy()
        
    @app.route('/api/quantum/entropy/stream', methods=['GET'])
    def quantum_entropy_stream_route():
        from routes import entropy_stream
        return entropy_stream()
    
    # API health check endpoint (not rate limited)
    @app.route('/api/health')
    @limiter.exempt
    def health_check():
        uptime_seconds = int(time.time() - APP_START_TIME)
        return jsonify({
            "status": "ok",
            "version": APP_VERSION,
            "uptime_s": uptime_seconds,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "host": platform.node()
        })
    
    # Serve embed widget
    @app.route('/embed')
    def serve_widget():
        return send_from_directory('static', 'quantonium-widget.html')
    
    # Route to help users find the embed page
    @app.route('/widget')
    def widget_redirect():
        return redirect('/embed')
        
    # Serve the wave visualization embed
    @app.route('/wave-embed')
    def wave_visualization_embed():
        return send_from_directory('static/wave_ui', 'embed.html')
        
    # Serve the Squarespace embed page
    @app.route('/squarespace-embed')
    def squarespace_embed():
        return send_from_directory('static', 'squarespace-embed.html')
        
    # Launch Desktop Resonance Analyzer page
    @app.route('/desktop-analyzer')
    def desktop_analyzer_launcher():
        return send_from_directory('static', 'launch_desktop_analyzer.html')
        
    # QuantoniumOS Desktop Launcher
    @app.route('/os')
    def os_launcher():
        app.logger.info("OS launcher route accessed - serving with enhanced cache control")
        response = send_from_directory('static', 'quantum-os.html')
        # Add cache control headers to prevent browser caching
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
        
    # Serve embed demo instructions
    @app.route('/embed-demo')
    def embed_demo():
        return send_from_directory('static', 'embed-demo.html')
    
    # Completely blocking direct access to resonance-encrypt
    @app.route('/resonance-encrypt')
    def resonance_encrypt():
        # Direct access is completely blocked, ONLY allow embedded access
        embedded = request.args.get('embedded', 'false').lower() == 'true'
        
        # Log all attempts to access this page
        app.logger.warning(f"Attempt to access resonance-encrypt directly: embedded={embedded}")
        
        if embedded:
            # Only allow if embedded and with proper timestamp
            timestamp = request.args.get('v', None)
            if timestamp and timestamp.isdigit():
                return send_from_directory('static', 'resonance-encrypt.html')
            
        # All other attempts are blocked and redirected to OS
        return redirect('/os')
    
    # 64-Perturbation Benchmark tool
    @app.route('/64-benchmark')
    def benchmark_redirect():
        # Check if the request is coming from an embedded source within the OS
        embedded = request.args.get('embedded', 'false').lower() == 'true'
        
        # If embedded, redirect to resonance-encrypt with embedded flag
        if embedded:
            return redirect('/resonance-encrypt?embedded=true')
        else:
            # Otherwise direct to the OS
            return redirect('/os')
    
    # Quantum Grid visualization (150-qubit support)
    @app.route('/quantum-grid')
    def quantum_grid():
        # Check if the request is coming from an embedded source within the OS
        embedded = request.args.get('embedded', 'false').lower() == 'true'
        
        # Only allow access if embedded within OS, otherwise redirect to OS
        if embedded:
            return send_from_directory('static', 'quantum-grid.html')
        else:
            return redirect('/os')
        
    # Resonance Fourier Transform
    @app.route('/resonance-transform')
    def resonance_transform():
        # Check if the request is coming from an embedded source within the OS
        embedded = request.args.get('embedded', 'false').lower() == 'true'
        
        # Only allow access if embedded within OS, otherwise redirect to OS
        if embedded:
            return send_from_directory('static', 'resonance-transform.html')
        else:
            return redirect('/os')
        
    # Container Operations
    @app.route('/container-operations')
    def container_operations():
        app.logger.info("Container operations route accessed")
        # Check if the request is coming from an embedded source within the OS
        embedded = request.args.get('embedded', 'false').lower() == 'true'
        
        # Only allow access if embedded within OS, otherwise redirect to OS
        if embedded:
            try:
                return send_from_directory('static', 'container-operations.html')
            except Exception as e:
                app.logger.error(f"Error serving container-operations.html: {e}")
                return f"Error: {e}", 500
        else:
            return redirect('/os')
        
    # Quantum Entropy Generator
    @app.route('/quantum-entropy')
    def quantum_entropy_app():
        # Check if the request is coming from an embedded source within the OS
        embedded = request.args.get('embedded', 'false').lower() == 'true'
        
        # Only allow access if embedded within OS, otherwise redirect to OS
        if embedded:
            return send_from_directory('static', 'quantum-entropy.html')
        else:
            return redirect('/os')
        
    # Comprehensive frontend for Squarespace embedding
    @app.route('/frontend')
    def frontend():
        return send_from_directory('static', 'quantonium-frontend.html')
        
    # Live Waveform Visualization UI
    @app.route('/wave')
    def wave_ui():
        return send_from_directory('static/wave_ui', 'index.html')
        
    # Wave UI static files
    @app.route('/wave_ui/<path:filename>')
    def wave_ui_static(filename):
        return send_from_directory('static/wave_ui', filename)
    
    # Image Resonance Analyzer UI
    @app.route('/resonance-analyzer')
    def resonance_analyzer():
        return send_from_directory('static/resonance_analyzer', 'index.html')
    
    # Removed Qubit Visualizer - merged into the Quantum Grid
        
    # Root route directly serves the QuantoniumOS interface
    @app.route('/')
    def root():
        app.logger.info("ROOT route function called - directly serving quantum-os.html")
        # Directly serve the quantum-os.html file with relative paths to avoid path resolution issues
        response = send_from_directory('static', 'quantum-os.html')
        # Add cache control headers to prevent browser caching
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
        
    # Legacy home route (kept for backward compatibility)
    @app.route('/home')
    def home():
        # Add launcher in the main HTML navigation
        root_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>QuantoniumOS</title>
            <style>
                body {
                    font-family: 'Segoe UI', Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background: #111;
                    color: #fff;
                }
                .header {
                    background: linear-gradient(135deg, #1a2a4a 0%, #162238 100%);
                    color: white;
                    padding: 20px;
                    text-align: center;
                }
                .nav {
                    background: #0d1117;
                    display: flex;
                    justify-content: center;
                    padding: 10px;
                }
                .nav a {
                    color: #fff;
                    text-decoration: none;
                    margin: 0 15px;
                    padding: 8px 12px;
                    border-radius: 4px;
                    transition: background 0.3s;
                }
                .nav a:hover {
                    background: rgba(255,255,255,0.1);
                }
                .content {
                    max-width: 800px;
                    margin: 20px auto;
                    padding: 20px;
                    background: #1e1e2e;
                    border-radius: 10px;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
                }
                .card {
                    background: #252836;
                    border-radius: 8px;
                    padding: 15px;
                    margin-bottom: 15px;
                }
                .footer {
                    text-align: center;
                    padding: 20px;
                    color: #6c7983;
                    font-size: 14px;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>QuantoniumOS</h1>
                <p>Cloud Runtime & Symbolic Computing Platform</p>
            </div>
            
            <div class="nav">
                <a href="/resonance-encrypt">Resonance Encryption</a>
                <a href="/quantum-grid">Quantum Grid</a>
                <a href="/resonance-analyzer">Image Resonance Analyzer</a>
                <a href="/os">QuantoniumOS</a>
            </div>
            
            <div class="content">
                <div class="card">
                    <h2>Welcome to QuantoniumOS</h2>
                    <p>A secure, high-performance quantum-inspired API for symbolic computing.</p>
                    <p>Please select one of the applications from the navigation menu above.</p>
                </div>
            </div>
            
            <div class="footer">
                <p>QuantoniumOS Cloud Runtime v0.3.0 | &copy; 2025 Quantonium Research</p>
            </div>
        </body>
        </html>
        """
        return root_html
        
    # Status endpoint without auth
    @app.route('/status')
    def status():
        uptime_seconds = int(time.time() - APP_START_TIME)
        return jsonify({
            "name": "Quantonium OS Cloud Runtime", 
            "status": "operational", 
            "version": APP_VERSION,
            "uptime_s": uptime_seconds
        })
    
    # Launch the desktop resonance analyzer application
    @app.route('/api/launch-desktop-analyzer', methods=['POST'])
    def launch_desktop_analyzer():
        """API endpoint to launch the desktop resonance analyzer application."""
        try:
            import subprocess
            import sys
            import os
            
            # Get the path to the launcher script
            script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attached_assets")
            launcher_script = os.path.join(script_dir, "launch_resonance_analyzer.py")
            
            if not os.path.exists(launcher_script):
                return jsonify({
                    "success": False,
                    "error": "Launcher script not found"
                }), 404
            
            # Launch the application using subprocess
            subprocess.Popen([sys.executable, launcher_script])
            
            return jsonify({
                "success": True,
                "message": "Desktop application launched successfully"
            })
            
        except Exception as e:
            logger.error(f"Error launching desktop analyzer: {str(e)}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
            
    # Launch any app from the attached_assets folder
    @app.route('/api/launch-app', methods=['POST'])
    def launch_app():
        """API endpoint to launch any app from the attached_assets folder."""
        try:
            import subprocess
            import sys
            import os
            
            # Get app name from request
            data = request.get_json()
            if not data or 'app' not in data:
                return jsonify({
                    "success": False,
                    "error": "Missing app parameter"
                }), 400
                
            app_name = data['app']
            
            # Special case for the full OS launcher
            if app_name == "quantonium_os_main.py":
                # Use the custom launcher for the full OS
                launcher_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantonium_launcher.py")
                if not os.path.exists(launcher_path):
                    return jsonify({
                        "success": False,
                        "error": "Quantonium launcher not found"
                    }), 404
                
                # Launch the OS launcher
                subprocess.Popen([sys.executable, launcher_path])
                
                return jsonify({
                    "success": True,
                    "message": "QuantoniumOS Desktop Environment launched successfully"
                })
            
            # Validate app name to prevent command injection
            if not app_name.endswith('.py') or '/' in app_name or '\\' in app_name:
                return jsonify({
                    "success": False,
                    "error": "Invalid app name"
                }), 400
            
            # Get the path to the app script
            script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attached_assets")
            app_script = os.path.join(script_dir, app_name)
            
            if not os.path.exists(app_script):
                return jsonify({
                    "success": False,
                    "error": f"App script not found: {app_name}"
                }), 404
            
            # Set PYTHONPATH to include attached_assets directory and full project directory
            env = os.environ.copy()
            proj_dir = os.path.dirname(os.path.abspath(__file__))
            env["PYTHONPATH"] = script_dir + os.pathsep + proj_dir + os.pathsep + env.get("PYTHONPATH", "")
            
            # Set environment variables for headless operation (needed for Qt applications)
            env["QT_QPA_PLATFORM"] = "offscreen"
            
            # Launch the application silently in the background
            subprocess.Popen(
                [sys.executable, app_script], 
                env=env, 
                cwd=script_dir,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE
            )
            
            # Redirect to the appropriate HTML interface based on the app name
            app_base_name = os.path.splitext(app_name)[0]
            
            # Map app names to their HTML interfaces
            html_mapping = {
                "q_wave_composer": "/quantum-rft",
                "q_resonance_analyzer": "/resonance-analyzer",
                "q_wave_debugger": "/quantum-encryption",
                "q_mail": "/quantum-mail",
                "q_browser": "/quantum-browser",
                "q_notes": "/quantum-notes",
                "q_vault": "/quantum-container",
                "quantum_desktop": "/os",
                "quantum_encryption": "/quantum-encryption",
                "quantum_rft": "/quantum-rft"
            }
            
            # Get the HTML interface or use a default
            html_interface = html_mapping.get(app_base_name, "/")
            
            return jsonify({
                "success": True,
                "redirect_to": html_interface
            })
            
        except Exception as e:
            logger.error(f"Error launching app: {str(e)}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    # Metrics endpoint - protected by API key
    @app.route('/api/metrics')
    def metrics():
        import os
        import psutil
        
        # Get process information
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Get rate limiter stats (if available from the limiter implementation)
        limiter_stats = {}
        if hasattr(limiter, 'limiter'):
            limiter_stats = {
                "limit": "60 per minute",
                "endpoints": len(limiter.limiter.get_application_limits()),
                "exempt_routes": len(limiter.exempt_routes)
            }
        
        # Build metrics response
        metrics_data = {
            "process": {
                "memory_rss_bytes": memory_info.rss,
                "cpu_percent": process.cpu_percent(),
                "open_files": len(process.open_files()),
                "threads": process.num_threads()
            },
            "uptime_s": int(time.time() - APP_START_TIME),
            "rate_limiter": limiter_stats,
            "version": APP_VERSION
        }
        
        return jsonify(metrics_data)
        
    # Serve OpenAPI spec as JSON
    @app.route('/openapi.json')
    @limiter.exempt
    def openapi_spec():
        try:
            with open('openapi.json', 'r') as f:
                spec = json.load(f)
                # Update the version from app version
                if 'info' in spec:
                    spec['info']['version'] = APP_VERSION
                return jsonify(spec)
        except Exception as e:
            logger.error(f"Error serving OpenAPI spec: {str(e)}")
            return jsonify({"error": "OpenAPI spec not available"}), 500
    
    # Quantum Encryption app
    @app.route('/quantum-encryption')
    def quantum_encryption():
        """Quantum Encryption app."""
        return send_from_directory('static', 'quantum-encryption.html')
        
    # Quantum RFT app
    @app.route('/quantum-rft')
    def quantum_rft():
        """Quantum RFT app."""
        return send_from_directory('static', 'quantum-rft.html')
        
    # Quantum Container app
    @app.route('/quantum-container')
    def quantum_container():
        """Quantum Container app."""
        return send_from_directory('static', 'quantum-container.html')
        
    # Quantum Entropy app
    @app.route('/quantum-entropy')
    def quantum_entropy():
        """Quantum Entropy app."""
        return send_from_directory('static', 'quantum-entropy.html')
        
    # Quantum Benchmark app
    @app.route('/quantum-benchmark')
    def quantum_benchmark_app():
        """Quantum Benchmark app."""
        return send_from_directory('static', 'quantum-benchmark.html')
        
    # Q-Browser app
    @app.route('/quantum-browser')
    def quantum_browser_app():
        """Quantum Browser app."""
        return send_from_directory('static', 'quantum-browser.html')
        
    # Q-Mail app
    @app.route('/quantum-mail')
    def quantum_mail_app():
        """Quantum Mail app."""
        return send_from_directory('static', 'quantum-mail.html')
        
    # Q-Notes app
    @app.route('/quantum-notes')
    def quantum_notes_app():
        """Quantum Notes app."""
        return send_from_directory('static', 'quantum-notes.html')
        
    # Deployment Test Page
    @app.route('/deployment-test')
    def deployment_test():
        """Special route to test deployment status."""
        app.logger.info("Deployment test page accessed")
        return send_from_directory('static', 'deployment-test.html')
    
    # Swagger UI Documentation
    @app.route('/docs')
    def api_docs():
        """API Documentation using Swagger UI."""
        swagger_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Quantonium OS API Documentation</title>
            <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.5.0/swagger-ui.css" />
            <style>
                html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
                *, *:before, *:after { box-sizing: inherit; }
                body { margin: 0; background: #fafafa; }
                .topbar { display: none; }
                .swagger-ui .info .title { color: #3b4151; }
                .swagger-ui .info { margin: 30px 0; }
                .swagger-ui .scheme-container { box-shadow: none; border-radius: 4px; }
                .version-stamp { text-align: right; padding: 10px; font-size: 0.8em; color: #888; }
            </style>
        </head>
        <body>
            <div id="swagger-ui"></div>
            <div class="version-stamp">
                Version: """ + APP_VERSION + """ | <a href="https://github.com/quantonium/quantonium-os/releases/tag/v""" + APP_VERSION + """">Release Notes</a>
            </div>
            <script src="https://unpkg.com/swagger-ui-dist@4.5.0/swagger-ui-bundle.js"></script>
            <script>
                window.onload = function() {
                    window.ui = SwaggerUIBundle({
                        url: "/openapi.json",
                        dom_id: '#swagger-ui',
                        deepLinking: true,
                        presets: [
                            SwaggerUIBundle.presets.apis,
                            SwaggerUIBundle.SwaggerUIStandalonePreset
                        ],
                        layout: "BaseLayout",
                        validatorUrl: null,
                        defaultModelsExpandDepth: 1,
                        defaultModelExpandDepth: 1,
                        supportedSubmitMethods: ['get', 'post'],
                        displayRequestDuration: true
                    });
                };
            </script>
        </body>
        </html>
        """
        return render_template_string(swagger_html)
    
    return app

# ✅ Gunicorn uses this
app = create_app()

# ✅ Development mode only - disabled in production
if __name__ == "__main__":
    # Note: This is for development only, should use gunicorn in production
    logger.warning("Running in development mode. Use gunicorn for production.")
    app.run(host="0.0.0.0", port=8080)
