# QuantoniumOS - Complete Route Structure & Security Features

## Main Application Routes (`main.py`)

### Core System Routes
- `GET /` - Main QuantoniumOS interface (serves quantum-os.html)
- `GET /home` - Alternative home route 
- `GET /health` - Health check endpoint
- `GET /status` - System status with uptime and metrics
- `GET /deployment-test` - Deployment validation
- `GET /docs` - API documentation (Swagger UI)
- `GET /openapi.json` - OpenAPI specification

### Quantum Application Interfaces
- `GET /quantonium-os-100x` - Enhanced 100x interface with advanced features
- `GET /quantum-encryption` - Quantum encryption interface
- `GET /quantum-rft` - Resonance Fourier Transform interface  
- `GET /quantum-container` - Container operations interface
- `GET /quantum-entropy` - Quantum random number generation
- `GET /quantum-benchmark` - Performance benchmarking interface
- `GET /quantum-browser` - Quantum browser application
- `GET /quantum-mail` - Quantum mail application
- `GET /quantum-notes` - Quantum notes application

### Specialized Applications
- `GET /resonance-encrypt` - Resonance encryption interface
- `GET /resonance-transform` - Resonance transform operations
- `GET /resonance-analyzer` - Resonance analysis tools
- `GET /container-operations` - Container management interface
- `GET /quantum-grid` - Interactive quantum grid visualization
- `GET /64-benchmark` - 64-qubit benchmark testing

### Embedded Widgets & Integrations
- `GET /embed` - Generic embed interface
- `GET /widget` - Embeddable quantum widget
- `GET /wave-embed` - Wave visualization embed
- `GET /squarespace-embed` - Squarespace integration widget
- `GET /embed-demo` - Demonstration embed interface

### Desktop Application Launchers
- `GET /desktop-analyzer` - Desktop resonance analyzer launcher
- `GET /os` - Full QuantoniumOS launcher
- `POST /api/launch-desktop-analyzer` - API to launch desktop analyzer
- `POST /api/launch-app` - Generic application launcher API

### Frontend & Development
- `GET /frontend` - Frontend development interface
- `GET /wave` - Wave analysis interface
- `GET /wave_ui/<path:filename>` - Wave UI static files

## API Routes

### Core Encryption API (`main.py`)
- `POST /api/encrypt` - Resonance-based encryption
- `POST /api/decrypt` - Secure decryption operations
- `POST /api/entropy` - Quantum entropy generation

### Quantum API Routes (`main.py`)
- `POST /api/quantum/encrypt` - Quantum encryption operations
- `POST /api/quantum/decrypt` - Quantum decryption operations  
- `POST /api/quantum/entropy` - Quantum entropy API
- `POST /api/quantum/entropy/sample` - Entropy sampling
- `GET /api/quantum/entropy/stream` - Real-time entropy stream

### System API
- `GET /api/health` - API health check
- `GET /api/metrics` - Application performance metrics

### Security API
- `GET /security/status` - Enterprise security status
- `POST /security/analyze` - Advanced security analysis
- `POST /quantum/container/secure` - Secure quantum container creation

### Extended Quantum API (`routes_quantum.py`)
- `POST /api/quantum/initialize` - Initialize quantum operations
- `POST /api/quantum/encrypt` - Advanced quantum encryption
- Additional quantum computation endpoints

## Authentication Routes (`auth/routes.py`)

### API Key Management
- `POST /auth/api-key` - Create new API key
- `GET /auth/api-key` - List API keys
- `PUT /auth/api-key/<key_id>` - Update API key
- `DELETE /auth/api-key/<key_id>` - Revoke API key

### Authentication Operations
- `POST /auth/validate` - Validate JWT token
- `POST /auth/rotate` - Rotate API keys
- `GET /auth/audit` - API usage audit logs

## Security Features Implementation

### 1. Request Security Middleware

#### Global Security Headers (`main.py:48-55`)
```python
@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache" 
    response.headers["Expires"] = "0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response
```

#### WordPress/PHP Attack Protection (`main.py:62-94`)
**Blocked Patterns:**
- `/wp-admin`, `/wp-login`, `/wp-content`, `/wp-includes`
- `/wordpress`, `/admin.php`, `/login.php`, `/xmlrpc.php` 
- All `.php` file extensions
- `phpmyadmin`, `acme-challenge` probes

#### Protected File Access (`main.py:108-125`)
**Proprietary Files Blocked:**
- `/static/circuit-designer.js` - Quantum circuit designer
- `/static/quantum-matrix.js` - Advanced quantum operations
- `/static/resonance-core.js` - Core resonance algorithms

### 2. Enterprise Security System (`enterprise_security.py`)

#### Web Application Firewall (WAF)
```python
class WebApplicationFirewall:
    - SQL injection detection
    - XSS attack prevention  
    - Path traversal protection
    - DDoS pattern analysis
    - Request anomaly detection
```

#### Database Encryption
```python
class DatabaseEncryption:
    - Field-level encryption for sensitive data
    - AES-256-GCM encryption
    - Secure key management
    - Encrypted database storage
```

#### Advanced Monitoring
```python
class AdvancedMonitoring:
    - Real-time security event logging
    - Threat pattern recognition
    - Automated alert system
    - Security dashboard metrics
```

### 3. Rate Limiting System (`redis_config.py`)

#### Rate Limiting Features:
- **Redis-based tracking** - Distributed rate limiting
- **Per-IP limits** - Prevent abuse from single sources
- **Per-API-key limits** - Individual key quotas
- **Configurable thresholds** - Adjustable limits per endpoint
- **Graceful degradation** - Fallback when Redis unavailable

#### Current Rate Limit Issues:
- Redis connection failures (`localhost:6379`)
- Rate limiting bypassed when Redis down
- Requires Redis service configuration

### 4. Authentication Security

#### JWT Token Security:
- Short-lived tokens (configurable expiration)
- Secure signing algorithms
- Token validation on protected routes
- Automatic token refresh

#### API Key Management:
- Cryptographically secure key generation
- Hashed storage (never plaintext)
- Key rotation capabilities
- Usage audit logging
- Configurable permissions per key

### 5. Request Analysis & Logging

#### Security Event Types:
- `AUDIT` - Normal operations tracking
- `WARNING` - Suspicious activity detection  
- `CRITICAL` - Active attack attempts
- `ACCESS_DENIED` - Blocked requests

#### Comprehensive Logging:
- Request/response pairs
- IP address tracking
- User agent analysis
- Timing information
- Error patterns

### 6. Container Security

#### Execution Environment:
- **Non-root execution** - UID 1001 for security
- **Read-only filesystem** - Prevents runtime modification
- **Capability dropping** - All Linux capabilities removed
- **Seccomp profiles** - Restricted syscall access

#### Resource Isolation:
- Memory limits
- CPU quotas  
- Network restrictions
- File system sandboxing

## Static File Protection

### Protected Assets:
1. **Circuit Designer** (`/static/circuit-designer.js`)
   - Proprietary quantum circuit design algorithms
   - Access blocked from external requests

2. **Quantum Matrix** (`/static/quantum-matrix.js`) 
   - Advanced quantum matrix operations
   - Contains proprietary algorithms

3. **Resonance Core** (`/static/resonance-core.js`)
   - Core resonance mathematical functions
   - Protected intellectual property

### Security Implementation:
```python
proprietary_files = [
    'circuit-designer.js',
    'quantum-matrix.js', 
    'resonance-core.js'
]

@app.before_request
def protect_proprietary_files():
    if request.path.startswith('/static/'):
        filename = request.path.split('/')[-1]
        if filename in proprietary_files:
            app.logger.warning(f"BLOCKED access to proprietary file: {request.path}")
            abort(403)
```

## Database Security

### Encryption at Rest:
- Sensitive fields encrypted with AES-256-GCM
- Master key management system
- Automatic key rotation
- Secure key derivation

### Access Control:
- Parameterized queries (SQL injection prevention)
- ORM usage (SQLAlchemy)
- Connection pooling with security
- Database audit logging

## Performance & Monitoring

### Metrics Collection:
- Request processing times
- Error rates and patterns
- Security event frequencies
- Resource utilization

### Health Monitoring:
- Application uptime tracking
- Database connectivity status
- Redis connection health
- External service dependencies

## Deployment Security

### Production Hardening:
- HTTPS enforcement
- Secure headers implementation
- CORS policy configuration
- Environment variable protection

### Container Security:
- Minimal attack surface
- Security scanning
- Vulnerability management
- Regular security updates

This comprehensive structure provides enterprise-grade security with quantum-inspired computational capabilities, ensuring both advanced functionality and robust protection against modern threats.