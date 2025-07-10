# QuantoniumOS - Complete Application Architecture

## Overview
QuantoniumOS is a hybrid quantum-classical computational platform providing secure API access to quantum-inspired algorithms, container validation systems, and advanced cryptographic operations.

## Core Application Structure

### 1. Main Application (`main.py`)
**Primary Flask Application with Security Middleware**

#### Core Routes:
- `/` - Main QuantoniumOS interface (quantum-os.html)
- `/quantonium-os-100x` - Enhanced 100x interface with advanced features
- `/health` - Health check endpoint
- `/status` - System status with uptime metrics
- `/metrics` - Application performance metrics
- `/deployment-test` - Deployment validation endpoint

#### Quantum Application Routes:
- `/quantum-encryption` - Quantum encryption interface
- `/quantum-rft` - Resonance Fourier Transform interface
- `/quantum-container` - Container operations interface
- `/quantum-entropy` - Quantum random number generation
- `/quantum-benchmark` - Performance benchmarking
- `/quantum-browser` - Quantum browser application
- `/quantum-mail` - Quantum mail application
- `/quantum-notes` - Quantum notes application

#### Embedded Interfaces:
- `/widget` - Embeddable quantum widget
- `/wave-visualization-embed` - Wave visualization widget
- `/squarespace-embed` - Squarespace integration
- `/embed-demo` - Demonstration embed

#### Application Launchers:
- `/desktop-analyzer-launcher` - Desktop resonance analyzer
- `/os-launcher` - Full OS launcher
- `/launch-desktop-analyzer` - API endpoint for desktop launch
- `/launch-app` - Generic app launcher API

### 2. API Routes (`routes/`)

#### Core API (`routes/api.py`):
- `/api/encrypt` - Resonance-based encryption
- `/api/decrypt` - Secure decryption
- `/api/rft` - Resonance Fourier Transform
- `/api/entropy` - Quantum entropy generation
- `/api/container/unlock` - Container validation and access

#### Quantum API (`routes_quantum.py`):
- `/api/quantum/encrypt` - Quantum encryption operations
- `/api/quantum/decrypt` - Quantum decryption operations
- `/api/quantum/entropy` - Quantum entropy API
- `/api/quantum/entropy/sample` - Entropy sampling
- `/api/quantum/entropy/stream` - Real-time entropy stream

### 3. Authentication System (`auth/`)

#### Components:
- **JWT Authentication** (`auth/jwt_auth.py`)
- **API Key Management** (`auth/models.py`)
- **Key Rotation Service** (`auth/key_rotation_service.py`)
- **Secret Manager** (`auth/secret_manager.py`)

#### Authentication Routes (`auth/routes.py`):
- `/auth/api-key` - API key management
- `/auth/validate` - Token validation
- `/auth/rotate` - Key rotation endpoint

#### Database Models:
```python
class APIKey(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key_id = db.Column(db.String(36), unique=True, nullable=False)
    key_hash = db.Column(db.String(256), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_used = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    rate_limit = db.Column(db.Integer, default=1000)
    
class APIKeyAuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    api_key_id = db.Column(db.Integer, db.ForeignKey('api_key.id'))
    action = db.Column(db.String(50), nullable=False)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
```

### 4. Security Layer (`security.py`, `enterprise_security.py`)

#### Security Features:
- **Web Application Firewall (WAF)**
- **DDoS Protection**
- **Rate Limiting** (Redis-based)
- **Request Analysis**
- **Attack Pattern Detection**
- **Database Encryption at Rest**
- **Real-time Security Monitoring**

#### Security Components:
```python
class WebApplicationFirewall:
    - SQL injection detection
    - XSS attack prevention
    - Path traversal protection
    - WordPress/PHP attack blocking
    - Suspicious pattern analysis

class DatabaseEncryption:
    - Field-level encryption
    - Key management
    - Secure data storage

class AdvancedMonitoring:
    - Real-time event logging
    - Security dashboard
    - Alert system
    - Audit trail
```

#### Protected Endpoints:
- Proprietary file access blocking
- Rate limiting per IP/API key
- Request pattern analysis
- Automated threat response

### 5. Encryption Modules (`encryption/`)

#### Core Encryption Components:
- **Quantum Engine Adapter** (`quantum_engine_adapter.py`)
- **Resonance Encryption** (`resonance_encrypt.py`)
- **Resonance Fourier Transform** (`resonance_fourier.py`)
- **Geometric Container** (`geometric_container.py`)
- **Wave Primitives** (`wave_primitives.py`)
- **Entropy QRNG** (`entropy_qrng.py`)

#### Encryption Algorithms:
- Resonance-based symmetric encryption
- Quantum-inspired key derivation
- Geometric waveform hashing
- True quantum random number generation

### 6. Container System (`core/`)

#### Container Components:
- **HPC Modules** (`core/HPC/`)
- **Protected Algorithms** (`core/protected/`)
- **Symbolic Interface** (`core/encryption/`)

#### Container Features:
- Secure execution environment
- Hash-based validation
- Isolated computation
- Resource management

### 6.1. 100-Qubit State-Vector Simulator

The quantum simulator is a key component of the QuantoniumOS backend. It employs a hybrid strategy to balance performance and resource consumption, enabling the simulation of a high number of qubits on a single node.

#### Simulation Strategy:
- **Dense State-Vector Simulation (for N ≤ 28 qubits):** For circuits with 28 or fewer qubits, the simulator uses a standard dense state-vector representation. The state of the system is stored as a complex vector of size 2^N, allowing for precise and fast operations on smaller quantum systems.
- **Sparse State-Vector Simulation (for N > 28 qubits):** When the number of qubits exceeds 28, the memory requirements for a dense vector become prohibitive. The simulator automatically switches to a sparse representation. In this mode, only the non-zero amplitudes of the state vector are stored in a dictionary or a similar hash map structure. This approach is highly effective for quantum states that are not maximally entangled and have a limited number of significant basis states.

#### Resource Bounds:
- The sparse simulator is configured with a maximum limit of **10^6 (one million) non-zero states**. If a simulation exceeds this limit, it will terminate with an error, preventing uncontrolled memory consumption. This bound allows for the simulation of large, but not fully random, quantum circuits, which is typical for many quantum algorithms.
- This hybrid approach allows QuantoniumOS to claim support for up to 100 qubits under the condition that the quantum state remains sparse enough to fit within the defined resource limits.

### 7. Applications (`apps/`)

#### Quantum Applications:
- **Quantum Nova System** (`apps/quantum_nova/`)
- **Resonance Analyzer** (`apps/resonance_analyzer/`)
- **Bridge Interface** (`apps/bridge.py`)

### 8. Desktop Environment (`attached_assets/`)

#### Qt Applications:
- **QuantoniumOS Desktop** (`quantonium_os_100x.py`)
- **Task Manager** (`qshll_task_manager.py`)
- **File Explorer** (`qshll_file_explorer.py`)
- **System Monitor** (`qshll_system_monitor.py`)
- **Resonance Analyzer** (`q_resonance_analyzer.py`)
- **Wave Debugger** (`q_wave_debugger.py`)
- **Mail Client** (`q_mail.py`)
- **Browser** (`q_browser.py`)
- **Notes App** (`q_notes.py`)
- **Vault** (`q_vault.py`)

#### Desktop Features:
- Multi-tab interface
- Particle effects and animations
- Quantum grid visualization
- Beige/terracotta design theme
- Full OS experience

### 9. Static Assets (`static/`)

#### Web Interfaces:
- `quantum-os.html` - Main OS interface
- `quantonium_os_web_100x.html` - Enhanced 100x interface
- `resonance-encrypt.html` - Encryption interface
- `wave-visualization.html` - Wave visualization
- `quantum-grid.html` - Quantum grid interface

#### Protected Assets:
- `circuit-designer.js` - Proprietary quantum circuit designer
- `quantum-matrix.js` - Advanced quantum matrix operations
- Various visualization and UI components

## Security Implementation

### 1. Multi-Layer Security Architecture

#### Request Processing Pipeline:
1. **Rate Limiting Check** (Redis-based)
2. **WAF Analysis** (Attack pattern detection)
3. **Authentication Validation** (JWT/API Key)
4. **Authorization Check** (Resource access control)
5. **Request Logging** (Security audit trail)
6. **Response Security Headers** (XSS, CSRF protection)

### 2. Attack Prevention Systems

#### WordPress/PHP Attack Blocking:
```python
wordpress_patterns = [
    '/wp-admin', '/wp-login', '/wp-content', '/wp-includes',
    '/wordpress', '/admin.php', '/login.php', '/xmlrpc.php'
]
```

#### File Access Protection:
- Block access to proprietary JavaScript files
- Prevent directory traversal attacks
- Validate file extensions and paths

#### SQL Injection Prevention:
- Parameterized queries
- Input sanitization
- ORM usage (SQLAlchemy)

### 3. Monitoring and Alerting

#### Security Event Types:
- `SecurityEventType.AUDIT` - Normal operations
- `SecurityEventType.WARNING` - Suspicious activity
- `SecurityEventType.CRITICAL` - Active attacks
- `SecurityEventType.ACCESS_DENIED` - Blocked requests

#### Real-time Monitoring:
- Request/response logging
- Performance metrics
- Error tracking
- Security dashboard

## Database Schema

### Core Tables:
- `api_key` - API key management
- `api_key_audit_log` - API usage audit trail
- Security event logs (JSON format)
- Rate limiting data (Redis)

## Configuration Management

### Environment Variables:
- `DATABASE_URL` - PostgreSQL connection
- `FLASK_SECRET_KEY` - Session security
- `REDIS_URL` - Rate limiting storage
- Security tokens and API keys

### Feature Flags:
- Development/production modes
- Debug logging levels
- Security enforcement levels

## Performance Optimizations

### Caching Strategy:
- Redis for rate limiting
- Static file caching
- Database query optimization

### Resource Management:
- Connection pooling
- Memory usage monitoring
- CPU optimization

## Deployment Architecture

### Container Security:
- Non-root execution (UID 1001)
- Read-only filesystem
- Capability dropping
- Seccomp profiles

### Production Features:
- Gunicorn WSGI server
- Health check endpoints
- Auto-scaling support
- Load balancing ready

## API Documentation

### OpenAPI Specification:
- Available at `/api-docs`
- Interactive Swagger UI
- Complete endpoint documentation
- Authentication examples

### Rate Limits:
- Default: 1000 requests/hour per API key
- Configurable per key
- Redis-based tracking
- Graceful degradation

## Recent Enhancements

### June 2025 Updates:
- QSS color scheme integration
- Enhanced particle effects
- Improved Qt application
- Security monitoring improvements
- Rate limiting refinements

This architecture provides a comprehensive, secure, and scalable platform for quantum-inspired computational operations with enterprise-grade security features.