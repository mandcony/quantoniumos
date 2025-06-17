# QuantoniumOS Cloud Runtime - replit.md

## Overview

QuantoniumOS is a hybrid computational framework that bridges classical and quantum computing paradigms through quantum-inspired symbolic computing. The platform provides a Flask-based API for accessing quantum-inspired computational resources with advanced authentication, cryptographic operations, and containerized security.

## System Architecture

### Backend Architecture
- **Flask Application**: Python-based web API with Gunicorn WSGI server
- **Database Layer**: PostgreSQL for data persistence with encryption at rest
- **Authentication**: JWT-based authentication with API key management
- **Security**: Containerized execution with seccomp profiles and capability dropping

### Frontend Architecture
- **Web Interface**: Static HTML/CSS/JavaScript applications
- **API Integration**: RESTful endpoints for quantum operations
- **Responsive Design**: Browser-compatible quantum visualization tools

### Container Security
- **Non-root execution**: UID 1001 for enhanced security
- **Read-only filesystem**: Prevents runtime modification
- **Capability dropping**: All Linux capabilities removed
- **Seccomp enforcement**: Restricted syscall access

## Key Components

### Core Modules
1. **Quantum Engine Adapter** (`encryption/quantum_engine_adapter.py`): Unified interface for quantum operations
2. **Resonance Fourier Transform** (`core/encryption/resonance_fourier.py`): Advanced signal analysis
3. **Symbolic Interface** (`core/protected/symbolic_interface.py`): Bridges API with protected algorithms
4. **Enterprise Security** (`enterprise_security.py`): Database encryption and WAF protection

### API Endpoints
- `/api/encrypt` - Resonance-based encryption
- `/api/decrypt` - Secure decryption operations  
- `/api/rft` - Resonance Fourier Transform
- `/api/entropy` - Quantum-inspired random number generation
- `/api/container/unlock` - Container validation and access
- `/api/quantum/*` - Quantum computing operations

### Authentication System
- **API Keys**: Database-stored with rotation capabilities
- **JWT Tokens**: Short-lived tokens for session management
- **Audit Logging**: Comprehensive security event tracking
- **Rate Limiting**: Per-IP and per-key request throttling

## Data Flow

1. **Request Authentication**: API key validation and JWT token generation
2. **Input Validation**: Pydantic-based request schema validation
3. **Security Processing**: Rate limiting and threat detection
4. **Quantum Operations**: Symbolic computation through protected modules
5. **Response Signing**: Cryptographic response validation
6. **Audit Logging**: Security event recording

## External Dependencies

### Python Packages
- Flask 3.1.0 - Web framework
- Gunicorn 23.0.0 - WSGI server
- PostgreSQL (psycopg2-binary) - Database connectivity
- Cryptography 44.0.2+ - Encryption operations
- PyJWT 2.10.1+ - JWT token handling
- Flask-CORS 5.0.1 - Cross-origin resource sharing
- Pydantic 2.11.3 - Data validation

### System Dependencies
- PostgreSQL 16 - Primary database
- Redis (optional) - Caching and rate limiting
- Python 3.11+ - Runtime environment

### External Services
- Replit deployment platform
- Optional Zenodo integration for research publications

## Deployment Strategy

### Development Environment
- **Local Flask Server**: Direct Python execution on port 5000
- **Hot Reload**: Automatic restart on code changes
- **Debug Mode**: Enhanced error reporting

### Production Deployment
- **Gunicorn WSGI**: Multi-worker production server
- **Docker Containerization**: Secure isolated execution
- **Auto-scaling**: Replit autoscale deployment target
- **Health Monitoring**: Built-in health check endpoints

### Security Hardening
- **Container Isolation**: Read-only filesystem with tmpfs mounts
- **Network Security**: CORS lockdown and security headers
- **Secrets Management**: Environment-based configuration
- **Vulnerability Scanning**: Automated dependency auditing

## Changelog

- June 17, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.