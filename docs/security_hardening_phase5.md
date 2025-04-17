# Quantonium OS Security Hardening - Phase 5: Authentication Framework

## Overview

Phase 5 implements an advanced authentication framework for the Quantonium OS API, focusing on robust API key management with JWT tokens, key rotation, revocation, and granular access control.

## Features Implemented

### 1. JWT/HMAC Authentication Framework

- **Salted Hash Storage**: API keys are stored as salted hashes, never in plain text
- **JWT Token Generation**: Short-lived JWT tokens with configurable expiry
- **Per-Key JWT Secrets**: Each API key has its own JWT signing secret for isolation
- **Bearer Token Authentication**: Support for `Authorization: Bearer <token>` header
- **Legacy API Key Support**: Backward compatibility with `X-API-Key` header

### 2. Key Management System

- **Key Database Model**: Complete key lifecycle management via SQLAlchemy models
- **Key Creation**: Generate cryptographically secure API keys with prefix for fast lookup
- **Key Rotation**: Seamless key rotation with immediate revocation of old keys
- **Key Revocation**: Immediate key revocation with reason tracking
- **Key Expiry**: Configurable automatic key expiration

### 3. Permissions & Access Control

- **Permission Strings**: Fine-grained permissions using space-separated strings
- **Admin Keys**: Super-user keys with all permissions
- **Permission Checking**: Decorator-based permission checks on endpoints
- **Scoped Access**: Keys can be limited to specific API operations

### 4. Audit Logging

- **Event Logging**: Comprehensive action logging for all key operations
- **Key ID in Logs**: API key ID included in all logs for traceability
- **Request Tracking**: API request logging with key information
- **IP & User Agent**: Client IP and user agent tracking

### 5. Admin Tools

- **CLI Interface**: Administrative CLI for key management
- **REST API**: Admin API endpoints for programmatic key management
- **Reporting**: Key usage and audit reporting

## Implementation Details

### Database Integration

The authentication framework uses PostgreSQL for secure storage of API keys and audit logs. Key setup:

1. **Database Configuration**: 
   ```python
   app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
   app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
   ```

2. **Table Creation**: 
   ```python
   # Create tables on startup
   with app.app_context():
       db.create_all()
   ```

3. **Models**:
   - `APIKey`: Stores API key metadata, permissions, and status
   - `APIKeyAuditLog`: Records all actions performed with keys

### JWT Token Flow

1. Client authenticates with API key via `X-API-Key` header
2. Server validates key and issues JWT token
3. Client uses token for subsequent requests via `Authorization: Bearer <token>` header
4. Server validates token signature and permissions
5. Token expiry forces periodic reauthentication

### Security Considerations

- API keys stored only as salted hashes using `werkzeug.security.generate_password_hash`
- Each API key has its own isolated JWT signing secret
- All key operations are logged with full audit trail
- Token expiry mitigates risks of token compromise
- Key rotation allows for regular key updates without service disruption

## CLI Commands

```bash
# Create a new API key
python -m auth.cli create --name "Production API Key" --permissions "api:read api:write"

# List all keys
python -m auth.cli list

# Revoke a key
python -m auth.cli revoke --id KEY_ID

# Rotate a key
python -m auth.cli rotate --id KEY_ID

# View audit logs
python -m auth.cli logs --id KEY_ID
```

## API Endpoints

- `POST /api/auth/token` - Get JWT token using API key
- `GET /api/auth/keys` - List all keys (admin only)
- `POST /api/auth/keys` - Create a new key (admin only)
- `POST /api/auth/keys/{key_id}/revoke` - Revoke a key (admin only)
- `POST /api/auth/keys/{key_id}/rotate` - Rotate a key (admin only)
- `GET /api/auth/profile` - Get current key profile

## First-Time Setup

```bash
# Create an admin key
python -m auth.cli create --name "Admin Key" --admin --permissions "api:read api:write api:admin"

# Create regular API keys for services
python -m auth.cli create --name "Production API" --permissions "api:read api:write"
```

## Testing

Tests ensure that the authentication framework works as expected:

- Unit tests for all authentication components
- Integration tests for the JWT token flow
- Tests for key rotation and revocation
- Performance tests for authentication overhead

## Security Response Procedures

In case of compromised API keys:

1. Immediately revoke the compromised key
2. Review audit logs for unauthorized access
3. Rotate all admin keys as a precaution
4. Generate new keys for affected services
5. Update documentation of the incident

## Next Steps

Phase 6 will focus on runtime isolation with SELinux/AppArmor profiles and seccomp rules to further enhance the security posture of the Quantonium OS.