# Quantonium OS Authentication Guide

## Overview

The Quantonium OS API uses a robust API key and JWT token-based authentication system. This document explains how to use the authentication system, manage API keys, and respond to security incidents.

## Authentication Flow

1. **API Key Creation**: Administrators create API keys via the CLI tool or admin API
2. **API Key Storage**: The key is stored as a salted hash in the database
3. **Token Generation**: Clients authenticate with their API key to get a short-lived JWT token
4. **Request Authentication**: Subsequent requests use the JWT token for authentication
5. **Token Renewal**: Tokens are renewed before expiry to maintain sessions

## API Key Format

API keys follow this format:

```
PREFIX.RANDOM_DATA
```

Example: `TcE8hG4L.UVZuhD7EF5C7VdXbvtNF2p7QrJZJKTm1Lza0FH2i3`

- The prefix (first 8 characters) is used for key lookup and is visible in logs
- The random part is used for verification and should never be logged

## Authentication Methods

### 1. API Key Authentication

Use the `X-API-Key` header with your API key:

```http
GET /api/encrypt
X-API-Key: PREFIX.RANDOM_DATA
```

### 2. JWT Token Authentication

First, obtain a token using your API key:

```http
POST /api/auth/token
X-API-Key: PREFIX.RANDOM_DATA
```

Response:

```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 3600,
  "token_type": "Bearer",
  "key_id": "550e8400-e29b-41d4-a716-446655440000",
  "permissions": "api:read api:write",
  "is_admin": false
}
```

Then use the token in the `Authorization` header:

```http
GET /api/encrypt
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## API Key Management

### CLI Tool

The system includes a comprehensive CLI tool for managing API keys:

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

### Admin API Endpoints

Administrators can also manage keys via API endpoints:

- `GET /api/auth/keys` - List all keys
- `POST /api/auth/keys` - Create a new key
- `POST /api/auth/keys/{key_id}/revoke` - Revoke a key
- `POST /api/auth/keys/{key_id}/rotate` - Rotate a key

## API Key Security Best Practices

1. **Regular Rotation**: Rotate API keys every 90 days
2. **Principle of Least Privilege**: Assign only necessary permissions
3. **Immediate Revocation**: Revoke keys immediately when no longer needed
4. **Audit Logging**: Monitor key usage regularly for unauthorized access
5. **Secure Storage**: Store API keys securely, never in source code
6. **Environment Segregation**: Use different keys for development, testing, and production

## Responding to Compromised API Keys

If you suspect an API key has been compromised:

1. **Immediate Revocation**: Revoke the compromised key immediately
   ```bash
   python -m auth.cli revoke --id KEY_ID
   ```

2. **Create Replacement**: Create a new key with the same permissions
   ```bash
   python -m auth.cli create --name "Replacement for compromised key" --permissions "api:read api:write"
   ```

3. **Audit Access**: Review audit logs to determine impact
   ```bash
   python -m auth.cli logs --id KEY_ID
   ```

4. **Investigate**: Determine how the key was compromised and address the underlying issue

5. **Document**: Record the incident and response for future reference

## Security Incident Contact

If you need to report a security issue, contact the security team at:

- **Email**: security@quantonium-os.internal
- **Incident Response Hotline**: 555-123-4567
- **Security Portal**: https://security.quantonium-os.internal