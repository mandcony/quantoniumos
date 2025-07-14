# Quantonium OS 0.3.0-rc1 Release Notes

## Overview

Quantonium OS 0.3.0-rc1 is a release candidate representing the culmination of our comprehensive security hardening initiative. This version introduces significant security improvements across the entire stack, from the API authentication framework to container-level isolation, while maintaining full backward compatibility with existing clients.

## Key Highlights

- **Enhanced Authentication**: New JWT-based authentication with fine-grained permissions
- **Runtime Isolation**: Seccomp profiles and capability restrictions for container security
- **Audit & Monitoring**: Comprehensive logging and performance tracking
- **API Hardening**: Rate limiting, CSP, and security headers
- **Container Security**: Read-only filesystem with non-root execution

## Upgrade Path

### For API Consumers

1. **Authentication Changes**
   - You can continue using your existing API keys with the `X-API-Key` header
   - For improved security, request a JWT token and use the `Authorization: Bearer <token>` header:
     ```
     curl -X POST https://api.quantonium.io/api/auth/token \
       -H "Content-Type: application/json" \
       -H "X-API-Key: your-api-key" \
       -d '{}'
     ```

2. **Rate Limits**
   - New rate limits are in effect: 3600 requests per hour per API key
   - The response headers include your current limit status:
     - `X-RateLimit-Limit`: Your total limit
     - `X-RateLimit-Remaining`: Requests remaining
     - `X-RateLimit-Reset`: When your limit resets

3. **Security Headers**
   - Ensure your client respects the Content Security Policy headers
   - CORS is now restricted to authorized domains

### For Self-Hosted Deployments

1. **Container Updates**
   - Pull the new container image: `docker pull ghcr.io/quantonium/quantonium:0.3.0-rc1`
   - Verify container signature: `cosign verify ghcr.io/quantonium/quantonium:0.3.0-rc1`
   - Use the provided docker-compose.yml file for proper security configuration

2. **Environment Variables**
   - Required: `QUANTONIUM_API_KEY`, `SESSION_SECRET`, `DATABASE_URL`
   - Optional: `CORS_ORIGINS` (comma-separated list of allowed origins)

3. **Database Migration**
   - A database migration will run automatically on first startup
   - New tables: `api_keys` and `api_key_audit_logs`

4. **API Key Management**
   - Use the CLI tool to create your first admin key:
     ```
     docker exec -it quantonium-runtime python -m auth.cli create --name "Admin Key" --admin
     ```

## Breaking Changes

1. **Default Container Configuration**
   - Root filesystem is now read-only by default
   - All Linux capabilities are dropped
   - Custom log paths must be mounted to `/app/logs`

2. **API Changes**
   - `/api/health` endpoint now returns additional monitoring data
   - New `/api/metrics` endpoint requires admin permissions

## New Features

### Authentication Framework

- **JWT Authentication**: Secure token-based access with short-lived credentials
- **Key Management**: API to create, list, revoke, and rotate API keys
- **Permission System**: Fine-grained access control for all endpoints
- **Audit Logging**: All authentication events are recorded

### Container Security

- **Seccomp Profile**: Restrict container to only necessary system calls
- **Capability Management**: All Linux capabilities dropped by default
- **Process Isolation**: PID namespace isolation implemented
- **Read-only Filesystem**: Container runs with immutable root

### Monitoring & Observability

- **Structured Logging**: JSON logs with request IDs and performance data
- **Health Endpoint**: Enhanced health check with uptime data
- **Metrics Endpoint**: System resource usage and performance metrics

## Known Issues

1. The penetration testing script may fail in certain CI environments due to Docker-in-Docker restrictions.
2. Key rotation can experience race conditions under extremely high loads.
3. The iframe integration requires specific CSP headers on the host website.

## What's Next

Assuming this release candidate passes validation, we plan to release 0.3.0 final in two weeks. Future development will focus on:

1. Performance optimization for high-volume deployments
2. Enhanced monitoring dashboards
3. Additional authentication methods (OAuth, SAML)

## Verification

Verify the container image signature:
```
cosign verify ghcr.io/quantonium/quantonium:0.3.0-rc1
```

## Documentation

- [Authentication Guide](./authentication_guide.md)
- [Security Hardening - Phase 5](./security_hardening_phase5.md)
- [Security Hardening - Phase 6](./security_hardening_phase6.md)
- [Container Deployment Guide](./container_deployment.md)
- [API Key Management](./api_key_management.md)