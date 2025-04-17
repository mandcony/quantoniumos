# Changelog

All notable changes to the Quantonium OS project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0-rc1] - 2025-04-17

### Security Overhaul

This release candidate represents a comprehensive security hardening initiative across the entire Quantonium OS infrastructure.

#### Added

- **Authentication Framework**
  - JWT/HMAC based authentication with per-key JWT secrets
  - API key management system with rotation and revocation capabilities
  - Fine-grained permission system for API endpoints
  - Administrative CLI tool for key management
  - Comprehensive audit logging for all authentication events
  - Database integration for secure key storage

- **Runtime Isolation**
  - Custom seccomp profile to restrict available syscalls
  - All Linux capabilities dropped for container isolation
  - Read-only container filesystem with tmpfs mounts for logs
  - Non-root user execution with UID 1001
  - PID namespace isolation for process security

- **Monitoring & Observability**
  - Structured JSON logging with request IDs and API key tracking
  - Response timing middleware for performance monitoring
  - Enhanced health check endpoint with uptime information
  - New metrics endpoint exposing runtime resource usage
  - New audit logging for all key operations

- **Container Security**
  - Pinned dependency versions to prevent supply chain attacks
  - Multistage Docker builds with minimal runtime image
  - Non-root container user
  - Docker security scanning with Dockle and Trivy
  - Penetration testing script to validate security controls

- **API Hardening**
  - HSTS enabled for secure transport
  - Content Security Policy implemented
  - Rate limiting on all endpoints
  - Security headers enforced
  - CORS restrictions for authorized domains only
  - Request validation for all inputs

- **Documentation**
  - New security documentation
  - Updated container deployment guides
  - Authentication system documentation
  - Key management guides
  - OpenAPI specification

#### Changed

- Flask app structure refactored for better security
- Authentication flow now uses JWT tokens
- API request validation improved
- Default port binding security enhanced
- Log handling for better security event visibility
- Database schema updated for API key storage

#### Fixed

- Timing attack vulnerabilities in authentication
- CSRF protection improved
- Secure random number generation
- Dependency vulnerabilities patched
- Docker container privilege escalation vectors

## [0.2.0] - 2025-03-15

- Initial public API release with core functionality
- Basic container support
- API key authentication

## [0.1.0] - 2025-02-01

- Initial internal release