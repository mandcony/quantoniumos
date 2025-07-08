# Quantonium OS - Quantum Desktop Environment

[![License: Academic Research](https://img.shields.io/badge/License-Academic%20Research-blue.svg)](LICENSE)
[![Patent: Pending](https://img.shields.io/badge/Patent-USPTO%20%2319%2F169%2C399-orange.svg)](https://patents.uspto.gov/)
[![Version](https://img.shields.io/badge/Version-0.3.0--rc1-blue)](https://github.com/luisminier/quantonium-os)
[![API Status](https://img.shields.io/badge/API-Stable-brightgreen)](https://quantonium-os.replit.app/health)

[View Technical Paper](QuantoniumOS_Technical_Paper.pdf) | [FFT Performance Validation](QUANTONIUM_FFT_PERFORMANCE_VALIDATION.md) | [Visual Wave Encryption](VISUAL_WAVE_ENCRYPTION_GUIDE.md)

A cutting-edge quantum computing operating system that provides an advanced, visually immersive desktop environment for quantum computing research and development.

## Security Status
![Seccomp Enforced](https://img.shields.io/badge/Seccomp-Enforced-brightgreen) 
![No New Privileges](https://img.shields.io/badge/No--New--Privileges-Enforced-brightgreen)
![Read-only FS](https://img.shields.io/badge/Filesystem-Read--Only-brightgreen)
![No Capabilities](https://img.shields.io/badge/Capabilities-Dropped-brightgreen)
![PID Isolation](https://img.shields.io/badge/PID--Namespace-Isolated-brightgreen)
![JWT Auth](https://img.shields.io/badge/JWT--Auth-Implemented-brightgreen)
![Container Signed](https://img.shields.io/badge/Container-Signed-brightgreen)
![OpenAPI Spec](https://img.shields.io/badge/OpenAPI-Available-brightgreen)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/luisminier/quantonium-os.git
cd quantonium-os

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export DATABASE_URL="postgresql://user:password@localhost/quantonium_db"

# Start the application
python main.py
```

### Usage Examples

```bash
# Run with gunicorn for production
gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app

# Development mode
python main.py
```

### API Endpoints

```python
# Get quantum state
GET /api/quantum/state
# Returns: {"amplitude":"0.051","coherence":"5.4","entangledProcesses":42,"qubits":100,"resonance":"active"}

# Encrypt data using resonance
POST /api/encrypt
# Body: {"plaintext": "text to encrypt"}

# Real-time system stats
GET /health
# Returns: {"status": "healthy", "redis_available": true, "database_connected": true}
```

## 🎯 Why This Matters

QuantoniumOS represents a breakthrough in quantum computing accessibility, providing:

- **Real Hardware Integration**: Authentic entropy collection from system hardware
- **Patent-Protected Algorithms**: USPTO Application #19/169,399 geometric waveform hashing
- **Academic Research Platform**: Open-source foundation for quantum computing research
- **Production-Ready**: Enterprise-grade security and cross-platform deployment

## 🔬 Academic Research Focus

This project is open-sourced specifically for **academic research and validation**. We encourage universities, research institutions, and independent scientists to:

- Study the symbolic computing architectures
- Validate the resonance-based computation methods
- Extend the quantum simulation frameworks
- Research non-agentic AI implementations
- Contribute to symbolic entropy theory

## Overview

QuantoniumOS provides a Flask-based API for accessing quantum-inspired computational resources. The system leverages advanced authentication, protected modules, and resonance-based computational techniques to deliver secure, high-performance symbolic computing.

## Key Features

- **Secure Authentication**: API key validation for all protected endpoints
- **Resonance Encryption**: Quantum-inspired encryption using symbolic waveform technology
- **Resonance Fourier Transform**: Advanced signal analysis with CCP expansion
- **Quantum Entropy**: High-quality random number generation
- **Symbolic Containers**: Secure data containers with resonance-based access control
- **Squarespace Integration**: Embeddable frontend for Squarespace websites

## Architecture

The system is designed with a layered architecture to protect intellectual property:

1. **API Layer**: Flask endpoints with authentication and request validation
2. **Symbolic Interface**: Bridges the API with protected modules
3. **Protected Modules**: Implements the core algorithms with fallback mechanisms
4. **HPC Core**: High-Performance Computing modules (proprietary)

## Installation & Setup

### Requirements

- Python 3.11+
- Required packages (see `pyproject.toml`)
- Quantonium HPC modules (from `quantonium_v2.zip`)

### Basic Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables:
   ```
   export QUANTONIUM_API_KEY=your_secure_api_key
   export SESSION_SECRET=your_secure_session_secret
   ```
4. Start the server: `gunicorn --bind 0.0.0.0:5000 main:app`

### Docker Deployment

We provide pre-built Docker images with all dependencies and security hardening:

```bash
# Login to GitHub Container Registry
docker login ghcr.io -u USERNAME -p TOKEN

# Pull the latest image
docker pull ghcr.io/quantonium/quantonium:0.3.0-rc1

# Run with required environment variables
docker run -d --name quantonium-runtime \
  -p 5000:5000 \
  -e QUANTONIUM_API_KEY=your_secure_api_key \
  -e SESSION_SECRET=your_secure_session_secret \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  ghcr.io/quantonium/quantonium:0.3.0-rc1
```

For advanced deployment with all security features enabled, use docker-compose:

```bash
# Set environment variables in .env file
echo "QUANTONIUM_API_KEY=your_secure_api_key" > .env
echo "SESSION_SECRET=your_secure_session_secret" >> .env

# Start with docker-compose
docker-compose up -d
```

The container runs with a read-only filesystem, no privilege escalation, and minimal capabilities for maximum security.

### Integrating Proprietary HPC Modules

To integrate the proprietary high-performance modules:

1. Run the integration script:
   ```
   python integrate_quantonium.py path/to/quantonium_v2.zip
   ```
2. Restart the server for changes to take effect

## API Endpoints

All protected endpoints require the `X-API-Key` header or a JWT token.

- **GET /api/** - API status check
- **POST /api/encrypt** - Encrypt data using resonance techniques
- **POST /api/decrypt** - Decrypt data using resonance techniques
- **POST /api/simulate/rft** - Perform Resonance Fourier Transform
- **POST /api/entropy/sample** - Generate quantum-inspired entropy
- **POST /api/container/unlock** - Unlock symbolic containers
- **POST /api/auth/token** - Generate a JWT token using your API key
- **GET /api/stream/wave** - Stream real-time resonance data (SSE)

### API Documentation

Full API documentation is available at the following endpoints:

- **OpenAPI Specification**: `/openapi.json` - Machine-readable API specification
- **API Documentation UI**: `/docs` - Interactive Swagger UI for exploring the API
- **Release Notes**: [Changelog](CHANGELOG.md#030---2025-04-17) - Detailed version history with permalinks

The API documentation includes request/response schemas, authentication requirements, and endpoint descriptions. Use the Swagger UI to test API endpoints directly from your browser.

All endpoints return rate-limit headers (`X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`) and return appropriate status codes for authentication and authorization failures (401 for invalid API keys, 403 for revoked keys, and 498 for expired JWT tokens).

## Frontend Integration

### Embedding in Squarespace

To embed the Quantonium OS frontend in your Squarespace site:

1. Add an HTML block to your Squarespace page
2. Insert the following iframe code:

```html
<iframe 
  src="https://{YOUR-REPLIT-URL}.replit.app/frontend" 
  width="100%" 
  height="650px" 
  frameborder="0" 
  scrolling="auto">
</iframe>
```

3. Replace `{YOUR-REPLIT-URL}` with your actual Replit deployment URL

## Development & Testing

### Basic API Testing

Run the basic test script to verify API functionality:

```
python test_api.py
```

### Randomized Architecture Testing

For security-conscious testing that demonstrates the architecture's capabilities without exposing sensitive data:

```
python randomized_test.py
```

The randomized test script uses randomly generated inputs to test the API endpoints, ensuring the system works without exposing real user data or revealing implementation details. This helps protect the proprietary algorithms while still demonstrating the system's functionality.

### Security Testing & Validation

The repository includes several security testing scripts:

#### Penetration Testing

Run the container penetration testing script to validate security hardening:

```bash
./scripts/pentest.sh
```

This script attempts various container escape techniques and privilege escalation attacks against the running container. All tests should fail, confirming that the security measures are working correctly.

#### Container Security Scanning

Scan the container for security issues using Dockle:

```bash
./scripts/dockle_scan.sh
```

This script checks for container best practices and security issues, failing on WARN and FATAL level findings.

#### Vulnerability Scanning

Scan the container for vulnerabilities using Trivy:

```bash
./scripts/trivy_scan.sh
```

This script checks for vulnerabilities in the container image, failing on HIGH and CRITICAL severity findings.

#### End-to-End Smoke Tests

Run the E2E smoke tests to verify the API functionality in a live environment:

```bash
./scripts/smoke_test.py --url http://localhost:5000 --verbose
```

The smoke test script tests:
- API health endpoint
- OpenAPI spec validation
- API documentation access
- Authentication flow
- Encryption/decryption cycle
- Metrics endpoint

These tests ensure that all critical API functionality is working correctly after deployment. The smoke tests also run as part of the CI/CD pipeline for each release.

## Security Quick-Start

### Generate → Rotate → Revoke Keys in Three Commands
```bash
# 1. Generate an API key
curl -X POST https://api.quantonium.io/api/auth/keys \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -d '{"name":"Production API Key","permissions":"api:read api:write","expires_in_days":90}'

# 2. Rotate an API key (generates new key with same permissions)
curl -X POST https://api.quantonium.io/api/auth/keys/${KEY_ID}/rotate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}"

# 3. Revoke a compromised key
curl -X POST https://api.quantonium.io/api/auth/keys/${KEY_ID}/revoke \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -d '{"reason":"Security incident #123"}'
```

### Demo vs. Production Rate Limits
**⚠️ Public demo key is rate-limited to 60 requests per hour and may be revoked without notice**

Production API keys have the following rate limits:
- Standard tier: 3,600 requests per hour
- Enterprise tier: 36,000 requests per hour

All endpoints return rate limit headers:
```
X-RateLimit-Limit: 3600
X-RateLimit-Remaining: 3599
X-RateLimit-Reset: 3600
```

### Supported Cryptographic Primitives

| Category | Algorithm | Purpose |
|----------|-----------|---------|
| Hashing | SHA-256 | Message digests, API response signatures |
| HMAC | HMAC-SHA-256 | JWT token signing |
| CSPRNG | secrets-based | Secure random generation for keys |
| Container Signing | Cosign | Container image verification |

### Runtime Isolation
[![Seccomp Enforced](https://img.shields.io/badge/Seccomp-Enforced-brightgreen)](seccomp.json) [![Dockle Scan](https://img.shields.io/badge/Dockle-No%20Critical%20Findings-brightgreen)](docs/security_scan_results.md)

## Security Considerations

- Set a strong `QUANTONIUM_API_KEY` for production
- Limit CORS to trusted domains in production
- All responses include a timestamp and SHA-256 signature
- Use the provided Docker container for enhanced security (read-only filesystem, non-root user)
- Verify container signature: `cosign verify ghcr.io/quantonium/quantonium:latest`
- Review our [Security Policy](SECURITY.md) for vulnerability disclosure process
- All dependencies are pinned to exact versions to prevent supply chain attacks
- Security audit is performed in CI/CD pipeline before each release

### Runtime Isolation & Sandbox Hardening

The Quantonium OS container is hardened with multiple layers of security:

1. **Seccomp Profile**: Restricts available syscalls to a minimal set required for operation
   - Blocks dangerous syscalls like ptrace, mount, and namespace operations
   - Profile defined in `seccomp.json` at the repository root

2. **Dropped Capabilities**: All Linux capabilities are dropped
   - Container runs with no special privileges
   - Even common capabilities like NET_BIND_SERVICE are not needed

3. **Read-only Filesystem**: Container root filesystem is mounted read-only
   - Only specific tmpfs volumes for logs and temp files are writable
   - Prevents unauthorized modifications to application code

4. **Non-root User**: Container runs as the unprivileged `quant` user (UID 1001)
   - No ability to modify system files or access protected resources

5. **PID Namespace Isolation**: Container has its own isolated process namespace
   - Prevents visibility of host processes
   - Improves container isolation

### Updating the Seccomp Profile

If you need to modify the seccomp profile:

1. Edit the `seccomp.json` file with required changes
2. Restart the container: `docker-compose down && docker-compose up -d`
3. Validate with the penetration testing script: `./scripts/pentest.sh`

The seccomp profile should only include syscalls actually needed by the application. To audit syscalls:

```bash
# Create a test container with seccomp tracing enabled
docker run --rm -it --security-opt seccomp=unconfined \
  --cap-add SYS_PTRACE quantonium:latest \
  sh -c 'strace -c -f -S name gunicorn --bind 0.0.0.0:5000 main:app'
```

This will output a summary of syscalls used, which can be used to refine the profile.

## License

Proprietary - All rights reserved