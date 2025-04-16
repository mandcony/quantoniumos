# Quantonium OS Cloud Runtime

A secure, high-performance quantum-inspired API for symbolic computing with advanced HPC modules.

## Overview

Quantonium OS Cloud Runtime provides a Flask-based API for accessing quantum-inspired computational resources. The system leverages advanced authentication, protected modules, and resonance-based computational techniques to deliver secure, high-performance symbolic computing.

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

### Integrating Proprietary HPC Modules

To integrate the proprietary high-performance modules:

1. Run the integration script:
   ```
   python integrate_quantonium.py path/to/quantonium_v2.zip
   ```
2. Restart the server for changes to take effect

## API Endpoints

All protected endpoints require the `X-API-Key` header.

- **GET /api/** - API status check
- **POST /api/encrypt** - Encrypt data using resonance techniques
- **POST /api/decrypt** - Decrypt data using resonance techniques
- **POST /api/simulate/rft** - Perform Resonance Fourier Transform
- **POST /api/entropy/sample** - Generate quantum-inspired entropy
- **POST /api/container/unlock** - Unlock symbolic containers

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

## Security Considerations

- Set a strong `QUANTONIUM_API_KEY` for production
- Limit CORS to trusted domains in production
- All responses include a timestamp and SHA-256 signature

## License

Proprietary - All rights reserved