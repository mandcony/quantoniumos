# QuantoniumOS Middleware

This directory contains middleware components for the QuantoniumOS system.

## Authentication Middleware

The `auth.py` file implements:

1. **Rate Limiter Middleware** - A WSGI middleware that provides IP-based rate limiting before requests reach the Flask application. This provides an additional layer of protection against brute force attacks and DoS attacks.

   - Current configuration: 30 requests per 60 seconds per IP address.
   - Responses with HTTP 429 Too Many Requests when limit is exceeded.
   - Includes Retry-After header to inform clients when they can retry.

2. **JWT Authentication Decorator** - A decorator function that enforces JWT authentication for protected routes.

   - Validates the presence and format of Authorization header.
   - Designed to be extensible for future JWT validation logic.

## Integration with Main Application

The middleware is integrated in the `main.py` file:

1. **WSGI Level Integration** - The rate limiter is applied at the WSGI level, so it can reject requests before they even reach the Flask application:

   ```python
   from middleware.auth import RateLimiter
   app.wsgi_app = RateLimiter(calls=30, period=60)(app.wsgi_app)
   ```

2. **Complementary to Flask-Limiter** - This middleware complements the Flask-Limiter configured in `security.py`, providing multiple layers of protection.

## Patent Compliance

This implementation satisfies the security hardening requirements specified in the Patent Manifest document, which calls for rate limiting to prevent brute force attacks against the Resonance API endpoints.