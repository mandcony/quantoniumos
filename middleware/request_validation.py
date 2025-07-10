"""
Quantonium OS - Input Validation Middleware

Implements NIST 800-53 SI-10 (Input Validation) compliant middleware
for validating and sanitizing all inputs to the application. This helps
protect against injection attacks, XSS, and other input-based vulnerabilities.
"""

import functools
import json
import logging
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from flask import abort, g, jsonify, request

# Import security logging utilities
from utils.security_logger import log_data_validation_failure

# Configure logger
logger = logging.getLogger("quantonium_validation")
logger.setLevel(logging.INFO)

# Regular expressions for validation
PATTERNS = {
    "alphanumeric": re.compile(r"^[a-zA-Z0-9]+$"),
    "email": re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"),
    "uuid": re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    ),
    "base64": re.compile(r"^[a-zA-Z0-9+/]+={0,2}$"),
    "numeric": re.compile(r"^[0-9]+$"),
    "decimal": re.compile(r"^[0-9]+(\.[0-9]+)?$"),
    "filepath": re.compile(r"^[a-zA-Z0-9_/.-]+$"),
    "api_key": re.compile(r"^[a-zA-Z0-9_-]{16,64}$"),
    "jwt": re.compile(r"^[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+$"),
    "hex": re.compile(r"^[0-9a-fA-F]+$"),
    "hostname": re.compile(
        r"^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9])$"
    ),
    "ip": re.compile(r"^(\d{1,3}\.){3}\d{1,3}$"),
}

# Common characters to check for in injection attacks
INJECTION_CHARS = [
    ";",
    "--",
    "/*",
    "*/",
    "union",
    "select",
    "from",
    "where",
    "drop",
    "truncate",
    "insert",
    "update",
    "delete",
    "exec",
    "script",
    "<script>",
    "</script>",
    "onerror",
    "onload",
    "eval(",
    "javascript:",
    "document.cookie",
    "alert(",
]


def validate_pattern(value: str, pattern_name: str) -> bool:
    """
    Validate a string against a named pattern.

    Args:
        value: String to validate
        pattern_name: Name of the pattern to validate against

    Returns:
        True if the string matches the pattern, False otherwise
    """
    pattern = PATTERNS.get(pattern_name)
    if not pattern:
        logger.error(f"Unknown pattern: {pattern_name}")
        return False

    if not isinstance(value, str):
        return False

    return bool(pattern.match(value))


def check_injection_attempts(value: str) -> List[str]:
    """
    Check a string for common injection patterns.

    Args:
        value: String to check

    Returns:
        List of injection patterns found (empty if none)
    """
    if not isinstance(value, str):
        return []

    found_patterns = []

    # Convert to lowercase for case-insensitive detection
    value_lower = value.lower()

    for pattern in INJECTION_CHARS:
        if pattern.lower() in value_lower:
            found_patterns.append(pattern)

    return found_patterns


def sanitize_string(value: str, allow_html: bool = False) -> str:
    """
    Sanitize a string to remove potential XSS or injection vectors.

    Args:
        value: String to sanitize
        allow_html: Whether to allow HTML tags

    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        return str(value)

    # Basic sanitization
    if not allow_html:
        # Replace < and > with their HTML entities
        value = value.replace("<", "&lt;").replace(">", "&gt;")

    # Replace other potentially harmful characters
    value = value.replace("&", "&amp;").replace('"', "&quot;").replace("'", "&#39;")

    return value


def validate_json_input(schema: Dict[str, Any]) -> Callable:
    """
    Decorator to validate JSON input against a schema.

    Args:
        schema: JSON schema definition

    Returns:
        Decorator function
    """

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not request.is_json:
                log_data_validation_failure(
                    input_source="request",
                    validation_error="Expected JSON content type",
                    client_ip=request.remote_addr,
                    resource=request.path,
                )
                return jsonify({"error": "Content-Type must be application/json"}), 415

            try:
                data = request.get_json()
            except Exception as e:
                log_data_validation_failure(
                    input_source="request",
                    validation_error=f"Invalid JSON: {str(e)}",
                    client_ip=request.remote_addr,
                    resource=request.path,
                )
                return jsonify({"error": "Invalid JSON format"}), 400

            if not data or not isinstance(data, dict):
                log_data_validation_failure(
                    input_source="request",
                    validation_error="JSON data must be a non-empty object",
                    client_ip=request.remote_addr,
                    resource=request.path,
                )
                return jsonify({"error": "JSON data must be a non-empty object"}), 400

            # Validate required fields
            errors = []
            for field, field_schema in schema.items():
                # Check if required and missing
                if field_schema.get("required", False) and field not in data:
                    errors.append(f"Missing required field: {field}")
                    continue

                # Skip validation for optional fields that aren't present
                if field not in data:
                    continue

                value = data[field]

                # Validate type
                expected_type = field_schema.get("type")
                if expected_type:
                    valid_type = False
                    if expected_type == "string" and isinstance(value, str):
                        valid_type = True
                    elif expected_type == "number" and isinstance(value, (int, float)):
                        valid_type = True
                    elif expected_type == "integer" and isinstance(value, int):
                        valid_type = True
                    elif expected_type == "boolean" and isinstance(value, bool):
                        valid_type = True
                    elif expected_type == "array" and isinstance(value, list):
                        valid_type = True
                    elif expected_type == "object" and isinstance(value, dict):
                        valid_type = True

                    if not valid_type:
                        errors.append(f"Field {field} must be of type {expected_type}")
                        continue

                # Validate string constraints
                if expected_type == "string" and isinstance(value, str):
                    # Validate pattern
                    pattern_name = field_schema.get("pattern")
                    if pattern_name and not validate_pattern(value, pattern_name):
                        errors.append(f"Field {field} has invalid format")

                    # Validate min length
                    min_length = field_schema.get("minLength")
                    if min_length is not None and len(value) < min_length:
                        errors.append(
                            f"Field {field} must be at least {min_length} characters"
                        )

                    # Validate max length
                    max_length = field_schema.get("maxLength")
                    if max_length is not None and len(value) > max_length:
                        errors.append(
                            f"Field {field} must be at most {max_length} characters"
                        )

                    # Check for injection attempts
                    injection_patterns = field_schema.get("checkInjection", False)
                    if injection_patterns:
                        found_patterns = check_injection_attempts(value)
                        if found_patterns:
                            errors.append(
                                f"Field {field} contains suspicious patterns: {', '.join(found_patterns)}"
                            )

                # Validate number constraints
                if expected_type in ("number", "integer") and isinstance(
                    value, (int, float)
                ):
                    # Validate minimum
                    minimum = field_schema.get("minimum")
                    if minimum is not None and value < minimum:
                        errors.append(f"Field {field} must be at least {minimum}")

                    # Validate maximum
                    maximum = field_schema.get("maximum")
                    if maximum is not None and value > maximum:
                        errors.append(f"Field {field} must be at most {maximum}")

                # Validate array constraints
                if expected_type == "array" and isinstance(value, list):
                    # Validate min items
                    min_items = field_schema.get("minItems")
                    if min_items is not None and len(value) < min_items:
                        errors.append(
                            f"Field {field} must have at least {min_items} items"
                        )

                    # Validate max items
                    max_items = field_schema.get("maxItems")
                    if max_items is not None and len(value) > max_items:
                        errors.append(
                            f"Field {field} must have at most {max_items} items"
                        )

                    # Validate items
                    items_schema = field_schema.get("items")
                    if items_schema and isinstance(items_schema, dict):
                        for i, item in enumerate(value):
                            # Validate item type
                            item_type = items_schema.get("type")
                            if item_type:
                                valid_type = False
                                if item_type == "string" and isinstance(item, str):
                                    valid_type = True
                                elif item_type == "number" and isinstance(
                                    item, (int, float)
                                ):
                                    valid_type = True
                                elif item_type == "integer" and isinstance(item, int):
                                    valid_type = True
                                elif item_type == "boolean" and isinstance(item, bool):
                                    valid_type = True
                                elif item_type == "array" and isinstance(item, list):
                                    valid_type = True
                                elif item_type == "object" and isinstance(item, dict):
                                    valid_type = True

                                if not valid_type:
                                    errors.append(
                                        f"Item {i} in field {field} must be of type {item_type}"
                                    )

            if errors:
                # Log validation failures
                log_data_validation_failure(
                    input_source="json_schema",
                    validation_error="; ".join(errors),
                    client_ip=request.remote_addr,
                    resource=request.path,
                    metadata={"errors": errors},
                )
                return jsonify({"error": "Validation failed", "details": errors}), 400

            # If validation passes, add validated data to request context
            g.validated_data = data

            return f(*args, **kwargs)

        return wrapper

    return decorator


def validate_route_params(**param_rules) -> Callable:
    """
    Decorator to validate URL parameters.

    Args:
        **param_rules: Rules for validating each parameter

    Returns:
        Decorator function
    """

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            errors = []

            for param_name, rules in param_rules.items():
                if param_name not in kwargs:
                    if rules.get("required", False):
                        errors.append(f"Missing required parameter: {param_name}")
                    continue

                value = kwargs[param_name]

                # Type validation
                param_type = rules.get("type")
                if param_type == "int":
                    try:
                        kwargs[param_name] = int(value)
                    except (ValueError, TypeError):
                        errors.append(f"Parameter {param_name} must be an integer")
                        continue
                elif param_type == "float":
                    try:
                        kwargs[param_name] = float(value)
                    except (ValueError, TypeError):
                        errors.append(f"Parameter {param_name} must be a number")
                        continue

                # Pattern validation
                pattern = rules.get("pattern")
                if pattern and isinstance(value, str):
                    if not validate_pattern(value, pattern):
                        errors.append(f"Parameter {param_name} has invalid format")

                # Check for injection attempts
                check_injection = rules.get("checkInjection", False)
                if check_injection and isinstance(value, str):
                    found_patterns = check_injection_attempts(value)
                    if found_patterns:
                        errors.append(
                            f"Parameter {param_name} contains suspicious patterns: {', '.join(found_patterns)}"
                        )

            if errors:
                # Log validation failures
                log_data_validation_failure(
                    input_source="route_params",
                    validation_error="; ".join(errors),
                    client_ip=request.remote_addr,
                    resource=request.path,
                    metadata={"errors": errors},
                )
                return jsonify({"error": "Validation failed", "details": errors}), 400

            return f(*args, **kwargs)

        return wrapper

    return decorator


def validate_form_data(**field_rules) -> Callable:
    """
    Decorator to validate form data.

    Args:
        **field_rules: Rules for validating each field

    Returns:
        Decorator function
    """

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            errors = []

            for field_name, rules in field_rules.items():
                if field_name not in request.form:
                    if rules.get("required", False):
                        errors.append(f"Missing required field: {field_name}")
                    continue

                value = request.form[field_name]

                # Pattern validation
                pattern = rules.get("pattern")
                if pattern:
                    if not validate_pattern(value, pattern):
                        errors.append(f"Field {field_name} has invalid format")

                # Check for injection attempts
                check_injection = rules.get("checkInjection", False)
                if check_injection:
                    found_patterns = check_injection_attempts(value)
                    if found_patterns:
                        errors.append(
                            f"Field {field_name} contains suspicious patterns: {', '.join(found_patterns)}"
                        )

            if errors:
                # Log validation failures
                log_data_validation_failure(
                    input_source="form_data",
                    validation_error="; ".join(errors),
                    client_ip=request.remote_addr,
                    resource=request.path,
                    metadata={"errors": errors},
                )
                return jsonify({"error": "Validation failed", "details": errors}), 400

            return f(*args, **kwargs)

        return wrapper

    return decorator


def sanitize_headers() -> Callable:
    """
    Decorator to sanitize and validate request headers.

    Returns:
        Decorator function
    """

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Check common header injection attack patterns
            for header, value in request.headers:
                injection_patterns = check_injection_attempts(value)
                if injection_patterns:
                    log_data_validation_failure(
                        input_source="headers",
                        validation_error=f"Header {header} contains suspicious patterns: {', '.join(injection_patterns)}",
                        client_ip=request.remote_addr,
                        resource=request.path,
                    )
                    return jsonify({"error": "Invalid header value"}), 400

            return f(*args, **kwargs)

        return wrapper

    return decorator
