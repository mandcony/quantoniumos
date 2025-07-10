"""
Quantonium OS - Security Logging Utilities - NIST Compliant

This module provides enhanced security logging capabilities that track
security-related events with detailed contextual information, following
NIST SP 800-53 Rev. 5 audit and accountability controls (AU family).
"""

import datetime
import hashlib
import json
import logging
import os
import platform
import socket
import sys
import time
import traceback
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# Import Flask request object for request logging
try:
    from flask import request
except ImportError:
    request = None
    print("Flask not available - request logging will be limited")

# Configure the security logger
logger = logging.getLogger("quantonium_security_events")
logger.setLevel(logging.INFO)

# Create console handler if not in production
if os.environ.get("FLASK_ENV") != "production":
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# Ensure log directory exists
log_dir = os.path.join(os.getcwd(), "logs", "security")
os.makedirs(log_dir, exist_ok=True)

# Add file handler for persistent security logs
try:
    # Create a rotating file handler that rotates daily
    from logging.handlers import TimedRotatingFileHandler

    security_log_file = os.path.join(log_dir, "security.log")
    file_handler = TimedRotatingFileHandler(
        security_log_file,
        when="midnight",
        interval=1,
        backupCount=90,  # Keep 90 days of logs (NIST AU-11)
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
except Exception as e:
    sys.stderr.write(f"Failed to configure security log file: {str(e)}\n")


# Security event types for categorization (aligned with NIST SP 800-53)
class SecurityEventType(str, Enum):
    """Enumeration of security event types based on NIST categories"""

    AUTHENTICATION = "authentication"  # AU-14, IA-2
    AUTHORIZATION = "authorization"  # AC-3
    ACCESS_DENIED = "access_denied"  # AC-7
    DATA_ACCESS = "data_access"  # AU-3
    CONFIGURATION_CHANGE = "configuration_change"  # CM-5
    CRYPTO_OPERATION = "crypto_operation"  # SC-13
    RATE_LIMIT = "rate_limit"  # SC-5
    SUSPICIOUS_ACTIVITY = "suspicious_activity"  # SI-4
    SYSTEM_ERROR = "system_error"  # AU-5
    AUDIT = "audit"  # AU-2
    DATA_VALIDATION = "data_validation"  # SI-10
    PRIVILEGED_FUNCTION = "privileged_function"  # AC-6
    BOUNDARY_VIOLATION = "boundary_violation"  # SC-7


# Outcome types
class SecurityOutcome(str, Enum):
    """Enumeration of security event outcomes"""

    SUCCESS = "success"
    FAILURE = "failure"
    BLOCKED = "blocked"
    WARNING = "warning"
    UNKNOWN = "unknown"
    BYPASS_ATTEMPT = "bypass_attempt"
    TAMPERING_ATTEMPT = "tampering_attempt"


# NIST Impact Levels
class ImpactLevel(str, Enum):
    """NIST FIPS 199 impact levels"""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


def get_system_info() -> Dict[str, str]:
    """
    Get basic system information for contextual logging.
    Does not include sensitive information.

    Returns:
        Dictionary with basic system information
    """
    return {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
    }


def generate_event_id() -> str:
    """
    Generate a unique event ID for correlation.

    Returns:
        Unique event ID
    """
    # Combine current time with a random UUID
    timestamp = datetime.datetime.now().isoformat()
    random_component = str(uuid.uuid4())
    raw_id = f"{timestamp}-{random_component}"

    # Hash to create a fixed-length ID
    return hashlib.sha256(raw_id.encode()).hexdigest()[:24]


def log_security_event(
    event_type: SecurityEventType,
    message: str,
    outcome: SecurityOutcome = SecurityOutcome.UNKNOWN,
    level: int = logging.INFO,
    target_resource: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    security_labels: Optional[List[str]] = None,
    impact_level: ImpactLevel = ImpactLevel.LOW,
    user_id: Optional[str] = None,
    source_ip: Optional[str] = None,
    correlation_id: Optional[str] = None,
    include_stack_trace: bool = False,
):
    """
    Log a security event with enriched context following NIST guidelines.

    Args:
        event_type: Type of security event from SecurityEventType enum
        message: Human-readable description of the event
        outcome: Outcome of the event from SecurityOutcome enum
        level: Logging level (default: INFO)
        target_resource: Resource that was targeted by the event
        metadata: Additional contextual information as a dictionary
        security_labels: List of security labels/tags for the event
        impact_level: NIST FIPS 199 impact level of the event
        user_id: ID of the user associated with the event (if applicable)
        source_ip: Source IP address for the event (if applicable)
        correlation_id: ID for correlating related events
        include_stack_trace: Whether to include stack trace for errors
    """
    # Generate an event ID for correlation if not provided
    event_id = correlation_id or generate_event_id()

    # Get current timestamp in ISO format with TZ info for audit compliance
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Create a comprehensive event record
    event_data = {
        "event_id": event_id,
        "timestamp": timestamp,
        "event_type": str(event_type),
        "message": message,
        "outcome": str(outcome),
        "impact_level": str(impact_level),
        "system_info": get_system_info(),
        "target_resource": target_resource,
    }

    # Add user information if available
    if user_id:
        event_data["user_id"] = user_id

    # Add source IP if available
    if source_ip:
        event_data["source_ip"] = source_ip

    # Add stack trace for errors if requested
    if include_stack_trace and level >= logging.ERROR:
        event_data["stack_trace"] = traceback.format_stack()

    # Add optional fields if provided
    if metadata:
        event_data["metadata"] = metadata

    if security_labels:
        event_data["security_labels"] = security_labels

    # Log the event at the appropriate level
    log_entry = json.dumps(event_data)

    if level == logging.DEBUG:
        logger.debug(log_entry)
    elif level == logging.INFO:
        logger.info(log_entry)
    elif level == logging.WARNING:
        logger.warning(log_entry)
    elif level == logging.ERROR:
        logger.error(log_entry)
    elif level == logging.CRITICAL:
        logger.critical(log_entry)
    else:
        logger.info(log_entry)

    # For high-impact security events, flush logs immediately
    if str(impact_level) == str(ImpactLevel.HIGH):
        for handler in logger.handlers:
            handler.flush()

    return event_id


def sanitize_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize potentially sensitive data from logs.

    Args:
        data: Dictionary with data to sanitize

    Returns:
        Sanitized dictionary
    """
    sensitive_keys = [
        "password",
        "token",
        "secret",
        "key",
        "credential",
        "auth",
        "apikey",
        "api_key",
        "private",
        "ssn",
        "creditcard",
        "credit_card",
        "card_number",
        "cvv",
        "jwt",
        "access_token",
    ]

    sanitized = {}

    for k, v in data.items():
        # Check if this key contains any sensitive keywords
        key_lower = k.lower()
        is_sensitive = any(s in key_lower for s in sensitive_keys)

        if is_sensitive:
            if isinstance(v, str):
                # Mask the value, keeping only type and length information
                sanitized[k] = f"[REDACTED:{type(v).__name__}:{len(v)}]"
            else:
                sanitized[k] = f"[REDACTED:{type(v).__name__}]"
        elif isinstance(v, dict):
            # Recursively sanitize nested dictionaries
            sanitized[k] = sanitize_sensitive_data(v)
        elif isinstance(v, list) and all(isinstance(i, dict) for i in v):
            # Sanitize list of dictionaries
            sanitized[k] = [sanitize_sensitive_data(i) for i in v]
        else:
            sanitized[k] = v

    return sanitized


def log_rate_limit_exceeded(
    client_ip: str,
    resource: str,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
):
    """
    Specialized method to log rate limit exceeded events for NIST AC-7.

    Args:
        client_ip: IP address of the client
        resource: Resource that triggered the rate limit
        message: Optional custom message
        metadata: Additional metadata
        user_id: ID of the user (if known)
    """
    if not message:
        message = f"Rate limit exceeded for {client_ip} on {resource}"

    if not metadata:
        metadata = {}

    metadata.update({"client_ip": client_ip, "rate_limited_resource": resource})

    # Sanitize any potentially sensitive data
    safe_metadata = sanitize_sensitive_data(metadata)

    log_security_event(
        event_type=SecurityEventType.RATE_LIMIT,
        message=message,
        outcome=SecurityOutcome.BLOCKED,
        level=logging.WARNING,
        target_resource=resource,
        metadata=safe_metadata,
        security_labels=["rate-limit-exceeded", "dos-protection"],
        impact_level=ImpactLevel.MODERATE,
        user_id=user_id,
        source_ip=client_ip,
    )


def log_suspicious_activity(
    activity_type: str,
    message: str,
    client_ip: Optional[str] = None,
    resource: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    impact_level: ImpactLevel = ImpactLevel.MODERATE,
):
    """
    Log potentially suspicious activity for review per NIST SI-4.

    Args:
        activity_type: Type of suspicious activity
        message: Description of the suspicious activity
        client_ip: IP address of the client
        resource: Resource involved in the activity
        metadata: Additional metadata
        user_id: ID of the user (if known)
        impact_level: NIST impact level of the suspicious activity
    """
    if not metadata:
        metadata = {}

    if client_ip:
        metadata["client_ip"] = client_ip

    # Sanitize any potentially sensitive data
    safe_metadata = sanitize_sensitive_data(metadata)

    log_security_event(
        event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
        message=message,
        outcome=SecurityOutcome.WARNING,
        level=logging.WARNING,
        target_resource=resource,
        metadata=safe_metadata,
        security_labels=["suspicious-activity", activity_type],
        impact_level=impact_level,
        user_id=user_id,
        source_ip=client_ip,
    )


def log_audit_event(
    action: str,
    message: str,
    user_id: Optional[str] = None,
    resource: Optional[str] = None,
    outcome: SecurityOutcome = SecurityOutcome.SUCCESS,
    metadata: Optional[Dict[str, Any]] = None,
    client_ip: Optional[str] = None,
):
    """
    Log an audit event for compliance purposes per NIST AU-2.

    Args:
        action: The action being audited
        message: Description of the audit event
        user_id: ID of the user performing the action
        resource: Resource affected by the action
        outcome: Outcome of the action
        metadata: Additional metadata
        client_ip: IP address of the client
    """
    if not metadata:
        metadata = {}

    metadata["audit_action"] = action

    # Sanitize any potentially sensitive data
    safe_metadata = sanitize_sensitive_data(metadata)

    log_security_event(
        event_type=SecurityEventType.AUDIT,
        message=message,
        outcome=outcome,
        level=logging.INFO,
        target_resource=resource,
        metadata=safe_metadata,
        security_labels=["audit", action],
        user_id=user_id,
        source_ip=client_ip,
    )


def log_data_validation_failure(
    input_source: str,
    validation_error: str,
    user_id: Optional[str] = None,
    client_ip: Optional[str] = None,
    resource: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Log input validation failures per NIST SI-10.

    Args:
        input_source: Source of the invalid input
        validation_error: Description of the validation error
        user_id: ID of the user submitting the input
        client_ip: IP address of the client
        resource: Resource targeted by the input
        metadata: Additional metadata
    """
    if not metadata:
        metadata = {}

    metadata["validation_source"] = input_source
    metadata["validation_error"] = validation_error

    # Sanitize any potentially sensitive data
    safe_metadata = sanitize_sensitive_data(metadata)

    message = f"Input validation failed for {input_source}: {validation_error}"

    log_security_event(
        event_type=SecurityEventType.DATA_VALIDATION,
        message=message,
        outcome=SecurityOutcome.FAILURE,
        level=logging.WARNING,
        target_resource=resource,
        metadata=safe_metadata,
        security_labels=["validation-failure", input_source],
        user_id=user_id,
        source_ip=client_ip,
    )


# Flask app setup function (compatibility with existing code)
def setup_security_logger(app, log_dir="/tmp/logs", log_level=logging.INFO):
    """
    Set up a security logger for the Flask application.

    This is a compatibility function for the existing application.
    The actual security logging functionality is handled by the log_security_event function.

    Args:
        app: Flask application
        log_dir: Directory for log files
        log_level: Logging level

    Returns:
        The configured logger
    """
    # Create the log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Add file handler for security log if not already added
    handler_exists = False
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename.endswith(
            "security.log"
        ):
            handler_exists = True
            break

    if not handler_exists:
        try:
            # Add a dedicated security log file
            security_file = os.path.join(log_dir, "security.log")
            handler = logging.FileHandler(security_file)
            handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            # Log initialization message
            logger.info("Security logger initialized")

            # Add security logging to the application if Flask is available
            if request is not None and hasattr(app, "before_request"):

                @app.before_request
                def log_request_info():
                    # Skip logging for static files and assets
                    if hasattr(request, "path"):
                        if request.path.startswith("/static/") or request.path.endswith(
                            (".js", ".css", ".png", ".jpg", ".ico")
                        ):
                            return

                        # Log basic request info
                        log_security_event(
                            event_type=SecurityEventType.AUDIT,
                            message=f"Request: {request.method} {request.path}",
                            outcome=SecurityOutcome.UNKNOWN,
                            level=logging.INFO,
                            target_resource=request.path,
                            metadata={
                                "method": request.method,
                                "ip": request.remote_addr,
                                "user_agent": request.headers.get(
                                    "User-Agent", "Unknown"
                                ),
                            },
                            security_labels=["http-request"],
                            impact_level=ImpactLevel.LOW,
                            source_ip=request.remote_addr,
                        )

        except Exception as e:
            logger.error(f"Error setting up security logger: {str(e)}")

    return logger
