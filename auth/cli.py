#!/usr/bin/env python3
"""
Quantonium OS - API Key Management CLI

This CLI tool enables operations staff to create, list, revoke, and rotate API keys
for the Quantonium OS platform without direct database access.

Usage:
    python -m auth.cli create --name "Production API Key" --admin
    python -m auth.cli list
    python -m auth.cli revoke --id KEY_ID
    python -m auth.cli rotate --id KEY_ID
"""

import argparse
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from tabulate import tabulate

from auth.models import APIKey, APIKeyAuditLog, db


def create_app():
    """Create a minimal Flask app for CLI database access"""
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    return app


@contextmanager
def app_context():
    """Context manager to handle Flask app context"""
    app = create_app()
    with app.app_context():
        yield db


def format_datetime(dt):
    """Format datetime for display"""
    if not dt:
        return "Never"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def create_key(args):
    """Create a new API key"""
    with app_context():
        # Create key with specified parameters
        expires_in_days = args.expires_in_days if args.expires_in_days else None

        key, raw_key = APIKey.create(
            name=args.name,
            description=args.description,
            expires_in_days=expires_in_days,
            permissions=args.permissions,
            is_admin=args.admin,
        )

        # Log creation
        APIKeyAuditLog.log(
            key,
            "created",
            details=f"Created by CLI user {os.environ.get('USER', 'unknown')}",
        )

        print("\nAPI Key created successfully:\n")

        data = {
            "key_id": key.key_id,
            "name": key.name,
            "permissions": key.permissions,
            "is_admin": key.is_admin,
            "expires_at": (
                format_datetime(key.expires_at) if key.expires_at else "Never"
            ),
            "created_at": format_datetime(key.created_at),
            "api_key": raw_key,  # This is displayed only once at creation time
        }

        # Print as table for readability
        rows = [[k, v] for k, v in data.items()]
        print(tabulate(rows, tablefmt="pretty"))

        print("\n⚠️  IMPORTANT: Save this API key now! ⚠️")
        print("The complete key will not be displayed again.")
        print(f"\nAPI Key: {raw_key}\n")

        if args.json:
            print("\nJSON output:")
            print(json.dumps(data, indent=2))


def list_keys(args):
    """List all API keys"""
    with app_context():
        keys = APIKey.query.all()

        if not keys:
            print("No API keys found.")
            return

        data = []
        for key in keys:
            data.append(
                {
                    "key_id": key.key_id,
                    "prefix": key.key_prefix,
                    "name": key.name,
                    "is_admin": "Yes" if key.is_admin else "No",
                    "active": "Yes" if key.is_active else "No",
                    "revoked": "Yes" if key.revoked else "No",
                    "created": format_datetime(key.created_at),
                    "expires": format_datetime(key.expires_at),
                    "last_used": format_datetime(key.last_used_at),
                    "use_count": key.use_count,
                }
            )

        if args.json:
            print(json.dumps(data, indent=2))
        else:
            headers = [
                "Key ID",
                "Prefix",
                "Name",
                "Admin",
                "Active",
                "Revoked",
                "Created",
                "Expires",
                "Last Used",
                "Use Count",
            ]
            rows = [
                [
                    d["key_id"],
                    d["prefix"],
                    d["name"],
                    d["is_admin"],
                    d["active"],
                    d["revoked"],
                    d["created"],
                    d["expires"],
                    d["last_used"],
                    d["use_count"],
                ]
                for d in data
            ]
            print(tabulate(rows, headers=headers, tablefmt="pretty"))


def revoke_key(args):
    """Revoke an API key"""
    with app_context():
        key = APIKey.query.filter_by(key_id=args.id).first()

        if not key:
            print(f"Error: No API key found with ID {args.id}")
            return

        if key.revoked:
            print(f"Key {key.key_id} ({key.name}) is already revoked.")
            return

        # Confirm unless --force is used
        if not args.force:
            confirm = input(f"Revoke API key {key.key_id} ({key.name})? [y/N]: ")
            if confirm.lower() not in ["y", "yes"]:
                print("Operation cancelled.")
                return

        # Revoke the key
        key.revoke()

        # Log revocation
        APIKeyAuditLog.log(
            key,
            "revoked",
            details=f"Revoked by CLI user {os.environ.get('USER', 'unknown')}",
        )

        print(f"API key {key.key_id} ({key.name}) has been revoked.")


def rotate_key(args):
    """Rotate an API key"""
    with app_context():
        key = APIKey.query.filter_by(key_id=args.id).first()

        if not key:
            print(f"Error: No API key found with ID {args.id}")
            return

        # Confirm unless --force is used
        if not args.force:
            confirm = input(f"Rotate API key {key.key_id} ({key.name})? [y/N]: ")
            if confirm.lower() not in ["y", "yes"]:
                print("Operation cancelled.")
                return

        # Rotate the key
        new_key, raw_key = key.rotate()

        # Log rotation
        APIKeyAuditLog.log(
            key,
            "rotated",
            details=f"Rotated to {new_key.key_id} by CLI user {os.environ.get('USER', 'unknown')}",
        )

        print("\nAPI Key rotated successfully. New key details:\n")

        data = {
            "key_id": new_key.key_id,
            "name": new_key.name,
            "permissions": new_key.permissions,
            "is_admin": new_key.is_admin,
            "expires_at": (
                format_datetime(new_key.expires_at) if new_key.expires_at else "Never"
            ),
            "created_at": format_datetime(new_key.created_at),
            "api_key": raw_key,  # This is displayed only once at creation time
        }

        # Print as table for readability
        rows = [[k, v] for k, v in data.items()]
        print(tabulate(rows, tablefmt="pretty"))

        print("\n⚠️  IMPORTANT: Save this new API key now! ⚠️")
        print("The complete key will not be displayed again.")
        print(f"\nAPI Key: {raw_key}\n")

        print(f"The previous key ({key.key_id}) has been deactivated.")

        if args.json:
            print("\nJSON output:")
            print(json.dumps(data, indent=2))


def view_logs(args):
    """View audit logs for an API key"""
    with app_context():
        # Get the key
        key = APIKey.query.filter_by(key_id=args.id).first() if args.id else None

        # Query for logs
        query = APIKeyAuditLog.query

        if key:
            query = query.filter_by(api_key_id=key.id)

        # Apply limit
        if args.limit:
            query = query.limit(args.limit)

        # Order by timestamp descending
        logs = query.order_by(APIKeyAuditLog.timestamp.desc()).all()

        if not logs:
            print("No audit logs found.")
            return

        data = []
        for log in logs:
            data.append(
                {
                    "timestamp": format_datetime(log.timestamp),
                    "key_id": log.key_id,
                    "action": log.action,
                    "ip_address": log.ip_address or "-",
                    "request_path": log.request_path or "-",
                    "status_code": log.status_code or "-",
                    "details": log.details or "-",
                }
            )

        if args.json:
            print(json.dumps(data, indent=2))
        else:
            headers = [
                "Timestamp",
                "Key ID",
                "Action",
                "IP Address",
                "Path",
                "Status",
                "Details",
            ]
            rows = [
                [
                    d["timestamp"],
                    d["key_id"],
                    d["action"],
                    d["ip_address"],
                    d["request_path"],
                    d["status_code"],
                    d["details"],
                ]
                for d in data
            ]
            print(tabulate(rows, headers=headers, tablefmt="pretty"))


def main():
    """Main CLI entrypoint"""
    parser = argparse.ArgumentParser(description="Quantonium OS API Key Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new API key")
    create_parser.add_argument("--name", required=True, help="Name for the API key")
    create_parser.add_argument("--description", help="Description for the API key")
    create_parser.add_argument(
        "--expires-in-days", type=int, help="Key expiration in days"
    )
    create_parser.add_argument(
        "--permissions",
        default="api:read api:write",
        help="Space-separated permission list",
    )
    create_parser.add_argument(
        "--admin", action="store_true", help="Create an admin key with all permissions"
    )
    create_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List all API keys")
    list_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )

    # Revoke command
    revoke_parser = subparsers.add_parser("revoke", help="Revoke an API key")
    revoke_parser.add_argument("--id", required=True, help="ID of the key to revoke")
    revoke_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompt"
    )

    # Rotate command
    rotate_parser = subparsers.add_parser(
        "rotate", help="Rotate an API key (create new, deactivate old)"
    )
    rotate_parser.add_argument("--id", required=True, help="ID of the key to rotate")
    rotate_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompt"
    )
    rotate_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )

    # Logs command
    logs_parser = subparsers.add_parser("logs", help="View audit logs")
    logs_parser.add_argument("--id", help="Filter logs by key ID")
    logs_parser.add_argument(
        "--limit", type=int, default=50, help="Maximum number of logs to display"
    )
    logs_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )

    args = parser.parse_args()

    # Handle commands
    if args.command == "create":
        create_key(args)
    elif args.command == "list":
        list_keys(args)
    elif args.command == "revoke":
        revoke_key(args)
    elif args.command == "rotate":
        rotate_key(args)
    elif args.command == "logs":
        view_logs(args)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
