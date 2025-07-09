#!/bin/bash
# Quantonium OS - Startup Script for Replit

echo "[+] Launching Quantonium OS Cloud Runtime with Gunicorn (gevent)..."
gunicorn --bind 0.0.0.0:5000 --workers 4 --worker-class gevent --preload app:app
