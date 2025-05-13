"""
QuantoniumOS API Module

This module provides API endpoints for interacting with the QuantoniumOS desktop components
from the web interface. It acts as a gateway between the web platform and the desktop environment.
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from flask import Blueprint, request, jsonify, current_app

# Import bridge module for launching desktop components
from apps.bridge import (
    launch_quantonium_os,
    launch_resonance_analyzer,
    launch_quantum_nova,
    launch_vibrational_engine,
    get_process_output,
    is_process_running,
    terminate_component,
    get_component_status,
    BridgeError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("quantonium_api")

# Create Blueprint for registration with Flask app
quantonium_api = Blueprint('quantonium_api', __name__)

# Active process tracking for the API endpoints
active_api_processes: Dict[str, Dict[str, Any]] = {}

# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------

@quantonium_api.route('/launch/os', methods=['POST'])
def api_launch_os():
    """Launch the full QuantoniumOS desktop environment."""
    try:
        process_id = launch_quantonium_os()
        active_api_processes[process_id] = {
            "type": "os",
            "launched_at": time.time(),
            "status": "running"
        }
        return jsonify({
            "success": True,
            "process_id": process_id,
            "message": "QuantoniumOS desktop environment launched successfully"
        })
    except BridgeError as e:
        logger.error(f"Error launching QuantoniumOS: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@quantonium_api.route('/launch/analyzer', methods=['POST'])
def api_launch_analyzer():
    """Launch the Resonance Analyzer desktop application."""
    try:
        process_id = launch_resonance_analyzer()
        active_api_processes[process_id] = {
            "type": "analyzer",
            "launched_at": time.time(),
            "status": "running"
        }
        return jsonify({
            "success": True,
            "process_id": process_id,
            "message": "Resonance Analyzer launched successfully"
        })
    except BridgeError as e:
        logger.error(f"Error launching Resonance Analyzer: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@quantonium_api.route('/launch/nova', methods=['POST'])
def api_launch_nova():
    """Launch the Quantum Nova System."""
    try:
        # Get qubit count from request data
        data = request.get_json() or {}
        qubit_count = int(data.get('qubit_count', 3))
        
        # Validate qubit count
        if qubit_count < 1 or qubit_count > 150:
            return jsonify({
                "success": False,
                "error": "Qubit count must be between 1 and 150"
            }), 400
        
        # Launch the system
        process_id = launch_quantum_nova(qubit_count)
        active_api_processes[process_id] = {
            "type": "nova",
            "qubit_count": qubit_count,
            "launched_at": time.time(),
            "status": "running"
        }
        
        return jsonify({
            "success": True,
            "process_id": process_id,
            "qubit_count": qubit_count,
            "message": f"Quantum Nova System launched with {qubit_count} qubits"
        })
    except BridgeError as e:
        logger.error(f"Error launching Quantum Nova: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": f"Invalid parameters: {str(e)}"
        }), 400

@quantonium_api.route('/launch/vibrational', methods=['POST'])
def api_launch_vibrational():
    """Launch the Vibrational Engine."""
    try:
        process_id = launch_vibrational_engine()
        active_api_processes[process_id] = {
            "type": "vibrational",
            "launched_at": time.time(),
            "status": "running"
        }
        return jsonify({
            "success": True,
            "process_id": process_id,
            "message": "Vibrational Engine launched successfully"
        })
    except BridgeError as e:
        logger.error(f"Error launching Vibrational Engine: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@quantonium_api.route('/status', methods=['GET'])
def api_status():
    """Get status of all running components."""
    process_id = request.args.get('process_id')
    
    if process_id:
        # Get status of a specific process
        is_running = is_process_running(process_id)
        api_info = active_api_processes.get(process_id, {})
        
        return jsonify({
            "process_id": process_id,
            "running": is_running,
            "info": api_info
        })
    else:
        # Get status of all processes
        status = get_component_status()
        
        # Add API-specific information
        for pid, info in status.items():
            if pid in active_api_processes:
                info.update(active_api_processes[pid])
        
        return jsonify({
            "processes": status,
            "count": len(status)
        })

@quantonium_api.route('/terminate', methods=['POST'])
def api_terminate():
    """Terminate a running component."""
    data = request.get_json() or {}
    process_id = data.get('process_id')
    
    if not process_id:
        return jsonify({
            "success": False,
            "error": "Process ID is required"
        }), 400
    
    try:
        result = terminate_component(process_id)
        
        if result:
            # Remove from API tracking
            if process_id in active_api_processes:
                del active_api_processes[process_id]
            
            return jsonify({
                "success": True,
                "message": f"Process {process_id} terminated successfully"
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Failed to terminate process {process_id}"
            }), 500
    except Exception as e:
        logger.error(f"Error terminating process {process_id}: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@quantonium_api.route('/output', methods=['GET'])
def api_output():
    """Get output from a running component."""
    process_id = request.args.get('process_id')
    
    if not process_id:
        return jsonify({
            "success": False,
            "error": "Process ID is required"
        }), 400
    
    try:
        output = get_process_output(process_id)
        
        return jsonify({
            "success": True,
            "process_id": process_id,
            "output": output,
            "line_count": len(output)
        })
    except BridgeError as e:
        logger.error(f"Error getting output for process {process_id}: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 404
    except Exception as e:
        logger.error(f"Error getting output for process {process_id}: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Add the blueprint to the Flask app
def register_api(app):
    """Register the quantonium_api blueprint with the Flask app."""
    app.register_blueprint(quantonium_api, url_prefix='/api/quantonium')
    logger.info("QuantoniumOS API endpoints registered")