"""
QuantoniumOS Bridge Module

This module provides the integration layer between the web API and desktop components,
allowing for seamless interaction between the web platform and desktop applications
while maintaining security and isolation.
"""

import os
import sys
import subprocess
import logging
import json
import tempfile
import time
import threading
import queue
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("quantonium_bridge")

# Constants
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "attached_assets")
PYTHON_EXEC = sys.executable

# Process tracking
active_processes: Dict[str, subprocess.Popen] = {}
process_outputs: Dict[str, queue.Queue] = {}
process_lock = threading.Lock()

class BridgeError(Exception):
    """Exception raised for errors in the Bridge module."""
    pass

def find_component_path(component_name: str) -> str:
    """
    Find the full path to a component script.
    
    Args:
        component_name: The name of the component script (e.g., "quantum_nova_system.py")
    
    Returns:
        The full path to the component script
    
    Raises:
        BridgeError: If the component script is not found
    """
    # First check in assets directory
    component_path = os.path.join(ASSETS_DIR, component_name)
    if os.path.exists(component_path):
        return component_path
    
    # Then check in apps directory
    app_dir = os.path.dirname(os.path.abspath(__file__))
    for root, _, files in os.walk(app_dir):
        if component_name in files:
            return os.path.join(root, component_name)
    
    # Not found
    raise BridgeError(f"Component not found: {component_name}")

def output_reader(process: subprocess.Popen, output_queue: queue.Queue, process_id: str):
    """
    Read output from a process and put it in a queue.
    
    Args:
        process: The subprocess.Popen object
        output_queue: The queue to store the output
        process_id: The ID of the process
    """
    try:
        for line in iter(process.stdout.readline, b''):
            line_text = line.decode('utf-8').rstrip()
            output_queue.put(line_text)
    except Exception as e:
        logger.error(f"Error reading output from process {process_id}: {str(e)}")
    finally:
        process.stdout.close()

def launch_component(
    component_name: str,
    args: List[str] = None,
    env_vars: Dict[str, str] = None,
    capture_output: bool = True,
    process_id: str = None
) -> str:
    """
    Launch a component script as a subprocess.
    
    Args:
        component_name: The name of the component script (e.g., "quantum_nova_system.py")
        args: Additional arguments to pass to the script
        env_vars: Additional environment variables to set
        capture_output: Whether to capture stdout/stderr
        process_id: Custom ID for the process (defaults to component_name)
    
    Returns:
        The process ID that can be used to query status or terminate
    
    Raises:
        BridgeError: If the component launch fails
    """
    try:
        # Find the component path
        component_path = find_component_path(component_name)
        logger.info(f"Component path: {component_path}")
        
        # Prepare command and environment
        cmd = [PYTHON_EXEC, component_path]
        if args:
            cmd.extend(args)
        
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
        
        # Set PYTHONPATH to include both assets and app directories
        python_path = [
            ASSETS_DIR,
            os.path.dirname(os.path.abspath(__file__))
        ]
        env["PYTHONPATH"] = os.pathsep.join(python_path) + os.pathsep + env.get("PYTHONPATH", "")
        
        # Generate process ID if not provided
        if not process_id:
            process_id = f"{component_name.split('.')[0]}_{int(time.time())}"
        
        # Launch the process
        logger.info(f"Launching component: {component_name} with args: {args}")
        
        if capture_output:
            process = subprocess.Popen(
                cmd,
                env=env,
                cwd=os.path.dirname(component_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Create a queue for the output
            output_q = queue.Queue()
            process_outputs[process_id] = output_q
            
            # Start a thread to read the output
            threading.Thread(
                target=output_reader,
                args=(process, output_q, process_id),
                daemon=True
            ).start()
        else:
            process = subprocess.Popen(
                cmd,
                env=env,
                cwd=os.path.dirname(component_path)
            )
        
        # Store the process
        with process_lock:
            active_processes[process_id] = process
        
        logger.info(f"Component launched with process ID: {process_id}")
        return process_id
    
    except Exception as e:
        logger.error(f"Error launching component {component_name}: {str(e)}")
        raise BridgeError(f"Failed to launch {component_name}: {str(e)}")

def terminate_component(process_id: str) -> bool:
    """
    Terminate a component process.
    
    Args:
        process_id: The ID of the process to terminate
    
    Returns:
        True if the process was terminated, False otherwise
    """
    with process_lock:
        if process_id not in active_processes:
            logger.warning(f"Process ID not found: {process_id}")
            return False
        
        process = active_processes[process_id]
        try:
            process.terminate()
            process.wait(timeout=5)
            logger.info(f"Process terminated: {process_id}")
            del active_processes[process_id]
            if process_id in process_outputs:
                del process_outputs[process_id]
            return True
        except Exception as e:
            logger.error(f"Error terminating process {process_id}: {str(e)}")
            try:
                process.kill()
                logger.info(f"Process killed: {process_id}")
                del active_processes[process_id]
                if process_id in process_outputs:
                    del process_outputs[process_id]
                return True
            except Exception as e2:
                logger.error(f"Error killing process {process_id}: {str(e2)}")
                return False

def get_process_output(process_id: str, timeout: float = 0.1) -> List[str]:
    """
    Get the output from a process.
    
    Args:
        process_id: The ID of the process
        timeout: Time to wait for output (in seconds)
    
    Returns:
        List of output lines
    
    Raises:
        BridgeError: If the process ID is not found
    """
    if process_id not in process_outputs:
        raise BridgeError(f"Process ID not found: {process_id}")
    
    output_q = process_outputs[process_id]
    lines = []
    try:
        while True:
            try:
                line = output_q.get(block=True, timeout=timeout)
                lines.append(line)
                output_q.task_done()
            except queue.Empty:
                break
    except Exception as e:
        logger.error(f"Error getting output from process {process_id}: {str(e)}")
    
    return lines

def is_process_running(process_id: str) -> bool:
    """
    Check if a process is still running.
    
    Args:
        process_id: The ID of the process
    
    Returns:
        True if the process is running, False otherwise
    """
    with process_lock:
        if process_id not in active_processes:
            return False
        
        process = active_processes[process_id]
        return process.poll() is None

def get_component_status(component_name: str = None) -> Dict[str, Any]:
    """
    Get the status of running components.
    
    Args:
        component_name: Optional filter for a specific component
    
    Returns:
        Dict with process IDs as keys and status info as values
    """
    status = {}
    with process_lock:
        for pid, process in active_processes.items():
            if component_name and not pid.startswith(component_name.split('.')[0]):
                continue
            
            is_running = process.poll() is None
            returncode = process.poll()
            
            status[pid] = {
                "running": is_running,
                "returncode": returncode if returncode is not None else None,
                "component": pid.split('_')[0],
                "start_time": int(pid.split('_')[1]) if '_' in pid else 0
            }
    
    return status

def cleanup_all_processes():
    """Clean up all running processes before exiting."""
    with process_lock:
        for pid, process in list(active_processes.items()):
            try:
                process.terminate()
                process.wait(timeout=2)
                logger.info(f"Process terminated during cleanup: {pid}")
            except Exception:
                try:
                    process.kill()
                    logger.info(f"Process killed during cleanup: {pid}")
                except Exception as e:
                    logger.error(f"Failed to kill process {pid} during cleanup: {str(e)}")

# Register cleanup handler
import atexit
atexit.register(cleanup_all_processes)

# -------------------------------------------------------------------------
# Specific component API wrappers
# -------------------------------------------------------------------------

def launch_quantonium_os() -> str:
    """
    Launch the full QuantoniumOS desktop environment.
    
    Returns:
        Process ID of the launched QuantoniumOS
    
    Raises:
        BridgeError: If the launch fails
    """
    return launch_component("quantonium_os_main.py", process_id="quantonium_os_main")

def launch_resonance_analyzer() -> str:
    """
    Launch the Resonance Analyzer desktop application.
    
    Returns:
        Process ID of the launched analyzer
    
    Raises:
        BridgeError: If the launch fails
    """
    return launch_component("q_resonance_analyzer.py", process_id="resonance_analyzer")

def launch_quantum_nova(qubit_count: int = 3) -> str:
    """
    Launch the Quantum Nova System with specified qubit count.
    
    Args:
        qubit_count: Number of qubits to initialize (default: 3)
    
    Returns:
        Process ID of the launched system
    
    Raises:
        BridgeError: If the launch fails
    """
    return launch_component(
        "quantum_nova_system.py",
        args=["--qubit-count", str(qubit_count)],
        process_id=f"quantum_nova_{qubit_count}"
    )

def launch_vibrational_engine() -> str:
    """
    Launch the Vibrational Engine.
    
    Returns:
        Process ID of the launched engine
    
    Raises:
        BridgeError: If the launch fails
    """
    return launch_component("vibrational_engine.py", process_id="vibrational_engine")

# -------------------------------------------------------------------------
# Test function
# -------------------------------------------------------------------------

def test_bridge():
    """Test the bridge functionality."""
    try:
        # Try to launch a simple component
        pid = launch_component("launch_resonance_analyzer.py", capture_output=True)
        logger.info(f"Test launch successful, process ID: {pid}")
        
        # Wait a bit and get output
        time.sleep(0.5)
        output = get_process_output(pid)
        logger.info(f"Process output: {output}")
        
        # Check status
        status = get_component_status()
        logger.info(f"Component status: {json.dumps(status, indent=2)}")
        
        # Terminate the process
        result = terminate_component(pid)
        logger.info(f"Terminate result: {result}")
        
        return True
    
    except Exception as e:
        logger.error(f"Bridge test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Run the test when the module is executed directly
    if test_bridge():
        print("✅ Bridge test successful")
    else:
        print("❌ Bridge test failed")
        sys.exit(1)