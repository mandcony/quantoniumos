"""
Quantonium OS - JSON Logger Test Suite

Tests for verifying structured JSON logging setup and functionality.
"""

import os
import json
import time
import tempfile
import unittest
from datetime import datetime

import flask
from flask import Flask, request, jsonify

from utils.json_logger import JSONFormatter, setup_json_logger

class TestJSONLogger(unittest.TestCase):
    """Test cases for JSON logging functionality"""
    
    def setUp(self):
        """Set up a test Flask app and temporary log file"""
        # Create a temporary directory for logs
        self.log_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.log_dir, 'quantonium_api.log')
        
        # Create a test Flask app
        self.app = Flask(__name__)
        
        # Set up the JSON logger
        self.log_handler = setup_json_logger(self.app, log_dir=self.log_dir)
        
        # Add some test routes
        @self.app.route('/test1')
        def test_route_1():
            return jsonify({"message": "Test 1"})
            
        @self.app.route('/test2')
        def test_route_2():
            return jsonify({"message": "Test 2"})
            
        @self.app.route('/test3', methods=['POST'])
        def test_route_3():
            return jsonify({"message": "Test 3", "data": request.get_json()})
        
        # Create a test client
        self.client = self.app.test_client()
    
    def tearDown(self):
        """Clean up temporary log files"""
        # Clean up any files in the temporary directory
        for filename in os.listdir(self.log_dir):
            file_path = os.path.join(self.log_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")
                
        # Remove the directory
        try:
            os.rmdir(self.log_dir)
        except Exception as e:
            print(f"Error removing directory {self.log_dir}: {e}")
    
    def test_log_file_created(self):
        """Test that the log file is created when a request is made"""
        # Make a test request
        self.client.get('/test1')
        
        # Check that the log file exists
        self.assertTrue(os.path.exists(self.log_file))
    
    def test_log_format_json(self):
        """Test that the log entries are valid JSON with required fields"""
        # Make a test request
        response = self.client.get('/test1')
        self.assertEqual(response.status_code, 200)
        
        # Read the log file and check the structure
        with open(self.log_file, 'r') as f:
            log_entry = json.loads(f.read().strip())
            
        # Check required fields
        self.assertIn('timestamp', log_entry)
        self.assertIn('epoch_ms', log_entry)
        self.assertIn('level', log_entry)
        self.assertIn('message', log_entry)
        self.assertIn('logger', log_entry)
        self.assertIn('route', log_entry)
        self.assertIn('status', log_entry)
        self.assertIn('elapsed_ms', log_entry)
        
        # Validate field values
        self.assertEqual(log_entry['route'], '/test1')
        self.assertEqual(log_entry['status'], 200)
        self.assertIsInstance(log_entry['elapsed_ms'], int)
    
    def test_multiple_requests_logged(self):
        """Test that multiple requests are properly logged"""
        # Make three test requests
        self.client.get('/test1')
        self.client.get('/test2')
        self.client.post('/test3', json={"key": "value"})
        
        # Read the log file and count entries
        with open(self.log_file, 'r') as f:
            log_lines = f.readlines()
        
        # There should be three entries
        self.assertEqual(len(log_lines), 3)
        
        # Parse each line as JSON and check route
        routes = []
        for line in log_lines:
            entry = json.loads(line)
            routes.append(entry['route'])
        
        # Check that all three routes are logged
        self.assertIn('/test1', routes)
        self.assertIn('/test2', routes)
        self.assertIn('/test3', routes)
    
    def test_request_body_hashing(self):
        """Test that request bodies are hashed in the log"""
        # Make a POST request with a JSON body
        test_data = {"secret": "confidential_data", "public": "public_data"}
        self.client.post('/test3', json=test_data)
        
        # Read the log file
        with open(self.log_file, 'r') as f:
            log_entry = json.loads(f.read().strip())
        
        # Check that the request body is hashed, not stored as plaintext
        self.assertIn('sha256_body', log_entry)
        self.assertNotIn('confidential_data', log_entry['sha256_body'])
        self.assertEqual(len(log_entry['sha256_body']), 64)  # SHA-256 is 64 hex chars
    
    def test_timing_header(self):
        """Test that the X-Request-Time header is added to responses"""
        # Make a test request
        response = self.client.get('/test1')
        
        # Check that the timing header is present
        self.assertIn('X-Request-Time', response.headers)
        
        # Check that the timing value is a valid integer
        timing_value = int(response.headers['X-Request-Time'])
        self.assertIsInstance(timing_value, int)
        self.assertGreaterEqual(timing_value, 0)  # Should be a non-negative integer


if __name__ == '__main__':
    unittest.main()