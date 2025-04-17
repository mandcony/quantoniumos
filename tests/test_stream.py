"""
Quantonium OS - SSE Stream Contract Tests

Tests for the Server-Sent Events (SSE) streaming endpoint that provides
real-time access to resonance data.
"""

import json
import time
import unittest
import threading
from datetime import datetime
from unittest.mock import patch

from flask import Flask
from flask.testing import FlaskClient

import routes
from backend.stream import resonance_data_generator, update_encrypt_data


class SSEContractTest(unittest.TestCase):
    """Test the SSE stream endpoint contracts and behavior"""
    
    def setUp(self):
        """Set up test client and app"""
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.app.register_blueprint(routes.api, url_prefix='/api')
        
        # Create test client
        self.client = self.app.test_client()
        
        # Sample waveform data for tests
        self.sample_waveform = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        # Mock the authentication requirement for testing
        self._patcher = patch('routes.require_jwt_auth', lambda f: f)
        self._patcher.start()
    
    def tearDown(self):
        """Clean up after tests"""
        self._patcher.stop()
    
    def test_stream_endpoint_headers(self):
        """Test that the stream endpoint returns the correct headers"""
        # Make request to the stream endpoint
        response = self.client.get('/api/stream/wave')
        
        # Check response code
        self.assertEqual(response.status_code, 200)
        
        # Check content type header
        self.assertEqual(response.headers['Content-Type'], 'text/event-stream')
        
        # Check cache headers
        self.assertEqual(response.headers['Cache-Control'], 'no-cache')
        self.assertEqual(response.headers['X-Accel-Buffering'], 'no')
        self.assertEqual(response.headers['Connection'], 'keep-alive')
    
    def test_stream_receives_events(self):
        """Test that the stream endpoint sends SSE events with correct format"""
        # Set up a background thread to consume events
        events = []
        stop_event = threading.Event()
        
        def consume_events():
            """Consume SSE events in a background thread"""
            response = self.client.get('/api/stream/wave', stream=True)
            for line in response.iter_lines():
                line = line.decode('utf-8')
                if stop_event.is_set():
                    break
                if line.startswith('data:'):
                    data = line[5:].strip()
                    events.append(json.loads(data))
                    if len(events) >= 3:  # Collect 3 events
                        break
        
        # Start thread to collect events
        thread = threading.Thread(target=consume_events)
        thread.daemon = True
        thread.start()
        
        # Wait for events to be collected (with timeout)
        start_time = time.time()
        while len(events) < 3 and time.time() - start_time < 2.0:
            time.sleep(0.1)
        
        # Signal thread to stop
        stop_event.set()
        thread.join(timeout=1.0)
        
        # Verify that events were received
        self.assertGreaterEqual(len(events), 1, "Should receive at least one event")
        
        # Verify event structure
        for event in events:
            self.assertIn('t', event, "Event missing timestamp")
            self.assertIn('amp', event, "Event missing amplitude data")
            self.assertIn('phase', event, "Event missing phase data")
            
            # Check data types
            self.assertIsInstance(event['t'], int, "Timestamp should be an integer")
            self.assertIsInstance(event['amp'], list, "Amplitude data should be a list")
            self.assertIsInstance(event['phase'], list, "Phase data should be a list")
            
            # Check data lengths match
            self.assertEqual(len(event['amp']), len(event['phase']), 
                            "Amplitude and phase arrays should have same length")
    
    def test_update_encrypt_data(self):
        """Test that update_encrypt_data properly updates the shared data state"""
        # Update encrypt data with test waveform
        update_encrypt_data(self.sample_waveform)
        
        # Collect one event from generator
        gen = resonance_data_generator()
        event_str = next(gen)
        
        # Parse the event data
        data_line = event_str.strip().split('\n')[0]
        self.assertTrue(data_line.startswith('data:'), "Event should start with 'data:'")
        
        event_data = json.loads(data_line[5:])  # Skip 'data:' prefix
        
        # Verify the event contains our sample waveform data
        self.assertEqual(len(event_data['amp']), len(self.sample_waveform),
                        "Waveform length should match")
        
        # Amplitudes should match our sample
        for i, val in enumerate(self.sample_waveform):
            self.assertEqual(event_data['amp'][i], val, 
                            f"Amplitude at index {i} should match")


if __name__ == '__main__':
    unittest.main()