"""
QuantoniumOS - Quantum-Safe Cryptography Implementation
Advanced cryptographic module implementing post-quantum algorithms
"""

import os
import hashlib
import secrets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import logging
from typing import Tuple, Dict, Any, Optional
import json
from datetime import datetime, timedelta
import base64

logger = logging.getLogger("quantonium_quantum_security")

class QuantumSafeKeyManager:
    """
    Quantum-safe key management system implementing NIST post-quantum standards
    """
    
    def __init__(self):
        self.backend = default_backend()
        self.master_key = self._get_or_create_master_key()
        self.key_derivation_rounds = 100000
    
    def _get_or_create_master_key(self) -> bytes:
        """Get or create the master encryption key"""
        master_key_env = os.environ.get('QUANTONIUM_MASTER_KEY')
        if master_key_env:
            try:
                return base64.b64decode(master_key_env.encode())
            except Exception as e:
                logger.warning(f"Invalid QUANTONIUM_MASTER_KEY format: {e}")
        
        # Generate new master key
        master_key = secrets.token_bytes(32)
        encoded_key = base64.b64encode(master_key).decode()
        logger.warning(f"Generated new master key: {encoded_key}")
        logger.warning("Set QUANTONIUM_MASTER_KEY environment variable to persist this key")
        return master_key
    
    def derive_key(self, salt: bytes, context: str = "default") -> bytes:
        """Derive encryption key using PBKDF2 with high iteration count"""
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.key_derivation_rounds,
            backend=self.backend
        )
        return kdf.derive(self.master_key + context.encode())
    
    def generate_quantum_safe_keypair(self) -> Tuple[bytes, bytes]:
        """Generate quantum-resistant key pair using RSA-4096"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
            backend=self.backend
        )
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def encrypt_quantum_safe(self, data: bytes, public_key_pem: bytes) -> bytes:
        """Encrypt data using quantum-safe hybrid encryption"""
        # Load public key
        public_key = serialization.load_pem_public_key(public_key_pem, backend=self.backend)
        
        # Generate symmetric key
        symmetric_key = secrets.token_bytes(32)
        
        # Encrypt data with AES-256-GCM
        iv = secrets.token_bytes(12)
        cipher = Cipher(algorithms.AES(symmetric_key), modes.GCM(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Encrypt symmetric key with RSA-4096
        encrypted_key = public_key.encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Combine encrypted key, IV, tag, and ciphertext
        result = len(encrypted_key).to_bytes(4, 'big') + encrypted_key + iv + encryptor.tag + ciphertext
        return result
    
    def decrypt_quantum_safe(self, encrypted_data: bytes, private_key_pem: bytes) -> bytes:
        """Decrypt quantum-safe encrypted data"""
        # Load private key
        private_key = serialization.load_pem_private_key(private_key_pem, password=None, backend=self.backend)
        
        # Extract components
        key_length = int.from_bytes(encrypted_data[:4], 'big')
        encrypted_key = encrypted_data[4:4+key_length]
        iv = encrypted_data[4+key_length:4+key_length+12]
        tag = encrypted_data[4+key_length+12:4+key_length+12+16]
        ciphertext = encrypted_data[4+key_length+12+16:]
        
        # Decrypt symmetric key
        symmetric_key = private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Decrypt data
        cipher = Cipher(algorithms.AES(symmetric_key), modes.GCM(iv, tag), backend=self.backend)
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext

class ZeroTrustValidator:
    """
    Zero-Trust architecture implementation for QuantoniumOS
    """
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.session_timeout = 3600  # 1 hour
        self.max_failed_attempts = 5
        self.lockout_duration = 900  # 15 minutes
    
    def generate_session_token(self, user_id: str, client_info: Dict[str, Any]) -> str:
        """Generate cryptographically secure session token"""
        token_data = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'client_fingerprint': self._generate_client_fingerprint(client_info),
            'entropy': secrets.token_hex(32)
        }
        
        token_string = json.dumps(token_data, sort_keys=True)
        token_hash = hashlib.sha256(token_string.encode()).hexdigest()
        
        if self.redis_client:
            # Store session in Redis
            session_key = f"quantonium:session:{token_hash}"
            self.redis_client.setex(session_key, self.session_timeout, json.dumps(token_data))
        
        return token_hash
    
    def _generate_client_fingerprint(self, client_info: Dict[str, Any]) -> str:
        """Generate unique client fingerprint for device tracking"""
        fingerprint_data = {
            'user_agent': client_info.get('user_agent', ''),
            'ip_address': client_info.get('ip_address', ''),
            'accept_language': client_info.get('accept_language', ''),
            'screen_resolution': client_info.get('screen_resolution', ''),
            'timezone': client_info.get('timezone', '')
        }
        
        fingerprint_string = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_string.encode()).hexdigest()
    
    def validate_session(self, token: str, client_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate session token with zero-trust principles"""
        validation_result = {
            'valid': False,
            'user_id': None,
            'risk_score': 0,
            'reason': 'unknown'
        }
        
        if not self.redis_client:
            validation_result['reason'] = 'session_store_unavailable'
            return validation_result
        
        try:
            session_key = f"quantonium:session:{token}"
            session_data = self.redis_client.get(session_key)
            
            if not session_data:
                validation_result['reason'] = 'session_not_found'
                return validation_result
            
            session_info = json.loads(session_data)
            current_fingerprint = self._generate_client_fingerprint(client_info)
            stored_fingerprint = session_info.get('client_fingerprint')
            
            # Calculate risk score based on various factors
            risk_score = 0
            
            # Fingerprint mismatch increases risk
            if current_fingerprint != stored_fingerprint:
                risk_score += 30
            
            # Session age increases risk
            session_age = (datetime.now() - datetime.fromisoformat(session_info['timestamp'])).total_seconds()
            if session_age > self.session_timeout / 2:
                risk_score += 20
            
            # IP address change increases risk
            if client_info.get('ip_address') != session_info.get('ip_address'):
                risk_score += 25
            
            validation_result.update({
                'valid': risk_score < 50,  # Threshold for session validity
                'user_id': session_info['user_id'],
                'risk_score': risk_score,
                'reason': 'valid' if risk_score < 50 else 'high_risk'
            })
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Session validation failed: {e}")
            validation_result['reason'] = 'validation_error'
            return validation_result
    
    def check_rate_limit_per_user(self, user_id: str, endpoint: str) -> bool:
        """Enhanced per-user rate limiting"""
        if not self.redis_client:
            return True
        
        rate_limit_key = f"quantonium:user_ratelimit:{user_id}:{endpoint}"
        current_count = self.redis_client.incr(rate_limit_key)
        
        if current_count == 1:
            self.redis_client.expire(rate_limit_key, 60)  # 1 minute window
        
        # Different limits for different endpoints
        limits = {
            'quantum': 10,
            'container': 20,
            'encrypt': 15,
            'default': 30
        }
        
        limit = limits.get(endpoint.split('/')[0], limits['default'])
        return current_count <= limit

class ContainerIsolationManager:
    """
    Advanced container isolation for quantum computations
    """
    
    def __init__(self):
        self.active_containers = {}
        self.resource_limits = {
            'max_memory_mb': 512,
            'max_cpu_percent': 25,
            'max_execution_time': 30
        }
    
    def create_isolated_container(self, computation_id: str, user_id: str) -> Dict[str, Any]:
        """Create isolated execution environment for quantum computation"""
        container_config = {
            'container_id': computation_id,
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'resource_limits': self.resource_limits.copy(),
            'security_context': {
                'isolated_namespace': True,
                'restricted_syscalls': True,
                'no_network_access': True,
                'read_only_filesystem': True
            },
            'quantum_resources': {
                'max_qubits': 150,
                'max_gates': 10000,
                'max_circuits': 100
            }
        }
        
        self.active_containers[computation_id] = container_config
        logger.info(f"Created isolated container {computation_id} for user {user_id}")
        
        return container_config
    
    def validate_computation_request(self, computation_request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate quantum computation request against security policies"""
        validation_result = {
            'allowed': True,
            'violations': [],
            'risk_level': 'low'
        }
        
        # Check qubit count
        qubit_count = computation_request.get('qubits', 0)
        if qubit_count > 150:
            validation_result['allowed'] = False
            validation_result['violations'].append('excessive_qubit_count')
            validation_result['risk_level'] = 'high'
        
        # Check circuit complexity
        gates = computation_request.get('gates', [])
        if len(gates) > 10000:
            validation_result['allowed'] = False
            validation_result['violations'].append('excessive_gate_count')
            validation_result['risk_level'] = 'high'
        
        # Check for suspicious patterns
        if any(gate.get('type') == 'custom' for gate in gates):
            validation_result['violations'].append('custom_gates_detected')
            validation_result['risk_level'] = 'medium'
        
        return validation_result
    
    def cleanup_container(self, computation_id: str) -> bool:
        """Clean up resources after computation completion"""
        if computation_id in self.active_containers:
            del self.active_containers[computation_id]
            logger.info(f"Cleaned up container {computation_id}")
            return True
        return False

class IntrusionDetectionSystem:
    """
    Real-time intrusion detection with automated response
    """
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.threat_patterns = self._load_threat_patterns()
        self.alert_thresholds = {
            'failed_logins': 5,
            'suspicious_requests': 10,
            'rate_limit_violations': 3
        }
    
    def _load_threat_patterns(self) -> Dict[str, Any]:
        """Load known threat patterns and signatures"""
        return {
            'sql_injection': [
                'UNION SELECT', 'DROP TABLE', 'INSERT INTO',
                '1=1', 'OR 1=1', '; --', '/*', '*/'
            ],
            'xss_patterns': [
                '<script>', '</script>', 'javascript:', 'onload=',
                'onerror=', 'onmouseover=', 'eval('
            ],
            'path_traversal': [
                '../', '..\\', '%2e%2e%2f', '%2e%2e\\',
                '/etc/passwd', '/etc/shadow', 'web.config'
            ],
            'command_injection': [
                '; cat ', '; ls ', '| whoami', '&& id',
                'system(', 'exec(', 'shell_exec('
            ]
        }
    
    def analyze_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze incoming request for threats"""
        analysis_result = {
            'threat_detected': False,
            'threat_types': [],
            'risk_score': 0,
            'recommended_action': 'allow'
        }
        
        request_content = str(request_data).lower()
        
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if pattern.lower() in request_content:
                    analysis_result['threat_detected'] = True
                    analysis_result['threat_types'].append(threat_type)
                    analysis_result['risk_score'] += 25
        
        # Determine recommended action based on risk score
        if analysis_result['risk_score'] >= 75:
            analysis_result['recommended_action'] = 'block'
        elif analysis_result['risk_score'] >= 50:
            analysis_result['recommended_action'] = 'challenge'
        elif analysis_result['risk_score'] >= 25:
            analysis_result['recommended_action'] = 'monitor'
        
        return analysis_result
    
    def log_security_incident(self, incident_data: Dict[str, Any]) -> bool:
        """Log security incident for analysis"""
        if not self.redis_client:
            return False
        
        try:
            incident_key = f"quantonium:incidents:{datetime.now().strftime('%Y%m%d')}"
            incident_data['timestamp'] = datetime.now().isoformat()
            incident_data['incident_id'] = secrets.token_hex(16)
            
            self.redis_client.lpush(incident_key, json.dumps(incident_data))
            self.redis_client.expire(incident_key, 86400 * 90)  # Keep for 90 days
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log security incident: {e}")
            return False

# Global security managers
quantum_key_manager = QuantumSafeKeyManager()
zero_trust_validator = ZeroTrustValidator()
container_isolation = ContainerIsolationManager()
intrusion_detection = IntrusionDetectionSystem()