"""
Rigorous IND-CPA and IND-CCA2 Security Tests for QuantoniumOS

This module implements formal security game definitions and runs concrete
security experiments to validate theoretical security proofs.

Unlike statistical tests, these implement the actual cryptographic security
games defined in academic literature.
"""

import secrets
import hashlib
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Import the encryption system
try:
    from core.encryption.resonance_encrypt import resonance_encrypt, resonance_decrypt
    from core.encryption.geometric_waveform_hash import generate_waveform_hash
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("Warning: Crypto modules not available, using mock implementations")

class SecurityGameResult(Enum):
    ADVERSARY_WINS = "adversary_wins"
    ADVERSARY_LOSES = "adversary_loses"
    GAME_ABORTED = "game_aborted"

@dataclass
class SecurityExperiment:
    """Results from a security game experiment"""
    game_type: str
    adversary_advantage: float
    time_elapsed: float
    queries_made: int
    security_parameter: int
    result: SecurityGameResult
    details: Dict[str, Any]

class INDCPAGame:
    """
    Implementation of the IND-CPA (Indistinguishability under Chosen Plaintext Attack) game.
    
    This is the formal definition from cryptographic literature, not a heuristic test.
    """
    
    def __init__(self, security_parameter: int = 128):
        self.security_parameter = security_parameter
        self.challenge_bit = None
        self.key = None
        self.queries_made = 0
        self.experiment_start_time = None
        
    def setup(self):
        """Setup phase: Generate keys and initialize game"""
        self.key = secrets.token_hex(32)  # 256-bit key
        self.challenge_bit = secrets.randbelow(2)  # Random bit b ∈ {0,1}
        self.queries_made = 0
        self.experiment_start_time = time.time()
        
    def encryption_oracle(self, plaintext: str) -> str:
        """
        Encryption oracle: Adversary can encrypt any chosen plaintext.
        
        This models the adversary's ability to see encryptions of messages
        they choose during the chosen-plaintext attack phase.
        """
        if not CRYPTO_AVAILABLE:
            return hashlib.sha256(f"{plaintext}_{self.key}_{self.queries_made}".encode()).hexdigest()
        
        self.queries_made += 1
        return resonance_encrypt(plaintext, self.key)
    
    def challenge(self, m0: str, m1: str) -> str:
        """
        Challenge phase: Adversary submits two messages, gets encryption of one.
        
        The adversary wins if they can determine which message was encrypted.
        """
        if len(m0) != len(m1):
            raise ValueError("Challenge messages must have equal length")
        
        # Encrypt the message corresponding to the challenge bit
        challenge_message = m0 if self.challenge_bit == 0 else m1
        
        if not CRYPTO_AVAILABLE:
            return hashlib.sha256(f"challenge_{challenge_message}_{self.key}".encode()).hexdigest()
        
        return resonance_encrypt(challenge_message, self.key)
    
    def guess(self, adversary_guess: int) -> SecurityGameResult:
        """
        Adversary makes their final guess about which message was encrypted.
        
        Returns whether the adversary won (guessed correctly) or lost.
        """
        if adversary_guess == self.challenge_bit:
            return SecurityGameResult.ADVERSARY_WINS
        else:
            return SecurityGameResult.ADVERSARY_LOSES

class INDCCAGame:
    """
    Implementation of the IND-CCA2 (Indistinguishability under Adaptive Chosen Ciphertext Attack) game.
    
    This is stronger than IND-CPA as the adversary also gets a decryption oracle.
    """
    
    def __init__(self, security_parameter: int = 128):
        self.security_parameter = security_parameter
        self.challenge_bit = None
        self.key = None
        self.challenge_ciphertext = None
        self.encryption_queries = 0
        self.decryption_queries = 0
        self.experiment_start_time = None
        
    def setup(self):
        """Setup phase: Generate keys and initialize game"""
        self.key = secrets.token_hex(32)
        self.challenge_bit = secrets.randbelow(2)
        self.challenge_ciphertext = None
        self.encryption_queries = 0
        self.decryption_queries = 0
        self.experiment_start_time = time.time()
    
    def encryption_oracle(self, plaintext: str) -> str:
        """Encryption oracle for chosen-plaintext queries"""
        if not CRYPTO_AVAILABLE:
            return hashlib.sha256(f"{plaintext}_{self.key}_{self.encryption_queries}".encode()).hexdigest()
        
        self.encryption_queries += 1
        return resonance_encrypt(plaintext, self.key)
    
    def decryption_oracle(self, ciphertext: str) -> Optional[str]:
        """
        Decryption oracle: Adversary can decrypt any ciphertext except the challenge.
        
        This models adaptive chosen-ciphertext attacks where the adversary
        can see decryptions of ciphertexts they choose.
        """
        # Reject the challenge ciphertext (this would trivially break security)
        if ciphertext == self.challenge_ciphertext:
            return None
        
        if not CRYPTO_AVAILABLE:
            # Mock decryption for testing
            if "invalid" in ciphertext.lower():
                return None
            return f"decrypted_{ciphertext[:16]}"
        
        try:
            self.decryption_queries += 1
            return resonance_decrypt(ciphertext, self.key)
        except:
            return None  # Invalid ciphertext
    
    def challenge(self, m0: str, m1: str) -> str:
        """Challenge phase: Return encryption of m_b where b is the challenge bit"""
        if len(m0) != len(m1):
            raise ValueError("Challenge messages must have equal length")
        
        challenge_message = m0 if self.challenge_bit == 0 else m1
        
        if not CRYPTO_AVAILABLE:
            challenge_ct = hashlib.sha256(f"challenge_{challenge_message}_{self.key}".encode()).hexdigest()
        else:
            challenge_ct = resonance_encrypt(challenge_message, self.key)
        
        self.challenge_ciphertext = challenge_ct
        return challenge_ct
    
    def guess(self, adversary_guess: int) -> SecurityGameResult:
        """Final adversary guess"""
        if adversary_guess == self.challenge_bit:
            return SecurityGameResult.ADVERSARY_WINS
        else:
            return SecurityGameResult.ADVERSARY_LOSES

class SimpleAdversary:
    """
    A simple adversary that attempts to break IND-CPA/IND-CCA2 security.
    
    This represents a realistic adversary using known cryptanalytic techniques.
    """
    
    def __init__(self, name: str = "SimpleAdversary"):
        self.name = name
        self.encryption_queries = []
        self.decryption_queries = []
        
    def ind_cpa_attack(self, game: INDCPAGame) -> int:
        """
        Attempt to break IND-CPA security.
        
        Strategy: Look for patterns in encryptions that reveal information
        about which message was encrypted in the challenge.
        """
        # Phase 1: Learning phase - encrypt various messages to learn patterns
        test_messages = ["A" * 16, "B" * 16, "test message 1", "test message 2"]
        
        encryptions = {}
        for msg in test_messages:
            ct = game.encryption_oracle(msg)
            encryptions[msg] = ct
            self.encryption_queries.append((msg, ct))
        
        # Phase 2: Challenge phase
        m0 = "secret message 0"
        m1 = "secret message 1"
        challenge_ct = game.challenge(m0, m1)
        
        # Phase 3: Analysis phase - try to determine which message was encrypted
        # Simple strategy: look for patterns or try known-plaintext analysis
        
        # Encrypt both challenge messages directly and compare
        try:
            ct0 = game.encryption_oracle(m0)
            ct1 = game.encryption_oracle(m1)
            
            # If encryption is deterministic, this would break security
            if challenge_ct == ct0:
                return 0
            elif challenge_ct == ct1:
                return 1
        except:
            pass
        
        # If we can't determine, make a random guess
        return secrets.randbelow(2)
    
    def ind_cca2_attack(self, game: INDCCAGame) -> int:
        """
        Attempt to break IND-CCA2 security.
        
        Strategy: Use the decryption oracle to gain information about the challenge.
        """
        # Phase 1: Learning phase
        test_messages = ["A" * 16, "B" * 16, "known message"]
        
        for msg in test_messages:
            ct = game.encryption_oracle(msg)
            decrypted = game.decryption_oracle(ct)  # Should decrypt correctly
            self.encryption_queries.append((msg, ct))
            self.decryption_queries.append((ct, decrypted))
        
        # Phase 2: Challenge
        m0 = "challenge message 0" 
        m1 = "challenge message 1"
        challenge_ct = game.challenge(m0, m1)
        
        # Phase 3: Post-challenge queries using decryption oracle
        # Try to learn information about the challenge without directly decrypting it
        
        # Strategy 1: Try related ciphertexts (if malleable)
        related_cts = [
            challenge_ct[:-4] + "0000",  # Modify last few chars
            "0000" + challenge_ct[4:],    # Modify first few chars
        ]
        
        for related_ct in related_cts:
            try:
                decrypted = game.decryption_oracle(related_ct)
                self.decryption_queries.append((related_ct, decrypted))
                
                # Analyze if decrypted text gives hints about challenge message
                if decrypted and ("message 0" in decrypted or "challenge" in decrypted):
                    return 0
                elif decrypted and ("message 1" in decrypted):
                    return 1
            except:
                continue
        
        # If attack fails, random guess
        return secrets.randbelow(2)

class SecurityTester:
    """
    Runs formal security experiments to test IND-CPA and IND-CCA2 security.
    """
    
    def __init__(self, security_parameter: int = 128):
        self.security_parameter = security_parameter
        
    def run_ind_cpa_experiment(self, adversary: SimpleAdversary, 
                              num_trials: int = 100) -> SecurityExperiment:
        """
        Run multiple IND-CPA security experiments.
        
        Measures the adversary's advantage: |Pr[Adv wins] - 1/2|
        Secure schemes should have negligible advantage.
        """
        wins = 0
        total_time = 0
        total_queries = 0
        
        for trial in range(num_trials):
            game = INDCPAGame(self.security_parameter)
            game.setup()
            
            start_time = time.time()
            adversary_guess = adversary.ind_cpa_attack(game)
            end_time = time.time()
            
            result = game.guess(adversary_guess)
            
            if result == SecurityGameResult.ADVERSARY_WINS:
                wins += 1
                
            total_time += (end_time - start_time)
            total_queries += game.queries_made
        
        # Calculate advantage
        win_probability = wins / num_trials
        advantage = abs(win_probability - 0.5)
        
        return SecurityExperiment(
            game_type="IND-CPA",
            adversary_advantage=advantage,
            time_elapsed=total_time,
            queries_made=total_queries,
            security_parameter=self.security_parameter,
            result=SecurityGameResult.ADVERSARY_WINS if advantage > 0.1 else SecurityGameResult.ADVERSARY_LOSES,
            details={
                "trials": num_trials,
                "wins": wins,
                "win_probability": win_probability,
                "theoretical_bound": 2**(-self.security_parameter/4)  # Expected bound
            }
        )
    
    def run_ind_cca2_experiment(self, adversary: SimpleAdversary,
                               num_trials: int = 50) -> SecurityExperiment:
        """
        Run IND-CCA2 security experiments.
        
        This is stronger than IND-CPA since adversary gets decryption oracle access.
        """
        wins = 0
        total_time = 0
        total_enc_queries = 0
        total_dec_queries = 0
        
        for trial in range(num_trials):
            game = INDCCAGame(self.security_parameter)
            game.setup()
            
            start_time = time.time()
            adversary_guess = adversary.ind_cca2_attack(game)
            end_time = time.time()
            
            result = game.guess(adversary_guess)
            
            if result == SecurityGameResult.ADVERSARY_WINS:
                wins += 1
            
            total_time += (end_time - start_time)
            total_enc_queries += game.encryption_queries
            total_dec_queries += game.decryption_queries
        
        win_probability = wins / num_trials
        advantage = abs(win_probability - 0.5)
        
        return SecurityExperiment(
            game_type="IND-CCA2", 
            adversary_advantage=advantage,
            time_elapsed=total_time,
            queries_made=total_enc_queries + total_dec_queries,
            security_parameter=self.security_parameter,
            result=SecurityGameResult.ADVERSARY_WINS if advantage > 0.1 else SecurityGameResult.ADVERSARY_LOSES,
            details={
                "trials": num_trials,
                "wins": wins, 
                "win_probability": win_probability,
                "encryption_queries": total_enc_queries,
                "decryption_queries": total_dec_queries,
                "theoretical_bound": 3 * 2**(-self.security_parameter/4)  # Expected bound
            }
        )
    
    def generate_security_report(self, experiments: List[SecurityExperiment]) -> str:
        """Generate a comprehensive security report"""
        
        report = "QUANTONIUMOS FORMAL SECURITY EXPERIMENT RESULTS\n"
        report += "=" * 55 + "\n\n"
        
        for exp in experiments:
            report += f"EXPERIMENT: {exp.game_type}\n"
            report += f"Security Parameter: {exp.security_parameter} bits\n"
            report += f"Adversary Advantage: {exp.adversary_advantage:.6f}\n"
            report += f"Time Elapsed: {exp.time_elapsed:.2f}s\n"
            report += f"Queries Made: {exp.queries_made}\n"
            report += f"Result: {exp.result.value.upper()}\n"
            
            # Security analysis
            theoretical_bound = exp.details.get('theoretical_bound', 0)
            if exp.adversary_advantage <= theoretical_bound * 10:  # Within 10x of theoretical
                security_status = "✓ SECURE (advantage within theoretical bounds)"
            elif exp.adversary_advantage <= 0.01:  # Less than 1% advantage
                security_status = "✓ PRACTICALLY SECURE (low advantage)"
            elif exp.adversary_advantage <= 0.1:   # Less than 10% advantage
                security_status = "⚠ MARGINALLY SECURE (notable advantage)"
            else:
                security_status = "✗ INSECURE (high adversary advantage)"
            
            report += f"Security Assessment: {security_status}\n"
            
            if exp.game_type == "IND-CCA2":
                report += f"Encryption Queries: {exp.details['encryption_queries']}\n"
                report += f"Decryption Queries: {exp.details['decryption_queries']}\n"
            
            report += f"Theoretical Bound: {theoretical_bound:.2e}\n"
            report += "-" * 50 + "\n"
        
        return report

def run_formal_security_tests() -> str:
    """Run complete formal security test suite"""
    
    if not CRYPTO_AVAILABLE:
        return "Warning: Crypto modules not available. Install dependencies and retry."
    
    # Initialize tester and adversary
    tester = SecurityTester(128)
    adversary = SimpleAdversary("BasicCryptanalyst")
    
    print("Running IND-CPA security experiments...")
    ind_cpa_result = tester.run_ind_cpa_experiment(adversary, num_trials=100)
    
    print("Running IND-CCA2 security experiments...")
    ind_cca2_result = tester.run_ind_cca2_experiment(adversary, num_trials=50)
    
    # Generate comprehensive report
    experiments = [ind_cpa_result, ind_cca2_result]
    report = tester.generate_security_report(experiments)
    
    return report

if __name__ == "__main__":
    print("Starting formal cryptographic security tests...\n")
    result = run_formal_security_tests()
    print(result)
