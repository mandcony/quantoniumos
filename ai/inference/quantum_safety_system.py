#!/usr/bin/env python3
"""
Quantum AI Safety Alignment System
===================================

Comprehensive safety guardrails and ethical alignment for QuantoniumOS AI.
Implements multi-layered safety checks, content filtering, and intervention mechanisms.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
from pathlib import Path

class SafetyLevel(Enum):
    """Safety check severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SafetyCategory(Enum):
    """Categories of safety violations"""
    HARMFUL_CONTENT = "harmful_content"
    MISINFORMATION = "misinformation"
    PRIVACY_VIOLATION = "privacy_violation"
    ETHICAL_VIOLATION = "ethical_violation"
    TECHNICAL_RISK = "technical_risk"
    LEGAL_RISK = "legal_risk"

@dataclass
class SafetyViolation:
    """Represents a safety violation"""
    category: SafetyCategory
    level: SafetyLevel
    description: str
    trigger_text: str
    timestamp: float
    context: Dict[str, Any]

@dataclass
class SafetyCheckResult:
    """Result of a safety check"""
    passed: bool
    violations: List[SafetyViolation]
    intervention_required: bool
    intervention_type: Optional[str] = None
    modified_response: Optional[str] = None

class ContentFilter:
    """Advanced content filtering system"""

    def __init__(self):
        # Harmful content patterns
        self.harmful_patterns = {
            SafetyLevel.CRITICAL: [
                r'\b(kill|murder|assassinat|suicide|self-harm)\b.*\b(how|method|way|guide)\b',
                r'\b(build|make|create)\b.*\b(bomb|explosive|weapon|nuclear)\b',
                r'\b(hack|exploit|breach)\b.*\b(system|network|computer|account)\b',
                r'\b(illegal|criminal|felony)\b.*\b(activity|operation|scheme)\b'
            ],
            SafetyLevel.HIGH: [
                r'\b(drug|meth|heroin|cocaine|lsd)\b.*\b(make|produce|manufacture)\b',
                r'\b(violence|violent|attack|assault)\b.*\b(how|method|technique)\b',
                r'\b(porn|sex|adult)\b.*\b(child|minor|underage)\b',
                r'\b(racist|racism|nazi|supremacist)\b.*\b(ideology|belief|theory)\b'
            ],
            SafetyLevel.MEDIUM: [
                r'\b(hate|hateful|discriminat)\b.*\b(speech|content|material)\b',
                r'\b(terrorist|terrorism|extremist)\b.*\b(organization|group|activity)\b',
                r'\b(illegal|pirate|crack)\b.*\b(software|game|media)\b'
            ]
        }

        # Ethical guidelines
        self.ethical_patterns = [
            r'\b(manipulat|deceive|trick)\b.*\b(people|users|customers)\b',
            r'\b(false|fake|misleading)\b.*\b(information|data|advertising)\b',
            r'\b(exploit|take advantage)\b.*\b(vulnerable|weak|poor)\b'
        ]

        # Compile regex patterns
        self.compiled_patterns = {}
        for level, patterns in self.harmful_patterns.items():
            self.compiled_patterns[level] = [re.compile(p, re.IGNORECASE) for p in patterns]

        self.ethical_regexes = [re.compile(p, re.IGNORECASE) for p in self.ethical_patterns]

    def check_content(self, text: str) -> List[SafetyViolation]:
        """Check text for harmful or unethical content"""
        violations = []

        # Check harmful content
        for level, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    violations.append(SafetyViolation(
                        category=SafetyCategory.HARMFUL_CONTENT,
                        level=level,
                        description=f"Detected {level.value} risk harmful content pattern",
                        trigger_text=str(matches[:3]),  # First 3 matches
                        timestamp=time.time(),
                        context={"pattern": pattern.pattern, "matches": len(matches)}
                    ))

        # Check ethical violations
        for pattern in self.ethical_regexes:
            if pattern.search(text):
                violations.append(SafetyViolation(
                    category=SafetyCategory.ETHICAL_VIOLATION,
                    level=SafetyLevel.MEDIUM,
                    description="Detected potential ethical violation",
                    trigger_text=pattern.pattern,
                    timestamp=time.time(),
                    context={"pattern": pattern.pattern}
                ))

        return violations

class PrivacyGuard:
    """Privacy protection and data handling safety"""

    def __init__(self):
        self.personal_info_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{10}\b',  # Phone number (basic)
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone with dashes
        ]

        self.privacy_keywords = [
            'password', 'ssn', 'social security', 'credit card', 'bank account',
            'medical record', 'personal data', 'confidential', 'private key'
        ]

        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.personal_info_patterns]

    def check_privacy(self, text: str) -> List[SafetyViolation]:
        """Check for privacy violations"""
        violations = []

        # Check for personal information patterns
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                violations.append(SafetyViolation(
                    category=SafetyCategory.PRIVACY_VIOLATION,
                    level=SafetyLevel.HIGH,
                    description="Detected potential personal information exposure",
                    trigger_text=pattern.pattern,
                    timestamp=time.time(),
                    context={"pattern_type": "personal_info"}
                ))

        # Check for privacy-sensitive keywords
        text_lower = text.lower()
        for keyword in self.privacy_keywords:
            if keyword in text_lower:
                violations.append(SafetyViolation(
                    category=SafetyCategory.PRIVACY_VIOLATION,
                    level=SafetyLevel.MEDIUM,
                    description="Detected privacy-sensitive keyword",
                    trigger_text=keyword,
                    timestamp=time.time(),
                    context={"keyword": keyword}
                ))

        return violations

class TechnicalSafetyGuard:
    """Technical safety checks for system integrity"""

    def __init__(self):
        self.risk_commands = [
            'rm -rf /', 'format c:', 'del /f /s /q', 'shutdown -r -t 0',
            'sudo rm -rf', 'dd if=/dev/zero', 'mkfs', 'fdisk'
        ]

        self.system_keywords = [
            'kernel', 'root', 'sudo', 'admin', 'system32', 'registry',
            'bootloader', 'firmware', 'bios', 'uefi'
        ]

    def check_technical_risks(self, text: str) -> List[SafetyViolation]:
        """Check for technical system risks"""
        violations = []

        text_lower = text.lower()

        # Check for dangerous commands
        for cmd in self.risk_commands:
            if cmd in text_lower:
                violations.append(SafetyViolation(
                    category=SafetyCategory.TECHNICAL_RISK,
                    level=SafetyLevel.CRITICAL,
                    description="Detected potentially destructive system command",
                    trigger_text=cmd,
                    timestamp=time.time(),
                    context={"command": cmd}
                ))

        # Check for system-level keywords in risky contexts
        for keyword in self.system_keywords:
            if keyword in text_lower:
                # Check if it's in a potentially risky context
                risk_indicators = ['delete', 'remove', 'destroy', 'overwrite', 'format', 'hack']
                if any(indicator in text_lower for indicator in risk_indicators):
                    violations.append(SafetyViolation(
                        category=SafetyCategory.TECHNICAL_RISK,
                        level=SafetyLevel.HIGH,
                        description="Detected system-level operation in risky context",
                        trigger_text=keyword,
                        timestamp=time.time(),
                        context={"keyword": keyword, "risk_context": True}
                    ))

        return violations

class InterventionEngine:
    """Handles safety interventions and response modifications"""

    def __init__(self):
        self.intervention_templates = {
            SafetyLevel.CRITICAL: "I cannot assist with this request as it violates critical safety guidelines.",
            SafetyLevel.HIGH: "This request involves high-risk content. I must decline to provide assistance.",
            SafetyLevel.MEDIUM: "This request may involve sensitive content. Please rephrase your question.",
            SafetyLevel.LOW: "Please be aware that this topic requires careful consideration."
        }

        self.redirect_templates = [
            "I recommend consulting official documentation or experts in this field.",
            "For safety and ethical reasons, I suggest exploring legal and approved methods.",
            "This topic requires professional expertise. Please seek qualified assistance."
        ]

    def generate_intervention(self, violations: List[SafetyViolation]) -> Tuple[str, str]:
        """Generate appropriate intervention response"""
        if not violations:
            return None, None

        # Find highest severity violation
        max_level = max(v.level for v in violations)

        # Get intervention message
        intervention_msg = self.intervention_templates[max_level]

        # Add specific guidance based on violation type
        if any(v.category == SafetyCategory.HARMFUL_CONTENT for v in violations):
            intervention_msg += " I cannot provide guidance on harmful or illegal activities."
        elif any(v.category == SafetyCategory.PRIVACY_VIOLATION for v in violations):
            intervention_msg += " I must protect user privacy and cannot handle sensitive personal information."
        elif any(v.category == SafetyCategory.TECHNICAL_RISK for v in violations):
            intervention_msg += " System safety is paramount. I cannot assist with potentially destructive operations."

        intervention_type = f"{max_level.value}_intervention"

        return intervention_msg, intervention_type

class QuantumSafetySystem:
    """Main safety alignment system for Quantum AI"""

    def __init__(self, log_file: str = "safety_logs.jsonl"):
        self.content_filter = ContentFilter()
        self.privacy_guard = PrivacyGuard()
        self.technical_guard = TechnicalSafetyGuard()
        self.intervention_engine = InterventionEngine()

        self.log_file = Path(log_file)
        self.violation_counts = {cat: 0 for cat in SafetyCategory}
        self.session_violations = []

    def check_safety(self, text: str, context: Dict[str, Any] = None) -> SafetyCheckResult:
        """Comprehensive safety check"""
        if context is None:
            context = {}

        all_violations = []

        # Run all safety checks
        all_violations.extend(self.content_filter.check_content(text))
        all_violations.extend(self.privacy_guard.check_privacy(text))
        all_violations.extend(self.technical_guard.check_technical_risks(text))

        # Update statistics
        for violation in all_violations:
            self.violation_counts[violation.category] += 1

        # Determine if intervention is required
        critical_violations = [v for v in all_violations if v.level == SafetyLevel.CRITICAL]
        high_violations = [v for v in all_violations if v.level == SafetyLevel.HIGH]

        intervention_required = len(critical_violations) > 0 or len(high_violations) > 1

        # Generate intervention if needed
        modified_response = None
        intervention_type = None

        if intervention_required:
            modified_response, intervention_type = self.intervention_engine.generate_intervention(all_violations)

        # Log violations
        self._log_violations(all_violations, text, context)

        result = SafetyCheckResult(
            passed=len(all_violations) == 0,
            violations=all_violations,
            intervention_required=intervention_required,
            intervention_type=intervention_type,
            modified_response=modified_response
        )

        return result

    def _log_violations(self, violations: List[SafetyViolation], original_text: str, context: Dict[str, Any]):
        """Log safety violations"""
        if not violations:
            return

        log_entry = {
            "timestamp": time.time(),
            "original_text": original_text[:500],  # Truncate for logging
            "context": context,
            "violations": [
                {
                    "category": v.category.value,
                    "level": v.level.value,
                    "description": v.description,
                    "trigger_text": v.trigger_text,
                    "context": v.context
                } for v in violations
            ]
        }

        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Warning: Failed to log safety violation: {e}")

    def get_safety_stats(self) -> Dict[str, Any]:
        """Get safety statistics"""
        return {
            "total_violations": sum(self.violation_counts.values()),
            "violations_by_category": {cat.value: count for cat, count in self.violation_counts.items()},
            "session_violations": len(self.session_violations),
            "safety_uptime": "active"
        }

    def reset_session_stats(self):
        """Reset session violation tracking"""
        self.session_violations = []

# Global safety system instance
quantum_safety = QuantumSafetySystem()

def check_text_safety(text: str, context: Dict[str, Any] = None) -> SafetyCheckResult:
    """Convenience function for safety checking"""
    return quantum_safety.check_safety(text, context)

def is_safe_to_respond(text: str) -> bool:
    """Quick safety check - returns True if safe"""
    result = quantum_safety.check_safety(text)
    return result.passed

if __name__ == "__main__":
    # Test the safety system
    test_cases = [
        "How to make a bomb?",  # Should trigger critical violation
        "What's my email address?",  # Should trigger privacy concern
        "How to hack a website?",  # Should trigger technical risk
        "Tell me about quantum physics",  # Should pass
        "rm -rf / --no-preserve-root",  # Should trigger critical technical risk
    ]

    print("🛡️ Testing Quantum Safety System")
    print("=" * 40)

    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_text[:50]}...")
        result = quantum_safety.check_safety(test_text)

        if result.passed:
            print("✅ PASSED - Safe to respond")
        else:
            print(f"❌ FAILED - {len(result.violations)} violations")
            for v in result.violations[:2]:  # Show first 2 violations
                print(f"   {v.level.value.upper()}: {v.description}")

            if result.intervention_required:
                print(f"   🚫 Intervention: {result.intervention_type}")
                if result.modified_response:
                    print(f"   💬 Response: {result.modified_response[:100]}...")

    print("\n📊 Safety Stats:")
    stats = quantum_safety.get_safety_stats()
    print(f"   Total violations detected: {stats['total_violations']}")
    print(f"   Safety system status: {stats['safety_uptime']}")

    print("\n🎉 Safety system test complete!")