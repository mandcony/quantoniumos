#!/usr/bin/env python3
"""
QuantoniumOS Content Safety Filter
==================================

Comprehensive content filtering for image generation and text responses
to ensure ethical, safe, and appropriate AI outputs.

Features:
- Image prompt filtering
- Text content validation  
- Constitutional AI compliance
- Real-time safety monitoring
"""

import re
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import os

class ContentSafetyFilter:
    """
    Advanced content safety filtering system for QuantoniumOS
    """
    
    def __init__(self):
        self.version = "1.0.0"
        self.filter_strength = "STRICT"  # STRICT, MODERATE, BASIC
        
        # Blocked content categories
        self.blocked_categories = {
            "explicit": {
                "keywords": [
                    "nsfw", "nude", "naked", "explicit", "sexual", "porn", "erotic",
                    "xxx", "adult", "mature", "inappropriate", "vulgar", "obscene"
                ],
                "severity": "HIGH",
                "action": "BLOCK"
            },
            "violence": {
                "keywords": [
                    "violence", "violent", "blood", "gore", "killing", "murder", 
                    "death", "weapon", "gun", "knife", "harm", "hurt", "torture",
                    "abuse", "fight", "assault", "attack"
                ],
                "severity": "HIGH", 
                "action": "BLOCK"
            },
            "hate": {
                "keywords": [
                    "hate", "racist", "discrimination", "slur", "offensive",
                    "bigot", "prejudice", "supremacist", "nazi", "terrorism"
                ],
                "severity": "CRITICAL",
                "action": "BLOCK"
            },
            "illegal": {
                "keywords": [
                    "drug", "cocaine", "heroin", "illegal", "crime", "criminal",
                    "fraud", "scam", "hack", "piracy", "counterfeit", "stolen"
                ],
                "severity": "HIGH",
                "action": "BLOCK"
            },
            "privacy": {
                "keywords": [
                    "personal information", "private", "confidential", "secret",
                    "password", "credit card", "ssn", "social security"
                ],
                "severity": "HIGH",
                "action": "WARN"
            },
            "misinformation": {
                "keywords": [
                    "fake news", "conspiracy", "hoax", "false claim", "misinformation",
                    "disinformation", "propaganda", "lie", "deception"
                ],
                "severity": "MEDIUM",
                "action": "WARN"
            }
        }
        
        # Safe content indicators
        self.safe_indicators = [
            "art", "landscape", "nature", "educational", "creative", "beautiful",
            "colorful", "abstract", "artistic", "design", "illustration",
            "concept", "fantasy", "science", "technology", "architecture"
        ]
        
        # Safety log
        self.safety_log = []
        self._ensure_log_directory()
        
    def _ensure_log_directory(self):
        """Ensure safety logs directory exists"""
        os.makedirs("logs/safety", exist_ok=True)
        
    def _log_safety_event(self, event_type: str, content: str, action: str, details: Dict = None):
        """Log safety filtering events"""
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "content_hash": hashlib.sha256(content.encode()).hexdigest()[:16],
            "action": action,
            "filter_version": self.version,
            "details": details or {}
        }
        
        self.safety_log.append(event)
        
        # Save to persistent log
        try:
            log_file = f"logs/safety/content_filter_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            print(f"⚠️ Safety log error: {e}")
    
    def filter_image_prompt(self, prompt: str) -> Tuple[bool, str, List[str]]:
        """
        Filter image generation prompts for safety
        
        Returns:
        - (allowed: bool, filtered_prompt: str, warnings: List[str])
        """
        
        original_prompt = prompt
        prompt_lower = prompt.lower()
        warnings = []
        blocked_reasons = []
        
        # Check against blocked categories
        for category, config in self.blocked_categories.items():
            for keyword in config["keywords"]:
                if keyword in prompt_lower:
                    if config["action"] == "BLOCK":
                        blocked_reasons.append(f"{category}: '{keyword}'")
                        self._log_safety_event(
                            "IMAGE_PROMPT_BLOCKED",
                            original_prompt,
                            "BLOCKED",
                            {"category": category, "keyword": keyword, "severity": config["severity"]}
                        )
                    elif config["action"] == "WARN":
                        warnings.append(f"Potentially sensitive content: {category}")
        
        # If blocked, return rejection
        if blocked_reasons:
            return False, "", [f"Content blocked: {', '.join(blocked_reasons)}"]
        
        # Enhanced prompt with safety guidance
        safe_prompt = self._enhance_prompt_safety(prompt)
        
        self._log_safety_event("IMAGE_PROMPT_ALLOWED", original_prompt, "ALLOWED", {"warnings": warnings})
        
        return True, safe_prompt, warnings
    
    def _enhance_prompt_safety(self, prompt: str) -> str:
        """Enhance prompt with safety-oriented terms"""
        
        # Add safety-enhancing qualifiers
        safety_enhancers = [
            "appropriate", "family-friendly", "safe", "tasteful", 
            "professional", "artistic", "creative"
        ]
        
        # Check if prompt already has safety indicators
        has_safety_terms = any(term in prompt.lower() for term in self.safe_indicators + safety_enhancers)
        
        if not has_safety_terms:
            # Add appropriate safety qualifier
            enhanced_prompt = f"appropriate, tasteful {prompt}"
            return enhanced_prompt
            
        return prompt
    
    def filter_text_response(self, text: str) -> Tuple[bool, str, List[str]]:
        """
        Filter AI text responses for safety and appropriateness
        
        Returns:
        - (allowed: bool, filtered_text: str, warnings: List[str])
        """
        
        original_text = text
        text_lower = text.lower()
        warnings = []
        
        # Check for problematic content
        for category, config in self.blocked_categories.items():
            for keyword in config["keywords"]:
                if keyword in text_lower:
                    if config["severity"] in ["CRITICAL", "HIGH"]:
                        self._log_safety_event(
                            "TEXT_RESPONSE_BLOCKED",
                            original_text,
                            "BLOCKED",
                            {"category": category, "keyword": keyword}
                        )
                        return False, "I can't provide that type of content. Let me help you with something else instead.", [f"Content blocked: {category}"]
                    else:
                        warnings.append(f"Content advisory: {category}")
        
        # Check for personal information leakage
        if self._contains_personal_info(text):
            warnings.append("Potential personal information detected")
        
        self._log_safety_event("TEXT_RESPONSE_ALLOWED", original_text, "ALLOWED", {"warnings": warnings})
        
        return True, text, warnings
    
    def _contains_personal_info(self, text: str) -> bool:
        """Check if text contains personal information patterns"""
        
        patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Phone number
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                return True
                
        return False
    
    def validate_quantum_model_safety(self, model_data: Dict) -> Dict[str, Any]:
        """Validate quantum model data for safety compliance"""
        
        safety_report = {
            "model_safe": True,
            "safety_score": 1.0,
            "findings": [],
            "recommendations": []
        }
        
        metadata = model_data.get('metadata', {})
        
        # Check for safety indicators in metadata
        if 'safety_validated' not in metadata:
            safety_report["findings"].append("⚠️ No explicit safety validation in metadata")
            safety_report["safety_score"] -= 0.1
            
        # Check encoding method safety
        encoding_method = metadata.get('quantum_encoding_method', '')
        if 'rft' in encoding_method.lower():
            safety_report["findings"].append("✅ Using safe RFT encoding method")
        else:
            safety_report["findings"].append("⚠️ Unknown encoding method")
            safety_report["safety_score"] -= 0.2
            
        # Check unitarity preservation
        if metadata.get('unitarity_preserved', False):
            safety_report["findings"].append("✅ Quantum unitarity preserved")
        else:
            safety_report["findings"].append("❌ Quantum unitarity not guaranteed")
            safety_report["safety_score"] -= 0.3
            safety_report["model_safe"] = False
            
        # Final assessment
        if safety_report["safety_score"] < 0.7:
            safety_report["model_safe"] = False
            safety_report["recommendations"].append("Model requires safety improvements before deployment")
        elif safety_report["safety_score"] < 0.9:
            safety_report["recommendations"].append("Consider additional safety validations")
        else:
            safety_report["recommendations"].append("Model meets safety standards")
            
        return safety_report
    
    def get_safety_statistics(self) -> Dict[str, Any]:
        """Get comprehensive safety filtering statistics"""
        
        stats = {
            "total_events": len(self.safety_log),
            "blocked_content": len([e for e in self.safety_log if e["action"] == "BLOCKED"]),
            "warnings_issued": len([e for e in self.safety_log if "warnings" in e.get("details", {})]),
            "filter_version": self.version,
            "filter_strength": self.filter_strength,
            "categories_monitored": len(self.blocked_categories),
            "last_updated": datetime.now().isoformat()
        }
        
        # Category breakdown
        category_stats = {}
        for event in self.safety_log:
            category = event.get("details", {}).get("category", "unknown")
            category_stats[category] = category_stats.get(category, 0) + 1
            
        stats["category_breakdown"] = category_stats
        
        return stats

# Integration helper for QuantoniumOS
def initialize_content_safety() -> ContentSafetyFilter:
    """Initialize content safety system for QuantoniumOS"""
    
    print("🛡️ Initializing QuantoniumOS Content Safety Filter...")
    filter_system = ContentSafetyFilter()
    
    print(f"✅ Content Safety Filter v{filter_system.version} ready")
    print(f"   Filter Strength: {filter_system.filter_strength}")
    print(f"   Categories Monitored: {len(filter_system.blocked_categories)}")
    print(f"   Safety Logging: Active")
    
    return filter_system

if __name__ == "__main__":
    # Test the content filter
    filter_system = initialize_content_safety()
    
    # Test image prompts
    test_prompts = [
        "beautiful landscape with mountains",
        "inappropriate content example",
        "artistic portrait of a person",
        "violent scene with weapons"
    ]
    
    print("\n🧪 Testing Image Prompt Filtering:")
    for prompt in test_prompts:
        allowed, filtered, warnings = filter_system.filter_image_prompt(prompt)
        status = "✅ ALLOWED" if allowed else "❌ BLOCKED"
        print(f"   {status}: '{prompt[:30]}...'")
        if warnings:
            print(f"      Warnings: {warnings}")
    
    # Show statistics
    stats = filter_system.get_safety_statistics()
    print(f"\n📊 Safety Statistics:")
    print(f"   Total Events: {stats['total_events']}")
    print(f"   Blocked Content: {stats['blocked_content']}")
    print(f"   Categories: {stats['categories_monitored']}")