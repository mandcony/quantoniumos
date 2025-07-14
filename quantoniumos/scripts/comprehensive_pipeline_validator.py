#!/usr/bin/env python3
"""
Comprehensive pipeline validation script.
Checks for all common CI/CD issues before pushing.
"""

import os
import json
import yaml
import sys
import subprocess
import re
from pathlib import Path

def check_deprecated_actions():
    """Check for deprecated GitHub Actions versions."""
    issues = []
    workflow_dir = Path(".github/workflows")
    
    if not workflow_dir.exists():
        return ["❌ .github/workflows directory not found"]
    
    for workflow_file in workflow_dir.glob("*.yml"):
        with open(workflow_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Check for deprecated upload-artifact@v3
        if "upload-artifact@v3" in content:
            issues.append(f"❌ {workflow_file.name}: uses deprecated upload-artifact@v3 (should be v4)")
            
        # Check for deprecated download-artifact@v3  
        if "download-artifact@v3" in content:
            issues.append(f"❌ {workflow_file.name}: uses deprecated download-artifact@v3 (should be v4)")
            
        # Check for deprecated setup-python@v3
        if "setup-python@v3" in content:
            issues.append(f"❌ {workflow_file.name}: uses deprecated setup-python@v3 (should be v4)")
            
        # Check for deprecated checkout@v3
        if "checkout@v3" in content:
            issues.append(f"❌ {workflow_file.name}: uses deprecated checkout@v3 (should be v4)")
    
    if not issues:
        print("✅ No deprecated GitHub Actions found")
    
    return issues

def check_docker_image_references():
    """Check for Docker image references that might not exist."""
    issues = []
    workflow_dir = Path(".github/workflows")
    
    for workflow_file in workflow_dir.glob("*.yml"):
        with open(workflow_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Check for quantonium:latest without building first
        if "quantonium:latest" in content:
            # Check if this workflow builds the image or pushes images
            if "docker build" not in content and "build-push-action" not in content:
                issues.append(f"❌ {workflow_file.name}: references quantonium:latest but doesn't build it")
    
    if not issues:
        print("✅ Docker image references look good")
    
    return issues

def check_yaml_syntax():
    """Check YAML syntax for all workflow files."""
    issues = []
    workflow_dir = Path(".github/workflows")
    
    for workflow_file in workflow_dir.glob("*.yml"):
        try:
            with open(workflow_file, 'r', encoding='utf-8', errors='ignore') as f:
                yaml.safe_load(f)
            print(f"✅ {workflow_file.name}: Valid YAML")
        except yaml.YAMLError as e:
            issues.append(f"❌ {workflow_file.name}: Invalid YAML - {e}")
    
    return issues

def check_requirements_sync():
    """Check if requirements.txt exists and is reasonable."""
    issues = []
    
    if not Path("requirements.txt").exists():
        issues.append("❌ requirements.txt not found")
        return issues
    
    with open("requirements.txt", 'r') as f:
        reqs = f.read().strip()
    
    if not reqs:
        issues.append("❌ requirements.txt is empty")
    
    # Check for critical packages
    critical_packages = ["flask", "cryptography", "pydantic"]
    for package in critical_packages:
        if package not in reqs.lower():
            issues.append(f"⚠️ {package} not found in requirements.txt")
    
    if not issues:
        print("✅ requirements.txt looks good")
    
    return issues

def check_dockerfile():
    """Check if Dockerfile exists and has basic structure."""
    issues = []
    
    if not Path("Dockerfile").exists():
        issues.append("❌ Dockerfile not found")
        return issues
    
    with open("Dockerfile", 'r') as f:
        content = f.read()
    
    required_elements = ["FROM", "COPY", "RUN", "EXPOSE", "CMD"]
    for element in required_elements:
        if element not in content:
            issues.append(f"⚠️ Dockerfile missing {element} instruction")
    
    if not issues:
        print("✅ Dockerfile structure looks good")
    
    return issues

def check_secrets_and_leaks():
    """Check for potential secret leaks."""
    issues = []
    sensitive_patterns = [
        r'password\s*=\s*["\'][^"\']{10,}["\']',  # Only flag longer passwords
        r'api_key\s*=\s*["\'][a-zA-Z0-9_-]{20,}["\']',  # Only flag realistic API keys
        r'secret\s*=\s*["\'][a-zA-Z0-9_-]{20,}["\']',  # Only flag realistic secrets
        r'token\s*=\s*["\'][a-zA-Z0-9_-]{20,}["\']',  # Only flag realistic tokens
        r'-----BEGIN [A-Z ]+ KEY-----',
    ]
    
    # Ignore patterns (obvious dev/test values)
    ignore_patterns = [
        r'dev.*key',
        r'test.*key',
        r'demo.*key',
        r'example.*key',
        r'not.*secret',
    ]
    
    # Check Python files
    for py_file in Path(".").rglob("*.py"):
        if ".git" in str(py_file) or "venv" in str(py_file):
            continue
            
        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        for pattern in sensitive_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                matched_text = match.group()
                # Check if it's an obvious dev value
                is_dev_value = any(re.search(ignore, matched_text, re.IGNORECASE) for ignore in ignore_patterns)
                if not is_dev_value:
                    issues.append(f"⚠️ Potential secret in {py_file}: {matched_text[:30]}...")
    
    if not issues:
        print("✅ No suspicious secrets found")
    
    return issues

def check_main_app():
    """Check if main.py has basic structure."""
    issues = []
    
    if not Path("main.py").exists():
        issues.append("❌ main.py not found")
        return issues
    
    with open("main.py", 'r') as f:
        content = f.read()
    
    if "app = Flask" not in content and "FastAPI" not in content:
        issues.append("❌ main.py doesn't appear to create a Flask or FastAPI app")
    
    if "/health" not in content:
        issues.append("⚠️ No health endpoint found in main.py")
    
    if not issues:
        print("✅ main.py structure looks good")
    
    return issues

def main():
    """Run all validation checks."""
    print("🔍 Running comprehensive pipeline validation...\n")
    
    all_issues = []
    
    # Run all checks
    checks = [
        ("GitHub Actions versions", check_deprecated_actions),
        ("Docker image references", check_docker_image_references),
        ("YAML syntax", check_yaml_syntax),
        ("Requirements sync", check_requirements_sync),
        ("Dockerfile", check_dockerfile),
        ("Secrets and leaks", check_secrets_and_leaks),
        ("Main application", check_main_app),
    ]
    
    for check_name, check_func in checks:
        print(f"\n📋 Checking {check_name}...")
        issues = check_func()
        if issues:
            all_issues.extend(issues)
            for issue in issues:
                print(f"  {issue}")
        else:
            print(f"  ✅ {check_name} passed")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    if all_issues:
        print(f"❌ Found {len(all_issues)} issues:")
        for issue in all_issues:
            print(f"  {issue}")
        print(f"\n🚨 Please fix these issues before pushing!")
        return 1
    else:
        print("✅ All checks passed! Pipeline should be bulletproof.")
        print("🚀 Ready to push!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
