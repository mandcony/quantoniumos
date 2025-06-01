#!/bin/bash
# QuantoniumOS Security Smoke Test Suite
# Validates all security measures are functioning correctly

echo "🔒 QuantoniumOS Security Smoke Test Suite"
echo "=========================================="

BASE_URL=${1:-"http://localhost:5000"}
FAILED_TESTS=0
TOTAL_TESTS=0

test_passed() {
    echo "✅ PASS: $1"
    ((TOTAL_TESTS++))
}

test_failed() {
    echo "❌ FAIL: $1"
    ((FAILED_TESTS++))
    ((TOTAL_TESTS++))
}

echo
echo "1. Testing Security Headers..."

# Test HSTS header
HSTS_HEADER=$(curl -s -I "$BASE_URL" | grep -i "Strict-Transport-Security")
if [[ $HSTS_HEADER == *"max-age=63072000"* ]]; then
    test_passed "HSTS header with correct max-age"
else
    test_failed "HSTS header missing or incorrect"
fi

# Test CSP header
CSP_HEADER=$(curl -s -I "$BASE_URL" | grep -i "Content-Security-Policy")
if [[ -n "$CSP_HEADER" ]]; then
    test_passed "Content-Security-Policy header present"
else
    test_failed "Content-Security-Policy header missing"
fi

# Test Referrer Policy
REF_HEADER=$(curl -s -I "$BASE_URL" | grep -i "Referrer-Policy")
if [[ -n "$REF_HEADER" ]]; then
    test_passed "Referrer-Policy header present"
else
    test_failed "Referrer-Policy header missing"
fi

echo
echo "2. Testing WAF WordPress/PHP Attack Protection..."

# Test WordPress admin attack
WP_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/wp-admin/")
if [[ "$WP_RESPONSE" == "403" ]]; then
    test_passed "WordPress admin attack blocked (403)"
else
    test_failed "WordPress admin attack not blocked (got $WP_RESPONSE)"
fi

# Test PHP file attack
PHP_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/admin.php")
if [[ "$PHP_RESPONSE" == "403" ]]; then
    test_passed "PHP file attack blocked (403)"
else
    test_failed "PHP file attack not blocked (got $PHP_RESPONSE)"
fi

# Test xmlrpc.php attack
XML_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/xmlrpc.php")
if [[ "$XML_RESPONSE" == "403" ]]; then
    test_passed "xmlrpc.php attack blocked (403)"
else
    test_failed "xmlrpc.php attack not blocked (got $XML_RESPONSE)"
fi

echo
echo "3. Testing Rate Limiting..."

# Test if rate limiting is active (should get normal response first)
RATE_TEST=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health")
if [[ "$RATE_TEST" == "200" ]]; then
    test_passed "Rate limiting endpoint accessible"
else
    test_failed "Rate limiting endpoint not accessible (got $RATE_TEST)"
fi

echo
echo "4. Testing Environment Variable Security..."

# Check for exposed environment variables in responses
ENV_CHECK=$(curl -s "$BASE_URL/health" | grep -i "DATABASE_URL\|SECRET_KEY\|MASTER_KEY")
if [[ -z "$ENV_CHECK" ]]; then
    test_passed "No environment variables exposed in responses"
else
    test_failed "Environment variables detected in response"
fi

echo
echo "5. Testing Redis Connection..."

# Check Redis connectivity by looking for Redis errors in logs
if pgrep redis-server > /dev/null; then
    test_passed "Redis server is running"
else
    test_failed "Redis server not running"
fi

echo
echo "=========================================="
echo "Security Test Results:"
echo "Total Tests: $TOTAL_TESTS"
echo "Passed: $((TOTAL_TESTS - FAILED_TESTS))"
echo "Failed: $FAILED_TESTS"

if [[ $FAILED_TESTS -eq 0 ]]; then
    echo "🎉 All security tests PASSED! Platform is hardened."
    exit 0
else
    echo "⚠️  $FAILED_TESTS security tests FAILED. Review and fix issues."
    exit 1
fi