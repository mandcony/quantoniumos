#!/bin/bash
# QuantoniumOS Security Operations Script
# Implements the security maintenance recommendations

echo "🔒 QUANTONIUMOS SECURITY OPERATIONS"
echo "=================================="

# Function to run pre-operation validation
run_validation_tests() {
    echo ""
    echo "🔬 RUNNING PRE-OPERATION VALIDATION..."
    echo "------------------------------------"
    
    cd /workspaces/quantoniumos
    
    echo "1. Security Analysis:"
    python3 quantonium_security_analysis.py | tail -10
    
    echo ""
    echo "2. AI Intelligence Analysis:"
    python3 quantonium_ai_intelligence_analysis.py | tail -10
    
    echo ""
    echo "✅ Validation tests completed"
}

# Function to make core files read-only
secure_core_files() {
    echo ""
    echo "🛡️ SECURING CORE ALGORITHM FILES..."
    echo "--------------------------------"
    
    core_files=(
        "core/canonical_true_rft.py"
        "core/enhanced_rft_crypto_v2.py"
        "core/enhanced_topological_qubit.py"
        "core/geometric_waveform_hash.py"
        "core/topological_quantum_kernel.py"
        "core/working_quantum_kernel.py"
    )
    
    for file in "${core_files[@]}"; do
        if [ -f "/workspaces/quantoniumos/$file" ]; then
            chmod 444 "/workspaces/quantoniumos/$file"
            echo "✅ $file → READ-ONLY"
        else
            echo "⚠️ $file → NOT FOUND"
        fi
    done
}

# Function to monitor system resources
monitor_resources() {
    echo ""
    echo "📊 SYSTEM RESOURCE MONITORING"
    echo "-----------------------------"
    
    # Memory usage
    memory_percent=$(free | grep Mem | awk '{printf("%.1f", $3/$2 * 100.0)}')
    echo "Memory Usage: ${memory_percent}%"
    
    # CPU usage
    cpu_percent=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    echo "CPU Usage: ${cpu_percent}%"
    
    # Disk usage
    disk_percent=$(df / | tail -1 | awk '{print $5}' | cut -d'%' -f1)
    echo "Disk Usage: ${disk_percent}%"
    
    # Check thresholds
    if (( $(echo "$memory_percent > 90" | bc -l) )); then
        echo "⚠️ HIGH MEMORY USAGE DETECTED!"
    fi
    
    if (( $(echo "$cpu_percent > 95" | bc -l) )); then
        echo "⚠️ HIGH CPU USAGE DETECTED!"
    fi
    
    if (( disk_percent > 95 )); then
        echo "⚠️ HIGH DISK USAGE DETECTED!"
    fi
}

# Function to check for suspicious activity
security_check() {
    echo ""
    echo "🔍 SECURITY STATUS CHECK"
    echo "-----------------------"
    
    # Check for unauthorized network connections
    echo "Network connections:"
    netstat -tuln | grep LISTEN | wc -l
    
    # Check running Python processes
    echo "QuantoniumOS processes:"
    ps aux | grep -i quantonium | grep -v grep | wc -l
    
    # Check file integrity
    echo "Core file integrity:"
    core_files_count=$(find /workspaces/quantoniumos/core -name "*.py" | wc -l)
    echo "Core files found: $core_files_count/6"
}

# Main menu
show_menu() {
    echo ""
    echo "🎯 SECURITY OPERATIONS MENU"
    echo "=========================="
    echo "1. Run Pre-Operation Validation"
    echo "2. Secure Core Files (Make Read-Only)"
    echo "3. Monitor System Resources"
    echo "4. Security Status Check"
    echo "5. Full Security Maintenance"
    echo "6. Exit"
    echo ""
    read -p "Select option (1-6): " choice
}

# Full security maintenance
full_maintenance() {
    echo ""
    echo "🔒 FULL SECURITY MAINTENANCE"
    echo "==========================="
    
    run_validation_tests
    secure_core_files
    monitor_resources
    security_check
    
    echo ""
    echo "✅ Full security maintenance completed!"
}

# Main script logic
if [ "$1" = "--auto" ]; then
    # Automated mode
    full_maintenance
else
    # Interactive mode
    while true; do
        show_menu
        case $choice in
            1) run_validation_tests ;;
            2) secure_core_files ;;
            3) monitor_resources ;;
            4) security_check ;;
            5) full_maintenance ;;
            6) echo "Goodbye!"; exit 0 ;;
            *) echo "Invalid option" ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
fi
