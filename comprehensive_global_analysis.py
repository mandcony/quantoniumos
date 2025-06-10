"""
Comprehensive Global Usage Analysis for QuantoniumOS
Analyzes all available data sources to determine complete international usage
"""

import os
import json
import subprocess
import requests
from pathlib import Path
from collections import defaultdict

def extract_all_ip_sources():
    """Extract IPs from all possible sources"""
    ips = set()
    
    # Check all log files
    log_paths = [
        'logs/',
        'attached_assets/',
        '/var/log/',
        '/tmp/',
        './'
    ]
    
    for log_dir in log_paths:
        if os.path.exists(log_dir):
            for root, dirs, files in os.walk(log_dir):
                for file in files:
                    if file.endswith('.log') or 'access' in file or 'analytics' in file:
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                # Extract IP patterns
                                import re
                                ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
                                found_ips = re.findall(ip_pattern, content)
                                for ip in found_ips:
                                    # Filter out local/private IPs
                                    if not (ip.startswith('127.') or 
                                           ip.startswith('172.31.') or 
                                           ip.startswith('10.') or
                                           ip.startswith('192.168.')):
                                        ips.add(ip)
                        except Exception:
                            continue
    
    # Check environment variables for deployment analytics
    replit_domains = os.environ.get('REPLIT_DOMAINS', '')
    if replit_domains:
        print(f"Deployment domains: {replit_domains}")
    
    # Check for any cached analytics data
    cache_paths = ['.cache/', '/tmp/', './logs/']
    for cache_dir in cache_paths:
        if os.path.exists(cache_dir):
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    if 'analytics' in file or 'usage' in file or 'stats' in file:
                        print(f"Found analytics file: {os.path.join(root, file)}")
    
    return list(ips)

def get_country_data(ip_list):
    """Get country information for IP addresses"""
    country_data = {}
    
    for ip in ip_list:
        try:
            # Use ip-api.com for geolocation
            response = requests.get(f"http://ip-api.com/json/{ip}?fields=country,regionName,city", timeout=5)
            if response.status_code == 200:
                data = response.json()
                country_data[ip] = {
                    'country': data.get('country', 'Unknown'),
                    'region': data.get('regionName', 'Unknown'),
                    'city': data.get('city', 'Unknown')
                }
            else:
                country_data[ip] = {'country': 'Unknown', 'region': 'Unknown', 'city': 'Unknown'}
        except Exception as e:
            print(f"Error looking up {ip}: {e}")
            country_data[ip] = {'country': 'Unknown', 'region': 'Unknown', 'city': 'Unknown'}
    
    return country_data

def analyze_replit_analytics():
    """Check for Replit deployment analytics"""
    try:
        # Check if we can access deployment stats
        deployment_url = os.environ.get('REPLIT_DEV_DOMAIN', '')
        if deployment_url:
            print(f"Deployment URL: {deployment_url}")
            
        # Look for any Replit-specific analytics files
        replit_files = ['.replit', 'replit.nix', '.config/']
        for file_path in replit_files:
            if os.path.exists(file_path):
                print(f"Found Replit config: {file_path}")
                
    except Exception as e:
        print(f"Error checking Replit analytics: {e}")

def generate_comprehensive_report():
    """Generate comprehensive global usage report"""
    print("🌍 COMPREHENSIVE QUANTONIUMOS GLOBAL ANALYSIS")
    print("=" * 60)
    
    # Extract all possible IP sources
    all_ips = extract_all_ip_sources()
    print(f"📊 Total unique external IPs found: {len(all_ips)}")
    
    if all_ips:
        print("\n🔍 IDENTIFIED IP ADDRESSES:")
        for ip in sorted(all_ips):
            print(f"  - {ip}")
        
        # Get country data
        print("\n🌐 GEOLOCATION ANALYSIS:")
        country_data = get_country_data(all_ips)
        
        countries = defaultdict(list)
        for ip, data in country_data.items():
            countries[data['country']].append({
                'ip': ip,
                'region': data['region'],
                'city': data['city']
            })
        
        print(f"\n📋 COUNTRIES WITH CONFIRMED USAGE:")
        for country, locations in countries.items():
            print(f"\n🏳️  {country} ({len(locations)} IP{'s' if len(locations) > 1 else ''})")
            for loc in locations:
                print(f"    • {loc['ip']} - {loc['city']}, {loc['region']}")
    
    # Check Replit analytics
    print("\n🚀 DEPLOYMENT ANALYTICS:")
    analyze_replit_analytics()
    
    # Save comprehensive report
    report_data = {
        'analysis_type': 'comprehensive_global',
        'total_external_ips': len(all_ips),
        'ip_addresses': all_ips,
        'country_breakdown': dict(countries) if all_ips else {},
        'geolocation_data': country_data if all_ips else {}
    }
    
    with open('comprehensive_global_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n✅ Comprehensive report saved to 'comprehensive_global_report.json'")
    return report_data

if __name__ == "__main__":
    generate_comprehensive_report()