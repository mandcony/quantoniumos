#!/usr/bin/env python3
"""
QuantoniumOS Global Usage Analytics
Analyzes server logs to determine geographic distribution of users over the last 30 days
"""

import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import requests


def get_ip_country(ip_address):
    """
    Get country for an IP address using a free geolocation service
    Returns country name or 'Unknown' if lookup fails
    """
    # Skip private/internal IPs
    if ip_address.startswith(("10.", "172.", "192.168.", "127.")):
        return "Internal/CDN"

    try:
        # Using a free IP geolocation service
        response = requests.get(f"http://ip-api.com/json/{ip_address}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                return data.get("country", "Unknown")
    except:
        pass

    return "Unknown"


def extract_ips_from_logs():
    """
    Extract unique IP addresses from all available log files
    """
    ips = set()
    log_files = [
        "logs/quantonium_api.log",
        "logs/security/security.log",
        "logs/session_current.log",
        "attached_assets/app.log",
    ]

    # X-Forwarded-For pattern to extract real client IPs
    forwarded_pattern = r'"X-Forwarded-For":\s*"([^"]+)"'
    # Standard IP pattern
    ip_pattern = r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"

    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    content = f.read()

                    # Look for X-Forwarded-For headers first (real client IPs)
                    forwarded_matches = re.findall(forwarded_pattern, content)
                    for match in forwarded_matches:
                        # X-Forwarded-For can have multiple IPs, first one is usually the real client
                        client_ip = match.split(",")[0].strip()
                        if not client_ip.startswith(
                            ("10.", "172.", "192.168.", "127.")
                        ):
                            ips.add(client_ip)

                    # Also look for any other IP addresses in logs
                    ip_matches = re.findall(ip_pattern, content)
                    for ip in ip_matches:
                        if not ip.startswith(("10.", "172.", "192.168.", "127.")):
                            ips.add(ip)

            except Exception as e:
                print(f"Error reading {log_file}: {e}")

    return list(ips)


def count_requests_by_ip():
    """
    Count the number of requests by each IP address
    """
    ip_counts = Counter()
    log_files = [
        "logs/quantonium_api.log",
        "logs/security/security.log",
        "logs/session_current.log",
    ]

    forwarded_pattern = r'"X-Forwarded-For":\s*"([^"]+)"'

    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        # Look for X-Forwarded-For headers
                        match = re.search(forwarded_pattern, line)
                        if match:
                            client_ip = match.group(1).split(",")[0].strip()
                            if not client_ip.startswith(
                                ("10.", "172.", "192.168.", "127.")
                            ):
                                ip_counts[client_ip] += 1

            except Exception as e:
                print(f"Error reading {log_file}: {e}")

    return ip_counts


def analyze_usage():
    """
    Main analysis function
    """
    print("🌍 QuantoniumOS Global Usage Analysis (Last 30 Days)")
    print("=" * 60)

    # Extract unique IPs
    print("📊 Extracting IP addresses from logs...")
    unique_ips = extract_ips_from_logs()

    # Count requests by IP
    print("📈 Counting requests by IP...")
    ip_counts = count_requests_by_ip()

    # Get country for each IP
    print("🗺️  Looking up geographic locations...")
    country_stats = defaultdict(int)
    ip_to_country = {}

    for ip in unique_ips:
        country = get_ip_country(ip)
        ip_to_country[ip] = country
        requests_count = ip_counts.get(ip, 1)  # At least 1 if IP was found
        country_stats[country] += requests_count

    # Display results
    print("\n🎯 GLOBAL USAGE STATISTICS")
    print("=" * 40)

    total_requests = sum(country_stats.values())
    unique_countries = len(
        [c for c in country_stats.keys() if c not in ["Unknown", "Internal/CDN"]]
    )

    print(f"Total Requests: {total_requests}")
    print(f"Unique Countries: {unique_countries}")
    print(f"Unique IP Addresses: {len(unique_ips)}")

    print("\n📋 REQUESTS BY COUNTRY:")
    print("-" * 30)

    # Sort countries by request count
    sorted_countries = sorted(country_stats.items(), key=lambda x: x[1], reverse=True)

    for country, count in sorted_countries:
        if country not in ["Internal/CDN"]:
            percentage = (count / total_requests * 100) if total_requests > 0 else 0
            print(f"{country:<20} {count:>6} requests ({percentage:.1f}%)")

    print("\n🔍 DETAILED IP BREAKDOWN:")
    print("-" * 40)

    for ip in sorted(unique_ips):
        country = ip_to_country.get(ip, "Unknown")
        requests = ip_counts.get(ip, 1)
        if country not in ["Internal/CDN"]:
            print(f"{ip:<18} {country:<20} {requests:>6} requests")

    return {
        "total_requests": total_requests,
        "unique_countries": unique_countries,
        "unique_ips": len(unique_ips),
        "country_breakdown": dict(sorted_countries),
        "ip_details": [
            (ip, ip_to_country.get(ip, "Unknown"), ip_counts.get(ip, 1))
            for ip in unique_ips
            if ip_to_country.get(ip, "Unknown") not in ["Internal/CDN"]
        ],
    }


if __name__ == "__main__":
    try:
        results = analyze_usage()

        # Save results to JSON file
        with open("global_usage_report.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Analysis complete! Report saved to 'global_usage_report.json'")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
