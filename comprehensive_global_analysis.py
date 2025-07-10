"""
Comprehensive Global Usage Analysis for QuantoniumOS
Analyzes all available data sources to determine complete international usage
"""

import glob
import gzip
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Set

import requests


def extract_all_ip_sources():
    """Extract IPs from all possible sources"""
    all_ips = set()
    ip_sources = defaultdict(list)

    # Known user IPs to exclude (Bronx/Juilliard)
    known_user_ips = {
        "206.55.217.10",  # Bronx location
        "162.84.145.191",  # Juilliard location
        "10.83.0.153",  # Internal Replit
        "10.83.6.20",  # Internal Replit
        "10.83.10.77",  # Internal Replit
        "172.31.128.92",  # Internal container
        "172.31.128.105",  # Internal container
    }

    # Search all log files
    log_patterns = [
        "logs/*.log",
        "logs/**/*.log",
        "attached_assets/logs/*.log",
        "attached_assets/logs/**/*.log",
        "*.log",
        "app.log",
    ]

    for pattern in log_patterns:
        for log_file in glob.glob(pattern, recursive=True):
            try:
                print(f"Analyzing: {log_file}")

                # Handle both regular and gzipped files
                if log_file.endswith(".gz"):
                    with gzip.open(
                        log_file, "rt", encoding="utf-8", errors="ignore"
                    ) as f:
                        content = f.read()
                else:
                    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                # Extract IPs from various log formats
                ip_patterns = [
                    r'"X-Forwarded-For":\s*"([^,\s"]+)',
                    r"X-Forwarded-For:\s*([^,\s]+)",
                    r'client_ip":\s*"([^"]+)',
                    r'source_ip":\s*"([^"]+)',
                    r"IP:\s*(\d+\.\d+\.\d+\.\d+)",
                    r"(\d+\.\d+\.\d+\.\d+).*GET",
                    r"(\d+\.\d+\.\d+\.\d+).*POST",
                ]

                for pattern in ip_patterns:
                    matches = re.findall(pattern, content)
                    for ip in matches:
                        ip = ip.strip()
                        if (
                            ip
                            and ip not in known_user_ips
                            and not ip.startswith("10.")
                            and not ip.startswith("172.")
                        ):
                            all_ips.add(ip)
                            ip_sources[ip].append(log_file)

            except Exception as e:
                print(f"Error reading {log_file}: {e}")

    return all_ips, ip_sources


def get_country_data(ip_list: Set[str]) -> Dict[str, Dict]:
    """Get country information for IP addresses"""
    country_data = {}

    for ip in ip_list:
        try:
            # Use ipapi.co for geolocation (free tier)
            response = requests.get(f"https://ipapi.co/{ip}/json/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                country_data[ip] = {
                    "country": data.get("country_name", "Unknown"),
                    "country_code": data.get("country_code", "XX"),
                    "city": data.get("city", "Unknown"),
                    "region": data.get("region", "Unknown"),
                    "timezone": data.get("timezone", "Unknown"),
                    "org": data.get("org", "Unknown"),
                    "latitude": data.get("latitude"),
                    "longitude": data.get("longitude"),
                }
                print(f"IP {ip}: {data.get('country_name', 'Unknown')}")
            else:
                print(f"Failed to get location for {ip}")
                country_data[ip] = {"country": "Unknown", "country_code": "XX"}
        except Exception as e:
            print(f"Error getting location for {ip}: {e}")
            country_data[ip] = {"country": "Unknown", "country_code": "XX"}

    return country_data


def analyze_replit_analytics():
    """Check for Replit deployment analytics"""
    analytics_data = {}

    # Check if there are any Replit-specific analytics files
    replit_files = [".replit", "replit.nix", ".replit.json"]

    for file_path in replit_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    content = f.read()
                    analytics_data[file_path] = len(content.split("\n"))
            except:
                pass

    return analytics_data


def count_unique_requests(ip_sources: Dict[str, List[str]]) -> Dict[str, int]:
    """Count unique requests per IP"""
    request_counts = {}

    for ip, sources in ip_sources.items():
        total_requests = 0

        for source_file in sources:
            try:
                if source_file.endswith(".gz"):
                    with gzip.open(
                        source_file, "rt", encoding="utf-8", errors="ignore"
                    ) as f:
                        content = f.read()
                else:
                    with open(source_file, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                # Count occurrences of this IP in the file
                ip_count = content.count(ip)
                total_requests += ip_count

            except Exception as e:
                print(f"Error counting requests in {source_file}: {e}")

        request_counts[ip] = total_requests

    return request_counts


def generate_comprehensive_report():
    """Generate comprehensive global usage report"""
    print("QuantoniumOS Global Usage Analysis")
    print("=" * 50)

    # Extract all external IPs
    external_ips, ip_sources = extract_all_ip_sources()

    if not external_ips:
        print("No external users found in current logs")
        print("This may be due to log rotation or recent system reset")
        return

    print(f"Found {len(external_ips)} unique external IP addresses")

    # Get geographical data
    print("\nGetting geographical information...")
    country_data = get_country_data(external_ips)

    # Count requests per IP
    print("\nCounting requests per IP...")
    request_counts = count_unique_requests(ip_sources)

    # Generate summary statistics
    countries = Counter()
    total_requests = 0

    for ip in external_ips:
        country = country_data.get(ip, {}).get("country", "Unknown")
        countries[country] += 1
        total_requests += request_counts.get(ip, 0)

    # Create comprehensive report
    report = {
        "analysis_date": datetime.now().isoformat(),
        "summary": {
            "total_external_users": len(external_ips),
            "total_countries": len(countries),
            "total_requests": total_requests,
            "average_requests_per_user": (
                total_requests / len(external_ips) if external_ips else 0
            ),
        },
        "countries": dict(countries),
        "detailed_users": [],
    }

    # Add detailed user information
    for ip in sorted(external_ips):
        user_data = country_data.get(ip, {})
        user_data["ip"] = ip
        user_data["request_count"] = request_counts.get(ip, 0)
        user_data["log_sources"] = ip_sources.get(ip, [])
        report["detailed_users"].append(user_data)

    # Save report
    with open("comprehensive_global_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("GLOBAL USAGE SUMMARY")
    print("=" * 50)
    print(f"Total External Users: {len(external_ips)}")
    print(f"Total Countries: {len(countries)}")
    print(f"Total Requests: {total_requests:,}")

    if countries:
        print("\nTop Countries:")
        for country, count in countries.most_common(10):
            print(f"  {country}: {count} users")

    print("\nDetailed User Information:")
    for user in sorted(
        report["detailed_users"], key=lambda x: x["request_count"], reverse=True
    ):
        print(
            f"  {user['ip']} ({user.get('country', 'Unknown')}): {user['request_count']} requests"
        )

    print(f"\nReport saved to: comprehensive_global_report.json")

    return report


if __name__ == "__main__":
    report = generate_comprehensive_report()
