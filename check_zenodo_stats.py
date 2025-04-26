#!/usr/bin/env python3
"""
QuantoniumOS Zenodo Statistics Checker

This script checks the Zenodo repository for statistics on QuantoniumOS publications.
"""

import requests
import json
from datetime import datetime

# Define Zenodo DOIs to check
# Note: These are the DOIs shown in your Zenodo statistics screenshot
ZENODO_DOIS = [
    "10.5281/zenodo.15284115"  # Version v9, April 2025
]

def get_zenodo_metadata(doi):
    """Get metadata for a Zenodo publication by DOI"""
    print(f"Checking Zenodo metadata for DOI: {doi}")
    
    # Format: https://zenodo.org/api/records/DOI_SUFFIX
    doi_suffix = doi.split('zenodo.')[1]
    url = f"https://zenodo.org/api/records/{doi_suffix}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching Zenodo metadata: Status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception fetching Zenodo metadata: {str(e)}")
        return None

def format_metadata(metadata):
    """Format Zenodo metadata for display"""
    if not metadata:
        return "No metadata available"
    
    try:
        # Extract key information
        publication_info = {
            "title": metadata.get("metadata", {}).get("title", "Unknown Title"),
            "publication_date": metadata.get("metadata", {}).get("publication_date", "Unknown Date"),
            "creators": [creator.get("name", "Unknown Creator") for creator in metadata.get("metadata", {}).get("creators", [])],
            "description": metadata.get("metadata", {}).get("description", "No description available")[:500] + "...",
            "keywords": metadata.get("metadata", {}).get("keywords", []),
            "access_right": metadata.get("metadata", {}).get("access_right", "Unknown"),
            "version": metadata.get("metadata", {}).get("version", "Unknown"),
            "doi": metadata.get("metadata", {}).get("doi", "Unknown DOI"),
            "conceptdoi": metadata.get("metadata", {}).get("conceptdoi", "Unknown Concept DOI"),
            "stats": {
                "views": metadata.get("stats", {}).get("views", 0),
                "downloads": metadata.get("stats", {}).get("downloads", 0),
                "unique_views": metadata.get("stats", {}).get("unique_views", 0),
                "unique_downloads": metadata.get("stats", {}).get("unique_downloads", 0),
                "version_views": metadata.get("stats", {}).get("version_views", 0),
                "version_downloads": metadata.get("stats", {}).get("version_downloads", 0),
            }
        }
        
        # Format for display
        formatted = f"""
=== {publication_info['title']} ===
DOI: {publication_info['doi']}
Version: {publication_info['version']}
Published: {publication_info['publication_date']}
Authors: {', '.join(publication_info['creators'])}
Access: {publication_info['access_right']}

STATISTICS:
- Views (all versions): {publication_info['stats']['views']}
- Views (this version): {publication_info['stats']['version_views']}
- Downloads (all versions): {publication_info['stats']['downloads']}
- Downloads (this version): {publication_info['stats']['version_downloads']}
- Unique views: {publication_info['stats']['unique_views']}
- Unique downloads: {publication_info['stats']['unique_downloads']}

KEYWORDS:
{', '.join(publication_info['keywords'])}

DESCRIPTION:
{publication_info['description']}
"""
        return formatted
    except Exception as e:
        print(f"Error formatting metadata: {str(e)}")
        return "Error formatting metadata"

def main():
    """Main function to check Zenodo statistics"""
    print("QuantoniumOS Zenodo Statistics Checker")
    print("======================================")
    print(f"Running check on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    for doi in ZENODO_DOIS:
        metadata = get_zenodo_metadata(doi)
        formatted_metadata = format_metadata(metadata)
        print(formatted_metadata)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()