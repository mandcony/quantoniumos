#!/usr/bin/env python3
"""
QuantoniumOS Mention Search Tool

This script searches for mentions of QuantoniumOS and related concepts
across academic and technical sources.
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import trafilatura
import time
import sys

# Constants
SEARCH_TERMS = [
    "QuantoniumOS",
    "Quantum resonance hash",
    "Resonance Fourier Transform",
    "wave-based mathematical principles",
    "Hybrid Computational Framework for Quantum and Resonance Simulation",
    "USPTO Application 19169399",
    "resonance techniques encryption",
    "150-qubit simulation"
]

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'

def search_scholar(term):
    """Search Google Scholar for mentions"""
    print(f"\nSearching Google Scholar for: {term}")
    
    url = f"https://scholar.google.com/scholar?q={term.replace(' ', '+')}"
    headers = {'User-Agent': USER_AGENT}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('div', class_='gs_ri')
        
        findings = []
        for result in results[:5]:  # Limit to first 5 results
            title_elem = result.find('h3', class_='gs_rt')
            if title_elem:
                title = title_elem.text.strip()
                
                # Extract author and publication info
                info_elem = result.find('div', class_='gs_a')
                info = info_elem.text.strip() if info_elem else "No author/publication info"
                
                # Extract snippet
                snippet_elem = result.find('div', class_='gs_rs')
                snippet = snippet_elem.text.strip() if snippet_elem else "No snippet available"
                
                findings.append({
                    "title": title,
                    "info": info,
                    "snippet": snippet
                })
        
        return findings
    except Exception as e:
        print(f"Error searching Google Scholar: {str(e)}")
        return []

def search_arxiv(term):
    """Search arXiv for mentions"""
    print(f"\nSearching arXiv for: {term}")
    
    url = f"http://export.arxiv.org/api/query?search_query=all:{term.replace(' ', '+')}&start=0&max_results=5"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'xml')
        entries = soup.find_all('entry')
        
        findings = []
        for entry in entries:
            title_elem = entry.find('title')
            title = title_elem.text.strip() if title_elem else "No title"
            
            authors = []
            author_elems = entry.find_all('author')
            for author in author_elems:
                name_elem = author.find('name')
                if name_elem:
                    authors.append(name_elem.text.strip())
            
            summary_elem = entry.find('summary')
            summary = summary_elem.text.strip() if summary_elem else "No summary available"
            
            findings.append({
                "title": title,
                "authors": ", ".join(authors),
                "summary": summary
            })
        
        return findings
    except Exception as e:
        print(f"Error searching arXiv: {str(e)}")
        return []

def search_web(term):
    """Search web for mentions using DuckDuckGo"""
    print(f"\nSearching web for: {term}")
    
    headers = {
        'User-Agent': USER_AGENT
    }
    
    try:
        # Use a more reliable search API endpoint
        url = f"https://html.duckduckgo.com/html/?q={term.replace(' ', '+')}"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('div', class_='result')
        
        findings = []
        for result in results[:5]:  # Limit to first 5 results
            title_elem = result.find('a', class_='result__a')
            if title_elem:
                title = title_elem.text.strip()
                link = title_elem.get('href', '')
                
                # Extract snippet
                snippet_elem = result.find('a', class_='result__snippet')
                snippet = snippet_elem.text.strip() if snippet_elem else "No snippet available"
                
                findings.append({
                    "title": title,
                    "url": link,
                    "snippet": snippet
                })
        
        return findings
    except Exception as e:
        print(f"Error searching web: {str(e)}")
        return []

def analyze_content(url):
    """Extract and analyze content from a URL"""
    print(f"Analyzing content from: {url}")
    
    headers = {'User-Agent': USER_AGENT}
    
    try:
        # Use trafilatura for better content extraction
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text:
                return text[:500] + "..." if len(text) > 500 else text
        
        # Fallback to basic requests if trafilatura fails
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text[:500] + "..." if len(text) > 500 else text
    except Exception as e:
        print(f"Error analyzing content: {str(e)}")
        return "Content analysis failed."

def main():
    all_results = {}
    
    print("QuantoniumOS Mention Search Tool")
    print("===============================")
    print("Searching for mentions of QuantoniumOS and related concepts...")
    
    for term in SEARCH_TERMS:
        all_results[term] = {
            "scholar": search_scholar(term),
            "arxiv": search_arxiv(term),
            "web": search_web(term)
        }
        
        # Add a delay to avoid rate limiting
        time.sleep(2)
    
    # Output results
    for term, sources in all_results.items():
        print("\n\n======================================")
        print(f"RESULTS FOR: {term}")
        print("======================================")
        
        # Scholar results
        print("\n--- GOOGLE SCHOLAR ---")
        if sources["scholar"]:
            for i, result in enumerate(sources["scholar"], 1):
                print(f"\n{i}. {result['title']}")
                print(f"   {result['info']}")
                print(f"   {result['snippet']}")
        else:
            print("No results found in Google Scholar")
        
        # arXiv results
        print("\n--- ARXIV ---")
        if sources["arxiv"]:
            for i, result in enumerate(sources["arxiv"], 1):
                print(f"\n{i}. {result['title']}")
                print(f"   Authors: {result['authors']}")
                print(f"   {result['summary'][:200]}...")
        else:
            print("No results found in arXiv")
        
        # Web results
        print("\n--- WEB ---")
        if sources["web"]:
            for i, result in enumerate(sources["web"], 1):
                print(f"\n{i}. {result['title']}")
                print(f"   URL: {result['url']}")
                print(f"   {result['snippet']}")
        else:
            print("No results found on the web")
    
    print("\n\nSearch complete. The above results show mentions and potential interest in QuantoniumOS technology.")

if __name__ == "__main__":
    main()