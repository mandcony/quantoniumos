#!/usr/bin/env python
"""
QuantoniumOS Route Analyzer

This script analyzes the Flask routes in your QuantoniumOS application
and generates a visual map and documentation of all routes.
"""

import os
import re
import sys
import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Set, Optional

# Visualization imports
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("Graphviz not available. Install with: pip install graphviz")
    print("You'll also need the Graphviz binary: https://graphviz.org/download/")

# Constants
ROUTE_PATTERN = r'@\w+\.route\s*\(\s*[\'"]([^\'"]+)[\'"]'
METHOD_PATTERN = r'methods\s*=\s*\[\s*([^\]]+)\]'
FUNCTION_PATTERN = r'def\s+(\w+)\s*\('
AUTH_PATTERN = r'@require_jwt_auth'

class RouteInfo:
    """Store information about a route"""
    def __init__(self, path: str, function_name: str, methods: List[str], 
                 requires_auth: bool, blueprint: str, file_path: str, line_number: int):
        self.path = path
        self.function_name = function_name
        self.methods = methods
        self.requires_auth = requires_auth
        self.blueprint = blueprint
        self.file_path = file_path
        self.line_number = line_number
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "function": self.function_name,
            "methods": self.methods,
            "requires_auth": self.requires_auth,
            "blueprint": self.blueprint,
            "file": self.file_path,
            "line": self.line_number
        }
    
    def get_full_path(self) -> str:
        """Get the full route path including blueprint prefix"""
        # Some blueprints might have a url_prefix, but we'll assume /blueprint for simplicity
        if self.blueprint != "app":
            return f"/{self.blueprint}{self.path}"
        return self.path

def find_flask_route_files(root_dir: str) -> List[str]:
    """Find all Python files that might contain Flask routes"""
    route_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if '@app.route' in content or '@api.route' in content or '@blueprint.route' in content or '.route(' in content:
                        route_files.append(path)
    return route_files

def extract_routes_from_file(file_path: str) -> List[RouteInfo]:
    """Extract route information from a Python file"""
    routes = []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Find blueprint definitions
    blueprint_pattern = r'(\w+)\s*=\s*Blueprint\s*\(\s*[\'"](\w+)[\'"]'
    blueprints = {}
    for match in re.finditer(blueprint_pattern, content):
        var_name, blueprint_name = match.groups()
        blueprints[var_name] = blueprint_name
    
    # Default app blueprint
    blueprints['app'] = 'app'
    
    # Find routes
    route_blocks = []
    current_block = []
    in_decorator_block = False
    
    for i, line in enumerate(lines):
        if '@' in line and '.route(' in line:
            in_decorator_block = True
            current_block = [line]
        elif in_decorator_block:
            current_block.append(line)
            if 'def ' in line:
                in_decorator_block = False
                route_blocks.append((i - len(current_block) + 1, current_block))
                current_block = []
    
    for line_num, block in route_blocks:
        block_text = '\n'.join(block)
        
        # Extract route path
        route_match = re.search(ROUTE_PATTERN, block_text)
        if not route_match:
            continue
            
        route_path = route_match.group(1)
        
        # Extract HTTP methods
        methods = ['GET']  # Default method is GET
        method_match = re.search(METHOD_PATTERN, block_text)
        if method_match:
            methods_str = method_match.group(1)
            methods = [m.strip(' \'"') for m in methods_str.split(',')]
        
        # Extract function name
        function_match = re.search(FUNCTION_PATTERN, block_text)
        if not function_match:
            continue
            
        function_name = function_match.group(1)
        
        # Check if authentication is required
        requires_auth = AUTH_PATTERN in block_text
        
        # Determine blueprint
        blueprint_name = 'app'  # Default
        for bp_var, bp_name in blueprints.items():
            if f'@{bp_var}.route' in block_text:
                blueprint_name = bp_name
                break
        
        routes.append(RouteInfo(
            route_path,
            function_name,
            methods,
            requires_auth,
            blueprint_name,
            file_path,
            line_num
        ))
    
    return routes

def generate_routes_documentation(routes: List[RouteInfo], output_path: str = 'routes_documentation.md'):
    """Generate Markdown documentation for all routes"""
    with open(output_path, 'w') as f:
        f.write("# QuantoniumOS API Routes Documentation\n\n")
        
        # Group routes by blueprint
        routes_by_blueprint = {}
        for route in routes:
            if route.blueprint not in routes_by_blueprint:
                routes_by_blueprint[route.blueprint] = []
            routes_by_blueprint[route.blueprint].append(route)
        
        # Write Table of Contents
        f.write("## Table of Contents\n\n")
        for blueprint in sorted(routes_by_blueprint.keys()):
            f.write(f"- [{blueprint.capitalize()} Routes](#{blueprint.lower()}-routes)\n")
        f.write("\n")
        
        # Write routes by blueprint
        for blueprint, blueprint_routes in sorted(routes_by_blueprint.items()):
            f.write(f"## {blueprint.capitalize()} Routes\n\n")
            
            # Table header
            f.write("| Method | Route | Function | Auth Required | File Location |\n")
            f.write("|--------|-------|----------|--------------|---------------|\n")
            
            # Sort routes by path
            blueprint_routes.sort(key=lambda r: r.path)
            
            for route in blueprint_routes:
                methods = ", ".join(route.methods)
                auth = "✓" if route.requires_auth else "✗"
                file_loc = f"{os.path.basename(route.file_path)}:{route.line_number}"
                
                f.write(f"| {methods} | {route.get_full_path()} | `{route.function_name}` | {auth} | {file_loc} |\n")
            
            f.write("\n")
    
    print(f"Documentation generated: {output_path}")

def generate_routes_visualization(routes: List[RouteInfo], output_path: str = 'routes_visualization'):
    """Generate a visual graph of routes using Graphviz"""
    if not GRAPHVIZ_AVAILABLE:
        print("Skipping visualization as Graphviz is not available")
        return
    
    dot = graphviz.Digraph(
        'QuantoniumOS_Routes', 
        comment='QuantoniumOS API Routes',
        format='png'
    )
    
    # Add main application node
    dot.node('app', 'QuantoniumOS', shape='box', style='filled', color='lightblue')
    
    # Add blueprint nodes
    blueprints = set(route.blueprint for route in routes)
    for blueprint in blueprints:
        if blueprint != 'app':
            dot.node(blueprint, blueprint, shape='box', style='filled', color='lightgreen')
            dot.edge('app', blueprint)
    
    # Add route nodes
    for route in routes:
        route_id = f"{route.blueprint}_{route.function_name}"
        label = f"{', '.join(route.methods)} {route.path}\\n{route.function_name}"
        
        color = 'lightyellow' if not route.requires_auth else 'lightpink'
        dot.node(route_id, label, shape='ellipse', style='filled', color=color)
        dot.edge(route.blueprint, route_id)
    
    # Add a legend
    with dot.subgraph(name='cluster_legend') as legend:
        legend.attr(label='Legend', style='filled', color='white')
        legend.node('app_node', 'Application', shape='box', style='filled', color='lightblue')
        legend.node('blueprint_node', 'Blueprint', shape='box', style='filled', color='lightgreen')
        legend.node('public_route', 'Public Route', shape='ellipse', style='filled', color='lightyellow')
        legend.node('auth_route', 'Auth Required', shape='ellipse', style='filled', color='lightpink')
    
    # Render the graph
    dot.render(output_path, cleanup=True)
    print(f"Visualization generated: {output_path}.png")

def analyze_routes(root_dir: str):
    """Analyze all routes in the application"""
    print(f"Analyzing routes in: {root_dir}")
    
    # Find all files that might contain routes
    route_files = find_flask_route_files(root_dir)
    print(f"Found {len(route_files)} potential route files")
    
    # Extract routes from files
    all_routes = []
    for file_path in route_files:
        routes = extract_routes_from_file(file_path)
        all_routes.extend(routes)
    
    print(f"Extracted {len(all_routes)} routes")
    
    # Generate documentation
    generate_routes_documentation(all_routes)
    
    # Generate visualization
    generate_routes_visualization(all_routes)
    
    # Save routes as JSON for further analysis
    with open('routes_data.json', 'w') as f:
        json.dump([route.to_dict() for route in all_routes], f, indent=2)
    
    print(f"Routes data saved to routes_data.json")
    
    return all_routes

if __name__ == "__main__":
    # Use the current directory or the provided path
    root_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    
    analyze_routes(root_dir)
