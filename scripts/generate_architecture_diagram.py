#!/usr/bin/env python3
"""
Generate ASCII architecture diagram for QuantoniumOS
Run: python scripts/generate_architecture_diagram.py
"""

def generate_ascii_architecture():
    diagram = """
+-------------------+    +--------------------+    +-------------------+
|   Web Browser     |    |   Desktop App      |    |   Mobile App      |
|  (React/HTML)     |    |    (PyQt5)         |    |   (PWA)           |
+---------+---------+    +---------+----------+    +---------+---------+
          |                        |                         |
          +------------------------+-------------------------+
                                   |
          +---------------------+--+--+---------------------+
          |                     |  |  |                     |
    +-----v-----+         +-----v--v-----+           +-----v-----+
    |   NGINX   |         |   Flask App  |           |   Auth    |
    | (Static)  |         |  (Python)    |           |  Service  |
    +-----------+         +-----+--------+           +-----------+
                                |
                    +-----------+-------+
                    |           |       |
            +-------v---+  +----v----+  +--v------+
            | Quantum   |  |  C++    |  | Redis   |
            | Engine    |  | Core    |  | Cache   |
            | (Python)  |  | (Eigen) |  |         |
            +-----------+  +---------+  +---------+
                                |
                        +-------v--------+
                        |  PostgreSQL    |
                        |   Database     |
                        +----------------+

Data Flow:
1. Frontend -> Flask routes (/api/*)
2. Flask -> C++ quantum core (via Python bindings)
3. C++ -> Mathematical computations (Eigen, OpenMP)
4. Results -> PostgreSQL (encrypted) + Redis (cached)
5. Response -> JSON API -> Frontend

Technology Stack:
- Frontend: HTML5, CSS3, JavaScript (vanilla + some React components)
- Backend: Flask (Python 3.9+), SQLAlchemy, Redis
- Quantum Core: C++ with Eigen library, Python bindings via pybind11
- Database: PostgreSQL with encrypted storage
- Security: JWT auth, WAF, rate limiting, TLS
- Desktop: PyQt5 with native OS integration
- Deployment: Docker, Gunicorn, NGINX reverse proxy
    """
    return diagram

def save_diagram_to_docs():
    """Save the diagram to documentation folder"""
    import os
    
    # Create docs folder if it doesn't exist
    docs_dir = "docs"
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
    
    # Save ASCII diagram
    with open(f"{docs_dir}/architecture.txt", "w", encoding='utf-8') as f:
        f.write(generate_ascii_architecture())
    
    print("✅ Architecture diagram saved to docs/architecture.txt")
    
    # Also create a markdown version for the docs
    md_content = f"""# QuantoniumOS Architecture

## System Overview

```
{generate_ascii_architecture()}
```

## Component Details

### Frontend Layer
- **Web Browser**: HTML5/CSS3/JavaScript interface
- **Desktop App**: PyQt5 native application
- **Mobile App**: Progressive Web App (PWA)

### Application Layer
- **Flask App**: Python web framework with blueprints
- **Auth Service**: JWT-based authentication and authorization
- **NGINX**: Static file serving and reverse proxy

### Core Processing Layer
- **Quantum Engine**: Python quantum algorithm implementations
- **C++ Core**: High-performance mathematical computations
- **Redis Cache**: In-memory data structure store

### Data Layer
- **PostgreSQL**: Primary database with encryption at rest
- **File Storage**: Encrypted local/cloud storage for large datasets

## Security Architecture

```
Internet → NGINX (TLS) → Flask (WAF) → Auth Service → Core APIs
                                   ↓
                              Rate Limiting (Redis)
                                   ↓
                           Database (Encrypted)
```

## API Request Flow

1. **Authentication**: JWT token validation
2. **Rate Limiting**: Redis-based request throttling
3. **WAF**: Web Application Firewall security checks
4. **Routing**: Flask blueprint-based request routing
5. **Processing**: Python/C++ quantum computations
6. **Caching**: Redis result caching for performance
7. **Storage**: Encrypted PostgreSQL persistence
8. **Response**: JSON API response with metadata

## Performance Characteristics

- **Latency**: < 100ms for cached quantum operations
- **Throughput**: 1000+ requests/second with Redis caching
- **Concurrency**: Multi-threaded C++ core with Python GIL bypass
- **Scalability**: Horizontal scaling via Docker containers

## Deployment Architecture

```
Load Balancer → [App Instance 1, App Instance 2, App Instance N]
                              ↓
                    [Redis Cluster] ← → [PostgreSQL Cluster]
```
"""
    
    with open(f"{docs_dir}/architecture.md", "w", encoding='utf-8') as f:
        f.write(md_content)
    
    print("✅ Architecture documentation saved to docs/architecture.md")

if __name__ == "__main__":
    print("Generating QuantoniumOS Architecture Diagram...")
    print(generate_ascii_architecture())
    save_diagram_to_docs()
