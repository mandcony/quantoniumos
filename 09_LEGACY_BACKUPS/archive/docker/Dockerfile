FROM python:3.9-slim-buster

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Clone QuantoniumOS repository
RUN git clone https://github.com/mandcony/quantoniumos.git .

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set up entrypoint
ENTRYPOINT ["/bin/bash"]
