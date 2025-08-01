# Stage 1: Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        make \
        cmake \
        libc6-dev \
        python3-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the dependency files first to leverage Docker layer caching
COPY pyproject.toml ./

# Install dependencies with proper error handling
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir . || { \
        echo "Error: Failed to install package dependencies"; \
        pip list; \
        exit 1; \
    }

# Copy the application code
COPY . .

# Check for vulnerabilities in dependencies but don't fail the build
RUN pip install pip-audit \
    && python -m pip_audit || echo "Vulnerabilities found but continuing build" \
    && pip uninstall -y pip-audit

# Compile any C/C++ modules if needed
RUN if [ "${SKIP_BUILD_ENGINE}" != "1" ] && [ -f build_engine.sh ]; then \
        chmod +x build_engine.sh && \
        ./build_engine.sh || exit 1; \
    else \
        echo "Skipping build_engine.sh execution"; \
    fi

# Stage 2: Runtime stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r quant --gid=1001 && \
    useradd -r -g quant --uid=1001 -d /app quant

WORKDIR /app

# Create log directory with appropriate permissions
RUN mkdir -p /tmp/logs && chown quant:quant /tmp/logs

# Copy from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder --chown=quant:quant /app /app

# Set secure environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/bin:${PATH}" \
    PYTHONPATH="/app" \
    GUNICORN_CMD_ARGS="--timeout 120 --max-requests 1000 --max-requests-jitter 50"

# Switch to non-root user
USER quant

# Expose the port the app runs on
EXPOSE 5000

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--limit-request-line", "16384", "--reuse-port", "--reload", "main:app"]