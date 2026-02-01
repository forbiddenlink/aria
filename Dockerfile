# AI Artist Dockerfile
# Multi-stage build for optimized production image

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.11-slim

# Create non-root user for security
RUN groupadd --gid 1000 aiartist \
    && useradd --uid 1000 --gid aiartist --shell /bin/bash --create-home aiartist

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /home/aiartist/.local

# Copy application code
COPY --chown=aiartist:aiartist . .

# Install package in non-editable mode for production
RUN pip install --no-cache-dir .

# Create necessary directories with proper ownership
RUN mkdir -p gallery models logs config data \
    && chown -R aiartist:aiartist /app

# Set environment variables
ENV PATH=/home/aiartist/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models/cache

# Create entrypoint script to handle volume permissions
RUN echo '#!/bin/sh\n\
# Ensure gallery directory is writable (Railway volumes may override permissions)\n\
mkdir -p /app/gallery/2026\n\
exec "$@"' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh \
    && chown aiartist:aiartist /app/entrypoint.sh

# Switch to non-root user
USER aiartist

# Expose port (Railway will set $PORT)
EXPOSE 8000

# Health check - Railway sets PORT, fallback to 8000
# Note: Railway's healthcheckPath in railway.toml handles this, so we disable Docker healthcheck
# HEALTHCHECK NONE means Railway's external healthcheck will be used instead

# Use entrypoint to handle volume permissions
ENTRYPOINT ["/app/entrypoint.sh"]

# Run web server - use explicit shell to properly expand PORT variable
CMD ["sh", "-c", "uvicorn ai_artist.web.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
