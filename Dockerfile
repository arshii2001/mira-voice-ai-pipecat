FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure default curriculum data is available (can be overridden by volume mount)
RUN mkdir -p /content/curriculum
COPY content/curriculum/ /content/curriculum/

# Copy and setup entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Copy .env file for default environment variables
# These can be overridden at runtime with: docker run -e VAR=value


# Expose port
EXPOSE 7860

# Health check (uses -k to skip SSL verification for self-signed certs)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -kf https://localhost:7860/health || curl -f http://localhost:7860/health || exit 1

# Use entrypoint to load .env, then run server
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "server.py"]
