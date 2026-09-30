FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# torch with CUDA 12.6 (host driver 566.07 supports CUDA up to 12.7); installed first so
# requirements.txt finds torch already satisfied and does not pull a CPU/other-CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy configuration and application code
COPY config.json .
COPY embedding_service.py .

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "embedding_service.py"]
