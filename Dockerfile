# Production-minded Dockerfile for Medical RAG Assistant
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system build dependencies and curl for health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies manifest first to optimize Docker layer caching
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install Python packages
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY app /app/app
COPY static /app/static
COPY ingest_script.py /app/ingest_script.py
COPY conftest.py /app/conftest.py
COPY tests /app/tests

# Expose FastAPI port
EXPOSE 8000

# Health check using FastAPI health endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Launch production server using Uvicorn
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
