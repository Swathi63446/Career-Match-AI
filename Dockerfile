FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing .pyc files and buffer outputs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install minimal build tools required for C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt-get/lists/*

# Copy dependency configuration files
COPY pyproject.toml uv.lock ./

# Install uv package manager and sync dependencies without PyTorch
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev

# Copy application source code
COPY . .

# Ensure vector store pre-indexing step runs during image build
RUN uv run python ingest.py

EXPOSE 5000

# Run single Gunicorn worker to lock memory footprint well below Render's 512MB limit
CMD ["uv", "run", "gunicorn", "--workers", "1", "--threads", "2", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]