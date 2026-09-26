# Use an official lightweight Python base image
FROM python:3.12-slim

# Prevent Python from writing bytecode files and ensure log outputs are sent to terminal in real time
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install essential system dependencies for building C-extensions (needed by packages like ChromaDB)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install the uv package manager
RUN pip install --no-cache-dir uv

# Copy dependency files first to utilize Docker layer caching
COPY pyproject.toml uv.lock* ./

# Install project dependencies using uv
RUN uv sync --no-cache

# Copy the rest of the application codebase
COPY . .

# Expose the default Flask port
EXPOSE 5000

# Execute the Flask application using uv
CMD ["uv", "run", "python", "app.py"]