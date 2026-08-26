# Use official Python 3.11 image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    git \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN pip install uv

# Set workdir
WORKDIR /app

# Copy project files
COPY . .

# Create virtual environment and install dependencies
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -e .

# Default command (can be changed as needed)
CMD ["/bin/bash"] 