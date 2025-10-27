# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install system dependencies (removed software-properties-common - not available in Debian Trixie)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV via pip (more reliable for Docker builds)
RUN pip install --no-cache-dir uv

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with UV (10-100x faster than pip!)
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/vectorstore

# Create a non-root user for security
RUN useradd -m -u 1000 chatbot && chown -R chatbot:chatbot /app
USER chatbot

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "enhanced_app.py", "--server.port=8501", "--server.address=0.0.0.0"]