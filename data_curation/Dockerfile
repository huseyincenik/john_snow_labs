FROM python:3.10-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies (includes DocETL dependencies)
RUN uv pip install --system -e .

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/input data/output data/ontology logs

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

