#!/bin/bash
# RAG QA Chatbot - Automatic Startup Script for Linux/Mac
# This script starts Docker containers and opens the browser

echo "========================================"
echo " RAG QA Chatbot - Starting Application"
echo "========================================"
echo ""

echo "[1/3] Starting Docker containers..."
docker-compose up -d

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to start Docker containers!"
    echo "Please make sure Docker is running."
    exit 1
fi

echo ""
echo "[2/3] Waiting for application to be ready..."
echo "This may take 30-60 seconds..."
sleep 30

echo ""
echo "[3/3] Opening browser..."

# Detect OS and open browser accordingly
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open http://localhost:8501
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v xdg-open > /dev/null; then
        xdg-open http://localhost:8501
    elif command -v gnome-open > /dev/null; then
        gnome-open http://localhost:8501
    else
        echo "Please manually open: http://localhost:8501"
    fi
else
    echo "Please manually open: http://localhost:8501"
fi

echo ""
echo "========================================"
echo " Application is starting!"
echo "========================================"
echo ""
echo "- Web UI: http://localhost:8501"
echo "- Ollama API: http://localhost:11434"
echo ""
echo "To view logs:    docker-compose logs -f"
echo "To stop:         docker-compose down"
echo ""
echo "The browser should open automatically."
echo "If not, manually navigate to: http://localhost:8501"
echo ""

