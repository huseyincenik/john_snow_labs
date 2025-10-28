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
echo "[2/4] Waiting for Ollama to download models and become healthy..."
echo "This may take 5-10 minutes for first time setup..."
echo ""

max_wait_time=900  # 15 minutes maximum
elapsed=0
check_interval=10

while [ $elapsed -lt $max_wait_time ]; do
    ollama_health=$(docker inspect --format='{{.State.Health.Status}}' rag-ollama 2>/dev/null)
    
    if [ "$ollama_health" = "healthy" ]; then
        echo "✅ Ollama is ready!"
        break
    fi
    
    # Show progress
    minutes=$((elapsed / 60))
    seconds=$((elapsed % 60))
    echo "⏳ Waiting for Ollama... (${minutes}m ${seconds}s elapsed)"
    
    sleep $check_interval
    elapsed=$((elapsed + check_interval))
done

if [ $elapsed -ge $max_wait_time ]; then
    echo "⚠️  Timeout waiting for Ollama. Check logs with: docker-compose logs ollama"
fi

echo ""
echo "[3/4] Waiting for RAG Chatbot to be ready..."

max_wait_time=60  # 1 minute for chatbot
elapsed=0

while [ $elapsed -lt $max_wait_time ]; do
    chatbot_health=$(docker inspect --format='{{.State.Health.Status}}' rag-qa-chatbot 2>/dev/null)
    
    if [ "$chatbot_health" = "healthy" ]; then
        echo "✅ RAG Chatbot is ready!"
        break
    fi
    
    echo "⏳ Waiting for chatbot... (${elapsed}s elapsed)"
    sleep 5
    elapsed=$((elapsed + 5))
done

echo ""
echo "[4/4] Opening browser..."
sleep 2

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
echo " ✅ Application is Ready!"
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

