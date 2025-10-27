#!/bin/bash
# RAG QA Chatbot - Stop Script for Linux/Mac
# This script stops and cleans up Docker containers

echo "========================================"
echo " RAG QA Chatbot - Stopping Application"
echo "========================================"
echo ""

echo "[1/2] Checking container status..."
docker-compose ps -q > /dev/null 2>&1

echo "[2/2] Stopping Docker containers..."
docker-compose down 2>/dev/null

# Don't exit with error if containers were already stopped
# Just inform the user

echo ""
echo "========================================"
echo " Application stopped successfully!"
echo "========================================"
echo ""
echo "To start again:  ./start.sh"
echo "To view logs:    docker-compose logs -f"
echo ""
echo "Note: Your data and vector store are preserved."
echo "To remove all data: docker-compose down -v"
echo ""

