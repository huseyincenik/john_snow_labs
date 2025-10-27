@echo off
REM RAG QA Chatbot - Stop Script for Windows
REM This script stops and cleans up Docker containers

echo ========================================
echo  RAG QA Chatbot - Stopping Application
echo ========================================
echo.

echo [1/2] Checking container status...
docker-compose ps -q >nul 2>&1

echo [2/2] Stopping Docker containers...
docker-compose down 2>nul

REM Don't exit with error if containers were already stopped
REM Just inform the user
echo.
echo ========================================
echo  Application stopped successfully!
echo ========================================
echo.
echo To start again:  start.bat
echo To view logs:    docker-compose logs -f
echo.
echo Note: Your data and vector store are preserved.
echo To remove all data: docker-compose down -v
echo.
pause

