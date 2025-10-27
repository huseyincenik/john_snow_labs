@echo off
REM RAG QA Chatbot - Automatic Startup Script for Windows
REM This script starts Docker containers and opens the browser

echo ========================================
echo  RAG QA Chatbot - Starting Application
echo ========================================
echo.

echo [1/3] Starting Docker containers...
docker-compose up -d

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to start Docker containers!
    echo Please make sure Docker Desktop is running.
    pause
    exit /b 1
)

echo.
echo [2/3] Waiting for application to be ready...
echo This may take 30-60 seconds...
timeout /t 30 /nobreak > nul

echo.
echo [3/3] Opening browser...
start http://localhost:8501

echo.
echo ========================================
echo  Application is starting!
echo ========================================
echo.
echo - Web UI: http://localhost:8501
echo - Ollama API: http://localhost:11434
echo.
echo To view logs:    docker-compose logs -f
echo To stop:         docker-compose down
echo.
echo The browser should open automatically.
echo If not, manually navigate to: http://localhost:8501
echo.
pause

