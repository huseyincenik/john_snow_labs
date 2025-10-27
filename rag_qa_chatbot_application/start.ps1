# RAG QA Chatbot - Automatic Startup Script for Windows PowerShell
# This script starts Docker containers and opens the browser

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " RAG QA Chatbot - Starting Application" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/3] Starting Docker containers..." -ForegroundColor Yellow
docker-compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Failed to start Docker containers!" -ForegroundColor Red
    Write-Host "Please make sure Docker Desktop is running." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[2/3] Waiting for application to be ready..." -ForegroundColor Yellow
Write-Host "This may take 30-60 seconds..." -ForegroundColor Gray
Start-Sleep -Seconds 30

Write-Host ""
Write-Host "[3/3] Opening browser..." -ForegroundColor Yellow
Start-Process "http://localhost:8501"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " Application is starting!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "- Web UI: http://localhost:8501" -ForegroundColor White
Write-Host "- Ollama API: http://localhost:11434" -ForegroundColor White
Write-Host ""
Write-Host "To view logs:    docker-compose logs -f" -ForegroundColor Gray
Write-Host "To stop:         docker-compose down" -ForegroundColor Gray
Write-Host ""
Write-Host "The browser should open automatically." -ForegroundColor Cyan
Write-Host "If not, manually navigate to: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"

