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
Write-Host "[2/4] Waiting for Ollama to download models and become healthy..." -ForegroundColor Yellow
Write-Host "This may take 5-10 minutes for first time setup..." -ForegroundColor Gray
Write-Host ""

$maxWaitTime = 900  # 15 minutes maximum
$elapsed = 0
$checkInterval = 10

while ($elapsed -lt $maxWaitTime) {
    $ollamaHealth = docker inspect --format='{{.State.Health.Status}}' rag-ollama 2>$null
    
    if ($ollamaHealth -eq "healthy") {
        Write-Host "✅ Ollama is ready!" -ForegroundColor Green
        break
    }
    
    # Show progress
    $minutes = [math]::Floor($elapsed / 60)
    $seconds = $elapsed % 60
    Write-Host "⏳ Waiting for Ollama... (${minutes}m ${seconds}s elapsed)" -ForegroundColor Gray
    
    Start-Sleep -Seconds $checkInterval
    $elapsed += $checkInterval
}

if ($elapsed -ge $maxWaitTime) {
    Write-Host "⚠️  Timeout waiting for Ollama. Check logs with: docker-compose logs ollama" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[3/4] Waiting for RAG Chatbot to be ready..." -ForegroundColor Yellow

$maxWaitTime = 60  # 1 minute for chatbot
$elapsed = 0

while ($elapsed -lt $maxWaitTime) {
    $chatbotHealth = docker inspect --format='{{.State.Health.Status}}' rag-qa-chatbot 2>$null
    
    if ($chatbotHealth -eq "healthy") {
        Write-Host "✅ RAG Chatbot is ready!" -ForegroundColor Green
        break
    }
    
    Write-Host "⏳ Waiting for chatbot... (${elapsed}s elapsed)" -ForegroundColor Gray
    Start-Sleep -Seconds 5
    $elapsed += 5
}

Write-Host ""
Write-Host "[4/4] Opening browser..." -ForegroundColor Yellow
Start-Sleep -Seconds 2
Start-Process "http://localhost:8501"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " ✅ Application is Ready!" -ForegroundColor Green
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

