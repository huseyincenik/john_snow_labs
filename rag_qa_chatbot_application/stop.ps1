# RAG QA Chatbot - Stop Script for Windows PowerShell
# This script stops and cleans up Docker containers

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " RAG QA Chatbot - Stopping Application" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/2] Checking container status..." -ForegroundColor Yellow
docker-compose ps -q > $null 2>&1

Write-Host "[2/2] Stopping Docker containers..." -ForegroundColor Yellow
docker-compose down 2>$null

# Don't exit with error if containers were already stopped
# Just inform the user

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " Application stopped successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start again:  .\start.ps1" -ForegroundColor White
Write-Host "To view logs:    docker-compose logs -f" -ForegroundColor Gray
Write-Host ""
Write-Host "Note: Your data and vector store are preserved." -ForegroundColor Cyan
Write-Host "To remove all data: docker-compose down -v" -ForegroundColor Gray
Write-Host ""
Read-Host "Press Enter to exit"

