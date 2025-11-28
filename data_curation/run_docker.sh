#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Clean up existing stack
echo "[1/2] Mevcut container ve imajlar temizleniyor"
docker compose down --volumes --remove-orphans >/dev/null 2>&1 || true
docker builder prune -f >/dev/null 2>&1 || true

echo "[2/2] Servisler ayağa kaldırılıyor"
docker compose up --build

