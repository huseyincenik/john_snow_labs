#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Load environment variables (especially QWEN_MODEL/QWEN_PORT) if present
if [ -f "config/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "config/.env"
  set +a
fi

IMAGE_NAME="data-curation-docetl"
QWEN_MODEL_NAME="${QWEN_MODEL:-qwen2.5:0.5b-instruct}"

echo "[1/3] Mevcut container ve imajlar temizleniyor"
docker compose down --volumes --remove-orphans >/dev/null 2>&1 || true
docker image rm -f "$IMAGE_NAME" >/dev/null 2>&1 || true
docker builder prune -f >/dev/null 2>&1 || true

echo "[2/3] Docker imajı oluşturuluyor: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" .

echo "[3/3] Qwen servisi başlatılıyor ve model doğrulanıyor"
docker compose up -d qwen

if ! docker compose exec -T qwen ollama list | grep -q "$QWEN_MODEL_NAME"; then
  echo "    -> '$QWEN_MODEL_NAME' modeli indiriliyor (birkaç dakika sürebilir)..."
  if ! docker compose exec -T qwen ollama pull "$QWEN_MODEL_NAME"; then
    echo "    -> Model indirilemedi. Lütfen 'docker compose exec qwen ollama list' ile mevcut modelleri kontrol edin"
    echo "       ve 'config/.env' içindeki QWEN_MODEL değerini geçerli bir modele güncelleyin."
    exit 1
  fi
else
  echo "    -> '$QWEN_MODEL_NAME' modeli zaten mevcut"
fi

echo "Tüm servisler ayağa kaldırılıyor (loglar bu terminalde takip edilecek)"
docker compose up --build

