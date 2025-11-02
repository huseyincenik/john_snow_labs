# Deployment Guide

Complete deployment instructions for different environments.

## Development Deployment

### Local Development with Docker

```bash
# Clone repository
git clone <repository-url>
cd rag_qa_chatbot_application

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Access application
open http://localhost:8501
```

**Note**: The application comes with a pre-loaded PubMed database in `current_db/` directory. This database is persistent and includes separate FAISS indices for both OpenAI and Qwen embeddings. User-uploaded documents are stored in a temporary in-memory database that doesn't persist after container restart.

### Local Development without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Install and start Ollama (separate terminal)
ollama serve
ollama pull qwen2.5:7b
ollama pull llama3

# Run application
streamlit run enhanced_app.py
```

## Production Deployment

### Option 1: Docker Compose (Recommended)

```bash
# Set environment variables
export OPENAI_API_KEY=your-key-here  # Optional

# Start in production mode
docker-compose -f docker-compose.yml up -d

# Enable auto-restart
docker-compose -f docker-compose.yml up -d --restart=always
```

### Option 2: Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-chatbot
spec:
  replicas: 2
  selector:
    matchLabels:
      app: rag-chatbot
  template:
    metadata:
      labels:
        app: rag-chatbot
    spec:
      containers:
        - name: ollama
          image: ollama/ollama:latest
          ports:
            - containerPort: 11434
          resources:
            limits:
              memory: "8Gi"
              cpu: "4"
        - name: chatbot
          image: rag-qa-chatbot:latest
          ports:
            - containerPort: 8501
          env:
            - name: OLLAMA_BASE_URL
              value: "http://localhost:11434"
          resources:
            limits:
              memory: "4Gi"
              cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: rag-chatbot-service
spec:
  type: LoadBalancer
  selector:
    app: rag-chatbot
  ports:
    - port: 80
      targetPort: 8501
```

Apply:

```bash
kubectl apply -f deployment.yaml
```

### Option 3: Cloud Deployment

#### AWS ECS

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

docker build -t rag-chatbot .
docker tag rag-chatbot:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/rag-chatbot:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/rag-chatbot:latest

# Create ECS task definition and service
aws ecs create-service --service-name rag-chatbot --task-definition rag-chatbot:1 --desired-count 2
```

#### Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/rag-chatbot
gcloud run deploy rag-chatbot \
  --image gcr.io/PROJECT-ID/rag-chatbot \
  --platform managed \
  --memory 8Gi \
  --cpu 4
```

#### Azure Container Instances

```bash
# Create container group
az container create \
  --resource-group myResourceGroup \
  --name rag-chatbot \
  --image rag-chatbot:latest \
  --cpu 4 \
  --memory 8 \
  --ports 8501 \
  --environment-variables OLLAMA_BASE_URL=http://ollama:11434
```

## Environment Configuration

### Required Environment Variables

```bash
# Application settings
PYTHONPATH=/app
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Ollama configuration
OLLAMA_BASE_URL=http://ollama:11434

# Optional: OpenAI API
OPENAI_API_KEY=sk-...
```

### Optional Environment Variables

```bash
# Cache settings
CACHE_ENABLED=true
CACHE_TTL_SECONDS=3600
CACHE_MAX_SIZE=100

# Model settings
DEFAULT_CHUNK_SIZE=800
DEFAULT_CHUNK_OVERLAP=100
DEFAULT_SEARCH_K=5
```

## Security Checklist

- [ ] Use HTTPS in production (reverse proxy)
- [ ] Set strong API keys
- [ ] Enable authentication (Streamlit auth)
- [ ] Implement rate limiting
- [ ] Set up firewall rules
- [ ] Enable logging and monitoring
- [ ] Regular security updates

## Monitoring

### Health Checks

```bash
# Application health
curl http://localhost:8501/_stcore/health

# Ollama health
curl http://localhost:11434/api/tags
```

### Logging

```bash
# Docker Compose logs
docker-compose logs -f

# Application logs
tail -f logs/app.log

# Export logs
docker-compose logs > deployment-logs.txt
```

### Metrics to Monitor

- Response time
- Cache hit rate
- Document processing time
- Memory usage
- CPU usage
- Disk space (for persistent `current_db/` and temporary `data/` directories)
- Database mode usage (Current DB, New DB, Current + New DB)
- Metadata structure compliance (JSON format, Chunk_Id removal)

## Backup and Recovery

### Backup Vector Store

```bash
# Backup persistent database (current_db)
tar -czf backup-current-db-$(date +%Y%m%d).tar.gz current_db/

# Backup runtime data (optional, contains temporary new DB data)
tar -czf backup-data-$(date +%Y%m%d).tar.gz data/

# Upload to S3
aws s3 cp backup-*.tar.gz s3://my-bucket/backups/
```

**Important Notes:**
- `current_db/` contains the persistent PubMed database (should be backed up)
- `data/` contains temporary in-memory databases and cache (optional to backup)
- New DB data is ephemeral and doesn't need backup

### Restore Vector Store

```bash
# Download from S3
aws s3 cp s3://my-bucket/backups/backup-current-db-20240101.tar.gz .

# Restore persistent database
tar -xzf backup-current-db-20240101.tar.gz
docker-compose restart rag-chatbot
```

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.yml
services:
  rag-chatbot:
    deploy:
      replicas: 3
    # ... rest of config
```

### Load Balancing

```nginx
# nginx.conf
upstream rag_chatbot {
    server chatbot1:8501;
    server chatbot2:8501;
    server chatbot3:8501;
}

server {
    listen 80;
    location / {
        proxy_pass http://rag_chatbot;
    }
}
```

## Performance Optimization

### Resource Limits

```yaml
# docker-compose.yml
services:
  ollama:
    deploy:
      resources:
        limits:
          cpus: "4"
          memory: 8G
        reservations:
          cpus: "2"
          memory: 4G
```

### Caching Strategy

- Enable semantic caching
- Set appropriate TTL (3600s recommended)
- Monitor cache hit rate
- Clear cache periodically

## Troubleshooting Production Issues

### Qwen2.5 API Connection Error

**Problem:** "Failed to connect to Ollama service" when initializing Qwen2.5

**Solutions:**

```bash
# 1. Check if Ollama container is running and healthy
docker-compose ps

# 2. Check Ollama logs to see if models are still downloading
docker-compose logs ollama

# 3. Verify Ollama service is responding
curl http://localhost:11434/api/tags

# 4. Check if qwen2.5:7b model is available
docker exec rag-ollama ollama list

# 5. Wait for model download (can take 5-10 minutes on first start)
# Monitor progress with:
docker-compose logs -f ollama

# 6. If model download failed, manually pull it
docker exec rag-ollama ollama pull qwen2.5:7b

# 7. Restart both services after model is ready
docker-compose restart
```

**Important Notes:**

- First startup can take 5-10 minutes to download Qwen2.5:7b model (4.7GB)
- Wait for the message "✅ All models downloaded and ready!" in logs
- The RAG chatbot container waits for Ollama to be healthy before starting

### High Memory Usage

```bash
# Check memory
docker stats

# Restart service
docker-compose restart rag-chatbot

# Increase limits in docker-compose.yml
# Add under ollama service:
#   deploy:
#     resources:
#       limits:
#         memory: 8G
```

### Slow Responses

```bash
# Check cache stats in UI
# Reduce number of sources
# Use OpenAI instead of Ollama

# For Qwen performance issues:
# 1. First query is always slow (model loading)
# 2. Subsequent queries should be faster
# 3. Consider using OpenAI for production
```

### Service Not Starting

```bash
# Check logs
docker-compose logs rag-chatbot
docker-compose logs ollama

# Verify ports are not in use
netstat -tulpn | grep 8501
netstat -tulpn | grep 11434

# Reset everything (WARNING: This deletes all data)
docker-compose down -v
docker-compose up -d

# Check health status
docker-compose ps
curl http://localhost:8501/_stcore/health
curl http://localhost:11434/api/tags
```

### Data and Logs Folder Issues

**Problem:** Old data or logs being included in Docker image

**Solution:**
The `.dockerignore` file is configured to exclude `data/`, `logs/`, and `current_db/` folders from the Docker image.
These folders are mounted as volumes from your host machine:

```yaml
volumes:
  - ./current_db:/app/current_db  # Persistent PubMed database
  - ./data:/app/data              # Runtime data (temporary new DB, cache)
  - ./logs:/app/logs              # Application logs
```

**Database Structure:**
- `current_db/`: Persistent PubMed database (separate indices for OpenAI and Qwen)
- `data/`: Runtime data including temporary new DB (in-memory, not persisted)
- `logs/`: Application logs

If you need to start fresh:

```bash
# Clear runtime data (WARNING: Deletes temporary new DB and cache)
rm -rf data/vectorstore/*
rm -rf data/cache/*

# Clear logs
rm -rf logs/*

# NOTE: current_db/ is preserved (contains pre-loaded PubMed database)

# Rebuild without cache
docker-compose build --no-cache
docker-compose up -d
```

**Important**: The `current_db/` directory contains the persistent PubMed database and should NOT be deleted unless you want to rebuild it from scratch.

## Maintenance

### Regular Tasks

- [ ] Weekly: Check logs for errors
- [ ] Weekly: Review cache performance
- [ ] Monthly: Update Docker images
- [ ] Monthly: Backup vector store
- [ ] Quarterly: Security audit
- [ ] Quarterly: Dependency updates

### Update Procedure

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Verify health
curl http://localhost:8501/_stcore/health
```

---

**For detailed feature documentation, see [README.md](README.md)**
