#!/bin/bash

# Exit on error
set -e

# Configuration
IMAGE_NAME="astro-rag-app"
DOCKER_HUB_REPO="astro-rag-repo"
DOCKER_HUB_USERNAME="${DOCKER_USERNAME:-$USER}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Astronomy RAG Chatbot Docker image...${NC}"

# Create a temporary Dockerfile
cat > Dockerfile << EOL
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application with Gunicorn for production
CMD ["gunicorn", "app:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120"]
EOL

# Create requirements.txt if it doesn't exist
if [ ! -f requirements.txt ]; then
    echo -e "${YELLOW}Creating requirements.txt...${NC}"
    cat > requirements.txt << EOL
fastapi==0.100.0
uvicorn==0.22.0
gunicorn==20.1.0
langchain==0.0.267
python-dotenv==1.0.0
requests==2.31.0
pydantic==2.0.3
jinja2==3.1.2
google-generativeai==0.3.2
langchain-google-genai==0.0.5
EOL
fi

# Build the Docker image
echo -e "${GREEN}Building Docker image: ${IMAGE_NAME}${NC}"
docker build -t ${IMAGE_NAME} .

# Tag the image for Docker Hub
echo -e "${GREEN}Tagging image for Docker Hub: ${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}${NC}"
docker tag ${IMAGE_NAME} ${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}:latest

echo -e "${GREEN}Build complete!${NC}"
echo -e "${YELLOW}To push to Docker Hub, run:${NC}"
echo -e "  docker login"
echo -e "  docker push ${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}:latest"
echo
echo -e "${YELLOW}To run locally:${NC}"
echo -e "  docker run -p 8000:8000 ${IMAGE_NAME}"
echo
echo -e "${YELLOW}To deploy to Google Cloud Run:${NC}"
echo -e "  gcloud run deploy --image ${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}:latest --platform managed --allow-unauthenticated" 