#!/bin/bash

# Start local WandB server
echo "Starting local WandB server..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

# Check if wandb-local container is already running
if docker ps | grep -q wandb-local; then
    echo "WandB local server is already running at http://localhost:8080"
else
    echo "Starting new WandB local server..."
    docker run --rm -d -v wandb:/vol -p 8080:8080 --name wandb-local wandb/local
    
    # Wait for server to start
    echo "Waiting for server to start..."
    sleep 10
    
    # Check if server is running
    if docker ps | grep -q wandb-local; then
        echo "✅ WandB local server started successfully!"
        echo "🌐 Access the web interface at: http://localhost:8080"
        echo "📝 Server logs: docker logs wandb-local"
    else
        echo "❌ Failed to start WandB server"
        exit 1
    fi
fi

echo ""
echo "Environment variables for your project:"
echo "export WANDB_BASE_URL=http://localhost:8080"
echo "export WANDB_MODE=offline  # Use offline mode to avoid auth issues"