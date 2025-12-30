#!/bin/bash

# Sail Projects Deployment Script
# Usage: ./deploy.sh [backend|frontend|all]

set -e

PROJECT_DIR="/opt/sail-projects"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check if running as root or with sudo
check_permissions() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root or with sudo"
    fi
}

# Create project directories
setup_directories() {
    log "Setting up project directories..."
    mkdir -p $PROJECT_DIR/{backend,frontend,nginx,ssl,scripts,logs}
    chown -R $USER:$USER $PROJECT_DIR
}

# Deploy backend
deploy_backend() {
    log "Deploying backend..."
    
    cd $BACKEND_DIR
    
    # Pull latest changes
    git pull origin main
    
    # Stop existing containers
    docker-compose -f docker-compose.prod.yml down
    
    # Build and start new containers
    docker-compose -f docker-compose.prod.yml up -d --build
    
    # Wait for health check
    log "Waiting for backend health check..."
    sleep 30
    
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log "Backend deployed successfully!"
    else
        error "Backend health check failed!"
    fi
}

# Deploy frontend
deploy_frontend() {
    log "Deploying frontend..."
    
    cd $FRONTEND_DIR
    
    # Pull latest changes
    git pull origin main
    
    # Build and deploy
    docker-compose -f docker-compose.prod.yml down frontend
    docker-compose -f docker-compose.prod.yml up -d --build frontend
    
    log "Frontend deployed successfully!"
}

# Backup data
backup_data() {
    log "Creating backup..."
    
    BACKUP_DIR="/opt/backups/sail-$(date +%Y%m%d-%H%M%S)"
    mkdir -p $BACKUP_DIR
    
    # Backup databases and data
    cp -r $BACKEND_DIR/app/data $BACKUP_DIR/
    
    log "Backup created at $BACKUP_DIR"
}

# Main deployment function
main() {
    check_permissions
    setup_directories
    
    case "${1:-all}" in
        "backend")
            backup_data
            deploy_backend
            ;;
        "frontend")
            deploy_frontend
            ;;
        "all")
            backup_data
            deploy_backend
            deploy_frontend
            ;;
        *)
            echo "Usage: $0 [backend|frontend|all]"
            exit 1
            ;;
    esac
    
    log "Deployment completed successfully!"
}

main "$@"