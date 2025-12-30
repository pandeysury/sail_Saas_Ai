# 🚀 Complete Linux Server Deployment Guide

## Project Overview
- **Backend**: sail_Saas_Ai (FastAPI + RAG System)
- **Frontend**: sail_Web_Ai (React + TypeScript)
- **Target**: Linux Server Deployment

## 📋 Pre-Deployment Checklist

### 1. Fix Critical Issues First
- [ ] Fix router configuration in main.py (see Code Issues panel)
- [ ] Improve HTML parsing in ChatSection.tsx
- [ ] Remove sensitive data from .env files
- [ ] Update API keys and endpoints

### 2. Environment Setup
- [ ] Clean .env files
- [ ] Update configuration for production
- [ ] Prepare Docker configurations

## 🔧 Step-by-Step Deployment Process

### Phase 1: GitHub Setup & Code Preparation

#### 1.1 Initialize Git Repositories
```bash
# Navigate to your project root
cd /home/ubuntu/Desktop/Rag_System_Project

# Initialize git for backend
cd sail_Saas_Ai
git init
git add .
git commit -m "Initial backend commit"

# Initialize git for frontend  
cd ../sail_Web_Ai
git init
git add .
git commit -m "Initial frontend commit"
```

#### 1.2 Create GitHub Repositories
1. Go to GitHub.com
2. Create two repositories:
   - `sail-saas-ai-backend`
   - `sail-web-ai-frontend`
3. Follow GitHub instructions to push your code

### Phase 2: Server Preparation

#### 2.1 Server Requirements
- Ubuntu 20.04+ or similar Linux distribution
- Docker & Docker Compose
- Nginx (for reverse proxy)
- SSL certificate (Let's Encrypt recommended)

#### 2.2 Install Dependencies on Server
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Nginx
sudo apt install nginx -y

# Install Certbot for SSL
sudo apt install certbot python3-certbot-nginx -y
```

### Phase 3: Production Configuration

#### 3.1 Environment Variables (Production)
Create secure .env files without sensitive data committed to GitHub.

#### 3.2 Docker Configuration
- Multi-stage build for optimized images
- Proper volume mounting for data persistence
- Health checks and restart policies

#### 3.3 Nginx Configuration
- Reverse proxy setup
- SSL termination
- Static file serving
- Load balancing (if needed)

### Phase 4: Deployment Automation

#### 4.1 Deployment Script
Create automated deployment script for easy updates.

#### 4.2 CI/CD Pipeline (Optional)
Set up GitHub Actions for automated deployments.

## 🔄 Development Workflow

### Local Development → GitHub → Server Deployment

1. **Local Changes**: Make changes on your local machine
2. **Git Commit**: Commit and push to GitHub
3. **Server Pull**: Pull changes on server and redeploy
4. **Zero Downtime**: Use Docker for seamless updates

## 📁 Recommended Directory Structure on Server

```
/opt/sail-projects/
├── backend/
│   ├── docker-compose.yml
│   ├── .env.production
│   └── app/
├── frontend/
│   ├── docker-compose.yml
│   ├── .env.production
│   └── dist/
├── nginx/
│   └── sites-available/
├── ssl/
└── scripts/
    ├── deploy.sh
    └── backup.sh
```

## 🛡️ Security Considerations

- [ ] Use environment variables for secrets
- [ ] Enable firewall (UFW)
- [ ] Regular security updates
- [ ] SSL/TLS encryption
- [ ] Database backups
- [ ] Access logging

## 📊 Monitoring & Maintenance

- [ ] Docker container health monitoring
- [ ] Log aggregation
- [ ] Performance monitoring
- [ ] Automated backups
- [ ] Update notifications

## 🚨 Troubleshooting

Common issues and solutions will be documented here as they arise.

---

**Next Steps**: Follow the detailed implementation guide below.