# 🚀 Beginner's Step-by-Step Deployment Guide

## Phase 1: Prepare Your Code for GitHub

### Step 1: Clean Up Sensitive Data
1. **IMPORTANT**: Never commit API keys to GitHub!
2. Replace your current .env files with the .env.production templates
3. Keep your actual API keys safe for server setup

### Step 2: Fix Code Issues
Before pushing to GitHub, fix the issues found in the code review:

**Backend Issue (sail_Saas_Ai/app/main.py):**
- Add missing router registrations for feedback and dashboard with client_id prefix
- See Code Issues panel for exact fix

**Frontend Issue (sail_Web_Ai/src/components/ChatSection.tsx):**
- Improve HTML parsing method
- See Code Issues panel for exact fix

### Step 3: Initialize Git and Push to GitHub

```bash
# Navigate to your project
cd /home/ubuntu/Desktop/Rag_System_Project

# Backend Repository
cd sail_Saas_Ai
git init
git add .
git commit -m "Initial backend commit"

# Create repository on GitHub: sail-saas-ai-backend
git remote add origin https://github.com/YOUR_USERNAME/sail-saas-ai-backend.git
git branch -M main
git push -u origin main

# Frontend Repository  
cd ../sail_Web_Ai
git init
git add .
git commit -m "Initial frontend commit"

# Create repository on GitHub: sail-web-ai-frontend
git remote add origin https://github.com/YOUR_USERNAME/sail-web-ai-frontend.git
git branch -M main
git push -u origin main
```

## Phase 2: Server Setup

### Step 1: Get a Linux Server
**Recommended Options:**
- **DigitalOcean**: $5-10/month droplet
- **AWS EC2**: t3.micro (free tier eligible)
- **Linode**: $5/month nanode
- **Vultr**: $2.50-5/month instance

**Minimum Requirements:**
- 1GB RAM (2GB recommended)
- 25GB SSD storage
- Ubuntu 20.04 or 22.04

### Step 2: Get a Domain Name
- Purchase from Namecheap, GoDaddy, or Cloudflare
- Point A record to your server IP
- Example: yourdomain.com → 123.456.789.123

### Step 3: Connect to Your Server
```bash
# SSH into your server
ssh root@your-server-ip

# Or if you have a user account
ssh username@your-server-ip
```

### Step 4: Install Required Software
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

# Install Git
sudo apt install git -y

# Install Certbot for SSL
sudo apt install certbot python3-certbot-nginx -y

# Logout and login again for Docker permissions
exit
ssh username@your-server-ip
```

## Phase 3: Deploy Your Applications

### Step 1: Create Project Structure
```bash
# Create project directory
sudo mkdir -p /opt/sail-projects
sudo chown $USER:$USER /opt/sail-projects
cd /opt/sail-projects

# Clone your repositories
git clone https://github.com/YOUR_USERNAME/sail-saas-ai-backend.git backend
git clone https://github.com/YOUR_USERNAME/sail-web-ai-frontend.git frontend
```

### Step 2: Configure Environment Variables
```bash
# Backend environment
cd /opt/sail-projects/backend
cp .env.production .env
nano .env  # Add your actual OpenAI API key and other settings

# Frontend environment
cd /opt/sail-projects/frontend
cp .env.production .env
nano .env  # Update API URL to your domain
```

### Step 3: Deploy Applications
```bash
# Copy deployment script
cp /opt/sail-projects/backend/deploy.sh /opt/sail-projects/
chmod +x /opt/sail-projects/deploy.sh

# Run deployment
sudo /opt/sail-projects/deploy.sh all
```

### Step 4: Configure Nginx
```bash
# Copy nginx configuration
sudo cp /opt/sail-projects/backend/nginx-config.conf /etc/nginx/sites-available/sail-projects

# Update domain name in config
sudo nano /etc/nginx/sites-available/sail-projects
# Replace 'yourdomain.com' with your actual domain

# Enable site
sudo ln -s /etc/nginx/sites-available/sail-projects /etc/nginx/sites-enabled/
sudo nginx -t  # Test configuration
sudo systemctl reload nginx
```

### Step 5: Setup SSL Certificate
```bash
# Get SSL certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Test auto-renewal
sudo certbot renew --dry-run
```

## Phase 4: Verify Deployment

### Step 1: Check Services
```bash
# Check Docker containers
docker ps

# Check logs
docker logs sail-backend
docker logs sail-frontend

# Check Nginx
sudo systemctl status nginx
```

### Step 2: Test Your Application
1. Visit https://yourdomain.com
2. Test frontend functionality
3. Test API endpoints
4. Check chat functionality

## Phase 5: Future Updates Workflow

### When You Want to Make Changes:

1. **Make changes locally**
2. **Test locally**
3. **Commit and push to GitHub:**
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin main
   ```
4. **Deploy to server:**
   ```bash
   # SSH to server
   ssh username@your-server-ip
   
   # Run deployment
   sudo /opt/sail-projects/deploy.sh all
   ```

## 🛠️ Troubleshooting

### Common Issues:

1. **Docker permission denied:**
   ```bash
   sudo usermod -aG docker $USER
   # Logout and login again
   ```

2. **Port already in use:**
   ```bash
   sudo lsof -i :8000  # Check what's using port 8000
   sudo kill -9 PID    # Kill the process
   ```

3. **SSL certificate issues:**
   ```bash
   sudo certbot certificates  # Check certificates
   sudo certbot renew        # Renew if needed
   ```

4. **Application not starting:**
   ```bash
   docker logs sail-backend   # Check backend logs
   docker logs sail-frontend  # Check frontend logs
   ```

## 📞 Support

If you encounter issues:
1. Check the logs first
2. Verify all environment variables are set
3. Ensure your domain DNS is pointing to the server
4. Check firewall settings (ports 80, 443, 22 should be open)

## 🎉 Success!

Once everything is working:
- Your backend API will be available at: https://yourdomain.com/api/
- Your frontend will be available at: https://yourdomain.com/
- You can make changes locally and deploy easily with the deployment script

Remember to:
- Keep your server updated
- Monitor your applications
- Backup your data regularly
- Keep your API keys secure