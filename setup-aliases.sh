#!/bin/bash
# setup-aliases.sh - Setup convenient aliases

echo "Setting up project aliases..."

# Add aliases to .bashrc
cat >> ~/.bashrc << 'EOF'

# RAG Project Aliases
alias rag-start='cd /home/ubuntu/Desktop/Rag_System_Project/sail_Saas_Ai && ./start-server.sh'
alias rag-stop='pkill -f "uvicorn.*app.main:app"'
alias rag-status='ps aux | grep uvicorn | grep -v grep'
alias rag-logs='tail -f /tmp/rag-server.log'
alias rag-cd='cd /home/ubuntu/Desktop/Rag_System_Project/sail_Saas_Ai'

EOF

echo "✅ Aliases added to ~/.bashrc"
echo "Run 'source ~/.bashrc' or restart terminal to use aliases"
echo ""
echo "Available commands:"
echo "  rag-start  - Start the server (kills existing first)"
echo "  rag-stop   - Stop the server"
echo "  rag-status - Check if server is running"
echo "  rag-cd     - Go to project directory"