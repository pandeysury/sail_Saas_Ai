# RAG FastAPI (Hybrid BM25 + Vector + Rerank)

Production-grade FastAPI service for Retrieval-Augmented Generation with hybrid search capabilities.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone the repository
git clone <your-repo-url>
cd latest_pull_backend

# Copy environment template
cp .env.example .env

# Add your OpenAI API key to .env
nano .env
# Replace: OPENAI_API_KEY=your_openai_api_key_here
# With: OPENAI_API_KEY=sk-your-actual-key
```

### 2. Run Locally (Recommended)
```bash
# Start the application
./start-local-no-docker.sh
```

### 3. Access Your Application
- **Main App**: http://localhost:8000
- **Dashboard**: http://localhost:8000/static/dashboard-test.html
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/healthz

## 📊 Features

### Core RAG Capabilities
- **Hybrid Search**: BM25 + Vector similarity + Reranking
- **Multi-tenant**: Isolated data per client
- **Conversational**: Chat history and context
- **Intent Detection**: Automatic query classification
- **Entity Recognition**: Enhanced query understanding

### Dashboard & Analytics
- **Real-time Metrics**: Query volume, quality, performance
- **Q&A Monitoring**: Full question-answer pairs with quality scores
- **Performance Analytics**: Response times and trends
- **Client Statistics**: Per-tenant usage analytics

### API Endpoints
- `POST /api/ask` — Main RAG query endpoint
- `GET /api/history` — Conversation history
- `GET /api/dashboard/*` — Analytics endpoints
- `GET /healthz` — Health check
- `GET /readyz` — Readiness check

## 🛠️ Development

### Docker Setup (Alternative)
```bash
# Fix Docker permissions (if needed)
./setup-docker.sh

# Start with Docker
docker-compose -f docker-compose.local.yml up --build
```

### Testing
```bash
# Add sample data for testing
python3 add_sample_data.py

# Test dashboard API
python3 test_dashboard_api.py
```

## 📁 Project Structure

```
latest_pull_backend/
├── app/
│   ├── routers/          # API endpoints
│   │   ├── query.py      # Main RAG endpoint
│   │   ├── chat.py       # Chat functionality
│   │   ├── dashboard.py  # Analytics API
│   │   └── feedback.py   # Feedback system
│   ├── services/         # Business logic
│   ├── utils/           # Utilities
│   ├── config.py        # Configuration
│   ├── main.py          # FastAPI app
│   └── memory_store.py  # Chat history
├── static/              # Frontend files
├── tools/               # Indexing scripts
├── scripts/             # Helper scripts
└── requirements.txt     # Dependencies
```

## 🔧 Configuration

Key environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `DEFAULT_CLIENT_ID` | Default tenant | `rsms` |
| `PORT` | Server port | `8000` |
| `ALLOW_ORIGINS` | CORS origins | `*` |
| `OPENAI_CHAT_MODEL` | Chat model | `gpt-4o-mini` |
| `INITIAL_RETRIEVE_K` | Initial retrieval count | `12` |

## 📚 Usage Examples

### Query the RAG System
```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What records should be maintained for passage planning?",
    "client_id": "rsms",
    "conversation_id": "test-123"
  }'
```

### Get Dashboard Metrics
```bash
curl http://localhost:8000/api/dashboard/overview
```

## 🚢 Maritime Domain

This system is optimized for maritime Safety Management System (SMS) documentation:

- **Passage Planning**: Route planning procedures and records
- **Bridge Operations**: Navigation and watchkeeping procedures  
- **Cargo Operations**: Loading, securing, and monitoring procedures
- **Emergency Response**: Fire, collision, and medical emergency procedures
- **Compliance**: SOLAS, MARPOL, and flag state requirements

## 🔍 Advanced Features

### Query Intent Detection
- **Recordkeeping**: "What records should be maintained..."
- **Procedural**: "How to conduct..."
- **Requirements**: "What are the requirements..."
- **Summarization**: "Summarize the procedure..."

### Multi-tenant Architecture
- Isolated data per client
- Client-scoped conversation history
- Per-tenant analytics and metrics

### Performance Optimization
- Hybrid retrieval (BM25 + Vector)
- LLM-based reranking
- Caching and connection pooling
- Async processing where possible

## 📈 Monitoring

The dashboard provides:
- **Overview Metrics**: Total queries, avg quality, response time
- **Q&A Details**: Full questions and answers with quality indicators
- **Performance Timeline**: Daily trends and patterns
- **Client Analytics**: Per-tenant usage statistics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license here]

## 🆘 Support

For issues and questions:
1. Check the [API documentation](http://localhost:8000/docs)
2. Review the dashboard for system health
3. Check logs for error details
4. Open an issue on GitHub
