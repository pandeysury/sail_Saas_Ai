# RAG FastAPI (Hybrid BM25 + Vector + Rerank)

Production-grade FastAPI service that loads the persisted hybrid retriever built by `indexer_hybrid_li.py`.

## Quickstart
1) Create your index with `indexer_hybrid_li.py` (same machine/image):
   - This produces `<CLIENT_ROOT>/index_store/{chroma, nodes.jsonl, settings.json}`.

2) Configure env:
   - Copy `.env.example` to `.env` and set `CLIENT_ROOT` and `OPENAI_API_KEY`.

3) Install & run (dev):
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

4) Production with Gunicorn:
```bash
gunicorn app.main:app -c gunicorn.conf.py
```

## Endpoints
- `GET /healthz` — liveness
- `GET /readyz` — retriever loaded?
- `POST /query` — RAG query
- `POST /rebuild` — optional rebuild BM25 from nodes.jsonl (no re-embed)
- `GET /config` — current settings snapshot

## Notes
- Vector index lives under `<CLIENT_ROOT>/index_store/chroma`.
- BM25 is rebuilt from `<CLIENT_ROOT>/index_store/nodes.jsonl` at startup.
- Reranker selection & K-values read from `<CLIENT_ROOT>/index_store/settings.json`.
- To re-index documents (incremental), run `indexer_hybrid_li.py` again; this service can stay up.


## Scripts (Indexer & CLI Query)
The project includes the two helper scripts under `scripts/`:

- `scripts/indexer_hybrid_li.py` — builds the hybrid index (HTMLNodeParser + SentenceSplitter, Vector + BM25, reranker, incremental)
- `scripts/query_hybrid_demo.py` — quick CLI to test the persisted index

### Run the indexer
```bash
python scripts/indexer_hybrid_li.py   --client_root "C:/sms/andriki"   --tags "h1,h2,h3,ul,li,p"   --bm25_k 12   --fusion_mode reciprocal_rerank   --chunk_size 900   --chunk_overlap 150   --reranker sbert
```

### Test queries from CLI
```bash
python scripts/query_hybrid_demo.py --client_root "C:/sms/andriki" --query "visitor induction checklist" --k 6
```


---

## Local HTML Tester (no React)
Point your `index.html` to the API and open it in a browser.
- Place your static files somewhere your web server can serve:
  - `/srv/www/index.html`, `/srv/www/script.js`, `/srv/www/style.css`
- Ensure the FastAPI runs at the expected base URL (`/api`), which this project exposes.
- The frontend calls:
  - `GET /api/history?conversation_id=...`
  - `POST /api/ask` with `{ conversation_id, client_id, question }`

This project now implements those exact endpoints and persists chat turns using SQLite (see `app/memory_store.py`).


### Conversation endpoints
- `GET /api/threads` → list conversations (id, last message, counts)
- `POST /api/clear_conversation` → body `{ "conversation_id": "xyz" }`
- `DELETE /api/conversation/{conversation_id}` → delete all rows
- `POST /api/cleanup` → body `{ "days_old": 30, "keep_persistent": true }`
- `GET /api/search_history?term=..."` → search across past messages



## Multitenancy & Isolation
- All conversations are **namespaced** as `client_id::conversation_id` in the DB.
- Every endpoint takes a `client_id` and only returns data for that client.
- If `client_id` is omitted, the backend uses `DEFAULT_CLIENT_ID` from `.env`.

### Endpoint parameters (client_id)
- `POST /api/ask` — body includes `client_id`
- `GET /api/history?conversation_id=...&client_id=...`
- `GET /api/threads?client_id=...`
- `DELETE /api/conversation/{conversation_id}?client_id=...`
- `POST /api/clear_conversation` — body includes `client_id`
- `GET /api/search_history?term=...&client_id=...`
- `POST /api/set_title` — body includes `client_id`
- `GET /api/thread/{conversation_id}?client_id=...`



## SIRE-driven enrichment (rules.yaml)
1. Build rules from your SIRE SQLite:
   ```bash
   python tools/sire_rules_builder.py --sqlite "C:\sms\mterms\sire_viq.sqlite" --out rules.yaml --topk 25
   ```
2. Index your HTML SOPs with enrichment:
   ```bash
   python tools/indexer_hybrid_li.py --docs "C:\sms\andriki\documents" --rules rules.yaml --out storage_hybrid
   ```
3. Point backend to the new storage (in `.env`):
   ```env
   STORAGE_DIR=storage_hybrid
   ```

### Optional: OpenAI LLM reranker
Enable LLM reranking over top passages:
```env
RERANK_MODE=llm
OPENAI_RERANK_MODEL=gpt-4o-mini
RERANK_MAX_PASSAGES=40
OPENAI_API_KEY=sk-...
```
