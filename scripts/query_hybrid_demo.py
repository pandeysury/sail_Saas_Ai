#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/query_hybrid_demo.py
CLI to query the SAME hybrid stack your API uses.

- Loads Chroma from <client>/index_store/chroma
- Rebuilds BM25 from <client>/index_store/chunks.jsonl
- Uses fusion (BM25 + Vector) and optional reranker per settings.json
"""

import json, argparse, logging
from pathlib import Path
from typing import List

from llama_index.core import StorageContext, VectorStoreIndex, TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

# Optional rerankers (only used if present in settings.json)
try:
    from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
    HAS_SBERT = True
except Exception:
    HAS_SBERT = False

try:
    from llama_index.postprocessor.openai_rerank import OpenAIReranker
    HAS_OPENAI_RERANK = True
except Exception:
    HAS_OPENAI_RERANK = False

import chromadb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
log = logging.getLogger("query-hybrid-demo")


def _load_nodes_from_chunks(chunks_path: Path) -> List[TextNode]:
    nodes: List[TextNode] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                nodes.append(
                    TextNode(
                        text=obj["text"],
                        metadata=obj.get("metadata", {}),
                        id_=obj.get("id_", None),
                    )
                )
            except Exception:
                continue
    return nodes


def build_retriever(client_root: Path):
    store_dir  = client_root / "index_store"
    chroma_dir = store_dir / "chroma"
    chunks_path = store_dir / "chunks.jsonl"
    settings_path = store_dir / "settings.json"

    settings = {}
    if settings_path.exists():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    # Defaults if not present
    bm25_k = int(settings.get("bm25_k", 8))
    fusion_mode = settings.get("fusion_mode", "reciprocal_rerank")
    reranker_name = settings.get("reranker", "none")  # "none" | "sbert" | "openai"

    # Vector store (Chroma)
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma_client.get_or_create_collection(
        name="docs", metadata={"hnsw:space": "cosine"}
    )
    vs = ChromaVectorStore(chroma_collection=collection)
    sc = StorageContext.from_defaults(vector_store=vs)

    # Important: attach embed model so scores/top_k match prod
    _ = OpenAIEmbedding(model=settings.get("embed_model", "text-embedding-3-large"))
    vindex = VectorStoreIndex.from_documents([], storage_context=sc)
    vec = vindex.as_retriever(similarity_top_k=bm25_k)

    # BM25 from chunks.jsonl
    nodes = _load_nodes_from_chunks(chunks_path)
    bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=bm25_k)

    # Optional reranker
    post = []
    if reranker_name == "sbert" and HAS_SBERT:
        post = [SentenceTransformerRerank(
            model="sentence-transformers/all-MiniLM-L6-v2", top_n=bm25_k
        )]
    elif reranker_name == "openai" and HAS_OPENAI_RERANK:
        post = [OpenAIReranker(top_n=bm25_k)]

    retriever = QueryFusionRetriever(
        retrievers=[bm25, vec],
        mode=fusion_mode,
        num_queries=1,
        postprocessors=post,
    )
    return retriever


def cli():
    ap = argparse.ArgumentParser(description="Query hybrid (BM25+Vector) like the API.")
    ap.add_argument("--client_root", required=True, help="Path to <CLIENTS_BASE>/<client>")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    r = build_retriever(Path(args.client_root))
    results = r.retrieve(args.query)[: args.k]

    print("\n=== Top Results ===")
    for i, n in enumerate(results, start=1):
        md = n.node.metadata or {}
        print(f"{i}. {md.get('file','')} :: {md.get('section_title','')}  [{md.get('slug_url','')}]")
        print(f"   score={getattr(n, 'score', None)}")
        print(f"   {n.node.get_content()[:300].replace('\\n',' ')}\n")


if __name__ == "__main__":
    cli()
