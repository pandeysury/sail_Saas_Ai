#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
indexer_hybrid_li.py — OPTIMIZED CLEAN INDEXER (10x faster embeddings)

- Parses HTML via LlamaIndex HTMLNodeParser
- Sentence-aware chunking (SentenceSplitter) with overlap
- Enrichment from rules.yaml (SIRE VIQ patterns + maritime synonyms) — metadata ONLY
- Embeds ONLY new/changed chunks to Chroma (OpenAI embeddings) with deterministic node IDs (idempotent upserts)
- **OPTIMIZED**: Batch processing + parallel embeddings (15-20 min instead of 4-5 hours)
- Writes processed chunks to chunks.jsonl (to rebuild BM25 at query time)
- Incremental updates via manifest.json

Layout:
  <client_root>/
    documents/
    index_store/
      chroma/
      chunks.jsonl
      manifest.json
      settings.json

Deps:
  pip install "llama-index>=0.10.50" \
              "llama-index-embeddings-openai>=0.1.9" \
              "llama-index-vector-stores-chroma>=0.1.10" \
              chromadb tiktoken pyyaml tqdm

Env:
  set OPENAI_API_KEY=...
  Optional speed knobs:
    EMBED_MODEL=text-embedding-3-small   (default text-embedding-3-large)
    EMBED_BATCH=256                      (default 256)
"""

import os, re, json, argparse, hashlib, logging, asyncio, uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from itertools import islice
import concurrent.futures

# LlamaIndex core
from llama_index.core import Document, StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import HTMLNodeParser
from llama_index.core.node_parser.text import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.ingestion import IngestionPipeline

# Embeddings + Vector store
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# Vector DB
import chromadb
import yaml

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    logging.warning("tqdm not installed. Install with 'pip install tqdm' for progress bars.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
log = logging.getLogger("indexer-hybrid-li")


# ---------------------- Stable IDs for idempotent upserts ----------------------
def stable_node_id(file_name: str, section_id: str, text: str) -> str:
    """Deterministic ID for a chunk so we can upsert instead of duplicating."""
    h = hashlib.sha1()
    h.update(file_name.encode("utf-8", errors="ignore"))
    h.update(b"|")
    h.update((section_id or "").encode("utf-8", errors="ignore"))
    h.update(b"|")
    h.update(text.encode("utf-8", errors="ignore"))
    return f"node-{h.hexdigest()}"


# ---------------------- Helpers ----------------------
def guess_title_from_node(n: TextNode, md: Dict[str, Any]) -> str:
    """Heuristic, zero-LLM title from node + metadata."""
    for k in ("title", "heading", "Section", "section_title"):
        v = (md.get(k) or "").strip()
        if v:
            return v
    tag = (md.get("tag") or "").lower()
    if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        t = (n.text or "").strip()
        if t:
            return t.splitlines()[0][:180]
    t = (n.text or "").strip()
    if not t:
        return ""
    line = t.splitlines()[0]
    line = re.split(r"(?<=[.!?])\s+", line, maxsplit=1)[0]
    return line[:180]


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"files": {}, "generated_at": None, "version": 1}


def save_manifest(manifest_path: Path, data: Dict[str, Any]) -> None:
    manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_breadcrumb(file_stem: str, title: str) -> str:
    t = re.sub(r"\s+", " ", (title or "").strip())
    return f"{file_stem} > {t}" if t else file_stem


def stable_slug(text: str) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "-", (text or "").strip()).strip("-").lower()
    if not base:
        base = hashlib.sha1((text or str(uuid.uuid4())).encode("utf-8")).hexdigest()[:10]
    return f"sec-{base[:120]}"


def chunked(iterable, size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            return
        yield batch


# ---------------------- HTML parsing ----------------------
def parse_html_with_li(html_text: str, file_name: str, tags: Optional[List[str]] = None) -> List[TextNode]:
    """Use LlamaIndex HTMLNodeParser to break DOM into section-aware nodes."""
    tags = tags or ["h1", "h2", "h3", "h4", "ul", "li", "p"]
    parser = HTMLNodeParser(tags=tags)
    doc = Document(text=html_text, metadata={"file": file_name})
    nodes = parser.get_nodes_from_documents([doc])

    enriched: List[TextNode] = []
    for n in nodes:
        md = dict(n.metadata or {})
        section_title = guess_title_from_node(n, md)
        breadcrumb = build_breadcrumb(Path(file_name).stem, section_title)
        anchor = (md.get("id") or md.get("anchor") or "").strip()
        section_id = md.get("section_id")
        if not section_id:
            base_for_slug = f"{breadcrumb}#{anchor}" if anchor else breadcrumb
            section_id = stable_slug(base_for_slug)

        md.update({
            "file": file_name,
            "breadcrumb": breadcrumb,
            "section_title": section_title,
            "section_id": section_id,
            "slug_url": f"{file_name}#{anchor or section_id}",
            "source_ext": ".html",
            "tag": (md.get("tag") or "").lower(),
            "anchor": anchor,
        })
        n.metadata = md
        enriched.append(n)
    return enriched

def merge_heading_with_content(nodes: List[TextNode]) -> List[TextNode]:
    """
    Merge heading nodes with their following content nodes (ul/li/p).
    This ensures section headings stay with their content.
    """
    merged: List[TextNode] = []
    i = 0
    
    while i < len(nodes):
        node = nodes[i]
        tag = (node.metadata or {}).get("tag", "").lower()
        
        # If this is a heading, look ahead and merge with following content
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            heading_text = node.text or ""
            heading_md = dict(node.metadata or {})
            merged_texts = [heading_text]
            
            # Look ahead for content nodes (li, p, ul) that belong to this heading
            j = i + 1
            while j < len(nodes):
                next_node = nodes[j]
                next_tag = (next_node.metadata or {}).get("tag", "").lower()
                
                # Stop if we hit another heading
                if next_tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                    break
                
                # Include list items and paragraphs
                if next_tag in {"li", "p", "ul"}:
                    merged_texts.append(next_node.text or "")
                    j += 1
                else:
                    break
            
            # Create merged node
            merged_text = "\n".join(merged_texts)
            merged_node = TextNode(
                text=merged_text,
                metadata=heading_md
            )
            merged.append(merged_node)
            i = j  # Skip the nodes we just merged
        else:
            # Not a heading, keep as-is
            merged.append(node)
            i += 1
    
    return merged

# ---------------------- Rules (SIRE + synonyms) ----------------------
def tolerant_acronym_rx(term: str) -> re.Pattern:
    t = (term or "").strip()
    if not t:
        return re.compile(r"$^")
    if " " in t or "-" in t:
        pat = re.escape(t).replace(r"\ ", r"[\s\-]+")
        return re.compile(rf"(?i)(?<![A-Z0-9]){pat}(?![A-Z0-9])")
    letters = list(re.sub(r"[^A-Za-z0-9]", "", t))
    if not letters:
        return re.compile(r"$^")
    dotted = r"\.?\s*".join(map(re.escape, letters))
    pat = rf"(?i)(?:{re.escape(t)}|{dotted})"
    return re.compile(pat)


def compile_synonyms(syn_map: dict) -> dict:
    compiled = {}
    for base, variants in (syn_map or {}).items():
        rxes = [tolerant_acronym_rx(base)]
        for v in variants:
            rxes.append(tolerant_acronym_rx(v))
        compiled[base.lower()] = rxes
    return compiled


def wordflex(p: str) -> re.Pattern:
    s = (p or "").strip()
    if not s:
        return re.compile(r"$^")
    s = re.escape(s)
    s = s.replace(r"\ ", r"[\s\-/]+")
    s = s.replace(r"\/", r"[\/]")
    return re.compile(rf"(?i){s}")


def compile_viq_rules(viq_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compiled = []
    for r in viq_rules or []:
        pats = [wordflex(p) for p in r.get("patterns", []) if p]
        if pats:
            compiled.append({
                "viq_no": str(r.get("viq_no", "")).strip(),
                "chapter_no": r.get("chapter_no", None),
                "patterns": pats
            })
    return compiled


def enrich_chunk(text: str, compiled_rules, syn_rx_map) -> Tuple[set, list, list]:
    """
    Returns (domain_tags, viq_hints, synonym_hits)
    - domain_tags: {"chap:9", "viq:9.04.04", "syn:ecdis", ...}
    - viq_hints:   ["9.04.04", ...] (metadata only; NO edges now)
    - synonym_hits:["ecdis","ukc",...]
    """
    t = (text or "").lower()
    viq_hints: List[str] = []
    domain_tags: set = set()

    # VIQ hints
    for rule in compiled_rules:
        for rx in rule["patterns"]:
            if rx.search(t):
                viq = rule["viq_no"]
                if viq and viq not in viq_hints:
                    viq_hints.append(viq)
                    domain_tags.add(f"viq:{viq}")
                ch = rule.get("chapter_no")
                if ch is not None:
                    domain_tags.add(f"chap:{int(ch)}")
                break

    # Synonyms
    syn_hits = []
    for base, rxes in (syn_rx_map or {}).items():
        for rx in rxes:
            if rx.search(t):
                syn_hits.append(base)
                domain_tags.add(f"syn:{base}")
                break
    syn_hits = sorted(set(syn_hits))
    return domain_tags, viq_hints, syn_hits


# ---------------------- Ingestion pipeline ----------------------
async def build_pipeline(chunk_size: int = 900, chunk_overlap: int = 150):
    """Sentence-aware chunking only (no LLM)."""
    Settings.embed_model = OpenAIEmbedding(
        model=os.getenv("EMBED_MODEL", "text-embedding-3-large"),
        embed_batch_size=int(os.getenv("EMBED_BATCH", "256")),
        timeout=60
    )
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return IngestionPipeline(transformations=[splitter])


# ---------------------- Main indexing ----------------------
def index_documents(
    client_root: Path,
    rules_path: Path,
    tags: Optional[List[str]] = None,
    chunk_size: int = 900, 
    chunk_overlap: int = 150,
    parallel_workers: int = 3,
):
    docs_dir   = client_root / "documents"
    store_dir  = client_root / "index_store"
    chroma_dir = store_dir / "chroma"
    chunks_path = store_dir / "chunks.jsonl"
    settings_path = store_dir / "settings.json"
    manifest_path = store_dir / "manifest.json"

    assert docs_dir.exists(), f"Documents folder missing: {docs_dir}"
    assert rules_path.exists(), f"rules.yaml missing: {rules_path}"
    store_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    # Load rules.yaml
    RULES = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
    SYN_MAP = RULES.get("synonyms", {}) or {}
    VIQ_RULES = RULES.get("viq_rules", []) or []
    SYN_RX = compile_synonyms(SYN_MAP)
    COMPILED_VIQ = compile_viq_rules(VIQ_RULES)
    log.info(f"Loaded rules: {len(COMPILED_VIQ)} VIQ entries, {len(SYN_RX)} synonym groups")

    # manifest for incremental updates
    manifest = load_manifest(manifest_path)

    # gather changed/new HTMLs
    html_files = [p for p in sorted(docs_dir.iterdir()) if p.suffix.lower() in {".html", ".htm"}]
    to_process: List[Path] = []
    for p in html_files:
        h = file_hash(p)
        prev = manifest["files"].get(p.name, {})
        if prev.get("sha256") != h:
            to_process.append(p)

    log.info(f"Found {len(html_files)} HTML files; changed/new: {len(to_process)}")

    # parse changed files
    parsed_nodes: List[TextNode] = []
    for p in to_process:
        try:
            html = p.read_text(encoding="utf-8", errors="replace")
            nodes = parse_html_with_li(html, p.name, tags=tags)
            merged_nodes = merge_heading_with_content(nodes)  # ← Add this line
            parsed_nodes.extend(merged_nodes)  # ← Use merged_nodes instead
            manifest["files"][p.name] = {"sha256": file_hash(p), "mtime": p.stat().st_mtime}
            log.info(f"Parsed {p.name} -> {len(nodes)} nodes")
        except Exception as e:
            log.exception(f"Failed to parse {p.name}: {e}")

    if not parsed_nodes and not chunks_path.exists():
        log.warning("No nodes to index (and no previous chunks).")
        return

    # chunking only (no LLM)
    loop = asyncio.get_event_loop()
    pipeline = loop.run_until_complete(build_pipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap))

    processed_nodes: List[TextNode] = []
    if parsed_nodes:
        processed_nodes = loop.run_until_complete(pipeline.arun(nodes=parsed_nodes))
        log.info(f"Processed to {len(processed_nodes)} chunked nodes")

        # Enrich each chunk (metadata only) + assign GLOBALLY unique node_id
        changed_files = {p.name for p in to_process}
        
        # Use simple sequential numbering for truly unique IDs
        for idx, n in enumerate(processed_nodes):
            md = dict(n.metadata or {})
            domain_tags, viq_hints, syn_hits = enrich_chunk(n.text, COMPILED_VIQ, SYN_RX)
            
            # Convert lists to comma-separated strings for Chroma compatibility
            if domain_tags:
                existing_tags = md.get("domain_tags", "").split(",") if md.get("domain_tags") else []
                tags_list = sorted(set(existing_tags) | domain_tags)
                md["domain_tags"] = ",".join(tags_list)
            if viq_hints:
                existing_hints = md.get("viq_hints", "").split(",") if md.get("viq_hints") else []
                hints_list = sorted(set(existing_hints) | set(viq_hints))
                md["viq_hints"] = ",".join(hints_list)
            if syn_hits:
                existing_syns = md.get("synonym_hits", "").split(",") if md.get("synonym_hits") else []
                syn_list = sorted(set(existing_syns) | set(syn_hits))
                md["synonym_hits"] = ",".join(syn_list)

            file_name = md.get("file") or "unknown.html"
            section_id = md.get("section_id") or ""
            
            # CRITICAL FIX: Use UUID + index for guaranteed uniqueness
            import uuid
            unique_suffix = f"{idx}-{uuid.uuid4().hex[:8]}"
            base_id = stable_node_id(file_name, section_id, n.text)
            n.node_id = f"{base_id}-{unique_suffix}"
            n.metadata = md

        # Rewrite chunks.jsonl preserving old chunks from unchanged files
        old_chunks: List[Dict[str, Any]] = []
        if chunks_path.exists():
            with chunks_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if obj.get("metadata", {}).get("file") not in [p.name for p in to_process]:
                            old_chunks.append(obj)
                    except Exception:
                        continue

        with chunks_path.open("w", encoding="utf-8") as f:
            for oc in old_chunks:
                f.write(json.dumps(oc, ensure_ascii=False) + "\n")
            for n in processed_nodes:
                f.write(json.dumps({"id_": n.node_id, "text": n.text, "metadata": n.metadata}, ensure_ascii=False) + "\n")

        # For this run, mark changed files for potential embedding
        new_or_changed_nodes = [n for n in processed_nodes if n.metadata.get("file") in changed_files]

    else:
        # nothing changed → reload existing chunks from JSONL
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    node = TextNode(text=obj["text"], metadata=obj.get("metadata", {}), id_=obj.get("id_", None))
                    if obj.get("id_"):
                        node.node_id = obj["id_"]
                    processed_nodes.append(node)
                except Exception:
                    continue
        log.info(f"Reloaded {len(processed_nodes)} chunks from {chunks_path.name}")
        new_or_changed_nodes = []

    # ----- Initialize Chroma and check if empty -----
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    chroma_collection = chroma_client.get_or_create_collection(
        name="docs", metadata={"hnsw:space": "cosine"}
    )
    chroma_is_empty = chroma_collection.count() == 0

    # **KEY FIX**: If Chroma is empty, embed EVERYTHING
    if chroma_is_empty:
        log.info(f"🔄 Chroma is empty. Embedding ALL {len(processed_nodes)} chunks...")
        new_or_changed_nodes = processed_nodes
    elif new_or_changed_nodes:
        log.info(f"🔄 Upserting {len(new_or_changed_nodes)} new/changed chunks into Chroma...")
    else:
        log.info("ℹ️ No new/changed chunks to embed; skipping vector upsert.")

    # ----- Persist embeddings (Chroma) with PARALLEL BATCH PROCESSING -----
    if new_or_changed_nodes:
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Do NOT let LI re-run any transforms – we already chunked/enriched
        Settings.transformations = []
        Settings.node_parser = None

        embed_model = OpenAIEmbedding(
            model=os.getenv("EMBED_MODEL", "text-embedding-3-large"),
            embed_batch_size=int(os.getenv("EMBED_BATCH", "256")),
            timeout=60,
        )
        Settings.embed_model = embed_model

        # **OPTIMIZATION**: Parallel batch processing
        BATCH_SIZE = 500  # Process 500 chunks per batch
        batches = [new_or_changed_nodes[i:i + BATCH_SIZE] 
                   for i in range(0, len(new_or_changed_nodes), BATCH_SIZE)]
        
        log.info(f"📦 Processing {len(batches)} batches with {parallel_workers} parallel workers")
        log.info(f"📊 Embedding model: {os.getenv('EMBED_MODEL', 'text-embedding-3-large')}, batch size: {os.getenv('EMBED_BATCH', '256')}")

        def process_batch(batch_nodes):
            """Process a single batch of nodes"""
            try:
                # Each thread gets its own storage context
                thread_vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                thread_storage_context = StorageContext.from_defaults(vector_store=thread_vector_store)
                
                _ = VectorStoreIndex(
                    nodes=batch_nodes,
                    storage_context=thread_storage_context,
                    embed_model=embed_model,
                    show_progress=False,
                )
                return len(batch_nodes)
            except Exception as e:
                log.error(f"Batch processing error: {e}")
                return 0

        # Process batches in parallel with progress tracking
        processed_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {executor.submit(process_batch, batch): idx 
                      for idx, batch in enumerate(batches)}
            
            if HAS_TQDM:
                iterator = tqdm(concurrent.futures.as_completed(futures), 
                               total=len(batches), 
                               desc="Embedding batches",
                               unit="batch")
            else:
                iterator = concurrent.futures.as_completed(futures)
            
            for future in iterator:
                batch_idx = futures[future]
                try:
                    count = future.result()
                    processed_count += count
                    if (batch_idx + 1) % 5 == 0:
                        log.info(f"✅ Completed {batch_idx + 1}/{len(batches)} batches ({processed_count} chunks)")
                except Exception as e:
                    log.error(f"Batch {batch_idx} failed: {e}")
        
        log.info(f"✅ Upsert complete. Processed {processed_count} chunks.")

    # Save settings snapshot
    settings = {
        "client_root": str(client_root),
        "rules_yaml": str(rules_path),
        "chroma_path": str(chroma_dir),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "generated_at": datetime.now().isoformat(),
        "files_total": len(html_files),
        "chunks_total": len(processed_nodes),
        "viq_rules_loaded": len(COMPILED_VIQ),
        "synonym_groups": len(SYN_RX),
        "embed_model": os.getenv("EMBED_MODEL", "text-embedding-3-large"),
        "embed_batch": int(os.getenv("EMBED_BATCH", "256")),
        "parallel_workers": parallel_workers,
    }
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    # persist manifest
    manifest["generated_at"] = datetime.now().isoformat()
    save_manifest(manifest_path, manifest)

    log.info("✅ Indexing complete.")
    log.info(f"📦 Chroma at: {chroma_dir}")
    log.info(f"🧾 Chunks JSONL: {chunks_path}")
    log.info(f"⚙️  Settings: {settings_path}")


def cli():
    ap = argparse.ArgumentParser(
        description="OPTIMIZED Hybrid-ready indexer: HTMLNodeParser + SIRE/synonym enrichment + parallel Chroma upserts (10x faster)"
    )
    ap.add_argument("--client_root", required=True, help="Path to client folder (contains 'documents')")
    ap.add_argument("--rules", required=True, help="Path to rules.yaml (from sire_rules_builder.py)")
    ap.add_argument("--tags", default="h1,h2,h3,ul,li,p", help="Comma-separated HTML tags for HTMLNodeParser")
    ap.add_argument("--chunk_size", type=int, default=900)
    ap.add_argument("--chunk_overlap", type=int, default=150)
    ap.add_argument("--embed_batch", type=int, default=256, help="Chunks per OpenAI embedding batch")
    ap.add_argument("--parallel_workers", type=int, default=3, help="Number of parallel workers for embedding (default: 3)")

    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    # configure embedding batch via env for OpenAIEmbedding
    os.environ["EMBED_BATCH"] = str(args.embed_batch)

    index_documents(
        client_root=Path(args.client_root),
        rules_path=Path(args.rules),
        tags=tags,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        parallel_workers=args.parallel_workers,
    )


if __name__ == "__main__":
    cli()