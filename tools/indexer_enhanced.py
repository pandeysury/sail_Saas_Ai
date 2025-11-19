#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
indexer_enhanced.py — ENHANCED INDEXER with Aspose HTML bookmark handling

ENHANCEMENTS from fixed_indexer.py:
  - Extracts bookmarks using BOTH 'name' and 'id' attributes (Aspose uses 'name')
  - Better TOC anchor detection for Aspose documents
  - Enhanced heading-to-bookmark mapping
  - Improved section content extraction

Key Aspose HTML patterns handled:
  1. <a name="_Toc194412259">Text</a>  (bookmarks use 'name', not 'id')
  2. <a href="#_Toc194412259">4.3.5Route Validation</a>  (missing spaces in TOC)
  3. Headings without id attributes - IDs are in separate <a> tags
"""

import os, re, json, argparse, hashlib, logging, asyncio, uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from itertools import islice
import concurrent.futures

import html as _html
import re as _re

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
def stable_node_id(file_name: str, section_id: str, text: str, chunk_index: int = 0) -> str:
    """
    Deterministic ID for a chunk so we can upsert instead of duplicating.
    
    CRITICAL: Includes chunk_index to ensure uniqueness when a section
    is split into multiple overlapping chunks.
    """
    h = hashlib.sha1()
    h.update(file_name.encode("utf-8", errors="ignore"))
    h.update(b"|")
    h.update((section_id or "").encode("utf-8", errors="ignore"))
    h.update(b"|")
    h.update(str(chunk_index).encode("utf-8"))  # Include chunk position
    h.update(b"|")
    h.update(text[:200].encode("utf-8", errors="ignore"))  # Use first 200 chars for additional uniqueness
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

# --- TOC-first anchor picking (regex-only; no external deps) ---



def _norm_txt(s: str) -> str:
    """
    Normalize text for TOC/heading matching.
    Handles Aspose's missing spaces after section numbers.
    
    Examples:
        "4.3.5Route Validation" → "route validation"  (Aspose format)
        "4.3.5 Route Validation" → "route validation" (normal format)
        "1.KEY RESPONSIBILITIES" → "key responsibilities"
    """
    text = (s or "").strip()
    
    # Remove leading section numbers (space OPTIONAL after numbers)
    # Matches: "4.3.5", "4.3.5 ", "1.", "1. ", "10.2.3", etc.
    text = _re.sub(r'^\s*[\d\.]+\s*', '', text)
    
    # Lowercase
    text = text.lower()
    
    # Normalize whitespace
    text = _re.sub(r'\s+', ' ', text).strip()
    
    return text

def _strip_tags(s: str) -> str:
    """
    Remove HTML tags and decode entities.
    Used for extracting clean text from TOC links.
    """
    if not s:
        return ""
    
    # Remove script/style blocks
    s = _re.sub(r'<\s*script[^>]*>.*?</\s*script\s*>', " ", s, flags=_re.I|_re.S)
    s = _re.sub(r'<\s*style[^>]*>.*?</\s*style\s*>', " ", s, flags=_re.I|_re.S)
    
    # Remove all HTML tags
    s = _re.sub(r'<[^>]+>', " ", s)
    
    # Decode HTML entities (&nbsp;, &amp;, etc.)
    s = _html.unescape(s)
    
    # Normalize whitespace
    return _re.sub(r'\s+', ' ', s).strip()

def build_anchor_index_from_html_regex(html_text: str):
    """
    ENHANCED: Extract TOC and heading IDs from Aspose HTML.
    
    Returns (toc_map, heading_map, bookmark_map):
      toc_map      : { normalized_text -> '_Toc…id' }  from <a href="#_Toc...">Text</a>
      heading_map  : { normalized_text -> id }         from <h1-h6 id="...">
      bookmark_map : { normalized_text -> id }         from <a name="_Toc...">Text</a> (Aspose style)
    
    KEY ENHANCEMENT: Handles Aspose bookmarks with 'name' attribute!
    
    Handles Aspose-specific patterns:
      - TOC links with missing spaces ("4.3.5Route" instead of "4.3.5 Route")
      - Bookmarks using <a name="_Toc..."> instead of <a id="_Toc...">
      - IDs in <a name> tags instead of heading element IDs
      - Multiple legacy _Toc IDs per document
    """
    toc_map = {}
    heading_map = {}
    bookmark_map = {}  # NEW: Track bookmarks with 'name' attribute

    if not html_text:
        return toc_map, heading_map, bookmark_map

    # ====================
    # 1) TOC LINKS: <a href="#_Toc123456789">Text</a>
    # ====================
    toc_count = 0
    for m in _re.finditer(r'<a[^>]+href="#(_Toc[0-9A-Za-z_:-]+)"[^>]*>(.*?)</a>',
                          html_text, flags=_re.I|_re.S):
        _id = m.group(1)
        _txt = _strip_tags(m.group(2))
        if _id and _txt:
            toc_map[_norm_txt(_txt)] = _id
            toc_count += 1

    # ====================
    # 2) BOOKMARK ANCHORS: <a name="_Toc123456789">Text</a> (ASPOSE STYLE!)
    # ====================
    # CRITICAL: Aspose puts text INSIDE <a name> tags!
    bookmark_count = 0
    for m in _re.finditer(r'<a[^>]*\bname="(_Toc[0-9A-Za-z_:-]+)"[^>]*>(.*?)</a>',
                          html_text, flags=_re.I|_re.S):
        _id = m.group(1)
        _txt = _strip_tags(m.group(2))  # Text is INSIDE the tag!
        
        if _id and _txt:
            normalized = _norm_txt(_txt)
            if normalized:  # Only if we got meaningful text after normalization
                bookmark_map[normalized] = _id
                bookmark_count += 1

    # ====================
    # 3) HEADING IDs: <h1-h6 id="...">Text</h1-h6>
    # ====================
    heading_count = 0
    for m in _re.finditer(r'<(h[1-6])[^>]+id="([^"]+)"[^>]*>(.*?)</\1>',
                          html_text, flags=_re.I|_re.S):
        _id = m.group(2)
        _txt = _strip_tags(m.group(3))
        if _id and _txt:
            heading_map[_norm_txt(_txt)] = _id
            heading_count += 1

    log.info(f"Anchor extraction: TOC={toc_count}, Bookmarks={bookmark_count}, Headings={heading_count}")
    
    return toc_map, heading_map, bookmark_map


def pick_anchor_for_chunk(
    node_title: str,
    toc_map: Dict[str, str],
    heading_map: Dict[str, str],
    bookmark_map: Dict[str, str]  # NEW parameter
) -> str:
    """
    ENHANCED: Priority system with FLEXIBLE matching for short titles.
    
    Priority:
      1. Exact match in TOC
      2. Exact match in bookmark
      3. Exact match in heading
      4. Partial match in bookmark (for short titles like "Ch.")
      5. Partial match in TOC
      6. Fallback slug
    """
    if not node_title:
        return stable_slug("untitled")
    
    norm = _norm_txt(node_title)
    
    # Empty after normalization
    if not norm:
        return stable_slug(node_title or "untitled")
    
    # Priority 1: Exact TOC match
    if norm in toc_map:
        return toc_map[norm]
    
    # Priority 2: Exact bookmark match
    if norm in bookmark_map:
        return bookmark_map[norm]
    
    # Priority 3: Exact heading match
    if norm in heading_map:
        return heading_map[norm]
    
    # Priority 4: Partial bookmark match (for short/truncated titles)
    # e.g., "ch" matches "ch 5 navigation"
    if len(norm) >= 2:  # Avoid single-char matches
        for key, value in bookmark_map.items():
            if key.startswith(norm + " ") or (len(key) > len(norm) and key.startswith(norm)):
                return value
    
    # Priority 5: Partial TOC match
    if len(norm) >= 2:
        for key, value in toc_map.items():
            if key.startswith(norm + " ") or (len(key) > len(norm) and key.startswith(norm)):
                return value
    
    # Priority 6: Fallback to deterministic slug
    return stable_slug(node_title)


# ---------------------- MERGE HEADING WITH CONTENT ----------------------
def merge_heading_with_content(nodes: List[TextNode]) -> List[TextNode]:
    """
    Enhanced merge function that properly handles:
      1. Bold <p> tags as headings (e.g., "Recordkeeping")
      2. Consecutive content chunks that belong together
      3. Lists and structured content preservation
    
    Strategy:
      - If node is heading-like (short, bold, or h-tag) → potential heading
      - If next node is content → merge heading + content
      - Otherwise keep as-is
    """
    if not nodes:
        return nodes

    merged = []
    skip_next = False

    for i, node in enumerate(nodes):
        if skip_next:
            skip_next = False
            continue

        # Get metadata
        md = node.metadata or {}
        tag = (md.get("tag") or "").lower()
        text = (node.text or "").strip()
        
        # Check if this looks like a heading
        is_heading = (
            # Explicit heading tags
            tag in ("h1", "h2", "h3", "h4", "h5", "h6") or
            # Bold paragraph with short text (likely a section heading)
            (tag == "p" and len(text) < 100 and (
                "<strong>" in str(node.text).lower() or
                "<b>" in str(node.text).lower() or
                text.isupper()  # ALL CAPS
            ))
        )
        
        # If this is a heading and there's a next node, try to merge
        if is_heading and i + 1 < len(nodes):
            next_node = nodes[i + 1]
            next_text = (next_node.text or "").strip()
            next_tag = (next_node.metadata.get("tag") or "").lower()
            
            # Don't merge if next is also a heading
            next_is_heading = next_tag in ("h1", "h2", "h3", "h4", "h5", "h6")
            
            if not next_is_heading and next_text:
                # Merge heading + content
                merged_text = f"{text}\n\n{next_text}"
                merged_node = TextNode(
                    text=merged_text,
                    metadata={
                        **md,
                        "section_title": text,  # Store original heading
                        "merged": True,
                        "original_tags": f"{tag}+{next_tag}"
                    }
                )
                merged.append(merged_node)
                skip_next = True
                continue
        
        # No merge - keep original
        merged.append(node)

    log.info(f"Merge: {len(nodes)} nodes → {len(merged)} nodes")
    return merged


# ---------------------- MAIN INDEXING FUNCTION ----------------------
def index_documents(
    client_root: Path,
    rules_path: Path,
    tags: List[str],
    chunk_size: int,
    chunk_overlap: int,
    parallel_workers: int = 3,
):
    """
    ENHANCED indexing with Aspose HTML bookmark handling.
    
    New features:
      - Extracts bookmarks using 'name' attribute (Aspose style)
      - Better TOC-to-content mapping
      - Improved anchor selection priority
    """
    # 1) Validate paths
    docs_dir = client_root / "documents"
    if not docs_dir.exists():
        log.error(f"documents folder not found: {docs_dir}")
        return

    index_dir = client_root / "index_store"
    index_dir.mkdir(exist_ok=True, parents=True)

    chroma_dir = index_dir / "chroma"
    chunks_path = index_dir / "chunks.jsonl"
    manifest_path = index_dir / "manifest.json"
    settings_path = index_dir / "settings.json"

    # 2) Load YAML rules (using ORIGINAL indexer's structure)
    if not rules_path.exists():
        log.error(f"Rules file not found: {rules_path}")
        return
    
    with rules_path.open("r", encoding="utf-8") as f:
        rules = yaml.safe_load(f) or {}

    # VIQ Rules - using original structure: {viq_no, chapter_no, patterns: [...]}
    def wordflex(p: str) -> re.Pattern:
        """Helper for flexible word matching"""
        s = (p or "").strip()
        if not s:
            return re.compile(r"$^")
        s = re.escape(s)
        s = s.replace(r"\ ", r"[\s\-/]+")
        s = s.replace(r"\/", r"[\/]")
        return re.compile(rf"(?i){s}")
    
    COMPILED_VIQ = []
    for vr in (rules.get("viq_rules") or []):
        pats = [wordflex(p) for p in vr.get("patterns", []) if p]
        if pats:
            COMPILED_VIQ.append({
                "viq_no": str(vr.get("viq_no", "")).strip(),
                "chapter_no": vr.get("chapter_no", None),
                "patterns": pats
            })
    log.info(f"Loaded {len(COMPILED_VIQ)} VIQ rules")

    # Synonym Map - using original structure: {synonyms: {base: [words]}}
    def compile_synonyms(syn_map: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
        """Compile synonym map to regex patterns"""
        compiled = {}
        for base, words in (syn_map or {}).items():
            rxes = [wordflex(w) for w in words if w]
            if rxes:
                compiled[base.lower()] = rxes
        return compiled
    
    SYN_MAP = rules.get("synonyms", {}) or {}
    SYN_RX = compile_synonyms(SYN_MAP)
    log.info(f"Loaded {len(SYN_RX)} synonym groups")

    # 3) Collect HTML files
    html_files = sorted(docs_dir.glob("*.html"))
    if not html_files:
        log.warning(f"No HTML files found in {docs_dir}")
        return
    log.info(f"Found {len(html_files)} HTML files")

    # 4) Incremental check
    manifest = load_manifest(manifest_path)
    old_files = manifest.get("files", {})

    to_process = []
    changed_files = set()
    for p in html_files:
        h = file_hash(p)
        old_h = old_files.get(p.name, {}).get("hash")
        if h != old_h:
            to_process.append(p)
            changed_files.add(p.name)
            old_files[p.name] = {
                "hash": h,
                "size": p.stat().st_size,
                "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat()
            }
        else:
            log.debug(f"Skip unchanged: {p.name}")

    if to_process:
        log.info(f"Processing {len(to_process)} new/changed files")
    else:
        log.info("All files unchanged; skipping parse/transform")

    # 5) Parse + chunk + enrich
    processed_nodes = []

    if to_process:
        all_docs = []
        # Store maps separately, not in document metadata
        file_anchor_maps = {}
        
        log.info(f"📄 Processing {len(to_process)} files...")
        for p in to_process:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
                # ENHANCED: Also extract bookmarks
                toc_map, heading_map, bookmark_map = build_anchor_index_from_html_regex(text)
                
                # Log anchor extraction per file
                log.info(f"Anchor extraction: TOC={len(toc_map)}, Bookmarks={len(bookmark_map)}, Headings={len(heading_map)}")
                
                # Store maps separately by filename
                file_anchor_maps[p.name] = {
                    "toc_map": toc_map,
                    "heading_map": heading_map,
                    "bookmark_map": bookmark_map
                }
                
                doc = Document(
                    text=text,
                    metadata={
                        "file": p.name,  # Only store filename, not the large maps!
                        "source": str(p)
                    }
                )
                all_docs.append(doc)
                manifest["files"][p.name] = {"sha256": file_hash(p), "mtime": p.stat().st_mtime}
                
                log.info(f"✅ Parsed {p.name}")
            except Exception as e:
                log.exception(f"❌ Failed to parse {p.name}: {e}")

        parser = HTMLNodeParser(tags=tags)
        all_nodes = parser.get_nodes_from_documents(all_docs)
        log.info(f"Parsed {len(all_nodes)} raw nodes")

        # Enhanced merge
        all_nodes = merge_heading_with_content(all_nodes)
        log.info(f"After merge: {len(all_nodes)} nodes")

        # Split using IngestionPipeline (like original indexer - handles metadata correctly)
        async def build_pipeline(chunk_size: int, chunk_overlap: int):
            """Sentence-aware chunking only (no LLM)."""
            Settings.embed_model = OpenAIEmbedding(
                model=os.getenv("EMBED_MODEL", "text-embedding-3-large"),
                embed_batch_size=int(os.getenv("EMBED_BATCH", "256")),
                timeout=60
            )
            splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            return IngestionPipeline(transformations=[splitter])
        
        # Use async pipeline like original
        loop = asyncio.get_event_loop()
        pipeline = loop.run_until_complete(build_pipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap))
        all_nodes = loop.run_until_complete(pipeline.arun(nodes=all_nodes))
        log.info(f"After split: {len(all_nodes)} chunks")

        # Enrich each chunk (AFTER chunking, like original indexer)
        for idx, n in enumerate(all_nodes):
            md = n.metadata or {}
            file_name = md.get("file", "unknown.html")
            
            # Get anchor maps from our separate storage (not from metadata!)
            anchor_maps = file_anchor_maps.get(file_name, {})
            toc_map = anchor_maps.get("toc_map", {})
            heading_map = anchor_maps.get("heading_map", {})
            bookmark_map = anchor_maps.get("bookmark_map", {})
            
            # Guess title
            title = guess_title_from_node(n, md)
            
            # ENHANCED: Pick best anchor using bookmark_map
            section_id = pick_anchor_for_chunk(title, toc_map, heading_map, bookmark_map)
            
            # Build metadata (WITHOUT the large maps!)
            md["section_id"] = section_id
            md["section_title"] = title
            md["breadcrumb"] = build_breadcrumb(Path(file_name).stem, title)
            md["slug_url"] = f"{file_name}#{section_id}"

            # VIQ + synonym enrichment (using original structure)
            text_lower = (n.text or "").lower()
            viq_hints = []
            domain_tags = set()
            
            # VIQ hints - original structure with patterns list
            for rule in COMPILED_VIQ:
                for rx in rule["patterns"]:
                    if rx.search(text_lower):
                        viq = rule["viq_no"]
                        if viq and viq not in viq_hints:
                            viq_hints.append(viq)
                            domain_tags.add(f"viq:{viq}")
                        ch = rule.get("chapter_no")
                        if ch is not None:
                            domain_tags.add(f"chap:{int(ch)}")
                        break
            
            # Synonyms - original structure with synonym map
            syn_hits = []
            for base, rxes in SYN_RX.items():
                for rx in rxes:
                    if rx.search(text_lower):
                        syn_hits.append(base)
                        domain_tags.add(f"syn:{base}")
                        break

            # Add enrichment to metadata (original format)
            if domain_tags:
                md["domain_tags"] = ",".join(sorted(domain_tags))
            if viq_hints:
                md["viq_hints"] = ",".join(viq_hints)
            if syn_hits:
                md["synonym_hits"] = ",".join(sorted(syn_hits))

            file_name = md.get("file") or "unknown.html"
            section_id = md.get("section_id") or ""
            
            # Generate stable, deterministic ID (no UUID!)
            n.node_id = stable_node_id(file_name, section_id, n.text, chunk_index=idx)
            n.metadata = md
        
        # CRITICAL: Assign enriched nodes to processed_nodes
        processed_nodes = all_nodes

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
        "aspose_bookmark_support": True,  # NEW flag
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
        description="ENHANCED Hybrid-ready indexer with Aspose HTML bookmark support"
    )
    ap.add_argument("--client_root", required=True, help="Path to client folder (contains 'documents')")
    ap.add_argument("--rules", required=True, help="Path to rules.yaml (from sire_rules_builder.py)")
    ap.add_argument("--tags", default="h1,h2,h3,ul,li,p", help="Comma-separated HTML tags for HTMLNodeParser")
    ap.add_argument("--chunk_size", type=int, default=500, help="Target tokens per chunk (default: 500)")
    ap.add_argument("--chunk_overlap", type=int, default=50, help="Overlap tokens (default: 50)")
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