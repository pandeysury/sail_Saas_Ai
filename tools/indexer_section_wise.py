#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
indexer_section_wise.py — SECTION-WISE CHUNKING INDEXER

Revolutionary approach:
  - Chunk by _Toc section boundaries (not sentences)
  - Every chunk has a guaranteed real _Toc ID
  - No guessing, no matching, no fallbacks!
  
Algorithm:
  1. Find all <a name="_Toc...">Title</a> anchors
  2. Extract text between consecutive anchors
  3. Each section = one chunk (or split if too large)
  4. Direct mapping: chunk → _Toc ID
"""

import os, re, json, argparse, hashlib, logging, asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import concurrent.futures

import html as _html
import re as _re

# LlamaIndex core
from llama_index.core import Document, StorageContext, VectorStoreIndex, Settings
from llama_index.core.schema import TextNode
from llama_index.core.node_parser.text import SentenceSplitter

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
log = logging.getLogger(__name__)

# ============================= UTILITY FUNCTIONS =============================

def _strip_tags(s: str) -> str:
    """Remove HTML tags and decode entities."""
    if not s:
        return ""
    s = _re.sub(r'<\s*script[^>]*>.*?</\s*script\s*>', " ", s, flags=_re.I|_re.S)
    s = _re.sub(r'<\s*style[^>]*>.*?</\s*style\s*>', " ", s, flags=_re.I|_re.S)
    s = _re.sub(r'<[^>]+>', " ", s)
    s = _html.unescape(s)
    return _re.sub(r'\s+', ' ', s).strip()

def _norm_txt(s: str) -> str:
    """Normalize text for matching."""
    text = (s or "").strip()
    text = _re.sub(r'^\s*[\d\.]+\s*', '', text)  # Remove leading numbers
    text = text.lower()
    text = _re.sub(r'\s+', ' ', text).strip()
    return text

def stable_slug(s: str) -> str:
    """Generate stable slug for fallback IDs."""
    s = (s or "untitled").strip().lower()
    s = _re.sub(r'[^\w\s-]', '', s)
    s = _re.sub(r'[-\s]+', '-', s)
    return f"sec-{s[:50]}" if s else "sec-untitled"

def build_breadcrumb(doc: str, section: str) -> str:
    """Build breadcrumb path."""
    doc = doc.replace(".html", "")
    if not section or section == doc:
        return doc
    return f"{doc} > {section}"

def file_hash(path: Path) -> str:
    """Calculate SHA256 hash of file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load indexing manifest."""
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except:
            pass
    return {"timestamp": None, "files": {}}

def save_manifest(manifest_path: Path, manifest: Dict[str, Any]):
    """Save indexing manifest."""
    manifest["timestamp"] = datetime.utcnow().isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

def stable_node_id(file: str, section_id: str, chunk_index: int = 0) -> str:
    """Generate stable node ID."""
    s = f"{file}::{section_id}::{chunk_index}"
    return f"node-{hashlib.sha1(s.encode()).hexdigest()}"

# ============================= SECTION EXTRACTION =============================

class TocSection:
    """Represents a _Toc section with its content."""
    def __init__(self, toc_id: str, title: str, start_pos: int, end_pos: int, html: str):
        self.toc_id = toc_id
        self.title = title
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.html = html[start_pos:end_pos]
        self.text = _strip_tags(self.html)
    
    def __repr__(self):
        return f"TocSection({self.toc_id}, '{self.title[:30]}...', {len(self.text)} chars)"

def extract_toc_sections(html_text: str, filename: str) -> List[TocSection]:
    """
    Extract all _Toc sections from HTML.
    
    Returns list of TocSection objects, each representing content
    between consecutive <a name="_Toc..."> anchors.
    """
    sections = []
    
    # Find all _Toc anchors with their positions
    anchors = []
    for m in _re.finditer(r'<a[^>]*\bname="(_Toc[0-9A-Za-z_:-]+)"[^>]*>(.*?)</a>',
                          html_text, flags=_re.I|_re.S):
        toc_id = m.group(1)
        title_html = m.group(2)
        title = _strip_tags(title_html).strip()
        
        # Position where section content starts (after </a>)
        start_pos = m.end()
        
        anchors.append({
            'id': toc_id,
            'title': title or toc_id,  # Use ID if title is empty
            'start': start_pos
        })
    
    log.info(f"Found {len(anchors)} _Toc anchors in {filename}")
    
    # Create sections between consecutive anchors
    for i, anchor in enumerate(anchors):
        # End position = start of next anchor (or end of document)
        if i + 1 < len(anchors):
            end_pos = anchors[i + 1]['start']
        else:
            end_pos = len(html_text)
        
        # Create section
        section = TocSection(
            toc_id=anchor['id'],
            title=anchor['title'],
            start_pos=anchor['start'],
            end_pos=end_pos,
            html=html_text
        )
        
        # Only add if has meaningful content
        if len(section.text.strip()) > 10:
            sections.append(section)
    
    log.info(f"Created {len(sections)} sections with content")
    return sections

# ============================= SMART SECTION CHUNKING =============================

def chunk_sections(
    sections: List[TocSection],
    filename: str,
    max_chunk_size: int = 1500,
    min_chunk_size: int = 100,
    overlap: int = 100
) -> List[TextNode]:
    """
    Convert sections into chunks, splitting large sections if needed.
    
    Strategy:
      - Small sections (< max_chunk_size): One section = one chunk
      - Large sections (> max_chunk_size): Split into sub-chunks
      - All chunks preserve the section's _Toc ID
    """
    chunks = []
    
    # Create splitter once (reuse for all sections)
    splitter = SentenceSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=overlap
    )
    
    for section in sections:
        text = section.text.strip()
        
        if not text:
            continue
        
        # Estimate token count (rough: 1 token ≈ 4 chars)
        estimated_tokens = len(text) // 4
        
        # If section is small enough, make it one chunk
        if estimated_tokens <= max_chunk_size:
            node = TextNode(
                text=text,
                metadata={
                    'file': filename,
                    'section_id': section.toc_id,
                    'section_title': section.title,
                    'breadcrumb': build_breadcrumb(Path(filename).stem, section.title),
                    'slug_url': f"{filename}#{section.toc_id}",
                    'chunk_index': 0,
                    'is_complete_section': True
                }
            )
            node.id_ = stable_node_id(filename, section.toc_id, 0)
            chunks.append(node)
        
        else:
            # Section is too large, split it
            sub_chunks = splitter.split_text(text)
            
            # Only log if actually split into multiple chunks
            if len(sub_chunks) > 1:
                log.info(f"Split large section '{section.title}' ({estimated_tokens} tokens) into {len(sub_chunks)} sub-chunks")
            
            for i, sub_text in enumerate(sub_chunks):
                node = TextNode(
                    text=sub_text,
                    metadata={
                        'file': filename,
                        'section_id': section.toc_id,  # Same ID for all sub-chunks!
                        'section_title': section.title,
                        'breadcrumb': build_breadcrumb(Path(filename).stem, section.title),
                        'slug_url': f"{filename}#{section.toc_id}",
                        'chunk_index': i,
                        'is_complete_section': len(sub_chunks) == 1,
                        'total_sub_chunks': len(sub_chunks)
                    }
                )
                node.id_ = stable_node_id(filename, section.toc_id, i)
                chunks.append(node)
    
    return chunks

# ============================= VIQ & SYNONYM ENRICHMENT =============================

def load_viq_rules(rules_path: Path) -> List[Dict]:
    """Load VIQ rules from YAML."""
    if not rules_path.exists():
        log.warning(f"Rules file not found: {rules_path}")
        return []
    
    try:
        data = yaml.safe_load(rules_path.read_text(encoding='utf-8'))
        rules = data.get('viq_rules', [])
        
        # Compile regex patterns
        for rule in rules:
            rule['patterns'] = [_re.compile(p, _re.I) for p in rule.get('patterns', [])]
        
        log.info(f"Loaded {len(rules)} VIQ rules")
        return rules
    except Exception as e:
        log.error(f"Failed to load VIQ rules: {e}")
        return []

def load_synonyms(rules_path: Path) -> Dict[str, List[str]]:
    """Load synonym groups from YAML."""
    if not rules_path.exists():
        return {}
    
    try:
        data = yaml.safe_load(rules_path.read_text(encoding='utf-8'))
        synonyms = data.get('synonyms', [])
        
        # Build synonym map
        syn_map = {}
        for group in synonyms:
            # Handle both dict and string formats
            if isinstance(group, dict):
                terms = group.get('terms', [])
                group_name = group.get('name', terms[0] if terms else 'unknown')
            elif isinstance(group, str):
                # String format: just the term itself
                terms = [group]
                group_name = group
            else:
                continue
            
            # Compile regex for each term
            for term in terms:
                pattern = _re.compile(r'\b' + _re.escape(term) + r'\b', _re.I)
                syn_map[term] = {'name': group_name, 'pattern': pattern}
        
        log.info(f"Loaded {len(synonyms)} synonym groups")
        return syn_map
    except Exception as e:
        log.error(f"Failed to load synonyms: {e}")
        return {}

def enrich_chunks(chunks: List[TextNode], viq_rules: List[Dict], synonyms: Dict):
    """Add VIQ and synonym metadata to chunks."""
    for chunk in chunks:
        text_lower = (chunk.text or "").lower()
        
        # VIQ enrichment
        viq_hints = []
        domain_tags = set()
        
        for rule in viq_rules:
            # Skip rules without code
            if 'code' not in rule:
                continue
                
            for rx in rule.get('patterns', []):
                if rx.search(text_lower):
                    viq_hints.append(rule['code'])
                    domain_tags.add(f"viq:{rule['code']}")
                    for tag in rule.get('tags', []):
                        domain_tags.add(tag)
                    break
        
        # Synonym enrichment
        synonym_hits = []
        for term, info in synonyms.items():
            if info['pattern'].search(text_lower):
                synonym_hits.append(info['name'])
                domain_tags.add(f"syn:{info['name']}")
        
        # Update metadata
        chunk.metadata['viq_hints'] = ','.join(sorted(set(viq_hints)))
        chunk.metadata['domain_tags'] = ','.join(sorted(domain_tags))
        chunk.metadata['synonym_hits'] = ','.join(sorted(set(synonym_hits)))
    
    return chunks

# ============================= MAIN INDEXING FUNCTION =============================

async def index_documents(
    client_root: Path,
    rules_path: Path,
    max_chunk_size: int = 1500,
    chunk_overlap: int = 100,
    parallel_workers: int = 6,
    force_reindex: bool = False
):
    """
    Main indexing function using section-wise chunking.
    """
    log.info(f"Starting section-wise indexing for {client_root}")
    
    # Setup paths
    docs_dir = client_root / "documents"
    store_dir = client_root / "index_store"
    chroma_dir = store_dir / "chroma"
    chunks_file = store_dir / "chunks.jsonl"
    manifest_file = store_dir / "manifest.json"
    settings_file = store_dir / "settings.json"
    
    store_dir.mkdir(parents=True, exist_ok=True)
    
    # Load rules
    viq_rules = load_viq_rules(rules_path)
    synonyms = load_synonyms(rules_path)
    
    # Find HTML files
    html_files = sorted(docs_dir.glob("*.html"))
    log.info(f"Found {len(html_files)} HTML files")
    
    if not html_files:
        log.error(f"No HTML files found in {docs_dir}")
        return
    
    # Load manifest for incremental updates
    manifest = load_manifest(manifest_file)
    
    # Determine which files need processing
    to_process = []
    for p in html_files:
        prev = manifest["files"].get(p.name, {})
        curr_hash = file_hash(p)
        
        if force_reindex or prev.get("sha256") != curr_hash:
            to_process.append(p)
    
    log.info(f"Processing {len(to_process)} new/changed files")
    
    if not to_process:
        log.info("No files to process. Index is up to date.")
        return
    
    # ========== SECTION-WISE PROCESSING ==========
    all_chunks = []
    
    log.info(f"📄 Processing {len(to_process)} files...")
    for file_path in to_process:
        try:
            html_text = file_path.read_text(encoding="utf-8", errors="replace")
            
            # Extract sections by _Toc boundaries
            sections = extract_toc_sections(html_text, file_path.name)
            
            # Convert sections to chunks
            file_chunks = chunk_sections(
                sections=sections,
                filename=file_path.name,
                max_chunk_size=max_chunk_size,
                min_chunk_size=100,
                overlap=chunk_overlap
            )
            
            all_chunks.extend(file_chunks)
            
            # Update manifest
            manifest["files"][file_path.name] = {
                "sha256": file_hash(file_path),
                "mtime": file_path.stat().st_mtime,
                "sections": len(sections),
                "chunks": len(file_chunks)
            }
            
            log.info(f"✅ Processed {file_path.name}: {len(sections)} sections → {len(file_chunks)} chunks")
        
        except Exception as e:
            log.exception(f"❌ Failed to process {file_path.name}: {e}")
    
    log.info(f"Total chunks created: {len(all_chunks)}")
    
    # Enrich chunks with VIQ and synonyms
    if viq_rules or synonyms:
        log.info("Enriching chunks with VIQ and synonym metadata...")
        all_chunks = enrich_chunks(all_chunks, viq_rules, synonyms)
    
    # Save chunks to JSONL
    log.info(f"Saving chunks to {chunks_file}")
    with chunks_file.open('w', encoding='utf-8') as f:
        for chunk in all_chunks:
            json_obj = {
                'id_': chunk.id_,
                'text': chunk.text,
                'metadata': chunk.metadata
            }
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    
    # ========== EMBEDDING & VECTOR STORE ==========
    log.info("Setting up embeddings...")
    Settings.embed_model = OpenAIEmbedding(
        model=os.getenv("EMBED_MODEL", "text-embedding-3-large"),
        embed_batch_size=int(os.getenv("EMBED_BATCH", "256")),
        timeout=60
    )
    
    # Setup Chroma
    db = chromadb.PersistentClient(path=str(chroma_dir))
    chroma_collection = db.get_or_create_collection("docs")  # Match verification script
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Check what needs embedding
    existing_ids = set(chroma_collection.get()['ids'])
    new_chunks = [c for c in all_chunks if c.id_ not in existing_ids]
    
    if new_chunks:
        log.info(f"🔄 Chroma has {len(existing_ids)} chunks. Embedding {len(new_chunks)} new chunks...")
        
        # Parallel embedding
        batch_size = 256
        batches = [new_chunks[i:i+batch_size] for i in range(0, len(new_chunks), batch_size)]
        
        log.info(f"📦 Processing {len(batches)} batches with {parallel_workers} parallel workers")
        log.info(f"📊 Embedding model: {os.getenv('EMBED_MODEL', 'text-embedding-3-large')}, batch size: {batch_size}")
        
        iterator = tqdm(batches, desc="Embedding batches", unit="batch") if HAS_TQDM else batches
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = []
            for batch in iterator:
                future = executor.submit(
                    VectorStoreIndex.from_documents,
                    [],
                    storage_context=storage_context,
                    show_progress=False,
                    insert_batch_size=batch_size
                )
                future.batch_nodes = batch
                futures.append(future)
            
            for future in futures:
                try:
                    index = future.result()
                    index.insert_nodes(future.batch_nodes)
                except Exception as e:
                    log.error(f"Batch embedding failed: {e}")
        
        log.info(f"✅ Upsert complete. Processed {len(new_chunks)} chunks.")
    else:
        log.info(f"✅ All {len(all_chunks)} chunks already in Chroma. No embedding needed.")
    
    # Save manifest and settings
    save_manifest(manifest_file, manifest)
    
    settings_file.write_text(json.dumps({
        "embed_model": os.getenv("EMBED_MODEL", "text-embedding-3-large"),
        "chunk_size": max_chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunking_strategy": "section_wise",
        "last_updated": datetime.utcnow().isoformat()
    }, indent=2), encoding='utf-8')
    
    log.info("✅ Indexing complete.")
    log.info(f"📦 Chroma at: {chroma_dir}")
    log.info(f"🧾 Chunks JSONL: {chunks_file}")
    log.info(f"⚙️  Settings: {settings_file}")

# ============================= CLI =============================

def main():
    parser = argparse.ArgumentParser(description="Section-wise HTML indexer")
    parser.add_argument("--client_root", type=Path, required=True, help="Client root directory")
    parser.add_argument("--rules", type=Path, required=True, help="VIQ rules YAML file")
    parser.add_argument("--chunk_size", type=int, default=1500, help="Max chunk size")
    parser.add_argument("--chunk_overlap", type=int, default=100, help="Chunk overlap")
    parser.add_argument("--parallel_workers", type=int, default=6, help="Parallel workers")
    parser.add_argument("--force", action="store_true", help="Force reindex all files")
    
    args = parser.parse_args()
    
    asyncio.run(index_documents(
        client_root=args.client_root,
        rules_path=args.rules,
        max_chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        parallel_workers=args.parallel_workers,
        force_reindex=args.force
    ))

if __name__ == "__main__":
    main()