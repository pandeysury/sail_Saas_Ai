# app/routers/query.py - COMPLETE: Section-Aware + Document Summarization + SQLite history persistence
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from loguru import logger
import os
import time
import re
from urllib.parse import quote
from collections import defaultdict

# ── NEW: SQLite persistence for conversation history ──────────────────────────
import sqlite3
import threading

from app.models import AskRequest, AskResponse, RefItem
from app.state import get_bundle
from app.rerankers.reranker_llm import OpenAILLMReranker, LLMRerankerConfig
from app.utils.query_logger import get_query_logger
from app.utils.entity_recognition import get_entity_recognizer
from dotenv import load_dotenv
load_dotenv()
router = APIRouter(prefix="/api", tags=["query"])

# In-memory conversation history storage (kept as-is for backward compatibility)
CONVERSATION_HISTORY = defaultdict(list)

# =============================================================================
# NEW: SQLite persistence setup (uses existing chat_history.db at project root)
# =============================================================================
# repo_root = <project>/  (two levels up from this file: app/routers/query.py)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DB_PATH = os.getenv("CHAT_HISTORY_DB", os.path.join(_REPO_ROOT, "chat_history.db"))

_DB_LOCK = threading.Lock()
def _db_connect() -> sqlite3.Connection:
    # Ensure folder exists (no-op for project root files)
    os.makedirs(os.path.dirname(_DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    # Reasonable defaults for small chat persistence
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def _init_db():
    with _DB_LOCK:
        conn = _db_connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_history_client_conv ON chat_history(client_id, conversation_id, created_at);"
            )
            conn.commit()
        finally:
            conn.close()
    logger.info(f"[HISTORY][SQLite] Ready at {_DB_PATH}")

def _db_insert_message(client_id: str, conversation_id: str, role: str, content: str):
    try:
        with _DB_LOCK:
            conn = _db_connect()
            try:
                conn.execute(
                    "INSERT INTO chat_history (client_id, conversation_id, role, content) VALUES (?, ?, ?, ?);",
                    (client_id, conversation_id, role, content),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception as e:
        logger.warning(f"[HISTORY][SQLite] Insert failed: {e}")

def _db_fetch_history(client_id: Optional[str], conversation_id: str) -> List[Dict[str, str]]:
    try:
        with _DB_LOCK:
            conn = _db_connect()
            try:
                if client_id:
                    cur = conn.execute(
                        """
                        SELECT role, content
                        FROM chat_history
                        WHERE client_id = ? AND conversation_id = ?
                        ORDER BY created_at ASC, id ASC;
                        """,
                        (client_id, conversation_id),
                    )
                else:
                    # Fallback mode if client_id not provided (not recommended)
                    cur = conn.execute(
                        """
                        SELECT role, content
                        FROM chat_history
                        WHERE conversation_id = ?
                        ORDER BY created_at ASC, id ASC;
                        """,
                        (conversation_id,),
                    )
                rows = cur.fetchall()
                return [{"role": r[0], "content": r[1]} for r in rows]
            finally:
                conn.close()
    except Exception as e:
        logger.warning(f"[HISTORY][SQLite] Fetch failed: {e}")
        return []

# Initialize DB at import
_init_db()

# ============================================================================
# DOC URL ROUTING
# ============================================================================
def _doc_url(client_id: str, slug_url: Optional[str]) -> str:
    """Convert slug_url to routable URL with proper encoding."""
    if not slug_url:
        return ""
    
    parts = slug_url.split('#', 1)
    filename = parts[0].lstrip('/')
    anchor = parts[1] if len(parts) > 1 else ""
    
    encoded_filename = quote(filename, safe='/')
    
    if anchor:
        # return f"/docs/{client_id}/{encoded_filename}#{anchor}"
        return f"/{client_id}/docs/{encoded_filename}#{anchor}"
    else:
        return f"/{client_id}/docs/{encoded_filename}"

# ============================================================================
# SECTION EXTRACTION
# ============================================================================
def _extract_section_from_chunk(text: str, target_section: str) -> Optional[str]:
    """Extract a specific section from a multi-section chunk."""
    lines = text.split('\n')
    
    section_markers = [
        'Process', 'Procedure', 'Responsibility',
        'Checking & Assurance', 'Checking', 'Assurance',
        'Recordkeeping', 'Records', 'Documentation',
        'Requirements', 'Compliance', 'Instructions',
        'Notes', 'References', 'Background'
    ]
    
    section_start = -1
    section_end = len(lines)
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        if target_section.lower() in line_stripped.lower():
            section_start = i
            logger.info(f"[SECTION] Found '{target_section}' at line {i}")
            continue
        
        if section_start >= 0 and i > section_start:
            if any(marker in line_stripped for marker in section_markers):
                if line_stripped and not line_stripped.startswith(('*', '-', '•', '·')):
                    section_end = i
                    logger.info(f"[SECTION] Section ends at line {i}")
                    break
    
    if section_start >= 0:
        section_lines = lines[section_start:section_end]
        section_text = '\n'.join(section_lines).strip()
        logger.info(f"[SECTION] Extracted {len(section_text)} chars from '{target_section}' section")
        return section_text
    
    return None

# ============================================================================
# SMART CHUNK REORDERING
# ============================================================================
def _reorder_chunks_by_intent(question: str, nodes: list, query_intent: dict) -> list:
    """Reorder chunks to prioritize sections matching query intent."""
    question_lower = question.lower()
    intent_type = query_intent.get('type', 'general')
    
    # Define section priorities based on intent
    section_priorities = {
        'recordkeeping': ['recordkeeping', 'records', 'documentation', 'maintain following'],
        'procedural': ['process', 'procedure', 'steps', 'how to', 'method'],
        'checking': ['checking', 'assurance', 'verification', 'validation'],
        'requirements': ['requirements', 'compliance', 'must', 'shall'],
        'summarization': [],  # No specific reordering for summarization
        'general': []
    }
    
    target_sections = section_priorities.get(intent_type, [])
    
    if not target_sections:
        return nodes
    
    logger.info(f"[REORDER] Looking for sections: {target_sections}")
    
    chunk_scores = []
    
    for node in nodes:
        if isinstance(node, dict):
            text = node.get("text", "")
        else:
            text = getattr(node.node, "text", "") or getattr(node.node, "get_content", lambda: "")()
        
        text_lower = text.lower()
        score = 0
        
        for section in target_sections:
            if section in text_lower:
                lines = text_lower.split('\n')
                for line in lines:
                    if section in line and len(line.strip()) < 50:
                        score += 10
                        logger.info(f"[REORDER] Found section header '{section}' in chunk")
                        break
                else:
                    score += 5
        
        chunk_scores.append((node, score))
    
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    
    if chunk_scores[0][1] > 0:
        logger.info(f"[REORDER] Reordered chunks - top score: {chunk_scores[0][1]}")
    
    return [node for node, score in chunk_scores]

# ============================================================================
# FORMAT CHUNK WITH SECTION MARKERS
# ============================================================================
def _format_chunk_with_sections(node, index: int) -> str:
    """Format chunk with visible section markers for LLM to parse."""
    if isinstance(node, dict):
        text = node.get("text", "")
        md = node.get("metadata", {})
    else:
        text = getattr(node.node, "text", "") or getattr(node.node, "get_content", lambda: "")()
        md = getattr(node.node, "metadata", {}) or {}
    
    title = md.get("section_title", "") or md.get("breadcrumb", "") or md.get("file", "Unknown")
    breadcrumb = md.get("breadcrumb", "N/A")
    
    section_markers = [
        'Process', 'Procedure', 'Responsibility',
        'Checking & Assurance', 'Checking', 'Assurance',
        'Recordkeeping', 'Records', 'Documentation',
        'Requirements', 'Compliance', 'Instructions'
    ]
    
    highlighted = text
    for marker in section_markers:
        highlighted = highlighted.replace(
            f'\n{marker}\n', 
            f'\n\n🔹 SECTION: {marker.upper()} 🔹\n'
        )
        highlighted = highlighted.replace(
            f'\n{marker}:', 
            f'\n\n🔹 SECTION: {marker.upper()} 🔹\n'
        )
        highlighted = highlighted.replace(
            f'\n{marker} ', 
            f'\n\n🔹 SECTION: {marker.upper()} 🔹\n'
        )
    
    return f"""═══════════════════════════════════════════════════════════════
DOCUMENT {index + 1}: {title}
Source: {breadcrumb}
═══════════════════════════════════════════════════════════════

{highlighted}

"""

# ============================================================================
# REFERENCE BUILDING
# ============================================================================
def _build_references(nodes, client_id: Optional[str] = None) -> List[RefItem]:
    """Convert retrieved nodes to RefItem objects with routable URLs."""
    refs: List[RefItem] = []
    for n in nodes:
        try:
            if isinstance(n, dict):
                md = n.get("metadata", {}) or {}
                score = n.get("score")
                text = n.get("text", "")
            else:
                md = getattr(n.node, "metadata", {}) or {}
                score = getattr(n, "score", None)
                text = getattr(n.node, "text", "")
            
            domain_tags_str = md.get("domain_tags", "")
            domain_tags = [t.strip() for t in domain_tags_str.split(",") if t.strip()] if domain_tags_str else []
            
            viq_hints_str = md.get("viq_hints", "")
            viq_hints = [v.strip() for v in viq_hints_str.split(",") if v.strip()] if viq_hints_str else []
            
            url_val = _doc_url(
                client_id or md.get("client_id") or "",
                md.get("slug_url") or md.get("file")
            )
            
            refs.append(RefItem(
                title=md.get("section_title") or md.get("breadcrumb") or md.get("file"),
                breadcrumb=md.get("breadcrumb"),
                url=url_val,
                score=float(score) if score is not None else None,
                viq=viq_hints,
                tags=domain_tags,
            ))
        except Exception as e:
            logger.warning(f"Failed to parse reference node: {e}")
            continue
    
    return refs

def _refs_html(refs: List[RefItem]) -> str:
    """Build the exact HTML block your UI expects, wrapped after [REFS]."""
    if not refs:
        return ""
    links = []
    for idx, r in enumerate(refs):
        display_text = r.breadcrumb or r.title or "Document"
        safe_url = r.url or ""
        safe_title = r.title or display_text
        links.append(
            f'<a href="#" class="ref-link" data-url="{safe_url}" data-title="{safe_title}" data-idx="{idx}">{display_text}</a>'
        )
    links_html = "".join(links)
    return (
        "[REFS]"
        '<div class="refs-section">'
        "<strong>References</strong>"
        f'<div class="refs-list">{links_html}</div>'
        "</div>"
    )

# ============================================================================
# RERANKING
# ============================================================================
# Fixed _apply_reranking function for query.py
# This matches your actual reranker_llm.py implementation
def _apply_reranking(question: str, nodes, use_reranker: bool = True) -> list:
    """Apply LLM-based reranking using your custom OpenAILLMReranker."""
    if not use_reranker or not nodes:
        return nodes
    
    try:
        logger.info(f"[RERANK] Starting rerank for {len(nodes)} nodes")
        
        # Build passages list for your custom reranker
        passages = []
        for i, n in enumerate(nodes):
            # Extract text and metadata from node
            if isinstance(n, dict):
                text = n.get("text", "")
                md = n.get("metadata", {})
                score = n.get("score", None)
            else:
                text = getattr(n.node, "text", "") or getattr(n.node, "get_content", lambda: "")()
                md = getattr(n.node, "metadata", {}) or {}
                score = getattr(n, "score", None)
            
            passages.append({
                "id": i,
                "text": text,
                "metadata": md,
                "score": score,
                "original_node": n,
            })
        
        # ✅ FIXED: Use correct parameter names that match your reranker_llm.py
        config = LLMRerankerConfig(
            model=os.getenv("OPENAI_RERANK_MODEL", "gpt-4o-mini"),  # ✅ Changed to 'model'
            max_passages=int(os.getenv("RERANK_MAX_PASSAGES", "20")),
            parallel=True
        )
        
        # Create reranker (client=None since your reranker creates it internally)
        reranker = OpenAILLMReranker(client=None, config=config)
        
        # Rerank passages
        reranked_passages = reranker.rerank(question, passages)
        
        # Rebuild nodes list from reranked passages
        reranked_nodes = []
        for passage in reranked_passages:
            original_idx = passage.get("id", 0)
            if 0 <= original_idx < len(nodes):
                node = nodes[original_idx]
                # Update score if reranker provided one
                if "score" in passage:
                    if isinstance(node, dict):
                        node["score"] = passage["score"]
                    else:
                        node.score = passage["score"]
                reranked_nodes.append(node)
        
        logger.info(f"[RERANK] ✅ Reranked {len(reranked_nodes)} nodes")
        
        # Log top scores
        if reranked_nodes:
            top_scores = []
            for n in reranked_nodes[:3]:
                if isinstance(n, dict):
                    score = n.get("score", 0)
                else:
                    score = getattr(n, "score", 0)
                top_scores.append(score)
            logger.info(f"[RERANK] Top 3 scores: {top_scores}")
        
        return reranked_nodes
        
    except Exception as e:
        logger.error(f"[RERANK] Reranking failed: {e}")
        logger.warning(f"[RERANK] ⚠️ Falling back to original order")
        return nodes

# ============================================================================
# ANSWER SYNTHESIS - SECTION-AWARE + SUMMARIZATION
# ============================================================================
def _synthesize_answer(question: str, nodes: list, use_llm: bool = True, query_intent: dict = None) -> str:
    """Generate answer from context with section-aware extraction and summarization support."""
    if not nodes:
        return "No relevant information found."
    
    is_compound = query_intent.get("is_compound", False) if query_intent else False
    is_summarization = query_intent.get("is_summarization", False) if query_intent else False
    intent_type = query_intent.get("type", "general") if query_intent else "general"
    
    combined_context = "\n".join([
        _format_chunk_with_sections(n, i)
        for i, n in enumerate(nodes)
    ])
    
    if not use_llm:
        return f"Based on the retrieved documents:\n\n{combined_context[:1500]}..."
    
    try:
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set")
            return f"Based on the retrieved documents:\n\n{combined_context[:1500]}..."
        
        client = OpenAI(api_key=api_key)
        
        # Base system prompt
        system_prompt = """You are a maritime documentation assistant with expertise in SMS procedures.

**CRITICAL: SECTION-LEVEL ANALYSIS**

Each document contains MULTIPLE sections marked with 🔹 SECTION markers:
- Process / Procedure
- Checking & Assurance
- Recordkeeping / Records
- Requirements
- And others

**YOUR TASK:**

1. **SCAN ALL SECTIONS** in each document
   - Don't stop reading after the first section
   - The answer might be at the END of a document
   - Section markers look like: 🔹 SECTION: RECORDKEEPING 🔹

2. **MATCH QUERY INTENT TO SECTION**
   - "What records..." → Look for 🔹 SECTION: RECORDKEEPING 🔹
   - "How to..." → Look for 🔹 SECTION: PROCESS 🔹 or 🔹 SECTION: PROCEDURE 🔹
   - "Checking..." → Look for 🔹 SECTION: CHECKING & ASSURANCE 🔹
   - "Requirements..." → Look for 🔹 SECTION: REQUIREMENTS 🔹

3. **EXTRACT THE RIGHT SECTION**
   - Find the section that matches what was asked
   - Extract the COMPLETE section content
   - Include ALL bullet points and details
   - Don't mix content from different sections

4. **FORMAT YOUR ANSWER**
   - Use markdown headers (###)
   - Use bullet points for lists
   - Be comprehensive
   - Cite: **Source:** [Document Title] - [Section Name]

5. **QUALITY CHECK**
   - Did I find the right section?
   - Am I answering what was asked, not just what the document title suggests?
   - If asked about "records", did I find and use the Recordkeeping section?
"""
        # Add compound question handling
        if is_compound:
            system_prompt += """

⚠️ **COMPOUND QUESTION DETECTED**:
- This question asks MULTIPLE things (e.g., "When & How", "What & Why")
- You MUST answer ALL parts
- Structure with clear sections for each part
"""

        # Add summarization handling
        if is_summarization:
            system_prompt += """

📄 **DOCUMENT SUMMARIZATION REQUESTED**:
- User wants an OVERVIEW or SUMMARY of a document/topic
- You have access to MULTIPLE chunks covering different aspects
- Your task is to synthesize a COMPREHENSIVE summary that:
  
  **Structure:**
  - Start with a brief introduction (1-2 sentences)
  - Organize by main topics/sections found across chunks
  - Use clear headers (###) for each major topic
  - Include key points from each section
  - End with a brief conclusion if relevant
  
  **Coverage:**
  - Scan ALL provided chunks
  - Identify the main topics/themes
  - Don't focus on just one chunk
  - Synthesize information across all chunks
  - Provide a holistic overview
  
  **Format:**
  - Use headers for main topics
  - Use bullet points for key details
  - Keep it concise but comprehensive
  - Aim for 200-400 words for summaries
  
  **Example Structure:**
  ### Overview
  [Brief intro about the document/topic]
  
  ### Main Topic 1
  - Key point 1
  - Key point 2
  
  ### Main Topic 2
  - Key point 1
  - Key point 2
  
  ### Summary
  [Brief conclusion if relevant]
  
  **Source:** [Document Name(s)]
"""
        user_prompt = f"""Question: {question}

Query Intent: {intent_type}

Documents (SCAN ALL SECTIONS - look for 🔹 markers):

{combined_context}

REMEMBER: {"Synthesize across all chunks to provide a comprehensive summary!" if is_summarization else "The answer might be in a section at the END of a document. Scan completely!"}

Answer:"""
        
        # Adjust max_tokens for summarization
        max_tokens = 1500 if is_summarization else 1200
        
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=max_tokens
        )
        
        answer = response.choices[0].message.content.strip()
        return answer
        
    except Exception as e:
        logger.exception(f"LLM synthesis failed: {e}")
        return f"Based on the retrieved documents:\n\n{combined_context[:1500]}..."

# ============================================================================
# QUERY ENHANCEMENT
# ============================================================================
def _enhance_query_for_forms(query: str) -> Optional[str]:
    """Enhance queries mentioning specific form numbers."""
    form_pattern = r'\b(NP\s*\d{3,4}[A-Z]?|CG[-\s]?\d{4}[A-Z]?|[A-Z]{2,3}[-\s]?\d{2,4}[A-Z]?)\b'
    
    forms_found = re.findall(form_pattern, query.upper())
    
    if not forms_found:
        return None
    
    if any('NP' in form and '133' in form for form in forms_found):
        enhanced = query.lower()
        enhanced = enhanced.replace('enter in', 'maintain in')
        enhanced = enhanced.replace('enter into', 'maintain in')
        context_keywords = ['passage planning', 'recordkeeping', 'ECDIS']
        
        for keyword in context_keywords:
            if keyword.lower() not in enhanced:
                enhanced += f" {keyword}"
        
        return enhanced
    
    return None

def _enhance_query_with_entities(query: str, entities: Dict[str, List[str]]) -> str:
    """Enhance query with detected entities."""
    query_lower = query.lower()
    enhancements = []
    
    if entities.get("regulations"):
        for reg in entities["regulations"]:
            if reg.lower() not in query_lower:
                enhancements.append(reg)
    
    if entities.get("equipment"):
        for equip in entities["equipment"][:2]:
            if equip.lower() not in query_lower:
                enhancements.append(equip)
    
    if entities.get("procedures"):
        for proc in entities["procedures"][:2]:
            if proc.lower() not in query_lower:
                enhancements.append(proc)
    
    if enhancements:
        return f"{query} {' '.join(enhancements)}"
    
    return query

# ============================================================================
# INTENT DETECTION - WITH SUMMARIZATION
# ============================================================================
def _detect_compound_question(query: str) -> bool:
    """Detect if query asks multiple things."""
    compound_patterns = [
        r'\band\b.*\b(what|how|when|where|why|who)',
        r'\b(what|how|when|where|why|who)\b.*\band\b',
        r'\bor\b.*\b(what|how|when|where|why|who)',
        r'[,;].*\b(what|how|when|where|why|who)'
    ]
    
    query_lower = query.lower()
    for pattern in compound_patterns:
        if re.search(pattern, query_lower):
            return True
    
    wh_count = len(re.findall(r'\b(what|how|when|where|why|who)\b', query_lower))
    return wh_count >= 2

def _calculate_intent_confidence(intent: Dict[str, Any], query: str) -> float:
    """Calculate confidence score for detected intent."""
    confidence = 0.0
    query_lower = query.lower()
    
    if intent["type"] == "recordkeeping":
        strong_signals = ['what records', 'which records', 'maintain', 'recordkeeping']
        confidence += sum(0.2 for signal in strong_signals if signal in query_lower)
    
    elif intent["type"] == "procedural":
        strong_signals = ['how to', 'procedure for', 'steps to']
        confidence += sum(0.2 for signal in strong_signals if signal in query_lower)
    
    elif intent["type"] == "summarization":
        strong_signals = ['summarize', 'summary', 'overview', 'explain the']
        confidence += sum(0.2 for signal in strong_signals if signal in query_lower)
        if intent.get("is_document_query"):
            confidence += 0.2
    
    if intent["is_compound"]:
        confidence += 0.1
    
    return min(confidence, 1.0)

def _detect_query_intent(query: str) -> Dict[str, Any]:
    """Detect query intent with confidence scoring - includes summarization."""
    query_lower = query.lower()
    
    recordkeeping_signals = [
        'what records', 'which records', 'maintain', 'keep', 
        'recordkeeping', 'log', 'register', 'what should be',
        'what to', 'document in', 'enter in', 'record in'
    ]
    
    procedural_signals = [
        'how to', 'procedure for', 'steps to', 'process for',
        'what is the procedure', 'how do i', 'how should',
        'method', 'way to', 'approach to'
    ]
    
    summarization_signals = [
        'summarize', 'summary of', 'summarise', 'overview of',
        'brief overview', 'explain the', 'what is covered in',
        'what does the', 'tell me about', 'explain', 'describe',
        'give me an overview', 'what are the main', 'key points',
        'main topics', 'contents of', 'what\'s in the', 'give me a summary'
    ]
    
    document_patterns = [
        r'passage planning\s*(document|manual|section|chapter|procedure)?',
        r'(chapter|section|doc|document|manual)\s+\d+',
        r'(the|this)\s+(document|manual|chapter|section|procedure)',
        r'np\s*\d+',
        r'(solas|ism|marpol|stcw)\s*(chapter|part|section)?'
    ]
    
    is_recordkeeping = any(signal in query_lower for signal in recordkeeping_signals)
    is_procedural = any(signal in query_lower for signal in procedural_signals)
    is_summarization = any(signal in query_lower for signal in summarization_signals)
    is_compound = _detect_compound_question(query)
    is_document_query = any(re.search(pattern, query_lower) for pattern in document_patterns)
    
    # Determine intent type with priority
    if is_summarization or (is_document_query and not is_procedural and not is_recordkeeping):
        intent_type = "summarization"
    elif is_recordkeeping:
        intent_type = "recordkeeping"
    elif is_procedural:
        intent_type = "procedural"
    else:
        intent_type = "general"
    
    intent = {
        "type": intent_type,
        "is_recordkeeping": is_recordkeeping,
        "is_procedural": is_procedural,
        "is_summarization": is_summarization,
        "is_document_query": is_document_query,
        "is_compound": is_compound,
        "confidence": "pending"
    }
    
    confidence_score = _calculate_intent_confidence(intent, query)
    intent["confidence"] = "high" if confidence_score >= 0.6 else ("medium" if confidence_score >= 0.3 else "low")
    intent["confidence_score"] = round(confidence_score, 2)
    
    return intent

# ============================================================================
# MAIN ASK ENDPOINT - Frontend Compatible
# ============================================================================
@router.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    """Main RAG endpoint - called by frontend as /api/ask"""
    return ask(req)

# ============================================================================
# MAIN ASK ENDPOINT
# ============================================================================
@router.post("/query", response_model=AskResponse)
def ask(req: AskRequest):
    """Main RAG endpoint with section-awareness and document summarization."""
    start_time = time.time()
    retrieval_time_ms = None
    reranking_time_ms = None
    synthesis_time_ms = None
    
    query_logger = get_query_logger()
    
    try:
        logger.info(f"[ASK] client={req.client_id}, question={req.question[:80]}...]")

        # Query enhancement & intent detection
        enhanced_query = _enhance_query_for_forms(req.question)
        
        if enhanced_query:
            retrieval_query = enhanced_query
            query_was_enhanced = True
            logger.info(f"[ASK] ✨ Query enhanced: '{retrieval_query}'")
        else:
            retrieval_query = req.question
            query_was_enhanced = False
        
        query_intent = _detect_query_intent(req.question)
        
        entity_recognizer = get_entity_recognizer()
        entities = entity_recognizer.extract_entities(req.question)
        query_intent["entities"] = entities
        
        if not query_was_enhanced and entity_recognizer.has_entities(entities):
            if os.getenv("ENABLE_ENTITY_ENHANCEMENT", "true").lower() == "true":
                entity_enhanced_query = _enhance_query_with_entities(retrieval_query, entities)
                if entity_enhanced_query != retrieval_query:
                    retrieval_query = entity_enhanced_query
                    query_was_enhanced = True
                    logger.info(f"[ASK] 🏷️  Query enhanced with entities")
        
        logger.info(f"[ASK] 🎯 Query intent: {query_intent['type']} (confidence: {query_intent['confidence']}, score: {query_intent.get('confidence_score', 0)})")
        
        if entity_recognizer.has_entities(entities):
            entity_summary = entity_recognizer.get_entity_summary(entities)
            logger.info(f"[ASK] 🏷️  Entities: {entity_summary}")
        
        if query_intent.get("is_compound"):
            logger.info(f"[ASK] 🔄 Compound question detected")
        
        if query_intent.get("is_summarization"):
            logger.info(f"[ASK] 📄 Document summarization requested")
        
        # Load tenant bundle
        bundle = get_bundle(req.client_id, req.index_name)
        retriever = bundle.get("retriever")
        
        if retriever is None:
            raise HTTPException(status_code=500, detail="Retriever not initialized")
        
        # Retrieval
        logger.info(f"[ASK] Retrieving chunks for: {retrieval_query[:50]}...]")
        retrieval_start = time.time()
        
        initial_k = int(os.getenv("INITIAL_RETRIEVE_K", "12"))
        retriever._similarity_top_k = initial_k
        
        #nodes = retriever.retrieve(retrieval_query)
        nodes = retriever.retrieve(retrieval_query)  # Remove top_k here
        retrieval_time_ms = int((time.time() - retrieval_start) * 1000)
        logger.info(f"[ASK] ✅ Retrieved {len(nodes)} chunks in {retrieval_time_ms}ms")
        
        if not nodes:
            no_results_answer = "I couldn't find any relevant information in the indexed documents."
            return AskResponse(
                answer=no_results_answer,
                references=[],
                meta={
                    "client_id": req.client_id,
                    "chunks_retrieved": 0,
                    "query_enhanced": query_was_enhanced,
                    "query_intent": query_intent
                }
            )
        
        # Reranking
        reranking_start = time.time()
        reranked_nodes = _apply_reranking(retrieval_query, nodes, use_reranker=True)
        reranking_time_ms = int((time.time() - reranking_start) * 1000)
        
        # Smart chunk reordering by intent
        reranked_nodes = _reorder_chunks_by_intent(req.question, reranked_nodes, query_intent)
        
        # Dynamic chunk selection based on intent
        if query_intent.get("type") == "summarization":
            final_k = int(os.getenv("SUMMARIZATION_K", "10"))
            logger.info(f"[ASK] 📄 Document summarization: using {final_k} chunks")
        elif query_intent.get("is_compound"):
            final_k = int(os.getenv("COMPOUND_SYNTHESIS_K", "8"))
            logger.info(f"[ASK] 🔄 Compound question: using {final_k} chunks")
        else:
            final_k = int(os.getenv("FINAL_SYNTHESIS_K", "5"))
        
        final_nodes = reranked_nodes[:final_k]
        logger.info(f"[ASK] 📊 Using top {len(final_nodes)} chunks for synthesis")
        
        # Section-aware answer synthesis
        synthesis_start = time.time()
        answer = _synthesize_answer(
            req.question,
            final_nodes,
            use_llm=True,
            query_intent=query_intent
        )
        synthesis_time_ms = int((time.time() - synthesis_start) * 1000)
        
        # Build response
        refs = _build_references(final_nodes, req.client_id)
        # Build references HTML block for history replay
        refs_block = _refs_html(refs)

        
        total_time_ms = int((time.time() - start_time) * 1000)
        
        # Log query
        try:
            query_logger.log_query(
                client_id=req.client_id,
                user_org=req.client_id,
                index_name=req.index_name or req.client_id,
                conversation_id=req.conversation_id,
                original_query=req.question,
                enhanced_query=retrieval_query if query_was_enhanced else None,
                answer=answer,
                chunks_retrieved=len(nodes),
                chunks_used=len(final_nodes),
                retrieval_time_ms=retrieval_time_ms,
                reranking_time_ms=reranking_time_ms,
                synthesis_time_ms=synthesis_time_ms,
                total_time_ms=total_time_ms,
                query_intent=query_intent.get("type"),
                is_compound=query_intent.get("is_compound", False),
                is_followup=False,
                confidence_score=query_intent.get("confidence_score"),
                entities_detected=entity_recognizer.has_entities(entities),
                status="success"
            )
        except Exception as log_error:
            logger.warning(f"[ASK] Query logging failed: {log_error}")
        
        logger.info(f"[ASK] ✅ Request completed in {total_time_ms}ms")
        
        # Store conversation history using proper memory store
        try:
            from app.memory_store import get_store
            store = get_store()
            
            full_conversation_id = f"{req.client_id}:{req.conversation_id}"
            store.store_message("user", req.question, full_conversation_id)
            store.store_message("assistant", answer, full_conversation_id)
            if refs_block:
                store.store_message("assistant", refs_block, full_conversation_id)
                
            logger.info(f"[HISTORY] Stored messages for {full_conversation_id}")
            
        except Exception as hist_err:
            logger.warning(f"[HISTORY] Failed to store: {hist_err}")


        return AskResponse(
            answer=answer,
            references=refs,
            meta={
                "client_id": req.client_id,
                "chunks_retrieved": len(nodes),
                "chunks_used": len(final_nodes),
                "reranked": True,
                "query_enhanced": query_was_enhanced,
                "query_intent": query_intent,
                "performance": {
                    "retrieval_ms": retrieval_time_ms,
                    "reranking_ms": reranking_time_ms,
                    "synthesis_ms": synthesis_time_ms,
                    "total_ms": total_time_ms
                },
                **bundle.get("settings", {})
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[ASK] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# ============================================================================
# CONVERSATION HISTORY ENDPOINT
# ============================================================================
@router.get("/history")
async def get_history(conversation_id: str, client_id: Optional[str] = None):
    """
    Retrieve conversation history for a given conversation_id.
    Returns messages in chronological order.
    
    Args:
        conversation_id: The conversation ID to retrieve history for
        client_id: Optional client ID to scope the history (recommended)
        
    Returns:
        List of messages with role (user/assistant) and content
    """
    try:
        from app.memory_store import get_store
        store = get_store()
        if client_id and not conversation_id.startswith(f"{client_id}:"):
            full_conversation_id = f"{client_id}:{conversation_id}"
        else:
            full_conversation_id = conversation_id
        messages = store.fetch_all(full_conversation_id)
        history = []
        for msg in messages:
            history.append({
                "role": msg.role,
                "content": msg.content
            })
        
        logger.info(f"[HISTORY] Returning {len(history)} messages for {full_conversation_id}")
        return history
        
    except Exception as e:
        logger.error(f"[HISTORY] Error fetching history: {e}")
        return []
