#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_np133c.py - Diagnostic script to find NP 133C in indexed chunks
"""

import json
import sys
from pathlib import Path

# Paths
CHUNKS_PATH = Path(r"C:\sms\andriki\index_store\chunks.jsonl")
DOCS_PATH = Path(r"C:\sms\andriki\documents")

print("=" * 80)
print("NP 133C DIAGNOSTIC SCRIPT")
print("=" * 80)

# ============================================================================
# Part 1: Search in indexed chunks
# ============================================================================
print("\n[1] Searching for NP 133C in indexed chunks...")
print(f"Chunks file: {CHUNKS_PATH}")

if not CHUNKS_PATH.exists():
    print(f"❌ ERROR: Chunks file not found!")
    sys.exit(1)

found_chunks = []
total_chunks = 0

with CHUNKS_PATH.open("r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        total_chunks += 1
        try:
            chunk = json.loads(line)
            text_lower = chunk.get("text", "").lower()
            
            # Search for NP 133C variants
            if any(variant in text_lower for variant in ["np 133c", "np133c", "np-133c"]):
                found_chunks.append({
                    "line_num": line_num,
                    "chunk": chunk
                })
        except Exception as e:
            print(f"⚠️  Error parsing line {line_num}: {e}")
            continue

print(f"\n✓ Scanned {total_chunks} total chunks")
print(f"✓ Found {len(found_chunks)} chunks mentioning NP 133C")

if found_chunks:
    print("\n" + "=" * 80)
    print("FOUND CHUNKS WITH NP 133C:")
    print("=" * 80)
    
    for i, item in enumerate(found_chunks, 1):
        chunk = item["chunk"]
        meta = chunk.get("meta", {})
        
        print(f"\n--- Chunk {i} (Line {item['line_num']}) ---")
        print(f"Doc ID: {meta.get('doc_id', 'N/A')}")
        print(f"Breadcrumb: {meta.get('breadcrumb', 'N/A')}")
        print(f"Section: {meta.get('section_title', 'N/A')}")
        print(f"Tokens: {chunk.get('tokens', 'N/A')}")
        print(f"\nText preview (first 500 chars):")
        print("-" * 40)
        print(chunk.get("text", "")[:500])
        print("-" * 40)
        
        # Check metadata
        print(f"\nMetadata:")
        print(f"  - VIQ hints: {meta.get('viq_hints', 'N/A')}")
        print(f"  - Synonyms: {meta.get('synonyms_hit', 'N/A')}")
        print(f"  - Domain tags: {meta.get('domain_tags', 'N/A')[:100]}...")
        
        if i >= 3:  # Show max 3 chunks
            print(f"\n... and {len(found_chunks) - 3} more chunks")
            break
else:
    print("\n❌ NO CHUNKS FOUND with NP 133C!")
    print("\nThis means either:")
    print("  1. The content wasn't indexed")
    print("  2. The HTML file containing NP 133C wasn't processed")
    print("  3. The section was skipped (e.g., empty after parsing)")

# ============================================================================
# Part 2: Search in source HTML files
# ============================================================================
print("\n" + "=" * 80)
print("[2] Searching for NP 133C in source HTML files...")
print(f"Documents folder: {DOCS_PATH}")
print("=" * 80)

if not DOCS_PATH.exists():
    print(f"❌ ERROR: Documents folder not found!")
    sys.exit(1)

html_files = list(DOCS_PATH.glob("*.html")) + list(DOCS_PATH.glob("*.htm"))
print(f"\n✓ Found {len(html_files)} HTML files")

found_in_files = []

for html_file in html_files:
    try:
        content = html_file.read_text(encoding="utf-8", errors="ignore")
        content_lower = content.lower()
        
        if any(variant in content_lower for variant in ["np 133c", "np133c", "np-133c"]):
            # Count occurrences
            count = sum(content_lower.count(variant) for variant in ["np 133c", "np133c", "np-133c"])
            found_in_files.append({
                "file": html_file.name,
                "count": count,
                "content": content
            })
    except Exception as e:
        print(f"⚠️  Error reading {html_file.name}: {e}")
        continue

if found_in_files:
    print(f"\n✓ Found NP 133C in {len(found_in_files)} HTML files:")
    
    for item in found_in_files:
        print(f"\n📄 File: {item['file']}")
        print(f"   Mentions: {item['count']} times")
        
        # Show context around first mention
        content_lower = item['content'].lower()
        idx = -1
        for variant in ["np 133c", "np133c", "np-133c"]:
            idx = content_lower.find(variant)
            if idx != -1:
                break
        
        if idx != -1:
            # Get 300 chars before and after
            start = max(0, idx - 300)
            end = min(len(item['content']), idx + 300)
            context = item['content'][start:end]
            
            # Clean HTML tags for readability
            import re
            context_clean = re.sub(r'<[^>]+>', ' ', context)
            context_clean = re.sub(r'\s+', ' ', context_clean).strip()
            
            print(f"\n   Context (first mention):")
            print(f"   {'-' * 60}")
            print(f"   ...{context_clean}...")
            print(f"   {'-' * 60}")
else:
    print("\n❌ NO HTML FILES contain NP 133C!")
    print("\nThis is very unusual. Possible reasons:")
    print("  1. Wrong documents folder")
    print("  2. Content uses different terminology")
    print("  3. Files are in a subdirectory")

# ============================================================================
# Part 3: Summary & Recommendations
# ============================================================================
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

if found_chunks and found_in_files:
    print("\n✅ GOOD NEWS: NP 133C is both indexed and in source files!")
    print("\nProblem: Vector search is NOT retrieving these chunks.")
    print("\nPossible reasons:")
    print("  1. Chunk is too large (diluted with other content)")
    print("  2. Semantic similarity between query and chunk is low")
    print("  3. Other chunks have higher similarity scores")
    print("\nRECOMMENDATION:")
    print("  → Review the chunk content above")
    print("  → Check if 'recordkeeping' or 'passage planning' keywords are present")
    print("  → Consider reducing chunk size further (600 → 500 tokens)")
    print("  → Add explicit metadata filter for record_book field")

elif not found_chunks and found_in_files:
    print("\n⚠️  PROBLEM: NP 133C exists in source files but NOT in chunks!")
    print("\nThis means the indexer skipped this content.")
    print("\nPossible reasons:")
    print("  1. Section generated 0 chunks (empty after parsing)")
    print("  2. File was processed but chunks weren't written")
    print("  3. Parsing error during that specific section")
    print("\nRECOMMENDATION:")
    print("  → Re-run indexer with --reset flag")
    print("  → Check indexer logs for errors on this file")
    print("  → Verify the HTML structure is parseable")

elif found_chunks and not found_in_files:
    print("\n⚠️  WEIRD: NP 133C in chunks but NOT in source files!")
    print("\nThis shouldn't happen unless:")
    print("  1. Wrong documents folder path")
    print("  2. Files were moved/renamed after indexing")
    print("\nRECOMMENDATION:")
    print("  → Double-check DOCS_PATH is correct")
    print("  → Look in subdirectories")

else:
    print("\n❌ CRITICAL: NP 133C not found ANYWHERE!")
    print("\nPossible reasons:")
    print("  1. Wrong client (should this be 'rsms' not 'andriki'?)")
    print("  2. Content uses different terminology (e.g., 'Navigation Record Book')")
    print("  3. Wrong folder paths in this script")
    print("\nRECOMMENDATION:")
    print("  → Verify you're checking the correct client folder")
    print("  → Search for alternative terms like 'navigation record' or 'logbook'")
    print("  → Check if NP 133C is referenced by full name instead")

print("\n" + "=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("""
1. Review the output above carefully
2. If chunks were found: Check their content and metadata
3. If no chunks found: Re-index with smaller chunk size
4. Share this output for further analysis

To re-index with better settings:
  python indexer61025_optimized.py \\
    --client_root "C:\\sms\\andriki" \\
    --rules "C:\\sms\\rules.yaml" \\
    --chunk_size 600 \\
    --chunk_overlap 80 \\
    --parallel_workers 6
""")
print("=" * 80)