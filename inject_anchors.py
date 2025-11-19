#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inject_anchors.py — Add stable heading anchors to Aspose HTML docs (idempotent)

What it does
- Finds headings: <h1>…<h6> and <p class="HeadingN"> (Aspose)
- If an element already has an id (e.g., _Toc1944…), it is preserved
- If missing, generates a deterministic id: sec-<slug-of-heading-text>
- Ensures uniqueness within the file by appending -2, -3, …
- Saves a per-file slug map: <file>.slugmap.json

Defaults are SAFE:
- Existing IDs are NOT overwritten (preserve=true).
- Only missing IDs are injected.

Usage
  python tools/inject_anchors.py --docs "C:\\sms\\andriki\\documents"
  # dry-run:
  python tools/inject_anchors.py --docs "/path/docs" --dry-run

Args
  --docs     : required; documents folder (root that contains .html/.htm)
  --dry-run  : don’t write files, just report
  --backup   : write a .bak before modifying a file
  --max-len  : max characters in generated slug part (default 120)
"""

import re
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict

try:
    from bs4 import BeautifulSoup
except ImportError as e:
    raise SystemExit("Missing dependency: beautifulsoup4\npip install beautifulsoup4") from e


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

HEADING_SELECTORS = [
    "h1", "h2", "h3", "h4", "h5", "h6",
    'p[class^="Heading"]',  # Aspose: <p class="Heading1">…</p>
]

def stable_slug(text: str, max_len: int = 120) -> str:
    """Deterministic, URL-safe slug (lowercase alnum and hyphens)."""
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    base = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    if not base:
        base = "untitled"
    if max_len and len(base) > max_len:
        base = base[:max_len].rstrip("-")
    return f"sec-{base}"

def load_html(path: Path) -> BeautifulSoup:
    html = path.read_text(encoding="utf-8", errors="ignore")
    return BeautifulSoup(html, "html.parser")

def save_html(path: Path, soup: BeautifulSoup, backup: bool):
    if backup and not path.with_suffix(path.suffix + ".bak").exists():
        path.with_suffix(path.suffix + ".bak").write_text(
            path.read_text(encoding="utf-8", errors="ignore"),
            encoding="utf-8"
        )
    path.write_text(str(soup), encoding="utf-8")

def process_file(html_path: Path, dry_run: bool, backup: bool, max_len: int) -> Dict:
    soup = load_html(html_path)

    # Track uniqueness within this file
    seen: Dict[str, int] = {}
    def uniq(s: str) -> str:
        c = seen.get(s, 0) + 1
        seen[s] = c
        return s if c == 1 else f"{s}-{c}"

    changed = 0
    slugs: List[Dict] = []

    headings = soup.select(",".join(HEADING_SELECTORS))
    for el in headings:
        text = el.get_text(" ", strip=True)
        if not text:
            continue

        existing_id = el.get("id")
        if existing_id:
            # Preserve existing anchor (e.g., _Toc…)
            uid = uniq(existing_id)
            if uid != existing_id:
                # VERY rare: duplicate existing ids within same file; make unique
                el["id"] = uid
                changed += 1
        else:
            # Generate deterministic id
            base = stable_slug(text, max_len=max_len)
            uid = uniq(base)
            el["id"] = uid
            changed += 1

        slugs.append({
            "title": text,
            "id": el.get("id"),
            "tag": el.name,
            "class": " ".join(el.get("class", [])) if isinstance(el.get("class"), list) else (el.get("class") or "")
        })

    # Write outputs
    if changed and not dry_run:
        save_html(html_path, soup, backup=backup)
        logging.info(f"Updated {html_path.name}: {changed} anchors")

    # Always emit slug map (handy for debugging)
    slugmap = {"file": html_path.name, "slugs": slugs}
    if not dry_run:
        html_path.with_suffix(html_path.suffix + ".slugmap.json").write_text(
            json.dumps(slugmap, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    return {"file": html_path.name, "changed": changed, "total_headings": len(headings)}

def main():
    ap = argparse.ArgumentParser(description="Inject stable anchors into HTML docs (Aspose-friendly).")
    ap.add_argument("--docs", required=True, help="Path to documents dir (e.g., C:/sms/andriki/documents)")
    ap.add_argument("--dry-run", action="store_true", help="Analyze only; do not write files")
    ap.add_argument("--backup", action="store_true", help="Write a .bak alongside modified files")
    ap.add_argument("--max-len", type=int, default=120, help="Max chars in slug (default: 120)")
    args = ap.parse_args()

    root = Path(args.docs).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Docs directory not found: {root}")

    html_files = list(root.rglob("*.html")) + list(root.rglob("*.htm"))
    if not html_files:
        raise SystemExit(f"No HTML files under: {root}")

    total_changed = 0
    for f in sorted(html_files):
        try:
            r = process_file(f, dry_run=args.dry_run, backup=args.backup, max_len=args.max_len)
            total_changed += r["changed"]
        except Exception as e:
            logging.error(f"{f}: {e}")

    logging.info(f"Done. Files modified: {total_changed}")

if __name__ == "__main__":
    main()
