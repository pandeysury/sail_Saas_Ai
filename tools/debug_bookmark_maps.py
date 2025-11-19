#!/usr/bin/env python3
"""
Debug script to see exactly what's in the anchor maps
"""
import re
import html as _html

def _strip_tags(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r'<\s*script[^>]*>.*?</\s*script\s*>', " ", s, flags=re.I|re.S)
    s = re.sub(r'<\s*style[^>]*>.*?</\s*style\s*>', " ", s, flags=re.I|re.S)
    s = re.sub(r'<[^>]+>', " ", s)
    s = _html.unescape(s)
    return re.sub(r'\s+', ' ', s).strip()

def _norm_txt(s: str) -> str:
    text = (s or "").strip()
    text = re.sub(r'^\s*[\d\.]+\s*', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

html_file = r"C:\sms\andriki\documents\5.06 Passage Planning.html"

with open(html_file, 'r', encoding='utf-8') as f:
    html_text = f.read()

print("="*80)
print("BOOKMARK EXTRACTION DEBUG")
print("="*80)

bookmark_map = {}
bookmark_count = 0

for m in re.finditer(r'<a[^>]*\bname="(_Toc[0-9A-Za-z_:-]+)"[^>]*>(.*?)</a>',
                     html_text, flags=re.I|re.S):
    _id = m.group(1)
    _txt = _strip_tags(m.group(2))  # Text is INSIDE the tag!
    
    if _id and _txt:
        normalized = _norm_txt(_txt)
        if normalized:
            bookmark_count += 1
            
            # Show if this key already exists (overwrite)
            if normalized in bookmark_map:
                print(f"\n⚠️  OVERWRITE: '{normalized}'")
                print(f"   Old: {bookmark_map[normalized]}")
                print(f"   New: {_id}")
            else:
                print(f"\n✅ New: '{normalized}' → {_id}")
            
            bookmark_map[normalized] = _id
            print(f"   Raw text: {_txt[:100]}")

print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"Total bookmarks found: {bookmark_count}")
print(f"Unique keys in map: {len(bookmark_map)}")
print(f"\nFinal bookmark_map:")
for key, value in bookmark_map.items():
    print(f"  '{key}' → {value}")

print(f"\n{'='*80}")
print("Now let's test matching...")
print(f"{'='*80}")

test_titles = ["Ch.", "Route", "Principles of passage planning", "Compliance"]

for title in test_titles:
    norm = _norm_txt(title)
    print(f"\nTitle: '{title}' → normalized: '{norm}'")
    
    # Exact match
    if norm in bookmark_map:
        print(f"  ✅ Exact match: {bookmark_map[norm]}")
    else:
        print(f"  ❌ No exact match")
        
        # Partial match
        found = False
        if len(norm) >= 2:
            for key, value in bookmark_map.items():
                if key.startswith(norm + " ") or (len(key) > len(norm) and key.startswith(norm)):
                    print(f"  ✅ Partial match: '{key}' → {value}")
                    found = True
                    break
        
        if not found:
            print(f"  ❌ No partial match either")