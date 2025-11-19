#!/usr/bin/env python3
"""Find the full NP 133C context in source HTML"""

import re
from pathlib import Path
from bs4 import BeautifulSoup

# Source file
html_path = Path(r"C:\sms\andriki\documents\5.06 Passage Planning.html")

if not html_path.exists():
    print(f"ERROR: File not found: {html_path}")
    exit(1)

print("=" * 80)
print("SEARCHING FOR NP 133C FULL CONTEXT")
print("=" * 80)

html = html_path.read_text(encoding="utf-8", errors="ignore")
soup = BeautifulSoup(html, "lxml")

# Find all text mentioning NP 133C
body_text = soup.get_text()
lines = body_text.split('\n')

# Find the line with "NP 133C"
for i, line in enumerate(lines):
    if 'np 133c' in line.lower() or 'np133c' in line.lower():
        print(f"\n--- Found at line {i} ---")
        
        # Get context: 5 lines before, the line, and 20 lines after
        start = max(0, i - 5)
        end = min(len(lines), i + 25)
        
        context = '\n'.join(lines[start:end])
        # Clean up whitespace
        context = re.sub(r'\n\s*\n', '\n', context)
        
        print(context)
        print("-" * 80)

print("\n" + "=" * 80)
print("Now let's find the HTML structure around 'Recordkeeping':")
print("=" * 80)

# Find the actual HTML section
html_lower = html.lower()
idx = html_lower.find('np 133c')

if idx != -1:
    # Get 2000 chars before and 3000 after
    start = max(0, idx - 2000)
    end = min(len(html), idx + 3000)
    
    html_snippet = html[start:end]
    
    # Parse this snippet
    snippet_soup = BeautifulSoup(html_snippet, "lxml")
    
    print("\nHTML Structure:")
    print("-" * 80)
    
    # Find the parent structure
    for tag in snippet_soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'ul', 'li', 'strong']):
        text = tag.get_text(strip=True)
        if text and len(text) > 5:  # Skip empty tags
            print(f"{tag.name:10s}: {text[:100]}")
    
    print("-" * 80)
    
    print("\n\nRAW HTML around NP 133C:")
    print("=" * 80)
    print(html_snippet)
    print("=" * 80)

print("\n\nCONCLUSION:")
print("=" * 80)
print("""
Based on the context above, identify:

1. What section heading contains "NP 133C"?
2. Is there a list of records after the "NP 133C" mention?
3. How far apart are they (in lines/characters)?

This will tell us if the content is being split into separate chunks!
""")