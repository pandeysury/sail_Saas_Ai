#!/usr/bin/env python3
"""
Show actual HTML structure around bookmarks to understand the pattern
"""
import re

html_file = r"C:\sms\andriki\documents\5.06 Passage Planning.html"

with open(html_file, 'r', encoding='utf-8') as f:
    html_text = f.read()

print("="*80)
print("HTML STRUCTURE AROUND BOOKMARKS")
print("="*80)

# Find first 10 bookmarks and show context
count = 0
for m in re.finditer(r'<a[^>]*\bname="(_Toc[0-9A-Za-z_:-]+)"[^>]*></a>',
                     html_text, flags=re.I|re.S):
    count += 1
    if count > 10:  # Only show first 10
        break
    
    _id = m.group(1)
    pos = m.end()
    
    # Show 800 chars AFTER the bookmark
    context = html_text[pos:pos+800]
    
    print(f"\n{'='*80}")
    print(f"Bookmark #{count}: {_id}")
    print(f"{'='*80}")
    print("Next 800 chars:")
    print(context)
    print(f"{'='*80}")

# Also search for key headings we care about
print("\n\n")
print("="*80)
print("SEARCHING FOR KEY HEADINGS IN HTML")
print("="*80)

key_phrases = [
    "Ch. 5",
    "NAVIGATION",
    "Route Validation", 
    "Principles of passage planning",
    "Compliance"
]

for phrase in key_phrases:
    print(f"\n\nSearching for: '{phrase}'")
    print("-" * 40)
    
    # Find all occurrences
    pattern = re.escape(phrase)
    for m in re.finditer(pattern, html_text, flags=re.I):
        pos = m.start()
        
        # Show 200 chars BEFORE and AFTER
        before = html_text[max(0, pos-200):pos]
        after = html_text[pos:pos+200]
        
        print(f"\nFound at position {pos}:")
        print("BEFORE:", before[-100:])
        print(">>>", html_text[pos:m.end()], "<<<")
        print("AFTER:", after[:100])