
# Sample parsing code based on your HTML structure

import re
from bs4 import BeautifulSoup

def parse_aspose_html(html_content: str):
    """Parse Aspose HTML with TOC extraction."""
    
    # STEP 1: Extract TOC map BEFORE any cleaning
    toc_map = extract_toc_map(html_content)
    print(f"Extracted {len(toc_map)} TOC mappings")
    
    # STEP 2: Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # STEP 3: Find all headings (including paragraph-style)
    headings = []
    
    # Standard headings
    for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        for h in soup.find_all(tag):
            headings.append({
                'tag': tag,
                'id': h.get('id', ''),
                'text': h.get_text(strip=True),
                'element': h
            })
    
    # Paragraph headings (Aspose style)
    for p in soup.find_all('p'):
        classes = ' '.join(p.get('class', []))
        if re.search(r'Heading\d', classes, re.I):
            headings.append({
                'tag': 'p',
                'id': p.get('id', ''),
                'text': p.get_text(strip=True),
                'element': p
            })
    
    # STEP 4: Match headings to TOC IDs
    for heading in headings:
        text = heading['text']
        normalized = normalize_text(text)
        
        if normalized in toc_map:
            heading['toc_id'] = toc_map[normalized]
            print(f"✅ Matched: '{text}' → {toc_map[normalized]}")
        else:
            heading['toc_id'] = heading['id'] or generate_fallback_id(text)
            print(f"⚠️  No TOC match for: '{text}', using fallback")
    
    return headings


def extract_toc_map(html: str) -> dict:
    """Extract TOC mappings from HTML."""
    toc_map = {}
    
    # Find all <a href="#_Toc...">Text</a>
    pattern = r'<a[^>]+href="#(_Toc[0-9]+)"[^>]*>(.*?)</a>'
    
    for match in re.finditer(pattern, html, re.I | re.S):
        toc_id = match.group(1)
        text = strip_html_tags(match.group(2))
        
        if text and toc_id:
            normalized = normalize_text(text)
            toc_map[normalized] = toc_id
    
    return toc_map


def normalize_text(text: str) -> str:
    """Normalize text for matching."""
    # Remove leading section numbers
    text = re.sub(r'^\s*[\d\.]+\s+', '', text)
    
    # Lowercase and normalize whitespace
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    
    return text


def strip_html_tags(html: str) -> str:
    """Remove HTML tags from text."""
    # Remove script/style content
    html = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.I|re.S)
    html = re.sub(r'<style[^>]*>.*?</style>', ' ', html, flags=re.I|re.S)
    
    # Remove tags
    html = re.sub(r'<[^>]+>', ' ', html)
    
    # Decode HTML entities
    import html as html_module
    html = html_module.unescape(html)
    
    # Normalize whitespace
    html = re.sub(r'\s+', ' ', html).strip()
    
    return html


def generate_fallback_id(text: str) -> str:
    """Generate fallback ID if no TOC match."""
    import hashlib
    
    # Create slug from text
    slug = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
    
    if not slug:
        # Use hash if text is empty or all non-alphanumeric
        slug = hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]
    
    return f'sec-{slug[:120]}'


# Example usage:
if __name__ == '__main__':
    with open('5.06 Passage Planning.html', 'r', encoding='utf-8') as f:
        html = f.read()
    
    headings = parse_aspose_html(html)
    
    print(f"\nFound {len(headings)} headings:")
    for h in headings[:10]:
        print(f"  - {h['text'][:60]} → {h['toc_id']}")
