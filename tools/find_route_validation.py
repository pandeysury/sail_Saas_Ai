import json

chunks_file = r"C:\sms\andriki\index_store\chunks.jsonl"

print("Searching for chunks with 'route validation' in title or text...")
print("="*80)

found = []
with open(chunks_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            chunk = json.loads(line)
            metadata = chunk.get('metadata', {})
            text = chunk.get('text', '').lower()
            title = metadata.get('section_title', '').lower()
            
            if 'route validation' in title or 'route validation' in text[:500]:
                found.append({
                    'title': metadata.get('section_title'),
                    'section_id': metadata.get('section_id'),
                    'breadcrumb': metadata.get('breadcrumb'),
                    'text_preview': chunk.get('text', '')[:200]
                })
        except:
            continue

print(f"\nFound {len(found)} chunks mentioning 'route validation':\n")

for i, chunk in enumerate(found[:5], 1):  # Show first 5
    print(f"{i}. Title: '{chunk['title']}'")
    print(f"   Section ID: {chunk['section_id']}")
    print(f"   Breadcrumb: {chunk['breadcrumb']}")
    print(f"   Preview: {chunk['text_preview']}")
    print()