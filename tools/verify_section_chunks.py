import json

chunks_file = r"C:\sms\rsms\index_store\chunks.jsonl"

print("="*80)
print("SECTION-WISE CHUNKS VERIFICATION")
print("="*80)

with open(chunks_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"\nTotal chunks: {len(lines)}\n")

# Count by section_id type
real_toc_ids = 0
fallback_ids = 0
real_toc_chunks = []

for line in lines:
    chunk = json.loads(line)
    metadata = chunk.get('metadata', {})
    section_id = metadata.get('section_id', '')
    
    if section_id.startswith('_Toc'):
        real_toc_ids += 1
        real_toc_chunks.append({
            'title': metadata.get('section_title'),
            'id': section_id,
            'url': metadata.get('slug_url')
        })
    elif section_id.startswith('sec-'):
        fallback_ids += 1

print(f"✅ Chunks with real _Toc IDs: {real_toc_ids} ({real_toc_ids/len(lines)*100:.1f}%)")
print(f"❌ Chunks with fallback IDs: {fallback_ids} ({fallback_ids/len(lines)*100:.1f}%)")

# Show first 10 chunks
print("\n" + "="*80)
print("FIRST 10 CHUNKS:")
print("="*80)

for i, line in enumerate(lines[:10], 1):
    chunk = json.loads(line)
    metadata = chunk.get('metadata', {})
    
    print(f"\n{i}. Title: '{metadata.get('section_title')}'")
    print(f"   Section ID: {metadata.get('section_id')}")
    print(f"   URL: {metadata.get('slug_url')}")
    print(f"   Is Complete: {metadata.get('is_complete_section')}")
    print(f"   Text preview: {chunk.get('text', '')[:100]}...")

# Search for "Route Validation"
print("\n" + "="*80)
print("SEARCHING FOR 'ROUTE VALIDATION':")
print("="*80)

for line in lines:
    chunk = json.loads(line)
    metadata = chunk.get('metadata', {})
    
    if 'route validation' in metadata.get('section_title', '').lower():
        print(f"\n✅ FOUND!")
        print(f"   Title: '{metadata.get('section_title')}'")
        print(f"   Section ID: {metadata.get('section_id')}")
        print(f"   URL: {metadata.get('slug_url')}")
        print(f"   Text: {chunk.get('text', '')[:200]}...")
        break
else:
    print("\n❌ Not found by title. Searching in text...")
    for line in lines:
        chunk = json.loads(line)
        if 'route validation' in chunk.get('text', '').lower()[:500]:
            metadata = chunk.get('metadata', {})
            print(f"\n✅ FOUND in text!")
            print(f"   Title: '{metadata.get('section_title')}'")
            print(f"   Section ID: {metadata.get('section_id')}")
            print(f"   URL: {metadata.get('slug_url')}")
            print(f"   Text: {chunk.get('text', '')[:200]}...")
            break

# Show all unique section IDs
print("\n" + "="*80)
print("ALL SECTION IDS:")
print("="*80)

section_ids = set()
for line in lines:
    chunk = json.loads(line)
    metadata = chunk.get('metadata', {})
    section_ids.add(metadata.get('section_id'))

for sid in sorted(section_ids):
    status = "✅" if sid.startswith('_Toc') else "❌"
    print(f"{status} {sid}")