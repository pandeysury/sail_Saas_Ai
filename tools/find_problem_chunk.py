import json

chunks_file = r"C:\sms\andriki\index_store\chunks.jsonl"

print("Searching for chunk with section_id='sec-b4613f8681'...")
print("="*80)

found = False
with open(chunks_file, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        try:
            chunk = json.loads(line)
            metadata = chunk.get('metadata', {})
            
            if metadata.get('section_id') == 'sec-b4613f8681':
                found = True
                print(f"\n✅ FOUND at line {line_num}!")
                print(f"\nSection Title: '{metadata.get('section_title')}'")
                print(f"Section ID: {metadata.get('section_id')}")
                print(f"Breadcrumb: {metadata.get('breadcrumb')}")
                print(f"Slug URL: {metadata.get('slug_url')}")
                print(f"\nText (first 500 chars):")
                print(chunk.get('text', '')[:500])
                print(f"\n{'='*80}")
                break
        except:
            continue

if not found:
    print("\n❌ Chunk not found!")

print("\n\nNow searching for chunks with title '!!'...")
print("="*80)

with open(chunks_file, 'r', encoding='utf-8') as f:
    count = 0
    for line in f:
        try:
            chunk = json.loads(line)
            metadata = chunk.get('metadata', {})
            
            if metadata.get('section_title') == '!!':
                count += 1
                if count <= 3:  # Show first 3
                    print(f"\n{count}. Section ID: {metadata.get('section_id')}")
                    print(f"   Breadcrumb: {metadata.get('breadcrumb')}")
                    print(f"   Text: {chunk.get('text', '')[:200]}")
        except:
            continue
    
    print(f"\n\nTotal chunks with title '!!': {count}")