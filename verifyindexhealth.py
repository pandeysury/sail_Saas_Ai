import chromadb
from pathlib import Path
client = chromadb.PersistentClient(path=r"C:\sms\andriki\index_store\chroma")
col = client.get_or_create_collection(name="docs")
print("Total in docs:", col.count())

import json
cnt = 0
with open(r"C:\sms\andriki\index_store\chunks.jsonl","r",encoding="utf-8",errors="ignore") as f:
    for _ in f: cnt += 1
print("Chunks.jsonl lines:", cnt)
