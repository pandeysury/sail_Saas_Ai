import sqlite3
from pathlib import Path

# Connect to your query logs database
db_path = Path("query_logs.db")  # Adjust path if needed
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get recent queries with enhancement info
query = """
SELECT 
    id,
    timestamp,
    client_id,
    original_query,
    enhanced_query,
    was_rewritten,
    answer,
    status
FROM query_logs 
ORDER BY timestamp DESC 
LIMIT 10;
"""

# ✅ EXECUTE THE QUERY (this was missing!)
cursor.execute(query)
results = cursor.fetchall()

# Print the results
print(f"{'ID':<5} {'Original Query':<50} {'Enhanced Query':<50} {'Enhanced?'}")
print("=" * 160)

for row in results:
    log_id, timestamp, client_id, orig, enhanced, rewritten, answer, status = row
    enhanced_text = enhanced if enhanced else "(not enhanced)"
    print(f"{log_id:<5} {orig[:50]:<50} {enhanced_text[:50]:<50} {rewritten}")

conn.close()