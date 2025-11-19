#!/usr/bin/env python3
"""
Query Logs Viewer - View and analyze RAG system query logs
Usage: python view_query_logs.py [command] [options]
"""
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.query_logger import get_query_logger


def show_recent_queries(client_id=None, limit=20):
    """Show recent queries"""
    logger = get_query_logger()
    queries = logger.get_recent_queries(client_id=client_id, limit=limit)
    
    if not queries:
        print("No queries found.")
        return
    
    print(f"\n{'='*100}")
    print(f"RECENT QUERIES (showing {len(queries)} of {limit} requested)")
    print(f"{'='*100}\n")
    
    for q in queries:
        print(f"ID: {q['id']} | {q['timestamp']} | Client: {q['client_id']}")
        print(f"Question: {q['original_query']}")
        
        if q['was_rewritten']:
            print(f"Enhanced: {q['enhanced_query']}")
        
        print(f"Answer: {q['answer'][:150]}{'...' if len(q['answer']) > 150 else ''}")
        print(f"Status: {q['status']} | Chunks: {q['chunks_retrieved']} → {q['chunks_reranked']} → {q['chunks_used']}")
        print(f"Top Score: {q['top_reference_score']} | Time: {q['total_time_ms']}ms")
        print(f"-" * 100)
    
    print()


def show_analytics(client_id=None, days=30):
    """Show analytics"""
    logger = get_query_logger()
    analytics = logger.get_query_analytics(client_id=client_id, days=days)
    
    if not analytics:
        print("No analytics available.")
        return
    
    print(f"\n{'='*60}")
    print(f"QUERY ANALYTICS (Last {days} days)")
    if client_id:
        print(f"Client: {client_id}")
    print(f"{'='*60}\n")
    
    print(f"Total Queries:     {analytics['total_queries']}")
    print(f"Success Rate:      {analytics['success_rate']}%")
    print(f"Rewrite Rate:      {analytics['rewrite_rate']}%")
    
    print(f"\nPERFORMANCE METRICS:")
    perf = analytics['performance']
    print(f"Avg Total Time:    {perf.get('avg_total_time', 0):.0f}ms")
    print(f"Avg Retrieval:     {perf.get('avg_retrieval_time', 0):.0f}ms")
    print(f"Avg Reranking:     {perf.get('avg_reranking_time', 0):.0f}ms")
    print(f"Avg Synthesis:     {perf.get('avg_synthesis_time', 0):.0f}ms")
    print(f"Avg Chunks:        {perf.get('avg_chunks_retrieved', 0):.1f}")
    print()


def export_logs(output_file="query_logs_export.csv", client_id=None):
    """Export logs to CSV"""
    logger = get_query_logger()
    count = logger.export_to_csv(output_file, client_id=client_id)
    print(f"\n✅ Exported {count} queries to {output_file}")


def show_query_details(query_id):
    """Show detailed information for a specific query"""
    import sqlite3
    import json
    
    logger = get_query_logger()
    
    with sqlite3.connect(str(logger.db_path)) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get main query info
        cursor.execute("SELECT * FROM query_logs WHERE id = ?", (query_id,))
        query = cursor.fetchone()
        
        if not query:
            print(f"Query ID {query_id} not found.")
            return
        
        query = dict(query)
        
        # Get references
        cursor.execute("""
            SELECT * FROM query_references 
            WHERE query_log_id = ? 
            ORDER BY reference_rank
        """, (query_id,))
        references = [dict(row) for row in cursor.fetchall()]
        
        # Display query details
        print(f"\n{'='*100}")
        print(f"QUERY DETAILS - ID: {query_id}")
        print(f"{'='*100}\n")
        
        print(f"Timestamp:         {query['timestamp']}")
        print(f"Client ID:         {query['client_id']}")
        print(f"User Org:          {query['user_org']}")
        print(f"Index Name:        {query['index_name']}")
        print(f"Conversation ID:   {query['conversation_id']}")
        
        print(f"\nQUERY:")
        print(f"Original:  {query['original_query']}")
        if query['was_rewritten']:
            print(f"Enhanced:  {query['enhanced_query']}")
        
        print(f"\nANSWER ({query['answer_length']} chars):")
        print(f"{query['answer']}\n")
        
        print(f"METRICS:")
        print(f"Status:            {query['status']}")
        print(f"Chunks Retrieved:  {query['chunks_retrieved']}")
        print(f"Chunks Reranked:   {query['chunks_reranked']}")
        print(f"Chunks Used:       {query['chunks_used']}")
        print(f"Reranker Enabled:  {query['reranker_enabled']}")
        
        print(f"\nPERFORMANCE:")
        print(f"Retrieval Time:    {query['retrieval_time_ms']}ms")
        print(f"Reranking Time:    {query['reranking_time_ms']}ms")
        print(f"Synthesis Time:    {query['synthesis_time_ms']}ms")
        print(f"Total Time:        {query['total_time_ms']}ms")
        
        if references:
            print(f"\nREFERENCES ({len(references)}):")
            print(f"{'-'*100}")
            for ref in references:
                print(f"\n#{ref['reference_rank']} | Score: {ref['score']}")
                print(f"Title:      {ref['title']}")
                print(f"Breadcrumb: {ref['breadcrumb']}")
                print(f"URL:        {ref['url']}")
                if ref['viq_codes']:
                    viq = json.loads(ref['viq_codes'])
                    print(f"VIQ Codes:  {', '.join(viq[:10])}{'...' if len(viq) > 10 else ''}")
                if ref['text_snippet']:
                    print(f"Snippet:    {ref['text_snippet'][:150]}...")
        
        if query['metadata']:
            print(f"\nMETADATA:")
            metadata = json.loads(query['metadata'])
            for key, value in metadata.items():
                print(f"{key}: {value}")
        
        print(f"\n{'='*100}\n")


def show_failed_queries(client_id=None, limit=20):
    """Show queries that failed or returned no results"""
    import sqlite3
    
    logger = get_query_logger()
    
    with sqlite3.connect(str(logger.db_path)) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        where_clause = "WHERE status != 'success'"
        if client_id:
            where_clause += f" AND client_id = '{client_id}'"
        
        cursor.execute(f"""
            SELECT * FROM query_logs 
            {where_clause}
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        
        queries = [dict(row) for row in cursor.fetchall()]
    
    if not queries:
        print("\n✅ No failed queries found!")
        return
    
    print(f"\n{'='*100}")
    print(f"FAILED/NO RESULTS QUERIES (showing {len(queries)})")
    print(f"{'='*100}\n")
    
    for q in queries:
        print(f"ID: {q['id']} | {q['timestamp']} | Status: {q['status']}")
        print(f"Client: {q['client_id']}")
        print(f"Question: {q['original_query']}")
        if q['error_message']:
            print(f"Error: {q['error_message']}")
        print(f"-" * 100)
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="View and analyze RAG system query logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python view_query_logs.py recent                     # Show 20 recent queries
  python view_query_logs.py recent --limit 50          # Show 50 recent queries
  python view_query_logs.py recent --client andriki    # Show queries for specific client
  python view_query_logs.py analytics                  # Show analytics for last 30 days
  python view_query_logs.py analytics --days 7         # Show analytics for last 7 days
  python view_query_logs.py details 123                # Show details for query ID 123
  python view_query_logs.py failed                     # Show failed queries
  python view_query_logs.py export                     # Export all logs to CSV
  python view_query_logs.py export --client andriki    # Export logs for specific client
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Recent queries command
    recent_parser = subparsers.add_parser('recent', help='Show recent queries')
    recent_parser.add_argument('--client', help='Filter by client ID')
    recent_parser.add_argument('--limit', type=int, default=20, help='Number of queries to show')
    
    # Analytics command
    analytics_parser = subparsers.add_parser('analytics', help='Show query analytics')
    analytics_parser.add_argument('--client', help='Filter by client ID')
    analytics_parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
    
    # Details command
    details_parser = subparsers.add_parser('details', help='Show detailed info for a query')
    details_parser.add_argument('query_id', type=int, help='Query ID to show details for')
    
    # Failed queries command
    failed_parser = subparsers.add_parser('failed', help='Show failed queries')
    failed_parser.add_argument('--client', help='Filter by client ID')
    failed_parser.add_argument('--limit', type=int, default=20, help='Number of queries to show')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export logs to CSV')
    export_parser.add_argument('--output', default='query_logs_export.csv', help='Output file path')
    export_parser.add_argument('--client', help='Filter by client ID')
    
    args = parser.parse_args()
    
    if args.command == 'recent':
        show_recent_queries(client_id=args.client, limit=args.limit)
    elif args.command == 'analytics':
        show_analytics(client_id=args.client, days=args.days)
    elif args.command == 'details':
        show_query_details(args.query_id)
    elif args.command == 'failed':
        show_failed_queries(client_id=args.client, limit=args.limit)
    elif args.command == 'export':
        export_logs(output_file=args.output, client_id=args.client)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()