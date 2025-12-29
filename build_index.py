"""
Build indexes for hybrid search.

Loads chunks from processed data and builds both vector and keyword indexes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.processing.chunker import DocumentChunker
from src.indexing.hybrid_search import HybridSearch


def main():
    """Build all search indexes."""
    print("=" * 50)
    print("Building Search Indexes")
    print("=" * 50)
    
    # Load chunks
    print("\n1. Loading chunks...")
    chunker = DocumentChunker()
    chunks = chunker.load_chunks()
    
    if not chunks:
        print("ERROR: No chunks found. Run the chunker first:")
        print("  python -m src.processing.chunker")
        return False
    
    print(f"   Loaded {len(chunks)} chunks")
    
    # Initialize hybrid search (this creates both stores)
    print("\n2. Initializing search stores...")
    hybrid = HybridSearch()
    
    # Check if already indexed
    stats = hybrid.get_stats()
    vector_count = stats["vector_store"]["count"]
    keyword_count = stats["keyword_store"]["document_count"]
    
    if vector_count > 0 and keyword_count > 0:
        print(f"   Vector store already has {vector_count} documents")
        print(f"   Keyword store already has {keyword_count} documents")
        
        reindex = input("\n   Re-index? (y/n): ").lower().strip() == 'y'
        if not reindex:
            print("   Skipping indexing.")
            return True
        
        print("\n   Clearing existing indexes...")
        hybrid.clear()
    
    # Build indexes
    print("\n3. Building indexes...")
    hybrid.add_documents(chunks)
    
    # Verify
    print("\n4. Verifying indexes...")
    stats = hybrid.get_stats()
    print(f"   Vector store: {stats['vector_store']['count']} documents")
    print(f"   Keyword store: {stats['keyword_store']['document_count']} documents")
    
    # Test search
    print("\n5. Testing search...")
    test_query = "How do I trade options?"
    results = hybrid.search(test_query, k=3)
    
    if results:
        print(f"   Query: '{test_query}'")
        print(f"   Found {len(results)} results:")
        for i, r in enumerate(results, 1):
            title = r.get("metadata", {}).get("title", "Unknown")
            score = r.get("rrf_score", 0)
            print(f"     {i}. {title} (score: {score:.4f})")
    else:
        print("   WARNING: No results found for test query!")
    
    print("\n" + "=" * 50)
    print("Index building complete!")
    print("=" * 50)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
