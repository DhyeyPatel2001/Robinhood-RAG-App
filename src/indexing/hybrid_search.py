"""
Hybrid search combining vector and keyword search with RRF.

Uses Reciprocal Rank Fusion (RRF) to merge results from both
vector (semantic) and keyword (lexical) search.
"""

import os
from typing import Optional
from collections import defaultdict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import settings
from src.indexing.vector_store import VectorStore
from src.indexing.keyword_store import KeywordStore
from src.processing.chunker import DocumentChunk


class HybridSearch:
    """Hybrid search combining vector and keyword retrieval with RRF."""
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        keyword_store: Optional[KeywordStore] = None,
        rrf_k: Optional[int] = None,
    ):
        """Initialize hybrid search.
        
        Args:
            vector_store: VectorStore instance (creates new if None)
            keyword_store: KeywordStore instance (creates new if None)
            rrf_k: Constant k for RRF formula (default: 60)
        """
        self.vector_store = vector_store or VectorStore()
        self.keyword_store = keyword_store or KeywordStore()
        self.rrf_k = rrf_k or settings.RRF_K
    
    def _reciprocal_rank_fusion(
        self,
        result_lists: list[list[dict]],
        k: int = 60,
    ) -> list[dict]:
        """Merge multiple result lists using Reciprocal Rank Fusion.
        
        RRF score for document d:
        RRF(d) = Σ 1 / (k + rank(d))
        
        where k is a constant (typically 60) and rank is the position
        in each result list (1-indexed).
        
        Args:
            result_lists: List of result lists from different retrievers
            k: Constant k for RRF formula
            
        Returns:
            Merged and ranked results
        """
        # Track scores and document info
        rrf_scores: dict[str, float] = defaultdict(float)
        doc_info: dict[str, dict] = {}
        
        for results in result_lists:
            for rank, doc in enumerate(results, start=1):
                doc_id = doc["id"]
                
                # Calculate RRF contribution
                rrf_scores[doc_id] += 1.0 / (k + rank)
                
                # Store document info (use first occurrence)
                if doc_id not in doc_info:
                    doc_info[doc_id] = doc
        
        # Sort by RRF score (descending)
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )
        
        # Build final results
        merged_results = []
        for doc_id in sorted_ids:
            doc = doc_info[doc_id].copy()
            doc["rrf_score"] = rrf_scores[doc_id]
            merged_results.append(doc)
        
        return merged_results
    
    def search(
        self,
        query: str,
        k: int = 5,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        filter_category: Optional[str] = None,
    ) -> list[dict]:
        """Perform hybrid search.
        
        Args:
            query: Search query
            k: Number of final results to return
            vector_weight: Weight for vector search (for future weighted RRF)
            keyword_weight: Weight for keyword search (for future weighted RRF)
            filter_category: Optional category filter
            
        Returns:
            Merged search results ranked by RRF score
        """
        # Get more results from each retriever than final k
        retrieval_k = k * 3
        
        # Parallel search (could use asyncio for true parallelism)
        vector_results = self.vector_store.search(
            query=query,
            k=retrieval_k,
            filter_category=filter_category,
        )
        
        keyword_results = self.keyword_store.search(
            query=query,
            k=retrieval_k,
        )
        
        # Apply category filter to keyword results if specified
        if filter_category:
            keyword_results = [
                r for r in keyword_results
                if r["metadata"].get("category") == filter_category
            ]
        
        # Merge with RRF
        merged = self._reciprocal_rank_fusion(
            [vector_results, keyword_results],
            k=self.rrf_k,
        )
        
        # Return top k
        return merged[:k]
    
    def add_documents(self, chunks: list[DocumentChunk]):
        """Add documents to both stores.
        
        Args:
            chunks: List of DocumentChunk objects
        """
        print("Adding documents to vector store...")
        self.vector_store.add_documents(chunks)
        
        print("\nAdding documents to keyword store...")
        self.keyword_store.add_documents(chunks)
        
        print("\nHybrid search ready!")
    
    def get_stats(self) -> dict:
        """Get statistics from both stores.
        
        Returns:
            Combined statistics
        """
        return {
            "vector_store": self.vector_store.get_stats(),
            "keyword_store": self.keyword_store.get_stats(),
            "rrf_k": self.rrf_k,
        }
    
    def clear(self):
        """Clear both stores."""
        self.vector_store.clear()
        self.keyword_store.clear()
        print("Hybrid search stores cleared")


def main():
    """Test hybrid search functionality."""
    from src.processing.chunker import DocumentChunker
    
    # Load chunks
    chunker = DocumentChunker()
    chunks = chunker.load_chunks()
    
    if not chunks:
        print("No chunks found. Run chunker first.")
        return
    
    # Initialize hybrid search
    print("Initializing hybrid search...")
    hybrid = HybridSearch()
    
    # Check if we need to index
    stats = hybrid.get_stats()
    if stats["vector_store"]["count"] == 0:
        print("Indexing documents...")
        hybrid.add_documents(chunks)
    
    # Test search
    test_queries = [
        "How do I trade options?",
        "margin requirements",
        "transfer cryptocurrency",
    ]
    
    for query in test_queries:
        print(f"\n{'=' * 50}")
        print(f"Query: {query}")
        print("=" * 50)
        
        results = hybrid.search(query, k=3)
        
        for i, result in enumerate(results):
            print(f"\n{i + 1}. RRF Score: {result.get('rrf_score', 0):.4f}")
            print(f"   Title: {result['metadata'].get('title', 'N/A')}")
            print(f"   Category: {result['metadata'].get('category', 'N/A')}")
            print(f"   Content: {result['content'][:150]}...")
    
    print("\n\nStats:", hybrid.get_stats())


if __name__ == "__main__":
    main()
