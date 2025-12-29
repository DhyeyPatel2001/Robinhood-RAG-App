"""
Keyword store using BM25 for lexical search.

Provides keyword-based retrieval using the BM25 algorithm.
"""

import os
import pickle
from typing import Optional

import nltk
from rank_bm25 import BM25Okapi

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import get_bm25_path
from src.processing.chunker import DocumentChunk


# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class KeywordStore:
    """BM25-based keyword store for lexical search."""
    
    INDEX_FILENAME = "bm25_index.pkl"
    
    def __init__(self, persist_dir: Optional[str] = None):
        """Initialize the keyword store.
        
        Args:
            persist_dir: Directory for index persistence
        """
        self.persist_dir = persist_dir or get_bm25_path()
        self.bm25: Optional[BM25Okapi] = None
        self.documents: list[dict] = []  # Store doc info for retrieval
        self.tokenized_corpus: list[list[str]] = []
        
        # Try to load existing index
        self._load()
    
    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Lowercase and tokenize
        tokens = nltk.word_tokenize(text.lower())
        
        # Remove short tokens and non-alphanumeric
        tokens = [t for t in tokens if len(t) > 2 and t.isalnum()]
        
        return tokens
    
    def add_documents(self, chunks: list[DocumentChunk]):
        """Add document chunks to the keyword index.
        
        Args:
            chunks: List of DocumentChunk objects
        """
        if not chunks:
            print("No chunks to add")
            return
        
        print(f"Building BM25 index with {len(chunks)} chunks...")
        
        # Store document info
        self.documents = [
            {
                "id": chunk.chunk_id,
                "content": chunk.content,
                "source_url": chunk.source_url,
                "title": chunk.title,
                "category": chunk.category,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
            }
            for chunk in chunks
        ]
        
        # Tokenize corpus
        self.tokenized_corpus = [
            self._tokenize(chunk.content)
            for chunk in chunks
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print(f"BM25 index built with {len(self.documents)} documents")
        
        # Save index
        self._save()
    
    def search(self, query: str, k: int = 5) -> list[dict]:
        """Search for matching documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with score, content, and metadata
        """
        if not self.bm25 or not self.documents:
            print("Index not built. Call add_documents first.")
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k results
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]
        
        # Format results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include docs with positive score
                doc = self.documents[idx]
                results.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "score": float(scores[idx]),
                    "metadata": {
                        "source_url": doc["source_url"],
                        "title": doc["title"],
                        "category": doc["category"],
                        "chunk_index": doc["chunk_index"],
                        "total_chunks": doc["total_chunks"],
                    },
                })
        
        return results
    
    def _save(self):
        """Save index to disk."""
        index_path = os.path.join(self.persist_dir, self.INDEX_FILENAME)
        
        data = {
            "documents": self.documents,
            "tokenized_corpus": self.tokenized_corpus,
        }
        
        with open(index_path, "wb") as f:
            pickle.dump(data, f)
        
        print(f"BM25 index saved to {index_path}")
    
    def _load(self):
        """Load index from disk."""
        index_path = os.path.join(self.persist_dir, self.INDEX_FILENAME)
        
        if not os.path.exists(index_path):
            return
        
        try:
            with open(index_path, "rb") as f:
                data = pickle.load(f)
            
            self.documents = data["documents"]
            self.tokenized_corpus = data["tokenized_corpus"]
            
            if self.tokenized_corpus:
                self.bm25 = BM25Okapi(self.tokenized_corpus)
                print(f"BM25 index loaded with {len(self.documents)} documents")
                
        except Exception as e:
            print(f"Error loading BM25 index: {e}")
    
    def get_stats(self) -> dict:
        """Get index statistics.
        
        Returns:
            Dictionary with index stats
        """
        return {
            "document_count": len(self.documents),
            "persist_dir": self.persist_dir,
            "index_loaded": self.bm25 is not None,
        }
    
    def clear(self):
        """Clear the index."""
        self.bm25 = None
        self.documents = []
        self.tokenized_corpus = []
        
        index_path = os.path.join(self.persist_dir, self.INDEX_FILENAME)
        if os.path.exists(index_path):
            os.remove(index_path)
        
        print("Keyword store cleared")


def main():
    """Test keyword store functionality."""
    from src.processing.chunker import DocumentChunker
    
    # Load chunks
    chunker = DocumentChunker()
    chunks = chunker.load_chunks()
    
    if not chunks:
        print("No chunks found. Run chunker first.")
        return
    
    # Initialize keyword store
    store = KeywordStore()
    
    # Add chunks if empty
    if not store.documents:
        store.add_documents(chunks)
    
    # Test search
    print("\nTesting search...")
    results = store.search("options trading margin", k=3)
    
    for i, result in enumerate(results):
        print(f"\n{i + 1}. Score: {result['score']:.4f}")
        print(f"   Title: {result['metadata'].get('title', 'N/A')}")
        print(f"   Content: {result['content'][:150]}...")
    
    print("\nStats:", store.get_stats())


if __name__ == "__main__":
    main()
