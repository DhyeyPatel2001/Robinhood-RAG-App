"""
Vector store using ChromaDB for semantic search.

Provides embedding-based similarity search over document chunks.
"""

import os
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import settings, get_chroma_path
from src.processing.chunker import DocumentChunk


class VectorStore:
    """ChromaDB-based vector store for semantic search."""
    
    COLLECTION_NAME = "robinhood_support"
    
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """Initialize the vector store.
        
        Args:
            persist_dir: Directory for ChromaDB persistence
            embedding_model: Sentence transformer model name
        """
        self.persist_dir = persist_dir or get_chroma_path()
        self.embedding_model_name = embedding_model or settings.EMBEDDING_MODEL
        
        # Initialize embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(
                anonymized_telemetry=False,
            ),
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        
        print(f"Vector store initialized with {self.collection.count()} documents")
    
    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()
    
    def add_documents(self, chunks: list[DocumentChunk], batch_size: int = 100):
        """Add document chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects
            batch_size: Number of chunks to add per batch
        """
        if not chunks:
            print("No chunks to add")
            return
        
        print(f"Adding {len(chunks)} chunks to vector store...")
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Prepare data
            ids = [chunk.chunk_id for chunk in batch]
            documents = [chunk.content for chunk in batch]
            metadatas = [
                {
                    "source_url": chunk.source_url,
                    "title": chunk.title,
                    "category": chunk.category,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                }
                for chunk in batch
            ]
            
            # Generate embeddings
            embeddings = self._embed(documents)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            
            print(f"  Added batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}")
        
        print(f"Vector store now contains {self.collection.count()} documents")
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_category: Optional[str] = None,
    ) -> list[dict]:
        """Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_category: Optional category filter
            
        Returns:
            List of search results with score, content, and metadata
        """
        # Generate query embedding
        query_embedding = self._embed([query])[0]
        
        # Build filter
        where_filter = None
        if filter_category:
            where_filter = {"category": filter_category}
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        
        # Format results
        formatted_results = []
        
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                # Convert distance to similarity score (cosine distance -> similarity)
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance  # Higher is better
                
                formatted_results.append({
                    "id": results["ids"][0][i] if results["ids"] else None,
                    "content": doc,
                    "score": score,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })
        
        return formatted_results
    
    def get_stats(self) -> dict:
        """Get collection statistics.
        
        Returns:
            Dictionary with collection stats
        """
        return {
            "name": self.COLLECTION_NAME,
            "count": self.collection.count(),
            "embedding_model": self.embedding_model_name,
            "persist_dir": self.persist_dir,
        }
    
    def clear(self):
        """Clear all documents from the collection."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("Vector store cleared")


def main():
    """Test vector store functionality."""
    from src.processing.chunker import DocumentChunker
    
    # Load chunks
    chunker = DocumentChunker()
    chunks = chunker.load_chunks()
    
    if not chunks:
        print("No chunks found. Run chunker first.")
        return
    
    # Initialize vector store
    store = VectorStore()
    
    # Add chunks if empty
    if store.collection.count() == 0:
        store.add_documents(chunks)
    
    # Test search
    print("\nTesting search...")
    results = store.search("How do I trade options?", k=3)
    
    for i, result in enumerate(results):
        print(f"\n{i + 1}. Score: {result['score']:.4f}")
        print(f"   Title: {result['metadata'].get('title', 'N/A')}")
        print(f"   Content: {result['content'][:150]}...")
    
    print("\nStats:", store.get_stats())


if __name__ == "__main__":
    main()
