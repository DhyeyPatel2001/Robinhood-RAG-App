"""
Text chunking for RAG using LangChain.

Loads scraped markdown documents and splits them into
fixed-size chunks with overlap for retrieval.
"""

import json
import os
import re
from typing import Optional
from dataclasses import dataclass, asdict

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import settings, get_raw_data_path, get_processed_data_path


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    
    chunk_id: str
    content: str
    source_url: str
    title: str
    category: str
    chunk_index: int
    total_chunks: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class DocumentChunker:
    """Chunks documents using LangChain text splitters."""
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        raw_dir: Optional[str] = None,
        processed_dir: Optional[str] = None,
    ):
        """Initialize the chunker.
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            raw_dir: Directory containing raw scraped markdown files
            processed_dir: Directory to save processed chunks
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.raw_dir = raw_dir or get_raw_data_path()
        self.processed_dir = processed_dir or get_processed_data_path()
        
        # Initialize text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False,
        )
    
    def _parse_frontmatter(self, content: str) -> tuple[dict, str]:
        """Parse YAML frontmatter from markdown content.
        
        Args:
            content: Full markdown content with frontmatter
            
        Returns:
            Tuple of (metadata dict, content without frontmatter)
        """
        metadata = {}
        body = content
        
        if content.startswith("---"):
            # Find the closing ---
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    metadata = yaml.safe_load(parts[1])
                    body = parts[2].strip()
                except yaml.YAMLError:
                    pass
        
        return metadata, body
    
    def _clean_content(self, content: str) -> str:
        """Clean markdown content for better chunking.
        
        Args:
            content: Raw markdown content
            
        Returns:
            Cleaned content
        """
        # Remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Remove CSS/style artifacts
        content = re.sub(r'\.css-[a-zA-Z0-9\-]+\{[^}]+\}', '', content)
        
        # Remove image markdown that's broken
        content = re.sub(r'!\[\]\([^)]*\)', '', content)
        
        # Remove empty links
        content = re.sub(r'\[\]\([^)]*\)', '', content)
        
        # Normalize whitespace
        content = content.strip()
        
        return content
    
    def load_documents(self) -> list[tuple[dict, str]]:
        """Load all scraped documents.
        
        Returns:
            List of (metadata, content) tuples
        """
        documents = []
        
        if not os.path.exists(self.raw_dir):
            print(f"Raw data directory not found: {self.raw_dir}")
            return documents
        
        for filename in os.listdir(self.raw_dir):
            if not filename.endswith(".md") or filename.startswith("_"):
                continue
            
            filepath = os.path.join(self.raw_dir, filename)
            
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                
                metadata, body = self._parse_frontmatter(content)
                body = self._clean_content(body)
                
                if len(body) > 50:  # Skip very short documents
                    documents.append((metadata, body))
                    
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        print(f"Loaded {len(documents)} documents from {self.raw_dir}")
        return documents
    
    def chunk_document(
        self,
        content: str,
        metadata: dict,
    ) -> list[DocumentChunk]:
        """Chunk a single document.
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            List of DocumentChunk objects
        """
        # Split content into chunks
        chunks = self.splitter.split_text(content)
        
        # Create chunk objects
        source_url = metadata.get("url", "unknown")
        title = metadata.get("title", "Untitled")
        category = metadata.get("category", "general")
        
        document_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk = DocumentChunk(
                chunk_id=f"{hash(source_url)}_{i}",
                content=chunk_text,
                source_url=source_url,
                title=title,
                category=category,
                chunk_index=i,
                total_chunks=len(chunks),
            )
            document_chunks.append(chunk)
        
        return document_chunks
    
    def process_all(self, save: bool = True) -> list[DocumentChunk]:
        """Process all documents and create chunks.
        
        Args:
            save: Whether to save chunks to disk
            
        Returns:
            List of all DocumentChunk objects
        """
        documents = self.load_documents()
        
        if not documents:
            print("No documents to process")
            return []
        
        all_chunks = []
        
        for metadata, content in documents:
            chunks = self.chunk_document(content, metadata)
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        print(f"Average chunks per document: {len(all_chunks) / len(documents):.1f}")
        
        if save:
            self.save_chunks(all_chunks)
        
        return all_chunks
    
    def save_chunks(self, chunks: list[DocumentChunk]):
        """Save chunks to JSON file.
        
        Args:
            chunks: List of DocumentChunk objects
        """
        output_path = os.path.join(self.processed_dir, "chunks.json")
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                [chunk.to_dict() for chunk in chunks],
                f,
                indent=2,
                ensure_ascii=False,
            )
        
        print(f"Saved {len(chunks)} chunks to {output_path}")
    
    def load_chunks(self) -> list[DocumentChunk]:
        """Load chunks from JSON file.
        
        Returns:
            List of DocumentChunk objects
        """
        input_path = os.path.join(self.processed_dir, "chunks.json")
        
        if not os.path.exists(input_path):
            print(f"No chunks file found at {input_path}")
            return []
        
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        chunks = [DocumentChunk(**item) for item in data]
        print(f"Loaded {len(chunks)} chunks from {input_path}")
        
        return chunks


def main():
    """Main entry point for chunking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chunk scraped documents")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap")
    parser.add_argument("--raw-dir", type=str, default=None, help="Raw data directory")
    parser.add_argument("--processed-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--no-save", action="store_true", help="Don't save to disk")
    
    args = parser.parse_args()
    
    chunker = DocumentChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
    )
    
    chunks = chunker.process_all(save=not args.no_save)
    
    # Print sample chunks
    if chunks:
        print("\nSample chunk:")
        print("-" * 50)
        sample = chunks[0]
        print(f"Title: {sample.title}")
        print(f"Category: {sample.category}")
        print(f"Chunk {sample.chunk_index + 1}/{sample.total_chunks}")
        print(f"Content preview: {sample.content[:200]}...")


if __name__ == "__main__":
    main()
