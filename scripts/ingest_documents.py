#!/usr/bin/env python3
"""
Document ingestion script for QdrantRAG-Pro.

This script processes and ingests documents into the Qdrant vector database,
generating embeddings and handling various document formats.
"""

import sys
import os
import json
import asyncio
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
import time

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config.settings import Settings
from core.database.qdrant_client import QdrantManager
from core.database.document_store import DocumentStore
from core.services.embedding_service import EmbeddingService
from core.models.document import Document, DocumentMetadata, DocumentType
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table
from rich.panel import Panel


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ingest_documents.log')
        ]
    )


def load_documents_from_json(file_path: Path) -> List[Dict[str, Any]]:
    """Load documents from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both single document and array of documents
    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("JSON file must contain a document object or array of documents")


def load_documents_from_directory(directory: Path) -> List[Dict[str, Any]]:
    """Load documents from directory containing text files."""
    documents = []
    
    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.md', '.json']:
            try:
                if file_path.suffix.lower() == '.json':
                    docs = load_documents_from_json(file_path)
                    documents.extend(docs)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Create document metadata
                    doc_type = DocumentType.MARKDOWN if file_path.suffix.lower() == '.md' else DocumentType.TEXT
                    
                    documents.append({
                        "content": content,
                        "metadata": {
                            "title": file_path.stem,
                            "source": str(file_path),
                            "document_type": doc_type,
                            "file_size": file_path.stat().st_size
                        }
                    })
            except Exception as e:
                logging.warning(f"Failed to load {file_path}: {e}")
    
    return documents


def create_sample_documents() -> List[Dict[str, Any]]:
    """Create sample documents for testing."""
    return [
        {
            "content": "Qdrant is a vector database designed for high-performance similarity search. It provides advanced filtering capabilities and supports various distance metrics including cosine similarity, dot product, and Euclidean distance. The database is optimized for machine learning applications and can handle large-scale vector collections efficiently.",
            "metadata": {
                "title": "Introduction to Qdrant Vector Database",
                "author": "QdrantRAG Team",
                "category": "technology",
                "tags": ["vector-database", "similarity-search", "machine-learning"],
                "language": "en",
                "document_type": "text"
            }
        },
        {
            "content": "Retrieval-Augmented Generation (RAG) is a powerful technique that combines the strengths of retrieval-based and generation-based approaches in natural language processing. RAG systems first retrieve relevant documents from a knowledge base and then use this information to generate more accurate and contextually appropriate responses. This approach helps reduce hallucination in large language models.",
            "metadata": {
                "title": "Understanding RAG Systems",
                "author": "AI Research Team",
                "category": "artificial-intelligence",
                "tags": ["rag", "nlp", "language-models"],
                "language": "en",
                "document_type": "text"
            }
        },
        {
            "content": "Hybrid search combines the benefits of both semantic search (using vector embeddings) and traditional keyword search (using BM25 or TF-IDF). This approach provides better search results by capturing both semantic meaning and exact term matches. The key is to properly weight the contributions of each search method based on the query type and domain requirements.",
            "metadata": {
                "title": "Hybrid Search Strategies",
                "author": "Search Engineering Team",
                "category": "search-technology",
                "tags": ["hybrid-search", "semantic-search", "keyword-search"],
                "language": "en",
                "document_type": "text"
            }
        },
        {
            "content": "OpenAI's text-embedding-3-small model provides high-quality vector embeddings with 1536 dimensions. It offers excellent performance for most text similarity tasks while being cost-effective. The model supports various text lengths and provides consistent embeddings for similar content. It's particularly well-suited for RAG applications and semantic search systems.",
            "metadata": {
                "title": "OpenAI Embedding Models Guide",
                "author": "ML Engineering Team",
                "category": "machine-learning",
                "tags": ["openai", "embeddings", "text-processing"],
                "language": "en",
                "document_type": "text"
            }
        },
        {
            "content": "Production deployment of RAG systems requires careful consideration of scalability, latency, and cost optimization. Key factors include vector database configuration, embedding model selection, caching strategies, and response generation optimization. Monitoring and observability are crucial for maintaining system performance and quality in production environments.",
            "metadata": {
                "title": "Production RAG Deployment Best Practices",
                "author": "DevOps Team",
                "category": "deployment",
                "tags": ["production", "deployment", "scalability"],
                "language": "en",
                "document_type": "text"
            }
        }
    ]


async def ingest_documents_batch(
    documents: List[Document],
    document_store: DocumentStore,
    embedding_service: EmbeddingService,
    console: Console,
    progress: Progress,
    task_id: TaskID
) -> Dict[str, Any]:
    """Ingest a batch of documents."""
    start_time = time.time()
    
    # Generate embeddings for all documents
    contents = [doc.content for doc in documents]
    embedding_results = await embedding_service.create_embeddings_batch(contents)
    
    # Process documents with chunking if needed
    ingestion_results = []
    total_tokens = 0
    total_chunks = 0
    
    for i, (document, embedding_result) in enumerate(zip(documents, embedding_results)):
        # Generate chunks and chunk embeddings if document is large
        chunk_embeddings = None
        if len(document.content.split()) > 500:  # Chunk if more than 500 words
            chunks = embedding_service.chunk_text(document.content)
            if len(chunks) > 1:
                document.chunks = chunks
                chunk_embedding_results = await embedding_service.create_embeddings_batch(chunks)
                chunk_embeddings = [result.embedding for result in chunk_embedding_results]
        
        # Ingest document
        result = document_store.ingest_document(
            document=document,
            embedding=embedding_result.embedding,
            chunk_embeddings=chunk_embeddings
        )
        
        ingestion_results.append(result)
        total_tokens += embedding_result.token_count
        total_chunks += result.chunk_count or 0
        
        # Update progress
        progress.update(task_id, advance=1)
        
        if result.success:
            console.print(f"  ✅ {document.metadata.title or document.id}")
        else:
            console.print(f"  ❌ {document.metadata.title or document.id}: {result.message}")
    
    processing_time = time.time() - start_time
    successful = sum(1 for r in ingestion_results if r.success)
    
    return {
        "total": len(documents),
        "successful": successful,
        "failed": len(documents) - successful,
        "total_tokens": total_tokens,
        "total_chunks": total_chunks,
        "processing_time": processing_time,
        "results": ingestion_results
    }


async def main():
    """Main ingestion function."""
    parser = argparse.ArgumentParser(description="Ingest documents into QdrantRAG-Pro")
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Path to documents (file or directory)"
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample documents for testing"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of documents to process in each batch"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Initialize console
    console = Console()
    
    console.print(Panel.fit(
        "[bold blue]QdrantRAG-Pro Document Ingestion[/bold blue]\n"
        "Processing and ingesting documents into vector database...",
        border_style="blue"
    ))
    
    try:
        # Load settings
        console.print("📋 Loading configuration...")
        settings = Settings()
        
        # Initialize services
        console.print("🔌 Initializing services...")
        qdrant_manager = QdrantManager(settings)
        embedding_service = EmbeddingService(settings)
        document_store = DocumentStore(qdrant_manager, settings)
        
        # Health check
        if not qdrant_manager.health_check():
            console.print("[red]❌ Cannot connect to Qdrant database![/red]")
            console.print("Please run: python scripts/setup_database.py")
            return 1
        
        # Load documents
        console.print("📄 Loading documents...")
        
        if args.create_sample:
            console.print("Creating sample documents...")
            raw_documents = create_sample_documents()
        elif args.data_path:
            if args.data_path.is_file():
                raw_documents = load_documents_from_json(args.data_path)
            elif args.data_path.is_dir():
                raw_documents = load_documents_from_directory(args.data_path)
            else:
                console.print(f"[red]❌ Path not found: {args.data_path}[/red]")
                return 1
        else:
            console.print("[red]❌ Please specify --data-path or --create-sample[/red]")
            return 1
        
        if not raw_documents:
            console.print("[yellow]⚠️  No documents found to ingest[/yellow]")
            return 0
        
        # Convert to Document objects
        documents = []
        for doc_data in raw_documents:
            metadata = DocumentMetadata(**doc_data.get("metadata", {}))
            document = Document(
                content=doc_data["content"],
                metadata=metadata
            )
            documents.append(document)
        
        console.print(f"📊 Found {len(documents)} documents to process")
        
        # Estimate costs
        contents = [doc.content for doc in documents]
        cost_estimate = embedding_service.estimate_cost(contents)
        
        cost_table = Table(title="Processing Estimate")
        cost_table.add_column("Metric", style="cyan")
        cost_table.add_column("Value", style="green")
        
        cost_table.add_row("Documents", str(len(documents)))
        cost_table.add_row("Total Tokens", f"{cost_estimate['total_tokens']:,}")
        cost_table.add_row("Estimated Cost", f"${cost_estimate['estimated_cost_usd']:.4f}")
        cost_table.add_row("Batch Count", str(cost_estimate['batch_count']))
        
        console.print(cost_table)
        
        # Process documents
        console.print("\n🚀 Starting document ingestion...")
        
        with Progress(console=console) as progress:
            task = progress.add_task("Processing documents...", total=len(documents))
            
            # Process in batches
            batch_size = args.batch_size
            all_results = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                console.print(f"\n📦 Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                
                batch_results = await ingest_documents_batch(
                    batch, document_store, embedding_service, console, progress, task
                )
                all_results.append(batch_results)
        
        # Summary
        total_processed = sum(r["total"] for r in all_results)
        total_successful = sum(r["successful"] for r in all_results)
        total_failed = sum(r["failed"] for r in all_results)
        total_tokens = sum(r["total_tokens"] for r in all_results)
        total_chunks = sum(r["total_chunks"] for r in all_results)
        total_time = sum(r["processing_time"] for r in all_results)
        
        summary_table = Table(title="Ingestion Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Documents", str(total_processed))
        summary_table.add_row("Successful", str(total_successful))
        summary_table.add_row("Failed", str(total_failed))
        summary_table.add_row("Success Rate", f"{(total_successful/total_processed)*100:.1f}%")
        summary_table.add_row("Total Tokens", f"{total_tokens:,}")
        summary_table.add_row("Total Chunks", str(total_chunks))
        summary_table.add_row("Processing Time", f"{total_time:.2f}s")
        
        console.print(summary_table)
        
        if total_successful > 0:
            console.print(Panel.fit(
                "[bold green]🎉 Document ingestion completed![/bold green]\n"
                "You can now search documents with:\n"
                "[cyan]python scripts/interactive_search.py[/cyan]",
                border_style="green"
            ))
        
        logger.info(f"Ingestion completed: {total_successful}/{total_processed} documents successful")
        return 0 if total_failed == 0 else 1
        
    except Exception as e:
        console.print(f"\n[red]❌ Ingestion failed: {e}[/red]")
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
