#!/usr/bin/env python3
"""
Interactive search interface for QdrantRAG-Pro.

This script provides a command-line interface for searching documents
and generating responses using the hybrid search and RAG capabilities.
"""

import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config.settings import Settings
from core.database.qdrant_client import QdrantManager
from core.database.document_store import DocumentStore
from core.services.embedding_service import EmbeddingService
from core.services.search_engine import HybridSearchEngine
from core.services.response_generator import ResponseGenerator
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.markdown import Markdown


class InteractiveSearchInterface:
    """Interactive search interface with rich console output."""
    
    def __init__(self):
        """Initialize the search interface."""
        self.console = Console()
        self.settings = Settings()
        
        # Initialize services
        self.qdrant_manager = QdrantManager(self.settings)
        self.embedding_service = EmbeddingService(self.settings)
        self.document_store = DocumentStore(self.qdrant_manager, self.settings)
        self.search_engine = HybridSearchEngine(
            self.qdrant_manager, 
            self.embedding_service, 
            self.settings
        )
        self.response_generator = ResponseGenerator(self.settings)
        
        # Search configuration
        self.current_filters = {}
        self.search_limit = 10
        self.vector_weight = self.settings.default_vector_weight
        self.keyword_weight = self.settings.default_keyword_weight
    
    def display_welcome(self):
        """Display welcome message and system status."""
        self.console.print(Panel.fit(
            "[bold blue]QdrantRAG-Pro Interactive Search[/bold blue]\n"
            "Advanced hybrid search with intelligent response generation\n\n"
            "Commands:\n"
            "• [cyan]search <query>[/cyan] - Search documents\n"
            "• [cyan]ask <question>[/cyan] - Get AI-generated answers\n"
            "• [cyan]config[/cyan] - Configure search settings\n"
            "• [cyan]stats[/cyan] - Show database statistics\n"
            "• [cyan]help[/cyan] - Show detailed help\n"
            "• [cyan]quit[/cyan] - Exit the application",
            border_style="blue"
        ))
    
    async def check_system_status(self) -> bool:
        """Check if all systems are ready."""
        try:
            # Check Qdrant connection
            if not self.qdrant_manager.health_check():
                self.console.print("[red]❌ Cannot connect to Qdrant database[/red]")
                return False
            
            # Check collection
            collection_info = self.qdrant_manager.get_collection_info()
            if not collection_info:
                self.console.print("[red]❌ Collection not found[/red]")
                self.console.print("Please run: python scripts/setup_database.py")
                return False
            
            if collection_info.vectors_count == 0:
                self.console.print("[yellow]⚠️  No documents found in database[/yellow]")
                self.console.print("Please run: python scripts/ingest_documents.py --create-sample")
                return False
            
            # Display status
            status_table = Table(title="System Status")
            status_table.add_column("Component", style="cyan")
            status_table.add_column("Status", style="green")
            
            status_table.add_row("Qdrant Database", "✅ Connected")
            status_table.add_row("Collection", f"✅ Ready ({collection_info.vectors_count} documents)")
            status_table.add_row("Embedding Service", "✅ Ready")
            status_table.add_row("Search Engine", "✅ Ready")
            status_table.add_row("Response Generator", "✅ Ready")
            
            self.console.print(status_table)
            return True
            
        except Exception as e:
            self.console.print(f"[red]❌ System check failed: {e}[/red]")
            return False
    
    async def search_documents(self, query: str) -> None:
        """Search documents and display results."""
        try:
            self.console.print(f"\n🔍 Searching for: [bold]{query}[/bold]")
            
            # Perform search
            results = await self.search_engine.search(
                query=query,
                limit=self.search_limit,
                vector_weight=self.vector_weight,
                keyword_weight=self.keyword_weight,
                filters=self.current_filters if self.current_filters else None
            )
            
            if not results:
                self.console.print("[yellow]No results found[/yellow]")
                return
            
            # Display results
            self.console.print(f"\n📊 Found {len(results)} results:")
            
            for i, result in enumerate(results, 1):
                # Create result panel
                metadata_items = []
                for key, value in result.metadata.items():
                    if key in ["title", "author", "category", "source"]:
                        metadata_items.append(f"{key}: {value}")
                
                metadata_str = " | ".join(metadata_items) if metadata_items else "No metadata"
                
                content_preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
                
                panel_content = f"""[bold]Content:[/bold]
{content_preview}

[bold]Metadata:[/bold] {metadata_str}
[bold]Score:[/bold] {result.combined_score:.3f} | [bold]Type:[/bold] {result.result_type}
[bold]Explanation:[/bold] {result.explanation}"""
                
                self.console.print(Panel(
                    panel_content,
                    title=f"Result {i}",
                    border_style="green" if i <= 3 else "blue"
                ))
            
        except Exception as e:
            self.console.print(f"[red]❌ Search failed: {e}[/red]")
    
    async def ask_question(self, question: str) -> None:
        """Ask a question and get an AI-generated response."""
        try:
            self.console.print(f"\n🤔 Question: [bold]{question}[/bold]")
            self.console.print("🔍 Searching for relevant information...")
            
            # Search for relevant documents
            search_results = await self.search_engine.search(
                query=question,
                limit=self.settings.max_sources_per_response,
                vector_weight=self.vector_weight,
                keyword_weight=self.keyword_weight,
                filters=self.current_filters if self.current_filters else None
            )
            
            if not search_results:
                self.console.print("[yellow]No relevant information found[/yellow]")
                return
            
            self.console.print("🧠 Generating response...")
            
            # Generate response
            response = await self.response_generator.generate_response(
                query=question,
                search_results=search_results
            )
            
            # Display response
            self.console.print("\n" + "="*80)
            self.console.print(Panel(
                Markdown(response.answer),
                title="🤖 AI Response",
                border_style="green"
            ))
            
            # Display analysis
            analysis_table = Table(title="Response Analysis")
            analysis_table.add_column("Metric", style="cyan")
            analysis_table.add_column("Value", style="green")
            
            analysis_table.add_row("Confidence", f"{response.confidence_score:.2f} ({response.confidence_level})")
            analysis_table.add_row("Source Coverage", f"{response.source_coverage:.2f}")
            analysis_table.add_row("Sources Used", str(len(response.sources_used)))
            analysis_table.add_row("Processing Time", f"{response.processing_time:.2f}s")
            
            if response.needs_review:
                analysis_table.add_row("Quality", "[yellow]Needs Review[/yellow]")
            else:
                analysis_table.add_row("Quality", "[green]Good[/green]")
            
            self.console.print(analysis_table)
            
            # Show reasoning if requested
            if Confirm.ask("Show detailed reasoning?", default=False):
                reasoning_content = "\n".join([f"{i+1}. {step}" for i, step in enumerate(response.reasoning_steps)])
                self.console.print(Panel(
                    reasoning_content,
                    title="🧠 Reasoning Steps",
                    border_style="blue"
                ))
            
            # Show sources if requested
            if Confirm.ask("Show source details?", default=False):
                for i, source_detail in enumerate(response.source_details, 1):
                    source_content = f"""ID: {source_detail['id']}
Type: {source_detail['type']}
Score: {source_detail['score']:.3f}"""
                    
                    if 'title' in source_detail:
                        source_content += f"\nTitle: {source_detail['title']}"
                    if 'author' in source_detail:
                        source_content += f"\nAuthor: {source_detail['author']}"
                    
                    self.console.print(Panel(
                        source_content,
                        title=f"Source {i}",
                        border_style="cyan"
                    ))
            
            # Show limitations if any
            if response.limitations:
                self.console.print(Panel(
                    response.limitations,
                    title="⚠️ Limitations",
                    border_style="yellow"
                ))
            
        except Exception as e:
            self.console.print(f"[red]❌ Question processing failed: {e}[/red]")

    def configure_search(self) -> None:
        """Configure search settings."""
        self.console.print("\n⚙️ Search Configuration")

        config_table = Table(title="Current Settings")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")

        config_table.add_row("Search Limit", str(self.search_limit))
        config_table.add_row("Vector Weight", f"{self.vector_weight:.2f}")
        config_table.add_row("Keyword Weight", f"{self.keyword_weight:.2f}")
        config_table.add_row("Active Filters", str(len(self.current_filters)))

        self.console.print(config_table)

        # Configuration options
        while True:
            choice = Prompt.ask(
                "\nWhat would you like to configure?",
                choices=["limit", "weights", "filters", "reset", "done"],
                default="done"
            )

            if choice == "limit":
                new_limit = Prompt.ask("Enter search limit", default=str(self.search_limit))
                try:
                    self.search_limit = int(new_limit)
                    self.console.print(f"✅ Search limit set to {self.search_limit}")
                except ValueError:
                    self.console.print("[red]❌ Invalid number[/red]")

            elif choice == "weights":
                self.console.print("Current weights: Vector={:.2f}, Keyword={:.2f}".format(
                    self.vector_weight, self.keyword_weight
                ))

                vector_weight = Prompt.ask("Enter vector weight (0.0-1.0)", default=str(self.vector_weight))
                try:
                    new_vector = float(vector_weight)
                    if 0 <= new_vector <= 1:
                        self.keyword_weight = 1.0 - new_vector
                        self.vector_weight = new_vector
                        self.console.print(f"✅ Weights updated: Vector={self.vector_weight:.2f}, Keyword={self.keyword_weight:.2f}")
                    else:
                        self.console.print("[red]❌ Weight must be between 0.0 and 1.0[/red]")
                except ValueError:
                    self.console.print("[red]❌ Invalid number[/red]")

            elif choice == "filters":
                self.configure_filters()

            elif choice == "reset":
                self.search_limit = 10
                self.vector_weight = self.settings.default_vector_weight
                self.keyword_weight = self.settings.default_keyword_weight
                self.current_filters = {}
                self.console.print("✅ Settings reset to defaults")

            elif choice == "done":
                break

    def configure_filters(self) -> None:
        """Configure search filters."""
        self.console.print("\n🔍 Filter Configuration")

        if self.current_filters:
            filter_table = Table(title="Active Filters")
            filter_table.add_column("Field", style="cyan")
            filter_table.add_column("Value", style="green")

            for key, value in self.current_filters.items():
                filter_table.add_row(key, str(value))

            self.console.print(filter_table)
        else:
            self.console.print("No active filters")

        while True:
            action = Prompt.ask(
                "Filter action",
                choices=["add", "remove", "clear", "done"],
                default="done"
            )

            if action == "add":
                field = Prompt.ask("Filter field (e.g., category, author, language)")
                value = Prompt.ask("Filter value")
                self.current_filters[field] = value
                self.console.print(f"✅ Added filter: {field} = {value}")

            elif action == "remove":
                if self.current_filters:
                    field = Prompt.ask("Field to remove", choices=list(self.current_filters.keys()))
                    del self.current_filters[field]
                    self.console.print(f"✅ Removed filter: {field}")
                else:
                    self.console.print("No filters to remove")

            elif action == "clear":
                self.current_filters = {}
                self.console.print("✅ All filters cleared")

            elif action == "done":
                break

    def show_statistics(self) -> None:
        """Show database and system statistics."""
        try:
            collection_info = self.qdrant_manager.get_collection_info()
            document_count = self.document_store.get_document_count()
            cache_stats = self.embedding_service.get_cache_stats()

            stats_table = Table(title="System Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green")

            stats_table.add_row("Total Vectors", str(collection_info.vectors_count))
            stats_table.add_row("Main Documents", str(document_count))
            stats_table.add_row("Vector Dimension", str(collection_info.config.params.vectors.size))
            stats_table.add_row("Distance Metric", str(collection_info.config.params.vectors.distance))

            if cache_stats.get("cache_enabled", False):
                stats_table.add_row("Cache Size", f"{cache_stats['size']}/{cache_stats['max_size']}")
                stats_table.add_row("Cache TTL", f"{cache_stats['ttl_hours']}h")
            else:
                stats_table.add_row("Cache", "Disabled")

            self.console.print(stats_table)

        except Exception as e:
            self.console.print(f"[red]❌ Failed to get statistics: {e}[/red]")

    def show_help(self) -> None:
        """Show detailed help information."""
        help_content = """
[bold blue]QdrantRAG-Pro Interactive Search Help[/bold blue]

[bold]Basic Commands:[/bold]
• [cyan]search <query>[/cyan] - Perform hybrid search on documents
• [cyan]ask <question>[/cyan] - Get AI-generated answers with sources
• [cyan]config[/cyan] - Configure search parameters and filters
• [cyan]stats[/cyan] - Show database and system statistics
• [cyan]help[/cyan] - Show this help message
• [cyan]quit[/cyan] or [cyan]exit[/cyan] - Exit the application

[bold]Search Features:[/bold]
• Hybrid search combines vector similarity and keyword matching
• Automatic query analysis adjusts search weights
• Advanced filtering by metadata fields
• Configurable result limits and scoring weights

[bold]AI Response Features:[/bold]
• Retrieval-augmented generation with source attribution
• Confidence scoring and quality analysis
• Step-by-step reasoning explanation
• Source coverage and limitation reporting

[bold]Configuration Options:[/bold]
• Search limit: Number of results to return (default: 10)
• Vector weight: Importance of semantic similarity (default: 0.7)
• Keyword weight: Importance of exact matches (default: 0.3)
• Filters: Restrict search by metadata fields

[bold]Examples:[/bold]
• search vector database performance
• ask How does Qdrant handle large datasets?
• config (then choose weights to adjust search behavior)

[bold]Tips:[/bold]
• Use technical terms for keyword-focused search
• Use natural language for semantic search
• Combine both for best results
• Configure filters to narrow search scope
"""

        self.console.print(Panel(help_content, border_style="blue"))

    async def run(self) -> None:
        """Run the interactive search interface."""
        self.display_welcome()

        # Check system status
        if not await self.check_system_status():
            return

        self.console.print("\n✅ System ready! Type 'help' for commands or start searching.")

        # Main interaction loop
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]qdrant-rag>[/bold cyan]").strip()

                if not user_input:
                    continue

                # Parse command
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if command in ["quit", "exit", "q"]:
                    self.console.print("👋 Goodbye!")
                    break

                elif command == "search":
                    if args:
                        await self.search_documents(args)
                    else:
                        self.console.print("[yellow]Please provide a search query[/yellow]")

                elif command == "ask":
                    if args:
                        await self.ask_question(args)
                    else:
                        self.console.print("[yellow]Please provide a question[/yellow]")

                elif command == "config":
                    self.configure_search()

                elif command == "stats":
                    self.show_statistics()

                elif command == "help":
                    self.show_help()

                else:
                    # Treat unknown commands as search queries
                    await self.search_documents(user_input)

            except KeyboardInterrupt:
                self.console.print("\n👋 Goodbye!")
                break
            except Exception as e:
                self.console.print(f"[red]❌ Error: {e}[/red]")


async def main():
    """Main function."""
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise in interactive mode
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and run interface
    interface = InteractiveSearchInterface()
    await interface.run()


if __name__ == "__main__":
    asyncio.run(main())
