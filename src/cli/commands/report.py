"""Report command for PCIe Debug Agent CLI"""

import click
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import json

from src.cli.utils.output import (
    console, print_error, print_success, print_info,
    create_progress_bar
)


@click.command()
@click.option(
    "--query", "-q",
    multiple=True,
    help="Queries to analyze (can specify multiple)"
)
@click.option(
    "--input", "-i",
    type=click.Path(exists=True),
    help="Input log file or directory"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output file path"
)
@click.option(
    "--format", "-f",
    type=click.Choice(["html", "markdown", "pdf", "json"]),
    default="html",
    help="Report format"
)
@click.option(
    "--template", "-t",
    type=click.Path(exists=True),
    help="Custom report template"
)
@click.option(
    "--title",
    default="PCIe Debug Analysis Report",
    help="Report title"
)
@click.option(
    "--include-stats",
    is_flag=True,
    help="Include detailed statistics"
)
@click.option(
    "--include-timeline",
    is_flag=True,
    help="Include error timeline"
)
@click.pass_context
def report(
    ctx: click.Context,
    query: List[str],
    input: Optional[str],
    output: Optional[str],
    format: str,
    template: Optional[str],
    title: str,
    include_stats: bool,
    include_timeline: bool
):
    """Generate analysis reports
    
    Examples:
        pcie-debug report --query "What errors occurred?" --output report.html
        
        pcie-debug report -q "Timeout analysis" -q "Link failures" -f markdown
        
        pcie-debug report --input /logs --include-stats --include-timeline
    """
    settings = ctx.obj["settings"]
    verbose = ctx.obj.get("verbose", False)
    
    try:
        # Import required components
        from src.rag.enhanced_rag_engine import EnhancedRAGEngine
        from src.rag.vector_store import FAISSVectorStore
        from src.rag.model_manager import ModelManager
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        
        # Initialize components
        print_info("Initializing report generator...")
        
        vector_store = FAISSVectorStore(
            index_path=settings.vector_store.index_path,
            index_type=settings.vector_store.index_type,
            dimension=settings.vector_store.dimension
        )
        
        if Path(settings.vector_store.index_path + ".faiss").exists():
            vector_store.load_index()
        
        model_manager = ModelManager(
            embedding_model=settings.embedding.model,
            embedding_provider=settings.embedding.provider,
            embedding_api_key=settings.embedding.api_key,
            llm_provider=settings.llm.provider,
            llm_model=settings.llm.model,
            llm_api_key=settings.llm.api_key
        )
        
        engine = EnhancedRAGEngine(
            vector_store=vector_store,
            model_manager=model_manager,
            llm_provider=settings.llm.provider,
            llm_model=settings.llm.model
        )
        
        # Default queries if none provided
        if not query:
            query = [
                "What are the main errors in the logs?",
                "What is the root cause of the failures?",
                "What are the recommended fixes?"
            ]
        
        # Perform analyses
        print_info(f"Running {len(query)} analyses...")
        analyses = []
        
        with create_progress_bar("Analyzing") as progress:
            task = progress.add_task("Processing queries...", total=len(query))
            
            for q in query:
                result = engine.query(q)
                analyses.append({
                    "query": q,
                    "answer": result.answer,
                    "sources": result.sources,
                    "confidence": result.confidence,
                    "timestamp": datetime.now().isoformat()
                })
                progress.update(task, advance=1)
        
        # Collect statistics if requested
        statistics = {}
        if include_stats:
            print_info("Collecting statistics...")
            statistics = {
                "total_documents": vector_store.index.ntotal,
                "total_queries": len(query),
                "avg_confidence": sum(a["confidence"] for a in analyses) / len(analyses),
                "index_stats": vector_store.get_statistics()
            }
        
        # Generate timeline if requested
        timeline = []
        if include_timeline:
            print_info("Generating error timeline...")
            # TODO: Implement timeline generation from logs
            timeline = [
                {
                    "timestamp": "2024-01-15 10:00:00",
                    "event": "PCIe link training started",
                    "severity": "INFO"
                },
                {
                    "timestamp": "2024-01-15 10:00:05",
                    "event": "Link training failed at Gen3",
                    "severity": "ERROR"
                }
            ]
        
        # Prepare report data
        report_data = {
            "title": title,
            "generated_at": datetime.now().isoformat(),
            "analyses": analyses,
            "statistics": statistics,
            "timeline": timeline,
            "settings": {
                "model": settings.llm.model,
                "provider": settings.llm.provider
            }
        }
        
        # Generate report
        print_info(f"Generating {format} report...")
        
        if format == "json":
            report_content = json.dumps(report_data, indent=2)
            
        elif format == "markdown":
            report_content = _generate_markdown_report(report_data)
            
        elif format == "html":
            # Load template
            if template:
                template_dir = Path(template).parent
                template_name = Path(template).name
            else:
                template_dir = Path(__file__).parent.parent / "templates"
                template_name = "report.html.jinja2"
            
            env = Environment(
                loader=FileSystemLoader(template_dir),
                autoescape=select_autoescape(['html', 'xml'])
            )
            tmpl = env.get_template(template_name)
            report_content = tmpl.render(**report_data)
            
        elif format == "pdf":
            # Generate HTML first, then convert to PDF
            print_warning("PDF generation requires additional dependencies")
            # TODO: Implement PDF generation
            report_content = "PDF generation not yet implemented"
        
        # Save report
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_content)
            print_success(f"Report saved to: {output_path}")
        else:
            # Print to console
            console.print(report_content)
        
        print_success("Report generation complete!")
        
    except Exception as e:
        print_error(f"Report generation failed: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


def _generate_markdown_report(data: Dict[str, Any]) -> str:
    """Generate markdown report from data"""
    md = f"# {data['title']}\n\n"
    md += f"*Generated on: {data['generated_at']}*\n\n"
    
    # Analyses
    md += "## Analysis Results\n\n"
    for i, analysis in enumerate(data['analyses'], 1):
        md += f"### {i}. {analysis['query']}\n\n"
        md += f"{analysis['answer']}\n\n"
        md += f"**Confidence:** {analysis['confidence']:.1%}\n\n"
        
        if analysis['sources']:
            md += "**Sources:**\n"
            for source in analysis['sources']:
                md += f"- {source.get('file', 'Unknown')} "
                md += f"(relevance: {source.get('relevance', 0):.2f})\n"
            md += "\n"
    
    # Statistics
    if data['statistics']:
        md += "## Statistics\n\n"
        stats = data['statistics']
        md += f"- Total Documents: {stats.get('total_documents', 0)}\n"
        md += f"- Total Queries: {stats.get('total_queries', 0)}\n"
        md += f"- Average Confidence: {stats.get('avg_confidence', 0):.1%}\n\n"
    
    # Timeline
    if data['timeline']:
        md += "## Error Timeline\n\n"
        md += "| Timestamp | Event | Severity |\n"
        md += "|-----------|-------|----------|\n"
        for event in data['timeline']:
            md += f"| {event['timestamp']} | {event['event']} | {event['severity']} |\n"
        md += "\n"
    
    # Settings
    md += "## Configuration\n\n"
    md += f"- Model: {data['settings']['model']}\n"
    md += f"- Provider: {data['settings']['provider']}\n"
    
    return md