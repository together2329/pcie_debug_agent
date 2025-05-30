"""Output formatting utilities for CLI"""

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.panel import Panel
from rich.markdown import Markdown
from rich import box
from typing import List, Dict, Any, Optional
import json
import yaml


# Global console instance
console = Console()


def print_banner():
    """Print CLI banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                  PCIe Debug Agent v1.0.0                  ║
    ║           AI-Powered PCIe Log Analysis Tool               ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")


def print_success(message: str):
    """Print success message"""
    console.print(f"✅ {message}", style="green")


def print_error(message: str):
    """Print error message"""
    console.print(f"❌ {message}", style="red")


def print_warning(message: str):
    """Print warning message"""
    console.print(f"⚠️  {message}", style="yellow")


def print_info(message: str):
    """Print info message"""
    console.print(f"ℹ️  {message}", style="blue")


def create_progress_bar(description: str = "Processing") -> Progress:
    """Create a progress bar"""
    return Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        TimeElapsedColumn(),
        console=console
    )


def format_json(data: Dict[str, Any], indent: int = 2) -> str:
    """Format data as JSON"""
    return json.dumps(data, indent=indent, sort_keys=True)


def format_yaml(data: Dict[str, Any]) -> str:
    """Format data as YAML"""
    return yaml.dump(data, default_flow_style=False, sort_keys=True)


def print_json(data: Dict[str, Any], title: Optional[str] = None):
    """Print formatted JSON"""
    json_str = format_json(data)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
    
    if title:
        panel = Panel(syntax, title=title, box=box.ROUNDED)
        console.print(panel)
    else:
        console.print(syntax)


def print_yaml(data: Dict[str, Any], title: Optional[str] = None):
    """Print formatted YAML"""
    yaml_str = format_yaml(data)
    syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)
    
    if title:
        panel = Panel(syntax, title=title, box=box.ROUNDED)
        console.print(panel)
    else:
        console.print(syntax)


def print_table(
    headers: List[str],
    rows: List[List[str]],
    title: Optional[str] = None
):
    """Print formatted table"""
    table = Table(title=title, box=box.ROUNDED)
    
    # Add headers
    for header in headers:
        table.add_column(header, style="cyan", no_wrap=True)
    
    # Add rows
    for row in rows:
        table.add_row(*[str(cell) for cell in row])
    
    console.print(table)


def print_search_results(results: List[Dict[str, Any]]):
    """Print formatted search results"""
    if not results:
        print_warning("No results found")
        return
    
    console.print(f"\n[bold]Found {len(results)} results:[/bold]\n")
    
    for i, result in enumerate(results, 1):
        # Result header
        console.print(f"[bold cyan]Result {i}[/bold cyan]", style="bold")
        console.print(f"Score: [green]{result.get('score', 0):.3f}[/green]")
        
        # Source info
        if metadata := result.get('metadata', {}):
            source = metadata.get('source', 'Unknown')
            line = metadata.get('line', '')
            console.print(f"Source: [blue]{source}[/blue]" + 
                         (f" (line {line})" if line else ""))
        
        # Content
        content = result.get('content', '')
        if len(content) > 200:
            content = content[:200] + "..."
        
        content_panel = Panel(
            content,
            title="Content",
            box=box.ROUNDED,
            border_style="dim"
        )
        console.print(content_panel)
        console.print()  # Empty line between results


def print_analysis_result(result: Dict[str, Any]):
    """Print formatted analysis result"""
    # Answer
    answer_panel = Panel(
        Markdown(result.get('answer', 'No answer available')),
        title="Analysis Result",
        box=box.DOUBLE,
        border_style="green"
    )
    console.print(answer_panel)
    
    # Confidence
    confidence = result.get('confidence', 0)
    console.print(f"\nConfidence: [{'green' if confidence > 0.8 else 'yellow'}]"
                 f"{confidence:.1%}[/]")
    
    # Sources
    if sources := result.get('sources', []):
        console.print("\n[bold]Sources:[/bold]")
        for source in sources:
            console.print(f"  • {source.get('file', 'Unknown')} "
                         f"(relevance: {source.get('relevance', 0):.2f})")
    
    # Metadata
    if metadata := result.get('metadata', {}):
        console.print(f"\n[dim]Response time: {metadata.get('response_time', 0):.2f}s[/dim]")
        console.print(f"[dim]Model: {metadata.get('model', 'Unknown')}[/dim]")


def print_code(code: str, language: str = "python", title: Optional[str] = None):
    """Print formatted code"""
    syntax = Syntax(
        code,
        language,
        theme="monokai",
        line_numbers=True,
        code_width=80
    )
    
    if title:
        panel = Panel(syntax, title=title, box=box.ROUNDED)
        console.print(panel)
    else:
        console.print(syntax)


def print_log_entry(entry: Dict[str, Any]):
    """Print formatted log entry"""
    timestamp = entry.get('timestamp', '')
    level = entry.get('level', 'INFO')
    message = entry.get('message', '')
    
    # Color based on level
    level_colors = {
        'ERROR': 'red',
        'WARNING': 'yellow',
        'INFO': 'blue',
        'DEBUG': 'dim'
    }
    color = level_colors.get(level, 'white')
    
    console.print(f"[{color}][{timestamp}] {level}: {message}[/{color}]")


def confirm(message: str, default: bool = False) -> bool:
    """Ask for user confirmation"""
    suffix = " [Y/n]" if default else " [y/N]"
    response = console.input(f"{message}{suffix}: ").lower().strip()
    
    if not response:
        return default
    
    return response in ['y', 'yes']


def select_option(
    message: str,
    options: List[str],
    default: Optional[int] = None
) -> int:
    """Select from a list of options"""
    console.print(f"\n{message}")
    
    for i, option in enumerate(options, 1):
        marker = " [default]" if default and i == default else ""
        console.print(f"  {i}. {option}{marker}")
    
    while True:
        response = console.input("\nSelect option: ").strip()
        
        if not response and default:
            return default - 1
        
        try:
            choice = int(response)
            if 1 <= choice <= len(options):
                return choice - 1
            else:
                print_error(f"Please select a number between 1 and {len(options)}")
        except ValueError:
            print_error("Please enter a valid number")


def print_stats(stats: Dict[str, Any], title: str = "Statistics"):
    """Print formatted statistics"""
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in stats.items():
        # Format key
        formatted_key = key.replace('_', ' ').title()
        
        # Format value
        if isinstance(value, float):
            formatted_value = f"{value:.2f}"
        elif isinstance(value, int) and value > 1000:
            formatted_value = f"{value:,}"
        else:
            formatted_value = str(value)
        
        table.add_row(formatted_key, formatted_value)
    
    console.print(table)