"""Main entry point for PCIe Debug Agent CLI"""

import click
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.cli.commands import analyze, index, search, report, config, test, model
from src.cli.utils.output import console, print_banner
from src.config.settings import Settings, load_settings


@click.group()
@click.version_option(version="1.0.0", prog_name="PCIe Debug Agent")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="Path to configuration file"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output"
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Suppress non-essential output"
)
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], verbose: bool, quiet: bool):
    """PCIe Debug Agent - AI-powered PCIe log analysis tool
    
    Analyze PCIe debug logs and code using advanced RAG technology.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Print banner unless quiet mode
    if not quiet:
        print_banner()
    
    # Load configuration
    try:
        if config:
            settings = load_settings(Path(config))
        else:
            # Try default locations
            default_config = Path("configs/settings.yaml")
            if default_config.exists():
                settings = load_settings(default_config)
            else:
                settings = load_settings(None)  # Use defaults with env vars
        
        settings.validate()
        ctx.obj["settings"] = settings
        
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        sys.exit(1)
    
    # Set verbosity
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    
    # Initialize logging
    import logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)


# Register commands
cli.add_command(analyze.analyze)
cli.add_command(index.index)
cli.add_command(search.search)
cli.add_command(report.report)
cli.add_command(config.config)
cli.add_command(test.test)
cli.add_command(model.model)

# Import and add vectordb command
try:
    from src.cli.commands import vectordb
    cli.add_command(vectordb.vectordb)
except ImportError:
    pass  # vectordb command not available

# Import and add pcie command
try:
    from src.cli.commands.pcie_rag import pcie
    cli.add_command(pcie)
except ImportError:
    pass  # pcie command not available


def main():
    """Main entry point"""
    try:
        cli(obj={})
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()