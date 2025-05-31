"""Main entry point for PCIe Debug Agent CLI with Hybrid LLM support"""

import click
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.cli.commands import index, search, report, config, test
from src.cli.commands.analyze_hybrid import analyze_hybrid
from src.cli.utils.output import console, print_banner
from src.config.settings import Settings, load_settings


@click.group()
@click.version_option(version="1.1.0", prog_name="PCIe Debug Agent")
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
    """PCIe Debug Agent - AI-powered PCIe log analysis tool with Hybrid LLM
    
    Analyze PCIe debug logs using intelligent model selection:
    - Quick analysis: Llama 3.2 3B for fast interactive debugging
    - Detailed analysis: DeepSeek Q4_1 for comprehensive investigation
    - Auto mode: Automatically selects the best model
    
    Version 1.1.0 - Now with Hybrid LLM support!
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Print banner unless quiet mode
    if not quiet:
        print_banner()
        # Show hybrid LLM status
        console.print("[cyan]🤖 Hybrid LLM Mode Enabled[/cyan]")
        console.print("[dim]Using Llama 3.2 3B + DeepSeek Q4_1[/dim]\n")
    
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
cli.add_command(analyze_hybrid, name="analyze")  # Use hybrid analyze as default
cli.add_command(index.index)
cli.add_command(search.search)
cli.add_command(report.report)
cli.add_command(config.config)
cli.add_command(test.test)

# Add a new command group for model management
@cli.group()
def model():
    """Manage hybrid LLM models"""
    pass

@model.command()
@click.pass_context
def status(ctx: click.Context):
    """Check status of hybrid LLM models"""
    from src.models.hybrid_llm_provider import HybridLLMProvider
    
    provider = HybridLLMProvider()
    status = provider.get_model_status()
    
    console.print("\n[bold cyan]Hybrid LLM Model Status[/bold cyan]")
    console.print("=" * 50)
    
    # Llama status
    llama_status = status["llama"]
    console.print(f"\n🦙 [bold]Llama 3.2 3B[/bold]")
    console.print(f"   Status: {'✅ Available' if llama_status['available'] else '❌ Not Available'}")
    console.print(f"   Use Case: {llama_status['use_case']}")
    console.print(f"   Speed: {llama_status['typical_response_time']}")
    
    # DeepSeek status
    deepseek_status = status["deepseek"]
    console.print(f"\n🤖 [bold]DeepSeek Q4_1[/bold]")
    console.print(f"   Status: {'✅ Available' if deepseek_status['available'] else '❌ Not Available'}")
    console.print(f"   Use Case: {deepseek_status['use_case']}")
    console.print(f"   Speed: {deepseek_status['typical_response_time']}")
    
    # Overall status
    console.print(f"\n🎯 [bold]Hybrid System[/bold]")
    console.print(f"   Status: {'✅ Ready' if status['hybrid_ready'] else '❌ Not Ready'}")
    
    if status['hybrid_ready']:
        console.print("\n[green]✅ Hybrid LLM system is ready for use![/green]")
    else:
        console.print("\n[yellow]⚠️  At least one model needs to be set up[/yellow]")

@model.command()
@click.option("--quick", is_flag=True, help="Test quick analysis (Llama)")
@click.option("--detailed", is_flag=True, help="Test detailed analysis (DeepSeek)")
@click.pass_context
def test(ctx: click.Context, quick: bool, detailed: bool):
    """Test hybrid LLM models with sample queries"""
    from src.models.hybrid_llm_provider import HybridLLMProvider
    
    provider = HybridLLMProvider()
    
    test_query = "What is a PCIe TLP error?"
    test_log = "[10:15:30] PCIe: TLP error detected on device 0000:01:00.0"
    
    if quick or (not quick and not detailed):
        console.print("\n[cyan]Testing Quick Analysis (Llama 3.2 3B)...[/cyan]")
        response = provider.quick_analysis(test_query, test_log)
        if response.response:
            console.print(f"✅ Model: {response.model_used}")
            console.print(f"⏱️  Time: {response.response_time:.2f}s")
            console.print(f"📝 Response: {response.response[:200]}...")
        else:
            console.print(f"❌ Failed: {response.error}")
    
    if detailed or (not quick and not detailed):
        console.print("\n[cyan]Testing Detailed Analysis (DeepSeek Q4_1)...[/cyan]")
        response = provider.detailed_analysis(test_query, test_log)
        if response.response:
            console.print(f"✅ Model: {response.model_used}")
            console.print(f"⏱️  Time: {response.response_time:.2f}s")
            console.print(f"📝 Response: {response.response[:200]}...")
        else:
            console.print(f"❌ Failed: {response.error}")


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