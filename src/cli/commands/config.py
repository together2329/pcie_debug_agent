"""Config command for PCIe Debug Agent CLI"""

import click
import sys
from pathlib import Path
import yaml
import json
from typing import Optional, Any

from src.cli.utils.output import (
    console, print_error, print_success, print_info, print_warning,
    print_yaml, print_json, confirm
)
from src.config.settings import Settings


@click.group()
def config():
    """Manage configuration settings"""
    pass


@config.command()
@click.option(
    "--format", "-f",
    type=click.Choice(["yaml", "json", "env"]),
    default="yaml",
    help="Output format"
)
@click.pass_context
def show(ctx: click.Context, format: str):
    """Show current configuration
    
    Examples:
        pcie-debug config show
        
        pcie-debug config show --format json
    """
    settings = ctx.obj["settings"]
    
    try:
        if format == "yaml":
            print_yaml(settings.to_dict(), title="Current Configuration")
        elif format == "json":
            print_json(settings.to_dict(), title="Current Configuration")
        elif format == "env":
            # Show as environment variables
            console.print("[bold]Environment Variables:[/bold]\n")
            env_vars = _settings_to_env_vars(settings.to_dict())
            for key, value in env_vars.items():
                console.print(f"{key}={value}")
                
    except Exception as e:
        print_error(f"Failed to show configuration: {e}")
        sys.exit(1)


@config.command()
@click.argument("key")
@click.argument("value")
@click.option(
    "--type", "-t",
    type=click.Choice(["string", "int", "float", "bool"]),
    default="string",
    help="Value type"
)
@click.pass_context
def set(ctx: click.Context, key: str, value: str, type: str):
    """Set a configuration value
    
    Examples:
        pcie-debug config set llm.model gpt-4
        
        pcie-debug config set chunk_size 1000 --type int
        
        pcie-debug config set rag.rerank true --type bool
    """
    settings = ctx.obj["settings"]
    config_path = ctx.obj.get("config_path", "configs/settings.yaml")
    
    try:
        # Convert value to correct type
        if type == "int":
            value = int(value)
        elif type == "float":
            value = float(value)
        elif type == "bool":
            value = value.lower() in ["true", "yes", "1", "on"]
        
        # Update settings
        _set_nested_value(settings, key, value)
        
        # Validate
        settings.validate()
        
        # Save to file
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(settings.to_dict(), f, default_flow_style=False)
        
        print_success(f"Set {key} = {value}")
        
    except ValueError as e:
        print_error(f"Invalid value: {e}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to set configuration: {e}")
        sys.exit(1)


@config.command()
@click.argument("key")
@click.pass_context
def get(ctx: click.Context, key: str):
    """Get a configuration value
    
    Examples:
        pcie-debug config get llm.model
        
        pcie-debug config get embedding_dimension
    """
    settings = ctx.obj["settings"]
    
    try:
        value = _get_nested_value(settings, key)
        console.print(f"{key} = {value}")
        
    except KeyError:
        print_error(f"Configuration key not found: {key}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to get configuration: {e}")
        sys.exit(1)


@config.command()
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output file path"
)
@click.pass_context
def init(ctx: click.Context, output: Optional[str]):
    """Initialize configuration file
    
    Examples:
        pcie-debug config init
        
        pcie-debug config init --output my-config.yaml
    """
    output_path = Path(output or "configs/settings.yaml")
    
    try:
        if output_path.exists():
            if not confirm(f"Configuration file {output_path} already exists. Overwrite?"):
                print_info("Operation cancelled")
                return
        
        # Create default configuration
        settings = Settings()
        
        # Interactive setup
        console.print("\n[bold]Configuration Setup[/bold]\n")
        
        # API Provider
        provider = click.prompt(
            "Select LLM provider",
            type=click.Choice(["openai", "anthropic", "ollama", "custom"]),
            default="openai"
        )
        settings.llm.provider = provider
        
        if provider == "openai":
            settings.llm.model = click.prompt(
                "OpenAI model",
                default="gpt-3.5-turbo"
            )
            api_key = click.prompt(
                "OpenAI API key (or set OPENAI_API_KEY env var)",
                default="",
                hide_input=True
            )
            if api_key:
                settings.llm.api_key = api_key
                
        elif provider == "anthropic":
            settings.llm.model = click.prompt(
                "Anthropic model",
                default="claude-2"
            )
            api_key = click.prompt(
                "Anthropic API key (or set ANTHROPIC_API_KEY env var)",
                default="",
                hide_input=True
            )
            if api_key:
                settings.llm.api_key = api_key
        
        # Embedding settings
        embedding_provider = click.prompt(
            "Select embedding provider",
            type=click.Choice(["local", "openai", "custom"]),
            default="local"
        )
        settings.embedding.provider = embedding_provider
        
        if embedding_provider == "local":
            settings.embedding.model = click.prompt(
                "Local embedding model",
                default="sentence-transformers/all-MiniLM-L6-v2"
            )
        elif embedding_provider == "openai":
            settings.embedding.model = click.prompt(
                "OpenAI embedding model",
                default="text-embedding-3-small"
            )
        
        # Save configuration
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(settings.to_dict(), f, default_flow_style=False)
        
        print_success(f"Configuration saved to: {output_path}")
        print_info("\nNext steps:")
        print_info("1. Set API keys as environment variables if not provided")
        print_info("2. Run 'pcie-debug index build' to index your logs")
        print_info("3. Run 'pcie-debug analyze' to start analyzing")
        
    except Exception as e:
        print_error(f"Failed to initialize configuration: {e}")
        sys.exit(1)


@config.command()
@click.pass_context
def validate(ctx: click.Context):
    """Validate current configuration
    
    Example:
        pcie-debug config validate
    """
    settings = ctx.obj["settings"]
    
    try:
        settings.validate()
        print_success("Configuration is valid!")
        
        # Check API keys
        warnings = []
        
        if settings.llm.provider != "ollama" and not settings.llm.api_key:
            warnings.append(f"No API key set for LLM provider '{settings.llm.provider}'")
        
        if settings.embedding.provider not in ["local", "ollama"] and not settings.embedding.api_key:
            warnings.append(f"No API key set for embedding provider '{settings.embedding.provider}'")
        
        if warnings:
            print_warning("\nWarnings:")
            for warning in warnings:
                print_warning(f"  - {warning}")
        
    except ValueError as e:
        print_error(f"Configuration validation failed: {e}")
        sys.exit(1)


def _get_nested_value(obj: Any, key: str) -> Any:
    """Get nested value from object using dot notation"""
    parts = key.split('.')
    current = obj
    
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(f"Key '{part}' not found")
    
    return current


def _set_nested_value(obj: Any, key: str, value: Any):
    """Set nested value in object using dot notation"""
    parts = key.split('.')
    current = obj
    
    for part in parts[:-1]:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            raise KeyError(f"Key '{part}' not found")
    
    final_key = parts[-1]
    if hasattr(current, final_key):
        setattr(current, final_key, value)
    else:
        raise KeyError(f"Key '{final_key}' not found")


def _settings_to_env_vars(settings_dict: dict, prefix: str = "") -> dict:
    """Convert settings dictionary to environment variables"""
    env_vars = {}
    
    for key, value in settings_dict.items():
        env_key = f"{prefix}{key}".upper().replace('.', '_')
        
        if isinstance(value, dict):
            # Recursive for nested dicts
            nested = _settings_to_env_vars(value, f"{env_key}_")
            env_vars.update(nested)
        else:
            env_vars[env_key] = str(value)
    
    return env_vars