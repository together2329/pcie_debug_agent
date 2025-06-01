# PCIe Debug Agent - Slash Commands Demo

The interactive mode now properly handles slash commands, similar to Claude Code's interface.

## Available Slash Commands

### Model Management
```bash
🔧 > /model
# Lists all available models with current selection

🔧 > /model mock-llm
# Switches to the mock model (built-in, always available)

🔧 > /model llama-3.2-3b
# Switches to Llama model (requires download)
```

### System Information
```bash
🔧 > /status
# Shows current model, document count, turns, memory, session

🔧 > /config
# Displays current configuration settings
```

### Search & Analysis
```bash
🔧 > /search completion timeout
# Searches knowledge base for relevant documents

🔧 > /analyze logs/pcie_error.log
# Analyzes a specific log file
```

### Memory & Sessions
```bash
🔧 > /memory
# Shows current memory entries

🔧 > /memory set key value
# Sets a memory entry

🔧 > /session
# Lists recent sessions

🔧 > /session save
# Saves current conversation
```

### Conversation Control
```bash
🔧 > /clear
# Clears current conversation history

🔧 > /help
# Shows all available commands

🔧 > /exit or /quit
# Exits the interactive shell
```

## Direct Queries

You can also type PCIe questions directly without any prefix:

```bash
🔧 > What causes PCIe completion timeout?
# Analyzes and provides detailed response

🔧 > How to debug link training failures?
# Provides debugging guidance
```

## Implementation Details

The slash command handling is implemented in `src/cli/interactive.py`:

```python
def default(self, line):
    """Handle non-command input as PCIe questions"""
    if not line.strip():
        return
    
    # Handle slash commands
    if line.startswith('/'):
        # Convert slash command to method call
        parts = line[1:].split(None, 1)
        cmd = parts[0]
        arg = parts[1] if len(parts) > 1 else ''
        
        # Map common slash commands to methods
        if hasattr(self, f'do_{cmd}'):
            return getattr(self, f'do_{cmd}')(arg)
        else:
            print(f"❌ Unknown command: /{cmd}")
            print("   Type /help for available commands")
            return
    
    # Process as PCIe debugging query
    self._process_query(line)
```

This ensures that:
- Lines starting with `/` are treated as commands
- The slash is stripped and the command is parsed
- The appropriate `do_<command>` method is called
- Unknown commands show an error message
- Non-slash input is processed as PCIe queries

## Current Status

✅ All slash commands are working correctly
✅ Model switching with `/model` functions properly
✅ Interactive mode handles both commands and queries
✅ System uses mock-llm by default (no setup required)