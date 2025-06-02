# CLI Fixes Applied

## Issues Fixed:

### 1. ✅ Fixed `/status` command not being recognized
- Added `onecmd` method to properly route slash commands
- Now all slash commands work correctly
- Unknown commands show helpful error messages

### 2. ✅ Fixed 'line' undefined error
- Changed all occurrences of `line` to `query` in `_process_regular_query` method
- Regular queries now work without errors

## Test the Fixes:

```bash
# Start the CLI
./pcie-debug

# Test commands that were broken:
/status          # Should now show system status
/rag_status      # Should show RAG status
/help            # Should show help

# Test regular queries:
What is PCIe link training?
Why timeout happened?
```

## What Was Fixed:

1. **In `_process_regular_query` method** (lines 2637-2678):
   - Changed parameter usage from `line` to `query`
   - Fixed undefined variable error

2. **Added `onecmd` method** (lines 2948-2992):
   - Properly routes slash commands to their handlers
   - Shows command suggestions when typing `/`
   - Provides helpful error messages

The CLI should now work properly with all commands!