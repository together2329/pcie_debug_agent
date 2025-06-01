"""
Memory management for PCIe Debug Agent
Similar to Claude Code's memory system
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional


class MemoryManager:
    """Manages persistent memory across sessions"""
    
    def __init__(self, memory_dir: Optional[Path] = None):
        self.memory_dir = memory_dir or Path.home() / ".pcie_debug" / "memory"
        self.memory_file = self.memory_dir / "memory.json"
        
        # Create directory if it doesn't exist
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing memory
        self._memory = self._load_memory()
    
    def _load_memory(self) -> Dict[str, Any]:
        """Load memory from disk"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}
    
    def _save_memory(self):
        """Save memory to disk"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self._memory, f, indent=2)
        except Exception as e:
            print(f"Failed to save memory: {e}")
    
    def set_memory(self, key: str, value: Any):
        """Set a memory entry"""
        self._memory[key] = {
            "value": value,
            "timestamp": time.time(),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self._save_memory()
    
    def get_memory(self, key: Optional[str] = None) -> Any:
        """Get memory entry or all memory"""
        if key is None:
            # Return all memory values (without metadata)
            return {k: v["value"] for k, v in self._memory.items()}
        
        entry = self._memory.get(key)
        return entry["value"] if entry else None
    
    def delete_memory(self, key: str) -> bool:
        """Delete a memory entry"""
        if key in self._memory:
            del self._memory[key]
            self._save_memory()
            return True
        return False
    
    def clear_memory(self):
        """Clear all memory"""
        self._memory.clear()
        self._save_memory()
    
    def list_memory(self) -> Dict[str, Dict[str, Any]]:
        """List all memory entries with metadata"""
        return self._memory.copy()
    
    def get_memory_summary(self) -> str:
        """Get a summary of current memory for prompts"""
        if not self._memory:
            return "No persistent memory entries."
        
        summary_parts = ["Persistent Memory:"]
        for key, entry in self._memory.items():
            value = entry["value"]
            if len(str(value)) > 100:
                value = str(value)[:100] + "..."
            summary_parts.append(f"- {key}: {value}")
        
        return "\n".join(summary_parts)
    
    def add_context_to_prompt(self, prompt: str) -> str:
        """Add memory context to a prompt"""
        memory_summary = self.get_memory_summary()
        if memory_summary == "No persistent memory entries.":
            return prompt
        
        return f"{memory_summary}\n\nUser Query: {prompt}"


# Predefined memory templates for PCIe debugging
PCIE_MEMORY_TEMPLATES = {
    "system_info": {
        "description": "Store system configuration details",
        "example": "Intel Z690 chipset, PCIe 4.0, 32GB DDR5"
    },
    "recent_issues": {
        "description": "Track recent PCIe issues encountered",
        "example": "Link training failures on slot 1, completion timeouts"
    },
    "solutions_tried": {
        "description": "Remember what solutions have been attempted",
        "example": "Reseated cards, updated drivers, checked power"
    },
    "hardware_config": {
        "description": "Store hardware configuration details",
        "example": "RTX 4090 in slot 1, NVMe SSD in M.2_1"
    },
    "error_patterns": {
        "description": "Remember common error patterns",
        "example": "AER errors occur during high GPU load"
    }
}