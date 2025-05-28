import re
import ast
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CodeChunk:
    """Class representing a chunk of code"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    start_line: int
    end_line: int

class SystemVerilogChunker:
    """Processes and chunks SystemVerilog code files"""
    
    # Regular expressions for SystemVerilog code analysis
    PATTERNS = {
        'module': r'^\s*module\s+(\w+)',
        'class': r'^\s*class\s+(\w+)',
        'function': r'^\s*(?:function|task)\s+(?:static\s+)?(?:virtual\s+)?(?:protected\s+)?(?:local\s+)?(\w+)',
        'interface': r'^\s*interface\s+(\w+)',
        'package': r'^\s*package\s+(\w+)',
        'block_comment': r'/\*[\s\S]*?\*/',
        'line_comment': r'//.*$'
    }
    
    def __init__(self, max_chunk_size: int = 500):
        """Initialize SystemVerilog chunker"""
        self.max_chunk_size = max_chunk_size
        self.logger = logging.getLogger(__name__)
    
    def chunk_sv_file(self, file_path: Path) -> List[CodeChunk]:
        """Process and chunk a SystemVerilog file"""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Extract metadata
            metadata = self._extract_metadata(file_path, lines)
            
            # Split content into chunks
            chunks = self._split_into_chunks(lines, metadata)
            
            return chunks
        
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            return []
    
    def _extract_metadata(self, file_path: Path, lines: List[str]) -> Dict[str, Any]:
        """Extract metadata from SystemVerilog file"""
        metadata = {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'last_modified': file_path.stat().st_mtime,
            'total_lines': len(lines)
        }
        
        # Extract module/class/interface names
        for line in lines:
            for pattern_name, pattern in self.PATTERNS.items():
                if pattern_name in ['module', 'class', 'interface']:
                    match = re.search(pattern, line)
                    if match:
                        metadata[f'{pattern_name}_name'] = match.group(1)
        
        return metadata
    
    def _split_into_chunks(self, lines: List[str], metadata: Dict[str, Any]) -> List[CodeChunk]:
        """Split code into logical chunks"""
        chunks = []
        current_chunk = []
        current_start_line = 1
        brace_count = 0
        in_block_comment = False
        
        for i, line in enumerate(lines, 1):
            # Handle block comments
            if '/*' in line:
                in_block_comment = True
            if '*/' in line:
                in_block_comment = False
            
            # Skip comments
            if in_block_comment or line.strip().startswith('//'):
                continue
            
            # Count braces for block detection
            brace_count += line.count('{') - line.count('}')
            
            # Add line to current chunk
            current_chunk.append(line)
            
            # Check if we should create a new chunk
            if (brace_count == 0 and  # End of a block
                len(current_chunk) > 0 and
                (len(current_chunk) >= self.max_chunk_size or
                 any(re.search(pattern, line) for pattern in self.PATTERNS.values()))):
                
                # Create chunk
                chunk_content = ''.join(current_chunk)
                chunk = CodeChunk(
                    content=chunk_content,
                    metadata=metadata.copy(),
                    chunk_id=f"{metadata['file_name']}_{len(chunks)}",
                    start_line=current_start_line,
                    end_line=i
                )
                chunks.append(chunk)
                
                # Reset for next chunk
                current_chunk = []
                current_start_line = i + 1
        
        # Add the last chunk if there's any content left
        if current_chunk:
            chunk_content = ''.join(current_chunk)
            chunk = CodeChunk(
                content=chunk_content,
                metadata=metadata.copy(),
                chunk_id=f"{metadata['file_name']}_{len(chunks)}",
                start_line=current_start_line,
                end_line=len(lines)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _clean_code(self, code: str) -> str:
        """Clean code by removing comments and extra whitespace"""
        # Remove block comments
        code = re.sub(self.PATTERNS['block_comment'], '', code)
        
        # Remove line comments
        code = re.sub(self.PATTERNS['line_comment'], '', code)
        
        # Remove extra whitespace
        code = re.sub(r'\s+', ' ', code)
        
        return code.strip()
    
    def _is_complete_block(self, code: str) -> bool:
        """Check if code chunk is a complete block"""
        brace_count = 0
        for char in code:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
        
        return brace_count == 0 