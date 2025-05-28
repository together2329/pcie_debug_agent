import os
import re
import glob
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class UVMError:
    """Class representing a UVM error"""
    timestamp: datetime
    severity: str
    component: str
    message: str
    file_path: str
    line_number: Optional[int] = None
    context: Optional[str] = None
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity,
            'component': self.component,
            'message': self.message,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'context': self.context,
            'stack_trace': self.stack_trace
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UVMError':
        """Create error from dictionary"""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            severity=data['severity'],
            component=data['component'],
            message=data['message'],
            file_path=data['file_path'],
            line_number=data.get('line_number'),
            context=data.get('context'),
            stack_trace=data.get('stack_trace')
        )

class LogCollector:
    """Collects and parses UVM simulation logs"""
    
    # Regular expressions for parsing UVM logs
    ERROR_PATTERNS = {
        'timestamp': r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})',
        'severity': r'(FATAL|ERROR|WARNING|INFO)',
        'component': r'@(\w+):',
        'message': r':\s*(.*?)(?=\n|$)',
        'file_location': r'at\s+([^(]+)\((\d+)\)',
        'stack_trace': r'(?:at\s+[^(]+\(\d+\)\n?)+'
    }
    
    def __init__(self, settings: Dict[str, Any]):
        """Initialize log collector"""
        self.settings = settings
        self.logger = logging.getLogger(__name__)
    
    def collect_logs(self, start_time: Optional[str] = None) -> List[Path]:
        """Collect log files from configured directories"""
        log_files = []
        
        for log_dir in self.settings['log_directories']:
            log_path = Path(log_dir)
            if not log_path.exists():
                self.logger.warning(f"Log directory does not exist: {log_dir}")
                continue
            
            # Find all log files
            for log_file in log_path.glob("**/*.log"):
                if start_time:
                    # Check file modification time
                    mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if mtime < datetime.fromisoformat(start_time):
                        continue
                log_files.append(log_file)
        
        return log_files
    
    def extract_errors(self, log_file: Path) -> List[UVMError]:
        """Extract errors from a log file"""
        errors = []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find all error blocks
            error_blocks = self._find_error_blocks(content)
            
            for block in error_blocks:
                error = self._parse_error_block(block, log_file)
                if error:
                    errors.append(error)
        
        except Exception as e:
            self.logger.error(f"Error processing log file {log_file}: {str(e)}")
        
        return errors
    
    def _find_error_blocks(self, content: str) -> List[str]:
        """Find error blocks in log content"""
        # Pattern to match error blocks
        pattern = r'(?:FATAL|ERROR|WARNING|INFO).*?(?=(?:FATAL|ERROR|WARNING|INFO)|$)'
        return re.findall(pattern, content, re.DOTALL)
    
    def _parse_error_block(self, block: str, log_file: Path) -> Optional[UVMError]:
        """Parse an error block into a UVMError object"""
        try:
            # Extract timestamp
            timestamp_match = re.search(self.ERROR_PATTERNS['timestamp'], block)
            if not timestamp_match:
                return None
            timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
            
            # Extract severity
            severity_match = re.search(self.ERROR_PATTERNS['severity'], block)
            if not severity_match:
                return None
            severity = severity_match.group(1)
            
            # Extract component
            component_match = re.search(self.ERROR_PATTERNS['component'], block)
            component = component_match.group(1) if component_match else "Unknown"
            
            # Extract message
            message_match = re.search(self.ERROR_PATTERNS['message'], block)
            if not message_match:
                return None
            message = message_match.group(1).strip()
            
            # Extract file location
            file_match = re.search(self.ERROR_PATTERNS['file_location'], block)
            file_path = file_match.group(1).strip() if file_match else str(log_file)
            line_number = int(file_match.group(2)) if file_match else None
            
            # Extract stack trace
            stack_match = re.search(self.ERROR_PATTERNS['stack_trace'], block)
            stack_trace = stack_match.group(0) if stack_match else None
            
            # Extract context (lines before the error)
            context_lines = block.split('\n')[:3]  # Get first 3 lines as context
            context = '\n'.join(context_lines) if context_lines else None
            
            return UVMError(
                timestamp=timestamp,
                severity=severity,
                component=component,
                message=message,
                file_path=file_path,
                line_number=line_number,
                context=context,
                stack_trace=stack_trace
            )
        
        except Exception as e:
            self.logger.error(f"Error parsing error block: {str(e)}")
            return None 