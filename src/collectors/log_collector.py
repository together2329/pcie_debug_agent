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
    """UVM 에러 데이터 구조"""
    timestamp: str
    severity: str  # ERROR, FATAL, WARNING
    component: str
    message: str
    file_path: str
    line_number: Optional[int]
    raw_content: str
    context_before: List[str]
    context_after: List[str]

class LogCollector:
    """로그 파일 수집 및 처리"""
    
    def __init__(self, config: Dict[str, Any]):
        self.log_directories = config.get('log_directories', [])
        self.error_patterns = config.get('error_patterns', {})
        self.context_lines = config.get('context_lines', 5)
        
    def collect_logs(self, 
                     start_time: Optional[datetime] = None,
                     file_pattern: str = "*.log") -> List[Path]:
        """
        지정된 디렉토리에서 로그 파일 수집
        
        Args:
            start_time: 이 시간 이후 수정된 파일만 수집
            file_pattern: 파일 패턴 (기본: *.log)
            
        Returns:
            수집된 로그 파일 경로 리스트
        """
        collected_files = []
        
        for directory in self.log_directories:
            if not os.path.exists(directory):
                logger.warning(f"Directory not found: {directory}")
                continue
                
            pattern = os.path.join(directory, "**", file_pattern)
            files = glob.glob(pattern, recursive=True)
            
            for file_path in files:
                if start_time:
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if mod_time < start_time:
                        continue
                        
                collected_files.append(Path(file_path))
                
        logger.info(f"Collected {len(collected_files)} log files")
        return collected_files
    
    def extract_errors(self, log_file: Path) -> List[UVMError]:
        """
        로그 파일에서 UVM 에러 추출
        
        Args:
            log_file: 로그 파일 경로
            
        Returns:
            추출된 UVM 에러 리스트
        """
        errors = []
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                for severity, pattern in self.error_patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        error = self._parse_error(
                            lines, i, severity, match, str(log_file)
                        )
                        errors.append(error)
                        
        except Exception as e:
            logger.error(f"Error processing file {log_file}: {e}")
            
        return errors
    
    def _parse_error(self, 
                     lines: List[str], 
                     line_idx: int, 
                     severity: str,
                     match: re.Match,
                     file_path: str) -> UVMError:
        """에러 상세 정보 파싱"""
        
        # 타임스탬프 추출
        timestamp_match = re.search(r'(\d+\.\d+[a-z]*)', lines[line_idx])
        timestamp = timestamp_match.group(1) if timestamp_match else "unknown"
        
        # 컴포넌트 추출
        component_match = re.search(r'@\s*(\S+)', lines[line_idx])
        component = component_match.group(1) if component_match else "unknown"
        
        # 에러 메시지
        message = match.group(1).strip()
        
        # 파일 및 라인 정보
        file_line_match = re.search(r'\[(\S+):(\d+)\]', lines[line_idx])
        if file_line_match:
            error_file = file_line_match.group(1)
            line_number = int(file_line_match.group(2))
        else:
            error_file = "unknown"
            line_number = None
            
        # 컨텍스트 추출
        start_idx = max(0, line_idx - self.context_lines)
        end_idx = min(len(lines), line_idx + self.context_lines + 1)
        
        context_before = [lines[i].strip() for i in range(start_idx, line_idx)]
        context_after = [lines[i].strip() for i in range(line_idx + 1, end_idx)]
        
        return UVMError(
            timestamp=timestamp,
            severity=severity.upper(),
            component=component,
            message=message,
            file_path=error_file,
            line_number=line_number,
            raw_content=lines[line_idx].strip(),
            context_before=context_before,
            context_after=context_after
        ) 