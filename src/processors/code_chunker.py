import re
import ast
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CodeChunk:
    """코드 청크 데이터 구조"""
    
    def __init__(self,
                 content: str,
                 metadata: Dict[str, Any],
                 chunk_id: str):
        self.content = content
        self.metadata = metadata
        self.chunk_id = chunk_id
        self.embedding = None
        
    def __repr__(self):
        return f"CodeChunk(id={self.chunk_id}, type={self.metadata.get('type', 'unknown')})"

class SystemVerilogChunker:
    """SystemVerilog 코드 청킹"""
    
    def __init__(self, 
                 max_chunk_size: int = 500,
                 min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
    def chunk_sv_file(self, sv_path: Path) -> List[CodeChunk]:
        """SystemVerilog 파일을 구조 단위로 청킹"""
        chunks = []
        
        try:
            with open(sv_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 모듈 추출
            modules = self._extract_modules(content)
            for module_name, module_content in modules:
                chunk_id = f"{sv_path.stem}_{module_name}"
                chunk = CodeChunk(
                    content=module_content,
                    metadata={
                        'source': str(sv_path),
                        'type': 'module',
                        'name': module_name,
                        'language': 'systemverilog'
                    },
                    chunk_id=chunk_id
                )
                chunks.append(chunk)
            
            # 클래스 추출
            classes = self._extract_classes(content)
            for class_name, class_content in classes:
                chunk_id = f"{sv_path.stem}_{class_name}"
                chunk = CodeChunk(
                    content=class_content,
                    metadata={
                        'source': str(sv_path),
                        'type': 'class',
                        'name': class_name,
                        'language': 'systemverilog'
                    },
                    chunk_id=chunk_id
                )
                chunks.append(chunk)
            
            # 함수/태스크 추출
            functions = self._extract_functions(content)
            for func_name, func_content in functions:
                if len(func_content.split()) >= self.min_chunk_size:
                    chunk_id = f"{sv_path.stem}_{func_name}"
                    chunk = CodeChunk(
                        content=func_content,
                        metadata={
                            'source': str(sv_path),
                            'type': 'function',
                            'name': func_name,
                            'language': 'systemverilog'
                        },
                        chunk_id=chunk_id
                    )
                    chunks.append(chunk)
                    
        except Exception as e:
            logger.error(f"Error processing SystemVerilog file {sv_path}: {e}")
            
        return chunks
    
    def _extract_modules(self, content: str) -> List[Tuple[str, str]]:
        """모듈 추출"""
        modules = []
        
        # 모듈 패턴 매칭
        module_pattern = r'module\s+(\w+)\s*(?:\#\s*\([^)]*\))?\s*\([^)]*\)\s*;(.*?)endmodule'
        matches = re.finditer(module_pattern, content, re.DOTALL | re.MULTILINE)
        
        for match in matches:
            module_name = match.group(1)
            module_content = match.group(0)
            
            # 크기 확인
            if len(module_content.split()) <= self.max_chunk_size:
                modules.append((module_name, module_content))
            else:
                # 큰 모듈은 내부 블록으로 분할
                sub_chunks = self._split_large_block(module_content, module_name)
                modules.extend(sub_chunks)
                
        return modules
    
    def _extract_classes(self, content: str) -> List[Tuple[str, str]]:
        """클래스 추출"""
        classes = []
        
        # 클래스 패턴 매칭
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+\w+)?\s*;(.*?)endclass'
        matches = re.finditer(class_pattern, content, re.DOTALL | re.MULTILINE)
        
        for match in matches:
            class_name = match.group(1)
            class_content = match.group(0)
            
            if len(class_content.split()) <= self.max_chunk_size:
                classes.append((class_name, class_content))
            else:
                # 큰 클래스는 메서드 단위로 분할
                methods = self._extract_class_methods(class_content, class_name)
                classes.extend(methods)
                
        return classes
    
    def _extract_functions(self, content: str) -> List[Tuple[str, str]]:
        """함수/태스크 추출"""
        functions = []
        
        # 함수 패턴
        func_pattern = r'(?:function|task)\s+(?:\w+\s+)?(\w+)\s*\([^)]*\)\s*;(.*?)(?:endfunction|endtask)'
        matches = re.finditer(func_pattern, content, re.DOTALL | re.MULTILINE)
        
        for match in matches:
            func_name = match.group(1)
            func_content = match.group(0)
            functions.append((func_name, func_content))
            
        return functions
    
    def _extract_class_methods(self, class_content: str, class_name: str) -> List[Tuple[str, str]]:
        """클래스 내부 메서드 추출"""
        methods = []
        
        # 클래스 내부 함수/태스크 패턴
        method_pattern = r'(?:function|task)\s+(?:\w+\s+)?(\w+)\s*\([^)]*\)\s*;(.*?)(?:endfunction|endtask)'
        matches = re.finditer(method_pattern, class_content, re.DOTALL | re.MULTILINE)
        
        for match in matches:
            method_name = match.group(1)
            method_content = match.group(0)
            full_name = f"{class_name}.{method_name}"
            methods.append((full_name, method_content))
            
        return methods
    
    def _split_large_block(self, content: str, block_name: str) -> List[Tuple[str, str]]:
        """큰 블록을 작은 청크로 분할"""
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_size = 0
        chunk_idx = 0
        
        for line in lines:
            line_size = len(line.split())
            
            if current_size + line_size > self.max_chunk_size and current_chunk:
                chunk_content = '\n'.join(current_chunk)
                chunk_name = f"{block_name}_part{chunk_idx}"
                chunks.append((chunk_name, chunk_content))
                
                current_chunk = []
                current_size = 0
                chunk_idx += 1
            
            current_chunk.append(line)
            current_size += line_size
            
        # 마지막 청크
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunk_name = f"{block_name}_part{chunk_idx}"
            chunks.append((chunk_name, chunk_content))
            
        return chunks 