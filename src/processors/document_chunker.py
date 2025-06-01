import re
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import PyPDF2
import markdown
from bs4 import BeautifulSoup
import logging
from dataclasses import dataclass
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Class representing a chunk of a document"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str

class DocumentChunker:
    """Processes and chunks documents into smaller pieces"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize document chunker"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = 50  # Minimum words per chunk
        self.logger = logging.getLogger(__name__)
    
    def chunk_documents(self, file_path: Path) -> List[DocumentChunk]:
        """Process and chunk a document file"""
        try:
            # Read file content based on file type
            content = self._read_file(file_path)
            if not content:
                return []
            
            # Extract metadata
            metadata = self._extract_metadata(file_path, content)
            
            # Split content into chunks
            chunks = self._split_into_chunks(content, metadata)
            
            return chunks
        
        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {str(e)}")
            return []
    
    def _read_file(self, file_path: Path) -> Optional[str]:
        """Read file content based on file type"""
        try:
            if file_path.suffix.lower() == '.pdf':
                return self._read_pdf(file_path)
            elif file_path.suffix.lower() == '.md':
                return self._read_markdown(file_path)
            elif file_path.suffix.lower() == '.txt':
                return self._read_text(file_path)
            else:
                self.logger.warning(f"Unsupported file type: {file_path.suffix}")
                return None
        
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            return None
    
    def _read_pdf(self, file_path: Path) -> str:
        """Read PDF file content"""
        text = []
        with open(file_path, 'rb') as f:
            pdf = PdfReader(f)
            for page in pdf.pages:
                text.append(page.extract_text())
        return '\n'.join(text)
    
    def _read_markdown(self, file_path: Path) -> str:
        """Read Markdown file content"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Convert markdown to HTML
        html = markdown.markdown(content)
        
        # Parse HTML and extract text
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()
    
    def _read_text(self, file_path: Path) -> str:
        """Read text file content"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _extract_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Extract metadata from document"""
        metadata = {
            'file_name': file_path.name,
            'file_type': file_path.suffix[1:],
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'last_modified': file_path.stat().st_mtime
        }
        
        # Extract title from content
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            metadata['title'] = title_match.group(1).strip()
        
        return metadata
    
    def _split_into_chunks(self, content: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Split content into overlapping chunks"""
        chunks = []
        
        # Split content into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence.split())
            
            # If adding this sentence would exceed chunk size, create a new chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk
                chunk_content = ' '.join(current_chunk)
                chunk = DocumentChunk(
                    content=chunk_content,
                    metadata=metadata.copy(),
                    chunk_id=f"{metadata['file_name']}_{len(chunks)}"
                )
                chunks.append(chunk)
                
                # Keep overlap
                overlap_words = []
                overlap_length = 0
                for word in reversed(current_chunk):
                    if overlap_length + len(word.split()) <= self.chunk_overlap:
                        overlap_words.insert(0, word)
                        overlap_length += len(word.split())
                    else:
                        break
                
                current_chunk = overlap_words
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add the last chunk if there's any content left
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunk = DocumentChunk(
                content=chunk_content,
                metadata=metadata.copy(),
                chunk_id=f"{metadata['file_name']}_{len(chunks)}"
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_text(self, text: str) -> List[str]:
        """Simple text chunking method for string input"""
        if not text:
            return []
        
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence.split())
            
            # If adding this sentence would exceed chunk size, create a new chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Keep overlap
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    current_chunk = current_chunk[-self.chunk_overlap:]
                    current_length = sum(len(s.split()) for s in current_chunk)
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add the last chunk if there's any content left
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def chunk_pdf(self, pdf_path: Path) -> List[DocumentChunk]:
        """PDF 문서 청킹"""
        chunks = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    
                    # 페이지 텍스트를 청크로 분할
                    page_chunks = self._split_text_into_chunks(
                        text,
                        {
                            'source': str(pdf_path),
                            'page': page_num + 1,
                            'type': 'pdf'
                        }
                    )
                    chunks.extend(page_chunks)
                    
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            
        return chunks
    
    def chunk_markdown(self, md_path: Path) -> List[DocumentChunk]:
        """Markdown 문서를 섹션 단위로 청킹"""
        chunks = []
        
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Markdown을 HTML로 변환
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, 'html.parser')
            
            # 헤딩 기준으로 섹션 분할
            sections = self._split_by_headings(soup)
            
            for section_idx, (heading, section_content) in enumerate(sections):
                chunks_from_section = self._split_text_into_chunks(
                    section_content,
                    {
                        'source': str(md_path),
                        'section': heading,
                        'section_idx': section_idx,
                        'type': 'markdown'
                    }
                )
                chunks.extend(chunks_from_section)
                
        except Exception as e:
            logger.error(f"Error processing Markdown {md_path}: {e}")
            
        return chunks
    
    def chunk_text(self, text_path: Path) -> List[DocumentChunk]:
        """일반 텍스트 파일 청킹"""
        chunks = []
        
        try:
            with open(text_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            chunks = self._split_text_into_chunks(
                content,
                {
                    'source': str(text_path),
                    'type': 'text'
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing text file {text_path}: {e}")
            
        return chunks
    
    def _split_text_into_chunks(self, 
                                text: str, 
                                base_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """텍스트를 청크로 분할"""
        
        # 문장 단위로 분할
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            # 현재 청크가 크기 제한을 초과하면 새 청크 시작
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                
                if len(chunk_text.split()) >= self.min_chunk_size:
                    chunk_id = f"{base_metadata['source']}_{len(chunks)}"
                    
                    chunk = DocumentChunk(
                        content=chunk_text,
                        metadata={
                            **base_metadata,
                            'chunk_idx': len(chunks),
                            'word_count': len(chunk_text.split())
                        },
                        chunk_id=chunk_id
                    )
                    chunks.append(chunk)
                
                # 오버랩 처리
                if self.chunk_overlap > 0:
                    overlap_sentences = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_sentences
                    current_size = sum(len(s.split()) for s in overlap_sentences)
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # 마지막 청크 처리
        if current_chunk and len(' '.join(current_chunk).split()) >= self.min_chunk_size:
            chunk_text = ' '.join(current_chunk)
            chunk_id = f"{base_metadata['source']}_{len(chunks)}"
            
            chunk = DocumentChunk(
                content=chunk_text,
                metadata={
                    **base_metadata,
                    'chunk_idx': len(chunks),
                    'word_count': len(chunk_text.split())
                },
                chunk_id=chunk_id
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """텍스트를 문장으로 분할"""
        # 간단한 문장 분할 (실제로는 더 정교한 방법 필요)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_by_headings(self, soup: BeautifulSoup) -> List[Tuple[str, str]]:
        """HTML을 헤딩 기준으로 섹션 분할"""
        sections = []
        current_heading = "Introduction"
        current_content = []
        
        for element in soup.children:
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                # 이전 섹션 저장
                if current_content:
                    content = ' '.join(str(e) for e in current_content)
                    sections.append((current_heading, content))
                
                current_heading = element.get_text().strip()
                current_content = []
            else:
                current_content.append(element.get_text())
        
        # 마지막 섹션 저장
        if current_content:
            content = ' '.join(str(e) for e in current_content)
            sections.append((current_heading, content))
        
        return sections 