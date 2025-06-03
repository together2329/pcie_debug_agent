import re
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    import PyPDF2
    from PyPDF2 import PdfReader
    PYMUPDF_AVAILABLE = False
import markdown
from bs4 import BeautifulSoup
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Class representing a chunk of a document"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str

class DocumentChunker:
    """Processes and chunks documents into smaller pieces with enhanced PDF parsing"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        """Initialize document chunker with optimized parameters"""
        self.chunk_size = chunk_size  # Increased from 500 to 1000 for better context
        self.chunk_overlap = chunk_overlap  # Increased overlap for 1000-word chunks (15%)
        self.min_chunk_size = 100  # Increased minimum for quality chunks
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"DocumentChunker initialized with PyMuPDF: {PYMUPDF_AVAILABLE}")
    
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
        """Read PDF file content with enhanced PyMuPDF parsing"""
        if PYMUPDF_AVAILABLE:
            return self._read_pdf_pymupdf(file_path)
        else:
            return self._read_pdf_pypdf2(file_path)
    
    def _read_pdf_pymupdf(self, file_path: Path) -> str:
        """Read PDF using PyMuPDF for superior text extraction"""
        text = []
        try:
            doc = fitz.open(str(file_path))
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                # Enhanced text extraction with better formatting
                page_text = page.get_text("text", flags=fitz.TEXTFLAGS_TEXT)
                
                # Clean up common PDF artifacts
                page_text = self._clean_pdf_text(page_text)
                
                if page_text.strip():  # Only add non-empty pages
                    text.append(page_text)
            doc.close()
            self.logger.debug(f"PyMuPDF extracted {len(text)} pages from {file_path.name}")
        except Exception as e:
            self.logger.error(f"PyMuPDF failed for {file_path}: {e}")
            # Fallback to PyPDF2 if available
            return self._read_pdf_pypdf2(file_path)
        
        return '\n\n'.join(text)
    
    def _read_pdf_pypdf2(self, file_path: Path) -> str:
        """Fallback PDF reading using PyPDF2"""
        text = []
        try:
            with open(file_path, 'rb') as f:
                if PYMUPDF_AVAILABLE:
                    # This shouldn't happen, but just in case
                    return ""  
                else:
                    pdf = PdfReader(f)
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text.append(page_text)
            self.logger.debug(f"PyPDF2 extracted {len(text)} pages from {file_path.name}")
        except Exception as e:
            self.logger.error(f"PyPDF2 failed for {file_path}: {e}")
            return ""
        
        return '\n\n'.join(text)
    
    def _clean_pdf_text(self, text: str) -> str:
        """Clean PDF text artifacts and improve readability"""
        if not text:
            return ""
        
        # Remove excessive whitespace while preserving paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words split across lines
        text = re.sub(r'\n([a-z])', r' \1', text)  # Join lines that don't start with capital
        
        # Preserve technical terms and PCIe-specific formatting
        text = re.sub(r'P C I e', 'PCIe', text)  # Fix separated PCIe
        text = re.sub(r'F L R', 'FLR', text)  # Fix separated FLR
        text = re.sub(r'C R S', 'CRS', text)  # Fix separated CRS
        text = re.sub(r'L T S S M', 'LTSSM', text)  # Fix separated LTSSM
        
        return text.strip()
    
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
        """Split content into overlapping chunks with smart boundary detection"""
        chunks = []
        
        # Enhanced sentence splitting with better boundary detection
        sentences = self._smart_sentence_split(content)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence.split())
            
            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk with smart boundary preservation
                chunk_content = ' '.join(current_chunk)
                
                # Ensure chunk meets minimum size requirement
                if current_length >= self.min_chunk_size:
                    chunk = DocumentChunk(
                        content=chunk_content,
                        metadata={**metadata, 'word_count': current_length, 'chunk_index': len(chunks)},
                        chunk_id=f"{metadata['file_name']}_{len(chunks)}"
                    )
                    chunks.append(chunk)
                
                # Smart overlap: preserve complete sentences and technical terms
                current_chunk, current_length = self._create_smart_overlap(current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add the last chunk if there's significant content
        if current_chunk and current_length >= self.min_chunk_size:
            chunk_content = ' '.join(current_chunk)
            chunk = DocumentChunk(
                content=chunk_content,
                metadata={**metadata, 'word_count': current_length, 'chunk_index': len(chunks)},
                chunk_id=f"{metadata['file_name']}_{len(chunks)}"
            )
            chunks.append(chunk)
        
        self.logger.debug(f"Created {len(chunks)} chunks from {metadata['file_name']}")
        return chunks
    
    def _smart_sentence_split(self, content: str) -> List[str]:
        """Enhanced sentence splitting that preserves technical terms and formatting"""
        # Protect technical terms from being split
        protected_terms = [
            r'PCIe\s+(?:Gen\d+|\d\.\d)',  # PCIe Gen3, PCIe 4.0
            r'\b(?:FLR|CRS|LTSSM|AER|MSI-X|TLP|DLLP)\b',  # Technical acronyms
            r'\b0x[0-9A-Fa-f]+\b',  # Hex values
            r'\b\d+(?:\.\d+)?\s*(?:GT/s|Gbps|MHz|GHz)\b',  # Speed/frequency values
            r'\bChapter\s+\d+(?:\.\d+)*\b',  # Chapter references
        ]
        
        # First, protect technical terms by replacing with placeholders
        protected_map = {}
        for i, pattern in enumerate(protected_terms):
            for match in re.finditer(pattern, content, re.IGNORECASE):
                placeholder = f"__PROTECTED_{i}_{len(protected_map)}__"
                protected_map[placeholder] = match.group(0)
                content = content.replace(match.group(0), placeholder, 1)
        
        # Split into sentences, being careful with abbreviations
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', content)
        
        # Restore protected terms
        for i, sentence in enumerate(sentences):
            for placeholder, original in protected_map.items():
                sentence = sentence.replace(placeholder, original)
            sentences[i] = sentence
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_smart_overlap(self, current_chunk: List[str]) -> Tuple[List[str], int]:
        """Create intelligent overlap that preserves context and technical terms"""
        if not current_chunk or self.chunk_overlap <= 0:
            return [], 0
        
        # Calculate overlap in words, not sentences
        total_words = sum(len(sentence.split()) for sentence in current_chunk)
        target_overlap_words = min(self.chunk_overlap, total_words // 2)
        
        overlap_sentences = []
        overlap_word_count = 0
        
        # Work backwards to find sentences that create good overlap
        for sentence in reversed(current_chunk):
            sentence_words = len(sentence.split())
            if overlap_word_count + sentence_words <= target_overlap_words:
                overlap_sentences.insert(0, sentence)
                overlap_word_count += sentence_words
            else:
                # If this sentence would exceed target, check if it contains technical terms
                if self._contains_technical_terms(sentence) and overlap_word_count < target_overlap_words * 0.5:
                    overlap_sentences.insert(0, sentence)
                    overlap_word_count += sentence_words
                break
        
        return overlap_sentences, overlap_word_count
    
    def _contains_technical_terms(self, text: str) -> bool:
        """Check if text contains important technical terms that should be preserved"""
        technical_patterns = [
            r'\b(?:PCIe|FLR|CRS|LTSSM|AER|MSI-X|TLP|DLLP|Gen\d+)\b',
            r'\b0x[0-9A-Fa-f]+\b',  # Hex values
            r'\b\d+(?:\.\d+)?\s*(?:GT/s|Gbps|MHz|GHz)\b',  # Technical measurements
            r'\b(?:completion|timeout|error|status|register|capability)\b',  # Key concepts
        ]
        
        for pattern in technical_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
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
                if PYMUPDF_AVAILABLE:
                    # Use PyMuPDF for better extraction
                    doc = fitz.open(str(pdf_path))
                    for page_num in range(doc.page_count):
                        page = doc.load_page(page_num)
                        text = page.get_text("text")
                        text = self._clean_pdf_text(text)
                        
                        # Split page text into chunks
                        page_chunks = self._split_text_into_chunks(
                            text,
                            {
                                'source': str(pdf_path),
                                'page': page_num + 1,
                                'type': 'pdf'
                            }
                        )
                        chunks.extend(page_chunks)
                    doc.close()
                else:
                    # Fallback to PyPDF2
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