import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
import PyPDF2
import markdown
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

class DocumentChunk:
    """문서 청크 데이터 구조"""
    
    def __init__(self, 
                 content: str,
                 metadata: Dict[str, Any],
                 chunk_id: str):
        self.content = content
        self.metadata = metadata
        self.chunk_id = chunk_id
        self.embedding = None  # 나중에 임베딩 추가
        
    def __repr__(self):
        return f"DocumentChunk(id={self.chunk_id}, size={len(self.content)})"

class DocumentChunker:
    """문서를 의미 단위로 청킹"""
    
    def __init__(self, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
    def chunk_documents(self, doc_path: Path) -> List[DocumentChunk]:
        """문서 타입에 따라 적절한 청킹 방법 선택"""
        
        if doc_path.suffix.lower() == '.pdf':
            return self.chunk_pdf(doc_path)
        elif doc_path.suffix.lower() in ['.md', '.markdown']:
            return self.chunk_markdown(doc_path)
        elif doc_path.suffix.lower() in ['.txt', '.log']:
            return self.chunk_text(doc_path)
        else:
            logger.warning(f"Unsupported document type: {doc_path.suffix}")
            return []
            
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