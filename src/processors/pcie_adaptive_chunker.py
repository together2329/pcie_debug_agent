"""
PCIe-Optimized Adaptive Document Chunker

Implements intelligent chunking strategy specifically designed for PCIe specifications:
- Respects semantic boundaries (headers, procedures)
- Maintains technical coherence 
- Optimizes for both context and retrieval precision
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
from bs4 import BeautifulSoup
import markdown

logger = logging.getLogger(__name__)

@dataclass
class PCIeChunk:
    """PCIe-specific document chunk with enhanced metadata"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    semantic_type: str  # 'header_section', 'procedure', 'specification', 'example'
    technical_level: int  # 1=overview, 2=detailed, 3=implementation
    pcie_concepts: List[str]  # Extracted PCIe concepts/terms

class PCIeAdaptiveChunker:
    """
    Adaptive chunker optimized for PCIe specification documents
    
    Strategy:
    1. Semantic boundaries first (headers, procedures)
    2. Size constraints second (target 1000 words)
    3. PCIe-specific intelligence (preserve technical coherence)
    """
    
    def __init__(self, 
                 target_size: int = 1000,
                 max_size: int = 1500, 
                 min_size: int = 200,
                 overlap_size: int = 200):
        """Initialize PCIe adaptive chunker"""
        self.target_size = target_size
        self.max_size = max_size
        self.min_size = min_size
        self.overlap_size = overlap_size
        self.logger = logging.getLogger(__name__)
        
        # PCIe-specific patterns
        self.pcie_concepts = {
            'ltssm_states': r'\b(Detect|Polling|Configuration|L0|L0s|L1|L2|Recovery|Hot Reset|Loopback|Disabled)\b',
            'tlp_types': r'\b(Memory Read|Memory Write|IO Read|IO Write|Configuration|Message|Completion)\b',
            'error_types': r'\b(AER|Correctable|Uncorrectable|Fatal|Non-Fatal|ECRC|LCRC)\b',
            'power_states': r'\b(ASPM|L0|L0s|L1|L2|L3|PME|D0|D1|D2|D3)\b',
            'flow_control': r'\b(Credit|FC|Posted|Non-Posted|Completion|InitFC|UpdateFC)\b',
            'link_training': r'\b(TS1|TS2|SKP|COM|PAD|STP|SDP|END|EDB)\b'
        }
        
        self.header_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headers
            r'^(.+)\n[=-]{3,}$',  # Underlined headers
            r'^\d+\.\d*\s+(.+)$',  # Numbered sections
            r'^[A-Z\s]{10,}$'  # ALL CAPS headers
        ]
        
        self.procedure_indicators = [
            'algorithm', 'procedure', 'steps', 'process', 'method',
            'implementation', 'sequence', 'workflow', 'protocol'
        ]
    
    def chunk_pcie_document(self, file_path: Path) -> List[PCIeChunk]:
        """Main entry point for PCIe document chunking"""
        try:
            content = self._read_file(file_path)
            if not content:
                return []
            
            # Extract base metadata
            base_metadata = self._extract_metadata(file_path, content)
            
            # Perform adaptive chunking
            chunks = self._adaptive_chunk_content(content, base_metadata)
            
            self.logger.info(f"Created {len(chunks)} PCIe-optimized chunks from {file_path}")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error chunking PCIe document {file_path}: {str(e)}")
            return []
    
    def _read_file(self, file_path: Path) -> Optional[str]:
        """Read file content based on type"""
        try:
            if file_path.suffix.lower() == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                self.logger.warning(f"Unsupported file type for PCIe chunker: {file_path.suffix}")
                return None
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            return None
    
    def _extract_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Extract enhanced metadata for PCIe documents"""
        metadata = {
            'file_name': file_path.name,
            'file_type': file_path.suffix[1:],
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'last_modified': file_path.stat().st_mtime,
            'chunking_strategy': 'pcie_adaptive'
        }
        
        # Extract PCIe document type
        filename_lower = file_path.name.lower()
        if 'physical' in filename_lower:
            metadata['pcie_layer'] = 'physical'
        elif 'transaction' in filename_lower or 'tlp' in filename_lower:
            metadata['pcie_layer'] = 'transaction'
        elif 'data_link' in filename_lower:
            metadata['pcie_layer'] = 'data_link'
        elif 'power' in filename_lower:
            metadata['pcie_layer'] = 'power_management'
        elif 'system' in filename_lower or 'architecture' in filename_lower:
            metadata['pcie_layer'] = 'system_architecture'
        elif 'software' in filename_lower:
            metadata['pcie_layer'] = 'software_interface'
        else:
            metadata['pcie_layer'] = 'general'
        
        # Extract title
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            metadata['title'] = title_match.group(1).strip()
        
        return metadata
    
    def _adaptive_chunk_content(self, content: str, base_metadata: Dict[str, Any]) -> List[PCIeChunk]:
        """Adaptive chunking with PCIe-specific intelligence"""
        
        # Step 1: Parse document structure
        sections = self._parse_document_structure(content)
        
        # Step 2: Apply adaptive chunking rules
        chunks = []
        for section in sections:
            section_chunks = self._chunk_section_adaptively(section, base_metadata, len(chunks))
            chunks.extend(section_chunks)
        
        # Step 3: Post-process for optimization
        optimized_chunks = self._optimize_chunks(chunks)
        
        return optimized_chunks
    
    def _parse_document_structure(self, content: str) -> List[Dict[str, Any]]:
        """Parse document into semantic sections"""
        sections = []
        
        # Split by headers first
        lines = content.split('\n')
        current_section = {
            'header': 'Document Start',
            'content': [],
            'level': 0,
            'line_start': 0
        }
        
        for i, line in enumerate(lines):
            header_match = self._detect_header(line)
            if header_match:
                # Save current section
                if current_section['content']:
                    current_section['content'] = '\n'.join(current_section['content'])
                    current_section['line_end'] = i
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    'header': header_match['text'],
                    'content': [],
                    'level': header_match['level'],
                    'line_start': i
                }
            else:
                current_section['content'].append(line)
        
        # Add final section
        if current_section['content']:
            current_section['content'] = '\n'.join(current_section['content'])
            current_section['line_end'] = len(lines)
            sections.append(current_section)
        
        return sections
    
    def _detect_header(self, line: str) -> Optional[Dict[str, Any]]:
        """Detect if line is a header and extract info"""
        line = line.strip()
        
        # Markdown headers
        md_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if md_match:
            return {
                'text': md_match.group(2),
                'level': len(md_match.group(1)),
                'type': 'markdown'
            }
        
        # Numbered sections
        num_match = re.match(r'^(\d+\.[\d\.]*)\s+(.+)$', line)
        if num_match:
            level = len(num_match.group(1).split('.'))
            return {
                'text': num_match.group(2),
                'level': level,
                'type': 'numbered'
            }
        
        # ALL CAPS headers (common in specs)
        if len(line) > 10 and line.isupper() and not re.search(r'[.!?]$', line):
            return {
                'text': line,
                'level': 2,
                'type': 'caps'
            }
        
        return None
    
    def _chunk_section_adaptively(self, section: Dict[str, Any], base_metadata: Dict[str, Any], chunk_offset: int) -> List[PCIeChunk]:
        """Apply adaptive chunking to a single section"""
        content = section['content']
        word_count = len(content.split())
        
        # Small sections: keep as single chunk (if above minimum)
        if word_count <= self.max_size and word_count >= self.min_size:
            return [self._create_pcie_chunk(
                content=content,
                section=section,
                base_metadata=base_metadata,
                chunk_index=chunk_offset,
                is_complete_section=True
            )]
        
        # Large sections: split intelligently
        elif word_count > self.max_size:
            return self._split_large_section(section, base_metadata, chunk_offset)
        
        # Tiny sections: will be merged in optimization step
        else:
            return [self._create_pcie_chunk(
                content=content,
                section=section,
                base_metadata=base_metadata,
                chunk_index=chunk_offset,
                is_complete_section=True,
                needs_merge=True
            )]
    
    def _split_large_section(self, section: Dict[str, Any], base_metadata: Dict[str, Any], chunk_offset: int) -> List[PCIeChunk]:
        """Split large sections while preserving semantic coherence"""
        content = section['content']
        chunks = []
        
        # Try to split by sub-sections first
        subsections = self._find_subsections(content)
        
        if len(subsections) > 1:
            # Split by subsections
            for i, subsection in enumerate(subsections):
                if len(subsection.split()) >= self.min_size:
                    chunk = self._create_pcie_chunk(
                        content=subsection,
                        section=section,
                        base_metadata=base_metadata,
                        chunk_index=chunk_offset + i,
                        subsection_index=i
                    )
                    chunks.append(chunk)
        else:
            # Split by sentences with intelligent breaks
            chunks = self._split_by_sentences_intelligent(content, section, base_metadata, chunk_offset)
        
        return chunks
    
    def _find_subsections(self, content: str) -> List[str]:
        """Find natural subsections within content"""
        # Look for sub-headers, procedures, examples
        lines = content.split('\n')
        subsections = []
        current_subsection = []
        
        for line in lines:
            line = line.strip()
            
            # Check for subsection indicators
            if (self._is_subsection_break(line) and current_subsection):
                subsections.append('\n'.join(current_subsection))
                current_subsection = [line]
            else:
                current_subsection.append(line)
        
        # Add final subsection
        if current_subsection:
            subsections.append('\n'.join(current_subsection))
        
        return subsections
    
    def _is_subsection_break(self, line: str) -> bool:
        """Detect if line indicates a subsection break"""
        line_lower = line.lower()
        
        # Sub-headers
        if re.match(r'^#{2,6}\s+', line) or re.match(r'^\d+\.\d+', line):
            return True
        
        # Procedure indicators
        for indicator in self.procedure_indicators:
            if indicator in line_lower:
                return True
        
        # Lists/enumerations
        if re.match(r'^\s*[•\-\*]\s+', line) or re.match(r'^\s*\d+\.\s+', line):
            return True
        
        return False
    
    def _split_by_sentences_intelligent(self, content: str, section: Dict[str, Any], base_metadata: Dict[str, Any], chunk_offset: int) -> List[PCIeChunk]:
        """Split by sentences with PCIe-aware intelligence"""
        sentences = re.split(r'(?<=[.!?])\s+', content)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_size = len(sentence.split())
            
            # Check if adding sentence would exceed target
            if current_size + sentence_size > self.target_size and current_chunk:
                # Look for good break point
                break_point = self._find_intelligent_break_point(current_chunk, sentence)
                
                if break_point >= 0:
                    # Create chunk up to break point
                    chunk_content = ' '.join(current_chunk[:break_point + 1])
                    chunk = self._create_pcie_chunk(
                        content=chunk_content,
                        section=section,
                        base_metadata=base_metadata,
                        chunk_index=chunk_offset + len(chunks)
                    )
                    chunks.append(chunk)
                    
                    # Continue with overlap
                    overlap_start = max(0, break_point + 1 - self.overlap_size // 10)  # Approximate sentence overlap
                    current_chunk = current_chunk[overlap_start:] + [sentence]
                    current_size = sum(len(s.split()) for s in current_chunk)
                else:
                    # No good break point, force split
                    chunk_content = ' '.join(current_chunk)
                    chunk = self._create_pcie_chunk(
                        content=chunk_content,
                        section=section,
                        base_metadata=base_metadata,
                        chunk_index=chunk_offset + len(chunks)
                    )
                    chunks.append(chunk)
                    current_chunk = [sentence]
                    current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            if len(chunk_content.split()) >= self.min_size:
                chunk = self._create_pcie_chunk(
                    content=chunk_content,
                    section=section,
                    base_metadata=base_metadata,
                    chunk_index=chunk_offset + len(chunks)
                )
                chunks.append(chunk)
        
        return chunks
    
    def _find_intelligent_break_point(self, sentences: List[str], next_sentence: str) -> int:
        """Find the best break point considering PCIe semantics"""
        if len(sentences) < 2:
            return -1
        
        # Prefer breaks after complete thoughts
        for i in reversed(range(len(sentences))):
            sentence = sentences[i].lower()
            
            # Good break points
            if any(phrase in sentence for phrase in [
                'therefore', 'thus', 'in conclusion', 'as a result',
                'the following', 'for example', 'note that'
            ]):
                return i
            
            # Bad break points (avoid splitting)
            if any(phrase in sentence for phrase in [
                'step', 'first', 'second', 'next', 'then',
                'refer to', 'see section', 'as shown'
            ]):
                continue
        
        # Default: split at 2/3 point
        return int(len(sentences) * 0.67)
    
    def _create_pcie_chunk(self, 
                          content: str, 
                          section: Dict[str, Any], 
                          base_metadata: Dict[str, Any], 
                          chunk_index: int,
                          subsection_index: Optional[int] = None,
                          is_complete_section: bool = False,
                          needs_merge: bool = False) -> PCIeChunk:
        """Create a PCIe-optimized chunk with enhanced metadata"""
        
        # Analyze content for PCIe concepts
        pcie_concepts = self._extract_pcie_concepts(content)
        semantic_type = self._classify_semantic_type(content, section)
        technical_level = self._assess_technical_level(content)
        
        # Build chunk metadata
        chunk_metadata = {
            **base_metadata,
            'section_header': section['header'],
            'section_level': section['level'],
            'chunk_index': chunk_index,
            'word_count': len(content.split()),
            'is_complete_section': is_complete_section,
            'needs_merge': needs_merge,
            'semantic_type': semantic_type,
            'technical_level': technical_level,
            'pcie_concepts': pcie_concepts
        }
        
        if subsection_index is not None:
            chunk_metadata['subsection_index'] = subsection_index
        
        chunk_id = f"pcie_{base_metadata['file_name']}_{chunk_index}"
        
        return PCIeChunk(
            content=content,
            metadata=chunk_metadata,
            chunk_id=chunk_id,
            semantic_type=semantic_type,
            technical_level=technical_level,
            pcie_concepts=pcie_concepts
        )
    
    def _extract_pcie_concepts(self, content: str) -> List[str]:
        """Extract PCIe-specific concepts from content"""
        concepts = []
        content_lower = content.lower()
        
        for concept_type, pattern in self.pcie_concepts.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                concepts.extend([f"{concept_type}:{match}" for match in matches])
        
        return list(set(concepts))  # Remove duplicates
    
    def _classify_semantic_type(self, content: str, section: Dict[str, Any]) -> str:
        """Classify the semantic type of content"""
        content_lower = content.lower()
        header_lower = section['header'].lower()
        
        # Check for specific patterns
        if any(word in header_lower for word in ['example', 'illustration', 'figure']):
            return 'example'
        elif any(word in content_lower for word in self.procedure_indicators):
            return 'procedure'
        elif any(word in header_lower for word in ['specification', 'format', 'structure']):
            return 'specification'
        elif section['level'] <= 2:
            return 'header_section'
        else:
            return 'content'
    
    def _assess_technical_level(self, content: str) -> int:
        """Assess technical complexity level (1=basic, 2=intermediate, 3=advanced)"""
        content_lower = content.lower()
        
        # Count technical indicators
        advanced_terms = ['implementation', 'algorithm', 'protocol', 'specification', 'compliance']
        intermediate_terms = ['procedure', 'method', 'process', 'configuration', 'register']
        basic_terms = ['overview', 'introduction', 'summary', 'definition', 'example']
        
        advanced_count = sum(1 for term in advanced_terms if term in content_lower)
        intermediate_count = sum(1 for term in intermediate_terms if term in content_lower)
        basic_count = sum(1 for term in basic_terms if term in content_lower)
        
        if advanced_count > intermediate_count and advanced_count > basic_count:
            return 3
        elif intermediate_count > basic_count:
            return 2
        else:
            return 1
    
    def _optimize_chunks(self, chunks: List[PCIeChunk]) -> List[PCIeChunk]:
        """Post-process chunks for optimization"""
        optimized = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # Check if current chunk needs merging
            if (hasattr(current_chunk.metadata, 'needs_merge') and 
                current_chunk.metadata.get('needs_merge', False) and
                i < len(chunks) - 1):
                
                next_chunk = chunks[i + 1]
                merged_chunk = self._merge_chunks(current_chunk, next_chunk)
                optimized.append(merged_chunk)
                i += 2  # Skip next chunk as it's been merged
            else:
                optimized.append(current_chunk)
                i += 1
        
        return optimized
    
    def _merge_chunks(self, chunk1: PCIeChunk, chunk2: PCIeChunk) -> PCIeChunk:
        """Merge two chunks intelligently"""
        merged_content = chunk1.content + "\n\n" + chunk2.content
        
        # Combine metadata
        merged_metadata = chunk1.metadata.copy()
        merged_metadata['word_count'] = len(merged_content.split())
        merged_metadata['is_merged'] = True
        merged_metadata['merged_from'] = [chunk1.chunk_id, chunk2.chunk_id]
        
        # Combine PCIe concepts
        merged_concepts = list(set(chunk1.pcie_concepts + chunk2.pcie_concepts))
        
        return PCIeChunk(
            content=merged_content,
            metadata=merged_metadata,
            chunk_id=f"merged_{chunk1.chunk_id}_{chunk2.chunk_id}",
            semantic_type=chunk1.semantic_type,  # Keep first chunk's type
            technical_level=max(chunk1.technical_level, chunk2.technical_level),
            pcie_concepts=merged_concepts
        )