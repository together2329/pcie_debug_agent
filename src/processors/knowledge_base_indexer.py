"""
Knowledge Base Indexer for PCIe Debug Agent
Processes and indexes PCIe documentation into the vector store
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import hashlib
import json
import re
from datetime import datetime

from src.processors.document_chunker import DocumentChunker
from src.vectorstore.faiss_store import FAISSVectorStore
from src.models.model_manager import ModelManager
from src.config.settings import Settings

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of documentation"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    source_file: str
    section: Optional[str] = None
    subsection: Optional[str] = None

@dataclass
class IndexingStats:
    """Statistics for indexing process"""
    total_files: int = 0
    total_chunks: int = 0
    successful_chunks: int = 0
    failed_chunks: int = 0
    processing_time: float = 0.0
    embedding_time: float = 0.0

class KnowledgeBaseIndexer:
    """Indexes PCIe knowledge base documents into vector store"""
    
    def __init__(self, 
                 settings: Settings,
                 vector_store: FAISSVectorStore,
                 model_manager: ModelManager):
        """
        Initialize the knowledge base indexer
        
        Args:
            settings: Application settings
            vector_store: FAISS vector store instance
            model_manager: Model manager for embeddings
        """
        self.settings = settings
        self.vector_store = vector_store
        self.model_manager = model_manager
        
        # Initialize document chunker
        self.chunker = DocumentChunker(
            chunk_size=settings.rag.chunk_size,
            chunk_overlap=settings.rag.chunk_overlap
        )
        
        # Knowledge base directory
        self.kb_dir = Path("data/knowledge_base")
        
        # Statistics
        self.stats = IndexingStats()
        
        # Content categories for metadata
        self.content_categories = {
            'error_scenarios': ['error', 'failure', 'problem', 'issue', 'debug'],
            'ltssm_states': ['ltssm', 'link training', 'state machine', 'polling', 'configuration'],
            'tlp_analysis': ['tlp', 'transaction layer', 'packet', 'header', 'completion'],
            'signal_integrity': ['signal integrity', 'si', 'eye diagram', 'jitter', 'impedance'],
            'power_management': ['power', 'aspm', 'd-state', 'l-state', 'wake'],
            'aer_handling': ['aer', 'advanced error reporting', 'correctable', 'uncorrectable']
        }
    
    def index_knowledge_base(self, force_reindex: bool = False) -> IndexingStats:
        """
        Index the entire knowledge base
        
        Args:
            force_reindex: Force reindexing even if already indexed
            
        Returns:
            Indexing statistics
        """
        start_time = datetime.now()
        logger.info("Starting knowledge base indexing")
        
        try:
            # Create knowledge base directory if it doesn't exist
            self.kb_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all markdown files in knowledge base
            md_files = list(self.kb_dir.glob("*.md"))
            self.stats.total_files = len(md_files)
            
            if not md_files:
                logger.warning("No markdown files found in knowledge base directory")
                return self.stats
            
            logger.info(f"Found {len(md_files)} knowledge base files")
            
            # Process each file
            all_chunks = []
            all_embeddings = []
            all_metadata = []
            
            for file_path in md_files:
                logger.info(f"Processing {file_path.name}")
                
                try:
                    chunks = self._process_document(file_path)
                    if chunks:
                        # Generate embeddings for this file's chunks
                        chunk_texts = [chunk.content for chunk in chunks]
                        embeddings = self._generate_embeddings(chunk_texts)
                        
                        # Collect for batch insertion
                        all_chunks.extend(chunk_texts)
                        all_embeddings.extend(embeddings)
                        all_metadata.extend([chunk.metadata for chunk in chunks])
                        
                        self.stats.successful_chunks += len(chunks)
                        logger.info(f"Successfully processed {len(chunks)} chunks from {file_path.name}")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {str(e)}")
                    self.stats.failed_chunks += 1
            
            # Add all chunks to vector store in one batch
            if all_chunks:
                logger.info(f"Adding {len(all_chunks)} chunks to vector store")
                self.vector_store.add_documents(all_embeddings, all_chunks, all_metadata)
                
                # Save the vector store
                if hasattr(self.vector_store, 'save'):
                    vector_store_dir = self.settings.data_dir / "vectorstore"
                    logger.info(f"Saving vector store to {vector_store_dir}")
                    self.vector_store.save(str(vector_store_dir))
                
                logger.info("Successfully added all chunks to vector store")
            
            # Update statistics
            self.stats.total_chunks = len(all_chunks)
            self.stats.processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Knowledge base indexing completed in {self.stats.processing_time:.2f}s")
            logger.info(f"Indexed {self.stats.successful_chunks} chunks from {self.stats.total_files} files")
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Error during knowledge base indexing: {str(e)}")
            raise
    
    def _process_document(self, file_path: Path) -> List[DocumentChunk]:
        """
        Process a single document into chunks
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of document chunks
        """
        try:
            # Read document content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata from document
            metadata = self._extract_document_metadata(file_path, content)
            
            # Split into sections based on headers
            sections = self._split_into_sections(content)
            
            chunks = []
            for section_title, section_content in sections:
                # Simple chunking by splitting on sentences and grouping
                section_chunks_text = self._simple_chunk_text(section_content)
                
                for i, chunk_text in enumerate(section_chunks_text):
                    # Create chunk metadata
                    chunk_metadata = {
                        **metadata,
                        'section': section_title,
                        'chunk_index': i,
                        'chunk_id': self._generate_chunk_id(file_path, section_title, i),
                        'content_length': len(chunk_text),
                        'content_categories': self._categorize_content(chunk_text)
                    }
                    
                    chunk = DocumentChunk(
                        content=chunk_text,
                        metadata=chunk_metadata,
                        chunk_id=chunk_metadata['chunk_id'],
                        source_file=str(file_path),
                        section=section_title
                    )
                    
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return []
    
    def _extract_document_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """
        Extract metadata from document
        
        Args:
            file_path: Path to the document
            content: Document content
            
        Returns:
            Document metadata
        """
        # Basic metadata
        metadata = {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_size': len(content),
            'indexed_at': datetime.now().isoformat(),
            'document_type': 'knowledge_base',
            'source': 'pcie_debug_agent'
        }
        
        # Extract title from first header
        title_match = re.search(r'^#\s+(.+)', content, re.MULTILINE)
        if title_match:
            metadata['title'] = title_match.group(1).strip()
        else:
            metadata['title'] = file_path.stem.replace('_', ' ').title()
        
        # Count sections and subsections
        headers = re.findall(r'^#+\s+(.+)', content, re.MULTILINE)
        metadata['section_count'] = len([h for h in headers if h.startswith('#')])
        metadata['subsection_count'] = len([h for h in headers if h.startswith('##')])
        
        # Extract key topics/tags
        metadata['topics'] = self._extract_topics(content)
        
        # Document complexity score
        metadata['complexity_score'] = self._calculate_complexity_score(content)
        
        return metadata
    
    def _split_into_sections(self, content: str) -> List[tuple]:
        """
        Split document into sections based on headers
        
        Args:
            content: Document content
            
        Returns:
            List of (section_title, section_content) tuples
        """
        sections = []
        current_section = ""
        current_title = "Introduction"
        
        lines = content.split('\n')
        
        for line in lines:
            # Check if this is a header
            header_match = re.match(r'^(#+)\s+(.+)', line)
            
            if header_match:
                # Save previous section if it has content
                if current_section.strip():
                    sections.append((current_title, current_section.strip()))
                
                # Start new section
                header_level = len(header_match.group(1))
                current_title = header_match.group(2).strip()
                current_section = ""
            else:
                # Add line to current section
                current_section += line + '\n'
        
        # Add the last section
        if current_section.strip():
            sections.append((current_title, current_section.strip()))
        
        return sections
    
    def _generate_chunk_id(self, file_path: Path, section: str, chunk_index: int) -> str:
        """
        Generate unique chunk ID
        
        Args:
            file_path: Source file path
            section: Section title
            chunk_index: Chunk index within section
            
        Returns:
            Unique chunk ID
        """
        base_string = f"{file_path.stem}_{section}_{chunk_index}"
        return hashlib.md5(base_string.encode()).hexdigest()[:12]
    
    def _categorize_content(self, content: str) -> List[str]:
        """
        Categorize content based on keywords
        
        Args:
            content: Content to categorize
            
        Returns:
            List of content categories
        """
        content_lower = content.lower()
        categories = []
        
        for category, keywords in self.content_categories.items():
            if any(keyword in content_lower for keyword in keywords):
                categories.append(category)
        
        return categories
    
    def _extract_topics(self, content: str) -> List[str]:
        """
        Extract key topics from content
        
        Args:
            content: Document content
            
        Returns:
            List of topics
        """
        # Extract terms from headers
        headers = re.findall(r'^#+\s+(.+)', content, re.MULTILINE)
        topics = []
        
        for header in headers:
            # Clean up header text
            clean_header = re.sub(r'[^\w\s]', '', header).strip()
            if clean_header:
                topics.append(clean_header.lower())
        
        # Extract common PCIe terms
        pcie_terms = [
            'pcie', 'tlp', 'dllp', 'ltssm', 'aer', 'aspm',
            'link training', 'completion timeout', 'error recovery',
            'signal integrity', 'power management', 'configuration space'
        ]
        
        content_lower = content.lower()
        for term in pcie_terms:
            if term in content_lower:
                topics.append(term)
        
        return list(set(topics))  # Remove duplicates
    
    def _calculate_complexity_score(self, content: str) -> float:
        """
        Calculate document complexity score
        
        Args:
            content: Document content
            
        Returns:
            Complexity score (0.0 to 1.0)
        """
        score = 0.0
        
        # Length factor
        length_score = min(len(content) / 50000, 1.0)  # Normalize to 50k chars
        score += length_score * 0.3
        
        # Technical term density
        technical_terms = [
            'protocol', 'specification', 'register', 'configuration',
            'algorithm', 'implementation', 'optimization', 'validation'
        ]
        term_count = sum(content.lower().count(term) for term in technical_terms)
        term_density = min(term_count / 100, 1.0)  # Normalize to 100 occurrences
        score += term_density * 0.4
        
        # Code/example density
        code_blocks = len(re.findall(r'```', content)) / 2  # Pairs of code blocks
        example_blocks = len(re.findall(r'example', content, re.IGNORECASE))
        code_score = min((code_blocks + example_blocks) / 20, 1.0)
        score += code_score * 0.3
        
        return min(score, 1.0)
    
    def _simple_chunk_text(self, text: str) -> List[str]:
        """
        Simple text chunking method
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence would exceed chunk size, start new chunk
            if current_length + sentence_length > self.settings.rag.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Keep overlap
                if self.settings.rag.chunk_overlap > 0 and len(current_chunk) > self.settings.rag.chunk_overlap:
                    overlap_sentences = current_chunk[-self.settings.rag.chunk_overlap:]
                    current_chunk = overlap_sentences
                    current_length = sum(len(s.split()) for s in overlap_sentences)
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add the last chunk if there's any content left
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text]
    
    def _generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        start_time = datetime.now()
        
        try:
            # Generate embeddings using model manager
            embeddings = self.model_manager.generate_embeddings(texts)
            
            # Update timing stats
            self.stats.embedding_time += (datetime.now() - start_time).total_seconds()
            
            # Convert to list of arrays
            return [embeddings[i] for i in range(len(texts))]
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def get_indexing_stats(self) -> Dict[str, Any]:
        """
        Get indexing statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_files': self.stats.total_files,
            'total_chunks': self.stats.total_chunks,
            'successful_chunks': self.stats.successful_chunks,
            'failed_chunks': self.stats.failed_chunks,
            'processing_time': self.stats.processing_time,
            'embedding_time': self.stats.embedding_time,
            'chunks_per_second': self.stats.successful_chunks / max(self.stats.processing_time, 1),
            'success_rate': self.stats.successful_chunks / max(self.stats.total_chunks, 1) * 100
        }
    
    def search_knowledge_base(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            # Generate query embedding
            query_embedding = self.model_manager.generate_embeddings([query])[0]
            
            # Search vector store
            results = self.vector_store.search(query_embedding, k)
            
            # Format results
            formatted_results = []
            for content, metadata, score in results:
                formatted_results.append({
                    'content': content,
                    'metadata': metadata,
                    'relevance_score': score,
                    'source': metadata.get('file_name', 'unknown'),
                    'section': metadata.get('section', 'unknown'),
                    'categories': metadata.get('content_categories', [])
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            return []
    
    def validate_index(self) -> Dict[str, Any]:
        """
        Validate the knowledge base index
        
        Returns:
            Validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Check if vector store exists and has content
            if not hasattr(self.vector_store, 'index') or self.vector_store.index is None:
                validation_results['is_valid'] = False
                validation_results['errors'].append("Vector store index not found")
                return validation_results
            
            # Check index size
            index_size = self.vector_store.index.ntotal if self.vector_store.index else 0
            validation_results['statistics']['index_size'] = index_size
            
            if index_size == 0:
                validation_results['is_valid'] = False
                validation_results['errors'].append("Vector store is empty")
                return validation_results
            
            # Test search functionality
            test_queries = [
                "PCIe link training failure",
                "TLP completion timeout error",
                "Signal integrity issues"
            ]
            
            search_results = []
            for query in test_queries:
                results = self.search_knowledge_base(query, k=3)
                search_results.append({
                    'query': query,
                    'result_count': len(results),
                    'top_score': results[0]['relevance_score'] if results else 0.0
                })
            
            validation_results['statistics']['search_tests'] = search_results
            
            # Check for minimum expected content
            expected_files = [
                'pcie_error_scenarios.md',
                'ltssm_states_guide.md',
                'tlp_error_analysis.md',
                'signal_integrity_troubleshooting.md',
                'power_management_issues.md',
                'aer_error_handling.md'
            ]
            
            missing_files = []
            for file_name in expected_files:
                file_path = self.kb_dir / file_name
                if not file_path.exists():
                    missing_files.append(file_name)
            
            if missing_files:
                validation_results['warnings'].append(f"Missing expected files: {missing_files}")
            
            validation_results['statistics']['expected_files'] = len(expected_files)
            validation_results['statistics']['missing_files'] = len(missing_files)
            
            logger.info("Knowledge base validation completed successfully")
            return validation_results
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
            logger.error(f"Error during knowledge base validation: {str(e)}")
            return validation_results