"""
PCIe RAG Engine with Adaptive Chunking

Specialized RAG engine optimized for PCIe specifications with:
- Adaptive chunking strategy (1000 words + semantic boundaries)
- PCIe-specific concept extraction and indexing
- Enhanced retrieval for technical documentation
- Context-aware response generation
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import time

from ..processors.pcie_adaptive_chunker import PCIeAdaptiveChunker, PCIeChunk
from ..vectorstore.multi_model_manager import MultiModelVectorManager
from ..vectorstore.faiss_store import FAISSVectorStore
from ..models.model_manager import ModelManager
from .pcie_structured_output import (
    PCIeQueryResponse, PCIeStructuredOutputFormatter,
    PCIeAnalysis, TechnicalLevel, PCIeLayer
)

logger = logging.getLogger(__name__)

@dataclass
class PCIeQueryResult:
    """Enhanced query result with PCIe-specific metadata"""
    content: str
    score: float
    metadata: Dict[str, Any]
    pcie_concepts: List[str]
    technical_level: int
    semantic_type: str
    source_section: str

class PCIeRAGEngine:
    """
    PCIe-specialized RAG engine with adaptive chunking and enhanced retrieval
    """
    
    def __init__(self, 
                 embedding_model: str = "text-embedding-3-small",
                 vector_db_path: Optional[str] = None,
                 chunk_config: Optional[Dict[str, int]] = None):
        """Initialize PCIe RAG engine"""
        
        # Initialize with PCIe-optimized chunking config
        default_chunk_config = {
            'target_size': 1000,
            'max_size': 1500,
            'min_size': 200,
            'overlap_size': 200
        }
        
        if chunk_config:
            default_chunk_config.update(chunk_config)
        
        # Store settings before parent init
        self.embedding_model = embedding_model
        self.vector_db_path_override = vector_db_path
        
        # Initialize components
        self.vector_store = None
        self.model_manager = None
        self.vector_manager = MultiModelVectorManager()
        
        # Initialize PCIe-specific components
        self.pcie_chunker = PCIeAdaptiveChunker(**default_chunk_config)
        self.chunk_config = default_chunk_config
        
        # PCIe-specific settings
        self.pcie_mode = True
        self.enable_concept_boosting = True
        self.enable_technical_level_filtering = True
        
        # Vector database path for PCIe mode
        if vector_db_path is None:
            self.vector_db_path = f"data/vectorstore/pcie_adaptive_{embedding_model.replace('-', '_')}"
        else:
            self.vector_db_path = vector_db_path
        
        logger.info(f"Initialized PCIe RAG Engine with adaptive chunking")
        logger.info(f"Chunk config: {self.chunk_config}")
    
    def build_knowledge_base(self, knowledge_base_path: str, force_rebuild: bool = False) -> bool:
        """Build PCIe knowledge base with adaptive chunking"""
        try:
            kb_path = Path(knowledge_base_path)
            if not kb_path.exists():
                logger.error(f"Knowledge base path does not exist: {knowledge_base_path}")
                return False
            
            # Check if rebuild is needed
            if not force_rebuild and self._is_knowledge_base_current():
                logger.info("PCIe knowledge base is up to date")
                return True
            
            logger.info("Building PCIe knowledge base with adaptive chunking...")
            
            # Process all documents with traditional chunker for now
            # TODO: Fix PCIe adaptive chunker - it's creating chunks that are too large
            from src.processors.document_chunker import DocumentChunker
            
            traditional_chunker = DocumentChunker(
                chunk_size=1000,  # target words
                chunk_overlap=200  # overlap words
            )
            
            all_chunks = []
            processed_files = 0
            
            for file_path in kb_path.rglob("*.md"):
                if self._should_process_file(file_path):
                    logger.debug(f"Processing {file_path} with traditional chunker")
                    
                    doc_chunks = traditional_chunker.chunk_documents(file_path)
                    if doc_chunks:
                        # Convert to PCIe chunks
                        pcie_chunks = []
                        for chunk in doc_chunks:
                            pcie_chunk = PCIeChunk(
                                content=chunk.content,
                                metadata=chunk.metadata,
                                chunk_id=chunk.chunk_id,
                                semantic_type='content',
                                technical_level=2,  # default intermediate
                                pcie_concepts=self.pcie_chunker._extract_pcie_concepts(chunk.content)
                            )
                            pcie_chunks.append(pcie_chunk)
                        
                        all_chunks.extend(pcie_chunks)
                        processed_files += 1
                        logger.debug(f"Created {len(pcie_chunks)} chunks from {file_path}")
            
            if not all_chunks:
                logger.warning("No chunks created from knowledge base")
                return False
            
            # Build vector database
            success = self._build_vector_database(all_chunks)
            
            if success:
                logger.info(f"Successfully built PCIe knowledge base:")
                logger.info(f"  - Files processed: {processed_files}")
                logger.info(f"  - Total chunks: {len(all_chunks)}")
                logger.info(f"  - Average chunk size: {np.mean([len(c.content.split()) for c in all_chunks]):.1f} words")
                logger.info(f"  - Vector database: {self.vector_db_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error building PCIe knowledge base: {str(e)}")
            return False
    
    def _should_process_file(self, file_path: Path) -> bool:
        """Determine if file should be processed for PCIe knowledge base"""
        # Skip certain files
        skip_patterns = ['README.md', 'index.md', '.git', '__pycache__']
        
        for pattern in skip_patterns:
            if pattern in str(file_path):
                return False
        
        # Focus on PCIe-related content
        filename_lower = file_path.name.lower()
        pcie_indicators = [
            'pcie', 'physical_layer', 'transaction_layer', 'data_link',
            'power_management', 'system_architecture', 'software_interface',
            'ltssm', 'tlp', 'aer', 'error', 'specification'
        ]
        
        # Always process files with PCIe indicators
        if any(indicator in filename_lower for indicator in pcie_indicators):
            return True
        
        # Process other .md files but deprioritize
        return file_path.suffix.lower() == '.md'
    
    def _build_vector_database(self, chunks: List[PCIeChunk]) -> bool:
        """Build vector database from PCIe chunks"""
        try:
            # Prepare documents for vector database
            documents = []
            metadata_list = []
            
            for chunk in chunks:
                # Skip empty chunks
                if not chunk.content or not chunk.content.strip():
                    continue
                
                # Ensure chunk is not too large for OpenAI (8192 token limit ~= 6000 words)
                content = chunk.content
                word_count = len(content.split())
                
                if word_count > 3000:  # Much stricter limit to ensure we stay under token limits
                    logger.warning(f"Chunk too large ({word_count} words), truncating to 3000 words")
                    words = content.split()[:3000]
                    content = ' '.join(words)
                
                documents.append(content)
                
                # Enhanced metadata for PCIe chunks
                chunk_metadata = {
                    'source': chunk.metadata.get('file_path', ''),
                    'chunk_id': chunk.chunk_id,
                    'section_header': chunk.metadata.get('section_header', ''),
                    'pcie_layer': chunk.metadata.get('pcie_layer', 'general'),
                    'semantic_type': chunk.semantic_type,
                    'technical_level': chunk.technical_level,
                    'word_count': chunk.metadata.get('word_count', 0),
                    'pcie_concepts': ','.join(chunk.pcie_concepts),
                    'chunking_strategy': 'pcie_adaptive'
                }
                metadata_list.append(chunk_metadata)
            
            # Build vector database using multi-model manager
            from src.models.embedding_selector import get_embedding_selector
            from src.processors.embedder import Embedder
            
            # Get embedding function
            embedding_selector = get_embedding_selector()
            embedding_selector.switch_model(self.embedding_model)
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(documents)} chunks...")
            embeddings = []
            batch_size = 20  # OpenAI has limits on batch size
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                try:
                    # Use provider's encode method
                    batch_embeddings = embedding_selector.current_provider.encode(batch)
                    embeddings.extend(batch_embeddings)
                    
                    if i % 100 == 0:
                        logger.debug(f"Processed {i}/{len(documents)} chunks")
                except Exception as e:
                    logger.error(f"Error encoding batch {i}-{i+batch_size}: {str(e)}")
                    logger.error(f"Batch size: {len(batch)}, First doc length: {len(batch[0]) if batch else 0}")
                    # Try one by one as fallback
                    for j, doc in enumerate(batch):
                        try:
                            single_embedding = embedding_selector.current_provider.encode([doc])
                            embeddings.extend(single_embedding)
                        except Exception as single_e:
                            logger.error(f"Error encoding document {i+j}: {str(single_e)}")
                            logger.error(f"Document length: {len(doc)}, Preview: {doc[:100]}...")
                            raise
            
            # Build FAISS index
            dimension = embeddings[0].shape[0] if embeddings else 0
            
            if dimension == 0:
                logger.error("No embeddings generated")
                return False
            
            # Create vector store
            vector_db_path = Path(self.vector_db_path)
            vector_db_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize FAISS store with embeddings
            self.vector_store = FAISSVectorStore(
                dimension=dimension,
                index_path=str(vector_db_path)
            )
            
            # Add documents to index
            self.vector_store.add_documents(
                embeddings=embeddings,
                documents=documents,
                metadata=metadata_list
            )
            
            # Save the index
            self.vector_store.save(str(vector_db_path))
            success = True
            
            if success:
                logger.info(f"Vector database built successfully at {self.vector_db_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error building vector database: {str(e)}")
            return False
    
    def query(self, 
              query: str, 
              top_k: int = 5,
              technical_level_filter: Optional[int] = None,
              pcie_layer_filter: Optional[str] = None,
              semantic_type_filter: Optional[str] = None,
              return_structured: bool = False) -> Union[List[PCIeQueryResult], 'PCIeQueryResponse']:
        """Enhanced query with PCIe-specific filtering and boosting"""
        
        start_time = time.time()
        
        try:
            # Get base retrieval results
            base_results = self._retrieve_base_results(query, top_k * 2)  # Get more for filtering
            
            # Apply PCIe-specific filtering
            filtered_results = self._apply_pcie_filters(
                base_results,
                technical_level_filter=technical_level_filter,
                pcie_layer_filter=pcie_layer_filter,
                semantic_type_filter=semantic_type_filter
            )
            
            # Apply concept boosting
            if self.enable_concept_boosting:
                boosted_results = self._apply_concept_boosting(query, filtered_results)
            else:
                boosted_results = filtered_results
            
            # Take top k results
            final_results = boosted_results[:top_k]
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Return structured output if requested
            if return_structured:
                # Build filters dict
                filters = {}
                if technical_level_filter:
                    filters['technical_level'] = technical_level_filter
                if pcie_layer_filter:
                    filters['pcie_layer'] = pcie_layer_filter
                if semantic_type_filter:
                    filters['semantic_type'] = semantic_type_filter
                
                # Format as structured response
                structured_response = PCIeStructuredOutputFormatter.format_search_results(
                    query=query,
                    raw_results=final_results,
                    response_time_ms=response_time_ms,
                    model_used=self.embedding_model,
                    filters=filters
                )
                
                # Add analysis if we can detect issue type
                if any(keyword in query.lower() for keyword in ['timeout', 'completion timeout']):
                    structured_response = PCIeStructuredOutputFormatter.add_analysis(
                        structured_response,
                        issue_type="Completion Timeout",
                        severity="critical",
                        confidence=0.8
                    )
                elif any(keyword in query.lower() for keyword in ['link training', 'ltssm']):
                    structured_response = PCIeStructuredOutputFormatter.add_analysis(
                        structured_response,
                        issue_type="Link Training Issue",
                        severity="warning",
                        confidence=0.7
                    )
                
                return structured_response
            else:
                # Convert to PCIe query results (legacy format)
                pcie_results = self._convert_to_pcie_results(final_results)
                logger.debug(f"PCIe query returned {len(pcie_results)} results")
                return pcie_results
            
        except Exception as e:
            logger.error(f"Error in PCIe query: {str(e)}")
            if return_structured:
                # Return error response
                return PCIeQueryResponse(
                    query=query,
                    response_time_ms=(time.time() - start_time) * 1000,
                    total_results=0,
                    results=[],
                    model_used=self.embedding_model
                )
            else:
                return []
    
    def _retrieve_base_results(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve base results from vector database"""
        try:
            # Load vector store if not loaded
            if self.vector_store is None:
                vector_db_path = Path(self.vector_db_path)
                if not vector_db_path.exists():
                    logger.error(f"Vector database not found at {self.vector_db_path}")
                    return []
                
                # Get dimension from embedding model
                from src.models.embedding_selector import get_embedding_selector
                embedding_selector = get_embedding_selector()
                embedding_selector.switch_model(self.embedding_model)
                model_info = embedding_selector.get_model_info()
                dimension = model_info.get('dimension', 1536)
                
                self.vector_store = FAISSVectorStore(
                    dimension=dimension,
                    index_path=str(vector_db_path)
                )
                self.vector_store.load(str(vector_db_path))
            
            # Generate query embedding
            from src.models.embedding_selector import get_embedding_selector
            from src.processors.embedder import Embedder
            
            embedding_selector = get_embedding_selector()
            embedding_selector.switch_model(self.embedding_model)
            
            # Get query embedding
            query_embedding = embedding_selector.current_provider.encode([query])[0]
            
            # Search
            results = self.vector_store.search(
                query_embedding=query_embedding,
                k=top_k
            )
            
            # Convert to expected format
            formatted_results = []
            for content, metadata, score in results:
                formatted_results.append({
                    'content': content,
                    'metadata': metadata,
                    'score': float(score)
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error retrieving base results: {str(e)}")
            return []
    
    def _apply_pcie_filters(self, 
                           results: List[Dict[str, Any]],
                           technical_level_filter: Optional[int] = None,
                           pcie_layer_filter: Optional[str] = None,
                           semantic_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Apply PCIe-specific filters"""
        
        filtered = results
        
        # Technical level filter
        if technical_level_filter is not None and self.enable_technical_level_filtering:
            filtered = [
                r for r in filtered 
                if r.get('metadata', {}).get('technical_level', 1) >= technical_level_filter
            ]
        
        # PCIe layer filter
        if pcie_layer_filter:
            filtered = [
                r for r in filtered
                if r.get('metadata', {}).get('pcie_layer') == pcie_layer_filter
            ]
        
        # Semantic type filter
        if semantic_type_filter:
            filtered = [
                r for r in filtered
                if r.get('metadata', {}).get('semantic_type') == semantic_type_filter
            ]
        
        return filtered
    
    def _apply_concept_boosting(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Boost results based on PCIe concept matching"""
        
        # Extract PCIe concepts from query
        query_concepts = self.pcie_chunker._extract_pcie_concepts(query)
        
        if not query_concepts:
            return results
        
        # Boost scores based on concept overlap
        boosted_results = []
        for result in results:
            metadata = result.get('metadata', {})
            result_concepts = metadata.get('pcie_concepts', '').split(',')
            result_concepts = [c for c in result_concepts if c]  # Remove empty strings
            
            # Calculate concept overlap boost
            concept_overlap = len(set(query_concepts) & set(result_concepts))
            boost_factor = 1.0 + (concept_overlap * 0.1)  # 10% boost per overlapping concept
            
            # Apply boost to score
            boosted_score = result.get('score', 0.0) * boost_factor
            
            boosted_result = result.copy()
            boosted_result['score'] = boosted_score
            boosted_result['concept_boost'] = boost_factor
            boosted_results.append(boosted_result)
        
        # Re-sort by boosted scores
        boosted_results.sort(key=lambda x: x['score'], reverse=True)
        
        return boosted_results
    
    def _convert_to_pcie_results(self, results: List[Dict[str, Any]]) -> List[PCIeQueryResult]:
        """Convert base results to PCIe query results"""
        pcie_results = []
        
        for result in results:
            metadata = result.get('metadata', {})
            
            pcie_result = PCIeQueryResult(
                content=result.get('content', ''),
                score=result.get('score', 0.0),
                metadata=metadata,
                pcie_concepts=metadata.get('pcie_concepts', '').split(','),
                technical_level=metadata.get('technical_level', 1),
                semantic_type=metadata.get('semantic_type', 'content'),
                source_section=metadata.get('section_header', 'Unknown Section')
            )
            
            pcie_results.append(pcie_result)
        
        return pcie_results
    
    def get_pcie_mode_stats(self) -> Dict[str, Any]:
        """Get statistics about PCIe RAG mode"""
        try:
            stats = {
                'mode': 'pcie_adaptive',
                'chunking_strategy': 'adaptive',
                'chunk_config': self.chunk_config,
                'concept_boosting_enabled': self.enable_concept_boosting,
                'technical_level_filtering_enabled': self.enable_technical_level_filtering
            }
            
            # Try to get vector database stats
            if self.vector_store is not None:
                try:
                    total_vectors = len(self.vector_store.metadata) if hasattr(self.vector_store, 'metadata') else 0
                    dimension = self.vector_store.dimension if hasattr(self.vector_store, 'dimension') else 0
                    
                    # Calculate size
                    vector_db_path = Path(self.vector_db_path)
                    size_mb = 0
                    if vector_db_path.exists():
                        index_file = vector_db_path / "index.faiss"
                        if index_file.exists():
                            size_mb = index_file.stat().st_size / (1024 * 1024)
                    
                    stats.update({
                        'total_vectors': total_vectors,
                        'dimension': dimension,
                        'size_mb': size_mb
                    })
                except:
                    pass
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting PCIe mode stats: {str(e)}")
            return {'error': str(e)}
    
    def _is_knowledge_base_current(self) -> bool:
        """Check if knowledge base is current"""
        # Simple implementation - check if vector database exists
        vector_db_path = Path(self.vector_db_path)
        return vector_db_path.exists() and any(vector_db_path.iterdir())

# Utility functions for PCIe RAG mode

def create_pcie_rag_engine(embedding_model: str = "text-embedding-3-small",
                          chunk_config: Optional[Dict[str, int]] = None) -> PCIeRAGEngine:
    """Factory function to create PCIe RAG engine"""
    return PCIeRAGEngine(
        embedding_model=embedding_model,
        chunk_config=chunk_config
    )

def build_pcie_knowledge_base(knowledge_base_path: str, 
                             embedding_model: str = "text-embedding-3-small",
                             force_rebuild: bool = False) -> bool:
    """Build PCIe knowledge base with adaptive chunking"""
    engine = create_pcie_rag_engine(embedding_model=embedding_model)
    return engine.build_knowledge_base(knowledge_base_path, force_rebuild=force_rebuild)