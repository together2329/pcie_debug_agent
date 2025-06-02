"""
Metadata-Enhanced RAG Engine
Combines traditional RAG with rich metadata for improved search and filtering
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import asyncio
from datetime import datetime

from src.vectorstore.faiss_store import FAISSVectorStore
from src.rag.enhanced_rag_engine import EnhancedRAGEngine, RAGQuery, RAGResponse
from src.rag.metadata_extractor import (
    MetadataExtractor, MetadataFilter, PCIeMetadata,
    PCIeDocumentType, PCIeVersion, ErrorSeverity
)
from src.models.model_manager import ModelManager

logger = logging.getLogger(__name__)


@dataclass
class MetadataRAGQuery(RAGQuery):
    """Extended RAG query with metadata filters"""
    # Metadata filters
    pcie_versions: Optional[List[str]] = None
    document_types: Optional[List[PCIeDocumentType]] = None
    error_severity: Optional[ErrorSeverity] = None
    components: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    
    # Advanced options
    use_metadata_boost: bool = True  # Boost relevance based on metadata match
    metadata_weight: float = 0.3  # Weight for metadata relevance (0.0-1.0)


class MetadataEnhancedRAGEngine(EnhancedRAGEngine):
    """RAG Engine enhanced with metadata extraction and filtering"""
    
    def __init__(self, 
                 vector_store: FAISSVectorStore,
                 model_manager: ModelManager,
                 metadata_extractor: Optional[MetadataExtractor] = None,
                 **kwargs):
        super().__init__(vector_store, model_manager, **kwargs)
        
        # Initialize metadata extractor
        self.metadata_extractor = metadata_extractor or MetadataExtractor(
            model_manager, 
            model_id=kwargs.get('llm_model', 'gpt-4o-mini')
        )
        
        # Metadata cache
        self.metadata_cache = {}
        
        # Enhanced metrics
        self.metrics.update({
            "metadata_extractions": 0,
            "metadata_filtered_queries": 0,
            "metadata_boost_applied": 0
        })
    
    async def add_document_with_metadata(self, 
                                       content: str, 
                                       file_path: str,
                                       quick_extract: bool = False) -> Dict[str, Any]:
        """Add document with automatically extracted metadata"""
        try:
            # Extract metadata
            if quick_extract:
                # Use regex-based quick extraction
                metadata_dict = self.metadata_extractor.extract_quick_metadata(content)
            else:
                # Use LLM-based extraction
                metadata = await self.metadata_extractor.extract_metadata(content, file_path)
                metadata_dict = self._metadata_to_dict(metadata)
            
            # Generate embedding
            embedding = self.model_manager.embed([content])[0]
            
            # Add to vector store with metadata
            self.vector_store.add_documents(
                embeddings=[embedding],
                documents=[content],
                metadata=[metadata_dict]
            )
            
            # Cache metadata
            doc_id = f"{file_path}:{len(self.vector_store.documents)-1}"
            self.metadata_cache[doc_id] = metadata_dict
            
            self.metrics["metadata_extractions"] += 1
            
            return {
                "status": "success",
                "document_id": doc_id,
                "metadata": metadata_dict
            }
            
        except Exception as e:
            logger.error(f"Error adding document with metadata: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def query_with_metadata(self, query: MetadataRAGQuery) -> RAGResponse:
        """Execute RAG query with metadata filtering and boosting"""
        start_time = datetime.now()
        
        try:
            # Get initial search results
            initial_results = self._search_documents(query)
            
            # Apply metadata filters
            if self._has_metadata_filters(query):
                filtered_results = self._apply_metadata_filters(initial_results, query)
                self.metrics["metadata_filtered_queries"] += 1
            else:
                filtered_results = initial_results
            
            # Apply metadata boosting if enabled
            if query.use_metadata_boost and self._has_metadata_filters(query):
                boosted_results = self._apply_metadata_boost(filtered_results, query)
                self.metrics["metadata_boost_applied"] += 1
            else:
                boosted_results = filtered_results
            
            # Re-rank results
            final_results = self._rerank_results(boosted_results, query)
            
            # Generate response
            response = self._generate_response(query.query, final_results)
            
            # Calculate metrics
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            return RAGResponse(
                answer=response["answer"],
                sources=response["sources"],
                confidence=response["confidence"],
                reasoning=response.get("reasoning"),
                metadata={
                    "query_time": elapsed_time,
                    "initial_results": len(initial_results),
                    "filtered_results": len(filtered_results),
                    "metadata_filters_applied": self._has_metadata_filters(query),
                    "metadata_boost_applied": query.use_metadata_boost
                }
            )
            
        except Exception as e:
            logger.error(f"Error in metadata-enhanced query: {str(e)}")
            return RAGResponse(
                answer=f"Error processing query: {str(e)}",
                sources=[],
                confidence=0.0
            )
    
    def _search_documents(self, query: RAGQuery) -> List[Tuple[str, Dict[str, Any], float]]:
        """Search documents using vector similarity"""
        # Generate query embedding
        query_embedding = self.model_manager.embed([query.query])[0]
        
        # Search in vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            k=query.context_window * 2  # Get more results for filtering
        )
        
        # Filter by minimum similarity
        filtered = []
        for doc, metadata, score in results:
            if score >= query.min_similarity:
                filtered.append((doc, metadata, score))
        
        return filtered
    
    def _has_metadata_filters(self, query: MetadataRAGQuery) -> bool:
        """Check if query has metadata filters"""
        return any([
            query.pcie_versions,
            query.document_types,
            query.error_severity,
            query.components,
            query.topics
        ])
    
    def _apply_metadata_filters(self, 
                               results: List[Tuple[str, Dict[str, Any], float]], 
                               query: MetadataRAGQuery) -> List[Tuple[str, Dict[str, Any], float]]:
        """Apply metadata filters to results"""
        filtered = results
        
        # Convert to document format for filtering
        documents = [
            {"content": doc, "metadata": meta, "score": score}
            for doc, meta, score in results
        ]
        
        # Apply filters
        if query.pcie_versions:
            documents = MetadataFilter.filter_by_pcie_version(documents, query.pcie_versions)
        
        if query.error_severity:
            documents = MetadataFilter.filter_by_error_severity(documents, query.error_severity)
        
        if query.components:
            documents = MetadataFilter.filter_by_components(documents, query.components)
        
        if query.topics:
            documents = MetadataFilter.filter_by_topics(documents, query.topics)
        
        # Convert back to tuple format
        filtered = [
            (doc["content"], doc["metadata"], doc["score"])
            for doc in documents
        ]
        
        return filtered
    
    def _apply_metadata_boost(self,
                            results: List[Tuple[str, Dict[str, Any], float]],
                            query: MetadataRAGQuery) -> List[Tuple[str, Dict[str, Any], float]]:
        """Boost document scores based on metadata relevance"""
        boosted = []
        
        for doc, metadata, base_score in results:
            # Calculate metadata relevance score
            metadata_score = self._calculate_metadata_relevance(metadata, query)
            
            # Combine scores
            final_score = (
                base_score * (1 - query.metadata_weight) + 
                metadata_score * query.metadata_weight
            )
            
            boosted.append((doc, metadata, final_score))
        
        # Re-sort by new scores
        boosted.sort(key=lambda x: x[2], reverse=True)
        
        return boosted
    
    def _calculate_metadata_relevance(self, 
                                    metadata: Dict[str, Any], 
                                    query: MetadataRAGQuery) -> float:
        """Calculate how well metadata matches query filters"""
        relevance_scores = []
        
        # PCIe version match
        if query.pcie_versions and metadata.get("pcie_version"):
            matches = sum(1 for v in query.pcie_versions if v in metadata["pcie_version"])
            relevance_scores.append(matches / len(query.pcie_versions))
        
        # Component match
        if query.components and metadata.get("components"):
            matches = sum(1 for c in query.components if c in metadata["components"])
            relevance_scores.append(matches / len(query.components))
        
        # Topic match
        if query.topics and metadata.get("topics"):
            matches = sum(1 for t in query.topics if t in metadata["topics"])
            relevance_scores.append(matches / len(query.topics))
        
        # Document type match
        if query.document_types and metadata.get("document_type"):
            if metadata["document_type"] in [dt.value for dt in query.document_types]:
                relevance_scores.append(1.0)
            else:
                relevance_scores.append(0.0)
        
        # Average relevance
        if relevance_scores:
            return sum(relevance_scores) / len(relevance_scores)
        else:
            return 0.5  # Neutral score if no filters
    
    def _rerank_results(self,
                       results: List[Tuple[str, Dict[str, Any], float]],
                       query: RAGQuery) -> List[Tuple[str, Dict[str, Any], float]]:
        """Re-rank results and limit to context window"""
        # Already sorted by score, just limit
        return results[:query.context_window]
    
    def _generate_response(self, 
                         query: str, 
                         results: List[Tuple[str, Dict[str, Any], float]]) -> Dict[str, Any]:
        """Generate response using LLM with context"""
        # Format context from results
        context_parts = []
        sources = []
        
        for i, (doc, metadata, score) in enumerate(results):
            # Add document context
            context_parts.append(f"[Document {i+1}]")
            if metadata.get("title"):
                context_parts.append(f"Title: {metadata['title']}")
            if metadata.get("pcie_version"):
                context_parts.append(f"PCIe Version: {', '.join(metadata['pcie_version'])}")
            context_parts.append(f"Content: {doc[:500]}...")  # Limit content length
            context_parts.append("")
            
            # Track source
            sources.append({
                "document_id": i + 1,
                "title": metadata.get("title", "Unknown"),
                "metadata": metadata,
                "relevance_score": float(score)
            })
        
        context = "\n".join(context_parts)
        
        # Generate answer using analyzer
        response = self.analyzer.analyze(
            query=query,
            context=context,
            analysis_type="answer_with_sources"
        )
        
        return {
            "answer": response,
            "sources": sources,
            "confidence": self._calculate_confidence(results)
        }
    
    def _calculate_confidence(self, results: List[Tuple[str, Dict[str, Any], float]]) -> float:
        """Calculate confidence score based on results"""
        if not results:
            return 0.0
        
        # Average of top scores
        top_scores = [score for _, _, score in results[:3]]
        avg_score = sum(top_scores) / len(top_scores)
        
        # Normalize to 0-1 range
        return min(avg_score, 1.0)
    
    def _metadata_to_dict(self, metadata: PCIeMetadata) -> Dict[str, Any]:
        """Convert PCIeMetadata to dictionary"""
        return {
            "document_type": metadata.document_type.value,
            "title": metadata.title,
            "summary": metadata.summary,
            "pcie_version": [v.value for v in metadata.pcie_version],
            "topics": metadata.topics,
            "error_codes": metadata.error_codes,
            "error_severity": metadata.error_severity.value if metadata.error_severity else None,
            "components": metadata.components,
            "speed": metadata.speed,
            "link_width": metadata.link_width,
            "keywords": metadata.keywords,
            "related_specs": metadata.related_specs,
            "confidence_score": metadata.confidence_score,
            "extraction_timestamp": metadata.extraction_timestamp
        }
    
    async def batch_process_documents(self, 
                                    file_paths: List[str], 
                                    quick_extract: bool = False) -> Dict[str, Any]:
        """Process multiple documents with metadata extraction"""
        results = {
            "processed": 0,
            "failed": 0,
            "documents": []
        }
        
        for file_path in file_paths:
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Add with metadata
                result = await self.add_document_with_metadata(
                    content, file_path, quick_extract
                )
                
                if result["status"] == "success":
                    results["processed"] += 1
                    results["documents"].append(result)
                else:
                    results["failed"] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                results["failed"] += 1
        
        return results