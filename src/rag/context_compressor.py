#!/usr/bin/env python3
"""
Context Compression Engine for RAG
Compresses retrieved context to fit token limits while preserving key information
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class CompressedContext:
    """Compressed context with metadata"""
    compressed_text: str
    original_length: int
    compressed_length: int
    compression_ratio: float
    key_sections: List[str]
    relevance_score: float

class ContextCompressor:
    """
    Intelligent context compression for RAG systems
    Uses multiple strategies to compress context while preserving relevance
    """
    
    def __init__(self, 
                 max_tokens: int = 8000,
                 compression_model: str = "all-MiniLM-L6-v2",
                 preserve_ratio: float = 0.7):
        """
        Initialize context compressor
        
        Args:
            max_tokens: Maximum tokens for compressed context
            compression_model: Sentence transformer for semantic similarity
            preserve_ratio: Ratio of most relevant content to preserve
        """
        self.max_tokens = max_tokens
        self.preserve_ratio = preserve_ratio
        
        # Load compression model
        try:
            self.embedder = SentenceTransformer(compression_model)
        except Exception as e:
            logger.warning(f"Failed to load compression model: {e}")
            self.embedder = None
        
        # PCIe-specific keywords for importance weighting
        self.pcie_keywords = {
            'critical': ['error', 'failure', 'critical', 'fatal', 'timeout', 'corrupt'],
            'components': ['tlp', 'ltssm', 'aer', 'dpc', 'link', 'lane', 'port'],
            'states': ['l0', 'l1', 'l2', 'd0', 'd1', 'd2', 'd3', 'recovery', 'training'],
            'speeds': ['gen1', 'gen2', 'gen3', 'gen4', 'gen5', 'gt/s', 'mhz'],
            'widths': ['x1', 'x2', 'x4', 'x8', 'x16', 'x32'],
            'protocols': ['config', 'memory', 'completion', 'message', 'vendor']
        }
    
    def compress_context(self, 
                        query: str,
                        retrieved_docs: List[Dict[str, Any]],
                        error_log: str = "") -> CompressedContext:
        """
        Compress context using multiple strategies
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents from RAG
            error_log: Error log context
            
        Returns:
            CompressedContext with optimized content
        """
        # Combine all content
        full_content = self._combine_content(retrieved_docs, error_log)
        original_length = len(full_content)
        
        if original_length <= self.max_tokens:
            return CompressedContext(
                compressed_text=full_content,
                original_length=original_length,
                compressed_length=original_length,
                compression_ratio=1.0,
                key_sections=[],
                relevance_score=1.0
            )
        
        # Apply compression strategies
        compressed_content = self._apply_compression_strategies(
            query, full_content, retrieved_docs
        )
        
        compressed_length = len(compressed_content)
        compression_ratio = compressed_length / original_length
        
        # Extract key sections for metadata
        key_sections = self._extract_key_sections(compressed_content)
        
        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(query, compressed_content)
        
        return CompressedContext(
            compressed_text=compressed_content,
            original_length=original_length,
            compressed_length=compressed_length,
            compression_ratio=compression_ratio,
            key_sections=key_sections,
            relevance_score=relevance_score
        )
    
    def _combine_content(self, 
                        retrieved_docs: List[Dict[str, Any]], 
                        error_log: str) -> str:
        """Combine all content sources"""
        content_parts = []
        
        # Add error log first (highest priority)
        if error_log.strip():
            content_parts.append(f"=== ERROR LOG ===\n{error_log}\n")
        
        # Add retrieved documents
        for i, doc in enumerate(retrieved_docs):
            content = doc.get('content', doc.get('text', ''))
            metadata = doc.get('metadata', {})
            source = metadata.get('source', f'Document {i+1}')
            
            content_parts.append(f"=== {source.upper()} ===\n{content}\n")
        
        return "\n".join(content_parts)
    
    def _apply_compression_strategies(self, 
                                   query: str,
                                   content: str,
                                   retrieved_docs: List[Dict[str, Any]]) -> str:
        """Apply multiple compression strategies"""
        
        # Strategy 1: Sentence-level relevance filtering
        content = self._filter_by_sentence_relevance(query, content)
        
        # Strategy 2: PCIe keyword importance weighting
        content = self._apply_keyword_weighting(content)
        
        # Strategy 3: Redundancy removal
        content = self._remove_redundancy(content)
        
        # Strategy 4: Section prioritization
        content = self._prioritize_sections(query, content)
        
        # Strategy 5: Token limit enforcement
        content = self._enforce_token_limit(content)
        
        return content
    
    def _filter_by_sentence_relevance(self, query: str, content: str) -> str:
        """Filter sentences by semantic relevance to query"""
        if not self.embedder:
            return content
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 10:  # Don't compress if too few sentences
            return content
        
        try:
            # Get embeddings
            query_embedding = self.embedder.encode([query])
            sentence_embeddings = self.embedder.encode(sentences)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]
            
            # Keep top sentences based on preserve_ratio
            num_keep = max(5, int(len(sentences) * self.preserve_ratio))
            top_indices = np.argsort(similarities)[-num_keep:]
            
            # Maintain original order
            top_indices = sorted(top_indices)
            relevant_sentences = [sentences[i] for i in top_indices]
            
            return '. '.join(relevant_sentences) + '.'
            
        except Exception as e:
            logger.warning(f"Sentence filtering failed: {e}")
            return content
    
    def _apply_keyword_weighting(self, content: str) -> str:
        """Prioritize content with PCIe-specific keywords"""
        paragraphs = content.split('\n\n')
        weighted_paragraphs = []
        
        for paragraph in paragraphs:
            score = self._calculate_keyword_score(paragraph)
            weighted_paragraphs.append((paragraph, score))
        
        # Sort by score and keep top paragraphs
        weighted_paragraphs.sort(key=lambda x: x[1], reverse=True)
        num_keep = max(3, int(len(weighted_paragraphs) * 0.8))
        
        return '\n\n'.join([p[0] for p in weighted_paragraphs[:num_keep]])
    
    def _calculate_keyword_score(self, text: str) -> float:
        """Calculate importance score based on PCIe keywords"""
        text_lower = text.lower()
        score = 0.0
        
        for category, keywords in self.pcie_keywords.items():
            category_score = sum(1 for kw in keywords if kw in text_lower)
            
            # Weight different categories
            weights = {
                'critical': 3.0,
                'components': 2.0, 
                'states': 1.5,
                'speeds': 1.0,
                'widths': 1.0,
                'protocols': 1.0
            }
            
            score += category_score * weights.get(category, 1.0)
        
        # Bonus for error-related content
        if any(word in text_lower for word in ['error', 'fail', 'issue', 'problem']):
            score += 2.0
        
        return score
    
    def _remove_redundancy(self, content: str) -> str:
        """Remove redundant information"""
        lines = content.split('\n')
        unique_lines = []
        seen_content = set()
        
        for line in lines:
            # Normalize line for comparison
            normalized = re.sub(r'\s+', ' ', line.strip().lower())
            
            # Skip very short lines or duplicates
            if len(normalized) < 20 or normalized in seen_content:
                continue
            
            seen_content.add(normalized)
            unique_lines.append(line)
        
        return '\n'.join(unique_lines)
    
    def _prioritize_sections(self, query: str, content: str) -> str:
        """Prioritize sections based on query relevance"""
        sections = re.split(r'=== .* ===', content)
        
        if len(sections) <= 2:
            return content
        
        # Always keep error log section first
        error_section = ""
        other_sections = []
        
        for section in sections:
            if 'error log' in section.lower() or any(
                word in section.lower() for word in ['error', 'fail', 'critical']
            ):
                error_section = section
            else:
                other_sections.append(section)
        
        # Combine prioritized content
        result = error_section
        if other_sections:
            result += '\n\n' + '\n\n'.join(other_sections[:2])  # Keep top 2 other sections
        
        return result
    
    def _enforce_token_limit(self, content: str) -> str:
        """Enforce maximum token limit with smart truncation"""
        # Rough token estimation (1 token ≈ 4 characters for English)
        estimated_tokens = len(content) // 4
        
        if estimated_tokens <= self.max_tokens:
            return content
        
        # Calculate target length
        target_length = self.max_tokens * 4
        
        # Smart truncation - prefer keeping complete sentences
        sentences = re.split(r'[.!?]+', content)
        result_sentences = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence) + 1  # +1 for punctuation
            if current_length + sentence_length > target_length:
                break
            result_sentences.append(sentence)
            current_length += sentence_length
        
        return '. '.join(result_sentences) + '.'
    
    def _extract_key_sections(self, content: str) -> List[str]:
        """Extract key sections for metadata"""
        sections = re.findall(r'=== (.*?) ===', content)
        return sections
    
    def _calculate_relevance_score(self, query: str, content: str) -> float:
        """Calculate overall relevance score"""
        if not self.embedder:
            return 0.8  # Default score
        
        try:
            query_embedding = self.embedder.encode([query])
            content_embedding = self.embedder.encode([content])
            
            similarity = cosine_similarity(query_embedding, content_embedding)[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Relevance calculation failed: {e}")
            return 0.8

def main():
    """Test the context compressor"""
    compressor = ContextCompressor(max_tokens=1000)
    
    # Test data
    query = "Why is PCIe link training failing?"
    retrieved_docs = [
        {
            'content': 'PCIe link training is a complex process involving LTSSM states. When training fails, it often indicates signal integrity issues or hardware problems.',
            'metadata': {'source': 'PCIe Specification'}
        }
    ]
    error_log = "[10:15:30] PCIe: Link training failed on device 0000:01:00.0"
    
    # Compress context
    result = compressor.compress_context(query, retrieved_docs, error_log)
    
    print(f"Original length: {result.original_length}")
    print(f"Compressed length: {result.compressed_length}")
    print(f"Compression ratio: {result.compression_ratio:.2f}")
    print(f"Relevance score: {result.relevance_score:.2f}")
    print(f"Compressed content:\n{result.compressed_text}")

if __name__ == "__main__":
    main()