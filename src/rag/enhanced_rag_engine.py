"""
Enhanced RAG Engine with embedding-based LLM functionality
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime

from src.vectorstore.faiss_store import FAISSVectorStore
from src.models.model_manager import ModelManager
from src.rag.retriever import Retriever
from src.rag.analyzer import Analyzer
from src.models.pcie_prompts import PCIePromptTemplates

logger = logging.getLogger(__name__)

@dataclass
class RAGQuery:
    """RAG 쿼리 객체"""
    query: str
    context_window: int = 5
    rerank: bool = True
    filters: Optional[Dict[str, Any]] = None
    min_similarity: float = 0.1  # Lowered from 0.5 to 0.1 for better recall
    include_metadata: bool = True

@dataclass
class RAGResponse:
    """RAG 응답 객체"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class EnhancedRAGEngine:
    """향상된 RAG 엔진 - Embedding 기반 LLM 통합"""
    
    def __init__(self,
                 vector_store: FAISSVectorStore,
                 model_manager: ModelManager,
                 llm_provider: str = "openai",
                 llm_model: str = "gpt-4",
                 temperature: float = 0.1,
                 max_tokens: int = 2000):
        
        self.vector_store = vector_store
        self.model_manager = model_manager
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 검색기 초기화
        self.retriever = Retriever(vector_store, model_manager)
        
        # 분석기 초기화
        self.analyzer = Analyzer(
            llm_provider=llm_provider,
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            model_manager=model_manager
        )
        
        # 성능 메트릭
        self.metrics = {
            "queries_processed": 0,
            "average_response_time": 0,
            "average_confidence": 0,
            "cache_hits": 0
        }
        
        # 쿼리 캐시 (선택사항)
        self.query_cache = {}
        self.cache_size = 100
        
    def query(self, rag_query: RAGQuery) -> RAGResponse:
        """
        RAG 쿼리 실행
        
        Args:
            rag_query: RAG 쿼리 객체
            
        Returns:
            RAG 응답 객체
        """
        start_time = datetime.now()
        
        try:
            # 캐시 확인
            cache_key = self._get_cache_key(rag_query)
            if cache_key in self.query_cache:
                self.metrics["cache_hits"] += 1
                cached_response = self.query_cache[cache_key]
                cached_response.metadata["from_cache"] = True
                return cached_response
            
            # 1. 쿼리 임베딩 생성
            query_embedding = self._generate_query_embedding(rag_query.query)
            
            # 2. 관련 문서 검색
            retrieved_docs = self._retrieve_documents(
                query_embedding,
                rag_query.context_window,
                rag_query.filters,
                rag_query.min_similarity
            )
            
            # 3. 문서 재순위화 (선택사항)
            if rag_query.rerank and len(retrieved_docs) > 0:
                retrieved_docs = self._rerank_documents(
                    rag_query.query,
                    retrieved_docs
                )
            
            # 4. 컨텍스트 구성
            context = self._build_context(retrieved_docs, rag_query.include_metadata)
            
            # 5. LLM 프롬프트 생성 및 호출
            try:
                prompt = self._create_prompt(rag_query.query, context)
                llm_response = self._call_llm(prompt)
                answer, reasoning = self._parse_llm_response(llm_response)
            except Exception as e:
                # Fall back to search results only when LLM is unavailable
                logger.warning(f"LLM call failed, providing search results only: {e}")
                answer = self._create_fallback_answer(rag_query.query, retrieved_docs)
                reasoning = "LLM unavailable - providing search results from knowledge base only."
            
            # 7. 신뢰도 계산
            confidence = self._calculate_confidence(retrieved_docs, answer)
            
            # 8. 응답 객체 생성
            response = RAGResponse(
                answer=answer,
                sources=retrieved_docs[:rag_query.context_window],
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    "query_time": (datetime.now() - start_time).total_seconds(),
                    "num_sources": len(retrieved_docs),
                    "model": self.llm_model,
                    "from_cache": False
                }
            )
            
            # 캐시 저장
            self._update_cache(cache_key, response)
            
            # 메트릭 업데이트
            self._update_metrics(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            return RAGResponse(
                answer=f"죄송합니다. 쿼리 처리 중 오류가 발생했습니다: {str(e)}",
                sources=[],
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    async def aquery(self, rag_query: RAGQuery) -> RAGResponse:
        """비동기 RAG 쿼리 실행"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            response = await loop.run_in_executor(executor, self.query, rag_query)
        return response
    
    def batch_query(self, queries: List[RAGQuery]) -> List[RAGResponse]:
        """배치 쿼리 처리"""
        responses = []
        
        # 쿼리 임베딩 일괄 생성
        query_texts = [q.query for q in queries]
        query_embeddings = self.model_manager.generate_embeddings(query_texts)
        
        for query, embedding in zip(queries, query_embeddings):
            # 개별 쿼리 처리 (임베딩은 재사용)
            response = self._process_single_query_with_embedding(query, embedding)
            responses.append(response)
            
        return responses
    
    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """쿼리 임베딩 생성"""
        embeddings = self.model_manager.generate_embeddings([query])
        return embeddings[0]
    
    def _retrieve_documents(self,
                          query_embedding: np.ndarray,
                          k: int,
                          filters: Optional[Dict[str, Any]],
                          min_similarity: float) -> List[Dict[str, Any]]:
        """문서 검색"""
        results = self.vector_store.search(query_embedding, k * 2)  # 필터링을 위해 더 많이 검색
        
        formatted_results = []
        for doc, metadata, score in results:
            # 유사도 필터링
            if score < min_similarity:
                continue
                
            # 메타데이터 필터링
            if filters:
                if not all(metadata.get(key) == value for key, value in filters.items()):
                    continue
            
            formatted_results.append({
                'content': doc,
                'metadata': metadata,
                'score': float(score)
            })
            
            if len(formatted_results) >= k:
                break
                
        return formatted_results
    
    def _rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """문서 재순위화"""
        # Cross-encoder 또는 더 정교한 재순위 모델 사용
        # 여기서는 간단한 구현
        for doc in documents:
            # 쿼리와 문서의 추가적인 관련성 점수 계산
            relevance_boost = self._calculate_relevance_boost(query, doc['content'])
            doc['rerank_score'] = doc['score'] * (1 + relevance_boost)
        
        # 재순위 점수로 정렬
        documents.sort(key=lambda x: x.get('rerank_score', x['score']), reverse=True)
        return documents
    
    def _calculate_relevance_boost(self, query: str, content: str) -> float:
        """관련성 부스트 계산"""
        # 간단한 키워드 매칭 기반 부스트
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())
        
        overlap = len(query_terms.intersection(content_terms))
        boost = overlap / len(query_terms) if query_terms else 0
        
        return boost * 0.5  # 최대 50% 부스트
    
    def _build_context(self, documents: List[Dict[str, Any]], include_metadata: bool) -> str:
        """Enhanced context construction with automatic citation tracking"""
        context_parts = []
        
        for i, doc in enumerate(documents):
            source_id = i + 1
            context_part = f"[Source {source_id}]\n"
            
            # Enhanced metadata extraction with citation information
            if include_metadata and doc.get('metadata'):
                meta = doc['metadata']
                citation_info = self._extract_citation_info(meta, doc.get('content', ''))
                
                if citation_info['file_name']:
                    context_part += f"File: {citation_info['file_name']}\n"
                if citation_info['section']:
                    context_part += f"Section: {citation_info['section']}\n"
                if citation_info['page']:
                    context_part += f"Page: {citation_info['page']}\n"
                if citation_info['spec_reference']:
                    context_part += f"Specification: {citation_info['spec_reference']}\n"
                if citation_info['authority_level']:
                    context_part += f"Authority: {citation_info['authority_level']}\n"
            
            # Add content with relevance scoring
            context_part += f"Content: {doc['content']}\n"
            context_part += f"Relevance Score: {doc['score']:.3f}\n"
            
            # Add citation instructions for LLM
            context_part += f"Citation Format: [Source {source_id}]\n"
            
            context_parts.append(context_part)
        
        # Add citation instructions at the end
        citation_instructions = self._generate_citation_instructions(len(documents))
        context_with_instructions = "\n---\n".join(context_parts) + "\n\n" + citation_instructions
        
        return context_with_instructions
    
    def _extract_citation_info(self, metadata: Dict[str, Any], content: str) -> Dict[str, str]:
        """Extract comprehensive citation information from metadata and content"""
        citation_info = {
            'file_name': '',
            'section': '',
            'page': '',
            'spec_reference': '',
            'authority_level': ''
        }
        
        # Basic metadata
        citation_info['file_name'] = metadata.get('file_name', metadata.get('source', ''))
        citation_info['section'] = metadata.get('section', '')
        citation_info['page'] = str(metadata.get('page', '')) if metadata.get('page') else ''
        
        # Extract specification references from content
        spec_patterns = [
            r'PCIe\s+(?:Base\s+)?Spec(?:ification)?\s+(\d+\.\d+)',
            r'PCI\s+Express\s+(?:Base\s+)?Specification\s+(\d+\.\d+)',
            r'Chapter\s+(\d+(?:\.\d+)*)',
            r'Section\s+(\d+(?:\.\d+)*)',
        ]
        
        for pattern in spec_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                citation_info['spec_reference'] = match.group(0)
                break
        
        # Determine authority level based on source type and content
        citation_info['authority_level'] = self._determine_authority_level(metadata, content)
        
        return citation_info
    
    def _determine_authority_level(self, metadata: Dict[str, Any], content: str) -> str:
        """Determine the authority level of the source"""
        file_name = metadata.get('file_name', '').lower()
        content_lower = content.lower()
        
        # Official specifications
        if any(term in file_name for term in ['pcie_spec', 'pci_express_spec', 'specification']):
            return 'Official Specification'
        
        # Standards documents
        if any(term in file_name for term in ['standard', 'ieee', 'iso']):
            return 'Industry Standard'
        
        # Technical documentation
        if any(term in file_name for term in ['manual', 'guide', 'reference']):
            return 'Technical Documentation'
        
        # Analysis of content for authority indicators
        authority_indicators = {
            'specification': 'Specification Document',
            'compliance': 'Compliance Guide',
            'debug': 'Debug Guide',
            'troubleshoot': 'Troubleshooting Manual',
            'error': 'Error Reference'
        }
        
        for indicator, level in authority_indicators.items():
            if indicator in content_lower:
                return level
        
        return 'General Reference'
    
    def _generate_citation_instructions(self, num_sources: int) -> str:
        """Generate citation instructions for the LLM"""
        instructions = """
**CITATION REQUIREMENTS:**
When providing your answer, you MUST cite sources using the format [Source X] where X is the source number.

Citation Guidelines:
1. Cite specific sources when making factual claims
2. Use [Source 1], [Source 2], etc. to reference the sources above
3. Prefer authoritative sources (specifications, official docs) over general references
4. Multiple sources can be cited for the same point: [Source 1, Source 3]
5. Include source citations naturally within your response text

Example: "PCIe completion timeouts typically occur when a request is not completed within the specified time limit [Source 1]. The timeout value is configurable in the device's configuration space [Source 2]."
"""
        
        if num_sources > 0:
            source_list = ", ".join([f"Source {i+1}" for i in range(num_sources)])
            instructions += f"\nAvailable sources: {source_list}"
        
        return instructions
    
    def _create_prompt(self, query: str, context: str) -> str:
        """LLM 프롬프트 생성 (PCIe 특화)"""
        # Use PCIe-specific prompt templates
        return PCIePromptTemplates.get_prompt_for_query_type(query, context)
    
    def _call_llm(self, prompt: str) -> str:
        """LLM 호출"""
        return self.model_manager.generate_completion(
            provider=self.llm_provider,
            model=self.llm_model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _parse_llm_response(self, response: str) -> Tuple[str, Optional[str]]:
        """LLM 응답 파싱"""
        # 응답에서 답변과 추론 부분 분리
        parts = response.split("Reasoning:")
        
        answer = parts[0].strip()
        reasoning = parts[1].strip() if len(parts) > 1 else None
        
        return answer, reasoning
    
    def _create_fallback_answer(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Create fallback answer when LLM is unavailable"""
        if not documents:
            return f"No relevant information found for query: '{query}'. Please check your question or try different keywords."
        
        # Remove duplicates based on content
        unique_docs = []
        seen_content = set()
        for doc in documents:
            content = doc.get('content', '')[:100]  # First 100 chars for comparison
            if content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(content)
        
        answer = f"Search results for: '{query}'\n\n"
        answer += "Based on the PCIe knowledge base, here are the most relevant findings:\n\n"
        
        for i, doc in enumerate(unique_docs[:3], 1):  # Top 3 unique results
            content = doc.get('content', '')
            score = doc.get('score', 0.0)
            source = doc.get('source', 'Unknown source')
            
            answer += f"{i}. **Relevance: {score:.1%}**\n"
            answer += f"   Source: {source}\n"
            answer += f"   Content: {content[:300]}...\n\n"
        
        answer += "💡 **Recommendation**: For detailed analysis and solutions, please:\n"
        answer += "- Set up a local model (download required model files), or\n"
        answer += "- Configure API keys for cloud models (OpenAI/Anthropic)\n"
        answer += "- Use '/model list' to see available options"
        
        return answer

    def _calculate_confidence(self, documents: List[Dict[str, Any]], answer: str) -> float:
        """Advanced confidence calculation with multi-layered PCIe domain intelligence"""
        if not documents:
            return 0.0
        
        try:
            return self._advanced_confidence_calculation(documents, answer)
        except Exception as e:
            logger.warning(f"Advanced confidence calculation failed, using fallback: {e}")
            return self._fallback_confidence_calculation(documents, answer)
    
    def _advanced_confidence_calculation(self, documents: List[Dict[str, Any]], answer: str) -> float:
        """Multi-layered confidence calculation with domain intelligence"""
        
        # Initialize confidence components
        confidence_components = {
            'base_similarity': 0.0,
            'technical_alignment': 0.0,
            'content_quality': 0.0,
            'domain_expertise': 0.0,
            'answer_completeness': 0.0,
            'source_reliability': 0.0
        }
        
        # 1. Base similarity score (30% weight)
        avg_score = np.mean([doc['score'] for doc in documents])
        confidence_components['base_similarity'] = min(avg_score * 0.3, 0.30)
        
        # 2. Technical alignment score (25% weight)
        confidence_components['technical_alignment'] = self._calculate_technical_alignment(documents, answer) * 0.25
        
        # 3. Content quality score (20% weight)
        confidence_components['content_quality'] = self._calculate_content_quality(documents, answer) * 0.20
        
        # 4. Domain expertise score (15% weight)
        confidence_components['domain_expertise'] = self._calculate_domain_expertise(documents, answer) * 0.15
        
        # 5. Answer completeness score (10% weight)
        confidence_components['answer_completeness'] = self._calculate_answer_completeness(documents, answer) * 0.10
        
        # 6. Source reliability score (10% weight)  
        confidence_components['source_reliability'] = self._calculate_source_reliability(documents) * 0.10
        
        # Apply domain-specific multipliers
        domain_multiplier = self._get_domain_multiplier(documents, answer)
        
        # Calculate total confidence
        base_confidence = sum(confidence_components.values())
        final_confidence = min(base_confidence * domain_multiplier, 1.0)
        
        # Log confidence breakdown for debugging
        self.logger.debug(f"Confidence breakdown: {confidence_components}, "
                         f"domain_multiplier: {domain_multiplier:.3f}, "
                         f"final: {final_confidence:.3f}")
        
        return final_confidence
    
    def _calculate_technical_alignment(self, documents: List[Dict[str, Any]], answer: str) -> float:
        """Calculate how well the answer aligns with PCIe technical concepts"""
        
        # PCIe technical concepts with weights
        technical_concepts = {
            # Core concepts (high weight)
            'pcie': 1.0, 'function level reset': 1.0, 'flr': 1.0,
            'configuration request retry status': 1.0, 'crs': 1.0,
            'link training state machine': 1.0, 'ltssm': 1.0,
            'completion timeout': 1.0, 'advanced error reporting': 1.0,
            
            # Specific technical terms (medium weight)
            'transaction layer packet': 0.8, 'tlp': 0.8,
            'data link layer packet': 0.8, 'dllp': 0.8,
            'message signaled interrupt': 0.8, 'msi': 0.8, 'msi-x': 0.8,
            'poisoned tlp': 0.8, 'malformed tlp': 0.8, 'ecrc error': 0.8,
            
            # General terms (lower weight)
            'endpoint': 0.6, 'root complex': 0.6, 'switch': 0.6,
            'configuration space': 0.6, 'capability': 0.6,
            'gen1': 0.5, 'gen2': 0.5, 'gen3': 0.5, 'gen4': 0.5, 'gen5': 0.5,
        }
        
        combined_text = answer.lower() + " " + " ".join([doc.get('content', '') for doc in documents[:3]]).lower()
        
        alignment_score = 0.0
        matched_concepts = 0
        
        for concept, weight in technical_concepts.items():
            if concept in combined_text:
                alignment_score += weight
                matched_concepts += 1
        
        # Normalize by number of possible matches and apply diminishing returns
        if matched_concepts > 0:
            normalized_score = min(alignment_score / (matched_concepts * 1.2), 1.0)
            return normalized_score
        
        return 0.1  # Small base score if no technical terms found
    
    def _calculate_content_quality(self, documents: List[Dict[str, Any]], answer: str) -> float:
        """Calculate content quality based on structure, citations, and coherence"""
        
        quality_score = 0.0
        
        # 1. Answer structure quality
        if len(answer) > 100:  # Reasonable length
            quality_score += 0.3
        
        # 2. Source citations present
        citation_count = 0
        for i in range(len(documents)):
            if f"[Source {i+1}]" in answer or f"source {i+1}" in answer.lower():
                citation_count += 1
        
        citation_ratio = min(citation_count / max(len(documents), 1), 1.0)
        quality_score += citation_ratio * 0.3
        
        # 3. Technical specificity
        technical_indicators = ['0x', 'bit', 'register', 'offset', 'specification', 'chapter', 'section']
        specificity_count = sum(1 for indicator in technical_indicators if indicator in answer.lower())
        quality_score += min(specificity_count / len(technical_indicators), 1.0) * 0.2
        
        # 4. Explanation depth
        explanation_indicators = ['because', 'due to', 'results in', 'causes', 'leads to', 'therefore']
        explanation_count = sum(1 for indicator in explanation_indicators if indicator in answer.lower())
        quality_score += min(explanation_count / 3, 1.0) * 0.2
        
        return min(quality_score, 1.0)
    
    def _calculate_domain_expertise(self, documents: List[Dict[str, Any]], answer: str) -> float:
        """Calculate domain expertise level of the answer"""
        
        expertise_score = 0.0
        
        # PCIe specification references
        spec_patterns = [
            r'pcie\s+(?:base\s+)?spec(?:ification)?',
            r'chapter\s+\d+',
            r'section\s+\d+(?:\.\d+)*',
            r'table\s+\d+',
            r'figure\s+\d+',
        ]
        
        combined_text = answer.lower()
        spec_matches = sum(1 for pattern in spec_patterns if re.search(pattern, combined_text))
        expertise_score += min(spec_matches / len(spec_patterns), 1.0) * 0.4
        
        # Error code specificity
        error_patterns = [
            r'0x[0-9a-f]+',  # Hex error codes
            r'bit\s+\d+',    # Bit positions
            r'register\s+0x[0-9a-f]+',  # Register addresses
        ]
        
        error_matches = sum(1 for pattern in error_patterns if re.search(pattern, combined_text))
        expertise_score += min(error_matches / len(error_patterns), 1.0) * 0.3
        
        # Troubleshooting depth
        troubleshooting_terms = [
            'debug', 'analyze', 'check', 'verify', 'confirm', 'investigate',
            'root cause', 'workaround', 'solution', 'fix'
        ]
        
        troubleshooting_count = sum(1 for term in troubleshooting_terms if term in combined_text)
        expertise_score += min(troubleshooting_count / len(troubleshooting_terms), 1.0) * 0.3
        
        return min(expertise_score, 1.0)
    
    def _calculate_answer_completeness(self, documents: List[Dict[str, Any]], answer: str) -> float:
        """Calculate how complete the answer is relative to available information"""
        
        if not documents:
            return 0.0
        
        # Check coverage of available information
        total_content_length = sum(len(doc.get('content', '')) for doc in documents)
        answer_length = len(answer)
        
        # Ideal answer should use substantial portion of available content
        coverage_ratio = min(answer_length / max(total_content_length * 0.1, 100), 1.0)
        
        # Check if answer addresses multiple aspects
        question_aspects = ['what', 'why', 'how', 'when', 'where']
        addressed_aspects = sum(1 for aspect in question_aspects 
                              if any(aspect in answer.lower() for aspect in question_aspects))
        
        aspect_score = min(addressed_aspects / len(question_aspects), 1.0)
        
        # Combine coverage and aspect scores
        completeness = (coverage_ratio * 0.7) + (aspect_score * 0.3)
        
        return min(completeness, 1.0)
    
    def _calculate_source_reliability(self, documents: List[Dict[str, Any]]) -> float:
        """Calculate reliability of source documents"""
        
        if not documents:
            return 0.0
        
        reliability_score = 0.0
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            doc_reliability = 0.0
            
            # File type reliability
            file_type = metadata.get('file_type', '').lower()
            if file_type == 'pdf':
                doc_reliability += 0.4  # PDFs often more authoritative
            elif file_type in ['md', 'txt']:
                doc_reliability += 0.3
            
            # Source name indicators
            source_name = metadata.get('file_name', '').lower()
            if any(indicator in source_name for indicator in ['spec', 'standard', 'official']):
                doc_reliability += 0.3
            elif any(indicator in source_name for indicator in ['manual', 'guide', 'doc']):
                doc_reliability += 0.2
            
            # Content length indicates depth
            content_length = len(doc.get('content', ''))
            if content_length > 500:
                doc_reliability += 0.3
            elif content_length > 200:
                doc_reliability += 0.2
            
            reliability_score += min(doc_reliability, 1.0)
        
        # Average reliability across documents
        return min(reliability_score / len(documents), 1.0)
    
    def _get_domain_multiplier(self, documents: List[Dict[str, Any]], answer: str) -> float:
        """Get domain-specific confidence multiplier"""
        
        combined_text = answer.lower() + " " + " ".join([doc.get('content', '') for doc in documents[:2]]).lower()
        
        # High-confidence domains
        if any(term in combined_text for term in ['completion timeout', 'flr', 'function level reset']):
            return 1.15  # High confidence for well-documented issues
        
        # Medium-confidence domains
        if any(term in combined_text for term in ['ltssm', 'link training', 'error', 'debug']):
            return 1.10
        
        # Complex domains that need careful handling
        if any(term in combined_text for term in ['compliance', 'specification', 'protocol']):
            return 1.05
        
        return 1.0  # Base multiplier
    
    def _fallback_confidence_calculation(self, documents: List[Dict[str, Any]], answer: str) -> float:
        """Fallback confidence calculation if advanced method fails"""
        
        avg_score = np.mean([doc['score'] for doc in documents])
        answer_length_factor = min(len(answer) / 500, 1.0)
        
        citation_factor = 1.0
        for i in range(len(documents)):
            if f"[Source {i+1}]" in answer:
                citation_factor += 0.1
        citation_factor = min(citation_factor, 1.5)
        
        confidence = avg_score * answer_length_factor * citation_factor
        return min(confidence, 1.0)
    
    def _get_cache_key(self, rag_query: RAGQuery) -> str:
        """캐시 키 생성"""
        key_parts = [
            rag_query.query,
            str(rag_query.context_window),
            str(rag_query.filters),
            str(rag_query.min_similarity)
        ]
        return "|".join(key_parts)
    
    def _update_cache(self, key: str, response: RAGResponse):
        """캐시 업데이트"""
        if len(self.query_cache) >= self.cache_size:
            # LRU 방식으로 가장 오래된 항목 제거
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        self.query_cache[key] = response
    
    def _update_metrics(self, response: RAGResponse):
        """메트릭 업데이트"""
        self.metrics["queries_processed"] += 1
        
        # 평균 응답 시간 업데이트
        query_time = response.metadata.get("query_time", 0)
        prev_avg = self.metrics["average_response_time"]
        n = self.metrics["queries_processed"]
        self.metrics["average_response_time"] = (prev_avg * (n-1) + query_time) / n
        
        # 평균 신뢰도 업데이트
        prev_conf = self.metrics["average_confidence"]
        self.metrics["average_confidence"] = (prev_conf * (n-1) + response.confidence) / n
    
    def get_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 반환"""
        return self.metrics.copy()
    
    def clear_cache(self):
        """캐시 초기화"""
        self.query_cache.clear()
        logger.info("Query cache cleared")
    
    def _process_single_query_with_embedding(self, 
                                           query: RAGQuery, 
                                           embedding: np.ndarray) -> RAGResponse:
        """임베딩이 주어진 단일 쿼리 처리"""
        # 이미 계산된 임베딩을 사용하여 쿼리 처리
        retrieved_docs = self._retrieve_documents(
            embedding,
            query.context_window,
            query.filters,
            query.min_similarity
        )
        
        if query.rerank and len(retrieved_docs) > 0:
            retrieved_docs = self._rerank_documents(query.query, retrieved_docs)
        
        context = self._build_context(retrieved_docs, query.include_metadata)
        prompt = self._create_prompt(query.query, context)
        llm_response = self._call_llm(prompt)
        answer, reasoning = self._parse_llm_response(llm_response)
        confidence = self._calculate_confidence(retrieved_docs, answer)
        
        return RAGResponse(
            answer=answer,
            sources=retrieved_docs[:query.context_window],
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "num_sources": len(retrieved_docs),
                "model": self.llm_model
            }
        ) 