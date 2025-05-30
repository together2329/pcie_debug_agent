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

logger = logging.getLogger(__name__)

@dataclass
class RAGQuery:
    """RAG 쿼리 객체"""
    query: str
    context_window: int = 5
    rerank: bool = True
    filters: Optional[Dict[str, Any]] = None
    min_similarity: float = 0.5
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
            max_tokens=max_tokens
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
            
            # 5. LLM 프롬프트 생성
            prompt = self._create_prompt(rag_query.query, context)
            
            # 6. LLM 호출
            llm_response = self._call_llm(prompt)
            
            # 7. 응답 파싱 및 신뢰도 계산
            answer, reasoning = self._parse_llm_response(llm_response)
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
        """컨텍스트 구성"""
        context_parts = []
        
        for i, doc in enumerate(documents):
            context_part = f"[Source {i+1}]\n"
            
            if include_metadata and doc.get('metadata'):
                meta = doc['metadata']
                if 'file_name' in meta:
                    context_part += f"File: {meta['file_name']}\n"
                if 'section' in meta:
                    context_part += f"Section: {meta['section']}\n"
                if 'page' in meta:
                    context_part += f"Page: {meta['page']}\n"
            
            context_part += f"Content: {doc['content']}\n"
            context_part += f"Relevance Score: {doc['score']:.3f}\n"
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """LLM 프롬프트 생성"""
        prompt = f"""You are an expert assistant with access to a knowledge base. 
Answer the following question based on the provided context. 
If the context doesn't contain sufficient information, say so clearly.

Question: {query}

Context from knowledge base:
{context}

Instructions:
1. Provide a comprehensive answer based on the context
2. Cite the source numbers [Source N] when using information
3. If you're making any inferences, clearly state them
4. Structure your answer with clear sections if needed
5. At the end, provide a brief reasoning for your answer

Answer:"""
        
        return prompt
    
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
    
    def _calculate_confidence(self, documents: List[Dict[str, Any]], answer: str) -> float:
        """신뢰도 계산"""
        if not documents:
            return 0.0
        
        # 평균 유사도 점수
        avg_score = np.mean([doc['score'] for doc in documents])
        
        # 답변 길이 기반 조정
        answer_length_factor = min(len(answer) / 500, 1.0)  # 500자를 기준으로
        
        # 소스 인용 확인
        citation_factor = 1.0
        for i in range(len(documents)):
            if f"[Source {i+1}]" in answer:
                citation_factor += 0.1
        citation_factor = min(citation_factor, 1.5)
        
        # 최종 신뢰도
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