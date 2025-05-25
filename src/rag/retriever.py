from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from src.vectorstore.faiss_store import FAISSVectorStore
from src.processors.embedder import Embedder
import logging

logger = logging.getLogger(__name__)

class Retriever:
    """RAG 검색 엔진"""
    
    def __init__(self,
                 vector_store: FAISSVectorStore,
                 embedder: Embedder,
                 reranker: Optional[Any] = None):
        
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        
    def retrieve(self,
                query: str,
                k: int = 10,
                filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        쿼리에 대해 관련 문서 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            filter_metadata: 메타데이터 필터
            
        Returns:
            검색 결과 리스트
        """
        # 쿼리 임베딩
        query_embedding = self.embedder.embed_single(query)
        
        # 벡터 검색
        results = self.vector_store.search(
            query_embedding,
            k=k * 2 if self.reranker else k,  # 재순위화를 위해 더 많이 검색
            filter_metadata=filter_metadata
        )
        
        # 결과 포맷팅
        formatted_results = []
        for score, doc, meta in results:
            formatted_results.append({
                'content': doc,
                'metadata': meta,
                'score': score,
                'query': query
            })
            
        # 재순위화 (선택)
        if self.reranker:
            formatted_results = self._rerank(query, formatted_results, k)
            
        return formatted_results[:k]
    
    def _rerank(self, query: str, results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """재순위화 (추후 구현)"""
        # 현재는 스코어 기준 정렬만 수행
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def retrieve_with_context(self,
                            query: str,
                            context: List[str],
                            k: int = 10) -> List[Dict[str, Any]]:
        """
        컨텍스트를 고려한 검색
        
        Args:
            query: 검색 쿼리
            context: 이전 분석 컨텍스트
            k: 반환할 문서 수
            
        Returns:
            검색 결과 리스트
        """
        # 컨텍스트를 쿼리에 통합
        enhanced_query = self._enhance_query_with_context(query, context)
        
        # 검색 수행
        return self.retrieve(enhanced_query, k)
    
    def _enhance_query_with_context(self, query: str, context: List[str]) -> str:
        """컨텍스트로 쿼리 향상"""
        if not context:
            return query
            
        # 최근 컨텍스트 2개만 사용
        recent_context = context[-2:]
        context_str = " ".join(recent_context)
        
        # 쿼리와 컨텍스트 결합
        enhanced = f"{query} Context: {context_str}"
        
        return enhanced 