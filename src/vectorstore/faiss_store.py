import faiss
import numpy as np
import pickle
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """FAISS 기반 벡터 저장소"""
    
    def __init__(self, 
                 dimension: int = 384,
                 index_type: str = "IndexFlatL2",
                 nlist: int = 100):
        
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        
        # 인덱스 생성
        self.index = self._create_index()
        
        # 메타데이터 저장
        self.documents = []
        self.metadata = []
        
    def _create_index(self) -> faiss.Index:
        """FAISS 인덱스 생성"""
        if self.index_type == "IndexFlatL2":
            # 정확한 L2 거리 검색
            index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IndexFlatIP":
            # 내적 (코사인 유사도용)
            index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexIVFFlat":
            # 근사 검색 (대규모 데이터용)
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
            
        return index
    
    def add_documents(self, 
                     embeddings: np.ndarray,
                     documents: List[str],
                     metadata: List[Dict[str, Any]]) -> None:
        """
        문서와 임베딩 추가
        
        Args:
            embeddings: 임베딩 배열 (shape: [n_docs, dimension])
            documents: 문서 텍스트 리스트
            metadata: 문서 메타데이터 리스트
        """
        # 입력 검증
        n_docs = len(documents)
        assert embeddings.shape[0] == n_docs, "Number of embeddings must match documents"
        assert len(metadata) == n_docs, "Number of metadata must match documents"
        
        # 임베딩 정규화 (코사인 유사도용)
        if self.index_type == "IndexFlatIP":
            faiss.normalize_L2(embeddings)
            
        # 인덱스 학습 (IVF 인덱스의 경우)
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)
            
        # 인덱스에 추가
        start_idx = len(self.documents)
        self.index.add(embeddings)
        
        # 메타데이터 저장
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        
        logger.info(f"Added {n_docs} documents to index. Total: {len(self.documents)}")
        
    def search(self, 
               query_embedding: np.ndarray,
               k: int = 10,
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[float, str, Dict[str, Any]]]:
        """
        유사도 검색
        
        Args:
            query_embedding: 쿼리 임베딩
            k: 반환할 결과 수
            filter_metadata: 메타데이터 필터 (선택)
            
        Returns:
            [(score, document, metadata), ...] 형태의 결과 리스트
        """
        # 쿼리 임베딩 정규화
        if self.index_type == "IndexFlatIP":
            query_embedding = query_embedding.copy()
            faiss.normalize_L2(query_embedding.reshape(1, -1))
            
        # 검색 수행
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        distances, indices = self.index.search(query_embedding, k * 2)  # 필터링을 위해 더 많이 검색
        
        # 결과 정리
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # 유효하지 않은 인덱스
                continue
                
            doc = self.documents[idx]
            meta = self.metadata[idx]
            
            # 메타데이터 필터링
            if filter_metadata:
                match = all(meta.get(key) == value for key, value in filter_metadata.items())
                if not match:
                    continue
                    
            # 거리를 유사도로 변환
            if self.index_type in ["IndexFlatL2", "IndexIVFFlat"]:
                score = 1 / (1 + dist)  # L2 거리를 유사도로
            else:
                score = dist  # 내적은 이미 유사도
                
            results.append((score, doc, meta))
            
            if len(results) >= k:
                break
                
        return results
    
    def save(self, path: Union[str, Path]) -> None:
        """인덱스와 메타데이터 저장"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # FAISS 인덱스 저장
        index_path = path / "faiss.index"
        faiss.write_index(self.index, str(index_path))
        
        # 메타데이터 저장
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'dimension': self.dimension,
                'index_type': self.index_type
            }, f)
            
        logger.info(f"Saved vector store to {path}")
        
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FAISSVectorStore':
        """저장된 인덱스 로드"""
        path = Path(path)
        
        # 메타데이터 로드
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            
        # 인스턴스 생성
        store = cls(
            dimension=data['dimension'],
            index_type=data['index_type']
        )
        
        # FAISS 인덱스 로드
        index_path = path / "faiss.index"
        store.index = faiss.read_index(str(index_path))
        
        # 메타데이터 복원
        store.documents = data['documents']
        store.metadata = data['metadata']
        
        logger.info(f"Loaded vector store from {path}")
        return store 