"""
Vector store implementation using FAISS
"""

import faiss
import numpy as np
from pathlib import Path
import pickle
import json
import logging
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """FAISS 기반 벡터 스토어"""
    
    def __init__(self, index_path: str, index_type: str = "L2", dimension: int = 1536):
        """
        Args:
            index_path: 인덱스 파일 경로
            index_type: 인덱스 타입 ("L2" 또는 "IP")
            dimension: 벡터 차원 (기본값: 1536)
        """
        self.index_path = Path(index_path)
        self.index_type = index_type
        self.dimension = dimension
        self.index = None
        self.metadata = []
        
        # 인덱스 로드 또는 생성
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """인덱스 파일 로드 또는 생성"""
        try:
            # 인덱스 경로가 디렉토리인 경우 처리
            if Path(self.index_path).is_dir():
                # Check for separate files format (new format)
                separate_files_path = Path(self.index_path)
                index_file = separate_files_path / "index.faiss"
                metadata_file = separate_files_path / "metadata.json"
                documents_file = separate_files_path / "documents.json"
                
                if index_file.exists() and metadata_file.exists():
                    # Load from separate files
                    try:
                        import json
                        self.index = faiss.read_index(str(index_file))
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            self.metadata = json.load(f)
                        # Update dimension from loaded index
                        self.dimension = self.index.d
                        logger.info(f"Loaded index from separate files in {self.index_path}")
                        return
                    except Exception as e:
                        logger.warning(f"Failed to load from separate files: {e}")
                
                # Fall back to single file format
                self.index_path = Path(self.index_path) / "index.faiss"
            
            # 인덱스 파일이 존재하는 경우 로드
            if Path(self.index_path).exists():
                with open(self.index_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        self.index = data.get('index')
                        self.metadata = data.get('metadata', [])
                    else:
                        # 구버전 호환성 - faiss.deserialize_index 시도
                        self.index = faiss.deserialize_index(data)
            else:
                # 인덱스 파일이 없는 경우 새로 생성
                if self.index_type == "L2":
                    self.index = faiss.IndexFlatL2(self.dimension)
                else:  # IP (Inner Product)
                    self.index = faiss.IndexFlatIP(self.dimension)
                # 부모 디렉토리 생성
                Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
                # 인덱스 저장
                self._save_index()
        except Exception as e:
            logger.error(f"인덱스 로드/생성 실패: {str(e)}")
            raise
    
    def add_documents(self, documents: List[str], embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        문서 추가
        
        Args:
            documents: 문서 리스트
            embeddings: 문서 임베딩 (numpy array)
            metadata: 메타데이터 리스트
        """
        if self.index is None:
            # 첫 번째 문서 추가 시 인덱스 생성
            self.dimension = embeddings.shape[1]
            if self.index_type == "L2":
                self.index = faiss.IndexFlatL2(self.dimension)
            else:  # IP (Inner Product)
                self.index = faiss.IndexFlatIP(self.dimension)
        
        # 문서 추가
        self.index.add(embeddings)
        
        # 메타데이터에 문서 내용 포함
        for i, doc in enumerate(documents):
            metadata[i]['content'] = doc
        
        self.metadata.extend(metadata)
        
        # 인덱스 저장
        self._save_index()
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        유사 문서 검색
        
        Args:
            query_embedding: 쿼리 임베딩
            k: 반환할 문서 수
            
        Returns:
            (문서, 메타데이터, 유사도 점수) 튜플 리스트
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # 검색
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        # 결과 포맷팅
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # 유효한 인덱스인 경우
                score = float(distances[0][i])
                if self.index_type == "L2":
                    # L2 거리를 유사도 점수로 변환 (0~1 범위)
                    score = 1 / (1 + score)
                results.append((
                    self.metadata[idx].get('content', ''),
                    self.metadata[idx],
                    score
                ))
        
        return results
    
    def _save_index(self):
        """인덱스 저장"""
        try:
            # Ensure index_path is a Path object
            index_path = Path(self.index_path)
            
            # Check if index_path is a directory (new format)
            if index_path.is_dir():
                # Save in separate files format
                directory = index_path
                directory.mkdir(parents=True, exist_ok=True)
                
                # Save index
                faiss.write_index(self.index, str(directory / "index.faiss"))
                
                # Save metadata
                with open(directory / "metadata.json", 'w', encoding='utf-8') as f:
                    json.dump(self.metadata, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Saved index to {directory} (separate files format)")
            else:
                # Save in single file format (old format)
                # 디렉토리 생성
                index_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 인덱스 저장
                with open(index_path, 'wb') as f:
                    pickle.dump({
                        'index': self.index,
                        'metadata': self.metadata
                    }, f)
                logger.info(f"Saved index to {index_path}")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise
    
    def clear(self):
        """인덱스 초기화"""
        self.index = None
        self.metadata = []
        self._save_index()
        logger.info("Index cleared") 