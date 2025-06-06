import numpy as np
from typing import List, Dict, Any, Union, Optional
from sentence_transformers import SentenceTransformer
import torch
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Embedder:
    """텍스트 임베딩 생성기"""
    
    def __init__(self, 
                 model_name: Union[str, Any] = 'sentence-transformers/all-MiniLM-L6-v2',
                 device: Optional[str] = None,
                 batch_size: int = 32):
        
        self.model_name = model_name
        self.batch_size = batch_size
        
        # 디바이스 설정
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # 모델 로드
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Loaded embedding model: {model_name} on {self.device}")
        logger.info(f"Embedding dimension: {self.embedding_dimension}")
        
    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        텍스트 리스트를 임베딩으로 변환
        
        Args:
            texts: 임베딩할 텍스트 리스트
            show_progress: 진행률 표시 여부
            
        Returns:
            임베딩 배열 (shape: [len(texts), embedding_dim])
        """
        if not texts:
            return np.array([])
            
        # 배치 처리
        embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), 
                     disable=not show_progress,
                     desc="Generating embeddings"):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
            
        return np.vstack(embeddings)
    
    def embed_single(self, text: str) -> np.ndarray:
        """단일 텍스트 임베딩"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_chunks(self, chunks: List[Union['DocumentChunk', 'CodeChunk']]) -> None:
        """청크 객체들에 임베딩 추가"""
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed_texts(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            
    def compute_similarity(self, query_embedding: np.ndarray, 
                          doc_embeddings: np.ndarray) -> np.ndarray:
        """코사인 유사도 계산"""
        # 정규화
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        
        # 코사인 유사도
        similarities = np.dot(doc_norms, query_norm)
        return similarities 