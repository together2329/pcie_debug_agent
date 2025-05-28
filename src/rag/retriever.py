from typing import List, Dict, Any, Optional
import logging
from src.vectorstore.faiss_store import FAISSVectorStore

class Retriever:
    """Retrieves relevant context for error analysis"""
    
    def __init__(self, 
                 vector_store: FAISSVectorStore,
                 embedder: Any):  # Type hint for embedder will be added later
        """Initialize retriever"""
        self.vector_store = vector_store
        self.embedder = embedder
        self.logger = logging.getLogger(__name__)
    
    def retrieve(self, 
                query: str, 
                k: int = 5,
                filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant context for a query"""
        try:
            # Generate query embedding
            query_embedding = self.embedder.generate_embeddings([query])[0]
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                k=k
            )
            
            # Format results
            formatted_results = []
            for doc, metadata, score in results:
                # Apply metadata filtering if specified
                if filter_metadata:
                    if not all(metadata.get(key) == value 
                             for key, value in filter_metadata.items()):
                        continue
                
                formatted_results.append({
                    'content': doc,
                    'metadata': metadata,
                    'score': score
                })
            
            return formatted_results
        
        except Exception as e:
            self.logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def retrieve_with_reranking(self,
                              query: str,
                              k: int = 5,
                              rerank_k: int = 20,
                              filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve and rerank results for better relevance"""
        try:
            # First retrieval with larger k
            initial_results = self.retrieve(
                query=query,
                k=rerank_k,
                filter_metadata=filter_metadata
            )
            
            if not initial_results:
                return []
            
            # Rerank using cross-encoder if available
            if hasattr(self.embedder, 'rerank'):
                reranked_results = self.embedder.rerank(
                    query=query,
                    documents=[r['content'] for r in initial_results]
                )
                
                # Combine reranking scores with initial scores
                for i, result in enumerate(initial_results):
                    result['rerank_score'] = reranked_results[i]
                    result['final_score'] = (result['score'] + result['rerank_score']) / 2
                
                # Sort by final score
                initial_results.sort(key=lambda x: x['final_score'], reverse=True)
            
            # Return top k results
            return initial_results[:k]
        
        except Exception as e:
            self.logger.error(f"Error in retrieve_with_reranking: {str(e)}")
            return []
    
    def retrieve_with_filters(self,
                            query: str,
                            k: int = 5,
                            file_types: Optional[List[str]] = None,
                            min_score: float = 0.0,
                            max_score: float = 1.0) -> List[Dict[str, Any]]:
        """Retrieve results with additional filtering options"""
        try:
            # Build metadata filter
            filter_metadata = {}
            if file_types:
                filter_metadata['file_type'] = file_types
            
            # Retrieve results
            results = self.retrieve(
                query=query,
                k=k * 2,  # Retrieve more to account for filtering
                filter_metadata=filter_metadata
            )
            
            # Apply score filtering
            filtered_results = [
                r for r in results
                if min_score <= r['score'] <= max_score
            ]
            
            # Return top k results
            return filtered_results[:k]
        
        except Exception as e:
            self.logger.error(f"Error in retrieve_with_filters: {str(e)}")
            return []
    
    def get_relevant_sections(self,
                            query: str,
                            document: str,
                            k: int = 3) -> List[str]:
        """Extract most relevant sections from a document"""
        try:
            # Split document into sections
            sections = self._split_into_sections(document)
            
            if not sections:
                return []
            
            # Generate embeddings for sections
            section_embeddings = self.embedder.generate_embeddings(sections)
            query_embedding = self.embedder.generate_embeddings([query])[0]
            
            # Calculate similarity scores
            scores = []
            for section, embedding in zip(sections, section_embeddings):
                score = self._calculate_similarity(query_embedding, embedding)
                scores.append((section, score))
            
            # Sort by score and return top k
            scores.sort(key=lambda x: x[1], reverse=True)
            return [section for section, _ in scores[:k]]
        
        except Exception as e:
            self.logger.error(f"Error getting relevant sections: {str(e)}")
            return []
    
    def _split_into_sections(self, document: str) -> List[str]:
        """Split document into sections"""
        # Simple section splitting by paragraphs
        sections = [s.strip() for s in document.split('\n\n') if s.strip()]
        return sections
    
    def _calculate_similarity(self, 
                            embedding1: List[float],
                            embedding2: List[float]) -> float:
        """Calculate similarity between two embeddings"""
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = sum(a * a for a in embedding1) ** 0.5
        norm2 = sum(b * b for b in embedding2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0 