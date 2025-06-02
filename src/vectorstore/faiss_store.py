import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import pickle
import json

class FAISSVectorStore:
    """FAISS-based vector store for efficient similarity search"""
    
    def __init__(self, dimension: int, index_type: str = "IndexFlatIP", index_path: Optional[str] = None):
        """Initialize FAISS vector store"""
        self.dimension = dimension
        self.index_type = index_type
        self.index_path = index_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize index
        if index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Store metadata
        self.metadata = []
        self.documents = []
        
        # Try to load existing index if path provided
        if index_path and Path(index_path).exists():
            try:
                loaded_store = self.load(index_path)
                self.index = loaded_store.index
                self.documents = loaded_store.documents
                self.metadata = loaded_store.metadata
            except Exception as e:
                self.logger.warning(f"Could not load existing index from {index_path}: {e}")
                # Continue with empty index
    
    def add_documents(self, 
                     embeddings: List[List[float]], 
                     documents: List[str],
                     metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add documents to the vector store"""
        try:
            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Normalize embeddings for cosine similarity
            if self.index_type == "IndexFlatIP":
                faiss.normalize_L2(embeddings_array)
            
            # Add to index
            self.index.add(embeddings_array)
            
            # Store documents and metadata
            self.documents.extend(documents)
            if metadata:
                self.metadata.extend(metadata)
            else:
                self.metadata.extend([{} for _ in documents])
        
        except Exception as e:
            self.logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def search(self, 
              query_embedding: List[float], 
              k: int = 5) -> List[Tuple[str, Dict[str, Any], float]]:
        """Search for similar documents"""
        try:
            # Convert query to numpy array
            query_array = np.array([query_embedding]).astype('float32')
            
            # Normalize query for cosine similarity
            if self.index_type == "IndexFlatIP":
                faiss.normalize_L2(query_array)
            
            # Search
            distances, indices = self.index.search(query_array, k)
            
            # Format results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.documents):  # Valid index
                    results.append((
                        self.documents[idx],
                        self.metadata[idx],
                        float(distance)
                    ))
            
            return results
        
        except Exception as e:
            # Only log meaningful errors with actual content
            error_msg = str(e).strip()
            if error_msg and len(error_msg) > 0:
                # Only log if it's not a spurious empty error
                self.logger.error(f"Error searching vector store: {error_msg}")
                if hasattr(self.logger, 'debug'):
                    self.logger.debug(f"Exception type: {type(e).__name__}")
            # Return partial results if available, otherwise empty list
            return results if 'results' in locals() else []
    
    def save(self, directory: str) -> None:
        """Save vector store to disk"""
        try:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            
            # Save index
            faiss.write_index(self.index, str(directory / "index.faiss"))
            
            # Save documents and metadata
            with open(directory / "documents.json", 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            
            with open(directory / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        except Exception as e:
            self.logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    @classmethod
    def load(cls, directory: str) -> 'FAISSVectorStore':
        """Load vector store from disk"""
        try:
            directory = Path(directory)
            
            # Load index
            index = faiss.read_index(str(directory / "index.faiss"))
            
            # Create instance
            store = cls(dimension=index.d, index_type="IndexFlatIP")
            store.index = index
            
            # Load documents and metadata
            with open(directory / "documents.json", 'r', encoding='utf-8') as f:
                store.documents = json.load(f)
            
            with open(directory / "metadata.json", 'r', encoding='utf-8') as f:
                store.metadata = json.load(f)
            
            return store
        
        except Exception as e:
            logging.error(f"Error loading vector store: {str(e)}")
            raise
    
    def clear(self) -> None:
        """Clear the vector store"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.metadata = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            'num_documents': len(self.documents),
            'dimension': self.dimension,
            'index_type': self.index_type,
            'total_size': self.index.ntotal
        } 