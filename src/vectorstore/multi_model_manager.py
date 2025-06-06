"""
Multi-Model Vector Database Manager
Manages separate vector databases for different embedding models
"""
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, List
import logging

from src.vectorstore.faiss_store import FAISSVectorStore
from src.processors.document_chunker import DocumentChunker
from src.models.embedding_selector import get_embedding_selector

logger = logging.getLogger(__name__)

class MultiModelVectorManager:
    """Manages vector databases for multiple embedding models"""
    
    def __init__(self, base_path: str = "data/vectorstore"):
        self.base_path = Path(base_path)
        self.embedding_selector = get_embedding_selector()
        
        # Create base directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Model-specific paths
        self.model_paths = {
            "text-embedding-3-small": self.base_path / "openai_small_1536d",
            "text-embedding-3-large": self.base_path / "openai_large_3072d", 
            "text-embedding-ada-002": self.base_path / "openai_ada_1536d",
            "all-MiniLM-L6-v2": self.base_path / "local_minilm_384d",
            "all-mpnet-base-v2": self.base_path / "local_mpnet_768d",
            "multi-qa-MiniLM-L6-cos-v1": self.base_path / "local_qa_384d",
            # PCIe-specific adaptive chunking paths
            "pcie_adaptive_text-embedding-3-small": self.base_path / "pcie_adaptive_openai_small_1536d",
            "pcie_adaptive_all-MiniLM-L6-v2": self.base_path / "pcie_adaptive_local_minilm_384d"
        }
        
    def get_model_path(self, model_name: str) -> Path:
        """Get storage path for a specific model"""
        if model_name in self.model_paths:
            return self.model_paths[model_name]
        else:
            # Generate path based on model info
            try:
                self.embedding_selector.switch_model(model_name)
                info = self.embedding_selector.get_model_info()
                provider = info.get('provider', 'unknown')
                dimension = info.get('dimension', 'unknownd')
                safe_name = model_name.replace('/', '_').replace('-', '_')
                return self.base_path / f"{provider}_{safe_name}_{dimension}d"
            except:
                return self.base_path / f"unknown_{model_name}"
    
    def exists(self, model_name: str) -> bool:
        """Check if vector database exists for model"""
        model_path = self.get_model_path(model_name)
        return (model_path / "index.faiss").exists() and (model_path / "metadata.json").exists()
    
    def load(self, model_name: str) -> Optional[FAISSVectorStore]:
        """Load vector database for specific model"""
        if not self.exists(model_name):
            return None
        
        model_path = self.get_model_path(model_name)
        try:
            return FAISSVectorStore.load(str(model_path))
        except Exception as e:
            logger.error(f"Failed to load vector database for {model_name}: {e}")
            return None
    
    def build(self, model_name: str, input_dir: str = "data/knowledge_base", 
              force: bool = False, chunk_size: int = 1000, chunk_overlap: int = 200) -> bool:
        """Build vector database for specific model"""
        model_path = self.get_model_path(model_name)
        
        # Check if already exists and not forcing rebuild
        if self.exists(model_name) and not force:
            logger.info(f"Vector database for {model_name} already exists. Use force=True to rebuild.")
            return True
        
        try:
            # Switch to the target model
            self.embedding_selector.switch_model(model_name)
            embedding_info = self.embedding_selector.get_model_info()
            embedding_provider = self.embedding_selector.get_current_provider()
            
            logger.info(f"Building vector database for {model_name}")
            logger.info(f"  Model: {embedding_info['model']}")
            logger.info(f"  Provider: {embedding_info['provider']}")
            logger.info(f"  Dimension: {embedding_info['dimension']}")
            logger.info(f"  Path: {model_path}")
            
            # Remove existing database if forcing rebuild
            if model_path.exists() and force:
                shutil.rmtree(model_path)
            
            # Create model-specific directory
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Process documents
            input_path = Path(input_dir)
            if not input_path.exists():
                logger.error(f"Input directory not found: {input_path}")
                return False
            
            # Create chunker
            chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            # Find and process documents
            documents = []
            text_files = list(input_path.glob('**/*.md')) + \
                        list(input_path.glob('**/*.txt')) + \
                        list(input_path.glob('**/*.log'))
            
            if not text_files:
                logger.error(f"No documents found in {input_path}")
                return False
            
            logger.info(f"Processing {len(text_files)} documents...")
            
            for file_path in text_files:
                try:
                    chunks = chunker.chunk_documents(file_path)
                    for chunk in chunks:
                        documents.append({
                            'content': chunk.content,
                            'source': str(file_path),
                            'chunk_id': chunk.chunk_id,
                            'metadata': chunk.metadata
                        })
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")
                    continue
            
            if not documents:
                logger.error("No documents were successfully processed")
                return False
            
            logger.info(f"Created {len(documents)} chunks from {len(text_files)} documents")
            
            # Create vector store
            store = FAISSVectorStore(dimension=embedding_info['dimension'])
            
            # Add documents in batches
            batch_size = 100
            logger.info(f"Generating embeddings and building index...")
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                texts = [doc['content'] for doc in batch]
                metadatas = [{'source': doc.get('source', 'unknown')} for doc in batch]
                
                # Generate embeddings
                embeddings = embedding_provider.encode(texts)
                
                # Add to vector store
                store.add_documents(embeddings, texts, metadatas)
                
                logger.info(f"  Processed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            # Save the index
            store.save(str(model_path))
            
            logger.info(f"✅ Vector database built successfully for {model_name}")
            logger.info(f"   Total vectors: {store.index.ntotal}")
            logger.info(f"   Location: {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build vector database for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def delete(self, model_name: str) -> bool:
        """Delete vector database for specific model"""
        model_path = self.get_model_path(model_name)
        
        if not model_path.exists():
            logger.info(f"Vector database for {model_name} does not exist")
            return True
        
        try:
            shutil.rmtree(model_path)
            logger.info(f"✅ Deleted vector database for {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vector database for {model_name}: {e}")
            return False
    
    def list_models(self) -> Dict[str, Dict]:
        """List all models and their database status"""
        models = {}
        
        # Check predefined models
        for model_name in self.model_paths.keys():
            models[model_name] = {
                'exists': self.exists(model_name),
                'path': str(self.get_model_path(model_name)),
                'size': self._get_database_size(model_name) if self.exists(model_name) else 0
            }
        
        # Check for other databases in the directory
        if self.base_path.exists():
            for path in self.base_path.iterdir():
                if path.is_dir() and path.name not in [p.name for p in self.model_paths.values()]:
                    # Try to extract model name from path
                    model_name = path.name
                    if model_name not in models:
                        models[model_name] = {
                            'exists': (path / "index.faiss").exists() and (path / "metadata.json").exists(),
                            'path': str(path),
                            'size': self._get_database_size_from_path(path)
                        }
        
        return models
    
    def _get_database_size(self, model_name: str) -> int:
        """Get database size in bytes"""
        model_path = self.get_model_path(model_name)
        return self._get_database_size_from_path(model_path)
    
    def _get_database_size_from_path(self, path: Path) -> int:
        """Get database size from path in bytes"""
        total_size = 0
        if path.exists():
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        return total_size
    
    def get_stats(self, model_name: str) -> Optional[Dict]:
        """Get statistics for a model's vector database"""
        if not self.exists(model_name):
            return None
        
        try:
            store = self.load(model_name)
            if store is None:
                return None
            
            return {
                'model_name': model_name,
                'path': str(self.get_model_path(model_name)),
                'total_vectors': store.index.ntotal,
                'dimension': store.dimension,
                'size_bytes': self._get_database_size(model_name),
                'size_mb': self._get_database_size(model_name) / (1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Failed to get stats for {model_name}: {e}")
            return None
    
    def cleanup_orphaned(self) -> List[str]:
        """Remove orphaned vector databases that don't match current models"""
        cleaned = []
        
        if not self.base_path.exists():
            return cleaned
        
        # Get list of valid model paths
        valid_paths = set(str(p) for p in self.model_paths.values())
        
        for path in self.base_path.iterdir():
            if path.is_dir() and str(path) not in valid_paths:
                # Check if it looks like a vector database
                if (path / "index.faiss").exists() or (path / "metadata.json").exists():
                    try:
                        shutil.rmtree(path)
                        cleaned.append(str(path))
                        logger.info(f"Cleaned up orphaned database: {path}")
                    except Exception as e:
                        logger.error(f"Failed to cleanup {path}: {e}")
        
        return cleaned

    def migrate_legacy(self) -> bool:
        """Migrate legacy single vector database to multi-model system"""
        legacy_path = self.base_path / "index.faiss"
        
        if not legacy_path.exists():
            logger.info("No legacy vector database found")
            return True
        
        try:
            # Try to determine which model the legacy database was built with
            # This is a best guess based on dimension
            legacy_store = FAISSVectorStore.load(str(self.base_path))
            dimension = legacy_store.dimension
            
            # Map dimension to likely model
            dimension_to_model = {
                384: "all-MiniLM-L6-v2",
                768: "all-mpnet-base-v2", 
                1536: "text-embedding-3-small",
                3072: "text-embedding-3-large"
            }
            
            likely_model = dimension_to_model.get(dimension)
            if not likely_model:
                logger.warning(f"Unknown dimension {dimension}, cannot migrate legacy database")
                return False
            
            logger.info(f"Migrating legacy database (dimension {dimension}) to {likely_model}")
            
            # Move to model-specific path
            target_path = self.get_model_path(likely_model)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy files to new location
            shutil.move(str(self.base_path / "index.faiss"), str(target_path / "index.faiss"))
            if (self.base_path / "documents.json").exists():
                shutil.move(str(self.base_path / "documents.json"), str(target_path / "documents.json"))
            if (self.base_path / "metadata.json").exists():
                shutil.move(str(self.base_path / "metadata.json"), str(target_path / "metadata.json"))
            
            logger.info(f"✅ Legacy database migrated to {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate legacy database: {e}")
            return False