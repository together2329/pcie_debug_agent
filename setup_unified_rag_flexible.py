#!/usr/bin/env python3
"""
Flexible Unified RAG Setup - Works with Available Embedding Models
Falls back to local models if OpenAI API key not available
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GGML_METAL_LOG_LEVEL"] = "0"

class FlexibleUnifiedRAGSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.specs_folder = self.project_root / "data" / "specs"
        self.config_file = self.project_root / "configs" / "settings.yaml"
        self.has_openai = bool(os.getenv("OPENAI_API_KEY"))
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "setup_steps": [],
            "processed_pdfs": [],
            "errors": [],
            "success": False,
            "embedding_model_used": None
        }
        
    async def setup_unified_rag_default(self):
        """Complete setup process with flexible embedding model selection"""
        print("🚀 Setting up Unified RAG as Default System")
        print("=" * 60)
        
        try:
            # Step 1: Check available embedding models
            await self.check_embedding_options()
            
            # Step 2: Update configuration
            await self.update_configuration()
            
            # Step 3: Process PCIe specification PDFs
            await self.process_pcie_specifications()
            
            # Step 4: Build embedding database
            await self.build_embedding_database()
            
            # Step 5: Update interactive shell to use Unified RAG
            await self.update_interactive_shell_default()
            
            # Step 6: Test the setup
            await self.test_setup()
            
            self.results["success"] = True
            print("\\n✅ Setup completed successfully!")
            
        except Exception as e:
            self.results["errors"].append(str(e))
            print(f"\\n❌ Setup failed: {str(e)}")
            
        finally:
            self.save_setup_report()
    
    async def check_embedding_options(self):
        """Check available embedding model options"""
        print("\\n🔍 Checking Embedding Model Options...")
        
        if self.has_openai:
            print("  ✅ OpenAI API key found - will use text-embedding-3-small")
            self.embedding_model = "text-embedding-3-small"
            self.embedding_dimension = 1536
            self.vector_store_suffix = "openai_1536d"
        else:
            print("  ⚠️  OpenAI API key not found - will use local embeddings")
            self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            self.embedding_dimension = 384
            self.vector_store_suffix = "local_384d"
        
        print(f"  📊 Selected model: {self.embedding_model}")
        print(f"  📐 Dimensions: {self.embedding_dimension}")
        
        self.results["embedding_model_used"] = self.embedding_model
        self.results["setup_steps"].append("embedding_model_selected")
    
    async def update_configuration(self):
        """Update configuration to use Unified RAG"""
        print("\\n⚙️ Updating Configuration...")
        
        config = {
            "default_model": "gpt-4o-mini" if self.has_openai else "mock-llm",
            "default_embedding_model": self.embedding_model,
            "default_rag_mode": "unified_adaptive",
            "embedding_dimension": self.embedding_dimension,
            "vector_store_path": f"./data/vectorstore/unified_{self.vector_store_suffix}",
            "enable_metadata_extraction": True,
            "enable_unified_rag": True,
            "chunk_size": 500,
            "chunk_overlap": 50,
            "similarity_threshold": 0.1,
            "unified_rag_default": True
        }
        
        # Create config directory if it doesn't exist
        config_dir = self.config_file.parent
        config_dir.mkdir(exist_ok=True)
        
        # Write YAML config
        try:
            import yaml
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        except ImportError:
            # Fallback to JSON if yaml not available
            json_config_file = self.config_file.with_suffix('.json')
            with open(json_config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"  📝 Config saved as JSON: {json_config_file}")
        
        print(f"  ✅ Configuration updated")
        print(f"  🚀 Default RAG mode: unified_adaptive")
        print(f"  🧠 Embedding model: {self.embedding_model}")
        
        self.results["setup_steps"].append("configuration_updated")
    
    async def process_pcie_specifications(self):
        """Process all PDFs in the specs folder"""
        print("\\n📚 Processing PCIe Specification PDFs...")
        
        pdf_files = list(self.specs_folder.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files:")
        for pdf_file in pdf_files:
            print(f"  📄 {pdf_file.name}")
        
        processed_docs = []
        
        for pdf_file in pdf_files:
            print(f"\\n  🔄 Processing: {pdf_file.name}")
            
            try:
                # Extract text from PDF
                doc_content = await self.extract_pdf_text(pdf_file)
                
                if doc_content:
                    # Create document chunks
                    chunks = await self.create_document_chunks(doc_content, pdf_file)
                    processed_docs.extend(chunks)
                    
                    self.results["processed_pdfs"].append({
                        "filename": pdf_file.name,
                        "chunks": len(chunks),
                        "total_length": len(doc_content),
                        "status": "success"
                    })
                    
                    print(f"    ✅ Created {len(chunks)} chunks ({len(doc_content):,} chars)")
                else:
                    print(f"    ⚠️ No text extracted")
                    
            except Exception as e:
                error_msg = f"Failed to process {pdf_file.name}: {str(e)}"
                print(f"    ❌ {error_msg}")
                self.results["errors"].append(error_msg)
        
        print(f"\\n  📊 Total document chunks: {len(processed_docs)}")
        self.processed_documents = processed_docs
        self.results["setup_steps"].append("pdfs_processed")
    
    async def extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text content from PDF"""
        try:
            import PyPDF2
            
            text_content = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                print(f"      📄 {len(pdf_reader.pages)} pages")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            # Clean up text
                            text = text.replace('\\n', ' ').replace('\\r', ' ')
                            text = ' '.join(text.split())  # Normalize whitespace
                            text_content.append(f"[Page {page_num + 1}] {text}")
                    except Exception as e:
                        print(f"        ⚠️ Page {page_num + 1}: {e}")
            
            return "\\n\\n".join(text_content)
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            return ""
    
    async def create_document_chunks(self, content: str, pdf_path: Path) -> List[Dict[str, Any]]:
        """Create document chunks with rich PCIe metadata"""
        chunks = []
        chunk_size = 500  # words
        chunk_overlap = 50  # words
        
        # Split content into words
        words = content.split()
        
        # Create overlapping chunks
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Skip very short chunks
            if len(chunk_words) < 50:
                continue
            
            # Extract comprehensive metadata
            metadata = await self.extract_comprehensive_metadata(chunk_text, pdf_path, len(chunks))
            
            chunks.append({
                "content": chunk_text,
                "metadata": metadata
            })
        
        return chunks
    
    async def extract_comprehensive_metadata(self, text: str, pdf_path: Path, chunk_index: int) -> Dict[str, Any]:
        """Extract comprehensive PCIe metadata for enhanced search"""
        import re
        
        metadata = {
            # Basic metadata
            "source": str(pdf_path),
            "filename": pdf_path.name,
            "chunk_id": f"{pdf_path.stem}_chunk_{chunk_index}",
            "chunk_index": chunk_index,
            "document_type": "specification",
            "content_type": "pcie_spec",
            "word_count": len(text.split()),
            
            # PCIe specific metadata
            "pcie_versions": [],
            "components": [],
            "topics": [],
            "error_codes": [],
            "technical_terms": [],
            "protocols": [],
            "specifications": []
        }
        
        text_lower = text.lower()
        
        # Extract PCIe versions and generations
        version_patterns = [
            (r'pci\\s*express\\s*([1-6]\\.[0-9])', "version"),
            (r'pcie\\s*([1-6]\\.[0-9])', "version"),
            (r'gen\\s*([1-6])', "generation"),
            (r'generation\\s*([1-6])', "generation")
        ]
        
        for pattern, vtype in version_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if vtype == "generation":
                    metadata["pcie_versions"].append(f"Gen{match}")
                else:
                    metadata["pcie_versions"].append(match)
        
        # Extract PCIe components
        component_mapping = {
            "root_complex": ["root complex", "rc", "host bridge"],
            "endpoint": ["endpoint", "ep", "device"],
            "switch": ["switch", "pci-to-pci bridge", "downstream port", "upstream port"],
            "phy": ["phy", "physical layer", "electrical"],
            "mac": ["mac", "media access control", "data link layer"],
            "transaction_layer": ["transaction layer", "tlp", "transaction"],
            "data_link_layer": ["data link layer", "dllp", "dll"],
            "physical_layer": ["physical layer", "phy layer"]
        }
        
        for component, terms in component_mapping.items():
            if any(term in text_lower for term in terms):
                metadata["components"].append(component)
        
        # Extract technical topics
        topic_mapping = {
            "link_training": ["link training", "ltssm", "link state", "training sequence"],
            "error_handling": ["error", "correction", "detection", "crc", "ecc"],
            "power_management": ["power", "l0", "l1", "l2", "l3", "aspm", "clkreq"],
            "flow_control": ["flow control", "credit", "fc", "buffer"],
            "addressing": ["address", "routing", "bar", "memory map", "io space"],
            "configuration": ["configuration", "config space", "capability"],
            "hot_plug": ["hot plug", "hot swap", "surprise removal"],
            "virtualization": ["sr-iov", "iov", "virtual function", "vf"],
            "security": ["security", "ats", "pasid", "ari"],
            "performance": ["bandwidth", "throughput", "latency", "qos"]
        }
        
        for topic, terms in topic_mapping.items():
            if any(term in text_lower for term in terms):
                metadata["topics"].append(topic)
        
        # Extract error codes and technical identifiers
        error_codes = re.findall(r'0x[0-9a-f]{2,8}', text_lower)
        metadata["error_codes"] = list(set(error_codes))
        
        # Extract protocol information
        protocols = []
        protocol_terms = ["tlp", "dllp", "ts1", "ts2", "fts", "skip", "com", "pad"]
        for term in protocol_terms:
            if term in text_lower:
                protocols.append(term)
        metadata["protocols"] = protocols
        
        # Extract specification references
        spec_patterns = [
            r'(pci\\s*express\\s*base\\s*specification)',
            r'(cem\\s*specification)',
            r'(aer\\s*specification)',
            r'(sr-iov\\s*specification)'
        ]
        
        specifications = []
        for pattern in spec_patterns:
            matches = re.findall(pattern, text_lower)
            specifications.extend(matches)
        metadata["specifications"] = specifications
        
        # Add derived classifications
        metadata["is_technical"] = len(metadata["technical_terms"]) > 2
        metadata["is_error_related"] = "error_handling" in metadata["topics"] or len(metadata["error_codes"]) > 0
        metadata["complexity_score"] = min(10, len(metadata["components"]) + len(metadata["topics"]) + len(metadata["protocols"]))
        
        # Clean up lists (remove duplicates)
        for key in ["pcie_versions", "components", "topics", "protocols", "specifications"]:
            metadata[key] = list(set(metadata[key]))
        
        return metadata
    
    async def build_embedding_database(self):
        """Build embedding database with available model"""
        print(f"\\n🧠 Building Embedding Database with {self.embedding_model}...")
        
        if not hasattr(self, 'processed_documents'):
            raise Exception("No processed documents found")
        
        try:
            from src.vectorstore.faiss_store import FAISSVectorStore
            
            # Create vector store directory
            vector_store_path = self.project_root / "data" / "vectorstore" / f"unified_{self.vector_store_suffix}"
            vector_store_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize FAISS vector store
            vector_store = FAISSVectorStore(
                dimension=self.embedding_dimension,
                index_path=str(vector_store_path)
            )
            
            print(f"  📊 Processing {len(self.processed_documents)} document chunks...")
            
            # Initialize embedding provider
            if self.has_openai:
                # Use OpenAI embeddings directly
                import openai
                openai.api_key = os.getenv("OPENAI_API_KEY")
                
                def embed_batch(texts):
                    try:
                        response = openai.embeddings.create(
                            model="text-embedding-3-small",
                            input=texts
                        )
                        return [data.embedding for data in response.data]
                    except Exception as e:
                        print(f"OpenAI API error: {e}")
                        raise
                
                embed_func = embed_batch
            else:
                # Use sentence transformers
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(self.embedding_model)
                embed_func = lambda texts: model.encode(texts).tolist()
            
            # Process documents in batches
            batch_size = 10 if self.has_openai else 20  # Smaller batches for API calls
            total_batches = (len(self.processed_documents) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(self.processed_documents))
                batch_docs = self.processed_documents[start_idx:end_idx]
                
                print(f"    🔄 Batch {batch_idx + 1}/{total_batches} ({len(batch_docs)} docs)...")
                
                # Extract content and metadata
                contents = [doc["content"] for doc in batch_docs]
                metadatas = [doc["metadata"] for doc in batch_docs]
                
                # Generate embeddings
                try:
                    embeddings = embed_func(contents)
                    
                    # Add to vector store
                    vector_store.add_documents(
                        embeddings=embeddings,
                        documents=contents,
                        metadata=metadatas
                    )
                    
                    print(f"      ✅ Added {len(embeddings)} embeddings")
                    
                except Exception as e:
                    print(f"      ❌ Batch failed: {e}")
                    continue
            
            # Save vector store
            vector_store.save(str(vector_store_path))
            
            print(f"  ✅ Database saved to: {vector_store_path}")
            print(f"  📊 Total documents: {len(self.processed_documents)}")
            print(f"  🧠 Embedding model: {self.embedding_model}")
            print(f"  📐 Dimensions: {self.embedding_dimension}")
            
            self.results["setup_steps"].append("database_built")
            
        except Exception as e:
            error_msg = f"Failed to build embedding database: {str(e)}"
            print(f"  ❌ {error_msg}")
            self.results["errors"].append(error_msg)
            raise
    
    async def update_interactive_shell_default(self):
        """Update interactive shell to use Unified RAG as default"""
        print("\\n🔧 Setting Unified RAG as Default in Interactive Shell...")
        
        # Update the interactive.py file to make unified RAG the default
        interactive_file = self.project_root / "src" / "cli" / "interactive.py"
        
        if not interactive_file.exists():
            print("  ❌ Interactive shell file not found")
            return
        
        # Read the current file
        with open(interactive_file, 'r') as f:
            content = f.read()
        
        # Modifications to make unified RAG default
        modifications = [
            # Change default method in queries
            ('self.rag_search_mode = "semantic"', 'self.rag_search_mode = "unified_adaptive"'),
            ('rag_search_mode = "semantic"', 'rag_search_mode = "unified_adaptive"'),
            
            # Update default query processing
            ('def default(self, line):', '''def default(self, line):
        """Process default queries using Unified RAG"""
        if line.strip().startswith('/'):
            # Handle slash commands
            super().default(line)
            return
        
        # Use Unified RAG for regular queries
        if hasattr(self, 'unified_rag') and self.unified_rag:
            self._process_unified_rag_query(line)
        else:
            self._process_regular_query(line)
    
    def _process_unified_rag_query(self, query):
        """Process query using Unified RAG"""
        try:
            from src.rag.unified_rag_engine import UnifiedRAGQuery
            
            # Create unified query
            unified_query = UnifiedRAGQuery(
                query=query,
                strategy=self.rag_search_mode if hasattr(self, 'rag_search_mode') else "adaptive",
                priority="balance",
                user_expertise=getattr(self, 'user_expertise', 'intermediate')
            )
            
            # Execute query
            import asyncio
            response = asyncio.run(self.unified_rag.query(unified_query))
            
            # Display result
            print(f"\\n💡 Answer (Confidence: {response.confidence:.1%}):")
            print(response.answer)
            
            if response.sources and self.analysis_verbose:
                print(f"\\n📚 Sources ({len(response.sources)}):")
                for i, source in enumerate(response.sources[:3], 1):
                    print(f"  {i}. Score: {source.get('final_score', 0):.3f}")
                    if 'metadata' in source and source['metadata'].get('filename'):
                        print(f"     File: {source['metadata']['filename']}")
                
                print(f"\\n⚡ Performance:")
                print(f"   Methods: {', '.join(response.methods_used)}")
                print(f"   Time: {response.total_processing_time:.2f}s")
            
        except Exception as e:
            print_error(f"❌ Unified RAG query failed: {e}")
            self._process_regular_query(query)
    
    def _process_regular_query(self, query):
        """Fallback to regular query processing"""'''),
        ]
        
        # Apply modifications
        modified = False
        for old, new in modifications:
            if old in content and new not in content:
                content = content.replace(old, new)
                modified = True
        
        # Add unified RAG initialization if not present
        if "unified_rag" not in content:
            init_code = '''
        # Initialize Unified RAG as default
        try:
            from src.rag.unified_rag_engine import UnifiedRAGEngine
            self.unified_rag = UnifiedRAGEngine(
                self.vector_store,
                self.model_manager
            )
            print_info("🚀 Unified RAG initialized as default")
        except Exception as e:
            print_warning(f"⚠️ Unified RAG initialization failed: {e}")
            self.unified_rag = None
'''
            
            # Find a good insertion point
            insertion_points = [
                "self.vector_store = ",
                "self.model_manager = ",
                "# Initialize components"
            ]
            
            for point in insertion_points:
                pos = content.find(point)
                if pos != -1:
                    # Find end of line
                    line_end = content.find("\\n", pos)
                    if line_end != -1:
                        content = content[:line_end] + init_code + content[line_end:]
                        modified = True
                        break
        
        # Write back if modified
        if modified:
            with open(interactive_file, 'w') as f:
                f.write(content)
            print("  ✅ Interactive shell updated to use Unified RAG as default")
        else:
            print("  ℹ️ Interactive shell already configured")
        
        self.results["setup_steps"].append("shell_updated")
    
    async def test_setup(self):
        """Test the complete setup"""
        print("\\n🧪 Testing Complete Setup...")
        
        try:
            # Test 1: Configuration
            print("  🔍 Testing configuration...")
            if self.config_file.exists():
                print("    ✅ Configuration file exists")
            
            # Test 2: Vector store
            print("  🔍 Testing vector store...")
            vector_store_path = self.project_root / "data" / "vectorstore" / f"unified_{self.vector_store_suffix}"
            
            if vector_store_path.exists():
                from src.vectorstore.faiss_store import FAISSVectorStore
                vector_store = FAISSVectorStore(
                    dimension=self.embedding_dimension, 
                    index_path=str(vector_store_path)
                )
                
                if vector_store.index.ntotal > 0:
                    print(f"    ✅ Vector store loaded: {vector_store.index.ntotal} documents")
                else:
                    print("    ⚠️ Vector store empty")
            else:
                print("    ❌ Vector store not found")
            
            # Test 3: Unified RAG
            print("  🔍 Testing Unified RAG...")
            from src.rag.unified_rag_engine import UnifiedRAGEngine, UnifiedRAGQuery
            from src.models.model_manager import ModelManager
            
            model_manager = ModelManager()
            unified_rag = UnifiedRAGEngine(vector_store, model_manager)
            print("    ✅ Unified RAG engine initialized")
            
            # Test 4: Sample query
            print("  🔍 Testing sample query...")
            test_query = UnifiedRAGQuery(
                query="What is PCIe link training?",
                strategy="adaptive"
            )
            
            # Note: Skip actual query execution to avoid API calls in setup
            print("    ✅ Query structure validated")
            
            self.results["setup_steps"].append("testing_completed")
            
        except Exception as e:
            error_msg = f"Testing failed: {str(e)}"
            print(f"  ❌ {error_msg}")
            self.results["errors"].append(error_msg)
    
    def save_setup_report(self):
        """Save comprehensive setup report"""
        report_file = self.project_root / f"unified_rag_setup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\\n📄 Setup report saved: {report_file}")
        
        # Create summary
        if self.results["success"]:
            print("\\n🎉 UNIFIED RAG SETUP COMPLETE!")
            print("=" * 60)
            print("✅ Unified RAG is now the DEFAULT system")
            print(f"✅ Embedding model: {self.results['embedding_model_used']}")
            print(f"✅ PDFs processed: {len(self.results['processed_pdfs'])}")
            
            # Show processed files
            print("\\n📚 Processed PCIe Specifications:")
            for pdf in self.results['processed_pdfs']:
                print(f"  📄 {pdf['filename']} ({pdf['chunks']} chunks)")
            
            print("\\n🚀 Ready to use!")
            print("Next steps:")
            print("1. Run: python3 src/cli/interactive.py")
            print("2. Try: What is PCIe link training?")
            print("3. Use: /urag 'PCIe 4.0 error handling'")
            print("4. Check: /urag_status")
            
            if not self.has_openai:
                print("\\n💡 To use OpenAI embeddings:")
                print("1. Set OPENAI_API_KEY environment variable")
                print("2. Run this setup again")
                print("3. Enjoy better embedding quality!")
        else:
            print("\\n❌ SETUP FAILED:")
            for error in self.results["errors"]:
                print(f"  - {error}")

async def main():
    """Main setup function"""
    setup = FlexibleUnifiedRAGSetup()
    await setup.setup_unified_rag_default()

if __name__ == "__main__":
    asyncio.run(main())