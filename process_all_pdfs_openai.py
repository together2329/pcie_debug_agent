#!/usr/bin/env python3
"""
Process all PCIe PDFs with OpenAI embeddings
Handles rate limiting and retries
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class OpenAIPDFProcessor:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.specs_folder = self.project_root / "data" / "specs"
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.processed_count = 0
        self.total_chunks = 0
        
    async def process_all_pdfs(self):
        """Process all PDFs with OpenAI embeddings"""
        print("🚀 Processing All PDFs with OpenAI Embeddings")
        print("=" * 60)
        
        if not self.api_key:
            print("❌ OPENAI_API_KEY not set!")
            return False
        
        print("✅ OpenAI API key found")
        
        try:
            import openai
            import PyPDF2
            from src.vectorstore.faiss_store import FAISSVectorStore
            
            # Set API key
            openai.api_key = self.api_key
            
            # Load existing vector store or create new
            output_dir = Path("data/vectorstore/unified_openai_1536d")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                vector_store = FAISSVectorStore.load(str(output_dir))
                print(f"✅ Loaded existing vector store: {len(vector_store.documents)} documents")
            except:
                vector_store = FAISSVectorStore(
                    dimension=1536,
                    index_path=str(output_dir)
                )
                print("✅ Created new vector store")
            
            # Get PDFs to process
            pdf_files = list(self.specs_folder.glob("*.pdf"))
            print(f"\n📚 Found {len(pdf_files)} PDFs to process")
            
            # Process each PDF
            for pdf_idx, pdf_file in enumerate(pdf_files, 1):
                print(f"\n[{pdf_idx}/{len(pdf_files)}] Processing: {pdf_file.name}")
                
                # Extract text
                chunks = await self.extract_pdf_chunks(pdf_file)
                if not chunks:
                    print(f"  ⚠️ No chunks extracted")
                    continue
                
                print(f"  📄 Extracted {len(chunks)} chunks")
                self.total_chunks += len(chunks)
                
                # Process in small batches to avoid rate limits
                batch_size = 5  # Small batches for rate limit management
                for batch_idx in range(0, len(chunks), batch_size):
                    batch_end = min(batch_idx + batch_size, len(chunks))
                    batch = chunks[batch_idx:batch_end]
                    
                    print(f"  🔄 Batch {batch_idx//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}...", end='', flush=True)
                    
                    # Extract content and metadata
                    contents = [chunk["content"] for chunk in batch]
                    metadatas = [chunk["metadata"] for chunk in batch]
                    
                    # Generate embeddings with retry
                    for attempt in range(3):
                        try:
                            response = openai.embeddings.create(
                                model="text-embedding-3-small",
                                input=contents
                            )
                            embeddings = [data.embedding for data in response.data]
                            break
                        except Exception as e:
                            if "rate_limit" in str(e).lower() and attempt < 2:
                                wait_time = 2 ** attempt
                                print(f" (rate limit, waiting {wait_time}s)", end='', flush=True)
                                time.sleep(wait_time)
                            else:
                                raise e
                    
                    # Add to vector store
                    vector_store.add_documents(
                        embeddings=embeddings,
                        documents=contents,
                        metadata=metadatas
                    )
                    
                    self.processed_count += len(batch)
                    print(f" ✅ ({self.processed_count} total)")
                    
                    # Small delay to avoid rate limits
                    if batch_idx + batch_size < len(chunks):
                        time.sleep(0.5)
                
                # Save after each PDF
                vector_store.save(str(output_dir))
                print(f"  💾 Saved to disk ({len(vector_store.documents)} total documents)")
            
            print(f"\n🎉 Processing complete!")
            print(f"📊 Statistics:")
            print(f"  - PDFs processed: {len(pdf_files)}")
            print(f"  - Total chunks: {self.total_chunks}")
            print(f"  - Documents in store: {len(vector_store.documents)}")
            print(f"  - Vector store: {output_dir}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def extract_pdf_chunks(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract chunks from a PDF file"""
        try:
            import PyPDF2
            import re
            
            chunks = []
            chunk_size = 500  # words
            chunk_overlap = 50
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract all text
                full_text = []
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            # Clean text
                            text = text.replace('\n', ' ').replace('\r', ' ')
                            text = ' '.join(text.split())
                            full_text.append(f"[Page {page_num + 1}] {text}")
                    except:
                        pass
                
                # Join all text
                combined_text = " ".join(full_text)
                words = combined_text.split()
                
                # Create chunks
                for i in range(0, len(words), chunk_size - chunk_overlap):
                    chunk_words = words[i:i + chunk_size]
                    if len(chunk_words) < 50:  # Skip very short chunks
                        continue
                    
                    chunk_text = " ".join(chunk_words)
                    
                    # Extract metadata
                    metadata = {
                        "source": str(pdf_path),
                        "filename": pdf_path.name,
                        "chunk_index": len(chunks),
                        "document_type": "specification",
                        "content_type": "pcie_spec"
                    }
                    
                    # Extract PCIe-specific metadata
                    if "link training" in chunk_text.lower():
                        metadata["topics"] = metadata.get("topics", []) + ["link_training"]
                    if "ltssm" in chunk_text.lower():
                        metadata["topics"] = metadata.get("topics", []) + ["ltssm"]
                    if "error" in chunk_text.lower():
                        metadata["topics"] = metadata.get("topics", []) + ["error_handling"]
                    if "gen" in chunk_text.lower() and re.search(r'gen\s*[1-6]', chunk_text.lower()):
                        metadata["topics"] = metadata.get("topics", []) + ["pcie_generations"]
                    
                    chunks.append({
                        "content": chunk_text,
                        "metadata": metadata
                    })
            
            return chunks
            
        except Exception as e:
            print(f"  ❌ Error extracting from {pdf_path.name}: {e}")
            return []

async def main():
    processor = OpenAIPDFProcessor()
    await processor.process_all_pdfs()

if __name__ == "__main__":
    asyncio.run(main())