#!/usr/bin/env python3
"""
Quick OpenAI embeddings setup - processes a subset of documents for testing
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def quick_openai_setup():
    print("🚀 Quick OpenAI Embeddings Setup")
    print("=" * 50)
    
    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set!")
        print("Run: export $(grep -v '^#' .env | xargs)")
        return False
    
    print("✅ OpenAI API key found")
    
    try:
        import openai
        import numpy as np
        from src.vectorstore.faiss_store import FAISSVectorStore
        
        # Set API key
        openai.api_key = api_key
        
        # Create output directory
        output_dir = Path("data/vectorstore/unified_openai_1536d")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS store
        vector_store = FAISSVectorStore(
            dimension=1536,  # OpenAI text-embedding-3-small
            index_path=str(output_dir)
        )
        
        # Test documents
        test_docs = [
            {
                "content": "PCIe link training is the process by which two PCIe devices establish communication. It involves detecting the presence of a link partner, negotiating link width and speed, and achieving bit lock and symbol lock.",
                "metadata": {
                    "source": "quick_test",
                    "topic": "link_training",
                    "filename": "test_doc.txt"
                }
            },
            {
                "content": "The Link Training and Status State Machine (LTSSM) manages PCIe link states including Detect, Polling, Configuration, L0 (normal operation), and various low-power states (L0s, L1, L2).",
                "metadata": {
                    "source": "quick_test", 
                    "topic": "ltssm",
                    "filename": "test_doc.txt"
                }
            },
            {
                "content": "PCIe Gen5 operates at 32 GT/s with 128b/130b encoding, while PCIe Gen6 doubles that to 64 GT/s using PAM4 signaling and advanced FEC (Forward Error Correction).",
                "metadata": {
                    "source": "quick_test",
                    "topic": "generations",
                    "filename": "test_doc.txt"
                }
            },
            {
                "content": "Flow control in PCIe uses credits to manage buffer space. Posted, Non-Posted, and Completion transactions each have separate credit pools for headers and data.",
                "metadata": {
                    "source": "quick_test",
                    "topic": "flow_control", 
                    "filename": "test_doc.txt"
                }
            },
            {
                "content": "Advanced Error Reporting (AER) provides detailed error logging including correctable errors (like bad TLP or bad DLLP) and uncorrectable errors (like poisoned TLP or completion timeout).",
                "metadata": {
                    "source": "quick_test",
                    "topic": "error_handling",
                    "filename": "test_doc.txt"
                }
            }
        ]
        
        print(f"\n📊 Processing {len(test_docs)} test documents...")
        
        # Extract content and metadata
        contents = [doc["content"] for doc in test_docs]
        metadatas = [doc["metadata"] for doc in test_docs]
        
        # Generate embeddings
        print("🔄 Generating OpenAI embeddings...")
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=contents
        )
        
        embeddings = [data.embedding for data in response.data]
        print(f"✅ Generated {len(embeddings)} embeddings")
        
        # Add to vector store
        vector_store.add_documents(
            embeddings=embeddings,
            documents=contents,
            metadata=metadatas
        )
        
        # Save vector store
        vector_store.save(str(output_dir))
        print(f"\n✅ Vector store saved to: {output_dir}")
        
        # Test search
        print("\n🧪 Testing search...")
        test_query = "How does PCIe link training work?"
        
        # Get embedding for query
        query_response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=test_query
        )
        query_embedding = query_response.data[0].embedding
        
        # Search
        results = vector_store.search(query_embedding, k=3)
        
        print(f"\nQuery: '{test_query}'")
        print("Results:")
        for i, (doc, meta, score) in enumerate(results, 1):
            print(f"\n{i}. Score: {score:.3f}")
            print(f"   Topic: {meta.get('topic', 'N/A')}")
            print(f"   Content: {doc[:100]}...")
        
        # Update config
        config_path = Path("configs/settings.yaml")
        config = {
            "default_model": "gpt-4o-mini",
            "default_embedding_model": "text-embedding-3-small",
            "default_rag_mode": "unified_adaptive",
            "embedding_dimension": 1536,
            "vector_store_path": "./data/vectorstore/unified_openai_1536d",
            "enable_unified_rag": True,
            "unified_rag_default": True
        }
        
        config_path.parent.mkdir(exist_ok=True)
        
        try:
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"\n✅ Config updated: {config_path}")
        except ImportError:
            # Fallback to JSON
            json_path = config_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"\n✅ Config saved as JSON: {json_path}")
        
        print("\n🎉 Quick setup complete!")
        print("\nNext steps:")
        print("1. Run full setup: python3 setup_unified_rag_flexible.py")
        print("2. Or test now: python3 src/cli/interactive.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Source .env if needed
    if not os.getenv("OPENAI_API_KEY"):
        print("Attempting to source .env file...")
        os.system("export $(grep -v '^#' .env | xargs)")
    
    quick_openai_setup()