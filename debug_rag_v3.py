#!/usr/bin/env python3
"""
Debug the RAG v3 system to see why enhanced logic isn't showing
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def debug_rag_v3_system():
    """Debug what the RAG v3 system is actually doing"""
    print("🔍 DEBUGGING RAG v3 SYSTEM")
    print("=" * 50)
    
    try:
        # Test direct classification
        from src.rag.pcie_knowledge_classifier import PCIeKnowledgeClassifier
        
        classifier = PCIeKnowledgeClassifier()
        test_content = "My PCIe device shows completion timeout errors during high traffic"
        
        result = classifier.classify_content(test_content)
        print(f"Direct classifier test:")
        print(f"  Content: {test_content}")
        print(f"  Category: {result.category.value}")
        print(f"  Facts: {len(result.facts)}")
        print(f"  Keywords: {result.keywords}")
        
        if result.category.value == "error_handling":
            print("✅ Direct classifier working correctly")
        else:
            print(f"❌ Direct classifier wrong: {result.category.value} (expected error_handling)")
            
        # Test if RAG v3 can be instantiated
        print(f"\n🔧 Testing RAG v3 instantiation:")
        try:
            from src.rag.enhanced_rag_engine_v3 import EnhancedRAGEngineV3, EnhancedRAGQuery
            
            # Create mock vector store
            class MockVectorStore:
                def __init__(self):
                    self.documents = [test_content]
                    self.metadata = [{"source": "test"}]
                    
                def search(self, query, k=5):
                    return [(test_content, {"source": "test"}, 0.8)]
            
            # Create mock model manager  
            class MockModelManager:
                def generate_embeddings(self, texts):
                    import numpy as np
                    return [np.random.random(384) for _ in texts]
                    
                def generate_completion(self, prompt, temperature=0.1, max_tokens=2000):
                    return "Enhanced test completion about PCIe timeout errors"
            
            vector_store = MockVectorStore()
            model_manager = MockModelManager()
            
            # Initialize v3 engine
            engine = EnhancedRAGEngineV3(
                vector_store=vector_store,
                model_manager=model_manager,
                use_hybrid_search=False  # Simplified for testing
            )
            print("✅ RAG v3 engine instantiated successfully")
            
            # Test a query
            query = EnhancedRAGQuery(
                query=test_content,
                verify_answer=True,
                normalize_question=True
            )
            
            response = engine.query(query)
            print(f"\n📊 RAG v3 Response:")
            print(f"  Confidence: {response.confidence:.1%}")
            print(f"  Knowledge items: {len(response.knowledge_items)}")
            
            if response.knowledge_items:
                for i, item in enumerate(response.knowledge_items[:3]):
                    print(f"  Item {i+1}: {item.category.value} ({len(item.facts)} facts)")
                    
            if response.normalized_question:
                print(f"  Normalized: {response.normalized_question.normalized_form}")
                print(f"  Concepts: {response.normalized_question.key_concepts}")
                
            # Check if we're getting the expected results
            if response.confidence > 0.7:
                print("✅ RAG v3 producing high confidence results")
            else:
                print(f"⚠️ RAG v3 confidence low: {response.confidence:.1%}")
                
            if response.knowledge_items and response.knowledge_items[0].category.value == "error_handling":
                print("✅ RAG v3 using enhanced classification correctly")
            else:
                category = response.knowledge_items[0].category.value if response.knowledge_items else "none"
                print(f"❌ RAG v3 classification wrong: {category}")
                
            return True
            
        except Exception as e:
            print(f"❌ RAG v3 instantiation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Debug test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the debug test"""
    success = debug_rag_v3_system()
    
    if success:
        print("\n🎯 RAG v3 Debug Complete - Components Working")
    else:
        print("\n❌ RAG v3 Debug Found Issues")
        
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)