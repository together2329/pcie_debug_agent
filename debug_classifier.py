#!/usr/bin/env python3
"""
Debug why the classifier is returning tlp instead of error_handling
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def debug_classifier_logic():
    """Debug the classifier step by step"""
    print("🔍 DEBUGGING CLASSIFIER LOGIC")
    print("=" * 50)
    
    try:
        from src.rag.pcie_knowledge_classifier import PCIeKnowledgeClassifier
        
        classifier = PCIeKnowledgeClassifier()
        
        # Test with the exact query
        test_query = "My PCIe device shows completion timeout errors during high traffic"
        print(f"Test query: {test_query}")
        
        # Test direct classification of the query
        result = classifier.classify_content(test_query)
        print(f"\nDirect query classification:")
        print(f"  Category: {result.category.value}")
        print(f"  Facts: {len(result.facts)}")
        
        # Now let's test what content the RAG system might be getting
        # Simulate retrieved content that might be coming from the vector store
        retrieved_contents = [
            "PCIe Transaction Layer Packet (TLP) header format specifications",
            "Memory read and write TLP formats for PCIe protocol",
            "TLP routing and addressing in PCIe switches",
            "Completion timeout handling in PCIe transaction layer",
            "Error reporting mechanisms for PCIe TLP processing"
        ]
        
        print(f"\n🔍 Testing various retrieved content:")
        for i, content in enumerate(retrieved_contents):
            result = classifier.classify_content(content)
            print(f"  {i+1}. Content: {content[:50]}...")
            print(f"     Category: {result.category.value}")
            
        # Test the category scoring directly
        print(f"\n🔧 Testing category scoring for query:")
        content_lower = test_query.lower()
        
        for category, keywords in classifier.category_keywords.items():
            score = sum(content_lower.count(keyword.lower()) for keyword in keywords)
            print(f"  {category.value}: score={score}")
            if score > 0:
                print(f"    Matching keywords: {[kw for kw in keywords if kw.lower() in content_lower]}")
                
        # Test with timeout-specific content
        timeout_content = "completion timeout errors during high traffic"
        result = classifier.classify_content(timeout_content)
        print(f"\nTimeout-specific content: {timeout_content}")
        print(f"  Category: {result.category.value}")
        print(f"  Facts: {len(result.facts)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the classifier debug"""
    success = debug_classifier_logic()
    
    if success:
        print("\n🎯 Classifier Debug Complete")
    else:
        print("\n❌ Classifier Debug Failed")
        
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)