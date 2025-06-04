#!/usr/bin/env python3
"""
Test Smart Context-Aware RAG
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from smart_context_agent import SmartContextAgent

def test_smart_rag():
    """Test smart RAG with problematic queries"""
    
    agent = SmartContextAgent()
    
    test_queries = [
        "why FLR UR return happened?",
        "what is CRS vs UR difference?", 
        "when does CA completion occur?",
        "explain LTSSM states in PCIe",
        "AER error handling for TLP",
        "malformed TLP causes UR response"
    ]
    
    print("🧪 Testing Smart Context-Aware RAG")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 40)
        
        result = agent.smart_search(query)
        
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Terms Found: {len(result.context_matches)}")
        
        if result.context_matches:
            print("Terminology:")
            for match in result.context_matches:
                print(f"  ✅ {match.term}: {match.definition}")
        
        print(f"Intent: {result.reasoning}")
        print()

def demonstrate_improvement():
    """Demonstrate the improvement over original system"""
    
    agent = SmartContextAgent()
    
    print("🎯 DEMONSTRATION: Smart RAG vs Original Response")
    print("=" * 70)
    
    query = "why FLR UR return happened?"
    
    print(f"\n❌ **Original System Response (INCORRECT):**")
    print("FLR UR return indicates uncorrectable error during Function-Level Reset")
    print("Expected CRS (Completion Request Status) but got uncorrectable error...")
    print("Confidence: ~76%")
    
    print(f"\n✅ **Smart Context-Aware Response (CORRECT):**")
    result = agent.smart_search(query)
    print(result.answer)
    print(f"Confidence: {result.confidence:.1%}")
    
    print(f"\n🔍 **Key Corrections:**")
    print("• UR = Unsupported Request (NOT uncorrectable error)")
    print("• CRS = Configuration Retry Status (NOT Completion Request Status)")
    print("• Context-aware analysis based on actual PCIe terminology")
    print("• Proper relationship between FLR and completion statuses")

if __name__ == "__main__":
    test_smart_rag()
    print("\n" + "="*60)
    demonstrate_improvement()