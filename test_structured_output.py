#!/usr/bin/env python3
"""
Test structured output functionality for PCIe mode
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_structured_classes():
    """Test structured output classes"""
    print("🧪 Testing Structured Output Classes")
    print("=" * 50)
    
    try:
        from src.rag.pcie_structured_output import (
            PCIeConcept, PCIeChunkMetadata, PCIeSearchResult,
            PCIeQueryResponse, PCIeStructuredOutputFormatter,
            TechnicalLevel, PCIeLayer, SemanticType
        )
        
        # Test PCIe concept
        concept = PCIeConcept(
            name="L0",
            category="LTSSM State",
            confidence=0.95
        )
        print(f"✅ PCIeConcept: {concept.name} ({concept.category})")
        
        # Test metadata
        metadata = PCIeChunkMetadata(
            source_file="ltssm_guide.md",
            chunk_id="chunk_001",
            technical_level=TechnicalLevel.INTERMEDIATE,
            pcie_layer=PCIeLayer.PHYSICAL,
            semantic_type=SemanticType.SPECIFICATION,
            pcie_concepts=[concept]
        )
        print(f"✅ PCIeChunkMetadata: {metadata.source_file}")
        
        # Test search result
        result = PCIeSearchResult(
            content="PCIe L0 state is the active state where data transfer occurs...",
            score=0.85,
            metadata=metadata,
            highlighted_terms=["L0", "active state"]
        )
        print(f"✅ PCIeSearchResult: Score {result.score:.3f}")
        
        # Test query response
        response = PCIeQueryResponse(
            query="What is PCIe L0 state?",
            total_results=1,
            results=[result],
            model_used="text-embedding-3-small"
        )
        print(f"✅ PCIeQueryResponse: {response.total_results} results")
        
        # Test JSON serialization
        json_output = response.to_json()
        parsed = json.loads(json_output)
        print(f"✅ JSON serialization: {len(json_output)} characters")
        
        # Test summary
        summary = response.to_summary()
        print(f"✅ Summary generation: {len(summary.split())} words")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_json_output():
    """Test CLI JSON output"""
    print("\n🖥️  Testing CLI JSON Output")
    print("=" * 50)
    
    try:
        from click.testing import CliRunner
        from src.cli.commands.pcie_rag import pcie
        
        runner = CliRunner()
        
        # Test JSON flag in help
        result = runner.invoke(pcie, ['query', '--help'])
        if '--json' in result.output:
            print("✅ --json flag available in CLI help")
        else:
            print("❌ --json flag not found in CLI help")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ CLI test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_interactive_json_command():
    """Test interactive shell JSON command"""
    print("\n💬 Testing Interactive JSON Command")
    print("=" * 50)
    
    try:
        from src.cli.interactive import PCIeDebugShell
        
        # Create shell instance
        shell = PCIeDebugShell(model_id="gpt-4o-mini", verbose=False)
        
        # Check if json_query command exists
        if hasattr(shell, 'do_json_query'):
            print("✅ /json_query command available")
        else:
            print("❌ /json_query command not found")
            return False
        
        # Test help
        shell.onecmd('/json_query')  # Should show usage
        print("✅ Command help displayed")
        
        return True
        
    except Exception as e:
        print(f"❌ Interactive test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_structured_output():
    """Show example of structured output"""
    print("\n📋 Structured Output Examples")
    print("=" * 50)
    
    # Example 1: CLI JSON query
    print("1. CLI JSON Query:")
    print("   ./pcie-debug pcie query \"PCIe LTSSM states\" --json --pretty")
    print()
    
    # Example 2: Interactive JSON query
    print("2. Interactive JSON Query:")
    print("   ./pcie-debug")
    print("   /rag_mode pcie")
    print("   /json_query What are PCIe completion timeout causes?")
    print()
    
    # Example 3: Python API
    print("3. Python API:")
    print("""
   from src.rag.pcie_rag_engine import create_pcie_rag_engine
   
   engine = create_pcie_rag_engine()
   response = engine.query(
       "PCIe link training failure",
       return_structured=True
   )
   
   # Get JSON
   json_data = response.to_json(indent=2)
   
   # Get summary
   summary = response.to_summary()
   
   # Access structured data
   for result in response.results:
       print(f"Score: {result.score}")
       print(f"Layer: {result.metadata.pcie_layer}")
       print(f"Concepts: {[c.name for c in result.metadata.pcie_concepts]}")
   """)

def main():
    """Run all structured output tests"""
    print("🚀 PCIe Structured Output Test Suite")
    print("=" * 60)
    
    tests = [
        ("Structured Classes", test_structured_classes),
        ("CLI JSON Output", test_cli_json_output),
        ("Interactive JSON Command", test_interactive_json_command)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"💥 {test_name} crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Structured Output Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    # Show examples
    demonstrate_structured_output()
    
    if passed == total:
        print("\n🎉 All structured output tests passed!")
        print("✅ JSON output is ready for use in PCIe mode")
    else:
        print("\n⚠️  Some tests failed. Check output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)