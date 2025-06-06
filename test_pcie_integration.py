#!/usr/bin/env python3
"""Test PCIe integration in interactive mode"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_interactive_integration():
    """Test PCIe mode integration in interactive shell"""
    print("🧪 Testing PCIe Integration")
    print("=" * 50)
    
    try:
        from src.cli.interactive import PCIeDebugShell
        
        # Create shell instance
        shell = PCIeDebugShell(model_id="gpt-4o-mini", verbose=True)
        
        print("✅ Interactive shell created successfully")
        
        # Test PCIe mode detection
        test_queries = [
            "What are LTSSM states?",
            "PCIe link training failed",
            "Explain TLP format",
            "How to debug completion timeout",
            "FLR compliance violations"
        ]
        
        print("\n🔍 Testing PCIe Query Detection:")
        for query in test_queries:
            is_pcie = shell._is_pcie_query(query)
            print(f"   '{query}' → {'✅ PCIe' if is_pcie else '❌ Not PCIe'}")
        
        # Test PCIe mode switching
        print(f"\n🔧 Testing PCIe Mode Switch:")
        print(f"   Current mode: {getattr(shell, 'rag_search_mode', 'none')}")
        
        # Switch to PCIe mode
        shell.onecmd("/rag_mode pcie")
        print(f"   After switch: {getattr(shell, 'rag_search_mode', 'none')}")
        
        # Test a simple query
        print("\n💬 Testing Query Processing:")
        shell.onecmd("What are PCIe power states?")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_integration():
    """Test CLI command integration"""
    print("\n🖥️  Testing CLI Integration")
    print("=" * 50)
    
    try:
        # Test CLI command availability
        from src.cli.main import cli
        from click.testing import CliRunner
        
        runner = CliRunner()
        
        # Test help
        result = runner.invoke(cli, ['--help'])
        if 'pcie' in result.output:
            print("✅ PCIe command available in CLI help")
        else:
            print("❌ PCIe command not found in CLI help")
            return False
        
        # Test pcie help
        result = runner.invoke(cli, ['pcie', '--help'])
        if result.exit_code == 0:
            print("✅ PCIe subcommand help works")
        else:
            print(f"❌ PCIe subcommand help failed: {result.output}")
            return False
        
        # Test pcie stats
        result = runner.invoke(cli, ['pcie', 'stats'])
        if result.exit_code == 0:
            print("✅ PCIe stats command works")
        else:
            print(f"❌ PCIe stats command failed: {result.output}")
            return False
        
        print("✅ CLI integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ CLI test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests"""
    print("🚀 PCIe Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Interactive Mode", test_interactive_integration),
        ("CLI Integration", test_cli_integration)
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
    print("\n📊 Integration Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All integration tests passed! PCIe mode is fully integrated.")
    else:
        print("⚠️  Some integration tests failed. Check output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)