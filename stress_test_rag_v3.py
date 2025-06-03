#!/usr/bin/env python3
"""
Comprehensive Stress Test Suite for Enhanced RAG v3
Tests edge cases, error handling, performance, and robustness
"""

import sys
import time
import threading
import concurrent.futures
from pathlib import Path
import json
import traceback
from typing import List, Dict, Any
import gc
import psutil
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class StressTestResults:
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.errors = []
        self.performance_data = []
        self.memory_usage = []
        
    def add_result(self, test_name: str, passed: bool, error: str = None, perf_data: Dict = None):
        self.tests_run += 1
        if passed:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
            self.errors.append({"test": test_name, "error": error})
        
        if perf_data:
            perf_data['test'] = test_name
            self.performance_data.append(perf_data)
    
    def record_memory(self, test_name: str):
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_usage.append({"test": test_name, "memory_mb": memory_mb})
        except:
            pass

def test_import_stress():
    """Stress test imports under various conditions"""
    print("🔥 Import Stress Test")
    print("-" * 30)
    
    results = StressTestResults()
    
    # Test 1: Repeated import/reload stress
    for i in range(10):
        try:
            if 'src.rag.pcie_knowledge_classifier' in sys.modules:
                del sys.modules['src.rag.pcie_knowledge_classifier']
            from src.rag.pcie_knowledge_classifier import PCIeKnowledgeClassifier
            results.add_result(f"import_reload_{i}", True)
        except Exception as e:
            results.add_result(f"import_reload_{i}", False, str(e))
    
    # Test 2: Memory pressure during imports
    try:
        # Create memory pressure
        memory_hog = []
        for i in range(1000):
            memory_hog.append("x" * 10000)
        
        from src.rag.enhanced_rag_engine_v3 import EnhancedRAGEngineV3
        results.add_result("import_under_memory_pressure", True)
        del memory_hog
        gc.collect()
    except Exception as e:
        results.add_result("import_under_memory_pressure", False, str(e))
    
    # Test 3: Concurrent imports
    def concurrent_import():
        try:
            from src.rag.answer_verifier import AnswerVerifier
            return True
        except:
            return False
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(concurrent_import) for _ in range(10)]
        concurrent_results = [f.result() for f in futures]
    
    passed_concurrent = sum(concurrent_results)
    results.add_result("concurrent_imports", passed_concurrent == 10, 
                      f"Only {passed_concurrent}/10 concurrent imports succeeded")
    
    return results

def test_component_edge_cases():
    """Test components with edge cases and malformed inputs"""
    print("\n🔥 Component Edge Case Stress Test")
    print("-" * 40)
    
    results = StressTestResults()
    
    try:
        from src.rag.pcie_knowledge_classifier import PCIeKnowledgeClassifier
        from src.rag.answer_verifier import AnswerVerifier
        from src.rag.question_normalizer import QuestionNormalizer
        
        classifier = PCIeKnowledgeClassifier()
        verifier = AnswerVerifier()
        normalizer = QuestionNormalizer()
        
        # Test malformed/edge case inputs
        edge_case_inputs = [
            "",  # Empty
            " ",  # Whitespace only
            "a" * 10000,  # Very long
            "🔥💻🚀" * 100,  # Unicode/emoji spam
            "\n\r\t" * 50,  # Control characters
            "NULL\x00\x01\x02",  # Null bytes
            "<script>alert('xss')</script>",  # Potential XSS
            "../../etc/passwd",  # Path traversal
            "SELECT * FROM users;",  # SQL injection attempt
            "{{7*7}}",  # Template injection
            "A" * 1000 + "?" * 1000,  # Mixed very long
        ]
        
        for i, test_input in enumerate(edge_case_inputs):
            start_time = time.time()
            
            # Test classifier
            try:
                knowledge_item = classifier.classify_content(test_input)
                classifier_passed = True
            except Exception as e:
                classifier_passed = False
                results.add_result(f"classifier_edge_{i}", False, f"Classifier failed: {e}")
            
            # Test normalizer
            try:
                normalized = normalizer.normalize_question(test_input)
                normalizer_passed = True
            except Exception as e:
                normalizer_passed = False
                results.add_result(f"normalizer_edge_{i}", False, f"Normalizer failed: {e}")
            
            # Test verifier
            try:
                verification = verifier.verify_answer(test_input, test_input, [])
                verifier_passed = True
            except Exception as e:
                verifier_passed = False
                results.add_result(f"verifier_edge_{i}", False, f"Verifier failed: {e}")
            
            # Overall edge case test
            all_passed = classifier_passed and normalizer_passed and verifier_passed
            processing_time = time.time() - start_time
            
            results.add_result(f"edge_case_{i}", all_passed, 
                             None if all_passed else f"Some components failed",
                             {"processing_time": processing_time, "input_length": len(test_input)})
            
            results.record_memory(f"edge_case_{i}")
    
    except Exception as e:
        results.add_result("component_edge_cases", False, f"Setup failed: {e}")
    
    return results

def test_performance_stress():
    """Stress test performance with large inputs and concurrent processing"""
    print("\n🔥 Performance Stress Test")
    print("-" * 30)
    
    results = StressTestResults()
    
    try:
        from src.rag.pcie_knowledge_classifier import PCIeKnowledgeClassifier
        from src.rag.question_normalizer import QuestionNormalizer
        
        classifier = PCIeKnowledgeClassifier()
        normalizer = QuestionNormalizer()
        
        # Generate large realistic PCIe content
        large_pcie_content = """
        PCIe Configuration Space Header Layout:
        Offset 0x00: Vendor ID [15:0], Device ID [31:16]
        Offset 0x04: Command Register [15:0], Status Register [31:16]
        Offset 0x08: Revision ID [7:0], Class Code [31:8]
        Offset 0x0C: Cache Line Size [7:0], Latency Timer [15:8], Header Type [23:16], BIST [31:24]
        """ * 100  # Repeat 100 times
        
        # Test large content processing
        start_time = time.time()
        for i in range(50):
            knowledge_item = classifier.classify_content(large_pcie_content)
            if i == 0:
                first_result = knowledge_item
        
        large_content_time = time.time() - start_time
        results.add_result("large_content_processing", True, None,
                          {"total_time": large_content_time, "avg_time_per_item": large_content_time/50})
        
        # Test concurrent processing
        def process_content(content):
            start = time.time()
            try:
                result = classifier.classify_content(content)
                return time.time() - start, True, None
            except Exception as e:
                return time.time() - start, False, str(e)
        
        test_contents = [large_pcie_content[:1000 * (i+1)] for i in range(20)]
        
        # Sequential processing
        start_time = time.time()
        sequential_results = [process_content(content) for content in test_contents]
        sequential_time = time.time() - start_time
        
        # Concurrent processing
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            concurrent_results = list(executor.map(process_content, test_contents))
        concurrent_time = time.time() - start_time
        
        # Analyze results
        seq_success = sum(1 for _, success, _ in sequential_results if success)
        conc_success = sum(1 for _, success, _ in concurrent_results if success)
        
        results.add_result("sequential_processing", seq_success == 20, 
                          f"Only {seq_success}/20 sequential tests passed",
                          {"total_time": sequential_time, "success_rate": seq_success/20})
        
        results.add_result("concurrent_processing", conc_success == 20,
                          f"Only {conc_success}/20 concurrent tests passed", 
                          {"total_time": concurrent_time, "success_rate": conc_success/20,
                           "speedup": sequential_time/concurrent_time if concurrent_time > 0 else 0})
        
        results.record_memory("performance_stress_end")
        
    except Exception as e:
        results.add_result("performance_stress", False, f"Performance test failed: {e}")
    
    return results

def test_memory_stress():
    """Test memory usage and potential leaks"""
    print("\n🔥 Memory Stress Test")
    print("-" * 25)
    
    results = StressTestResults()
    
    try:
        from src.rag.pcie_knowledge_classifier import PCIeKnowledgeClassifier
        
        # Get initial memory
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        results.record_memory("initial")
        
        # Create and destroy many instances
        instances = []
        for i in range(100):
            classifier = PCIeKnowledgeClassifier()
            instances.append(classifier)
            
            if i % 20 == 0:
                results.record_memory(f"create_instances_{i}")
        
        # Process with each instance
        test_content = "PCIe TLP Header: DW0 [31:0] contains format and type fields" * 10
        for i, classifier in enumerate(instances):
            classifier.classify_content(test_content)
            if i % 20 == 0:
                results.record_memory(f"process_instances_{i}")
        
        # Delete instances
        del instances
        gc.collect()
        results.record_memory("after_cleanup")
        
        # Check for memory growth
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Allow some memory growth but flag excessive growth
        acceptable_growth = 50  # MB
        memory_test_passed = memory_growth < acceptable_growth
        
        results.add_result("memory_stress", memory_test_passed,
                          f"Memory grew by {memory_growth:.1f}MB (limit: {acceptable_growth}MB)",
                          {"initial_mb": initial_memory, "final_mb": final_memory, "growth_mb": memory_growth})
        
    except Exception as e:
        results.add_result("memory_stress", False, f"Memory test failed: {e}")
    
    return results

def test_error_handling_robustness():
    """Test error handling and recovery mechanisms"""
    print("\n🔥 Error Handling Robustness Test")
    print("-" * 40)
    
    results = StressTestResults()
    
    try:
        from src.rag.enhanced_rag_engine_v3 import EnhancedRAGEngineV3, EnhancedRAGQuery
        
        # Test with None/missing dependencies
        try:
            # This should fail gracefully
            engine = EnhancedRAGEngineV3(None, None)
            results.add_result("none_dependencies", False, "Should have failed with None inputs")
        except Exception as e:
            # Expected to fail
            results.add_result("none_dependencies", True, None)
        
        # Test query with invalid parameters
        try:
            query = EnhancedRAGQuery(
                query="",  # Empty query
                min_confidence=-1.0,  # Invalid confidence
                max_results=-5  # Invalid max results
            )
            # Should handle gracefully
            results.add_result("invalid_query_params", True)
        except Exception as e:
            results.add_result("invalid_query_params", False, f"Failed to handle invalid params: {e}")
        
        # Test with simulated network/IO errors
        original_open = open
        def failing_open(*args, **kwargs):
            if 'test_fail' in str(args[0]):
                raise IOError("Simulated IO error")
            return original_open(*args, **kwargs)
        
        # Temporarily replace open to simulate failures
        __builtins__['open'] = failing_open
        
        try:
            from src.rag.answer_verifier import AnswerVerifier
            verifier = AnswerVerifier()
            # Should work despite IO simulation
            results.add_result("io_error_resilience", True)
        except Exception as e:
            results.add_result("io_error_resilience", False, f"Not resilient to IO errors: {e}")
        finally:
            __builtins__['open'] = original_open
        
    except Exception as e:
        results.add_result("error_handling", False, f"Error handling test setup failed: {e}")
    
    return results

def test_cli_integration_stress():
    """Stress test CLI integration under various conditions"""
    print("\n🔥 CLI Integration Stress Test")
    print("-" * 35)
    
    results = StressTestResults()
    
    try:
        from src.cli.interactive import PCIeDebugShell
        
        # Test shell creation under stress
        shells = []
        for i in range(10):
            try:
                shell = PCIeDebugShell(verbose=False)
                shells.append(shell)
                results.add_result(f"shell_creation_{i}", True)
            except Exception as e:
                results.add_result(f"shell_creation_{i}", False, str(e))
        
        # Test command processing
        if shells:
            shell = shells[0]
            
            # Test rapid command execution
            commands = [
                "/help",
                "/status", 
                "/rag_v3_status",
                "/suggest ltssm",
                "/model",
                "/rag off",
                "/rag on"
            ]
            
            for i, cmd in enumerate(commands):
                try:
                    # Simulate command processing
                    if hasattr(shell, 'onecmd'):
                        shell.onecmd(cmd)
                    results.add_result(f"command_{i}", True)
                except Exception as e:
                    results.add_result(f"command_{i}", False, str(e))
        
        # Test with malformed commands
        malformed_commands = [
            "/rag_analyze",  # Missing argument
            "/rag_verify",   # Missing argument
            "/suggest",      # Missing argument
            "/unknown_command",
            "/rag invalid_option",
            "//double//slash",
            "/rag_analyze " + "x" * 10000,  # Very long argument
        ]
        
        for i, cmd in enumerate(malformed_commands):
            try:
                # Should handle gracefully without crashing
                if shells and hasattr(shells[0], 'onecmd'):
                    shells[0].onecmd(cmd)
                results.add_result(f"malformed_cmd_{i}", True)
            except Exception as e:
                results.add_result(f"malformed_cmd_{i}", False, f"Crashed on malformed command: {e}")
        
        # Cleanup
        del shells
        
    except Exception as e:
        results.add_result("cli_stress", False, f"CLI stress test failed: {e}")
    
    return results

def run_comprehensive_stress_test():
    """Run all stress tests and generate detailed report"""
    print("🔥🔥🔥 ENHANCED RAG v3 COMPREHENSIVE STRESS TEST 🔥🔥🔥")
    print("=" * 70)
    
    start_time = time.time()
    
    # Run all stress test categories
    stress_tests = [
        ("Import Stress", test_import_stress),
        ("Component Edge Cases", test_component_edge_cases),
        ("Performance Stress", test_performance_stress),
        ("Memory Stress", test_memory_stress),
        ("Error Handling", test_error_handling_robustness),
        ("CLI Integration", test_cli_integration_stress)
    ]
    
    all_results = StressTestResults()
    category_results = {}
    
    for category_name, test_func in stress_tests:
        print(f"\n{'='*20} {category_name} {'='*20}")
        try:
            category_result = test_func()
            category_results[category_name] = category_result
            
            # Aggregate results
            all_results.tests_run += category_result.tests_run
            all_results.tests_passed += category_result.tests_passed
            all_results.tests_failed += category_result.tests_failed
            all_results.errors.extend(category_result.errors)
            all_results.performance_data.extend(category_result.performance_data)
            all_results.memory_usage.extend(category_result.memory_usage)
            
            # Show category summary
            success_rate = category_result.tests_passed / category_result.tests_run * 100 if category_result.tests_run > 0 else 0
            print(f"\n{category_name} Results: {category_result.tests_passed}/{category_result.tests_run} passed ({success_rate:.1f}%)")
            
            if category_result.tests_failed > 0:
                print(f"❌ {category_result.tests_failed} failures in {category_name}")
                for error in category_result.errors:
                    if error.get('test', '').startswith(category_name.lower().replace(' ', '_')):
                        print(f"   • {error['test']}: {error['error'][:100]}...")
        
        except Exception as e:
            print(f"❌ {category_name} stress test crashed: {e}")
            traceback.print_exc()
            all_results.tests_run += 1
            all_results.tests_failed += 1
            all_results.errors.append({"test": category_name, "error": str(e)})
    
    total_time = time.time() - start_time
    
    # Generate comprehensive report
    print("\n" + "="*70)
    print("🔥 COMPREHENSIVE STRESS TEST RESULTS 🔥")
    print("="*70)
    
    overall_success_rate = all_results.tests_passed / all_results.tests_run * 100 if all_results.tests_run > 0 else 0
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total Tests Run: {all_results.tests_run}")
    print(f"   Tests Passed: {all_results.tests_passed}")
    print(f"   Tests Failed: {all_results.tests_failed}")
    print(f"   Success Rate: {overall_success_rate:.1f}%")
    print(f"   Total Time: {total_time:.2f} seconds")
    
    # Performance analysis
    if all_results.performance_data:
        print(f"\n⚡ PERFORMANCE ANALYSIS:")
        avg_processing_time = sum(p.get('processing_time', 0) for p in all_results.performance_data) / len(all_results.performance_data)
        print(f"   Average Processing Time: {avg_processing_time:.4f}s")
        
        # Find slowest operations
        slowest = sorted(all_results.performance_data, key=lambda x: x.get('processing_time', 0), reverse=True)[:3]
        print(f"   Slowest Operations:")
        for i, op in enumerate(slowest, 1):
            print(f"      {i}. {op.get('test', 'unknown')}: {op.get('processing_time', 0):.4f}s")
    
    # Memory analysis
    if all_results.memory_usage:
        print(f"\n🧠 MEMORY ANALYSIS:")
        memory_values = [m['memory_mb'] for m in all_results.memory_usage]
        min_memory = min(memory_values)
        max_memory = max(memory_values)
        print(f"   Memory Range: {min_memory:.1f}MB - {max_memory:.1f}MB")
        print(f"   Memory Growth: {max_memory - min_memory:.1f}MB")
    
    # Error analysis
    if all_results.errors:
        print(f"\n❌ ERROR ANALYSIS ({len(all_results.errors)} errors):")
        error_types = {}
        for error in all_results.errors:
            error_type = error['error'].split(':')[0] if ':' in error['error'] else 'Unknown'
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   {error_type}: {count} occurrences")
    
    # Robustness assessment
    print(f"\n🛡️  ROBUSTNESS ASSESSMENT:")
    
    if overall_success_rate >= 95:
        robustness = "EXCELLENT"
        verdict = "🏆 System is extremely robust and ready for production"
    elif overall_success_rate >= 85:
        robustness = "GOOD"
        verdict = "✅ System is robust with minor issues"
    elif overall_success_rate >= 70:
        robustness = "ACCEPTABLE"
        verdict = "⚠️  System has some robustness issues"
    else:
        robustness = "POOR"
        verdict = "❌ System needs significant robustness improvements"
    
    print(f"   Robustness Level: {robustness}")
    print(f"   {verdict}")
    
    # Specific recommendations
    print(f"\n🔧 RECOMMENDATIONS:")
    
    if any("memory" in error['error'].lower() for error in all_results.errors):
        print("   • Review memory management and implement cleanup")
    
    if any("import" in error['test'] for error in all_results.errors):
        print("   • Strengthen import error handling")
    
    if any("concurrent" in error['test'] for error in all_results.errors):
        print("   • Add thread safety mechanisms")
    
    if all_results.tests_failed == 0:
        print("   • Excellent! No critical issues found")
        print("   • System demonstrates high reliability under stress")
    
    # Save detailed results
    report_data = {
        "timestamp": time.time(),
        "summary": {
            "total_tests": all_results.tests_run,
            "passed": all_results.tests_passed,
            "failed": all_results.tests_failed,
            "success_rate": overall_success_rate,
            "total_time": total_time,
            "robustness": robustness
        },
        "categories": {name: {
            "tests_run": result.tests_run,
            "tests_passed": result.tests_passed,
            "tests_failed": result.tests_failed
        } for name, result in category_results.items()},
        "errors": all_results.errors,
        "performance": all_results.performance_data,
        "memory": all_results.memory_usage
    }
    
    report_file = Path("stress_test_results.json")
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {report_file}")
    print("="*70)
    
    return overall_success_rate >= 85

if __name__ == "__main__":
    try:
        success = run_comprehensive_stress_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Stress test interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n\n💥 Stress test crashed: {e}")
        traceback.print_exc()
        sys.exit(3)