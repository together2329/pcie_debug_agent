#!/usr/bin/env python3
"""
Functional Test Suite for VCD Analysis System
Tests end-to-end functionality and robustness
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import os

# Import our modules
from vcd_error_analyzer import PCIeVCDErrorAnalyzer
from vcd_rag_analyzer import PCIeVCDAnalyzer
from vcd_analysis_demo import MockRAGEngine


class VCDAnalysisFunctionalTest:
    """Comprehensive functional testing suite"""
    
    def __init__(self):
        self.test_results = {}
        self.failed_tests = []
        
    def test_vcd_generation(self) -> bool:
        """Test VCD file generation"""
        print("\n1. Testing VCD Generation...")
        print("-" * 40)
        
        try:
            # Remove existing VCD if present
            if Path("pcie_waveform.vcd").exists():
                os.remove("pcie_waveform.vcd")
                
            # Generate new VCD
            result = subprocess.run(
                ["./generate_vcd.sh"], 
                capture_output=True, 
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and Path("pcie_waveform.vcd").exists():
                vcd_size = Path("pcie_waveform.vcd").stat().st_size
                print(f"✓ VCD generated successfully ({vcd_size} bytes)")
                return True
            else:
                print(f"✗ VCD generation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"✗ VCD generation error: {e}")
            return False
            
    def test_vcd_parsing(self) -> bool:
        """Test VCD file parsing"""
        print("\n2. Testing VCD Parsing...")
        print("-" * 40)
        
        try:
            analyzer = PCIeVCDAnalyzer("pcie_waveform.vcd")
            vcd_data = analyzer.parse_vcd()
            
            # Verify parsed data
            events = len(vcd_data['events'])
            transactions = len(vcd_data['transactions'])
            errors = len(vcd_data['errors'])
            state_changes = len(vcd_data['state_changes'])
            
            print(f"✓ VCD parsed successfully:")
            print(f"  - Events: {events}")
            print(f"  - Transactions: {transactions}")
            print(f"  - Errors: {errors}")
            print(f"  - State changes: {state_changes}")
            
            # Basic validation
            if events > 0 and transactions > 0:
                return True
            else:
                print("✗ Insufficient data extracted")
                return False
                
        except Exception as e:
            print(f"✗ VCD parsing failed: {e}")
            return False
            
    def test_error_analysis(self) -> bool:
        """Test error analysis functionality"""
        print("\n3. Testing Error Analysis...")
        print("-" * 40)
        
        try:
            analyzer = PCIeVCDErrorAnalyzer("pcie_waveform.vcd")
            results = analyzer.analyze_errors()
            
            error_contexts = len(results['error_contexts'])
            root_causes = len(results['root_causes'])
            recommendations = len(results['recommendations'])
            
            print(f"✓ Error analysis completed:")
            print(f"  - Error contexts: {error_contexts}")
            print(f"  - Root causes: {root_causes}")
            print(f"  - Recommendations: {recommendations}")
            
            # Verify timing metrics
            if results['timing_metrics']:
                print(f"  - Timing analysis: Available")
            else:
                print(f"  - Timing analysis: Not available")
                
            return True
            
        except Exception as e:
            print(f"✗ Error analysis failed: {e}")
            return False
            
    def test_ai_integration(self) -> bool:
        """Test AI integration functionality"""
        print("\n4. Testing AI Integration...")
        print("-" * 40)
        
        try:
            # Create AI engine
            ai_engine = MockRAGEngine()
            
            # Test knowledge addition
            test_chunks = [
                "Test error: CRC error at 1000ns",
                "Test pattern: Recovery cycle detected",
                "Test recommendation: Check signal integrity"
            ]
            ai_engine.add_knowledge(test_chunks)
            
            # Test queries
            test_queries = [
                "What errors occurred?",
                "What is the root cause?",
                "What are the recommendations?"
            ]
            
            query_responses = 0
            for query in test_queries:
                response = ai_engine.query(query)
                if response and len(response) > 10:
                    query_responses += 1
                    
            print(f"✓ AI integration working:")
            print(f"  - Knowledge chunks: {len(test_chunks)}")
            print(f"  - Successful queries: {query_responses}/{len(test_queries)}")
            
            return query_responses >= len(test_queries) // 2
            
        except Exception as e:
            print(f"✗ AI integration failed: {e}")
            return False
            
    def test_report_generation(self) -> bool:
        """Test report generation"""
        print("\n5. Testing Report Generation...")
        print("-" * 40)
        
        try:
            analyzer = PCIeVCDErrorAnalyzer("pcie_waveform.vcd")
            results = analyzer.analyze_errors()
            
            # Generate report
            report = analyzer.generate_rag_enhanced_report()
            
            # Verify report content
            required_sections = [
                "Executive Summary",
                "Error Timeline Analysis", 
                "Timing Analysis",
                "Root Cause Analysis",
                "Recommendations"
            ]
            
            sections_found = 0
            for section in required_sections:
                if section in report:
                    sections_found += 1
                    
            print(f"✓ Report generated:")
            print(f"  - Length: {len(report)} characters")
            print(f"  - Sections: {sections_found}/{len(required_sections)}")
            
            # Save test report
            test_report_file = "functional_test_report.txt"
            with open(test_report_file, 'w') as f:
                f.write(report)
            print(f"  - Saved to: {test_report_file}")
            
            return sections_found >= len(required_sections) // 2
            
        except Exception as e:
            print(f"✗ Report generation failed: {e}")
            return False
            
    def test_performance(self) -> bool:
        """Test performance characteristics"""
        print("\n6. Testing Performance...")
        print("-" * 40)
        
        try:
            # Test analysis speed
            start_time = time.time()
            
            analyzer = PCIeVCDErrorAnalyzer("pcie_waveform.vcd")
            results = analyzer.analyze_errors()
            
            analysis_time = time.time() - start_time
            
            # Performance thresholds
            max_analysis_time = 5.0  # seconds
            
            print(f"✓ Performance test:")
            print(f"  - Analysis time: {analysis_time:.2f}s")
            print(f"  - Threshold: {max_analysis_time}s")
            
            if analysis_time <= max_analysis_time:
                print(f"  - Status: PASS")
                return True
            else:
                print(f"  - Status: FAIL (too slow)")
                return False
                
        except Exception as e:
            print(f"✗ Performance test failed: {e}")
            return False
            
    def test_edge_cases(self) -> bool:
        """Test edge cases and error handling"""
        print("\n7. Testing Edge Cases...")
        print("-" * 40)
        
        tests_passed = 0
        total_tests = 0
        
        # Test 1: Non-existent VCD file
        total_tests += 1
        try:
            analyzer = PCIeVCDErrorAnalyzer("nonexistent.vcd")
            results = analyzer.analyze_errors()
            print("✗ Should have failed with non-existent file")
        except Exception:
            print("✓ Correctly handled non-existent file")
            tests_passed += 1
            
        # Test 2: Empty query to AI
        total_tests += 1
        try:
            ai_engine = MockRAGEngine()
            response = ai_engine.query("")
            if response:
                print("✓ Handled empty query gracefully")
                tests_passed += 1
            else:
                print("✗ Empty query handling failed")
        except Exception:
            print("✗ Empty query caused exception")
            
        # Test 3: Large query
        total_tests += 1
        try:
            ai_engine = MockRAGEngine()
            large_query = "What " * 1000 + "happened?"
            response = ai_engine.query(large_query)
            if response:
                print("✓ Handled large query")
                tests_passed += 1
            else:
                print("✗ Large query failed")
        except Exception:
            print("✗ Large query caused exception")
            
        print(f"✓ Edge case tests: {tests_passed}/{total_tests} passed")
        return tests_passed >= total_tests // 2
        
    def test_integration_workflow(self) -> bool:
        """Test complete end-to-end workflow"""
        print("\n8. Testing Integration Workflow...")
        print("-" * 40)
        
        try:
            # Step 1: Parse VCD
            vcd_analyzer = PCIeVCDAnalyzer("pcie_waveform.vcd")
            vcd_data = vcd_analyzer.parse_vcd()
            
            # Step 2: Analyze errors
            error_analyzer = PCIeVCDErrorAnalyzer("pcie_waveform.vcd")
            error_results = error_analyzer.analyze_errors()
            
            # Step 3: Create AI knowledge
            ai_engine = MockRAGEngine()
            
            # Generate knowledge from analysis
            knowledge = []
            knowledge.append(f"VCD Analysis: {len(vcd_data['errors'])} errors found")
            
            for context in error_results['error_contexts']:
                knowledge.append(f"Error: {context.error_type} at {context.error_time}ns")
                
            ai_engine.add_knowledge(knowledge)
            
            # Step 4: Query and analyze
            workflow_queries = [
                "What errors were found in the analysis?",
                "What is the primary issue?",
                "What should be done to fix this?"
            ]
            
            successful_queries = 0
            for query in workflow_queries:
                response = ai_engine.query(query)
                if response and "error" in response.lower():
                    successful_queries += 1
                    
            # Step 5: Generate final report
            report = error_analyzer.generate_rag_enhanced_report()
            
            print(f"✓ Integration workflow completed:")
            print(f"  - VCD events: {len(vcd_data['events'])}")
            print(f"  - Error contexts: {len(error_results['error_contexts'])}")
            print(f"  - Knowledge chunks: {len(knowledge)}")
            print(f"  - Successful queries: {successful_queries}/{len(workflow_queries)}")
            print(f"  - Report generated: {len(report)} chars")
            
            return successful_queries >= len(workflow_queries) // 2
            
        except Exception as e:
            print(f"✗ Integration workflow failed: {e}")
            return False
            
    def run_all_tests(self):
        """Run complete test suite"""
        print("="*60)
        print("VCD Analysis System - Functional Test Suite")
        print("="*60)
        
        tests = [
            ("VCD Generation", self.test_vcd_generation),
            ("VCD Parsing", self.test_vcd_parsing),
            ("Error Analysis", self.test_error_analysis),
            ("AI Integration", self.test_ai_integration),
            ("Report Generation", self.test_report_generation),
            ("Performance", self.test_performance),
            ("Edge Cases", self.test_edge_cases),
            ("Integration Workflow", self.test_integration_workflow)
        ]
        
        passed_tests = 0
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                if result:
                    passed_tests += 1
                    self.test_results[test_name] = "PASS"
                else:
                    self.failed_tests.append(test_name)
                    self.test_results[test_name] = "FAIL"
            except Exception as e:
                print(f"✗ {test_name} crashed: {e}")
                self.failed_tests.append(test_name)
                self.test_results[test_name] = f"CRASH: {e}"
                
        # Print summary
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        
        print(f"\n📊 Overall Results:")
        print(f"   Tests Passed: {passed_tests}/{len(tests)}")
        print(f"   Success Rate: {passed_tests/len(tests)*100:.1f}%")
        
        print(f"\n✅ Passed Tests:")
        for test_name, result in self.test_results.items():
            if result == "PASS":
                print(f"   ✓ {test_name}")
                
        if self.failed_tests:
            print(f"\n❌ Failed Tests:")
            for test_name in self.failed_tests:
                result = self.test_results[test_name]
                print(f"   ✗ {test_name}: {result}")
                
        print(f"\n🎯 System Status:")
        if passed_tests == len(tests):
            print("   🟢 ALL TESTS PASSED - System fully functional")
        elif passed_tests >= len(tests) * 0.8:
            print("   🟡 MOSTLY FUNCTIONAL - Minor issues detected")
        elif passed_tests >= len(tests) * 0.5:
            print("   🟠 PARTIALLY FUNCTIONAL - Several issues need fixing")
        else:
            print("   🔴 SYSTEM NEEDS ATTENTION - Major issues detected")
            
        return passed_tests >= len(tests) * 0.8


def main():
    """Run the functional test suite"""
    tester = VCDAnalysisFunctionalTest()
    success = tester.run_all_tests()
    
    if success:
        print(f"\n🎉 VCD Analysis System is working well!")
    else:
        print(f"\n⚠️  VCD Analysis System needs attention.")
        
    return success


if __name__ == "__main__":
    main()