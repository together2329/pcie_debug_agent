#!/usr/bin/env python3
"""
Automated RAG Quality Test Suite
Tests RAG responses against known PCIe scenarios and benchmarks quality improvements
"""

import json
import time
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics
import re

logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    """Individual test case for RAG evaluation"""
    id: str
    category: str
    question: str
    expected_keywords: List[str]
    expected_spec_refs: List[str]
    expected_concepts: List[str]
    difficulty: str  # basic, intermediate, advanced, expert
    compliance_critical: bool = False
    expected_confidence_min: float = 0.6

@dataclass
class TestResult:
    """Result of a single test case"""
    test_id: str
    passed: bool
    confidence: float
    response_time: float
    keyword_coverage: float
    spec_coverage: float
    concept_coverage: float
    answer_length: int
    answer: str
    issues: List[str]
    score: float

@dataclass
class TestSuiteReport:
    """Complete test suite results"""
    timestamp: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_score: float
    avg_confidence: float
    avg_response_time: float
    critical_failures: int
    category_scores: Dict[str, float]
    results: List[TestResult]

class PCIeRAGTestSuite:
    """Automated test suite for PCIe RAG system quality"""
    
    def __init__(self):
        self.test_cases = self._load_test_cases()
        self.quality_metrics = {
            'keyword_weight': 0.25,
            'spec_weight': 0.25,
            'concept_weight': 0.25,
            'confidence_weight': 0.15,
            'speed_weight': 0.10
        }
    
    def _load_test_cases(self) -> List[TestCase]:
        """Load comprehensive PCIe test cases"""
        return [
            # FLR/CRS Compliance Tests
            TestCase(
                id="flr_crs_001",
                category="compliance",
                question="why dut send successful return of completion during flr ? I expect crs return",
                expected_keywords=["flr", "function level reset", "crs", "configuration request retry", "compliance", "specification"],
                expected_spec_refs=["6.6.1.2", "base specification"],
                expected_concepts=["reset behavior", "configuration space", "device readiness"],
                difficulty="expert",
                compliance_critical=True,
                expected_confidence_min=0.8
            ),
            TestCase(
                id="flr_crs_002", 
                category="compliance",
                question="What should device return during FLR to configuration reads?",
                expected_keywords=["crs", "configuration request retry", "flr", "not ready"],
                expected_spec_refs=["6.6.1.2"],
                expected_concepts=["reset sequence", "configuration access"],
                difficulty="intermediate",
                compliance_critical=True
            ),
            
            # Completion Timeout Tests
            TestCase(
                id="cto_001",
                category="error_handling", 
                question="PCIe completion timeout during memory read operations",
                expected_keywords=["completion timeout", "memory read", "non-posted", "target device"],
                expected_spec_refs=["2.2.9"],
                expected_concepts=["timeout handling", "credit management", "transaction flow"],
                difficulty="intermediate"
            ),
            TestCase(
                id="cto_002",
                category="debug",
                question="How to debug completion timeout errors in PCIe?",
                expected_keywords=["debug", "timeout", "completion", "device control", "traffic analysis"],
                expected_spec_refs=["device control 2"],
                expected_concepts=["debugging methodology", "register analysis", "traffic monitoring"],
                difficulty="advanced"
            ),
            
            # LTSSM Tests
            TestCase(
                id="ltssm_001",
                category="ltssm",
                question="PCIe link stuck in Polling.Compliance state",
                expected_keywords=["ltssm", "polling", "compliance", "link training", "electrical idle"],
                expected_spec_refs=["4.2"],
                expected_concepts=["state machine", "link negotiation", "speed negotiation"],
                difficulty="advanced"
            ),
            
            # TLP Tests  
            TestCase(
                id="tlp_001",
                category="tlp",
                question="What is the TLP header format for 3DW memory read?",
                expected_keywords=["tlp", "header", "3dw", "memory read", "format", "type"],
                expected_spec_refs=["2.2"],
                expected_concepts=["packet structure", "addressing", "transaction layer"],
                difficulty="basic"
            ),
            
            # Error Handling Tests
            TestCase(
                id="aer_001",
                category="error_handling",
                question="AER correctable error reporting configuration",
                expected_keywords=["aer", "correctable", "error reporting", "capability", "mask"],
                expected_spec_refs=["6.2"],
                expected_concepts=["error detection", "reporting mechanism", "error masking"],
                difficulty="intermediate"
            ),
            
            # Power Management Tests
            TestCase(
                id="pm_001",
                category="power_management", 
                question="PCIe ASPM L1 entry conditions and requirements",
                expected_keywords=["aspm", "l1", "power management", "entry", "exit", "latency"],
                expected_spec_refs=["5.4"],
                expected_concepts=["power states", "link power management", "latency requirements"],
                difficulty="advanced"
            ),
            
            # Configuration Tests
            TestCase(
                id="config_001",
                category="configuration",
                question="PCIe configuration space capability structure",
                expected_keywords=["configuration", "capability", "structure", "pointer", "next"],
                expected_spec_refs=["7.5"],
                expected_concepts=["capability list", "configuration registers", "device identification"],
                difficulty="basic"
            ),
            
            # Advanced Features
            TestCase(
                id="sriov_001",
                category="advanced_features",
                question="SR-IOV virtual function enumeration process",
                expected_keywords=["sr-iov", "virtual function", "enumeration", "vf", "pf"],
                expected_spec_refs=["sr-iov specification"],
                expected_concepts=["virtualization", "function discovery", "resource allocation"],
                difficulty="expert"
            )
        ]
    
    def run_full_suite(self, rag_engine) -> TestSuiteReport:
        """Run complete test suite and generate report"""
        print("🧪 Starting PCIe RAG Quality Test Suite")
        print(f"   Running {len(self.test_cases)} test cases...")
        print("=" * 60)
        
        start_time = datetime.now()
        results = []
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n[{i}/{len(self.test_cases)}] Testing: {test_case.id}")
            print(f"   Category: {test_case.category}")
            print(f"   Question: {test_case.question[:60]}...")
            
            result = self._run_single_test(test_case, rag_engine)
            results.append(result)
            
            # Show immediate result
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"   Result: {status} (Score: {result.score:.2f}, Time: {result.response_time:.2f}s)")
            
            if result.issues:
                print(f"   Issues: {'; '.join(result.issues[:2])}")
        
        # Generate comprehensive report
        report = self._generate_report(results, start_time)
        self._save_report(report)
        self._print_summary(report)
        
        return report
    
    def _run_single_test(self, test_case: TestCase, rag_engine) -> TestResult:
        """Run a single test case"""
        start_time = time.time()
        issues = []
        
        try:
            # Query the RAG engine
            if hasattr(rag_engine, 'query'):
                # Use different query interfaces based on engine type
                if hasattr(rag_engine, 'production_rag'):
                    from production_rag_fix import ProductionRAGQuery
                    query = ProductionRAGQuery(query=test_case.question)
                    response = rag_engine.production_rag.query(query)
                    answer = response.answer
                    confidence = response.confidence
                else:
                    # Standard RAG engine
                    from src.rag.enhanced_rag_engine import RAGQuery
                    query = RAGQuery(query=test_case.question)
                    response = rag_engine.query(query)
                    answer = response.answer
                    confidence = response.confidence
            else:
                # Fallback: simulate response
                answer = f"Test response for: {test_case.question}"
                confidence = 0.5
                issues.append("RAG engine not available - using fallback")
        
        except Exception as e:
            answer = f"Error: {str(e)}"
            confidence = 0.0
            issues.append(f"Query failed: {str(e)}")
        
        response_time = time.time() - start_time
        
        # Evaluate response quality
        keyword_coverage = self._calculate_keyword_coverage(answer, test_case.expected_keywords)
        spec_coverage = self._calculate_spec_coverage(answer, test_case.expected_spec_refs)
        concept_coverage = self._calculate_concept_coverage(answer, test_case.expected_concepts)
        
        # Calculate overall score
        score = self._calculate_test_score(
            keyword_coverage, spec_coverage, concept_coverage, 
            confidence, response_time, test_case
        )
        
        # Determine pass/fail
        passed = (
            score >= 0.6 and 
            confidence >= test_case.expected_confidence_min and
            keyword_coverage >= 0.4 and
            not any("error" in issue.lower() for issue in issues)
        )
        
        # Add quality issues
        if keyword_coverage < 0.4:
            issues.append(f"Low keyword coverage: {keyword_coverage:.1%}")
        if spec_coverage < 0.3:
            issues.append(f"Missing spec references: {spec_coverage:.1%}")
        if confidence < test_case.expected_confidence_min:
            issues.append(f"Low confidence: {confidence:.1%} < {test_case.expected_confidence_min:.1%}")
        if response_time > 5.0:
            issues.append(f"Slow response: {response_time:.1f}s")
        
        return TestResult(
            test_id=test_case.id,
            passed=passed,
            confidence=confidence,
            response_time=response_time,
            keyword_coverage=keyword_coverage,
            spec_coverage=spec_coverage,
            concept_coverage=concept_coverage,
            answer_length=len(answer),
            answer=answer,
            issues=issues,
            score=score
        )
    
    def _calculate_keyword_coverage(self, answer: str, expected_keywords: List[str]) -> float:
        """Calculate how many expected keywords are present"""
        if not expected_keywords:
            return 1.0
        
        answer_lower = answer.lower()
        found_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
        return found_keywords / len(expected_keywords)
    
    def _calculate_spec_coverage(self, answer: str, expected_specs: List[str]) -> float:
        """Calculate specification reference coverage"""
        if not expected_specs:
            return 1.0
        
        answer_lower = answer.lower()
        found_specs = sum(1 for spec in expected_specs if spec.lower() in answer_lower)
        return found_specs / len(expected_specs)
    
    def _calculate_concept_coverage(self, answer: str, expected_concepts: List[str]) -> float:
        """Calculate conceptual coverage"""
        if not expected_concepts:
            return 1.0
        
        answer_lower = answer.lower()
        found_concepts = 0
        
        for concept in expected_concepts:
            # Check for concept or related terms
            concept_words = concept.lower().split()
            if any(word in answer_lower for word in concept_words):
                found_concepts += 1
        
        return found_concepts / len(expected_concepts)
    
    def _calculate_test_score(self, keyword_cov: float, spec_cov: float, concept_cov: float, 
                             confidence: float, response_time: float, test_case: TestCase) -> float:
        """Calculate weighted test score"""
        # Base quality score
        quality_score = (
            keyword_cov * self.quality_metrics['keyword_weight'] +
            spec_cov * self.quality_metrics['spec_weight'] +
            concept_cov * self.quality_metrics['concept_weight'] +
            confidence * self.quality_metrics['confidence_weight']
        )
        
        # Speed bonus/penalty
        speed_factor = min(1.0, 3.0 / max(response_time, 0.1))  # Bonus for <3s response
        speed_score = speed_factor * self.quality_metrics['speed_weight']
        
        total_score = quality_score + speed_score
        
        # Critical compliance penalty
        if test_case.compliance_critical and (keyword_cov < 0.6 or confidence < 0.7):
            total_score *= 0.7  # 30% penalty for critical failures
        
        return min(total_score, 1.0)
    
    def _generate_report(self, results: List[TestResult], start_time: datetime) -> TestSuiteReport:
        """Generate comprehensive test report"""
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        
        overall_score = statistics.mean([r.score for r in results])
        avg_confidence = statistics.mean([r.confidence for r in results])
        avg_response_time = statistics.mean([r.response_time for r in results])
        
        critical_failures = sum(1 for r in results 
                               if not r.passed and any("critical" in issue.lower() for issue in r.issues))
        
        # Category scores
        category_scores = {}
        test_categories = {tc.category for tc in self.test_cases}
        for category in test_categories:
            category_results = [r for r, tc in zip(results, self.test_cases) if tc.category == category]
            if category_results:
                category_scores[category] = statistics.mean([r.score for r in category_results])
        
        return TestSuiteReport(
            timestamp=datetime.now().isoformat(),
            total_tests=len(results),
            passed_tests=passed,
            failed_tests=failed,
            overall_score=overall_score,
            avg_confidence=avg_confidence,
            avg_response_time=avg_response_time,
            critical_failures=critical_failures,
            category_scores=category_scores,
            results=results
        )
    
    def _save_report(self, report: TestSuiteReport):
        """Save detailed report to JSON"""
        filename = f"rag_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        print(f"\n📄 Detailed report saved: {filename}")
    
    def _print_summary(self, report: TestSuiteReport):
        """Print test summary"""
        print(f"\n{'='*60}")
        print("🧪 PCIe RAG Test Suite Results")
        print(f"{'='*60}")
        
        print(f"\n📊 Overall Results:")
        print(f"   Tests Run: {report.total_tests}")
        print(f"   Passed: {report.passed_tests} ({report.passed_tests/report.total_tests*100:.1f}%)")
        print(f"   Failed: {report.failed_tests} ({report.failed_tests/report.total_tests*100:.1f}%)")
        print(f"   Overall Score: {report.overall_score:.2f}/1.00")
        
        print(f"\n⚡ Performance:")
        print(f"   Average Confidence: {report.avg_confidence:.1%}")
        print(f"   Average Response Time: {report.avg_response_time:.2f}s")
        print(f"   Critical Failures: {report.critical_failures}")
        
        print(f"\n📈 Category Scores:")
        for category, score in sorted(report.category_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"   {category.replace('_', ' ').title()}: {score:.2f}")
        
        print(f"\n🎯 Recommendations:")
        if report.overall_score < 0.7:
            print("   • Overall quality needs improvement")
        if report.avg_confidence < 0.6:
            print("   • Improve answer confidence through better knowledge base")
        if report.avg_response_time > 3.0:
            print("   • Optimize response time performance")
        if report.critical_failures > 0:
            print("   • Address critical compliance test failures immediately")
        
        # Show worst performing categories
        worst_categories = sorted(report.category_scores.items(), key=lambda x: x[1])[:2]
        if worst_categories and worst_categories[0][1] < 0.6:
            print(f"   • Focus improvement on: {', '.join([c[0] for c in worst_categories])}")

def run_automated_tests():
    """Quick test runner"""
    print("🚀 Starting Automated PCIe RAG Tests...")
    
    test_suite = PCIeRAGTestSuite()
    
    # Mock RAG engine for demonstration
    class MockRAGEngine:
        def query(self, query):
            class MockResponse:
                def __init__(self):
                    self.answer = "This is a mock response for testing purposes."
                    self.confidence = 0.75
            return MockResponse()
    
    mock_engine = MockRAGEngine()
    report = test_suite.run_full_suite(mock_engine)
    
    return report

if __name__ == "__main__":
    run_automated_tests()