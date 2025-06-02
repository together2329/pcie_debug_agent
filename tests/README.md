# PCIe Debug Agent - Quality Test Suite

## 🔬 Comprehensive Testing Framework

This directory contains a comprehensive quality testing framework for the PCIe Debug Agent's hybrid LLM system.

## 📋 Test Components

### 1. **PCIe Error Test Scenarios** (`test_pcie_error_scenarios.py`)
- **15 comprehensive test cases** covering real-world PCIe errors
- Categories: Link Training, TLP Errors, Power Management, AER, Thermal, DPC, etc.
- Each test case includes:
  - Realistic error logs
  - Expected keywords
  - Expected root causes
  - Expected solutions
  - Complexity ratings

### 2. **Hybrid Quality Testing** (`test_hybrid_quality.py`)
- Tests response quality across all error scenarios
- Measures:
  - Keyword coverage
  - Root cause identification accuracy
  - Solution validity
  - Response time
  - Model performance comparison

### 3. **Edge Case Testing** (`test_edge_cases.py`)
- Tests system robustness with:
  - Empty/minimal inputs
  - Oversized inputs (10KB+ logs)
  - Malformed inputs (binary, unicode, control chars)
  - Concurrent requests
  - Timeout handling
  - Rapid model switching

### 4. **Accuracy Validation** (`test_accuracy_validation.py`)
- Validates response correctness:
  - Technical accuracy (no misinformation)
  - Diagnostic accuracy (correct problem identification)
  - Solution validity (safe and effective)
  - Factual correctness (no hallucinations)
  - Completeness and relevance

## 🚀 Quick Start

### Run Quick Tests
```bash
# Quick smoke test (2-3 minutes)
python run_quality_tests.py --quick

# Specific test suite
python tests/test_hybrid_quality.py --test quick
python tests/test_edge_cases.py
python tests/test_accuracy_validation.py --cases 3
```

### Run Full Test Suite
```bash
# Complete test suite (10-15 minutes)
python run_quality_tests.py --full

# Category-based testing (5-7 minutes)
python run_quality_tests.py
```

### Custom Testing
```bash
# Skip certain tests
python run_quality_tests.py --skip-edge
python run_quality_tests.py --skip-accuracy

# Test specific analysis type
python tests/test_hybrid_quality.py --test full --analysis detailed
```

## 📊 Test Results

### Quality Metrics
- **Overall Quality Score**: 0-1.0 scale
- **Response Time**: Average time per query
- **Keyword Coverage**: % of expected PCIe terms found
- **Root Cause Accuracy**: % of causes correctly identified
- **Solution Accuracy**: % of valid solutions provided

### Grading Scale
- **A+ (100%)**: Production ready, excellent quality
- **A (90%+)**: Very good, minor issues only
- **B (80%+)**: Good, some improvements needed
- **C (70%+)**: Fair, significant improvements needed
- **F (<70%)**: Poor, major issues to address

## 📋 Test Case Examples

### Link Training Failure
```python
error_log = """
[10:15:30] PCIe: 0000:01:00.0 - Link training failed
[10:15:31] PCIe: LTSSM stuck in Recovery.RcvrLock state
[10:15:32] PCIe: Signal quality degraded on lanes 0-3
"""
query = "Why is link training failing and how to fix it?"
```

### Malformed TLP
```python
error_log = """
[12:34:56] PCIe: Malformed TLP detected
[12:34:57] PCIe: TLP Type: 0x7F (Invalid)
[12:34:58] PCIe: ECRC check failed
"""
query = "What caused the malformed TLP and is data corrupted?"
```

## 🎯 Expected Results

### Good Response Characteristics
- ✅ Identifies specific error type
- ✅ Provides accurate root cause analysis
- ✅ Suggests practical solutions
- ✅ Uses correct PCIe terminology
- ✅ No hallucinations or false information

### Poor Response Characteristics
- ❌ Generic or vague answers
- ❌ Incorrect technical information
- ❌ Missing root cause analysis
- ❌ Dangerous or invalid solutions
- ❌ Off-topic content

## 📈 Performance Benchmarks

### Target Metrics
- **Quick Analysis**: <15s response time
- **Detailed Analysis**: <60s response time
- **Accuracy Score**: >80%
- **Success Rate**: >95%
- **Concurrent Handling**: 5+ simultaneous requests

## 🔧 Extending Tests

### Adding New Test Cases
```python
PCIeErrorTestCase(
    id="NEW-001",
    category="New Category",
    severity="high",
    error_log="Your error log here",
    expected_keywords=["keyword1", "keyword2"],
    expected_root_causes=["cause1", "cause2"],
    expected_solutions=["solution1", "solution2"],
    query="Your test query?",
    complexity="moderate"
)
```

### Custom Validation
```python
def validate_custom_aspect(response: str, test_case) -> float:
    # Your validation logic
    score = 1.0
    # Adjust score based on criteria
    return score
```

## 📝 Test Reports

Reports are saved as JSON files with timestamps:
- `comprehensive_test_report_YYYYMMDD_HHMMSS.json`
- `hybrid_quality_test_results_YYYYMMDD_HHMMSS.json`
- `edge_case_test_results_YYYYMMDD_HHMMSS.json`

## 🚨 Common Issues

### No Models Available
```bash
# Check model status
./pcie-debug-hybrid model status

# Ensure models are downloaded
ollama pull deepseek-r1
```

### Test Timeouts
- Increase timeout values in test configurations
- Use `--quick` flag for faster testing
- Check system resources

### Low Accuracy Scores
- Review model prompts
- Check context window settings
- Verify model configurations

## 🎉 Success Criteria

The system is considered production-ready when:
1. All test suites pass (100% success rate)
2. Average accuracy score >85%
3. Response times meet targets
4. Edge cases handled gracefully
5. No critical failures or crashes

---

**Happy Testing!** 🚀