# 🧪 Testing Guide for Enhanced RAG Implementation

This guide shows how to test the Phase 1-3 enhanced RAG implementation to verify all improvements are working correctly.

## 🚀 Quick Start Testing

### 1. Run the Validation Benchmark Suite

This is the most comprehensive test that validates all improvements:

```bash
# Run the complete validation benchmark
python3 validation_benchmark_system.py
```

**Expected Output:**
- Baseline performance scores
- Phase 1 improvements (+25% confidence)
- Phase 2 improvements (+63.6% confidence)
- Phase 3 improvements (+75.1% confidence)
- Detailed report saved to `rag_validation_report.md`

### 2. Run the Unified Integration Test

Test the complete integrated system:

```bash
# Test the unified RAG integration
python3 test_unified_rag_system.py
```

**Expected Output:**
- 8/8 tests passing (100% success rate)
- Average confidence: ~0.800
- All components integrated successfully
- Results saved to `unified_rag_test_results_*.json`

## 🔍 Component-Level Testing

### Test Phase 1 Improvements

#### 1. Enhanced PDF Parsing
```python
# Test PDF parsing improvements
from src.processors.document_chunker import DocumentChunker

chunker = DocumentChunker()
# Test with a PCIe PDF document
chunks = chunker.chunk_document("path/to/pcie_spec.pdf")

# Verify:
# - Chunk size is ~1000 tokens (not 500)
# - Text extraction quality improved
# - Fallback to PyPDF2 if PyMuPDF fails
```

#### 2. Phrase Matching Boost
```python
# Test phrase matching in hybrid search
from src.rag.hybrid_search import HybridSearch

# Query with PCIe technical terms
results = hybrid_search.search("PCIe FLR compliance requirements")

# Verify:
# - Technical terms get boosted scores
# - 280+ PCIe terms recognized
# - Better ranking for technical content
```

#### 3. Multi-layered Confidence Scoring
```python
# Test confidence calculation
from src.rag.enhanced_rag_engine import EnhancedRAGEngine

rag = EnhancedRAGEngine()
result = rag.query("What is PCIe completion timeout?")

# Check result['confidence'] includes:
# - Base similarity score
# - Technical alignment
# - Content quality
# - Domain expertise
# - Answer completeness
# - Source reliability
```

### Test Phase 2 Features

#### 1. Query Expansion
```python
# Test query expansion engine
from src.rag.query_expansion_engine import QueryExpansionEngine

expander = QueryExpansionEngine()
expanded = expander.expand_query("What is FLR?")

# Verify expansion includes:
# - "Function Level Reset"
# - Related PCIe terms
# - Context hints
```

#### 2. Compliance Intelligence
```python
# Test compliance detection
from src.rag.compliance_intelligence import ComplianceIntelligence

compliance = ComplianceIntelligence()
query = "device sends completion during FLR"
result = compliance.check_compliance(query)

# Verify detection of:
# - FLR violation
# - Severity level (HIGH)
# - Specification reference
```

#### 3. Model Ensemble
```python
# Test model ensemble system
from src.rag.model_ensemble import ModelEnsemble

ensemble = ModelEnsemble()
# Should combine multiple embedding models
embeddings = ensemble.get_ensemble_embeddings("test query")

# Verify:
# - Multiple models used
# - Weighted combinations
# - Better coverage than single model
```

### Test Phase 3 Intelligence

#### 1. Quality Monitoring
```python
# Test quality monitoring
from src.rag.quality_monitor import ContinuousQualityMonitor

monitor = ContinuousQualityMonitor()
monitor.start_monitoring()

# Execute some queries
# Check monitor.get_real_time_metrics() for:
# - Response times
# - Confidence scores
# - Success rates
# - Quality trends
```

#### 2. Meta-RAG Coordination
```python
# Test meta-RAG coordinator
from src.rag.meta_rag_coordinator import MetaRAGCoordinator

coordinator = MetaRAGCoordinator()
result = coordinator.execute_query("complex PCIe query")

# Verify:
# - Appropriate engine selection
# - Strategy application
# - Result fusion if multiple engines used
```

## 🎯 Integration Testing

### Full System Test Script

Create and run test script:

```bash
python3 test_full_system.py
```

## 🔬 Performance Testing

### Benchmark Response Times
```bash
python3 benchmark_performance.py
```

### Load Testing
```bash
python3 load_test.py
```

## 🎪 Interactive Testing

### Using the Enhanced CLI

```bash
# Start the interactive shell
python3 src/cli/main.py

# Test Phase 1 improvements
/rag "What is PCIe FLR?"
# Check: Confidence score, source citations

# Test Phase 2 improvements  
/rag "device sends completion during FLR"
# Check: Compliance violation detected

# Test Phase 3 improvements
/rag --status
# Check: System health, quality metrics

/rag --test
# Run comprehensive test suite
```

## 📊 Validation Checklist

### Phase 1 Validation ✓
- [ ] PDF parsing uses PyMuPDF (check logs)
- [ ] Chunks are ~1000 tokens (not 500)
- [ ] PCIe technical terms boost search results
- [ ] Confidence scores have 6 components
- [ ] Sources are automatically cited

### Phase 2 Validation ✓
- [ ] Queries get expanded (FLR → Function Level Reset)
- [ ] Compliance violations detected for known issues
- [ ] Multiple embedding models used (if configured)
- [ ] Complex queries routed to appropriate tier
- [ ] Categories and intents properly classified

### Phase 3 Validation ✓
- [ ] Quality metrics displayed in real-time
- [ ] Meta-coordination between engines works
- [ ] Performance analytics available
- [ ] Response optimization based on patterns
- [ ] Context maintained across session

## 🐛 Troubleshooting

### If Tests Fail

1. **Import Errors**
   ```bash
   # Ensure you're in the project root
   cd /path/to/pcie_debug_agent
   
   # Add src to Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install pymupdf  # For PDF parsing
   ```

3. **Database Issues**
   ```bash
   # Remove old databases
   rm *.db
   
   # Rebuild vector database
   pcie-debug vectordb build --force
   ```

4. **Check Logs**
   ```bash
   # Check for errors
   tail -f logs/pcie_debug.log
   ```

## 🎉 Expected Results

After running all tests, you should see:

1. **Validation Benchmark**: Shows progressive improvements across phases
2. **Integration Tests**: 100% pass rate
3. **Confidence Scores**: 0.678 → 0.888 → 0.950 progression
4. **Response Times**: Maintained under 1 second
5. **Component Status**: All 20+ components active

The enhanced RAG system is working correctly when all these tests pass\!
EOF < /dev/null