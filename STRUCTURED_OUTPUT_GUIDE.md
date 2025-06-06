# 📊 PCIe Structured Output Guide

Complete guide for using structured JSON output in PCIe Debug Agent

## 🚀 Quick Start

### CLI JSON Output
```bash
# Basic JSON query
./pcie-debug pcie query "PCIe LTSSM states" --json

# Pretty-formatted JSON
./pcie-debug pcie query "completion timeout" --json --pretty

# With filters and JSON output
./pcie-debug pcie query "link training" --technical-level 3 --layer physical --json --pretty
```

### Interactive Mode JSON
```bash
./pcie-debug
/rag_mode pcie
/json_query What causes PCIe link training failures?
```

## 📋 Structured Data Format

### Response Structure
```json
{
  "query": "Your PCIe question",
  "timestamp": "2025-06-06T00:00:03.820706",
  "response_time_ms": 729.59,
  "total_results": 5,
  "results": [
    {
      "content": "Detailed PCIe technical content...",
      "score": 0.564,
      "metadata": {
        "source_file": "ltssm_timeout_conditions.md",
        "chunk_id": "chunk_001",
        "technical_level": 2,
        "pcie_layer": "general",
        "semantic_type": "content",
        "pcie_concepts": [
          {
            "name": "LTSSM",
            "category": "LTSSM State",
            "confidence": 1.0
          }
        ],
        "section_title": "Link Training States",
        "page_number": 42
      },
      "highlighted_terms": ["LTSSM", "timeout"],
      "relevance_explanation": "Why this result is relevant"
    }
  ],
  "analysis": {
    "issue_type": "Link Training Issue",
    "severity": "warning",
    "root_cause": "Signal integrity problems",
    "affected_components": ["LTSSM", "Physical Layer"],
    "recommended_actions": [
      "Check PCIe lane connectivity",
      "Verify reference clock",
      "Review LTSSM state transitions"
    ],
    "related_specs": ["pcie_spec.pdf"],
    "confidence": 0.7
  },
  "model_used": "text-embedding-3-small",
  "search_mode": "pcie",
  "filters_applied": {
    "technical_level": 3,
    "pcie_layer": "physical"
  }
}
```

### Metadata Fields

#### Technical Levels
- `1` - Basic (overview, general concepts)
- `2` - Intermediate (implementation details)
- `3` - Advanced (deep technical, debugging)

#### PCIe Layers
- `physical` - Physical layer (LTSSM, electrical)
- `data_link` - Data link layer (DLLP, flow control)
- `transaction` - Transaction layer (TLP, completion)
- `software` - Software interface (drivers, OS)
- `power_management` - Power management (ASPM, L-states)
- `system_architecture` - System-level architecture
- `general` - Cross-layer or general concepts

#### Semantic Types
- `header_section` - Document headers/titles
- `procedure` - Step-by-step procedures
- `specification` - Technical specifications
- `example` - Code examples or samples
- `content` - General content

#### PCIe Concept Categories
- `LTSSM State` - Link training states (L0, L1, Recovery, etc.)
- `TLP Type` - Transaction layer packets
- `Error Code` - Error conditions and codes
- `Power State` - Power management states
- `General` - Other PCIe concepts

## 🎯 Usage Examples

### 1. Debugging Completion Timeouts
```bash
./pcie-debug pcie query "completion timeout causes" --json --pretty
```

This returns structured analysis with:
- Issue type: "Completion Timeout"
- Severity: "critical"
- Root cause analysis
- Recommended debugging actions

### 2. LTSSM State Analysis
```bash
./pcie-debug pcie query "LTSSM state transitions" --technical-level 3 --json
```

Returns advanced technical content about:
- State machine transitions
- Timeout conditions
- Debug procedures

### 3. Layer-Specific Queries
```bash
# Physical layer issues
./pcie-debug pcie query "signal integrity" --layer physical --json

# Power management
./pcie-debug pcie query "ASPM L1" --layer power_management --json

# Transaction layer
./pcie-debug pcie query "TLP format" --layer transaction --json
```

### 4. Interactive Structured Queries
```bash
./pcie-debug
/rag_mode pcie
/json_query How to debug PCIe hot-plug issues?
```

## 📊 Data Processing Examples

### Python Processing
```python
import json
import subprocess

# Run query and get JSON
result = subprocess.run([
    './pcie-debug', 'pcie', 'query', 
    'PCIe completion timeout', '--json'
], capture_output=True, text=True)

data = json.loads(result.stdout)

# Extract key information
print(f"Query: {data['query']}")
print(f"Results: {data['total_results']}")
print(f"Response time: {data['response_time_ms']:.1f}ms")

# Process results
for i, result in enumerate(data['results']):
    print(f"\nResult {i+1}:")
    print(f"  Score: {result['score']:.3f}")
    print(f"  Layer: {result['metadata']['pcie_layer']}")
    print(f"  Technical Level: {result['metadata']['technical_level']}")
    
    # Extract PCIe concepts
    concepts = [c['name'] for c in result['metadata']['pcie_concepts']]
    print(f"  Concepts: {concepts[:5]}")  # Show first 5

# Check for analysis
if data['analysis']:
    analysis = data['analysis']
    print(f"\nAnalysis:")
    print(f"  Issue: {analysis['issue_type']}")
    print(f"  Severity: {analysis['severity']}")
    print(f"  Confidence: {analysis['confidence']:.1%}")
    print(f"  Actions: {analysis['recommended_actions']}")
```

### Shell Processing with jq
```bash
# Extract just the issue analysis
./pcie-debug pcie query "link training failure" --json | \
  jq '.analysis'

# Get top 3 results with scores
./pcie-debug pcie query "LTSSM timeout" --json | \
  jq '.results[:3] | .[] | {score, source: .metadata.source_file}'

# Extract all PCIe concepts mentioned
./pcie-debug pcie query "power management" --json | \
  jq '.results[].metadata.pcie_concepts[].name' | sort -u

# Get response time and result count
./pcie-debug pcie query "completion timeout" --json | \
  jq '{query, response_time_ms, total_results}'
```

## 🔧 Advanced Features

### 1. Confidence Scoring
Each result includes confidence scores:
- `score` - Relevance score (0.0-1.0)
- `analysis.confidence` - Analysis confidence (0.0-1.0)
- `pcie_concepts[].confidence` - Concept extraction confidence

### 2. Automatic Issue Detection
The system automatically detects issue types:
- **Completion Timeout** - For timeout-related queries
- **Link Training Issue** - For LTSSM/link training queries
- **Power Management Issue** - For ASPM/power state queries

### 3. Cross-Reference Tracking
- `affected_components` - List of PCIe components involved
- `related_specs` - Relevant specification documents
- `recommended_actions` - Actionable debugging steps

### 4. Temporal Information
- `timestamp` - When query was processed
- `response_time_ms` - Processing time in milliseconds
- Performance tracking for optimization

## 🎯 Best Practices

### 1. Query Optimization
- Use specific PCIe terminology for better results
- Combine filters for targeted searches
- Use technical level filtering to match expertise

### 2. Result Processing
- Check `total_results` to understand scope
- Sort by `score` for relevance ranking
- Use `pcie_concepts` for automated categorization

### 3. Error Handling
```python
# Always check for results
if data['total_results'] == 0:
    print("No results found for query")
else:
    # Process results...
    pass

# Check analysis availability
if data['analysis']:
    # Use structured analysis
    pass
else:
    # Fallback to result content
    pass
```

### 4. Performance Monitoring
- Monitor `response_time_ms` for performance
- Use `model_used` for model tracking
- Track `filters_applied` for query optimization

## 📈 Integration Examples

### 1. Automated Bug Report Generation
```python
def generate_bug_report(issue_description):
    """Generate structured bug report from PCIe issue"""
    result = query_pcie_structured(issue_description)
    
    if result['analysis']:
        return {
            'title': f"PCIe {result['analysis']['issue_type']}",
            'severity': result['analysis']['severity'],
            'description': issue_description,
            'root_cause': result['analysis']['root_cause'],
            'debug_steps': result['analysis']['recommended_actions'],
            'related_docs': result['analysis']['related_specs'],
            'confidence': result['analysis']['confidence']
        }
    
    return None
```

### 2. Knowledge Base Search API
```python
class PCIeKnowledgeAPI:
    def search(self, query, filters=None):
        """Search PCIe knowledge base with structured output"""
        cmd = ['./pcie-debug', 'pcie', 'query', query, '--json']
        
        if filters:
            if filters.get('technical_level'):
                cmd.extend(['--technical-level', str(filters['technical_level'])])
            if filters.get('layer'):
                cmd.extend(['--layer', filters['layer']])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return json.loads(result.stdout)
    
    def get_concepts(self, query):
        """Extract PCIe concepts from query results"""
        data = self.search(query)
        concepts = set()
        
        for result in data['results']:
            for concept in result['metadata']['pcie_concepts']:
                concepts.add((concept['name'], concept['category']))
        
        return list(concepts)
```

## ✅ Summary

The structured output system provides:

1. **📊 Rich Metadata** - Technical levels, layers, concepts
2. **🔍 Automated Analysis** - Issue detection and recommendations  
3. **⚡ Performance Tracking** - Response times and confidence scores
4. **🎯 Precise Filtering** - Layer and expertise-based searches
5. **🔗 Cross-References** - Component and specification linking
6. **📈 Machine-Readable** - JSON format for automation

Perfect for:
- **Automated debugging workflows**
- **Knowledge base integration**
- **Bug report generation** 
- **Performance monitoring**
- **Educational tools**