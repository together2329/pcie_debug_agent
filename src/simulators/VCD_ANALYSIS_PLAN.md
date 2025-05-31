# VCD Waveform Analysis Integration Plan

## Overview
Integrate VCD (Value Change Dump) waveform analysis into the PCIe Debug Agent's RAG/LLM system to provide intelligent, context-aware debugging based on signal timing and behavior.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  PCIe Debug Agent with VCD Analysis         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐     ┌─────────────────┐     ┌─────────┐ │
│  │   VCD File   │────▶│   VCD Parser    │────▶│  Signal │ │
│  │ (Waveform)   │     │   & Analyzer    │     │Database │ │
│  └──────────────┘     └─────────────────┘     └─────────┘ │
│                               │                      │      │
│                               ▼                      ▼      │
│                      ┌─────────────────┐    ┌─────────────┐│
│                      │ Pattern Detect  │    │   Vector    ││
│                      │   & Anomaly     │────▶│   Store     ││
│                      └─────────────────┘    └─────────────┘│
│                               │                      │      │
│                               ▼                      ▼      │
│                      ┌─────────────────┐    ┌─────────────┐│
│                      │  RAG Engine     │◀───│  LLM with   ││
│                      │  Integration    │    │VCD Context  ││
│                      └─────────────────┘    └─────────────┘│
│                               │                             │
│                               ▼                             │
│                      ┌─────────────────┐                   │
│                      │ Intelligent     │                   │
│                      │ Debug Reports   │                   │
│                      └─────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Enhanced VCD Parser
Build a comprehensive VCD parser that extracts meaningful information for AI analysis.

**Key Features:**
- Signal hierarchy extraction
- Time-based event detection
- State transition tracking
- Protocol-aware parsing (PCIe specific)

**Implementation:**
```python
class VCDAnalyzer:
    def __init__(self, vcd_file):
        self.signals = {}
        self.events = []
        self.patterns = []
        
    def extract_pcie_events(self):
        # Extract PCIe-specific events
        # - TLP transactions
        # - Error occurrences
        # - State transitions
        # - Recovery sequences
        
    def detect_anomalies(self):
        # Identify unusual patterns
        # - Timing violations
        # - Unexpected state transitions
        # - Protocol violations
```

### Phase 2: VCD-to-Text Conversion
Convert waveform data into structured text that can be embedded and searched by the RAG engine.

**Format Example:**
```
[Time: 1000ns] PCIe Transaction Event
  - Type: Memory Read (MRd)
  - Address: 0x1000
  - Tag: 1
  - LTSSM State: L0
  - Link Status: UP
  - Duration: 110ns
  
[Time: 1500ns] Error Event
  - Type: CRC Error
  - Signal: error_valid=1, error_type=1
  - Recovery: LTSSM transition L0→RECOVERY→L0
  - Impact: 500ns link downtime
```

### Phase 3: Pattern Recognition
Implement algorithms to detect common PCIe issues from waveform patterns.

**Patterns to Detect:**
1. **Link Training Issues**
   - Stuck in POLLING state
   - Repeated recovery cycles
   - Speed negotiation failures

2. **Transaction Errors**
   - Missing completions
   - Malformed TLP patterns
   - Credit flow problems

3. **Timing Violations**
   - Setup/hold violations
   - Timeout patterns
   - Clock domain issues

### Phase 4: RAG Integration

#### 4.1 VCD Data Ingestion
```python
class VCDRAGIntegration:
    def __init__(self, rag_engine, vcd_analyzer):
        self.rag_engine = rag_engine
        self.vcd_analyzer = vcd_analyzer
        
    def ingest_waveform(self, vcd_file):
        # Parse VCD file
        events = self.vcd_analyzer.extract_events(vcd_file)
        
        # Convert to text chunks
        text_chunks = []
        for event in events:
            chunk = self.format_event_for_rag(event)
            text_chunks.append(chunk)
            
        # Generate embeddings
        embeddings = self.rag_engine.generate_embeddings(text_chunks)
        
        # Store in vector database
        self.rag_engine.store_with_metadata(
            chunks=text_chunks,
            embeddings=embeddings,
            metadata={'source': 'vcd', 'file': vcd_file}
        )
```

#### 4.2 Query Enhancement
```python
def enhance_query_with_waveform_context(query, vcd_data):
    """Add waveform context to user queries"""
    
    context = f"""
    Waveform Analysis Context:
    - Simulation Duration: {vcd_data['duration']}
    - Total Errors: {vcd_data['error_count']}
    - Error Types: {vcd_data['error_types']}
    - State Transitions: {vcd_data['state_changes']}
    - Transaction Summary: {vcd_data['transaction_summary']}
    
    User Query: {query}
    """
    return context
```

### Phase 5: LLM Prompts for Waveform Analysis

#### 5.1 Specialized Prompts
```python
WAVEFORM_ANALYSIS_PROMPTS = {
    "timing_analysis": """
    Analyze the following PCIe waveform timing data:
    {waveform_data}
    
    Focus on:
    1. Setup and hold time violations
    2. Clock domain crossing issues
    3. Recovery time requirements
    4. Transaction latencies
    
    Provide specific timing fixes.
    """,
    
    "error_correlation": """
    Given the waveform events:
    {waveform_events}
    
    Correlate the errors with:
    1. Preceding transactions
    2. State machine transitions
    3. Signal conditions
    
    Identify root cause and prevention.
    """,
    
    "protocol_compliance": """
    Check PCIe protocol compliance in waveform:
    {protocol_events}
    
    Verify:
    1. TLP format compliance
    2. Flow control rules
    3. Ordering requirements
    4. Power state transitions
    """
}
```

### Phase 6: Automated Anomaly Detection

```python
class WaveformAnomalyDetector:
    def __init__(self):
        self.rules = {
            'excessive_recovery': {
                'condition': lambda events: count_recovery_cycles(events) > 5,
                'severity': 'high',
                'recommendation': 'Check signal integrity'
            },
            'completion_timeout': {
                'condition': lambda events: detect_missing_completions(events),
                'severity': 'critical',
                'recommendation': 'Verify device response'
            },
            'link_flapping': {
                'condition': lambda events: detect_link_instability(events),
                'severity': 'high',
                'recommendation': 'Check power delivery'
            }
        }
        
    def analyze(self, vcd_file):
        anomalies = []
        events = parse_vcd(vcd_file)
        
        for rule_name, rule in self.rules.items():
            if rule['condition'](events):
                anomalies.append({
                    'type': rule_name,
                    'severity': rule['severity'],
                    'recommendation': rule['recommendation'],
                    'evidence': extract_evidence(events, rule_name)
                })
                
        return anomalies
```

### Phase 7: Timing Analysis

```python
class PCIeTimingAnalyzer:
    def analyze_transaction_timing(self, vcd_data):
        """Analyze PCIe transaction timing from VCD"""
        
        metrics = {
            'avg_completion_time': [],
            'max_completion_time': 0,
            'timeout_events': [],
            'recovery_duration': []
        }
        
        # Extract timing measurements
        for transaction in vcd_data.transactions:
            if transaction.type == 'read':
                completion_time = transaction.completion_time - transaction.start_time
                metrics['avg_completion_time'].append(completion_time)
                
        return metrics
```

### Phase 8: Report Generation with Waveforms

```python
def generate_waveform_report(analysis_results, vcd_file):
    """Generate comprehensive report with waveform insights"""
    
    report = f"""
    PCIe Debug Report with Waveform Analysis
    ========================================
    
    Source: {vcd_file}
    Generated: {datetime.now()}
    
    ## Executive Summary
    - Total Simulation Time: {analysis_results['duration']}ns
    - Errors Detected: {analysis_results['error_count']}
    - Critical Issues: {analysis_results['critical_issues']}
    
    ## Waveform Analysis
    
    ### 1. Error Correlation
    {format_error_timeline(analysis_results['errors'])}
    
    ### 2. Timing Analysis
    - Average Transaction Latency: {analysis_results['avg_latency']}ns
    - Maximum Latency: {analysis_results['max_latency']}ns
    - Timing Violations: {analysis_results['timing_violations']}
    
    ### 3. State Machine Analysis
    {format_state_transitions(analysis_results['states'])}
    
    ### 4. AI Recommendations
    {analysis_results['ai_recommendations']}
    
    ### 5. Waveform Excerpts
    {generate_ascii_waveform(analysis_results['key_events'])}
    """
    
    return report
```

## Integration Example

```python
# Complete workflow
def analyze_pcie_simulation(vcd_file, user_query):
    # 1. Parse VCD
    vcd_analyzer = VCDAnalyzer(vcd_file)
    waveform_data = vcd_analyzer.parse()
    
    # 2. Detect anomalies
    anomaly_detector = WaveformAnomalyDetector()
    anomalies = anomaly_detector.analyze(waveform_data)
    
    # 3. Extract timing
    timing_analyzer = PCIeTimingAnalyzer()
    timing_metrics = timing_analyzer.analyze_transaction_timing(waveform_data)
    
    # 4. Integrate with RAG
    rag_integration = VCDRAGIntegration(rag_engine, vcd_analyzer)
    rag_integration.ingest_waveform(vcd_file)
    
    # 5. Query with context
    enhanced_query = enhance_query_with_waveform_context(
        user_query, 
        {
            'anomalies': anomalies,
            'timing': timing_metrics,
            'waveform_summary': waveform_data.summary()
        }
    )
    
    # 6. Get AI analysis
    ai_response = rag_engine.query(enhanced_query)
    
    # 7. Generate report
    report = generate_waveform_report({
        'duration': waveform_data.duration,
        'error_count': len(anomalies),
        'critical_issues': [a for a in anomalies if a['severity'] == 'critical'],
        'errors': waveform_data.errors,
        'avg_latency': np.mean(timing_metrics['avg_completion_time']),
        'max_latency': timing_metrics['max_completion_time'],
        'timing_violations': timing_metrics['violations'],
        'states': waveform_data.state_transitions,
        'ai_recommendations': ai_response.answer,
        'key_events': waveform_data.get_key_events()
    }, vcd_file)
    
    return report
```

## Benefits

1. **Precise Debugging**: Correlate errors with exact timing and signal conditions
2. **Pattern Recognition**: Automatically identify common PCIe issues
3. **Intelligent Analysis**: LLM provides context-aware recommendations
4. **Comprehensive Reports**: Combine waveform data with AI insights
5. **Proactive Detection**: Find issues before they cause failures

## Next Steps

1. Implement enhanced VCD parser with PCIe awareness
2. Create pattern library for common PCIe issues
3. Build real-time waveform streaming for live analysis
4. Add support for multiple waveform formats (FST, VPD)
5. Integrate with existing debugging workflows

This plan enables the PCIe Debug Agent to provide unprecedented insight by combining traditional waveform analysis with modern AI capabilities.