"""
PCIe-specific prompt templates for local LLM
"""

from typing import Dict, List, Any
import json


class PCIePromptTemplates:
    """PCIe debugging prompt templates optimized for local LLM"""
    
    SYSTEM_PROMPT = """You are a PCIe debugging expert specialized in analyzing PCIe logs and hardware issues. 
You have deep knowledge of:
- PCIe protocol specifications (Gen 1-5)
- Link training and initialization
- Transaction Layer Packets (TLPs)
- Data Link Layer Packets (DLLPs)
- Physical layer signaling
- Common PCIe errors and their causes
- Hardware debugging techniques

Provide clear, technical, and actionable analysis based on the given context."""

    @staticmethod
    def create_analysis_prompt(query: str, context: str) -> str:
        """Create analysis prompt for PCIe debugging"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{PCIePromptTemplates.SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

Based on the PCIe log data below, please analyze the following query:

**Query**: {query}

**PCIe Log Context**:
{context}

Please provide:
1. **Analysis**: What the logs indicate about the PCIe issue
2. **Root Cause**: Most likely cause of the problem
3. **Impact**: How this affects system operation
4. **Recommendations**: Specific debugging steps or fixes

Focus on technical accuracy and cite specific log entries when relevant.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    @staticmethod
    def create_error_classification_prompt(log_entries: List[str]) -> str:
        """Create prompt for classifying PCIe errors"""
        log_text = "\\n".join(log_entries)
        
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{PCIePromptTemplates.SYSTEM_PROMPT}

Classify PCIe errors into categories: LINK_TRAINING, TIMEOUT, CORRECTABLE_ERROR, UNCORRECTABLE_ERROR, CONFIGURATION, POWER_MANAGEMENT, or OTHER.<|eot_id|><|start_header_id|>user<|end_header_id|>

Analyze these PCIe log entries and classify each error:

{log_text}

For each error, provide:
- **Category**: One of the standard PCIe error categories
- **Severity**: LOW, MEDIUM, HIGH, CRITICAL
- **Description**: Brief technical explanation
- **Device**: Which PCIe device is affected (if identifiable)

Format as JSON array.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    @staticmethod
    def create_link_training_analysis_prompt(context: str) -> str:
        """Create prompt for PCIe link training analysis"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{PCIePromptTemplates.SYSTEM_PROMPT}

Focus on PCIe link training sequence analysis.<|eot_id|><|start_header_id|>user<|end_header_id|>

Analyze the PCIe link training sequence from these logs:

{context}

Please examine:
1. **Link Training States**: Which LTSSM states were traversed
2. **Speed Negotiation**: What PCIe generations were attempted
3. **Width Negotiation**: Lane count negotiations
4. **Failure Points**: Where link training failed (if applicable)
5. **Timing Issues**: Any timing-related problems
6. **Recovery Attempts**: Link recovery mechanisms triggered

Provide specific technical details about the link training process.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    @staticmethod
    def create_performance_analysis_prompt(context: str) -> str:
        """Create prompt for PCIe performance analysis"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{PCIePromptTemplates.SYSTEM_PROMPT}

Focus on PCIe performance analysis and optimization.<|eot_id|><|start_header_id|>user<|end_header_id|>

Analyze PCIe performance from these logs:

{context}

Please evaluate:
1. **Throughput**: Actual vs theoretical bandwidth utilization
2. **Latency**: Transaction completion times
3. **Bottlenecks**: Performance limiting factors
4. **Error Impact**: How errors affect performance
5. **Configuration**: PCIe configuration affecting performance
6. **Optimization**: Recommendations for performance improvement

Include specific metrics and technical recommendations.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    @staticmethod
    def create_device_enumeration_prompt(context: str) -> str:
        """Create prompt for PCIe device enumeration analysis"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{PCIePromptTemplates.SYSTEM_PROMPT}

Focus on PCIe device enumeration and configuration space analysis.<|eot_id|><|start_header_id|>user<|end_header_id|>

Analyze PCIe device enumeration from these logs:

{context}

Please examine:
1. **Device Discovery**: Which devices were found during enumeration
2. **Configuration Space**: Key configuration registers and their values
3. **Resource Allocation**: Memory and I/O space assignments
4. **Capabilities**: PCIe capabilities advertised by devices
5. **Enumeration Issues**: Any problems during device discovery
6. **Topology**: PCIe hierarchy and device relationships

Provide detailed analysis of the enumeration process.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    @staticmethod
    def create_power_management_prompt(context: str) -> str:
        """Create prompt for PCIe power management analysis"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{PCIePromptTemplates.SYSTEM_PROMPT}

Focus on PCIe power management states and transitions.<|eot_id|><|start_header_id|>user<|end_header_id|>

Analyze PCIe power management from these logs:

{context}

Please evaluate:
1. **Power States**: D0, D1, D2, D3hot, D3cold transitions
2. **Link States**: L0, L0s, L1, L2, L3 link power states
3. **ASPM**: Active State Power Management configuration
4. **Wake Events**: Power management wake-up events
5. **State Transitions**: Power state change sequences
6. **Issues**: Power management related problems

Provide detailed analysis of power management behavior.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    @staticmethod
    def create_tlp_analysis_prompt(context: str) -> str:
        """Create prompt for Transaction Layer Packet analysis"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{PCIePromptTemplates.SYSTEM_PROMPT}

Focus on PCIe Transaction Layer Packet (TLP) analysis.<|eot_id|><|start_header_id|>user<|end_header_id|>

Analyze PCIe TLP transactions from these logs:

{context}

Please examine:
1. **TLP Types**: Memory reads/writes, configuration, completion, etc.
2. **TLP Format**: Header format, addressing, payload analysis
3. **Transaction Flow**: Request-completion pairs
4. **Errors**: TLP-related errors and malformed packets
5. **Performance**: Transaction efficiency and timing
6. **Routing**: How TLPs are routed through the PCIe fabric

Provide detailed TLP-level analysis with packet-specific insights.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    @staticmethod
    def create_summary_prompt(analysis_results: List[str]) -> str:
        """Create prompt for summarizing multiple analysis results"""
        combined_analysis = "\\n\\n".join(analysis_results)
        
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{PCIePromptTemplates.SYSTEM_PROMPT}

Provide executive summary of PCIe analysis results.<|eot_id|><|start_header_id|>user<|end_header_id|>

Based on the following detailed PCIe analysis results, provide a comprehensive summary:

{combined_analysis}

Please create:
1. **Executive Summary**: High-level overview of findings
2. **Critical Issues**: Most important problems identified
3. **System Impact**: Overall impact on system operation
4. **Priority Actions**: Recommended actions in order of priority
5. **Long-term Recommendations**: Strategic improvements

Keep the summary concise but technically accurate.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    @staticmethod
    def get_prompt_for_query_type(query: str, context: str) -> str:
        """
        Select appropriate prompt template based on query content
        
        Args:
            query: User's query
            context: Log context
            
        Returns:
            Formatted prompt string
        """
        query_lower = query.lower()
        
        # Determine query type and use appropriate template
        if any(term in query_lower for term in ['link training', 'ltssm', 'link state']):
            return PCIePromptTemplates.create_link_training_analysis_prompt(context)
        elif any(term in query_lower for term in ['performance', 'throughput', 'bandwidth', 'latency']):
            return PCIePromptTemplates.create_performance_analysis_prompt(context)
        elif any(term in query_lower for term in ['enumeration', 'device discovery', 'configuration space']):
            return PCIePromptTemplates.create_device_enumeration_prompt(context)
        elif any(term in query_lower for term in ['power', 'aspm', 'l0s', 'l1', 'd0', 'd3']):
            return PCIePromptTemplates.create_power_management_prompt(context)
        elif any(term in query_lower for term in ['tlp', 'transaction', 'packet', 'memory read', 'memory write']):
            return PCIePromptTemplates.create_tlp_analysis_prompt(context)
        else:
            # Default to general analysis
            return PCIePromptTemplates.create_analysis_prompt(query, context)

    @staticmethod
    def get_available_templates() -> Dict[str, str]:
        """Get list of available prompt templates"""
        return {
            "general_analysis": "General PCIe log analysis",
            "error_classification": "Classify and categorize PCIe errors",
            "link_training": "Analyze PCIe link training sequences",
            "performance": "Performance analysis and optimization",
            "device_enumeration": "Device discovery and enumeration",
            "power_management": "Power state analysis",
            "tlp_analysis": "Transaction Layer Packet analysis",
            "summary": "Executive summary of multiple analyses"
        }