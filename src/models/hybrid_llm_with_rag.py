#!/usr/bin/env python3
"""
Hybrid LLM Provider with Enhanced RAG and Context Compression
Integrates RAG context compression with the hybrid LLM system
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time

from src.models.hybrid_llm_provider import HybridLLMProvider, AnalysisRequest, AnalysisResponse
from src.rag.enhanced_rag_engine_v2 import EnhancedRAGEngineV2, EnhancedRAGQuery
from src.rag.context_compressor import ContextCompressor

logger = logging.getLogger(__name__)

@dataclass
class RAGAnalysisRequest:
    """Analysis request with RAG enhancement"""
    query: str
    error_log: str = ""
    analysis_type: str = "auto"  # auto, quick, detailed
    max_response_time: float = 60.0
    # RAG parameters
    use_rag: bool = True
    compress_context: bool = True
    context_window: int = 5
    max_context_tokens: int = 6000

class HybridLLMWithRAG:
    """
    Hybrid LLM Provider enhanced with RAG and Context Compression
    
    Architecture:
    1. Query Analysis -> Determine analysis type
    2. RAG Retrieval -> Find relevant documentation  
    3. Context Compression -> Optimize context for token limits
    4. Hybrid LLM -> Generate response with appropriate model
    5. Quality Enhancement -> Post-process and validate
    """
    
    def __init__(self, 
                 vector_store_path: Optional[str] = None,
                 enable_compression: bool = True):
        """
        Initialize hybrid LLM with RAG
        
        Args:
            vector_store_path: Path to FAISS vector store
            enable_compression: Enable context compression
        """
        
        # Initialize base hybrid provider
        self.hybrid_provider = HybridLLMProvider()
        
        # Initialize enhanced RAG engine
        self.rag_engine = EnhancedRAGEngineV2(
            vector_store=None,  # Would load from vector_store_path
            model_manager=None,  # Would initialize embeddings
            llm_provider="local"
        )
        
        self.enable_compression = enable_compression
        
        # Quality enhancement settings
        self.quality_settings = {
            "min_response_length": 100,
            "required_pcie_terms": ["pcie", "link", "error", "ltssm", "tlp"],
            "response_structure": ["analysis", "cause", "solution"]
        }
        
        logger.info("Hybrid LLM with RAG initialized")
    
    def analyze_pcie_error_with_rag(self, request: RAGAnalysisRequest) -> AnalysisResponse:
        """
        Analyze PCIe error with RAG enhancement
        
        Args:
            request: RAG analysis request
            
        Returns:
            Enhanced analysis response
        """
        start_time = time.time()
        
        try:
            # Step 1: Determine optimal analysis strategy
            analysis_type = self._determine_analysis_type(request)
            
            # Step 2: Retrieve and compress context using RAG
            enhanced_context = ""
            compression_info = None
            
            if request.use_rag:
                rag_query = EnhancedRAGQuery(
                    query=request.query,
                    error_log=request.error_log,
                    analysis_type=analysis_type,
                    compress_context=request.compress_context,
                    max_context_tokens=request.max_context_tokens,
                    context_window=request.context_window
                )
                
                rag_response = self.rag_engine.query(rag_query)
                
                if rag_response.compression_info:
                    enhanced_context = rag_response.compression_info.compressed_text
                    compression_info = rag_response.compression_info
                else:
                    enhanced_context = "\\n".join([
                        source.get('content', '') for source in rag_response.sources
                    ])
            
            # Step 3: Create enhanced prompt with compressed context
            enhanced_prompt = self._create_enhanced_prompt(
                request.query, 
                request.error_log, 
                enhanced_context,
                analysis_type
            )
            
            # Step 4: Generate response using hybrid LLM
            base_request = AnalysisRequest(
                query=enhanced_prompt,
                error_log=request.error_log,
                analysis_type=analysis_type,
                max_response_time=request.max_response_time
            )
            
            response = self.hybrid_provider.analyze_pcie_error(base_request)
            
            # Step 5: Enhance response quality
            enhanced_response = self._enhance_response_quality(response, request)
            
            # Step 6: Add RAG metadata
            if enhanced_response.metadata is None:
                enhanced_response.metadata = {}
            
            enhanced_response.metadata.update({
                "rag_enabled": request.use_rag,
                "context_compressed": request.compress_context,
                "compression_ratio": compression_info.compression_ratio if compression_info else 1.0,
                "context_relevance": compression_info.relevance_score if compression_info else 0.0,
                "total_processing_time": time.time() - start_time
            })
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"RAG-enhanced analysis failed: {e}")
            
            # Fallback to basic hybrid analysis
            base_request = AnalysisRequest(
                query=request.query,
                error_log=request.error_log,
                analysis_type=request.analysis_type,
                max_response_time=request.max_response_time
            )
            
            response = self.hybrid_provider.analyze_pcie_error(base_request)
            
            if response.metadata is None:
                response.metadata = {}
            response.metadata["rag_fallback"] = str(e)
            
            return response
    
    def _determine_analysis_type(self, request: RAGAnalysisRequest) -> str:
        """Determine optimal analysis type based on request"""
        if request.analysis_type != "auto":
            return request.analysis_type
        
        # Analyze query complexity
        query_lower = request.query.lower()
        error_log_length = len(request.error_log)
        
        # Complex indicators
        complex_keywords = [
            "root cause", "detailed", "comprehensive", "analysis",
            "debug", "investigate", "deep dive", "troubleshoot"
        ]
        
        # Quick indicators  
        quick_keywords = [
            "quick", "simple", "what is", "basic", "summary"
        ]
        
        has_complex = any(kw in query_lower for kw in complex_keywords)
        has_quick = any(kw in query_lower for kw in quick_keywords)
        
        if has_quick or (error_log_length < 100 and not has_complex):
            return "quick"
        elif has_complex or error_log_length > 1000:
            return "detailed"
        else:
            return "quick"  # Default to quick for better performance
    
    def _create_enhanced_prompt(self, 
                              query: str,
                              error_log: str,
                              context: str,
                              analysis_type: str) -> str:
        """Create enhanced prompt with RAG context"""
        
        prompt_parts = [
            "You are an expert PCIe (PCI Express) debug engineer with deep knowledge of hardware and protocol specifications.",
            "",
            f"ANALYSIS TYPE: {analysis_type.upper()}",
            "",
            "QUERY:",
            query,
            ""
        ]
        
        if error_log.strip():
            prompt_parts.extend([
                "ERROR LOG:",
                error_log,
                ""
            ])
        
        if context.strip():
            prompt_parts.extend([
                "RELEVANT CONTEXT:",
                context,
                ""
            ])
        
        if analysis_type == "detailed":
            prompt_parts.extend([
                "Please provide a DETAILED analysis including:",
                "1. PROBLEM IDENTIFICATION: What specific PCIe issue is occurring?",
                "2. ROOT CAUSE ANALYSIS: Why is this happening? Include technical details.",
                "3. STEP-BY-STEP RESOLUTION: Specific actions to resolve the issue.",
                "4. PREVENTION: How to prevent this issue in the future.",
                "5. RELATED ISSUES: What other problems might be related?",
                "",
                "Use precise PCIe terminology and provide actionable technical guidance."
            ])
        else:
            prompt_parts.extend([
                "Please provide a QUICK analysis including:",
                "1. What is the main PCIe issue?",
                "2. Most likely cause?", 
                "3. Primary resolution steps?",
                "",
                "Be concise but technically accurate."
            ])
        
        return "\\n".join(prompt_parts)
    
    def _enhance_response_quality(self, 
                                response: AnalysisResponse,
                                request: RAGAnalysisRequest) -> AnalysisResponse:
        """Enhance response quality with post-processing"""
        
        if not response.response or response.error:
            return response
        
        enhanced_response = response.response
        
        # Quality check 1: Ensure minimum length
        if len(enhanced_response) < self.quality_settings["min_response_length"]:
            enhanced_response += "\\n\\nFor more detailed analysis, please provide additional error logs or specific questions about the PCIe issue."
        
        # Quality check 2: Ensure PCIe terminology
        response_lower = enhanced_response.lower()
        missing_terms = [
            term for term in self.quality_settings["required_pcie_terms"]
            if term not in response_lower and term in request.query.lower()
        ]
        
        if missing_terms and "pcie" not in response_lower:
            enhanced_response = f"PCIe Analysis: {enhanced_response}"
        
        # Quality check 3: Structure validation
        if request.analysis_type == "detailed":
            if not any(word in response_lower for word in ["cause", "analysis", "root"]):
                enhanced_response += "\\n\\nNote: For comprehensive root cause analysis, please ensure error logs contain sufficient detail about the PCIe link state and error conditions."
        
        # Create enhanced response object
        enhanced = AnalysisResponse(
            response=enhanced_response,
            model_used=response.model_used,
            response_time=response.response_time,
            fallback_used=response.fallback_used,
            error=response.error,
            metadata=response.metadata
        )
        
        return enhanced
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        base_status = self.hybrid_provider.get_model_status()
        
        rag_status = {
            "rag_engine": "initialized",
            "context_compression": self.enable_compression,
            "rag_metrics": self.rag_engine.get_metrics()
        }
        
        return {
            **base_status,
            "rag_system": rag_status
        }

def main():
    """Test RAG-enhanced hybrid LLM"""
    print("🚀 Testing Hybrid LLM with RAG Enhancement")
    print("=" * 60)
    
    # Initialize system
    hybrid_rag = HybridLLMWithRAG(enable_compression=True)
    
    # Test request
    request = RAGAnalysisRequest(
        query="Why is PCIe link training failing and how can I fix it?",
        error_log="[10:15:30.123] PCIe: 0000:01:00.0 - Link training failed\\n[10:15:31.456] PCIe: LTSSM stuck in Recovery.RcvrLock state\\n[10:15:32.789] PCIe: Signal quality degraded on lanes 0-3",
        analysis_type="detailed",
        use_rag=True,
        compress_context=True
    )
    
    # Execute analysis
    print(f"Query: {request.query}")
    print(f"Analysis Type: {request.analysis_type}")
    print(f"RAG Enabled: {request.use_rag}")
    print(f"Compression: {request.compress_context}")
    print()
    
    start_time = time.time()
    response = hybrid_rag.analyze_pcie_error_with_rag(request)
    total_time = time.time() - start_time
    
    print("Results:")
    print(f"Model Used: {response.model_used}")
    print(f"Response Time: {total_time:.2f}s")
    print(f"Response Length: {len(response.response)} chars")
    
    if response.metadata:
        print(f"RAG Enabled: {response.metadata.get('rag_enabled', False)}")
        print(f"Compression Ratio: {response.metadata.get('compression_ratio', 1.0):.2f}")
        print(f"Context Relevance: {response.metadata.get('context_relevance', 0.0):.2f}")
    
    print()
    print("Response Preview:")
    print("-" * 40)
    print(response.response[:300] + "..." if len(response.response) > 300 else response.response)
    
    # System status
    print()
    print("System Status:")
    status = hybrid_rag.get_system_status()
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()