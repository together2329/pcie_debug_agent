"""
Hybrid LLM Provider for PCIe Error Analysis
Combines Llama 3.2 3B (fast) + DeepSeek Q4_1 (detailed) for optimal debugging
"""

import logging
import time
import subprocess
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass
from pathlib import Path

from .local_llm_provider import LocalLLMProvider

logger = logging.getLogger(__name__)

@dataclass
class AnalysisRequest:
    """Request for PCIe error analysis"""
    query: str
    error_log: str = ""
    analysis_type: Literal["quick", "detailed", "auto"] = "auto"
    max_response_time: float = 30.0  # seconds
    context: str = ""

@dataclass
class AnalysisResponse:
    """Response from hybrid analysis"""
    response: str
    model_used: str
    analysis_type: str
    response_time: float
    confidence_score: float
    fallback_used: bool = False
    error: Optional[str] = None

class HybridLLMProvider:
    """
    Hybrid LLM Provider that intelligently chooses between:
    - Llama 3.2 3B: Fast interactive analysis
    - DeepSeek Q4_1: Detailed comprehensive analysis
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        
        # Initialize Llama provider (fast model)
        self.llama_provider = None
        self.llama_available = self._init_llama()
        
        # Initialize DeepSeek availability (detailed model)
        self.deepseek_available = self._check_deepseek()
        
        # Configuration for different analysis types
        self.analysis_configs = {
            "quick": {
                "preferred_model": "llama",
                "max_tokens": 500,
                "temperature": 0.1,
                "timeout": 15.0
            },
            "detailed": {
                "preferred_model": "deepseek",
                "max_tokens": 2000,
                "temperature": 0.1,
                "timeout": 180.0
            },
            "auto": {
                "quick_threshold": 30.0,  # seconds
                "detailed_threshold": 100,  # characters in query
                "fallback_enabled": True
            }
        }
        
        logger.info(f"Hybrid LLM Provider initialized - Llama: {self.llama_available}, DeepSeek: {self.deepseek_available}")
    
    def _init_llama(self) -> bool:
        """Initialize Llama provider with fixed configuration"""
        try:
            self.llama_provider = LocalLLMProvider(
                models_dir=str(self.models_dir),
                n_ctx=16384,  # Increased context window
                n_gpu_layers=-1,
                verbose=False
            )
            
            if self.llama_provider.model_path.exists():
                logger.info("Llama 3.2 3B model available for fast analysis")
                return True
            else:
                logger.warning("Llama 3.2 3B model not found")
                return False
                
        except Exception as e:
            logger.warning(f"Failed to initialize Llama provider: {e}")
            return False
    
    def _check_deepseek(self) -> bool:
        """Check if DeepSeek model is available via Ollama"""
        try:
            result = subprocess.run(
                ['ollama', 'list'], 
                capture_output=True, 
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and 'deepseek-r1' in result.stdout:
                logger.info("DeepSeek Q4_1 model available for detailed analysis")
                return True
            else:
                logger.warning("DeepSeek model not found in Ollama")
                return False
                
        except Exception as e:
            logger.warning(f"Failed to check DeepSeek availability: {e}")
            return False
    
    def analyze_pcie_error(self, request: AnalysisRequest) -> AnalysisResponse:
        """
        Analyze PCIe error using hybrid approach
        
        Args:
            request: Analysis request with query, error log, and preferences
            
        Returns:
            Analysis response with results and metadata
        """
        start_time = time.time()
        
        # Determine analysis type if auto
        if request.analysis_type == "auto":
            analysis_type = self._determine_analysis_type(request)
        else:
            analysis_type = request.analysis_type
        
        logger.info(f"Starting {analysis_type} PCIe error analysis")
        
        # Try preferred model for analysis type
        config = self.analysis_configs[analysis_type]
        preferred_model = config.get("preferred_model", "llama")
        
        if preferred_model == "llama" and self.llama_available:
            response = self._analyze_with_llama(request, config)
        elif preferred_model == "deepseek" and self.deepseek_available:
            response = self._analyze_with_deepseek(request, config)
        else:
            # Fallback to available model
            response = self._analyze_with_fallback(request, analysis_type)
        
        response.analysis_type = analysis_type
        response.response_time = time.time() - start_time
        
        return response
    
    def _determine_analysis_type(self, request: AnalysisRequest) -> str:
        """Intelligently determine analysis type based on request"""
        auto_config = self.analysis_configs["auto"]
        
        # Check if user wants quick response
        if request.max_response_time <= auto_config["quick_threshold"]:
            return "quick"
        
        # Check query complexity
        query_length = len(request.query + request.error_log)
        if query_length > auto_config["detailed_threshold"]:
            return "detailed"
        
        # Check for complex PCIe scenarios
        complex_keywords = [
            "correlation", "root cause", "sequence", "workflow", 
            "multiple errors", "intermittent", "design"
        ]
        
        query_text = (request.query + request.error_log).lower()
        complex_count = sum(1 for kw in complex_keywords if kw in query_text)
        
        if complex_count >= 2:
            return "detailed"
        
        return "quick"
    
    def _analyze_with_llama(self, request: AnalysisRequest, config: Dict) -> AnalysisResponse:
        """Analyze using Llama 3.2 3B for fast response"""
        try:
            # Prepare optimized prompt for Llama
            prompt = self._format_prompt_for_llama(request)
            
            # Generate response with simplified approach
            if self.llama_provider.llm is None:
                self.llama_provider.load_model()
            
            response = self.llama_provider.llm(
                prompt,
                max_tokens=config.get("max_tokens", 500),
                temperature=config.get("temperature", 0.1),
                echo=False
            )
            
            generated_text = response["choices"][0]["text"].strip()
            confidence_score = self._calculate_confidence(generated_text, "llama")
            
            return AnalysisResponse(
                response=generated_text,
                model_used="llama-3.2-3b",
                analysis_type="",  # Will be set by caller
                response_time=0,  # Will be set by caller
                confidence_score=confidence_score,
                fallback_used=False
            )
            
        except Exception as e:
            logger.error(f"Llama analysis failed: {e}")
            return AnalysisResponse(
                response="",
                model_used="llama-3.2-3b",
                analysis_type="",
                response_time=0,
                confidence_score=0.0,
                fallback_used=False,
                error=str(e)
            )
    
    def _analyze_with_deepseek(self, request: AnalysisRequest, config: Dict) -> AnalysisResponse:
        """Analyze using DeepSeek Q4_1 for detailed response"""
        try:
            # Prepare comprehensive prompt for DeepSeek
            prompt = self._format_prompt_for_deepseek(request)
            
            result = subprocess.run(
                ['ollama', 'run', 'deepseek-r1:latest', prompt],
                capture_output=True,
                text=True,
                timeout=config.get("timeout", 180)
            )
            
            if result.returncode == 0:
                generated_text = result.stdout.strip()
                confidence_score = self._calculate_confidence(generated_text, "deepseek")
                
                return AnalysisResponse(
                    response=generated_text,
                    model_used="deepseek-q4_1",
                    analysis_type="",
                    response_time=0,
                    confidence_score=confidence_score,
                    fallback_used=False
                )
            else:
                raise Exception(f"DeepSeek failed with code {result.returncode}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error("DeepSeek analysis timed out")
            return AnalysisResponse(
                response="",
                model_used="deepseek-q4_1",
                analysis_type="",
                response_time=0,
                confidence_score=0.0,
                fallback_used=False,
                error="Timeout"
            )
        except Exception as e:
            logger.error(f"DeepSeek analysis failed: {e}")
            return AnalysisResponse(
                response="",
                model_used="deepseek-q4_1", 
                analysis_type="",
                response_time=0,
                confidence_score=0.0,
                fallback_used=False,
                error=str(e)
            )
    
    def _analyze_with_fallback(self, request: AnalysisRequest, analysis_type: str) -> AnalysisResponse:
        """Fallback analysis when preferred model is unavailable"""
        logger.warning(f"Using fallback for {analysis_type} analysis")
        
        # Try available models in order of preference
        if analysis_type == "detailed" and self.llama_available:
            # Use Llama for detailed analysis if DeepSeek unavailable
            config = self.analysis_configs["detailed"]
            config["max_tokens"] = 1500  # Increase tokens for Llama detailed analysis
            response = self._analyze_with_llama(request, config)
            response.fallback_used = True
            return response
            
        elif analysis_type == "quick" and self.deepseek_available:
            # Use DeepSeek for quick analysis if Llama unavailable (not ideal but works)
            config = self.analysis_configs["quick"]
            config["timeout"] = 60  # Shorter timeout for "quick" DeepSeek
            response = self._analyze_with_deepseek(request, config)
            response.fallback_used = True
            return response
        
        # No models available
        return AnalysisResponse(
            response="No LLM models available for analysis",
            model_used="none",
            analysis_type=analysis_type,
            response_time=0,
            confidence_score=0.0,
            fallback_used=True,
            error="No models available"
        )
    
    def _format_prompt_for_llama(self, request: AnalysisRequest) -> str:
        """Format prompt optimized for Llama fast analysis"""
        prompt = f"""PCIe Error Analysis:

{request.error_log}

Query: {request.query}

{request.context}

Provide concise analysis with:
1. Root cause
2. Fix steps
3. Prevention"""
        
        return prompt
    
    def _format_prompt_for_deepseek(self, request: AnalysisRequest) -> str:
        """Format prompt optimized for DeepSeek detailed analysis"""
        prompt = f"""Comprehensive PCIe Error Analysis:

Error Log:
{request.error_log}

Analysis Request: {request.query}

Context: {request.context}

Please provide detailed analysis including:
1. Comprehensive root cause analysis
2. Step-by-step failure sequence explanation
3. Detailed debugging methodology
4. Prevention and monitoring strategies
5. Related PCIe protocol considerations
6. Hardware and software recommendations"""
        
        return prompt
    
    def _calculate_confidence(self, response: str, model: str) -> float:
        """Calculate confidence score based on response quality"""
        if not response:
            return 0.0
        
        # Basic scoring based on response characteristics
        score = 0.0
        
        # Length scoring (reasonable responses should be substantial)
        if len(response) > 100:
            score += 0.3
        if len(response) > 500:
            score += 0.2
        
        # PCIe keyword presence
        pcie_keywords = [
            "PCIe", "TLP", "LTSSM", "link training", "recovery", 
            "enumeration", "configuration", "protocol", "error"
        ]
        keyword_count = sum(1 for kw in pcie_keywords if kw.lower() in response.lower())
        score += min(keyword_count * 0.1, 0.4)
        
        # Structure scoring (numbered lists, clear sections)
        if any(marker in response for marker in ["1.", "2.", "3.", "•", "-"]):
            score += 0.1
        
        return min(score, 1.0)
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of both models"""
        return {
            "llama": {
                "available": self.llama_available,
                "model": "Llama 3.2 3B Instruct",
                "use_case": "Fast interactive analysis",
                "typical_response_time": "< 5 seconds"
            },
            "deepseek": {
                "available": self.deepseek_available,
                "model": "DeepSeek Q4_1",
                "use_case": "Detailed comprehensive analysis", 
                "typical_response_time": "30-180 seconds"
            },
            "hybrid_ready": self.llama_available or self.deepseek_available
        }
    
    def quick_analysis(self, query: str, error_log: str = "") -> AnalysisResponse:
        """Convenience method for quick analysis"""
        request = AnalysisRequest(
            query=query,
            error_log=error_log,
            analysis_type="quick",
            max_response_time=15.0
        )
        return self.analyze_pcie_error(request)
    
    def detailed_analysis(self, query: str, error_log: str = "", context: str = "") -> AnalysisResponse:
        """Convenience method for detailed analysis"""
        request = AnalysisRequest(
            query=query,
            error_log=error_log,
            analysis_type="detailed",
            max_response_time=300.0,
            context=context
        )
        return self.analyze_pcie_error(request)