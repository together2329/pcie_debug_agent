#!/usr/bin/env python3
"""
Update ModelManager to support hybrid LLM provider
"""

import os
import shutil

def update_model_manager():
    """Add hybrid provider support to ModelManager"""
    
    # Read the current model_manager.py
    file_path = "src/models/model_manager.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add hybrid to SUPPORTED_MODELS
    if '"hybrid"' not in content:
        # Find the line with "local" models and add "hybrid" after it
        local_models_end = content.find('        }\n    }')
        if local_models_end > 0:
            # Insert hybrid models section
            hybrid_section = ''',
        "hybrid": {
            "llama-3.2-3b-quick": ModelInfo(
                name="Llama 3.2 3B (Quick Analysis)",
                provider="hybrid",
                description="Fast PCIe error analysis",
                context_window=16384,
                memory_usage="~1.8GB"
            ),
            "deepseek-r1-detailed": ModelInfo(
                name="DeepSeek R1 (Detailed Analysis)",
                provider="hybrid",
                description="Comprehensive PCIe analysis",
                context_window=32768,
                memory_usage="~5.2GB"
            ),
            "auto-hybrid": ModelInfo(
                name="Auto Hybrid Selection",
                provider="hybrid",
                description="Automatically selects best model",
                context_window=16384,
                memory_usage="~1.8-5.2GB"
            )
        }'''
            content = content[:local_models_end] + hybrid_section + content[local_models_end:]
    
    # Add hybrid provider initialization
    if 'provider == "hybrid"' not in content:
        # Find the local provider section
        local_section_start = content.find('elif provider == "local":')
        if local_section_start > 0:
            # Find the end of local section (before else:)
            else_start = content.find('else:', local_section_start)
            if else_start > 0:
                # Insert hybrid section
                hybrid_init_section = '''
        elif provider == "hybrid":
            # Initialize hybrid model handler
            try:
                from src.models.hybrid_llm_provider import HybridLLMProvider
                self._clients["hybrid"] = HybridLLMProvider(**kwargs)
            except ImportError as e:
                logger.warning(f"Hybrid LLM not available: {e}")
                self._clients["hybrid"] = None
                
        '''
                content = content[:else_start] + hybrid_init_section + content[else_start:]
    
    # Add hybrid support to generate_completion
    if 'provider == "hybrid"' not in content[content.find('def generate_completion'):]:
        # Find the local provider completion section
        local_completion_start = content.find('elif provider == "local":', content.find('def generate_completion'))
        if local_completion_start > 0:
            # Find the end of local section
            else_completion = content.find('else:', local_completion_start)
            if else_completion > 0:
                # Insert hybrid completion section
                hybrid_completion_section = '''
            elif provider == "hybrid":
                if self._clients["hybrid"] is None:
                    raise RuntimeError("Hybrid models not available")
                    
                # Use HybridLLMProvider
                hybrid_provider = self._clients["hybrid"]
                
                # Determine analysis type from model name or kwargs
                analysis_type = "auto"
                if "quick" in model.lower():
                    analysis_type = "quick"
                elif "detailed" in model.lower():
                    analysis_type = "detailed"
                
                from src.models.hybrid_llm_provider import AnalysisRequest
                request = AnalysisRequest(
                    query=prompt,
                    analysis_type=analysis_type,
                    max_response_time=kwargs.get("timeout", 30.0)
                )
                
                response = hybrid_provider.analyze_pcie_error(request)
                return response.response if response.response else "Analysis failed"
                
            '''
                content = content[:else_completion] + hybrid_completion_section + content[else_completion:]
    
    # Update recommendations
    if '"hybrid_analysis"' not in content:
        # Find the recommendations section
        recommendations_end = content.find('return recommendations.get(use_case, recommendations["fast_analysis"])')
        if recommendations_end > 0:
            # Insert hybrid recommendation
            hybrid_recommendation = ''',
            "hybrid_analysis": {
                "embedding": "sentence-transformers/all-MiniLM-L12-v2",
                "llm_provider": "hybrid",
                "llm_model": "auto-hybrid",
                "reason": "Best of both worlds - fast for simple, detailed for complex"
            }'''
            # Find the last recommendation
            last_recommendation = content.rfind('}', 0, recommendations_end - 50)
            if last_recommendation > 0:
                content = content[:last_recommendation+1] + hybrid_recommendation + content[last_recommendation+1:]
    
    # Backup original file
    shutil.copy(file_path, file_path + '.backup')
    
    # Write updated content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated {file_path} with hybrid LLM support")
    print("📝 Original file backed up as model_manager.py.backup")

if __name__ == "__main__":
    update_model_manager()