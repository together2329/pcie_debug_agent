from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class Analyzer:
    """LLM 기반 에러 분석기"""
    
    def __init__(self, 
                 llm_provider: str = "openai",
                 model: str = "gpt-4",
                 temperature: float = 0.1,
                 max_tokens: int = 2000,
                 api_key: Optional[str] = None,
                 model_manager: Optional[Any] = None):
        
        self.llm_provider = llm_provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_manager = model_manager
        
        # Store API key for later use if needed
        self.api_key = api_key
            
    def analyze(self,
               query: str,
               context: str,
               analysis_type: str = "answer_with_sources") -> str:
        """
        Generic analysis method for RAG queries
        
        Args:
            query: User query
            context: Retrieved context documents
            analysis_type: Type of analysis to perform
            
        Returns:
            Analysis response string
        """
        # Create analysis-specific prompt based on analysis_type
        if analysis_type == "error_analysis":
            prompt = f"""You are a PCIe debugging expert specializing in error analysis. 

QUERY: {query}

CONTEXT FROM KNOWLEDGE BASE:
{context}

ANALYSIS INSTRUCTIONS:
1. Focus specifically on the technical issue described in the query
2. Identify any PCIe compliance violations or protocol errors
3. Explain expected vs actual behavior for the specific scenario
4. Provide concrete debugging steps for this exact issue
5. Reference relevant PCIe specification sections if applicable

Provide a focused, technical answer addressing the specific PCIe issue."""
        
        elif analysis_type == "debug_analysis":
            prompt = f"""You are a PCIe debugging expert. Provide specific debugging guidance.

PROBLEM: {query}

KNOWLEDGE BASE CONTEXT:
{context}

DEBUGGING FOCUS:
- Identify root cause of the specific issue
- Explain why the observed behavior occurs
- Provide step-by-step debugging approach
- Suggest specific register checks or log analysis
- Reference PCIe protocol requirements

Answer with specific debugging guidance for this exact scenario."""
        
        elif analysis_type == "compliance_analysis":
            prompt = f"""You are a PCIe compliance expert. Analyze the compliance issue.

COMPLIANCE QUESTION: {query}

RELEVANT STANDARDS:
{context}

COMPLIANCE ANALYSIS:
- Identify specific PCIe specification requirements
- Explain compliance violations if any
- Detail expected protocol behavior
- Provide compliance verification steps

Focus on PCIe specification compliance for this specific scenario."""
        
        else:
            # Default technical analysis prompt
            prompt = f"""You are a PCIe technical expert. Provide detailed technical analysis.

TECHNICAL QUERY: {query}

RELEVANT DOCUMENTATION:
{context}

Provide a comprehensive technical explanation focusing on:
1. The specific PCIe concept or mechanism involved
2. Technical details and protocol behavior
3. Implementation considerations
4. Common issues and troubleshooting tips

Answer with precise technical details for this query."""

        try:
            return self._call_llm(prompt)
        except Exception as e:
            logger.warning(f"LLM call failed for analysis: {e}")
            # Provide fallback analysis based on context
            return self._create_fallback_analysis(query, context, analysis_type)
    
    def _create_fallback_analysis(self, query: str, context: str, analysis_type: str) -> str:
        """Create fallback analysis when LLM is unavailable"""
        if not context.strip():
            return f"No relevant information found for query: '{query}'. Please check knowledge base."
        
        fallback = f"Analysis for: {query}\n\n"
        fallback += "Based on available knowledge base content:\n\n"
        
        # Extract key information from context
        context_lines = context.split('\n')
        relevant_lines = [line for line in context_lines if line.strip() and not line.startswith('[Source')]
        
        if relevant_lines:
            fallback += "Key findings:\n"
            for i, line in enumerate(relevant_lines[:5], 1):
                if len(line.strip()) > 20:  # Skip very short lines
                    fallback += f"{i}. {line.strip()}\n"
        
        fallback += f"\n💡 Note: This is a basic analysis. For detailed technical analysis, please configure an LLM model."
        return fallback
            
    def analyze_error(self,
                     error: 'UVMError',
                     context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        단일 에러 분석
        
        Args:
            error: UVM 에러 객체
            context: 검색된 관련 문서
            
        Returns:
            분석 결과 딕셔너리
        """
        # 프롬프트 생성
        prompt = self._create_analysis_prompt(error, context)
        
        # LLM 호출
        response = self._call_llm(prompt)
        
        # 결과 파싱
        analysis = self._parse_response(response)
        
        # 메타데이터 추가
        analysis['metadata'] = {
            'error_id': f"{error.component}_{error.timestamp}",
            'analyzed_at': datetime.now().isoformat(),
            'model': self.model,
            'context_docs': len(context)
        }
        
        return analysis
    
    def _create_analysis_prompt(self, error: 'UVMError', context: List[Dict[str, Any]]) -> str:
        """분석 프롬프트 생성"""
        
        # 컨텍스트 문서 정리
        context_str = "\n\n".join([
            f"[Document {i+1}]\n"
            f"Source: {doc['metadata'].get('source', 'Unknown')}\n"
            f"Type: {doc['metadata'].get('type', 'Unknown')}\n"
            f"Content: {doc['content'][:500]}..."
            for i, doc in enumerate(context[:5])  # 상위 5개만
        ])
        
        prompt = f"""You are a UVM (Universal Verification Methodology) expert analyzing verification errors.

## Error Information
- Timestamp: {error.timestamp}
- Severity: {error.severity}
- Component: {error.component}
- Message: {error.message}
- File: {error.file_path}:{error.line_number}

## Error Context (before/after lines)
Before:
{chr(10).join(error.context_before[-3:])}

Error Line:
{error.raw_content}

After:
{chr(10).join(error.context_after[:3])}

## Related Documentation/Code
{context_str}

## Analysis Required
Please provide a detailed analysis including:

1. **Root Cause Analysis**
   - What is the fundamental cause of this error?
   - Which UVM/SystemVerilog concepts are involved?

2. **Component Analysis**
   - Which components are affected?
   - What is the data/control flow leading to this error?

3. **Suggested Fixes**
   - Provide specific code changes or configuration fixes
   - Include example code if applicable

4. **Prevention**
   - How can similar errors be prevented in the future?
   - Any best practices to follow?

5. **Additional Investigation**
   - What additional information would help diagnose this issue?
   - Suggested debug steps or checks

Please format your response as a JSON object with the above sections."""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """LLM API 호출"""
        try:
            # Use model_manager if available
            if self.model_manager and hasattr(self.model_manager, 'generate_completion'):
                return self.model_manager.generate_completion(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            
            # Otherwise, try to use the model selector directly
            try:
                from src.models.model_selector import get_model_selector
                model_selector = get_model_selector()
                return model_selector.generate_completion(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            except Exception as e:
                logger.error(f"Failed to use model selector: {e}")
                
                # Final fallback
                return "Unable to generate response due to LLM configuration issues."
                
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return "{}"
            
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """LLM 응답 파싱"""
        try:
            # JSON 파싱 시도
            return json.loads(response)
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 텍스트로 처리
            logger.warning("Failed to parse JSON response, using text format")
            return {
                "root_cause": response,
                "component_analysis": "Unable to parse structured response",
                "suggested_fixes": "See root_cause section",
                "prevention": "See root_cause section",
                "additional_investigation": "Manual review required"
            } 