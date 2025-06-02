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
        prompt = f"""You are a PCIe debugging expert. Based on the following context, answer the user's query.

Query: {query}

Context:
{context}

Please provide a detailed technical answer based on the context provided."""

        return self._call_llm(prompt)
            
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