"""
Model manager for handling embedding and LLM models
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import openai
import cohere
import anthropic
import requests
import json

logger = logging.getLogger(__name__)

class ModelManager:
    """임베딩 및 LLM 모델 관리자"""
    
    def __init__(self,
                 embedding_model: str,
                 embedding_provider: str,
                 embedding_api_key: Optional[str] = None,
                 embedding_api_base_url: Optional[str] = None,
                 llm_provider: str = "openai",
                 llm_model: str = "gpt-4",
                 llm_api_key: Optional[str] = None,
                 llm_api_base_url: Optional[str] = None):
        """
        Args:
            embedding_model: 임베딩 모델 이름
            embedding_provider: 임베딩 제공자
            embedding_api_key: 임베딩 API 키
            embedding_api_base_url: 임베딩 API 기본 URL
            llm_provider: LLM 제공자
            llm_model: LLM 모델 이름
            llm_api_key: LLM API 키
            llm_api_base_url: LLM API 기본 URL
        """
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.embedding_api_key = embedding_api_key
        self.embedding_api_base_url = embedding_api_base_url
        
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_api_base_url = llm_api_base_url
        
        # API 클라이언트 초기화
        self._initialize_clients()
    
    def _initialize_clients(self):
        """API 클라이언트 초기화"""
        # OpenAI 클라이언트
        if self.embedding_provider == "openai" or self.llm_provider == "openai":
            openai.api_key = self.embedding_api_key if self.embedding_provider == "openai" else self.llm_api_key
            if self.embedding_api_base_url:
                openai.api_base = self.embedding_api_base_url
        
        # Cohere 클라이언트
        if self.embedding_provider == "cohere":
            self.cohere_client = cohere.Client(self.embedding_api_key)
        
        # Anthropic 클라이언트
        if self.llm_provider == "anthropic":
            self.anthropic_client = anthropic.Client(api_key=self.llm_api_key)
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        텍스트 임베딩 생성
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            임베딩 벡터 (numpy array)
        """
        try:
            if self.embedding_provider == "openai":
                response = openai.Embedding.create(
                    input=texts,
                    model=self.embedding_model
                )
                embeddings = [data['embedding'] for data in response['data']]
            
            elif self.embedding_provider == "cohere":
                response = self.cohere_client.embed(
                    texts=texts,
                    model=self.embedding_model
                )
                embeddings = response.embeddings
            
            elif self.embedding_provider == "custom":
                # 커스텀 API 호출
                headers = {
                    "Authorization": f"Bearer {self.embedding_api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": self.embedding_model,
                    "input": texts
                }
                response = requests.post(
                    self.embedding_api_base_url,
                    headers=headers,
                    json=data
                )
                response.raise_for_status()
                embeddings = response.json()['data']
            
            else:
                raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def generate_completion(self,
                          provider: str,
                          model: str,
                          prompt: str,
                          temperature: float = 0.1,
                          max_tokens: int = 2000) -> str:
        """
        LLM 완성 생성
        
        Args:
            provider: LLM 제공자
            model: LLM 모델
            prompt: 프롬프트
            temperature: 온도
            max_tokens: 최대 토큰 수
            
        Returns:
            생성된 텍스트
        """
        try:
            if provider == "openai":
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            
            elif provider == "anthropic":
                response = self.anthropic_client.completion(
                    prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                    model=model,
                    temperature=temperature,
                    max_tokens_to_sample=max_tokens
                )
                return response.completion
            
            elif provider == "custom":
                # 커스텀 API 호출
                headers = {
                    "Authorization": f"Bearer {self.llm_api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                response = requests.post(
                    self.llm_api_base_url,
                    headers=headers,
                    json=data
                )
                response.raise_for_status()
                return response.json()['choices'][0]['text']
            
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
            
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise 