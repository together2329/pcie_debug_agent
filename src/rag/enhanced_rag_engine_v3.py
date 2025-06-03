"""
Enhanced RAG Engine v3 - Integrated PCIe Domain Intelligence
Combines all improvements: domain classification, answer verification, question normalization
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime

# Import our new enhanced components
from src.rag.pcie_knowledge_classifier import PCIeKnowledgeClassifier, PCIeKnowledgeItem
from src.rag.answer_verifier import AnswerVerifier, AnswerVerification
from src.rag.question_normalizer import QuestionNormalizer, NormalizedQuestion
from src.rag.hybrid_search import HybridSearchEngine, HybridSearchResult

# Import existing components
from src.vectorstore.faiss_store import FAISSVectorStore
from src.rag.retriever import Retriever
from src.rag.analyzer import Analyzer

logger = logging.getLogger(__name__)

@dataclass
class EnhancedRAGQuery:
    """Enhanced RAG query with domain intelligence"""
    query: str
    context_window: int = 5
    use_hybrid_search: bool = True
    verify_answer: bool = True
    normalize_question: bool = True
    min_confidence: float = 0.3
    max_results: int = 10
    category_filter: Optional[str] = None
    include_metadata: bool = True

@dataclass
class EnhancedRAGResponse:
    """Enhanced RAG response with comprehensive analysis"""
    answer: str
    confidence: float
    verification: Optional[AnswerVerification]
    normalized_question: Optional[NormalizedQuestion]
    sources: List[Dict[str, Any]]
    knowledge_items: List[PCIeKnowledgeItem]
    reasoning: Optional[str] = None
    suggestions: List[str] = None
    metadata: Optional[Dict[str, Any]] = None

class EnhancedRAGEngineV3:
    """Enhanced RAG Engine with PCIe domain intelligence and answer verification"""
    
    def __init__(self,
                 vector_store: FAISSVectorStore,
                 model_manager,
                 llm_provider: str = "openai",
                 llm_model: str = "gpt-4",
                 temperature: float = 0.1,
                 max_tokens: int = 2000,
                 use_hybrid_search: bool = True):
        
        self.vector_store = vector_store
        self.model_manager = model_manager
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize enhanced components
        self.knowledge_classifier = PCIeKnowledgeClassifier()
        self.answer_verifier = AnswerVerifier()
        self.question_normalizer = QuestionNormalizer()
        
        # Initialize hybrid search if requested
        self.hybrid_search = None
        if use_hybrid_search:
            try:
                self.hybrid_search = HybridSearchEngine(vector_store)
                logger.info("Hybrid search engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize hybrid search: {e}")
        
        # Initialize existing components
        self.retriever = Retriever(vector_store, model_manager)
        self.analyzer = Analyzer(
            llm_provider=llm_provider,
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            model_manager=model_manager
        )
        
        # Enhanced metrics
        self.metrics = {
            "queries_processed": 0,
            "average_response_time": 0,
            "average_confidence": 0,
            "cache_hits": 0,
            "verification_successes": 0,
            "normalization_successes": 0,
            "hybrid_search_used": 0
        }
        
        # Query cache with enhanced keys
        self.query_cache = {}
        self.cache_size = 100
        
    def query(self, rag_query: EnhancedRAGQuery) -> EnhancedRAGResponse:
        """
        Enhanced RAG query with domain intelligence
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Question normalization
            normalized_question = None
            if rag_query.normalize_question:
                normalized_question = self.question_normalizer.normalize_question(rag_query.query)
                self.metrics["normalization_successes"] += 1
                
                # Use normalized form for better search
                search_query = normalized_question.normalized_form
                logger.info(f"Normalized question: {rag_query.query} -> {search_query}")
            else:
                search_query = rag_query.query
            
            # Step 2: Enhanced retrieval
            retrieved_results = self._enhanced_retrieval(search_query, rag_query)
            
            # Step 3: Knowledge classification - Enhanced to consider query context
            knowledge_items = []
            
            # First, classify the query itself for primary category
            query_classification = self.knowledge_classifier.classify_content(
                search_query,
                {'source': 'user_query'}
            )
            knowledge_items.append(query_classification)
            
            # Then classify retrieved content in context of the query
            for result in retrieved_results:
                # Combine query context with retrieved content for better classification
                contextual_content = f"{search_query} | {result.get('content', '')}"
                knowledge_item = self.knowledge_classifier.classify_content(
                    contextual_content,
                    result.get('metadata', {})
                )
                knowledge_items.append(knowledge_item)
            
            # Step 4: Generate answer
            answer = self._generate_enhanced_answer(
                search_query, 
                retrieved_results, 
                knowledge_items,
                normalized_question
            )
            
            # Step 5: Answer verification
            verification = None
            if rag_query.verify_answer:
                expected_keywords = []
                if normalized_question:
                    expected_keywords = normalized_question.key_concepts
                
                verification = self.answer_verifier.verify_answer(
                    rag_query.query,
                    answer,
                    retrieved_results,
                    expected_keywords
                )
                
                if verification.confidence >= rag_query.min_confidence:
                    self.metrics["verification_successes"] += 1
                else:
                    logger.warning(f"Low confidence answer: {verification.confidence:.2f}")
            
            # Step 6: Generate suggestions
            suggestions = []
            if normalized_question and normalized_question.key_concepts:
                suggestions = self.question_normalizer.suggest_related_questions(
                    normalized_question.key_concepts
                )
            
            # Step 7: Calculate final confidence
            final_confidence = self._calculate_final_confidence(verification, knowledge_items)
            
            # Update metrics
            self.metrics["queries_processed"] += 1
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(response_time, final_confidence)
            
            return EnhancedRAGResponse(
                answer=answer,
                confidence=final_confidence,
                verification=verification,
                normalized_question=normalized_question,
                sources=retrieved_results,
                knowledge_items=knowledge_items,
                suggestions=suggestions,
                metadata={
                    "response_time": response_time,
                    "search_method": "hybrid" if self.hybrid_search and rag_query.use_hybrid_search else "semantic",
                    "results_count": len(retrieved_results)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced RAG query: {str(e)}")
            return EnhancedRAGResponse(
                answer=f"I encountered an error processing your query: {str(e)}",
                confidence=0.0,
                verification=None,
                normalized_question=None,
                sources=[],
                knowledge_items=[],
                metadata={"error": str(e)}
            )
    
    def _enhanced_retrieval(self, query: str, rag_query: EnhancedRAGQuery) -> List[Dict[str, Any]]:
        """Enhanced retrieval using hybrid search or semantic search"""
        
        if self.hybrid_search and rag_query.use_hybrid_search:
            try:
                # Generate query embedding for hybrid search
                if hasattr(self.model_manager, 'generate_embeddings'):
                    query_embedding = self.model_manager.generate_embeddings([query])[0]
                elif hasattr(self.model_manager, 'embed'):
                    query_embedding = self.model_manager.embed([query])[0]
                else:
                    # Fallback: try using embedding selector
                    from src.models.embedding_selector import get_embedding_selector
                    embedding_selector = get_embedding_selector()
                    provider = embedding_selector.get_current_provider()
                    query_embedding = provider.encode([query])[0]
                
                # Use hybrid search with correct method
                hybrid_results = self.hybrid_search.hybrid_search(
                    query=query,
                    query_embedding=query_embedding,
                    k=rag_query.max_results,
                    alpha=0.7  # Weight for semantic vs keyword
                )
                
                self.metrics["hybrid_search_used"] += 1
                
                # Convert to standard format
                results = []
                for result in hybrid_results:
                    results.append({
                        'content': result.content,
                        'metadata': result.metadata,
                        'score': result.combined_score,
                        'semantic_score': result.semantic_score,
                        'keyword_score': result.keyword_score
                    })
                
                logger.info(f"Hybrid search returned {len(results)} results")
                return results
                
            except Exception as e:
                logger.warning(f"Hybrid search failed, falling back to semantic: {e}")
        
        # Fallback to standard semantic retrieval
        try:
            results = self.retriever.retrieve(
                query=query,
                k=rag_query.max_results
            )
            logger.info(f"Semantic search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def _generate_enhanced_answer(self, 
                                 query: str, 
                                 retrieved_results: List[Dict],
                                 knowledge_items: List[PCIeKnowledgeItem],
                                 normalized_question: Optional[NormalizedQuestion]) -> str:
        """Generate answer using enhanced context and domain knowledge"""
        
        # Build enhanced context
        context_parts = []
        
        # Add classified knowledge
        for item in knowledge_items:
            if item.facts:
                fact_text = "; ".join([fact.content for fact in item.facts[:3]])
                context_parts.append(f"[{item.category.value}] {fact_text}")
        
        # Add original retrieved content
        for result in retrieved_results[:5]:  # Limit context length
            context_parts.append(result.get('content', ''))
        
        context = "\n\n".join(context_parts)
        
        # Enhanced prompt with domain awareness
        if normalized_question:
            intent_guidance = f"Intent: {normalized_question.intent.value}"
            concepts_guidance = f"Key concepts: {', '.join(normalized_question.key_concepts)}"
            prompt_prefix = f"{intent_guidance}. {concepts_guidance}. "
        else:
            prompt_prefix = ""
        
        enhanced_prompt = f"""
{prompt_prefix}
You are a PCIe expert. Answer this question using the provided technical context.

Question: {query}

Context:
{context}

Provide a comprehensive, technically accurate answer. Include specific details like:
- Register offsets and bit fields when relevant
- Timeout values and timing specifications
- Error codes and their meanings
- State transitions and conditions
- Protocol-specific information

Answer:"""
        
        try:
            # Generate answer using the analyzer
            answer = self.analyzer.analyze(query, context, "enhanced_analysis")
            return answer
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "I couldn't generate a complete answer due to a technical issue."
    
    def _calculate_final_confidence(self, 
                                   verification: Optional[AnswerVerification],
                                   knowledge_items: List[PCIeKnowledgeItem]) -> float:
        """Calculate final confidence score"""
        
        confidence_factors = []
        
        # Verification confidence
        if verification:
            confidence_factors.append(verification.confidence * 0.5)
        
        # Knowledge quality confidence
        if knowledge_items:
            avg_fact_confidence = np.mean([
                np.mean([fact.confidence for fact in item.facts]) if item.facts else 0.5
                for item in knowledge_items
            ])
            confidence_factors.append(avg_fact_confidence * 0.3)
        
        # Source quality confidence
        source_quality = 0.7  # Base quality score
        confidence_factors.append(source_quality * 0.2)
        
        if confidence_factors:
            return min(sum(confidence_factors), 1.0)
        else:
            return 0.5  # Default medium confidence
    
    def _update_metrics(self, response_time: float, confidence: float):
        """Update performance metrics"""
        # Update average response time
        prev_avg = self.metrics["average_response_time"]
        count = self.metrics["queries_processed"]
        self.metrics["average_response_time"] = (prev_avg * (count - 1) + response_time) / count
        
        # Update average confidence
        prev_conf = self.metrics["average_confidence"]
        self.metrics["average_confidence"] = (prev_conf * (count - 1) + confidence) / count
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "metrics": self.metrics,
            "components": {
                "knowledge_classifier": "active",
                "answer_verifier": "active",
                "question_normalizer": "active",
                "hybrid_search": "active" if self.hybrid_search else "disabled",
                "vector_store": "active" if self.vector_store else "inactive"
            },
            "cache_stats": {
                "size": len(self.query_cache),
                "max_size": self.cache_size,
                "hit_rate": self.metrics["cache_hits"] / max(self.metrics["queries_processed"], 1)
            }
        }
    
    def benchmark_performance(self, test_queries: List[str]) -> Dict[str, Any]:
        """Benchmark system performance with test queries"""
        results = {
            "total_queries": len(test_queries),
            "successful_queries": 0,
            "high_confidence_answers": 0,
            "verified_answers": 0,
            "average_response_time": 0,
            "average_confidence": 0,
            "detailed_results": []
        }
        
        start_time = datetime.now()
        
        for i, query in enumerate(test_queries):
            try:
                rag_query = EnhancedRAGQuery(query=query)
                response = self.query(rag_query)
                
                results["successful_queries"] += 1
                
                if response.confidence >= 0.7:
                    results["high_confidence_answers"] += 1
                
                if response.verification and response.verification.confidence >= 0.7:
                    results["verified_answers"] += 1
                
                results["detailed_results"].append({
                    "query": query,
                    "confidence": response.confidence,
                    "verification_confidence": response.verification.confidence if response.verification else None,
                    "response_time": response.metadata.get("response_time", 0)
                })
                
            except Exception as e:
                logger.error(f"Benchmark query {i} failed: {e}")
                results["detailed_results"].append({
                    "query": query,
                    "error": str(e)
                })
        
        total_time = (datetime.now() - start_time).total_seconds()
        results["total_time"] = total_time
        
        if results["successful_queries"] > 0:
            results["average_response_time"] = total_time / results["successful_queries"]
            results["average_confidence"] = np.mean([
                r.get("confidence", 0) for r in results["detailed_results"] 
                if "confidence" in r
            ])
        
        return results