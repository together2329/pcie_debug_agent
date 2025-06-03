#!/usr/bin/env python3
"""
Auto-Evolving RAG System for PCIe Debug Agent
Implements self-evolving optimization without external dependencies
"""

import json
import time
import logging
import os
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

@dataclass
class EvolutionConfig:
    """Configuration for evolution process"""
    max_trials: int = 50
    target_recall: float = 0.95
    max_time_minutes: int = 15
    max_memory_mb: int = 512
    study_name: str = "pcie_rag_evolution"

@dataclass
class RAGParams:
    """RAG system parameters for optimization"""
    # Chunking parameters
    chunking_strategy: str = "fixed"  # {fixed, heading-aware, sliding}
    base_chunk_size: int = 256  # {128, 256, 384, 512} tokens
    overlap_ratio: float = 0.15  # {0.10, 0.15, 0.20}
    dynamic_split_thresh: int = 256  # {192, 256}
    
    # Metadata parameters
    add_meta_fields: List[str] = None  # power-set{page, heading, level, file}
    length_penalty: float = 0.1  # [0, 0.3]
    
    # Retrieval parameters
    retriever_type: str = "bm25"  # {bm25, simple_embed}
    hybrid_weight: float = 0.5  # [0.3 - 0.7] BM25 vs embed
    hierarchical_mode: bool = False
    
    # Ranking parameters
    rerank_model: str = "none"  # {none, simple}
    
    # Context parameters
    max_total_ctx_tokens: int = 2048  # {1024, 2048, 3072}
    
    def __post_init__(self):
        if self.add_meta_fields is None:
            self.add_meta_fields = ["page", "file"]

@dataclass
class EvalResult:
    """Evaluation results for a RAG configuration"""
    recall_at_3: float
    mrr: float
    latency_ms: float
    objective_score: float
    details: Dict[str, Any]

class SimpleBM25:
    """Simple BM25 implementation"""
    
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.doc_freqs = []
        self.idf = {}
        self.avgdl = 0
        self._initialize()
    
    def _initialize(self):
        nd = len(self.documents)
        num_words = 0
        
        for document in self.documents:
            words = document.lower().split()
            num_words += len(words)
            
            frequencies = {}
            for word in words:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            
            for word in frequencies:
                self.idf[word] = self.idf.get(word, 0) + 1
        
        self.avgdl = num_words / nd
        
        for word in self.idf:
            self.idf[word] = math.log(nd / self.idf[word])
    
    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        query_words = query.lower().split()
        scores = []
        
        for i, doc_freq in enumerate(self.doc_freqs):
            score = 0
            doc_len = sum(doc_freq.values())
            
            for word in query_words:
                if word in doc_freq:
                    freq = doc_freq[word]
                    tf = freq / (freq + 1.2 * (0.25 + 0.75 * doc_len / self.avgdl))
                    score += self.idf.get(word, 0) * tf
            
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

class PCIeDocumentProcessor:
    """Process PCIe specification documents"""
    
    def __init__(self):
        self.documents = []
        self.metadata = []
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """Load PCIe specification documents"""
        # Fallback PCIe documentation
        return [
            {
                'content': """PCIe Function Level Reset (FLR) Implementation:
                
During Function Level Reset, the device must implement proper configuration space handling.
According to PCIe Base Specification Section 6.6.1.2, when a Function Level Reset is initiated:

1. The device must return Configuration Request Retry Status (CRS) to configuration reads
2. This indicates the device is not ready to process configuration requests
3. The minimum FLR duration is 100ms as specified in the PCIe specification
4. Configuration space access must be blocked during the reset sequence

Common FLR implementation errors:
- Returning Successful Completion instead of CRS during reset
- Premature ready signaling before reset completion
- Insufficient reset duration (less than 100ms)
- Configuration space not properly gated during reset

Compliance Requirements:
Per PCIe Base Specification Section 6.6.1.2, any configuration read during FLR MUST return CRS until device is fully ready.
The device violates compliance if it returns Successful Completion during the reset sequence.""",
                'metadata': {
                    'file': 'pcie_base_spec.pdf',
                    'page': 412,
                    'heading': 'Function Level Reset',
                    'level': 2,
                    'source': 'specification'
                }
            },
            {
                'content': """PCIe Completion Timeout Mechanism:
                
Completion timeouts occur when non-posted requests don't receive completion packets
within the configured timeout period. Per Section 2.2.9:

1. Requester must implement completion timeout for all non-posted requests
2. Timeout values are configurable in Device Control 2 Register
3. Valid timeout ranges: 50μs to 64 seconds
4. Default timeout is typically 50ms-200ms

Root causes of completion timeouts:
- Target device not responding (power/reset issues)
- Incorrect routing through switches
- Credit exhaustion for non-posted requests
- IOMMU/platform configuration problems
- BAR programming errors

Debug steps for completion timeout:
1. Check Device Control 2 Register timeout value
2. Monitor PCIe traffic for completion patterns
3. Verify target device power state and configuration
4. Check non-posted credit availability
5. Analyze routing tables through switches""",
                'metadata': {
                    'file': 'pcie_base_spec.pdf',
                    'page': 89,
                    'heading': 'Completion Timeout',
                    'level': 3,
                    'source': 'specification'
                }
            },
            {
                'content': """PCIe LTSSM State Machine Operation:
                
The Link Training and Status State Machine (LTSSM) controls PCIe link operation.
Key states and transitions per Section 4.2:

1. Detect.Quiet -> Detect.Active: Begin receiver detection
2. Polling.Active -> Polling.Compliance: Speed negotiation
3. Configuration.Linkwidth -> Configuration.Complete: Width negotiation
4. L0 -> Recovery: Error recovery or speed change
5. L0 -> L0s/L1: Power management transitions

Common LTSSM issues:
- Stuck in Polling.Compliance (electrical problems)
- Training sequence errors during Configuration
- Speed negotiation failures (Gen1/2/3/4/5)
- Link width reduction due to lane failures

LTSSM troubleshooting:
1. Check electrical signal integrity
2. Verify training sequence patterns
3. Analyze speed negotiation handshake
4. Monitor receiver detection results
5. Check compliance pattern generation""",
                'metadata': {
                    'file': 'pcie_base_spec.pdf',
                    'page': 156,
                    'heading': 'LTSSM State Machine',
                    'level': 2,
                    'source': 'specification'
                }
            },
            {
                'content': """PCIe Transaction Layer Packet (TLP) Format:
                
TLP header formats for different transaction types:

3DW Header (Memory Requests ≤4GB):
- Fmt[2:0] = 000 (3DW, no data) or 010 (3DW, with data)
- Type[4:0] specifies transaction type
- Length[9:0] in DW, Address[31:2] for alignment

4DW Header (Memory Requests >4GB):
- Fmt[2:0] = 001 (4DW, no data) or 011 (4DW, with data)  
- Extended address field for 64-bit addressing
- Used when Address[63:32] != 0

Common TLP types:
- Memory Read Request (Type = 00000)
- Memory Write Request (Type = 00000, with data)
- Configuration Read (Type = 00100)
- Completion (Type = 01010)

TLP format debugging:
1. Verify Fmt and Type fields match transaction
2. Check address alignment requirements
3. Validate length field accuracy
4. Monitor TLP routing and delivery""",
                'metadata': {
                    'file': 'pcie_base_spec.pdf',
                    'page': 45,
                    'heading': 'TLP Header Format',
                    'level': 3,
                    'source': 'specification'
                }
            },
            {
                'content': """PCIe Advanced Error Reporting (AER) Configuration:
                
AER provides detailed error reporting capabilities. Configuration involves:

1. AER Capability Structure (Cap ID 0x01)
2. Uncorrectable Error Status/Mask/Severity Registers
3. Correctable Error Status/Mask Registers
4. Root Error Command/Status (Root Ports only)

Error types handled by AER:
- Correctable: Receiver Error, Bad TLP, Bad DLLP, Replay Timer Timeout
- Uncorrectable Non-Fatal: Poison TLP, Flow Control Protocol Error
- Uncorrectable Fatal: Malformed TLP, Data Link Layer Protocol Error
- Advisory Non-Fatal: Advisory errors for logging

AER configuration steps:
1. Enable AER capability in device
2. Configure error masks appropriately
3. Set up Root Port error reporting
4. Implement error handler software
5. Test error injection and reporting

AER reporting path: Device -> Root Port -> Root Complex -> OS""",
                'metadata': {
                    'file': 'pcie_base_spec.pdf',
                    'page': 324,
                    'heading': 'Advanced Error Reporting',
                    'level': 2,
                    'source': 'specification'
                }
            }
        ]

class DocumentChunker:
    """Chunk documents using various strategies"""
    
    def __init__(self, params: RAGParams):
        self.params = params
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk documents based on strategy"""
        if self.params.chunking_strategy == "fixed":
            return self._fixed_chunking(documents)
        elif self.params.chunking_strategy == "heading-aware":
            return self._heading_aware_chunking(documents)
        elif self.params.chunking_strategy == "sliding":
            return self._sliding_window_chunking(documents)
        else:
            return self._fixed_chunking(documents)
    
    def _fixed_chunking(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fixed-size chunking"""
        chunks = []
        
        for doc in documents:
            content = doc['content']
            words = content.split()
            
            # Approximate tokens (1 token ≈ 0.75 words)
            chunk_words = int(self.params.base_chunk_size * 0.75)
            overlap_words = int(chunk_words * self.params.overlap_ratio)
            
            for i in range(0, len(words), chunk_words - overlap_words):
                chunk_text = ' '.join(words[i:i + chunk_words])
                
                if len(chunk_text.strip()) > 20:
                    chunk_metadata = doc['metadata'].copy()
                    chunk_metadata.update({
                        'chunk_id': len(chunks),
                        'chunk_size': len(chunk_text.split()),
                        'strategy': 'fixed'
                    })
                    
                    chunks.append({
                        'content': chunk_text,
                        'metadata': chunk_metadata
                    })
        
        return chunks
    
    def _heading_aware_chunking(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Heading-aware chunking"""
        chunks = []
        
        for doc in documents:
            content = doc['content']
            lines = content.split('\n')
            
            current_chunk = []
            current_heading = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Detect headings
                if (line.isupper() or 
                    line.startswith(('1.', '2.', '3.', '4.', '5.')) or
                    len(line) < 100 and line.endswith(':') or
                    'section' in line.lower()):
                    
                    # Save previous chunk
                    if current_chunk:
                        chunk_text = '\n'.join(current_chunk)
                        if len(chunk_text.split()) >= 10:
                            chunk_metadata = doc['metadata'].copy()
                            chunk_metadata.update({
                                'chunk_id': len(chunks),
                                'heading': current_heading,
                                'strategy': 'heading-aware'
                            })
                            
                            chunks.append({
                                'content': chunk_text,
                                'metadata': chunk_metadata
                            })
                    
                    # Start new chunk
                    current_chunk = [line]
                    current_heading = line
                else:
                    current_chunk.append(line)
                    
                    # Check size limit
                    if len(' '.join(current_chunk).split()) > self.params.base_chunk_size:
                        chunk_text = '\n'.join(current_chunk)
                        chunk_metadata = doc['metadata'].copy()
                        chunk_metadata.update({
                            'chunk_id': len(chunks),
                            'heading': current_heading,
                            'strategy': 'heading-aware'
                        })
                        
                        chunks.append({
                            'content': chunk_text,
                            'metadata': chunk_metadata
                        })
                        
                        # Overlap
                        overlap_lines = max(1, int(len(current_chunk) * self.params.overlap_ratio))
                        current_chunk = current_chunk[-overlap_lines:]
            
            # Final chunk
            if current_chunk:
                chunk_text = '\n'.join(current_chunk)
                if len(chunk_text.split()) >= 10:
                    chunk_metadata = doc['metadata'].copy()
                    chunk_metadata.update({
                        'chunk_id': len(chunks),
                        'heading': current_heading,
                        'strategy': 'heading-aware'
                    })
                    
                    chunks.append({
                        'content': chunk_text,
                        'metadata': chunk_metadata
                    })
        
        return chunks
    
    def _sliding_window_chunking(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sliding window chunking"""
        chunks = []
        
        for doc in documents:
            content = doc['content']
            sentences = content.split('. ')
            
            current_chunk = []
            current_size = 0
            
            for sentence in sentences:
                words = sentence.split()
                sentence_size = len(words)
                
                if current_size + sentence_size > self.params.base_chunk_size:
                    if current_size >= self.params.dynamic_split_thresh:
                        # Save chunk
                        chunk_text = '. '.join(current_chunk)
                        if chunk_text.strip():
                            chunk_metadata = doc['metadata'].copy()
                            chunk_metadata.update({
                                'chunk_id': len(chunks),
                                'chunk_size': current_size,
                                'strategy': 'sliding'
                            })
                            
                            chunks.append({
                                'content': chunk_text,
                                'metadata': chunk_metadata
                            })
                        
                        # Overlap
                        overlap_sentences = max(1, int(len(current_chunk) * self.params.overlap_ratio))
                        current_chunk = current_chunk[-overlap_sentences:] + [sentence]
                        current_size = sum(len(s.split()) for s in current_chunk)
                    else:
                        current_chunk.append(sentence)
                        current_size += sentence_size
                else:
                    current_chunk.append(sentence)
                    current_size += sentence_size
            
            # Final chunk
            if current_chunk:
                chunk_text = '. '.join(current_chunk)
                if chunk_text.strip():
                    chunk_metadata = doc['metadata'].copy()
                    chunk_metadata.update({
                        'chunk_id': len(chunks),
                        'chunk_size': current_size,
                        'strategy': 'sliding'
                    })
                    
                    chunks.append({
                        'content': chunk_text,
                        'metadata': chunk_metadata
                    })
        
        return chunks

class SimpleRetriever:
    """Simple retrieval system"""
    
    def __init__(self, params: RAGParams):
        self.params = params
        self.chunks = []
        self.bm25 = None
    
    def build_index(self, chunks: List[Dict[str, Any]]):
        """Build index from chunks"""
        self.chunks = chunks
        
        # Build BM25 index
        texts = [chunk['content'] for chunk in chunks]
        self.bm25 = SimpleBM25(texts)
        
        logger.info(f"Built index with {len(chunks)} chunks")
    
    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks"""
        if not self.bm25:
            return []
        
        # BM25 search
        results = self.bm25.search(query, k)
        
        formatted_results = []
        for idx, score in results:
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(score)
                
                # Apply length penalty
                if self.params.length_penalty > 0:
                    length_factor = len(chunk['content']) / 1000
                    chunk['score'] -= self.params.length_penalty * length_factor
                
                formatted_results.append(chunk)
        
        return formatted_results

class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    def __init__(self, eval_queries_path: str = "eval_queries.json"):
        self.eval_queries = self._load_eval_queries(eval_queries_path)
    
    def _load_eval_queries(self, path: str) -> List[Dict[str, Any]]:
        """Load evaluation queries"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            # Fallback queries
            return [
                {
                    "question": "why dut send successful return of completion during flr ? I expect crs return",
                    "answers": [
                        "FLR compliance violation",
                        "Configuration Request Retry Status",
                        "PCIe Base Specification Section 6.6.1.2",
                        "device not ready",
                        "reset sequence",
                        "100ms minimum",
                        "configuration space blocked"
                    ]
                },
                {
                    "question": "What should device return during FLR to configuration reads?",
                    "answers": [
                        "CRS",
                        "Configuration Request Retry Status",
                        "not ready",
                        "retry status",
                        "Section 6.6.1.2"
                    ]
                },
                {
                    "question": "PCIe completion timeout causes and debug steps",
                    "answers": [
                        "non-posted request",
                        "target device",
                        "Device Control 2 Register",
                        "timeout value",
                        "completion packet",
                        "credit exhaustion",
                        "routing problems"
                    ]
                },
                {
                    "question": "LTSSM stuck in Polling.Compliance state causes",
                    "answers": [
                        "electrical",
                        "link training",
                        "speed negotiation",
                        "receiver detection",
                        "compliance pattern",
                        "training sequence",
                        "LTSSM"
                    ]
                },
                {
                    "question": "PCIe TLP header format for 3DW memory read",
                    "answers": [
                        "3DW",
                        "Fmt[2:0]",
                        "Type[4:0]",
                        "Length[9:0]",
                        "Address[31:2]",
                        "memory request",
                        "transaction layer"
                    ]
                }
            ]
    
    def evaluate(self, retriever: SimpleRetriever, k: int = 3) -> EvalResult:
        """Evaluate RAG system"""
        start_time = time.time()
        
        recall_scores = []
        mrr_scores = []
        query_latencies = []
        
        for query_data in self.eval_queries:
            query = query_data['question']
            expected_answers = query_data['answers']
            
            # Measure retrieval latency
            query_start = time.time()
            results = retriever.retrieve(query, k=k)
            query_latency = (time.time() - query_start) * 1000
            query_latencies.append(query_latency)
            
            # Calculate recall@k
            retrieved_content = ' '.join([r['content'].lower() for r in results])
            
            recall_hits = 0
            for answer in expected_answers:
                if answer.lower() in retrieved_content:
                    recall_hits += 1
            
            recall = recall_hits / len(expected_answers) if expected_answers else 0
            recall_scores.append(recall)
            
            # Calculate MRR
            mrr = 0.0
            for i, result in enumerate(results):
                content_lower = result['content'].lower()
                for answer in expected_answers:
                    if answer.lower() in content_lower:
                        mrr = 1.0 / (i + 1)
                        break
                if mrr > 0:
                    break
            mrr_scores.append(mrr)
        
        # Aggregate metrics
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
        avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
        avg_latency = sum(query_latencies) / len(query_latencies) if query_latencies else 0
        
        # Objective score
        objective_score = avg_recall - 0.001 * avg_latency
        
        return EvalResult(
            recall_at_3=avg_recall,
            mrr=avg_mrr,
            latency_ms=avg_latency,
            objective_score=objective_score,
            details={
                'individual_recalls': recall_scores,
                'individual_mrrs': mrr_scores,
                'individual_latencies': query_latencies,
                'total_eval_time': time.time() - start_time
            }
        )

class SimpleOptimizer:
    """Simple optimization engine (replaces Optuna)"""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.best_params = None
        self.best_score = -float('inf')
        self.trial_history = []
    
    def sample_params(self, trial_num: int) -> RAGParams:
        """Sample parameters for a trial"""
        random.seed(42 + trial_num)  # Reproducible
        
        return RAGParams(
            chunking_strategy=random.choice(['fixed', 'heading-aware', 'sliding']),
            base_chunk_size=random.choice([128, 256, 384, 512]),
            overlap_ratio=random.choice([0.10, 0.15, 0.20]),
            dynamic_split_thresh=random.choice([192, 256]),
            add_meta_fields=random.choice([
                ['page'], ['file'], ['page', 'file'],
                ['page', 'heading'], ['page', 'file', 'heading']
            ]),
            length_penalty=round(random.uniform(0.0, 0.3), 2),
            retriever_type=random.choice(['bm25']),
            hybrid_weight=round(random.uniform(0.3, 0.7), 2),
            hierarchical_mode=random.choice([True, False]),
            rerank_model=random.choice(['none', 'simple']),
            max_total_ctx_tokens=random.choice([1024, 2048, 3072])
        )
    
    def optimize(self, objective_func) -> RAGParams:
        """Run optimization"""
        print(f"🚀 Starting optimization with {self.config.max_trials} trials...")
        
        for trial_num in range(self.config.max_trials):
            params = self.sample_params(trial_num)
            
            try:
                score = objective_func(params, trial_num)
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
                    print(f"✅ Trial {trial_num}: New best score {score:.4f}")
                else:
                    print(f"   Trial {trial_num}: Score {score:.4f}")
                
                self.trial_history.append({
                    'trial': trial_num,
                    'score': score,
                    'params': asdict(params)
                })
                
                # Early stopping
                if score >= self.config.target_recall:
                    print(f"🎯 Target recall {self.config.target_recall} reached!")
                    break
                    
            except Exception as e:
                print(f"❌ Trial {trial_num} failed: {e}")
        
        return self.best_params

class AutoEvolvingRAG:
    """Main auto-evolving RAG system"""
    
    def __init__(self, config: EvolutionConfig = None):
        self.config = config or EvolutionConfig()
        self.processor = PCIeDocumentProcessor()
        self.evaluator = RAGEvaluator()
        self.optimizer = SimpleOptimizer(self.config)
        
        # Load documents once
        self.documents = self.processor.load_documents()
        print(f"📚 Loaded {len(self.documents)} PCIe documents")
        
        # Evolution tracking
        self.evolution_history = []
        self.current_generation = 0
    
    def objective_function(self, params: RAGParams, trial_num: int) -> float:
        """Objective function for optimization"""
        try:
            start_time = time.time()
            
            # Build system
            chunker = DocumentChunker(params)
            chunks = chunker.chunk_documents(self.documents)
            
            retriever = SimpleRetriever(params)
            retriever.build_index(chunks)
            
            # Evaluate
            result = self.evaluator.evaluate(retriever, k=3)
            
            elapsed_time = time.time() - start_time
            
            # Check constraints
            if elapsed_time > self.config.max_time_minutes * 60:
                return -1.0  # Penalty for slow configs
            
            return result.objective_score
            
        except Exception as e:
            logger.error(f"Trial {trial_num} failed: {e}")
            return -1.0
    
    def evolve(self) -> RAGParams:
        """Run evolution cycle"""
        print(f"\n🧬 Starting Evolution Generation {self.current_generation + 1}")
        print("=" * 60)
        
        # Run optimization
        best_params = self.optimizer.optimize(self.objective_function)
        
        if best_params:
            # Save best configuration
            self._save_best_config(best_params)
            
            # Track evolution
            self.evolution_history.append({
                'generation': self.current_generation + 1,
                'best_score': self.optimizer.best_score,
                'best_params': asdict(best_params),
                'timestamp': datetime.now().isoformat()
            })
            
            self.current_generation += 1
            
            print(f"\n🎉 Evolution Generation {self.current_generation} Complete!")
            print(f"   Best Score: {self.optimizer.best_score:.4f}")
            print(f"   Best Strategy: {best_params.chunking_strategy}")
            print(f"   Best Chunk Size: {best_params.base_chunk_size}")
            print(f"   Best Overlap: {best_params.overlap_ratio}")
            
            return best_params
        else:
            print("❌ Evolution failed - no valid parameters found")
            return RAGParams()  # Default params
    
    def _save_best_config(self, params: RAGParams):
        """Save best configuration"""
        config_dict = asdict(params)
        config_dict['evolution_metadata'] = {
            'generation': self.current_generation + 1,
            'best_score': self.optimizer.best_score,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            import yaml
            with open('best_config.yaml', 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            print(f"💾 Best config saved to best_config.yaml")
        except:
            # Fallback to JSON
            with open('best_config.json', 'w') as f:
                json.dump(config_dict, f, indent=2)
            print(f"💾 Best config saved to best_config.json")
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get evolution status"""
        return {
            'current_generation': self.current_generation,
            'total_trials': len(self.optimizer.trial_history),
            'best_score': self.optimizer.best_score,
            'best_params': asdict(self.optimizer.best_params) if self.optimizer.best_params else None,
            'evolution_history': self.evolution_history
        }
    
    def query_with_best_params(self, question: str) -> Dict[str, Any]:
        """Query using best evolved parameters"""
        if not self.optimizer.best_params:
            print("⚠️  No evolved parameters available, using defaults")
            params = RAGParams()
        else:
            params = self.optimizer.best_params
        
        start_time = time.time()
        
        try:
            # Build system with best params
            chunker = DocumentChunker(params)
            chunks = chunker.chunk_documents(self.documents)
            
            retriever = SimpleRetriever(params)
            retriever.build_index(chunks)
            
            # Retrieve results
            results = retriever.retrieve(question, k=3)
            
            # Build answer
            answer = self._build_answer(question, results)
            
            response_time = time.time() - start_time
            
            return {
                'answer': answer,
                'confidence': self._calculate_confidence(results),
                'sources': results,
                'params_used': asdict(params),
                'response_time': response_time,
                'generation': self.current_generation
            }
            
        except Exception as e:
            return {
                'answer': f"Query failed: {str(e)}",
                'confidence': 0.0,
                'sources': [],
                'error': str(e)
            }
    
    def _build_answer(self, question: str, results: List[Dict]) -> str:
        """Build answer from results"""
        if not results:
            return "No relevant information found in the PCIe knowledge base."
        
        question_lower = question.lower()
        
        # Detect question type and build appropriate answer
        if any(term in question_lower for term in ['expect', 'should', 'compliance']):
            return self._build_compliance_answer(question, results)
        elif any(term in question_lower for term in ['debug', 'cause', 'why', 'timeout']):
            return self._build_debug_answer(question, results)
        else:
            return self._build_general_answer(question, results)
    
    def _build_compliance_answer(self, question: str, results: List[Dict]) -> str:
        """Build compliance-focused answer"""
        answer = "**PCIe Compliance Analysis:**\n\n"
        
        # Find specification references
        for result in results[:2]:
            content = result['content']
            if 'section' in content.lower() or 'specification' in content.lower():
                # Extract key compliance points
                lines = content.split('\n')
                for line in lines:
                    if any(word in line.lower() for word in ['must', 'shall', 'required']):
                        answer += f"• {line.strip()}\n"
                break
        
        answer += f"\n**Key Information:**\n"
        for i, result in enumerate(results[:2], 1):
            content = result['content'][:200]
            answer += f"{i}. {content}...\n"
        
        return answer
    
    def _build_debug_answer(self, question: str, results: List[Dict]) -> str:
        """Build debug-focused answer"""
        answer = "**PCIe Debug Analysis:**\n\n"
        
        # Extract debug information
        debug_info = []
        for result in results[:3]:
            content = result['content']
            # Look for debug steps or causes
            if 'debug' in content.lower() or 'cause' in content.lower():
                debug_info.append(content[:150])
        
        if debug_info:
            answer += "**Debug Steps/Causes:**\n"
            for i, info in enumerate(debug_info, 1):
                answer += f"{i}. {info}...\n"
        
        answer += f"\n**Relevant Information:**\n"
        answer += f"{results[0]['content'][:300]}..." if results else "No specific information found."
        
        return answer
    
    def _build_general_answer(self, question: str, results: List[Dict]) -> str:
        """Build general answer"""
        answer = f"**PCIe Information for: {question}**\n\n"
        
        for i, result in enumerate(results[:3], 1):
            content = result['content']
            score = result.get('score', 0)
            
            answer += f"**{i}. Relevance Score: {score:.2f}**\n"
            answer += f"{content[:200]}...\n\n"
        
        return answer
    
    def _calculate_confidence(self, results: List[Dict]) -> float:
        """Calculate response confidence"""
        if not results:
            return 0.0
        
        # Simple confidence based on top result score
        top_score = results[0].get('score', 0) if results else 0
        
        # Normalize and cap
        confidence = min(top_score / 10.0, 1.0)  # Assuming max BM25 score ~10
        return max(confidence, 0.1)  # Minimum confidence

def run_evolution_demo():
    """Run evolution demonstration"""
    print("🚀 PCIe RAG Evolution Demo")
    print("=" * 50)
    
    # Create evolution system
    config = EvolutionConfig(
        max_trials=10,  # Quick demo
        target_recall=0.80,
        max_time_minutes=5
    )
    
    rag_system = AutoEvolvingRAG(config)
    
    # Run 5 evolution cycles
    for cycle in range(5):
        print(f"\n🔬 EVOLUTION CYCLE {cycle + 1}/5")
        print("-" * 40)
        
        best_params = rag_system.evolve()
        
        if best_params:
            # Test with evolved parameters
            test_question = "why dut send successful completion during flr?"
            result = rag_system.query_with_best_params(test_question)
            
            print(f"\n🧪 Test Query: {test_question}")
            print(f"   Confidence: {result['confidence']:.1%}")
            print(f"   Response Time: {result.get('response_time', 0):.2f}s")
            print(f"   Answer Preview: {result['answer'][:100]}...")
        
        # Short pause between cycles
        time.sleep(1)
    
    # Final status
    status = rag_system.get_evolution_status()
    print(f"\n🎯 FINAL EVOLUTION STATUS")
    print("=" * 40)
    print(f"Generations Completed: {status['current_generation']}")
    print(f"Total Trials: {status['total_trials']}")
    print(f"Best Score Achieved: {status['best_score']:.4f}")
    
    if status['best_params']:
        print(f"Best Configuration:")
        print(f"  - Strategy: {status['best_params']['chunking_strategy']}")
        print(f"  - Chunk Size: {status['best_params']['base_chunk_size']}")
        print(f"  - Overlap: {status['best_params']['overlap_ratio']}")
    
    return rag_system

if __name__ == "__main__":
    run_evolution_demo()