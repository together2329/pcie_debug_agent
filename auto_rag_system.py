#!/usr/bin/env python3
"""
AutoRAG-Agent: Self-Evolving Context System for PCIe-Spec RAG (v0.1)

A closed-loop pipeline that:
1. Builds a RAG system for PCIe specifications
2. Evaluates on fixed Q-A sets  
3. Mutates hyper-parameters & design choices
4. Keeps the best model and repeats
"""

import json
import time
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import os
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvolutionConfig:
    """Configuration for evolution process"""
    max_trials: int = 50
    target_recall: float = 0.95
    max_time_minutes: int = 15
    max_memory_mb: int = 512
    study_name: str = "pcie_rag_evolution"
    storage_url: Optional[str] = None

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
    retriever_type: str = "faiss_flat"  # {faiss_flat, faiss_hnsw}
    hybrid_weight: float = 0.5  # [0.3 - 0.7] BM25 vs embed
    hierarchical_mode: bool = False
    
    # Ranking parameters
    rerank_model: str = "none"  # {none, "cross-encoder/ms-marco-MiniLM-L-6"}
    
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

class PCIeDocumentProcessor:
    """Process PCIe specification documents"""
    
    def __init__(self, specs_dir: str = "./specs"):
        self.specs_dir = Path(specs_dir)
        self.documents = []
        self.metadata = []
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """Load PCIe specification documents"""
        if not self.specs_dir.exists():
            logger.warning(f"Specs directory {self.specs_dir} not found")
            return self._load_fallback_docs()
        
        documents = []
        for pdf_file in self.specs_dir.glob("*.pdf"):
            try:
                docs = self._extract_from_pdf(pdf_file)
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
        
        if not documents:
            return self._load_fallback_docs()
        
        return documents
    
    def _extract_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract text from PDF with metadata"""
        try:
            import PyMuPDF as fitz
            
            doc = fitz.open(pdf_path)
            documents = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if len(text.strip()) > 50:  # Skip mostly empty pages
                    documents.append({
                        'content': text,
                        'metadata': {
                            'file': pdf_path.name,
                            'page': page_num + 1,
                            'source': 'pdf'
                        }
                    })
            
            doc.close()
            return documents
            
        except ImportError:
            logger.warning("PyMuPDF not available, using fallback documents")
            return self._load_fallback_docs()
    
    def _load_fallback_docs(self) -> List[Dict[str, Any]]:
        """Fallback PCIe documentation when PDFs not available"""
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
- Configuration space not properly gated during reset""",
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
- BAR programming errors""",
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
- Link width reduction due to lane failures""",
                'metadata': {
                    'file': 'pcie_base_spec.pdf',
                    'page': 156,
                    'heading': 'LTSSM State Machine',
                    'level': 2,
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

AER reporting path: Device -> Root Port -> Root Complex -> OS""",
                'metadata': {
                    'file': 'pcie_base_spec.pdf',
                    'page': 324,
                    'heading': 'Advanced Error Reporting',
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
- Completion (Type = 01010)""",
                'metadata': {
                    'file': 'pcie_base_spec.pdf',
                    'page': 45,
                    'heading': 'TLP Header Format',
                    'level': 3,
                    'source': 'specification'
                }
            },
            {
                'content': """PCIe Power Management and ASPM:
                
Active State Power Management (ASPM) provides link-level power savings:

L0s (Low-power active state):
- Enter when link is idle for configured period
- Quick entry/exit (μs range)
- Transmitter enters electrical idle

L1 (Link power management state):
- Deeper power savings than L0s
- Both transmitter and receiver enter low power
- Longer entry/exit latency (ms range)
- Requires handshake between link partners

Configuration:
- Link Capabilities Register reports L0s/L1 support
- Link Control Register enables ASPM L0s/L1
- ASPM must be supported by both link partners""",
                'metadata': {
                    'file': 'pcie_base_spec.pdf',
                    'page': 278,
                    'heading': 'ASPM Configuration',
                    'level': 3,
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
            raise ValueError(f"Unknown chunking strategy: {self.params.chunking_strategy}")
    
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
                
                if len(chunk_text.strip()) > 20:  # Skip very short chunks
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
        """Heading-aware chunking that respects document structure"""
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
                
                # Detect headings (simple heuristic)
                if (line.isupper() or 
                    line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) or
                    len(line) < 100 and line.endswith(':') or
                    'section' in line.lower() or 'chapter' in line.lower()):
                    
                    # Save previous chunk
                    if current_chunk:
                        chunk_text = '\n'.join(current_chunk)
                        if len(chunk_text.split()) >= 10:  # Minimum chunk size
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
                    
                    # Check if chunk is getting too large
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
                        
                        # Start overlap
                        overlap_lines = max(1, int(len(current_chunk) * self.params.overlap_ratio))
                        current_chunk = current_chunk[-overlap_lines:]
            
            # Save final chunk
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
        """Sliding window chunking with dynamic splits"""
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
                    # Dynamic split decision
                    if current_size >= self.params.dynamic_split_thresh:
                        # Save current chunk
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
                        
                        # Start new chunk with overlap
                        overlap_sentences = max(1, int(len(current_chunk) * self.params.overlap_ratio))
                        current_chunk = current_chunk[-overlap_sentences:] + [sentence]
                        current_size = sum(len(s.split()) for s in current_chunk)
                    else:
                        # Continue current chunk
                        current_chunk.append(sentence)
                        current_size += sentence_size
                else:
                    current_chunk.append(sentence)
                    current_size += sentence_size
            
            # Save final chunk
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

class EmbeddingRetriever:
    """FAISS-based retrieval system"""
    
    def __init__(self, params: RAGParams):
        self.params = params
        self.model_name = os.getenv('EMBED_MODEL', 'all-MiniLM-L6-v2')
        self.model = SentenceTransformer(self.model_name)
        self.index = None
        self.chunks = []
        self.embeddings = None
    
    def build_index(self, chunks: List[Dict[str, Any]]):
        """Build FAISS index from chunks"""
        self.chunks = chunks
        
        # Generate embeddings
        texts = [chunk['content'] for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        
        if self.params.retriever_type == "faiss_flat":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        elif self.params.retriever_type == "faiss_hnsw":
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # HNSW with M=32
        else:
            raise ValueError(f"Unknown retriever type: {self.params.retriever_type}")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        logger.info(f"Built {self.params.retriever_type} index with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for query"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(score)
                
                # Apply length penalty if configured
                if self.params.length_penalty > 0:
                    length_factor = len(chunk['content']) / 1000  # Normalize by 1000 chars
                    chunk['score'] -= self.params.length_penalty * length_factor
                
                results.append(chunk)
        
        return results

class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    def __init__(self, eval_queries_path: str = "eval_queries.json"):
        self.eval_queries = self._load_eval_queries(eval_queries_path)
    
    def _load_eval_queries(self, path: str) -> List[Dict[str, Any]]:
        """Load evaluation queries"""
        eval_path = Path(path)
        
        if eval_path.exists():
            with open(eval_path, 'r') as f:
                return json.load(f)
        else:
            # Fallback evaluation queries for PCIe
            logger.warning(f"Eval queries file {path} not found, using fallback queries")
            return [
                {
                    "question": "why dut send successful return of completion during flr ? I expect crs return",
                    "answers": [
                        "FLR compliance violation",
                        "Configuration Request Retry Status",
                        "PCIe Base Specification Section 6.6.1.2",
                        "device not ready",
                        "reset sequence"
                    ]
                },
                {
                    "question": "What should device return during FLR to configuration reads?",
                    "answers": [
                        "CRS",
                        "Configuration Request Retry Status",
                        "not ready",
                        "retry status"
                    ]
                },
                {
                    "question": "PCIe completion timeout causes and debug steps",
                    "answers": [
                        "non-posted request",
                        "target device",
                        "Device Control 2 Register",
                        "timeout value",
                        "completion packet"
                    ]
                },
                {
                    "question": "LTSSM stuck in Polling.Compliance state causes",
                    "answers": [
                        "electrical",
                        "link training",
                        "speed negotiation",
                        "receiver detection",
                        "compliance pattern"
                    ]
                },
                {
                    "question": "PCIe TLP header format for 3DW memory read",
                    "answers": [
                        "3DW",
                        "Fmt[2:0]",
                        "Type[4:0]",
                        "Length[9:0]",
                        "Address[31:2]"
                    ]
                },
                {
                    "question": "AER correctable error reporting configuration",
                    "answers": [
                        "AER capability",
                        "Correctable Error Status",
                        "mask register",
                        "error reporting",
                        "Root Port"
                    ]
                },
                {
                    "question": "PCIe ASPM L1 entry conditions and requirements", 
                    "answers": [
                        "L1",
                        "power management",
                        "link partners",
                        "Link Control Register",
                        "electrical idle"
                    ]
                }
            ]
    
    def evaluate(self, retriever: EmbeddingRetriever, k: int = 3) -> EvalResult:
        """Evaluate RAG system on all queries"""
        start_time = time.time()
        
        recall_scores = []
        mrr_scores = []
        query_latencies = []
        
        for query_data in tqdm(self.eval_queries, desc="Evaluating"):
            query = query_data['question']
            expected_answers = query_data['answers']
            
            # Measure retrieval latency
            query_start = time.time()
            results = retriever.retrieve(query, k=k)
            query_latency = (time.time() - query_start) * 1000  # ms
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
        avg_recall = np.mean(recall_scores)
        avg_mrr = np.mean(mrr_scores)
        avg_latency = np.mean(query_latencies)
        
        # Calculate objective score (recall@3 - 0.001*latency_ms)
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

class AutoRAGAgent:
    """Main auto-evolving RAG agent"""
    
    def __init__(self, config: EvolutionConfig = None):
        self.config = config or EvolutionConfig()
        self.processor = PCIeDocumentProcessor()
        self.evaluator = RAGEvaluator()
        self.best_params = None
        self.best_score = 0.0
        
        # Load documents once
        self.documents = self.processor.load_documents()
        logger.info(f"Loaded {len(self.documents)} documents")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function"""
        try:
            # Sample parameters
            params = self._sample_params(trial)
            
            # Build and evaluate RAG system
            start_time = time.time()
            
            # Chunk documents
            chunker = DocumentChunker(params)
            chunks = chunker.chunk_documents(self.documents)
            
            # Build retriever
            retriever = EmbeddingRetriever(params)
            retriever.build_index(chunks)
            
            # Evaluate
            result = self.evaluator.evaluate(retriever, k=3)
            
            # Check time/memory constraints
            elapsed_time = time.time() - start_time
            if elapsed_time > self.config.max_time_minutes * 60:
                logger.warning(f"Trial exceeded time limit: {elapsed_time:.1f}s")
                return -1.0  # Penalty for slow configurations
            
            # Store additional metrics
            trial.set_user_attr('recall_at_3', result.recall_at_3)
            trial.set_user_attr('mrr', result.mrr)
            trial.set_user_attr('latency_ms', result.latency_ms)
            trial.set_user_attr('num_chunks', len(chunks))
            trial.set_user_attr('eval_time', elapsed_time)
            
            logger.info(f"Trial {trial.number}: Recall@3={result.recall_at_3:.3f}, "
                       f"MRR={result.mrr:.3f}, Latency={result.latency_ms:.1f}ms, "
                       f"Objective={result.objective_score:.3f}")
            
            return result.objective_score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return -1.0
    
    def _sample_params(self, trial: optuna.Trial) -> RAGParams:
        """Sample parameters for a trial"""
        return RAGParams(
            # Chunking
            chunking_strategy=trial.suggest_categorical(
                'chunking_strategy', ['fixed', 'heading-aware', 'sliding']
            ),
            base_chunk_size=trial.suggest_categorical(
                'base_chunk_size', [128, 256, 384, 512]
            ),
            overlap_ratio=trial.suggest_categorical(
                'overlap_ratio', [0.10, 0.15, 0.20]
            ),
            dynamic_split_thresh=trial.suggest_categorical(
                'dynamic_split_thresh', [192, 256]
            ),
            
            # Metadata
            add_meta_fields=trial.suggest_categorical(
                'add_meta_fields', [
                    ['page'], ['file'], ['page', 'file'], 
                    ['page', 'heading'], ['page', 'file', 'heading'],
                    ['page', 'file', 'heading', 'level']
                ]
            ),
            length_penalty=trial.suggest_float('length_penalty', 0.0, 0.3),
            
            # Retrieval
            retriever_type=trial.suggest_categorical(
                'retriever_type', ['faiss_flat', 'faiss_hnsw']
            ),
            hybrid_weight=trial.suggest_float('hybrid_weight', 0.3, 0.7),
            hierarchical_mode=trial.suggest_categorical('hierarchical_mode', [True, False]),
            
            # Ranking
            rerank_model=trial.suggest_categorical(
                'rerank_model', ['none', 'cross-encoder/ms-marco-MiniLM-L-6']
            ),
            
            # Context
            max_total_ctx_tokens=trial.suggest_categorical(
                'max_total_ctx_tokens', [1024, 2048, 3072]
            )
        )
    
    def evolve(self) -> RAGParams:
        """Run evolution process"""
        logger.info("🚀 Starting AutoRAG evolution process...")
        
        # Create Optuna study
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name=self.config.study_name,
            storage=self.config.storage_url
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.config.max_trials,
            timeout=self.config.max_time_minutes * 60
        )
        
        # Get best configuration
        best_trial = study.best_trial
        self.best_params = self._sample_params(best_trial)
        self.best_score = best_trial.value
        
        logger.info(f"🎯 Evolution complete! Best objective score: {self.best_score:.3f}")
        logger.info(f"   Recall@3: {best_trial.user_attrs.get('recall_at_3', 'N/A'):.3f}")
        logger.info(f"   MRR: {best_trial.user_attrs.get('mrr', 'N/A'):.3f}")
        logger.info(f"   Latency: {best_trial.user_attrs.get('latency_ms', 'N/A'):.1f}ms")
        
        # Save best configuration
        self._save_best_config()
        
        return self.best_params
    
    def _save_best_config(self):
        """Save best configuration to YAML"""
        if self.best_params:
            config_dict = asdict(self.best_params)
            config_dict['evolution_metadata'] = {
                'best_score': self.best_score,
                'timestamp': datetime.now().isoformat(),
                'num_documents': len(self.documents)
            }
            
            with open('best_config.yaml', 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info("💾 Best configuration saved to best_config.yaml")

def main():
    """Main execution function"""
    print("🤖 AutoRAG-Agent: Self-Evolving PCIe-Spec RAG System")
    print("=" * 60)
    
    # Initialize evolution config
    config = EvolutionConfig(
        max_trials=20,  # Reduced for demo
        target_recall=0.90,
        max_time_minutes=10
    )
    
    # Create and run AutoRAG agent
    agent = AutoRAGAgent(config)
    best_params = agent.evolve()
    
    print(f"\n🎉 Evolution completed!")
    print(f"Best configuration:")
    for key, value in asdict(best_params).items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()