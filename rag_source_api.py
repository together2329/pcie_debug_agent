#!/usr/bin/env python3
"""
RAG Source API - Get detailed source information for queries
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GGML_METAL_LOG_LEVEL"] = "0"

from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.rag.vector_store import FAISSVectorStore
from src.config.settings import load_settings
from sentence_transformers import SentenceTransformer
import numpy as np

class RAGSourceAPI:
    """API for retrieving and analyzing sources"""
    
    def __init__(self):
        settings = load_settings()
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        vector_store_path = Path("data/vectorstore")
        self.vector_store = FAISSVectorStore(
            index_path=str(vector_store_path),
            dimension=384  # MiniLM-L6-v2 dimension
        )
        
        # Load knowledge base metadata
        self.kb_metadata = self._load_kb_metadata()
    
    def _load_kb_metadata(self) -> Dict[str, Any]:
        """Load knowledge base file information"""
        kb_path = Path("data/knowledge_base")
        metadata = {}
        
        if kb_path.exists():
            for file_path in kb_path.glob("*.md"):
                with open(file_path, 'r') as f:
                    content = f.read()
                    metadata[file_path.name] = {
                        'path': str(file_path),
                        'size': len(content),
                        'lines': content.count('\n'),
                        'sections': self._extract_sections(content)
                    }
        
        return metadata
    
    def _extract_sections(self, content: str) -> List[str]:
        """Extract section headers from markdown"""
        sections = []
        for line in content.split('\n'):
            if line.startswith('#'):
                sections.append(line.strip('#').strip())
        return sections
    
    def search_sources(self, query: str, k: int = 10, threshold: float = 0.5) -> Dict[str, Any]:
        """Search for relevant sources with detailed information"""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k=k)
        
        # Process and enrich results
        enriched_results = []
        source_distribution = {}
        
        for i, result in enumerate(results):
            score = result.get('score', 0)
            if score < threshold:
                continue
                
            metadata = result.get('metadata', {})
            content = result.get('content', result.get('text', ''))
            
            # Extract source information
            source_file = metadata.get('source', 'Unknown')
            chunk_id = metadata.get('chunk_id', f'chunk_{i}')
            
            # Count source distribution
            if source_file not in source_distribution:
                source_distribution[source_file] = 0
            source_distribution[source_file] += 1
            
            # Analyze content relevance
            content_analysis = self._analyze_content_relevance(query, content)
            
            enriched_result = {
                'rank': i + 1,
                'score': float(score),
                'source': {
                    'file': source_file,
                    'chunk_id': chunk_id,
                    'kb_info': self.kb_metadata.get(source_file, {})
                },
                'content': {
                    'text': content,
                    'length': len(content),
                    'preview': content[:200] + '...' if len(content) > 200 else content
                },
                'relevance_analysis': content_analysis,
                'metadata': metadata
            }
            
            enriched_results.append(enriched_result)
        
        # Calculate statistics
        statistics = {
            'total_results': len(enriched_results),
            'avg_score': sum(r['score'] for r in enriched_results) / len(enriched_results) if enriched_results else 0,
            'source_distribution': source_distribution,
            'unique_sources': len(source_distribution)
        }
        
        return {
            'query': query,
            'results': enriched_results,
            'statistics': statistics
        }
    
    def _analyze_content_relevance(self, query: str, content: str) -> Dict[str, Any]:
        """Analyze how content is relevant to query"""
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Extract key terms from query
        query_terms = [term for term in query_lower.split() if len(term) > 3]
        
        # Find matching terms
        matching_terms = []
        term_positions = {}
        
        for term in query_terms:
            if term in content_lower:
                matching_terms.append(term)
                # Find position
                pos = content_lower.find(term)
                term_positions[term] = pos
        
        # Identify key PCIe concepts
        pcie_concepts = {
            'link_training': ['link training', 'ltssm', 'training'],
            'errors': ['error', 'failure', 'timeout', 'invalid'],
            'tlp': ['tlp', 'transaction', 'packet', 'completion'],
            'power': ['power', 'aspm', 'l0s', 'l1', 'd3'],
            'signal': ['signal', 'integrity', 'eye', 'jitter'],
            'aer': ['aer', 'advanced error', 'reporting']
        }
        
        found_concepts = []
        for concept, keywords in pcie_concepts.items():
            if any(kw in content_lower for kw in keywords):
                found_concepts.append(concept)
        
        return {
            'matching_query_terms': matching_terms,
            'match_percentage': len(matching_terms) / len(query_terms) * 100 if query_terms else 0,
            'term_positions': term_positions,
            'pcie_concepts_found': found_concepts,
            'relevance_indicators': {
                'has_error_context': any(word in content_lower for word in ['error', 'fail', 'issue']),
                'has_solution': any(word in content_lower for word in ['fix', 'solve', 'resolution', 'check']),
                'has_explanation': any(word in content_lower for word in ['because', 'due to', 'caused by'])
            }
        }
    
    def get_source_context(self, source_file: str, chunk_id: str = None) -> Dict[str, Any]:
        """Get full context for a specific source"""
        # Search for all chunks from this source
        all_results = self.vector_store.search(
            self.embedding_model.encode(""),  # Empty query to get all
            k=1000  # Get many results
        )
        
        # Filter by source
        source_chunks = []
        for result in all_results:
            metadata = result.get('metadata', {})
            if metadata.get('source') == source_file:
                if chunk_id is None or metadata.get('chunk_id') == chunk_id:
                    source_chunks.append({
                        'chunk_id': metadata.get('chunk_id'),
                        'content': result.get('content', ''),
                        'position': metadata.get('chunk_index', 0)
                    })
        
        # Sort by position
        source_chunks.sort(key=lambda x: x.get('position', 0))
        
        return {
            'source_file': source_file,
            'kb_info': self.kb_metadata.get(source_file, {}),
            'total_chunks': len(source_chunks),
            'chunks': source_chunks
        }
    
    def explain_source_selection(self, query: str, top_k: int = 5) -> str:
        """Generate human-readable explanation of source selection"""
        results = self.search_sources(query, k=top_k)
        
        explanation = f"🔍 Source Selection for: '{query}'\n"
        explanation += "="*60 + "\n\n"
        
        explanation += f"📊 Found {results['statistics']['total_results']} relevant sources\n"
        explanation += f"   Average relevance score: {results['statistics']['avg_score']:.3f}\n\n"
        
        explanation += "📚 Top Sources Selected:\n"
        for result in results['results'][:top_k]:
            explanation += f"\n{result['rank']}. {result['source']['file']} "
            explanation += f"(Score: {result['score']:.3f})\n"
            
            relevance = result['relevance_analysis']
            explanation += f"   Why selected:\n"
            explanation += f"   - Query term matches: {relevance['match_percentage']:.0f}%\n"
            explanation += f"   - Matching terms: {', '.join(relevance['matching_query_terms'])}\n"
            explanation += f"   - PCIe concepts: {', '.join(relevance['pcie_concepts_found'])}\n"
            
            if relevance['relevance_indicators']['has_solution']:
                explanation += "   - ✅ Contains solution/fix information\n"
            if relevance['relevance_indicators']['has_error_context']:
                explanation += "   - ✅ Discusses error scenarios\n"
            if relevance['relevance_indicators']['has_explanation']:
                explanation += "   - ✅ Provides explanations\n"
        
        explanation += "\n" + "="*60 + "\n"
        explanation += "💡 These sources were selected because they contain the most\n"
        explanation += "   relevant information to answer your specific query.\n"
        
        return explanation

# Convenience functions for direct use
def search_pcie_sources(query: str, detailed: bool = True) -> Dict[str, Any]:
    """Quick function to search PCIe sources"""
    api = RAGSourceAPI()
    results = api.search_sources(query)
    
    if detailed:
        print(api.explain_source_selection(query))
    
    return results

def get_source_info(source_file: str) -> Dict[str, Any]:
    """Get information about a specific source file"""
    api = RAGSourceAPI()
    return api.get_source_context(source_file)

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Searching for: {query}\n")
        results = search_pcie_sources(query, detailed=True)
        
        # Save results
        with open('source_search_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\n💾 Detailed results saved to: source_search_results.json")
    else:
        print("Usage: python rag_source_api.py <your query>")
        print("Example: python rag_source_api.py Why is link training failing")