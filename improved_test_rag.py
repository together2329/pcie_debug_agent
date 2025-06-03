#!/usr/bin/env python3
"""
Improved test for finding the specific multi-function error handling answer
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the classes from the previous test
exec(open('test_heading_aware_rag.py').read())

def search_for_target_content():
    """Search for the specific multi-function error handling content"""
    print("🎯 Searching for Multi-Function Error Handling Content")
    print("=" * 60)
    
    # Initialize components
    parser = SimplePDFParser()
    chunker = SimpleChunker()
    vector_store = SimpleVectorStore()
    
    # Process PDFs (focus on the most relevant ones first)
    pdf_dir = Path("data/specs")
    priority_pdfs = [
        "System Architecture.pdf",
        "PCIe 6.2 - 7 Chapter - Software.pdf", 
        "Transaction Layer.pdf"
    ]
    
    all_chunks = []
    
    for pdf_name in priority_pdfs:
        pdf_path = pdf_dir / pdf_name
        if pdf_path.exists():
            print(f"\n📖 Processing: {pdf_name}")
            sections = parser.parse_pdf(str(pdf_path))
            
            for section in sections:
                chunks = chunker.split_section(section)
                all_chunks.extend(chunks)
                vector_store.add_chunks(chunks)
    
    print(f"\n📊 Total chunks indexed: {len(all_chunks)}")
    
    # Try multiple search queries
    search_queries = [
        "how multi function handle error",
        "multi-function device error handling",
        "Multi-Function Device PCI Express errors",
        "Physical Layer errors Data Link Layer errors",
        "ECRC Check Failed Unsupported Request",
        "at most one error reporting Message",
        "Software is responsible for scanning all Functions"
    ]
    
    target_keywords = [
        "multi-function device",
        "physical layer errors",
        "data link layer errors", 
        "ecrc check failed",
        "unsupported request",
        "receiver overflow",
        "flow control protocol error",
        "malformed tlp",
        "unexpected completion",
        "at most one error",
        "software is responsible",
        "scanning all functions"
    ]
    
    best_matches = []
    
    for query in search_queries:
        print(f"\n🔍 Query: '{query}'")
        results = vector_store.search(query, k=15)
        
        for chunk, score in results:
            content_lower = chunk.text.lower()
            
            # Count keyword matches
            keyword_count = sum(1 for keyword in target_keywords if keyword in content_lower)
            
            if keyword_count >= 3:  # At least 3 keywords match
                print(f"   ✅ Strong match (score: {score:.4f}, keywords: {keyword_count})")
                print(f"   File: {Path(chunk.metadata['file_path']).name}")
                print(f"   Heading: {chunk.metadata['heading'][:80]}...")
                
                # Check if we've already found this chunk
                chunk_id = f"{chunk.metadata['file_path']}_{chunk.metadata['chunk_index']}"
                if chunk_id not in [m[0] for m in best_matches]:
                    best_matches.append((chunk_id, chunk, score, keyword_count))
    
    # Sort by keyword count and score
    best_matches.sort(key=lambda x: (x[3], x[2]), reverse=True)
    
    print(f"\n🏆 Top Matches:")
    print("=" * 60)
    
    target_found = False
    
    for i, (chunk_id, chunk, score, keyword_count) in enumerate(best_matches[:5]):
        print(f"\n[{i+1}] Keywords: {keyword_count}, Score: {score:.4f}")
        print(f"File: {Path(chunk.metadata['file_path']).name}")
        print(f"Heading: {chunk.metadata['heading']}")
        print(f"Content:")
        print("-" * 40)
        print(chunk.text)
        print("-" * 40)
        
        # Check for the specific target content
        content = chunk.text
        if ("Multi-Function Device" in content and 
            "PCI Express errors" in content and
            ("Physical Layer errors" in content or 
             "Data Link Layer errors" in content or
             "at most one error" in content)):
            print("🎯 TARGET CONTENT FOUND!")
            target_found = True
    
    # If not found in top matches, search more broadly
    if not target_found:
        print(f"\n🔍 Searching more broadly...")
        
        # Search for exact phrases
        exact_phrases = [
            "Multi-Function Device other than an SR-IOV device",
            "PCI Express errors not specific to any single Function",
            "Physical Layer errors",
            "Data Link Layer errors", 
            "ECRC Check Failed",
            "at most one error reporting Message"
        ]
        
        for chunk in all_chunks:
            content = chunk.text
            phrase_count = sum(1 for phrase in exact_phrases if phrase in content)
            
            if phrase_count >= 2:
                print(f"\n📋 Potential match ({phrase_count} exact phrases):")
                print(f"File: {Path(chunk.metadata['file_path']).name}")
                print(f"Heading: {chunk.metadata['heading']}")
                print(f"Content: {content[:500]}...")
                
                if phrase_count >= 3:
                    print("🎯 STRONG TARGET CANDIDATE!")
                    print("Full content:")
                    print("-" * 40)
                    print(content)
                    print("-" * 40)
                    target_found = True
                    break
    
    return target_found

def test_with_existing_rag():
    """Test with the existing RAG system for comparison"""
    print(f"\n🔄 Testing with Existing RAG System")
    print("=" * 60)
    
    try:
        from src.cli.interactive import PCIeDebugShell
        
        shell = PCIeDebugShell(verbose=False)
        shell.analysis_verbose = True
        
        print(f"Vector store has {len(shell.vector_store.documents)} documents")
        
        # Test the query
        shell.onecmd("how multi function handle error?")
        
    except Exception as e:
        print(f"Error with existing RAG: {e}")

if __name__ == "__main__":
    # Test the new implementation
    found_target = search_for_target_content()
    
    # Test with existing RAG for comparison
    test_with_existing_rag()
    
    if found_target:
        print("\n🎉 SUCCESS: Target content found with heading-aware approach!")
    else:
        print("\n⚠️  Target content not clearly found. May need larger chunk sizes or better parsing.")