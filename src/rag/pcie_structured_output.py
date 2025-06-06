"""
PCIe Structured Output Module

Provides structured data classes and JSON serialization for PCIe RAG responses
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json
from datetime import datetime


class PCIeLayer(str, Enum):
    """PCIe protocol layers"""
    PHYSICAL = "physical"
    DATA_LINK = "data_link"
    TRANSACTION = "transaction"
    SOFTWARE = "software"
    POWER_MANAGEMENT = "power_management"
    SYSTEM_ARCHITECTURE = "system_architecture"
    GENERAL = "general"
    UNKNOWN = "unknown"


class TechnicalLevel(int, Enum):
    """Technical complexity levels"""
    BASIC = 1
    INTERMEDIATE = 2
    ADVANCED = 3


class SemanticType(str, Enum):
    """Semantic content types"""
    HEADER_SECTION = "header_section"
    PROCEDURE = "procedure"
    SPECIFICATION = "specification"
    EXAMPLE = "example"
    CONTENT = "content"
    UNKNOWN = "unknown"


@dataclass
class PCIeConcept:
    """Represents a PCIe concept extracted from content"""
    name: str
    category: str  # e.g., "LTSSM State", "TLP Type", "Error Code"
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PCIeChunkMetadata:
    """Metadata for a PCIe document chunk"""
    source_file: str
    chunk_id: str
    technical_level: TechnicalLevel
    pcie_layer: PCIeLayer
    semantic_type: SemanticType
    pcie_concepts: List[PCIeConcept] = field(default_factory=list)
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "chunk_id": self.chunk_id,
            "technical_level": self.technical_level.value,
            "pcie_layer": self.pcie_layer.value,
            "semantic_type": self.semantic_type.value,
            "pcie_concepts": [c.to_dict() for c in self.pcie_concepts],
            "section_title": self.section_title,
            "page_number": self.page_number
        }


@dataclass
class PCIeSearchResult:
    """Structured search result for PCIe queries"""
    content: str
    score: float
    metadata: PCIeChunkMetadata
    highlighted_terms: List[str] = field(default_factory=list)
    relevance_explanation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata.to_dict(),
            "highlighted_terms": self.highlighted_terms,
            "relevance_explanation": self.relevance_explanation
        }


@dataclass 
class PCIeAnalysis:
    """Structured analysis of PCIe issue"""
    issue_type: str  # e.g., "Link Training Failure", "Completion Timeout"
    severity: str  # "critical", "warning", "info"
    root_cause: Optional[str] = None
    affected_components: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    related_specs: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PCIeQueryResponse:
    """Complete structured response for PCIe queries"""
    query: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    response_time_ms: float = 0.0
    total_results: int = 0
    results: List[PCIeSearchResult] = field(default_factory=list)
    analysis: Optional[PCIeAnalysis] = None
    model_used: str = "unknown"
    search_mode: str = "pcie"
    filters_applied: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "query": self.query,
            "timestamp": self.timestamp,
            "response_time_ms": self.response_time_ms,
            "total_results": self.total_results,
            "results": [r.to_dict() for r in self.results],
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "model_used": self.model_used,
            "search_mode": self.search_mode,
            "filters_applied": self.filters_applied
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to formatted JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def to_summary(self) -> str:
        """Generate human-readable summary"""
        lines = [
            f"Query: {self.query}",
            f"Results: {self.total_results} found in {self.response_time_ms:.1f}ms",
            f"Model: {self.model_used} (mode: {self.search_mode})"
        ]
        
        if self.filters_applied:
            lines.append(f"Filters: {', '.join(f'{k}={v}' for k, v in self.filters_applied.items())}")
        
        if self.analysis:
            lines.extend([
                f"\nAnalysis:",
                f"  Issue Type: {self.analysis.issue_type}",
                f"  Severity: {self.analysis.severity}",
                f"  Confidence: {self.analysis.confidence:.1%}"
            ])
            if self.analysis.root_cause:
                lines.append(f"  Root Cause: {self.analysis.root_cause}")
        
        if self.results:
            lines.append(f"\nTop Results:")
            for i, result in enumerate(self.results[:3], 1):
                lines.append(f"  {i}. [{result.score:.3f}] {result.metadata.source_file}")
                if result.metadata.section_title:
                    lines.append(f"     Section: {result.metadata.section_title}")
                if result.metadata.pcie_concepts:
                    concepts = [c.name for c in result.metadata.pcie_concepts[:3]]
                    lines.append(f"     Concepts: {', '.join(concepts)}")
        
        return "\n".join(lines)


class PCIeStructuredOutputFormatter:
    """Formats PCIe RAG responses as structured output"""
    
    @staticmethod
    def format_search_results(
        query: str,
        raw_results: List[Dict[str, Any]],
        response_time_ms: float,
        model_used: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> PCIeQueryResponse:
        """Convert raw search results to structured response"""
        
        # Convert raw results to structured results
        structured_results = []
        for raw in raw_results:
            # Extract metadata
            meta = raw.get('metadata', {})
            
            # Create PCIe concepts
            concepts = []
            for concept_name in meta.get('pcie_concepts', []):
                if concept_name:  # Skip empty strings
                    concepts.append(PCIeConcept(
                        name=concept_name,
                        category=PCIeStructuredOutputFormatter._categorize_concept(concept_name)
                    ))
            
            # Create chunk metadata
            chunk_meta = PCIeChunkMetadata(
                source_file=meta.get('source', 'unknown'),
                chunk_id=meta.get('chunk_id', 'unknown'),
                technical_level=TechnicalLevel(meta.get('technical_level', 1)),
                pcie_layer=PCIeLayer(meta.get('pcie_layer', 'unknown')),
                semantic_type=SemanticType(meta.get('semantic_type', 'content')),
                pcie_concepts=concepts,
                section_title=meta.get('section_title'),
                page_number=meta.get('page_number')
            )
            
            # Create search result
            result = PCIeSearchResult(
                content=raw.get('content', ''),
                score=float(raw.get('score', 0.0)),
                metadata=chunk_meta,
                highlighted_terms=raw.get('highlighted_terms', [])
            )
            
            structured_results.append(result)
        
        # Create response
        response = PCIeQueryResponse(
            query=query,
            response_time_ms=response_time_ms,
            total_results=len(structured_results),
            results=structured_results,
            model_used=model_used,
            filters_applied=filters or {}
        )
        
        return response
    
    @staticmethod
    def _categorize_concept(concept_name: str) -> str:
        """Categorize a PCIe concept"""
        concept_lower = concept_name.lower()
        
        if any(state in concept_lower for state in ['detect', 'polling', 'config', 'l0', 'l1', 'l2', 'recovery']):
            return "LTSSM State"
        elif any(tlp in concept_lower for tlp in ['tlp', 'dllp', 'completion', 'request', 'message']):
            return "TLP Type"
        elif any(err in concept_lower for err in ['timeout', 'error', 'violation', 'malformed']):
            return "Error Code"
        elif any(pwr in concept_lower for pwr in ['aspm', 'power', 'd0', 'd1', 'd2', 'd3']):
            return "Power State"
        else:
            return "General"
    
    @staticmethod
    def add_analysis(
        response: PCIeQueryResponse,
        issue_type: str,
        severity: str = "info",
        root_cause: Optional[str] = None,
        confidence: float = 0.0
    ) -> PCIeQueryResponse:
        """Add analysis to response"""
        
        analysis = PCIeAnalysis(
            issue_type=issue_type,
            severity=severity,
            root_cause=root_cause,
            confidence=confidence
        )
        
        # Extract affected components from results
        components = set()
        specs = set()
        
        for result in response.results:
            # Extract components from concepts
            for concept in result.metadata.pcie_concepts:
                if concept.category in ["LTSSM State", "TLP Type"]:
                    components.add(concept.name)
            
            # Extract spec references
            if "PCIe" in result.content and "specification" in result.content.lower():
                specs.add(result.metadata.source_file)
        
        analysis.affected_components = list(components)
        analysis.related_specs = list(specs)
        
        # Add recommendations based on issue type
        if "timeout" in issue_type.lower():
            analysis.recommended_actions = [
                "Check PCIe link training status",
                "Verify signal integrity",
                "Review timeout timer values",
                "Check for hardware compatibility"
            ]
        elif "link training" in issue_type.lower():
            analysis.recommended_actions = [
                "Verify PCIe lane connectivity",
                "Check reference clock",
                "Review LTSSM state transitions",
                "Validate electrical parameters"
            ]
        
        response.analysis = analysis
        return response