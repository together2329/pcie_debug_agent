"""
LLM-powered metadata extraction for enhanced RAG functionality
Automatically extracts rich metadata from PCIe-related documents
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from src.models.model_manager import ModelManager

logger = logging.getLogger(__name__)


class PCIeDocumentType(Enum):
    """Types of PCIe documents"""
    SPECIFICATION = "specification"
    ERROR_LOG = "error_log"
    DEBUG_LOG = "debug_log"
    TROUBLESHOOTING = "troubleshooting"
    TUTORIAL = "tutorial"
    CODE = "code"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class PCIeVersion(Enum):
    """PCIe specification versions"""
    PCIE_1_0 = "1.0"
    PCIE_1_1 = "1.1"
    PCIE_2_0 = "2.0"
    PCIE_2_1 = "2.1"
    PCIE_3_0 = "3.0"
    PCIE_3_1 = "3.1"
    PCIE_4_0 = "4.0"
    PCIE_5_0 = "5.0"
    PCIE_6_0 = "6.0"
    GENERIC = "generic"


class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"
    UNKNOWN = "unknown"


@dataclass
class PCIeMetadata:
    """Rich metadata for PCIe documents"""
    # Basic metadata
    document_type: PCIeDocumentType
    title: str
    summary: str
    
    # PCIe specific metadata
    pcie_version: List[PCIeVersion]
    topics: List[str]  # e.g., ["link_training", "error_handling", "TLP"]
    
    # Error/Debug specific
    error_codes: List[str] = None
    error_severity: ErrorSeverity = None
    components: List[str] = None  # e.g., ["root_complex", "endpoint", "switch"]
    
    # Technical details
    speed: Optional[str] = None  # e.g., "Gen3 x16"
    link_width: Optional[int] = None  # e.g., 1, 4, 8, 16
    
    # Context
    keywords: List[str] = None
    related_specs: List[str] = None  # e.g., ["PCIe Base Spec 4.0", "CEM Spec"]
    
    # Quality metadata
    confidence_score: float = 0.0
    extraction_timestamp: str = None
    
    def __post_init__(self):
        if self.extraction_timestamp is None:
            self.extraction_timestamp = datetime.now().isoformat()
        if self.keywords is None:
            self.keywords = []
        if self.error_codes is None:
            self.error_codes = []
        if self.components is None:
            self.components = []
        if self.related_specs is None:
            self.related_specs = []


class MetadataExtractor:
    """Extract rich metadata from documents using LLM"""
    
    def __init__(self, model_manager: ModelManager, model_id: str = "gpt-4o-mini"):
        self.model_manager = model_manager
        self.model_id = model_id
        self.extraction_prompt_template = """
Analyze the following PCIe-related document and extract structured metadata.

Document content:
{content}

Extract the following metadata in JSON format:
1. document_type: One of [specification, error_log, debug_log, troubleshooting, tutorial, code, configuration, unknown]
2. title: A descriptive title for this content
3. summary: A 1-2 sentence summary
4. pcie_version: List of PCIe versions mentioned (e.g., ["3.0", "4.0"])
5. topics: List of main topics covered (e.g., ["link_training", "error_handling", "TLP", "power_management"])
6. error_codes: List of any error codes mentioned (e.g., ["0x123", "LTSSM_TIMEOUT"])
7. error_severity: If errors present, severity level [critical, error, warning, info, debug, unknown]
8. components: PCIe components mentioned (e.g., ["root_complex", "endpoint", "switch", "phy"])
9. speed: PCIe speed if mentioned (e.g., "Gen3 x16", "Gen4 x8")
10. link_width: Link width if mentioned (1, 4, 8, 16)
11. keywords: Important technical keywords
12. related_specs: Related specifications mentioned
13. confidence_score: Your confidence in this extraction (0.0-1.0)

Respond with valid JSON only.
"""
        
        self.batch_size = 5  # Process multiple documents in parallel
        
    async def extract_metadata(self, content: str, 
                             file_path: Optional[str] = None) -> PCIeMetadata:
        """Extract metadata from a single document"""
        try:
            # Prepare the prompt
            prompt = self.extraction_prompt_template.format(
                content=content[:3000]  # Limit content size for LLM
            )
            
            # Get LLM response
            response = await self._call_llm(prompt)
            
            # Parse JSON response
            metadata_dict = json.loads(response)
            
            # Convert to PCIeMetadata object
            metadata = self._dict_to_metadata(metadata_dict)
            
            # Add file path context if available
            if file_path:
                metadata.keywords.append(f"source:{file_path}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            # Return basic metadata on error
            return PCIeMetadata(
                document_type=PCIeDocumentType.UNKNOWN,
                title="Unknown Document",
                summary="Failed to extract metadata",
                pcie_version=[PCIeVersion.GENERIC],
                topics=["unknown"],
                confidence_score=0.0
            )
    
    async def extract_batch(self, documents: List[Tuple[str, str]]) -> List[PCIeMetadata]:
        """Extract metadata from multiple documents in batch"""
        import asyncio
        
        tasks = []
        for content, file_path in documents:
            task = self.extract_metadata(content, file_path)
            tasks.append(task)
        
        # Process in batches to avoid overwhelming the LLM
        results = []
        for i in range(0, len(tasks), self.batch_size):
            batch = tasks[i:i + self.batch_size]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
        
        return results
    
    def extract_quick_metadata(self, content: str) -> Dict[str, Any]:
        """Quick metadata extraction without LLM (for real-time processing)"""
        import re
        
        metadata = {
            "quick_extract": True,
            "timestamp": datetime.now().isoformat()
        }
        
        # Extract PCIe versions
        version_pattern = r'PCIe?\s*([1-6]\.[0-9]|Gen[1-6])'
        versions = re.findall(version_pattern, content, re.IGNORECASE)
        metadata["pcie_versions"] = list(set(versions))
        
        # Extract error codes
        error_pattern = r'(0x[0-9A-Fa-f]{2,8}|Error:\s*\w+|LTSSM_\w+)'
        errors = re.findall(error_pattern, content)
        metadata["error_codes"] = list(set(errors))[:10]  # Limit to 10
        
        # Extract speeds
        speed_pattern = r'(Gen[1-6]\s*x\d+|[0-9]+\s*GT/s)'
        speeds = re.findall(speed_pattern, content, re.IGNORECASE)
        metadata["speeds"] = list(set(speeds))
        
        # Extract components
        component_keywords = [
            "root complex", "endpoint", "switch", "bridge",
            "phy", "controller", "device", "host"
        ]
        found_components = []
        content_lower = content.lower()
        for comp in component_keywords:
            if comp in content_lower:
                found_components.append(comp.replace(" ", "_"))
        metadata["components"] = found_components
        
        # Detect document type
        if "error" in content_lower or "fail" in content_lower:
            metadata["likely_type"] = "error_log"
        elif "specification" in content_lower or "section" in content_lower:
            metadata["likely_type"] = "specification"
        elif "debug" in content_lower or "trace" in content_lower:
            metadata["likely_type"] = "debug_log"
        else:
            metadata["likely_type"] = "unknown"
        
        return metadata
    
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM for metadata extraction"""
        try:
            model = self.model_manager.get_model(self.model_id)
            response = await model.agenerate(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise
    
    def _dict_to_metadata(self, data: Dict[str, Any]) -> PCIeMetadata:
        """Convert dictionary to PCIeMetadata object"""
        # Convert string enums to enum objects
        doc_type = PCIeDocumentType(data.get("document_type", "unknown"))
        
        pcie_versions = []
        for v in data.get("pcie_version", ["generic"]):
            try:
                pcie_versions.append(PCIeVersion(v))
            except:
                pcie_versions.append(PCIeVersion.GENERIC)
        
        error_severity = None
        if data.get("error_severity"):
            try:
                error_severity = ErrorSeverity(data["error_severity"])
            except:
                error_severity = ErrorSeverity.UNKNOWN
        
        return PCIeMetadata(
            document_type=doc_type,
            title=data.get("title", "Untitled"),
            summary=data.get("summary", "No summary"),
            pcie_version=pcie_versions,
            topics=data.get("topics", []),
            error_codes=data.get("error_codes", []),
            error_severity=error_severity,
            components=data.get("components", []),
            speed=data.get("speed"),
            link_width=data.get("link_width"),
            keywords=data.get("keywords", []),
            related_specs=data.get("related_specs", []),
            confidence_score=data.get("confidence_score", 0.5)
        )


class MetadataFilter:
    """Filter documents based on metadata"""
    
    @staticmethod
    def filter_by_pcie_version(documents: List[Dict[str, Any]], 
                              versions: List[str]) -> List[Dict[str, Any]]:
        """Filter documents by PCIe version"""
        filtered = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            doc_versions = metadata.get("pcie_version", [])
            if any(v in doc_versions for v in versions):
                filtered.append(doc)
        return filtered
    
    @staticmethod
    def filter_by_error_severity(documents: List[Dict[str, Any]], 
                                min_severity: ErrorSeverity) -> List[Dict[str, Any]]:
        """Filter documents by minimum error severity"""
        severity_order = {
            ErrorSeverity.DEBUG: 0,
            ErrorSeverity.INFO: 1,
            ErrorSeverity.WARNING: 2,
            ErrorSeverity.ERROR: 3,
            ErrorSeverity.CRITICAL: 4
        }
        
        min_level = severity_order.get(min_severity, 0)
        filtered = []
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            doc_severity = metadata.get("error_severity")
            if doc_severity and severity_order.get(doc_severity, 0) >= min_level:
                filtered.append(doc)
        
        return filtered
    
    @staticmethod
    def filter_by_components(documents: List[Dict[str, Any]], 
                           components: List[str]) -> List[Dict[str, Any]]:
        """Filter documents by PCIe components"""
        filtered = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            doc_components = metadata.get("components", [])
            if any(c in doc_components for c in components):
                filtered.append(doc)
        return filtered
    
    @staticmethod
    def filter_by_topics(documents: List[Dict[str, Any]], 
                        topics: List[str]) -> List[Dict[str, Any]]:
        """Filter documents by topics"""
        filtered = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            doc_topics = metadata.get("topics", [])
            if any(t in doc_topics for t in topics):
                filtered.append(doc)
        return filtered