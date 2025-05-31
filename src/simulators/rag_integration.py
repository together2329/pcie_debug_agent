"""
Integration between PCIe simulator and RAG engine for real-time analysis
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rag.enhanced_rag_engine import EnhancedRAGEngine, RAGQuery
from src.rag.vector_store import FAISSVectorStore
from src.config.settings import load_settings
from src.models.model_manager import ModelManager
from src.collectors.log_collector import LogCollector


class SimulatorLogCapture:
    """Captures and processes simulator logs in real-time"""
    
    def __init__(self):
        self.log_buffer = []
        self.error_buffer = []
        self.transaction_buffer = []
        self.callbacks = []
        
    def add_callback(self, callback):
        """Add callback for log processing"""
        self.callbacks.append(callback)
        
    def parse_simulator_log(self, log_line: str) -> Dict[str, Any]:
        """Parse a simulator log line into structured format"""
        # Extract timestamp
        timestamp_match = re.match(r'\[(\d+)\]', log_line)
        timestamp = int(timestamp_match.group(1)) if timestamp_match else 0
        
        # Determine log type and extract details
        log_entry = {
            "timestamp": timestamp,
            "raw": log_line,
            "time_ns": timestamp,
            "type": "info"
        }
        
        # Parse PCIe specific logs
        if "PCIe:" in log_line:
            if "ERROR" in log_line:
                log_entry["type"] = "error"
                
                # Extract error type
                if "CRC error" in log_line:
                    log_entry["error_type"] = "CRC_ERROR"
                elif "timeout" in log_line or "Timeout" in log_line:
                    log_entry["error_type"] = "TIMEOUT"
                elif "ECRC error" in log_line:
                    log_entry["error_type"] = "ECRC_ERROR"
                elif "Malformed TLP" in log_line:
                    log_entry["error_type"] = "MALFORMED_TLP"
                elif "Unsupported" in log_line:
                    log_entry["error_type"] = "UNSUPPORTED_REQUEST"
                    
            elif "TLP Received" in log_line:
                log_entry["type"] = "transaction"
                # Extract TLP details
                tlp_match = re.search(r'Type=(\w+).*Addr=0x(\w+).*Data=0x(\w+).*Tag=(\d+)', log_line)
                if tlp_match:
                    log_entry["tlp_type"] = tlp_match.group(1)
                    log_entry["address"] = tlp_match.group(2)
                    log_entry["data"] = tlp_match.group(3)
                    log_entry["tag"] = int(tlp_match.group(4))
                    
            elif "Completion" in log_line:
                log_entry["type"] = "completion"
                cpl_match = re.search(r'Tag=(\d+).*Status=(\w+)', log_line)
                if cpl_match:
                    log_entry["tag"] = int(cpl_match.group(1))
                    log_entry["status"] = cpl_match.group(2)
                    
            elif "LTSSM" in log_line:
                log_entry["type"] = "link_state"
                if "entering" in log_line:
                    state_match = re.search(r'entering (\w+)', log_line)
                    if state_match:
                        log_entry["ltssm_state"] = state_match.group(1)
                        
            elif "Link UP" in log_line:
                log_entry["type"] = "link_status"
                link_match = re.search(r'Speed Gen(\d).*Width x(\d+)', log_line)
                if link_match:
                    log_entry["link_speed"] = f"Gen{link_match.group(1)}"
                    log_entry["link_width"] = f"x{link_match.group(2)}"
                    
        return log_entry
    
    def process_log(self, log_line: str):
        """Process a single log line"""
        parsed = self.parse_simulator_log(log_line)
        
        # Add to appropriate buffer
        self.log_buffer.append(parsed)
        
        if parsed["type"] == "error":
            self.error_buffer.append(parsed)
        elif parsed["type"] == "transaction":
            self.transaction_buffer.append(parsed)
            
        # Trigger callbacks
        for callback in self.callbacks:
            callback(parsed)
            
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors detected"""
        error_counts = {}
        for error in self.error_buffer:
            error_type = error.get("error_type", "UNKNOWN")
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
        return {
            "total_errors": len(self.error_buffer),
            "error_counts": error_counts,
            "first_error_time": self.error_buffer[0]["timestamp"] if self.error_buffer else None,
            "last_error_time": self.error_buffer[-1]["timestamp"] if self.error_buffer else None
        }


class SimulatorRAGBridge:
    """Bridge between simulator and RAG engine"""
    
    def __init__(self, rag_engine: EnhancedRAGEngine):
        self.rag_engine = rag_engine
        self.log_capture = SimulatorLogCapture()
        self.log_batch = []
        self.batch_size = 10
        
        # Register callback
        self.log_capture.add_callback(self._on_log_received)
        
    def _on_log_received(self, parsed_log: Dict[str, Any]):
        """Handle incoming log from simulator"""
        # Format log for RAG engine
        formatted_log = self._format_log_for_rag(parsed_log)
        
        # Add to batch
        self.log_batch.append(formatted_log)
        
        # Process batch if full
        if len(self.log_batch) >= self.batch_size:
            self._process_log_batch()
            
    def _format_log_for_rag(self, parsed_log: Dict[str, Any]) -> str:
        """Format parsed log for RAG engine ingestion"""
        timestamp = parsed_log["timestamp"]
        log_type = parsed_log["type"]
        
        if log_type == "error":
            error_type = parsed_log.get("error_type", "UNKNOWN")
            return f"[{timestamp}ns] PCIe ERROR: {error_type} - {parsed_log['raw']}"
            
        elif log_type == "transaction":
            tlp_type = parsed_log.get("tlp_type", "?")
            addr = parsed_log.get("address", "?")
            tag = parsed_log.get("tag", "?")
            return f"[{timestamp}ns] PCIe TLP: Type={tlp_type} Addr=0x{addr} Tag={tag}"
            
        elif log_type == "link_state":
            state = parsed_log.get("ltssm_state", "?")
            return f"[{timestamp}ns] PCIe LTSSM: State={state}"
            
        else:
            return f"[{timestamp}ns] {parsed_log['raw']}"
            
    def _process_log_batch(self):
        """Process accumulated logs"""
        if not self.log_batch:
            return
            
        # Generate embeddings for batch
        embeddings = self.rag_engine.model_manager.generate_embeddings(self.log_batch)
        
        # Create metadata
        metadata = []
        for i, log in enumerate(self.log_batch):
            meta = {
                "source": "simulator",
                "timestamp": datetime.now().isoformat(),
                "log_index": i,
                "type": "runtime_log"
            }
            metadata.append(meta)
            
        # Add to vector store
        self.rag_engine.vector_store.add_documents(
            self.log_batch,
            embeddings,
            metadata
        )
        
        # Clear batch
        self.log_batch = []
        
    def analyze_simulation(self, query: str) -> Dict[str, Any]:
        """Analyze simulation results based on query"""
        # Ensure all logs are processed
        self._process_log_batch()
        
        # Get error summary
        error_summary = self.log_capture.get_error_summary()
        
        # Create enhanced query with context
        context = f"""
        Simulation Summary:
        - Total Errors: {error_summary['total_errors']}
        - Error Types: {json.dumps(error_summary['error_counts'])}
        - Total Transactions: {len(self.log_capture.transaction_buffer)}
        
        Query: {query}
        """
        
        # Create RAG query
        rag_query = RAGQuery(
            query=context,
            context_window=5,
            min_similarity=0.3,
            rerank=True
        )
        
        # Get analysis from RAG engine
        result = self.rag_engine.query(rag_query)
        
        # Add simulation-specific data
        result.metadata = {
            "error_summary": error_summary,
            "transaction_count": len(self.log_capture.transaction_buffer),
            "total_logs": len(self.log_capture.log_buffer)
        }
        
        return result
        
    def get_simulation_report(self) -> str:
        """Generate comprehensive simulation report"""
        error_summary = self.log_capture.get_error_summary()
        
        report = f"""
PCIe Simulation Analysis Report
==============================

Simulation Statistics:
- Total Log Entries: {len(self.log_capture.log_buffer)}
- Total Transactions: {len(self.log_capture.transaction_buffer)}
- Total Errors: {error_summary['total_errors']}

Error Breakdown:
"""
        for error_type, count in error_summary['error_counts'].items():
            report += f"- {error_type}: {count} occurrences\n"
            
        # Add sample errors
        if self.log_capture.error_buffer:
            report += "\nSample Errors:\n"
            for error in self.log_capture.error_buffer[:5]:
                report += f"- [{error['timestamp']}ns] {error.get('error_type', 'UNKNOWN')}\n"
                
        # Add LLM analysis
        analysis_result = self.analyze_simulation(
            "Analyze the PCIe errors and provide recommendations for fixing them"
        )
        
        report += f"\nAI Analysis:\n{analysis_result.answer}\n"
        
        return report


def create_rag_bridge() -> SimulatorRAGBridge:
    """Create and initialize RAG bridge for simulator integration"""
    # Load settings
    config_path = Path("configs/settings.yaml")
    settings = load_settings(config_path)
    
    # Initialize model manager
    model_manager = ModelManager()
    model_manager.load_embedding_model(settings.embedding.model)
    
    # Initialize LLM
    model_manager.initialize_llm(
        provider=settings.llm.provider,
        model_name=settings.llm.model,
        models_dir=settings.local_llm.models_dir
    )
    
    # Create vector store
    vector_store = FAISSVectorStore(
        index_path=str(settings.vector_store.index_path),
        index_type=settings.vector_store.index_type,
        dimension=settings.embedding.dimension
    )
    
    # Create RAG engine
    rag_engine = EnhancedRAGEngine(
        vector_store=vector_store,
        model_manager=model_manager,
        llm_provider=settings.llm.provider,
        llm_model=settings.llm.model
    )
    
    # Create bridge
    return SimulatorRAGBridge(rag_engine)


# Example usage in test
if __name__ == "__main__":
    # This would be called from within cocotb test
    bridge = create_rag_bridge()
    
    # Simulate some logs
    sample_logs = [
        "[1000] PCIe: TLP Received - Type=0 Addr=0x00001000 Data=0x00000000 Tag=0",
        "[1500] PCIe: ERROR - CRC error injected",
        "[2000] PCIe: ERROR - Link training timeout in POLLING",
        "[3000] PCIe: Completion generated - Tag=0 Status=CA"
    ]
    
    for log in sample_logs:
        bridge.log_capture.process_log(log)
        
    # Analyze
    result = bridge.analyze_simulation("What errors occurred and why?")
    print(result.answer)