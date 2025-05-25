import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from config.settings import Settings
from collectors.log_collector import LogCollector, UVMError
from processors.document_chunker import DocumentChunker
from processors.code_chunker import SystemVerilogChunker
from processors.embedder import Embedder
from vectorstore.faiss_store import FAISSVectorStore
from rag.retriever import Retriever
from rag.analyzer import Analyzer
from reports.report_generator import ReportGenerator

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO"):
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def collect_errors(settings: Settings, start_time: str = None) -> List[UVMError]:
    """로그 파일에서 에러 수집"""
    collector = LogCollector()
    errors = []
    
    for log_dir in settings.log_directories:
        try:
            dir_errors = collector.collect_logs(
                log_dir=log_dir,
                start_time=start_time,
                file_pattern="*.log"
            )
            errors.extend(dir_errors)
        except Exception as e:
            logger.error(f"Error collecting logs from {log_dir}: {e}")
            
    return errors

def process_documents(settings: Settings) -> List[Dict[str, Any]]:
    """문서 처리 및 임베딩"""
    # 문서 청커 초기화
    doc_chunker = DocumentChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    
    # 코드 청커 초기화
    code_chunker = SystemVerilogChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    
    # 임베더 초기화
    embedder = Embedder(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size
    )
    
    # 벡터 스토어 초기화
    vector_store = FAISSVectorStore(
        dimension=settings.embedding_dimension,
        index_type="IndexFlatIP"
    )
    
    # 문서 처리
    chunks = []
    
    # 스펙 문서 처리
    for spec_file in Path(settings.data_dir).glob("specs/**/*"):
        if spec_file.suffix in ['.pdf', '.md', '.txt']:
            try:
                doc_chunks = doc_chunker.chunk_documents([spec_file])
                chunks.extend(doc_chunks)
            except Exception as e:
                logger.error(f"Error processing spec file {spec_file}: {e}")
    
    # 코드 파일 처리
    for code_file in Path(settings.data_dir).glob("testbench/**/*.sv"):
        try:
            code_chunks = code_chunker.chunk_sv_file(code_file)
            chunks.extend(code_chunks)
        except Exception as e:
            logger.error(f"Error processing code file {code_file}: {e}")
    
    # 임베딩 생성
    embedder.embed_chunks(chunks)
    
    # 벡터 스토어에 추가
    vector_store.add_documents(
        embeddings=[chunk.embedding for chunk in chunks],
        documents=[chunk.content for chunk in chunks],
        metadata=[chunk.metadata for chunk in chunks]
    )
    
    return chunks

def analyze_errors(settings: Settings,
                  errors: List[UVMError],
                  chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """에러 분석"""
    # 임베더 초기화
    embedder = Embedder(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size
    )
    
    # 벡터 스토어 초기화
    vector_store = FAISSVectorStore(
        dimension=settings.embedding_dimension,
        index_type="IndexFlatIP"
    )
    
    # 검색기 초기화
    retriever = Retriever(
        vector_store=vector_store,
        embedder=embedder
    )
    
    # 분석기 초기화
    analyzer = Analyzer(
        llm_provider=settings.llm_provider,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.openai_api_key
    )
    
    # 에러 분석
    analysis_results = []
    
    for error in errors:
        try:
            # 관련 문서 검색
            context = retriever.retrieve(
                query=error.message,
                k=settings.retrieval_top_k
            )
            
            # 에러 분석
            analysis = analyzer.analyze_error(error, context)
            analysis_results.append(analysis)
            
        except Exception as e:
            logger.error(f"Error analyzing error {error}: {e}")
            
    return analysis_results

def generate_reports(settings: Settings,
                    errors: List[UVMError],
                    analysis_results: List[Dict[str, Any]]):
    """리포트 생성"""
    report_generator = ReportGenerator()
    
    # HTML 리포트 생성
    html_report = report_generator.generate_report(
        errors=errors,
        analysis_results=analysis_results,
        output_path=settings.report_dir / "report.html",
        format="html"
    )
    
    # Markdown 리포트 생성
    md_report = report_generator.generate_report(
        errors=errors,
        analysis_results=analysis_results,
        output_path=settings.report_dir / "report.md",
        format="markdown"
    )
    
    # 요약 리포트 생성
    summary_report = report_generator.generate_summary(
        errors=errors,
        analysis_results=analysis_results,
        output_path=settings.report_dir / "summary.yaml"
    )
    
    return html_report, md_report, summary_report

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="UVM 에러 분석 도구")
    parser.add_argument("--config", type=str, default="configs/settings.yaml",
                      help="설정 파일 경로")
    parser.add_argument("--start-time", type=str,
                      help="분석 시작 시간 (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--log-level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="로깅 레벨")
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(args.log_level)
    
    try:
        # 설정 로드
        settings = Settings.from_yaml(args.config)
        
        # 에러 수집
        logger.info("Collecting errors...")
        errors = collect_errors(settings, args.start_time)
        logger.info(f"Collected {len(errors)} errors")
        
        # 문서 처리
        logger.info("Processing documents...")
        chunks = process_documents(settings)
        logger.info(f"Processed {len(chunks)} chunks")
        
        # 에러 분석
        logger.info("Analyzing errors...")
        analysis_results = analyze_errors(settings, errors, chunks)
        logger.info(f"Analyzed {len(analysis_results)} errors")
        
        # 리포트 생성
        logger.info("Generating reports...")
        html_report, md_report, summary_report = generate_reports(
            settings, errors, analysis_results
        )
        logger.info(f"Generated reports: {html_report}, {md_report}, {summary_report}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main() 