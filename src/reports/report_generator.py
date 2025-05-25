import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from pathlib import Path
import jinja2
import yaml

logger = logging.getLogger(__name__)

class ReportGenerator:
    """에러 분석 리포트 생성기"""
    
    def __init__(self, template_dir: str = "templates"):
        self.template_dir = Path(template_dir)
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            autoescape=True
        )
        
    def generate_report(self,
                       errors: List[Dict[str, Any]],
                       analysis_results: List[Dict[str, Any]],
                       output_path: str,
                       format: str = "html") -> str:
        """
        분석 리포트 생성
        
        Args:
            errors: 원본 에러 목록
            analysis_results: 분석 결과 목록
            output_path: 출력 파일 경로
            format: 출력 형식 (html/markdown)
            
        Returns:
            생성된 리포트 파일 경로
        """
        # 데이터 준비
        report_data = self._prepare_report_data(errors, analysis_results)
        
        # 템플릿 선택
        template = self._get_template(format)
        
        # 리포트 생성
        report_content = template.render(**report_data)
        
        # 파일 저장
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        logger.info(f"Report generated: {output_path}")
        return str(output_path)
    
    def _prepare_report_data(self,
                           errors: List[Dict[str, Any]],
                           analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """리포트 데이터 준비"""
        
        # 에러 통계
        error_stats = {
            'total': len(errors),
            'by_severity': {},
            'by_component': {},
            'by_file': {}
        }
        
        for error in errors:
            # 심각도별 통계
            severity = error.get('severity', 'unknown')
            error_stats['by_severity'][severity] = error_stats['by_severity'].get(severity, 0) + 1
            
            # 컴포넌트별 통계
            component = error.get('component', 'unknown')
            error_stats['by_component'][component] = error_stats['by_component'].get(component, 0) + 1
            
            # 파일별 통계
            file_path = error.get('file_path', 'unknown')
            error_stats['by_file'][file_path] = error_stats['by_file'].get(file_path, 0) + 1
        
        # 분석 결과 통계
        analysis_stats = {
            'total': len(analysis_results),
            'by_root_cause': {},
            'by_component': {}
        }
        
        for result in analysis_results:
            # 근본 원인별 통계
            root_cause = result.get('root_cause', {}).get('summary', 'unknown')
            analysis_stats['by_root_cause'][root_cause] = analysis_stats['by_root_cause'].get(root_cause, 0) + 1
            
            # 영향받는 컴포넌트별 통계
            components = result.get('component_analysis', {}).get('affected_components', [])
            for component in components:
                analysis_stats['by_component'][component] = analysis_stats['by_component'].get(component, 0) + 1
        
        return {
            'generated_at': datetime.now().isoformat(),
            'error_stats': error_stats,
            'analysis_stats': analysis_stats,
            'errors': errors,
            'analysis_results': analysis_results
        }
    
    def _get_template(self, format: str) -> jinja2.Template:
        """템플릿 선택"""
        template_name = f"report.{format}.jinja2"
        try:
            return self.env.get_template(template_name)
        except jinja2.TemplateNotFound:
            logger.warning(f"Template {template_name} not found, using default")
            return self.env.get_template("report.html.jinja2")
    
    def generate_summary(self,
                        errors: List[Dict[str, Any]],
                        analysis_results: List[Dict[str, Any]],
                        output_path: str) -> str:
        """
        요약 리포트 생성 (YAML 형식)
        
        Args:
            errors: 원본 에러 목록
            analysis_results: 분석 결과 목록
            output_path: 출력 파일 경로
            
        Returns:
            생성된 요약 파일 경로
        """
        # 데이터 준비
        summary_data = self._prepare_summary_data(errors, analysis_results)
        
        # YAML 형식으로 저장
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(summary_data, f, allow_unicode=True, sort_keys=False)
            
        logger.info(f"Summary generated: {output_path}")
        return str(output_path)
    
    def _prepare_summary_data(self,
                            errors: List[Dict[str, Any]],
                            analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """요약 데이터 준비"""
        
        # 에러 요약
        error_summary = {
            'total_errors': len(errors),
            'severity_distribution': {},
            'component_distribution': {},
            'critical_errors': []
        }
        
        for error in errors:
            # 심각도별 분포
            severity = error.get('severity', 'unknown')
            error_summary['severity_distribution'][severity] = error_summary['severity_distribution'].get(severity, 0) + 1
            
            # 컴포넌트별 분포
            component = error.get('component', 'unknown')
            error_summary['component_distribution'][component] = error_summary['component_distribution'].get(component, 0) + 1
            
            # 심각한 에러 추출
            if severity in ['FATAL', 'ERROR']:
                error_summary['critical_errors'].append({
                    'component': component,
                    'message': error.get('message', ''),
                    'file': error.get('file_path', ''),
                    'line': error.get('line_number', '')
                })
        
        # 분석 요약
        analysis_summary = {
            'total_analyzed': len(analysis_results),
            'root_causes': {},
            'common_fixes': [],
            'prevention_guidelines': []
        }
        
        for result in analysis_results:
            # 근본 원인 요약
            root_cause = result.get('root_cause', {})
            if 'summary' in root_cause:
                cause = root_cause['summary']
                analysis_summary['root_causes'][cause] = analysis_summary['root_causes'].get(cause, 0) + 1
            
            # 일반적인 수정사항
            fixes = result.get('suggested_fixes', {}).get('common_fixes', [])
            analysis_summary['common_fixes'].extend(fixes)
            
            # 예방 가이드라인
            prevention = result.get('prevention', {}).get('guidelines', [])
            analysis_summary['prevention_guidelines'].extend(prevention)
        
        # 중복 제거
        analysis_summary['common_fixes'] = list(set(analysis_summary['common_fixes']))
        analysis_summary['prevention_guidelines'] = list(set(analysis_summary['prevention_guidelines']))
        
        return {
            'generated_at': datetime.now().isoformat(),
            'error_summary': error_summary,
            'analysis_summary': analysis_summary
        } 