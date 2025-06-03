#!/usr/bin/env python3
"""
Continuous Quality Monitoring System for Phase 3 Intelligence Layer

Provides real-time quality monitoring, performance tracking, and
automatic system health assessment for the RAG system.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import statistics
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class QualityMetric(Enum):
    """Quality metrics for monitoring"""
    RESPONSE_TIME = "response_time"
    CONFIDENCE_SCORE = "confidence_score"
    RELEVANCE_SCORE = "relevance_score"
    COMPLETENESS_SCORE = "completeness_score"
    USER_SATISFACTION = "user_satisfaction"
    ERROR_RATE = "error_rate"
    SYSTEM_AVAILABILITY = "system_availability"
    MEMORY_USAGE = "memory_usage"
    TOKEN_EFFICIENCY = "token_efficiency"
    COMPLIANCE_ACCURACY = "compliance_accuracy"

class AlertLevel(Enum):
    """Alert levels for quality monitoring"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class QualityMeasurement:
    """Individual quality measurement"""
    metric: QualityMetric
    value: float
    timestamp: datetime
    context: Dict[str, Any]
    session_id: Optional[str] = None
    query_id: Optional[str] = None

@dataclass
class QualityAlert:
    """Quality alert for threshold violations"""
    alert_id: str
    level: AlertLevel
    metric: QualityMetric
    current_value: float
    threshold: float
    message: str
    timestamp: datetime
    context: Dict[str, Any]
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None

@dataclass
class QualityReport:
    """Comprehensive quality report"""
    report_id: str
    start_time: datetime
    end_time: datetime
    total_queries: int
    avg_response_time: float
    avg_confidence: float
    error_rate: float
    user_satisfaction: float
    system_health_score: float
    recommendations: List[str]
    alerts: List[QualityAlert]
    detailed_metrics: Dict[str, Dict[str, float]]

class QualityThresholds:
    """Quality thresholds configuration"""
    
    def __init__(self):
        self.thresholds = {
            QualityMetric.RESPONSE_TIME: {
                'warning': 5.0,      # 5 seconds
                'critical': 10.0,    # 10 seconds
                'emergency': 30.0    # 30 seconds
            },
            QualityMetric.CONFIDENCE_SCORE: {
                'warning': 0.6,      # Below 60%
                'critical': 0.4,     # Below 40%
                'emergency': 0.2     # Below 20%
            },
            QualityMetric.ERROR_RATE: {
                'warning': 0.05,     # 5% error rate
                'critical': 0.10,    # 10% error rate
                'emergency': 0.20    # 20% error rate
            },
            QualityMetric.USER_SATISFACTION: {
                'warning': 0.7,      # Below 70%
                'critical': 0.5,     # Below 50%
                'emergency': 0.3     # Below 30%
            },
            QualityMetric.COMPLIANCE_ACCURACY: {
                'warning': 0.8,      # Below 80%
                'critical': 0.6,     # Below 60%
                'emergency': 0.4     # Below 40%
            },
            QualityMetric.SYSTEM_AVAILABILITY: {
                'warning': 0.95,     # Below 95%
                'critical': 0.90,    # Below 90%
                'emergency': 0.80    # Below 80%
            }
        }

class ContinuousQualityMonitor:
    """Continuous quality monitoring system"""
    
    def __init__(self, window_size_minutes: int = 60):
        self.window_size = timedelta(minutes=window_size_minutes)
        self.thresholds = QualityThresholds()
        
        # Data storage
        self.measurements = defaultdict(deque)  # metric -> deque of measurements
        self.alerts = deque(maxlen=1000)  # Recent alerts
        self.session_data = {}  # session_id -> session metrics
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0
        
        # Real-time metrics
        self.current_metrics = {}
        self.trend_analysis = {}
        
        # Background monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Callback for alerts
        self.alert_callbacks = []
        
    def start_monitoring(self):
        """Start continuous monitoring in background"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Continuous quality monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Continuous quality monitoring stopped")
    
    def record_measurement(self, metric: QualityMetric, value: float, 
                         context: Dict[str, Any] = None, 
                         session_id: str = None, query_id: str = None):
        """Record a quality measurement"""
        measurement = QualityMeasurement(
            metric=metric,
            value=value,
            timestamp=datetime.now(),
            context=context or {},
            session_id=session_id,
            query_id=query_id
        )
        
        # Add to measurements
        self.measurements[metric].append(measurement)
        
        # Update session data
        if session_id:
            if session_id not in self.session_data:
                self.session_data[session_id] = defaultdict(list)
            self.session_data[session_id][metric].append(value)
        
        # Check for threshold violations
        self._check_thresholds(measurement)
        
        # Update current metrics
        self._update_current_metrics(metric)
        
        # Clean old data
        self._cleanup_old_data()
    
    def record_query_execution(self, response_time: float, confidence: float, 
                             success: bool, context: Dict[str, Any] = None,
                             session_id: str = None, query_id: str = None):
        """Record complete query execution metrics"""
        self.total_queries += 1
        
        if success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1
        
        # Record individual metrics
        self.record_measurement(QualityMetric.RESPONSE_TIME, response_time, 
                              context, session_id, query_id)
        self.record_measurement(QualityMetric.CONFIDENCE_SCORE, confidence, 
                              context, session_id, query_id)
        
        # Calculate error rate
        error_rate = self.failed_queries / self.total_queries if self.total_queries > 0 else 0
        self.record_measurement(QualityMetric.ERROR_RATE, error_rate, 
                              context, session_id, query_id)
        
        # Calculate system availability
        availability = self.successful_queries / self.total_queries if self.total_queries > 0 else 1.0
        self.record_measurement(QualityMetric.SYSTEM_AVAILABILITY, availability, 
                              context, session_id, query_id)
    
    def record_user_feedback(self, satisfaction_score: float, query_id: str = None,
                           session_id: str = None, feedback_context: Dict[str, Any] = None):
        """Record user satisfaction feedback"""
        self.record_measurement(QualityMetric.USER_SATISFACTION, satisfaction_score,
                              feedback_context, session_id, query_id)
    
    def record_compliance_check(self, accuracy: float, context: Dict[str, Any] = None,
                              session_id: str = None, query_id: str = None):
        """Record compliance accuracy measurement"""
        self.record_measurement(QualityMetric.COMPLIANCE_ACCURACY, accuracy,
                              context, session_id, query_id)
    
    def _check_thresholds(self, measurement: QualityMeasurement):
        """Check if measurement violates quality thresholds"""
        metric_thresholds = self.thresholds.thresholds.get(measurement.metric)
        if not metric_thresholds:
            return
        
        # Determine alert level
        alert_level = None
        threshold_value = None
        
        # For metrics where lower is worse (confidence, satisfaction, availability)
        if measurement.metric in [QualityMetric.CONFIDENCE_SCORE, 
                                QualityMetric.USER_SATISFACTION,
                                QualityMetric.COMPLIANCE_ACCURACY,
                                QualityMetric.SYSTEM_AVAILABILITY]:
            if measurement.value < metric_thresholds['emergency']:
                alert_level = AlertLevel.EMERGENCY
                threshold_value = metric_thresholds['emergency']
            elif measurement.value < metric_thresholds['critical']:
                alert_level = AlertLevel.CRITICAL
                threshold_value = metric_thresholds['critical']
            elif measurement.value < metric_thresholds['warning']:
                alert_level = AlertLevel.WARNING
                threshold_value = metric_thresholds['warning']
        
        # For metrics where higher is worse (response time, error rate)
        else:
            if measurement.value > metric_thresholds['emergency']:
                alert_level = AlertLevel.EMERGENCY
                threshold_value = metric_thresholds['emergency']
            elif measurement.value > metric_thresholds['critical']:
                alert_level = AlertLevel.CRITICAL
                threshold_value = metric_thresholds['critical']
            elif measurement.value > metric_thresholds['warning']:
                alert_level = AlertLevel.WARNING
                threshold_value = metric_thresholds['warning']
        
        # Create alert if threshold violated
        if alert_level:
            alert = QualityAlert(
                alert_id=f"{measurement.metric.value}_{int(time.time())}",
                level=alert_level,
                metric=measurement.metric,
                current_value=measurement.value,
                threshold=threshold_value,
                message=self._generate_alert_message(measurement.metric, measurement.value, threshold_value, alert_level),
                timestamp=measurement.timestamp,
                context=measurement.context
            )
            
            self.alerts.append(alert)
            self._trigger_alert_callbacks(alert)
            
            logger.warning(f"Quality alert: {alert.message}")
    
    def _generate_alert_message(self, metric: QualityMetric, value: float, 
                              threshold: float, level: AlertLevel) -> str:
        """Generate human-readable alert message"""
        metric_name = metric.value.replace('_', ' ').title()
        
        if metric in [QualityMetric.CONFIDENCE_SCORE, QualityMetric.USER_SATISFACTION,
                     QualityMetric.COMPLIANCE_ACCURACY, QualityMetric.SYSTEM_AVAILABILITY]:
            return f"{level.value.upper()}: {metric_name} dropped to {value:.2%} (threshold: {threshold:.2%})"
        else:
            return f"{level.value.upper()}: {metric_name} increased to {value:.2f} (threshold: {threshold:.2f})"
    
    def _update_current_metrics(self, metric: QualityMetric):
        """Update current metric values and trends"""
        measurements = self.measurements[metric]
        if not measurements:
            return
        
        # Calculate current statistics
        recent_values = [m.value for m in measurements if 
                        datetime.now() - m.timestamp <= timedelta(minutes=10)]
        
        if recent_values:
            self.current_metrics[metric] = {
                'current': recent_values[-1],
                'avg_10min': statistics.mean(recent_values),
                'min_10min': min(recent_values),
                'max_10min': max(recent_values),
                'count_10min': len(recent_values)
            }
            
            # Trend analysis (comparing last 10 mins to previous 10 mins)
            if len(measurements) > len(recent_values):
                previous_values = [m.value for m in measurements if 
                                 timedelta(minutes=20) >= datetime.now() - m.timestamp > timedelta(minutes=10)]
                if previous_values:
                    current_avg = statistics.mean(recent_values)
                    previous_avg = statistics.mean(previous_values)
                    trend = (current_avg - previous_avg) / previous_avg * 100 if previous_avg != 0 else 0
                    self.trend_analysis[metric] = trend
    
    def _cleanup_old_data(self):
        """Remove measurements older than window size"""
        cutoff_time = datetime.now() - self.window_size
        
        for metric in self.measurements:
            while (self.measurements[metric] and 
                   self.measurements[metric][0].timestamp < cutoff_time):
                self.measurements[metric].popleft()
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Generate periodic health reports
                self._generate_health_check()
                
                # Check for anomalies
                self._detect_anomalies()
                
                # Sleep for monitoring interval
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _generate_health_check(self):
        """Generate periodic system health check"""
        health_score = self.calculate_system_health_score()
        
        context = {
            'total_queries': self.total_queries,
            'success_rate': self.successful_queries / self.total_queries if self.total_queries > 0 else 1.0,
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
        }
        
        self.record_measurement(QualityMetric.SYSTEM_AVAILABILITY, health_score, context)
    
    def _detect_anomalies(self):
        """Detect statistical anomalies in metrics"""
        for metric, measurements in self.measurements.items():
            if len(measurements) < 10:  # Need sufficient data
                continue
            
            values = [m.value for m in measurements]
            
            # Simple anomaly detection using standard deviation
            if len(values) >= 10:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values)
                
                # Check if latest value is anomalous (> 2 standard deviations)
                latest_value = values[-1]
                if abs(latest_value - mean_val) > 2 * std_val:
                    logger.warning(f"Anomaly detected in {metric.value}: {latest_value} (mean: {mean_val:.2f}, std: {std_val:.2f})")
    
    def calculate_system_health_score(self) -> float:
        """Calculate overall system health score (0-1)"""
        if not self.current_metrics:
            return 1.0  # Default to healthy if no data
        
        # Weight different metrics
        weights = {
            QualityMetric.RESPONSE_TIME: 0.2,
            QualityMetric.CONFIDENCE_SCORE: 0.3,
            QualityMetric.ERROR_RATE: 0.2,
            QualityMetric.USER_SATISFACTION: 0.2,
            QualityMetric.COMPLIANCE_ACCURACY: 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in self.current_metrics:
                metric_data = self.current_metrics[metric]
                
                # Normalize metric to 0-1 scale (higher is better)
                if metric == QualityMetric.RESPONSE_TIME:
                    # Invert response time (faster is better)
                    normalized = max(0, 1 - metric_data['avg_10min'] / 10.0)  # 10s = 0 score
                elif metric == QualityMetric.ERROR_RATE:
                    # Invert error rate (lower is better)
                    normalized = max(0, 1 - metric_data['avg_10min'])
                else:
                    # For confidence, satisfaction, compliance (higher is better)
                    normalized = min(1.0, metric_data['avg_10min'])
                
                total_score += normalized * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 1.0
    
    def get_quality_report(self, hours: int = 24) -> QualityReport:
        """Generate comprehensive quality report"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Filter measurements within time range
        period_measurements = {}
        for metric, measurements in self.measurements.items():
            period_measurements[metric] = [
                m for m in measurements 
                if start_time <= m.timestamp <= end_time
            ]
        
        # Calculate aggregated metrics
        detailed_metrics = {}
        for metric, measurements in period_measurements.items():
            if measurements:
                values = [m.value for m in measurements]
                detailed_metrics[metric.value] = {
                    'count': len(values),
                    'average': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'median': statistics.median(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0
                }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(detailed_metrics)
        
        # Get recent alerts
        recent_alerts = [alert for alert in self.alerts 
                        if start_time <= alert.timestamp <= end_time]
        
        # Calculate summary metrics
        response_times = [m.value for m in period_measurements.get(QualityMetric.RESPONSE_TIME, [])]
        confidences = [m.value for m in period_measurements.get(QualityMetric.CONFIDENCE_SCORE, [])]
        error_rates = [m.value for m in period_measurements.get(QualityMetric.ERROR_RATE, [])]
        satisfactions = [m.value for m in period_measurements.get(QualityMetric.USER_SATISFACTION, [])]
        
        report = QualityReport(
            report_id=f"quality_report_{int(time.time())}",
            start_time=start_time,
            end_time=end_time,
            total_queries=len(response_times),
            avg_response_time=statistics.mean(response_times) if response_times else 0.0,
            avg_confidence=statistics.mean(confidences) if confidences else 0.0,
            error_rate=error_rates[-1] if error_rates else 0.0,
            user_satisfaction=statistics.mean(satisfactions) if satisfactions else 0.0,
            system_health_score=self.calculate_system_health_score(),
            recommendations=recommendations,
            alerts=recent_alerts,
            detailed_metrics=detailed_metrics
        )
        
        return report
    
    def _generate_recommendations(self, detailed_metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate actionable recommendations based on metrics"""
        recommendations = []
        
        # Response time recommendations
        if 'response_time' in detailed_metrics:
            avg_response = detailed_metrics['response_time']['average']
            if avg_response > 5.0:
                recommendations.append("Response times are elevated. Consider optimizing retrieval algorithms or adding caching.")
            if avg_response > 10.0:
                recommendations.append("URGENT: Response times are critical. Investigate system bottlenecks immediately.")
        
        # Confidence recommendations
        if 'confidence_score' in detailed_metrics:
            avg_confidence = detailed_metrics['confidence_score']['average']
            if avg_confidence < 0.6:
                recommendations.append("Confidence scores are low. Review query expansion and retrieval quality.")
            if avg_confidence < 0.4:
                recommendations.append("CRITICAL: Confidence scores are very low. Model retraining may be required.")
        
        # Error rate recommendations
        if 'error_rate' in detailed_metrics:
            error_rate = detailed_metrics['error_rate']['max']  # Use max for worst case
            if error_rate > 0.05:
                recommendations.append("Error rate is elevated. Review error handling and system stability.")
            if error_rate > 0.1:
                recommendations.append("HIGH: Error rate is concerning. Investigate root causes immediately.")
        
        # User satisfaction recommendations
        if 'user_satisfaction' in detailed_metrics:
            avg_satisfaction = detailed_metrics['user_satisfaction']['average']
            if avg_satisfaction < 0.7:
                recommendations.append("User satisfaction is below target. Review response quality and relevance.")
        
        # General recommendations
        if not recommendations:
            recommendations.append("System performance is within normal parameters. Continue monitoring.")
        
        recommendations.append("Regular monitoring and proactive optimization recommended.")
        
        return recommendations
    
    def add_alert_callback(self, callback):
        """Add callback function to be called when alerts are triggered"""
        self.alert_callbacks.append(callback)
    
    def _trigger_alert_callbacks(self, alert: QualityAlert):
        """Trigger all registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics"""
        return {
            'current_metrics': self.current_metrics,
            'trend_analysis': self.trend_analysis,
            'system_health_score': self.calculate_system_health_score(),
            'total_queries': self.total_queries,
            'success_rate': self.successful_queries / self.total_queries if self.total_queries > 0 else 1.0,
            'active_alerts': len([a for a in self.alerts if not a.resolved]),
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
        }
    
    def export_metrics(self, filename: str):
        """Export metrics to JSON file"""
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_queries': self.total_queries,
            'successful_queries': self.successful_queries,
            'failed_queries': self.failed_queries,
            'measurements': {
                metric.value: [asdict(m) for m in measurements]
                for metric, measurements in self.measurements.items()
            },
            'alerts': [asdict(alert) for alert in self.alerts],
            'current_metrics': self.current_metrics,
            'trend_analysis': self.trend_analysis
        }
        
        # Convert datetime objects to strings
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=convert_datetime)
        
        logger.info(f"Metrics exported to {filename}")


# Usage example and integration
class QualityMonitorIntegration:
    """Integration helper for RAG system quality monitoring"""
    
    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        self.monitor = ContinuousQualityMonitor()
        
        # Set up alert callbacks
        self.monitor.add_alert_callback(self._handle_quality_alert)
        
        # Start monitoring
        self.monitor.start_monitoring()
    
    def _handle_quality_alert(self, alert: QualityAlert):
        """Handle quality alerts with automatic responses"""
        if alert.level == AlertLevel.CRITICAL:
            logger.critical(f"CRITICAL QUALITY ALERT: {alert.message}")
            # Potentially switch to backup models or reduce complexity
            
        elif alert.level == AlertLevel.WARNING:
            logger.warning(f"Quality warning: {alert.message}")
            # Log for review and potential optimization
    
    def monitor_query(self, query: str, response: Dict[str, Any], 
                     execution_time: float, session_id: str = None):
        """Monitor a single query execution"""
        query_id = f"query_{int(time.time() * 1000)}"
        
        # Record response time
        self.monitor.record_measurement(
            QualityMetric.RESPONSE_TIME, 
            execution_time,
            context={'query_length': len(query), 'response_length': len(str(response))},
            session_id=session_id,
            query_id=query_id
        )
        
        # Record confidence if available
        if 'confidence' in response:
            self.monitor.record_measurement(
                QualityMetric.CONFIDENCE_SCORE,
                response['confidence'],
                context={'query': query[:100]},  # First 100 chars for context
                session_id=session_id,
                query_id=query_id
            )
        
        # Record success/failure
        success = 'error' not in response
        self.monitor.record_query_execution(
            execution_time, 
            response.get('confidence', 0.0),
            success,
            context={'query_category': response.get('category', 'unknown')},
            session_id=session_id,
            query_id=query_id
        )
        
        return query_id
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        real_time = self.monitor.get_real_time_metrics()
        recent_report = self.monitor.get_quality_report(hours=1)
        
        return {
            'real_time_metrics': real_time,
            'hourly_summary': {
                'total_queries': recent_report.total_queries,
                'avg_response_time': recent_report.avg_response_time,
                'avg_confidence': recent_report.avg_confidence,
                'error_rate': recent_report.error_rate,
                'system_health': recent_report.system_health_score
            },
            'active_alerts': len([a for a in self.monitor.alerts if not a.resolved]),
            'recommendations': recent_report.recommendations[:3]  # Top 3 recommendations
        }