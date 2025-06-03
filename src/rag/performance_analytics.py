#!/usr/bin/env python3
"""
Performance Analytics Dashboard for Phase 3 Intelligence Layer

Provides comprehensive performance analytics, visual metrics, and
real-time monitoring capabilities for the RAG system.
"""

import time
import json
import statistics
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import math

logger = logging.getLogger(__name__)

class AnalyticsTimeframe(Enum):
    """Time frame options for analytics"""
    LAST_HOUR = "last_hour"
    LAST_4_HOURS = "last_4_hours"
    LAST_24_HOURS = "last_24_hours"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    ALL_TIME = "all_time"

class MetricType(Enum):
    """Types of metrics to analyze"""
    RESPONSE_TIME = "response_time"
    CONFIDENCE_SCORE = "confidence_score"
    USER_SATISFACTION = "user_satisfaction"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    SYSTEM_HEALTH = "system_health"
    ENGINE_PERFORMANCE = "engine_performance"
    QUERY_COMPLEXITY = "query_complexity"

@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""
    timestamp: datetime
    response_time: float
    confidence: float
    success: bool
    engine_used: str
    query_category: str
    query_intent: str
    user_satisfaction: Optional[float] = None
    complexity_score: Optional[float] = None

@dataclass
class TrendData:
    """Trend analysis data"""
    metric: str
    timeframe: AnalyticsTimeframe
    data_points: List[Tuple[datetime, float]]
    trend_direction: str  # "up", "down", "stable"
    trend_strength: float  # 0.0 to 1.0
    percentage_change: float
    statistical_significance: float

@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    report_id: str
    generated_at: datetime
    timeframe: AnalyticsTimeframe
    summary_metrics: Dict[str, float]
    trend_analysis: List[TrendData]
    engine_comparison: Dict[str, Dict[str, float]]
    category_performance: Dict[str, Dict[str, float]]
    recommendations: List[str]
    alerts: List[str]
    visualizations: Dict[str, Any]

class PerformanceAnalytics:
    """Advanced performance analytics and dashboard system"""
    
    def __init__(self, quality_monitor=None, meta_coordinator=None):
        self.quality_monitor = quality_monitor
        self.meta_coordinator = meta_coordinator
        
        # Historical data storage
        self.performance_snapshots = deque(maxlen=10000)  # Last 10k snapshots
        self.hourly_aggregates = defaultdict(dict)  # Hour -> metrics
        self.daily_aggregates = defaultdict(dict)   # Day -> metrics
        
        # Analytics configuration
        self.trend_sensitivity = 0.05  # 5% change threshold
        self.significance_threshold = 0.95  # 95% confidence
        
        # Cached analytics
        self._cached_reports = {}
        self._cache_expiry = {}
        
    def record_performance_snapshot(self, response_time: float, confidence: float,
                                  success: bool, engine_used: str,
                                  query_category: str, query_intent: str,
                                  user_satisfaction: float = None,
                                  complexity_score: float = None):
        """Record a performance snapshot"""
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            response_time=response_time,
            confidence=confidence,
            success=success,
            engine_used=engine_used,
            query_category=query_category,
            query_intent=query_intent,
            user_satisfaction=user_satisfaction,
            complexity_score=complexity_score
        )
        
        self.performance_snapshots.append(snapshot)
        self._update_aggregates(snapshot)
        
        # Clear relevant caches
        self._clear_expired_caches()
    
    def _update_aggregates(self, snapshot: PerformanceSnapshot):
        """Update hourly and daily aggregates"""
        
        # Hourly aggregates
        hour_key = snapshot.timestamp.replace(minute=0, second=0, microsecond=0)
        if hour_key not in self.hourly_aggregates:
            self.hourly_aggregates[hour_key] = {
                'count': 0,
                'response_times': [],
                'confidences': [],
                'successes': [],
                'engines': defaultdict(int),
                'categories': defaultdict(int),
                'satisfactions': []
            }
        
        hourly = self.hourly_aggregates[hour_key]
        hourly['count'] += 1
        hourly['response_times'].append(snapshot.response_time)
        hourly['confidences'].append(snapshot.confidence)
        hourly['successes'].append(snapshot.success)
        hourly['engines'][snapshot.engine_used] += 1
        hourly['categories'][snapshot.query_category] += 1
        
        if snapshot.user_satisfaction is not None:
            hourly['satisfactions'].append(snapshot.user_satisfaction)
        
        # Daily aggregates
        day_key = snapshot.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        if day_key not in self.daily_aggregates:
            self.daily_aggregates[day_key] = {
                'count': 0,
                'response_times': [],
                'confidences': [],
                'successes': [],
                'engines': defaultdict(int),
                'categories': defaultdict(int),
                'satisfactions': []
            }
        
        daily = self.daily_aggregates[day_key]
        daily['count'] += 1
        daily['response_times'].append(snapshot.response_time)
        daily['confidences'].append(snapshot.confidence)
        daily['successes'].append(snapshot.success)
        daily['engines'][snapshot.engine_used] += 1
        daily['categories'][snapshot.query_category] += 1
        
        if snapshot.user_satisfaction is not None:
            daily['satisfactions'].append(snapshot.user_satisfaction)
    
    def generate_performance_report(self, timeframe: AnalyticsTimeframe = AnalyticsTimeframe.LAST_24_HOURS) -> PerformanceReport:
        """Generate comprehensive performance report"""
        
        # Check cache
        cache_key = f"report_{timeframe.value}"
        if (cache_key in self._cached_reports and 
            cache_key in self._cache_expiry and
            datetime.now() < self._cache_expiry[cache_key]):
            return self._cached_reports[cache_key]
        
        # Get data for timeframe
        snapshots = self._get_snapshots_for_timeframe(timeframe)
        
        if not snapshots:
            return self._empty_report(timeframe)
        
        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(snapshots)
        
        # Perform trend analysis
        trend_analysis = self._analyze_trends(timeframe)
        
        # Engine comparison
        engine_comparison = self._compare_engines(snapshots)
        
        # Category performance
        category_performance = self._analyze_category_performance(snapshots)
        
        # Generate recommendations
        recommendations = self._generate_performance_recommendations(
            summary_metrics, trend_analysis, engine_comparison
        )
        
        # Generate alerts
        alerts = self._generate_performance_alerts(summary_metrics, trend_analysis)
        
        # Create visualizations
        visualizations = self._create_visualizations(snapshots, timeframe)
        
        report = PerformanceReport(
            report_id=f"perf_report_{int(time.time())}",
            generated_at=datetime.now(),
            timeframe=timeframe,
            summary_metrics=summary_metrics,
            trend_analysis=trend_analysis,
            engine_comparison=engine_comparison,
            category_performance=category_performance,
            recommendations=recommendations,
            alerts=alerts,
            visualizations=visualizations
        )
        
        # Cache report
        self._cached_reports[cache_key] = report
        self._cache_expiry[cache_key] = datetime.now() + timedelta(minutes=5)
        
        return report
    
    def _get_snapshots_for_timeframe(self, timeframe: AnalyticsTimeframe) -> List[PerformanceSnapshot]:
        """Get snapshots for specified timeframe"""
        
        now = datetime.now()
        
        if timeframe == AnalyticsTimeframe.LAST_HOUR:
            cutoff = now - timedelta(hours=1)
        elif timeframe == AnalyticsTimeframe.LAST_4_HOURS:
            cutoff = now - timedelta(hours=4)
        elif timeframe == AnalyticsTimeframe.LAST_24_HOURS:
            cutoff = now - timedelta(hours=24)
        elif timeframe == AnalyticsTimeframe.LAST_WEEK:
            cutoff = now - timedelta(days=7)
        elif timeframe == AnalyticsTimeframe.LAST_MONTH:
            cutoff = now - timedelta(days=30)
        else:  # ALL_TIME
            cutoff = datetime.min
        
        return [s for s in self.performance_snapshots if s.timestamp >= cutoff]
    
    def _calculate_summary_metrics(self, snapshots: List[PerformanceSnapshot]) -> Dict[str, float]:
        """Calculate summary metrics from snapshots"""
        
        if not snapshots:
            return {}
        
        response_times = [s.response_time for s in snapshots]
        confidences = [s.confidence for s in snapshots]
        successes = [s.success for s in snapshots]
        satisfactions = [s.user_satisfaction for s in snapshots if s.user_satisfaction is not None]
        
        metrics = {
            'total_queries': len(snapshots),
            'avg_response_time': statistics.mean(response_times),
            'median_response_time': statistics.median(response_times),
            'p95_response_time': self._percentile(response_times, 95),
            'p99_response_time': self._percentile(response_times, 99),
            'avg_confidence': statistics.mean(confidences),
            'median_confidence': statistics.median(confidences),
            'success_rate': sum(successes) / len(successes),
            'error_rate': 1 - (sum(successes) / len(successes)),
            'throughput_qps': len(snapshots) / ((snapshots[-1].timestamp - snapshots[0].timestamp).total_seconds() / 3600) if len(snapshots) > 1 else 0
        }
        
        if satisfactions:
            metrics.update({
                'avg_user_satisfaction': statistics.mean(satisfactions),
                'median_user_satisfaction': statistics.median(satisfactions)
            })
        
        # Add variance and standard deviation
        if len(response_times) > 1:
            metrics['response_time_std'] = statistics.stdev(response_times)
            metrics['confidence_std'] = statistics.stdev(confidences)
        
        return metrics
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100.0) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _analyze_trends(self, timeframe: AnalyticsTimeframe) -> List[TrendData]:
        """Analyze performance trends"""
        
        trends = []
        
        # Get appropriate aggregates based on timeframe
        if timeframe in [AnalyticsTimeframe.LAST_HOUR, AnalyticsTimeframe.LAST_4_HOURS]:
            # Use raw snapshots for short timeframes
            snapshots = self._get_snapshots_for_timeframe(timeframe)
            if len(snapshots) < 10:  # Need sufficient data
                return trends
            
            # Group by 5-minute intervals
            interval_data = self._group_by_intervals(snapshots, minutes=5)
        else:
            # Use hourly aggregates for longer timeframes
            interval_data = self._get_hourly_aggregates_for_timeframe(timeframe)
        
        if len(interval_data) < 3:  # Need at least 3 points for trend
            return trends
        
        # Analyze trends for each metric
        metrics_to_analyze = ['response_time', 'confidence', 'success_rate']
        
        for metric in metrics_to_analyze:
            trend_data = self._calculate_trend(interval_data, metric, timeframe)
            if trend_data:
                trends.append(trend_data)
        
        return trends
    
    def _group_by_intervals(self, snapshots: List[PerformanceSnapshot], 
                          minutes: int = 5) -> List[Tuple[datetime, Dict[str, float]]]:
        """Group snapshots by time intervals"""
        
        if not snapshots:
            return []
        
        intervals = {}
        
        for snapshot in snapshots:
            # Round to nearest interval
            interval_time = self._round_to_interval(snapshot.timestamp, minutes)
            
            if interval_time not in intervals:
                intervals[interval_time] = {
                    'response_times': [],
                    'confidences': [],
                    'successes': []
                }
            
            intervals[interval_time]['response_times'].append(snapshot.response_time)
            intervals[interval_time]['confidences'].append(snapshot.confidence)
            intervals[interval_time]['successes'].append(snapshot.success)
        
        # Calculate aggregates for each interval
        result = []
        for timestamp, data in sorted(intervals.items()):
            aggregated = {
                'response_time': statistics.mean(data['response_times']),
                'confidence': statistics.mean(data['confidences']),
                'success_rate': sum(data['successes']) / len(data['successes'])
            }
            result.append((timestamp, aggregated))
        
        return result
    
    def _round_to_interval(self, timestamp: datetime, minutes: int) -> datetime:
        """Round timestamp to nearest interval"""
        minute = (timestamp.minute // minutes) * minutes
        return timestamp.replace(minute=minute, second=0, microsecond=0)
    
    def _get_hourly_aggregates_for_timeframe(self, timeframe: AnalyticsTimeframe) -> List[Tuple[datetime, Dict[str, float]]]:
        """Get hourly aggregates for timeframe"""
        
        snapshots = self._get_snapshots_for_timeframe(timeframe)
        if not snapshots:
            return []
        
        start_time = snapshots[0].timestamp
        end_time = snapshots[-1].timestamp
        
        result = []
        for timestamp, data in self.hourly_aggregates.items():
            if start_time <= timestamp <= end_time and data['count'] > 0:
                aggregated = {
                    'response_time': statistics.mean(data['response_times']),
                    'confidence': statistics.mean(data['confidences']),
                    'success_rate': sum(data['successes']) / len(data['successes'])
                }
                result.append((timestamp, aggregated))
        
        return sorted(result)
    
    def _calculate_trend(self, interval_data: List[Tuple[datetime, Dict[str, float]]], 
                        metric: str, timeframe: AnalyticsTimeframe) -> Optional[TrendData]:
        """Calculate trend for a specific metric"""
        
        if len(interval_data) < 3:
            return None
        
        # Extract data points
        data_points = [(timestamp, data[metric]) for timestamp, data in interval_data if metric in data]
        
        if len(data_points) < 3:
            return None
        
        # Calculate linear regression
        x_values = [(dp[0] - data_points[0][0]).total_seconds() for dp in data_points]
        y_values = [dp[1] for dp in data_points]
        
        slope, correlation = self._linear_regression(x_values, y_values)
        
        # Determine trend direction and strength
        if abs(slope) < self.trend_sensitivity:
            direction = "stable"
            strength = 0.0
        elif slope > 0:
            direction = "up"
            strength = min(1.0, abs(correlation))
        else:
            direction = "down"
            strength = min(1.0, abs(correlation))
        
        # Calculate percentage change
        if len(y_values) >= 2:
            first_half = statistics.mean(y_values[:len(y_values)//2])
            second_half = statistics.mean(y_values[len(y_values)//2:])
            percentage_change = ((second_half - first_half) / first_half * 100) if first_half != 0 else 0
        else:
            percentage_change = 0
        
        # Statistical significance (simplified)
        significance = abs(correlation) if abs(correlation) > 0.5 else 0.0
        
        return TrendData(
            metric=metric,
            timeframe=timeframe,
            data_points=data_points,
            trend_direction=direction,
            trend_strength=strength,
            percentage_change=percentage_change,
            statistical_significance=significance
        )
    
    def _linear_regression(self, x_values: List[float], y_values: List[float]) -> Tuple[float, float]:
        """Calculate linear regression slope and correlation"""
        
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0, 0.0
        
        n = len(x_values)
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(y_values)
        
        # Calculate slope
        numerator = sum((x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0.0
        
        # Calculate correlation coefficient
        x_std = statistics.stdev(x_values) if len(x_values) > 1 else 1.0
        y_std = statistics.stdev(y_values) if len(y_values) > 1 else 1.0
        
        correlation = (numerator / (n - 1)) / (x_std * y_std) if x_std != 0 and y_std != 0 else 0.0
        
        return slope, correlation
    
    def _compare_engines(self, snapshots: List[PerformanceSnapshot]) -> Dict[str, Dict[str, float]]:
        """Compare performance across different engines"""
        
        engine_data = defaultdict(lambda: {
            'response_times': [],
            'confidences': [],
            'successes': [],
            'satisfactions': []
        })
        
        for snapshot in snapshots:
            engine_data[snapshot.engine_used]['response_times'].append(snapshot.response_time)
            engine_data[snapshot.engine_used]['confidences'].append(snapshot.confidence)
            engine_data[snapshot.engine_used]['successes'].append(snapshot.success)
            
            if snapshot.user_satisfaction is not None:
                engine_data[snapshot.engine_used]['satisfactions'].append(snapshot.user_satisfaction)
        
        # Calculate metrics for each engine
        comparison = {}
        for engine, data in engine_data.items():
            if not data['response_times']:  # Skip engines with no data
                continue
            
            metrics = {
                'query_count': len(data['response_times']),
                'avg_response_time': statistics.mean(data['response_times']),
                'avg_confidence': statistics.mean(data['confidences']),
                'success_rate': sum(data['successes']) / len(data['successes']),
                'p95_response_time': self._percentile(data['response_times'], 95)
            }
            
            if data['satisfactions']:
                metrics['avg_satisfaction'] = statistics.mean(data['satisfactions'])
            
            # Calculate efficiency score (higher is better)
            confidence_score = metrics['avg_confidence']
            speed_score = max(0, 1 - metrics['avg_response_time'] / 10.0)  # 10s = 0 score
            success_score = metrics['success_rate']
            
            metrics['efficiency_score'] = (confidence_score + speed_score + success_score) / 3
            
            comparison[engine] = metrics
        
        return comparison
    
    def _analyze_category_performance(self, snapshots: List[PerformanceSnapshot]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by query category"""
        
        category_data = defaultdict(lambda: {
            'response_times': [],
            'confidences': [],
            'successes': [],
            'satisfactions': []
        })
        
        for snapshot in snapshots:
            category_data[snapshot.query_category]['response_times'].append(snapshot.response_time)
            category_data[snapshot.query_category]['confidences'].append(snapshot.confidence)
            category_data[snapshot.query_category]['successes'].append(snapshot.success)
            
            if snapshot.user_satisfaction is not None:
                category_data[snapshot.query_category]['satisfactions'].append(snapshot.user_satisfaction)
        
        # Calculate metrics for each category
        analysis = {}
        for category, data in category_data.items():
            if not data['response_times']:
                continue
            
            metrics = {
                'query_count': len(data['response_times']),
                'avg_response_time': statistics.mean(data['response_times']),
                'avg_confidence': statistics.mean(data['confidences']),
                'success_rate': sum(data['successes']) / len(data['successes'])
            }
            
            if data['satisfactions']:
                metrics['avg_satisfaction'] = statistics.mean(data['satisfactions'])
            
            analysis[category] = metrics
        
        return analysis
    
    def _generate_performance_recommendations(self, summary_metrics: Dict[str, float],
                                            trend_analysis: List[TrendData],
                                            engine_comparison: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate performance improvement recommendations"""
        
        recommendations = []
        
        # Response time recommendations
        avg_response_time = summary_metrics.get('avg_response_time', 0)
        if avg_response_time > 5.0:
            recommendations.append(f"Response time averaging {avg_response_time:.2f}s is elevated. Consider caching frequently accessed content.")
        
        p95_response_time = summary_metrics.get('p95_response_time', 0)
        if p95_response_time > 10.0:
            recommendations.append(f"95th percentile response time ({p95_response_time:.2f}s) is high. Investigate slow queries and optimize retrieval.")
        
        # Confidence recommendations
        avg_confidence = summary_metrics.get('avg_confidence', 0)
        if avg_confidence < 0.7:
            recommendations.append(f"Average confidence ({avg_confidence:.2%}) is below target. Review query expansion and model selection.")
        
        # Error rate recommendations
        error_rate = summary_metrics.get('error_rate', 0)
        if error_rate > 0.05:
            recommendations.append(f"Error rate ({error_rate:.2%}) is elevated. Review error handling and system stability.")
        
        # Trend-based recommendations
        for trend in trend_analysis:
            if trend.trend_direction == "down" and trend.trend_strength > 0.5:
                if trend.metric == "confidence":
                    recommendations.append(f"Confidence scores trending down ({trend.percentage_change:.1f}%). Monitor model performance.")
                elif trend.metric == "success_rate":
                    recommendations.append(f"Success rate declining ({trend.percentage_change:.1f}%). Investigate recent failures.")
            elif trend.trend_direction == "up" and trend.metric == "response_time" and trend.trend_strength > 0.5:
                recommendations.append(f"Response times trending up ({trend.percentage_change:.1f}%). Monitor system load and performance.")
        
        # Engine comparison recommendations
        if len(engine_comparison) > 1:
            # Find best and worst performing engines
            engines_by_efficiency = sorted(engine_comparison.items(), 
                                         key=lambda x: x[1].get('efficiency_score', 0), reverse=True)
            
            if len(engines_by_efficiency) >= 2:
                best_engine = engines_by_efficiency[0]
                worst_engine = engines_by_efficiency[-1]
                
                efficiency_gap = best_engine[1]['efficiency_score'] - worst_engine[1]['efficiency_score']
                if efficiency_gap > 0.2:  # 20% efficiency gap
                    recommendations.append(f"Consider routing more queries to {best_engine[0]} (efficiency: {best_engine[1]['efficiency_score']:.2%}) vs {worst_engine[0]} (efficiency: {worst_engine[1]['efficiency_score']:.2%})")
        
        # General recommendations
        if not recommendations:
            recommendations.append("System performance is within acceptable parameters. Continue monitoring for optimization opportunities.")
        
        recommendations.append("Regular performance review and optimization recommended.")
        
        return recommendations
    
    def _generate_performance_alerts(self, summary_metrics: Dict[str, float],
                                   trend_analysis: List[TrendData]) -> List[str]:
        """Generate performance alerts"""
        
        alerts = []
        
        # Critical thresholds
        avg_response_time = summary_metrics.get('avg_response_time', 0)
        if avg_response_time > 15.0:
            alerts.append(f"CRITICAL: Average response time ({avg_response_time:.2f}s) exceeds 15 seconds")
        
        error_rate = summary_metrics.get('error_rate', 0)
        if error_rate > 0.2:
            alerts.append(f"CRITICAL: Error rate ({error_rate:.2%}) exceeds 20%")
        
        avg_confidence = summary_metrics.get('avg_confidence', 0)
        if avg_confidence < 0.4:
            alerts.append(f"CRITICAL: Average confidence ({avg_confidence:.2%}) below 40%")
        
        # Warning thresholds
        if 10.0 < avg_response_time <= 15.0:
            alerts.append(f"WARNING: Average response time ({avg_response_time:.2f}s) elevated")
        
        if 0.1 < error_rate <= 0.2:
            alerts.append(f"WARNING: Error rate ({error_rate:.2%}) elevated")
        
        if 0.4 <= avg_confidence < 0.6:
            alerts.append(f"WARNING: Average confidence ({avg_confidence:.2%}) low")
        
        # Trend alerts
        for trend in trend_analysis:
            if trend.trend_strength > 0.7 and abs(trend.percentage_change) > 20:
                direction = "increasing" if trend.trend_direction == "up" else "decreasing"
                alerts.append(f"TREND ALERT: {trend.metric} {direction} by {abs(trend.percentage_change):.1f}%")
        
        return alerts
    
    def _create_visualizations(self, snapshots: List[PerformanceSnapshot],
                             timeframe: AnalyticsTimeframe) -> Dict[str, Any]:
        """Create visualization data for dashboard"""
        
        visualizations = {}
        
        # Time series data for charts
        time_series = self._create_time_series_data(snapshots)
        visualizations['time_series'] = time_series
        
        # Distribution histograms
        histograms = self._create_histograms(snapshots)
        visualizations['histograms'] = histograms
        
        # Engine performance radar chart
        engine_radar = self._create_engine_radar_data(snapshots)
        visualizations['engine_radar'] = engine_radar
        
        # Category performance pie chart
        category_pie = self._create_category_pie_data(snapshots)
        visualizations['category_pie'] = category_pie
        
        # Heatmap data for hourly patterns
        heatmap = self._create_hourly_heatmap(snapshots)
        visualizations['hourly_heatmap'] = heatmap
        
        return visualizations
    
    def _create_time_series_data(self, snapshots: List[PerformanceSnapshot]) -> Dict[str, List]:
        """Create time series data for line charts"""
        
        if not snapshots:
            return {}
        
        # Group by 5-minute intervals
        intervals = self._group_by_intervals(snapshots, minutes=5)
        
        timestamps = [interval[0].isoformat() for interval in intervals]
        response_times = [interval[1]['response_time'] for interval in intervals]
        confidences = [interval[1]['confidence'] for interval in intervals]
        success_rates = [interval[1]['success_rate'] for interval in intervals]
        
        return {
            'timestamps': timestamps,
            'response_time': response_times,
            'confidence': confidences,
            'success_rate': success_rates
        }
    
    def _create_histograms(self, snapshots: List[PerformanceSnapshot]) -> Dict[str, Dict]:
        """Create histogram data for distributions"""
        
        if not snapshots:
            return {}
        
        response_times = [s.response_time for s in snapshots]
        confidences = [s.confidence for s in snapshots]
        
        return {
            'response_time': self._create_histogram_bins(response_times, 20),
            'confidence': self._create_histogram_bins(confidences, 20)
        }
    
    def _create_histogram_bins(self, data: List[float], num_bins: int) -> Dict:
        """Create histogram bins for data"""
        
        if not data:
            return {'bins': [], 'counts': []}
        
        min_val = min(data)
        max_val = max(data)
        bin_width = (max_val - min_val) / num_bins
        
        bins = []
        counts = []
        
        for i in range(num_bins):
            bin_start = min_val + i * bin_width
            bin_end = min_val + (i + 1) * bin_width
            
            count = sum(1 for x in data if bin_start <= x < bin_end)
            
            bins.append(f"{bin_start:.2f}-{bin_end:.2f}")
            counts.append(count)
        
        return {'bins': bins, 'counts': counts}
    
    def _create_engine_radar_data(self, snapshots: List[PerformanceSnapshot]) -> Dict:
        """Create radar chart data for engine comparison"""
        
        engine_data = defaultdict(lambda: {
            'response_times': [],
            'confidences': [],
            'successes': []
        })
        
        for snapshot in snapshots:
            engine_data[snapshot.engine_used]['response_times'].append(snapshot.response_time)
            engine_data[snapshot.engine_used]['confidences'].append(snapshot.confidence)
            engine_data[snapshot.engine_used]['successes'].append(snapshot.success)
        
        radar_data = {}
        for engine, data in engine_data.items():
            if not data['response_times']:
                continue
            
            # Normalize metrics to 0-1 scale
            speed_score = max(0, 1 - statistics.mean(data['response_times']) / 10.0)
            confidence_score = statistics.mean(data['confidences'])
            reliability_score = sum(data['successes']) / len(data['successes'])
            
            radar_data[engine] = {
                'speed': speed_score,
                'confidence': confidence_score,
                'reliability': reliability_score
            }
        
        return radar_data
    
    def _create_category_pie_data(self, snapshots: List[PerformanceSnapshot]) -> Dict:
        """Create pie chart data for category distribution"""
        
        category_counts = defaultdict(int)
        for snapshot in snapshots:
            category_counts[snapshot.query_category] += 1
        
        return {
            'labels': list(category_counts.keys()),
            'values': list(category_counts.values())
        }
    
    def _create_hourly_heatmap(self, snapshots: List[PerformanceSnapshot]) -> Dict:
        """Create heatmap data for hourly patterns"""
        
        # Create 24x7 grid (hour x day of week)
        heatmap_data = [[0 for _ in range(7)] for _ in range(24)]
        
        for snapshot in snapshots:
            hour = snapshot.timestamp.hour
            day_of_week = snapshot.timestamp.weekday()
            heatmap_data[hour][day_of_week] += 1
        
        return {
            'data': heatmap_data,
            'x_labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            'y_labels': [f"{h:02d}:00" for h in range(24)]
        }
    
    def _empty_report(self, timeframe: AnalyticsTimeframe) -> PerformanceReport:
        """Create empty report when no data available"""
        
        return PerformanceReport(
            report_id=f"empty_report_{int(time.time())}",
            generated_at=datetime.now(),
            timeframe=timeframe,
            summary_metrics={},
            trend_analysis=[],
            engine_comparison={},
            category_performance={},
            recommendations=["No data available for the selected timeframe."],
            alerts=[],
            visualizations={}
        )
    
    def _clear_expired_caches(self):
        """Clear expired cached reports"""
        
        now = datetime.now()
        expired_keys = [key for key, expiry in self._cache_expiry.items() if now >= expiry]
        
        for key in expired_keys:
            if key in self._cached_reports:
                del self._cached_reports[key]
            del self._cache_expiry[key]
    
    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time data for dashboard display"""
        
        # Get recent snapshots (last hour)
        recent_snapshots = self._get_snapshots_for_timeframe(AnalyticsTimeframe.LAST_HOUR)
        
        if not recent_snapshots:
            return {
                'status': 'no_data',
                'message': 'No recent performance data available'
            }
        
        # Calculate current metrics
        current_metrics = self._calculate_summary_metrics(recent_snapshots[-10:])  # Last 10 queries
        
        # Get trends
        trends = self._analyze_trends(AnalyticsTimeframe.LAST_HOUR)
        
        # Real-time status
        latest_snapshot = recent_snapshots[-1]
        status = "healthy"
        if latest_snapshot.response_time > 10:
            status = "degraded"
        if not latest_snapshot.success:
            status = "error"
        
        return {
            'status': status,
            'last_updated': datetime.now().isoformat(),
            'current_metrics': current_metrics,
            'trends': [asdict(trend) for trend in trends],
            'recent_activity': len(recent_snapshots),
            'system_health': {
                'response_time': latest_snapshot.response_time,
                'confidence': latest_snapshot.confidence,
                'success': latest_snapshot.success,
                'engine': latest_snapshot.engine_used
            }
        }
    
    def export_analytics_data(self, filename: str, timeframe: AnalyticsTimeframe = AnalyticsTimeframe.LAST_24_HOURS):
        """Export analytics data to JSON file"""
        
        report = self.generate_performance_report(timeframe)
        snapshots = self._get_snapshots_for_timeframe(timeframe)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'timeframe': timeframe.value,
            'report': asdict(report),
            'raw_snapshots': [asdict(s) for s in snapshots],
            'aggregates': {
                'hourly': {k.isoformat(): v for k, v in self.hourly_aggregates.items()},
                'daily': {k.isoformat(): v for k, v in self.daily_aggregates.items()}
            }
        }
        
        # Convert datetime objects to strings
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=convert_datetime)
        
        logger.info(f"Analytics data exported to {filename}")


# Integration helper
class AnalyticsIntegration:
    """Integration helper for RAG system analytics"""
    
    def __init__(self, quality_monitor=None, meta_coordinator=None):
        self.analytics = PerformanceAnalytics(quality_monitor, meta_coordinator)
        
    def track_query_performance(self, query: str, result: Dict[str, Any], 
                              execution_time: float, engine_used: str,
                              session_id: str = None):
        """Track query performance for analytics"""
        
        # Extract metadata from result
        confidence = result.get('confidence', 0.0)
        success = 'error' not in result
        category = result.get('category', 'unknown')
        intent = result.get('intent', 'unknown')
        
        # Record snapshot
        self.analytics.record_performance_snapshot(
            response_time=execution_time,
            confidence=confidence,
            success=success,
            engine_used=engine_used,
            query_category=category,
            query_intent=intent
        )
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        
        real_time = self.analytics.get_real_time_dashboard_data()
        report_24h = self.analytics.generate_performance_report(AnalyticsTimeframe.LAST_24_HOURS)
        
        return {
            'real_time': real_time,
            'daily_summary': {
                'total_queries': report_24h.summary_metrics.get('total_queries', 0),
                'avg_response_time': report_24h.summary_metrics.get('avg_response_time', 0),
                'avg_confidence': report_24h.summary_metrics.get('avg_confidence', 0),
                'success_rate': report_24h.summary_metrics.get('success_rate', 1.0),
                'system_health': len(report_24h.alerts) == 0
            },
            'visualizations': report_24h.visualizations,
            'recommendations': report_24h.recommendations[:5],
            'alerts': report_24h.alerts
        }