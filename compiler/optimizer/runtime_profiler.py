"""
Runtime Profiling and Dynamic Optimization System
================================================

Provides runtime performance monitoring and adaptive optimization for
matrix operations and SIMD code generation. This system collects
execution statistics and dynamically adjusts optimization strategies.

Features:
- Real-time performance monitoring
- Adaptive optimization strategy selection
- Hot path detection and optimization
- Memory access pattern analysis
- Performance regression detection
- Optimization feedback loop
"""

import time
import threading
import json
import os
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics

from ..simd.simd_core import SIMDProcessor, DataType
from ..backend.simd_codegen import SIMDCodeGenerator, MatrixDimensions


class ProfilingEventType(Enum):
    """Types of profiling events"""
    FUNCTION_CALL = "function_call"
    MATRIX_MULTIPLY = "matrix_multiply"
    VECTOR_OPERATION = "vector_operation"
    MEMORY_ACCESS = "memory_access"
    CACHE_MISS = "cache_miss"
    OPTIMIZATION_APPLIED = "optimization_applied"


@dataclass
class ProfilingEvent:
    """Single profiling event record"""
    event_type: ProfilingEventType
    timestamp: float
    duration_ms: float
    function_name: str
    operation_details: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class FunctionProfile:
    """Profile data for a single function"""
    name: str
    call_count: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    hotness_score: float = 0.0
    
    # Matrix operation specific metrics
    matrix_ops_count: int = 0
    total_gflops: float = 0.0
    avg_gflops: float = 0.0
    
    # Optimization metrics
    vectorization_opportunities: int = 0
    optimizations_applied: int = 0
    optimization_benefit: float = 0.0
    
    # Recent performance history
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update(self, execution_time_ms: float, gflops: float = 0.0):
        """Update profile with new execution data"""
        self.call_count += 1
        self.total_time_ms += execution_time_ms
        self.avg_time_ms = self.total_time_ms / self.call_count
        self.min_time_ms = min(self.min_time_ms, execution_time_ms)
        self.max_time_ms = max(self.max_time_ms, execution_time_ms)
        self.recent_times.append(execution_time_ms)
        
        if gflops > 0:
            self.matrix_ops_count += 1
            self.total_gflops += gflops
            self.avg_gflops = self.total_gflops / self.matrix_ops_count
        
        # Update hotness score (weighted by frequency and total time)
        self.hotness_score = self.call_count * self.total_time_ms
    
    def get_trend(self) -> str:
        """Get performance trend (improving, degrading, stable)"""
        if len(self.recent_times) < 10:
            return "insufficient_data"
        
        recent_avg = statistics.mean(list(self.recent_times)[-10:])
        older_avg = statistics.mean(list(self.recent_times)[-20:-10]) if len(self.recent_times) >= 20 else recent_avg
        
        if recent_avg < older_avg * 0.95:
            return "improving"
        elif recent_avg > older_avg * 1.05:
            return "degrading"
        else:
            return "stable"


class RuntimeProfiler:
    """
    Runtime profiler that monitors performance and collects optimization data.
    
    Thread-safe profiler that can be integrated into the runtime system
    to collect real-time performance data for adaptive optimization.
    """
    
    def __init__(self, simd_processor: Optional[SIMDProcessor] = None):
        self.simd_processor = simd_processor or SIMDProcessor()
        
        # Profiling data storage
        self.function_profiles: Dict[str, FunctionProfile] = {}
        self.events: deque = deque(maxlen=10000)  # Keep recent events
        self.optimization_history: Dict[str, List[Dict]] = defaultdict(list)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Configuration
        self.profiling_enabled = True
        self.detailed_profiling = False  # More expensive profiling
        self.profile_memory = True
        self.profile_cache_behavior = False  # Very expensive
        
        # Hot path detection
        self.hot_functions: Set[str] = set()
        self.hot_path_threshold = 1000.0  # Hotness score threshold
        
        # Performance regression detection
        self.regression_threshold = 1.2  # 20% degradation threshold
        self.performance_alerts: List[Dict] = []
        
        # Start profiling thread
        self.profiling_thread = None
        self.should_stop = False
        self.start_profiling_thread()
    
    def start_profiling_thread(self):
        """Start background profiling thread"""
        if self.profiling_thread is None:
            self.profiling_thread = threading.Thread(
                target=self._profiling_loop,
                daemon=True,
                name="RuntimeProfiler"
            )
            self.profiling_thread.start()
    
    def stop_profiling(self):
        """Stop profiling and cleanup"""
        self.should_stop = True
        if self.profiling_thread:
            self.profiling_thread.join(timeout=1.0)
    
    def _profiling_loop(self):
        """Background profiling loop for periodic analysis"""
        while not self.should_stop:
            try:
                # Perform periodic analysis every 5 seconds
                time.sleep(5.0)
                
                with self._lock:
                    self._update_hot_paths()
                    self._detect_performance_regressions()
                    self._cleanup_old_data()
                
            except Exception as e:
                print(f"Profiling error: {e}")
    
    def record_function_call(self, function_name: str, 
                           execution_time_ms: float,
                           operation_details: Optional[Dict] = None) -> str:
        """
        Record a function call for profiling.
        
        Returns a profile ID that can be used for additional updates.
        """
        if not self.profiling_enabled:
            return ""
        
        with self._lock:
            # Get or create function profile
            if function_name not in self.function_profiles:
                self.function_profiles[function_name] = FunctionProfile(function_name)
            
            profile = self.function_profiles[function_name]
            profile.update(execution_time_ms)
            
            # Record event
            event = ProfilingEvent(
                event_type=ProfilingEventType.FUNCTION_CALL,
                timestamp=time.time(),
                duration_ms=execution_time_ms,
                function_name=function_name,
                operation_details=operation_details or {}
            )
            self.events.append(event)
            
            return f"{function_name}_{profile.call_count}"
    
    def record_matrix_operation(self, function_name: str,
                              execution_time_ms: float,
                              dimensions: Tuple[int, int, int],
                              gflops: float) -> None:
        """Record matrix operation performance data"""
        if not self.profiling_enabled:
            return
        
        m, k, n = dimensions
        
        with self._lock:
            # Update function profile
            if function_name not in self.function_profiles:
                self.function_profiles[function_name] = FunctionProfile(function_name)
            
            profile = self.function_profiles[function_name]
            profile.update(execution_time_ms, gflops)
            
            # Record detailed event
            event = ProfilingEvent(
                event_type=ProfilingEventType.MATRIX_MULTIPLY,
                timestamp=time.time(),
                duration_ms=execution_time_ms,
                function_name=function_name,
                operation_details={
                    'matrix_dimensions': (m, k, n),
                    'total_elements': m * k + k * n + m * n,
                    'computation_intensity': (2 * m * n * k) / (m * k + k * n + m * n)
                },
                performance_metrics={
                    'gflops': gflops,
                    'elements_per_second': (m * n) / (execution_time_ms / 1000.0),
                    'cache_efficiency': self._estimate_cache_efficiency(m, k, n)
                }
            )
            self.events.append(event)
    
    def record_optimization_applied(self, function_name: str,
                                  optimization_type: str,
                                  before_time_ms: float,
                                  after_time_ms: float,
                                  details: Optional[Dict] = None) -> None:
        """Record successful optimization application"""
        
        speedup = before_time_ms / max(after_time_ms, 0.001)
        benefit = before_time_ms - after_time_ms
        
        with self._lock:
            # Update function profile
            if function_name in self.function_profiles:
                profile = self.function_profiles[function_name]
                profile.optimizations_applied += 1
                profile.optimization_benefit += benefit
            
            # Record optimization history
            opt_record = {
                'timestamp': time.time(),
                'optimization_type': optimization_type,
                'speedup': speedup,
                'benefit_ms': benefit,
                'details': details or {}
            }
            self.optimization_history[function_name].append(opt_record)
            
            # Record event
            event = ProfilingEvent(
                event_type=ProfilingEventType.OPTIMIZATION_APPLIED,
                timestamp=time.time(),
                duration_ms=0,  # Instantaneous event
                function_name=function_name,
                operation_details={
                    'optimization_type': optimization_type,
                    'before_time_ms': before_time_ms,
                    'after_time_ms': after_time_ms,
                    **(details or {})
                },
                performance_metrics={
                    'speedup_factor': speedup,
                    'benefit_ms': benefit
                }
            )
            self.events.append(event)
    
    def get_hot_functions(self, top_n: int = 10) -> List[FunctionProfile]:
        """Get the hottest functions by execution time and frequency"""
        with self._lock:
            profiles = list(self.function_profiles.values())
            profiles.sort(key=lambda p: p.hotness_score, reverse=True)
            return profiles[:top_n]
    
    def get_optimization_candidates(self, min_calls: int = 10,
                                  min_time_ms: float = 1.0) -> List[FunctionProfile]:
        """Get functions that are good candidates for optimization"""
        candidates = []
        
        with self._lock:
            for profile in self.function_profiles.values():
                if (profile.call_count >= min_calls and 
                    profile.avg_time_ms >= min_time_ms and
                    profile.optimizations_applied == 0):  # Not yet optimized
                    candidates.append(profile)
        
        # Sort by potential benefit (hotness score)
        candidates.sort(key=lambda p: p.hotness_score, reverse=True)
        return candidates
    
    def get_performance_regression_alerts(self) -> List[Dict]:
        """Get current performance regression alerts"""
        with self._lock:
            return self.performance_alerts.copy()
    
    def _update_hot_paths(self):
        """Update hot path detection"""
        new_hot_functions = set()
        
        for profile in self.function_profiles.values():
            if profile.hotness_score >= self.hot_path_threshold:
                new_hot_functions.add(profile.name)
        
        # Detect newly hot functions
        newly_hot = new_hot_functions - self.hot_functions
        for func_name in newly_hot:
            print(f"🔥 Hot path detected: {func_name}")
        
        self.hot_functions = new_hot_functions
    
    def _detect_performance_regressions(self):
        """Detect performance regressions"""
        new_alerts = []
        
        for profile in self.function_profiles.values():
            trend = profile.get_trend()
            
            if trend == "degrading":
                recent_avg = statistics.mean(list(profile.recent_times)[-10:])
                baseline_avg = profile.avg_time_ms
                
                if recent_avg > baseline_avg * self.regression_threshold:
                    regression_factor = recent_avg / baseline_avg
                    alert = {
                        'timestamp': time.time(),
                        'function_name': profile.name,
                        'regression_factor': regression_factor,
                        'recent_avg_ms': recent_avg,
                        'baseline_avg_ms': baseline_avg,
                        'severity': 'high' if regression_factor > 1.5 else 'medium'
                    }
                    new_alerts.append(alert)
        
        # Add new alerts and keep recent ones
        self.performance_alerts.extend(new_alerts)
        self.performance_alerts = self.performance_alerts[-50:]  # Keep recent 50
    
    def _cleanup_old_data(self):
        """Clean up old profiling data to prevent memory growth"""
        current_time = time.time()
        cutoff_time = current_time - 3600  # Keep 1 hour of detailed events
        
        # Clean old events
        while self.events and self.events[0].timestamp < cutoff_time:
            self.events.popleft()
        
        # Clean old optimization history
        for func_name, history in self.optimization_history.items():
            self.optimization_history[func_name] = [
                record for record in history 
                if record['timestamp'] >= cutoff_time
            ]
    
    def _estimate_cache_efficiency(self, m: int, k: int, n: int) -> float:
        """Estimate cache efficiency for matrix operations"""
        # Rough approximation based on data size vs cache size
        data_size_bytes = (m * k + k * n + m * n) * 4  # float32
        l1_cache_size = getattr(self.simd_processor.capabilities, 'cache_sizes', {}).get('L1', 32 * 1024)
        
        if data_size_bytes <= l1_cache_size:
            return 0.95  # Excellent cache efficiency
        elif data_size_bytes <= l1_cache_size * 4:
            return 0.7   # Good cache efficiency  
        else:
            return 0.3   # Poor cache efficiency
    
    def export_profile_data(self, filepath: str) -> bool:
        """Export profiling data to file for analysis"""
        try:
            with self._lock:
                export_data = {
                    'timestamp': time.time(),
                    'function_profiles': {
                        name: {
                            'name': profile.name,
                            'call_count': profile.call_count,
                            'total_time_ms': profile.total_time_ms,
                            'avg_time_ms': profile.avg_time_ms,
                            'min_time_ms': profile.min_time_ms,
                            'max_time_ms': profile.max_time_ms,
                            'hotness_score': profile.hotness_score,
                            'matrix_ops_count': profile.matrix_ops_count,
                            'avg_gflops': profile.avg_gflops,
                            'optimizations_applied': profile.optimizations_applied,
                            'optimization_benefit': profile.optimization_benefit,
                            'trend': profile.get_trend()
                        }
                        for name, profile in self.function_profiles.items()
                    },
                    'optimization_history': dict(self.optimization_history),
                    'performance_alerts': self.performance_alerts,
                    'hot_functions': list(self.hot_functions),
                    'simd_capabilities': {
                        'instruction_sets': self.simd_processor.get_available_instruction_sets(),
                        'vector_width': self.simd_processor.get_vector_width(DataType.FLOAT32),
                        'cache_sizes': getattr(self.simd_processor.capabilities, 'cache_sizes', {})
                    }
                }
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
                    
                return True
                
        except Exception as e:
            print(f"Failed to export profile data: {e}")
            return False
    
    def import_profile_data(self, filepath: str) -> bool:
        """Import profiling data from file"""
        try:
            with open(filepath, 'r') as f:
                import_data = json.load(f)
            
            with self._lock:
                # Import function profiles
                for name, data in import_data.get('function_profiles', {}).items():
                    profile = FunctionProfile(name)
                    profile.call_count = data['call_count']
                    profile.total_time_ms = data['total_time_ms']
                    profile.avg_time_ms = data['avg_time_ms']
                    profile.min_time_ms = data['min_time_ms']
                    profile.max_time_ms = data['max_time_ms']
                    profile.hotness_score = data['hotness_score']
                    profile.matrix_ops_count = data.get('matrix_ops_count', 0)
                    profile.avg_gflops = data.get('avg_gflops', 0.0)
                    profile.optimizations_applied = data.get('optimizations_applied', 0)
                    profile.optimization_benefit = data.get('optimization_benefit', 0.0)
                    
                    self.function_profiles[name] = profile
                
                # Import optimization history
                self.optimization_history.update(import_data.get('optimization_history', {}))
                
                # Import alerts and hot functions
                self.performance_alerts = import_data.get('performance_alerts', [])
                self.hot_functions = set(import_data.get('hot_functions', []))
                
                return True
                
        except Exception as e:
            print(f"Failed to import profile data: {e}")
            return False


class AdaptiveOptimizer:
    """
    Adaptive optimizer that uses runtime profiling data to make optimization decisions.
    
    This system analyzes profiling data and dynamically adjusts optimization
    strategies based on actual runtime performance.
    """
    
    def __init__(self, profiler: RuntimeProfiler, simd_codegen: SIMDCodeGenerator):
        self.profiler = profiler
        self.simd_codegen = simd_codegen
        
        # Optimization strategies
        self.strategies = {
            'aggressive': {'min_benefit': 1.1, 'confidence_threshold': 0.3},
            'standard': {'min_benefit': 1.5, 'confidence_threshold': 0.5},
            'conservative': {'min_benefit': 2.0, 'confidence_threshold': 0.8}
        }
        
        self.current_strategy = 'standard'
        
        # Adaptation parameters
        self.adaptation_enabled = True
        self.min_samples_for_adaptation = 50
        
    def should_optimize_function(self, function_name: str) -> Tuple[bool, str]:
        """
        Determine if a function should be optimized based on profiling data.
        
        Returns: (should_optimize, reason)
        """
        
        profile = self.profiler.function_profiles.get(function_name)
        if not profile:
            return False, "no_profile_data"
        
        strategy = self.strategies[self.current_strategy]
        
        # Check if function is hot enough
        if profile.hotness_score < 100.0:
            return False, "not_hot_enough"
        
        # Check if already optimized
        if profile.optimizations_applied > 0:
            # Check if re-optimization is beneficial
            recent_trend = profile.get_trend()
            if recent_trend == "degrading":
                return True, "performance_regression_detected"
            else:
                return False, "already_optimized"
        
        # Check minimum call count
        if profile.call_count < 10:
            return False, "insufficient_samples"
        
        # Estimate potential benefit
        estimated_benefit = self._estimate_optimization_benefit(profile)
        
        if estimated_benefit >= strategy['min_benefit']:
            return True, f"estimated_benefit_{estimated_benefit:.2f}x"
        
        return False, f"insufficient_benefit_{estimated_benefit:.2f}x"
    
    def get_optimal_strategy_for_matrix(self, dimensions: Tuple[int, int, int],
                                      historical_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Get optimal optimization strategy for matrix operations based on dimensions and history.
        """
        
        m, k, n = dimensions
        
        # Base strategy from SIMD codegen
        base_perf = self.simd_codegen.estimate_performance(
            MatrixDimensions(m, k, n)
        )
        
        # Adjust based on historical performance if available
        if historical_data:
            # Analyze what worked well for similar matrices
            similar_ops = [
                op for op in historical_data 
                if self._matrices_are_similar(op.get('dimensions', (0,0,0)), dimensions)
            ]
            
            if similar_ops:
                # Find best performing strategy
                best_strategy = max(similar_ops, 
                                  key=lambda op: op.get('gflops', 0))
                
                return {
                    'use_vectorization': True,
                    'use_blocking': best_strategy.get('used_blocking', True),
                    'block_size': best_strategy.get('block_size', 64),
                    'use_strassen': best_strategy.get('used_strassen', False),
                    'expected_gflops': best_strategy.get('gflops', base_perf['estimated_gflops']),
                    'confidence': 0.8,
                    'source': 'historical_data'
                }
        
        # Fall back to theoretical estimation
        return {
            'use_vectorization': True,
            'use_blocking': m * k * n > 100000,
            'block_size': self.simd_codegen.optimal_block_size,
            'use_strassen': (m == k == n and m >= 1024 and (m & (m-1)) == 0),
            'expected_gflops': base_perf['estimated_gflops'],
            'confidence': 0.5,
            'source': 'theoretical_estimation'
        }
    
    def adapt_optimization_strategy(self) -> None:
        """Adapt optimization strategy based on recent performance data"""
        if not self.adaptation_enabled:
            return
        
        recent_optimizations = []
        current_time = time.time()
        
        # Collect recent optimization results
        for func_name, history in self.profiler.optimization_history.items():
            recent_opts = [
                opt for opt in history 
                if current_time - opt['timestamp'] < 3600  # Last hour
            ]
            recent_optimizations.extend(recent_opts)
        
        if len(recent_optimizations) < self.min_samples_for_adaptation:
            return
        
        # Analyze success rate of current strategy
        successful_opts = [opt for opt in recent_optimizations if opt['speedup'] >= 1.2]
        success_rate = len(successful_opts) / len(recent_optimizations)
        
        # Adapt strategy based on success rate
        if success_rate < 0.5:
            # Low success rate - become more conservative
            if self.current_strategy == 'aggressive':
                self.current_strategy = 'standard'
                print("📉 Adapting to standard optimization strategy")
            elif self.current_strategy == 'standard':
                self.current_strategy = 'conservative'
                print("📉 Adapting to conservative optimization strategy")
        
        elif success_rate > 0.8:
            # High success rate - become more aggressive
            if self.current_strategy == 'conservative':
                self.current_strategy = 'standard'
                print("📈 Adapting to standard optimization strategy")
            elif self.current_strategy == 'standard':
                self.current_strategy = 'aggressive'
                print("📈 Adapting to aggressive optimization strategy")
    
    def _estimate_optimization_benefit(self, profile: FunctionProfile) -> float:
        """Estimate potential optimization benefit for a function"""
        
        # Base benefit estimation
        base_benefit = 1.0
        
        # Matrix operations have higher vectorization potential
        if profile.matrix_ops_count > 0:
            base_benefit = 4.0  # Assume 4x speedup potential
        
        # Functions with poor current performance have more potential
        if profile.avg_gflops > 0 and profile.avg_gflops < 50:
            base_benefit *= 1.5  # More room for improvement
        
        # Hot functions justify more aggressive optimization
        if profile.hotness_score > 10000:
            base_benefit *= 1.2
        
        return base_benefit
    
    def _matrices_are_similar(self, dims1: Tuple[int, int, int], 
                            dims2: Tuple[int, int, int],
                            tolerance: float = 0.5) -> bool:
        """Check if two matrix dimension sets are similar"""
        
        m1, k1, n1 = dims1
        m2, k2, n2 = dims2
        
        # Compare relative sizes
        size1 = m1 * k1 * n1
        size2 = m2 * k2 * n2
        
        if size1 == 0 or size2 == 0:
            return False
        
        size_ratio = min(size1, size2) / max(size1, size2)
        return size_ratio >= tolerance


# Context manager for profiling function calls
class ProfiledExecution:
    """Context manager for profiling function execution"""
    
    def __init__(self, profiler: RuntimeProfiler, function_name: str, 
                 operation_details: Optional[Dict] = None):
        self.profiler = profiler
        self.function_name = function_name
        self.operation_details = operation_details
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            execution_time_ms = (time.perf_counter() - self.start_time) * 1000
            self.profiler.record_function_call(
                self.function_name, 
                execution_time_ms,
                self.operation_details
            )


# Factory function for creating profiler instances
def create_runtime_profiler(enable_detailed_profiling: bool = False,
                          enable_memory_profiling: bool = True,
                          simd_processor: Optional[SIMDProcessor] = None) -> RuntimeProfiler:
    """Create and configure a runtime profiler instance"""
    
    profiler = RuntimeProfiler(simd_processor)
    profiler.detailed_profiling = enable_detailed_profiling
    profiler.profile_memory = enable_memory_profiling
    
    return profiler
