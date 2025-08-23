"""
JIT Runtime Profiler for NeuralScript
=====================================

Advanced runtime profiling system that identifies hot code paths and 
functions suitable for JIT compilation. Tracks execution frequencies,
timing data, and provides intelligent compilation decisions.

Features:
- Function call frequency tracking
- Execution time measurement
- Hot path detection algorithms
- Adaptive threshold adjustment
- Memory-efficient profiling with minimal overhead
- Integration with existing memory management and SIMD systems
"""

import time
import threading
from typing import Dict, List, Set, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum, auto
import statistics
import weakref


class HotspotCategory(Enum):
    """Categories of hot code paths"""
    FUNCTION_CALL = auto()      # Frequently called functions
    LOOP_BODY = auto()          # Inner loop bodies
    MATH_OPERATION = auto()     # Mathematical computations
    MATRIX_OPERATION = auto()   # Matrix/tensor operations
    MEMORY_INTENSIVE = auto()   # Memory allocation heavy code
    SIMD_CANDIDATE = auto()     # Code suitable for SIMD optimization
    COMPUTE_INTENSIVE = auto()  # CPU-intensive computations


@dataclass
class ExecutionSample:
    """Single execution measurement sample"""
    timestamp: float
    execution_time_ns: int
    call_count: int = 1
    memory_allocations: int = 0
    simd_operations: int = 0
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass 
class FunctionProfile:
    """Comprehensive profile data for a function or code block"""
    name: str
    total_calls: int = 0
    total_execution_time_ns: int = 0
    recent_samples: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Performance metrics
    average_execution_time_ns: float = 0.0
    calls_per_second: float = 0.0
    peak_calls_per_second: float = 0.0
    
    # Categorization
    hotspot_categories: Set[HotspotCategory] = field(default_factory=set)
    jit_eligibility_score: float = 0.0
    
    # JIT compilation tracking
    is_jit_compiled: bool = False
    jit_compilation_time: Optional[float] = None
    jit_speedup_factor: float = 1.0
    
    # Code characteristics
    has_loops: bool = False
    has_math_ops: bool = False
    has_matrix_ops: bool = False
    memory_allocation_rate: float = 0.0
    simd_potential: float = 0.0
    
    def update_sample(self, sample: ExecutionSample):
        """Update profile with new execution sample"""
        self.total_calls += sample.call_count
        self.total_execution_time_ns += sample.execution_time_ns
        self.recent_samples.append(sample)
        
        # Update derived metrics
        self._recalculate_metrics()
        self._update_categorization(sample)
    
    def _recalculate_metrics(self):
        """Recalculate derived performance metrics"""
        if self.total_calls > 0:
            self.average_execution_time_ns = self.total_execution_time_ns / self.total_calls
        
        if len(self.recent_samples) >= 2:
            # Calculate calls per second using recent samples
            recent_window = 10.0  # seconds
            cutoff_time = time.time() - recent_window
            
            recent_calls = sum(
                sample.call_count for sample in self.recent_samples 
                if sample.timestamp >= cutoff_time
            )
            
            if recent_calls > 0:
                self.calls_per_second = recent_calls / recent_window
                self.peak_calls_per_second = max(self.peak_calls_per_second, self.calls_per_second)
    
    def _update_categorization(self, sample: ExecutionSample):
        """Update hotspot categorization based on sample data"""
        # High frequency function calls
        if self.calls_per_second > 100:  # 100+ calls per second
            self.hotspot_categories.add(HotspotCategory.FUNCTION_CALL)
        
        # Memory intensive operations
        if sample.memory_allocations > 10:
            self.hotspot_categories.add(HotspotCategory.MEMORY_INTENSIVE)
            self.memory_allocation_rate = sample.memory_allocations / max(sample.execution_time_ns, 1)
        
        # SIMD candidates
        if sample.simd_operations > 0:
            self.hotspot_categories.add(HotspotCategory.SIMD_CANDIDATE)
            self.simd_potential = min(1.0, sample.simd_operations / 10.0)
        
        # Matrix operations (high SIMD potential)
        if self.has_matrix_ops:
            self.hotspot_categories.add(HotspotCategory.MATRIX_OPERATION)
            self.simd_potential = max(self.simd_potential, 0.8)
        
        # Mathematical operations
        if self.has_math_ops:
            self.hotspot_categories.add(HotspotCategory.MATH_OPERATION)
        
        # Loop bodies (high optimization potential)
        if self.has_loops and self.calls_per_second > 1000:
            self.hotspot_categories.add(HotspotCategory.LOOP_BODY)
        
        # Calculate JIT eligibility score
        self._calculate_jit_eligibility()
    
    def _calculate_jit_eligibility(self):
        """Calculate overall JIT compilation eligibility score (0-1)"""
        score = 0.0
        
        # Frequency component (0-0.4)
        frequency_score = min(0.4, self.calls_per_second / 1000.0)
        score += frequency_score
        
        # Execution time component (0-0.3)
        if self.average_execution_time_ns > 1_000_000:  # > 1ms
            time_score = min(0.3, self.average_execution_time_ns / 10_000_000)  # Scale to 10ms
            score += time_score
        
        # Category bonuses (0-0.3)
        category_bonus = 0.0
        if HotspotCategory.LOOP_BODY in self.hotspot_categories:
            category_bonus += 0.1
        if HotspotCategory.MATH_OPERATION in self.hotspot_categories:
            category_bonus += 0.1
        if HotspotCategory.SIMD_CANDIDATE in self.hotspot_categories:
            category_bonus += 0.1
        
        score += category_bonus
        
        self.jit_eligibility_score = min(1.0, score)
    
    def should_jit_compile(self, threshold: float = 0.6) -> bool:
        """Determine if this function should be JIT compiled"""
        return (
            not self.is_jit_compiled and
            self.jit_eligibility_score >= threshold and
            self.total_calls > 50  # Minimum sample size
        )
    
    def get_optimization_hints(self) -> List[str]:
        """Get optimization hints for JIT compilation"""
        hints = []
        
        if HotspotCategory.SIMD_CANDIDATE in self.hotspot_categories:
            hints.append("vectorize_loops")
            hints.append("enable_simd")
        
        if HotspotCategory.MATRIX_OPERATION in self.hotspot_categories:
            hints.append("optimize_matrix_ops")
            hints.append("cache_blocking")
        
        if HotspotCategory.MEMORY_INTENSIVE in self.hotspot_categories:
            hints.append("optimize_allocations")
            hints.append("memory_prefetch")
        
        if HotspotCategory.LOOP_BODY in self.hotspot_categories:
            hints.append("unroll_loops")
            hints.append("strength_reduction")
        
        if self.calls_per_second > 500:
            hints.append("aggressive_inline")
        
        return hints


class JITRuntimeProfiler:
    """
    Advanced runtime profiler for JIT compilation decisions.
    
    Tracks function execution patterns, identifies hot paths, and provides
    intelligent recommendations for JIT compilation with minimal overhead.
    """
    
    def __init__(self, enable_profiling: bool = True):
        self.enable_profiling = enable_profiling
        
        # Profiling data
        self.function_profiles: Dict[str, FunctionProfile] = {}
        self.call_stack: List[Tuple[str, float]] = []  # (function_name, start_time)
        
        # Hot path detection
        self.hot_functions: Set[str] = set()
        self.jit_candidates: Set[str] = set()
        self.compilation_queue: deque = deque()
        
        # Configuration
        self.sampling_rate = 1.0  # Profile every call initially
        self.adaptive_sampling = True
        self.jit_threshold = 0.6
        self.min_samples_for_jit = 50
        
        # Performance tracking
        self.profiler_overhead_ns = 0
        self.total_profiling_calls = 0
        self.last_analysis_time = time.time()
        self.analysis_interval = 5.0  # seconds
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background analysis
        self._analysis_thread: Optional[threading.Thread] = None
        self._should_stop_analysis = False
        
        if self.enable_profiling:
            self._start_background_analysis()
    
    def _start_background_analysis(self):
        """Start background thread for periodic analysis"""
        if self._analysis_thread is None:
            self._analysis_thread = threading.Thread(
                target=self._analysis_loop,
                daemon=True,
                name="JITProfilerAnalysis"
            )
            self._analysis_thread.start()
    
    def _analysis_loop(self):
        """Background analysis loop"""
        while not self._should_stop_analysis:
            try:
                time.sleep(self.analysis_interval)
                self._analyze_profiles()
                self._update_sampling_rate()
                self._identify_jit_candidates()
                
            except Exception as e:
                print(f"JIT profiler analysis error: {e}")
    
    def start_function_call(self, function_name: str, **metadata):
        """Record the start of a function call"""
        if not self.enable_profiling:
            return
        
        # Adaptive sampling to reduce overhead
        if self.adaptive_sampling and len(self.function_profiles) > 100:
            import random
            if random.random() > self.sampling_rate:
                return
        
        start_time = time.perf_counter_ns()
        
        with self._lock:
            self.call_stack.append((function_name, start_time))
            
            # Initialize profile if needed
            if function_name not in self.function_profiles:
                profile = FunctionProfile(name=function_name)
                
                # Set code characteristics from metadata
                profile.has_loops = metadata.get('has_loops', False)
                profile.has_math_ops = metadata.get('has_math_ops', False)
                profile.has_matrix_ops = metadata.get('has_matrix_ops', False)
                
                self.function_profiles[function_name] = profile
    
    def end_function_call(self, function_name: str, **metadata):
        """Record the end of a function call"""
        if not self.enable_profiling:
            return
        
        end_time = time.perf_counter_ns()
        
        with self._lock:
            # Find matching start call
            for i in range(len(self.call_stack) - 1, -1, -1):
                stack_name, start_time = self.call_stack[i]
                if stack_name == function_name:
                    # Remove from call stack
                    del self.call_stack[i]
                    
                    # Calculate execution time
                    execution_time = end_time - start_time
                    
                    # Create execution sample
                    sample = ExecutionSample(
                        timestamp=time.time(),
                        execution_time_ns=execution_time,
                        memory_allocations=metadata.get('memory_allocations', 0),
                        simd_operations=metadata.get('simd_operations', 0)
                    )
                    
                    # Update profile
                    if function_name in self.function_profiles:
                        self.function_profiles[function_name].update_sample(sample)
                    
                    self.total_profiling_calls += 1
                    break
    
    def _analyze_profiles(self):
        """Analyze all function profiles and update hot path detection"""
        with self._lock:
            current_time = time.time()
            
            # Update hot functions based on current metrics
            new_hot_functions = set()
            
            for name, profile in self.function_profiles.items():
                # Hot function criteria
                is_hot = (
                    profile.calls_per_second > 50 or  # High frequency
                    profile.average_execution_time_ns > 5_000_000 or  # Long execution
                    profile.jit_eligibility_score > 0.4  # High JIT potential
                )
                
                if is_hot:
                    new_hot_functions.add(name)
            
            self.hot_functions = new_hot_functions
            self.last_analysis_time = current_time
    
    def _update_sampling_rate(self):
        """Adaptively update sampling rate based on overhead"""
        if not self.adaptive_sampling:
            return
        
        # Calculate profiler overhead
        total_functions = len(self.function_profiles)
        
        if total_functions > 1000:
            # Reduce sampling for very large codebases
            self.sampling_rate = max(0.01, 50.0 / total_functions)
        elif total_functions > 100:
            self.sampling_rate = max(0.1, 10.0 / total_functions)
        else:
            self.sampling_rate = 1.0
    
    def _identify_jit_candidates(self):
        """Identify functions that should be JIT compiled"""
        with self._lock:
            new_candidates = set()
            
            for name, profile in self.function_profiles.items():
                if profile.should_jit_compile(self.jit_threshold):
                    new_candidates.add(name)
                    
                    # Add to compilation queue if not already there
                    if name not in self.jit_candidates:
                        self.compilation_queue.append(name)
            
            self.jit_candidates = new_candidates
    
    def get_jit_candidates(self) -> List[Tuple[str, FunctionProfile]]:
        """Get list of functions ready for JIT compilation"""
        with self._lock:
            candidates = []
            
            for name in list(self.compilation_queue):
                if name in self.function_profiles:
                    profile = self.function_profiles[name]
                    if profile.should_jit_compile(self.jit_threshold):
                        candidates.append((name, profile))
            
            return candidates
    
    def mark_jit_compiled(self, function_name: str, compilation_time: float, speedup_factor: float = 1.0):
        """Mark a function as successfully JIT compiled"""
        with self._lock:
            if function_name in self.function_profiles:
                profile = self.function_profiles[function_name]
                profile.is_jit_compiled = True
                profile.jit_compilation_time = compilation_time
                profile.jit_speedup_factor = speedup_factor
            
            # Remove from compilation queue
            try:
                while function_name in self.compilation_queue:
                    self.compilation_queue.remove(function_name)
            except ValueError:
                pass
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """Get comprehensive profiling summary"""
        with self._lock:
            profiles_list = list(self.function_profiles.values())
            
            if not profiles_list:
                return {'error': 'No profiling data available'}
            
            # Calculate summary statistics
            total_calls = sum(p.total_calls for p in profiles_list)
            total_execution_time = sum(p.total_execution_time_ns for p in profiles_list)
            
            hot_functions_data = []
            for name in self.hot_functions:
                if name in self.function_profiles:
                    profile = self.function_profiles[name]
                    hot_functions_data.append({
                        'name': name,
                        'calls_per_second': profile.calls_per_second,
                        'average_time_ms': profile.average_execution_time_ns / 1_000_000,
                        'jit_eligibility': profile.jit_eligibility_score,
                        'categories': [cat.name for cat in profile.hotspot_categories],
                        'is_jit_compiled': profile.is_jit_compiled
                    })
            
            jit_compiled_count = sum(1 for p in profiles_list if p.is_jit_compiled)
            
            return {
                'total_functions_tracked': len(self.function_profiles),
                'total_calls_recorded': total_calls,
                'total_execution_time_ms': total_execution_time / 1_000_000,
                'hot_functions_count': len(self.hot_functions),
                'jit_candidates_count': len(self.jit_candidates),
                'jit_compiled_count': jit_compiled_count,
                'compilation_queue_length': len(self.compilation_queue),
                'current_sampling_rate': self.sampling_rate,
                'profiler_overhead_calls': self.total_profiling_calls,
                'hot_functions': sorted(hot_functions_data, key=lambda x: x['jit_eligibility'], reverse=True)[:20],
                'top_candidates_for_jit': [name for name in list(self.compilation_queue)[:10]]
            }
    
    def get_function_profile(self, function_name: str) -> Optional[FunctionProfile]:
        """Get detailed profile for a specific function"""
        with self._lock:
            return self.function_profiles.get(function_name)
    
    def reset_profiling_data(self):
        """Reset all profiling data (useful for testing)"""
        with self._lock:
            self.function_profiles.clear()
            self.call_stack.clear()
            self.hot_functions.clear()
            self.jit_candidates.clear()
            self.compilation_queue.clear()
            self.total_profiling_calls = 0
    
    def set_jit_threshold(self, threshold: float):
        """Set JIT compilation eligibility threshold"""
        self.jit_threshold = max(0.0, min(1.0, threshold))
    
    def enable_adaptive_sampling(self, enable: bool = True):
        """Enable or disable adaptive sampling"""
        self.adaptive_sampling = enable
        if not enable:
            self.sampling_rate = 1.0
    
    def cleanup(self):
        """Clean up profiler resources"""
        self._should_stop_analysis = True
        if self._analysis_thread:
            self._analysis_thread.join(timeout=1.0)


# Global profiler instance
_global_jit_profiler: Optional[JITRuntimeProfiler] = None


def get_jit_profiler() -> JITRuntimeProfiler:
    """Get the global JIT runtime profiler instance"""
    global _global_jit_profiler
    if _global_jit_profiler is None:
        _global_jit_profiler = JITRuntimeProfiler()
    return _global_jit_profiler


def profile_function_call(function_name: str, **metadata):
    """Decorator/context manager for profiling function calls"""
    class FunctionCallProfiler:
        def __init__(self, name: str, metadata: Dict[str, Any]):
            self.name = name
            self.metadata = metadata
        
        def __enter__(self):
            get_jit_profiler().start_function_call(self.name, **self.metadata)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            get_jit_profiler().end_function_call(self.name, **self.metadata)
    
    return FunctionCallProfiler(function_name, metadata)


# Convenience functions
def start_profiling():
    """Start JIT profiling globally"""
    profiler = get_jit_profiler()
    profiler.enable_profiling = True
    return profiler


def stop_profiling() -> Dict[str, Any]:
    """Stop JIT profiling and return summary"""
    profiler = get_jit_profiler()
    profiler.enable_profiling = False
    return profiler.get_profiling_summary()


def get_jit_candidates() -> List[Tuple[str, FunctionProfile]]:
    """Get current JIT compilation candidates"""
    return get_jit_profiler().get_jit_candidates()
