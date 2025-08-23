"""
SIMD Optimizer for Automatic Vectorization

Advanced optimization engine for automatic vectorization decisions,
performance analysis, and code generation optimization.
"""

import numpy as np
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import math

from .simd_core import SIMDProcessor, SIMDConfiguration, SIMDInstructionSet, DataType


class OptimizationLevel(Enum):
    """Optimization levels for vectorization"""
    NONE = 0        # No optimization
    BASIC = 1       # Basic vectorization
    AGGRESSIVE = 2  # Aggressive vectorization with unrolling
    MAXIMUM = 3     # Maximum optimization with all techniques


class VectorizationStrategy(Enum):
    """Vectorization strategies"""
    AUTO = "auto"              # Automatic strategy selection
    LOOP_VECTORIZATION = "loop"    # Loop-based vectorization
    BLOCK_PROCESSING = "block"     # Block-based processing
    STREAMING = "streaming"        # Streaming SIMD operations
    KERNEL_FUSION = "fusion"       # Kernel fusion optimization


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization decisions"""
    execution_time: float = 0.0
    throughput: float = 0.0  # Operations per second
    efficiency: float = 0.0  # Percentage of peak performance
    cache_hits: int = 0
    cache_misses: int = 0
    memory_bandwidth: float = 0.0  # GB/s
    flop_rate: float = 0.0  # FLOPS
    vectorization_factor: float = 1.0
    
    def __post_init__(self):
        if self.execution_time > 0:
            self.efficiency = min(100.0, (self.throughput / self.execution_time) * 100)


@dataclass
class OptimizationHint:
    """Optimization hints for specific operations"""
    operation_name: str
    data_size: int
    data_type: DataType
    access_pattern: str  # "sequential", "random", "stride"
    reuse_factor: float = 1.0
    compute_intensity: float = 1.0  # FLOPS per byte
    preferred_strategy: Optional[VectorizationStrategy] = None
    memory_bound: bool = False
    compute_bound: bool = False


class VectorizationAnalyzer:
    """Analyzes code patterns for vectorization opportunities"""
    
    def __init__(self, simd_processor: Optional[SIMDProcessor] = None):
        self.simd = simd_processor or SIMDProcessor()
        self._analysis_cache = {}
        self._lock = threading.RLock()
    
    def analyze_loop_pattern(self, array_size: int, data_type: DataType,
                           access_pattern: str = "sequential") -> Dict[str, Any]:
        """
        Analyze a loop pattern for vectorization potential
        
        Args:
            array_size: Size of the array being processed
            data_type: Data type of array elements
            access_pattern: Memory access pattern
        
        Returns:
            Analysis results with vectorization recommendations
        """
        cache_key = (array_size, data_type, access_pattern)
        
        with self._lock:
            if cache_key in self._analysis_cache:
                return self._analysis_cache[cache_key]
        
        # Calculate vectorization parameters
        vector_width = self.simd.get_vector_width(data_type)
        chunk_size = self.simd.calculate_optimal_chunk_size(array_size, data_type)
        
        # Estimate performance characteristics
        sequential_cost = array_size  # Scalar operations
        vectorized_cost = math.ceil(array_size / vector_width)  # Vector operations
        
        speedup_potential = sequential_cost / vectorized_cost if vectorized_cost > 0 else 1.0
        
        # Memory access analysis
        type_sizes = {
            DataType.FLOAT32: 4, DataType.FLOAT64: 8,
            DataType.INT32: 4, DataType.INT64: 8,
            DataType.INT16: 2, DataType.INT8: 1,
            DataType.UINT32: 4, DataType.UINT64: 8,
            DataType.UINT16: 2, DataType.UINT8: 1,
        }
        element_size = type_sizes.get(data_type, 4)
        cache_line_elements = 64 // element_size  # Assuming 64-byte cache lines
        cache_efficiency = 1.0
        
        if access_pattern == "sequential":
            cache_efficiency = min(1.0, cache_line_elements / vector_width)
        elif access_pattern == "stride":
            cache_efficiency = 0.5  # Reduced efficiency for strided access
        elif access_pattern == "random":
            cache_efficiency = 0.1  # Poor cache efficiency
        
        # Vectorization recommendation
        should_vectorize = (
            array_size >= self.simd.config.vectorization_threshold and
            speedup_potential > 1.2 and  # At least 20% improvement
            cache_efficiency > 0.3
        )
        
        analysis = {
            'should_vectorize': should_vectorize,
            'vector_width': vector_width,
            'chunk_size': chunk_size,
            'speedup_potential': speedup_potential,
            'cache_efficiency': cache_efficiency,
            'recommended_strategy': self._recommend_strategy(array_size, access_pattern),
            'unroll_factor': min(4, max(1, chunk_size // vector_width)),
            'memory_bound': cache_efficiency < 0.5,
            'compute_bound': speedup_potential > 2.0
        }
        
        with self._lock:
            self._analysis_cache[cache_key] = analysis
        
        return analysis
    
    def _recommend_strategy(self, array_size: int, access_pattern: str) -> VectorizationStrategy:
        """Recommend the best vectorization strategy"""
        if array_size < 1000:
            return VectorizationStrategy.LOOP_VECTORIZATION
        elif array_size < 100000:
            if access_pattern == "sequential":
                return VectorizationStrategy.BLOCK_PROCESSING
            else:
                return VectorizationStrategy.LOOP_VECTORIZATION
        else:
            if access_pattern == "sequential":
                return VectorizationStrategy.STREAMING
            else:
                return VectorizationStrategy.BLOCK_PROCESSING


class PerformanceProfiler:
    """Profiles SIMD operations for optimization decisions"""
    
    def __init__(self, simd_processor: Optional[SIMDProcessor] = None):
        self.simd = simd_processor or SIMDProcessor()
        self.metrics_history = []
        self._lock = threading.RLock()
    
    def profile_operation(self, operation: Callable, *args, **kwargs) -> PerformanceMetrics:
        """
        Profile a SIMD operation to gather performance metrics
        
        Args:
            operation: The operation to profile
            *args: Operation arguments
            **kwargs: Operation keyword arguments
        
        Returns:
            Performance metrics
        """
        # Warm-up run
        try:
            operation(*args, **kwargs)
        except Exception:
            pass
        
        # Timing runs
        times = []
        for _ in range(3):  # Multiple runs for accuracy
            start_time = time.perf_counter()
            try:
                result = operation(*args, **kwargs)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except Exception as e:
                # Return default metrics if operation fails
                return PerformanceMetrics(execution_time=float('inf'))
        
        avg_time = sum(times) / len(times)
        
        # Estimate data size and compute metrics
        data_size = self._estimate_data_size(args)
        flop_count = self._estimate_flop_count(operation.__name__, data_size)
        
        metrics = PerformanceMetrics(
            execution_time=avg_time,
            throughput=data_size / avg_time if avg_time > 0 else 0.0,
            flop_rate=flop_count / avg_time if avg_time > 0 else 0.0,
            memory_bandwidth=(data_size * 4) / avg_time / (1024**3) if avg_time > 0 else 0.0,  # Assuming float32
            vectorization_factor=self._estimate_vectorization_factor(operation.__name__)
        )
        
        with self._lock:
            self.metrics_history.append(metrics)
        
        return metrics
    
    def _estimate_data_size(self, args) -> int:
        """Estimate the total data size processed"""
        total_size = 0
        for arg in args:
            if hasattr(arg, 'size'):
                total_size += arg.size
            elif hasattr(arg, '__len__'):
                total_size += len(arg)
            else:
                total_size += 1
        return total_size
    
    def _estimate_flop_count(self, operation_name: str, data_size: int) -> int:
        """Estimate floating point operations count"""
        # Simple heuristics for common operations
        flop_counts = {
            'add': 1, 'subtract': 1, 'multiply': 1, 'divide': 1,
            'sqrt': 4, 'exp': 8, 'log': 8, 'sin': 10, 'cos': 10,
            'matrix_multiply': data_size * 2,  # Approximate for square matrices
            'dot_product': data_size * 2,
            'conv1d': data_size * 4,
            'conv2d': data_size * 6
        }
        
        base_flops = flop_counts.get(operation_name.split('_')[-1], 1)
        return base_flops * data_size
    
    def _estimate_vectorization_factor(self, operation_name: str) -> float:
        """Estimate how well the operation vectorizes"""
        # Heuristic based on operation type
        if any(op in operation_name.lower() for op in ['add', 'multiply', 'subtract']):
            return 4.0  # Simple arithmetic vectorizes well
        elif any(op in operation_name.lower() for op in ['sqrt', 'abs']):
            return 3.0  # Math functions vectorize moderately
        elif any(op in operation_name.lower() for op in ['sin', 'cos', 'exp', 'log']):
            return 2.0  # Transcendental functions vectorize less efficiently
        else:
            return 1.5  # Default modest vectorization


class AutoVectorizer:
    """Automatic vectorization optimizer"""
    
    def __init__(self, simd_processor: Optional[SIMDProcessor] = None,
                 optimization_level: OptimizationLevel = OptimizationLevel.BASIC):
        self.simd = simd_processor or SIMDProcessor()
        self.optimization_level = optimization_level
        self.analyzer = VectorizationAnalyzer(simd_processor)
        self.profiler = PerformanceProfiler(simd_processor)
        self._optimization_cache = {}
        self._lock = threading.RLock()
    
    def optimize_operation(self, operation_hint: OptimizationHint) -> Dict[str, Any]:
        """
        Generate optimization recommendations for an operation
        
        Args:
            operation_hint: Hint about the operation to optimize
        
        Returns:
            Optimization recommendations
        """
        cache_key = (
            operation_hint.operation_name,
            operation_hint.data_size,
            operation_hint.data_type,
            operation_hint.access_pattern
        )
        
        with self._lock:
            if cache_key in self._optimization_cache:
                return self._optimization_cache[cache_key]
        
        # Analyze the operation
        analysis = self.analyzer.analyze_loop_pattern(
            operation_hint.data_size,
            operation_hint.data_type,
            operation_hint.access_pattern
        )
        
        # Generate optimization strategy
        optimization = self._generate_optimization_strategy(operation_hint, analysis)
        
        with self._lock:
            self._optimization_cache[cache_key] = optimization
        
        return optimization
    
    def _generate_optimization_strategy(self, hint: OptimizationHint,
                                      analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed optimization strategy"""
        strategy = {
            'vectorize': analysis['should_vectorize'],
            'vector_width': analysis['vector_width'],
            'chunk_size': analysis['chunk_size'],
            'unroll_factor': 1,
            'use_prefetch': False,
            'blocking_strategy': None,
            'memory_alignment': 32,  # 32-byte alignment for AVX
            'parallel_threshold': 10000,
            'estimated_speedup': analysis['speedup_potential']
        }
        
        # Adjust based on optimization level
        if self.optimization_level.value >= OptimizationLevel.BASIC.value:
            strategy['unroll_factor'] = analysis['unroll_factor']
            strategy['use_prefetch'] = hint.data_size > 10000
        
        if self.optimization_level.value >= OptimizationLevel.AGGRESSIVE.value:
            strategy['unroll_factor'] = min(8, strategy['unroll_factor'] * 2)
            strategy['blocking_strategy'] = self._determine_blocking_strategy(hint)
            strategy['parallel_threshold'] = 5000  # Lower threshold for parallelization
        
        if self.optimization_level.value >= OptimizationLevel.MAXIMUM.value:
            strategy['unroll_factor'] = min(16, strategy['unroll_factor'] * 2)
            strategy['use_kernel_fusion'] = True
            strategy['use_fma'] = self.simd.capabilities.has_fma
            strategy['parallel_threshold'] = 1000
        
        # Memory-bound optimizations
        if analysis['memory_bound']:
            strategy['use_prefetch'] = True
            strategy['blocking_strategy'] = 'cache_blocking'
            strategy['vector_width'] = min(strategy['vector_width'], 4)  # Reduce pressure
        
        # Compute-bound optimizations
        if analysis['compute_bound']:
            strategy['unroll_factor'] = min(16, strategy['unroll_factor'] * 2)
            strategy['use_fma'] = True
        
        return strategy
    
    def _determine_blocking_strategy(self, hint: OptimizationHint) -> Optional[str]:
        """Determine the best blocking strategy for the operation"""
        if hint.data_size < 1000:
            return None
        elif hint.memory_bound:
            return 'cache_blocking'
        elif hint.compute_bound:
            return 'register_blocking'
        else:
            return 'hierarchical_blocking'


class KernelFusionOptimizer:
    """Optimizes multiple operations through kernel fusion"""
    
    def __init__(self, simd_processor: Optional[SIMDProcessor] = None):
        self.simd = simd_processor or SIMDProcessor()
        self.fusion_opportunities = []
        self._lock = threading.RLock()
    
    def analyze_fusion_opportunity(self, operations: List[str],
                                 data_flow: List[Tuple[int, int]]) -> Dict[str, Any]:
        """
        Analyze potential for fusing multiple operations
        
        Args:
            operations: List of operation names
            data_flow: List of (producer, consumer) index pairs
        
        Returns:
            Fusion analysis and recommendations
        """
        if len(operations) < 2:
            return {'fusable': False, 'reason': 'insufficient_operations'}
        
        # Check for data dependencies
        can_fuse = self._check_fusion_compatibility(operations, data_flow)
        
        if not can_fuse:
            return {'fusable': False, 'reason': 'incompatible_dependencies'}
        
        # Estimate fusion benefits
        individual_cost = sum(self._estimate_operation_cost(op) for op in operations)
        fused_cost = self._estimate_fused_cost(operations)
        
        speedup = individual_cost / fused_cost if fused_cost > 0 else 1.0
        memory_reduction = self._estimate_memory_reduction(operations)
        
        return {
            'fusable': speedup > 1.1,  # At least 10% improvement
            'estimated_speedup': speedup,
            'memory_reduction': memory_reduction,
            'recommended_fusion': self._recommend_fusion_pattern(operations),
            'complexity': len(operations)
        }
    
    def _check_fusion_compatibility(self, operations: List[str],
                                   data_flow: List[Tuple[int, int]]) -> bool:
        """Check if operations can be safely fused"""
        # Simple compatibility check - can be made more sophisticated
        element_wise_ops = ['add', 'multiply', 'subtract', 'divide', 'relu', 'sigmoid']
        reduction_ops = ['sum', 'max', 'min', 'dot_product']
        
        has_element_wise = any(op in ' '.join(operations) for op in element_wise_ops)
        has_reduction = any(op in ' '.join(operations) for op in reduction_ops)
        
        # Element-wise operations can usually be fused
        if has_element_wise and not has_reduction:
            return True
        
        # More complex analysis needed for mixed operations
        return len(operations) <= 3
    
    def _estimate_operation_cost(self, operation: str) -> float:
        """Estimate computational cost of an operation"""
        costs = {
            'add': 1.0, 'subtract': 1.0, 'multiply': 1.0, 'divide': 4.0,
            'sqrt': 8.0, 'exp': 16.0, 'log': 16.0, 'sin': 20.0, 'cos': 20.0,
            'relu': 2.0, 'sigmoid': 12.0, 'tanh': 16.0
        }
        
        for op_name, cost in costs.items():
            if op_name in operation.lower():
                return cost
        
        return 5.0  # Default cost
    
    def _estimate_fused_cost(self, operations: List[str]) -> float:
        """Estimate cost of fused operations"""
        total_cost = sum(self._estimate_operation_cost(op) for op in operations)
        
        # Fusion reduces overhead but may increase register pressure
        fusion_efficiency = 0.8  # 20% overhead reduction
        register_penalty = 1.0 + (len(operations) - 2) * 0.1  # Penalty for complexity
        
        return total_cost * fusion_efficiency * register_penalty
    
    def _estimate_memory_reduction(self, operations: List[str]) -> float:
        """Estimate memory access reduction from fusion"""
        # Each fused operation saves intermediate memory accesses
        return max(0.0, (len(operations) - 1) * 0.5)  # 50% reduction per intermediate
    
    def _recommend_fusion_pattern(self, operations: List[str]) -> str:
        """Recommend specific fusion pattern"""
        if len(operations) <= 3:
            return 'simple_fusion'
        elif any('conv' in op.lower() for op in operations):
            return 'convolution_fusion'
        elif any('matrix' in op.lower() for op in operations):
            return 'linear_algebra_fusion'
        else:
            return 'element_wise_fusion'


class AdaptiveOptimizer:
    """Adaptive optimizer that learns from runtime performance"""
    
    def __init__(self, simd_processor: Optional[SIMDProcessor] = None):
        self.simd = simd_processor or SIMDProcessor()
        self.auto_vectorizer = AutoVectorizer(simd_processor)
        self.fusion_optimizer = KernelFusionOptimizer(simd_processor)
        self.performance_history = {}
        self._adaptation_lock = threading.RLock()
    
    def adapt_strategy(self, operation_name: str, hint: OptimizationHint,
                      actual_performance: PerformanceMetrics) -> Dict[str, Any]:
        """
        Adapt optimization strategy based on actual performance
        
        Args:
            operation_name: Name of the operation
            hint: Original optimization hint
            actual_performance: Measured performance metrics
        
        Returns:
            Updated optimization strategy
        """
        with self._adaptation_lock:
            if operation_name not in self.performance_history:
                self.performance_history[operation_name] = []
            
            self.performance_history[operation_name].append({
                'hint': hint,
                'performance': actual_performance,
                'timestamp': time.time()
            })
        
        # Analyze performance trend
        history = self.performance_history[operation_name]
        if len(history) < 3:
            # Not enough data for adaptation
            return self.auto_vectorizer.optimize_operation(hint)
        
        # Calculate performance trend
        recent_performance = [h['performance'].efficiency for h in history[-5:]]
        avg_efficiency = sum(recent_performance) / len(recent_performance)
        
        # Adapt strategy based on performance
        adapted_strategy = self._adapt_based_on_efficiency(hint, avg_efficiency)
        
        return adapted_strategy
    
    def _adapt_based_on_efficiency(self, hint: OptimizationHint,
                                  avg_efficiency: float) -> Dict[str, Any]:
        """Adapt strategy based on measured efficiency"""
        base_strategy = self.auto_vectorizer.optimize_operation(hint)
        
        if avg_efficiency < 30:  # Poor performance
            # Reduce aggressiveness
            base_strategy['unroll_factor'] = max(1, base_strategy['unroll_factor'] // 2)
            base_strategy['vector_width'] = min(4, base_strategy['vector_width'])
            base_strategy['use_prefetch'] = False
        
        elif avg_efficiency > 80:  # Good performance
            # Increase aggressiveness
            base_strategy['unroll_factor'] = min(16, base_strategy['unroll_factor'] * 2)
            base_strategy['use_prefetch'] = True
            base_strategy['use_fma'] = True
        
        return base_strategy
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        with self._adaptation_lock:
            operations_count = len(self.performance_history)
            total_measurements = sum(len(history) for history in self.performance_history.values())
            
            # Calculate average efficiency by operation
            operation_efficiencies = {}
            for op_name, history in self.performance_history.items():
                efficiencies = [h['performance'].efficiency for h in history]
                operation_efficiencies[op_name] = sum(efficiencies) / len(efficiencies)
            
            # Find best and worst performing operations
            if operation_efficiencies:
                best_operation = max(operation_efficiencies.items(), key=lambda x: x[1])
                worst_operation = min(operation_efficiencies.items(), key=lambda x: x[1])
            else:
                best_operation = worst_operation = None
        
        report = {
            'total_operations_tracked': operations_count,
            'total_measurements': total_measurements,
            'operation_efficiencies': operation_efficiencies,
            'best_performing_operation': best_operation,
            'worst_performing_operation': worst_operation,
            'simd_utilization': self.simd.get_performance_report(),
            'optimization_recommendations': self._generate_global_recommendations()
        }
        
        return report
    
    def _generate_global_recommendations(self) -> List[str]:
        """Generate global optimization recommendations"""
        recommendations = []
        
        # Analyze overall performance patterns
        with self._adaptation_lock:
            if not self.performance_history:
                return ["Insufficient performance data for recommendations"]
            
            all_efficiencies = []
            for history in self.performance_history.values():
                all_efficiencies.extend(h['performance'].efficiency for h in history)
            
            avg_global_efficiency = sum(all_efficiencies) / len(all_efficiencies)
        
        if avg_global_efficiency < 40:
            recommendations.extend([
                "Consider reducing vectorization aggressiveness",
                "Check for memory bottlenecks",
                "Verify data alignment and access patterns"
            ])
        
        elif avg_global_efficiency < 70:
            recommendations.extend([
                "Moderate performance - consider targeted optimizations",
                "Analyze cache utilization patterns",
                "Consider loop unrolling adjustments"
            ])
        
        else:
            recommendations.extend([
                "Good SIMD utilization achieved",
                "Consider more aggressive optimization levels",
                "Explore kernel fusion opportunities"
            ])
        
        return recommendations
