"""
Memory Optimizer for NeuralScript

Provides intelligent memory optimization strategies, allocation pattern
analysis, and automatic tuning for optimal performance.
"""

import time
import math
from typing import Dict, List, Optional, Set, Tuple, Any, NamedTuple
from enum import Enum, auto
from dataclasses import dataclass
from collections import defaultdict, deque

from .object_model import ObjectType
from .heap_manager import HeapManager, HeapRegionType, AllocationStrategy


class OptimizationHint(Enum):
    """Optimization hints for different allocation patterns"""
    SEQUENTIAL_ACCESS = auto()      # Objects accessed sequentially
    RANDOM_ACCESS = auto()          # Objects accessed randomly
    TEMPORAL_LOCALITY = auto()      # Objects used together in time
    SPATIAL_LOCALITY = auto()       # Objects used together in memory
    SHORT_LIVED = auto()            # Objects with short lifespans
    LONG_LIVED = auto()             # Objects with long lifespans
    LARGE_ALLOCATIONS = auto()      # Pattern involves large objects
    FREQUENT_ALLOCATIONS = auto()   # High allocation frequency
    BATCH_PROCESSING = auto()       # Batch processing workload
    STREAMING_DATA = auto()         # Streaming data processing


@dataclass
class OptimizationStrategy:
    """A memory optimization strategy"""
    name: str
    description: str
    applicable_hints: List[OptimizationHint]
    priority: int = 1  # Higher numbers = higher priority
    
    def applies_to(self, hints: List[OptimizationHint]) -> bool:
        """Check if this strategy applies to given hints"""
        return any(hint in self.applicable_hints for hint in hints)


@dataclass
class AllocationPattern:
    """Detected allocation pattern"""
    object_type: ObjectType
    average_size: float
    allocation_rate: float  # allocations per second
    survival_rate: float    # fraction that survive GC
    access_pattern: str     # 'sequential', 'random', 'clustered'
    temporal_clustering: float  # 0.0 to 1.0, higher = more clustered in time
    spatial_clustering: float   # 0.0 to 1.0, higher = more clustered in memory
    optimization_hints: List[OptimizationHint]


@dataclass
class OptimizationRecommendation:
    """A specific optimization recommendation"""
    strategy: OptimizationStrategy
    confidence: float  # 0.0 to 1.0
    estimated_improvement: float  # Estimated improvement factor
    parameters: Dict[str, Any]
    reasoning: str


class PatternAnalyzer:
    """Analyzes allocation patterns to identify optimization opportunities"""
    
    def __init__(self):
        self._allocation_history: deque = deque(maxlen=10000)
        self._access_history: deque = deque(maxlen=10000)
        self._survival_tracking: Dict[int, float] = {}  # address -> allocation_time
    
    def record_allocation(self, address: int, obj_type: ObjectType, size: int):
        """Record an allocation for pattern analysis"""
        timestamp = time.time()
        self._allocation_history.append((timestamp, address, obj_type, size))
        self._survival_tracking[address] = timestamp
    
    def record_deallocation(self, address: int):
        """Record a deallocation for survival analysis"""
        if address in self._survival_tracking:
            del self._survival_tracking[address]
    
    def record_access(self, address: int, access_type: str = 'read'):
        """Record object access for access pattern analysis"""
        timestamp = time.time()
        self._access_history.append((timestamp, address, access_type))
    
    def analyze_patterns(self) -> Dict[ObjectType, AllocationPattern]:
        """Analyze allocation patterns by object type"""
        if not self._allocation_history:
            return {}
        
        patterns = {}
        
        # Group allocations by type
        allocations_by_type = defaultdict(list)
        for timestamp, address, obj_type, size in self._allocation_history:
            allocations_by_type[obj_type].append((timestamp, address, size))
        
        # Analyze each object type
        for obj_type, allocations in allocations_by_type.items():
            pattern = self._analyze_type_pattern(obj_type, allocations)
            patterns[obj_type] = pattern
        
        return patterns
    
    def _analyze_type_pattern(self, obj_type: ObjectType, 
                            allocations: List[Tuple[float, int, int]]) -> AllocationPattern:
        """Analyze allocation pattern for a specific object type"""
        if not allocations:
            return AllocationPattern(
                object_type=obj_type,
                average_size=0,
                allocation_rate=0,
                survival_rate=0,
                access_pattern='unknown',
                temporal_clustering=0,
                spatial_clustering=0,
                optimization_hints=[]
            )
        
        # Calculate basic statistics
        sizes = [size for _, _, size in allocations]
        timestamps = [ts for ts, _, _ in allocations]
        addresses = [addr for _, addr, _ in allocations]
        
        average_size = sum(sizes) / len(sizes)
        
        # Calculate allocation rate
        time_span = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1.0
        allocation_rate = len(allocations) / time_span if time_span > 0 else 0
        
        # Calculate survival rate
        still_alive = sum(1 for _, addr, _ in allocations if addr in self._survival_tracking)
        survival_rate = still_alive / len(allocations)
        
        # Analyze access patterns
        access_pattern = self._analyze_access_pattern(addresses)
        
        # Calculate temporal clustering
        temporal_clustering = self._calculate_temporal_clustering(timestamps)
        
        # Calculate spatial clustering (based on address proximity)
        spatial_clustering = self._calculate_spatial_clustering(addresses)
        
        # Generate optimization hints
        hints = self._generate_hints(average_size, allocation_rate, survival_rate,
                                   access_pattern, temporal_clustering, spatial_clustering)
        
        return AllocationPattern(
            object_type=obj_type,
            average_size=average_size,
            allocation_rate=allocation_rate,
            survival_rate=survival_rate,
            access_pattern=access_pattern,
            temporal_clustering=temporal_clustering,
            spatial_clustering=spatial_clustering,
            optimization_hints=hints
        )
    
    def _analyze_access_pattern(self, addresses: List[int]) -> str:
        """Analyze spatial access patterns"""
        if len(addresses) < 3:
            return 'unknown'
        
        # Look for sequential access patterns
        sequential_count = 0
        for i in range(1, len(addresses)):
            if abs(addresses[i] - addresses[i-1]) < 1024:  # Within 1KB
                sequential_count += 1
        
        sequential_ratio = sequential_count / (len(addresses) - 1)
        
        if sequential_ratio > 0.7:
            return 'sequential'
        elif sequential_ratio < 0.3:
            return 'random'
        else:
            return 'clustered'
    
    def _calculate_temporal_clustering(self, timestamps: List[float]) -> float:
        """Calculate how clustered allocations are in time"""
        if len(timestamps) < 3:
            return 0.0
        
        # Calculate gaps between allocations
        gaps = []
        for i in range(1, len(timestamps)):
            gaps.append(timestamps[i] - timestamps[i-1])
        
        if not gaps:
            return 0.0
        
        # Calculate coefficient of variation (lower = more clustered)
        mean_gap = sum(gaps) / len(gaps)
        if mean_gap == 0:
            return 1.0
        
        variance = sum((gap - mean_gap) ** 2 for gap in gaps) / len(gaps)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_gap
        
        # Convert to clustering score (1.0 = highly clustered, 0.0 = evenly spaced)
        return max(0.0, 1.0 - min(cv, 2.0) / 2.0)
    
    def _calculate_spatial_clustering(self, addresses: List[int]) -> float:
        """Calculate how clustered allocations are in memory"""
        if len(addresses) < 3:
            return 0.0
        
        sorted_addresses = sorted(addresses)
        gaps = []
        
        for i in range(1, len(sorted_addresses)):
            gaps.append(sorted_addresses[i] - sorted_addresses[i-1])
        
        if not gaps:
            return 0.0
        
        # Calculate clustering based on gap distribution
        mean_gap = sum(gaps) / len(gaps)
        small_gaps = sum(1 for gap in gaps if gap < mean_gap / 2)
        clustering = small_gaps / len(gaps)
        
        return clustering
    
    def _generate_hints(self, average_size: float, allocation_rate: float,
                       survival_rate: float, access_pattern: str,
                       temporal_clustering: float, spatial_clustering: float) -> List[OptimizationHint]:
        """Generate optimization hints based on pattern analysis"""
        hints = []
        
        # Size-based hints
        if average_size > 64 * 1024:  # > 64KB
            hints.append(OptimizationHint.LARGE_ALLOCATIONS)
        
        # Rate-based hints
        if allocation_rate > 1000:  # > 1000 allocs/sec
            hints.append(OptimizationHint.FREQUENT_ALLOCATIONS)
        
        # Lifetime hints
        if survival_rate < 0.1:
            hints.append(OptimizationHint.SHORT_LIVED)
        elif survival_rate > 0.8:
            hints.append(OptimizationHint.LONG_LIVED)
        
        # Access pattern hints
        if access_pattern == 'sequential':
            hints.append(OptimizationHint.SEQUENTIAL_ACCESS)
        elif access_pattern == 'random':
            hints.append(OptimizationHint.RANDOM_ACCESS)
        
        # Clustering hints
        if temporal_clustering > 0.7:
            hints.append(OptimizationHint.TEMPORAL_LOCALITY)
        if spatial_clustering > 0.7:
            hints.append(OptimizationHint.SPATIAL_LOCALITY)
        
        # Workload hints
        if temporal_clustering > 0.8 and allocation_rate > 100:
            hints.append(OptimizationHint.BATCH_PROCESSING)
        
        if allocation_rate > 500 and survival_rate < 0.2:
            hints.append(OptimizationHint.STREAMING_DATA)
        
        return hints


class MemoryOptimizer:
    """
    Advanced memory optimizer for NeuralScript.
    
    Analyzes allocation patterns, provides optimization recommendations,
    and automatically applies memory optimization strategies.
    """
    
    def __init__(self, heap_manager: HeapManager):
        self.heap_manager = heap_manager
        self.pattern_analyzer = PatternAnalyzer()
        
        # Optimization strategies
        self._strategies = self._initialize_strategies()
        
        # Performance tracking
        self._optimization_history = []
        self._performance_metrics = {}
        
        # Configuration
        self._auto_optimize = True
        self._min_confidence = 0.7
        self._max_optimizations_per_cycle = 3
    
    def _initialize_strategies(self) -> List[OptimizationStrategy]:
        """Initialize built-in optimization strategies"""
        return [
            # Memory layout optimizations
            OptimizationStrategy(
                name="Sequential Allocation Pool",
                description="Use sequential allocation pool for objects with sequential access",
                applicable_hints=[OptimizationHint.SEQUENTIAL_ACCESS, OptimizationHint.SPATIAL_LOCALITY],
                priority=3
            ),
            
            OptimizationStrategy(
                name="Large Object Heap",
                description="Use dedicated heap for large objects",
                applicable_hints=[OptimizationHint.LARGE_ALLOCATIONS],
                priority=4
            ),
            
            OptimizationStrategy(
                name="Bump Pointer Allocation",
                description="Use bump pointer allocation for frequent short-lived objects",
                applicable_hints=[OptimizationHint.FREQUENT_ALLOCATIONS, OptimizationHint.SHORT_LIVED],
                priority=3
            ),
            
            # Temporal optimizations
            OptimizationStrategy(
                name="Generation Tuning",
                description="Adjust generation sizes based on survival rates",
                applicable_hints=[OptimizationHint.SHORT_LIVED, OptimizationHint.LONG_LIVED],
                priority=2
            ),
            
            OptimizationStrategy(
                name="Batch Collection",
                description="Batch allocations to reduce collection frequency",
                applicable_hints=[OptimizationHint.BATCH_PROCESSING, OptimizationHint.TEMPORAL_LOCALITY],
                priority=2
            ),
            
            # Access pattern optimizations
            OptimizationStrategy(
                name="Prefetch Optimization",
                description="Optimize memory layout for prefetching",
                applicable_hints=[OptimizationHint.SEQUENTIAL_ACCESS],
                priority=1
            ),
            
            OptimizationStrategy(
                name="Cache-Friendly Layout",
                description="Arrange objects for better cache locality",
                applicable_hints=[OptimizationHint.SPATIAL_LOCALITY, OptimizationHint.TEMPORAL_LOCALITY],
                priority=2
            ),
            
            # Specialized optimizations
            OptimizationStrategy(
                name="Streaming Buffer Pool",
                description="Use circular buffer pool for streaming data",
                applicable_hints=[OptimizationHint.STREAMING_DATA],
                priority=4
            ),
            
            OptimizationStrategy(
                name="Arena Allocation",
                description="Use arena allocation for related objects",
                applicable_hints=[OptimizationHint.BATCH_PROCESSING, OptimizationHint.SPATIAL_LOCALITY],
                priority=3
            )
        ]
    
    def record_allocation(self, address: int, obj_type: ObjectType, size: int):
        """Record allocation for pattern analysis"""
        self.pattern_analyzer.record_allocation(address, obj_type, size)
    
    def record_deallocation(self, address: int):
        """Record deallocation for pattern analysis"""
        self.pattern_analyzer.record_deallocation(address)
    
    def record_access(self, address: int, access_type: str = 'read'):
        """Record object access for pattern analysis"""
        self.pattern_analyzer.record_access(address, access_type)
    
    def analyze_patterns(self) -> Dict[ObjectType, AllocationPattern]:
        """Analyze current allocation patterns"""
        return self.pattern_analyzer.analyze_patterns()
    
    def get_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Get optimization recommendations based on current patterns"""
        patterns = self.analyze_patterns()
        recommendations = []
        
        for obj_type, pattern in patterns.items():
            # Find applicable strategies
            applicable_strategies = [
                strategy for strategy in self._strategies
                if strategy.applies_to(pattern.optimization_hints)
            ]
            
            # Generate recommendations
            for strategy in applicable_strategies:
                confidence = self._calculate_confidence(strategy, pattern)
                if confidence >= self._min_confidence:
                    improvement = self._estimate_improvement(strategy, pattern)
                    params = self._generate_parameters(strategy, pattern)
                    reasoning = self._generate_reasoning(strategy, pattern)
                    
                    recommendations.append(OptimizationRecommendation(
                        strategy=strategy,
                        confidence=confidence,
                        estimated_improvement=improvement,
                        parameters=params,
                        reasoning=reasoning
                    ))
        
        # Sort by priority and confidence
        recommendations.sort(key=lambda r: (r.strategy.priority, r.confidence), reverse=True)
        return recommendations[:self._max_optimizations_per_cycle]
    
    def _calculate_confidence(self, strategy: OptimizationStrategy, 
                            pattern: AllocationPattern) -> float:
        """Calculate confidence score for applying a strategy to a pattern"""
        # Base confidence from hint matching
        matching_hints = sum(1 for hint in pattern.optimization_hints 
                           if hint in strategy.applicable_hints)
        total_hints = len(strategy.applicable_hints)
        base_confidence = matching_hints / total_hints if total_hints > 0 else 0
        
        # Adjust based on pattern strength
        if OptimizationHint.FREQUENT_ALLOCATIONS in pattern.optimization_hints:
            if pattern.allocation_rate > 1000:
                base_confidence *= 1.2
        
        if OptimizationHint.LARGE_ALLOCATIONS in pattern.optimization_hints:
            if pattern.average_size > 1024 * 1024:  # > 1MB
                base_confidence *= 1.3
        
        if OptimizationHint.SPATIAL_LOCALITY in pattern.optimization_hints:
            base_confidence *= (1 + pattern.spatial_clustering * 0.5)
        
        if OptimizationHint.TEMPORAL_LOCALITY in pattern.optimization_hints:
            base_confidence *= (1 + pattern.temporal_clustering * 0.5)
        
        return min(1.0, base_confidence)
    
    def _estimate_improvement(self, strategy: OptimizationStrategy,
                            pattern: AllocationPattern) -> float:
        """Estimate performance improvement from applying a strategy"""
        base_improvement = 1.1  # Base 10% improvement
        
        # Strategy-specific improvements
        if strategy.name == "Large Object Heap":
            # Large objects benefit significantly from dedicated heap
            if pattern.average_size > 1024 * 1024:
                base_improvement = 1.5
        
        elif strategy.name == "Sequential Allocation Pool":
            # Sequential access patterns benefit from locality
            if pattern.access_pattern == 'sequential':
                base_improvement = 1.3
        
        elif strategy.name == "Bump Pointer Allocation":
            # High allocation rates benefit from fast allocation
            if pattern.allocation_rate > 1000:
                base_improvement = 1.4
        
        elif strategy.name == "Streaming Buffer Pool":
            # Streaming data benefits significantly
            if OptimizationHint.STREAMING_DATA in pattern.optimization_hints:
                base_improvement = 1.6
        
        # Adjust based on pattern characteristics
        if pattern.spatial_clustering > 0.8:
            base_improvement *= 1.1
        
        if pattern.temporal_clustering > 0.8:
            base_improvement *= 1.1
        
        return base_improvement
    
    def _generate_parameters(self, strategy: OptimizationStrategy,
                           pattern: AllocationPattern) -> Dict[str, Any]:
        """Generate parameters for applying a strategy"""
        params = {}
        
        if strategy.name == "Large Object Heap":
            params['min_object_size'] = max(64 * 1024, pattern.average_size * 0.8)
            params['initial_heap_size'] = pattern.average_size * 100  # Room for ~100 objects
        
        elif strategy.name == "Sequential Allocation Pool":
            params['pool_size'] = pattern.average_size * 50  # Room for ~50 objects
            params['alignment'] = 64  # Cache line alignment
        
        elif strategy.name == "Bump Pointer Allocation":
            params['arena_size'] = min(1024 * 1024, pattern.average_size * 1000)
            params['reset_threshold'] = 0.9  # Reset when 90% full
        
        elif strategy.name == "Generation Tuning":
            if OptimizationHint.SHORT_LIVED in pattern.optimization_hints:
                params['young_gen_multiplier'] = 1.5
            if OptimizationHint.LONG_LIVED in pattern.optimization_hints:
                params['old_gen_multiplier'] = 1.3
        
        elif strategy.name == "Streaming Buffer Pool":
            params['buffer_count'] = 4
            params['buffer_size'] = pattern.average_size * 10
        
        return params
    
    def _generate_reasoning(self, strategy: OptimizationStrategy,
                          pattern: AllocationPattern) -> str:
        """Generate human-readable reasoning for the recommendation"""
        reasons = []
        
        if pattern.allocation_rate > 1000:
            reasons.append(f"High allocation rate ({pattern.allocation_rate:.0f}/sec)")
        
        if pattern.average_size > 64 * 1024:
            reasons.append(f"Large average object size ({pattern.average_size / 1024:.0f}KB)")
        
        if pattern.survival_rate < 0.1:
            reasons.append(f"Very short object lifetimes ({pattern.survival_rate:.1%} survival)")
        elif pattern.survival_rate > 0.8:
            reasons.append(f"Long object lifetimes ({pattern.survival_rate:.1%} survival)")
        
        if pattern.spatial_clustering > 0.7:
            reasons.append(f"Strong spatial locality ({pattern.spatial_clustering:.1%})")
        
        if pattern.temporal_clustering > 0.7:
            reasons.append(f"Strong temporal locality ({pattern.temporal_clustering:.1%})")
        
        if pattern.access_pattern == 'sequential':
            reasons.append("Sequential access pattern detected")
        
        reasoning = f"{strategy.description}. Applicable because: " + ", ".join(reasons)
        return reasoning
    
    def optimize(self) -> List[str]:
        """Perform automatic optimization based on current patterns"""
        if not self._auto_optimize:
            return []
        
        recommendations = self.get_optimization_recommendations()
        applied_optimizations = []
        
        for recommendation in recommendations:
            try:
                success = self._apply_optimization(recommendation)
                if success:
                    applied_optimizations.append(recommendation.strategy.name)
                    self._optimization_history.append({
                        'timestamp': time.time(),
                        'strategy': recommendation.strategy.name,
                        'confidence': recommendation.confidence,
                        'estimated_improvement': recommendation.estimated_improvement,
                        'parameters': recommendation.parameters
                    })
            except Exception as e:
                print(f"Failed to apply optimization {recommendation.strategy.name}: {e}")
        
        return applied_optimizations
    
    def _apply_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply a specific optimization recommendation"""
        strategy = recommendation.strategy
        params = recommendation.parameters
        
        try:
            if strategy.name == "Large Object Heap":
                return self._apply_large_object_optimization(params)
            
            elif strategy.name == "Sequential Allocation Pool":
                return self._apply_sequential_pool_optimization(params)
            
            elif strategy.name == "Bump Pointer Allocation":
                return self._apply_bump_pointer_optimization(params)
            
            elif strategy.name == "Generation Tuning":
                return self._apply_generation_tuning(params)
            
            # Add more strategy implementations as needed
            
            return False
            
        except Exception:
            return False
    
    def _apply_large_object_optimization(self, params: Dict[str, Any]) -> bool:
        """Apply large object heap optimization"""
        min_size = params.get('min_object_size', 64 * 1024)
        initial_size = params.get('initial_heap_size', 1024 * 1024)
        
        # Configure heap manager for large objects
        self.heap_manager.set_large_object_threshold(min_size)
        self.heap_manager.set_allocation_strategy(
            HeapRegionType.LARGE_OBJECT,
            AllocationStrategy.FIRST_FIT  # Good for large objects
        )
        
        return True
    
    def _apply_sequential_pool_optimization(self, params: Dict[str, Any]) -> bool:
        """Apply sequential allocation pool optimization"""
        pool_size = params.get('pool_size', 1024 * 1024)
        alignment = params.get('alignment', 64)
        
        # Create optimized pool for sequential allocations
        self.heap_manager.create_specialized_pool(
            "sequential_pool",
            pool_size,
            alignment=alignment
        )
        
        return True
    
    def _apply_bump_pointer_optimization(self, params: Dict[str, Any]) -> bool:
        """Apply bump pointer allocation optimization"""
        arena_size = params.get('arena_size', 1024 * 1024)
        
        # Configure young generation to use bump pointer allocation
        self.heap_manager.set_allocation_strategy(
            HeapRegionType.YOUNG_GENERATION,
            AllocationStrategy.SEQUENTIAL
        )
        
        return True
    
    def _apply_generation_tuning(self, params: Dict[str, Any]) -> bool:
        """Apply generational GC tuning"""
        young_multiplier = params.get('young_gen_multiplier', 1.0)
        old_multiplier = params.get('old_gen_multiplier', 1.0)
        
        # Adjust generation sizes
        current_stats = self.heap_manager.get_heap_statistics()
        
        for pool_name, pool_stats in current_stats['pools'].items():
            if 'young' in pool_name.lower():
                new_size = int(pool_stats['total_size'] * young_multiplier)
                self.heap_manager.resize_pool(pool_name, new_size)
            elif 'old' in pool_name.lower():
                new_size = int(pool_stats['total_size'] * old_multiplier)
                self.heap_manager.resize_pool(pool_name, new_size)
        
        return True
    
    def get_optimization_report(self) -> str:
        """Generate a report on applied optimizations and their effectiveness"""
        if not self._optimization_history:
            return "No optimizations have been applied yet."
        
        report_lines = []
        report_lines.append("Memory Optimization Report")
        report_lines.append("=" * 40)
        
        # Summary statistics
        total_optimizations = len(self._optimization_history)
        recent_optimizations = [opt for opt in self._optimization_history
                              if time.time() - opt['timestamp'] < 3600]  # Last hour
        
        report_lines.append(f"Total optimizations applied: {total_optimizations}")
        report_lines.append(f"Recent optimizations (last hour): {len(recent_optimizations)}")
        
        # Most common optimizations
        strategy_counts = defaultdict(int)
        for opt in self._optimization_history:
            strategy_counts[opt['strategy']] += 1
        
        report_lines.append("\nMost Applied Optimizations:")
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            report_lines.append(f"  {strategy}: {count} times")
        
        # Recent optimizations detail
        if recent_optimizations:
            report_lines.append("\nRecent Optimizations:")
            for opt in recent_optimizations[-5:]:  # Last 5
                timestamp = time.strftime('%H:%M:%S', time.localtime(opt['timestamp']))
                improvement = opt['estimated_improvement']
                confidence = opt['confidence']
                report_lines.append(
                    f"  {timestamp}: {opt['strategy']} "
                    f"(confidence: {confidence:.1%}, estimated improvement: {improvement:.1f}x)"
                )
        
        return "\n".join(report_lines)
    
    def set_auto_optimize(self, enabled: bool):
        """Enable or disable automatic optimization"""
        self._auto_optimize = enabled
    
    def set_optimization_parameters(self, min_confidence: float = None,
                                  max_optimizations_per_cycle: int = None):
        """Configure optimization parameters"""
        if min_confidence is not None:
            self._min_confidence = min_confidence
        if max_optimizations_per_cycle is not None:
            self._max_optimizations_per_cycle = max_optimizations_per_cycle
    
    def clear_history(self):
        """Clear optimization history"""
        self._optimization_history.clear()
        self._performance_metrics.clear()
        # Note: Don't clear pattern analyzer history as it's needed for analysis
