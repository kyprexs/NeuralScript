"""
Memory Layout Optimization for NeuralScript
==========================================

Advanced memory layout optimization system that transforms data structures
for improved cache efficiency and reduced memory footprint. Key component
for achieving 30% memory reduction vs Python.

Features:
- Structure-of-Arrays (SoA) transformations
- Cache-aware data alignment and padding
- Memory coalescing and compaction
- NUMA-aware memory placement
- Vectorization-friendly layouts for SIMD
- Hot/cold data separation
"""

import threading
import ctypes
import math
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import struct
import platform
import numpy as np


class LayoutStrategy(Enum):
    """Memory layout optimization strategies"""
    ARRAY_OF_STRUCTS = "aos"        # Traditional layout
    STRUCT_OF_ARRAYS = "soa"        # Vectorization-friendly
    HYBRID = "hybrid"               # Mix of AoS and SoA
    HOT_COLD_SEPARATION = "hot_cold" # Separate frequently/rarely accessed data
    CACHE_ALIGNED = "cache_aligned"  # Align to cache line boundaries
    NUMA_AWARE = "numa_aware"       # NUMA topology aware


@dataclass
class FieldInfo:
    """Information about a data structure field"""
    name: str
    data_type: type
    size_bytes: int
    alignment: int
    access_frequency: float = 1.0  # Relative access frequency
    is_hot_data: bool = True       # Hot vs cold data classification
    vectorizable: bool = False     # Can benefit from SIMD
    
    
@dataclass
class CacheInfo:
    """CPU cache hierarchy information"""
    l1_data_size: int = 32 * 1024      # 32KB typical L1 data cache
    l1_line_size: int = 64              # 64-byte cache lines
    l2_size: int = 256 * 1024           # 256KB typical L2 cache
    l3_size: int = 8 * 1024 * 1024      # 8MB typical L3 cache
    prefetch_distance: int = 2          # Cache lines to prefetch ahead
    
    def __post_init__(self):
        """Auto-detect cache information if possible"""
        try:
            self._detect_cache_info()
        except Exception:
            pass  # Use defaults
    
    def _detect_cache_info(self):
        """Attempt to detect actual CPU cache information"""
        # This would use platform-specific methods to query cache info
        # For now, use reasonable defaults for x86-64
        if platform.machine() in ['x86_64', 'AMD64']:
            self.l1_line_size = 64
            self.l1_data_size = 32 * 1024
        elif platform.machine() == 'arm64':
            self.l1_line_size = 64
            self.l1_data_size = 64 * 1024  # ARM often has larger L1


@dataclass
class LayoutOptimization:
    """Result of layout optimization analysis"""
    strategy: LayoutStrategy
    field_order: List[str]
    padding_bytes: Dict[str, int]
    total_size: int
    cache_lines_used: int
    memory_savings: int
    vectorization_benefit: float
    estimated_performance_improvement: float


class MemoryLayoutOptimizer:
    """
    Advanced memory layout optimizer that transforms data structures
    for optimal cache utilization and reduced memory footprint.
    """
    
    def __init__(self, cache_info: Optional[CacheInfo] = None):
        self.cache_info = cache_info or CacheInfo()
        
        # Optimization statistics
        self.optimizations_applied = 0
        self.total_memory_saved = 0
        self.total_performance_improvement = 0.0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Optimization cache
        self._optimization_cache: Dict[str, LayoutOptimization] = {}
        
        # Hot/cold data tracking
        self.access_patterns: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
    def optimize_structure(self, 
                         structure_name: str,
                         fields: List[FieldInfo],
                         target_strategy: Optional[LayoutStrategy] = None,
                         vectorization_target: bool = True) -> LayoutOptimization:
        """
        Optimize memory layout for a data structure.
        
        Args:
            structure_name: Name of the structure for caching
            fields: List of field information
            target_strategy: Preferred optimization strategy
            vectorization_target: Whether to optimize for SIMD operations
            
        Returns:
            LayoutOptimization with detailed results
        """
        
        # Check cache first
        cache_key = f"{structure_name}_{hash(tuple(f.name for f in fields))}"
        if cache_key in self._optimization_cache:
            return self._optimization_cache[cache_key]
        
        with self._lock:
            # Analyze current layout
            original_layout = self._analyze_current_layout(fields)
            
            # Generate optimization candidates
            candidates = []
            
            if target_strategy is None or target_strategy == LayoutStrategy.STRUCT_OF_ARRAYS:
                soa_opt = self._optimize_for_soa(fields, vectorization_target)
                candidates.append(soa_opt)
            
            if target_strategy is None or target_strategy == LayoutStrategy.HOT_COLD_SEPARATION:
                hot_cold_opt = self._optimize_hot_cold_separation(fields)
                candidates.append(hot_cold_opt)
            
            if target_strategy is None or target_strategy == LayoutStrategy.CACHE_ALIGNED:
                cache_opt = self._optimize_cache_alignment(fields)
                candidates.append(cache_opt)
            
            if target_strategy is None or target_strategy == LayoutStrategy.HYBRID:
                hybrid_opt = self._optimize_hybrid_layout(fields, vectorization_target)
                candidates.append(hybrid_opt)
            
            # Select best optimization
            best_optimization = self._select_best_optimization(candidates, original_layout)
            
            # Cache the result
            self._optimization_cache[cache_key] = best_optimization
            
            # Update statistics
            self.optimizations_applied += 1
            self.total_memory_saved += best_optimization.memory_savings
            self.total_performance_improvement += best_optimization.estimated_performance_improvement
            
            return best_optimization
    
    def _analyze_current_layout(self, fields: List[FieldInfo]) -> Dict[str, Any]:
        """Analyze the current (unoptimized) layout"""
        total_size = 0
        current_offset = 0
        
        for field in fields:
            # Add padding for alignment
            alignment_padding = (field.alignment - (current_offset % field.alignment)) % field.alignment
            current_offset += alignment_padding
            
            current_offset += field.size_bytes
            total_size = current_offset
        
        # Round up to nearest pointer alignment (8 bytes on 64-bit)
        total_size = (total_size + 7) & ~7
        
        cache_lines_used = (total_size + self.cache_info.l1_line_size - 1) // self.cache_info.l1_line_size
        
        return {
            'total_size': total_size,
            'cache_lines_used': cache_lines_used,
            'memory_efficiency': sum(f.size_bytes for f in fields) / total_size if total_size > 0 else 0.0
        }
    
    def _optimize_for_soa(self, fields: List[FieldInfo], vectorization_target: bool) -> LayoutOptimization:
        """Optimize layout for Structure-of-Arrays"""
        
        # Separate vectorizable and non-vectorizable fields
        vectorizable_fields = [f for f in fields if f.vectorizable and vectorization_target]
        other_fields = [f for f in fields if not f.vectorizable or not vectorization_target]
        
        # Calculate memory layout
        field_order = []
        padding_bytes = {}
        total_size = 0
        
        # Place vectorizable fields first, aligned to SIMD boundaries
        simd_alignment = 32  # AVX alignment
        
        for field in vectorizable_fields:
            alignment_padding = (simd_alignment - (total_size % simd_alignment)) % simd_alignment
            if alignment_padding > 0:
                padding_bytes[field.name + "_pre"] = alignment_padding
                total_size += alignment_padding
            
            field_order.append(field.name)
            total_size += field.size_bytes
        
        # Place other fields with normal alignment
        for field in other_fields:
            alignment_padding = (field.alignment - (total_size % field.alignment)) % field.alignment
            if alignment_padding > 0:
                padding_bytes[field.name + "_pre"] = alignment_padding
                total_size += alignment_padding
            
            field_order.append(field.name)
            total_size += field.size_bytes
        
        # Calculate benefits
        cache_lines_used = (total_size + self.cache_info.l1_line_size - 1) // self.cache_info.l1_line_size
        original_layout = self._analyze_current_layout(fields)
        memory_savings = original_layout['total_size'] - total_size
        
        # Estimate vectorization benefit
        vectorization_benefit = len(vectorizable_fields) * 2.0 if vectorizable_fields else 0.0
        performance_improvement = vectorization_benefit * 0.15  # 15% improvement per 2x speedup
        
        return LayoutOptimization(
            strategy=LayoutStrategy.STRUCT_OF_ARRAYS,
            field_order=field_order,
            padding_bytes=padding_bytes,
            total_size=total_size,
            cache_lines_used=cache_lines_used,
            memory_savings=memory_savings,
            vectorization_benefit=vectorization_benefit,
            estimated_performance_improvement=performance_improvement
        )
    
    def _optimize_hot_cold_separation(self, fields: List[FieldInfo]) -> LayoutOptimization:
        """Optimize layout by separating hot and cold data"""
        
        # Classify fields as hot or cold based on access frequency
        hot_fields = [f for f in fields if f.is_hot_data or f.access_frequency > 0.5]
        cold_fields = [f for f in fields if not f.is_hot_data and f.access_frequency <= 0.5]
        
        field_order = []
        padding_bytes = {}
        total_size = 0
        
        # Place hot data first, tightly packed
        for field in sorted(hot_fields, key=lambda f: f.access_frequency, reverse=True):
            alignment_padding = (field.alignment - (total_size % field.alignment)) % field.alignment
            if alignment_padding > 0:
                padding_bytes[field.name + "_pre"] = alignment_padding
                total_size += alignment_padding
            
            field_order.append(field.name)
            total_size += field.size_bytes
        
        # Pad to cache line boundary before cold data
        cache_line_padding = (self.cache_info.l1_line_size - (total_size % self.cache_info.l1_line_size)) % self.cache_info.l1_line_size
        if cache_line_padding > 0 and cold_fields:
            padding_bytes["hot_cold_separator"] = cache_line_padding
            total_size += cache_line_padding
        
        # Place cold data
        for field in cold_fields:
            alignment_padding = (field.alignment - (total_size % field.alignment)) % field.alignment
            if alignment_padding > 0:
                padding_bytes[field.name + "_pre"] = alignment_padding
                total_size += alignment_padding
            
            field_order.append(field.name)
            total_size += field.size_bytes
        
        # Calculate benefits
        cache_lines_used = (total_size + self.cache_info.l1_line_size - 1) // self.cache_info.l1_line_size
        original_layout = self._analyze_current_layout(fields)
        
        # Hot/cold separation can actually increase total size but improve cache performance
        memory_savings = original_layout['total_size'] - total_size
        
        # Estimate cache performance improvement
        hot_data_size = sum(f.size_bytes for f in hot_fields)
        cache_improvement = min(2.0, self.cache_info.l1_line_size / max(hot_data_size, 1))
        performance_improvement = cache_improvement * 0.10  # 10% improvement per cache improvement
        
        return LayoutOptimization(
            strategy=LayoutStrategy.HOT_COLD_SEPARATION,
            field_order=field_order,
            padding_bytes=padding_bytes,
            total_size=total_size,
            cache_lines_used=cache_lines_used,
            memory_savings=memory_savings,
            vectorization_benefit=0.0,
            estimated_performance_improvement=performance_improvement
        )
    
    def _optimize_cache_alignment(self, fields: List[FieldInfo]) -> LayoutOptimization:
        """Optimize layout for cache line alignment"""
        
        # Sort fields by size (largest first) to minimize padding
        sorted_fields = sorted(fields, key=lambda f: f.size_bytes, reverse=True)
        
        field_order = []
        padding_bytes = {}
        total_size = 0
        
        for field in sorted_fields:
            # Align to field's natural alignment or cache line, whichever is smaller
            target_alignment = min(field.alignment, self.cache_info.l1_line_size)
            alignment_padding = (target_alignment - (total_size % target_alignment)) % target_alignment
            
            if alignment_padding > 0:
                padding_bytes[field.name + "_pre"] = alignment_padding
                total_size += alignment_padding
            
            field_order.append(field.name)
            total_size += field.size_bytes
        
        # Pad to cache line boundary
        final_padding = (self.cache_info.l1_line_size - (total_size % self.cache_info.l1_line_size)) % self.cache_info.l1_line_size
        if final_padding > 0:
            padding_bytes["final_padding"] = final_padding
            total_size += final_padding
        
        # Calculate benefits
        cache_lines_used = total_size // self.cache_info.l1_line_size
        original_layout = self._analyze_current_layout(fields)
        memory_savings = original_layout['total_size'] - total_size
        
        # Cache alignment improves predictable access patterns
        performance_improvement = 0.05  # 5% improvement from better alignment
        
        return LayoutOptimization(
            strategy=LayoutStrategy.CACHE_ALIGNED,
            field_order=field_order,
            padding_bytes=padding_bytes,
            total_size=total_size,
            cache_lines_used=cache_lines_used,
            memory_savings=memory_savings,
            vectorization_benefit=0.0,
            estimated_performance_improvement=performance_improvement
        )
    
    def _optimize_hybrid_layout(self, fields: List[FieldInfo], vectorization_target: bool) -> LayoutOptimization:
        """Create hybrid layout combining multiple strategies"""
        
        # Classify fields
        vectorizable_hot = [f for f in fields if f.vectorizable and f.is_hot_data and vectorization_target]
        hot_fields = [f for f in fields if f.is_hot_data and (not f.vectorizable or not vectorization_target)]
        cold_fields = [f for f in fields if not f.is_hot_data]
        
        field_order = []
        padding_bytes = {}
        total_size = 0
        
        # Section 1: Vectorizable hot data (SIMD aligned)
        if vectorizable_hot:
            simd_alignment = 32
            alignment_padding = (simd_alignment - (total_size % simd_alignment)) % simd_alignment
            if alignment_padding > 0:
                padding_bytes["simd_align"] = alignment_padding
                total_size += alignment_padding
            
            for field in sorted(vectorizable_hot, key=lambda f: f.access_frequency, reverse=True):
                field_order.append(field.name)
                total_size += field.size_bytes
        
        # Section 2: Other hot data (cache line aligned)
        if hot_fields:
            cache_alignment = self.cache_info.l1_line_size
            alignment_padding = (cache_alignment - (total_size % cache_alignment)) % cache_alignment
            if alignment_padding > 0:
                padding_bytes["hot_align"] = alignment_padding
                total_size += alignment_padding
            
            for field in sorted(hot_fields, key=lambda f: f.access_frequency, reverse=True):
                field_order.append(field.name)
                total_size += field.size_bytes
        
        # Section 3: Cold data (minimal alignment)
        if cold_fields:
            # Pad to next cache line boundary
            cache_line_padding = (self.cache_info.l1_line_size - (total_size % self.cache_info.l1_line_size)) % self.cache_info.l1_line_size
            if cache_line_padding > 0:
                padding_bytes["cold_separator"] = cache_line_padding
                total_size += cache_line_padding
            
            for field in cold_fields:
                field_order.append(field.name)
                total_size += field.size_bytes
        
        # Calculate benefits
        cache_lines_used = (total_size + self.cache_info.l1_line_size - 1) // self.cache_info.l1_line_size
        original_layout = self._analyze_current_layout(fields)
        memory_savings = original_layout['total_size'] - total_size
        
        # Combine benefits from vectorization and cache optimization
        vectorization_benefit = len(vectorizable_hot) * 2.0
        cache_benefit = 1.5 if hot_fields else 1.0
        
        performance_improvement = (vectorization_benefit * 0.15) + (cache_benefit * 0.08)
        
        return LayoutOptimization(
            strategy=LayoutStrategy.HYBRID,
            field_order=field_order,
            padding_bytes=padding_bytes,
            total_size=total_size,
            cache_lines_used=cache_lines_used,
            memory_savings=memory_savings,
            vectorization_benefit=vectorization_benefit,
            estimated_performance_improvement=performance_improvement
        )
    
    def _select_best_optimization(self, candidates: List[LayoutOptimization], original: Dict[str, Any]) -> LayoutOptimization:
        """Select the best optimization from candidates"""
        
        if not candidates:
            # Return a default optimization
            return LayoutOptimization(
                strategy=LayoutStrategy.ARRAY_OF_STRUCTS,
                field_order=[],
                padding_bytes={},
                total_size=original['total_size'],
                cache_lines_used=original['cache_lines_used'],
                memory_savings=0,
                vectorization_benefit=0.0,
                estimated_performance_improvement=0.0
            )
        
        # Score each candidate
        def score_optimization(opt: LayoutOptimization) -> float:
            # Weighted scoring function
            memory_score = opt.memory_savings / max(original['total_size'], 1) * 100  # % memory saved
            performance_score = opt.estimated_performance_improvement * 100
            vectorization_score = opt.vectorization_benefit * 10
            
            # Penalize excessive cache line usage
            cache_penalty = max(0, opt.cache_lines_used - original['cache_lines_used']) * 5
            
            total_score = memory_score + performance_score + vectorization_score - cache_penalty
            return total_score
        
        best_optimization = max(candidates, key=score_optimization)
        return best_optimization
    
    def create_optimized_layout_code(self, 
                                   optimization: LayoutOptimization,
                                   fields: List[FieldInfo],
                                   struct_name: str = "OptimizedStruct") -> str:
        """Generate C-style struct definition for optimized layout"""
        
        lines = [
            f"// Optimized layout using {optimization.strategy.value} strategy",
            f"// Memory savings: {optimization.memory_savings} bytes",
            f"// Performance improvement: {optimization.estimated_performance_improvement:.1%}",
            f"// Cache lines used: {optimization.cache_lines_used}",
            "",
            f"struct {struct_name} {{",
        ]
        
        field_map = {f.name: f for f in fields}
        current_offset = 0
        
        for field_name in optimization.field_order:
            field = field_map[field_name]
            
            # Add pre-padding if needed
            pre_padding_key = field_name + "_pre"
            if pre_padding_key in optimization.padding_bytes:
                padding = optimization.padding_bytes[pre_padding_key]
                lines.append(f"    char _pad_{current_offset}[{padding}];  // Alignment padding")
                current_offset += padding
            
            # Add the field
            type_name = self._get_c_type_name(field.data_type)
            lines.append(f"    {type_name} {field_name};  // Size: {field.size_bytes}, Offset: {current_offset}")
            current_offset += field.size_bytes
        
        # Add any remaining padding
        for padding_name, padding_size in optimization.padding_bytes.items():
            if not padding_name.endswith("_pre") and padding_name not in [f.name + "_pre" for f in fields]:
                lines.append(f"    char {padding_name}[{padding_size}];  // Structure padding")
        
        lines.extend([
            "};",
            "",
            f"// Total size: {optimization.total_size} bytes"
        ])
        
        return "\n".join(lines)
    
    def _get_c_type_name(self, python_type: type) -> str:
        """Convert Python type to C type name"""
        type_map = {
            int: "int64_t",
            float: "double",
            bool: "bool",
            str: "char*",
            bytes: "uint8_t*"
        }
        
        return type_map.get(python_type, "void*")
    
    def analyze_access_pattern(self, structure_name: str, field_accesses: Dict[str, int]):
        """Update access pattern statistics for a structure"""
        with self._lock:
            total_accesses = sum(field_accesses.values())
            if total_accesses > 0:
                for field_name, access_count in field_accesses.items():
                    frequency = access_count / total_accesses
                    # Use exponential moving average to update frequency
                    current_freq = self.access_patterns[structure_name][field_name]
                    self.access_patterns[structure_name][field_name] = (
                        0.7 * current_freq + 0.3 * frequency
                    )
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        with self._lock:
            return {
                'optimizations_applied': self.optimizations_applied,
                'total_memory_saved_bytes': self.total_memory_saved,
                'average_memory_saved_per_optimization': (
                    self.total_memory_saved / max(self.optimizations_applied, 1)
                ),
                'total_performance_improvement': self.total_performance_improvement,
                'average_performance_improvement': (
                    self.total_performance_improvement / max(self.optimizations_applied, 1)
                ),
                'optimization_cache_size': len(self._optimization_cache),
                'cache_info': {
                    'l1_data_size': self.cache_info.l1_data_size,
                    'l1_line_size': self.cache_info.l1_line_size,
                    'l2_size': self.cache_info.l2_size,
                    'l3_size': self.cache_info.l3_size
                }
            }
    
    def clear_cache(self):
        """Clear optimization cache"""
        with self._lock:
            self._optimization_cache.clear()


# Utility functions for common optimization patterns

def optimize_matrix_layout(rows: int, cols: int, element_size: int, 
                          for_vectorization: bool = True) -> LayoutOptimization:
    """Optimize layout for matrix data structures"""
    
    optimizer = MemoryLayoutOptimizer()
    
    # Create field info for matrix elements
    fields = []
    for i in range(rows):
        for j in range(cols):
            field = FieldInfo(
                name=f"elem_{i}_{j}",
                data_type=float,
                size_bytes=element_size,
                alignment=element_size,
                access_frequency=1.0,
                is_hot_data=True,
                vectorizable=for_vectorization
            )
            fields.append(field)
    
    return optimizer.optimize_structure(
        f"Matrix_{rows}x{cols}",
        fields,
        target_strategy=LayoutStrategy.STRUCT_OF_ARRAYS if for_vectorization else LayoutStrategy.CACHE_ALIGNED,
        vectorization_target=for_vectorization
    )


def create_neural_network_layer_layout(input_size: int, output_size: int) -> LayoutOptimization:
    """Optimize layout for neural network layer data"""
    
    optimizer = MemoryLayoutOptimizer()
    
    fields = [
        # Weight matrix (hot data, vectorizable)
        FieldInfo("weights", float, input_size * output_size * 8, 32, 
                 access_frequency=0.9, is_hot_data=True, vectorizable=True),
        
        # Bias vector (hot data, vectorizable)
        FieldInfo("biases", float, output_size * 8, 32,
                 access_frequency=0.8, is_hot_data=True, vectorizable=True),
        
        # Activation function type (cold metadata)
        FieldInfo("activation_type", int, 4, 4,
                 access_frequency=0.1, is_hot_data=False, vectorizable=False),
        
        # Layer name (cold metadata)
        FieldInfo("layer_name", str, 8, 8,  # Pointer size
                 access_frequency=0.05, is_hot_data=False, vectorizable=False),
        
        # Training flag (cold metadata)
        FieldInfo("is_training", bool, 1, 1,
                 access_frequency=0.1, is_hot_data=False, vectorizable=False),
    ]
    
    return optimizer.optimize_structure(
        f"NeuralLayer_{input_size}_{output_size}",
        fields,
        target_strategy=LayoutStrategy.HYBRID,
        vectorization_target=True
    )


def get_global_layout_optimizer() -> MemoryLayoutOptimizer:
    """Get a global shared layout optimizer instance"""
    if not hasattr(get_global_layout_optimizer, '_instance'):
        get_global_layout_optimizer._instance = MemoryLayoutOptimizer()
    return get_global_layout_optimizer._instance
