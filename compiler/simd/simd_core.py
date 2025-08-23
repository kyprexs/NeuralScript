"""
Core SIMD Processing System

Main SIMD processor implementation with hardware detection,
capability analysis, and vectorization orchestration.
"""

import platform
import subprocess
import threading
import time
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum, auto
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np


class SIMDInstructionSet(Enum):
    """Supported SIMD instruction sets"""
    # x86 instruction sets
    SSE = auto()
    SSE2 = auto() 
    SSE3 = auto()
    SSSE3 = auto()
    SSE4_1 = auto()
    SSE4_2 = auto()
    AVX = auto()
    AVX2 = auto()
    AVX512F = auto()
    AVX512DQ = auto()
    AVX512BW = auto()
    AVX512VL = auto()
    
    # ARM instruction sets
    NEON = auto()
    SVE = auto()
    SVE2 = auto()
    
    # Fallback
    SCALAR = auto()


class DataType(Enum):
    """Supported data types for SIMD operations"""
    FLOAT32 = auto()
    FLOAT64 = auto()
    INT8 = auto()
    INT16 = auto()
    INT32 = auto()
    INT64 = auto()
    UINT8 = auto()
    UINT16 = auto()
    UINT32 = auto()
    UINT64 = auto()
    COMPLEX64 = auto()
    COMPLEX128 = auto()


@dataclass
class SIMDCapabilities:
    """Hardware SIMD capabilities"""
    instruction_sets: Set[SIMDInstructionSet]
    vector_widths: Dict[SIMDInstructionSet, int]  # bits
    supported_types: Dict[SIMDInstructionSet, Set[DataType]]
    cache_sizes: Dict[str, int]  # L1, L2, L3 cache sizes
    cpu_features: Set[str]
    max_threads: int
    
    @property
    def best_instruction_set(self) -> SIMDInstructionSet:
        """Property access to best instruction set"""
        return self.get_best_instruction_set()
    
    @property
    def has_fma(self) -> bool:
        """Check if FMA is supported"""
        return 'fma' in self.cpu_features or 'avx2' in self.cpu_features or 'avx512f' in self.cpu_features
    
    @property
    def thread_count(self) -> int:
        """Alias for max_threads"""
        return self.max_threads
    
    @property
    def supported_data_types(self) -> Set[DataType]:
        """Get all supported data types across instruction sets"""
        all_types = set()
        for types_set in self.supported_types.values():
            all_types.update(types_set)
        return all_types
    
    def get_best_instruction_set(self) -> SIMDInstructionSet:
        """Get the best available instruction set"""
        # Priority order for x86
        priority_order = [
            SIMDInstructionSet.AVX512F,
            SIMDInstructionSet.AVX2,
            SIMDInstructionSet.AVX,
            SIMDInstructionSet.SSE4_2,
            SIMDInstructionSet.SSE4_1,
            SIMDInstructionSet.SSSE3,
            SIMDInstructionSet.SSE3,
            SIMDInstructionSet.SSE2,
            SIMDInstructionSet.SSE,
            SIMDInstructionSet.NEON,  # ARM
            SIMDInstructionSet.SVE2,
            SIMDInstructionSet.SVE,
            SIMDInstructionSet.SCALAR
        ]
        
        for instruction_set in priority_order:
            if instruction_set in self.instruction_sets:
                return instruction_set
        
        return SIMDInstructionSet.SCALAR
    
    def get_vector_width(self, instruction_set: SIMDInstructionSet) -> int:
        """Get vector width in bits for instruction set"""
        return self.vector_widths.get(instruction_set, 64)
    
    def supports_type(self, instruction_set: SIMDInstructionSet, data_type: DataType) -> bool:
        """Check if instruction set supports data type"""
        supported = self.supported_types.get(instruction_set, set())
        return data_type in supported


@dataclass  
class SIMDConfiguration:
    """Configuration for SIMD operations"""
    preferred_instruction_set: Optional[SIMDInstructionSet] = None
    auto_vectorize: bool = True
    vectorization_threshold: int = 4  # Minimum elements to vectorize
    alignment_bytes: int = 32  # Memory alignment for vectors
    use_fma: bool = True  # Use fused multiply-add
    parallel_threshold: int = 10000  # Elements threshold for parallelization
    max_unroll_factor: int = 8
    enable_optimizations: bool = True
    debug_mode: bool = False


class HardwareDetector:
    """Hardware capability detection"""
    
    @staticmethod
    def detect_cpu_features() -> Set[str]:
        """Detect CPU features using platform-specific methods"""
        features = set()
        
        try:
            if platform.system() == 'Windows':
                # Use wmic or PowerShell
                result = subprocess.run(['wmic', 'cpu', 'get', 'Name'], 
                                      capture_output=True, text=True, timeout=5)
                if 'Intel' in result.stdout:
                    features.add('intel')
                elif 'AMD' in result.stdout:
                    features.add('amd')
                    
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Try to detect instruction sets via cpuinfo or other methods
        try:
            if hasattr(subprocess, 'check_output'):
                # Linux cpuinfo
                if platform.system() == 'Linux':
                    with open('/proc/cpuinfo', 'r') as f:
                        cpuinfo = f.read().lower()
                        
                    if 'avx512f' in cpuinfo:
                        features.update(['avx512f', 'avx512dq', 'avx512bw'])
                    if 'avx2' in cpuinfo:
                        features.add('avx2')
                    if 'avx' in cpuinfo:
                        features.add('avx')
                    if 'sse4_2' in cpuinfo:
                        features.add('sse4_2')
                    if 'sse4_1' in cpuinfo:
                        features.add('sse4_1')
                    if 'ssse3' in cpuinfo:
                        features.add('ssse3')
                    if 'sse3' in cpuinfo:
                        features.add('sse3')
                    if 'sse2' in cpuinfo:
                        features.add('sse2')
                    if 'neon' in cpuinfo:
                        features.add('neon')
                        
        except Exception:
            pass
        
        # Fallback feature detection
        if not features:
            # Assume basic SSE2 support on x86-64
            if platform.machine() in ['x86_64', 'AMD64']:
                features.update(['sse', 'sse2'])
            elif 'arm' in platform.machine().lower():
                features.add('neon')
        
        return features
    
    @staticmethod
    def detect_cache_sizes() -> Dict[str, int]:
        """Detect CPU cache sizes"""
        cache_sizes = {
            'L1': 32 * 1024,   # 32KB default
            'L2': 256 * 1024,  # 256KB default  
            'L3': 8 * 1024 * 1024  # 8MB default
        }
        
        try:
            if platform.system() == 'Linux':
                # Try to read cache info from sysfs
                for level in ['L1', 'L2', 'L3']:
                    try:
                        cache_path = f'/sys/devices/system/cpu/cpu0/cache/index{level[1]}/size'
                        with open(cache_path, 'r') as f:
                            size_str = f.read().strip()
                            if size_str.endswith('K'):
                                cache_sizes[level] = int(size_str[:-1]) * 1024
                            elif size_str.endswith('M'):
                                cache_sizes[level] = int(size_str[:-1]) * 1024 * 1024
                    except (FileNotFoundError, ValueError):
                        continue
                        
        except Exception:
            pass
        
        return cache_sizes
    
    @staticmethod
    def create_capabilities() -> SIMDCapabilities:
        """Create capabilities object from hardware detection"""
        features = HardwareDetector.detect_cpu_features()
        cache_sizes = HardwareDetector.detect_cache_sizes()
        
        # Map features to instruction sets
        instruction_sets = set()
        vector_widths = {}
        supported_types = {}
        
        # x86 instruction sets
        if 'sse' in features:
            instruction_sets.add(SIMDInstructionSet.SSE)
            vector_widths[SIMDInstructionSet.SSE] = 128
            supported_types[SIMDInstructionSet.SSE] = {
                DataType.FLOAT32, DataType.INT32, DataType.INT16, DataType.INT8
            }
            
        if 'sse2' in features:
            instruction_sets.add(SIMDInstructionSet.SSE2)
            vector_widths[SIMDInstructionSet.SSE2] = 128
            supported_types[SIMDInstructionSet.SSE2] = {
                DataType.FLOAT32, DataType.FLOAT64, DataType.INT64, 
                DataType.INT32, DataType.INT16, DataType.INT8
            }
            
        if 'sse3' in features:
            instruction_sets.add(SIMDInstructionSet.SSE3)
            vector_widths[SIMDInstructionSet.SSE3] = 128
            supported_types[SIMDInstructionSet.SSE3] = supported_types[SIMDInstructionSet.SSE2]
            
        if 'ssse3' in features:
            instruction_sets.add(SIMDInstructionSet.SSSE3)
            vector_widths[SIMDInstructionSet.SSSE3] = 128
            supported_types[SIMDInstructionSet.SSSE3] = supported_types[SIMDInstructionSet.SSE2]
            
        if 'sse4_1' in features:
            instruction_sets.add(SIMDInstructionSet.SSE4_1)
            vector_widths[SIMDInstructionSet.SSE4_1] = 128
            supported_types[SIMDInstructionSet.SSE4_1] = supported_types[SIMDInstructionSet.SSE2]
            
        if 'sse4_2' in features:
            instruction_sets.add(SIMDInstructionSet.SSE4_2)
            vector_widths[SIMDInstructionSet.SSE4_2] = 128
            supported_types[SIMDInstructionSet.SSE4_2] = supported_types[SIMDInstructionSet.SSE2]
            
        if 'avx' in features:
            instruction_sets.add(SIMDInstructionSet.AVX)
            vector_widths[SIMDInstructionSet.AVX] = 256
            supported_types[SIMDInstructionSet.AVX] = {
                DataType.FLOAT32, DataType.FLOAT64, DataType.INT32, DataType.INT64
            }
            
        if 'avx2' in features:
            instruction_sets.add(SIMDInstructionSet.AVX2)
            vector_widths[SIMDInstructionSet.AVX2] = 256
            supported_types[SIMDInstructionSet.AVX2] = {
                DataType.FLOAT32, DataType.FLOAT64, DataType.INT64,
                DataType.INT32, DataType.INT16, DataType.INT8,
                DataType.UINT64, DataType.UINT32, DataType.UINT16, DataType.UINT8
            }
            
        if 'avx512f' in features:
            instruction_sets.add(SIMDInstructionSet.AVX512F)
            vector_widths[SIMDInstructionSet.AVX512F] = 512
            supported_types[SIMDInstructionSet.AVX512F] = {
                DataType.FLOAT32, DataType.FLOAT64, DataType.INT32, DataType.INT64
            }
            
        # ARM instruction sets
        if 'neon' in features:
            instruction_sets.add(SIMDInstructionSet.NEON)
            vector_widths[SIMDInstructionSet.NEON] = 128
            supported_types[SIMDInstructionSet.NEON] = {
                DataType.FLOAT32, DataType.INT32, DataType.INT16, DataType.INT8,
                DataType.UINT32, DataType.UINT16, DataType.UINT8
            }
        
        # Always add scalar fallback
        instruction_sets.add(SIMDInstructionSet.SCALAR)
        vector_widths[SIMDInstructionSet.SCALAR] = 64
        supported_types[SIMDInstructionSet.SCALAR] = {
            DataType.FLOAT32, DataType.FLOAT64, DataType.INT64,
            DataType.INT32, DataType.INT16, DataType.INT8,
            DataType.UINT64, DataType.UINT32, DataType.UINT16, DataType.UINT8,
            DataType.COMPLEX64, DataType.COMPLEX128
        }
        
        return SIMDCapabilities(
            instruction_sets=instruction_sets,
            vector_widths=vector_widths,
            supported_types=supported_types,
            cache_sizes=cache_sizes,
            cpu_features=features,
            max_threads=max(1, (threading.active_count() * 2) if hasattr(threading, 'active_count') else 4)
        )


class SIMDProcessor:
    """
    Main SIMD processor for NeuralScript.
    
    Handles SIMD operation dispatching, optimization, and execution
    with automatic hardware detection and adaptive vectorization.
    """
    
    def __init__(self, config: Optional[SIMDConfiguration] = None):
        self.config = config or SIMDConfiguration()
        self.capabilities = HardwareDetector.create_capabilities()
        
        # Select optimal instruction set
        if self.config.preferred_instruction_set:
            if self.config.preferred_instruction_set in self.capabilities.instruction_sets:
                self.instruction_set = self.config.preferred_instruction_set
            else:
                self.instruction_set = self.capabilities.get_best_instruction_set()
        else:
            self.instruction_set = self.capabilities.get_best_instruction_set()
        
        # Performance statistics
        self._operation_count = 0
        self._total_execution_time = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize subsystems
        self._initialize_subsystems()
        
        if self.config.debug_mode:
            print(f"SIMD Processor initialized with {self.instruction_set.name}")
            print(f"Vector width: {self.capabilities.get_vector_width(self.instruction_set)} bits")
    
    def _initialize_subsystems(self):
        """Initialize SIMD subsystems"""
        # This will be expanded when we add the other modules
        pass
    
    def get_vector_width(self, data_type: DataType) -> int:
        """Get vector width for data type in elements"""
        vector_bits = self.capabilities.get_vector_width(self.instruction_set)
        
        type_sizes = {
            DataType.FLOAT32: 32,
            DataType.FLOAT64: 64,
            DataType.INT8: 8,
            DataType.INT16: 16,
            DataType.INT32: 32,
            DataType.INT64: 64,
            DataType.UINT8: 8,
            DataType.UINT16: 16,
            DataType.UINT32: 32,
            DataType.UINT64: 64,
            DataType.COMPLEX64: 64,
            DataType.COMPLEX128: 128,
        }
        
        element_bits = type_sizes.get(data_type, 32)
        return vector_bits // element_bits
    
    def supports_type(self, data_type: DataType) -> bool:
        """Check if current instruction set supports data type"""
        return self.capabilities.supports_type(self.instruction_set, data_type)
    
    def should_vectorize(self, length: int, data_type: DataType) -> bool:
        """Determine if operation should be vectorized"""
        if not self.config.auto_vectorize:
            return False
        
        if not self.supports_type(data_type):
            return False
        
        if length < self.config.vectorization_threshold:
            return False
        
        vector_width = self.get_vector_width(data_type)
        return length >= vector_width
    
    def get_optimal_chunk_size(self, length: int, data_type: DataType) -> int:
        """Get optimal chunk size for processing"""
        vector_width = self.get_vector_width(data_type)
        
        # Consider cache line size
        cache_line_size = 64  # bytes
        element_size = {
            DataType.FLOAT32: 4, DataType.FLOAT64: 8,
            DataType.INT32: 4, DataType.INT64: 8,
            DataType.INT16: 2, DataType.INT8: 1,
        }.get(data_type, 4)
        
        elements_per_cache_line = cache_line_size // element_size
        
        # Optimal chunk size balances vector width and cache efficiency
        optimal_chunk = max(vector_width, elements_per_cache_line)
        
        # Ensure chunk size divides evenly or handle remainder
        if length < optimal_chunk:
            return length
        
        return optimal_chunk
    
    def calculate_optimal_chunk_size(self, length: int, data_type: DataType) -> int:
        """Alias for get_optimal_chunk_size for compatibility"""
        return self.get_optimal_chunk_size(length, data_type)
    
    def benchmark_operation(self, operation_name: str, iterations: int = 1000) -> Dict[str, float]:
        """Benchmark SIMD operation performance"""
        # Create test data
        size = 1024
        if operation_name.startswith('vector'):
            a = np.random.random(size).astype(np.float32)
            b = np.random.random(size).astype(np.float32) 
        elif operation_name.startswith('matrix'):
            a = np.random.random((32, 32)).astype(np.float32)
            b = np.random.random((32, 32)).astype(np.float32)
        else:
            return {'error': 'Unknown operation'}
        
        # Warmup
        for _ in range(10):
            if operation_name == 'vector_add':
                _ = a + b
            elif operation_name == 'vector_mul':
                _ = a * b
            elif operation_name == 'matrix_mul':
                _ = np.dot(a, b)
        
        # Benchmark
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            if operation_name == 'vector_add':
                result = a + b
            elif operation_name == 'vector_mul':
                result = a * b
            elif operation_name == 'matrix_mul':
                result = np.dot(a, b)
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        if operation_name.startswith('vector'):
            operations_per_second = size / avg_time
        elif operation_name.startswith('matrix'):
            # Matrix multiplication: 2 * N^3 operations for NxN matrices
            n = a.shape[0]
            operations_per_second = (2 * n * n * n) / avg_time
        else:
            operations_per_second = 1 / avg_time
        
        return {
            'total_time_seconds': total_time,
            'average_time_seconds': avg_time,
            'operations_per_second': operations_per_second,
            'iterations': iterations,
            'instruction_set': self.instruction_set.name,
            'vector_width': self.capabilities.get_vector_width(self.instruction_set)
        }
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self._lock:
            avg_execution_time = (self._total_execution_time / max(1, self._operation_count))
            cache_hit_rate = (self._cache_hits / max(1, self._cache_hits + self._cache_misses))
            
            return {
                'total_operations': self._operation_count,
                'total_execution_time': self._total_execution_time,
                'average_execution_time': avg_execution_time,
                'operations_per_second': 1.0 / max(avg_execution_time, 1e-9),
                'cache_hit_rate': cache_hit_rate,
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses,
                'instruction_set': self.instruction_set.name,
                'capabilities': {
                    'vector_width_bits': self.capabilities.get_vector_width(self.instruction_set),
                    'supported_instruction_sets': [iset.name for iset in self.capabilities.instruction_sets],
                    'cpu_features': list(self.capabilities.cpu_features),
                    'cache_sizes': self.capabilities.cache_sizes
                }
            }
    
    def reset_statistics(self):
        """Reset performance statistics"""
        with self._lock:
            self._operation_count = 0
            self._total_execution_time = 0.0
            self._cache_hits = 0
            self._cache_misses = 0
    
    def generate_report(self) -> str:
        """Generate comprehensive SIMD system report"""
        stats = self.get_performance_statistics()
        
        report_lines = []
        report_lines.append("NeuralScript SIMD Processor Report")
        report_lines.append("=" * 50)
        
        # Hardware capabilities
        report_lines.append("\nHardware Capabilities:")
        report_lines.append(f"  • Active Instruction Set: {stats['instruction_set']}")
        report_lines.append(f"  • Vector Width: {stats['capabilities']['vector_width_bits']} bits")
        report_lines.append(f"  • CPU Features: {', '.join(stats['capabilities']['cpu_features'])}")
        
        cache_info = stats['capabilities']['cache_sizes']
        report_lines.append(f"  • Cache Sizes: L1={cache_info['L1']//1024}KB, "
                          f"L2={cache_info['L2']//1024}KB, L3={cache_info['L3']//1024//1024}MB")
        
        # Performance statistics
        report_lines.append("\nPerformance Statistics:")
        if stats['total_operations'] > 0:
            report_lines.append(f"  • Total Operations: {stats['total_operations']:,}")
            report_lines.append(f"  • Operations/Second: {stats['operations_per_second']:,.0f}")
            report_lines.append(f"  • Average Execution Time: {stats['average_execution_time']*1000:.3f} ms")
            report_lines.append(f"  • Cache Hit Rate: {stats['cache_hit_rate']*100:.1f}%")
        else:
            report_lines.append("  • No operations executed yet")
        
        # Configuration
        report_lines.append("\nConfiguration:")
        report_lines.append(f"  • Auto-vectorization: {'Enabled' if self.config.auto_vectorize else 'Disabled'}")
        report_lines.append(f"  • Vectorization Threshold: {self.config.vectorization_threshold} elements")
        report_lines.append(f"  • Memory Alignment: {self.config.alignment_bytes} bytes")
        report_lines.append(f"  • FMA Support: {'Enabled' if self.config.use_fma else 'Disabled'}")
        
        return "\n".join(report_lines)

    def get_performance_report(self) -> Dict[str, Any]:
        """Structured performance report used by other modules and demos"""
        stats = self.get_performance_statistics()
        best_iset = self.capabilities.best_instruction_set.name
        
        # Vector widths mapping by instruction set name
        vector_widths = {}
        for iset, width in self.capabilities.vector_widths.items():
            vector_widths[iset.name] = width
        
        supported_data_types = []
        for dt in self.capabilities.supported_data_types:
            supported_data_types.append(dt.name)
        
        return {
            'active_instruction_set': self.instruction_set.name,
            'capabilities': {
                'best_instruction_set': best_iset,
                'instruction_sets': [iset.name for iset in self.capabilities.instruction_sets],
                'vector_widths': vector_widths,
                'supported_data_types': supported_data_types,
                'has_fma': self.capabilities.has_fma,
                'cpu_features': list(self.capabilities.cpu_features),
                'cache_sizes': self.capabilities.cache_sizes,
                'thread_count': self.capabilities.thread_count,
            },
            'performance_stats': {
                'operation_count': stats['total_operations'],
                'total_execution_time': stats['total_execution_time'],
                'avg_operation_time': stats['average_execution_time'],
            },
            'configuration': {
                'auto_vectorize': self.config.auto_vectorize,
                'vectorization_threshold': self.config.vectorization_threshold,
                'use_fma': self.config.use_fma,
                'max_unroll_factor': self.config.max_unroll_factor,
                'alignment_bytes': self.config.alignment_bytes,
            },
        }
    
    def get_available_instruction_sets(self) -> List[str]:
        """Get list of available SIMD instruction sets as strings"""
        return [iset.name for iset in self.capabilities.instruction_sets]
