"""
SIMD Vectorization System for NeuralScript

High-performance SIMD (Single Instruction, Multiple Data) vectorization
for mathematical operations optimized for scientific computing and 
machine learning workloads.

Author: Assistant for NeuralScript
"""

from .simd_core import (
    SIMDProcessor, SIMDConfiguration, SIMDCapabilities,
    DataType, SIMDInstructionSet as InstructionSet
)
from .vector_operations import (
    VectorOperations, VectorMath, VectorComparison,
    VectorLogical, VectorTranscendental
)
from .matrix_operations import (
    MatrixOperations, MatrixDecomposition,
    MatrixSolvers, ConvolutionOperations, ActivationFunctions,
    BatchOperations
)
from .optimizer import (
    AutoVectorizer, OptimizationLevel, VectorizationStrategy,
    PerformanceProfiler, OptimizationHint, AdaptiveOptimizer
)

__all__ = [
    # Core SIMD system
    'SIMDProcessor', 'SIMDConfiguration', 'SIMDCapabilities',
    'DataType', 'InstructionSet',
    
    # Vector operations
    'VectorOperations', 'VectorMath', 'VectorComparison',
    'VectorLogical', 'VectorTranscendental',
    
    # Matrix operations
    'MatrixOperations', 'MatrixDecomposition', 'MatrixSolvers',
    'ConvolutionOperations', 'ActivationFunctions', 'BatchOperations',
    
    # Optimization system
    'AutoVectorizer', 'OptimizationLevel', 'VectorizationStrategy',
    'PerformanceProfiler', 'OptimizationHint', 'AdaptiveOptimizer',
]

# Version information
__version__ = "1.0.0"
__author__ = "NeuralScript Team"
__description__ = "Production-grade SIMD vectorization system"

# Performance benchmarks (operations per second)
PERFORMANCE_TARGETS = {
    'vector_add_f32': 10_000_000_000,      # 10 GFLOPS
    'vector_mul_f32': 8_000_000_000,       # 8 GFLOPS  
    'vector_fma_f32': 15_000_000_000,      # 15 GFLOPS
    'matrix_mul_f32': 5_000_000_000,       # 5 GFLOPS
    'vector_sqrt_f32': 2_000_000_000,      # 2 GFLOPS
    'vector_sin_f32': 500_000_000,         # 500 MFLOPS
}

# Supported instruction sets
SUPPORTED_INSTRUCTION_SETS = [
    'SSE', 'SSE2', 'SSE3', 'SSSE3', 'SSE4.1', 'SSE4.2',
    'AVX', 'AVX2', 'AVX512F', 'AVX512DQ', 'AVX512BW',
    'NEON', 'SVE', 'SVE2'  # ARM support
]
