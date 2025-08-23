#!/usr/bin/env python3
"""
SIMD System Demonstration and Test Suite

Comprehensive test and demonstration of the NeuralScript SIMD vectorization system.
Shows all components working together with performance analysis.
"""

import numpy as np
import time
import sys
import os

# Add the compiler directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'compiler'))

from compiler.simd import (
    SIMDProcessor, VectorOperations, VectorMath, VectorTranscendental,
    MatrixOperations, ConvolutionOperations, ActivationFunctions,
    AutoVectorizer, OptimizationHint, OptimizationLevel,
    DataType, VectorizationStrategy, PerformanceProfiler
)

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'=' * 80}")
    print(f" {title}")
    print('=' * 80)

def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n{'-' * 60}")
    print(f" {title}")
    print('-' * 60)

def demonstrate_hardware_detection():
    """Demonstrate hardware detection and SIMD capabilities"""
    print_section("SIMD Hardware Detection and Capabilities")
    
    simd = SIMDProcessor()
    capabilities = simd.capabilities
    
    print(f"CPU Information:")
    print(f"  Best Instruction Set: {capabilities.best_instruction_set}")
    print(f"  Available Instruction Sets: {list(capabilities.instruction_sets)}")
    print(f"  Vector Widths: {capabilities.vector_widths}")
    print(f"  Supported Data Types: {list(capabilities.supported_data_types)}")
    print(f"  Has FMA Support: {capabilities.has_fma}")
    print(f"  Thread Count: {capabilities.thread_count}")
    
    print(f"\nCache Information:")
    for level, size in capabilities.cache_sizes.items():
        print(f"  {level}: {size // 1024} KB")
    
    print(f"\nCPU Features: {list(capabilities.cpu_features)}")
    
    return simd

def demonstrate_vector_operations(simd: SIMDProcessor):
    """Demonstrate basic vector operations"""
    print_section("Vector Operations Demonstration")
    
    vec_ops = VectorOperations(simd)
    
    # Create test vectors
    size = 10000
    a = np.random.random(size).astype(np.float32)
    b = np.random.random(size).astype(np.float32)
    c = np.random.random(size).astype(np.float32)
    
    print_subsection("Basic Arithmetic Operations")
    
    # Test basic operations
    start_time = time.perf_counter()
    result_add = vec_ops.add(a, b)
    add_time = time.perf_counter() - start_time
    print(f"Vector Addition ({size} elements): {add_time*1000:.3f} ms")
    
    start_time = time.perf_counter()
    result_mul = vec_ops.multiply(a, b)
    mul_time = time.perf_counter() - start_time
    print(f"Vector Multiplication ({size} elements): {mul_time*1000:.3f} ms")
    
    start_time = time.perf_counter()
    result_fma = vec_ops.fused_multiply_add(a, b, c)
    fma_time = time.perf_counter() - start_time
    print(f"Fused Multiply-Add ({size} elements): {fma_time*1000:.3f} ms")
    
    print_subsection("Vector Geometry Operations")
    
    # 3D vector operations
    vec3d_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    vec3d_b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    
    dot_result = vec_ops.dot_product(vec3d_a, vec3d_b)
    cross_result = vec_ops.cross_product(vec3d_a, vec3d_b)
    magnitude = vec_ops.magnitude(vec3d_a)
    normalized = vec_ops.normalize(vec3d_a)
    
    print(f"Dot Product: {dot_result}")
    print(f"Cross Product: {cross_result}")
    print(f"Magnitude: {magnitude:.3f}")
    print(f"Normalized Vector: {normalized}")

def demonstrate_math_functions(simd: SIMDProcessor):
    """Demonstrate advanced mathematical functions"""
    print_section("Advanced Mathematical Functions")
    
    vec_math = VectorMath(simd)
    vec_transcendental = VectorTranscendental(simd)
    
    # Create test data
    size = 5000
    x = np.linspace(0.1, 10.0, size, dtype=np.float32)
    angles = np.linspace(0, 2*np.pi, size, dtype=np.float32)
    
    print_subsection("Mathematical Functions Performance")
    
    # Test mathematical functions
    functions = [
        ('Square Root', lambda: vec_math.sqrt(x)),
        ('Reciprocal Sqrt', lambda: vec_math.rsqrt(x)),
        ('Exponential', lambda: vec_math.exp(x)),
        ('Natural Log', lambda: vec_math.log(x)),
        ('Power (x^2.5)', lambda: vec_math.power(x, 2.5)),
        ('Absolute Value', lambda: vec_math.abs(x - 5.0))
    ]
    
    for name, func in functions:
        start_time = time.perf_counter()
        result = func()
        end_time = time.perf_counter()
        print(f"{name}: {(end_time-start_time)*1000:.3f} ms")
    
    print_subsection("Transcendental Functions Performance")
    
    # Test transcendental functions
    trig_functions = [
        ('Sine', lambda: vec_transcendental.sin(angles)),
        ('Cosine', lambda: vec_transcendental.cos(angles)),
        ('Tangent', lambda: vec_transcendental.tan(angles)),
        ('Sin/Cos Combined', lambda: vec_transcendental.sincos(angles)),
        ('Atan2', lambda: vec_transcendental.atan2(x[:len(angles)], angles))
    ]
    
    for name, func in trig_functions:
        start_time = time.perf_counter()
        result = func()
        end_time = time.perf_counter()
        print(f"{name}: {(end_time-start_time)*1000:.3f} ms")

def demonstrate_matrix_operations(simd: SIMDProcessor):
    """Demonstrate matrix operations"""
    print_section("Matrix Operations Demonstration")
    
    mat_ops = MatrixOperations(simd)
    
    # Create test matrices
    size = 500
    A = np.random.random((size, size)).astype(np.float32)
    B = np.random.random((size, size)).astype(np.float32)
    v = np.random.random(size).astype(np.float32)
    
    print_subsection("Linear Algebra Operations")
    
    # Matrix multiplication
    start_time = time.perf_counter()
    C = mat_ops.matrix_multiply(A, B)
    mm_time = time.perf_counter() - start_time
    print(f"Matrix Multiplication ({size}x{size}): {mm_time*1000:.1f} ms")
    
    # Matrix-vector multiplication
    start_time = time.perf_counter()
    result_mv = mat_ops.matrix_vector_multiply(A, v)
    mv_time = time.perf_counter() - start_time
    print(f"Matrix-Vector Multiply ({size}x{size}): {mv_time*1000:.1f} ms")
    
    # Matrix transpose
    start_time = time.perf_counter()
    A_T = mat_ops.transpose(A)
    t_time = time.perf_counter() - start_time
    print(f"Matrix Transpose ({size}x{size}): {t_time*1000:.1f} ms")
    
    print_subsection("Matrix Decomposition (smaller matrix for speed)")
    
    # Use smaller matrix for expensive operations
    small_size = 100
    A_small = A[:small_size, :small_size].copy()
    
    from compiler.simd.matrix_operations import MatrixDecomposition, MatrixSolvers
    decomp = MatrixDecomposition(simd)
    solvers = MatrixSolvers(simd)
    
    # QR decomposition
    start_time = time.perf_counter()
    Q, R = decomp.qr_decomposition(A_small)
    qr_time = time.perf_counter() - start_time
    print(f"QR Decomposition ({small_size}x{small_size}): {qr_time*1000:.1f} ms")
    
    # SVD
    start_time = time.perf_counter()
    U, s, Vt = decomp.svd(A_small)
    svd_time = time.perf_counter() - start_time
    print(f"SVD ({small_size}x{small_size}): {svd_time*1000:.1f} ms")

def demonstrate_ml_operations(simd: SIMDProcessor):
    """Demonstrate machine learning specific operations"""
    print_section("Machine Learning Operations")
    
    from compiler.simd.matrix_operations import (
        ConvolutionOperations, ActivationFunctions, BatchOperations
    )
    
    conv_ops = ConvolutionOperations(simd)
    activations = ActivationFunctions(simd)
    batch_ops = BatchOperations(simd)
    
    print_subsection("Convolution Operations")
    
    # 1D convolution
    signal = np.random.random(1000).astype(np.float32)
    kernel_1d = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    
    start_time = time.perf_counter()
    conv_1d_result = conv_ops.conv1d(signal, kernel_1d)
    conv1d_time = time.perf_counter() - start_time
    print(f"1D Convolution (signal: {len(signal)}, kernel: {len(kernel_1d)}): {conv1d_time*1000:.1f} ms")
    
    # 2D convolution
    image = np.random.random((64, 64)).astype(np.float32)
    kernel_2d = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    
    start_time = time.perf_counter()
    conv_2d_result = conv_ops.conv2d(image, kernel_2d)
    conv2d_time = time.perf_counter() - start_time
    print(f"2D Convolution (image: {image.shape}, kernel: {kernel_2d.shape}): {conv2d_time*1000:.1f} ms")
    
    print_subsection("Activation Functions")
    
    # Test activation functions
    data = np.random.randn(10000).astype(np.float32)
    
    activations_list = [
        ('ReLU', lambda: activations.relu(data)),
        ('Leaky ReLU', lambda: activations.leaky_relu(data)),
        ('Sigmoid', lambda: activations.sigmoid(data)),
        ('Tanh', lambda: activations.tanh(data)),
        ('Softmax', lambda: activations.softmax(data.reshape(100, 100)))
    ]
    
    for name, func in activations_list:
        start_time = time.perf_counter()
        result = func()
        end_time = time.perf_counter()
        print(f"{name}: {(end_time-start_time)*1000:.3f} ms")
    
    print_subsection("Batch Operations")
    
    # Batch matrix multiplication
    batch_size = 32
    batch_A = np.random.random((batch_size, 64, 32)).astype(np.float32)
    batch_B = np.random.random((batch_size, 32, 16)).astype(np.float32)
    
    start_time = time.perf_counter()
    batch_result = batch_ops.batch_matrix_multiply(batch_A, batch_B)
    batch_time = time.perf_counter() - start_time
    print(f"Batch Matrix Multiply ({batch_size} matrices): {batch_time*1000:.1f} ms")
    
    # Batch normalization
    batch_data = np.random.random((batch_size, 128)).astype(np.float32)
    start_time = time.perf_counter()
    normalized = batch_ops.batch_normalize(batch_data)
    norm_time = time.perf_counter() - start_time
    print(f"Batch Normalization ({batch_data.shape}): {norm_time*1000:.1f} ms")

def demonstrate_optimization_system(simd: SIMDProcessor):
    """Demonstrate the optimization and vectorization system"""
    print_section("Automatic Vectorization and Optimization")
    
    # Create optimizer with different levels
    from compiler.simd.optimizer import (
        AutoVectorizer, PerformanceProfiler, AdaptiveOptimizer
    )
    
    profiler = PerformanceProfiler(simd)
    optimizer = AutoVectorizer(simd, OptimizationLevel.AGGRESSIVE)
    adaptive = AdaptiveOptimizer(simd)
    
    print_subsection("Optimization Analysis")
    
    # Create optimization hints for different scenarios
    test_scenarios = [
        {
            'name': 'Small Vector Addition',
            'hint': OptimizationHint(
                operation_name='vector_add',
                data_size=100,
                data_type=DataType.FLOAT32,
                access_pattern='sequential'
            )
        },
        {
            'name': 'Large Matrix Multiplication',
            'hint': OptimizationHint(
                operation_name='matrix_multiply',
                data_size=250000,  # 500x500 matrix
                data_type=DataType.FLOAT32,
                access_pattern='sequential',
                compute_bound=True
            )
        },
        {
            'name': 'Random Access Pattern',
            'hint': OptimizationHint(
                operation_name='lookup_operation',
                data_size=10000,
                data_type=DataType.FLOAT32,
                access_pattern='random',
                memory_bound=True
            )
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\nScenario: {scenario['name']}")
        optimization = optimizer.optimize_operation(scenario['hint'])
        
        print(f"  Should Vectorize: {optimization['vectorize']}")
        print(f"  Vector Width: {optimization['vector_width']}")
        print(f"  Unroll Factor: {optimization['unroll_factor']}")
        print(f"  Use Prefetch: {optimization['use_prefetch']}")
        print(f"  Estimated Speedup: {optimization['estimated_speedup']:.2f}x")
        if 'blocking_strategy' in optimization and optimization['blocking_strategy']:
            print(f"  Blocking Strategy: {optimization['blocking_strategy']}")
    
    print_subsection("Performance Profiling")
    
    # Profile some actual operations
    vec_ops = VectorOperations(simd)
    
    # Create test data
    a = np.random.random(5000).astype(np.float32)
    b = np.random.random(5000).astype(np.float32)
    
    # Profile vector addition
    metrics = profiler.profile_operation(vec_ops.add, a, b)
    print(f"Vector Addition Profiling:")
    print(f"  Execution Time: {metrics.execution_time*1000:.3f} ms")
    print(f"  Throughput: {metrics.throughput:.0f} ops/sec")
    print(f"  Memory Bandwidth: {metrics.memory_bandwidth:.2f} GB/s")
    print(f"  FLOP Rate: {metrics.flop_rate:.0f} FLOPS")
    print(f"  Vectorization Factor: {metrics.vectorization_factor:.1f}")

def benchmark_comparison(simd: SIMDProcessor):
    """Compare SIMD operations against scalar equivalents"""
    print_section("Performance Comparison: SIMD vs Scalar")
    
    vec_ops = VectorOperations(simd)
    
    # Test different sizes
    sizes = [1000, 10000, 100000]
    
    for size in sizes:
        print_subsection(f"Array Size: {size:,} elements")
        
        # Create test data
        a = np.random.random(size).astype(np.float32)
        b = np.random.random(size).astype(np.float32)
        
        # SIMD vectorized operation
        start_time = time.perf_counter()
        result_simd = vec_ops.add(a, b)
        simd_time = time.perf_counter() - start_time
        
        # Scalar operation (pure Python would be too slow, so use numpy without explicit vectorization)
        start_time = time.perf_counter()
        result_scalar = np.add(a, b)  # This is still vectorized, but represents the baseline
        scalar_time = time.perf_counter() - start_time
        
        # Calculate speedup
        speedup = scalar_time / simd_time if simd_time > 0 else 1.0
        
        print(f"  SIMD Time: {simd_time*1000:.3f} ms")
        print(f"  Baseline Time: {scalar_time*1000:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Throughput: {size/simd_time/1e6:.1f} Mops/sec")

def generate_comprehensive_report(simd: SIMDProcessor):
    """Generate a comprehensive performance and capability report"""
    print_section("Comprehensive SIMD System Report")
    
    report = simd.get_performance_report()
    
    print("System Configuration:")
    print(f"  Active Instruction Set: {report['active_instruction_set']}")
    print(f"  Hardware Capabilities: {report['capabilities']['best_instruction_set']}")
    print(f"  Vector Width Support: {report['capabilities']['vector_widths']}")
    
    print(f"\nPerformance Statistics:")
    print(f"  Total Operations: {report['performance_stats']['operation_count']:,}")
    print(f"  Total Execution Time: {report['performance_stats']['total_execution_time']:.3f} s")
    print(f"  Average Operation Time: {report['performance_stats']['avg_operation_time']*1000:.3f} ms")
    
    if report['performance_stats']['operation_count'] > 0:
        throughput = report['performance_stats']['operation_count'] / report['performance_stats']['total_execution_time']
        print(f"  Operations per Second: {throughput:.0f}")
    
    print(f"\nConfiguration:")
    config = report['configuration']
    print(f"  Vectorization Threshold: {config['vectorization_threshold']}")
    print(f"  Auto Vectorize: {config['auto_vectorize']}")
    print(f"  Use FMA: {config['use_fma']}")
    print(f"  Max Unroll Factor: {config['max_unroll_factor']}")
    print(f"  Alignment Requirement: {config['alignment_bytes']} bytes")

def main():
    """Main demonstration function"""
    print("NeuralScript SIMD Vectorization System")
    print("High-Performance Computing for Machine Learning")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    
    try:
        # Initialize the SIMD system
        simd = demonstrate_hardware_detection()
        
        # Demonstrate all major components
        demonstrate_vector_operations(simd)
        demonstrate_math_functions(simd)
        demonstrate_matrix_operations(simd)
        demonstrate_ml_operations(simd)
        demonstrate_optimization_system(simd)
        benchmark_comparison(simd)
        generate_comprehensive_report(simd)
        
        print_section("SIMD System Demonstration Complete")
        print("All components successfully tested and demonstrated!")
        print(f"Total SIMD operations executed: {simd._operation_count:,}")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
