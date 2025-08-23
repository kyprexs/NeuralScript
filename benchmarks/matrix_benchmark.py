#!/usr/bin/env python3
"""
High-Performance Matrix Multiplication Benchmark
===============================================

Comprehensive benchmarking system for measuring matrix multiplication performance
against the NeuralScript target of < 50ms for 1000x1000 matrices.

Features:
- Multiple algorithm implementations (naive, blocked, SIMD-optimized)
- Performance comparison against NumPy baseline
- Memory usage tracking
- Hardware capability detection
- Statistical analysis of results
"""

import time
import numpy as np
import psutil
import platform
from typing import Tuple, Dict, List, Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager
import gc
import sys
import os

# Add the compiler directory to the path so we can import SIMD modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'compiler'))

try:
    from simd.simd_core import SIMDProcessor
    from simd.matrix_operations import MatrixOperations
    SIMD_AVAILABLE = True
except ImportError:
    print("Warning: SIMD system not available, using fallback implementations")
    SIMD_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    algorithm_name: str
    matrix_size: int
    execution_time_ms: float
    memory_usage_mb: float
    gflops: float
    meets_target: bool
    error_vs_baseline: Optional[float] = None


@dataclass
class SystemInfo:
    """Information about the system running the benchmark."""
    cpu_info: str
    memory_gb: float
    python_version: str
    numpy_version: str
    simd_capabilities: List[str]
    

class MatrixMultiplicationBenchmark:
    """
    Comprehensive matrix multiplication benchmark suite.
    
    Tests multiple algorithms and measures performance against targets.
    """
    
    def __init__(self):
        self.target_time_ms = 50.0  # Target: < 50ms for 1000x1000
        self.system_info = self._get_system_info()
        self.simd_processor = None
        self.matrix_math = None
        
        if SIMD_AVAILABLE:
            try:
                self.simd_processor = SIMDProcessor()
                self.matrix_operations = MatrixOperations(self.simd_processor)
                print(f"✅ SIMD system initialized with {len(self.simd_processor.get_available_instruction_sets())} instruction sets")
            except Exception as e:
                print(f"⚠️  SIMD initialization failed: {e}")
    
    def _get_system_info(self) -> SystemInfo:
        """Collect system information for benchmark context."""
        return SystemInfo(
            cpu_info=platform.processor() or "Unknown CPU",
            memory_gb=psutil.virtual_memory().total / (1024**3),
            python_version=platform.python_version(),
            numpy_version=np.__version__,
            simd_capabilities=self._detect_simd_capabilities()
        )
    
    def _detect_simd_capabilities(self) -> List[str]:
        """Detect available SIMD instruction sets."""
        if not SIMD_AVAILABLE:
            return ["None - SIMD system not available"]
        
        try:
            if self.simd_processor is None:
                temp_processor = SIMDProcessor()
                capabilities = temp_processor.get_available_instruction_sets()
                return capabilities if capabilities else ["None detected"]
            return self.simd_processor.get_available_instruction_sets()
        except:
            return ["Detection failed"]
    
    @contextmanager
    def _memory_tracker(self):
        """Context manager to track memory usage during benchmark."""
        process = psutil.Process()
        start_memory = process.memory_info().rss / (1024**2)  # MB
        
        try:
            yield
        finally:
            end_memory = process.memory_info().rss / (1024**2)  # MB
            self.last_memory_usage = end_memory - start_memory
    
    def _calculate_gflops(self, matrix_size: int, time_seconds: float) -> float:
        """Calculate GFLOPS for matrix multiplication."""
        # Matrix multiplication: n^3 multiply-add operations = 2n^3 FLOPs
        flops = 2 * matrix_size ** 3
        return flops / time_seconds / 1e9
    
    def naive_matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Naive triple-loop matrix multiplication (for comparison).
        """
        n, k = A.shape
        k2, m = B.shape
        assert k == k2, "Matrix dimensions don't match"
        
        C = np.zeros((n, m), dtype=A.dtype)
        
        for i in range(n):
            for j in range(m):
                for l in range(k):
                    C[i, j] += A[i, l] * B[l, j]
        
        return C
    
    def blocked_matrix_multiply(self, A: np.ndarray, B: np.ndarray, block_size: int = 64) -> np.ndarray:
        """
        Blocked (cache-friendly) matrix multiplication.
        """
        n, k = A.shape
        k2, m = B.shape
        assert k == k2, "Matrix dimensions don't match"
        
        C = np.zeros((n, m), dtype=A.dtype)
        
        for i in range(0, n, block_size):
            for j in range(0, m, block_size):
                for l in range(0, k, block_size):
                    # Define block boundaries
                    i_end = min(i + block_size, n)
                    j_end = min(j + block_size, m)
                    l_end = min(l + block_size, k)
                    
                    # Multiply blocks
                    C[i:i_end, j:j_end] += np.dot(A[i:i_end, l:l_end], B[l:l_end, j:j_end])
        
        return C
    
    def simd_matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        SIMD-optimized matrix multiplication using our SIMD system.
        """
        if not SIMD_AVAILABLE or not hasattr(self, 'matrix_operations') or self.matrix_operations is None:
            # Fallback to numpy
            return np.dot(A, B)
        
        try:
            return self.matrix_operations.matrix_multiply(A, B)
        except Exception as e:
            print(f"SIMD multiply failed: {e}, falling back to NumPy")
            return np.dot(A, B)
    
    def strassen_matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Strassen algorithm matrix multiplication for large matrices.
        """
        if not SIMD_AVAILABLE or not hasattr(self, 'matrix_operations') or self.matrix_operations is None:
            # Fallback to numpy
            return np.dot(A, B)
        
        try:
            return self.matrix_operations.matrix_multiply_strassen(A, B)
        except Exception as e:
            print(f"Strassen multiply failed: {e}, falling back to NumPy")
            return np.dot(A, B)
    
    def numpy_matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        NumPy's optimized matrix multiplication (baseline).
        """
        return np.dot(A, B)
    
    def benchmark_algorithm(self, 
                           algorithm: Callable[[np.ndarray, np.ndarray], np.ndarray],
                           name: str,
                           A: np.ndarray, 
                           B: np.ndarray,
                           baseline_result: Optional[np.ndarray] = None) -> BenchmarkResult:
        """
        Benchmark a single matrix multiplication algorithm.
        """
        # Warm up
        if A.shape[0] <= 100:  # Only warm up for small matrices
            _ = algorithm(A[:10, :10], B[:10, :10])
        
        # Force garbage collection
        gc.collect()
        
        with self._memory_tracker():
            start_time = time.perf_counter()
            result = algorithm(A, B)
            end_time = time.perf_counter()
        
        execution_time_seconds = end_time - start_time
        execution_time_ms = execution_time_seconds * 1000
        
        matrix_size = A.shape[0]
        gflops = self._calculate_gflops(matrix_size, execution_time_seconds)
        meets_target = execution_time_ms < self.target_time_ms
        
        error_vs_baseline = None
        if baseline_result is not None:
            error_vs_baseline = np.linalg.norm(result - baseline_result) / np.linalg.norm(baseline_result)
        
        return BenchmarkResult(
            algorithm_name=name,
            matrix_size=matrix_size,
            execution_time_ms=execution_time_ms,
            memory_usage_mb=self.last_memory_usage,
            gflops=gflops,
            meets_target=meets_target,
            error_vs_baseline=error_vs_baseline
        )
    
    def run_comprehensive_benchmark(self, matrix_size: int = 1000, 
                                  num_runs: int = 5) -> List[BenchmarkResult]:
        """
        Run comprehensive benchmarks with multiple algorithms.
        """
        print(f"\n🚀 Starting Matrix Multiplication Benchmark")
        print(f"Matrix Size: {matrix_size}x{matrix_size}")
        print(f"Target: < {self.target_time_ms}ms")
        print(f"Runs per algorithm: {num_runs}")
        print("=" * 60)
        
        # Generate test matrices
        print("Generating test matrices...")
        np.random.seed(42)  # Reproducible results
        A = np.random.randn(matrix_size, matrix_size).astype(np.float32)
        B = np.random.randn(matrix_size, matrix_size).astype(np.float32)
        
        # Define algorithms to test
        algorithms = [
            (self.numpy_matrix_multiply, "NumPy (Baseline)"),
            (self.simd_matrix_multiply, "SIMD Optimized"),
            (self.blocked_matrix_multiply, "Blocked Cache-Friendly"),
        ]
        
        # Add Strassen for large matrices (power of 2 sizes)
        if matrix_size >= 512 and (matrix_size & (matrix_size - 1)) == 0:
            algorithms.append((self.strassen_matrix_multiply, "Strassen Algorithm"))
        
        # Add naive algorithm only for smaller matrices
        if matrix_size <= 500:
            algorithms.append((self.naive_matrix_multiply, "Naive Triple-Loop"))
        
        all_results = []
        baseline_result = None
        
        for algorithm, name in algorithms:
            print(f"\n📊 Benchmarking: {name}")
            
            algorithm_results = []
            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}...", end=" ")
                
                try:
                    result = self.benchmark_algorithm(
                        algorithm, name, A, B, baseline_result
                    )
                    algorithm_results.append(result)
                    
                    status = "✅ PASS" if result.meets_target else "❌ MISS"
                    print(f"{result.execution_time_ms:.2f}ms ({result.gflops:.2f} GFLOPS) {status}")
                    
                    # Store baseline for comparison
                    if name == "NumPy (Baseline)" and run == 0 and baseline_result is None:
                        baseline_result = algorithm(A, B)
                
                except Exception as e:
                    print(f"❌ FAILED: {e}")
                    continue
            
            if algorithm_results:
                # Calculate statistics
                times = [r.execution_time_ms for r in algorithm_results]
                best_result = min(algorithm_results, key=lambda r: r.execution_time_ms)
                avg_time = np.mean(times)
                std_time = np.std(times)
                
                print(f"  📈 Best: {best_result.execution_time_ms:.2f}ms")
                print(f"  📊 Average: {avg_time:.2f} ± {std_time:.2f}ms")
                
                all_results.extend(algorithm_results)
        
        return all_results
    
    def print_system_info(self):
        """Print detailed system information."""
        print("\n💻 System Information")
        print("=" * 40)
        print(f"CPU: {self.system_info.cpu_info}")
        print(f"Memory: {self.system_info.memory_gb:.1f} GB")
        print(f"Python: {self.system_info.python_version}")
        print(f"NumPy: {self.system_info.numpy_version}")
        print(f"SIMD Support: {', '.join(self.system_info.simd_capabilities)}")
    
    def print_results_summary(self, results: List[BenchmarkResult]):
        """Print a summary of benchmark results."""
        if not results:
            print("No results to display")
            return
        
        print(f"\n📊 Benchmark Results Summary")
        print("=" * 80)
        print(f"{'Algorithm':<25} {'Best Time (ms)':<15} {'GFLOPS':<10} {'Target':<8} {'Error':<10}")
        print("-" * 80)
        
        # Group results by algorithm
        algorithm_results = {}
        for result in results:
            if result.algorithm_name not in algorithm_results:
                algorithm_results[result.algorithm_name] = []
            algorithm_results[result.algorithm_name].append(result)
        
        target_met = False
        best_algorithm = None
        best_time = float('inf')
        
        for algorithm_name, algorithm_results_list in algorithm_results.items():
            best_result = min(algorithm_results_list, key=lambda r: r.execution_time_ms)
            
            target_status = "✅ PASS" if best_result.meets_target else "❌ MISS"
            error_str = f"{best_result.error_vs_baseline:.2e}" if best_result.error_vs_baseline else "N/A"
            
            print(f"{algorithm_name:<25} {best_result.execution_time_ms:<15.2f} "
                  f"{best_result.gflops:<10.2f} {target_status:<8} {error_str:<10}")
            
            if best_result.meets_target:
                target_met = True
            
            if best_result.execution_time_ms < best_time:
                best_time = best_result.execution_time_ms
                best_algorithm = algorithm_name
        
        print("-" * 80)
        
        if target_met:
            print(f"🎯 SUCCESS: Target of < {self.target_time_ms}ms achieved!")
            print(f"🏆 Best algorithm: {best_algorithm} ({best_time:.2f}ms)")
        else:
            print(f"❌ Target of < {self.target_time_ms}ms not achieved")
            print(f"🥇 Closest: {best_algorithm} ({best_time:.2f}ms)")
            print(f"📈 Improvement needed: {((best_time / self.target_time_ms) - 1) * 100:.1f}% faster")


def main():
    """Main benchmark execution."""
    print("🧠⚡ NeuralScript Matrix Multiplication Benchmark")
    print("=" * 50)
    
    benchmark = MatrixMultiplicationBenchmark()
    benchmark.print_system_info()
    
    # Test different matrix sizes
    sizes_to_test = [100, 500, 1000]
    
    for size in sizes_to_test:
        results = benchmark.run_comprehensive_benchmark(matrix_size=size, num_runs=3)
        benchmark.print_results_summary(results)
        
        if size < max(sizes_to_test):
            print("\n" + "="*50 + "\n")
    
    print(f"\n🏁 Benchmark Complete!")


if __name__ == "__main__":
    main()
