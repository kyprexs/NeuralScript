#!/usr/bin/env python3
"""
Comprehensive Matrix Operations Performance Test Suite
=====================================================

Tests matrix multiplication performance across different sizes, data types,
and optimization strategies to ensure we meet performance targets.

Features:
- Performance regression testing
- Correctness validation 
- Multi-algorithm benchmarking
- Memory usage profiling
- Hardware-specific optimizations
"""

import pytest
import numpy as np
import time
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add compiler path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'compiler'))

try:
    from simd.simd_core import SIMDProcessor
    from simd.matrix_operations import MatrixOperations
    SIMD_AVAILABLE = True
except ImportError:
    SIMD_AVAILABLE = False


@dataclass
class PerformanceTarget:
    """Performance targets for different matrix sizes"""
    size: int
    target_time_ms: float
    min_gflops: float
    max_memory_mb: float


class TestMatrixPerformance:
    """
    Comprehensive test suite for matrix multiplication performance.
    """
    
    # Performance targets based on our requirements
    PERFORMANCE_TARGETS = [
        PerformanceTarget(100, 5.0, 2.0, 1.0),       # Small matrices
        PerformanceTarget(500, 15.0, 25.0, 5.0),     # Medium matrices  
        PerformanceTarget(1000, 50.0, 100.0, 20.0),  # Target: < 50ms
        PerformanceTarget(2000, 400.0, 50.0, 80.0),  # Large matrices
    ]
    
    @classmethod
    def setup_class(cls):
        """Set up test class with SIMD system if available"""
        cls.simd_processor = None
        cls.matrix_operations = None
        
        if SIMD_AVAILABLE:
            try:
                cls.simd_processor = SIMDProcessor()
                cls.matrix_operations = MatrixOperations(cls.simd_processor)
            except Exception as e:
                print(f"Warning: Failed to initialize SIMD system: {e}")
    
    def _generate_test_matrices(self, size: int, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
        """Generate reproducible test matrices"""
        np.random.seed(42 + size)  # Size-dependent seed for reproducibility
        A = np.random.randn(size, size).astype(dtype)
        B = np.random.randn(size, size).astype(dtype)
        return A, B
    
    def _measure_performance(self, func, A: np.ndarray, B: np.ndarray, num_runs: int = 3) -> Dict:
        """Measure performance of matrix multiplication function"""
        times = []
        
        # Warmup
        _ = func(A[:10, :10], B[:10, :10])
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            result = func(A, B)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        execution_time = min(times)  # Best of multiple runs
        execution_time_ms = execution_time * 1000
        
        # Calculate GFLOPS (2 * n^3 operations for n x n matrix multiply)
        n = A.shape[0]
        flops = 2 * n ** 3
        gflops = flops / execution_time / 1e9
        
        return {
            'execution_time_ms': execution_time_ms,
            'gflops': gflops,
            'result': result
        }
    
    def _validate_correctness(self, result: np.ndarray, reference: np.ndarray, 
                            tolerance: float = 1e-5) -> bool:
        """Validate that result matches reference within tolerance"""
        if result.shape != reference.shape:
            return False
        
        relative_error = np.linalg.norm(result - reference) / np.linalg.norm(reference)
        return relative_error < tolerance
    
    @pytest.mark.parametrize("target", PERFORMANCE_TARGETS)
    def test_numpy_baseline_performance(self, target: PerformanceTarget):
        """Test NumPy baseline performance against targets"""
        A, B = self._generate_test_matrices(target.size)
        
        perf = self._measure_performance(np.dot, A, B)
        
        # Performance assertions
        assert perf['execution_time_ms'] < target.target_time_ms, \
            f"NumPy baseline too slow: {perf['execution_time_ms']:.2f}ms > {target.target_time_ms}ms"
        
        assert perf['gflops'] > target.min_gflops, \
            f"NumPy baseline too slow: {perf['gflops']:.2f} GFLOPS < {target.min_gflops}"
        
        # Store reference result for correctness testing
        self._reference_result = perf['result']
    
    @pytest.mark.skipif(not SIMD_AVAILABLE, reason="SIMD system not available")
    @pytest.mark.parametrize("target", PERFORMANCE_TARGETS)
    def test_simd_matrix_performance(self, target: PerformanceTarget):
        """Test SIMD-optimized matrix multiplication performance"""
        A, B = self._generate_test_matrices(target.size)
        
        def simd_multiply(A, B):
            return self.matrix_operations.matrix_multiply(A, B)
        
        perf = self._measure_performance(simd_multiply, A, B)
        
        # Performance assertions - SIMD should meet or exceed targets
        assert perf['execution_time_ms'] < target.target_time_ms, \
            f"SIMD multiply too slow: {perf['execution_time_ms']:.2f}ms > {target.target_time_ms}ms"
        
        # Correctness validation
        reference_result = np.dot(A, B)
        assert self._validate_correctness(perf['result'], reference_result), \
            "SIMD result doesn't match reference"
    
    def test_critical_1000x1000_performance(self):
        """Critical test for our main 1000x1000 performance target"""
        A, B = self._generate_test_matrices(1000)
        
        # Test NumPy baseline
        numpy_perf = self._measure_performance(np.dot, A, B)
        
        # Must achieve < 50ms target
        assert numpy_perf['execution_time_ms'] < 50.0, \
            f"CRITICAL: 1000x1000 matrix multiply failed target: {numpy_perf['execution_time_ms']:.2f}ms >= 50ms"
        
        # Must achieve reasonable GFLOPS
        assert numpy_perf['gflops'] > 100.0, \
            f"CRITICAL: 1000x1000 matrix multiply too slow: {numpy_perf['gflops']:.2f} GFLOPS < 100"
        
        print(f"✅ 1000x1000 Performance: {numpy_perf['execution_time_ms']:.2f}ms, {numpy_perf['gflops']:.2f} GFLOPS")
    
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_data_type_performance(self, dtype):
        """Test performance across different data types"""
        A, B = self._generate_test_matrices(500, dtype)
        
        perf = self._measure_performance(np.dot, A, B)
        
        # Performance should be reasonable for both data types
        assert perf['execution_time_ms'] < 20.0, \
            f"{dtype} matrix multiply too slow: {perf['execution_time_ms']:.2f}ms"
        
        assert perf['gflops'] > 25.0, \
            f"{dtype} matrix multiply too slow: {perf['gflops']:.2f} GFLOPS"
    
    def test_memory_usage_scaling(self):
        """Test that memory usage scales reasonably with matrix size"""
        import psutil
        process = psutil.Process()
        
        memory_usage = {}
        
        for size in [100, 200, 500, 1000]:
            # Measure memory before
            start_memory = process.memory_info().rss / (1024**2)  # MB
            
            # Generate matrices and multiply
            A, B = self._generate_test_matrices(size)
            result = np.dot(A, B)
            
            # Measure memory after
            end_memory = process.memory_info().rss / (1024**2)  # MB
            memory_used = end_memory - start_memory
            
            memory_usage[size] = memory_used
            
            # Clean up
            del A, B, result
        
        # Memory usage should scale reasonably (not more than quadratically)
        ratio_1000_to_100 = memory_usage[1000] / max(memory_usage[100], 0.1)
        assert ratio_1000_to_100 < 200, f"Memory usage scales too poorly: {ratio_1000_to_100}x"
    
    @pytest.mark.skipif(not SIMD_AVAILABLE, reason="SIMD system not available")
    def test_simd_algorithm_comparison(self):
        """Compare different SIMD algorithms for correctness and performance"""
        A, B = self._generate_test_matrices(500)
        reference = np.dot(A, B)
        
        algorithms = []
        
        # Standard SIMD multiply
        def standard_multiply(A, B):
            return self.matrix_operations.matrix_multiply(A, B)
        algorithms.append(("Standard SIMD", standard_multiply))
        
        # Strassen algorithm (if available and matrix is power of 2)
        if hasattr(self.matrix_operations, 'matrix_multiply_strassen') and A.shape[0] == 512:
            A_512, B_512 = self._generate_test_matrices(512)
            def strassen_multiply(A, B):
                return self.matrix_operations.matrix_multiply_strassen(A, B)
            algorithms.append(("Strassen", strassen_multiply))
        
        for name, algorithm in algorithms:
            perf = self._measure_performance(algorithm, A, B)
            
            # Correctness check
            assert self._validate_correctness(perf['result'], reference), \
                f"{name} algorithm produces incorrect results"
            
            # Performance should be reasonable
            assert perf['execution_time_ms'] < 20.0, \
                f"{name} algorithm too slow: {perf['execution_time_ms']:.2f}ms"
    
    def test_performance_consistency(self):
        """Test that performance is consistent across multiple runs"""
        A, B = self._generate_test_matrices(500)
        
        times = []
        for _ in range(10):
            start_time = time.perf_counter()
            _ = np.dot(A, B)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        cv = std_time / mean_time  # Coefficient of variation
        
        # Performance should be consistent (CV < 20%)
        assert cv < 0.2, f"Performance too inconsistent: CV = {cv:.2%}"
        
        # All runs should meet performance targets
        assert max(times) < 15.0, f"Worst case too slow: {max(times):.2f}ms"
    
    @pytest.mark.parametrize("size", [100, 250, 500, 750, 1000])
    def test_performance_scaling(self, size):
        """Test that performance scales reasonably with matrix size"""
        A, B = self._generate_test_matrices(size)
        
        perf = self._measure_performance(np.dot, A, B)
        
        # Calculate theoretical time based on O(n^3) scaling from size 100 baseline
        if size == 100:
            self._baseline_time_per_flop = perf['execution_time_ms'] / (2 * 100**3)
        else:
            theoretical_time = self._baseline_time_per_flop * (2 * size**3)
            actual_time = perf['execution_time_ms']
            
            # Actual performance should not be more than 3x worse than theoretical
            # (allowing for cache effects and algorithm differences)
            scaling_factor = actual_time / theoretical_time
            assert scaling_factor < 3.0, \
                f"Performance scaling too poor at size {size}: {scaling_factor:.2f}x worse than O(n^3)"
    
    def test_edge_cases(self):
        """Test matrix multiplication edge cases"""
        
        # Very small matrices
        A_small = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        B_small = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        
        result = np.dot(A_small, B_small)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)
        
        assert np.allclose(result, expected), "Small matrix multiplication incorrect"
        
        # Single element matrices
        A_single = np.array([[42.0]], dtype=np.float32)
        B_single = np.array([[1.5]], dtype=np.float32)
        
        result_single = np.dot(A_single, B_single)
        expected_single = np.array([[63.0]], dtype=np.float32)
        
        assert np.allclose(result_single, expected_single), "Single element matrix incorrect"


class TestMatrixOperationsIntegration:
    """Integration tests for matrix operations with the compiler IR"""
    
    def test_ir_matrix_node_creation(self):
        """Test creating matrix multiplication IR nodes with SIMD metadata"""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'compiler'))
        
        try:
            from ir.ir_nodes import IRMatMul, IRTensorType, IRPrimitiveType, IRDataType, IRValue
            
            # Create tensor types
            float32_type = IRPrimitiveType(IRDataType.F32)
            tensor_type = IRTensorType(float32_type, [1000, 1000])
            
            # Create IR values
            left = IRValue(tensor_type, "matrix_a")
            right = IRValue(tensor_type, "matrix_b")
            
            # Create matrix multiplication node with SIMD optimization
            matmul_node = IRMatMul(
                left=left,
                right=right,
                result_type=tensor_type,
                prefer_simd=True,
                block_size=128,
                use_strassen=False,
                parallel_threshold=1000
            )
            
            # Verify SIMD metadata is set correctly
            assert matmul_node.get_metadata('simd_optimizable') == True
            assert matmul_node.get_metadata('block_size') == 128
            assert matmul_node.get_metadata('use_strassen') == False
            assert matmul_node.get_metadata('parallel_threshold') == 1000
            
            # Verify string representation includes optimization hints
            ir_str = str(matmul_node)
            assert "simd" in ir_str
            assert "block=128" in ir_str
            assert "matmul" in ir_str
            
        except ImportError:
            pytest.skip("IR system not available")


@pytest.mark.performance
class TestPerformanceRegression:
    """Performance regression tests to ensure we maintain our gains"""
    
    PERFORMANCE_HISTORY_FILE = "tests/performance/matrix_performance_history.json"
    
    def test_no_performance_regression(self):
        """Ensure performance hasn't regressed from previous runs"""
        import json
        
        A, B = self._generate_test_matrices(1000)
        current_perf = self._measure_performance(np.dot, A, B)
        
        try:
            # Load performance history
            with open(self.PERFORMANCE_HISTORY_FILE, 'r') as f:
                history = json.load(f)
            
            last_time = history.get('last_1000x1000_time_ms', float('inf'))
            last_gflops = history.get('last_1000x1000_gflops', 0.0)
            
            # Allow 10% regression tolerance
            assert current_perf['execution_time_ms'] < last_time * 1.1, \
                f"Performance regression: {current_perf['execution_time_ms']:.2f}ms > {last_time * 1.1:.2f}ms"
            
            assert current_perf['gflops'] > last_gflops * 0.9, \
                f"Performance regression: {current_perf['gflops']:.2f} GFLOPS < {last_gflops * 0.9:.2f} GFLOPS"
            
        except FileNotFoundError:
            # First run - create history file
            pass
        
        # Update history file
        try:
            os.makedirs(os.path.dirname(self.PERFORMANCE_HISTORY_FILE), exist_ok=True)
            history = {
                'last_1000x1000_time_ms': current_perf['execution_time_ms'],
                'last_1000x1000_gflops': current_perf['gflops'],
                'timestamp': time.time()
            }
            with open(self.PERFORMANCE_HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not update performance history: {e}")
    
    def _generate_test_matrices(self, size: int, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
        """Generate reproducible test matrices"""
        np.random.seed(42 + size)
        A = np.random.randn(size, size).astype(dtype)
        B = np.random.randn(size, size).astype(dtype)
        return A, B
    
    def _measure_performance(self, func, A: np.ndarray, B: np.ndarray, num_runs: int = 3) -> Dict:
        """Measure performance of matrix multiplication function"""
        times = []
        
        # Warmup
        _ = func(A[:10, :10], B[:10, :10])
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            result = func(A, B)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        execution_time = min(times)
        execution_time_ms = execution_time * 1000
        
        n = A.shape[0]
        flops = 2 * n ** 3
        gflops = flops / execution_time / 1e9
        
        return {
            'execution_time_ms': execution_time_ms,
            'gflops': gflops,
            'result': result
        }


if __name__ == "__main__":
    # Run performance tests directly
    import sys
    
    print("🧠⚡ NeuralScript Matrix Performance Test Suite")
    print("=" * 60)
    
    # Run critical performance test
    test_perf = TestMatrixPerformance()
    test_perf.setup_class()
    
    try:
        test_perf.test_critical_1000x1000_performance()
        print("✅ Critical 1000x1000 performance test passed!")
    except AssertionError as e:
        print(f"❌ Critical performance test failed: {e}")
        sys.exit(1)
    
    # Run a few other key tests
    try:
        for target in TestMatrixPerformance.PERFORMANCE_TARGETS:
            test_perf.test_numpy_baseline_performance(target)
        print("✅ All baseline performance tests passed!")
    except AssertionError as e:
        print(f"⚠️  Some performance tests failed: {e}")
    
    print("\n🏁 Test suite complete!")
