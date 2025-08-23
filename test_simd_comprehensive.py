#!/usr/bin/env python3
"""
Comprehensive SIMD System Test Suite

Full validation of the NeuralScript SIMD vectorization system.
Tests all components, edge cases, and performance characteristics.
"""

import numpy as np
import pytest
import time
import sys
import os
import unittest
from typing import Dict, List, Any

# Add the compiler directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'compiler'))

from compiler.simd import (
    SIMDProcessor, SIMDConfiguration, VectorOperations, VectorMath,
    MatrixOperations, ActivationFunctions, AutoVectorizer, OptimizationHint,
    DataType, OptimizationLevel, PerformanceProfiler
)


class TestSIMDCore(unittest.TestCase):
    """Test the core SIMD processor functionality"""
    
    def setUp(self):
        self.simd = SIMDProcessor()
    
    def test_hardware_detection(self):
        """Test hardware capability detection"""
        self.assertIsNotNone(self.simd.capabilities)
        self.assertGreater(len(self.simd.capabilities.instruction_sets), 0)
        self.assertIn('SCALAR', [iset.name for iset in self.simd.capabilities.instruction_sets])
        
        # Test cache sizes are reasonable
        self.assertGreater(self.simd.capabilities.cache_sizes['L1'], 0)
        self.assertGreater(self.simd.capabilities.cache_sizes['L2'], 0)
        self.assertGreater(self.simd.capabilities.cache_sizes['L3'], 0)
    
    def test_vector_width_calculation(self):
        """Test vector width calculation for different data types"""
        for data_type in [DataType.FLOAT32, DataType.FLOAT64, DataType.INT32]:
            width = self.simd.get_vector_width(data_type)
            self.assertGreater(width, 0)
            self.assertIsInstance(width, int)
    
    def test_vectorization_decision(self):
        """Test vectorization decision logic"""
        # Small arrays should not be vectorized
        self.assertFalse(self.simd.should_vectorize(2, DataType.FLOAT32))
        
        # Large arrays should be vectorized
        self.assertTrue(self.simd.should_vectorize(1000, DataType.FLOAT32))
    
    def test_performance_tracking(self):
        """Test performance statistics tracking"""
        initial_count = self.simd._operation_count
        
        # Simulate some operations
        self.simd._operation_count += 5
        self.simd._total_execution_time += 0.1
        
        stats = self.simd.get_performance_statistics()
        self.assertEqual(stats['total_operations'], initial_count + 5)
        self.assertGreaterEqual(stats['total_execution_time'], 0.1)
    
    def test_configuration_options(self):
        """Test SIMD configuration options"""
        config = SIMDConfiguration(
            auto_vectorize=False,
            vectorization_threshold=16,
            alignment_bytes=64
        )
        simd_custom = SIMDProcessor(config)
        
        self.assertFalse(simd_custom.config.auto_vectorize)
        self.assertEqual(simd_custom.config.vectorization_threshold, 16)
        self.assertEqual(simd_custom.config.alignment_bytes, 64)


class TestVectorOperations(unittest.TestCase):
    """Test vector operations functionality"""
    
    def setUp(self):
        self.simd = SIMDProcessor()
        self.vec_ops = VectorOperations(self.simd)
        self.vec_math = VectorMath(self.simd)
    
    def test_basic_arithmetic(self):
        """Test basic arithmetic operations"""
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        
        # Test addition
        result = self.vec_ops.add(a, b)
        expected = np.array([6.0, 8.0, 10.0, 12.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test multiplication
        result = self.vec_ops.multiply(a, b)
        expected = np.array([5.0, 12.0, 21.0, 32.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test subtraction
        result = self.vec_ops.subtract(b, a)
        expected = np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test division
        result = self.vec_ops.divide(b, a)
        expected = np.array([5.0, 3.0, 7.0/3.0, 2.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_fused_multiply_add(self):
        """Test fused multiply-add operation"""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        c = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        
        result = self.vec_ops.fused_multiply_add(a, b, c)
        expected = a * b + c
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_dot_product(self):
        """Test dot product calculation"""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        
        result = self.vec_ops.dot_product(a, b)
        expected = np.dot(a, b)
        self.assertAlmostEqual(result, expected, places=5)
    
    def test_cross_product(self):
        """Test 3D cross product"""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        result = self.vec_ops.cross_product(a, b)
        expected = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_vector_normalization(self):
        """Test vector normalization"""
        a = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        
        # Test magnitude calculation
        magnitude = self.vec_ops.magnitude(a)
        self.assertAlmostEqual(magnitude, 5.0, places=5)
        
        # Test normalization
        normalized = self.vec_ops.normalize(a)
        expected = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(normalized, expected)
        
        # Verify normalized vector has unit length
        self.assertAlmostEqual(self.vec_ops.magnitude(normalized), 1.0, places=5)
    
    def test_mathematical_functions(self):
        """Test advanced mathematical functions"""
        x = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
        
        # Test square root
        result = self.vec_math.sqrt(x)
        expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test power function
        result = self.vec_math.power(x, 0.5)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test exponential and logarithm
        x_small = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        exp_result = self.vec_math.exp(x_small)
        log_result = self.vec_math.log(exp_result)
        np.testing.assert_array_almost_equal(log_result, x_small, decimal=5)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        # Empty arrays
        empty = np.array([], dtype=np.float32)
        result = self.vec_ops.add(empty, empty)
        self.assertEqual(len(result), 0)
        
        # Single element arrays
        single_a = np.array([5.0], dtype=np.float32)
        single_b = np.array([3.0], dtype=np.float32)
        result = self.vec_ops.add(single_a, single_b)
        np.testing.assert_array_almost_equal(result, np.array([8.0]))
        
        # Test with different array sizes (should handle broadcasting)
        try:
            a = np.array([1.0, 2.0], dtype=np.float32)
            b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            # This should raise an error or handle gracefully
            result = self.vec_ops.add(a, b)
        except (ValueError, Exception):
            pass  # Expected for incompatible shapes


class TestMatrixOperations(unittest.TestCase):
    """Test matrix operations functionality"""
    
    def setUp(self):
        self.simd = SIMDProcessor()
        self.mat_ops = MatrixOperations(self.simd)
    
    def test_matrix_multiplication(self):
        """Test matrix multiplication"""
        A = np.array([[1, 2], [3, 4]], dtype=np.float32)
        B = np.array([[5, 6], [7, 8]], dtype=np.float32)
        
        result = self.mat_ops.matrix_multiply(A, B)
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_matrix_vector_multiply(self):
        """Test matrix-vector multiplication"""
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        v = np.array([1, 2, 3], dtype=np.float32)
        
        result = self.mat_ops.matrix_vector_multiply(A, v)
        expected = np.array([14, 32], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_matrix_transpose(self):
        """Test matrix transpose"""
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        
        result = self.mat_ops.transpose(A)
        expected = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_matrix_decomposition(self):
        """Test matrix decomposition operations"""
        from compiler.simd.matrix_operations import MatrixDecomposition
        
        decomp = MatrixDecomposition(self.simd)
        
        # Test QR decomposition
        A = np.array([[1, 1], [1, 2], [1, 3]], dtype=np.float32)
        Q, R = decomp.qr_decomposition(A)
        
        # Verify Q*R = A
        reconstructed = np.dot(Q, R)
        np.testing.assert_array_almost_equal(reconstructed, A, decimal=4)
        
        # Verify Q is orthogonal (Q^T * Q = I)
        QTQ = np.dot(Q.T, Q)
        identity = np.eye(Q.shape[1], dtype=np.float32)
        np.testing.assert_array_almost_equal(QTQ, identity, decimal=4)


class TestActivationFunctions(unittest.TestCase):
    """Test ML activation functions"""
    
    def setUp(self):
        self.simd = SIMDProcessor()
        self.activations = ActivationFunctions(self.simd)
    
    def test_relu(self):
        """Test ReLU activation function"""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        
        result = self.activations.relu(x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_leaky_relu(self):
        """Test Leaky ReLU activation function"""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        alpha = 0.1
        
        result = self.activations.leaky_relu(x, alpha=alpha)
        expected = np.array([-0.2, -0.1, 0.0, 1.0, 2.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_sigmoid(self):
        """Test sigmoid activation function"""
        x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        
        result = self.activations.sigmoid(x)
        
        # Check sigmoid properties
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))
        self.assertAlmostEqual(result[0], 0.5, places=4)  # sigmoid(0) = 0.5
    
    def test_softmax(self):
        """Test softmax activation function"""
        x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        
        result = self.activations.softmax(x, axis=1)
        
        # Check softmax properties
        self.assertTrue(np.all(result >= 0.0))
        row_sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.array([1.0, 1.0]))


class TestOptimizationSystem(unittest.TestCase):
    """Test the optimization and vectorization system"""
    
    def setUp(self):
        self.simd = SIMDProcessor()
        self.optimizer = AutoVectorizer(self.simd, OptimizationLevel.AGGRESSIVE)
        self.profiler = PerformanceProfiler(self.simd)
    
    def test_optimization_analysis(self):
        """Test optimization hint analysis"""
        hint = OptimizationHint(
            operation_name='vector_add',
            data_size=1000,
            data_type=DataType.FLOAT32,
            access_pattern='sequential'
        )
        
        optimization = self.optimizer.optimize_operation(hint)
        
        self.assertIn('vectorize', optimization)
        self.assertIn('vector_width', optimization)
        self.assertIn('estimated_speedup', optimization)
        self.assertGreater(optimization['estimated_speedup'], 1.0)
    
    def test_performance_profiling(self):
        """Test performance profiling functionality"""
        vec_ops = VectorOperations(self.simd)
        
        a = np.random.random(100).astype(np.float32)
        b = np.random.random(100).astype(np.float32)
        
        metrics = self.profiler.profile_operation(vec_ops.add, a, b)
        
        self.assertGreater(metrics.execution_time, 0)
        self.assertGreater(metrics.throughput, 0)
        self.assertGreaterEqual(metrics.efficiency, 0)
        self.assertGreater(metrics.vectorization_factor, 0)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def setUp(self):
        self.simd = SIMDProcessor()
        self.vec_ops = VectorOperations(self.simd)
        self.mat_ops = MatrixOperations(self.simd)
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        # Test cross product with wrong dimensions
        with self.assertRaises(ValueError):
            a = np.array([1, 2], dtype=np.float32)  # 2D instead of 3D
            b = np.array([3, 4], dtype=np.float32)
            self.vec_ops.cross_product(a, b)
        
        # Test matrix multiplication with incompatible shapes
        with self.assertRaises(ValueError):
            A = np.array([[1, 2]], dtype=np.float32)  # 1x2
            B = np.array([[1], [2], [3]], dtype=np.float32)  # 3x1
            self.mat_ops.matrix_multiply(A, B)
    
    def test_data_type_consistency(self):
        """Test data type handling"""
        # Test with different input types
        a_int = np.array([1, 2, 3])  # int64
        b_float = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        # Should handle type conversion gracefully
        result = self.vec_ops.add(a_int, b_float)
        self.assertEqual(result.dtype, np.float32)


class BenchmarkTests:
    """Performance benchmark tests"""
    
    def __init__(self):
        self.simd = SIMDProcessor()
        self.vec_ops = VectorOperations(self.simd)
        self.mat_ops = MatrixOperations(self.simd)
    
    def benchmark_vector_operations(self):
        """Benchmark vector operations performance"""
        sizes = [100, 1000, 10000, 100000]
        operations = [
            ('add', lambda a, b: self.vec_ops.add(a, b)),
            ('multiply', lambda a, b: self.vec_ops.multiply(a, b)),
            ('dot_product', lambda a, b: self.vec_ops.dot_product(a, b)),
        ]
        
        print("\nVector Operations Benchmark:")
        print("Size\t\tOperation\t\tTime (ms)\t\tThroughput (MOps/s)")
        print("-" * 70)
        
        for size in sizes:
            a = np.random.random(size).astype(np.float32)
            b = np.random.random(size).astype(np.float32)
            
            for op_name, op_func in operations:
                # Warm up
                for _ in range(3):
                    op_func(a, b)
                
                # Benchmark
                start_time = time.perf_counter()
                for _ in range(10):
                    result = op_func(a, b)
                end_time = time.perf_counter()
                
                avg_time = (end_time - start_time) / 10
                throughput = size / avg_time / 1e6
                
                print(f"{size:8d}\t\t{op_name:12s}\t\t{avg_time*1000:8.3f}\t\t{throughput:8.1f}")
    
    def benchmark_matrix_operations(self):
        """Benchmark matrix operations performance"""
        sizes = [64, 128, 256, 512]
        
        print("\nMatrix Operations Benchmark:")
        print("Size\t\tOperation\t\tTime (ms)\t\tGFLOPS")
        print("-" * 60)
        
        for size in sizes:
            A = np.random.random((size, size)).astype(np.float32)
            B = np.random.random((size, size)).astype(np.float32)
            
            # Matrix multiplication benchmark
            start_time = time.perf_counter()
            C = self.mat_ops.matrix_multiply(A, B)
            end_time = time.perf_counter()
            
            elapsed_time = end_time - start_time
            flops = 2 * size * size * size  # 2*N^3 operations
            gflops = flops / elapsed_time / 1e9
            
            print(f"{size:4d}x{size:<4d}\t\tmatmul\t\t\t{elapsed_time*1000:8.3f}\t\t{gflops:8.2f}")


def run_all_tests():
    """Run the complete test suite"""
    print("=" * 80)
    print(" NeuralScript SIMD Comprehensive Test Suite")
    print("=" * 80)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSIMDCore,
        TestVectorOperations, 
        TestMatrixOperations,
        TestActivationFunctions,
        TestOptimizationSystem,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Run benchmarks
    if result.wasSuccessful():
        print("\n" + "=" * 80)
        print(" Performance Benchmarks")
        print("=" * 80)
        
        benchmark = BenchmarkTests()
        benchmark.benchmark_vector_operations()
        benchmark.benchmark_matrix_operations()
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n" + "✅" * 20)
        print("All tests passed! SIMD system is fully validated.")
        print("✅" * 20)
        sys.exit(0)
    else:
        print("\n" + "❌" * 20)
        print("Some tests failed. Please review the results above.")
        print("❌" * 20)
        sys.exit(1)
