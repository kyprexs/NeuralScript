"""
SIMD Matrix Operations for Linear Algebra

High-performance matrix operations optimized for machine learning
and scientific computing workloads using SIMD vectorization.
"""

import numpy as np
import threading
import time
from typing import Union, Optional, Tuple, List
from enum import Enum

from .simd_core import SIMDProcessor, DataType, SIMDConfiguration


ArrayLike = Union[np.ndarray, List[List[float]], List[List[int]]]


class MatrixLayout(Enum):
    """Matrix memory layout options"""
    ROW_MAJOR = "row_major"
    COLUMN_MAJOR = "column_major"
    AUTO = "auto"


class MatrixOperations:
    """
    High-performance SIMD matrix operations.
    
    Provides vectorized implementations of linear algebra operations
    with automatic SIMD optimization and cache-efficient algorithms.
    """
    
    def __init__(self, simd_processor: Optional[SIMDProcessor] = None):
        self.simd = simd_processor or SIMDProcessor()
        self._operation_cache = {}
        self._lock = threading.RLock()
    
    def matrix_multiply(self, a: ArrayLike, b: ArrayLike, 
                       out: Optional[np.ndarray] = None,
                       layout: MatrixLayout = MatrixLayout.AUTO) -> np.ndarray:
        """
        Optimized matrix multiplication using SIMD.
        
        Args:
            a: Left matrix (M x K)
            b: Right matrix (K x N)
            out: Optional output matrix (M x N)
            layout: Memory layout preference
        
        Returns:
            Result matrix (M x N)
        """
        a_matrix = np.asarray(a, dtype=np.float32)
        b_matrix = np.asarray(b, dtype=np.float32)
        
        if len(a_matrix.shape) != 2 or len(b_matrix.shape) != 2:
            raise ValueError("Matrix multiplication requires 2D arrays")
        
        if a_matrix.shape[1] != b_matrix.shape[0]:
            raise ValueError(f"Incompatible shapes: {a_matrix.shape} @ {b_matrix.shape}")
        
        if out is None:
            out = np.empty((a_matrix.shape[0], b_matrix.shape[1]), dtype=np.float32)
        
        start_time = time.perf_counter()
        
        # Use numpy's optimized BLAS implementation
        np.dot(a_matrix, b_matrix, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def matrix_vector_multiply(self, matrix: ArrayLike, vector: ArrayLike,
                              out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Matrix-vector multiplication: Mv
        
        Args:
            matrix: Input matrix (M x N)
            vector: Input vector (N,)
            out: Optional output vector (M,)
        
        Returns:
            Result vector (M,)
        """
        matrix_array = np.asarray(matrix, dtype=np.float32)
        vector_array = np.asarray(vector, dtype=np.float32)
        
        if len(matrix_array.shape) != 2 or len(vector_array.shape) != 1:
            raise ValueError("Matrix-vector multiply requires 2D matrix and 1D vector")
        
        if matrix_array.shape[1] != vector_array.shape[0]:
            raise ValueError(f"Incompatible shapes: {matrix_array.shape} @ {vector_array.shape}")
        
        if out is None:
            out = np.empty(matrix_array.shape[0], dtype=np.float32)
        
        start_time = time.perf_counter()
        np.dot(matrix_array, vector_array, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def transpose(self, a: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Optimized matrix transpose"""
        a_matrix = np.asarray(a, dtype=np.float32)
        
        if len(a_matrix.shape) != 2:
            raise ValueError("Transpose requires 2D array")
        
        if out is None:
            out = np.empty(a_matrix.shape[::-1], dtype=np.float32)
        
        start_time = time.perf_counter()
        result = np.transpose(a_matrix)
        out[:] = result
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def matrix_add(self, a: ArrayLike, b: ArrayLike, 
                   out: Optional[np.ndarray] = None) -> np.ndarray:
        """Element-wise matrix addition"""
        a_matrix = np.asarray(a, dtype=np.float32)
        b_matrix = np.asarray(b, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_matrix)
        
        start_time = time.perf_counter()
        np.add(a_matrix, b_matrix, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def matrix_subtract(self, a: ArrayLike, b: ArrayLike, 
                       out: Optional[np.ndarray] = None) -> np.ndarray:
        """Element-wise matrix subtraction"""
        a_matrix = np.asarray(a, dtype=np.float32)
        b_matrix = np.asarray(b, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_matrix)
        
        start_time = time.perf_counter()
        np.subtract(a_matrix, b_matrix, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def matrix_scale(self, a: ArrayLike, scalar: float, 
                    out: Optional[np.ndarray] = None) -> np.ndarray:
        """Matrix scalar multiplication"""
        a_matrix = np.asarray(a, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_matrix)
        
        start_time = time.perf_counter()
        np.multiply(a_matrix, scalar, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out


class MatrixDecomposition:
    """Advanced matrix decomposition operations"""
    
    def __init__(self, simd_processor: Optional[SIMDProcessor] = None):
        self.simd = simd_processor or SIMDProcessor()
        self._lock = threading.RLock()
    
    def lu_decomposition(self, a: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """LU decomposition with partial pivoting"""
        a_matrix = np.asarray(a, dtype=np.float32)
        
        if len(a_matrix.shape) != 2 or a_matrix.shape[0] != a_matrix.shape[1]:
            raise ValueError("LU decomposition requires square matrix")
        
        start_time = time.perf_counter()
        
        # Use scipy's optimized LU decomposition
        try:
            from scipy.linalg import lu
            P, L, U = lu(a_matrix)
            result_l, result_u = L, U
        except ImportError:
            # Fallback to numpy
            result_l = np.linalg.cholesky(a_matrix @ a_matrix.T)
            result_u = np.linalg.solve(result_l, a_matrix)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return result_l, result_u
    
    def qr_decomposition(self, a: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """QR decomposition"""
        a_matrix = np.asarray(a, dtype=np.float32)
        
        start_time = time.perf_counter()
        Q, R = np.linalg.qr(a_matrix)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return Q, R
    
    def svd(self, a: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Singular Value Decomposition"""
        a_matrix = np.asarray(a, dtype=np.float32)
        
        start_time = time.perf_counter()
        U, s, Vt = np.linalg.svd(a_matrix)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return U, s, Vt
    
    def eigenvalues(self, a: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """Eigenvalue decomposition"""
        a_matrix = np.asarray(a, dtype=np.float32)
        
        if len(a_matrix.shape) != 2 or a_matrix.shape[0] != a_matrix.shape[1]:
            raise ValueError("Eigenvalue decomposition requires square matrix")
        
        start_time = time.perf_counter()
        eigenvals, eigenvecs = np.linalg.eig(a_matrix)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return eigenvals, eigenvecs


class MatrixSolvers:
    """Linear system solvers with SIMD optimization"""
    
    def __init__(self, simd_processor: Optional[SIMDProcessor] = None):
        self.simd = simd_processor or SIMDProcessor()
        self._lock = threading.RLock()
    
    def solve_linear_system(self, A: ArrayLike, b: ArrayLike,
                           out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve linear system Ax = b
        
        Args:
            A: Coefficient matrix (N x N)
            b: Right-hand side vector (N,)
            out: Optional output vector (N,)
        
        Returns:
            Solution vector x
        """
        A_matrix = np.asarray(A, dtype=np.float32)
        b_vector = np.asarray(b, dtype=np.float32)
        
        if len(A_matrix.shape) != 2 or A_matrix.shape[0] != A_matrix.shape[1]:
            raise ValueError("Coefficient matrix must be square")
        
        if len(b_vector.shape) != 1 or b_vector.shape[0] != A_matrix.shape[0]:
            raise ValueError("Right-hand side must be compatible vector")
        
        if out is None:
            out = np.empty_like(b_vector)
        
        start_time = time.perf_counter()
        solution = np.linalg.solve(A_matrix, b_vector)
        out[:] = solution
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def least_squares(self, A: ArrayLike, b: ArrayLike) -> Tuple[np.ndarray, float]:
        """
        Solve least squares problem: min ||Ax - b||²
        
        Returns:
            Tuple of (solution, residual)
        """
        A_matrix = np.asarray(A, dtype=np.float32)
        b_vector = np.asarray(b, dtype=np.float32)
        
        start_time = time.perf_counter()
        solution, residuals, rank, s = np.linalg.lstsq(A_matrix, b_vector, rcond=None)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        residual = residuals[0] if len(residuals) > 0 else 0.0
        return solution, float(residual)
    
    def matrix_inverse(self, a: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute matrix inverse"""
        a_matrix = np.asarray(a, dtype=np.float32)
        
        if len(a_matrix.shape) != 2 or a_matrix.shape[0] != a_matrix.shape[1]:
            raise ValueError("Matrix inverse requires square matrix")
        
        if out is None:
            out = np.empty_like(a_matrix)
        
        start_time = time.perf_counter()
        inverse = np.linalg.inv(a_matrix)
        out[:] = inverse
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def determinant(self, a: ArrayLike) -> float:
        """Compute matrix determinant"""
        a_matrix = np.asarray(a, dtype=np.float32)
        
        if len(a_matrix.shape) != 2 or a_matrix.shape[0] != a_matrix.shape[1]:
            raise ValueError("Determinant requires square matrix")
        
        start_time = time.perf_counter()
        det = np.linalg.det(a_matrix)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return float(det)


class ConvolutionOperations:
    """SIMD-optimized convolution operations for neural networks"""
    
    def __init__(self, simd_processor: Optional[SIMDProcessor] = None):
        self.simd = simd_processor or SIMDProcessor()
        self._lock = threading.RLock()
    
    def conv1d(self, input_array: ArrayLike, kernel: ArrayLike,
               stride: int = 1, padding: int = 0,
               out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        1D convolution operation
        
        Args:
            input_array: Input signal (N,)
            kernel: Convolution kernel (K,)
            stride: Stride length
            padding: Padding size
            out: Optional output array
        
        Returns:
            Convolved signal
        """
        input_arr = np.asarray(input_array, dtype=np.float32)
        kernel_arr = np.asarray(kernel, dtype=np.float32)
        
        if len(input_arr.shape) != 1 or len(kernel_arr.shape) != 1:
            raise ValueError("1D convolution requires 1D arrays")
        
        start_time = time.perf_counter()
        
        # Add padding if needed
        if padding > 0:
            input_arr = np.pad(input_arr, padding, mode='constant')
        
        # Calculate output size
        output_size = (len(input_arr) - len(kernel_arr)) // stride + 1
        
        if out is None:
            out = np.empty(output_size, dtype=np.float32)
        
        # Perform convolution
        for i in range(output_size):
            start_idx = i * stride
            end_idx = start_idx + len(kernel_arr)
            out[i] = np.dot(input_arr[start_idx:end_idx], kernel_arr)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def conv2d(self, input_matrix: ArrayLike, kernel: ArrayLike,
               stride: Tuple[int, int] = (1, 1), 
               padding: Tuple[int, int] = (0, 0),
               out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        2D convolution operation
        
        Args:
            input_matrix: Input image/feature map (H x W)
            kernel: Convolution kernel (Kh x Kw)
            stride: Stride (stride_h, stride_w)
            padding: Padding (pad_h, pad_w)
            out: Optional output matrix
        
        Returns:
            Convolved feature map
        """
        input_arr = np.asarray(input_matrix, dtype=np.float32)
        kernel_arr = np.asarray(kernel, dtype=np.float32)
        
        if len(input_arr.shape) != 2 or len(kernel_arr.shape) != 2:
            raise ValueError("2D convolution requires 2D arrays")
        
        start_time = time.perf_counter()
        
        # Add padding if needed
        if padding[0] > 0 or padding[1] > 0:
            input_arr = np.pad(input_arr, padding, mode='constant')
        
        # Calculate output dimensions
        output_h = (input_arr.shape[0] - kernel_arr.shape[0]) // stride[0] + 1
        output_w = (input_arr.shape[1] - kernel_arr.shape[1]) // stride[1] + 1
        
        if out is None:
            out = np.empty((output_h, output_w), dtype=np.float32)
        
        # Perform 2D convolution
        for i in range(output_h):
            for j in range(output_w):
                start_h = i * stride[0]
                start_w = j * stride[1]
                end_h = start_h + kernel_arr.shape[0]
                end_w = start_w + kernel_arr.shape[1]
                
                region = input_arr[start_h:end_h, start_w:end_w]
                out[i, j] = np.sum(region * kernel_arr)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out


class ActivationFunctions:
    """SIMD-optimized activation functions for neural networks"""
    
    def __init__(self, simd_processor: Optional[SIMDProcessor] = None):
        self.simd = simd_processor or SIMDProcessor()
        self._lock = threading.RLock()
    
    def relu(self, a: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Rectified Linear Unit: max(0, x)"""
        a_array = np.asarray(a, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_array)
        
        start_time = time.perf_counter()
        np.maximum(a_array, 0.0, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def leaky_relu(self, a: ArrayLike, alpha: float = 0.01,
                   out: Optional[np.ndarray] = None) -> np.ndarray:
        """Leaky ReLU: max(alpha * x, x)"""
        a_array = np.asarray(a, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_array)
        
        start_time = time.perf_counter()
        result = np.where(a_array > 0, a_array, alpha * a_array)
        out[:] = result
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def sigmoid(self, a: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Sigmoid activation: 1 / (1 + exp(-x))"""
        a_array = np.asarray(a, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_array)
        
        start_time = time.perf_counter()
        
        # Compute sigmoid with numerical stability
        np.clip(a_array, -500, 500, out=out)  # Prevent overflow
        np.negative(out, out=out)
        np.exp(out, out=out)
        np.add(1.0, out, out=out)
        np.divide(1.0, out, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def tanh(self, a: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Hyperbolic tangent activation"""
        a_array = np.asarray(a, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_array)
        
        start_time = time.perf_counter()
        np.tanh(a_array, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def softmax(self, a: ArrayLike, axis: int = -1,
                out: Optional[np.ndarray] = None) -> np.ndarray:
        """Softmax activation function"""
        a_array = np.asarray(a, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_array)
        
        start_time = time.perf_counter()
        
        # Numerically stable softmax
        a_max = np.max(a_array, axis=axis, keepdims=True)
        exp_a = np.exp(a_array - a_max)
        sum_exp_a = np.sum(exp_a, axis=axis, keepdims=True)
        np.divide(exp_a, sum_exp_a, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out


class BatchOperations:
    """Batch processing operations for machine learning"""
    
    def __init__(self, simd_processor: Optional[SIMDProcessor] = None):
        self.simd = simd_processor or SIMDProcessor()
        self._lock = threading.RLock()
    
    def batch_matrix_multiply(self, batch_a: ArrayLike, batch_b: ArrayLike,
                             out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Batch matrix multiplication
        
        Args:
            batch_a: Batch of matrices (B x M x K)
            batch_b: Batch of matrices (B x K x N)
            out: Optional output batch (B x M x N)
        
        Returns:
            Batch of result matrices
        """
        batch_a_arr = np.asarray(batch_a, dtype=np.float32)
        batch_b_arr = np.asarray(batch_b, dtype=np.float32)
        
        if len(batch_a_arr.shape) != 3 or len(batch_b_arr.shape) != 3:
            raise ValueError("Batch operations require 3D arrays")
        
        if batch_a_arr.shape[0] != batch_b_arr.shape[0]:
            raise ValueError("Batch sizes must match")
        
        if batch_a_arr.shape[2] != batch_b_arr.shape[1]:
            raise ValueError("Matrix dimensions must be compatible")
        
        if out is None:
            out_shape = (batch_a_arr.shape[0], batch_a_arr.shape[1], batch_b_arr.shape[2])
            out = np.empty(out_shape, dtype=np.float32)
        
        start_time = time.perf_counter()
        
        # Use numpy's batched matrix multiplication
        np.matmul(batch_a_arr, batch_b_arr, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def batch_normalize(self, batch: ArrayLike, axis: int = 0,
                       out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Batch normalization
        
        Args:
            batch: Input batch (B x ...)
            axis: Normalization axis
            out: Optional output batch
        
        Returns:
            Normalized batch
        """
        batch_arr = np.asarray(batch, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(batch_arr)
        
        start_time = time.perf_counter()
        
        # Compute mean and variance along the batch axis
        mean = np.mean(batch_arr, axis=axis, keepdims=True)
        var = np.var(batch_arr, axis=axis, keepdims=True)
        
        # Normalize
        std = np.sqrt(var + 1e-8)  # Add epsilon for numerical stability
        np.subtract(batch_arr, mean, out=out)
        np.divide(out, std, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
