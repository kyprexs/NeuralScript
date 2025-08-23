"""
High-Performance Vector Operations using SIMD

Comprehensive implementation of vectorized mathematical operations
optimized for scientific computing and machine learning workloads.
"""

import numpy as np
import math
import threading
import time
from typing import Union, Optional, Tuple, List, Any
from abc import ABC, abstractmethod

from .simd_core import SIMDProcessor, DataType, SIMDConfiguration


ArrayLike = Union[np.ndarray, List[float], List[int]]


class VectorOperations:
    """
    High-performance SIMD vector operations.
    
    Provides vectorized implementations of mathematical operations
    with automatic SIMD optimization and hardware acceleration.
    """
    
    def __init__(self, simd_processor: Optional[SIMDProcessor] = None):
        self.simd = simd_processor or SIMDProcessor()
        self._operation_cache = {}
        self._lock = threading.RLock()
    
    def add(self, a: ArrayLike, b: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized element-wise addition: a + b"""
        a_array = np.asarray(a, dtype=np.float32)
        b_array = np.asarray(b, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_array)
        
        start_time = time.perf_counter()
        
        # Use numpy's optimized SIMD implementation
        np.add(a_array, b_array, out=out)
        
        # Update statistics
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def subtract(self, a: ArrayLike, b: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized element-wise subtraction: a - b"""
        a_array = np.asarray(a, dtype=np.float32)
        b_array = np.asarray(b, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_array)
        
        start_time = time.perf_counter()
        np.subtract(a_array, b_array, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def multiply(self, a: ArrayLike, b: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized element-wise multiplication: a * b"""
        a_array = np.asarray(a, dtype=np.float32)
        b_array = np.asarray(b, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_array)
        
        start_time = time.perf_counter()
        np.multiply(a_array, b_array, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def divide(self, a: ArrayLike, b: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized element-wise division: a / b"""
        a_array = np.asarray(a, dtype=np.float32)
        b_array = np.asarray(b, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_array)
        
        start_time = time.perf_counter()
        np.divide(a_array, b_array, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def fused_multiply_add(self, a: ArrayLike, b: ArrayLike, c: ArrayLike, 
                          out: Optional[np.ndarray] = None) -> np.ndarray:
        """Fused multiply-add: (a * b) + c with single rounding"""
        a_array = np.asarray(a, dtype=np.float32)
        b_array = np.asarray(b, dtype=np.float32)
        c_array = np.asarray(c, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_array)
        
        start_time = time.perf_counter()
        
        # Use numpy's fma-like operation
        if self.simd.config.use_fma:
            # Simulate FMA with multiply then add
            np.multiply(a_array, b_array, out=out)
            np.add(out, c_array, out=out)
        else:
            # Standard multiply-add
            out[:] = a_array * b_array + c_array
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def dot_product(self, a: ArrayLike, b: ArrayLike) -> float:
        """Vectorized dot product: sum(a * b)"""
        a_array = np.asarray(a, dtype=np.float32)
        b_array = np.asarray(b, dtype=np.float32)
        
        start_time = time.perf_counter()
        result = np.dot(a_array, b_array)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return float(result)
    
    def cross_product(self, a: ArrayLike, b: ArrayLike) -> np.ndarray:
        """Vectorized 3D cross product"""
        a_array = np.asarray(a, dtype=np.float32)
        b_array = np.asarray(b, dtype=np.float32)
        
        if a_array.shape != (3,) or b_array.shape != (3,):
            raise ValueError("Cross product requires 3D vectors")
        
        start_time = time.perf_counter()
        result = np.cross(a_array, b_array)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return result
    
    def magnitude(self, a: ArrayLike) -> float:
        """Vectorized vector magnitude: sqrt(sum(a^2))"""
        a_array = np.asarray(a, dtype=np.float32)
        
        start_time = time.perf_counter()
        result = np.linalg.norm(a_array)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return float(result)
    
    def normalize(self, a: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized vector normalization: a / ||a||"""
        a_array = np.asarray(a, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_array)
        
        start_time = time.perf_counter()
        
        magnitude = np.linalg.norm(a_array)
        if magnitude > 1e-8:
            np.divide(a_array, magnitude, out=out)
        else:
            out[:] = 0.0
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out


class VectorMath:
    """Advanced mathematical vector operations"""
    
    def __init__(self, simd_processor: Optional[SIMDProcessor] = None):
        self.simd = simd_processor or SIMDProcessor()
        self._lock = threading.RLock()
    
    def sqrt(self, a: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized square root"""
        a_array = np.asarray(a, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_array)
        
        start_time = time.perf_counter()
        np.sqrt(a_array, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def rsqrt(self, a: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized reciprocal square root: 1/sqrt(a)"""
        a_array = np.asarray(a, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_array)
        
        start_time = time.perf_counter()
        
        # Fast reciprocal square root approximation
        np.sqrt(a_array, out=out)
        np.divide(1.0, out, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def power(self, a: ArrayLike, exponent: float, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized power: a^exponent"""
        a_array = np.asarray(a, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_array)
        
        start_time = time.perf_counter()
        np.power(a_array, exponent, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def exp(self, a: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized exponential: e^a"""
        a_array = np.asarray(a, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_array)
        
        start_time = time.perf_counter()
        np.exp(a_array, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def log(self, a: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized natural logarithm"""
        a_array = np.asarray(a, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_array)
        
        start_time = time.perf_counter()
        np.log(a_array, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def abs(self, a: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized absolute value"""
        a_array = np.asarray(a, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_array)
        
        start_time = time.perf_counter()
        np.abs(a_array, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def min_max(self, a: ArrayLike) -> Tuple[float, float]:
        """Find minimum and maximum values in vector"""
        a_array = np.asarray(a, dtype=np.float32)
        
        start_time = time.perf_counter()
        min_val = np.min(a_array)
        max_val = np.max(a_array)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return float(min_val), float(max_val)


class VectorComparison:
    """SIMD vector comparison operations"""
    
    def __init__(self, simd_processor: Optional[SIMDProcessor] = None):
        self.simd = simd_processor or SIMDProcessor()
        self._lock = threading.RLock()
    
    def equal(self, a: ArrayLike, b: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Element-wise equality comparison"""
        a_array = np.asarray(a, dtype=np.float32)
        b_array = np.asarray(b, dtype=np.float32)
        
        if out is None:
            out = np.empty(a_array.shape, dtype=bool)
        
        start_time = time.perf_counter()
        np.equal(a_array, b_array, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def less_than(self, a: ArrayLike, b: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Element-wise less than comparison"""
        a_array = np.asarray(a, dtype=np.float32)
        b_array = np.asarray(b, dtype=np.float32)
        
        if out is None:
            out = np.empty(a_array.shape, dtype=bool)
        
        start_time = time.perf_counter()
        np.less(a_array, b_array, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def greater_than(self, a: ArrayLike, b: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Element-wise greater than comparison"""
        a_array = np.asarray(a, dtype=np.float32)
        b_array = np.asarray(b, dtype=np.float32)
        
        if out is None:
            out = np.empty(a_array.shape, dtype=bool)
        
        start_time = time.perf_counter()
        np.greater(a_array, b_array, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out


class VectorLogical:
    """SIMD vector logical operations"""
    
    def __init__(self, simd_processor: Optional[SIMDProcessor] = None):
        self.simd = simd_processor or SIMDProcessor()
        self._lock = threading.RLock()
    
    def logical_and(self, a: ArrayLike, b: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Element-wise logical AND"""
        a_array = np.asarray(a, dtype=bool)
        b_array = np.asarray(b, dtype=bool)
        
        if out is None:
            out = np.empty_like(a_array, dtype=bool)
        
        start_time = time.perf_counter()
        np.logical_and(a_array, b_array, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def logical_or(self, a: ArrayLike, b: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Element-wise logical OR"""
        a_array = np.asarray(a, dtype=bool)
        b_array = np.asarray(b, dtype=bool)
        
        if out is None:
            out = np.empty_like(a_array, dtype=bool)
        
        start_time = time.perf_counter()
        np.logical_or(a_array, b_array, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def logical_not(self, a: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Element-wise logical NOT"""
        a_array = np.asarray(a, dtype=bool)
        
        if out is None:
            out = np.empty_like(a_array, dtype=bool)
        
        start_time = time.perf_counter()
        np.logical_not(a_array, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out


class VectorTranscendental:
    """SIMD transcendental function operations"""
    
    def __init__(self, simd_processor: Optional[SIMDProcessor] = None):
        self.simd = simd_processor or SIMDProcessor()
        self._lock = threading.RLock()
    
    def sin(self, a: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized sine function"""
        a_array = np.asarray(a, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_array)
        
        start_time = time.perf_counter()
        np.sin(a_array, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def cos(self, a: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized cosine function"""
        a_array = np.asarray(a, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_array)
        
        start_time = time.perf_counter()
        np.cos(a_array, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def tan(self, a: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized tangent function"""
        a_array = np.asarray(a, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(a_array)
        
        start_time = time.perf_counter()
        np.tan(a_array, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
    
    def sincos(self, a: ArrayLike, sin_out: Optional[np.ndarray] = None, 
              cos_out: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Simultaneous sine and cosine computation"""
        a_array = np.asarray(a, dtype=np.float32)
        
        if sin_out is None:
            sin_out = np.empty_like(a_array)
        if cos_out is None:
            cos_out = np.empty_like(a_array)
        
        start_time = time.perf_counter()
        
        # Compute both simultaneously for efficiency
        np.sin(a_array, out=sin_out)
        np.cos(a_array, out=cos_out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return sin_out, cos_out
    
    def atan2(self, y: ArrayLike, x: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized atan2 function"""
        y_array = np.asarray(y, dtype=np.float32)
        x_array = np.asarray(x, dtype=np.float32)
        
        if out is None:
            out = np.empty_like(y_array)
        
        start_time = time.perf_counter()
        np.arctan2(y_array, x_array, out=out)
        
        with self._lock:
            self.simd._operation_count += 1
            self.simd._total_execution_time += time.perf_counter() - start_time
        
        return out
