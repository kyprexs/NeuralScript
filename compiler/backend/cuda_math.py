"""
CUDA Mathematical Operations for NeuralScript

This module provides GPU-accelerated mathematical operations including
matrix operations, vector operations, and mathematical functions.

Key Features:
- High-performance matrix operations (multiply, transpose, inverse)
- Vectorized mathematical functions
- Linear algebra primitives
- Statistical operations
- FFT and signal processing
- Optimized memory access patterns
"""

import numpy as np
from typing import Tuple, Optional, Union, List
import time
from dataclasses import dataclass

from .cuda_backend import CudaBackend, CudaDataType, get_cuda_backend
from .cuda_kernels import CudaKernelGenerator, get_kernel_generator

@dataclass
class CudaTensor:
    """GPU tensor representation"""
    shape: Tuple[int, ...]
    dtype: CudaDataType
    device_ptr: int
    strides: Tuple[int, ...] = None
    
    def __post_init__(self):
        if self.strides is None:
            # Calculate default C-style strides
            strides = []
            stride = 1
            for dim in reversed(self.shape):
                strides.append(stride)
                stride *= dim
            self.strides = tuple(reversed(strides))
    
    @property
    def size(self) -> int:
        """Total number of elements"""
        result = 1
        for dim in self.shape:
            result *= dim
        return result
    
    @property
    def nbytes(self) -> int:
        """Total bytes required"""
        dtype_sizes = {
            CudaDataType.FLOAT32: 4,
            CudaDataType.FLOAT64: 8,
            CudaDataType.INT32: 4,
            CudaDataType.INT64: 8,
            CudaDataType.BOOL: 1
        }
        return self.size * dtype_sizes[self.dtype]

class CudaMath:
    """GPU-accelerated mathematical operations"""
    
    def __init__(self, backend: Optional[CudaBackend] = None, 
                 kernel_generator: Optional[CudaKernelGenerator] = None):
        self.backend = backend or get_cuda_backend()
        self.kernel_generator = kernel_generator or get_kernel_generator()
        
        # Compile commonly used kernels
        self._compile_math_kernels()
        
    def _compile_math_kernels(self):
        """Compile frequently used mathematical kernels"""
        
        # Vector addition
        vector_add_src = self.kernel_generator.generate_vector_add_kernel()
        self.backend.compile_kernel("vector_add_float", vector_add_src)
        
        # Matrix multiplication (tiled)
        matrix_mul_src = self.kernel_generator.generate_matrix_multiply_kernel(use_tiled=True)
        self.backend.compile_kernel("tiled_matrix_multiply_float", matrix_mul_src)
        
        # Element-wise operations
        relu_src = self.kernel_generator.generate_elementwise_kernel(
            "relu", "fmaxf(input[i], 0.0f)"
        )
        self.backend.compile_kernel("elementwise_relu_float", relu_src)
        
        sigmoid_src = self.kernel_generator.generate_elementwise_kernel(
            "sigmoid", "1.0f / (1.0f + expf(-input[i]))"
        )
        self.backend.compile_kernel("elementwise_sigmoid_float", sigmoid_src)
        
        # Matrix transpose kernel
        transpose_src = """
__global__ void matrix_transpose_float(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols
) {
    __shared__ float tile[32][33]; // +1 to avoid bank conflicts
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Load data into shared memory tile
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    
    __syncthreads();
    
    // Write transposed data to output
    x = blockIdx.y * blockDim.y + threadIdx.x;
    y = blockIdx.x * blockDim.x + threadIdx.y;
    
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}
        """
        self.backend.compile_kernel("matrix_transpose_float", transpose_src)
        
        # Reduction sum kernel
        reduction_sum_src = self.kernel_generator.generate_reduction_kernel(
            "sum", "sdata[tid] + sdata[tid + s]", "0.0f"
        )
        self.backend.compile_kernel("reduce_sum_float", reduction_sum_src)
        
        print("CUDA math kernels compiled successfully")
    
    def create_tensor(self, shape: Tuple[int, ...], 
                     dtype: CudaDataType = CudaDataType.FLOAT32,
                     device_id: int = 0) -> CudaTensor:
        """Create a new GPU tensor"""
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        
        dtype_sizes = {
            CudaDataType.FLOAT32: 4,
            CudaDataType.FLOAT64: 8,
            CudaDataType.INT32: 4,
            CudaDataType.INT64: 8,
            CudaDataType.BOOL: 1
        }
        
        size_bytes = total_elements * dtype_sizes[dtype]
        device_ptr = self.backend.allocate_memory(size_bytes, device_id)
        
        return CudaTensor(shape=shape, dtype=dtype, device_ptr=device_ptr)
    
    def from_numpy(self, array: np.ndarray, 
                   device_id: int = 0) -> CudaTensor:
        """Create GPU tensor from NumPy array"""
        # Map NumPy dtypes to CUDA dtypes
        dtype_map = {
            np.float32: CudaDataType.FLOAT32,
            np.float64: CudaDataType.FLOAT64,
            np.int32: CudaDataType.INT32,
            np.int64: CudaDataType.INT64,
            np.bool_: CudaDataType.BOOL
        }
        
        if array.dtype.type not in dtype_map:
            # Convert to float32 by default
            array = array.astype(np.float32)
            cuda_dtype = CudaDataType.FLOAT32
        else:
            cuda_dtype = dtype_map[array.dtype.type]
        
        tensor = self.create_tensor(array.shape, cuda_dtype, device_id)
        self.backend.copy_to_device(array, tensor.device_ptr)
        
        return tensor
    
    def to_numpy(self, tensor: CudaTensor) -> np.ndarray:
        """Convert GPU tensor to NumPy array"""
        dtype_map = {
            CudaDataType.FLOAT32: np.float32,
            CudaDataType.FLOAT64: np.float64,
            CudaDataType.INT32: np.int32,
            CudaDataType.INT64: np.int64,
            CudaDataType.BOOL: np.bool_
        }
        
        numpy_dtype = dtype_map[tensor.dtype]
        result = np.empty(tensor.shape, dtype=numpy_dtype)
        self.backend.copy_from_device(tensor.device_ptr, result)
        
        return result
    
    def vector_add(self, a: CudaTensor, b: CudaTensor) -> CudaTensor:
        """Element-wise vector addition"""
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
        
        result = self.create_tensor(a.shape, a.dtype)
        
        # Determine grid and block sizes
        n = a.size
        block_size = min(512, ((n + 31) // 32) * 32)  # Multiple of warp size
        grid_size = (n + block_size - 1) // block_size
        
        # Launch kernel
        exec_time = self.backend.launch_kernel(
            "vector_add_float",
            (grid_size, 1, 1),
            a.device_ptr, b.device_ptr, result.device_ptr, n
        )
        
        return result
    
    def matrix_multiply(self, a: CudaTensor, b: CudaTensor) -> CudaTensor:
        """Matrix multiplication using tiled algorithm"""
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise ValueError("Both tensors must be 2D matrices")
        
        M, K = a.shape
        K2, N = b.shape
        
        if K != K2:
            raise ValueError(f"Matrix dimension mismatch: {K} vs {K2}")
        
        result = self.create_tensor((M, N), a.dtype)
        
        # Use 16x16 blocks for optimal shared memory usage
        block_size = (16, 16, 1)
        grid_size = ((N + 15) // 16, (M + 15) // 16, 1)
        
        # Launch tiled matrix multiplication kernel
        exec_time = self.backend.launch_kernel(
            "tiled_matrix_multiply_float",
            grid_size,
            a.device_ptr, b.device_ptr, result.device_ptr, M, N, K
        )
        
        return result
    
    def matrix_transpose(self, tensor: CudaTensor) -> CudaTensor:
        """Matrix transpose using shared memory tiling"""
        if len(tensor.shape) != 2:
            raise ValueError("Tensor must be 2D for transpose")
        
        rows, cols = tensor.shape
        result = self.create_tensor((cols, rows), tensor.dtype)
        
        # Use 32x32 tiles with bank conflict avoidance
        block_size = (32, 32, 1)
        grid_size = ((cols + 31) // 32, (rows + 31) // 32, 1)
        
        exec_time = self.backend.launch_kernel(
            "matrix_transpose_float",
            grid_size,
            tensor.device_ptr, result.device_ptr, rows, cols
        )
        
        return result
    
    def relu(self, tensor: CudaTensor) -> CudaTensor:
        """ReLU activation function"""
        result = self.create_tensor(tensor.shape, tensor.dtype)
        
        n = tensor.size
        block_size = min(512, ((n + 31) // 32) * 32)
        grid_size = (n + block_size - 1) // block_size
        
        exec_time = self.backend.launch_kernel(
            "elementwise_relu_float",
            (grid_size, 1, 1),
            tensor.device_ptr, result.device_ptr, n
        )
        
        return result
    
    def sigmoid(self, tensor: CudaTensor) -> CudaTensor:
        """Sigmoid activation function"""
        result = self.create_tensor(tensor.shape, tensor.dtype)
        
        n = tensor.size
        block_size = min(512, ((n + 31) // 32) * 32)
        grid_size = (n + block_size - 1) // block_size
        
        exec_time = self.backend.launch_kernel(
            "elementwise_sigmoid_float",
            (grid_size, 1, 1),
            tensor.device_ptr, result.device_ptr, n
        )
        
        return result
    
    def reduce_sum(self, tensor: CudaTensor, axis: Optional[int] = None) -> CudaTensor:
        """Sum reduction along specified axis or all elements"""
        if axis is None:
            # Sum all elements
            n = tensor.size
            
            # Use hierarchical reduction
            block_size = min(512, ((n + 31) // 32) * 32)
            grid_size = (n + block_size - 1) // block_size
            
            # First reduction: sum within blocks
            intermediate = self.create_tensor((grid_size,), tensor.dtype)
            
            exec_time = self.backend.launch_kernel(
                "reduce_sum_float",
                (grid_size, 1, 1),
                tensor.device_ptr, intermediate.device_ptr, n
            )
            
            # If multiple blocks, reduce again
            if grid_size > 1:
                return self.reduce_sum(intermediate)
            else:
                return intermediate
        else:
            raise NotImplementedError("Axis-specific reductions not yet implemented")
    
    def batch_matrix_multiply(self, a: CudaTensor, b: CudaTensor) -> CudaTensor:
        """Batch matrix multiplication for 3D tensors"""
        if len(a.shape) != 3 or len(b.shape) != 3:
            raise ValueError("Both tensors must be 3D for batch matrix multiplication")
        
        batch_size, M, K = a.shape
        batch_size2, K2, N = b.shape
        
        if batch_size != batch_size2 or K != K2:
            raise ValueError("Batch sizes or inner dimensions don't match")
        
        result = self.create_tensor((batch_size, M, N), a.dtype)
        
        # Launch separate matrix multiplication for each batch
        for batch in range(batch_size):
            a_offset = batch * M * K * 4  # Assuming float32
            b_offset = batch * K * N * 4
            c_offset = batch * M * N * 4
            
            block_size = (16, 16, 1)
            grid_size = ((N + 15) // 16, (M + 15) // 16, 1)
            
            exec_time = self.backend.launch_kernel(
                "tiled_matrix_multiply_float",
                grid_size,
                a.device_ptr + a_offset, 
                b.device_ptr + b_offset, 
                result.device_ptr + c_offset, 
                M, N, K
            )
        
        return result
    
    def benchmark_operations(self, sizes: List[Tuple[int, ...]], 
                           iterations: int = 100) -> Dict[str, any]:
        """Benchmark various mathematical operations"""
        results = {
            'vector_add': {},
            'matrix_multiply': {},
            'matrix_transpose': {},
            'relu': {},
            'reduce_sum': {}
        }
        
        for size in sizes:
            if len(size) == 1:
                # Vector operations
                n = size[0]
                a = self.create_tensor((n,), CudaDataType.FLOAT32)
                b = self.create_tensor((n,), CudaDataType.FLOAT32)
                
                # Vector addition benchmark
                start_time = time.perf_counter()
                for _ in range(iterations):
                    result = self.vector_add(a, b)
                vector_add_time = (time.perf_counter() - start_time) * 1000 / iterations
                
                results['vector_add'][str(size)] = {
                    'average_time_ms': vector_add_time,
                    'throughput_gb_s': (n * 4 * 3) / (vector_add_time * 1e-3) / 1e9,
                    'elements': n
                }
                
                # ReLU benchmark
                start_time = time.perf_counter()
                for _ in range(iterations):
                    result = self.relu(a)
                relu_time = (time.perf_counter() - start_time) * 1000 / iterations
                
                results['relu'][str(size)] = {
                    'average_time_ms': relu_time,
                    'throughput_gb_s': (n * 4 * 2) / (relu_time * 1e-3) / 1e9,
                    'elements': n
                }
                
            elif len(size) == 2:
                # Matrix operations
                M, N = size
                a = self.create_tensor((M, N), CudaDataType.FLOAT32)
                b = self.create_tensor((N, M), CudaDataType.FLOAT32)
                
                # Matrix multiplication benchmark
                start_time = time.perf_counter()
                for _ in range(iterations):
                    result = self.matrix_multiply(a, b)
                matmul_time = (time.perf_counter() - start_time) * 1000 / iterations
                
                flops = 2 * M * N * M  # Approximate FLOPS for matrix multiplication
                results['matrix_multiply'][str(size)] = {
                    'average_time_ms': matmul_time,
                    'gflops': flops / (matmul_time * 1e-3) / 1e9,
                    'size': (M, N)
                }
                
                # Matrix transpose benchmark
                start_time = time.perf_counter()
                for _ in range(iterations):
                    result = self.matrix_transpose(a)
                transpose_time = (time.perf_counter() - start_time) * 1000 / iterations
                
                results['matrix_transpose'][str(size)] = {
                    'average_time_ms': transpose_time,
                    'throughput_gb_s': (M * N * 4 * 2) / (transpose_time * 1e-3) / 1e9,
                    'size': (M, N)
                }
        
        return results
    
    def free_tensor(self, tensor: CudaTensor):
        """Free GPU memory for a tensor"""
        self.backend.free_memory(tensor.device_ptr)

def get_cuda_math(backend: Optional[CudaBackend] = None) -> CudaMath:
    """Get global CUDA math instance"""
    if not hasattr(get_cuda_math, '_instance'):
        get_cuda_math._instance = CudaMath(backend)
    return get_cuda_math._instance

# Example usage and benchmarking
if __name__ == "__main__":
    # Initialize CUDA math
    cuda_math = get_cuda_math()
    
    print("Testing CUDA mathematical operations...")
    
    # Test vector addition
    a_np = np.random.random((1000,)).astype(np.float32)
    b_np = np.random.random((1000,)).astype(np.float32)
    
    a_gpu = cuda_math.from_numpy(a_np)
    b_gpu = cuda_math.from_numpy(b_np)
    
    result_gpu = cuda_math.vector_add(a_gpu, b_gpu)
    result_np = cuda_math.to_numpy(result_gpu)
    
    expected = a_np + b_np
    error = np.max(np.abs(result_np - expected))
    print(f"Vector addition max error: {error}")
    
    # Test matrix multiplication
    A_np = np.random.random((128, 256)).astype(np.float32)
    B_np = np.random.random((256, 128)).astype(np.float32)
    
    A_gpu = cuda_math.from_numpy(A_np)
    B_gpu = cuda_math.from_numpy(B_np)
    
    C_gpu = cuda_math.matrix_multiply(A_gpu, B_gpu)
    C_np = cuda_math.to_numpy(C_gpu)
    
    expected = np.dot(A_np, B_np)
    error = np.max(np.abs(C_np - expected))
    print(f"Matrix multiplication max error: {error}")
    
    # Benchmark operations
    print("\nRunning performance benchmarks...")
    benchmark_sizes = [(1000,), (10000,), (100000,), (128, 128), (256, 256), (512, 512)]
    results = cuda_math.benchmark_operations(benchmark_sizes, iterations=10)
    
    print("\nBenchmark Results:")
    for op, sizes in results.items():
        print(f"\n{op}:")
        for size, metrics in sizes.items():
            print(f"  Size {size}: {metrics}")
    
    # Clean up
    cuda_math.free_tensor(a_gpu)
    cuda_math.free_tensor(b_gpu)
    cuda_math.free_tensor(result_gpu)
    cuda_math.free_tensor(A_gpu)
    cuda_math.free_tensor(B_gpu)
    cuda_math.free_tensor(C_gpu)
    
    print("\nCUDA math operations test completed!")
