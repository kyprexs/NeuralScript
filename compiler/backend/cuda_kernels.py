"""
CUDA Kernel Generation for NeuralScript

This module provides CUDA kernel generation from NeuralScript IR, including
automatic kernel synthesis for common operations and optimization.

Key Features:
- IR-to-CUDA kernel translation
- Automatic kernel optimization
- Template-based kernel generation  
- Memory coalescing optimization
- Warp-level primitives
- Multi-dimensional indexing
"""

import re
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .cuda_backend import CudaBackend, CudaDataType, get_cuda_backend

class KernelOptimizationLevel(Enum):
    """CUDA kernel optimization levels"""
    NONE = 0
    BASIC = 1
    AGGRESSIVE = 2
    MAXIMUM = 3

@dataclass  
class KernelTemplate:
    """Template for generating CUDA kernels"""
    name: str
    source_template: str
    parameters: List[str]
    optimization_hints: Dict[str, Any]
    memory_pattern: str  # "sequential", "strided", "random"
    compute_intensity: float  # FLOPS per memory access

class CudaKernelGenerator:
    """Generates optimized CUDA kernels from NeuralScript operations"""
    
    def __init__(self, backend: Optional[CudaBackend] = None):
        self.backend = backend or get_cuda_backend()
        self.kernel_templates = {}
        self.optimization_level = KernelOptimizationLevel.AGGRESSIVE
        
        # Initialize built-in kernel templates
        self._initialize_templates()
        
    def _initialize_templates(self):
        """Initialize built-in kernel templates"""
        
        # Vector addition kernel
        vector_add_template = KernelTemplate(
            name="vector_add",
            source_template="""
__global__ void vector_add_{dtype}(
    const {dtype}* __restrict__ a,
    const {dtype}* __restrict__ b,
    {dtype}* __restrict__ c,
    int n
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {{
        c[i] = a[i] + b[i];
    }}
}}
            """.strip(),
            parameters=["a", "b", "c", "n"],
            optimization_hints={"vectorize": True, "memory_coalesced": True},
            memory_pattern="sequential",
            compute_intensity=1.0
        )
        
        # Matrix multiplication kernel  
        matrix_mul_template = KernelTemplate(
            name="matrix_multiply",
            source_template="""
__global__ void matrix_multiply_{dtype}(
    const {dtype}* __restrict__ A,
    const {dtype}* __restrict__ B, 
    {dtype}* __restrict__ C,
    int M, int N, int K
) {{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {{
        {dtype} sum = 0.0;
        for (int k = 0; k < K; k++) {{
            sum += A[row * K + k] * B[k * N + col];
        }}
        C[row * N + col] = sum;
    }}
}}
            """.strip(),
            parameters=["A", "B", "C", "M", "N", "K"],
            optimization_hints={"shared_memory": True, "tiled": True},
            memory_pattern="strided",
            compute_intensity=2.0
        )
        
        # Optimized tiled matrix multiplication
        tiled_matrix_mul_template = KernelTemplate(
            name="tiled_matrix_multiply",
            source_template="""
#define TILE_SIZE {tile_size}

__global__ void tiled_matrix_multiply_{dtype}(
    const {dtype}* __restrict__ A,
    const {dtype}* __restrict__ B,
    {dtype}* __restrict__ C,
    int M, int N, int K
) {{
    __shared__ {dtype} As[TILE_SIZE][TILE_SIZE];
    __shared__ {dtype} Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    {dtype} sum = 0.0;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {{
        // Load tile into shared memory
        if (row < M && tile * TILE_SIZE + tx < K)
            As[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0;
            
        if (col < N && tile * TILE_SIZE + ty < K)
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0;
            
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {{
            sum += As[ty][k] * Bs[k][tx];
        }}
        
        __syncthreads();
    }}
    
    if (row < M && col < N) {{
        C[row * N + col] = sum;
    }}
}}
            """.strip(),
            parameters=["A", "B", "C", "M", "N", "K"],
            optimization_hints={"shared_memory": True, "tiled": True, "tile_size": 32},
            memory_pattern="tiled",
            compute_intensity=4.0
        )
        
        # Element-wise operations
        elementwise_template = KernelTemplate(
            name="elementwise_op",
            source_template="""
__global__ void elementwise_{op_name}_{dtype}(
    const {dtype}* __restrict__ input,
    {dtype}* __restrict__ output,
    int n
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {{
        output[i] = {operation};
    }}
}}
            """.strip(),
            parameters=["input", "output", "n"],
            optimization_hints={"vectorize": True, "memory_coalesced": True},
            memory_pattern="sequential",
            compute_intensity=1.0
        )
        
        # Reduction kernel
        reduction_template = KernelTemplate(
            name="reduction",
            source_template="""
__global__ void reduce_{op_name}_{dtype}(
    const {dtype}* __restrict__ input,
    {dtype}* __restrict__ output,
    int n
) {{
    extern __shared__ {dtype} sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : {identity};
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = 1; s < blockDim.x; s *= 2) {{
        if (tid % (2*s) == 0) {{
            sdata[tid] = {reduction_op};
        }}
        __syncthreads();
    }}
    
    // Write result for this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}}
            """.strip(),
            parameters=["input", "output", "n"],
            optimization_hints={"shared_memory": True, "reduction": True},
            memory_pattern="sequential",
            compute_intensity=0.5
        )
        
        self.kernel_templates = {
            "vector_add": vector_add_template,
            "matrix_multiply": matrix_mul_template,
            "tiled_matrix_multiply": tiled_matrix_mul_template,
            "elementwise_op": elementwise_template,
            "reduction": reduction_template
        }
    
    def generate_vector_add_kernel(self, dtype: CudaDataType = CudaDataType.FLOAT32) -> str:
        """Generate vector addition kernel"""
        template = self.kernel_templates["vector_add"]
        return template.source_template.format(dtype=dtype.value)
    
    def generate_matrix_multiply_kernel(self, 
                                      dtype: CudaDataType = CudaDataType.FLOAT32,
                                      use_tiled: bool = True,
                                      tile_size: int = 32) -> str:
        """Generate matrix multiplication kernel"""
        if use_tiled and self.optimization_level.value >= KernelOptimizationLevel.BASIC.value:
            template = self.kernel_templates["tiled_matrix_multiply"]
            return template.source_template.format(dtype=dtype.value, tile_size=tile_size)
        else:
            template = self.kernel_templates["matrix_multiply"]
            return template.source_template.format(dtype=dtype.value)
    
    def generate_elementwise_kernel(self, 
                                   operation_name: str,
                                   operation_code: str,
                                   dtype: CudaDataType = CudaDataType.FLOAT32) -> str:
        """Generate element-wise operation kernel"""
        template = self.kernel_templates["elementwise_op"]
        return template.source_template.format(
            op_name=operation_name,
            dtype=dtype.value,
            operation=operation_code
        )
    
    def generate_reduction_kernel(self,
                                 operation_name: str,
                                 reduction_operation: str,
                                 identity_value: str,
                                 dtype: CudaDataType = CudaDataType.FLOAT32) -> str:
        """Generate reduction kernel"""
        template = self.kernel_templates["reduction"]
        return template.source_template.format(
            op_name=operation_name,
            dtype=dtype.value,
            reduction_op=reduction_operation,
            identity=identity_value
        )
    
    def generate_convolution_kernel(self,
                                   dtype: CudaDataType = CudaDataType.FLOAT32,
                                   kernel_size: int = 3,
                                   stride: int = 1,
                                   padding: int = 0) -> str:
        """Generate 2D convolution kernel"""
        return f"""
__global__ void conv2d_{dtype.value}(
    const {dtype.value}* __restrict__ input,
    const {dtype.value}* __restrict__ kernel,
    {dtype.value}* __restrict__ output,
    int input_height, int input_width,
    int kernel_height, int kernel_width,
    int output_height, int output_width,
    int stride, int padding
) {{
    int output_y = blockIdx.y * blockDim.y + threadIdx.y;
    int output_x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (output_y < output_height && output_x < output_width) {{
        {dtype.value} sum = 0.0;
        
        for (int ky = 0; ky < kernel_height; ky++) {{
            for (int kx = 0; kx < kernel_width; kx++) {{
                int input_y = output_y * stride + ky - padding;
                int input_x = output_x * stride + kx - padding;
                
                if (input_y >= 0 && input_y < input_height && 
                    input_x >= 0 && input_x < input_width) {{
                    
                    int input_idx = input_y * input_width + input_x;
                    int kernel_idx = ky * kernel_width + kx;
                    sum += input[input_idx] * kernel[kernel_idx];
                }}
            }}
        }}
        
        int output_idx = output_y * output_width + output_x;
        output[output_idx] = sum;
    }}
}}
        """.strip()
    
    def generate_activation_kernels(self, dtype: CudaDataType = CudaDataType.FLOAT32) -> Dict[str, str]:
        """Generate common activation function kernels"""
        activations = {}
        
        # ReLU
        relu_code = f"fmaxf(input[i], 0.0f)" if dtype == CudaDataType.FLOAT32 else f"fmax(input[i], 0.0)"
        activations["relu"] = self.generate_elementwise_kernel("relu", relu_code, dtype)
        
        # Sigmoid
        sigmoid_code = f"1.0f / (1.0f + expf(-input[i]))" if dtype == CudaDataType.FLOAT32 else f"1.0 / (1.0 + exp(-input[i]))"
        activations["sigmoid"] = self.generate_elementwise_kernel("sigmoid", sigmoid_code, dtype)
        
        # Tanh
        tanh_code = f"tanhf(input[i])" if dtype == CudaDataType.FLOAT32 else f"tanh(input[i])"
        activations["tanh"] = self.generate_elementwise_kernel("tanh", tanh_code, dtype)
        
        return activations
    
    def optimize_kernel_parameters(self, 
                                  kernel_name: str,
                                  problem_size: Tuple[int, ...],
                                  device_id: int = 0) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Automatically optimize kernel launch parameters"""
        device_info = self.backend.get_device_info(device_id)
        
        if kernel_name in ["vector_add", "elementwise_op"]:
            return self._optimize_1d_kernel(problem_size[0], device_info)
        elif kernel_name in ["matrix_multiply", "tiled_matrix_multiply"]:
            return self._optimize_2d_kernel(problem_size, device_info)
        else:
            # Default parameters
            return ((256, 1, 1), ((problem_size[0] + 255) // 256, 1, 1))
    
    def _optimize_1d_kernel(self, n: int, device_info) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Optimize parameters for 1D kernels"""
        # Use multiple of warp size
        block_size = min(512, ((n + device_info.warp_size - 1) // device_info.warp_size) * device_info.warp_size)
        block_size = max(block_size, device_info.warp_size)
        
        grid_size = (n + block_size - 1) // block_size
        grid_size = min(grid_size, device_info.multiprocessor_count * 4)  # 4 blocks per SM
        
        return ((block_size, 1, 1), (grid_size, 1, 1))
    
    def _optimize_2d_kernel(self, problem_size: Tuple[int, int], device_info) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Optimize parameters for 2D kernels"""
        M, N = problem_size[:2]
        
        # Use square blocks for better cache locality
        block_x = block_y = 16
        if M < 16 or N < 16:
            block_x = block_y = 8
        
        grid_x = (N + block_x - 1) // block_x
        grid_y = (M + block_y - 1) // block_y
        
        return ((block_x, block_y, 1), (grid_x, grid_y, 1))
    
    def compile_and_cache_kernel(self, 
                                kernel_name: str, 
                                source_code: str, 
                                dtype: CudaDataType = CudaDataType.FLOAT32) -> str:
        """Compile kernel and return unique identifier"""
        # Add headers and type definitions
        full_source = self._add_kernel_headers(source_code, dtype)
        
        # Compile kernel
        kernel = self.backend.compile_kernel(kernel_name, full_source)
        
        # Return kernel identifier for later use
        return kernel_name
    
    def _add_kernel_headers(self, source_code: str, dtype: CudaDataType) -> str:
        """Add necessary headers and definitions to kernel source"""
        headers = """
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// Type definitions for better readability
"""
        
        if dtype == CudaDataType.FLOAT32:
            headers += """
#define FLOAT_TYPE float
#define MATH_FUNC(func) func##f
"""
        else:
            headers += """
#define FLOAT_TYPE double  
#define MATH_FUNC(func) func
"""
        
        return headers + "\n" + source_code
    
    def benchmark_kernel(self, 
                        kernel_name: str,
                        test_data: Dict[str, np.ndarray],
                        iterations: int = 100) -> Dict[str, float]:
        """Benchmark a compiled kernel"""
        if kernel_name not in self.backend.compiled_kernels:
            raise ValueError(f"Kernel {kernel_name} not compiled")
        
        # Allocate GPU memory for test data
        gpu_buffers = {}
        for name, data in test_data.items():
            gpu_ptr = self.backend.allocate_memory(data.nbytes)
            self.backend.copy_to_device(data, gpu_ptr)
            gpu_buffers[name] = gpu_ptr
        
        # Determine optimal launch parameters
        problem_size = tuple(test_data[list(test_data.keys())[0]].shape)
        block_size, grid_size = self.optimize_kernel_parameters(kernel_name, problem_size)
        
        # Warm up
        for _ in range(10):
            self.backend.launch_kernel(kernel_name, grid_size, *gpu_buffers.values())
        
        # Benchmark
        execution_times = []
        for _ in range(iterations):
            exec_time = self.backend.launch_kernel(kernel_name, grid_size, *gpu_buffers.values())
            execution_times.append(exec_time)
        
        # Clean up
        for gpu_ptr in gpu_buffers.values():
            self.backend.free_memory(gpu_ptr)
        
        return {
            'average_time_ms': np.mean(execution_times),
            'min_time_ms': np.min(execution_times),
            'max_time_ms': np.max(execution_times),
            'std_time_ms': np.std(execution_times),
            'total_iterations': iterations
        }

def get_kernel_generator(backend: Optional[CudaBackend] = None) -> CudaKernelGenerator:
    """Get global kernel generator instance"""
    if not hasattr(get_kernel_generator, '_instance'):
        get_kernel_generator._instance = CudaKernelGenerator(backend)
    return get_kernel_generator._instance

# Example usage
if __name__ == "__main__":
    generator = get_kernel_generator()
    
    # Generate vector addition kernel
    vector_add_src = generator.generate_vector_add_kernel()
    print("Vector Add Kernel:")
    print(vector_add_src)
    print()
    
    # Generate matrix multiplication kernel  
    matrix_mul_src = generator.generate_matrix_multiply_kernel(use_tiled=True)
    print("Tiled Matrix Multiply Kernel:")
    print(matrix_mul_src[:500] + "...")
    print()
    
    # Generate activation kernels
    activations = generator.generate_activation_kernels()
    print("Generated activation kernels:", list(activations.keys()))
