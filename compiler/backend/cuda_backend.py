"""
CUDA Backend for NeuralScript GPU Acceleration

This module provides the main CUDA backend system for GPU acceleration in NeuralScript.
It includes device detection, memory management, kernel compilation, and execution.

Key Features:
- Automatic CUDA device detection and capabilities
- Dynamic kernel compilation with NVCC
- GPU memory management with pools
- Host-device memory transfers
- Multi-GPU support
- Performance profiling and optimization
"""

import os
import sys
import subprocess
import tempfile
import ctypes
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass
import threading
import time
import json
from pathlib import Path

try:
    import pycuda.driver as cuda
    import pycuda.compiler as cuda_compiler
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("PyCUDA not available. CUDA backend will use fallback mode.")

class CudaDataType(Enum):
    """CUDA data types supported by NeuralScript"""
    FLOAT32 = "float"
    FLOAT64 = "double" 
    INT32 = "int"
    INT64 = "long long"
    BOOL = "bool"

@dataclass
class CudaDeviceInfo:
    """Information about a CUDA device"""
    device_id: int
    name: str
    compute_capability: Tuple[int, int]
    memory_total: int
    memory_free: int
    multiprocessor_count: int
    max_threads_per_block: int
    max_block_dim: Tuple[int, int, int]
    max_grid_dim: Tuple[int, int, int]
    warp_size: int
    clock_rate: int
    memory_clock_rate: int
    memory_bus_width: int

@dataclass
class CudaKernel:
    """Compiled CUDA kernel information"""
    name: str
    source_code: str
    compiled_function: Any
    block_size: Tuple[int, int, int]
    grid_size: Tuple[int, int, int]
    shared_memory_size: int
    registers_per_thread: int

class CudaMemoryPool:
    """GPU memory pool for efficient allocation"""
    
    def __init__(self, device_id: int, initial_size: int = 1024 * 1024 * 1024):  # 1GB
        self.device_id = device_id
        self.pool = {}  # size -> list of free blocks
        self.allocated = {}  # ptr -> (size, in_use)
        self.total_allocated = 0
        self.peak_allocated = 0
        self.allocation_count = 0
        self.free_count = 0
        self._lock = threading.Lock()
        
    def allocate(self, size: int, alignment: int = 256) -> int:
        """Allocate GPU memory from pool"""
        # Align size to boundary
        aligned_size = ((size + alignment - 1) // alignment) * alignment
        
        with self._lock:
            # Try to find existing free block
            if aligned_size in self.pool and self.pool[aligned_size]:
                ptr = self.pool[aligned_size].pop()
                self.allocated[ptr] = (aligned_size, True)
                self.allocation_count += 1
                return ptr
            
            # Allocate new block
            if CUDA_AVAILABLE:
                ptr = cuda.mem_alloc(aligned_size)
                self.allocated[int(ptr)] = (aligned_size, True)
                self.total_allocated += aligned_size
                self.peak_allocated = max(self.peak_allocated, self.total_allocated)
                self.allocation_count += 1
                return int(ptr)
            else:
                # Fallback: simulate GPU allocation
                ptr = id(bytearray(aligned_size))
                self.allocated[ptr] = (aligned_size, True)
                self.allocation_count += 1
                return ptr
    
    def free(self, ptr: int):
        """Return memory to pool"""
        with self._lock:
            if ptr in self.allocated:
                size, in_use = self.allocated[ptr]
                if in_use:
                    self.allocated[ptr] = (size, False)
                    if size not in self.pool:
                        self.pool[size] = []
                    self.pool[size].append(ptr)
                    self.free_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics"""
        with self._lock:
            active_memory = sum(size for size, in_use in self.allocated.values() if in_use)
            return {
                'device_id': self.device_id,
                'total_allocated': self.total_allocated,
                'active_memory': active_memory,
                'peak_allocated': self.peak_allocated,
                'allocation_count': self.allocation_count,
                'free_count': self.free_count,
                'pool_blocks': sum(len(blocks) for blocks in self.pool.values())
            }

class CudaBackend:
    """Main CUDA backend for GPU acceleration"""
    
    def __init__(self, enable_profiling: bool = True):
        self.enable_profiling = enable_profiling
        self.devices: List[CudaDeviceInfo] = []
        self.current_device = 0
        self.memory_pools: Dict[int, CudaMemoryPool] = {}
        self.compiled_kernels: Dict[str, CudaKernel] = {}
        self.kernel_cache_dir = Path("cuda_kernel_cache")
        self.kernel_cache_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.kernel_execution_times = {}
        self.memory_transfer_times = {}
        self.compilation_times = {}
        
        # Initialize CUDA
        self._initialize_cuda()
        
    def _initialize_cuda(self):
        """Initialize CUDA devices and capabilities"""
        global CUDA_AVAILABLE
        if not CUDA_AVAILABLE:
            print("CUDA not available, using CPU fallback mode")
            return
            
        try:
            cuda.init()
            device_count = cuda.Device.count()
            print(f"Found {device_count} CUDA devices")
            
            for i in range(device_count):
                device = cuda.Device(i)
                context = device.make_context()
                
                # Get device properties
                attrs = device.get_attributes()
                memory_info = cuda.mem_get_info()
                
                device_info = CudaDeviceInfo(
                    device_id=i,
                    name=device.name(),
                    compute_capability=device.compute_capability(),
                    memory_total=memory_info[1],
                    memory_free=memory_info[0],
                    multiprocessor_count=attrs[cuda.device_attribute.MULTIPROCESSOR_COUNT],
                    max_threads_per_block=attrs[cuda.device_attribute.MAX_THREADS_PER_BLOCK],
                    max_block_dim=(
                        attrs[cuda.device_attribute.MAX_BLOCK_DIM_X],
                        attrs[cuda.device_attribute.MAX_BLOCK_DIM_Y],
                        attrs[cuda.device_attribute.MAX_BLOCK_DIM_Z]
                    ),
                    max_grid_dim=(
                        attrs[cuda.device_attribute.MAX_GRID_DIM_X],
                        attrs[cuda.device_attribute.MAX_GRID_DIM_Y],
                        attrs[cuda.device_attribute.MAX_GRID_DIM_Z]
                    ),
                    warp_size=attrs[cuda.device_attribute.WARP_SIZE],
                    clock_rate=attrs[cuda.device_attribute.CLOCK_RATE],
                    memory_clock_rate=attrs[cuda.device_attribute.MEMORY_CLOCK_RATE],
                    memory_bus_width=attrs[cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH]
                )
                
                self.devices.append(device_info)
                self.memory_pools[i] = CudaMemoryPool(i)
                
                context.pop()
                
            if self.devices:
                print(f"CUDA backend initialized with {len(self.devices)} devices")
                self._print_device_info()
            
        except Exception as e:
            print(f"Failed to initialize CUDA: {e}")
            CUDA_AVAILABLE = False
    
    def _print_device_info(self):
        """Print detailed information about available CUDA devices"""
        for device in self.devices:
            print(f"\nDevice {device.device_id}: {device.name}")
            print(f"  Compute Capability: {device.compute_capability[0]}.{device.compute_capability[1]}")
            print(f"  Memory: {device.memory_total / (1024**3):.1f} GB")
            print(f"  Multiprocessors: {device.multiprocessor_count}")
            print(f"  Max Threads/Block: {device.max_threads_per_block}")
            print(f"  Warp Size: {device.warp_size}")
    
    def set_device(self, device_id: int):
        """Set the current CUDA device"""
        if device_id < len(self.devices):
            self.current_device = device_id
            if CUDA_AVAILABLE:
                cuda.Device(device_id).make_context()
        else:
            raise ValueError(f"Device {device_id} not available")
    
    def get_device_info(self, device_id: Optional[int] = None) -> CudaDeviceInfo:
        """Get information about a CUDA device"""
        if device_id is None:
            device_id = self.current_device
        if device_id < len(self.devices):
            return self.devices[device_id]
        else:
            raise ValueError(f"Device {device_id} not available")
    
    def allocate_memory(self, size: int, device_id: Optional[int] = None) -> int:
        """Allocate GPU memory"""
        if device_id is None:
            device_id = self.current_device
        
        if device_id in self.memory_pools:
            return self.memory_pools[device_id].allocate(size)
        else:
            raise ValueError(f"Device {device_id} not available")
    
    def free_memory(self, ptr: int, device_id: Optional[int] = None):
        """Free GPU memory"""
        if device_id is None:
            device_id = self.current_device
        
        if device_id in self.memory_pools:
            self.memory_pools[device_id].free(ptr)
    
    def copy_to_device(self, host_data: np.ndarray, device_ptr: int) -> float:
        """Copy data from host to device, returns transfer time in ms"""
        if not CUDA_AVAILABLE:
            return 0.0
            
        start_time = time.perf_counter()
        
        if CUDA_AVAILABLE:
            cuda.memcpy_htod(device_ptr, host_data)
        
        transfer_time = (time.perf_counter() - start_time) * 1000
        
        if self.enable_profiling:
            key = f"H2D_{host_data.nbytes}"
            if key not in self.memory_transfer_times:
                self.memory_transfer_times[key] = []
            self.memory_transfer_times[key].append(transfer_time)
        
        return transfer_time
    
    def copy_from_device(self, device_ptr: int, host_data: np.ndarray) -> float:
        """Copy data from device to host, returns transfer time in ms"""
        if not CUDA_AVAILABLE:
            return 0.0
            
        start_time = time.perf_counter()
        
        if CUDA_AVAILABLE:
            cuda.memcpy_dtoh(host_data, device_ptr)
        
        transfer_time = (time.perf_counter() - start_time) * 1000
        
        if self.enable_profiling:
            key = f"D2H_{host_data.nbytes}"
            if key not in self.memory_transfer_times:
                self.memory_transfer_times[key] = []
            self.memory_transfer_times[key].append(transfer_time)
        
        return transfer_time
    
    def compile_kernel(self, kernel_name: str, source_code: str, 
                      include_dirs: List[str] = None, 
                      optimize: bool = True) -> CudaKernel:
        """Compile a CUDA kernel from source code"""
        if not CUDA_AVAILABLE:
            # Return dummy kernel for fallback mode
            return CudaKernel(
                name=kernel_name,
                source_code=source_code,
                compiled_function=None,
                block_size=(256, 1, 1),
                grid_size=(1, 1, 1),
                shared_memory_size=0,
                registers_per_thread=32
            )
        
        start_time = time.perf_counter()
        
        # Check cache first
        cache_key = f"{kernel_name}_{hash(source_code)}"
        cache_file = self.kernel_cache_dir / f"{cache_key}.ptx"
        
        options = ['-std=c++17']
        if optimize:
            options.extend(['-O3', '-use_fast_math'])
        if include_dirs:
            options.extend([f'-I{dir}' for dir in include_dirs])
        
        try:
            if cache_file.exists():
                # Load from cache
                with open(cache_file, 'r') as f:
                    ptx_code = f.read()
                module = cuda.module_from_buffer(ptx_code.encode())
            else:
                # Compile and cache
                module = cuda_compiler.SourceModule(source_code, options=options)
                if hasattr(module, 'get_code'):
                    ptx_code = module.get_code()
                    with open(cache_file, 'w') as f:
                        f.write(ptx_code.decode())
            
            compiled_function = module.get_function(kernel_name)
            
            # Auto-determine optimal block size
            block_size = self._determine_optimal_block_size(compiled_function, kernel_name)
            
            kernel = CudaKernel(
                name=kernel_name,
                source_code=source_code,
                compiled_function=compiled_function,
                block_size=block_size,
                grid_size=(1, 1, 1),  # Will be set during launch
                shared_memory_size=0,
                registers_per_thread=32  # Approximate
            )
            
            self.compiled_kernels[kernel_name] = kernel
            
            compile_time = (time.perf_counter() - start_time) * 1000
            if self.enable_profiling:
                self.compilation_times[kernel_name] = compile_time
            
            print(f"Compiled CUDA kernel '{kernel_name}' in {compile_time:.2f}ms")
            return kernel
            
        except Exception as e:
            print(f"Failed to compile CUDA kernel '{kernel_name}': {e}")
            raise
    
    def _determine_optimal_block_size(self, function: Any, kernel_name: str) -> Tuple[int, int, int]:
        """Determine optimal block size for a kernel"""
        device = self.get_device_info()
        max_threads = device.max_threads_per_block
        warp_size = device.warp_size
        
        # Start with a reasonable default
        threads_per_block = min(256, max_threads)
        
        # Ensure it's a multiple of warp size
        threads_per_block = (threads_per_block // warp_size) * warp_size
        
        return (threads_per_block, 1, 1)
    
    def launch_kernel(self, kernel_name: str, grid_size: Tuple[int, int, int], 
                     *args, **kwargs) -> float:
        """Launch a CUDA kernel, returns execution time in ms"""
        if kernel_name not in self.compiled_kernels:
            raise ValueError(f"Kernel '{kernel_name}' not compiled")
        
        kernel = self.compiled_kernels[kernel_name]
        
        if not CUDA_AVAILABLE:
            # Simulate kernel execution for fallback
            time.sleep(0.001)  # Simulate 1ms execution
            return 1.0
        
        start_time = time.perf_counter()
        
        try:
            # Update grid size
            kernel.grid_size = grid_size
            
            # Launch kernel
            kernel.compiled_function(
                *args,
                block=kernel.block_size,
                grid=grid_size,
                **kwargs
            )
            
            # Synchronize
            cuda.Context.synchronize()
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            if self.enable_profiling:
                if kernel_name not in self.kernel_execution_times:
                    self.kernel_execution_times[kernel_name] = []
                self.kernel_execution_times[kernel_name].append(execution_time)
            
            return execution_time
            
        except Exception as e:
            print(f"Failed to launch kernel '{kernel_name}': {e}")
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            'cuda_available': CUDA_AVAILABLE,
            'devices': len(self.devices),
            'current_device': self.current_device,
            'kernel_execution_times': {},
            'memory_transfer_times': {},
            'compilation_times': self.compilation_times,
            'memory_pool_stats': {}
        }
        
        # Average kernel execution times
        for kernel, times in self.kernel_execution_times.items():
            stats['kernel_execution_times'][kernel] = {
                'average_ms': np.mean(times),
                'min_ms': np.min(times),
                'max_ms': np.max(times),
                'total_executions': len(times)
            }
        
        # Average memory transfer times
        for transfer, times in self.memory_transfer_times.items():
            stats['memory_transfer_times'][transfer] = {
                'average_ms': np.mean(times),
                'min_ms': np.min(times),
                'max_ms': np.max(times),
                'total_transfers': len(times)
            }
        
        # Memory pool statistics
        for device_id, pool in self.memory_pools.items():
            stats['memory_pool_stats'][device_id] = pool.get_stats()
        
        return stats
    
    def export_performance_report(self, filename: str = "cuda_performance_report.json"):
        """Export detailed performance report"""
        stats = self.get_performance_stats()
        
        # Add device information
        stats['device_info'] = []
        for device in self.devices:
            device_dict = {
                'device_id': device.device_id,
                'name': device.name,
                'compute_capability': device.compute_capability,
                'memory_total_gb': device.memory_total / (1024**3),
                'multiprocessor_count': device.multiprocessor_count,
                'max_threads_per_block': device.max_threads_per_block,
                'warp_size': device.warp_size,
                'clock_rate_mhz': device.clock_rate / 1000,
                'memory_clock_rate_mhz': device.memory_clock_rate / 1000,
                'memory_bus_width': device.memory_bus_width
            }
            stats['device_info'].append(device_dict)
        
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"CUDA performance report exported to {filename}")

def get_cuda_backend(enable_profiling: bool = True) -> CudaBackend:
    """Get global CUDA backend instance"""
    if not hasattr(get_cuda_backend, '_instance'):
        get_cuda_backend._instance = CudaBackend(enable_profiling=enable_profiling)
    return get_cuda_backend._instance

# Example usage
if __name__ == "__main__":
    # Initialize CUDA backend
    backend = get_cuda_backend(enable_profiling=True)
    
    # Print device information
    if backend.devices:
        for i, device in enumerate(backend.devices):
            print(f"Device {i}: {device.name}")
            print(f"  Memory: {device.memory_total / (1024**3):.1f} GB")
            print(f"  Compute: {device.compute_capability}")
    
    # Simple kernel test
    kernel_source = """
    __global__ void vector_add(float* a, float* b, float* c, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = a[idx] + b[idx];
        }
    }
    """
    
    try:
        kernel = backend.compile_kernel("vector_add", kernel_source)
        print("Successfully compiled test kernel!")
        
        # Export performance report
        backend.export_performance_report()
        
    except Exception as e:
        print(f"Kernel compilation failed: {e}")
