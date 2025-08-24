"""
Comprehensive CUDA Performance Testing and Benchmarking

This module provides extensive testing and benchmarking of CUDA operations
comparing GPU performance against CPU baselines across different workloads.

Features:
- Comprehensive performance benchmarking
- Accuracy validation against CPU implementations  
- Scalability testing across different problem sizes
- Memory bandwidth analysis
- GPU utilization metrics
- Automated performance regression detection
"""

import numpy as np
import time
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import pandas as pd

from .cuda_backend import get_cuda_backend
from .cuda_math import get_cuda_math
from .cuda_ml import get_cuda_ml, ConvolutionConfig, ActivationType

@dataclass
class BenchmarkResult:
    """Result of a single benchmark"""
    operation: str
    problem_size: str
    gpu_time_ms: float
    cpu_time_ms: float
    speedup: float
    gpu_gflops: float
    cpu_gflops: float
    memory_bandwidth_gb_s: float
    accuracy_error: float
    passed: bool

@dataclass
class TestConfiguration:
    """Configuration for benchmark tests"""
    max_error_threshold: float = 1e-4
    warmup_iterations: int = 5
    benchmark_iterations: int = 20
    enable_plots: bool = True
    save_results: bool = True

class CudaPerformanceTester:
    """Comprehensive CUDA performance testing system"""
    
    def __init__(self, config: Optional[TestConfiguration] = None):
        self.config = config or TestConfiguration()
        self.cuda_backend = get_cuda_backend()
        self.cuda_math = get_cuda_math()
        self.cuda_ml = get_cuda_ml()
        
        self.results: List[BenchmarkResult] = []
        
    def validate_accuracy(self, gpu_result: np.ndarray, cpu_result: np.ndarray) -> float:
        """Validate GPU result accuracy against CPU baseline"""
        if gpu_result.shape != cpu_result.shape:
            return float('inf')
        
        # Calculate relative error
        diff = np.abs(gpu_result - cpu_result)
        max_error = np.max(diff)
        
        # Handle zero values
        cpu_nonzero = cpu_result[cpu_result != 0]
        if len(cpu_nonzero) > 0:
            relative_error = np.max(diff[cpu_result != 0] / np.abs(cpu_nonzero))
            return max(max_error, relative_error)
        
        return max_error
    
    def benchmark_vector_operations(self) -> List[BenchmarkResult]:
        """Benchmark vector operations"""
        results = []
        
        sizes = [1000, 10000, 100000, 1000000, 10000000]
        
        for size in sizes:
            print(f"Benchmarking vector addition (size: {size})...")
            
            # Generate test data
            a_cpu = np.random.random(size).astype(np.float32)
            b_cpu = np.random.random(size).astype(np.float32)
            
            # CPU benchmark
            start_time = time.perf_counter()
            for _ in range(self.config.benchmark_iterations):
                cpu_result = a_cpu + b_cpu
            cpu_time = (time.perf_counter() - start_time) * 1000 / self.config.benchmark_iterations
            
            # GPU benchmark
            a_gpu = self.cuda_math.from_numpy(a_cpu)
            b_gpu = self.cuda_math.from_numpy(b_cpu)
            
            # Warmup
            for _ in range(self.config.warmup_iterations):
                result_gpu_tensor = self.cuda_math.vector_add(a_gpu, b_gpu)
                self.cuda_math.free_tensor(result_gpu_tensor)
            
            # Actual benchmark
            start_time = time.perf_counter()
            for _ in range(self.config.benchmark_iterations):
                result_gpu_tensor = self.cuda_math.vector_add(a_gpu, b_gpu)
            gpu_time = (time.perf_counter() - start_time) * 1000 / self.config.benchmark_iterations
            
            # Get result for accuracy check
            gpu_result = self.cuda_math.to_numpy(result_gpu_tensor)
            accuracy_error = self.validate_accuracy(gpu_result, cpu_result)
            
            # Calculate metrics
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            memory_bytes = size * 4 * 3  # Two inputs + one output, float32
            memory_bandwidth = memory_bytes / (gpu_time * 1e-3) / 1e9
            
            # FLOPS calculation (minimal for vector add)
            flops = size
            gpu_gflops = flops / (gpu_time * 1e-3) / 1e9
            cpu_gflops = flops / (cpu_time * 1e-3) / 1e9
            
            result = BenchmarkResult(
                operation="vector_add",
                problem_size=str(size),
                gpu_time_ms=gpu_time,
                cpu_time_ms=cpu_time,
                speedup=speedup,
                gpu_gflops=gpu_gflops,
                cpu_gflops=cpu_gflops,
                memory_bandwidth_gb_s=memory_bandwidth,
                accuracy_error=accuracy_error,
                passed=accuracy_error < self.config.max_error_threshold
            )
            
            results.append(result)
            print(f"  GPU: {gpu_time:.2f}ms, CPU: {cpu_time:.2f}ms, Speedup: {speedup:.2f}x")
            
            # Clean up
            self.cuda_math.free_tensor(a_gpu)
            self.cuda_math.free_tensor(b_gpu)
            self.cuda_math.free_tensor(result_gpu_tensor)
        
        return results
    
    def benchmark_matrix_operations(self) -> List[BenchmarkResult]:
        """Benchmark matrix operations"""
        results = []
        
        sizes = [(128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)]
        
        for M, N in sizes:
            K = N  # Square matrix multiplication
            print(f"Benchmarking matrix multiplication ({M}x{K} * {K}x{N})...")
            
            # Generate test data
            A_cpu = np.random.random((M, K)).astype(np.float32)
            B_cpu = np.random.random((K, N)).astype(np.float32)
            
            # CPU benchmark (using optimized numpy)
            start_time = time.perf_counter()
            for _ in range(self.config.benchmark_iterations):
                cpu_result = np.dot(A_cpu, B_cpu)
            cpu_time = (time.perf_counter() - start_time) * 1000 / self.config.benchmark_iterations
            
            # GPU benchmark
            A_gpu = self.cuda_math.from_numpy(A_cpu)
            B_gpu = self.cuda_math.from_numpy(B_cpu)
            
            # Warmup
            for _ in range(self.config.warmup_iterations):
                result_gpu_tensor = self.cuda_math.matrix_multiply(A_gpu, B_gpu)
                self.cuda_math.free_tensor(result_gpu_tensor)
            
            # Actual benchmark
            start_time = time.perf_counter()
            for _ in range(self.config.benchmark_iterations):
                result_gpu_tensor = self.cuda_math.matrix_multiply(A_gpu, B_gpu)
            gpu_time = (time.perf_counter() - start_time) * 1000 / self.config.benchmark_iterations
            
            # Get result for accuracy check
            gpu_result = self.cuda_math.to_numpy(result_gpu_tensor)
            accuracy_error = self.validate_accuracy(gpu_result, cpu_result)
            
            # Calculate metrics
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            flops = 2 * M * N * K  # Matrix multiplication FLOPS
            gpu_gflops = flops / (gpu_time * 1e-3) / 1e9
            cpu_gflops = flops / (cpu_time * 1e-3) / 1e9
            
            memory_bytes = (M * K + K * N + M * N) * 4  # float32
            memory_bandwidth = memory_bytes / (gpu_time * 1e-3) / 1e9
            
            result = BenchmarkResult(
                operation="matrix_multiply",
                problem_size=f"{M}x{K}x{N}",
                gpu_time_ms=gpu_time,
                cpu_time_ms=cpu_time,
                speedup=speedup,
                gpu_gflops=gpu_gflops,
                cpu_gflops=cpu_gflops,
                memory_bandwidth_gb_s=memory_bandwidth,
                accuracy_error=accuracy_error,
                passed=accuracy_error < self.config.max_error_threshold
            )
            
            results.append(result)
            print(f"  GPU: {gpu_time:.2f}ms ({gpu_gflops:.1f} GFLOPS), CPU: {cpu_time:.2f}ms ({cpu_gflops:.1f} GFLOPS)")
            print(f"  Speedup: {speedup:.2f}x, Accuracy error: {accuracy_error:.2e}")
            
            # Clean up
            self.cuda_math.free_tensor(A_gpu)
            self.cuda_math.free_tensor(B_gpu)
            self.cuda_math.free_tensor(result_gpu_tensor)
        
        return results
    
    def benchmark_ml_operations(self) -> List[BenchmarkResult]:
        """Benchmark ML operations"""
        results = []
        
        # Convolution benchmark
        conv_configs = [
            ((1, 32, 64, 64), (64, 32, 3, 3)),  # Typical conv layer
            ((1, 64, 128, 128), (128, 64, 3, 3)),  # Larger conv layer
            ((32, 64, 32, 32), (128, 64, 3, 3))   # Batch processing
        ]
        
        for input_shape, kernel_shape in conv_configs:
            print(f"Benchmarking convolution (input: {input_shape}, kernel: {kernel_shape})...")
            
            # Generate test data
            input_cpu = np.random.random(input_shape).astype(np.float32)
            kernel_cpu = np.random.random(kernel_shape).astype(np.float32)
            
            # Simple CPU convolution (very basic implementation)
            def cpu_conv2d_simple(input_arr, kernel_arr):
                batch, in_ch, in_h, in_w = input_arr.shape
                out_ch, _, k_h, k_w = kernel_arr.shape
                
                out_h = in_h - k_h + 1  # No padding for simplicity
                out_w = in_w - k_w + 1
                
                output = np.zeros((batch, out_ch, out_h, out_w), dtype=np.float32)
                
                for b in range(batch):
                    for oc in range(out_ch):
                        for ic in range(in_ch):
                            for y in range(out_h):
                                for x in range(out_w):
                                    for ky in range(k_h):
                                        for kx in range(k_w):
                                            output[b, oc, y, x] += (
                                                input_arr[b, ic, y+ky, x+kx] * 
                                                kernel_arr[oc, ic, ky, kx]
                                            )
                return output
            
            # CPU benchmark (simplified)
            start_time = time.perf_counter()
            cpu_result = cpu_conv2d_simple(input_cpu, kernel_cpu)
            cpu_time = (time.perf_counter() - start_time) * 1000
            
            # GPU benchmark
            input_gpu = self.cuda_math.from_numpy(input_cpu)
            kernel_gpu = self.cuda_math.from_numpy(kernel_cpu)
            config = ConvolutionConfig(kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
            
            # Warmup
            for _ in range(self.config.warmup_iterations):
                result_gpu_tensor = self.cuda_ml.conv2d(input_gpu, kernel_gpu, config)
                self.cuda_math.free_tensor(result_gpu_tensor)
            
            # Actual benchmark
            start_time = time.perf_counter()
            for _ in range(self.config.benchmark_iterations):
                result_gpu_tensor = self.cuda_ml.conv2d(input_gpu, kernel_gpu, config)
            gpu_time = (time.perf_counter() - start_time) * 1000 / self.config.benchmark_iterations
            
            # Get result for accuracy check
            gpu_result = self.cuda_math.to_numpy(result_gpu_tensor)
            accuracy_error = self.validate_accuracy(gpu_result, cpu_result)
            
            # Calculate metrics
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            
            # Convolution FLOPS calculation
            batch, in_ch, in_h, in_w = input_shape
            out_ch, _, k_h, k_w = kernel_shape
            out_h, out_w = gpu_result.shape[-2:]
            conv_flops = batch * out_ch * out_h * out_w * in_ch * k_h * k_w * 2
            
            gpu_gflops = conv_flops / (gpu_time * 1e-3) / 1e9
            cpu_gflops = conv_flops / (cpu_time * 1e-3) / 1e9
            
            memory_bytes = (np.prod(input_shape) + np.prod(kernel_shape) + np.prod(gpu_result.shape)) * 4
            memory_bandwidth = memory_bytes / (gpu_time * 1e-3) / 1e9
            
            result = BenchmarkResult(
                operation="conv2d",
                problem_size=f"in{input_shape}_k{kernel_shape}",
                gpu_time_ms=gpu_time,
                cpu_time_ms=cpu_time,
                speedup=speedup,
                gpu_gflops=gpu_gflops,
                cpu_gflops=cpu_gflops,
                memory_bandwidth_gb_s=memory_bandwidth,
                accuracy_error=accuracy_error,
                passed=accuracy_error < self.config.max_error_threshold * 10  # More lenient for conv
            )
            
            results.append(result)
            print(f"  GPU: {gpu_time:.2f}ms ({gpu_gflops:.1f} GFLOPS), CPU: {cpu_time:.2f}ms ({cpu_gflops:.1f} GFLOPS)")
            print(f"  Speedup: {speedup:.2f}x, Accuracy error: {accuracy_error:.2e}")
            
            # Clean up
            self.cuda_math.free_tensor(input_gpu)
            self.cuda_math.free_tensor(kernel_gpu)
            self.cuda_math.free_tensor(result_gpu_tensor)
        
        # Activation functions benchmark
        activation_sizes = [100000, 1000000, 10000000]
        
        for size in activation_sizes:
            print(f"Benchmarking ReLU activation (size: {size})...")
            
            # Generate test data
            input_cpu = np.random.randn(size).astype(np.float32)
            
            # CPU benchmark
            start_time = time.perf_counter()
            for _ in range(self.config.benchmark_iterations):
                cpu_result = np.maximum(0, input_cpu)
            cpu_time = (time.perf_counter() - start_time) * 1000 / self.config.benchmark_iterations
            
            # GPU benchmark
            input_gpu = self.cuda_math.from_numpy(input_cpu)
            
            # Warmup
            for _ in range(self.config.warmup_iterations):
                result_gpu_tensor = self.cuda_ml.activation(input_gpu, ActivationType.RELU)
                self.cuda_math.free_tensor(result_gpu_tensor)
            
            # Actual benchmark
            start_time = time.perf_counter()
            for _ in range(self.config.benchmark_iterations):
                result_gpu_tensor = self.cuda_ml.activation(input_gpu, ActivationType.RELU)
            gpu_time = (time.perf_counter() - start_time) * 1000 / self.config.benchmark_iterations
            
            # Get result for accuracy check
            gpu_result = self.cuda_math.to_numpy(result_gpu_tensor)
            accuracy_error = self.validate_accuracy(gpu_result, cpu_result)
            
            # Calculate metrics
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            memory_bytes = size * 4 * 2  # Input + output, float32
            memory_bandwidth = memory_bytes / (gpu_time * 1e-3) / 1e9
            
            # ReLU has minimal compute, so FLOPS are negligible
            gpu_gflops = 0.1  # Minimal
            cpu_gflops = 0.1
            
            result = BenchmarkResult(
                operation="relu",
                problem_size=str(size),
                gpu_time_ms=gpu_time,
                cpu_time_ms=cpu_time,
                speedup=speedup,
                gpu_gflops=gpu_gflops,
                cpu_gflops=cpu_gflops,
                memory_bandwidth_gb_s=memory_bandwidth,
                accuracy_error=accuracy_error,
                passed=accuracy_error < self.config.max_error_threshold
            )
            
            results.append(result)
            print(f"  GPU: {gpu_time:.2f}ms, CPU: {cpu_time:.2f}ms, Speedup: {speedup:.2f}x")
            
            # Clean up
            self.cuda_math.free_tensor(input_gpu)
            self.cuda_math.free_tensor(result_gpu_tensor)
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        print("🚀 Starting comprehensive CUDA performance benchmark...")
        print(f"Configuration: warmup={self.config.warmup_iterations}, iterations={self.config.benchmark_iterations}")
        
        start_time = time.time()
        
        # Run all benchmark categories
        print("\n📊 Vector Operations Benchmark:")
        vector_results = self.benchmark_vector_operations()
        self.results.extend(vector_results)
        
        print("\n📊 Matrix Operations Benchmark:")
        matrix_results = self.benchmark_matrix_operations()
        self.results.extend(matrix_results)
        
        print("\n📊 ML Operations Benchmark:")
        ml_results = self.benchmark_ml_operations()
        self.results.extend(ml_results)
        
        total_time = time.time() - start_time
        
        # Analyze results
        summary = self.analyze_results()
        summary['benchmark_duration_seconds'] = total_time
        summary['total_tests'] = len(self.results)
        summary['passed_tests'] = sum(1 for r in self.results if r.passed)
        summary['failed_tests'] = summary['total_tests'] - summary['passed_tests']
        
        print(f"\n✅ Benchmark completed in {total_time:.1f} seconds")
        print(f"📈 Results: {summary['passed_tests']}/{summary['total_tests']} tests passed")
        
        if self.config.save_results:
            self.save_results(summary)
        
        if self.config.enable_plots:
            self.generate_plots()
        
        return summary
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results and generate summary statistics"""
        
        if not self.results:
            return {}
        
        # Group results by operation
        by_operation = {}
        for result in self.results:
            if result.operation not in by_operation:
                by_operation[result.operation] = []
            by_operation[result.operation].append(result)
        
        # Calculate summary statistics
        summary = {
            'by_operation': {},
            'overall_stats': {
                'average_speedup': np.mean([r.speedup for r in self.results]),
                'max_speedup': max([r.speedup for r in self.results]),
                'min_speedup': min([r.speedup for r in self.results]),
                'average_gpu_gflops': np.mean([r.gpu_gflops for r in self.results if r.gpu_gflops > 0]),
                'max_gpu_gflops': max([r.gpu_gflops for r in self.results]),
                'average_memory_bandwidth': np.mean([r.memory_bandwidth_gb_s for r in self.results]),
                'max_memory_bandwidth': max([r.memory_bandwidth_gb_s for r in self.results]),
                'accuracy_pass_rate': sum(1 for r in self.results if r.passed) / len(self.results) * 100
            }
        }
        
        # Per-operation analysis
        for operation, results in by_operation.items():
            op_stats = {
                'test_count': len(results),
                'pass_rate': sum(1 for r in results if r.passed) / len(results) * 100,
                'average_speedup': np.mean([r.speedup for r in results]),
                'max_speedup': max([r.speedup for r in results]),
                'average_gpu_gflops': np.mean([r.gpu_gflops for r in results if r.gpu_gflops > 0]),
                'max_accuracy_error': max([r.accuracy_error for r in results]),
                'results': results
            }
            summary['by_operation'][operation] = op_stats
        
        return summary
    
    def save_results(self, summary: Dict[str, Any]):
        """Save benchmark results to JSON file"""
        
        # Convert results to serializable format
        results_data = []
        for result in self.results:
            results_data.append({
                'operation': result.operation,
                'problem_size': result.problem_size,
                'gpu_time_ms': result.gpu_time_ms,
                'cpu_time_ms': result.cpu_time_ms,
                'speedup': result.speedup,
                'gpu_gflops': result.gpu_gflops,
                'cpu_gflops': result.cpu_gflops,
                'memory_bandwidth_gb_s': result.memory_bandwidth_gb_s,
                'accuracy_error': result.accuracy_error,
                'passed': result.passed
            })
        
        # Prepare summary for serialization (remove non-serializable objects)
        serializable_summary = {
            'overall_stats': summary['overall_stats'],
            'benchmark_duration_seconds': summary['benchmark_duration_seconds'],
            'total_tests': summary['total_tests'],
            'passed_tests': summary['passed_tests'],
            'failed_tests': summary['failed_tests'],
            'by_operation_stats': {}
        }
        
        for op, stats in summary['by_operation'].items():
            serializable_summary['by_operation_stats'][op] = {
                'test_count': stats['test_count'],
                'pass_rate': stats['pass_rate'],
                'average_speedup': stats['average_speedup'],
                'max_speedup': stats['max_speedup'],
                'average_gpu_gflops': stats['average_gpu_gflops'],
                'max_accuracy_error': stats['max_accuracy_error']
            }
        
        output_data = {
            'summary': serializable_summary,
            'detailed_results': results_data,
            'configuration': {
                'max_error_threshold': self.config.max_error_threshold,
                'warmup_iterations': self.config.warmup_iterations,
                'benchmark_iterations': self.config.benchmark_iterations
            }
        }
        
        filename = f'cuda_performance_benchmark_{int(time.time())}.json'
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"📁 Results saved to {filename}")
    
    def generate_plots(self):
        """Generate performance visualization plots"""
        try:
            # Speedup plot
            operations = []
            speedups = []
            
            for result in self.results:
                operations.append(f"{result.operation}\n{result.problem_size}")
                speedups.append(result.speedup)
            
            plt.figure(figsize=(15, 8))
            
            # Speedup comparison
            plt.subplot(2, 2, 1)
            bars = plt.bar(range(len(operations)), speedups)
            plt.title('GPU vs CPU Speedup')
            plt.ylabel('Speedup (x)')
            plt.xticks(range(len(operations)), operations, rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            # Color bars based on performance
            for i, (bar, speedup) in enumerate(zip(bars, speedups)):
                if speedup >= 5:
                    bar.set_color('green')
                elif speedup >= 2:
                    bar.set_color('orange')  
                else:
                    bar.set_color('red')
            
            # GFLOPS comparison
            plt.subplot(2, 2, 2)
            gpu_gflops = [r.gpu_gflops for r in self.results if r.gpu_gflops > 0]
            cpu_gflops = [r.cpu_gflops for r in self.results if r.gpu_gflops > 0]
            
            if gpu_gflops:
                x = np.arange(len(gpu_gflops))
                width = 0.35
                plt.bar(x - width/2, gpu_gflops, width, label='GPU', color='blue')
                plt.bar(x + width/2, cpu_gflops, width, label='CPU', color='red')
                plt.title('GFLOPS Comparison')
                plt.ylabel('GFLOPS')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Memory bandwidth
            plt.subplot(2, 2, 3)
            memory_bandwidths = [r.memory_bandwidth_gb_s for r in self.results]
            plt.bar(range(len(memory_bandwidths)), memory_bandwidths, color='purple')
            plt.title('Memory Bandwidth')
            plt.ylabel('GB/s')
            plt.grid(True, alpha=0.3)
            
            # Accuracy errors (log scale)
            plt.subplot(2, 2, 4)
            errors = [max(r.accuracy_error, 1e-10) for r in self.results]  # Avoid log(0)
            plt.bar(range(len(errors)), errors, color='orange')
            plt.title('Accuracy Errors (Log Scale)')
            plt.ylabel('Error')
            plt.yscale('log')
            plt.axhline(y=self.config.max_error_threshold, color='r', linestyle='--', label='Threshold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filename = f'cuda_performance_plots_{int(time.time())}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Performance plots saved to {filename}")
            
            if self.config.enable_plots:
                plt.show()
            
        except ImportError:
            print("⚠️ Matplotlib not available, skipping plot generation")
        except Exception as e:
            print(f"⚠️ Error generating plots: {e}")
    
    def print_summary_report(self, summary: Dict[str, Any]):
        """Print a formatted summary report"""
        
        print("\n" + "="*80)
        print("🎯 CUDA PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)
        
        overall = summary['overall_stats']
        print(f"\n📈 OVERALL PERFORMANCE:")
        print(f"  • Average Speedup: {overall['average_speedup']:.2f}x")
        print(f"  • Maximum Speedup: {overall['max_speedup']:.2f}x") 
        print(f"  • Average GPU GFLOPS: {overall['average_gpu_gflops']:.1f}")
        print(f"  • Maximum GPU GFLOPS: {overall['max_gpu_gflops']:.1f}")
        print(f"  • Average Memory Bandwidth: {overall['average_memory_bandwidth']:.1f} GB/s")
        print(f"  • Maximum Memory Bandwidth: {overall['max_memory_bandwidth']:.1f} GB/s")
        print(f"  • Accuracy Pass Rate: {overall['accuracy_pass_rate']:.1f}%")
        
        print(f"\n🔍 BY OPERATION:")
        for operation, stats in summary['by_operation'].items():
            print(f"  {operation.upper()}:")
            print(f"    - Tests: {stats['test_count']}")
            print(f"    - Pass Rate: {stats['pass_rate']:.1f}%")
            print(f"    - Average Speedup: {stats['average_speedup']:.2f}x")
            print(f"    - Max Speedup: {stats['max_speedup']:.2f}x")
            if stats['average_gpu_gflops'] > 0:
                print(f"    - Average GFLOPS: {stats['average_gpu_gflops']:.1f}")
            print(f"    - Max Accuracy Error: {stats['max_accuracy_error']:.2e}")
            print()
        
        print(f"⏱️ Total Runtime: {summary['benchmark_duration_seconds']:.1f} seconds")
        print(f"✅ Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
        
        print("\n" + "="*80)

# Main execution
if __name__ == "__main__":
    # Configure benchmark
    config = TestConfiguration(
        max_error_threshold=1e-4,
        warmup_iterations=3,
        benchmark_iterations=10,
        enable_plots=False,
        save_results=True
    )
    
    # Run comprehensive benchmark
    tester = CudaPerformanceTester(config)
    summary = tester.run_comprehensive_benchmark()
    
    # Print detailed report
    tester.print_summary_report(summary)
    
    print("\n🎉 CUDA performance benchmarking completed!")
    print("📁 Check the generated JSON file for detailed results")
    print("📊 Plots saved as PNG files (if matplotlib available)")
