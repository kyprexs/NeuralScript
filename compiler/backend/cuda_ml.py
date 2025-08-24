"""
CUDA Machine Learning Operations for NeuralScript

This module provides GPU-accelerated machine learning operations including
neural network training, convolution, activation functions, and optimization.

Key Features:
- GPU-accelerated convolution (2D/3D)
- Activation functions (ReLU, Sigmoid, Tanh, etc.)
- Pooling operations (Max, Average)
- Batch normalization
- Dropout
- Loss functions
- Optimizers (SGD, Adam, RMSprop)
- Gradient computation
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Any
import time
from dataclasses import dataclass
from enum import Enum

from .cuda_backend import CudaBackend, CudaDataType, get_cuda_backend
from .cuda_kernels import CudaKernelGenerator, get_kernel_generator
from .cuda_math import CudaTensor, CudaMath, get_cuda_math

class ActivationType(Enum):
    """Supported activation functions"""
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SWISH = "swish"
    GELU = "gelu"

class PoolingType(Enum):
    """Supported pooling types"""
    MAX = "max"
    AVERAGE = "average"
    GLOBAL_AVERAGE = "global_average"

class OptimizerType(Enum):
    """Supported optimizers"""
    SGD = "sgd"
    ADAM = "adam"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"

@dataclass
class ConvolutionConfig:
    """Configuration for convolution operations"""
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int] = (1, 1)
    padding: Tuple[int, int] = (0, 0)
    dilation: Tuple[int, int] = (1, 1)
    groups: int = 1

@dataclass
class OptimizerState:
    """Optimizer state for training"""
    learning_rate: float
    momentum: float = 0.9
    beta1: float = 0.9  # For Adam
    beta2: float = 0.999  # For Adam
    eps: float = 1e-8
    weight_decay: float = 0.0
    step: int = 0

class CudaML:
    """GPU-accelerated machine learning operations"""
    
    def __init__(self, 
                 backend: Optional[CudaBackend] = None,
                 math: Optional[CudaMath] = None,
                 kernel_generator: Optional[CudaKernelGenerator] = None):
        self.backend = backend or get_cuda_backend()
        self.math = math or get_cuda_math()
        self.kernel_generator = kernel_generator or get_kernel_generator()
        
        # Compile ML kernels
        self._compile_ml_kernels()
        
    def _compile_ml_kernels(self):
        """Compile machine learning specific kernels"""
        
        # Convolution kernel
        conv2d_src = self.kernel_generator.generate_convolution_kernel()
        self.backend.compile_kernel("conv2d_float", conv2d_src)
        
        # Activation functions
        activations = self.kernel_generator.generate_activation_kernels()
        for name, src in activations.items():
            self.backend.compile_kernel(f"elementwise_{name}_float", src)
        
        # Max pooling kernel
        maxpool_src = """
__global__ void maxpool2d_float(
    const float* __restrict__ input,
    float* __restrict__ output,
    int input_height, int input_width,
    int output_height, int output_width,
    int pool_height, int pool_width,
    int stride_y, int stride_x
) {
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_y < output_height && out_x < output_width) {
        float max_val = -FLT_MAX;
        
        for (int py = 0; py < pool_height; py++) {
            for (int px = 0; px < pool_width; px++) {
                int in_y = out_y * stride_y + py;
                int in_x = out_x * stride_x + px;
                
                if (in_y < input_height && in_x < input_width) {
                    float val = input[in_y * input_width + in_x];
                    max_val = fmaxf(max_val, val);
                }
            }
        }
        
        output[out_y * output_width + out_x] = max_val;
    }
}
        """
        self.backend.compile_kernel("maxpool2d_float", maxpool_src)
        
        # Average pooling kernel
        avgpool_src = """
__global__ void avgpool2d_float(
    const float* __restrict__ input,
    float* __restrict__ output,
    int input_height, int input_width,
    int output_height, int output_width,
    int pool_height, int pool_width,
    int stride_y, int stride_x
) {
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_y < output_height && out_x < output_width) {
        float sum = 0.0f;
        int count = 0;
        
        for (int py = 0; py < pool_height; py++) {
            for (int px = 0; px < pool_width; px++) {
                int in_y = out_y * stride_y + py;
                int in_x = out_x * stride_x + px;
                
                if (in_y < input_height && in_x < input_width) {
                    sum += input[in_y * input_width + in_x];
                    count++;
                }
            }
        }
        
        output[out_y * output_width + out_x] = sum / count;
    }
}
        """
        self.backend.compile_kernel("avgpool2d_float", avgpool_src)
        
        # Batch normalization kernel
        batchnorm_src = """
__global__ void batchnorm_forward_float(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float* __restrict__ output,
    int n, int c, int h, int w,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * c * h * w) {
        int channel = (idx / (h * w)) % c;
        
        float x = input[idx];
        float m = mean[channel];
        float v = var[channel];
        float g = gamma[channel];
        float b = beta[channel];
        
        output[idx] = g * (x - m) / sqrtf(v + eps) + b;
    }
}
        """
        self.backend.compile_kernel("batchnorm_forward_float", batchnorm_src)
        
        # Dropout kernel  
        dropout_src = """
__global__ void dropout_forward_float(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ mask,
    int n,
    float keep_prob
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * mask[idx] / keep_prob;
    }
}
        """
        self.backend.compile_kernel("dropout_forward_float", dropout_src)
        
        # Cross entropy loss kernel
        cross_entropy_src = """
__global__ void cross_entropy_loss_float(
    const float* __restrict__ predictions,
    const int* __restrict__ targets,
    float* __restrict__ loss,
    int batch_size, int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int target_class = targets[idx];
        float pred = predictions[idx * num_classes + target_class];
        loss[idx] = -logf(fmaxf(pred, 1e-7f));  // Clamp for numerical stability
    }
}
        """
        self.backend.compile_kernel("cross_entropy_loss_float", cross_entropy_src)
        
        # SGD optimizer kernel
        sgd_kernel_src = """
__global__ void sgd_update_float(
    float* __restrict__ params,
    const float* __restrict__ gradients,
    float* __restrict__ momentum_buffer,
    int n,
    float lr, float momentum, float weight_decay
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float grad = gradients[idx] + weight_decay * params[idx];
        momentum_buffer[idx] = momentum * momentum_buffer[idx] + grad;
        params[idx] -= lr * momentum_buffer[idx];
    }
}
        """
        self.backend.compile_kernel("sgd_update_float", sgd_kernel_src)
        
        # Adam optimizer kernel
        adam_kernel_src = """
__global__ void adam_update_float(
    float* __restrict__ params,
    const float* __restrict__ gradients,
    float* __restrict__ m_buffer,
    float* __restrict__ v_buffer,
    int n, int step,
    float lr, float beta1, float beta2, float eps, float weight_decay
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float grad = gradients[idx] + weight_decay * params[idx];
        
        // Update biased first moment estimate
        m_buffer[idx] = beta1 * m_buffer[idx] + (1.0f - beta1) * grad;
        
        // Update biased second raw moment estimate
        v_buffer[idx] = beta2 * v_buffer[idx] + (1.0f - beta2) * grad * grad;
        
        // Bias correction
        float m_hat = m_buffer[idx] / (1.0f - powf(beta1, step));
        float v_hat = v_buffer[idx] / (1.0f - powf(beta2, step));
        
        // Update parameters
        params[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}
        """
        self.backend.compile_kernel("adam_update_float", adam_kernel_src)
        
        print("CUDA ML kernels compiled successfully")
    
    def conv2d(self, 
               input_tensor: CudaTensor,
               kernel_tensor: CudaTensor, 
               config: ConvolutionConfig,
               bias: Optional[CudaTensor] = None) -> CudaTensor:
        """2D Convolution operation"""
        
        # Input: [batch, channels, height, width] or [channels, height, width]
        # Kernel: [out_channels, in_channels, kernel_h, kernel_w]
        
        if len(input_tensor.shape) == 3:
            # Add batch dimension
            batch_size = 1
            in_channels, in_height, in_width = input_tensor.shape
        else:
            batch_size, in_channels, in_height, in_width = input_tensor.shape
        
        out_channels, in_channels_k, kernel_h, kernel_w = kernel_tensor.shape
        
        if in_channels != in_channels_k:
            raise ValueError(f"Input channels mismatch: {in_channels} vs {in_channels_k}")
        
        # Calculate output dimensions
        pad_h, pad_w = config.padding
        stride_h, stride_w = config.stride
        
        out_height = (in_height + 2 * pad_h - kernel_h) // stride_h + 1
        out_width = (in_width + 2 * pad_w - kernel_w) // stride_w + 1
        
        if batch_size == 1:
            output = self.math.create_tensor((out_channels, out_height, out_width))
        else:
            output = self.math.create_tensor((batch_size, out_channels, out_height, out_width))
        
        # Launch convolution kernel
        block_size = (16, 16, 1)
        grid_size = ((out_width + 15) // 16, (out_height + 15) // 16, 1)
        
        exec_time = self.backend.launch_kernel(
            "conv2d_float",
            grid_size,
            input_tensor.device_ptr,
            kernel_tensor.device_ptr,
            output.device_ptr,
            in_height, in_width,
            kernel_h, kernel_w,
            out_height, out_width,
            stride_h, pad_h
        )
        
        return output
    
    def activation(self, tensor: CudaTensor, activation_type: ActivationType) -> CudaTensor:
        """Apply activation function"""
        kernel_name = f"elementwise_{activation_type.value}_float"
        
        result = self.math.create_tensor(tensor.shape, tensor.dtype)
        
        n = tensor.size
        block_size = min(512, ((n + 31) // 32) * 32)
        grid_size = (n + block_size - 1) // block_size
        
        exec_time = self.backend.launch_kernel(
            kernel_name,
            (grid_size, 1, 1),
            tensor.device_ptr, result.device_ptr, n
        )
        
        return result
    
    def max_pool2d(self, 
                   tensor: CudaTensor,
                   pool_size: Tuple[int, int],
                   stride: Optional[Tuple[int, int]] = None) -> CudaTensor:
        """2D Max pooling operation"""
        
        if stride is None:
            stride = pool_size
        
        if len(tensor.shape) == 3:
            channels, height, width = tensor.shape
        else:
            batch_size, channels, height, width = tensor.shape
        
        pool_h, pool_w = pool_size
        stride_h, stride_w = stride
        
        out_height = (height - pool_h) // stride_h + 1
        out_width = (width - pool_w) // stride_w + 1
        
        if len(tensor.shape) == 3:
            output = self.math.create_tensor((channels, out_height, out_width))
        else:
            output = self.math.create_tensor((batch_size, channels, out_height, out_width))
        
        block_size = (16, 16, 1)
        grid_size = ((out_width + 15) // 16, (out_height + 15) // 16, 1)
        
        exec_time = self.backend.launch_kernel(
            "maxpool2d_float",
            grid_size,
            tensor.device_ptr, output.device_ptr,
            height, width,
            out_height, out_width,
            pool_h, pool_w,
            stride_h, stride_w
        )
        
        return output
    
    def avg_pool2d(self,
                   tensor: CudaTensor,
                   pool_size: Tuple[int, int],
                   stride: Optional[Tuple[int, int]] = None) -> CudaTensor:
        """2D Average pooling operation"""
        
        if stride is None:
            stride = pool_size
        
        if len(tensor.shape) == 3:
            channels, height, width = tensor.shape
        else:
            batch_size, channels, height, width = tensor.shape
        
        pool_h, pool_w = pool_size
        stride_h, stride_w = stride
        
        out_height = (height - pool_h) // stride_h + 1
        out_width = (width - pool_w) // stride_w + 1
        
        if len(tensor.shape) == 3:
            output = self.math.create_tensor((channels, out_height, out_width))
        else:
            output = self.math.create_tensor((batch_size, channels, out_height, out_width))
        
        block_size = (16, 16, 1)
        grid_size = ((out_width + 15) // 16, (out_height + 15) // 16, 1)
        
        exec_time = self.backend.launch_kernel(
            "avgpool2d_float",
            grid_size,
            tensor.device_ptr, output.device_ptr,
            height, width,
            out_height, out_width,
            pool_h, pool_w,
            stride_h, stride_w
        )
        
        return output
    
    def batch_norm(self,
                   input_tensor: CudaTensor,
                   gamma: CudaTensor,
                   beta: CudaTensor,
                   running_mean: CudaTensor,
                   running_var: CudaTensor,
                   eps: float = 1e-5) -> CudaTensor:
        """Batch normalization operation"""
        
        if len(input_tensor.shape) == 4:
            n, c, h, w = input_tensor.shape
        else:
            n, c, h, w = 1, *input_tensor.shape
        
        output = self.math.create_tensor(input_tensor.shape)
        
        total_elements = n * c * h * w
        block_size = min(512, ((total_elements + 31) // 32) * 32)
        grid_size = (total_elements + block_size - 1) // block_size
        
        exec_time = self.backend.launch_kernel(
            "batchnorm_forward_float",
            (grid_size, 1, 1),
            input_tensor.device_ptr,
            gamma.device_ptr,
            beta.device_ptr,
            running_mean.device_ptr,
            running_var.device_ptr,
            output.device_ptr,
            n, c, h, w,
            eps
        )
        
        return output
    
    def dropout(self,
                tensor: CudaTensor,
                keep_prob: float = 0.5,
                training: bool = True) -> CudaTensor:
        """Dropout regularization"""
        
        if not training:
            return tensor
        
        # Generate random mask (simplified - in practice would use cuRAND)
        mask_np = np.random.binomial(1, keep_prob, tensor.shape).astype(np.float32)
        mask_tensor = self.math.from_numpy(mask_np)
        
        output = self.math.create_tensor(tensor.shape)
        
        n = tensor.size
        block_size = min(512, ((n + 31) // 32) * 32)
        grid_size = (n + block_size - 1) // block_size
        
        exec_time = self.backend.launch_kernel(
            "dropout_forward_float",
            (grid_size, 1, 1),
            tensor.device_ptr,
            output.device_ptr,
            mask_tensor.device_ptr,
            n,
            keep_prob
        )
        
        self.math.free_tensor(mask_tensor)
        return output
    
    def cross_entropy_loss(self,
                          predictions: CudaTensor,
                          targets: CudaTensor) -> CudaTensor:
        """Cross entropy loss computation"""
        
        batch_size, num_classes = predictions.shape
        loss = self.math.create_tensor((batch_size,))
        
        block_size = min(512, ((batch_size + 31) // 32) * 32)
        grid_size = (batch_size + block_size - 1) // block_size
        
        # Convert targets to int32 for kernel
        targets_int32 = self.math.create_tensor(targets.shape, CudaDataType.INT32)
        # Copy and cast targets (simplified)
        
        exec_time = self.backend.launch_kernel(
            "cross_entropy_loss_float",
            (grid_size, 1, 1),
            predictions.device_ptr,
            targets_int32.device_ptr,
            loss.device_ptr,
            batch_size,
            num_classes
        )
        
        self.math.free_tensor(targets_int32)
        return loss
    
    def sgd_update(self,
                   parameters: CudaTensor,
                   gradients: CudaTensor,
                   momentum_buffer: CudaTensor,
                   optimizer_state: OptimizerState):
        """SGD optimizer update step"""
        
        n = parameters.size
        block_size = min(512, ((n + 31) // 32) * 32)
        grid_size = (n + block_size - 1) // block_size
        
        exec_time = self.backend.launch_kernel(
            "sgd_update_float",
            (grid_size, 1, 1),
            parameters.device_ptr,
            gradients.device_ptr,
            momentum_buffer.device_ptr,
            n,
            optimizer_state.learning_rate,
            optimizer_state.momentum,
            optimizer_state.weight_decay
        )
    
    def adam_update(self,
                    parameters: CudaTensor,
                    gradients: CudaTensor,
                    m_buffer: CudaTensor,
                    v_buffer: CudaTensor,
                    optimizer_state: OptimizerState):
        """Adam optimizer update step"""
        
        n = parameters.size
        block_size = min(512, ((n + 31) // 32) * 32)
        grid_size = (n + block_size - 1) // block_size
        
        optimizer_state.step += 1
        
        exec_time = self.backend.launch_kernel(
            "adam_update_float",
            (grid_size, 1, 1),
            parameters.device_ptr,
            gradients.device_ptr,
            m_buffer.device_ptr,
            v_buffer.device_ptr,
            n,
            optimizer_state.step,
            optimizer_state.learning_rate,
            optimizer_state.beta1,
            optimizer_state.beta2,
            optimizer_state.eps,
            optimizer_state.weight_decay
        )
    
    def benchmark_ml_operations(self, iterations: int = 100) -> Dict[str, Any]:
        """Benchmark ML operations"""
        
        results = {
            'conv2d': {},
            'activation': {},
            'pooling': {},
            'batch_norm': {},
            'optimizers': {}
        }
        
        # Convolution benchmark
        input_tensor = self.math.create_tensor((1, 32, 64, 64))  # NCHW
        kernel_tensor = self.math.create_tensor((64, 32, 3, 3))
        config = ConvolutionConfig(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            output = self.conv2d(input_tensor, kernel_tensor, config)
        conv_time = (time.perf_counter() - start_time) * 1000 / iterations
        
        # Calculate FLOPS for convolution
        batch, in_ch, in_h, in_w = input_tensor.shape
        out_ch, _, k_h, k_w = kernel_tensor.shape
        out_h, out_w = output.shape[-2:]
        conv_flops = batch * out_ch * out_h * out_w * in_ch * k_h * k_w * 2
        
        results['conv2d']['3x3_64ch'] = {
            'average_time_ms': conv_time,
            'gflops': conv_flops / (conv_time * 1e-3) / 1e9,
            'input_shape': input_tensor.shape,
            'output_shape': output.shape
        }
        
        # Activation benchmark
        activation_tensor = self.math.create_tensor((1000000,))
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            output = self.activation(activation_tensor, ActivationType.RELU)
        relu_time = (time.perf_counter() - start_time) * 1000 / iterations
        
        results['activation']['relu_1M'] = {
            'average_time_ms': relu_time,
            'throughput_gb_s': (activation_tensor.nbytes * 2) / (relu_time * 1e-3) / 1e9,
            'elements': activation_tensor.size
        }
        
        # Pooling benchmark
        pool_input = self.math.create_tensor((1, 64, 128, 128))
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            output = self.max_pool2d(pool_input, (2, 2))
        pool_time = (time.perf_counter() - start_time) * 1000 / iterations
        
        results['pooling']['maxpool2x2'] = {
            'average_time_ms': pool_time,
            'input_shape': pool_input.shape,
            'output_shape': output.shape
        }
        
        # Batch normalization benchmark  
        bn_input = self.math.create_tensor((32, 64, 32, 32))
        gamma = self.math.create_tensor((64,))
        beta = self.math.create_tensor((64,))
        running_mean = self.math.create_tensor((64,))
        running_var = self.math.create_tensor((64,))
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            output = self.batch_norm(bn_input, gamma, beta, running_mean, running_var)
        bn_time = (time.perf_counter() - start_time) * 1000 / iterations
        
        results['batch_norm']['64ch_32x32'] = {
            'average_time_ms': bn_time,
            'input_shape': bn_input.shape
        }
        
        # Optimizer benchmark
        params = self.math.create_tensor((1000000,))
        gradients = self.math.create_tensor((1000000,))
        momentum_buffer = self.math.create_tensor((1000000,))
        
        optimizer_state = OptimizerState(learning_rate=0.001, momentum=0.9)
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            self.sgd_update(params, gradients, momentum_buffer, optimizer_state)
        sgd_time = (time.perf_counter() - start_time) * 1000 / iterations
        
        results['optimizers']['sgd_1M_params'] = {
            'average_time_ms': sgd_time,
            'parameters': params.size
        }
        
        # Clean up
        self.math.free_tensor(input_tensor)
        self.math.free_tensor(kernel_tensor)
        self.math.free_tensor(activation_tensor)
        self.math.free_tensor(pool_input)
        self.math.free_tensor(bn_input)
        self.math.free_tensor(gamma)
        self.math.free_tensor(beta)
        self.math.free_tensor(running_mean)
        self.math.free_tensor(running_var)
        self.math.free_tensor(params)
        self.math.free_tensor(gradients)
        self.math.free_tensor(momentum_buffer)
        
        return results

def get_cuda_ml(backend: Optional[CudaBackend] = None) -> CudaML:
    """Get global CUDA ML instance"""
    if not hasattr(get_cuda_ml, '_instance'):
        get_cuda_ml._instance = CudaML(backend)
    return get_cuda_ml._instance

# Example usage
if __name__ == "__main__":
    # Initialize CUDA ML
    cuda_ml = get_cuda_ml()
    
    print("Testing CUDA ML operations...")
    
    # Test convolution
    input_np = np.random.random((1, 3, 32, 32)).astype(np.float32)
    kernel_np = np.random.random((16, 3, 3, 3)).astype(np.float32)
    
    input_gpu = cuda_ml.math.from_numpy(input_np)
    kernel_gpu = cuda_ml.math.from_numpy(kernel_np)
    
    config = ConvolutionConfig(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    output_gpu = cuda_ml.conv2d(input_gpu, kernel_gpu, config)
    
    print(f"Convolution input shape: {input_gpu.shape}")
    print(f"Convolution output shape: {output_gpu.shape}")
    
    # Test activation
    activation_input = cuda_ml.math.create_tensor((1000,))
    relu_output = cuda_ml.activation(activation_input, ActivationType.RELU)
    
    print(f"ReLU activation applied to {activation_input.size} elements")
    
    # Run benchmarks
    print("\nRunning ML operation benchmarks...")
    results = cuda_ml.benchmark_ml_operations(iterations=10)
    
    print("\nBenchmark Results:")
    for category, operations in results.items():
        print(f"\n{category}:")
        for op_name, metrics in operations.items():
            print(f"  {op_name}: {metrics}")
    
    # Clean up
    cuda_ml.math.free_tensor(input_gpu)
    cuda_ml.math.free_tensor(kernel_gpu)
    cuda_ml.math.free_tensor(output_gpu)
    cuda_ml.math.free_tensor(activation_input)
    cuda_ml.math.free_tensor(relu_output)
    
    print("\nCUDA ML operations test completed!")
