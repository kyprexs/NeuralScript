"""
Native SIMD Code Generation for Matrix Operations
================================================

Generates optimized SIMD assembly instructions for matrix multiplication
and other linear algebra operations, targeting specific hardware capabilities.

Features:
- AVX/AVX2/AVX512 instruction generation
- SSE optimization for older hardware
- Cache-blocking code generation
- Automatic vectorization patterns
- Performance-optimized memory access patterns
"""

import platform
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass

try:
    import llvmlite.binding as llvm
    import llvmlite.ir as ll
    HAS_LLVMLITE = True
except ImportError:
    HAS_LLVMLITE = False
    # Mock for when llvmlite is unavailable
    class ll:
        class Module: pass
        class Builder: pass
        class IntType: pass
        class FloatType: pass
        class VectorType: pass

from ..ir.ir_nodes import IRMatMul, IRValue, IRTensorType
from ..simd.simd_core import SIMDProcessor, SIMDInstructionSet, DataType


class SIMDCodegenStrategy(Enum):
    """Code generation strategies for different SIMD capabilities"""
    SCALAR = "scalar"
    SSE = "sse"
    AVX = "avx"
    AVX2 = "avx2"
    AVX512 = "avx512"


@dataclass
class MatrixDimensions:
    """Matrix dimensions for code generation"""
    m: int  # Rows in A and C
    k: int  # Columns in A, rows in B
    n: int  # Columns in B and C


class SIMDCodeGenerator:
    """
    Generates native SIMD code for matrix operations.
    
    Produces highly optimized LLVM IR that compiles to efficient
    vectorized assembly code for maximum performance.
    """
    
    def __init__(self, simd_processor: Optional[SIMDProcessor] = None):
        if not HAS_LLVMLITE:
            raise ImportError("llvmlite required for SIMD code generation")
        
        self.simd_processor = simd_processor or SIMDProcessor()
        self.strategy = self._determine_strategy()
        
        # SIMD vector types for different instruction sets
        self.vector_types = {
            SIMDCodegenStrategy.SSE: ll.VectorType(ll.FloatType(), 4),      # 128-bit = 4 floats
            SIMDCodegenStrategy.AVX: ll.VectorType(ll.FloatType(), 8),      # 256-bit = 8 floats
            SIMDCodegenStrategy.AVX2: ll.VectorType(ll.FloatType(), 8),     # 256-bit = 8 floats
            SIMDCodegenStrategy.AVX512: ll.VectorType(ll.FloatType(), 16),  # 512-bit = 16 floats
        }
        
        # Cache line and block size optimizations
        self.cache_line_size = 64  # bytes
        self.l1_cache_size = 32 * 1024  # 32KB typical L1 data cache
        self.optimal_block_size = self._calculate_optimal_block_size()
        
    def _determine_strategy(self) -> SIMDCodegenStrategy:
        """Determine the best SIMD code generation strategy for this hardware"""
        instruction_sets = self.simd_processor.get_available_instruction_sets()
        
        if "AVX512F" in instruction_sets:
            return SIMDCodegenStrategy.AVX512
        elif "AVX2" in instruction_sets:
            return SIMDCodegenStrategy.AVX2
        elif "AVX" in instruction_sets:
            return SIMDCodegenStrategy.AVX
        elif any(s.startswith("SSE") for s in instruction_sets):
            return SIMDCodegenStrategy.SSE
        else:
            return SIMDCodegenStrategy.SCALAR
    
    def _calculate_optimal_block_size(self) -> int:
        """Calculate optimal block size for cache efficiency"""
        # For matrix multiplication, we want 3 blocks (A, B, C) to fit in L1 cache
        # Each float32 = 4 bytes, so L1 can hold ~8K floats
        # For 3 square blocks: 3 * block_size^2 * 4 <= L1_size
        # block_size <= sqrt(L1_size / 12)
        
        max_elements_per_block = self.l1_cache_size // (3 * 4)  # 3 blocks, 4 bytes per float
        block_size = int(max_elements_per_block ** 0.5)
        
        # Round down to nearest multiple of vector width for optimal SIMD
        vector_width = self._get_vector_width()
        block_size = (block_size // vector_width) * vector_width
        
        return max(32, min(block_size, 256))  # Clamp to reasonable range
    
    def _get_vector_width(self) -> int:
        """Get vector width in float32 elements for current strategy"""
        width_map = {
            SIMDCodegenStrategy.SCALAR: 1,
            SIMDCodegenStrategy.SSE: 4,
            SIMDCodegenStrategy.AVX: 8,
            SIMDCodegenStrategy.AVX2: 8,
            SIMDCodegenStrategy.AVX512: 16,
        }
        return width_map.get(self.strategy, 1)
    
    def generate_matrix_multiply(self, builder: 'll.Builder', 
                               matmul_ir: IRMatMul,
                               a_ptr: Any, b_ptr: Any, c_ptr: Any,
                               dims: MatrixDimensions) -> None:
        """
        Generate optimized SIMD code for matrix multiplication.
        
        Args:
            builder: LLVM IR builder
            matmul_ir: IR node containing optimization hints
            a_ptr, b_ptr, c_ptr: LLVM pointers to matrix data
            dims: Matrix dimensions
        """
        
        # Check if we should use advanced algorithms
        use_strassen = matmul_ir.get_metadata('use_strassen', False)
        block_size = matmul_ir.get_metadata('block_size', self.optimal_block_size)
        
        if use_strassen and dims.m == dims.k == dims.n and dims.m >= 512:
            self._generate_strassen_multiply(builder, a_ptr, b_ptr, c_ptr, dims.m)
        elif dims.m * dims.k * dims.n >= 1000000:  # Large matrices
            self._generate_blocked_multiply(builder, a_ptr, b_ptr, c_ptr, dims, block_size)
        else:
            self._generate_simple_multiply(builder, a_ptr, b_ptr, c_ptr, dims)
    
    def _generate_simple_multiply(self, builder: 'll.Builder',
                                a_ptr: Any, b_ptr: Any, c_ptr: Any,
                                dims: MatrixDimensions) -> None:
        """Generate simple matrix multiplication for small matrices"""
        
        i32 = ll.IntType(32)
        f32 = ll.FloatType()
        
        # Create loop variables
        i_var = builder.alloca(i32, name="i")
        j_var = builder.alloca(i32, name="j") 
        k_var = builder.alloca(i32, name="k")
        
        # Initialize loop counters
        builder.store(ll.Constant(i32, 0), i_var)
        
        # Create basic blocks for nested loops
        i_loop_header = builder.function.append_basic_block("i_loop_header")
        i_loop_body = builder.function.append_basic_block("i_loop_body")
        j_loop_header = builder.function.append_basic_block("j_loop_header")
        j_loop_body = builder.function.append_basic_block("j_loop_body")
        k_loop_header = builder.function.append_basic_block("k_loop_header")
        k_loop_body = builder.function.append_basic_block("k_loop_body")
        k_loop_end = builder.function.append_basic_block("k_loop_end")
        j_loop_end = builder.function.append_basic_block("j_loop_end")
        i_loop_end = builder.function.append_basic_block("i_loop_end")
        exit_block = builder.function.append_basic_block("exit")
        
        # Generate the triple nested loop structure
        builder.branch(i_loop_header)
        
        # i loop header
        builder.position_at_start(i_loop_header)
        i_val = builder.load(i_var)
        i_cond = builder.icmp_signed('<', i_val, ll.Constant(i32, dims.m))
        builder.cbranch(i_cond, i_loop_body, exit_block)
        
        # i loop body - initialize j
        builder.position_at_start(i_loop_body)
        builder.store(ll.Constant(i32, 0), j_var)
        builder.branch(j_loop_header)
        
        # j loop header  
        builder.position_at_start(j_loop_header)
        j_val = builder.load(j_var)
        j_cond = builder.icmp_signed('<', j_val, ll.Constant(i32, dims.n))
        builder.cbranch(j_cond, j_loop_body, j_loop_end)
        
        # j loop body - initialize k and accumulator
        builder.position_at_start(j_loop_body)
        builder.store(ll.Constant(i32, 0), k_var)
        
        # Initialize C[i][j] = 0 (accumulator)
        i_val = builder.load(i_var)
        j_val = builder.load(j_var)
        c_idx = builder.add(builder.mul(i_val, ll.Constant(i32, dims.n)), j_val)
        c_ptr_ij = builder.gep(c_ptr, [c_idx])
        builder.store(ll.Constant(f32, 0.0), c_ptr_ij)
        
        builder.branch(k_loop_header)
        
        # k loop header
        builder.position_at_start(k_loop_header)
        k_val = builder.load(k_var)
        k_cond = builder.icmp_signed('<', k_val, ll.Constant(i32, dims.k))
        builder.cbranch(k_cond, k_loop_body, k_loop_end)
        
        # k loop body - the actual computation
        builder.position_at_start(k_loop_body)
        i_val = builder.load(i_var)
        j_val = builder.load(j_var)
        k_val = builder.load(k_var)
        
        # Load A[i][k]
        a_idx = builder.add(builder.mul(i_val, ll.Constant(i32, dims.k)), k_val)
        a_ptr_ik = builder.gep(a_ptr, [a_idx])
        a_val = builder.load(a_ptr_ik)
        
        # Load B[k][j]
        b_idx = builder.add(builder.mul(k_val, ll.Constant(i32, dims.n)), j_val)
        b_ptr_kj = builder.gep(b_ptr, [b_idx])
        b_val = builder.load(b_ptr_kj)
        
        # Multiply A[i][k] * B[k][j]
        product = builder.fmul(a_val, b_val)
        
        # Load current C[i][j]
        c_idx = builder.add(builder.mul(i_val, ll.Constant(i32, dims.n)), j_val)
        c_ptr_ij = builder.gep(c_ptr, [c_idx])
        c_current = builder.load(c_ptr_ij)
        
        # Add to accumulator: C[i][j] += A[i][k] * B[k][j]
        c_new = builder.fadd(c_current, product)
        builder.store(c_new, c_ptr_ij)
        
        # Increment k
        k_next = builder.add(k_val, ll.Constant(i32, 1))
        builder.store(k_next, k_var)
        builder.branch(k_loop_header)
        
        # End k loop
        builder.position_at_start(k_loop_end)
        j_val = builder.load(j_var)
        j_next = builder.add(j_val, ll.Constant(i32, 1))
        builder.store(j_next, j_var)
        builder.branch(j_loop_header)
        
        # End j loop
        builder.position_at_start(j_loop_end)
        i_val = builder.load(i_var)
        i_next = builder.add(i_val, ll.Constant(i32, 1))
        builder.store(i_next, i_var)
        builder.branch(i_loop_header)
        
        # Exit
        builder.position_at_start(exit_block)
    
    def _generate_blocked_multiply(self, builder: 'll.Builder',
                                 a_ptr: Any, b_ptr: Any, c_ptr: Any,
                                 dims: MatrixDimensions, block_size: int) -> None:
        """Generate cache-blocked matrix multiplication"""
        
        i32 = ll.IntType(32)
        f32 = ll.FloatType()
        
        # Create loop variables for blocking
        ii_var = builder.alloca(i32, name="ii")  # Block row
        jj_var = builder.alloca(i32, name="jj")  # Block column
        kk_var = builder.alloca(i32, name="kk")  # Block depth
        
        # Initialize C to zero first
        self._generate_matrix_zero(builder, c_ptr, dims.m, dims.n)
        
        # Triple-blocked loop structure
        builder.store(ll.Constant(i32, 0), ii_var)
        
        # Create block loop structure
        ii_loop = builder.function.append_basic_block("ii_loop")
        ii_body = builder.function.append_basic_block("ii_body") 
        jj_loop = builder.function.append_basic_block("jj_loop")
        jj_body = builder.function.append_basic_block("jj_body")
        kk_loop = builder.function.append_basic_block("kk_loop")
        kk_body = builder.function.append_basic_block("kk_body")
        inner_multiply = builder.function.append_basic_block("inner_multiply")
        kk_end = builder.function.append_basic_block("kk_end")
        jj_end = builder.function.append_basic_block("jj_end")
        ii_end = builder.function.append_basic_block("ii_end")
        
        builder.branch(ii_loop)
        
        # ii (block row) loop
        builder.position_at_start(ii_loop)
        ii_val = builder.load(ii_var)
        ii_cond = builder.icmp_signed('<', ii_val, ll.Constant(i32, dims.m))
        builder.cbranch(ii_cond, ii_body, ii_end)
        
        builder.position_at_start(ii_body)
        builder.store(ll.Constant(i32, 0), jj_var)
        builder.branch(jj_loop)
        
        # jj (block column) loop
        builder.position_at_start(jj_loop)
        jj_val = builder.load(jj_var)
        jj_cond = builder.icmp_signed('<', jj_val, ll.Constant(i32, dims.n))
        builder.cbranch(jj_cond, jj_body, jj_end)
        
        builder.position_at_start(jj_body)
        builder.store(ll.Constant(i32, 0), kk_var)
        builder.branch(kk_loop)
        
        # kk (block depth) loop
        builder.position_at_start(kk_loop)
        kk_val = builder.load(kk_var)
        kk_cond = builder.icmp_signed('<', kk_val, ll.Constant(i32, dims.k))
        builder.cbranch(kk_cond, kk_body, kk_end)
        
        # Generate the inner block multiplication
        builder.position_at_start(kk_body)
        ii_val = builder.load(ii_var)
        jj_val = builder.load(jj_var)
        kk_val = builder.load(kk_var)
        
        # Calculate block boundaries
        ii_end_val = builder.select(
            builder.icmp_signed('<', builder.add(ii_val, ll.Constant(i32, block_size)), ll.Constant(i32, dims.m)),
            builder.add(ii_val, ll.Constant(i32, block_size)),
            ll.Constant(i32, dims.m)
        )
        
        jj_end_val = builder.select(
            builder.icmp_signed('<', builder.add(jj_val, ll.Constant(i32, block_size)), ll.Constant(i32, dims.n)),
            builder.add(jj_val, ll.Constant(i32, block_size)),
            ll.Constant(i32, dims.n)
        )
        
        kk_end_val = builder.select(
            builder.icmp_signed('<', builder.add(kk_val, ll.Constant(i32, block_size)), ll.Constant(i32, dims.k)),
            builder.add(kk_val, ll.Constant(i32, block_size)),
            ll.Constant(i32, dims.k)
        )
        
        # Generate vectorized inner block multiplication
        self._generate_vectorized_block_multiply(
            builder, a_ptr, b_ptr, c_ptr, dims,
            ii_val, jj_val, kk_val,
            ii_end_val, jj_end_val, kk_end_val
        )
        
        # Increment kk by block_size
        kk_next = builder.add(kk_val, ll.Constant(i32, block_size))
        builder.store(kk_next, kk_var)
        builder.branch(kk_loop)
        
        # End loops
        builder.position_at_start(kk_end)
        jj_val = builder.load(jj_var)
        jj_next = builder.add(jj_val, ll.Constant(i32, block_size))
        builder.store(jj_next, jj_var)
        builder.branch(jj_loop)
        
        builder.position_at_start(jj_end)
        ii_val = builder.load(ii_var)
        ii_next = builder.add(ii_val, ll.Constant(i32, block_size))
        builder.store(ii_next, ii_var)
        builder.branch(ii_loop)
        
        builder.position_at_start(ii_end)
        # Matrix multiply complete
    
    def _generate_vectorized_block_multiply(self, builder: 'll.Builder',
                                          a_ptr: Any, b_ptr: Any, c_ptr: Any,
                                          dims: MatrixDimensions,
                                          i_start: Any, j_start: Any, k_start: Any,
                                          i_end: Any, j_end: Any, k_end: Any) -> None:
        """Generate vectorized inner loop for block multiplication"""
        
        i32 = ll.IntType(32)
        f32 = ll.FloatType()
        vector_width = self._get_vector_width()
        
        if self.strategy == SIMDCodegenStrategy.SCALAR:
            # Fallback to scalar implementation
            self._generate_scalar_block_multiply(
                builder, a_ptr, b_ptr, c_ptr, dims,
                i_start, j_start, k_start, i_end, j_end, k_end
            )
            return
        
        # Use SIMD vectors
        vector_type = self.vector_types[self.strategy]
        
        # Create loop variables for the inner loops
        i_var = builder.alloca(i32, name="block_i")
        j_var = builder.alloca(i32, name="block_j")
        k_var = builder.alloca(i32, name="block_k")
        
        builder.store(i_start, i_var)
        
        # Create loop blocks
        i_loop = builder.function.append_basic_block("vector_i_loop")
        i_body = builder.function.append_basic_block("vector_i_body")
        j_loop = builder.function.append_basic_block("vector_j_loop")
        j_body = builder.function.append_basic_block("vector_j_body")
        k_loop = builder.function.append_basic_block("vector_k_loop")
        k_body = builder.function.append_basic_block("vector_k_body")
        k_exit = builder.function.append_basic_block("vector_k_exit")
        j_exit = builder.function.append_basic_block("vector_j_exit")
        i_exit = builder.function.append_basic_block("vector_i_exit")
        
        builder.branch(i_loop)
        
        # i loop
        builder.position_at_start(i_loop)
        i_val = builder.load(i_var)
        i_cond = builder.icmp_signed('<', i_val, i_end)
        builder.cbranch(i_cond, i_body, i_exit)
        
        builder.position_at_start(i_body)
        builder.store(j_start, j_var)
        builder.branch(j_loop)
        
        # j loop (vectorized)
        builder.position_at_start(j_loop)
        j_val = builder.load(j_var)
        # Process vector_width elements at a time
        j_vec_end = builder.add(j_val, ll.Constant(i32, vector_width))
        j_cond = builder.icmp_signed('<=', j_vec_end, j_end)
        builder.cbranch(j_cond, j_body, j_exit)
        
        builder.position_at_start(j_body)
        builder.store(k_start, k_var)
        
        # Load vector from C[i][j:j+vector_width]
        i_val = builder.load(i_var)
        j_val = builder.load(j_var)
        c_base_idx = builder.add(builder.mul(i_val, ll.Constant(i32, dims.n)), j_val)
        c_vec_ptr = builder.gep(c_ptr, [c_base_idx])
        c_vec_ptr_cast = builder.bitcast(c_vec_ptr, ll.PointerType(vector_type))
        c_vec = builder.load(c_vec_ptr_cast)
        
        builder.branch(k_loop)
        
        # k loop (vectorized inner product)
        builder.position_at_start(k_loop)
        k_val = builder.load(k_var)
        k_cond = builder.icmp_signed('<', k_val, k_end)
        builder.cbranch(k_cond, k_body, k_exit)
        
        builder.position_at_start(k_body)
        i_val = builder.load(i_var)
        j_val = builder.load(j_var)
        k_val = builder.load(k_var)
        
        # Load A[i][k] (broadcast to vector)
        a_idx = builder.add(builder.mul(i_val, ll.Constant(i32, dims.k)), k_val)
        a_ptr_ik = builder.gep(a_ptr, [a_idx])
        a_scalar = builder.load(a_ptr_ik)
        
        # Broadcast A[i][k] to vector
        a_vec = ll.Constant(vector_type, ll.Undefined)
        for lane in range(vector_width):
            a_vec = builder.insert_element(a_vec, a_scalar, ll.Constant(i32, lane))
        
        # Load vector B[k][j:j+vector_width]
        b_base_idx = builder.add(builder.mul(k_val, ll.Constant(i32, dims.n)), j_val)
        b_vec_ptr = builder.gep(b_ptr, [b_base_idx])
        b_vec_ptr_cast = builder.bitcast(b_vec_ptr, ll.PointerType(vector_type))
        b_vec = builder.load(b_vec_ptr_cast)
        
        # Vectorized FMA: C_vec += A_vec * B_vec
        product_vec = builder.fmul(a_vec, b_vec)
        c_vec = builder.fadd(c_vec, product_vec)
        
        # Increment k
        k_next = builder.add(k_val, ll.Constant(i32, 1))
        builder.store(k_next, k_var)
        builder.branch(k_loop)
        
        # Store result vector back to C
        builder.position_at_start(k_exit)
        i_val = builder.load(i_var)
        j_val = builder.load(j_var)
        c_base_idx = builder.add(builder.mul(i_val, ll.Constant(i32, dims.n)), j_val)
        c_vec_ptr = builder.gep(c_ptr, [c_base_idx])
        c_vec_ptr_cast = builder.bitcast(c_vec_ptr, ll.PointerType(vector_type))
        builder.store(c_vec, c_vec_ptr_cast)
        
        # Increment j by vector width
        j_val = builder.load(j_var)
        j_next = builder.add(j_val, ll.Constant(i32, vector_width))
        builder.store(j_next, j_var)
        builder.branch(j_loop)
        
        # End loops
        builder.position_at_start(j_exit)
        i_val = builder.load(i_var)
        i_next = builder.add(i_val, ll.Constant(i32, 1))
        builder.store(i_next, i_var)
        builder.branch(i_loop)
        
        builder.position_at_start(i_exit)
    
    def _generate_scalar_block_multiply(self, builder: 'll.Builder',
                                      a_ptr: Any, b_ptr: Any, c_ptr: Any,
                                      dims: MatrixDimensions,
                                      i_start: Any, j_start: Any, k_start: Any,
                                      i_end: Any, j_end: Any, k_end: Any) -> None:
        """Fallback scalar implementation for block multiplication"""
        # Similar to _generate_simple_multiply but with block bounds
        # Implementation details would be similar to the simple version
        # but using the block start/end values instead of 0/dims
        pass
    
    def _generate_matrix_zero(self, builder: 'll.Builder', 
                            matrix_ptr: Any, rows: int, cols: int) -> None:
        """Generate code to zero-initialize a matrix"""
        i32 = ll.IntType(32)
        f32 = ll.FloatType()
        
        # Use vectorized zeroing if possible
        if self.strategy != SIMDCodegenStrategy.SCALAR:
            vector_type = self.vector_types[self.strategy]
            vector_width = self._get_vector_width()
            zero_vector = ll.Constant(vector_type, [0.0] * vector_width)
            
            total_elements = rows * cols
            vector_elements = (total_elements // vector_width) * vector_width
            
            # Vectorized zero loop
            i_var = builder.alloca(i32, name="zero_i")
            builder.store(ll.Constant(i32, 0), i_var)
            
            loop_header = builder.function.append_basic_block("zero_loop")
            loop_body = builder.function.append_basic_block("zero_body")
            loop_exit = builder.function.append_basic_block("zero_exit")
            
            builder.branch(loop_header)
            
            builder.position_at_start(loop_header)
            i_val = builder.load(i_var)
            i_cond = builder.icmp_signed('<', i_val, ll.Constant(i32, vector_elements))
            builder.cbranch(i_cond, loop_body, loop_exit)
            
            builder.position_at_start(loop_body)
            i_val = builder.load(i_var)
            vec_ptr = builder.gep(matrix_ptr, [i_val])
            vec_ptr_cast = builder.bitcast(vec_ptr, ll.PointerType(vector_type))
            builder.store(zero_vector, vec_ptr_cast)
            
            i_next = builder.add(i_val, ll.Constant(i32, vector_width))
            builder.store(i_next, i_var)
            builder.branch(loop_header)
            
            builder.position_at_start(loop_exit)
            
            # Handle remaining scalar elements
            for i in range(vector_elements, total_elements):
                scalar_ptr = builder.gep(matrix_ptr, [ll.Constant(i32, i)])
                builder.store(ll.Constant(f32, 0.0), scalar_ptr)
        else:
            # Scalar zeroing
            for i in range(rows * cols):
                scalar_ptr = builder.gep(matrix_ptr, [ll.Constant(i32, i)])
                builder.store(ll.Constant(f32, 0.0), scalar_ptr)
    
    def _generate_strassen_multiply(self, builder: 'll.Builder',
                                  a_ptr: Any, b_ptr: Any, c_ptr: Any,
                                  n: int) -> None:
        """Generate Strassen algorithm for large square matrices"""
        # For now, fall back to blocked multiply
        # Strassen requires complex recursive structure that's better
        # implemented in higher-level optimizations
        dims = MatrixDimensions(n, n, n)
        self._generate_blocked_multiply(builder, a_ptr, b_ptr, c_ptr, dims, self.optimal_block_size)
    
    def get_required_intrinsics(self) -> List[str]:
        """Get list of required LLVM intrinsics for current strategy"""
        if self.strategy == SIMDCodegenStrategy.AVX512:
            return [
                "llvm.x86.avx512.mask.vfmadd.ps.512",
                "llvm.x86.avx512.mask.load.ps.512",
                "llvm.x86.avx512.mask.store.ps.512"
            ]
        elif self.strategy == SIMDCodegenStrategy.AVX2:
            return [
                "llvm.x86.fma.vfmadd.ps.256",
                "llvm.x86.avx.loadu.ps.256",
                "llvm.x86.avx.storeu.ps.256"
            ]
        elif self.strategy == SIMDCodegenStrategy.AVX:
            return [
                "llvm.x86.avx.loadu.ps.256",
                "llvm.x86.avx.storeu.ps.256"
            ]
        elif self.strategy == SIMDCodegenStrategy.SSE:
            return [
                "llvm.x86.sse.loadu.ps",
                "llvm.x86.sse.storeu.ps"
            ]
        else:
            return []
    
    def estimate_performance(self, dims: MatrixDimensions) -> Dict[str, float]:
        """Estimate performance characteristics for given matrix dimensions"""
        
        # Calculate theoretical FLOPS
        total_flops = 2 * dims.m * dims.n * dims.k
        
        # Estimate cycles based on hardware capabilities
        vector_width = self._get_vector_width()
        theoretical_vectors_per_cycle = 1  # Conservative estimate
        
        # Assume 3.0 GHz CPU (typical modern processor)
        cpu_frequency_ghz = 3.0
        cycles_per_second = cpu_frequency_ghz * 1e9
        
        # Calculate theoretical performance
        vectorized_ops_per_cycle = theoretical_vectors_per_cycle * vector_width
        theoretical_gflops = (vectorized_ops_per_cycle * cycles_per_second) / 1e9
        
        # Account for memory bandwidth limitations and cache efficiency
        cache_efficiency = 0.7 if dims.m * dims.k * 4 < self.l1_cache_size else 0.3
        memory_efficiency = 0.8  # Account for memory latency
        
        practical_gflops = theoretical_gflops * cache_efficiency * memory_efficiency
        estimated_time_ms = (total_flops / practical_gflops / 1e6)
        
        return {
            'estimated_time_ms': estimated_time_ms,
            'estimated_gflops': practical_gflops,
            'theoretical_gflops': theoretical_gflops,
            'vector_width': vector_width,
            'cache_efficiency': cache_efficiency,
            'strategy': self.strategy.value
        }


class MatrixCodegenOptimizer:
    """
    High-level optimizer for matrix operations.
    
    Analyzes matrix operations and selects optimal code generation strategies.
    """
    
    def __init__(self, simd_codegen: SIMDCodeGenerator):
        self.simd_codegen = simd_codegen
        self.optimization_history: Dict[Tuple[int, int, int], Dict] = {}
    
    def optimize_matrix_multiply(self, matmul_ir: IRMatMul) -> Dict[str, Any]:
        """
        Analyze matrix multiply IR and determine optimal code generation strategy.
        
        Returns optimization decisions and performance estimates.
        """
        
        # Extract matrix dimensions from IR tensor types
        left_shape = matmul_ir.left.type.shape if hasattr(matmul_ir.left.type, 'shape') else [1000, 1000]
        right_shape = matmul_ir.right.type.shape if hasattr(matmul_ir.right.type, 'shape') else [1000, 1000]
        
        dims = MatrixDimensions(
            m=left_shape[0],
            k=left_shape[1], 
            n=right_shape[1]
        )
        
        # Check if we have historical data for this size
        dims_key = (dims.m, dims.k, dims.n)
        if dims_key in self.optimization_history:
            return self.optimization_history[dims_key]
        
        # Estimate performance for different strategies
        perf_estimate = self.simd_codegen.estimate_performance(dims)
        
        # Determine optimal strategy
        strategy_decisions = {
            'use_vectorization': dims.m * dims.k * dims.n >= 1000,
            'use_blocking': dims.m * dims.k * dims.n >= 100000,
            'use_strassen': (dims.m == dims.k == dims.n and 
                           dims.m >= 1024 and 
                           (dims.m & (dims.m - 1)) == 0),  # Power of 2
            'optimal_block_size': self.simd_codegen.optimal_block_size,
            'parallel_worthwhile': dims.m * dims.k * dims.n >= 1000000,
        }
        
        optimization_plan = {
            'dimensions': dims,
            'performance_estimate': perf_estimate,
            'strategy_decisions': strategy_decisions,
            'recommended_approach': self._select_approach(strategy_decisions),
            'expected_speedup': self._calculate_expected_speedup(dims, strategy_decisions)
        }
        
        # Cache the decision
        self.optimization_history[dims_key] = optimization_plan
        
        return optimization_plan
    
    def _select_approach(self, decisions: Dict[str, Any]) -> str:
        """Select the best overall approach based on strategy decisions"""
        if decisions['use_strassen']:
            return "strassen_blocked_vectorized"
        elif decisions['use_blocking'] and decisions['use_vectorization']:
            return "blocked_vectorized"
        elif decisions['use_vectorization']:
            return "simple_vectorized"
        else:
            return "scalar"
    
    def _calculate_expected_speedup(self, dims: MatrixDimensions, decisions: Dict[str, Any]) -> float:
        """Calculate expected speedup from optimizations"""
        speedup = 1.0
        
        if decisions['use_vectorization']:
            speedup *= self.simd_codegen._get_vector_width()  # Vector speedup
        
        if decisions['use_blocking']:
            speedup *= 1.5  # Cache locality improvement
        
        if decisions['use_strassen'] and dims.m >= 2048:
            # Strassen theoretical speedup: n^2.807 vs n^3
            speedup *= (dims.m ** 0.193)  # Approximate improvement factor
        
        return min(speedup, 20.0)  # Cap at 20x to be realistic
