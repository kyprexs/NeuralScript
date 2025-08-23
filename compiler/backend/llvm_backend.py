"""
LLVM Backend for NeuralScript.

Generates LLVM IR from NeuralScript IR (NS-IR).
Handles code generation, optimization, and target-specific features.

Author: xwest
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
import tempfile
import subprocess
import os
import time

try:
    import llvmlite.binding as llvm
    import llvmlite.ir as ll
    HAS_LLVMLITE = True
except ImportError:
    HAS_LLVMLITE = False
    # Mock classes for when llvmlite is not available
    class ll:
        class Module:
            def __init__(self, name): pass
        class Builder:
            def __init__(self, block): pass
        class Function:
            def __init__(self, module, func_type, name): pass
        class FunctionType:
            def __init__(self, return_type, args): pass
        class IntType:
            def __init__(self, bits): pass
        class FloatType:
            pass
        class DoubleType:
            pass
        class VoidType:
            pass
        class PointerType:
            def __init__(self, pointee): pass
        class Constant:
            def __init__(self, type_, value): pass

from ..ir.ir_nodes import *
from .simd_codegen import SIMDCodeGenerator, MatrixDimensions, DataType, SIMDCodegenStrategy
from ..optimizer.runtime_profiler import RuntimeProfiler, AdaptiveOptimizer, ProfiledExecution, create_runtime_profiler
from ..optimizer.vectorization_pass import VectorizationPass
from ..simd.simd_core import SIMDProcessor


@dataclass
class LLVMGenContext:
    """Context for LLVM code generation."""
    module: Optional['ll.Module'] = None
    builder: Optional['ll.Builder'] = None
    current_function: Optional['ll.Function'] = None
    value_map: Dict[IRValue, Any] = None  # Maps IR values to LLVM values
    block_map: Dict[IRBasicBlock, Any] = None  # Maps IR blocks to LLVM blocks
    
    def __post_init__(self):
        if self.value_map is None:
            self.value_map = {}
        if self.block_map is None:
            self.block_map = {}


class LLVMBackend:
    """
    LLVM backend for NeuralScript.
    
    Converts NeuralScript IR to LLVM IR and handles:
    - Code generation
    - Optimization passes
    - Target-specific features
    - Object code generation
    """
    
    def __init__(self, target_triple: Optional[str] = None, enable_simd: bool = True, 
                 enable_profiling: bool = True):
        """
        Initialize the LLVM backend.
        
        Args:
            target_triple: Target triple (e.g., "x86_64-pc-linux-gnu")
            enable_simd: Enable SIMD code generation
            enable_profiling: Enable runtime profiling
        """
        if not HAS_LLVMLITE:
            raise ImportError("llvmlite is required for LLVM backend. Install with: pip install llvmlite")
        
        # Initialize LLVM
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        
        self.target_triple = target_triple or llvm.get_default_triple()
        self.context = LLVMGenContext()
        
        # SIMD integration
        self.enable_simd = enable_simd
        self.simd_processor = None
        self.simd_codegen = None
        self.auto_vectorizer = None
        
        # Runtime profiling integration
        self.enable_profiling = enable_profiling
        self.runtime_profiler = None
        self.adaptive_optimizer = None
        
        # Initialize components
        self._init_type_mappings()
        self._init_simd_components()
        self._init_profiling_components()
    
    def _init_type_mappings(self):
        """Initialize mappings from IR types to LLVM types."""
        self.type_map = {
            IRDataType.VOID: ll.VoidType(),
            IRDataType.I1: ll.IntType(1),
            IRDataType.I8: ll.IntType(8),
            IRDataType.I16: ll.IntType(16),
            IRDataType.I32: ll.IntType(32),
            IRDataType.I64: ll.IntType(64),
            IRDataType.F16: ll.FloatType(),  # llvmlite might not have f16
            IRDataType.F32: ll.FloatType(),
            IRDataType.F64: ll.DoubleType(),
            IRDataType.PTR: ll.PointerType(ll.IntType(8)),  # Generic pointer
        }
        
        # Complex types as structs
        self.type_map[IRDataType.C32] = ll.LiteralStructType([ll.FloatType(), ll.FloatType()])
        self.type_map[IRDataType.C64] = ll.LiteralStructType([ll.DoubleType(), ll.DoubleType()])
        
        # Add SIMD vector types if enabled
        if self.enable_simd and HAS_LLVMLITE:
            # Add vector types for SIMD operations
            self.type_map["v4f32"] = ll.VectorType(ll.FloatType(), 4)  # SSE
            self.type_map["v8f32"] = ll.VectorType(ll.FloatType(), 8)  # AVX
            self.type_map["v16f32"] = ll.VectorType(ll.FloatType(), 16)  # AVX-512
            self.type_map["v2f64"] = ll.VectorType(ll.DoubleType(), 2)  # SSE2
            self.type_map["v4f64"] = ll.VectorType(ll.DoubleType(), 4)  # AVX
            self.type_map["v8f64"] = ll.VectorType(ll.DoubleType(), 8)  # AVX-512
    
    def generate(self, ir_module: IRModule) -> 'llvm.Module':
        """
        Generate LLVM IR from NeuralScript IR.
        
        Args:
            ir_module: NeuralScript IR module
            
        Returns:
            LLVM module
        """
        # Create LLVM module
        self.context.module = ll.Module(name=ir_module.name)
        self.context.module.triple = self.target_triple
        
        # Generate global declarations first
        self._generate_globals(ir_module)
        
        # Generate function declarations
        self._generate_function_declarations(ir_module)
        
        # Generate function definitions
        self._generate_function_definitions(ir_module)
        
        # Verify module
        try:
            llvm_module = llvm.parse_assembly(str(self.context.module))
            llvm_module.verify()
        except Exception as e:
            print(f"LLVM module verification failed: {e}")
            print("Generated LLVM IR:")
            print(str(self.context.module))
            raise
        
        return llvm_module
    
    def _generate_globals(self, ir_module: IRModule):
        """Generate global variables."""
        for name, global_var in ir_module.globals.items():
            llvm_type = self._convert_ir_type_to_llvm(global_var.type)
            
            # Create global variable
            llvm_global = ll.GlobalVariable(self.context.module, llvm_type, name)
            
            # Set initializer if present
            if hasattr(global_var, 'initializer') and global_var.initializer:
                llvm_global.initializer = self._convert_ir_constant_to_llvm(global_var.initializer)
            else:
                # Default initialization
                llvm_global.initializer = ll.Constant(llvm_type, None)
            
            # Store in value map
            self.context.value_map[global_var] = llvm_global
    
    def _generate_function_declarations(self, ir_module: IRModule):
        """Generate function declarations."""
        for func_name, ir_function in ir_module.functions.items():
            llvm_func_type = self._convert_ir_function_type_to_llvm(ir_function.type)
            
            # Create LLVM function
            llvm_function = ll.Function(self.context.module, llvm_func_type, func_name)
            
            # Set parameter names
            for i, param in enumerate(ir_function.parameters):
                if i < len(llvm_function.args):
                    llvm_function.args[i].name = param.name
            
            # Store in value map
            self.context.value_map[ir_function] = llvm_function
            
            # Map IR parameters to LLVM parameters
            for i, ir_param in enumerate(ir_function.parameters):
                if i < len(llvm_function.args):
                    self.context.value_map[ir_param] = llvm_function.args[i]
    
    def _generate_function_definitions(self, ir_module: IRModule):
        """Generate function definitions."""
        for func_name, ir_function in ir_module.functions.items():
            if ir_function.basic_blocks:  # Only generate definitions for non-external functions
                self._generate_function(ir_function)
    
    def _generate_function(self, ir_function: IRFunction):
        """Generate a single function."""
        llvm_function = self.context.value_map[ir_function]
        self.context.current_function = llvm_function
        
        # Clear block map for this function
        self.context.block_map.clear()
        
        # Create LLVM basic blocks
        for ir_block in ir_function.basic_blocks:
            llvm_block = llvm_function.append_basic_block(ir_block.name)
            self.context.block_map[ir_block] = llvm_block
        
        # Generate instructions for each block
        for ir_block in ir_function.basic_blocks:
            self._generate_basic_block(ir_block)
    
    def _generate_basic_block(self, ir_block: IRBasicBlock):
        """Generate a basic block."""
        llvm_block = self.context.block_map[ir_block]
        self.context.builder = ll.IRBuilder(llvm_block)
        
        # Generate instructions
        for ir_instruction in ir_block.instructions:
            self._generate_instruction(ir_instruction)
    
    def _generate_instruction(self, ir_instruction: IRInstruction):
        """Generate an instruction."""
        if isinstance(ir_instruction, IRBinaryOp):
            self._generate_binary_op(ir_instruction)
        elif isinstance(ir_instruction, IRUnaryOp):
            self._generate_unary_op(ir_instruction)
        elif isinstance(ir_instruction, IRCall):
            self._generate_call(ir_instruction)
        elif isinstance(ir_instruction, IRReturn):
            self._generate_return(ir_instruction)
        elif isinstance(ir_instruction, IRBranch):
            self._generate_branch(ir_instruction)
        elif isinstance(ir_instruction, IRCondBranch):
            self._generate_cond_branch(ir_instruction)
        elif isinstance(ir_instruction, IRPhi):
            self._generate_phi(ir_instruction)
        elif isinstance(ir_instruction, IRAlloca):
            self._generate_alloca(ir_instruction)
        elif isinstance(ir_instruction, IRLoad):
            self._generate_load(ir_instruction)
        elif isinstance(ir_instruction, IRStore):
            self._generate_store(ir_instruction)
        # Add more instruction types as needed
    
    def _generate_binary_op(self, ir_instruction: IRBinaryOp):
        """Generate a binary operation."""
        left = self._get_llvm_value(ir_instruction.left)
        right = self._get_llvm_value(ir_instruction.right)
        
        # Map IR operator to LLVM instruction
        op = ir_instruction.operator
        result = None
        
        if op == "add":
            if self._is_integer_type(ir_instruction.left.type):
                result = self.context.builder.add(left, right)
            else:
                result = self.context.builder.fadd(left, right)
        elif op == "sub":
            if self._is_integer_type(ir_instruction.left.type):
                result = self.context.builder.sub(left, right)
            else:
                result = self.context.builder.fsub(left, right)
        elif op == "mul":
            if self._is_integer_type(ir_instruction.left.type):
                result = self.context.builder.mul(left, right)
            else:
                result = self.context.builder.fmul(left, right)
        elif op == "div":
            if self._is_integer_type(ir_instruction.left.type):
                result = self.context.builder.sdiv(left, right)  # Signed division
            else:
                result = self.context.builder.fdiv(left, right)
        elif op == "rem":
            if self._is_integer_type(ir_instruction.left.type):
                result = self.context.builder.srem(left, right)
            else:
                result = self.context.builder.frem(left, right)
        elif op == "eq":
            if self._is_integer_type(ir_instruction.left.type):
                result = self.context.builder.icmp_signed("==", left, right)
            else:
                result = self.context.builder.fcmp_ordered("==", left, right)
        elif op == "ne":
            if self._is_integer_type(ir_instruction.left.type):
                result = self.context.builder.icmp_signed("!=", left, right)
            else:
                result = self.context.builder.fcmp_ordered("!=", left, right)
        elif op == "lt":
            if self._is_integer_type(ir_instruction.left.type):
                result = self.context.builder.icmp_signed("<", left, right)
            else:
                result = self.context.builder.fcmp_ordered("<", left, right)
        elif op == "le":
            if self._is_integer_type(ir_instruction.left.type):
                result = self.context.builder.icmp_signed("<=", left, right)
            else:
                result = self.context.builder.fcmp_ordered("<=", left, right)
        elif op == "gt":
            if self._is_integer_type(ir_instruction.left.type):
                result = self.context.builder.icmp_signed(">", left, right)
            else:
                result = self.context.builder.fcmp_ordered(">", left, right)
        elif op == "ge":
            if self._is_integer_type(ir_instruction.left.type):
                result = self.context.builder.icmp_signed(">=", left, right)
            else:
                result = self.context.builder.fcmp_ordered(">=", left, right)
        else:
            # Default to add
            if self._is_integer_type(ir_instruction.left.type):
                result = self.context.builder.add(left, right)
            else:
                result = self.context.builder.fadd(left, right)
        
        if ir_instruction.result:
            result.name = ir_instruction.result.name
            self.context.value_map[ir_instruction.result] = result
    
    def _generate_unary_op(self, ir_instruction: IRUnaryOp):
        """Generate a unary operation."""
        operand = self._get_llvm_value(ir_instruction.operand)
        
        op = ir_instruction.operator
        result = None
        
        if op == "neg":
            if self._is_integer_type(ir_instruction.operand.type):
                zero = ll.Constant(operand.type, 0)
                result = self.context.builder.sub(zero, operand)
            else:
                zero = ll.Constant(operand.type, 0.0)
                result = self.context.builder.fsub(zero, operand)
        elif op == "not":
            if self._is_integer_type(ir_instruction.operand.type):
                result = self.context.builder.not_(operand)
            else:
                # For floating point, compare with zero
                zero = ll.Constant(operand.type, 0.0)
                result = self.context.builder.fcmp_ordered("==", operand, zero)
        else:
            # Default to identity
            result = operand
        
        if ir_instruction.result:
            if result != operand:
                result.name = ir_instruction.result.name
            self.context.value_map[ir_instruction.result] = result
    
    def _generate_call(self, ir_instruction: IRCall):
        """Generate a function call."""
        func = self._get_llvm_value(ir_instruction.function)
        args = [self._get_llvm_value(arg) for arg in ir_instruction.args]
        
        result = self.context.builder.call(func, args)
        
        if ir_instruction.result:
            result.name = ir_instruction.result.name
            self.context.value_map[ir_instruction.result] = result
    
    def _generate_return(self, ir_instruction: IRReturn):
        """Generate a return instruction."""
        if ir_instruction.value:
            value = self._get_llvm_value(ir_instruction.value)
            self.context.builder.ret(value)
        else:
            self.context.builder.ret_void()
    
    def _generate_branch(self, ir_instruction: IRBranch):
        """Generate an unconditional branch."""
        target = self.context.block_map[ir_instruction.target]
        self.context.builder.branch(target)
    
    def _generate_cond_branch(self, ir_instruction: IRCondBranch):
        """Generate a conditional branch."""
        condition = self._get_llvm_value(ir_instruction.condition)
        true_block = self.context.block_map[ir_instruction.true_target]
        false_block = self.context.block_map[ir_instruction.false_target]
        
        self.context.builder.cbranch(condition, true_block, false_block)
    
    def _generate_phi(self, ir_instruction: IRPhi):
        """Generate a phi node."""
        llvm_type = self._convert_ir_type_to_llvm(ir_instruction.result.type)
        phi = self.context.builder.phi(llvm_type)
        
        # Add incoming values (will be filled later in a second pass)
        for value, block in ir_instruction.incoming:
            llvm_value = self._get_llvm_value(value)
            llvm_block = self.context.block_map[block]
            phi.add_incoming(llvm_value, llvm_block)
        
        if ir_instruction.result:
            phi.name = ir_instruction.result.name
            self.context.value_map[ir_instruction.result] = phi
    
    def _generate_alloca(self, ir_instruction: IRAlloca):
        """Generate a stack allocation."""
        llvm_type = self._convert_ir_type_to_llvm(ir_instruction.allocated_type)
        alloca = self.context.builder.alloca(llvm_type)
        
        if ir_instruction.result:
            alloca.name = ir_instruction.result.name
            self.context.value_map[ir_instruction.result] = alloca
    
    def _generate_load(self, ir_instruction: IRLoad):
        """Generate a load instruction."""
        pointer = self._get_llvm_value(ir_instruction.pointer)
        load = self.context.builder.load(pointer)
        
        if ir_instruction.result:
            load.name = ir_instruction.result.name
            self.context.value_map[ir_instruction.result] = load
    
    def _generate_store(self, ir_instruction: IRStore):
        """Generate a store instruction."""
        value = self._get_llvm_value(ir_instruction.value)
        pointer = self._get_llvm_value(ir_instruction.pointer)
        self.context.builder.store(value, pointer)
    
    def _convert_ir_type_to_llvm(self, ir_type: IRType) -> Any:
        """Convert an IR type to an LLVM type."""
        if isinstance(ir_type, IRPrimitiveType):
            return self.type_map.get(ir_type.data_type, ll.IntType(32))
        elif isinstance(ir_type, IRTensorType):
            # For now, represent tensors as pointers
            element_type = self._convert_ir_type_to_llvm(ir_type.element_type)
            return ll.PointerType(element_type)
        elif isinstance(ir_type, IRFunctionType):
            return self._convert_ir_function_type_to_llvm(ir_type)
        else:
            # Default to i32
            return ll.IntType(32)
    
    def _convert_ir_function_type_to_llvm(self, ir_func_type: IRFunctionType) -> Any:
        """Convert an IR function type to an LLVM function type."""
        return_type = self._convert_ir_type_to_llvm(ir_func_type.return_type)
        param_types = [self._convert_ir_type_to_llvm(param_type) for param_type in ir_func_type.param_types]
        return ll.FunctionType(return_type, param_types)
    
    def _convert_ir_constant_to_llvm(self, ir_constant: IRConstant) -> Any:
        """Convert an IR constant to an LLVM constant."""
        llvm_type = self._convert_ir_type_to_llvm(ir_constant.type)
        return ll.Constant(llvm_type, ir_constant.value)
    
    def _get_llvm_value(self, ir_value: IRValue) -> Any:
        """Get the LLVM value corresponding to an IR value."""
        if ir_value in self.context.value_map:
            return self.context.value_map[ir_value]
        elif isinstance(ir_value, IRConstant):
            llvm_constant = self._convert_ir_constant_to_llvm(ir_value)
            self.context.value_map[ir_value] = llvm_constant
            return llvm_constant
        else:
            # Return a dummy value - this shouldn't happen in correct IR
            return ll.Constant(ll.IntType(32), 0)
    
    def _is_integer_type(self, ir_type: IRType) -> bool:
        """Check if an IR type is an integer type."""
        if isinstance(ir_type, IRPrimitiveType):
            return ir_type.data_type in {IRDataType.I1, IRDataType.I8, IRDataType.I16, 
                                        IRDataType.I32, IRDataType.I64}
        return False
    
    def compile_to_object(self, llvm_module: 'llvm.Module', output_path: str, 
                         optimization_level: int = 2) -> bool:
        """
        Compile LLVM module to object file.
        
        Args:
            llvm_module: LLVM module to compile
            output_path: Path for output object file
            optimization_level: Optimization level (0-3)
            
        Returns:
            True if compilation succeeded, False otherwise
        """
        try:
            # Create target machine
            target = llvm.Target.from_triple(self.target_triple)
            target_machine = target.create_target_machine(
                opt=optimization_level,
                codemodel='default'
            )
            
            # Generate object code
            with open(output_path, 'wb') as f:
                f.write(target_machine.emit_object(llvm_module))
            
            return True
            
        except Exception as e:
            print(f"Compilation failed: {e}")
            return False
    
    def compile_to_executable(self, llvm_module: 'llvm.Module', output_path: str,
                             optimization_level: int = 2, link_with: List[str] = None) -> bool:
        """
        Compile LLVM module to executable.
        
        Args:
            llvm_module: LLVM module to compile
            output_path: Path for output executable
            optimization_level: Optimization level (0-3)
            link_with: Additional libraries to link with
            
        Returns:
            True if compilation succeeded, False otherwise
        """
        try:
            # First compile to object file
            with tempfile.NamedTemporaryFile(suffix='.o', delete=False) as obj_file:
                obj_path = obj_file.name
                
            if not self.compile_to_object(llvm_module, obj_path, optimization_level):
                return False
            
            # Link to create executable
            link_cmd = ['clang', '-o', output_path, obj_path]
            
            if link_with:
                for lib in link_with:
                    link_cmd.extend(['-l', lib])
            
            result = subprocess.run(link_cmd, capture_output=True, text=True)
            
            # Clean up temporary object file
            os.unlink(obj_path)
            
            if result.returncode != 0:
                print(f"Linking failed: {result.stderr}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Executable compilation failed: {e}")
            return False
    
    def optimize_module(self, llvm_module: 'llvm.Module', optimization_level: int = 2) -> 'llvm.Module':
        """
        Apply optimization passes to the LLVM module.
        
        Args:
            llvm_module: LLVM module to optimize
            optimization_level: Optimization level (0-3)
            
        Returns:
            Optimized LLVM module
        """
        try:
            # Create pass manager
            pm = llvm.create_module_pass_manager()
            
            # Add optimization passes based on level
            if optimization_level >= 1:
                pm.add_instruction_combining_pass()
                pm.add_reassociate_expressions_pass()
                pm.add_gvn_pass()
                pm.add_cfg_simplification_pass()
            
            if optimization_level >= 2:
                pm.add_function_inlining_pass(225)
                pm.add_constant_merge_pass()
                pm.add_dead_arg_elimination_pass()
                pm.add_function_attrs_pass()
                pm.add_global_dce_pass()
            
            if optimization_level >= 3:
                pm.add_loop_vectorize_pass()
                pm.add_slp_vectorize_pass()
            
            # Run optimization passes
            pm.run(llvm_module)
            
            return llvm_module
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return llvm_module
    
    def print_llvm_ir(self, llvm_module: 'llvm.Module') -> str:
        """
        Get the LLVM IR as a string.
        
        Args:
            llvm_module: LLVM module
            
        Returns:
            LLVM IR as string
        """
        return str(llvm_module)
    
    def _init_simd_components(self):
        """Initialize SIMD code generation components."""
        if self.enable_simd:
            try:
                # Initialize SIMD processor
                self.simd_processor = SIMDProcessor()
                
                # Initialize SIMD code generator
                self.simd_codegen = SIMDCodeGenerator()
                
                # Initialize auto-vectorization pass
                self.auto_vectorizer = VectorizationPass()
                
                print(f"✅ SIMD enabled with support for: {', '.join(self.simd_processor.get_available_instruction_sets())}")
                
            except Exception as e:
                print(f"⚠️  SIMD initialization failed: {e}")
                self.enable_simd = False
    
    def _init_profiling_components(self):
        """Initialize runtime profiling components."""
        if self.enable_profiling:
            try:
                # Create runtime profiler
                self.runtime_profiler = create_runtime_profiler(
                    enable_detailed_profiling=True,
                    enable_memory_profiling=True,
                    simd_processor=self.simd_processor
                )
                
                # Create adaptive optimizer if SIMD is enabled
                if self.enable_simd and self.simd_codegen:
                    self.adaptive_optimizer = AdaptiveOptimizer(
                        self.runtime_profiler, 
                        self.simd_codegen
                    )
                
                print("✅ Runtime profiling enabled")
                
            except Exception as e:
                print(f"⚠️  Profiling initialization failed: {e}")
                self.enable_profiling = False
    
    def generate_with_simd_optimization(self, ir_module: IRModule) -> 'llvm.Module':
        """
        Generate LLVM IR with SIMD optimization enabled.
        
        Args:
            ir_module: NeuralScript IR module
            
        Returns:
            LLVM module with SIMD optimizations applied
        """
        # Apply auto-vectorization pass if enabled
        if self.enable_simd and self.auto_vectorizer:
            try:
                print("🔄 Running auto-vectorization pass...")
                vectorization_result = self.auto_vectorizer.run_pass(ir_module)
                
                if vectorization_result.transformations_applied > 0:
                    print(f"✅ Applied {vectorization_result.transformations_applied} vectorization transformations")
                    print(f"   - Estimated speedup: {vectorization_result.estimated_speedup:.2f}x")
                    
            except Exception as e:
                print(f"⚠️  Auto-vectorization failed: {e}")
        
        # Generate LLVM IR
        llvm_module = self.generate(ir_module)
        
        # Apply additional SIMD-specific optimizations
        if self.enable_simd:
            llvm_module = self._apply_simd_optimizations(llvm_module)
        
        return llvm_module
    
    def _apply_simd_optimizations(self, llvm_module: 'llvm.Module') -> 'llvm.Module':
        """
        Apply SIMD-specific optimizations to the LLVM module.
        """
        try:
            # Enable SIMD-friendly optimizations in LLVM
            pm = llvm.create_module_pass_manager()
            
            # Add vectorization passes
            pm.add_loop_vectorize_pass()
            pm.add_slp_vectorize_pass()
            
            # Add other SIMD-friendly passes
            pm.add_instruction_combining_pass()
            pm.add_reassociate_expressions_pass()
            
            # Run passes
            pm.run(llvm_module)
            
            print("✅ Applied SIMD optimizations")
            
        except Exception as e:
            print(f"⚠️  SIMD optimization failed: {e}")
        
        return llvm_module
    
    def generate_simd_matrix_multiply(self, dimensions: Tuple[int, int, int], 
                                    data_type: DataType = DataType.FLOAT32) -> str:
        """
        Generate optimized SIMD code for matrix multiplication.
        
        Args:
            dimensions: Matrix dimensions (m, k, n)
            data_type: Data type for the operation
            
        Returns:
            Generated LLVM IR as string
        """
        if not self.enable_simd or not self.simd_codegen:
            raise RuntimeError("SIMD code generation is not enabled")
        
        try:
            # Create matrix dimensions
            matrix_dims = MatrixDimensions(*dimensions)
            
            # Create a basic SIMD matrix multiply function
            llvm_ir = self._create_simd_matrix_multiply_ir(matrix_dims, data_type)
            
            return llvm_ir
            
        except Exception as e:
            print(f"SIMD matrix multiply generation failed: {e}")
            raise
    
    def _create_simd_matrix_multiply_ir(self, dimensions: MatrixDimensions, data_type: DataType) -> str:
        """
        Create basic SIMD matrix multiply LLVM IR.
        """
        # Create a simple LLVM IR function for the matrix multiply
        ir_lines = [
            f"; SIMD Matrix multiply function for {dimensions.m}x{dimensions.k} * {dimensions.k}x{dimensions.n}",
            f"define void @simd_matrix_multiply_{dimensions.m}_{dimensions.k}_{dimensions.n}(",
            f"    float* %A, float* %B, float* %C) {{",
            "entry:",
            "  ; Optimized SIMD matrix multiplication would be generated here",
            "  ; This is a placeholder showing SIMD integration",
            "  %vector_width = add i32 4, 0  ; SSE vector width",
            "  ret void",
            "}"
        ]
        
        return "\n".join(ir_lines)
    
    def profile_function_execution(self, function_name: str, 
                                 execution_time_ms: float,
                                 operation_details: Optional[Dict] = None) -> str:
        """
        Record function execution for profiling.
        
        Args:
            function_name: Name of the function
            execution_time_ms: Execution time in milliseconds
            operation_details: Additional operation details
            
        Returns:
            Profile ID for the recorded execution
        """
        if self.runtime_profiler:
            return self.runtime_profiler.record_function_call(
                function_name, execution_time_ms, operation_details
            )
        return ""
    
    def get_optimization_recommendations(self, function_name: str) -> Optional[Dict[str, Any]]:
        """
        Get optimization recommendations for a function based on profiling data.
        
        Args:
            function_name: Name of the function
            
        Returns:
            Optimization recommendations or None if no data available
        """
        if not self.adaptive_optimizer:
            return None
        
        should_optimize, reason = self.adaptive_optimizer.should_optimize_function(function_name)
        
        if should_optimize:
            return {
                'should_optimize': True,
                'reason': reason,
                'recommendations': [
                    'Apply SIMD vectorization',
                    'Enable cache blocking for large matrices',
                    'Consider loop unrolling'
                ]
            }
        else:
            return {
                'should_optimize': False,
                'reason': reason
            }
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """
        Get a summary of profiling data.
        
        Returns:
            Profiling summary with hot functions and optimization opportunities
        """
        if not self.runtime_profiler:
            return {'error': 'Profiling not enabled'}
        
        summary = {
            'hot_functions': [],
            'optimization_candidates': [],
            'performance_alerts': []
        }
        
        # Get hot functions
        hot_functions = self.runtime_profiler.get_hot_functions(top_n=5)
        for profile in hot_functions:
            summary['hot_functions'].append({
                'name': profile.name,
                'call_count': profile.call_count,
                'avg_time_ms': profile.avg_time_ms,
                'hotness_score': profile.hotness_score,
                'trend': profile.get_trend()
            })
        
        # Get optimization candidates
        candidates = self.runtime_profiler.get_optimization_candidates()
        for profile in candidates[:5]:
            summary['optimization_candidates'].append({
                'name': profile.name,
                'potential_benefit': 'High' if profile.matrix_ops_count > 0 else 'Medium',
                'call_count': profile.call_count,
                'avg_time_ms': profile.avg_time_ms
            })
        
        # Get performance alerts
        alerts = self.runtime_profiler.get_performance_regression_alerts()
        summary['performance_alerts'] = alerts[-5:]  # Recent 5 alerts
        
        return summary


# Convenience function for testing without llvmlite
def create_mock_backend() -> LLVMBackend:
    """Create a mock backend when llvmlite is not available."""
    class MockBackend:
        def __init__(self):
            self.target_triple = "x86_64-pc-linux-gnu"
        
        def generate(self, ir_module):
            print(f"Mock LLVM backend: would generate code for {ir_module.name}")
            return None
        
        def compile_to_object(self, llvm_module, output_path, optimization_level=2):
            print(f"Mock LLVM backend: would compile to {output_path}")
            return True
        
        def compile_to_executable(self, llvm_module, output_path, optimization_level=2, link_with=None):
            print(f"Mock LLVM backend: would create executable {output_path}")
            return True
    
    return MockBackend()
