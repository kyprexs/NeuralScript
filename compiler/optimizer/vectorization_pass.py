"""
Auto-Vectorization Compiler Pass
===============================

Automatically detects and optimizes vectorizable operations in NeuralScript IR.
This pass identifies matrix operations, loops, and other patterns that can
benefit from SIMD vectorization and applies appropriate optimizations.

Features:
- Automatic loop vectorization detection
- Matrix operation pattern recognition  
- SIMD width and strategy selection
- Dependency analysis for safe vectorization
- Performance cost modeling
- Optimization decision making
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ..ir.ir_nodes import *
from ..backend.simd_codegen import SIMDCodeGenerator, MatrixCodegenOptimizer


class VectorizationOpportunity(Enum):
    """Types of vectorization opportunities"""
    MATRIX_MULTIPLY = "matrix_multiply"
    ELEMENT_WISE_OP = "element_wise_op" 
    REDUCTION = "reduction"
    SIMPLE_LOOP = "simple_loop"
    STRIDED_ACCESS = "strided_access"
    GATHER_SCATTER = "gather_scatter"


@dataclass
class VectorizationCandidate:
    """A candidate for vectorization optimization"""
    opportunity_type: VectorizationOpportunity
    ir_node: IRNode
    estimated_benefit: float  # Expected speedup factor
    vectorization_width: int
    dependencies: Set[IRValue]
    memory_accesses: List[Tuple[IRValue, str]]  # (pointer, access_pattern)
    constraints: List[str]  # Constraints that must be satisfied
    confidence: float  # Confidence in the analysis (0.0 to 1.0)


class VectorizationAnalyzer:
    """
    Analyzes IR to find vectorization opportunities.
    
    Performs dependency analysis, memory access pattern recognition,
    and cost-benefit analysis for potential vectorizations.
    """
    
    def __init__(self, simd_codegen: SIMDCodeGenerator):
        self.simd_codegen = simd_codegen
        self.vector_width = simd_codegen._get_vector_width()
        
    def analyze_function(self, ir_function: IRFunction) -> List[VectorizationCandidate]:
        """Analyze a function for vectorization opportunities"""
        candidates = []
        
        for basic_block in ir_function.basic_blocks:
            candidates.extend(self._analyze_basic_block(basic_block))
        
        # Sort by estimated benefit
        candidates.sort(key=lambda c: c.estimated_benefit, reverse=True)
        
        return candidates
    
    def _analyze_basic_block(self, block: IRBasicBlock) -> List[VectorizationCandidate]:
        """Analyze a basic block for vectorization opportunities"""
        candidates = []
        
        for instruction in block.instructions:
            if isinstance(instruction, IRMatMul):
                candidate = self._analyze_matrix_multiply(instruction)
                if candidate:
                    candidates.append(candidate)
            
            elif isinstance(instruction, IRBinaryOp):
                candidate = self._analyze_binary_op(instruction)
                if candidate:
                    candidates.append(candidate)
            
            elif isinstance(instruction, IRCall):
                candidate = self._analyze_call(instruction)
                if candidate:
                    candidates.append(candidate)
        
        # Look for loop patterns
        loop_candidates = self._detect_vectorizable_loops(block)
        candidates.extend(loop_candidates)
        
        return candidates
    
    def _analyze_matrix_multiply(self, matmul: IRMatMul) -> Optional[VectorizationCandidate]:
        """Analyze matrix multiplication for vectorization"""
        
        # Extract dimensions if available
        try:
            left_shape = matmul.left.type.shape if hasattr(matmul.left.type, 'shape') else None
            right_shape = matmul.right.type.shape if hasattr(matmul.right.type, 'shape') else None
            
            if left_shape and right_shape:
                m, k = left_shape
                k2, n = right_shape
                
                # Calculate potential benefit
                total_ops = 2 * m * n * k
                vector_speedup = min(self.vector_width, n)  # Limited by innermost dimension
                cache_benefit = 1.2 if m * k > 10000 else 1.0
                
                estimated_benefit = vector_speedup * cache_benefit
                
                # Determine constraints
                constraints = []
                if k != k2:
                    constraints.append("dimension_mismatch")
                
                confidence = 0.9 if not constraints else 0.5
                
                return VectorizationCandidate(
                    opportunity_type=VectorizationOpportunity.MATRIX_MULTIPLY,
                    ir_node=matmul,
                    estimated_benefit=estimated_benefit,
                    vectorization_width=self.vector_width,
                    dependencies={matmul.left, matmul.right},
                    memory_accesses=[
                        (matmul.left, "sequential_read"),
                        (matmul.right, "strided_read"), 
                        (matmul.result, "sequential_write")
                    ],
                    constraints=constraints,
                    confidence=confidence
                )
        except Exception:
            pass
        
        # Fallback analysis for unknown dimensions
        return VectorizationCandidate(
            opportunity_type=VectorizationOpportunity.MATRIX_MULTIPLY,
            ir_node=matmul,
            estimated_benefit=4.0,  # Conservative estimate
            vectorization_width=self.vector_width,
            dependencies={matmul.left, matmul.right},
            memory_accesses=[],
            constraints=[],
            confidence=0.3
        )
    
    def _analyze_binary_op(self, binop: IRBinaryOp) -> Optional[VectorizationCandidate]:
        """Analyze binary operations for element-wise vectorization"""
        
        # Check if this is an element-wise operation on arrays/tensors
        left_type = binop.left.type
        right_type = binop.right.type
        
        if (isinstance(left_type, IRTensorType) and 
            isinstance(right_type, IRTensorType) and
            left_type.shape == right_type.shape):
            
            # Element-wise operations are highly vectorizable
            total_elements = left_type.total_elements
            
            if total_elements >= self.vector_width:
                vectorization_factor = min(self.vector_width, total_elements)
                estimated_benefit = vectorization_factor * 0.95  # High efficiency
                
                return VectorizationCandidate(
                    opportunity_type=VectorizationOpportunity.ELEMENT_WISE_OP,
                    ir_node=binop,
                    estimated_benefit=estimated_benefit,
                    vectorization_width=self.vector_width,
                    dependencies={binop.left, binop.right},
                    memory_accesses=[
                        (binop.left, "sequential_read"),
                        (binop.right, "sequential_read"),
                        (binop.result, "sequential_write")
                    ],
                    constraints=[],
                    confidence=0.95
                )
        
        return None
    
    def _analyze_call(self, call: IRCall) -> Optional[VectorizationCandidate]:
        """Analyze function calls for vectorizable math functions"""
        
        # Check if this is a math function that can be vectorized
        math_functions = {
            "sin", "cos", "tan", "exp", "log", "sqrt", "abs",
            "sigmoid", "tanh", "relu", "softmax"
        }
        
        func_name = getattr(call.function, 'name', '')
        if func_name in math_functions:
            
            # Check if operating on arrays/tensors
            if call.args and isinstance(call.args[0].type, IRTensorType):
                tensor_type = call.args[0].type
                total_elements = tensor_type.total_elements
                
                if total_elements >= self.vector_width:
                    # Math functions vectorize very well
                    vectorization_factor = min(self.vector_width, total_elements)
                    estimated_benefit = vectorization_factor * 0.8  # Good but not perfect
                    
                    return VectorizationCandidate(
                        opportunity_type=VectorizationOpportunity.ELEMENT_WISE_OP,
                        ir_node=call,
                        estimated_benefit=estimated_benefit,
                        vectorization_width=self.vector_width,
                        dependencies=set(call.args),
                        memory_accesses=[
                            (call.args[0], "sequential_read"),
                            (call.result, "sequential_write")
                        ],
                        constraints=[],
                        confidence=0.85
                    )
        
        return None
    
    def _detect_vectorizable_loops(self, block: IRBasicBlock) -> List[VectorizationCandidate]:
        """Detect loop patterns that can be vectorized"""
        candidates = []
        
        # Look for loop structures (simplified pattern matching)
        # In a real implementation, this would use more sophisticated
        # control flow analysis
        
        # For now, look for simple patterns like:
        # for i in range(n): array[i] = some_operation(array[i])
        
        return candidates  # Placeholder for now
    
    def check_vectorization_safety(self, candidate: VectorizationCandidate,
                                 all_candidates: List[VectorizationCandidate]) -> bool:
        """Check if vectorization is safe (no data races, dependencies)"""
        
        # Basic dependency analysis
        for other in all_candidates:
            if other == candidate:
                continue
            
            # Check for conflicting memory accesses
            if self._has_conflicting_accesses(candidate, other):
                return False
        
        # Check for specific constraints
        for constraint in candidate.constraints:
            if constraint == "dimension_mismatch":
                return False
        
        return True
    
    def _has_conflicting_accesses(self, candidate1: VectorizationCandidate,
                                candidate2: VectorizationCandidate) -> bool:
        """Check if two candidates have conflicting memory accesses"""
        
        accesses1 = {access[0] for access in candidate1.memory_accesses}
        accesses2 = {access[0] for access in candidate2.memory_accesses}
        
        # Simple overlap check - in reality would need more sophisticated analysis
        return bool(accesses1 & accesses2)


class VectorizationPass:
    """
    Compiler pass that applies SIMD vectorization optimizations.
    
    This pass analyzes the IR, identifies vectorization opportunities,
    and transforms the IR to use optimized SIMD operations.
    """
    
    def __init__(self, simd_codegen: Optional[SIMDCodeGenerator] = None):
        self.simd_codegen = simd_codegen or SIMDCodeGenerator()
        self.analyzer = VectorizationAnalyzer(self.simd_codegen)
        self.optimizer = MatrixCodegenOptimizer(self.simd_codegen)
        
        # Pass statistics
        self.stats = {
            'functions_analyzed': 0,
            'candidates_found': 0,
            'optimizations_applied': 0,
            'estimated_speedup': 1.0
        }
    
    def run_on_module(self, ir_module: IRModule) -> IRModule:
        """Run vectorization pass on entire module"""
        
        for func_name, ir_function in ir_module.functions.items():
            self.run_on_function(ir_function)
        
        return ir_module
    
    def run_on_function(self, ir_function: IRFunction) -> IRFunction:
        """Run vectorization pass on a single function"""
        
        self.stats['functions_analyzed'] += 1
        
        # Analyze function for vectorization opportunities
        candidates = self.analyzer.analyze_function(ir_function)
        self.stats['candidates_found'] += len(candidates)
        
        if not candidates:
            return ir_function
        
        # Apply safe and beneficial optimizations
        applied_optimizations = []
        
        for candidate in candidates:
            if (candidate.confidence >= 0.5 and 
                candidate.estimated_benefit >= 2.0 and
                self.analyzer.check_vectorization_safety(candidate, candidates)):
                
                success = self._apply_vectorization(candidate)
                if success:
                    applied_optimizations.append(candidate)
                    self.stats['optimizations_applied'] += 1
        
        # Update estimated speedup
        if applied_optimizations:
            total_benefit = sum(opt.estimated_benefit for opt in applied_optimizations)
            avg_benefit = total_benefit / len(applied_optimizations)
            self.stats['estimated_speedup'] *= min(avg_benefit, 10.0)
        
        return ir_function
    
    def _apply_vectorization(self, candidate: VectorizationCandidate) -> bool:
        """Apply vectorization transformation to a candidate"""
        
        try:
            if candidate.opportunity_type == VectorizationOpportunity.MATRIX_MULTIPLY:
                return self._vectorize_matrix_multiply(candidate)
            
            elif candidate.opportunity_type == VectorizationOpportunity.ELEMENT_WISE_OP:
                return self._vectorize_element_wise(candidate)
            
            else:
                # Other vectorization types not implemented yet
                return False
                
        except Exception as e:
            print(f"Vectorization failed for {candidate.opportunity_type}: {e}")
            return False
    
    def _vectorize_matrix_multiply(self, candidate: VectorizationCandidate) -> bool:
        """Apply matrix multiplication vectorization"""
        
        matmul_ir = candidate.ir_node
        if not isinstance(matmul_ir, IRMatMul):
            return False
        
        # Get optimization plan from the matrix optimizer
        optimization_plan = self.optimizer.optimize_matrix_multiply(matmul_ir)
        
        # Update the IR node with optimization metadata
        strategy = optimization_plan['strategy_decisions']
        
        matmul_ir.prefer_simd = strategy.get('use_vectorization', True)
        matmul_ir.block_size = strategy.get('optimal_block_size', 64)
        matmul_ir.use_strassen = strategy.get('use_strassen', False)
        matmul_ir.parallel_threshold = strategy.get('parallel_threshold', 1000000)
        
        # Update metadata
        matmul_ir.set_metadata('simd_optimizable', matmul_ir.prefer_simd)
        matmul_ir.set_metadata('block_size', matmul_ir.block_size)
        matmul_ir.set_metadata('use_strassen', matmul_ir.use_strassen)
        matmul_ir.set_metadata('vectorization_width', candidate.vectorization_width)
        matmul_ir.set_metadata('expected_speedup', candidate.estimated_benefit)
        
        return True
    
    def _vectorize_element_wise(self, candidate: VectorizationCandidate) -> bool:
        """Apply element-wise operation vectorization"""
        
        ir_node = candidate.ir_node
        
        # Add vectorization metadata
        ir_node.set_metadata('simd_optimizable', True)
        ir_node.set_metadata('vectorization_width', candidate.vectorization_width)
        ir_node.set_metadata('access_pattern', 'sequential')
        ir_node.set_metadata('expected_speedup', candidate.estimated_benefit)
        
        return True
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate a report of optimization results"""
        
        return {
            'pass_name': 'SIMD Vectorization Pass',
            'statistics': self.stats.copy(),
            'simd_strategy': self.simd_codegen.strategy.value,
            'vector_width': self.simd_codegen._get_vector_width(),
            'cache_block_size': self.simd_codegen.optimal_block_size,
            'performance_summary': {
                'functions_optimized': self.stats['optimizations_applied'] > 0,
                'total_functions': self.stats['functions_analyzed'],
                'optimization_rate': (
                    self.stats['optimizations_applied'] / 
                    max(1, self.stats['candidates_found'])
                ),
                'estimated_total_speedup': self.stats['estimated_speedup']
            }
        }


class AdvancedVectorizationPass(VectorizationPass):
    """
    Advanced vectorization pass with more sophisticated analysis.
    
    Includes features like:
    - Cross-function vectorization analysis
    - Advanced dependency analysis 
    - Cost-benefit modeling with profiling data
    - Adaptive optimization based on runtime feedback
    """
    
    def __init__(self, simd_codegen: Optional[SIMDCodeGenerator] = None,
                 profiling_data: Optional[Dict] = None):
        super().__init__(simd_codegen)
        self.profiling_data = profiling_data or {}
        self.cross_function_analysis = True
        
    def run_on_module(self, ir_module: IRModule) -> IRModule:
        """Enhanced module-level analysis with cross-function optimization"""
        
        if self.cross_function_analysis:
            # Analyze call graph for vectorization opportunities
            call_graph = self._build_call_graph(ir_module)
            hotspots = self._identify_hotspots(call_graph)
            
            # Prioritize functions based on profiling data and call frequency
            function_priority = self._prioritize_functions(ir_module, hotspots)
            
            # Process functions in priority order
            for func_name in function_priority:
                if func_name in ir_module.functions:
                    self.run_on_function(ir_module.functions[func_name])
        else:
            # Fall back to base implementation
            super().run_on_module(ir_module)
        
        return ir_module
    
    def _build_call_graph(self, ir_module: IRModule) -> Dict[str, List[str]]:
        """Build call graph for cross-function analysis"""
        call_graph = {}
        
        for func_name, function in ir_module.functions.items():
            calls = []
            for block in function.basic_blocks:
                for instruction in block.instructions:
                    if isinstance(instruction, IRCall):
                        callee = getattr(instruction.function, 'name', '')
                        if callee and callee in ir_module.functions:
                            calls.append(callee)
            call_graph[func_name] = calls
        
        return call_graph
    
    def _identify_hotspots(self, call_graph: Dict[str, List[str]]) -> Dict[str, float]:
        """Identify hot functions based on call frequency"""
        hotspots = {}
        
        # Use profiling data if available
        if self.profiling_data:
            for func_name, profile in self.profiling_data.items():
                execution_time = profile.get('total_time', 0)
                call_count = profile.get('call_count', 0)
                hotspots[func_name] = execution_time * call_count
        else:
            # Estimate hotness based on call graph
            for func_name, callees in call_graph.items():
                # Functions that are called frequently or call many others
                hotness = len(callees) + sum(1 for calls in call_graph.values() 
                                           if func_name in calls)
                hotspots[func_name] = float(hotness)
        
        return hotspots
    
    def _prioritize_functions(self, ir_module: IRModule, 
                            hotspots: Dict[str, float]) -> List[str]:
        """Prioritize functions for optimization"""
        
        function_scores = []
        
        for func_name, function in ir_module.functions.items():
            hotness = hotspots.get(func_name, 0.0)
            
            # Estimate vectorization potential
            candidates = self.analyzer.analyze_function(function)
            vectorization_potential = sum(c.estimated_benefit * c.confidence 
                                        for c in candidates)
            
            total_score = hotness * vectorization_potential
            function_scores.append((func_name, total_score))
        
        # Sort by score descending
        function_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [name for name, score in function_scores]


# Factory function for creating vectorization passes
def create_vectorization_pass(optimization_level: int = 2,
                            simd_codegen: Optional[SIMDCodeGenerator] = None,
                            profiling_data: Optional[Dict] = None) -> VectorizationPass:
    """
    Create appropriate vectorization pass based on optimization level.
    
    Args:
        optimization_level: 0=none, 1=basic, 2=standard, 3=aggressive
        simd_codegen: SIMD code generator (optional)
        profiling_data: Runtime profiling data (optional)
    
    Returns:
        Configured vectorization pass
    """
    
    if optimization_level <= 0:
        # Return no-op pass
        pass_instance = VectorizationPass(simd_codegen)
        pass_instance.run_on_function = lambda f: f  # No-op
        pass_instance.run_on_module = lambda m: m    # No-op
        return pass_instance
    
    elif optimization_level <= 2:
        # Standard vectorization pass
        return VectorizationPass(simd_codegen)
    
    else:
        # Advanced pass with cross-function analysis
        return AdvancedVectorizationPass(simd_codegen, profiling_data)
