"""
Automatic Differentiation Engine for NeuralScript

Implements both forward-mode and reverse-mode automatic differentiation
with support for higher-order derivatives and complex mathematical functions.

Author: xwest
"""

from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import uuid
from abc import ABC, abstractmethod

from ..parser.ast_nodes import *
from ..analyzer.semantic_analyzer import AnalysisResult, SymbolType
from ..ir.ir_nodes import *


class ADMode(Enum):
    """Automatic differentiation modes."""
    FORWARD = "forward"      # Forward-mode AD (tangent-linear)
    REVERSE = "reverse"      # Reverse-mode AD (adjoint)
    MIXED = "mixed"          # Mixed-mode for higher-order derivatives


@dataclass
class Derivative:
    """Represents a derivative with respect to a variable."""
    variable: str
    order: int = 1
    mode: ADMode = ADMode.REVERSE


@dataclass
class DualNumber:
    """Dual number for forward-mode AD: f + ε * f'"""
    primal: float  # Function value
    tangent: float  # Derivative value
    
    def __add__(self, other: Union['DualNumber', float]) -> 'DualNumber':
        if isinstance(other, DualNumber):
            return DualNumber(self.primal + other.primal, self.tangent + other.tangent)
        return DualNumber(self.primal + other, self.tangent)
    
    def __mul__(self, other: Union['DualNumber', float]) -> 'DualNumber':
        if isinstance(other, DualNumber):
            return DualNumber(
                self.primal * other.primal, 
                self.primal * other.tangent + self.tangent * other.primal
            )
        return DualNumber(self.primal * other, self.tangent * other)
    
    def __sub__(self, other: Union['DualNumber', float]) -> 'DualNumber':
        if isinstance(other, DualNumber):
            return DualNumber(self.primal - other.primal, self.tangent - other.tangent)
        return DualNumber(self.primal - other, self.tangent)
    
    def __truediv__(self, other: Union['DualNumber', float]) -> 'DualNumber':
        if isinstance(other, DualNumber):
            return DualNumber(
                self.primal / other.primal,
                (self.tangent * other.primal - self.primal * other.tangent) / (other.primal ** 2)
            )
        return DualNumber(self.primal / other, self.tangent / other)


@dataclass
class ADVariable:
    """Variable in the AD computation graph."""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    requires_grad: bool = True
    grad: Optional[Any] = None
    
    def __post_init__(self):
        self.id = uuid.uuid4()


@dataclass
class ADOperation:
    """Operation node in the AD computation graph."""
    op_type: str
    inputs: List[ADVariable]
    output: ADVariable
    backward_fn: Optional[callable] = None
    
    def __post_init__(self):
        self.id = uuid.uuid4()


class ADComputationGraph:
    """Computation graph for automatic differentiation."""
    
    def __init__(self):
        self.variables: Dict[str, ADVariable] = {}
        self.operations: List[ADOperation] = []
        self.topological_order: List[ADOperation] = []
        
    def add_variable(self, name: str, shape: Tuple[int, ...], dtype: str, 
                    requires_grad: bool = True) -> ADVariable:
        """Add a variable to the computation graph."""
        var = ADVariable(name, shape, dtype, requires_grad)
        self.variables[name] = var
        return var
    
    def add_operation(self, op_type: str, inputs: List[ADVariable], 
                     output: ADVariable, backward_fn: callable = None) -> ADOperation:
        """Add an operation to the computation graph."""
        operation = ADOperation(op_type, inputs, output, backward_fn)
        self.operations.append(operation)
        return operation
    
    def topological_sort(self) -> List[ADOperation]:
        """Perform topological sort for reverse-mode AD."""
        if self.topological_order:
            return self.topological_order
            
        # Simple topological sort implementation
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(op: ADOperation):
            if op.id in temp_visited:
                raise ValueError("Cycle detected in computation graph")
            if op.id in visited:
                return
                
            temp_visited.add(op.id)
            
            # Visit dependencies (operations that produce inputs to this op)
            for input_var in op.inputs:
                for other_op in self.operations:
                    if other_op.output.id == input_var.id:
                        visit(other_op)
            
            temp_visited.remove(op.id)
            visited.add(op.id)
            result.append(op)
        
        for operation in self.operations:
            if operation.id not in visited:
                visit(operation)
        
        self.topological_order = result
        return result
    
    def backward(self, loss_var: ADVariable):
        """Perform reverse-mode automatic differentiation."""
        # Initialize gradients
        for var in self.variables.values():
            if var.requires_grad:
                var.grad = 0.0
        
        # Set loss gradient to 1
        loss_var.grad = 1.0
        
        # Reverse topological order for backpropagation
        ops = self.topological_sort()
        
        for op in reversed(ops):
            if op.backward_fn:
                op.backward_fn()


class AutodiffEngine:
    """
    Main automatic differentiation engine.
    
    Transforms NeuralScript AST and IR to support automatic differentiation,
    including gradient computation, Jacobian matrices, and higher-order derivatives.
    """
    
    def __init__(self):
        self.computation_graph = ADComputationGraph()
        self.function_derivatives: Dict[str, Dict[str, callable]] = {}
        self.mode = ADMode.REVERSE
        
        # Initialize built-in function derivatives
        self._init_builtin_derivatives()
    
    def _init_builtin_derivatives(self):
        """Initialize derivatives for built-in mathematical functions."""
        import math
        
        # Elementary functions and their derivatives
        self.function_derivatives = {
            "sin": {
                "forward": lambda x: DualNumber(math.sin(x.primal), math.cos(x.primal) * x.tangent),
                "backward": lambda x, grad: grad * math.cos(x)
            },
            "cos": {
                "forward": lambda x: DualNumber(math.cos(x.primal), -math.sin(x.primal) * x.tangent),
                "backward": lambda x, grad: -grad * math.sin(x)
            },
            "exp": {
                "forward": lambda x: DualNumber(math.exp(x.primal), math.exp(x.primal) * x.tangent),
                "backward": lambda x, grad: grad * math.exp(x)
            },
            "log": {
                "forward": lambda x: DualNumber(math.log(x.primal), x.tangent / x.primal),
                "backward": lambda x, grad: grad / x
            },
            "sqrt": {
                "forward": lambda x: DualNumber(math.sqrt(x.primal), x.tangent / (2 * math.sqrt(x.primal))),
                "backward": lambda x, grad: grad / (2 * math.sqrt(x))
            },
            "pow": {
                "forward": lambda x, n: DualNumber(
                    x.primal ** n, 
                    n * (x.primal ** (n-1)) * x.tangent
                ),
                "backward": lambda x, n, grad: grad * n * (x ** (n-1))
            }
        }
    
    def differentiate_function(self, func_def: FunctionDef, 
                             derivatives: List[Derivative]) -> FunctionDef:
        """
        Generate a differentiated version of a function.
        
        Args:
            func_def: The original function definition
            derivatives: List of derivatives to compute
            
        Returns:
            New function definition that computes derivatives
        """
        if self.mode == ADMode.FORWARD:
            return self._differentiate_forward(func_def, derivatives)
        elif self.mode == ADMode.REVERSE:
            return self._differentiate_reverse(func_def, derivatives)
        else:
            return self._differentiate_mixed(func_def, derivatives)
    
    def _differentiate_forward(self, func_def: FunctionDef, 
                              derivatives: List[Derivative]) -> FunctionDef:
        """Implement forward-mode automatic differentiation."""
        # Create new function name
        derivative_names = "_".join([f"d{d.variable}" for d in derivatives])
        new_name = f"{func_def.name}_{derivative_names}"
        
        # Transform function body for forward-mode AD
        new_body = self._transform_statements_forward(func_def.body, derivatives)
        
        # Create new parameters (original + tangent inputs)
        new_params = func_def.params.copy()
        for deriv in derivatives:
            tangent_param = Parameter(
                name=f"d_{deriv.variable}",
                type_annotation=None,  # Will be inferred
                default_value=None
            )
            new_params.append(tangent_param)
        
        # Create differentiated function
        diff_func = FunctionDef(
            name=new_name,
            type_params=func_def.type_params,
            params=new_params,
            return_type=None,  # Will be inferred
            body=new_body,
            span=func_def.span,
            is_differentiable=True
        )
        
        return diff_func
    
    def _differentiate_reverse(self, func_def: FunctionDef, 
                              derivatives: List[Derivative]) -> FunctionDef:
        """Implement reverse-mode automatic differentiation."""
        # Create new function name
        derivative_names = "_".join([f"d{d.variable}" for d in derivatives])
        new_name = f"{func_def.name}_{derivative_names}_reverse"
        
        # Transform function body for reverse-mode AD
        new_body = self._transform_statements_reverse(func_def.body, derivatives)
        
        # Add gradient outputs to return type
        # This would return both the original value and gradients
        
        diff_func = FunctionDef(
            name=new_name,
            type_params=func_def.type_params,
            params=func_def.params,
            return_type=None,  # Will be a tuple of (value, gradients)
            body=new_body,
            span=func_def.span,
            is_differentiable=True
        )
        
        return diff_func
    
    def _differentiate_mixed(self, func_def: FunctionDef, 
                            derivatives: List[Derivative]) -> FunctionDef:
        """Implement mixed-mode AD for higher-order derivatives."""
        # For now, default to reverse mode
        return self._differentiate_reverse(func_def, derivatives)
    
    def _transform_statements_forward(self, stmt: Statement, 
                                    derivatives: List[Derivative]) -> Statement:
        """Transform statements for forward-mode AD."""
        if isinstance(stmt, BlockStatement):
            new_statements = []
            for s in stmt.statements:
                new_statements.append(self._transform_statements_forward(s, derivatives))
            return BlockStatement(new_statements, stmt.span)
        
        elif isinstance(stmt, ReturnStatement) and stmt.value:
            # Transform return expression
            new_expr = self._transform_expression_forward(stmt.value, derivatives)
            return ReturnStatement(new_expr, stmt.span)
        
        elif isinstance(stmt, VariableDecl) and stmt.initializer:
            # Transform variable initialization
            new_init = self._transform_expression_forward(stmt.initializer, derivatives)
            return VariableDecl(
                stmt.name, stmt.type_annotation, new_init, stmt.span,
                is_mutable=stmt.is_mutable, is_const=stmt.is_const
            )
        
        return stmt
    
    def _transform_statements_reverse(self, stmt: Statement, 
                                    derivatives: List[Derivative]) -> Statement:
        """Transform statements for reverse-mode AD."""
        # For reverse mode, we need to build a computation graph
        # and then add backward pass logic
        
        if isinstance(stmt, BlockStatement):
            # Add computation graph building
            new_statements = []
            
            # Forward pass (build computation graph)
            for s in stmt.statements:
                new_statements.append(self._transform_statements_reverse(s, derivatives))
            
            # Add backward pass
            backward_stmts = self._generate_backward_pass(derivatives)
            new_statements.extend(backward_stmts)
            
            return BlockStatement(new_statements, stmt.span)
        
        return stmt
    
    def _transform_expression_forward(self, expr: Expression, 
                                    derivatives: List[Derivative]) -> Expression:
        """Transform expressions for forward-mode AD."""
        if isinstance(expr, BinaryOp):
            # Transform binary operations using dual number arithmetic
            left_dual = self._transform_expression_forward(expr.left, derivatives)
            right_dual = self._transform_expression_forward(expr.right, derivatives)
            
            # Create dual number operation
            return self._create_dual_binary_op(expr.operator, left_dual, right_dual, expr.span)
        
        elif isinstance(expr, FunctionCall):
            # Transform function calls using chain rule
            return self._transform_function_call_forward(expr, derivatives)
        
        elif isinstance(expr, Identifier):
            # Check if this is a differentiated variable
            for deriv in derivatives:
                if expr.name == deriv.variable:
                    # Return dual number with tangent = 1
                    return self._create_dual_variable(expr.name, 1.0, expr.span)
            
            # Regular variable, tangent = 0
            return self._create_dual_variable(expr.name, 0.0, expr.span)
        
        return expr
    
    def _create_dual_binary_op(self, operator: str, left: Expression, 
                              right: Expression, span: SourceSpan) -> Expression:
        """Create a dual number binary operation."""
        # This would create IR that implements dual number arithmetic
        # For now, return a placeholder function call
        
        dual_op_name = f"dual_{operator}"
        return FunctionCall(
            function=Identifier(dual_op_name, span),
            args=[left, right],
            span=span
        )
    
    def _create_dual_variable(self, name: str, tangent: float, span: SourceSpan) -> Expression:
        """Create a dual number variable."""
        return FunctionCall(
            function=Identifier("make_dual", span),
            args=[
                Identifier(name, span),
                Literal(tangent, "float", span)
            ],
            span=span
        )
    
    def _transform_function_call_forward(self, call: FunctionCall, 
                                       derivatives: List[Derivative]) -> Expression:
        """Transform function calls for forward-mode AD."""
        if isinstance(call.function, Identifier):
            func_name = call.function.name
            
            if func_name in self.function_derivatives:
                # Use pre-computed derivative
                deriv_func_name = f"{func_name}_dual"
                return FunctionCall(
                    function=Identifier(deriv_func_name, call.span),
                    args=[self._transform_expression_forward(arg, derivatives) 
                          for arg in call.args],
                    span=call.span
                )
        
        # Default transformation
        return FunctionCall(
            function=call.function,
            args=[self._transform_expression_forward(arg, derivatives) 
                  for arg in call.args],
            span=call.span
        )
    
    def _generate_backward_pass(self, derivatives: List[Derivative]) -> List[Statement]:
        """Generate statements for the backward pass in reverse-mode AD."""
        # This would generate code to traverse the computation graph backwards
        # and accumulate gradients
        
        statements = []
        
        # Initialize gradient accumulators
        for deriv in derivatives:
            grad_init = VariableDecl(
                name=f"grad_{deriv.variable}",
                type_annotation=None,
                initializer=Literal(0.0, "float", SourceSpan(
                    SourceLocation("generated", 0, 0, 0),
                    SourceLocation("generated", 0, 0, 0)
                )),
                span=SourceSpan(
                    SourceLocation("generated", 0, 0, 0),
                    SourceLocation("generated", 0, 0, 0)
                )
            )
            statements.append(grad_init)
        
        # Add backward pass computation (placeholder)
        backward_call = FunctionCall(
            function=Identifier("compute_gradients", SourceSpan(
                SourceLocation("generated", 0, 0, 0),
                SourceLocation("generated", 0, 0, 0)
            )),
            args=[],
            span=SourceSpan(
                SourceLocation("generated", 0, 0, 0),
                SourceLocation("generated", 0, 0, 0)
            )
        )
        
        statements.append(backward_call)
        
        return statements
    
    def generate_gradient_ir(self, func_ir: IRFunction, 
                           variables: List[str]) -> IRFunction:
        """Generate IR for gradient computation."""
        # Create gradient function name
        grad_name = f"{func_ir.name}_grad"
        
        # Create new function type (returns gradients)
        param_types = [param.type for param in func_ir.parameters]
        grad_return_type = IRTensorType(
            IRPrimitiveType(IRDataType.F64), 
            [len(variables)]  # Vector of gradients
        )
        grad_func_type = IRFunctionType(grad_return_type, param_types)
        
        # Create gradient function
        grad_func = IRFunction(grad_name, grad_func_type)
        
        # Create entry block
        entry_block = IRBasicBlock("entry")
        grad_func.add_basic_block(entry_block)
        
        # Add gradient computation logic (simplified)
        # In a real implementation, this would implement reverse-mode AD
        
        # Allocate gradient storage
        grad_alloca = IRAlloca(grad_return_type)
        entry_block.add_instruction(grad_alloca)
        
        # Compute gradients (placeholder - would be actual AD logic)
        for i, var_name in enumerate(variables):
            # This would contain the actual gradient computation
            pass
        
        # Return gradients
        grad_load = IRLoad(grad_alloca.result, grad_return_type)
        entry_block.add_instruction(grad_load)
        
        grad_return = IRReturn(grad_load.result)
        entry_block.add_instruction(grad_return)
        
        return grad_func
    
    def compute_jacobian(self, func_def: FunctionDef, 
                        input_vars: List[str], 
                        output_vars: List[str]) -> FunctionDef:
        """Compute Jacobian matrix of a function."""
        jacobian_name = f"{func_def.name}_jacobian"
        
        # For each output variable, compute derivatives w.r.t. all input variables
        derivatives = []
        for out_var in output_vars:
            for in_var in input_vars:
                derivatives.append(Derivative(f"{out_var}_{in_var}"))
        
        # Generate Jacobian computation function
        jacobian_func = self.differentiate_function(func_def, derivatives)
        jacobian_func.name = jacobian_name
        
        return jacobian_func
    
    def compute_hessian(self, func_def: FunctionDef, 
                       variables: List[str]) -> FunctionDef:
        """Compute Hessian matrix (second-order derivatives)."""
        hessian_name = f"{func_def.name}_hessian"
        
        # Compute all second-order partial derivatives
        derivatives = []
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if j >= i:  # Hessian is symmetric, compute upper triangle
                    derivatives.append(Derivative(f"{var1}_{var2}", order=2))
        
        hessian_func = self.differentiate_function(func_def, derivatives)
        hessian_func.name = hessian_name
        
        return hessian_func
    
    def set_mode(self, mode: ADMode):
        """Set the automatic differentiation mode."""
        self.mode = mode
    
    def create_differentiable_version(self, module: IRModule) -> IRModule:
        """Create a differentiable version of an IR module."""
        diff_module = IRModule(f"{module.name}_diff")
        
        # Copy original functions
        for func_name, func in module.functions.items():
            diff_module.add_function(func)
            
            # Generate gradient function
            if not func_name.endswith("_grad"):
                grad_func = self.generate_gradient_ir(func, ["x", "y"])  # Example variables
                diff_module.add_function(grad_func)
        
        return diff_module


# Utility functions for AD

def gradient(func: callable, variables: List[str], mode: ADMode = ADMode.REVERSE):
    """
    Decorator to automatically generate gradient functions.
    
    @gradient(variables=["x", "y"])
    def f(x, y):
        return x * x + y * y
    """
    def decorator(original_func):
        engine = AutodiffEngine()
        engine.set_mode(mode)
        
        # This would integrate with the compiler pipeline
        # to generate differentiated versions
        
        def gradient_wrapper(*args, **kwargs):
            # Execute forward pass and compute gradients
            result = original_func(*args, **kwargs)
            
            # Compute and return gradients
            gradients = {}
            for var in variables:
                # Placeholder - would compute actual gradient
                gradients[var] = 0.0
            
            return result, gradients
        
        return gradient_wrapper
    
    return decorator


def jacobian(func: callable, input_vars: List[str], output_vars: List[str]):
    """Decorator to compute Jacobian matrix."""
    def decorator(original_func):
        def jacobian_wrapper(*args, **kwargs):
            result = original_func(*args, **kwargs)
            
            # Compute Jacobian (placeholder)
            jac_matrix = [[0.0 for _ in input_vars] for _ in output_vars]
            
            return result, jac_matrix
        
        return jacobian_wrapper
    
    return decorator


def hessian(func: callable, variables: List[str]):
    """Decorator to compute Hessian matrix."""
    def decorator(original_func):
        def hessian_wrapper(*args, **kwargs):
            result = original_func(*args, **kwargs)
            
            # Compute Hessian (placeholder)
            n = len(variables)
            hess_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
            
            return result, hess_matrix
        
        return hessian_wrapper
    
    return decorator
