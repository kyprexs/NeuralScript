"""
IR Generator for NeuralScript.

Converts a semantically analyzed AST into NeuralScript IR (NS-IR).
Handles lowering of high-level constructs into IR form.

Author: xwest
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
import uuid

from ..parser.ast_nodes import *
from ..analyzer.semantic_analyzer import AnalysisResult, SymbolType
from .ir_nodes import *


@dataclass
class IRGenContext:
    """Context for IR generation."""
    current_function: Optional[IRFunction] = None
    current_block: Optional[IRBasicBlock] = None
    value_map: Dict[str, IRValue] = None  # Maps AST names to IR values
    type_map: Dict[SymbolType, IRType] = None  # Maps semantic types to IR types
    break_target: Optional[IRBasicBlock] = None
    continue_target: Optional[IRBasicBlock] = None
    
    def __post_init__(self):
        if self.value_map is None:
            self.value_map = {}
        if self.type_map is None:
            self.type_map = {}


class IRGenerator:
    """
    Generates NeuralScript IR from a semantically analyzed AST.
    
    The generator performs the following transformations:
    - Converts high-level AST nodes to IR instructions
    - Handles control flow with basic blocks and branches
    - Maps semantic types to IR types
    - Generates SSA form with phi nodes
    - Handles tensor operations and mathematical functions
    """
    
    def __init__(self):
        """Initialize the IR generator."""
        self.module: Optional[IRModule] = None
        self.context = IRGenContext()
        
        # Initialize type mappings
        self._init_type_mappings()
    
    def _init_type_mappings(self):
        """Initialize mappings from semantic types to IR types."""
        self.primitive_type_map = {
            "i8": IRPrimitiveType(IRDataType.I8),
            "i16": IRPrimitiveType(IRDataType.I16),
            "i32": IRPrimitiveType(IRDataType.I32),
            "i64": IRPrimitiveType(IRDataType.I64),
            "f16": IRPrimitiveType(IRDataType.F16),
            "f32": IRPrimitiveType(IRDataType.F32),
            "f64": IRPrimitiveType(IRDataType.F64),
            "c32": IRPrimitiveType(IRDataType.C32),
            "c64": IRPrimitiveType(IRDataType.C64),
            "bool": IRPrimitiveType(IRDataType.I1),
            "void": IRPrimitiveType(IRDataType.VOID),
            "ptr": IRPrimitiveType(IRDataType.PTR),
        }
    
    def generate(self, analysis_result: AnalysisResult) -> IRModule:
        """
        Generate IR from a semantically analyzed AST.
        
        Args:
            analysis_result: Results from semantic analysis
            
        Returns:
            IRModule containing the generated IR
        """
        self.module = IRModule("main")
        self.analysis_result = analysis_result
        
        # Build type mappings from semantic analysis
        self._build_type_mappings(analysis_result)
        
        # Generate IR for all top-level items
        for item in analysis_result.ast.items:
            self._generate_item(item)
        
        return self.module
    
    def _build_type_mappings(self, analysis_result: AnalysisResult):
        """Build mappings from semantic types to IR types."""
        # Map all annotated types from the analysis
        for ast_node, symbol_type in analysis_result.type_annotations.items():
            ir_type = self._convert_semantic_type_to_ir(symbol_type)
            self.context.type_map[symbol_type] = ir_type
    
    def _convert_semantic_type_to_ir(self, semantic_type: SymbolType) -> IRType:
        """Convert a semantic type to an IR type."""
        if semantic_type.name in self.primitive_type_map:
            return self.primitive_type_map[semantic_type.name]
        
        elif semantic_type.kind == "tensor":
            # Extract element type and shape from tensor type
            if semantic_type.parameters:
                element_type = self._convert_semantic_type_to_ir(semantic_type.parameters[0])
                # For now, use dynamic shape - in a full implementation we'd extract actual shape
                return IRTensorType(element_type, [])
            else:
                return IRTensorType(IRPrimitiveType(IRDataType.F32), [])
        
        elif semantic_type.kind == "function":
            # Convert function type
            if semantic_type.parameters:
                return_type = self._convert_semantic_type_to_ir(semantic_type.parameters[0])
                param_types = [self._convert_semantic_type_to_ir(param) for param in semantic_type.parameters[1:]]
                return IRFunctionType(return_type, param_types)
            else:
                return IRFunctionType(IRPrimitiveType(IRDataType.VOID), [])
        
        else:
            # Default to void for unknown types
            return IRPrimitiveType(IRDataType.VOID)
    
    def _generate_item(self, item: Item):
        """Generate IR for a top-level item."""
        if isinstance(item, FunctionDef):
            self._generate_function(item)
        elif isinstance(item, StructDef):
            self._generate_struct(item)
        # Add other item types as needed
    
    def _generate_function(self, func_def: FunctionDef):
        """Generate IR for a function definition."""
        # Get function type from semantic analysis
        func_symbol = self.analysis_result.symbol_table.lookup_symbol(func_def.name, func_def.span.start)
        func_ir_type = self._convert_semantic_type_to_ir(func_symbol.symbol_type)
        
        # Create IR function
        ir_function = IRFunction(func_def.name, func_ir_type)
        self.module.add_function(ir_function)
        
        # Set context
        old_function = self.context.current_function
        old_block = self.context.current_block
        self.context.current_function = ir_function
        
        # Create entry block
        entry_block = IRBasicBlock("entry")
        ir_function.add_basic_block(entry_block)
        self.context.current_block = entry_block
        
        # Map parameters to IR values
        for i, param in enumerate(func_def.params):
            if i < len(ir_function.parameters):
                self.context.value_map[param.name] = ir_function.parameters[i]
        
        # Generate function body
        self._generate_statement(func_def.body)
        
        # Ensure function is properly terminated
        if not self.context.current_block.instructions or \
           not isinstance(self.context.current_block.instructions[-1], (IRReturn, IRBranch, IRCondBranch)):
            # Add default return
            if func_ir_type.return_type.data_type == IRDataType.VOID:
                self.context.current_block.add_instruction(IRReturn())
            else:
                # Return zero/null value
                zero_value = self._create_zero_constant(func_ir_type.return_type)
                self.context.current_block.add_instruction(IRReturn(zero_value))
        
        # Restore context
        self.context.current_function = old_function
        self.context.current_block = old_block
    
    def _generate_struct(self, struct_def: StructDef):
        """Generate IR for a struct definition."""
        # Structs are handled by type system - no separate IR generation needed
        pass
    
    def _generate_statement(self, stmt: Statement):
        """Generate IR for a statement."""
        if isinstance(stmt, VariableDecl):
            self._generate_variable_decl(stmt)
        elif isinstance(stmt, BlockStatement):
            self._generate_block_statement(stmt)
        elif isinstance(stmt, IfStatement):
            self._generate_if_statement(stmt)
        elif isinstance(stmt, WhileLoop):
            self._generate_while_loop(stmt)
        elif isinstance(stmt, ForLoop):
            self._generate_for_loop(stmt)
        elif isinstance(stmt, ReturnStatement):
            self._generate_return_statement(stmt)
        elif isinstance(stmt, Expression):
            # Expression statement - evaluate and discard result
            self._generate_expression(stmt)
    
    def _generate_variable_decl(self, var_decl: VariableDecl):
        """Generate IR for a variable declaration."""
        # Get variable type
        var_type = self.analysis_result.type_annotations.get(var_decl)
        if var_type is None:
            var_type = SymbolType("i32", "primitive")  # Default
        
        ir_type = self._convert_semantic_type_to_ir(var_type)
        
        # Allocate stack space
        alloca_instr = IRAlloca(ir_type)
        self.context.current_block.add_instruction(alloca_instr)
        
        # Store pointer in value map
        self.context.value_map[var_decl.name] = alloca_instr.result
        
        # Generate initializer if present
        if var_decl.initializer:
            init_value = self._generate_expression(var_decl.initializer)
            store_instr = IRStore(init_value, alloca_instr.result)
            self.context.current_block.add_instruction(store_instr)
    
    def _generate_block_statement(self, block: BlockStatement):
        """Generate IR for a block statement."""
        for stmt in block.statements:
            self._generate_statement(stmt)
    
    def _generate_if_statement(self, if_stmt: IfStatement):
        """Generate IR for an if statement."""
        # Generate condition
        condition = self._generate_expression(if_stmt.condition)
        
        # Create basic blocks
        then_block = IRBasicBlock(f"if_then_{uuid.uuid4().hex[:8]}")
        else_block = IRBasicBlock(f"if_else_{uuid.uuid4().hex[:8]}")
        merge_block = IRBasicBlock(f"if_merge_{uuid.uuid4().hex[:8]}")
        
        self.context.current_function.add_basic_block(then_block)
        self.context.current_function.add_basic_block(else_block)
        self.context.current_function.add_basic_block(merge_block)
        
        # Conditional branch
        cond_br = IRCondBranch(condition, then_block, else_block)
        self.context.current_block.add_instruction(cond_br)
        
        # Generate then branch
        self.context.current_block = then_block
        self._generate_statement(if_stmt.then_branch)
        
        # Branch to merge if not already terminated
        if not self.context.current_block.instructions or \
           not isinstance(self.context.current_block.instructions[-1], (IRReturn, IRBranch, IRCondBranch)):
            self.context.current_block.add_instruction(IRBranch(merge_block))
        
        # Generate else branch
        self.context.current_block = else_block
        if if_stmt.else_branch:
            self._generate_statement(if_stmt.else_branch)
        
        # Branch to merge if not already terminated
        if not self.context.current_block.instructions or \
           not isinstance(self.context.current_block.instructions[-1], (IRReturn, IRBranch, IRCondBranch)):
            self.context.current_block.add_instruction(IRBranch(merge_block))
        
        # Continue with merge block
        self.context.current_block = merge_block
    
    def _generate_while_loop(self, while_loop: WhileLoop):
        """Generate IR for a while loop."""
        # Create basic blocks
        header_block = IRBasicBlock(f"while_header_{uuid.uuid4().hex[:8]}")
        body_block = IRBasicBlock(f"while_body_{uuid.uuid4().hex[:8]}")
        exit_block = IRBasicBlock(f"while_exit_{uuid.uuid4().hex[:8]}")
        
        self.context.current_function.add_basic_block(header_block)
        self.context.current_function.add_basic_block(body_block)
        self.context.current_function.add_basic_block(exit_block)
        
        # Branch to header
        self.context.current_block.add_instruction(IRBranch(header_block))
        
        # Generate header (condition check)
        self.context.current_block = header_block
        condition = self._generate_expression(while_loop.condition)
        cond_br = IRCondBranch(condition, body_block, exit_block)
        self.context.current_block.add_instruction(cond_br)
        
        # Generate body
        old_break = self.context.break_target
        old_continue = self.context.continue_target
        self.context.break_target = exit_block
        self.context.continue_target = header_block
        
        self.context.current_block = body_block
        self._generate_statement(while_loop.body)
        
        # Branch back to header if not already terminated
        if not self.context.current_block.instructions or \
           not isinstance(self.context.current_block.instructions[-1], (IRReturn, IRBranch, IRCondBranch)):
            self.context.current_block.add_instruction(IRBranch(header_block))
        
        # Restore break/continue targets
        self.context.break_target = old_break
        self.context.continue_target = old_continue
        
        # Continue with exit block
        self.context.current_block = exit_block
    
    def _generate_for_loop(self, for_loop: ForLoop):
        """Generate IR for a for loop."""
        # For now, treat as while loop - full implementation would handle iterators
        # This is a simplified version
        
        # Create basic blocks
        header_block = IRBasicBlock(f"for_header_{uuid.uuid4().hex[:8]}")
        body_block = IRBasicBlock(f"for_body_{uuid.uuid4().hex[:8]}")
        exit_block = IRBasicBlock(f"for_exit_{uuid.uuid4().hex[:8]}")
        
        self.context.current_function.add_basic_block(header_block)
        self.context.current_function.add_basic_block(body_block)
        self.context.current_function.add_basic_block(exit_block)
        
        # Initialize loop variable (simplified)
        # In a full implementation, we'd generate proper iterator protocol
        
        # Branch to header
        self.context.current_block.add_instruction(IRBranch(header_block))
        
        # For now, just generate body once (simplified)
        self.context.current_block = body_block
        self._generate_statement(for_loop.body)
        
        # Branch to exit
        self.context.current_block.add_instruction(IRBranch(exit_block))
        
        # Continue with exit block
        self.context.current_block = exit_block
    
    def _generate_return_statement(self, return_stmt: ReturnStatement):
        """Generate IR for a return statement."""
        if return_stmt.value:
            return_value = self._generate_expression(return_stmt.value)
            self.context.current_block.add_instruction(IRReturn(return_value))
        else:
            self.context.current_block.add_instruction(IRReturn())
    
    def _generate_expression(self, expr: Expression) -> IRValue:
        """Generate IR for an expression and return the result value."""
        if isinstance(expr, Literal):
            return self._generate_literal(expr)
        elif isinstance(expr, Identifier):
            return self._generate_identifier(expr)
        elif isinstance(expr, BinaryOp):
            return self._generate_binary_op(expr)
        elif isinstance(expr, UnaryOp):
            return self._generate_unary_op(expr)
        elif isinstance(expr, FunctionCall):
            return self._generate_function_call(expr)
        elif isinstance(expr, TensorLiteral):
            return self._generate_tensor_literal(expr)
        elif isinstance(expr, UnitLiteral):
            return self._generate_unit_literal(expr)
        elif isinstance(expr, ComplexLiteral):
            return self._generate_complex_literal(expr)
        else:
            # Unknown expression - return zero constant
            return self._create_zero_constant(IRPrimitiveType(IRDataType.I32))
    
    def _generate_literal(self, literal: Literal) -> IRValue:
        """Generate IR for a literal."""
        if literal.literal_type == "integer":
            return IRConstant(IRPrimitiveType(IRDataType.I32), literal.value)
        elif literal.literal_type == "float":
            return IRConstant(IRPrimitiveType(IRDataType.F64), literal.value)
        elif literal.literal_type == "string":
            # Strings need more complex handling - simplified here
            return IRConstant(IRPrimitiveType(IRDataType.PTR), literal.value)
        elif literal.literal_type == "character":
            return IRConstant(IRPrimitiveType(IRDataType.I8), ord(literal.value))
        elif literal.literal_type == "boolean":
            return IRConstant(IRPrimitiveType(IRDataType.I1), 1 if literal.value else 0)
        else:
            return IRConstant(IRPrimitiveType(IRDataType.I32), 0)
    
    def _generate_identifier(self, identifier: Identifier) -> IRValue:
        """Generate IR for an identifier."""
        # Look up value in value map
        if identifier.name in self.context.value_map:
            value = self.context.value_map[identifier.name]
            
            # If it's a pointer (from alloca), load the value
            if isinstance(value.type, IRPrimitiveType) and value.type.data_type == IRDataType.PTR:
                # Need to determine the type to load - simplified here
                load_type = IRPrimitiveType(IRDataType.I32)  # Default
                load_instr = IRLoad(value, load_type)
                self.context.current_block.add_instruction(load_instr)
                return load_instr.result
            else:
                return value
        else:
            # Undefined identifier - return zero constant
            return self._create_zero_constant(IRPrimitiveType(IRDataType.I32))
    
    def _generate_binary_op(self, binary_op: BinaryOp) -> IRValue:
        """Generate IR for a binary operation."""
        left = self._generate_expression(binary_op.left)
        right = self._generate_expression(binary_op.right)
        
        # Map operator to IR instruction
        op_map = {
            "+": "add",
            "-": "sub", 
            "*": "mul",
            "/": "div",
            "%": "rem",
            "==": "eq",
            "!=": "ne",
            "<": "lt",
            "<=": "le",
            ">": "gt",
            ">=": "ge",
            # Add more operators as needed
        }
        
        ir_op = op_map.get(binary_op.operator, "add")  # Default to add
        
        # Determine result type - simplified
        result_type = left.type
        
        # Create instruction
        instr = IRBinaryOp(ir_op, left, right, result_type)
        self.context.current_block.add_instruction(instr)
        return instr.result
    
    def _generate_unary_op(self, unary_op: UnaryOp) -> IRValue:
        """Generate IR for a unary operation."""
        operand = self._generate_expression(unary_op.operand)
        
        # Map operator to IR instruction
        op_map = {
            "-": "neg",
            "!": "not",
            "+": "pos",  # Usually a no-op
        }
        
        ir_op = op_map.get(unary_op.operator, "neg")
        result_type = operand.type
        
        # Create instruction
        instr = IRUnaryOp(ir_op, operand, result_type)
        self.context.current_block.add_instruction(instr)
        return instr.result
    
    def _generate_function_call(self, call: FunctionCall) -> IRValue:
        """Generate IR for a function call."""
        # Generate function value
        func_value = self._generate_expression(call.function)
        
        # Generate arguments
        args = []
        for arg in call.args:
            arg_value = self._generate_expression(arg)
            args.append(arg_value)
        
        # Determine return type - simplified
        if isinstance(func_value.type, IRFunctionType):
            result_type = func_value.type.return_type
        else:
            result_type = IRPrimitiveType(IRDataType.VOID)
        
        # Create call instruction
        call_instr = IRCall(func_value, args, result_type if result_type.data_type != IRDataType.VOID else None)
        self.context.current_block.add_instruction(call_instr)
        return call_instr.result if call_instr.result else self._create_zero_constant(IRPrimitiveType(IRDataType.I32))
    
    def _generate_tensor_literal(self, tensor: TensorLiteral) -> IRValue:
        """Generate IR for a tensor literal."""
        # Simplified tensor creation
        # In a full implementation, we'd create proper tensor construction
        
        # For now, create a simple tensor type
        element_type = IRPrimitiveType(IRDataType.F32)  # Default
        tensor_type = IRTensorType(element_type, [len(tensor.elements)])
        
        # Create constant for now - real implementation would construct tensor
        return IRConstant(tensor_type, tensor.elements)
    
    def _generate_unit_literal(self, unit: UnitLiteral) -> IRValue:
        """Generate IR for a unit literal."""
        # For now, treat as regular numeric literal
        # Full implementation would track units in type system
        
        if isinstance(unit.value, int):
            return IRConstant(IRPrimitiveType(IRDataType.I32), unit.value)
        else:
            return IRConstant(IRPrimitiveType(IRDataType.F64), unit.value)
    
    def _generate_complex_literal(self, complex_lit: ComplexLiteral) -> IRValue:
        """Generate IR for a complex literal."""
        # Create complex constant
        return IRConstant(IRPrimitiveType(IRDataType.C64), (complex_lit.real, complex_lit.imag))
    
    def _create_zero_constant(self, ir_type: IRType) -> IRValue:
        """Create a zero constant of the given type."""
        if isinstance(ir_type, IRPrimitiveType):
            if ir_type.data_type in {IRDataType.I8, IRDataType.I16, IRDataType.I32, IRDataType.I64, IRDataType.I1}:
                return IRConstant(ir_type, 0)
            elif ir_type.data_type in {IRDataType.F16, IRDataType.F32, IRDataType.F64}:
                return IRConstant(ir_type, 0.0)
            elif ir_type.data_type in {IRDataType.C32, IRDataType.C64}:
                return IRConstant(ir_type, (0.0, 0.0))
            elif ir_type.data_type == IRDataType.PTR:
                return IRConstant(ir_type, None)
        
        # Default
        return IRConstant(IRPrimitiveType(IRDataType.I32), 0)
