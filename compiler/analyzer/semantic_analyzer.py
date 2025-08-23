"""
Main semantic analyzer for NeuralScript.

Coordinates all semantic analysis passes including:
- Symbol table construction
- Type checking and inference  
- Dimensional analysis
- Tensor shape verification
- Borrow checking
- Compile-time evaluation

Author: xwest
"""

from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass
import copy

from ..lexer.tokens import SourceLocation
from ..parser.ast_nodes import *
from .symbol_table import SymbolTable, Symbol, SymbolType, SymbolKind, Visibility, ScopeKind
from .errors import (
    SemanticError, SemanticWarning, create_type_mismatch_error,
    create_undefined_symbol_error, create_dimensional_mismatch_error,
    create_shape_mismatch_error, create_arity_mismatch_error
)


@dataclass
class AnalysisResult:
    """Results of semantic analysis."""
    ast: Program
    symbol_table: SymbolTable
    errors: List[SemanticError]
    warnings: List[SemanticWarning]
    type_annotations: Dict[ASTNode, SymbolType]  # Type information for AST nodes
    
    def has_errors(self) -> bool:
        """Check if analysis found any errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if analysis found any warnings."""
        return len(self.warnings) > 0


class SemanticAnalyzer:
    """
    Main semantic analyzer for NeuralScript.
    
    Performs comprehensive semantic analysis on the AST including:
    - Symbol resolution and scope checking
    - Type inference and checking
    - Dimensional analysis for units
    - Tensor shape verification
    - Memory safety analysis (borrow checking)
    - Compile-time constant evaluation
    """
    
    def __init__(self):
        """Initialize the semantic analyzer."""
        self.symbol_table = SymbolTable()
        self.errors: List[SemanticError] = []
        self.warnings: List[SemanticWarning] = []
        self.type_annotations: Dict[ASTNode, SymbolType] = {}
        
        # Analysis state
        self.current_function_return_type: Optional[SymbolType] = None
        self.generic_substitutions: Dict[str, SymbolType] = {}
        
        # Initialize built-in type mappings
        self._init_builtin_types()
    
    def _init_builtin_types(self):
        """Initialize built-in type mappings."""
        self.builtin_types = {
            "i8": SymbolType("i8", "primitive"),
            "i16": SymbolType("i16", "primitive"), 
            "i32": SymbolType("i32", "primitive"),
            "i64": SymbolType("i64", "primitive"),
            "u8": SymbolType("u8", "primitive"),
            "u16": SymbolType("u16", "primitive"),
            "u32": SymbolType("u32", "primitive"),
            "u64": SymbolType("u64", "primitive"),
            "f16": SymbolType("f16", "primitive"),
            "f32": SymbolType("f32", "primitive"),
            "f64": SymbolType("f64", "primitive"),
            "c32": SymbolType("c32", "primitive"),
            "c64": SymbolType("c64", "primitive"),
            "bool": SymbolType("bool", "primitive"),
            "char": SymbolType("char", "primitive"),
            "str": SymbolType("str", "primitive"),
        }
    
    def analyze(self, ast: Program) -> AnalysisResult:
        """
        Perform complete semantic analysis on the AST.
        
        Args:
            ast: The abstract syntax tree to analyze
            
        Returns:
            AnalysisResult containing analysis results and any errors/warnings
        """
        try:
            # Multi-pass analysis
            self._analyze_pass1_symbol_collection(ast)
            self._analyze_pass2_type_checking(ast)
            self._analyze_pass3_advanced_checking(ast)
            
        except SemanticError as e:
            self.errors.append(e)
        
        return AnalysisResult(
            ast=ast,
            symbol_table=self.symbol_table,
            errors=self.errors,
            warnings=self.warnings,
            type_annotations=self.type_annotations
        )
    
    def _analyze_pass1_symbol_collection(self, ast: Program):
        """
        Pass 1: Collect all symbols and build symbol table.
        
        This pass collects function signatures, type definitions, and 
        variable declarations to enable forward references.
        """
        for item in ast.items:
            try:
                self._collect_item_symbols(item)
            except SemanticError as e:
                self.errors.append(e)
    
    def _analyze_pass2_type_checking(self, ast: Program):
        """
        Pass 2: Perform type checking and inference.
        
        This pass validates types, infers missing types, and checks
        type compatibility throughout the program.
        """
        for item in ast.items:
            try:
                self._check_item_types(item)
            except SemanticError as e:
                self.errors.append(e)
    
    def _analyze_pass3_advanced_checking(self, ast: Program):
        """
        Pass 3: Advanced semantic checking.
        
        This pass performs dimensional analysis, tensor shape checking,
        borrow checking, and other advanced semantic validations.
        """
        for item in ast.items:
            try:
                self._check_item_advanced(item)
            except SemanticError as e:
                self.errors.append(e)
    
    # ========================================================================
    # Pass 1: Symbol Collection
    # ========================================================================
    
    def _collect_item_symbols(self, item: Item):
        """Collect symbols from a top-level item."""
        if isinstance(item, FunctionDef):
            self._collect_function_symbols(item)
        elif isinstance(item, StructDef):
            self._collect_struct_symbols(item)
        elif isinstance(item, TraitDef):
            self._collect_trait_symbols(item)
        # Add other item types as needed
    
    def _collect_function_symbols(self, func: FunctionDef):
        """Collect symbols from a function definition."""
        # Create function type
        param_types = []
        for param in func.params:
            if param.type_annotation:
                param_type = self._resolve_type_reference(param.type_annotation)
                param_types.append(param_type)
            else:
                # Will be inferred in pass 2
                param_types.append(SymbolType("unknown", "infer"))
        
        return_type = None
        if func.return_type:
            return_type = self._resolve_type_reference(func.return_type)
        else:
            return_type = SymbolType("unknown", "infer")
        
        func_type = SymbolType("function", "function", [return_type] + param_types)
        
        # Add function to symbol table
        self.symbol_table.define_function(
            func.name, 
            func_type, 
            func.span.start,
            generic_params=[tp.name for tp in func.type_params] if func.type_params else None
        )
    
    def _collect_struct_symbols(self, struct: StructDef):
        """Collect symbols from a struct definition."""
        # Create struct type
        field_types = []
        for field in struct.fields:
            field_type = self._resolve_type_reference(field.type_annotation)
            field_types.append(field_type)
        
        struct_type = SymbolType("struct", "composite", field_types)
        
        # Add struct to symbol table
        self.symbol_table.define_type(
            struct.name,
            struct_type,
            struct.span.start,
            generic_params=[tp.name for tp in struct.type_params] if struct.type_params else None
        )
    
    def _collect_trait_symbols(self, trait: TraitDef):
        """Collect symbols from a trait definition."""
        trait_type = SymbolType("trait", "trait")
        
        # Add trait to symbol table
        self.symbol_table.define_type(
            trait.name,
            trait_type,
            trait.span.start,
            generic_params=[tp.name for tp in trait.type_params] if trait.type_params else None
        )
    
    # ========================================================================
    # Pass 2: Type Checking
    # ========================================================================
    
    def _check_item_types(self, item: Item):
        """Check types for a top-level item."""
        if isinstance(item, FunctionDef):
            self._check_function_types(item)
        elif isinstance(item, StructDef):
            self._check_struct_types(item)
        elif isinstance(item, TraitDef):
            self._check_trait_types(item)
    
    def _check_function_types(self, func: FunctionDef):
        """Check types in a function definition."""
        # Enter function scope
        func_scope = self.symbol_table.enter_scope(ScopeKind.FUNCTION, func.name)
        
        try:
            # Add generic parameters
            if func.type_params:
                for type_param in func.type_params:
                    self.symbol_table.add_generic_param(type_param.name)
            
            # Add parameters to scope
            for param in func.params:
                param_type = self._resolve_type_reference(param.type_annotation) if param.type_annotation else None
                if param_type is None:
                    # Try to infer from usage (simplified)
                    param_type = SymbolType("unknown", "infer")
                
                self.symbol_table.define_variable(
                    param.name, 
                    param_type, 
                    func.span.start,
                    is_mutable=param.is_mutable
                )
            
            # Set return type for checking return statements
            if func.return_type:
                self.current_function_return_type = self._resolve_type_reference(func.return_type)
            else:
                self.current_function_return_type = SymbolType("void", "primitive")
            
            # Check function body
            self._check_statement_types(func.body)
            
        finally:
            # Exit function scope
            self.symbol_table.exit_scope()
            self.current_function_return_type = None
    
    def _check_struct_types(self, struct: StructDef):
        """Check types in a struct definition."""
        # Enter struct scope
        struct_scope = self.symbol_table.enter_scope(ScopeKind.STRUCT, struct.name)
        
        try:
            # Add generic parameters
            if struct.type_params:
                for type_param in struct.type_params:
                    self.symbol_table.add_generic_param(type_param.name)
            
            # Check field types
            for field in struct.fields:
                field_type = self._resolve_type_reference(field.type_annotation)
                # Field types are already collected, just validate them here
                
        finally:
            # Exit struct scope
            self.symbol_table.exit_scope()
    
    def _check_trait_types(self, trait: TraitDef):
        """Check types in a trait definition."""
        # Enter trait scope
        trait_scope = self.symbol_table.enter_scope(ScopeKind.TRAIT, trait.name)
        
        try:
            # Add generic parameters
            if trait.type_params:
                for type_param in trait.type_params:
                    self.symbol_table.add_generic_param(type_param.name)
            
            # Check method signatures
            for method in trait.methods:
                self._check_function_types(method)
                
        finally:
            # Exit trait scope
            self.symbol_table.exit_scope()
    
    def _check_statement_types(self, stmt: Statement):
        """Check types in a statement."""
        if isinstance(stmt, VariableDecl):
            self._check_variable_decl_types(stmt)
        elif isinstance(stmt, BlockStatement):
            self._check_block_statement_types(stmt)
        elif isinstance(stmt, IfStatement):
            self._check_if_statement_types(stmt)
        elif isinstance(stmt, WhileLoop):
            self._check_while_loop_types(stmt)
        elif isinstance(stmt, ForLoop):
            self._check_for_loop_types(stmt)
        elif isinstance(stmt, ReturnStatement):
            self._check_return_statement_types(stmt)
        elif isinstance(stmt, Expression):
            # Expression statements
            self._check_expression_types(stmt)
    
    def _check_variable_decl_types(self, var_decl: VariableDecl):
        """Check types in a variable declaration."""
        var_type = None
        
        if var_decl.type_annotation:
            var_type = self._resolve_type_reference(var_decl.type_annotation)
        
        if var_decl.initializer:
            init_type = self._check_expression_types(var_decl.initializer)
            
            if var_type is None:
                # Infer type from initializer
                var_type = init_type
            else:
                # Check type compatibility
                if not self._types_compatible(var_type, init_type):
                    self.errors.append(create_type_mismatch_error(
                        str(var_type), str(init_type),
                        var_decl.span.start, var_decl
                    ))
        
        if var_type is None:
            var_type = SymbolType("unknown", "infer")
        
        # Add variable to symbol table
        self.symbol_table.define_variable(
            var_decl.name,
            var_type,
            var_decl.span.start,
            is_mutable=var_decl.is_mutable,
            is_const=var_decl.is_const
        )
        
        # Store type annotation
        self.type_annotations[var_decl] = var_type
    
    def _check_block_statement_types(self, block: BlockStatement):
        """Check types in a block statement."""
        # Enter new scope for block
        block_scope = self.symbol_table.enter_scope(ScopeKind.BLOCK, "block")
        
        try:
            for stmt in block.statements:
                self._check_statement_types(stmt)
        finally:
            self.symbol_table.exit_scope()
    
    def _check_if_statement_types(self, if_stmt: IfStatement):
        """Check types in an if statement."""
        # Check condition type
        condition_type = self._check_expression_types(if_stmt.condition)
        
        # Condition should be boolean
        bool_type = SymbolType("bool", "primitive")
        if not self._types_compatible(bool_type, condition_type):
            self.errors.append(create_type_mismatch_error(
                "bool", str(condition_type),
                if_stmt.condition.span.start, if_stmt.condition
            ))
        
        # Check branches
        self._check_statement_types(if_stmt.then_branch)
        if if_stmt.else_branch:
            self._check_statement_types(if_stmt.else_branch)
    
    def _check_while_loop_types(self, while_loop: WhileLoop):
        """Check types in a while loop."""
        # Check condition type
        condition_type = self._check_expression_types(while_loop.condition)
        
        # Condition should be boolean
        bool_type = SymbolType("bool", "primitive")
        if not self._types_compatible(bool_type, condition_type):
            self.errors.append(create_type_mismatch_error(
                "bool", str(condition_type),
                while_loop.condition.span.start, while_loop.condition
            ))
        
        # Enter loop scope and check body
        loop_scope = self.symbol_table.enter_scope(ScopeKind.LOOP, "while")
        try:
            self._check_statement_types(while_loop.body)
        finally:
            self.symbol_table.exit_scope()
    
    def _check_for_loop_types(self, for_loop: ForLoop):
        """Check types in a for loop."""
        # Check iterable type
        iterable_type = self._check_expression_types(for_loop.iterable)
        
        # TODO: Check if iterable_type is actually iterable
        # For now, assume element type is the same as iterable type
        element_type = iterable_type
        
        # Enter loop scope
        loop_scope = self.symbol_table.enter_scope(ScopeKind.LOOP, "for")
        try:
            # Add loop variable
            self.symbol_table.define_variable(
                for_loop.variable,
                element_type,
                for_loop.span.start
            )
            
            # Check body
            self._check_statement_types(for_loop.body)
        finally:
            self.symbol_table.exit_scope()
    
    def _check_return_statement_types(self, return_stmt: ReturnStatement):
        """Check types in a return statement."""
        if return_stmt.value:
            return_type = self._check_expression_types(return_stmt.value)
        else:
            return_type = SymbolType("void", "primitive")
        
        # Check against function return type
        if self.current_function_return_type:
            if not self._types_compatible(self.current_function_return_type, return_type):
                self.errors.append(create_type_mismatch_error(
                    str(self.current_function_return_type), str(return_type),
                    return_stmt.span.start, return_stmt
                ))
    
    def _check_expression_types(self, expr: Expression) -> SymbolType:
        """Check types in an expression and return the expression's type."""
        if isinstance(expr, Literal):
            return self._check_literal_types(expr)
        elif isinstance(expr, Identifier):
            return self._check_identifier_types(expr)
        elif isinstance(expr, BinaryOp):
            return self._check_binary_op_types(expr)
        elif isinstance(expr, UnaryOp):
            return self._check_unary_op_types(expr)
        elif isinstance(expr, FunctionCall):
            return self._check_function_call_types(expr)
        elif isinstance(expr, TensorLiteral):
            return self._check_tensor_literal_types(expr)
        elif isinstance(expr, UnitLiteral):
            return self._check_unit_literal_types(expr)
        elif isinstance(expr, ComplexLiteral):
            return self._check_complex_literal_types(expr)
        else:
            # Unknown expression type
            return SymbolType("unknown", "primitive")
    
    def _check_literal_types(self, literal: Literal) -> SymbolType:
        """Check types for a literal expression."""
        type_map = {
            "integer": "i32",  # Default integer type
            "float": "f64",    # Default float type
            "string": "str",
            "character": "char",
            "boolean": "bool"
        }
        
        type_name = type_map.get(literal.literal_type, "unknown")
        symbol_type = SymbolType(type_name, "primitive")
        self.type_annotations[literal] = symbol_type
        return symbol_type
    
    def _check_identifier_types(self, identifier: Identifier) -> SymbolType:
        """Check types for an identifier expression."""
        try:
            symbol = self.symbol_table.lookup_symbol(identifier.name, identifier.span.start)
            self.type_annotations[identifier] = symbol.symbol_type
            return symbol.symbol_type
        except SemanticError as e:
            self.errors.append(e)
            unknown_type = SymbolType("unknown", "primitive")
            self.type_annotations[identifier] = unknown_type
            return unknown_type
    
    def _check_binary_op_types(self, binary_op: BinaryOp) -> SymbolType:
        """Check types for a binary operation."""
        left_type = self._check_expression_types(binary_op.left)
        right_type = self._check_expression_types(binary_op.right)
        
        result_type = self._infer_binary_op_result_type(
            binary_op.operator, left_type, right_type
        )
        
        if result_type is None:
            self.errors.append(SemanticError(
                f"Invalid operation: {left_type} {binary_op.operator} {right_type}",
                binary_op.span.start,
                binary_op,
                code="S005"
            ))
            result_type = SymbolType("unknown", "primitive")
        
        self.type_annotations[binary_op] = result_type
        return result_type
    
    def _check_unary_op_types(self, unary_op: UnaryOp) -> SymbolType:
        """Check types for a unary operation."""
        operand_type = self._check_expression_types(unary_op.operand)
        
        result_type = self._infer_unary_op_result_type(
            unary_op.operator, operand_type
        )
        
        if result_type is None:
            self.errors.append(SemanticError(
                f"Invalid operation: {unary_op.operator}{operand_type}",
                unary_op.span.start,
                unary_op,
                code="S005"
            ))
            result_type = SymbolType("unknown", "primitive")
        
        self.type_annotations[unary_op] = result_type
        return result_type
    
    def _check_function_call_types(self, call: FunctionCall) -> SymbolType:
        """Check types for a function call."""
        # Get function type
        func_type = self._check_expression_types(call.function)
        
        # Check argument types
        arg_types = []
        for arg in call.args:
            arg_type = self._check_expression_types(arg)
            arg_types.append(arg_type)
        
        # For now, assume the function returns the first parameter type
        # In a full implementation, this would properly handle function signatures
        if func_type.parameters:
            result_type = func_type.parameters[0]  # Return type is first parameter
        else:
            result_type = SymbolType("unknown", "primitive")
        
        self.type_annotations[call] = result_type
        return result_type
    
    def _check_tensor_literal_types(self, tensor: TensorLiteral) -> SymbolType:
        """Check types for a tensor literal."""
        # Determine element type from first element
        if tensor.elements and tensor.elements[0]:
            first_element_type = self._check_expression_types(tensor.elements[0][0])
        else:
            first_element_type = SymbolType("f64", "primitive")  # Default
        
        # Create tensor type with dimensions
        tensor_type = SymbolType(
            "Tensor",
            "tensor",
            [first_element_type],  # Element type as parameter
        )
        
        self.type_annotations[tensor] = tensor_type
        return tensor_type
    
    def _check_unit_literal_types(self, unit: UnitLiteral) -> SymbolType:
        """Check types for a unit literal."""
        # Unit literals have the base type plus unit information
        if isinstance(unit.value, int):
            base_type = SymbolType("i32", "primitive")
        else:
            base_type = SymbolType("f64", "primitive")
        
        unit_type = SymbolType(f"{base_type.name}_{unit.unit}", "unit", [base_type])
        self.type_annotations[unit] = unit_type
        return unit_type
    
    def _check_complex_literal_types(self, complex_lit: ComplexLiteral) -> SymbolType:
        """Check types for a complex literal."""
        complex_type = SymbolType("c64", "primitive")  # Default complex type
        self.type_annotations[complex_lit] = complex_type
        return complex_type
    
    # ========================================================================
    # Pass 3: Advanced Checking
    # ========================================================================
    
    def _check_item_advanced(self, item: Item):
        """Perform advanced checking on items."""
        # For now, just placeholder
        pass
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _resolve_type_reference(self, type_ref: TypeRef) -> SymbolType:
        """Resolve a type reference to a SymbolType."""
        if isinstance(type_ref, SimpleTypeRef):
            # Check if it's a built-in type
            if type_ref.name in self.builtin_types:
                return self.builtin_types[type_ref.name]
            
            # Look up in symbol table
            try:
                return self.symbol_table.lookup_type(type_ref.name, type_ref.span.start)
            except SemanticError as e:
                self.errors.append(e)
                return SymbolType("unknown", "primitive")
        
        elif isinstance(type_ref, TensorTypeRef):
            # Handle tensor types
            element_type = self._resolve_type_reference(type_ref.element_type)
            
            # Extract dimension information (simplified)
            dimensions = []
            for dim_expr in type_ref.dimensions:
                if isinstance(dim_expr, Literal) and dim_expr.literal_type == "integer":
                    dimensions.append(dim_expr.value)
                else:
                    dimensions.append(-1)  # Unknown dimension
            
            return SymbolType("Tensor", "tensor", [element_type])
        
        return SymbolType("unknown", "primitive")
    
    def _types_compatible(self, expected: SymbolType, actual: SymbolType) -> bool:
        """Check if two types are compatible."""
        # Simplified type compatibility check
        if expected.name == actual.name:
            return True
        
        # Allow unknown types to be compatible with anything (for inference)
        if expected.name == "unknown" or actual.name == "unknown":
            return True
        
        # Allow numeric type promotion (simplified)
        numeric_types = {"i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f16", "f32", "f64"}
        if expected.name in numeric_types and actual.name in numeric_types:
            return True
        
        return False
    
    def _infer_binary_op_result_type(self, operator: str, left: SymbolType, right: SymbolType) -> Optional[SymbolType]:
        """Infer the result type of a binary operation."""
        # Simplified binary operation type inference
        arithmetic_ops = {"+", "-", "*", "/", "%", "**", "⊙", "⊗", "⋅", "×"}
        comparison_ops = {"==", "!=", "<", ">", "<=", ">=", "≡", "≠", "≤", "≥", "≈"}
        logical_ops = {"&&", "||", "∧", "∨", "⊕", "↔"}
        
        if operator in arithmetic_ops:
            # Arithmetic operations preserve the "larger" type
            if left.name == "f64" or right.name == "f64":
                return SymbolType("f64", "primitive")
            elif left.name == "f32" or right.name == "f32":
                return SymbolType("f32", "primitive")
            else:
                return SymbolType("i32", "primitive")
        
        elif operator in comparison_ops:
            return SymbolType("bool", "primitive")
        
        elif operator in logical_ops:
            return SymbolType("bool", "primitive")
        
        elif operator == "=":
            # Assignment returns the right-hand type
            return right
        
        return None
    
    def _infer_unary_op_result_type(self, operator: str, operand: SymbolType) -> Optional[SymbolType]:
        """Infer the result type of a unary operation."""
        if operator in {"-", "+"}:
            # Numeric unary operations preserve type
            return operand
        elif operator in {"!", "¬"}:
            # Logical not returns boolean
            return SymbolType("bool", "primitive")
        elif operator in {"∇", "∂"}:
            # Gradient operations (simplified)
            return operand
        elif operator in {"√", "∛"}:
            # Root operations return floating point
            return SymbolType("f64", "primitive")
        
        return None
