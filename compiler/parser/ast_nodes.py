"""
Abstract Syntax Tree node definitions for NeuralScript.

Defines a comprehensive set of AST node types for representing NeuralScript programs.
Each node includes source location information and supports the visitor pattern.

Author: xwest
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Union, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid

from ..lexer.tokens import SourceLocation, Token, TokenType


class ASTNodeType(Enum):
    """Enumeration of all AST node types."""
    
    # Top-level
    PROGRAM = "Program"
    
    # Items (top-level declarations)
    FUNCTION_DEF = "FunctionDef"
    STRUCT_DEF = "StructDef"
    ENUM_DEF = "EnumDef"
    TRAIT_DEF = "TraitDef"
    IMPL_BLOCK = "ImplBlock"
    TYPE_ALIAS = "TypeAlias"
    MODULE = "Module"
    USE_STATEMENT = "UseStatement"
    
    # Statements
    VARIABLE_DECL = "VariableDecl"
    ASSIGNMENT = "Assignment"
    EXPRESSION_STMT = "ExpressionStatement"
    IF_STATEMENT = "IfStatement"
    WHILE_LOOP = "WhileLoop"
    FOR_LOOP = "ForLoop"
    LOOP_STATEMENT = "LoopStatement"
    MATCH_STATEMENT = "MatchStatement"
    RETURN_STATEMENT = "ReturnStatement"
    BREAK_STATEMENT = "BreakStatement"
    CONTINUE_STATEMENT = "ContinueStatement"
    BLOCK_STATEMENT = "BlockStatement"
    
    # Expressions
    BINARY_OP = "BinaryOp"
    UNARY_OP = "UnaryOp"
    FUNCTION_CALL = "FunctionCall"
    METHOD_CALL = "MethodCall"
    FIELD_ACCESS = "FieldAccess"
    INDEX_ACCESS = "IndexAccess"
    SLICE_ACCESS = "SliceAccess"
    TENSOR_ACCESS = "TensorAccess"
    
    # Literals
    LITERAL = "Literal"
    IDENTIFIER = "Identifier"
    ARRAY_LITERAL = "ArrayLiteral"
    TENSOR_LITERAL = "TensorLiteral"
    STRUCT_LITERAL = "StructLiteral"
    TUPLE_LITERAL = "TupleLiteral"
    UNIT_LITERAL = "UnitLiteral"
    COMPLEX_LITERAL = "ComplexLiteral"
    
    # Types
    TYPE_REF = "TypeRef"
    GENERIC_TYPE = "GenericType"
    ARRAY_TYPE = "ArrayType"
    TENSOR_TYPE = "TensorType"
    FUNCTION_TYPE = "FunctionType"
    TUPLE_TYPE = "TupleType"
    UNIT_TYPE = "UnitType"
    
    # Patterns (for match expressions)
    PATTERN = "Pattern"
    LITERAL_PATTERN = "LiteralPattern"
    IDENTIFIER_PATTERN = "IdentifierPattern"
    TUPLE_PATTERN = "TuplePattern"
    STRUCT_PATTERN = "StructPattern"
    WILDCARD_PATTERN = "WildcardPattern"
    
    # Mathematical constructs
    GRADIENT = "Gradient"
    PARTIAL_DERIVATIVE = "PartialDerivative"
    SUMMATION = "Summation"
    INTEGRAL = "Integral"
    
    # Concurrency
    ASYNC_BLOCK = "AsyncBlock"
    AWAIT_EXPRESSION = "AwaitExpression"
    ACTOR_DEF = "ActorDef"
    CHANNEL_SEND = "ChannelSend"
    CHANNEL_RECV = "ChannelRecv"


@dataclass
class SourceSpan:
    """Represents a span of source code (start and end locations)."""
    start: SourceLocation
    end: SourceLocation
    
    def __str__(self) -> str:
        if self.start.filename == self.end.filename:
            return f"{self.start.filename}:{self.start.line}:{self.start.column}-{self.end.line}:{self.end.column}"
        return f"{self.start}-{self.end}"


class ASTVisitor(ABC):
    """Abstract visitor interface for traversing AST nodes."""
    
    @abstractmethod
    def visit(self, node: 'ASTNode') -> Any:
        """Visit a generic AST node."""
        pass


class ASTNode(ABC):
    """Base class for all AST nodes."""
    
    def __init__(self, node_type: ASTNodeType, span: SourceSpan):
        self.node_type = node_type
        self.span = span
        self.parent: Optional['ASTNode'] = None
        self.attributes: Dict[str, Any] = {}
        # Generate unique ID for hashability
        self._id = uuid.uuid4()
    
    @abstractmethod
    def accept(self, visitor: ASTVisitor) -> Any:
        """Accept a visitor (visitor pattern)."""
        pass
    
    @abstractmethod
    def children(self) -> List['ASTNode']:
        """Get all child nodes."""
        pass
    
    def set_parent(self, parent: 'ASTNode'):
        """Set the parent node."""
        self.parent = parent
        
    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get an attribute value."""
        return self.attributes.get(key, default)
        
    def set_attribute(self, key: str, value: Any):
        """Set an attribute value."""
        self.attributes[key] = value
    
    def __str__(self) -> str:
        return f"{self.node_type.value}@{self.span}"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(span={self.span})"
    
    def __hash__(self) -> int:
        """Hash based on unique ID for use in dictionaries."""
        return hash(self._id)
    
    def __eq__(self, other) -> bool:
        """Equality based on unique ID."""
        if not isinstance(other, ASTNode):
            return False
        return self._id == other._id


# ============================================================================
# Top-level nodes
# ============================================================================

class Program(ASTNode):
    """Root AST node representing a complete program."""
    items: List['Item']
    
    def __init__(self, items: List['Item'], span: SourceSpan):
        super().__init__(ASTNodeType.PROGRAM, span)
        self.items = items
        for item in items:
            item.set_parent(self)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
    
    def children(self) -> List[ASTNode]:
        return self.items


# ============================================================================
# Items (top-level declarations)
# ============================================================================

class Item(ASTNode):
    """Base class for top-level items."""
    pass


class FunctionDef(Item):
    """Function definition."""
    name: str
    type_params: Optional[List['TypeParam']]
    params: List['Parameter']
    return_type: Optional['TypeRef']
    body: 'BlockStatement'
    is_async: bool = False
    is_differentiable: bool = False
    visibility: str = "private"  # "pub", "private"
    
    def __init__(self, name: str, type_params: Optional[List['TypeParam']], 
                 params: List['Parameter'], return_type: Optional['TypeRef'],
                 body: 'BlockStatement', span: SourceSpan, **kwargs):
        super().__init__(ASTNodeType.FUNCTION_DEF, span)
        self.name = name
        self.type_params = type_params or []
        self.params = params
        self.return_type = return_type
        self.body = body
        self.is_async = kwargs.get('is_async', False)
        self.is_differentiable = kwargs.get('is_differentiable', False)
        self.visibility = kwargs.get('visibility', 'private')
        
        # Set parents
        body.set_parent(self)
        for param in params:
            param.set_parent(self)
        if return_type:
            return_type.set_parent(self)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
    
    def children(self) -> List[ASTNode]:
        children = [self.body] + self.params
        if self.return_type:
            children.append(self.return_type)
        return children


@dataclass
class Parameter:
    """Function parameter."""
    name: str
    type_annotation: Optional['TypeRef']
    default_value: Optional['Expression']
    is_mutable: bool = False
    
    def set_parent(self, parent: ASTNode):
        if self.type_annotation:
            self.type_annotation.set_parent(parent)
        if self.default_value:
            self.default_value.set_parent(parent)


@dataclass
class TypeParam:
    """Generic type parameter."""
    name: str
    bounds: List['TypeRef']
    default: Optional['TypeRef'] = None


class StructDef(Item):
    """Struct definition."""
    name: str
    type_params: Optional[List[TypeParam]]
    fields: List['StructField']
    visibility: str = "private"
    
    def __init__(self, name: str, type_params: Optional[List[TypeParam]], 
                 fields: List['StructField'], span: SourceSpan, **kwargs):
        super().__init__(ASTNodeType.STRUCT_DEF, span)
        self.name = name
        self.type_params = type_params or []
        self.fields = fields
        self.visibility = kwargs.get('visibility', 'private')
        
        for field in fields:
            field.set_parent(self)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
    
    def children(self) -> List[ASTNode]:
        return []  # Fields are not ASTNodes in this simplified version


@dataclass
class StructField:
    """Struct field."""
    name: str
    type_annotation: 'TypeRef'
    visibility: str = "private"
    
    def set_parent(self, parent: ASTNode):
        self.type_annotation.set_parent(parent)


class TraitDef(Item):
    """Trait definition."""
    name: str
    type_params: Optional[List[TypeParam]]
    methods: List[FunctionDef]
    visibility: str = "private"
    
    def __init__(self, name: str, type_params: Optional[List[TypeParam]], 
                 methods: List[FunctionDef], span: SourceSpan, **kwargs):
        super().__init__(ASTNodeType.TRAIT_DEF, span)
        self.name = name
        self.type_params = type_params or []
        self.methods = methods
        self.visibility = kwargs.get('visibility', 'private')
        
        for method in methods:
            method.set_parent(self)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
    
    def children(self) -> List[ASTNode]:
        return self.methods


# ============================================================================
# Statements
# ============================================================================

class Statement(ASTNode):
    """Base class for statements."""
    pass


class VariableDecl(Statement):
    """Variable declaration statement."""
    name: str
    type_annotation: Optional['TypeRef']
    initializer: Optional['Expression']
    is_mutable: bool = False
    is_const: bool = False
    
    def __init__(self, name: str, type_annotation: Optional['TypeRef'],
                 initializer: Optional['Expression'], span: SourceSpan, **kwargs):
        super().__init__(ASTNodeType.VARIABLE_DECL, span)
        self.name = name
        self.type_annotation = type_annotation
        self.initializer = initializer
        self.is_mutable = kwargs.get('is_mutable', False)
        self.is_const = kwargs.get('is_const', False)
        
        if type_annotation:
            type_annotation.set_parent(self)
        if initializer:
            initializer.set_parent(self)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
    
    def children(self) -> List[ASTNode]:
        children = []
        if self.type_annotation:
            children.append(self.type_annotation)
        if self.initializer:
            children.append(self.initializer)
        return children


class BlockStatement(Statement):
    """Block statement containing multiple statements."""
    statements: List[Statement]
    
    def __init__(self, statements: List[Statement], span: SourceSpan):
        super().__init__(ASTNodeType.BLOCK_STATEMENT, span)
        self.statements = statements
        for stmt in statements:
            stmt.set_parent(self)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
    
    def children(self) -> List[ASTNode]:
        return self.statements


class IfStatement(Statement):
    """If statement with optional else clause."""
    condition: 'Expression'
    then_branch: Statement
    else_branch: Optional[Statement] = None
    
    def __init__(self, condition: 'Expression', then_branch: Statement, 
                 else_branch: Optional[Statement], span: SourceSpan):
        super().__init__(ASTNodeType.IF_STATEMENT, span)
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch
        
        condition.set_parent(self)
        then_branch.set_parent(self)
        if else_branch:
            else_branch.set_parent(self)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
    
    def children(self) -> List[ASTNode]:
        children = [self.condition, self.then_branch]
        if self.else_branch:
            children.append(self.else_branch)
        return children


class WhileLoop(Statement):
    """While loop statement."""
    condition: 'Expression'
    body: Statement
    
    def __init__(self, condition: 'Expression', body: Statement, span: SourceSpan):
        super().__init__(ASTNodeType.WHILE_LOOP, span)
        self.condition = condition
        self.body = body
        
        condition.set_parent(self)
        body.set_parent(self)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
    
    def children(self) -> List[ASTNode]:
        return [self.condition, self.body]


class ForLoop(Statement):
    """For loop statement."""
    variable: str
    iterable: 'Expression'
    body: Statement
    
    def __init__(self, variable: str, iterable: 'Expression', body: Statement, span: SourceSpan):
        super().__init__(ASTNodeType.FOR_LOOP, span)
        self.variable = variable
        self.iterable = iterable
        self.body = body
        
        iterable.set_parent(self)
        body.set_parent(self)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
    
    def children(self) -> List[ASTNode]:
        return [self.iterable, self.body]


class ReturnStatement(Statement):
    """Return statement."""
    value: Optional['Expression']
    
    def __init__(self, value: Optional['Expression'], span: SourceSpan):
        super().__init__(ASTNodeType.RETURN_STATEMENT, span)
        self.value = value
        
        if value:
            value.set_parent(self)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
    
    def children(self) -> List[ASTNode]:
        return [self.value] if self.value else []


# ============================================================================
# Expressions
# ============================================================================

class Expression(ASTNode):
    """Base class for expressions."""
    pass


class BinaryOp(Expression):
    """Binary operation expression."""
    left: Expression
    operator: str
    right: Expression
    
    def __init__(self, left: Expression, operator: str, right: Expression, span: SourceSpan):
        super().__init__(ASTNodeType.BINARY_OP, span)
        self.left = left
        self.operator = operator
        self.right = right
        
        left.set_parent(self)
        right.set_parent(self)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
    
    def children(self) -> List[ASTNode]:
        return [self.left, self.right]


class UnaryOp(Expression):
    """Unary operation expression."""
    operator: str
    operand: Expression
    
    def __init__(self, operator: str, operand: Expression, span: SourceSpan):
        super().__init__(ASTNodeType.UNARY_OP, span)
        self.operator = operator
        self.operand = operand
        
        operand.set_parent(self)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
    
    def children(self) -> List[ASTNode]:
        return [self.operand]


class FunctionCall(Expression):
    """Function call expression."""
    function: Expression
    args: List[Expression]
    type_args: Optional[List['TypeRef']] = None
    
    def __init__(self, function: Expression, args: List[Expression], 
                 span: SourceSpan, type_args: Optional[List['TypeRef']] = None):
        super().__init__(ASTNodeType.FUNCTION_CALL, span)
        self.function = function
        self.args = args
        self.type_args = type_args or []
        
        function.set_parent(self)
        for arg in args:
            arg.set_parent(self)
        for type_arg in self.type_args:
            type_arg.set_parent(self)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
    
    def children(self) -> List[ASTNode]:
        return [self.function] + self.args + self.type_args


class Identifier(Expression):
    """Identifier expression."""
    name: str
    
    def __init__(self, name: str, span: SourceSpan):
        super().__init__(ASTNodeType.IDENTIFIER, span)
        self.name = name
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
    
    def children(self) -> List[ASTNode]:
        return []


class Literal(Expression):
    """Literal value expression."""
    value: Any
    literal_type: str  # "integer", "float", "string", "boolean", etc.
    
    def __init__(self, value: Any, literal_type: str, span: SourceSpan):
        super().__init__(ASTNodeType.LITERAL, span)
        self.value = value
        self.literal_type = literal_type
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
    
    def children(self) -> List[ASTNode]:
        return []


class TensorLiteral(Expression):
    """Tensor literal expression."""
    elements: List[List[Expression]]  # Nested lists for multidimensional tensors
    dimensions: List[int]
    
    def __init__(self, elements: List[List[Expression]], dimensions: List[int], span: SourceSpan):
        super().__init__(ASTNodeType.TENSOR_LITERAL, span)
        self.elements = elements
        self.dimensions = dimensions
        
        # Set parents for all nested expressions
        def set_parents(expr_list):
            for expr in expr_list:
                if isinstance(expr, Expression):
                    expr.set_parent(self)
                elif isinstance(expr, list):
                    set_parents(expr)
        
        set_parents(elements)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
    
    def children(self) -> List[ASTNode]:
        # Flatten nested structure to get all expression children
        children = []
        def collect_children(expr_list):
            for expr in expr_list:
                if isinstance(expr, Expression):
                    children.append(expr)
                elif isinstance(expr, list):
                    collect_children(expr)
        collect_children(self.elements)
        return children


class UnitLiteral(Expression):
    """Unit literal with dimensional analysis."""
    value: Union[int, float]
    unit: str
    
    def __init__(self, value: Union[int, float], unit: str, span: SourceSpan):
        super().__init__(ASTNodeType.UNIT_LITERAL, span)
        self.value = value
        self.unit = unit
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
    
    def children(self) -> List[ASTNode]:
        return []


class ComplexLiteral(Expression):
    """Complex number literal."""
    real: float
    imaginary: float
    
    def __init__(self, real: float, imaginary: float, span: SourceSpan):
        super().__init__(ASTNodeType.COMPLEX_LITERAL, span)
        self.real = real
        self.imaginary = imaginary
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
    
    def children(self) -> List[ASTNode]:
        return []


# ============================================================================
# Mathematical constructs
# ============================================================================

class Gradient(Expression):
    """Gradient operator (∇) expression."""
    function: Expression
    variables: List[Expression]
    
    def __init__(self, function: Expression, variables: List[Expression], span: SourceSpan):
        super().__init__(ASTNodeType.GRADIENT, span)
        self.function = function
        self.variables = variables
        
        function.set_parent(self)
        for var in variables:
            var.set_parent(self)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
    
    def children(self) -> List[ASTNode]:
        return [self.function] + self.variables


class Summation(Expression):
    """Summation (∑) expression."""
    variable: str
    start: Expression
    end: Expression
    expression: Expression
    
    def __init__(self, variable: str, start: Expression, end: Expression, 
                 expression: Expression, span: SourceSpan):
        super().__init__(ASTNodeType.SUMMATION, span)
        self.variable = variable
        self.start = start
        self.end = end
        self.expression = expression
        
        start.set_parent(self)
        end.set_parent(self)
        expression.set_parent(self)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
    
    def children(self) -> List[ASTNode]:
        return [self.start, self.end, self.expression]


# ============================================================================
# Type system
# ============================================================================

class TypeRef(ASTNode):
    """Base class for type references."""
    pass


class SimpleTypeRef(TypeRef):
    """Simple type reference (e.g., 'i32', 'f64', 'String')."""
    name: str
    
    def __init__(self, name: str, span: SourceSpan):
        super().__init__(ASTNodeType.TYPE_REF, span)
        self.name = name
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
    
    def children(self) -> List[ASTNode]:
        return []


class TensorTypeRef(TypeRef):
    """Tensor type reference with dimensions."""
    element_type: TypeRef
    dimensions: List[Expression]  # Can be literals or const expressions
    
    def __init__(self, element_type: TypeRef, dimensions: List[Expression], span: SourceSpan):
        super().__init__(ASTNodeType.TENSOR_TYPE, span)
        self.element_type = element_type
        self.dimensions = dimensions
        
        element_type.set_parent(self)
        for dim in dimensions:
            dim.set_parent(self)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
    
    def children(self) -> List[ASTNode]:
        return [self.element_type] + self.dimensions


# Alias for the main AST type
AST = Program
