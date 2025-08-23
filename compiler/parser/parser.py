"""
NeuralScript Pratt Parser Implementation

Implements a top-down operator precedence (Pratt) parser for NeuralScript.
Handles mathematical expressions with Unicode operators, function calls,
tensor operations, and complex language constructs.

Author: xwest
"""

from typing import List, Optional, Dict, Callable, Any, Union
from enum import IntEnum

from ..lexer.tokens import Token, TokenType, SourceLocation
from .ast_nodes import *
from .errors import (
    ParseError, ParseWarning, create_unexpected_token_error,
    create_missing_token_error, create_unexpected_eof_error,
    create_invalid_expression_error, SyntaxErrorRecovery
)


class Precedence(IntEnum):
    """Operator precedence levels for Pratt parsing."""
    NONE = 0
    ASSIGNMENT = 1      # =, +=, -=, etc.
    OR = 2              # ||, ∨
    AND = 3             # &&, ∧  
    EQUALITY = 4        # ==, !=, ≡, ≠
    COMPARISON = 5      # <, >, <=, >=, ≤, ≥, ∈, ∉
    TERM = 6            # +, -, ⊕, ⊖
    FACTOR = 7          # *, /, %, ⊙, ⊗, ⋅, ×, ÷
    UNARY = 8           # !, -, +, ¬, ∇, ∂, √
    CALL = 9            # function calls, field access
    PRIMARY = 10        # literals, identifiers, parentheses


class Parser:
    """
    NeuralScript Pratt parser.
    
    Implements top-down operator precedence parsing with error recovery
    and comprehensive AST generation.
    """
    
    def __init__(self, tokens: List[Token]):
        """
        Initialize parser with a list of tokens.
        
        Args:
            tokens: List of tokens from the lexer
        """
        self.tokens = tokens
        self.current = 0
        self.errors: List[ParseError] = []
        self.warnings: List[ParseWarning] = []
        
        # Initialize parsing tables
        self._init_parsing_tables()
    
    def _init_parsing_tables(self):
        """Initialize operator precedence and parsing function tables."""
        
        # Prefix parsing functions (for tokens that can start expressions)
        self.prefix_parsers: Dict[TokenType, Callable[[], Expression]] = {
            # Literals
            TokenType.INTEGER_DECIMAL: self._parse_integer_literal,
            TokenType.INTEGER_BINARY: self._parse_integer_literal,
            TokenType.INTEGER_OCTAL: self._parse_integer_literal,
            TokenType.INTEGER_HEXADECIMAL: self._parse_integer_literal,
            TokenType.FLOAT: self._parse_float_literal,
            TokenType.COMPLEX: self._parse_complex_literal,
            TokenType.STRING: self._parse_string_literal,
            TokenType.CHARACTER: self._parse_character_literal,
            TokenType.UNIT_LITERAL: self._parse_unit_literal,
            TokenType.TRUE: self._parse_boolean_literal,
            TokenType.FALSE: self._parse_boolean_literal,
            
            # Identifiers
            TokenType.IDENTIFIER: self._parse_identifier,
            
            # Mathematical keywords that can be used as identifiers
            TokenType.SUM: self._parse_math_keyword_as_identifier,
            TokenType.PROD: self._parse_math_keyword_as_identifier,
            TokenType.GRAD: self._parse_math_keyword_as_identifier,
            TokenType.DIV: self._parse_math_keyword_as_identifier,
            TokenType.CURL: self._parse_math_keyword_as_identifier,
            TokenType.LAPLACE: self._parse_math_keyword_as_identifier,
            TokenType.INTEGRAL: self._parse_math_keyword_as_identifier,
            TokenType.DIFF: self._parse_math_keyword_as_identifier,
            
            # Unary operators
            TokenType.MINUS: self._parse_unary,
            TokenType.PLUS: self._parse_unary,
            TokenType.LOGICAL_NOT: self._parse_unary,
            TokenType.NOT_SYMBOL: self._parse_unary,
            TokenType.NABLA: self._parse_gradient,
            TokenType.PARTIAL: self._parse_partial_derivative,
            TokenType.SQUARE_ROOT: self._parse_unary,
            TokenType.CUBE_ROOT: self._parse_unary,
            
            # Mathematical operators
            TokenType.SUMMATION: self._parse_summation,
            TokenType.INTEGRAL_OP: self._parse_integral,
            
            # Grouping
            TokenType.LEFT_PAREN: self._parse_grouping,
            TokenType.LEFT_BRACKET: self._parse_tensor_literal,
            
            # Tensor literal syntax: tensor![...]
            # This would be handled via identifier parsing for 'tensor'
        }
        
        # Infix parsing functions (for binary operators)
        self.infix_parsers: Dict[TokenType, Callable[[Expression], Expression]] = {
            # Arithmetic operators
            TokenType.PLUS: self._parse_binary,
            TokenType.MINUS: self._parse_binary,
            TokenType.MULTIPLY: self._parse_binary,
            TokenType.DIVIDE: self._parse_binary,
            TokenType.MODULO: self._parse_binary,
            TokenType.POWER: self._parse_binary,
            TokenType.FLOOR_DIVIDE: self._parse_binary,
            
            # Mathematical operators (Unicode)
            TokenType.DOT_PRODUCT: self._parse_binary,
            TokenType.TENSOR_PRODUCT: self._parse_binary,
            TokenType.ELEMENT_WISE_ADD: self._parse_binary,
            TokenType.ELEMENT_WISE_SUB: self._parse_binary,
            TokenType.SCALAR_PRODUCT: self._parse_binary,
            TokenType.CROSS_PRODUCT: self._parse_binary,
            TokenType.DIVISION_SYMBOL: self._parse_binary,
            
            # Comparison operators
            TokenType.EQUAL: self._parse_binary,
            TokenType.NOT_EQUAL: self._parse_binary,
            TokenType.LESS_THAN: self._parse_binary,
            TokenType.GREATER_THAN: self._parse_binary,
            TokenType.LESS_EQUAL: self._parse_binary,
            TokenType.GREATER_EQUAL: self._parse_binary,
            TokenType.EQUIVALENT: self._parse_binary,
            TokenType.NOT_EQUIVALENT: self._parse_binary,
            TokenType.LESS_EQUAL_MATH: self._parse_binary,
            TokenType.GREATER_EQUAL_MATH: self._parse_binary,
            TokenType.APPROXIMATELY: self._parse_binary,
            TokenType.ELEMENT_OF: self._parse_binary,
            TokenType.NOT_ELEMENT_OF: self._parse_binary,
            
            # Logical operators
            TokenType.LOGICAL_AND: self._parse_binary,
            TokenType.LOGICAL_OR: self._parse_binary,
            TokenType.AND_SYMBOL: self._parse_binary,
            TokenType.OR_SYMBOL: self._parse_binary,
            TokenType.XOR_SYMBOL: self._parse_binary,
            TokenType.BICONDITIONAL: self._parse_binary,
            
            # Assignment operators
            TokenType.ASSIGN: self._parse_assignment,
            TokenType.PLUS_ASSIGN: self._parse_assignment,
            TokenType.MINUS_ASSIGN: self._parse_assignment,
            TokenType.MULTIPLY_ASSIGN: self._parse_assignment,
            TokenType.DIVIDE_ASSIGN: self._parse_assignment,
            TokenType.MODULO_ASSIGN: self._parse_assignment,
            TokenType.POWER_ASSIGN: self._parse_assignment,
            
            # Function call and member access
            TokenType.LEFT_PAREN: self._parse_function_call,
            TokenType.DOT: self._parse_field_access,
            TokenType.LEFT_BRACKET: self._parse_index_access,
        }
        
        # Operator precedence table
        self.precedences: Dict[TokenType, Precedence] = {
            # Assignment (right associative, handled specially)
            TokenType.ASSIGN: Precedence.ASSIGNMENT,
            TokenType.PLUS_ASSIGN: Precedence.ASSIGNMENT,
            TokenType.MINUS_ASSIGN: Precedence.ASSIGNMENT,
            TokenType.MULTIPLY_ASSIGN: Precedence.ASSIGNMENT,
            TokenType.DIVIDE_ASSIGN: Precedence.ASSIGNMENT,
            TokenType.MODULO_ASSIGN: Precedence.ASSIGNMENT,
            TokenType.POWER_ASSIGN: Precedence.ASSIGNMENT,
            
            # Logical OR
            TokenType.LOGICAL_OR: Precedence.OR,
            TokenType.OR_SYMBOL: Precedence.OR,
            
            # Logical AND
            TokenType.LOGICAL_AND: Precedence.AND,
            TokenType.AND_SYMBOL: Precedence.AND,
            
            # Equality
            TokenType.EQUAL: Precedence.EQUALITY,
            TokenType.NOT_EQUAL: Precedence.EQUALITY,
            TokenType.EQUIVALENT: Precedence.EQUALITY,
            TokenType.NOT_EQUIVALENT: Precedence.EQUALITY,
            TokenType.APPROXIMATELY: Precedence.EQUALITY,
            
            # Comparison
            TokenType.LESS_THAN: Precedence.COMPARISON,
            TokenType.GREATER_THAN: Precedence.COMPARISON,
            TokenType.LESS_EQUAL: Precedence.COMPARISON,
            TokenType.GREATER_EQUAL: Precedence.COMPARISON,
            TokenType.LESS_EQUAL_MATH: Precedence.COMPARISON,
            TokenType.GREATER_EQUAL_MATH: Precedence.COMPARISON,
            TokenType.ELEMENT_OF: Precedence.COMPARISON,
            TokenType.NOT_ELEMENT_OF: Precedence.COMPARISON,
            
            # Addition/Subtraction
            TokenType.PLUS: Precedence.TERM,
            TokenType.MINUS: Precedence.TERM,
            TokenType.ELEMENT_WISE_ADD: Precedence.TERM,
            TokenType.ELEMENT_WISE_SUB: Precedence.TERM,
            
            # Multiplication/Division
            TokenType.MULTIPLY: Precedence.FACTOR,
            TokenType.DIVIDE: Precedence.FACTOR,
            TokenType.MODULO: Precedence.FACTOR,
            TokenType.FLOOR_DIVIDE: Precedence.FACTOR,
            TokenType.DOT_PRODUCT: Precedence.FACTOR,
            TokenType.TENSOR_PRODUCT: Precedence.FACTOR,
            TokenType.SCALAR_PRODUCT: Precedence.FACTOR,
            TokenType.CROSS_PRODUCT: Precedence.FACTOR,
            TokenType.DIVISION_SYMBOL: Precedence.FACTOR,
            
            # Exponentiation (right associative, higher precedence)
            TokenType.POWER: Precedence.UNARY,
            
            # Function calls and member access
            TokenType.LEFT_PAREN: Precedence.CALL,
            TokenType.DOT: Precedence.CALL,
            TokenType.LEFT_BRACKET: Precedence.CALL,
        }
    
    def parse(self) -> Program:
        """
        Parse the token stream into an AST.
        
        Returns:
            Program AST node representing the entire program
            
        Raises:
            ParseError: If parsing fails fatally
        """
        items = []
        
        while not self._is_at_end():
            try:
                # Skip newlines at top level
                if self._check(TokenType.NEWLINE):
                    self._advance()
                    continue
                
                item = self._parse_item()
                if item:
                    items.append(item)
                    
            except ParseError as e:
                self.errors.append(e)
                # Try to recover by synchronizing to next statement
                self.current = SyntaxErrorRecovery.synchronize_to_statement_boundary(
                    self.tokens, self.current
                )
        
        # Create source span for entire program
        start_location = self.tokens[0].location if self.tokens else SourceLocation("<empty>", 1, 1, 0)
        end_location = self.tokens[-1].location if self.tokens else start_location
        program_span = SourceSpan(start_location, end_location)
        
        program = Program(items, program_span)
        
        # If we have errors, raise the first one
        if self.errors:
            raise self.errors[0]
            
        return program
    
    def _parse_item(self) -> Optional[Item]:
        """Parse a top-level item (function, struct, etc.)."""
        # Parse attributes first
        attributes = []
        while self._check(TokenType.AT):
            attributes.append(self._parse_attribute())
        
        # Check for visibility modifier
        visibility = "private"
        if self._match(TokenType.PUB):
            visibility = "pub"
        
        # Parse the actual item
        if self._check(TokenType.FN) or self._check(TokenType.ASYNC):
            func = self._parse_function(visibility)
            # Apply attributes to function
            for attr in attributes:
                if attr == "differentiable":
                    func.is_differentiable = True
            return func
        elif self._check(TokenType.STRUCT):
            return self._parse_struct(visibility)
        elif self._check(TokenType.TRAIT):
            return self._parse_trait(visibility)
        elif self._check(TokenType.IMPL):
            return self._parse_impl_block()
        elif self._check(TokenType.TYPE):
            return self._parse_type_alias(visibility)
        elif self._check(TokenType.USE):
            return self._parse_use_statement()
        else:
            # Not a top-level item - might be a statement at global scope
            # This is an error in most languages, but let's be flexible
            raise create_unexpected_token_error(
                "top-level item (fn, struct, trait, etc.)",
                self._peek()
            )
    
    def _parse_function(self, visibility: str = "private") -> FunctionDef:
        """Parse a function definition."""
        start_token = self._peek()
        
        # Handle async functions
        is_async = False
        if self._match(TokenType.ASYNC):
            is_async = True
        
        self._consume(TokenType.FN, "Expected 'fn' keyword")
        
        # Function name
        name_token = self._consume(TokenType.IDENTIFIER, "Expected function name")
        name = name_token.lexeme
        
        # Generic parameters (optional)
        type_params = None
        if self._match(TokenType.LESS_THAN):
            type_params = self._parse_type_parameters()
            self._consume(TokenType.GREATER_THAN, "Expected '>' after type parameters")
        
        # Function parameters
        self._consume(TokenType.LEFT_PAREN, "Expected '(' after function name")
        params = self._parse_parameter_list()
        self._consume(TokenType.RIGHT_PAREN, "Expected ')' after parameters")
        
        # Return type (optional)
        return_type = None
        if self._match(TokenType.ARROW):
            return_type = self._parse_type_reference()
        
        # Function body
        body = self._parse_block_statement()
        
        # Create source span
        end_location = self._previous().location
        span = SourceSpan(start_token.location, end_location)
        
        return FunctionDef(
            name=name,
            type_params=type_params,
            params=params,
            return_type=return_type,
            body=body,
            span=span,
            is_async=is_async,
            visibility=visibility
        )
    
    def _parse_struct(self, visibility: str = "private") -> StructDef:
        """Parse a struct definition."""
        start_token = self._consume(TokenType.STRUCT, "Expected 'struct' keyword")
        
        # Struct name
        name_token = self._consume(TokenType.IDENTIFIER, "Expected struct name")
        name = name_token.lexeme
        
        # Generic parameters (optional)
        type_params = None
        if self._match(TokenType.LESS_THAN):
            type_params = self._parse_type_parameters()
            self._consume(TokenType.GREATER_THAN, "Expected '>' after type parameters")
        
        # Struct fields
        self._consume(TokenType.LEFT_BRACE, "Expected '{' before struct fields")
        fields = self._parse_struct_fields()
        self._consume(TokenType.RIGHT_BRACE, "Expected '}' after struct fields")
        
        # Create source span
        end_location = self._previous().location
        span = SourceSpan(start_token.location, end_location)
        
        return StructDef(
            name=name,
            type_params=type_params,
            fields=fields,
            span=span,
            visibility=visibility
        )
    
    def _parse_trait(self, visibility: str = "private") -> TraitDef:
        """Parse a trait definition."""
        start_token = self._consume(TokenType.TRAIT, "Expected 'trait' keyword")
        
        # Trait name
        name_token = self._consume(TokenType.IDENTIFIER, "Expected trait name")
        name = name_token.lexeme
        
        # Generic parameters (optional)
        type_params = None
        if self._match(TokenType.LESS_THAN):
            type_params = self._parse_type_parameters()
            self._consume(TokenType.GREATER_THAN, "Expected '>' after type parameters")
        
        # Trait methods
        self._consume(TokenType.LEFT_BRACE, "Expected '{' before trait methods")
        methods = []
        
        while not self._check(TokenType.RIGHT_BRACE) and not self._is_at_end():
            # Skip newlines
            if self._match(TokenType.NEWLINE):
                continue
                
            method = self._parse_function("public")  # Trait methods are public by default
            methods.append(method)
        
        self._consume(TokenType.RIGHT_BRACE, "Expected '}' after trait methods")
        
        # Create source span
        end_location = self._previous().location
        span = SourceSpan(start_token.location, end_location)
        
        return TraitDef(
            name=name,
            type_params=type_params,
            methods=methods,
            span=span,
            visibility=visibility
        )
    
    def _parse_parameter_list(self) -> List[Parameter]:
        """Parse function parameter list."""
        params = []
        
        if not self._check(TokenType.RIGHT_PAREN):
            params.append(self._parse_parameter())
            
            while self._match(TokenType.COMMA):
                # Allow trailing commas
                if self._check(TokenType.RIGHT_PAREN):
                    break
                params.append(self._parse_parameter())
        
        return params
    
    def _parse_parameter(self) -> Parameter:
        """Parse a single function parameter."""
        # Check for mutability
        is_mutable = self._match(TokenType.MUT)
        
        # Parameter name
        name_token = self._consume(TokenType.IDENTIFIER, "Expected parameter name")
        name = name_token.lexeme
        
        # Type annotation
        type_annotation = None
        if self._match(TokenType.COLON):
            type_annotation = self._parse_type_reference()
        
        # Default value
        default_value = None
        if self._match(TokenType.ASSIGN):
            default_value = self._parse_expression()
        
        return Parameter(
            name=name,
            type_annotation=type_annotation,
            default_value=default_value,
            is_mutable=is_mutable
        )
    
    def _parse_struct_fields(self) -> List[StructField]:
        """Parse struct field list."""
        fields = []
        
        while not self._check(TokenType.RIGHT_BRACE) and not self._is_at_end():
            # Skip newlines
            if self._match(TokenType.NEWLINE):
                continue
                
            # Field visibility
            field_visibility = "private"
            if self._match(TokenType.PUB):
                field_visibility = "pub"
            
            # Field name
            name_token = self._consume(TokenType.IDENTIFIER, "Expected field name")
            name = name_token.lexeme
            
            # Type annotation (required for struct fields)
            self._consume(TokenType.COLON, "Expected ':' after field name")
            type_annotation = self._parse_type_reference()
            
            # Optional comma
            self._match(TokenType.COMMA)
            
            fields.append(StructField(
                name=name,
                type_annotation=type_annotation,
                visibility=field_visibility
            ))
        
        return fields
    
    def _parse_type_parameters(self) -> List[TypeParam]:
        """Parse generic type parameters."""
        # Simplified implementation - just names for now
        type_params = []
        
        if not self._check(TokenType.GREATER_THAN):
            name_token = self._consume(TokenType.IDENTIFIER, "Expected type parameter name")
            type_params.append(TypeParam(name_token.lexeme, [], None))
            
            while self._match(TokenType.COMMA):
                if self._check(TokenType.GREATER_THAN):
                    break
                name_token = self._consume(TokenType.IDENTIFIER, "Expected type parameter name")
                type_params.append(TypeParam(name_token.lexeme, [], None))
        
        return type_params
    
    def _parse_type_reference(self) -> TypeRef:
        """Parse a type reference."""
        if self._check(TokenType.IDENTIFIER):
            name_token = self._advance()
            name = name_token.lexeme
            
            # Check for tensor type syntax: Tensor<f32, [3, 4]>
            if name == "Tensor" and self._match(TokenType.LESS_THAN):
                element_type = self._parse_type_reference()
                self._consume(TokenType.COMMA, "Expected ',' after tensor element type")
                
                # Parse dimension list [3, 4, ...]
                self._consume(TokenType.LEFT_BRACKET, "Expected '[' before tensor dimensions")
                dimensions = []
                
                if not self._check(TokenType.RIGHT_BRACKET):
                    dimensions.append(self._parse_expression())
                    while self._match(TokenType.COMMA):
                        if self._check(TokenType.RIGHT_BRACKET):
                            break
                        dimensions.append(self._parse_expression())
                
                self._consume(TokenType.RIGHT_BRACKET, "Expected ']' after tensor dimensions")
                self._consume(TokenType.GREATER_THAN, "Expected '>' after tensor type")
                
                span = SourceSpan(name_token.location, self._previous().location)
                return TensorTypeRef(element_type, dimensions, span)
            
            # Simple type reference
            span = SourceSpan(name_token.location, name_token.location)
            return SimpleTypeRef(name, span)
        
        raise create_unexpected_token_error("type name", self._peek())
    
    def _parse_block_statement(self) -> BlockStatement:
        """Parse a block statement."""
        start_token = self._consume(TokenType.LEFT_BRACE, "Expected '{'")
        statements = []
        
        while not self._check(TokenType.RIGHT_BRACE) and not self._is_at_end():
            # Skip newlines
            if self._match(TokenType.NEWLINE):
                continue
                
            stmt = self._parse_statement()
            if stmt:
                statements.append(stmt)
        
        end_token = self._consume(TokenType.RIGHT_BRACE, "Expected '}'")
        span = SourceSpan(start_token.location, end_token.location)
        
        return BlockStatement(statements, span)
    
    def _parse_statement(self) -> Optional[Statement]:
        """Parse a statement."""
        try:
            if self._check(TokenType.LET) or self._check(TokenType.CONST):
                return self._parse_variable_declaration()
            elif self._check(TokenType.IF):
                return self._parse_if_statement()
            elif self._check(TokenType.WHILE):
                return self._parse_while_statement()
            elif self._check(TokenType.FOR):
                return self._parse_for_statement()
            elif self._check(TokenType.RETURN):
                return self._parse_return_statement()
            elif self._check(TokenType.LEFT_BRACE):
                return self._parse_block_statement()
            else:
                # Expression statement
                expr = self._parse_expression()
                self._consume_statement_terminator()
                return self._create_expression_statement(expr)
                
        except ParseError as e:
            self.errors.append(e)
            # Synchronize to next statement boundary
            self.current = SyntaxErrorRecovery.synchronize_to_statement_boundary(
                self.tokens, self.current
            )
            return None
    
    def _parse_variable_declaration(self) -> VariableDecl:
        """Parse a variable declaration statement."""
        start_token = self._peek()
        
        # Check for const vs let
        is_const = self._match(TokenType.CONST)
        is_mutable = False
        
        if not is_const:
            self._consume(TokenType.LET, "Expected 'let' or 'const'")
            is_mutable = self._match(TokenType.MUT)
        
        # Variable name - allow mathematical keywords as identifiers in this context
        name_token = self._consume_variable_name("Expected variable name")
        name = name_token.lexeme
        
        # Type annotation (optional)
        type_annotation = None
        if self._match(TokenType.COLON):
            type_annotation = self._parse_type_reference()
        
        # Initializer (optional for declarations, required for const)
        initializer = None
        if self._match(TokenType.ASSIGN):
            initializer = self._parse_expression()
        elif is_const:
            raise create_missing_token_error(TokenType.ASSIGN, self._peek().location)
        
        self._consume_statement_terminator()
        
        span = SourceSpan(start_token.location, self._previous().location)
        
        return VariableDecl(
            name=name,
            type_annotation=type_annotation,
            initializer=initializer,
            span=span,
            is_mutable=is_mutable,
            is_const=is_const
        )
    
    def _parse_if_statement(self) -> IfStatement:
        """Parse an if statement."""
        start_token = self._consume(TokenType.IF, "Expected 'if'")
        
        # Condition
        condition = self._parse_expression()
        
        # Then branch
        then_branch = self._parse_statement()
        
        # Else branch (optional)
        else_branch = None
        if self._match(TokenType.ELSE):
            else_branch = self._parse_statement()
        
        span = SourceSpan(start_token.location, self._previous().location)
        
        return IfStatement(condition, then_branch, else_branch, span)
    
    def _parse_while_statement(self) -> WhileLoop:
        """Parse a while loop statement."""
        start_token = self._consume(TokenType.WHILE, "Expected 'while'")
        
        # Condition
        condition = self._parse_expression()
        
        # Body
        body = self._parse_statement()
        
        span = SourceSpan(start_token.location, self._previous().location)
        
        return WhileLoop(condition, body, span)
    
    def _parse_for_statement(self) -> ForLoop:
        """Parse a for loop statement."""
        start_token = self._consume(TokenType.FOR, "Expected 'for'")
        
        # Loop variable
        var_token = self._consume(TokenType.IDENTIFIER, "Expected loop variable")
        variable = var_token.lexeme
        
        # 'in' keyword
        self._consume(TokenType.IN, "Expected 'in' after loop variable")
        
        # Iterable expression
        iterable = self._parse_expression()
        
        # Body
        body = self._parse_statement()
        
        span = SourceSpan(start_token.location, self._previous().location)
        
        return ForLoop(variable, iterable, body, span)
    
    def _parse_return_statement(self) -> ReturnStatement:
        """Parse a return statement."""
        start_token = self._consume(TokenType.RETURN, "Expected 'return'")
        
        # Return value (optional)
        value = None
        if not self._check_statement_terminator():
            value = self._parse_expression()
        
        self._consume_statement_terminator()
        
        span = SourceSpan(start_token.location, self._previous().location)
        
        return ReturnStatement(value, span)
    
    def _parse_expression(self) -> Expression:
        """Parse an expression using Pratt parsing."""
        return self._parse_precedence(Precedence.ASSIGNMENT)
    
    def _parse_precedence(self, precedence: Precedence) -> Expression:
        """Parse expression with given minimum precedence."""
        # Get prefix parser
        prefix_parser = self.prefix_parsers.get(self._peek().type)
        if prefix_parser is None:
            raise create_invalid_expression_error(
                f"Unexpected token '{self._peek().lexeme}' in expression",
                self._peek().location,
                self._peek()
            )
        
        # Parse left side with prefix parser
        left = prefix_parser()
        
        # Parse infix operators
        while precedence <= self._get_precedence(self._peek().type):
            infix_parser = self.infix_parsers.get(self._peek().type)
            if infix_parser is None:
                break
            left = infix_parser(left)
        
        return left
    
    def _get_precedence(self, token_type: TokenType) -> Precedence:
        """Get precedence for a token type."""
        return self.precedences.get(token_type, Precedence.NONE)
    
    # Prefix parsers (tokens that can start expressions)
    
    def _parse_integer_literal(self) -> Literal:
        """Parse integer literal."""
        token = self._advance()
        span = SourceSpan(token.location, token.location)
        return Literal(token.value, "integer", span)
    
    def _parse_float_literal(self) -> Literal:
        """Parse float literal."""
        token = self._advance()
        span = SourceSpan(token.location, token.location)
        return Literal(token.value, "float", span)
    
    def _parse_complex_literal(self) -> ComplexLiteral:
        """Parse complex number literal."""
        token = self._advance()
        complex_value = token.value  # This is a Python complex number
        span = SourceSpan(token.location, token.location)
        return ComplexLiteral(complex_value.real, complex_value.imag, span)
    
    def _parse_string_literal(self) -> Literal:
        """Parse string literal."""
        token = self._advance()
        span = SourceSpan(token.location, token.location)
        return Literal(token.value, "string", span)
    
    def _parse_character_literal(self) -> Literal:
        """Parse character literal."""
        token = self._advance()
        span = SourceSpan(token.location, token.location)
        return Literal(token.value, "character", span)
    
    def _parse_unit_literal(self) -> UnitLiteral:
        """Parse unit literal."""
        token = self._advance()
        value, unit = token.value  # Tuple from lexer
        span = SourceSpan(token.location, token.location)
        return UnitLiteral(value, unit, span)
    
    def _parse_boolean_literal(self) -> Literal:
        """Parse boolean literal."""
        token = self._advance()
        span = SourceSpan(token.location, token.location)
        return Literal(token.value, "boolean", span)
    
    def _parse_identifier(self) -> Identifier:
        """Parse identifier."""
        token = self._advance()
        span = SourceSpan(token.location, token.location)
        return Identifier(token.lexeme, span)
    
    def _parse_math_keyword_as_identifier(self) -> Identifier:
        """Parse mathematical keyword as identifier."""
        token = self._advance()
        span = SourceSpan(token.location, token.location)
        return Identifier(token.lexeme, span)
    
    def _parse_unary(self) -> UnaryOp:
        """Parse unary operation."""
        operator_token = self._advance()
        operator = operator_token.lexeme
        
        # Parse operand with unary precedence
        operand = self._parse_precedence(Precedence.UNARY)
        
        span = SourceSpan(operator_token.location, operand.span.end)
        return UnaryOp(operator, operand, span)
    
    def _parse_gradient(self) -> Gradient:
        """Parse gradient operator (∇)."""
        start_token = self._advance()  # Consume ∇
        
        # Parse function expression
        function = self._parse_precedence(Precedence.UNARY)
        
        # For now, assume single variable gradients
        # In a full implementation, this would parse variable lists
        variables = []
        
        span = SourceSpan(start_token.location, function.span.end)
        return Gradient(function, variables, span)
    
    def _parse_partial_derivative(self) -> UnaryOp:
        """Parse partial derivative operator (∂)."""
        # For now, treat as unary operator
        # In full implementation, would parse ∂f/∂x syntax
        return self._parse_unary()
    
    def _parse_summation(self) -> Summation:
        """Parse summation operator (∑)."""
        start_token = self._advance()  # Consume ∑
        
        # Simplified summation parsing - would need full syntax in real implementation
        # For now, just parse the next expression
        expression = self._parse_precedence(Precedence.UNARY)
        
        # Create dummy bounds for now
        start_expr = Literal(0, "integer", SourceSpan(start_token.location, start_token.location))
        end_expr = Literal(10, "integer", SourceSpan(start_token.location, start_token.location))
        
        span = SourceSpan(start_token.location, expression.span.end)
        return Summation("i", start_expr, end_expr, expression, span)
    
    def _parse_integral(self) -> UnaryOp:
        """Parse integral operator (∫)."""
        # Simplified - treat as unary for now
        return self._parse_unary()
    
    def _parse_grouping(self) -> Expression:
        """Parse parenthesized expression."""
        self._advance()  # Consume (
        
        expr = self._parse_expression()
        
        self._consume(TokenType.RIGHT_PAREN, "Expected ')' after expression")
        
        return expr
    
    def _parse_tensor_literal(self) -> TensorLiteral:
        """Parse tensor literal [1, 2, 3] or [[1, 2], [3, 4]]."""
        start_token = self._advance()  # Consume [
        
        elements = []
        dimensions = []
        
        if not self._check(TokenType.RIGHT_BRACKET):
            # Parse first element to determine if this is nested
            first_element = self._parse_expression()
            elements.append([first_element])
            
            while self._match(TokenType.COMMA):
                if self._check(TokenType.RIGHT_BRACKET):
                    break
                element = self._parse_expression()
                elements[0].append(element)
        
        self._consume(TokenType.RIGHT_BRACKET, "Expected ']' after tensor elements")
        
        # Calculate dimensions (simplified)
        if elements:
            dimensions = [len(elements[0])]
        
        span = SourceSpan(start_token.location, self._previous().location)
        return TensorLiteral(elements, dimensions, span)
    
    # Infix parsers (binary operators and postfix operations)
    
    def _parse_binary(self, left: Expression) -> BinaryOp:
        """Parse binary operation."""
        operator_token = self._advance()
        operator = operator_token.lexeme
        
        # Get precedence and handle right associativity for ** operator
        precedence = self._get_precedence(operator_token.type)
        if operator_token.type == TokenType.POWER:
            # Right associative
            right = self._parse_precedence(precedence)
        else:
            # Left associative
            right = self._parse_precedence(Precedence(precedence + 1))
        
        span = SourceSpan(left.span.start, right.span.end)
        return BinaryOp(left, operator, right, span)
    
    def _parse_assignment(self, left: Expression) -> BinaryOp:
        """Parse assignment operation."""
        # Assignment is right associative
        operator_token = self._advance()
        operator = operator_token.lexeme
        
        right = self._parse_precedence(Precedence.ASSIGNMENT)
        
        span = SourceSpan(left.span.start, right.span.end)
        return BinaryOp(left, operator, right, span)
    
    def _parse_function_call(self, left: Expression) -> FunctionCall:
        """Parse function call."""
        self._advance()  # Consume (
        
        args = []
        if not self._check(TokenType.RIGHT_PAREN):
            args.append(self._parse_expression())
            while self._match(TokenType.COMMA):
                if self._check(TokenType.RIGHT_PAREN):
                    break
                args.append(self._parse_expression())
        
        end_token = self._consume(TokenType.RIGHT_PAREN, "Expected ')' after arguments")
        
        span = SourceSpan(left.span.start, end_token.location)
        return FunctionCall(left, args, span)
    
    def _parse_field_access(self, left: Expression) -> BinaryOp:
        """Parse field access (dot operator)."""
        operator_token = self._advance()  # Consume .
        
        field_token = self._consume(TokenType.IDENTIFIER, "Expected field name after '.'")
        field = Identifier(field_token.lexeme, SourceSpan(field_token.location, field_token.location))
        
        span = SourceSpan(left.span.start, field.span.end)
        return BinaryOp(left, ".", field, span)
    
    def _parse_index_access(self, left: Expression) -> BinaryOp:
        """Parse index access (array[index])."""
        self._advance()  # Consume [
        
        index = self._parse_expression()
        
        end_token = self._consume(TokenType.RIGHT_BRACKET, "Expected ']' after index")
        
        span = SourceSpan(left.span.start, end_token.location)
        return BinaryOp(left, "[]", index, span)
    
    # Utility methods
    
    def _match(self, token_type: TokenType) -> bool:
        """Check if current token matches type and consume if so."""
        if self._check(token_type):
            self._advance()
            return True
        return False
    
    def _check(self, token_type: TokenType) -> bool:
        """Check if current token matches type without consuming."""
        if self._is_at_end():
            return False
        return self._peek().type == token_type
    
    def _advance(self) -> Token:
        """Consume and return current token."""
        if not self._is_at_end():
            self.current += 1
        return self._previous()
    
    def _is_at_end(self) -> bool:
        """Check if we've reached the end of tokens."""
        return self.current >= len(self.tokens) or self._peek().type == TokenType.EOF
    
    def _peek(self) -> Token:
        """Return current token without consuming."""
        if self.current < len(self.tokens):
            return self.tokens[self.current]
        # Return EOF token if past end
        return Token(TokenType.EOF, "", None, SourceLocation("<eof>", 0, 0, 0))
    
    def _previous(self) -> Token:
        """Return previous token."""
        if self.current > 0:
            return self.tokens[self.current - 1]
        return self.tokens[0] if self.tokens else self._peek()
    
    def _consume(self, token_type: TokenType, message: str) -> Token:
        """Consume token of expected type or raise error."""
        if self._check(token_type):
            return self._advance()
        
        current_token = self._peek()
        raise create_unexpected_token_error(token_type, current_token)
    
    def _consume_variable_name(self, message: str) -> Token:
        """Consume a variable name, allowing mathematical keywords as identifiers."""
        current_token = self._peek()
        
        # Allow regular identifiers
        if current_token.type == TokenType.IDENTIFIER:
            return self._advance()
        
        # Allow mathematical keywords as identifiers in variable declaration context
        from ..lexer.tokens import MATH_KEYWORDS
        if current_token.lexeme in MATH_KEYWORDS:
            # Create a new token with IDENTIFIER type but same lexeme
            identifier_token = Token(
                TokenType.IDENTIFIER,
                current_token.lexeme,
                current_token.lexeme,  # Set value to lexeme for identifiers
                current_token.location
            )
            self._advance()  # Consume the original token
            return identifier_token
        
        raise create_unexpected_token_error(TokenType.IDENTIFIER, current_token)
    
    def _check_statement_terminator(self) -> bool:
        """Check for statement terminators."""
        return (self._check(TokenType.SEMICOLON) or 
                self._check(TokenType.NEWLINE) or
                self._check(TokenType.EOF) or
                self._check(TokenType.RIGHT_BRACE))
    
    def _consume_statement_terminator(self):
        """Consume optional statement terminator."""
        if self._match(TokenType.SEMICOLON) or self._match(TokenType.NEWLINE):
            return
        # Allow implicit terminators at end of block or file
    
    def _create_expression_statement(self, expr: Expression) -> Statement:
        """Create an expression statement wrapper."""
        # For now, just return the expression as-is
        # In a full implementation, would have ExpressionStatement AST node
        return expr
    
    # Placeholder implementations for incomplete features
    
    def _parse_impl_block(self) -> Item:
        """Parse impl block (placeholder)."""
        raise create_unexpected_token_error("complete implementation", self._peek())
    
    def _parse_type_alias(self, visibility: str) -> Item:
        """Parse type alias (placeholder)."""
        raise create_unexpected_token_error("complete implementation", self._peek())
    
    def _parse_use_statement(self) -> Item:
        """Parse use statement (placeholder)."""
        raise create_unexpected_token_error("complete implementation", self._peek())
    
    def _parse_attribute(self) -> str:
        """Parse an attribute like @differentiable."""
        self._consume(TokenType.AT, "Expected '@'")
        
        attr_token = self._consume(TokenType.IDENTIFIER, "Expected attribute name")
        return attr_token.lexeme


def parse_string(source: str, filename: str = "<string>") -> Program:
    """
    Convenience function to parse a source string.
    
    Args:
        source: Source code string
        filename: Filename for error reporting
        
    Returns:
        Program AST
        
    Raises:
        ParseError: If parsing fails
    """
    from ..lexer import tokenize_string
    
    tokens = tokenize_string(source, filename)
    parser = Parser(tokens)
    return parser.parse()


def parse_file(filepath: str) -> Program:
    """
    Convenience function to parse a source file.
    
    Args:
        filepath: Path to source file
        
    Returns:
        Program AST
        
    Raises:
        ParseError: If parsing fails
        IOError: If file cannot be read
    """
    from ..lexer import tokenize_file
    
    tokens = tokenize_file(filepath)
    parser = Parser(tokens)
    return parser.parse()
