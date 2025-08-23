"""
Semantic analysis error handling for NeuralScript.

Provides comprehensive error reporting for semantic analysis including
type errors, scope resolution errors, dimensional analysis errors, and
memory safety violations.

Author: xwest
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

from ..lexer.tokens import SourceLocation
from ..lexer.errors import Diagnostic
from ..parser.ast_nodes import ASTNode


class SemanticError(Exception):
    """
    Exception raised when semantic analysis encounters a fatal error.
    
    Contains detailed diagnostic information for error reporting.
    """
    
    def __init__(
        self, 
        message: str, 
        location: SourceLocation,
        node: Optional[ASTNode] = None,
        code: Optional[str] = None,
        help_text: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        related_locations: Optional[List[SourceLocation]] = None
    ):
        super().__init__(message)
        self.diagnostic = Diagnostic(
            message=message,
            location=location, 
            severity="error",
            code=code,
            help_text=help_text,
            suggestions=suggestions
        )
        self.node = node
        self.related_locations = related_locations or []
    
    def __str__(self) -> str:
        result = str(self.diagnostic)
        
        # Add related locations if any
        if self.related_locations:
            result += "\nRelated locations:\n"
            for loc in self.related_locations:
                result += f"  --> {loc}\n"
        
        return result


class SemanticWarning:
    """
    Represents a semantic warning that doesn't stop compilation.
    """
    
    def __init__(
        self, 
        message: str,
        location: SourceLocation,
        node: Optional[ASTNode] = None,
        code: Optional[str] = None,
        help_text: Optional[str] = None,
        suggestions: Optional[List[str]] = None
    ):
        self.diagnostic = Diagnostic(
            message=message,
            location=location,
            severity="warning", 
            code=code,
            help_text=help_text,
            suggestions=suggestions
        )
        self.node = node
    
    def __str__(self) -> str:
        return str(self.diagnostic)


# Semantic error codes for categorization
SEMANTIC_ERROR_CODES = {
    # Type errors
    "S001": "Type mismatch",
    "S002": "Undefined type", 
    "S003": "Type inference failed",
    "S004": "Generic type constraint violation",
    "S005": "Invalid type conversion",
    
    # Symbol resolution errors
    "S010": "Undefined symbol",
    "S011": "Symbol redefinition",
    "S012": "Symbol not in scope",
    "S013": "Circular dependency",
    "S014": "Import resolution failed",
    
    # Dimensional analysis errors
    "S020": "Dimensional mismatch",
    "S021": "Invalid unit operation",
    "S022": "Unit conversion failed",
    "S023": "Dimensionless expected",
    
    # Tensor shape errors
    "S030": "Shape mismatch",
    "S031": "Invalid tensor operation",
    "S032": "Shape inference failed",
    "S033": "Broadcast incompatible",
    "S034": "Rank mismatch",
    
    # Memory safety errors  
    "S040": "Use after move",
    "S041": "Multiple mutable borrows",
    "S042": "Borrow outlives owner",
    "S043": "Dangling reference",
    "S044": "Invalid ownership transfer",
    
    # Function/method errors
    "S050": "Arity mismatch",
    "S051": "Missing required parameter",
    "S052": "Duplicate parameter name",
    "S053": "Invalid return type",
    "S054": "Missing return statement",
    
    # Control flow errors
    "S060": "Unreachable code",
    "S061": "Missing else branch",
    "S062": "Invalid break/continue",
    "S063": "Pattern exhaustiveness",
    
    # Compile-time evaluation errors
    "S070": "Non-constant expression in constant context",
    "S071": "Overflow in constant evaluation",
    "S072": "Division by zero in constant",
    "S073": "Invalid constant operation",
}


# Helper functions for creating specific semantic errors

def create_type_mismatch_error(
    expected: str, 
    actual: str, 
    location: SourceLocation,
    node: Optional[ASTNode] = None
) -> SemanticError:
    """Create a type mismatch error."""
    return SemanticError(
        message=f"Type mismatch: expected {expected}, found {actual}",
        location=location,
        node=node,
        code="S001",
        help_text=f"The expression has type '{actual}' but '{expected}' was expected.",
        suggestions=[
            f"Convert {actual} to {expected}",
            "Check the types in this expression",
            "Ensure all operands have compatible types"
        ]
    )


def create_undefined_symbol_error(
    symbol: str, 
    location: SourceLocation,
    node: Optional[ASTNode] = None,
    similar_names: Optional[List[str]] = None
) -> SemanticError:
    """Create an undefined symbol error."""
    suggestions = []
    if similar_names:
        suggestions.extend([f"Did you mean '{name}'?" for name in similar_names[:3]])
    
    suggestions.extend([
        f"Declare '{symbol}' before using it",
        "Check for typos in the symbol name",
        "Ensure the symbol is imported if from another module"
    ])
    
    return SemanticError(
        message=f"Undefined symbol: '{symbol}'",
        location=location,
        node=node,
        code="S010",
        help_text=f"The symbol '{symbol}' is not defined in the current scope.",
        suggestions=suggestions
    )


def create_dimensional_mismatch_error(
    expected_unit: str,
    actual_unit: str,
    operation: str,
    location: SourceLocation,
    node: Optional[ASTNode] = None
) -> SemanticError:
    """Create a dimensional analysis error."""
    return SemanticError(
        message=f"Dimensional mismatch in {operation}: expected {expected_unit}, found {actual_unit}",
        location=location,
        node=node,
        code="S020",
        help_text=f"The operation '{operation}' requires compatible units.",
        suggestions=[
            f"Convert {actual_unit} to {expected_unit}",
            "Check the units in this calculation",
            "Use dimensionless values if appropriate"
        ]
    )


def create_shape_mismatch_error(
    expected_shape: List[int],
    actual_shape: List[int],
    operation: str,
    location: SourceLocation,
    node: Optional[ASTNode] = None
) -> SemanticError:
    """Create a tensor shape mismatch error."""
    return SemanticError(
        message=f"Shape mismatch in {operation}: expected {expected_shape}, found {actual_shape}",
        location=location,
        node=node,
        code="S030",
        help_text=f"The operation '{operation}' requires compatible tensor shapes.",
        suggestions=[
            f"Reshape tensor to {expected_shape}",
            "Check tensor dimensions in this operation",
            "Use broadcasting if supported"
        ]
    )


def create_borrow_checker_error(
    error_type: str,
    symbol: str,
    location: SourceLocation,
    original_location: Optional[SourceLocation] = None,
    node: Optional[ASTNode] = None
) -> SemanticError:
    """Create a borrow checker error."""
    error_messages = {
        "use_after_move": f"Use of moved value '{symbol}'",
        "multiple_mutable_borrow": f"Cannot borrow '{symbol}' as mutable more than once",
        "borrow_outlives_owner": f"Borrowed value '{symbol}' outlives its owner",
        "dangling_reference": f"Reference to '{symbol}' may be dangling"
    }
    
    help_messages = {
        "use_after_move": f"The value '{symbol}' was moved and can no longer be used.",
        "multiple_mutable_borrow": f"Only one mutable borrow of '{symbol}' is allowed at a time.",
        "borrow_outlives_owner": f"The borrowed reference to '{symbol}' must not outlive the owner.",
        "dangling_reference": f"The reference to '{symbol}' may point to deallocated memory."
    }
    
    code_map = {
        "use_after_move": "S040",
        "multiple_mutable_borrow": "S041", 
        "borrow_outlives_owner": "S042",
        "dangling_reference": "S043"
    }
    
    related_locations = [original_location] if original_location else []
    
    return SemanticError(
        message=error_messages.get(error_type, f"Borrow checker error: {error_type}"),
        location=location,
        node=node,
        code=code_map.get(error_type, "S044"),
        help_text=help_messages.get(error_type, "Memory safety violation detected."),
        suggestions=[
            "Check the lifetime of borrowed values",
            "Avoid using values after they've been moved",
            "Consider using reference counting (Rc/Arc) for shared ownership"
        ],
        related_locations=related_locations
    )


def create_arity_mismatch_error(
    function_name: str,
    expected_args: int,
    actual_args: int,
    location: SourceLocation,
    node: Optional[ASTNode] = None
) -> SemanticError:
    """Create a function arity mismatch error."""
    return SemanticError(
        message=f"Function '{function_name}' expects {expected_args} arguments, got {actual_args}",
        location=location,
        node=node,
        code="S050",
        help_text=f"The function call has the wrong number of arguments.",
        suggestions=[
            f"Provide exactly {expected_args} arguments",
            "Check the function signature",
            "Ensure all required parameters are provided"
        ]
    )


def create_unreachable_code_warning(
    location: SourceLocation,
    node: Optional[ASTNode] = None
) -> SemanticWarning:
    """Create an unreachable code warning."""
    return SemanticWarning(
        message="Unreachable code detected",
        location=location,
        node=node,
        code="S060",
        help_text="This code will never be executed.",
        suggestions=[
            "Remove the unreachable code",
            "Check control flow logic",
            "Move code to a reachable location"
        ]
    )


def create_pattern_exhaustiveness_error(
    missing_patterns: List[str],
    location: SourceLocation,
    node: Optional[ASTNode] = None
) -> SemanticError:
    """Create a pattern matching exhaustiveness error."""
    missing_str = ", ".join(missing_patterns)
    return SemanticError(
        message=f"Non-exhaustive pattern match, missing: {missing_str}",
        location=location,
        node=node,
        code="S063",
        help_text="Pattern matching must cover all possible cases.",
        suggestions=[
            f"Add patterns for: {missing_str}",
            "Add a wildcard pattern (_) to catch all remaining cases",
            "Use a default case if appropriate"
        ]
    )
