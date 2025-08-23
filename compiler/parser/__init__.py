"""
NeuralScript Parser Package

Implements a Pratt-based recursive descent parser for the NeuralScript language.
Produces richly annotated Abstract Syntax Trees with full source location information.

Key Features:
- Top-down operator precedence (Pratt parsing)
- Mathematical expression parsing with Unicode operators
- Rich AST nodes with source spans for IDE features
- Error recovery and synchronization
- Comprehensive error diagnostics

Author: xwest
"""

from .ast_nodes import *
from .parser import Parser
from .errors import ParseError, ParseWarning

__all__ = [
    # Core parser
    "Parser",
    
    # AST nodes
    "AST", "ASTNode", 
    "Program", "Item", "Statement", "Expression",
    "FunctionDef", "StructDef", "TraitDef", "ImplBlock",
    "VariableDecl", "Assignment", "IfStatement", "WhileLoop", "ForLoop",
    "BinaryOp", "UnaryOp", "FunctionCall", "Literal", "Identifier",
    "TensorLiteral", "UnitLiteral", "ComplexLiteral",
    
    # Error handling
    "ParseError", "ParseWarning",
]
