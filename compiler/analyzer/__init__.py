"""
NeuralScript Semantic Analyzer Package

Implements comprehensive semantic analysis including:
- Multi-pass symbol resolution
- Hindley-Milner type inference
- Dimensional analysis and unit checking
- Tensor shape verification
- Ownership and borrowing analysis
- Compile-time evaluation
- Rich error diagnostics

Author: xwest
"""

from .semantic_analyzer import SemanticAnalyzer
from .symbol_table import SymbolTable, Symbol, Scope
from .errors import SemanticError, SemanticWarning

__all__ = [
    # Main analyzer
    "SemanticAnalyzer",
    
    # Symbol management
    "SymbolTable", "Symbol", "Scope",
    
    # Error handling
    "SemanticError", "SemanticWarning",
]
