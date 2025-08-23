"""
NeuralScript Compiler Package

A from-scratch implementation of a high-performance compiler for the NeuralScript
programming language, designed specifically for data science, machine learning,
and scientific computing applications.

Architecture:
    compiler/
    ├── lexer/           # Tokenization and lexical analysis
    ├── parser/          # Syntax analysis and AST generation  
    ├── analyzer/        # Semantic analysis and type checking
    ├── ir/              # Intermediate representation
    ├── optimizer/       # Code optimization passes
    └── codegen/         # Native code generation

Author: xwest
License: MIT
"""

__version__ = "0.1.0-alpha"
__author__ = "xwest"
__email__ = "dev@neuralscript.org"
__license__ = "MIT"

# Core compiler exports (only what exists)
from .lexer import Lexer
from .parser import Parser
from .analyzer import SemanticAnalyzer
from .ir import IRGenerator

__all__ = [
    # Core classes  
    "Lexer",
    "Parser", 
    "SemanticAnalyzer",
    "IRGenerator",
    
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]
