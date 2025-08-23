"""
NeuralScript Lexer Package

Implements a from-scratch lexical analyzer (tokenizer) for the NeuralScript language.
Features support for Unicode mathematical symbols, dimensional units, complex numbers,
and context-sensitive keyword recognition.

Key Features:
- UTF-8 Unicode support with mathematical symbols (∑, ∂, ∇, etc.)
- Custom numeric literals (complex: 3+4i, units: 5.0m/s²)
- Context-sensitive keywords
- Error recovery and diagnostics
- Zero-copy tokenization where possible
- Source location tracking for IDE features

Author: xwest
"""

from .tokens import Token, TokenType, SourceLocation
from .lexer import Lexer
from .errors import LexerError, LexerWarning

__all__ = [
    "Lexer", 
    "Token", 
    "TokenType", 
    "SourceLocation",
    "LexerError",
    "LexerWarning",
]
