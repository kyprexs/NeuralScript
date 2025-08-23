"""
Error handling for the NeuralScript lexer.

Provides comprehensive error reporting with source location information,
error recovery suggestions, and IDE-friendly diagnostics.

Author: xwest
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from .tokens import SourceLocation


@dataclass
class Diagnostic:
    """Base class for lexer diagnostics (errors, warnings, info)."""
    message: str
    location: SourceLocation
    severity: str  # "error", "warning", "info", "hint"
    code: Optional[str] = None
    help_text: Optional[str] = None
    suggestions: Optional[List[str]] = None
    
    def __str__(self) -> str:
        severity_prefix = self.severity.upper()
        result = f"{severity_prefix}: {self.message}\n"
        result += f"  --> {self.location}\n"
        
        if self.help_text:
            result += f"  help: {self.help_text}\n"
        
        if self.suggestions:
            result += "  suggestions:\n"
            for suggestion in self.suggestions:
                result += f"    - {suggestion}\n"
        
        return result


class LexerError(Exception):
    """
    Exception raised when the lexer encounters a fatal error.
    
    Contains detailed diagnostic information for error reporting.
    """
    
    def __init__(
        self, 
        message: str, 
        location: SourceLocation,
        code: Optional[str] = None,
        help_text: Optional[str] = None,
        suggestions: Optional[List[str]] = None
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
    
    def __str__(self) -> str:
        return str(self.diagnostic)


class LexerWarning:
    """
    Represents a lexer warning that doesn't stop compilation.
    """
    
    def __init__(
        self, 
        message: str,
        location: SourceLocation,
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
    
    def __str__(self) -> str:
        return str(self.diagnostic)


class ErrorRecovery:
    """
    Utilities for error recovery in the lexer.
    
    Provides strategies to continue lexing after encountering errors,
    allowing the collection of multiple errors in a single pass.
    """
    
    @staticmethod
    def suggest_keyword_corrections(invalid_word: str) -> List[str]:
        """Suggest corrections for misspelled keywords using edit distance."""
        from .tokens import KEYWORDS
        
        suggestions = []
        for keyword in KEYWORDS.keys():
            distance = ErrorRecovery._edit_distance(invalid_word.lower(), keyword)
            if distance <= 2:  # Allow up to 2 character differences
                suggestions.append(keyword)
        
        return sorted(suggestions, key=lambda k: ErrorRecovery._edit_distance(invalid_word.lower(), k))[:3]
    
    @staticmethod
    def suggest_operator_corrections(invalid_op: str) -> List[str]:
        """Suggest corrections for invalid operators."""
        from .tokens import OPERATORS
        
        suggestions = []
        for operator in OPERATORS.keys():
            if len(operator) == len(invalid_op):
                distance = ErrorRecovery._edit_distance(invalid_op, operator)
                if distance <= 1:
                    suggestions.append(operator)
        
        return suggestions[:3]
    
    @staticmethod
    def suggest_unicode_alternatives(char: str) -> List[str]:
        """Suggest Unicode mathematical alternatives for ASCII characters."""
        unicode_alternatives = {
            '*': ['⊙', '⊗', '⋅', '×'],
            '+': ['⊕'],
            '-': ['⊖'],
            '/': ['÷'],
            '=': ['≡', '≈'],
            '!': ['≠', '¬'],
            '<': ['≤', '∈'],
            '>': ['≥'],
            '&': ['∧'],
            '|': ['∨'],
        }
        
        return unicode_alternatives.get(char, [])
    
    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return ErrorRecovery._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


# Common error codes for categorization
ERROR_CODES = {
    "L001": "Invalid character",
    "L002": "Unterminated string literal",
    "L003": "Invalid numeric literal",
    "L004": "Invalid Unicode sequence",
    "L005": "Unexpected character in identifier",
    "L006": "Invalid escape sequence",
    "L007": "Number literal overflow",
    "L008": "Invalid unit specification",
    "L009": "Mismatched quote marks",
    "L010": "Invalid character in numeric literal",
}

# Helper functions for creating common errors
def create_invalid_character_error(char: str, location: SourceLocation) -> LexerError:
    """Create an error for an invalid character."""
    suggestions = ErrorRecovery.suggest_unicode_alternatives(char)
    help_text = None
    
    if suggestions:
        help_text = f"Did you mean one of these Unicode alternatives: {', '.join(suggestions)}?"
    elif char.isprintable():
        help_text = f"The character '{char}' is not valid in NeuralScript source code."
    else:
        help_text = f"Non-printable character (Unicode: U+{ord(char):04X}) is not allowed."
    
    return LexerError(
        message=f"Invalid character: '{char}'",
        location=location,
        code="L001",
        help_text=help_text,
        suggestions=suggestions
    )


def create_unterminated_string_error(quote_type: str, location: SourceLocation) -> LexerError:
    """Create an error for an unterminated string literal."""
    return LexerError(
        message=f"Unterminated string literal",
        location=location,
        code="L002",
        help_text=f"String literals must be closed with a matching {quote_type} quote.",
        suggestions=[f"Add a closing {quote_type} quote", "Check for unescaped quotes in the string"]
    )


def create_invalid_number_error(lexeme: str, location: SourceLocation, reason: str) -> LexerError:
    """Create an error for an invalid numeric literal."""
    return LexerError(
        message=f"Invalid numeric literal: '{lexeme}'",
        location=location,
        code="L003", 
        help_text=reason,
        suggestions=["Check the numeric format", "Ensure proper use of underscores for readability"]
    )


def create_invalid_unicode_error(sequence: str, location: SourceLocation) -> LexerError:
    """Create an error for an invalid Unicode sequence."""
    return LexerError(
        message=f"Invalid Unicode sequence: '{sequence}'",
        location=location,
        code="L004",
        help_text="Unicode sequences must be valid UTF-8.",
        suggestions=["Check the Unicode encoding", "Use proper escape sequences"]
    )
