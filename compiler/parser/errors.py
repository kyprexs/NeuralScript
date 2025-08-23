"""
Error handling for the NeuralScript parser.

Provides comprehensive error reporting with source location information,
error recovery strategies, and IDE-friendly diagnostics for syntax errors.

Author: xwest
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

from ..lexer.tokens import Token, TokenType, SourceLocation
from ..lexer.errors import Diagnostic


class ParseError(Exception):
    """
    Exception raised when the parser encounters a fatal syntax error.
    
    Contains detailed diagnostic information for error reporting.
    """
    
    def __init__(
        self, 
        message: str, 
        location: SourceLocation,
        token: Optional[Token] = None,
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
        self.token = token
    
    def __str__(self) -> str:
        return str(self.diagnostic)


class ParseWarning:
    """
    Represents a parser warning that doesn't stop compilation.
    """
    
    def __init__(
        self, 
        message: str,
        location: SourceLocation,
        token: Optional[Token] = None,
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
        self.token = token
    
    def __str__(self) -> str:
        return str(self.diagnostic)


class SyntaxErrorRecovery:
    """
    Utilities for error recovery in the parser.
    
    Provides strategies to continue parsing after encountering syntax errors,
    allowing the collection of multiple errors in a single pass.
    """
    
    # Token types that typically indicate statement boundaries for recovery
    STATEMENT_BOUNDARIES = {
        TokenType.SEMICOLON,
        TokenType.NEWLINE,
        TokenType.LEFT_BRACE,
        TokenType.RIGHT_BRACE,
        TokenType.LET,
        TokenType.CONST,
        TokenType.FN,
        TokenType.STRUCT,
        TokenType.TRAIT,
        TokenType.IMPL,
        TokenType.IF,
        TokenType.WHILE,
        TokenType.FOR,
        TokenType.RETURN,
        TokenType.EOF,
    }
    
    # Token types that indicate expression boundaries
    EXPRESSION_BOUNDARIES = {
        TokenType.SEMICOLON,
        TokenType.COMMA,
        TokenType.RIGHT_PAREN,
        TokenType.RIGHT_BRACKET,
        TokenType.RIGHT_BRACE,
        TokenType.NEWLINE,
        TokenType.EOF,
    }
    
    @staticmethod
    def suggest_missing_token(expected: TokenType, found: Token) -> List[str]:
        """Suggest what token might be missing."""
        suggestions = []
        
        token_suggestions = {
            TokenType.SEMICOLON: ["Add a semicolon ';' to end the statement"],
            TokenType.RIGHT_PAREN: ["Add a closing parenthesis ')'"],
            TokenType.RIGHT_BRACKET: ["Add a closing bracket ']'"],
            TokenType.RIGHT_BRACE: ["Add a closing brace '}'"],
            TokenType.LEFT_BRACE: ["Add an opening brace '{' to start a block"],
            TokenType.COLON: ["Add a colon ':' after the type annotation"],
            TokenType.ARROW: ["Add an arrow '->' before the return type"],
            TokenType.ASSIGN: ["Add an assignment operator '='"],
        }
        
        if expected in token_suggestions:
            suggestions.extend(token_suggestions[expected])
        
        return suggestions
    
    @staticmethod
    def suggest_operator_corrections(invalid_op: str) -> List[str]:
        """Suggest corrections for invalid operators in expressions."""
        corrections = {
            "=": ["Use '==' for comparison", "Use '=' for assignment"],
            "!": ["Use '!=' for not equal", "Use '¬' for logical not"],
            "&": ["Use '&&' for logical and", "Use '∧' for logical and", "Use '&' for bitwise and"],
            "|": ["Use '||' for logical or", "Use '∨' for logical or", "Use '|' for bitwise or"],
            "*": ["Use '*' for multiplication", "Use '⊙' for dot product", "Use '⊗' for tensor product"],
        }
        
        return corrections.get(invalid_op, [])
    
    @staticmethod
    def suggest_keyword_in_context(context: str, found_token: Token) -> List[str]:
        """Suggest appropriate keywords based on parsing context."""
        suggestions = []
        
        context_keywords = {
            "function_declaration": ["fn", "async fn", "pub fn"],
            "variable_declaration": ["let", "let mut", "const"],
            "type_declaration": ["struct", "enum", "trait", "type"],
            "control_flow": ["if", "while", "for", "loop", "match"],
            "expression": ["true", "false", "return", "break", "continue"],
        }
        
        if context in context_keywords:
            suggestions.extend([f"Did you mean '{keyword}'?" for keyword in context_keywords[context]])
        
        return suggestions
    
    @staticmethod
    def synchronize_to_statement_boundary(tokens: List[Token], current_pos: int) -> int:
        """
        Synchronize parser to the next likely statement boundary.
        
        Returns the position to resume parsing from.
        """
        while current_pos < len(tokens):
            token = tokens[current_pos]
            
            if token.type in SyntaxErrorRecovery.STATEMENT_BOUNDARIES:
                # Skip the boundary token and return next position
                return current_pos + 1
                
            current_pos += 1
        
        return current_pos
    
    @staticmethod
    def synchronize_to_expression_boundary(tokens: List[Token], current_pos: int) -> int:
        """
        Synchronize parser to the next likely expression boundary.
        
        Returns the position to resume parsing from.
        """
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        
        while current_pos < len(tokens):
            token = tokens[current_pos]
            
            # Track nesting depth
            if token.type == TokenType.LEFT_PAREN:
                paren_depth += 1
            elif token.type == TokenType.RIGHT_PAREN:
                paren_depth -= 1
            elif token.type == TokenType.LEFT_BRACKET:
                bracket_depth += 1
            elif token.type == TokenType.RIGHT_BRACKET:
                bracket_depth -= 1
            elif token.type == TokenType.LEFT_BRACE:
                brace_depth += 1
            elif token.type == TokenType.RIGHT_BRACE:
                brace_depth -= 1
            
            # If we're at depth 0 and hit a boundary, stop here
            if (paren_depth == 0 and bracket_depth == 0 and brace_depth == 0 and
                token.type in SyntaxErrorRecovery.EXPRESSION_BOUNDARIES):
                return current_pos
                
            current_pos += 1
        
        return current_pos


# Common parser error codes for categorization
PARSER_ERROR_CODES = {
    "P001": "Unexpected token",
    "P002": "Expected token not found",
    "P003": "Missing semicolon",
    "P004": "Unclosed delimiter",
    "P005": "Invalid expression",
    "P006": "Missing function body",
    "P007": "Invalid type annotation",
    "P008": "Malformed function signature",
    "P009": "Invalid operator usage",
    "P010": "Unexpected end of input",
    "P011": "Invalid tensor literal",
    "P012": "Mismatched parentheses",
}


# Helper functions for creating common parser errors

def create_unexpected_token_error(expected: Union[TokenType, str], found: Token) -> ParseError:
    """Create an error for an unexpected token."""
    expected_str = expected.name if isinstance(expected, TokenType) else expected
    found_str = found.type.name
    
    suggestions = SyntaxErrorRecovery.suggest_missing_token(expected, found) if isinstance(expected, TokenType) else []
    
    return ParseError(
        message=f"Expected {expected_str}, found {found_str}",
        location=found.location,
        token=found,
        code="P001",
        help_text=f"The parser expected to see {expected_str} at this position, but found {found_str} instead.",
        suggestions=suggestions
    )


def create_missing_token_error(expected: TokenType, location: SourceLocation) -> ParseError:
    """Create an error for a missing expected token."""
    expected_str = expected.name
    
    return ParseError(
        message=f"Expected {expected_str}",
        location=location,
        code="P002",
        help_text=f"The parser expected to see {expected_str} at this position.",
        suggestions=SyntaxErrorRecovery.suggest_missing_token(expected, None)
    )


def create_unclosed_delimiter_error(delimiter: str, open_location: SourceLocation, 
                                  current_location: SourceLocation) -> ParseError:
    """Create an error for an unclosed delimiter."""
    closing_delimiters = {
        "(": ")",
        "[": "]", 
        "{": "}",
        '"': '"',
        "'": "'",
    }
    
    closing = closing_delimiters.get(delimiter, delimiter)
    
    return ParseError(
        message=f"Unclosed delimiter '{delimiter}'",
        location=current_location,
        code="P004",
        help_text=f"The opening '{delimiter}' at {open_location} was never closed.",
        suggestions=[f"Add a closing '{closing}'", "Check for missing delimiters"]
    )


def create_invalid_expression_error(reason: str, location: SourceLocation, 
                                  token: Optional[Token] = None) -> ParseError:
    """Create an error for an invalid expression."""
    return ParseError(
        message=f"Invalid expression: {reason}",
        location=location,
        token=token,
        code="P005",
        help_text=reason,
        suggestions=["Check the expression syntax", "Ensure all operators have operands"]
    )


def create_unexpected_eof_error(expected: str, location: SourceLocation) -> ParseError:
    """Create an error for unexpected end of input."""
    return ParseError(
        message=f"Unexpected end of input, expected {expected}",
        location=location,
        code="P010",
        help_text=f"The parser reached the end of the file while expecting {expected}.",
        suggestions=[f"Add the missing {expected}", "Check for incomplete statements"]
    )
