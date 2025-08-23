"""
NeuralScript Lexer - handles tokenizing source code

Built this over several months. Started simple but got pretty complex
when I added all the Unicode math stuff and unit literals.

TODO: Clean up some of the regex patterns, they're getting messy
TODO: Maybe cache compiled patterns better? 

xwest - started this in early 2024
"""

import re
import unicodedata
from typing import Iterator, List, Optional, Tuple, Union, Set
from io import StringIO
from dataclasses import dataclass

from .tokens import (
    Token, TokenType, SourceLocation, KEYWORDS, OPERATORS, CORE_KEYWORDS, MATH_KEYWORDS,
    MATHEMATICAL_IDENTIFIER_CHARS, FRACTION_VALUES
)
from .errors import (
    LexerError, LexerWarning, create_invalid_character_error,
    create_unterminated_string_error, create_invalid_number_error,
    create_invalid_unicode_error, ErrorRecovery
)


class Lexer:
    """
    NeuralScript lexical analyzer.
    
    Converts source code text into a stream of tokens, handling Unicode
    mathematical symbols, dimensional units, complex numbers, and providing
    comprehensive error recovery.
    """
    
    def __init__(self, source: str, filename: str = "<unknown>"):
        """
        Initialize the lexer with source code.
        
        Args:
            source: Source code string (UTF-8)
            filename: Name of source file for error reporting
        """
        self.source = source
        self.filename = filename
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
        self.errors: List[LexerError] = []
        self.warnings: List[LexerWarning] = []
        
        # Precompile regex patterns for efficiency
        self._compile_patterns()
        
        # Cache for identifier validation
        self._identifier_cache: Set[str] = set()
        
    def _compile_patterns(self):
        """Compile regex patterns used by the lexer."""
        
        # Integer patterns
        self.decimal_pattern = re.compile(r'\d[\d_]*')
        self.binary_pattern = re.compile(r'0[bB][01][01_]*')
        self.octal_pattern = re.compile(r'0[oO][0-7][0-7_]*') 
        self.hex_pattern = re.compile(r'0[xX][0-9a-fA-F][0-9a-fA-F_]*')
        
        # Float patterns
        self.float_pattern = re.compile(
            r'(?:\d[\d_]*)?\.(?:\d[\d_]*)?(?:[eE][+-]?\d[\d_]*)?(?:f32|f64|f128)?|'
            r'\d[\d_]*[eE][+-]?\d[\d_]*(?:f32|f64|f128)?|'
            r'\d[\d_]*(?:f32|f64|f128)'
        )
        
        # Complex pattern (e.g., 3+4i, 2.5-1.8i, 100_+50_i)
        self.complex_pattern = re.compile(
            r'(?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:[eE][+-]?\d[\d_]*)?[+-](?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:[eE][+-]?\d[\d_]*)?i'
        )
        
        # Unit pattern (e.g., 5.0m, 9.8m/s², 3.2kg⋅m/s, 100.0_m, 42_kg)
        self.unit_pattern = re.compile(
            r'(?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:[eE][+-]?\d[\d_]*)?[a-zA-ZΑ-Ωα-ω][a-zA-ZΑ-Ωα-ω⁰-⁹₀-₉⋅/²³]*'
        )
        
        # Identifier pattern (including Unicode mathematical symbols)
        self.identifier_pattern = re.compile(
            r'[a-zA-Z_αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]'
            r'[a-zA-Z0-9_αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹′″‴]*'
        )
        
        # String patterns
        self.string_pattern = re.compile(r'"([^"\\\\]|\\\\.)*"?')
        self.raw_string_pattern = re.compile(r'r"([^"])*"?')
        self.format_string_pattern = re.compile(r'f"([^"\\\\]|\\\\.)*"?')
        self.char_pattern = re.compile(r"'([^'\\\\]|\\\\.)?'?")
        
        # Comment patterns
        self.line_comment_pattern = re.compile(r'//.*$', re.MULTILINE)
        self.block_comment_pattern = re.compile(r'/\*.*?\*/', re.DOTALL)
        
        # Whitespace pattern
        self.whitespace_pattern = re.compile(r'[ \t]+')
        
    def tokenize(self) -> List[Token]:
        """
        Tokenize the entire source code.
        
        Returns:
            List of tokens including EOF token
        """
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens.clear()
        self.errors.clear()
        self.warnings.clear()
        
        while self.pos < len(self.source):
            try:
                self._skip_whitespace_and_comments()
                
                if self.pos >= len(self.source):
                    break
                    
                # Try to tokenize the next token
                token = self._next_token()
                if token:
                    self.tokens.append(token)
                    
            except LexerError as e:
                self.errors.append(e)
                # Try to recover by skipping the problematic character
                self._advance()
                
        # Add EOF token
        eof_location = SourceLocation(self.filename, self.line, self.column, self.pos)
        self.tokens.append(Token(TokenType.EOF, "", None, eof_location))
        
        return self.tokens
    
    def _next_token(self) -> Optional[Token]:
        """Get the next token from the source."""
        if self.pos >= len(self.source):
            return None
            
        start_pos = self.pos
        start_line = self.line
        start_column = self.column
        
        current_char = self.source[self.pos]
        
        # Newlines
        if current_char == '\n':
            self._advance()
            return Token(
                TokenType.NEWLINE, 
                '\n', 
                None,
                SourceLocation(self.filename, start_line, start_column, start_pos)
            )
        
        # Try complex numbers first (before regular numbers)
        if self._try_complex_literal():
            return self._tokenize_complex_literal(start_line, start_column, start_pos)
            
        # Try unit literals
        if self._try_unit_literal():
            return self._tokenize_unit_literal(start_line, start_column, start_pos)
            
        # Numbers (integers and floats)
        if current_char.isdigit() or (current_char == '.' and self._peek().isdigit()):
            return self._tokenize_number(start_line, start_column, start_pos)
            
        # Identifiers and keywords
        if self._is_identifier_start(current_char):
            return self._tokenize_identifier_or_keyword(start_line, start_column, start_pos)
            
        # String literals
        if current_char == '"':
            return self._tokenize_string(start_line, start_column, start_pos)
        elif current_char == 'r' and self._peek() == '"':
            return self._tokenize_raw_string(start_line, start_column, start_pos)
        elif current_char == 'f' and self._peek() == '"':
            return self._tokenize_format_string(start_line, start_column, start_pos)
            
        # Character literals
        if current_char == "'":
            return self._tokenize_character(start_line, start_column, start_pos)
            
        # Operators and punctuation (multi-character first)
        for op_len in [3, 2, 1]:  # Check longer operators first
            if self.pos + op_len <= len(self.source):
                potential_op = self.source[self.pos:self.pos + op_len]
                if potential_op in OPERATORS:
                    self._advance_by(op_len)
                    token_type = OPERATORS[potential_op]
                    
                    # Handle fraction literals with their numeric values
                    if token_type == TokenType.FRACTION:
                        value = FRACTION_VALUES.get(potential_op, None)
                    else:
                        value = None
                    
                    return Token(
                        token_type,
                        potential_op,
                        value,
                        SourceLocation(self.filename, start_line, start_column, start_pos)
                    )
        
        # Single characters that aren't recognized
        raise create_invalid_character_error(
            current_char,
            SourceLocation(self.filename, start_line, start_column, start_pos)
        )
    
    def _try_complex_literal(self) -> bool:
        """Check if current position starts a complex literal."""
        remaining = self.source[self.pos:]
        match = self.complex_pattern.match(remaining)
        return match is not None
        
    def _try_unit_literal(self) -> bool:
        """Check if current position starts a unit literal."""
        remaining = self.source[self.pos:]
        match = self.unit_pattern.match(remaining)
        return match is not None
    
    def _tokenize_complex_literal(self, line: int, column: int, offset: int) -> Token:
        """Tokenize a complex number literal."""
        remaining = self.source[self.pos:]
        match = self.complex_pattern.match(remaining)
        
        if not match:
            raise create_invalid_number_error(
                remaining[:10], 
                SourceLocation(self.filename, line, column, offset),
                "Invalid complex number format"
            )
            
        lexeme = match.group(0)
        self._advance_by(len(lexeme))
        
        # Parse the complex number
        try:
            # Remove 'i' and parse real/imaginary parts
            without_i = lexeme[:-1]
            if '+' in without_i[1:]:  # Skip first char for negative numbers
                parts = without_i.split('+')
                real = float(parts[0].replace('_', ''))
                imag = float(parts[1].replace('_', ''))
            else:
                parts = without_i.rsplit('-', 1)
                if len(parts) == 2:
                    real = float(parts[0].replace('_', ''))
                    imag = -float(parts[1].replace('_', ''))
                else:
                    real = 0.0
                    imag = float(parts[0].replace('_', ''))
            
            complex_value = complex(real, imag)
            
        except ValueError:
            raise create_invalid_number_error(
                lexeme,
                SourceLocation(self.filename, line, column, offset),
                "Cannot parse complex number components"
            )
        
        return Token(
            TokenType.COMPLEX,
            lexeme,
            complex_value,
            SourceLocation(self.filename, line, column, offset)
        )
    
    def _tokenize_unit_literal(self, line: int, column: int, offset: int) -> Token:
        """Tokenize a dimensional unit literal."""
        remaining = self.source[self.pos:]
        match = self.unit_pattern.match(remaining)
        
        if not match:
            raise create_invalid_number_error(
                remaining[:10],
                SourceLocation(self.filename, line, column, offset),
                "Invalid unit literal format"
            )
            
        lexeme = match.group(0)
        self._advance_by(len(lexeme))
        
        # Parse numeric part and unit part
        # This took me way too long to get right with the underscores...
        i = 0
        while i < len(lexeme) and (lexeme[i].isdigit() or lexeme[i] in '.eE+-_'):
            i += 1
            
        numeric_part = lexeme[:i]
        unit_part = lexeme[i:]
        
        try:
            # Remove underscores before parsing (finally works!)
            clean_numeric = numeric_part.replace('_', '')
            if '.' in clean_numeric or 'e' in clean_numeric.lower():
                numeric_value = float(clean_numeric)
            else:
                numeric_value = int(clean_numeric)
        except ValueError:
            # Debug: print(f"Failed to parse: '{lexeme}', numeric: '{numeric_part}'")
            raise create_invalid_number_error(
                lexeme,
                SourceLocation(self.filename, line, column, offset),
                "Invalid numeric part in unit literal"
            )
        
        # Store as tuple (value, unit_string)
        unit_value = (numeric_value, unit_part)
        
        return Token(
            TokenType.UNIT_LITERAL,
            lexeme,
            unit_value,
            SourceLocation(self.filename, line, column, offset)
        )
    
    def _tokenize_number(self, line: int, column: int, offset: int) -> Token:
        """Tokenize integer or float literals."""
        remaining = self.source[self.pos:]
        
        # Try binary first
        if remaining.startswith(('0b', '0B')):
            return self._tokenize_integer(self.binary_pattern, TokenType.INTEGER_BINARY, 2, line, column, offset)
        
        # Try octal
        if remaining.startswith(('0o', '0O')):
            return self._tokenize_integer(self.octal_pattern, TokenType.INTEGER_OCTAL, 8, line, column, offset)
            
        # Try hexadecimal
        if remaining.startswith(('0x', '0X')):
            return self._tokenize_integer(self.hex_pattern, TokenType.INTEGER_HEXADECIMAL, 16, line, column, offset)
            
        # Try float
        float_match = self.float_pattern.match(remaining)
        if float_match and ('.' in float_match.group(0) or 'e' in float_match.group(0).lower() or 
                           float_match.group(0).endswith(('f32', 'f64', 'f128'))):
            return self._tokenize_float(float_match, line, column, offset)
            
        # Default to decimal integer
        return self._tokenize_integer(self.decimal_pattern, TokenType.INTEGER_DECIMAL, 10, line, column, offset)
    
    def _tokenize_integer(self, pattern, token_type: TokenType, base: int, line: int, column: int, offset: int) -> Token:
        """Tokenize an integer literal."""
        remaining = self.source[self.pos:]
        match = pattern.match(remaining)
        
        if not match:
            raise create_invalid_number_error(
                remaining[:10],
                SourceLocation(self.filename, line, column, offset),
                f"Invalid integer literal in base {base}"
            )
            
        lexeme = match.group(0)
        self._advance_by(len(lexeme))
        
        # Remove underscores and base prefix for parsing
        clean_lexeme = lexeme.replace('_', '')
        if base == 2:
            clean_lexeme = clean_lexeme[2:]  # Remove '0b'
        elif base == 8:
            clean_lexeme = clean_lexeme[2:]  # Remove '0o'
        elif base == 16:
            clean_lexeme = clean_lexeme[2:]  # Remove '0x'
            
        try:
            value = int(clean_lexeme, base)
        except ValueError:
            raise create_invalid_number_error(
                lexeme,
                SourceLocation(self.filename, line, column, offset),
                f"Cannot parse integer in base {base}"
            )
            
        return Token(token_type, lexeme, value, SourceLocation(self.filename, line, column, offset))
    
    def _tokenize_float(self, match, line: int, column: int, offset: int) -> Token:
        """Tokenize a floating-point literal."""
        lexeme = match.group(0)
        self._advance_by(len(lexeme))
        
        # FIXME: The type suffix parsing is a bit messy here
        clean_lexeme = lexeme
        if lexeme.endswith(('f32', 'f64', 'f128')):
            clean_lexeme = lexeme[:-3] if lexeme.endswith('f32') else lexeme[:-3]
            
        try:
            value = float(clean_lexeme.replace('_', ''))
        except ValueError:
            raise create_invalid_number_error(
                lexeme,
                SourceLocation(self.filename, line, column, offset),
                "Cannot parse floating-point number"
            )
            
        return Token(TokenType.FLOAT, lexeme, value, SourceLocation(self.filename, line, column, offset))
    
    def _tokenize_identifier_or_keyword(self, line: int, column: int, offset: int, allow_math_keywords: bool = False) -> Token:
        """Tokenize an identifier or keyword.
        
        Args:
            line: Source line number
            column: Source column number
            offset: Source offset
            allow_math_keywords: If True, mathematical keywords can be used as identifiers
        """
        start_pos = self.pos
        
        # First character is already validated as identifier start
        self._advance()
        
        # Continue while we have identifier continuation characters
        while self.pos < len(self.source) and self._is_identifier_continue(self.source[self.pos]):
            self._advance()
            
        lexeme = self.source[start_pos:self.pos]
        
        # Check if it's a keyword, respecting context
        if allow_math_keywords and lexeme in MATH_KEYWORDS:
            # In variable declaration context, treat math keywords as identifiers
            token_type = TokenType.IDENTIFIER
            value = lexeme
        else:
            # Normal keyword recognition
            token_type = KEYWORDS.get(lexeme, TokenType.IDENTIFIER)
            value = None if token_type != TokenType.IDENTIFIER else lexeme
        
        # Handle boolean literals
        if token_type in (TokenType.TRUE, TokenType.FALSE):
            value = token_type == TokenType.TRUE
            
        return Token(token_type, lexeme, value, SourceLocation(self.filename, line, column, offset))
    
    def tokenize_with_context(self, context_hints: Optional[List[str]] = None) -> List[Token]:
        """Tokenize with parsing context hints for context-sensitive keywords.
        
        Args:
            context_hints: List of context hints like ['variable_declaration']
        """
        # For now, use standard tokenization - context will be handled by parser
        # The parser will need to re-tokenize specific sections with context
        return self.tokenize()
    
    def _tokenize_string(self, line: int, column: int, offset: int) -> Token:
        """Tokenize a regular string literal."""
        start_pos = self.pos
        self._advance()  # Skip opening quote
        
        value_parts = []
        
        while self.pos < len(self.source) and self.source[self.pos] != '"':
            if self.source[self.pos] == '\\':
                # Handle escape sequences
                if self.pos + 1 >= len(self.source):
                    break
                self._advance()  # Skip backslash
                escaped_char = self._handle_escape_sequence()
                value_parts.append(escaped_char)
            elif self.source[self.pos] == '\n':
                # Newlines in strings
                value_parts.append('\n')
                self._advance()
            else:
                value_parts.append(self.source[self.pos])
                self._advance()
        
        if self.pos >= len(self.source) or self.source[self.pos] != '"':
            raise create_unterminated_string_error(
                '"',
                SourceLocation(self.filename, line, column, offset)
            )
            
        self._advance()  # Skip closing quote
        
        lexeme = self.source[start_pos:self.pos]
        value = ''.join(value_parts)
        
        return Token(TokenType.STRING, lexeme, value, SourceLocation(self.filename, line, column, offset))
    
    def _tokenize_raw_string(self, line: int, column: int, offset: int) -> Token:
        """Tokenize a raw string literal."""
        start_pos = self.pos
        self._advance()  # Skip 'r'
        self._advance()  # Skip opening quote
        
        value_parts = []
        
        while self.pos < len(self.source) and self.source[self.pos] != '"':
            value_parts.append(self.source[self.pos])
            self._advance()
        
        if self.pos >= len(self.source) or self.source[self.pos] != '"':
            raise create_unterminated_string_error(
                'r"',
                SourceLocation(self.filename, line, column, offset)
            )
            
        self._advance()  # Skip closing quote
        
        lexeme = self.source[start_pos:self.pos]
        value = ''.join(value_parts)
        
        return Token(TokenType.STRING, lexeme, value, SourceLocation(self.filename, line, column, offset))
    
    def _tokenize_format_string(self, line: int, column: int, offset: int) -> Token:
        """Tokenize a format string literal (f-string)."""
        start_pos = self.pos
        self._advance()  # Skip 'f'
        self._advance()  # Skip opening quote
        
        # TODO: For now, treat as regular string (format string parsing is complex)
        # In a full implementation, this would parse format expressions
        # Need to handle {} expressions properly eventually
        value_parts = []
        
        while self.pos < len(self.source) and self.source[self.pos] != '"':
            if self.source[self.pos] == '\\':
                if self.pos + 1 >= len(self.source):
                    break
                self._advance()
                escaped_char = self._handle_escape_sequence()
                value_parts.append(escaped_char)
            else:
                value_parts.append(self.source[self.pos])
                self._advance()
        
        if self.pos >= len(self.source) or self.source[self.pos] != '"':
            raise create_unterminated_string_error(
                'f"',
                SourceLocation(self.filename, line, column, offset)
            )
            
        self._advance()  # Skip closing quote
        
        lexeme = self.source[start_pos:self.pos]
        value = ''.join(value_parts)
        
        return Token(TokenType.STRING, lexeme, value, SourceLocation(self.filename, line, column, offset))
    
    def _tokenize_character(self, line: int, column: int, offset: int) -> Token:
        """Tokenize a character literal."""
        start_pos = self.pos
        self._advance()  # Skip opening quote
        
        if self.pos >= len(self.source):
            raise create_unterminated_string_error(
                "'",
                SourceLocation(self.filename, line, column, offset)
            )
        
        char_value = None
        
        if self.source[self.pos] == '\\':
            # Escape sequence
            self._advance()
            char_value = self._handle_escape_sequence()
        elif self.source[self.pos] == "'":
            raise LexerError(
                "Empty character literal",
                SourceLocation(self.filename, line, column, offset),
                code="L009"
            )
        else:
            char_value = self.source[self.pos]
            self._advance()
        
        if self.pos >= len(self.source) or self.source[self.pos] != "'":
            raise create_unterminated_string_error(
                "'",
                SourceLocation(self.filename, line, column, offset)
            )
            
        self._advance()  # Skip closing quote
        
        lexeme = self.source[start_pos:self.pos]
        
        return Token(TokenType.CHARACTER, lexeme, char_value, SourceLocation(self.filename, line, column, offset))
    
    def _handle_escape_sequence(self) -> str:
        """Handle escape sequences in strings and characters."""
        if self.pos >= len(self.source):
            return '\\'
            
        escape_char = self.source[self.pos]
        self._advance()
        
        escape_sequences = {
            'n': '\n',
            't': '\t', 
            'r': '\r',
            '\\': '\\',
            '"': '"',
            "'": "'",
            '0': '\0',
        }
        
        if escape_char in escape_sequences:
            return escape_sequences[escape_char]
        elif escape_char == 'x':
            # Hexadecimal escape \xHH
            if self.pos + 1 < len(self.source):
                hex_digits = self.source[self.pos:self.pos + 2]
                if all(c in '0123456789abcdefABCDEF' for c in hex_digits):
                    self._advance_by(2)
                    return chr(int(hex_digits, 16))
        elif escape_char == 'u':
            # Unicode escape \uHHHH
            if self.pos + 3 < len(self.source):
                hex_digits = self.source[self.pos:self.pos + 4]
                if all(c in '0123456789abcdefABCDEF' for c in hex_digits):
                    self._advance_by(4)
                    return chr(int(hex_digits, 16))
        
        # Invalid escape sequence - return as literal
        return escape_char
    
    def _is_identifier_start(self, char: str) -> bool:
        """Check if character can start an identifier."""
        return (char.isalpha() or char == '_' or 
                char in MATHEMATICAL_IDENTIFIER_CHARS or
                unicodedata.category(char) in ('Lu', 'Ll', 'Lt', 'Lm', 'Lo', 'Nl'))
    
    def _is_identifier_continue(self, char: str) -> bool:
        """Check if character can continue an identifier."""
        return (char.isalnum() or char == '_' or
                char in MATHEMATICAL_IDENTIFIER_CHARS or
                unicodedata.category(char) in ('Lu', 'Ll', 'Lt', 'Lm', 'Lo', 'Nl', 'Mn', 'Mc', 'Nd', 'Pc'))
    
    def _skip_whitespace_and_comments(self):
        """Skip whitespace and comments."""
        while self.pos < len(self.source):
            # Skip whitespace
            if self.source[self.pos].isspace() and self.source[self.pos] != '\n':
                self._advance()
                continue
                
            # Skip line comments //
            if (self.pos + 1 < len(self.source) and 
                self.source[self.pos:self.pos + 2] == '//'):
                while self.pos < len(self.source) and self.source[self.pos] != '\n':
                    self._advance()
                continue
                
            # Skip block comments /* */
            if (self.pos + 1 < len(self.source) and
                self.source[self.pos:self.pos + 2] == '/*'):
                self._advance_by(2)
                while (self.pos + 1 < len(self.source) and 
                       self.source[self.pos:self.pos + 2] != '*/'):
                    self._advance()
                if self.pos + 1 < len(self.source):
                    self._advance_by(2)  # Skip closing */
                continue
                
            break
    
    def _advance(self):
        """Advance position by one character, updating line/column."""
        if self.pos < len(self.source):
            if self.source[self.pos] == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            self.pos += 1
    
    def _advance_by(self, count: int):
        """Advance position by multiple characters."""
        for _ in range(count):
            if self.pos < len(self.source):
                self._advance()
    
    def _peek(self, offset: int = 1) -> str:
        """Peek at character ahead without advancing."""
        peek_pos = self.pos + offset
        if peek_pos < len(self.source):
            return self.source[peek_pos]
        return '\0'
    
    def has_errors(self) -> bool:
        """Check if lexer encountered any errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if lexer encountered any warnings."""
        return len(self.warnings) > 0
    
    def get_diagnostics(self) -> List[Union[LexerError, LexerWarning]]:
        """Get all diagnostics (errors and warnings)."""
        return self.errors + self.warnings


def tokenize_string(source: str, filename: str = "<string>") -> List[Token]:
    """
    Convenience function to tokenize a source string.
    
    Args:
        source: Source code string
        filename: Filename for error reporting
        
    Returns:
        List of tokens
        
    Raises:
        LexerError: If lexing fails
    """
    lexer = Lexer(source, filename)
    tokens = lexer.tokenize()
    
    if lexer.has_errors():
        # Raise the first error encountered
        raise lexer.errors[0]
        
    return tokens


def tokenize_file(filepath: str) -> List[Token]:
    """
    Convenience function to tokenize a source file.
    
    Args:
        filepath: Path to source file
        
    Returns:
        List of tokens
        
    Raises:
        LexerError: If lexing fails
        IOError: If file cannot be read
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
        
    return tokenize_string(source, filepath)
