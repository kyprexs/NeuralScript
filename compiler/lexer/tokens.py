"""
Token definitions for the NeuralScript lexer.

This module defines all token types supported by NeuralScript, including:
- Keywords (both ASCII and Unicode mathematical keywords)  
- Operators (ASCII and Unicode mathematical operators)
- Literals (integers, floats, complex, strings, units)
- Identifiers (including Unicode mathematical symbols)
- Punctuation and delimiters

Author: xwest
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Optional, Union


class TokenType(Enum):
    """
    Enumeration of all token types in NeuralScript.
    
    Organized by category for clarity and maintainability.
    """
    
    # ========================================================================
    # Special Tokens
    # ========================================================================
    EOF = auto()                    # End of file
    NEWLINE = auto()               # Newline (significant in some contexts)
    INDENT = auto()                # Indentation increase  
    DEDENT = auto()                # Indentation decrease
    COMMENT = auto()               # Comments (for IDE features)
    
    # ========================================================================
    # Literals
    # ========================================================================
    
    # Integer literals
    INTEGER_DECIMAL = auto()        # 42, 1_000_000
    INTEGER_BINARY = auto()         # 0b101010
    INTEGER_OCTAL = auto()          # 0o52
    INTEGER_HEXADECIMAL = auto()    # 0x2A
    
    # Floating-point literals
    FLOAT = auto()                  # 3.14, 1.23e-4, 2.5f32
    
    # Complex literals  
    COMPLEX = auto()                # 3+4i, 2.5-1.8i
    
    # String literals
    STRING = auto()                 # "hello", r"raw", f"formatted {var}"
    CHARACTER = auto()              # 'a', '\n'
    
    # Unit literals (dimensional analysis)
    UNIT_LITERAL = auto()           # 5.0m, 9.8m/s², 3.2kg⋅m/s
    
    # Fraction literals (Unicode mathematical)
    FRACTION = auto()               # ½, ¼, ¾, etc.
    
    # Boolean literals
    TRUE = auto()                   # true
    FALSE = auto()                  # false
    
    # ========================================================================
    # Identifiers and Keywords
    # ========================================================================
    IDENTIFIER = auto()             # variable_name, α, θ₁
    
    # Control flow keywords
    IF = auto()                     # if
    ELSE = auto()                   # else
    ELIF = auto()                   # elif (equivalent to else if)
    WHILE = auto()                  # while
    FOR = auto()                    # for
    LOOP = auto()                   # loop (infinite loop)
    BREAK = auto()                  # break
    CONTINUE = auto()               # continue
    MATCH = auto()                  # match (pattern matching)
    RETURN = auto()                 # return
    YIELD = auto()                  # yield (generators)
    
    # Declaration keywords
    LET = auto()                    # let (variable declaration)
    MUT = auto()                    # mut (mutable)
    CONST = auto()                  # const (compile-time constant)
    STATIC = auto()                 # static
    
    # Function and type keywords
    FN = auto()                     # fn (function)
    STRUCT = auto()                 # struct
    ENUM = auto()                   # enum  
    TRAIT = auto()                  # trait (interface)
    IMPL = auto()                   # impl (implementation)
    TYPE = auto()                   # type (type alias)
    
    # Module system keywords
    MOD = auto()                    # mod (module)
    USE = auto()                    # use (import)
    PUB = auto()                    # pub (public visibility)
    SUPER = auto()                  # super
    SELF_LOWER = auto()             # self
    SELF_UPPER = auto()             # Self
    
    # Concurrency keywords
    ASYNC = auto()                  # async
    AWAIT = auto()                  # await
    
    # Memory management keywords
    UNSAFE = auto()                 # unsafe
    
    # Other keywords
    IN = auto()                     # in (for loops, membership)
    WHERE = auto()                  # where (generic constraints)
    
    # Mathematical keywords (can use Unicode equivalents)
    GRAD = auto()                   # grad (gradient)
    DIV = auto()                    # div (divergence)  
    CURL = auto()                   # curl
    LAPLACE = auto()                # laplace
    SUM = auto()                    # sum
    PROD = auto()                   # prod (product)
    INTEGRAL = auto()               # integral
    DIFF = auto()                   # diff (differentiation)
    
    # ========================================================================
    # Operators
    # ========================================================================
    
    # Arithmetic operators (ASCII)
    PLUS = auto()                   # +
    MINUS = auto()                  # -
    MULTIPLY = auto()               # *
    DIVIDE = auto()                 # /
    MODULO = auto()                 # %
    POWER = auto()                  # ** (exponentiation)
    FLOOR_DIVIDE = auto()           # //
    
    # Arithmetic operators (Unicode mathematical)
    DOT_PRODUCT = auto()            # ⊙ (tensor contraction)
    TENSOR_PRODUCT = auto()         # ⊗ (tensor product)
    ELEMENT_WISE_ADD = auto()       # ⊕ (element-wise addition)
    ELEMENT_WISE_SUB = auto()       # ⊖ (element-wise subtraction)
    SCALAR_PRODUCT = auto()         # ⋅ (scalar/dot product)
    CROSS_PRODUCT = auto()          # × (cross product)
    DIVISION_SYMBOL = auto()        # ÷ (division symbol)
    
    # Assignment operators
    ASSIGN = auto()                 # =
    PLUS_ASSIGN = auto()            # +=
    MINUS_ASSIGN = auto()           # -=
    MULTIPLY_ASSIGN = auto()        # *=
    DIVIDE_ASSIGN = auto()          # /=
    MODULO_ASSIGN = auto()          # %=
    POWER_ASSIGN = auto()           # **=
    
    # Comparison operators (ASCII)
    EQUAL = auto()                  # ==
    NOT_EQUAL = auto()              # !=
    LESS_THAN = auto()              # <
    GREATER_THAN = auto()           # >
    LESS_EQUAL = auto()             # <=
    GREATER_EQUAL = auto()          # >=
    
    # Comparison operators (Unicode mathematical)
    EQUIVALENT = auto()             # ≡ (mathematical equivalence)
    NOT_EQUIVALENT = auto()         # ≠ (not equal)
    LESS_EQUAL_MATH = auto()        # ≤ (less than or equal)
    GREATER_EQUAL_MATH = auto()     # ≥ (greater than or equal)
    APPROXIMATELY = auto()          # ≈ (approximately equal)
    ELEMENT_OF = auto()             # ∈ (element of set)
    NOT_ELEMENT_OF = auto()         # ∉ (not element of set)
    
    # Logical operators (ASCII)
    LOGICAL_AND = auto()            # &&
    LOGICAL_OR = auto()             # ||
    LOGICAL_NOT = auto()            # !
    
    # Logical operators (Unicode mathematical)
    AND_SYMBOL = auto()             # ∧ (logical and)
    OR_SYMBOL = auto()              # ∨ (logical or)
    NOT_SYMBOL = auto()             # ¬ (logical not)
    XOR_SYMBOL = auto()             # ⊕ (exclusive or)
    BICONDITIONAL = auto()          # ↔ (if and only if)
    
    # Mathematical operators (Unicode)
    NABLA = auto()                  # ∇ (gradient/del operator)
    PARTIAL = auto()                # ∂ (partial derivative)
    SUMMATION = auto()              # ∑ (summation)
    PRODUCT_OP = auto()             # ∏ (product)
    INTEGRAL_OP = auto()            # ∫ (integral)
    SQUARE_ROOT = auto()            # √ (square root)
    CUBE_ROOT = auto()              # ∛ (cube root)
    INFINITY = auto()               # ∞ (infinity)
    
    # Bitwise operators
    BIT_AND = auto()                # &
    BIT_OR = auto()                 # |
    BIT_XOR = auto()                # ^
    BIT_NOT = auto()                # ~
    LEFT_SHIFT = auto()             # <<
    RIGHT_SHIFT = auto()            # >>
    
    # ========================================================================
    # Punctuation and Delimiters
    # ========================================================================
    LEFT_PAREN = auto()             # (
    RIGHT_PAREN = auto()            # )
    LEFT_BRACKET = auto()           # [
    RIGHT_BRACKET = auto()          # ]
    LEFT_BRACE = auto()             # {
    RIGHT_BRACE = auto()            # }
    
    SEMICOLON = auto()              # ;
    COMMA = auto()                  # ,
    DOT = auto()                    # .
    COLON = auto()                  # :
    DOUBLE_COLON = auto()           # ::
    ARROW = auto()                  # ->
    FAT_ARROW = auto()              # =>
    QUESTION = auto()               # ?
    AT = auto()                     # @ (decorators/attributes)
    HASH = auto()                   # # (attributes)
    DOLLAR = auto()                 # $ (special syntax)
    
    # Range operators
    RANGE_EXCLUSIVE = auto()        # .. (exclusive range)
    RANGE_INCLUSIVE = auto()        # ..= (inclusive range)
    
    # ========================================================================
    # Error and Recovery Tokens
    # ========================================================================
    INVALID = auto()                # Invalid/unrecognized token
    UNTERMINATED_STRING = auto()    # Unterminated string literal
    INVALID_NUMBER = auto()         # Malformed numeric literal
    INVALID_UNICODE = auto()        # Invalid Unicode sequence


@dataclass(frozen=True)
class SourceLocation:
    """
    Represents a location in the source code.
    
    Used for error reporting, IDE features, and debugging information.
    """
    filename: str
    line: int
    column: int
    offset: int  # Byte offset from start of file
    
    def __str__(self) -> str:
        return f"{self.filename}:{self.line}:{self.column}"
    
    def __repr__(self) -> str:
        return f"SourceLocation({self.filename!r}, {self.line}, {self.column}, {self.offset})"


@dataclass(frozen=True)
class Token:
    """
    Represents a lexical token in the NeuralScript language.
    
    Contains the token type, lexeme (raw text), semantic value,
    and source location for comprehensive error reporting and IDE support.
    """
    type: TokenType
    lexeme: str                     # Raw text from source
    value: Any                      # Parsed/semantic value (e.g., int for INTEGER)
    location: SourceLocation        # Source location
    
    def __str__(self) -> str:
        if self.value is not None and self.value != self.lexeme:
            return f"{self.type.name}({self.lexeme!r} -> {self.value!r})"
        return f"{self.type.name}({self.lexeme!r})"
    
    def __repr__(self) -> str:
        return (f"Token({self.type.name}, {self.lexeme!r}, "
                f"{self.value!r}, {self.location!r})")
    
    @property
    def is_literal(self) -> bool:
        """Check if this token is a literal value."""
        return self.type in {
            TokenType.INTEGER_DECIMAL, TokenType.INTEGER_BINARY, 
            TokenType.INTEGER_OCTAL, TokenType.INTEGER_HEXADECIMAL,
            TokenType.FLOAT, TokenType.COMPLEX, TokenType.STRING,
            TokenType.CHARACTER, TokenType.UNIT_LITERAL, TokenType.FRACTION,
            TokenType.TRUE, TokenType.FALSE
        }
    
    @property
    def is_keyword(self) -> bool:
        """Check if this token is a keyword."""
        return self.type.name in KEYWORDS
    
    @property
    def is_operator(self) -> bool:
        """Check if this token is an operator."""
        return self.type.name in OPERATORS
    
    @property
    def is_identifier(self) -> bool:
        """Check if this token is an identifier."""
        return self.type == TokenType.IDENTIFIER


# Lookup tables for efficient token recognition
# These will be used by the lexer for fast keyword/operator recognition

# Core language keywords that are always reserved
CORE_KEYWORDS = {
    # Control flow
    "if": TokenType.IF,
    "else": TokenType.ELSE, 
    "elif": TokenType.ELIF,
    "while": TokenType.WHILE,
    "for": TokenType.FOR,
    "loop": TokenType.LOOP,
    "break": TokenType.BREAK,
    "continue": TokenType.CONTINUE,
    "match": TokenType.MATCH,
    "return": TokenType.RETURN,
    "yield": TokenType.YIELD,
    
    # Declarations
    "let": TokenType.LET,
    "mut": TokenType.MUT,
    "const": TokenType.CONST,
    "static": TokenType.STATIC,
    
    # Functions and types
    "fn": TokenType.FN,
    "struct": TokenType.STRUCT,
    "enum": TokenType.ENUM,
    "trait": TokenType.TRAIT,
    "impl": TokenType.IMPL,
    "type": TokenType.TYPE,
    
    # Module system
    "mod": TokenType.MOD,
    "use": TokenType.USE,
    "import": TokenType.USE,  # import is alias for use
    "pub": TokenType.PUB,
    "super": TokenType.SUPER,
    "self": TokenType.SELF_LOWER,
    "Self": TokenType.SELF_UPPER,
    
    # Concurrency
    "async": TokenType.ASYNC,
    "await": TokenType.AWAIT,
    
    # Memory management
    "unsafe": TokenType.UNSAFE,
    
    # Other
    "in": TokenType.IN,
    "where": TokenType.WHERE,
    "true": TokenType.TRUE,
    "false": TokenType.FALSE,
}

# Mathematical keywords that can be used as identifiers in most contexts
MATH_KEYWORDS = {
    "grad": TokenType.GRAD,
    "div": TokenType.DIV,
    "curl": TokenType.CURL,
    "laplace": TokenType.LAPLACE,
    "sum": TokenType.SUM,
    "prod": TokenType.PROD,
    "integral": TokenType.INTEGRAL,
    "diff": TokenType.DIFF,
}

# All keywords combined (for backward compatibility)
KEYWORDS = {**CORE_KEYWORDS, **MATH_KEYWORDS}

OPERATORS = {
    # Arithmetic (ASCII)
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "*": TokenType.MULTIPLY,
    "/": TokenType.DIVIDE,
    "%": TokenType.MODULO,
    "**": TokenType.POWER,
    "//": TokenType.FLOOR_DIVIDE,
    
    # Arithmetic (Unicode)
    "⊙": TokenType.DOT_PRODUCT,
    "⊗": TokenType.TENSOR_PRODUCT,
    "⊕": TokenType.ELEMENT_WISE_ADD,
    "⊖": TokenType.ELEMENT_WISE_SUB,
    "⋅": TokenType.SCALAR_PRODUCT,
    "×": TokenType.CROSS_PRODUCT,
    "÷": TokenType.DIVISION_SYMBOL,
    
    # Assignment
    "=": TokenType.ASSIGN,
    "+=": TokenType.PLUS_ASSIGN,
    "-=": TokenType.MINUS_ASSIGN,
    "*=": TokenType.MULTIPLY_ASSIGN,
    "/=": TokenType.DIVIDE_ASSIGN,
    "%=": TokenType.MODULO_ASSIGN,
    "**=": TokenType.POWER_ASSIGN,
    
    # Comparison (ASCII)
    "==": TokenType.EQUAL,
    "!=": TokenType.NOT_EQUAL,
    "<": TokenType.LESS_THAN,
    ">": TokenType.GREATER_THAN,
    "<=": TokenType.LESS_EQUAL,
    ">=": TokenType.GREATER_EQUAL,
    
    # Comparison (Unicode)
    "≡": TokenType.EQUIVALENT,
    "≠": TokenType.NOT_EQUIVALENT,
    "≤": TokenType.LESS_EQUAL_MATH,
    "≥": TokenType.GREATER_EQUAL_MATH,
    "≈": TokenType.APPROXIMATELY,
    "∈": TokenType.ELEMENT_OF,
    "∉": TokenType.NOT_ELEMENT_OF,
    
    # Logical (ASCII)
    "&&": TokenType.LOGICAL_AND,
    "||": TokenType.LOGICAL_OR,
    "!": TokenType.LOGICAL_NOT,
    
    # Logical (Unicode)
    "∧": TokenType.AND_SYMBOL,
    "∨": TokenType.OR_SYMBOL,
    "¬": TokenType.NOT_SYMBOL,
    # Note: ⊕ used for XOR, conflicts with ELEMENT_WISE_ADD - context dependent
    "↔": TokenType.BICONDITIONAL,
    
    # Mathematical (Unicode)
    "∇": TokenType.NABLA,
    "∂": TokenType.PARTIAL,
    "∑": TokenType.SUMMATION,
    "∏": TokenType.PRODUCT_OP,
    "∫": TokenType.INTEGRAL_OP,
    "√": TokenType.SQUARE_ROOT,
    "∛": TokenType.CUBE_ROOT,
    "∞": TokenType.INFINITY,
    
    # Common fraction literals
    "½": TokenType.FRACTION,
    "⅓": TokenType.FRACTION, 
    "⅔": TokenType.FRACTION,
    "¼": TokenType.FRACTION,
    "¾": TokenType.FRACTION,
    "⅕": TokenType.FRACTION,
    "⅖": TokenType.FRACTION,
    "⅗": TokenType.FRACTION,
    "⅘": TokenType.FRACTION,
    "⅙": TokenType.FRACTION,
    "⅚": TokenType.FRACTION,
    "⅛": TokenType.FRACTION,
    "⅜": TokenType.FRACTION,
    "⅝": TokenType.FRACTION,
    "⅞": TokenType.FRACTION,
    
    # Bitwise
    "&": TokenType.BIT_AND,
    "|": TokenType.BIT_OR,
    "^": TokenType.BIT_XOR,
    "~": TokenType.BIT_NOT,
    "<<": TokenType.LEFT_SHIFT,
    ">>": TokenType.RIGHT_SHIFT,
    
    # Punctuation
    "(": TokenType.LEFT_PAREN,
    ")": TokenType.RIGHT_PAREN,
    "[": TokenType.LEFT_BRACKET,
    "]": TokenType.RIGHT_BRACKET,
    "{": TokenType.LEFT_BRACE,
    "}": TokenType.RIGHT_BRACE,
    ";": TokenType.SEMICOLON,
    ",": TokenType.COMMA,
    ".": TokenType.DOT,
    ":": TokenType.COLON,
    "::": TokenType.DOUBLE_COLON,
    "->": TokenType.ARROW,
    "=>": TokenType.FAT_ARROW,
    "?": TokenType.QUESTION,
    "@": TokenType.AT,
    "#": TokenType.HASH,
    "$": TokenType.DOLLAR,
    "..": TokenType.RANGE_EXCLUSIVE,
    "..=": TokenType.RANGE_INCLUSIVE,
}

# Unicode mathematical symbols that can be used in identifiers
MATHEMATICAL_IDENTIFIER_CHARS = {
    # Greek letters (commonly used in mathematics)
    'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
    'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ', 'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω',
    
    # Mathematical subscripts and superscripts (for variable names like θ₁, x²)
    '₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉',
    '⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹',
    
    # Other common mathematical symbols used in identifiers
    '′', '″', '‴',  # Prime symbols (x', x'', x''')
}

# Fraction symbol to value mapping
FRACTION_VALUES = {
    '½': 0.5,
    '⅓': 1/3, 
    '⅔': 2/3,
    '¼': 0.25,
    '¾': 0.75,
    '⅕': 0.2,
    '⅖': 0.4,
    '⅗': 0.6,
    '⅘': 0.8,
    '⅙': 1/6,
    '⅚': 5/6,
    '⅛': 0.125,
    '⅜': 0.375,
    '⅝': 0.625,
    '⅞': 0.875,
}
