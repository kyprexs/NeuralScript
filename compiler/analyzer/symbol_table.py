"""
Symbol table and scope management for NeuralScript semantic analysis.

Implements hierarchical symbol tables with support for:
- Lexical scoping
- Symbol visibility rules
- Forward declarations
- Generic type parameters
- Import resolution

Author: xwest
"""

from typing import Dict, List, Optional, Set, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid

from ..lexer.tokens import SourceLocation
from ..parser.ast_nodes import ASTNode, TypeRef
from .errors import SemanticError, create_undefined_symbol_error


class SymbolKind(Enum):
    """Types of symbols in the symbol table."""
    VARIABLE = "variable"
    FUNCTION = "function"
    TYPE = "type"
    TRAIT = "trait" 
    MODULE = "module"
    GENERIC_PARAM = "generic_param"
    FIELD = "field"
    METHOD = "method"
    CONSTANT = "constant"


class Visibility(Enum):
    """Symbol visibility levels."""
    PRIVATE = "private"
    PUBLIC = "public"
    PROTECTED = "protected"  # For potential inheritance features


@dataclass
class SymbolType:
    """Represents a type in the type system."""
    name: str
    kind: str  # "primitive", "composite", "generic", "tensor", etc.
    parameters: List['SymbolType'] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        if self.parameters:
            param_str = ", ".join(str(p) for p in self.parameters)
            return f"{self.name}<{param_str}>"
        return self.name
    
    def is_generic(self) -> bool:
        """Check if this type contains generic parameters."""
        if self.kind == "generic":
            return True
        return any(param.is_generic() for param in self.parameters)
    
    def substitute_generics(self, substitutions: Dict[str, 'SymbolType']) -> 'SymbolType':
        """Substitute generic type parameters with concrete types."""
        if self.name in substitutions:
            return substitutions[self.name]
        
        new_params = []
        for param in self.parameters:
            new_params.append(param.substitute_generics(substitutions))
        
        return SymbolType(
            name=self.name,
            kind=self.kind,
            parameters=new_params,
            constraints=self.constraints.copy()
        )
    
    def __hash__(self) -> int:
        """Hash based on type structure for use in dictionaries."""
        return hash((self.name, self.kind, tuple(self.parameters), tuple(self.constraints)))
    
    def __eq__(self, other) -> bool:
        """Equality based on type structure."""
        if not isinstance(other, SymbolType):
            return False
        return (self.name == other.name and 
                self.kind == other.kind and 
                self.parameters == other.parameters and 
                self.constraints == other.constraints)


@dataclass
class Symbol:
    """Represents a symbol in the symbol table."""
    name: str
    kind: SymbolKind
    symbol_type: Optional[SymbolType]
    location: SourceLocation
    visibility: Visibility = Visibility.PRIVATE
    is_mutable: bool = False
    is_const: bool = False
    ast_node: Optional[ASTNode] = None
    
    # Additional attributes for different symbol kinds
    generic_params: List[str] = field(default_factory=list)
    value: Optional[Any] = None  # For constants
    module_path: Optional[str] = None  # For imports
    
    # Memory safety attributes
    is_moved: bool = False
    borrowers: Set[str] = field(default_factory=set)
    owner: Optional[str] = None
    
    def __str__(self) -> str:
        type_str = f": {self.symbol_type}" if self.symbol_type else ""
        return f"{self.name}{type_str}"
    
    def can_access_from(self, scope: 'Scope') -> bool:
        """Check if this symbol can be accessed from the given scope."""
        if self.visibility == Visibility.PUBLIC:
            return True
        
        if self.visibility == Visibility.PRIVATE:
            # Private symbols can only be accessed from the same module/scope
            return scope.is_same_module_as(self)
        
        return False


class ScopeKind(Enum):
    """Types of scopes."""
    MODULE = "module"
    FUNCTION = "function"
    BLOCK = "block"
    STRUCT = "struct"
    TRAIT = "trait"
    IMPL = "impl"
    LOOP = "loop"


@dataclass
class Scope:
    """Represents a lexical scope."""
    kind: ScopeKind
    name: str
    symbols: Dict[str, Symbol] = field(default_factory=dict)
    parent: Optional['Scope'] = None
    children: List['Scope'] = field(default_factory=list)
    generic_params: Set[str] = field(default_factory=set)
    
    # Scope-specific attributes
    return_type: Optional[SymbolType] = None  # For function scopes
    loop_labels: Set[str] = field(default_factory=set)  # For loop scopes
    module_path: Optional[str] = None  # For module scopes
    
    def __post_init__(self):
        """Initialize scope after creation."""
        if self.parent:
            self.parent.children.append(self)
    
    def define_symbol(self, symbol: Symbol) -> None:
        """Define a symbol in this scope."""
        if symbol.name in self.symbols:
            existing = self.symbols[symbol.name]
            # Allow function overloading based on arity/types
            if symbol.kind == SymbolKind.FUNCTION and existing.kind == SymbolKind.FUNCTION:
                # TODO: Implement proper function overloading logic
                pass
            else:
                raise SemanticError(
                    f"Symbol '{symbol.name}' is already defined in this scope",
                    symbol.location,
                    code="S011"
                )
        
        self.symbols[symbol.name] = symbol
    
    def lookup_symbol(self, name: str) -> Optional[Symbol]:
        """Look up a symbol in this scope and parent scopes."""
        # Check current scope
        if name in self.symbols:
            return self.symbols[name]
        
        # Check parent scopes
        if self.parent:
            return self.parent.lookup_symbol(name)
        
        return None
    
    def lookup_symbol_local(self, name: str) -> Optional[Symbol]:
        """Look up a symbol only in this scope (no parent traversal)."""
        return self.symbols.get(name)
    
    def get_all_symbols(self) -> Dict[str, Symbol]:
        """Get all symbols visible in this scope."""
        result = {}
        
        # Add parent symbols first
        if self.parent:
            result.update(self.parent.get_all_symbols())
        
        # Add local symbols (can override parent symbols)
        result.update(self.symbols)
        
        return result
    
    def get_similar_names(self, name: str, max_distance: int = 2) -> List[str]:
        """Get symbol names similar to the given name (for error suggestions)."""
        def levenshtein_distance(s1: str, s2: str) -> int:
            """Calculate edit distance between two strings."""
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
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
        
        similar_names = []
        all_symbols = self.get_all_symbols()
        
        for symbol_name in all_symbols.keys():
            distance = levenshtein_distance(name.lower(), symbol_name.lower())
            if distance <= max_distance:
                similar_names.append((symbol_name, distance))
        
        # Sort by distance and return names only
        similar_names.sort(key=lambda x: x[1])
        return [name for name, _ in similar_names[:5]]
    
    def is_same_module_as(self, symbol: Symbol) -> bool:
        """Check if this scope is in the same module as the symbol."""
        # Find the module scope
        current = self
        while current and current.kind != ScopeKind.MODULE:
            current = current.parent
        
        if current and symbol.module_path:
            return current.module_path == symbol.module_path
        
        return True  # Assume same module if no module info available
    
    def add_generic_param(self, name: str):
        """Add a generic type parameter to this scope."""
        self.generic_params.add(name)
    
    def is_generic_param(self, name: str) -> bool:
        """Check if a name is a generic parameter in this scope hierarchy."""
        if name in self.generic_params:
            return True
        
        if self.parent:
            return self.parent.is_generic_param(name)
        
        return False
    
    def __str__(self) -> str:
        symbol_count = len(self.symbols)
        return f"Scope({self.kind.value}, {self.name}, {symbol_count} symbols)"


class SymbolTable:
    """
    Manages hierarchical symbol tables and scopes.
    
    Provides symbol resolution, scope management, and visibility checking.
    """
    
    def __init__(self):
        """Initialize the symbol table with a global scope."""
        self.global_scope = Scope(ScopeKind.MODULE, "global")
        self.current_scope = self.global_scope
        self.scopes: List[Scope] = [self.global_scope]
        
        # Built-in types and symbols
        self._initialize_builtins()
    
    def _initialize_builtins(self):
        """Initialize built-in types and functions."""
        builtin_types = [
            # Primitive types
            ("i8", "primitive"), ("i16", "primitive"), ("i32", "primitive"), ("i64", "primitive"),
            ("u8", "primitive"), ("u16", "primitive"), ("u32", "primitive"), ("u64", "primitive"),
            ("f16", "primitive"), ("f32", "primitive"), ("f64", "primitive"),
            ("c32", "primitive"), ("c64", "primitive"),
            ("bool", "primitive"), ("char", "primitive"), ("str", "primitive"),
            
            # Container types
            ("Array", "generic"), ("Vec", "generic"), ("Matrix", "generic"), ("Tensor", "generic"),
            
            # Unit types
            ("Meter", "unit"), ("Kilogram", "unit"), ("Second", "unit"),
        ]
        
        dummy_location = SourceLocation("builtin", 0, 0, 0)
        
        for type_name, type_kind in builtin_types:
            symbol_type = SymbolType(type_name, type_kind)
            symbol = Symbol(
                name=type_name,
                kind=SymbolKind.TYPE,
                symbol_type=symbol_type,
                location=dummy_location,
                visibility=Visibility.PUBLIC
            )
            self.global_scope.define_symbol(symbol)
        
        # Built-in functions
        builtin_functions = [
            "print", "println", "len", "sum", "max", "min",
            "sin", "cos", "tan", "exp", "log", "sqrt",
            "tensor", "zeros", "ones", "eye", "rand"
        ]
        
        for func_name in builtin_functions:
            # Create generic function type (will be refined later)
            func_type = SymbolType("function", "function")
            symbol = Symbol(
                name=func_name,
                kind=SymbolKind.FUNCTION,
                symbol_type=func_type,
                location=dummy_location,
                visibility=Visibility.PUBLIC
            )
            self.global_scope.define_symbol(symbol)
    
    def enter_scope(self, kind: ScopeKind, name: str) -> Scope:
        """Enter a new scope."""
        new_scope = Scope(kind, name, parent=self.current_scope)
        self.current_scope = new_scope
        self.scopes.append(new_scope)
        return new_scope
    
    def exit_scope(self) -> Optional[Scope]:
        """Exit the current scope and return to parent."""
        if self.current_scope.parent:
            old_scope = self.current_scope
            self.current_scope = self.current_scope.parent
            return old_scope
        return None
    
    def define_symbol(self, symbol: Symbol) -> None:
        """Define a symbol in the current scope."""
        self.current_scope.define_symbol(symbol)
    
    def lookup_symbol(self, name: str, location: SourceLocation) -> Symbol:
        """Look up a symbol and raise error if not found."""
        symbol = self.current_scope.lookup_symbol(name)
        
        if symbol is None:
            similar_names = self.current_scope.get_similar_names(name)
            raise create_undefined_symbol_error(name, location, similar_names=similar_names)
        
        # Check if symbol can be accessed
        if not symbol.can_access_from(self.current_scope):
            raise SemanticError(
                f"Symbol '{name}' is not accessible from current scope",
                location,
                code="S012"
            )
        
        return symbol
    
    def lookup_symbol_safe(self, name: str) -> Optional[Symbol]:
        """Look up a symbol without raising errors."""
        return self.current_scope.lookup_symbol(name)
    
    def lookup_type(self, name: str, location: SourceLocation) -> SymbolType:
        """Look up a type symbol and return its type."""
        symbol = self.lookup_symbol(name, location)
        
        if symbol.kind != SymbolKind.TYPE:
            raise SemanticError(
                f"'{name}' is not a type",
                location,
                code="S002"
            )
        
        return symbol.symbol_type
    
    def define_variable(self, name: str, var_type: SymbolType, location: SourceLocation,
                       is_mutable: bool = False, is_const: bool = False) -> Symbol:
        """Define a variable symbol."""
        symbol = Symbol(
            name=name,
            kind=SymbolKind.VARIABLE,
            symbol_type=var_type,
            location=location,
            is_mutable=is_mutable,
            is_const=is_const
        )
        self.define_symbol(symbol)
        return symbol
    
    def define_function(self, name: str, func_type: SymbolType, location: SourceLocation,
                       generic_params: Optional[List[str]] = None) -> Symbol:
        """Define a function symbol."""
        symbol = Symbol(
            name=name,
            kind=SymbolKind.FUNCTION,
            symbol_type=func_type,
            location=location,
            generic_params=generic_params or []
        )
        self.define_symbol(symbol)
        return symbol
    
    def define_type(self, name: str, type_def: SymbolType, location: SourceLocation,
                   generic_params: Optional[List[str]] = None) -> Symbol:
        """Define a type symbol."""
        symbol = Symbol(
            name=name,
            kind=SymbolKind.TYPE,
            symbol_type=type_def,
            location=location,
            generic_params=generic_params or []
        )
        self.define_symbol(symbol)
        return symbol
    
    def get_current_function_scope(self) -> Optional[Scope]:
        """Get the current function scope, if any."""
        current = self.current_scope
        while current:
            if current.kind == ScopeKind.FUNCTION:
                return current
            current = current.parent
        return None
    
    def get_current_loop_scope(self) -> Optional[Scope]:
        """Get the current loop scope, if any."""
        current = self.current_scope
        while current:
            if current.kind == ScopeKind.LOOP:
                return current
            current = current.parent
        return None
    
    def is_in_function(self) -> bool:
        """Check if currently inside a function."""
        return self.get_current_function_scope() is not None
    
    def is_in_loop(self) -> bool:
        """Check if currently inside a loop."""
        return self.get_current_loop_scope() is not None
    
    def add_generic_param(self, name: str):
        """Add a generic parameter to the current scope."""
        self.current_scope.add_generic_param(name)
    
    def resolve_type_from_ast(self, type_ref: TypeRef) -> SymbolType:
        """Resolve a type reference from the AST to a SymbolType."""
        # This is a simplified implementation
        # In a full implementation, this would handle generic types, 
        # tensor types with dimensions, etc.
        
        if hasattr(type_ref, 'name'):
            # Simple type reference
            return self.lookup_type(type_ref.name, type_ref.span.start)
        
        # For now, return a generic type
        return SymbolType("unknown", "primitive")
    
    def __str__(self) -> str:
        return f"SymbolTable(current: {self.current_scope})"
