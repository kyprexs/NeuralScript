"""
Dimensional Analysis Engine for NeuralScript

Provides compile-time dimensional analysis and unit checking to ensure
physical unit consistency and automatic unit conversions.

Author: xwest
"""

from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import math
import re

from ..parser.ast_nodes import *
from ..analyzer.semantic_analyzer import AnalysisResult, SymbolType
from ..analyzer.errors import SemanticError


class BaseUnit(Enum):
    """SI base units."""
    METER = "m"          # Length
    KILOGRAM = "kg"      # Mass  
    SECOND = "s"         # Time
    AMPERE = "A"         # Electric current
    KELVIN = "K"         # Temperature
    MOLE = "mol"         # Amount of substance
    CANDELA = "cd"       # Luminous intensity
    
    # Additional common base units
    RADIAN = "rad"       # Angle (dimensionless but tracked)
    STERADIAN = "sr"     # Solid angle


@dataclass(frozen=True)
class Dimension:
    """
    Represents a physical dimension as powers of base units.
    
    For example:
    - Velocity: m^1 * s^-1
    - Force: kg^1 * m^1 * s^-2
    - Energy: kg^1 * m^2 * s^-2
    """
    meter: Fraction = Fraction(0)      # Length
    kilogram: Fraction = Fraction(0)   # Mass
    second: Fraction = Fraction(0)     # Time
    ampere: Fraction = Fraction(0)     # Current
    kelvin: Fraction = Fraction(0)     # Temperature
    mole: Fraction = Fraction(0)       # Amount
    candela: Fraction = Fraction(0)    # Luminosity
    radian: Fraction = Fraction(0)     # Angle
    
    def __mul__(self, other: 'Dimension') -> 'Dimension':
        """Multiply dimensions (add exponents)."""
        return Dimension(
            meter=self.meter + other.meter,
            kilogram=self.kilogram + other.kilogram,
            second=self.second + other.second,
            ampere=self.ampere + other.ampere,
            kelvin=self.kelvin + other.kelvin,
            mole=self.mole + other.mole,
            candela=self.candela + other.candela,
            radian=self.radian + other.radian
        )
    
    def __truediv__(self, other: 'Dimension') -> 'Dimension':
        """Divide dimensions (subtract exponents)."""
        return Dimension(
            meter=self.meter - other.meter,
            kilogram=self.kilogram - other.kilogram,
            second=self.second - other.second,
            ampere=self.ampere - other.ampere,
            kelvin=self.kelvin - other.kelvin,
            mole=self.mole - other.mole,
            candela=self.candela - other.candela,
            radian=self.radian - other.radian
        )
    
    def __pow__(self, exponent: Union[int, float, Fraction]) -> 'Dimension':
        """Raise dimension to a power."""
        exp = Fraction(exponent)
        return Dimension(
            meter=self.meter * exp,
            kilogram=self.kilogram * exp,
            second=self.second * exp,
            ampere=self.ampere * exp,
            kelvin=self.kelvin * exp,
            mole=self.mole * exp,
            candela=self.candela * exp,
            radian=self.radian * exp
        )
    
    def is_dimensionless(self) -> bool:
        """Check if the dimension is dimensionless."""
        return all(power == 0 for power in [
            self.meter, self.kilogram, self.second, self.ampere,
            self.kelvin, self.mole, self.candela, self.radian
        ])
    
    def is_compatible(self, other: 'Dimension') -> bool:
        """Check if two dimensions are the same."""
        return self == other
    
    def __str__(self) -> str:
        """String representation of dimension."""
        if self.is_dimensionless():
            return "1"
        
        parts = []
        units = [
            (self.meter, "m"),
            (self.kilogram, "kg"), 
            (self.second, "s"),
            (self.ampere, "A"),
            (self.kelvin, "K"),
            (self.mole, "mol"),
            (self.candela, "cd"),
            (self.radian, "rad")
        ]
        
        for power, symbol in units:
            if power != 0:
                if power == 1:
                    parts.append(symbol)
                else:
                    parts.append(f"{symbol}^{power}")
        
        return " * ".join(parts) if parts else "1"


@dataclass
class Unit:
    """
    Represents a physical unit with name, symbol, dimension, and conversion factor.
    """
    name: str
    symbol: str
    dimension: Dimension
    scale_factor: float = 1.0  # Conversion factor to base unit
    offset: float = 0.0        # Offset for non-linear scales (like Celsius)
    
    def convert_to(self, other: 'Unit', value: float) -> float:
        """Convert a value from this unit to another unit."""
        if not self.dimension.is_compatible(other.dimension):
            raise ValueError(f"Cannot convert {self.name} to {other.name}: incompatible dimensions")
        
        # Convert to base unit, then to target unit
        base_value = (value + self.offset) * self.scale_factor
        target_value = base_value / other.scale_factor - other.offset
        return target_value
    
    def __str__(self) -> str:
        return self.symbol


@dataclass
class UnitRegistry:
    """Registry of all available units and their relationships."""
    
    def __init__(self):
        self.units: Dict[str, Unit] = {}
        self.dimensions: Dict[str, Dimension] = {}
        self.prefixes: Dict[str, float] = {}
        
        self._init_base_units()
        self._init_derived_units()
        self._init_prefixes()
        self._init_common_dimensions()
    
    def _init_base_units(self):
        """Initialize SI base units."""
        base_units = [
            Unit("meter", "m", Dimension(meter=Fraction(1))),
            Unit("kilogram", "kg", Dimension(kilogram=Fraction(1))),
            Unit("second", "s", Dimension(second=Fraction(1))),
            Unit("ampere", "A", Dimension(ampere=Fraction(1))),
            Unit("kelvin", "K", Dimension(kelvin=Fraction(1))),
            Unit("mole", "mol", Dimension(mole=Fraction(1))),
            Unit("candela", "cd", Dimension(candela=Fraction(1))),
            Unit("radian", "rad", Dimension(radian=Fraction(1))),
        ]
        
        for unit in base_units:
            self.units[unit.name] = unit
            self.units[unit.symbol] = unit
    
    def _init_derived_units(self):
        """Initialize derived SI units and common units."""
        derived_units = [
            # Area
            Unit("square_meter", "m²", Dimension(meter=Fraction(2))),
            
            # Volume
            Unit("cubic_meter", "m³", Dimension(meter=Fraction(3))),
            Unit("liter", "L", Dimension(meter=Fraction(3)), scale_factor=0.001),
            
            # Velocity
            Unit("meter_per_second", "m/s", 
                 Dimension(meter=Fraction(1), second=Fraction(-1))),
            Unit("kilometer_per_hour", "km/h",
                 Dimension(meter=Fraction(1), second=Fraction(-1)), 
                 scale_factor=1000/3600),
            
            # Acceleration
            Unit("meter_per_second_squared", "m/s²",
                 Dimension(meter=Fraction(1), second=Fraction(-2))),
            
            # Force
            Unit("newton", "N", 
                 Dimension(kilogram=Fraction(1), meter=Fraction(1), second=Fraction(-2))),
            
            # Pressure
            Unit("pascal", "Pa",
                 Dimension(kilogram=Fraction(1), meter=Fraction(-1), second=Fraction(-2))),
            Unit("atmosphere", "atm",
                 Dimension(kilogram=Fraction(1), meter=Fraction(-1), second=Fraction(-2)),
                 scale_factor=101325),
            
            # Energy
            Unit("joule", "J",
                 Dimension(kilogram=Fraction(1), meter=Fraction(2), second=Fraction(-2))),
            Unit("calorie", "cal",
                 Dimension(kilogram=Fraction(1), meter=Fraction(2), second=Fraction(-2)),
                 scale_factor=4.184),
            Unit("electronvolt", "eV",
                 Dimension(kilogram=Fraction(1), meter=Fraction(2), second=Fraction(-2)),
                 scale_factor=1.602176634e-19),
            
            # Power
            Unit("watt", "W",
                 Dimension(kilogram=Fraction(1), meter=Fraction(2), second=Fraction(-3))),
            
            # Electric charge
            Unit("coulomb", "C",
                 Dimension(ampere=Fraction(1), second=Fraction(1))),
            
            # Voltage
            Unit("volt", "V",
                 Dimension(kilogram=Fraction(1), meter=Fraction(2), 
                          second=Fraction(-3), ampere=Fraction(-1))),
            
            # Frequency
            Unit("hertz", "Hz", Dimension(second=Fraction(-1))),
            
            # Temperature (additional scales)
            Unit("celsius", "°C", Dimension(kelvin=Fraction(1)), 
                 scale_factor=1.0, offset=273.15),
            Unit("fahrenheit", "°F", Dimension(kelvin=Fraction(1)),
                 scale_factor=5/9, offset=459.67*5/9),
            
            # Angle
            Unit("degree", "°", Dimension(radian=Fraction(1)), 
                 scale_factor=math.pi/180),
            
            # Dimensionless
            Unit("dimensionless", "1", Dimension()),
            Unit("percent", "%", Dimension(), scale_factor=0.01),
        ]
        
        for unit in derived_units:
            self.units[unit.name] = unit
            self.units[unit.symbol] = unit
    
    def _init_prefixes(self):
        """Initialize SI prefixes."""
        self.prefixes = {
            "yotta": 1e24, "Y": 1e24,
            "zetta": 1e21, "Z": 1e21,
            "exa": 1e18, "E": 1e18,
            "peta": 1e15, "P": 1e15,
            "tera": 1e12, "T": 1e12,
            "giga": 1e9, "G": 1e9,
            "mega": 1e6, "M": 1e6,
            "kilo": 1e3, "k": 1e3,
            "hecto": 1e2, "h": 1e2,
            "deca": 1e1, "da": 1e1,
            
            "deci": 1e-1, "d": 1e-1,
            "centi": 1e-2, "c": 1e-2,
            "milli": 1e-3, "m": 1e-3,
            "micro": 1e-6, "μ": 1e-6, "u": 1e-6,
            "nano": 1e-9, "n": 1e-9,
            "pico": 1e-12, "p": 1e-12,
            "femto": 1e-15, "f": 1e-15,
            "atto": 1e-18, "a": 1e-18,
            "zepto": 1e-21, "z": 1e-21,
            "yocto": 1e-24, "y": 1e-24,
        }
    
    def _init_common_dimensions(self):
        """Initialize common physical dimension names."""
        self.dimensions = {
            "length": Dimension(meter=Fraction(1)),
            "mass": Dimension(kilogram=Fraction(1)),
            "time": Dimension(second=Fraction(1)),
            "area": Dimension(meter=Fraction(2)),
            "volume": Dimension(meter=Fraction(3)),
            "velocity": Dimension(meter=Fraction(1), second=Fraction(-1)),
            "acceleration": Dimension(meter=Fraction(1), second=Fraction(-2)),
            "force": Dimension(kilogram=Fraction(1), meter=Fraction(1), second=Fraction(-2)),
            "energy": Dimension(kilogram=Fraction(1), meter=Fraction(2), second=Fraction(-2)),
            "power": Dimension(kilogram=Fraction(1), meter=Fraction(2), second=Fraction(-3)),
            "pressure": Dimension(kilogram=Fraction(1), meter=Fraction(-1), second=Fraction(-2)),
            "frequency": Dimension(second=Fraction(-1)),
            "dimensionless": Dimension(),
        }
    
    def get_unit(self, unit_str: str) -> Optional[Unit]:
        """Get a unit by name or symbol, including prefixed units."""
        # Direct lookup
        if unit_str in self.units:
            return self.units[unit_str]
        
        # Try to parse prefixed units
        for prefix, scale in self.prefixes.items():
            if unit_str.startswith(prefix):
                base_unit_str = unit_str[len(prefix):]
                if base_unit_str in self.units:
                    base_unit = self.units[base_unit_str]
                    return Unit(
                        name=f"{prefix}{base_unit.name}",
                        symbol=unit_str,
                        dimension=base_unit.dimension,
                        scale_factor=base_unit.scale_factor * scale,
                        offset=base_unit.offset
                    )
        
        return None
    
    def parse_compound_unit(self, unit_expr: str) -> Optional[Unit]:
        """Parse compound units like 'kg*m/s²' or 'm/s'."""
        # This is a simplified parser - a full implementation would handle
        # complex expressions with parentheses, etc.
        
        # Handle division
        if '/' in unit_expr:
            parts = unit_expr.split('/')
            if len(parts) == 2:
                numerator_unit = self.parse_unit_product(parts[0])
                denominator_unit = self.parse_unit_product(parts[1])
                
                if numerator_unit and denominator_unit:
                    return Unit(
                        name=unit_expr,
                        symbol=unit_expr,
                        dimension=numerator_unit.dimension / denominator_unit.dimension,
                        scale_factor=numerator_unit.scale_factor / denominator_unit.scale_factor
                    )
        
        # Handle multiplication only
        return self.parse_unit_product(unit_expr)
    
    def parse_unit_product(self, unit_expr: str) -> Optional[Unit]:
        """Parse unit products like 'kg*m' or 'kg·m'."""
        # Split by multiplication symbols
        parts = re.split(r'[*·×]', unit_expr)
        
        if len(parts) == 1:
            # Single unit (possibly with exponent)
            return self.parse_unit_with_power(parts[0].strip())
        
        # Multiple units
        result_dimension = Dimension()
        result_scale = 1.0
        name_parts = []
        
        for part in parts:
            unit = self.parse_unit_with_power(part.strip())
            if not unit:
                return None
            
            result_dimension = result_dimension * unit.dimension
            result_scale *= unit.scale_factor
            name_parts.append(unit.symbol)
        
        return Unit(
            name=" * ".join(name_parts),
            symbol=" * ".join(name_parts),
            dimension=result_dimension,
            scale_factor=result_scale
        )
    
    def parse_unit_with_power(self, unit_str: str) -> Optional[Unit]:
        """Parse a unit with possible exponent like 'm²' or 's^-1'."""
        # Handle superscript exponents
        superscript_map = {
            '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5',
            '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9', '⁻': '-'
        }
        
        # Check for superscript
        for sup, normal in superscript_map.items():
            if sup in unit_str:
                base_unit_str = unit_str
                exponent_str = ""
                
                # Extract exponent
                for char in unit_str:
                    if char in superscript_map:
                        exponent_str += superscript_map[char]
                        base_unit_str = base_unit_str.replace(char, "")
                
                if exponent_str:
                    base_unit = self.get_unit(base_unit_str)
                    if base_unit:
                        exponent = int(exponent_str) if exponent_str else 1
                        return Unit(
                            name=f"{base_unit.name}^{exponent}",
                            symbol=unit_str,
                            dimension=base_unit.dimension ** exponent,
                            scale_factor=base_unit.scale_factor ** exponent
                        )
        
        # Handle caret notation like 's^-1'
        if '^' in unit_str:
            parts = unit_str.split('^')
            if len(parts) == 2:
                base_unit = self.get_unit(parts[0])
                if base_unit:
                    try:
                        exponent = int(parts[1])
                        return Unit(
                            name=f"{base_unit.name}^{exponent}",
                            symbol=unit_str,
                            dimension=base_unit.dimension ** exponent,
                            scale_factor=base_unit.scale_factor ** exponent
                        )
                    except ValueError:
                        pass
        
        # No exponent, just return the unit
        return self.get_unit(unit_str)


class DimensionalAnalyzer:
    """
    Dimensional analysis engine for NeuralScript.
    
    Performs compile-time dimensional analysis to ensure unit consistency
    and provides automatic unit conversions where appropriate.
    """
    
    def __init__(self):
        self.registry = UnitRegistry()
        self.variable_units: Dict[str, Unit] = {}
        self.errors: List[SemanticError] = []
        
    def analyze_program(self, ast: Program) -> List[SemanticError]:
        """Analyze an entire program for dimensional consistency."""
        self.errors = []
        
        for item in ast.items:
            if isinstance(item, FunctionDef):
                self._analyze_function(item)
        
        return self.errors
    
    def _analyze_function(self, func: FunctionDef):
        """Analyze a function for dimensional consistency."""
        # Create new scope for this function
        old_variables = self.variable_units.copy()
        
        try:
            # Analyze parameters
            for param in func.params:
                if param.type_annotation:
                    unit = self._extract_unit_from_type(param.type_annotation)
                    if unit:
                        self.variable_units[param.name] = unit
            
            # Analyze function body
            self._analyze_statement(func.body)
            
        finally:
            # Restore previous scope
            self.variable_units = old_variables
    
    def _analyze_statement(self, stmt: Statement):
        """Analyze a statement for dimensional consistency."""
        if isinstance(stmt, BlockStatement):
            for s in stmt.statements:
                self._analyze_statement(s)
        
        elif isinstance(stmt, VariableDecl):
            if stmt.initializer:
                init_unit = self._analyze_expression(stmt.initializer)
                
                # Check type annotation compatibility
                if stmt.type_annotation:
                    declared_unit = self._extract_unit_from_type(stmt.type_annotation)
                    if declared_unit and init_unit:
                        if not declared_unit.dimension.is_compatible(init_unit.dimension):
                            self.errors.append(SemanticError(
                                f"Unit mismatch: cannot assign {init_unit} to {declared_unit}",
                                stmt.span.start,
                                stmt,
                                code="U001"
                            ))
                        else:
                            # Store the declared unit
                            self.variable_units[stmt.name] = declared_unit
                    elif declared_unit:
                        self.variable_units[stmt.name] = declared_unit
                elif init_unit:
                    self.variable_units[stmt.name] = init_unit
        
        elif isinstance(stmt, ReturnStatement) and stmt.value:
            self._analyze_expression(stmt.value)
    
    def _analyze_expression(self, expr: Expression) -> Optional[Unit]:
        """Analyze an expression and return its unit."""
        if isinstance(expr, UnitLiteral):
            # Parse the unit string
            return self.registry.parse_compound_unit(expr.unit)
        
        elif isinstance(expr, Literal):
            # Dimensionless literal
            return self.registry.units.get("dimensionless")
        
        elif isinstance(expr, Identifier):
            return self.variable_units.get(expr.name)
        
        elif isinstance(expr, BinaryOp):
            return self._analyze_binary_op(expr)
        
        elif isinstance(expr, UnaryOp):
            return self._analyze_unary_op(expr)
        
        elif isinstance(expr, FunctionCall):
            return self._analyze_function_call(expr)
        
        return None
    
    def _analyze_binary_op(self, binary_op: BinaryOp) -> Optional[Unit]:
        """Analyze binary operations for dimensional consistency."""
        left_unit = self._analyze_expression(binary_op.left)
        right_unit = self._analyze_expression(binary_op.right)
        
        if not left_unit or not right_unit:
            return None
        
        operator = binary_op.operator
        
        if operator in ['+', '-']:
            # Addition/subtraction: operands must have same dimension
            if not left_unit.dimension.is_compatible(right_unit.dimension):
                self.errors.append(SemanticError(
                    f"Unit mismatch in {operator}: {left_unit} {operator} {right_unit}",
                    binary_op.span.start,
                    binary_op,
                    code="U002"
                ))
                return None
            
            # Result has the same dimension as operands
            return left_unit
        
        elif operator == '*':
            # Multiplication: dimensions multiply
            result_dimension = left_unit.dimension * right_unit.dimension
            return Unit(
                name=f"({left_unit.symbol} * {right_unit.symbol})",
                symbol=f"({left_unit.symbol} * {right_unit.symbol})",
                dimension=result_dimension,
                scale_factor=left_unit.scale_factor * right_unit.scale_factor
            )
        
        elif operator == '/':
            # Division: dimensions divide
            result_dimension = left_unit.dimension / right_unit.dimension
            return Unit(
                name=f"({left_unit.symbol} / {right_unit.symbol})",
                symbol=f"({left_unit.symbol} / {right_unit.symbol})",
                dimension=result_dimension,
                scale_factor=left_unit.scale_factor / right_unit.scale_factor
            )
        
        elif operator == '**' or operator == '^':
            # Exponentiation: dimension raised to power
            # Right operand must be dimensionless
            if not right_unit.dimension.is_dimensionless():
                self.errors.append(SemanticError(
                    f"Exponent must be dimensionless, got {right_unit}",
                    binary_op.span.start,
                    binary_op,
                    code="U003"
                ))
                return None
            
            # For now, assume integer powers (would need to evaluate right side)
            # This is a simplification
            return left_unit
        
        elif operator in ['==', '!=', '<', '<=', '>', '>=']:
            # Comparison: operands must have same dimension, result is dimensionless
            if not left_unit.dimension.is_compatible(right_unit.dimension):
                self.errors.append(SemanticError(
                    f"Unit mismatch in comparison: {left_unit} {operator} {right_unit}",
                    binary_op.span.start,
                    binary_op,
                    code="U004"
                ))
                return None
            
            return self.registry.units.get("dimensionless")
        
        return None
    
    def _analyze_unary_op(self, unary_op: UnaryOp) -> Optional[Unit]:
        """Analyze unary operations."""
        operand_unit = self._analyze_expression(unary_op.operand)
        
        if not operand_unit:
            return None
        
        operator = unary_op.operator
        
        if operator in ['+', '-']:
            # Unary plus/minus: preserve unit
            return operand_unit
        
        elif operator == '!':
            # Logical not: operand should be dimensionless, result is dimensionless
            if not operand_unit.dimension.is_dimensionless():
                self.errors.append(SemanticError(
                    f"Logical not requires dimensionless operand, got {operand_unit}",
                    unary_op.span.start,
                    unary_op,
                    code="U005"
                ))
            
            return self.registry.units.get("dimensionless")
        
        return operand_unit
    
    def _analyze_function_call(self, call: FunctionCall) -> Optional[Unit]:
        """Analyze function calls for dimensional consistency."""
        if isinstance(call.function, Identifier):
            func_name = call.function.name
            
            # Built-in mathematical functions
            if func_name in ["sin", "cos", "tan"]:
                # Trigonometric functions: argument should be angle, result is dimensionless
                if call.args:
                    arg_unit = self._analyze_expression(call.args[0])
                    if arg_unit and not arg_unit.dimension.is_compatible(
                        self.registry.dimensions["dimensionless"]
                    ) and not arg_unit.dimension.is_compatible(
                        Dimension(radian=Fraction(1))
                    ):
                        self.errors.append(SemanticError(
                            f"{func_name} expects angle or dimensionless argument, got {arg_unit}",
                            call.span.start,
                            call,
                            code="U006"
                        ))
                
                return self.registry.units.get("dimensionless")
            
            elif func_name in ["exp", "log", "log10"]:
                # Exponential/logarithmic: argument must be dimensionless
                if call.args:
                    arg_unit = self._analyze_expression(call.args[0])
                    if arg_unit and not arg_unit.dimension.is_dimensionless():
                        self.errors.append(SemanticError(
                            f"{func_name} expects dimensionless argument, got {arg_unit}",
                            call.span.start,
                            call,
                            code="U007"
                        ))
                
                return self.registry.units.get("dimensionless")
            
            elif func_name == "sqrt":
                # Square root: result dimension is operand dimension / 2
                if call.args:
                    arg_unit = self._analyze_expression(call.args[0])
                    if arg_unit:
                        result_dimension = arg_unit.dimension ** Fraction(1, 2)
                        return Unit(
                            name=f"sqrt({arg_unit.symbol})",
                            symbol=f"√({arg_unit.symbol})",
                            dimension=result_dimension,
                            scale_factor=math.sqrt(arg_unit.scale_factor)
                        )
            
            elif func_name == "abs":
                # Absolute value: preserve unit
                if call.args:
                    return self._analyze_expression(call.args[0])
        
        return None
    
    def _extract_unit_from_type(self, type_ref: TypeRef) -> Optional[Unit]:
        """Extract unit information from a type annotation."""
        if hasattr(type_ref, 'name'):
            # Look for unit information in type name
            # This is a simplified approach - a full implementation might have
            # special unit type syntax
            
            # Check if the type name contains unit information
            type_name = type_ref.name
            
            # Pattern like "meters" or "m_per_s"
            if type_name.endswith('_unit'):
                unit_part = type_name[:-5]  # Remove '_unit' suffix
                return self.registry.get_unit(unit_part)
            
            # Direct unit name
            return self.registry.get_unit(type_name)
        
        return None
    
    def suggest_conversion(self, from_unit: Unit, to_dimension: Dimension) -> Optional[Unit]:
        """Suggest a unit conversion to match a target dimension."""
        if from_unit.dimension.is_compatible(to_dimension):
            return from_unit
        
        # Look for a unit in the registry with the target dimension
        for unit in self.registry.units.values():
            if unit.dimension.is_compatible(to_dimension):
                return unit
        
        return None
    
    def check_assignment_compatibility(self, target_unit: Unit, source_unit: Unit) -> bool:
        """Check if a source unit can be assigned to a target unit."""
        return target_unit.dimension.is_compatible(source_unit.dimension)
    
    def generate_conversion_factor(self, from_unit: Unit, to_unit: Unit) -> Optional[float]:
        """Generate the conversion factor from one unit to another."""
        if not from_unit.dimension.is_compatible(to_unit.dimension):
            return None
        
        return from_unit.scale_factor / to_unit.scale_factor


def create_unit_type(unit_str: str, base_type: str = "f64") -> SymbolType:
    """Create a SymbolType with unit information."""
    return SymbolType(
        name=f"{base_type}_{unit_str}",
        kind="unit",
        parameters=[SymbolType(base_type, "primitive")],
        constraints=[f"unit={unit_str}"]
    )


# Integration with the compiler pipeline

def add_dimensional_analysis_pass(analysis_result: AnalysisResult) -> AnalysisResult:
    """Add dimensional analysis as an additional semantic analysis pass."""
    analyzer = DimensionalAnalyzer()
    unit_errors = analyzer.analyze_program(analysis_result.ast)
    
    # Add unit errors to existing errors
    analysis_result.errors.extend(unit_errors)
    
    return analysis_result
