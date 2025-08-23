"""
NeuralScript Intermediate Representation (NS-IR) Package

Implements a Static Single Assignment (SSA) form intermediate representation
specifically designed for numerical computing, tensor operations, and automatic
differentiation. The IR is optimized for vectorization and GPU code generation.

Key Features:
- SSA form for optimization
- First-class tensor operations
- Automatic differentiation nodes
- GPU kernel representations
- Mathematical function primitives
- Parallel constructs

Author: xwest
"""

from .ir_nodes import *
from .ir_generator import IRGenerator

__all__ = [
    # Core IR components
    "IRGenerator",
    
    # IR nodes (from ir_nodes module)
    # All exported via from .ir_nodes import *
]
