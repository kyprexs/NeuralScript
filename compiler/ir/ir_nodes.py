"""
NeuralScript Intermediate Representation (NS-IR) Nodes

Defines all IR node types used in the NeuralScript intermediate representation.
The IR is in SSA form and designed for numerical computing optimization.

Author: xwest
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid


class IRNodeType(Enum):
    """Enumeration of IR node types."""
    
    # Basic nodes
    MODULE = "module"
    FUNCTION = "function"
    BASIC_BLOCK = "basic_block"
    VALUE = "value"
    CONSTANT = "constant"
    
    # Instructions
    BINARY_OP = "binary_op"
    UNARY_OP = "unary_op"
    CALL = "call"
    RETURN = "return"
    BRANCH = "branch"
    COND_BRANCH = "cond_branch"
    PHI = "phi"
    ALLOCA = "alloca"
    LOAD = "load"
    STORE = "store"
    
    # Tensor operations
    TENSOR_OP = "tensor_op"
    MATMUL = "matmul"
    CONV2D = "conv2d"
    POOLING = "pooling"
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    SLICE = "slice"
    CONCAT = "concat"
    BROADCAST = "broadcast"
    REDUCE = "reduce"
    
    # Mathematical operations
    MATH_OP = "math_op"
    TRIG_OP = "trig_op"
    EXP_OP = "exp_op"
    LOG_OP = "log_op"
    SQRT_OP = "sqrt_op"
    
    # Automatic differentiation
    GRADIENT = "gradient"
    JACOBIAN = "jacobian"
    HESSIAN = "hessian"
    BACKWARD = "backward"
    FORWARD = "forward"
    
    # Control flow
    LABEL = "label"
    GOTO = "goto"
    LOOP = "loop"
    PARALLEL = "parallel"
    
    # GPU operations
    KERNEL = "kernel"
    LAUNCH = "launch"
    SYNC = "sync"
    BARRIER = "barrier"


class IRDataType(Enum):
    """IR data types."""
    VOID = "void"
    I1 = "i1"      # boolean
    I8 = "i8"
    I16 = "i16"
    I32 = "i32"
    I64 = "i64"
    F16 = "f16"
    F32 = "f32"
    F64 = "f64"
    C32 = "c32"    # complex float
    C64 = "c64"    # complex double
    PTR = "ptr"    # pointer


@dataclass
class IRType:
    """Base class for IR types."""
    name: str
    size: int  # Size in bytes
    
    def __str__(self) -> str:
        return self.name


@dataclass
class IRPrimitiveType(IRType):
    """Primitive IR type."""
    data_type: IRDataType
    
    def __init__(self, data_type: IRDataType):
        size_map = {
            IRDataType.VOID: 0,
            IRDataType.I1: 1,
            IRDataType.I8: 1,
            IRDataType.I16: 2,
            IRDataType.I32: 4,
            IRDataType.I64: 8,
            IRDataType.F16: 2,
            IRDataType.F32: 4,
            IRDataType.F64: 8,
            IRDataType.C32: 8,
            IRDataType.C64: 16,
            IRDataType.PTR: 8,
        }
        super().__init__(data_type.value, size_map[data_type])
        self.data_type = data_type


@dataclass
class IRTensorType(IRType):
    """Tensor IR type with shape information."""
    element_type: IRPrimitiveType
    shape: List[int]
    
    def __init__(self, element_type: IRPrimitiveType, shape: List[int]):
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        size = total_elements * element_type.size
        name = f"tensor<{element_type.name}, {shape}>"
        super().__init__(name, size)
        self.element_type = element_type
        self.shape = shape
    
    @property
    def rank(self) -> int:
        """Number of dimensions."""
        return len(self.shape)
    
    @property
    def total_elements(self) -> int:
        """Total number of elements."""
        total = 1
        for dim in self.shape:
            total *= dim
        return total


@dataclass
class IRFunctionType(IRType):
    """Function IR type."""
    return_type: IRType
    param_types: List[IRType]
    
    def __init__(self, return_type: IRType, param_types: List[IRType]):
        name = f"({', '.join(t.name for t in param_types)}) -> {return_type.name}"
        super().__init__(name, 8)  # Function pointer size
        self.return_type = return_type
        self.param_types = param_types


class IRNode(ABC):
    """Base class for all IR nodes."""
    
    def __init__(self, node_type: IRNodeType, name: str = ""):
        self.node_type = node_type
        self.name = name or f"{node_type.value}_{uuid.uuid4().hex[:8]}"
        self.metadata: Dict[str, Any] = {}
        self.parent: Optional['IRNode'] = None
        self.uses: Set['IRValue'] = set()
        self.def_: Optional['IRValue'] = None
        # Generate unique ID for hashability
        self._id = uuid.uuid4()
    
    @abstractmethod
    def __str__(self) -> str:
        pass
    
    def set_metadata(self, key: str, value: Any):
        """Set metadata for this node."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata for this node."""
        return self.metadata.get(key, default)
    
    def __hash__(self) -> int:
        """Hash based on unique ID for use in dictionaries."""
        return hash(self._id)
    
    def __eq__(self, other) -> bool:
        """Equality based on unique ID."""
        if not isinstance(other, IRNode):
            return False
        return self._id == other._id


class IRValue:
    """Represents a value in SSA form."""
    
    def __init__(self, ir_type: IRType, name: str = ""):
        self.type = ir_type
        self.name = name or f"v_{uuid.uuid4().hex[:8]}"
        self.def_node: Optional[IRNode] = None
        self.users: Set[IRNode] = set()
    
    def add_user(self, user: IRNode):
        """Add a user of this value."""
        self.users.add(user)
    
    def remove_user(self, user: IRNode):
        """Remove a user of this value."""
        self.users.discard(user)
    
    def replace_all_uses_with(self, new_value: 'IRValue'):
        """Replace all uses of this value with another value."""
        for user in self.users.copy():
            # This would need to be implemented per instruction type
            pass
    
    def __str__(self) -> str:
        return f"%{self.name}: {self.type.name}"


@dataclass
class IRConstant(IRValue):
    """Constant value in IR."""
    value: Any
    
    def __init__(self, ir_type: IRType, value: Any):
        super().__init__(ir_type, f"const_{value}")
        self.value = value
    
    def __str__(self) -> str:
        return f"{self.type.name} {self.value}"


class IRBasicBlock(IRNode):
    """Basic block containing a sequence of instructions."""
    
    def __init__(self, name: str = ""):
        super().__init__(IRNodeType.BASIC_BLOCK, name)
        self.instructions: List['IRInstruction'] = []
        self.predecessors: Set['IRBasicBlock'] = set()
        self.successors: Set['IRBasicBlock'] = set()
        self.function: Optional['IRFunction'] = None
    
    def add_instruction(self, instruction: 'IRInstruction'):
        """Add an instruction to this basic block."""
        instruction.parent = self
        self.instructions.append(instruction)
    
    def insert_instruction(self, index: int, instruction: 'IRInstruction'):
        """Insert an instruction at a specific position."""
        instruction.parent = self
        self.instructions.insert(index, instruction)
    
    def remove_instruction(self, instruction: 'IRInstruction'):
        """Remove an instruction from this basic block."""
        if instruction in self.instructions:
            instruction.parent = None
            self.instructions.remove(instruction)
    
    def add_predecessor(self, pred: 'IRBasicBlock'):
        """Add a predecessor basic block."""
        self.predecessors.add(pred)
        pred.successors.add(self)
    
    def add_successor(self, succ: 'IRBasicBlock'):
        """Add a successor basic block."""
        self.successors.add(succ)
        succ.predecessors.add(self)
    
    def __str__(self) -> str:
        lines = [f"{self.name}:"]
        for instr in self.instructions:
            lines.append(f"  {instr}")
        return "\n".join(lines)


class IRFunction(IRNode):
    """Function in IR."""
    
    def __init__(self, name: str, func_type: IRFunctionType):
        super().__init__(IRNodeType.FUNCTION, name)
        self.type = func_type
        self.basic_blocks: List[IRBasicBlock] = []
        self.parameters: List[IRValue] = []
        self.entry_block: Optional[IRBasicBlock] = None
        self.module: Optional['IRModule'] = None
        
        # Create parameter values
        for i, param_type in enumerate(func_type.param_types):
            param = IRValue(param_type, f"arg{i}")
            self.parameters.append(param)
    
    def add_basic_block(self, block: IRBasicBlock):
        """Add a basic block to this function."""
        block.function = self
        self.basic_blocks.append(block)
        
        # Set entry block if this is the first block
        if self.entry_block is None:
            self.entry_block = block
    
    def remove_basic_block(self, block: IRBasicBlock):
        """Remove a basic block from this function."""
        if block in self.basic_blocks:
            block.function = None
            self.basic_blocks.remove(block)
            
            if self.entry_block == block:
                self.entry_block = self.basic_blocks[0] if self.basic_blocks else None
    
    def __str__(self) -> str:
        param_strs = [f"%{param.name}: {param.type.name}" for param in self.parameters]
        lines = [f"define {self.type.return_type.name} @{self.name}({', '.join(param_strs)}) {{"]
        
        for block in self.basic_blocks:
            lines.append(str(block))
        
        lines.append("}")
        return "\n".join(lines)


class IRModule(IRNode):
    """Top-level IR module."""
    
    def __init__(self, name: str):
        super().__init__(IRNodeType.MODULE, name)
        self.functions: Dict[str, IRFunction] = {}
        self.globals: Dict[str, IRValue] = {}
        self.types: Dict[str, IRType] = {}
    
    def add_function(self, function: IRFunction):
        """Add a function to this module."""
        function.module = self
        self.functions[function.name] = function
    
    def get_function(self, name: str) -> Optional[IRFunction]:
        """Get a function by name."""
        return self.functions.get(name)
    
    def add_global(self, name: str, value: IRValue):
        """Add a global value."""
        self.globals[name] = value
    
    def __str__(self) -> str:
        lines = [f"; Module: {self.name}"]
        
        # Globals
        for name, global_val in self.globals.items():
            lines.append(f"@{name} = global {global_val.type.name}")
        
        if self.globals:
            lines.append("")
        
        # Functions
        for func in self.functions.values():
            lines.append(str(func))
            lines.append("")
        
        return "\n".join(lines)


# ============================================================================
# Instructions
# ============================================================================

class IRInstruction(IRNode):
    """Base class for IR instructions."""
    
    def __init__(self, node_type: IRNodeType, operands: List[IRValue], 
                 result_type: Optional[IRType] = None):
        super().__init__(node_type)
        self.operands = operands
        self.result: Optional[IRValue] = None
        
        # Track uses
        for operand in operands:
            operand.add_user(self)
        
        # Create result value if needed
        if result_type:
            self.result = IRValue(result_type)
            self.result.def_node = self


class IRBinaryOp(IRInstruction):
    """Binary operation instruction."""
    operator: str
    left: IRValue
    right: IRValue
    
    def __init__(self, operator: str, left: IRValue, right: IRValue, result_type: IRType):
        super().__init__(IRNodeType.BINARY_OP, [left, right], result_type)
        self.operator = operator
        self.left = left
        self.right = right
    
    def __str__(self) -> str:
        return f"%{self.result.name} = {self.operator} {self.left} {self.right}"


class IRUnaryOp(IRInstruction):
    """Unary operation instruction."""
    operator: str
    operand: IRValue
    
    def __init__(self, operator: str, operand: IRValue, result_type: IRType):
        super().__init__(IRNodeType.UNARY_OP, [operand], result_type)
        self.operator = operator
        self.operand = operand
    
    def __str__(self) -> str:
        return f"%{self.result.name} = {self.operator} {self.operand}"


class IRCall(IRInstruction):
    """Function call instruction."""
    function: IRValue
    args: List[IRValue]
    
    def __init__(self, function: IRValue, args: List[IRValue], result_type: Optional[IRType] = None):
        operands = [function] + args
        super().__init__(IRNodeType.CALL, operands, result_type)
        self.function = function
        self.args = args
    
    def __str__(self) -> str:
        arg_strs = [str(arg) for arg in self.args]
        if self.result:
            return f"%{self.result.name} = call {self.function}({', '.join(arg_strs)})"
        else:
            return f"call {self.function}({', '.join(arg_strs)})"


class IRReturn(IRInstruction):
    """Return instruction."""
    value: Optional[IRValue]
    
    def __init__(self, value: Optional[IRValue] = None):
        operands = [value] if value else []
        super().__init__(IRNodeType.RETURN, operands)
        self.value = value
    
    def __str__(self) -> str:
        if self.value:
            return f"ret {self.value}"
        else:
            return "ret void"


class IRBranch(IRInstruction):
    """Unconditional branch instruction."""
    target: IRBasicBlock
    
    def __init__(self, target: IRBasicBlock):
        super().__init__(IRNodeType.BRANCH, [])
        self.target = target
    
    def __str__(self) -> str:
        return f"br label %{self.target.name}"


class IRCondBranch(IRInstruction):
    """Conditional branch instruction."""
    condition: IRValue
    true_target: IRBasicBlock
    false_target: IRBasicBlock
    
    def __init__(self, condition: IRValue, true_target: IRBasicBlock, false_target: IRBasicBlock):
        super().__init__(IRNodeType.COND_BRANCH, [condition])
        self.condition = condition
        self.true_target = true_target
        self.false_target = false_target
    
    def __str__(self) -> str:
        return f"br {self.condition}, label %{self.true_target.name}, label %{self.false_target.name}"


class IRPhi(IRInstruction):
    """Phi node for SSA form."""
    incoming: List[Tuple[IRValue, IRBasicBlock]]
    
    def __init__(self, result_type: IRType, incoming: List[Tuple[IRValue, IRBasicBlock]] = None):
        values = [val for val, _ in incoming] if incoming else []
        super().__init__(IRNodeType.PHI, values, result_type)
        self.incoming = incoming or []
    
    def add_incoming(self, value: IRValue, block: IRBasicBlock):
        """Add an incoming value from a block."""
        self.incoming.append((value, block))
        self.operands.append(value)
        value.add_user(self)
    
    def __str__(self) -> str:
        incoming_strs = [f"[{val}, %{block.name}]" for val, block in self.incoming]
        return f"%{self.result.name} = phi {self.result.type.name} {', '.join(incoming_strs)}"


class IRAlloca(IRInstruction):
    """Stack allocation instruction."""
    allocated_type: IRType
    
    def __init__(self, allocated_type: IRType):
        ptr_type = IRPrimitiveType(IRDataType.PTR)
        super().__init__(IRNodeType.ALLOCA, [], ptr_type)
        self.allocated_type = allocated_type
    
    def __str__(self) -> str:
        return f"%{self.result.name} = alloca {self.allocated_type.name}"


class IRLoad(IRInstruction):
    """Load instruction."""
    pointer: IRValue
    
    def __init__(self, pointer: IRValue, result_type: IRType):
        super().__init__(IRNodeType.LOAD, [pointer], result_type)
        self.pointer = pointer
    
    def __str__(self) -> str:
        return f"%{self.result.name} = load {self.result.type.name}, {self.pointer}"


class IRStore(IRInstruction):
    """Store instruction."""
    value: IRValue
    pointer: IRValue
    
    def __init__(self, value: IRValue, pointer: IRValue):
        super().__init__(IRNodeType.STORE, [value, pointer])
        self.value = value
        self.pointer = pointer
    
    def __str__(self) -> str:
        return f"store {self.value}, {self.pointer}"


# ============================================================================
# Tensor Operations
# ============================================================================

class IRTensorOp(IRInstruction):
    """Base class for tensor operations."""
    pass


@dataclass
class IRMatMul(IRTensorOp):
    """Matrix multiplication operation."""
    left: IRValue
    right: IRValue
    
    def __init__(self, left: IRValue, right: IRValue, result_type: IRTensorType):
        super().__init__(IRNodeType.MATMUL, [left, right], result_type)
        self.left = left
        self.right = right
    
    def __str__(self) -> str:
        return f"%{self.result.name} = matmul {self.left}, {self.right}"


@dataclass
class IRConv2D(IRTensorOp):
    """2D convolution operation."""
    input: IRValue
    kernel: IRValue
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    
    def __init__(self, input: IRValue, kernel: IRValue, stride: Tuple[int, int], 
                 padding: Tuple[int, int], result_type: IRTensorType):
        super().__init__(IRNodeType.CONV2D, [input, kernel], result_type)
        self.input = input
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
    
    def __str__(self) -> str:
        return f"%{self.result.name} = conv2d {self.input}, {self.kernel}, stride={self.stride}, padding={self.padding}"


@dataclass
class IRReshape(IRTensorOp):
    """Tensor reshape operation."""
    input: IRValue
    new_shape: List[int]
    
    def __init__(self, input: IRValue, new_shape: List[int], result_type: IRTensorType):
        super().__init__(IRNodeType.RESHAPE, [input], result_type)
        self.input = input
        self.new_shape = new_shape
    
    def __str__(self) -> str:
        return f"%{self.result.name} = reshape {self.input}, {self.new_shape}"


@dataclass
class IRTranspose(IRTensorOp):
    """Tensor transpose operation."""
    input: IRValue
    axes: Optional[List[int]]
    
    def __init__(self, input: IRValue, axes: Optional[List[int]], result_type: IRTensorType):
        super().__init__(IRNodeType.TRANSPOSE, [input], result_type)
        self.input = input
        self.axes = axes
    
    def __str__(self) -> str:
        if self.axes:
            return f"%{self.result.name} = transpose {self.input}, axes={self.axes}"
        else:
            return f"%{self.result.name} = transpose {self.input}"


# ============================================================================
# Mathematical Operations
# ============================================================================

@dataclass
class IRMathOp(IRInstruction):
    """Mathematical operation."""
    operation: str
    operand: IRValue
    
    def __init__(self, operation: str, operand: IRValue, result_type: IRType):
        super().__init__(IRNodeType.MATH_OP, [operand], result_type)
        self.operation = operation
        self.operand = operand
    
    def __str__(self) -> str:
        return f"%{self.result.name} = {self.operation} {self.operand}"


# ============================================================================
# Automatic Differentiation
# ============================================================================

@dataclass
class IRGradient(IRInstruction):
    """Gradient computation."""
    function: IRValue
    variables: List[IRValue]
    
    def __init__(self, function: IRValue, variables: List[IRValue], result_type: IRType):
        operands = [function] + variables
        super().__init__(IRNodeType.GRADIENT, operands, result_type)
        self.function = function
        self.variables = variables
    
    def __str__(self) -> str:
        var_strs = [str(var) for var in self.variables]
        return f"%{self.result.name} = gradient {self.function}, [{', '.join(var_strs)}]"


@dataclass
class IRBackward(IRInstruction):
    """Backward pass for automatic differentiation."""
    loss: IRValue
    
    def __init__(self, loss: IRValue):
        super().__init__(IRNodeType.BACKWARD, [loss])
        self.loss = loss
    
    def __str__(self) -> str:
        return f"backward {self.loss}"


# ============================================================================
# GPU Operations
# ============================================================================

@dataclass
class IRKernel(IRFunction):
    """GPU kernel function."""
    grid_size: Tuple[int, int, int]
    block_size: Tuple[int, int, int]
    
    def __init__(self, name: str, func_type: IRFunctionType, 
                 grid_size: Tuple[int, int, int], block_size: Tuple[int, int, int]):
        super().__init__(name, func_type)
        self.node_type = IRNodeType.KERNEL
        self.grid_size = grid_size
        self.block_size = block_size
    
    def __str__(self) -> str:
        result = super().__str__()
        return f"__kernel__ {result} grid={self.grid_size} block={self.block_size}"


@dataclass
class IRLaunch(IRInstruction):
    """GPU kernel launch."""
    kernel: IRValue
    args: List[IRValue]
    grid_size: Tuple[int, int, int]
    block_size: Tuple[int, int, int]
    
    def __init__(self, kernel: IRValue, args: List[IRValue], 
                 grid_size: Tuple[int, int, int], block_size: Tuple[int, int, int]):
        operands = [kernel] + args
        super().__init__(IRNodeType.LAUNCH, operands)
        self.kernel = kernel
        self.args = args
        self.grid_size = grid_size
        self.block_size = block_size
    
    def __str__(self) -> str:
        arg_strs = [str(arg) for arg in self.args]
        return f"launch {self.kernel}<<<{self.grid_size}, {self.block_size}>>>({', '.join(arg_strs)})"
