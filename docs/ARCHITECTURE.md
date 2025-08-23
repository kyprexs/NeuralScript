# NeuralScript Architecture v0.1.0

## Executive Summary

NeuralScript is architected as a high-performance, statically-typed programming language specifically optimized for data science, machine learning, and scientific computing workloads. The architecture emphasizes mathematical expressiveness, compile-time safety, and runtime performance through native code generation.

## Design Goals

### Primary Objectives
1. **Performance**: Match or exceed C++ performance for numerical kernels
2. **Expressiveness**: Natural mathematical syntax with Unicode support
3. **Safety**: Eliminate common errors through static analysis
4. **Productivity**: Rich tooling ecosystem and excellent developer experience

### Success Metrics
- **Compilation Speed**: < 5 seconds for 10,000 lines of code
- **Runtime Performance**: Within 10% of hand-optimized C++ for BLAS operations
- **Memory Efficiency**: 30% less memory usage than Python for equivalent programs
- **Developer Experience**: Sub-second IDE response times, comprehensive error messages

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    NeuralScript Toolchain                   │
├─────────────────────────────────────────────────────────────┤
│  IDE/Editor     │  REPL/Notebook  │  Package Manager (nspm) │
│   Extensions    │    Interface     │    & Registry          │
├─────────────────────────────────────────────────────────────┤
│                 Language Server Protocol (LSP)              │
├─────────────────────────────────────────────────────────────┤
│               NeuralScript Compiler (nsc)                   │
│  ┌─────────┬─────────┬──────────┬──────────┬──────────────┐ │
│  │ Lexer   │ Parser  │ Analyzer │ AutoDiff │   Codegen    │ │
│  │         │         │          │  Pass    │   Backend    │ │
│  └─────────┴─────────┴──────────┴──────────┴──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                 NeuralScript Runtime                        │
│  ┌──────────────┬─────────────────┬─────────────────────┐   │
│  │ Memory Mgmt  │   Concurrency   │  Foreign Function   │   │
│  │ (GC + Owned) │     Runtime     │    Interface        │   │
│  └──────────────┴─────────────────┴─────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                    Standard Library                         │
│  ┌─────────┬─────────┬──────────────┬───────────────────┐  │
│  │  Core   │  Math   │ Machine      │   Scientific      │  │
│  │ Types   │ & Linalg│  Learning    │   Computing       │  │
│  └─────────┴─────────┴──────────────┴───────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                  Native/GPU Backends                        │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐ │
│  │ x86-64/ARM  │    CUDA     │   OpenCL    │   Vulkan    │ │
│  │   Native    │   Backend   │   Backend   │  Compute    │ │
│  └─────────────┴─────────────┴─────────────┴─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Compiler Architecture

### Compilation Pipeline

```
Source Code (.ns)
       ↓
┌─────────────────┐
│     Lexer       │ ← UTF-8 with mathematical symbols
│                 │   Unicode normalization
└─────────────────┘   Error recovery
       ↓
┌─────────────────┐
│     Parser      │ ← Pratt parser for operators
│  (Pratt-based)  │   Rich AST with source spans
└─────────────────┘   Macro expansion
       ↓
┌─────────────────┐
│ Semantic        │ ← Symbol resolution
│ Analysis        │   Type inference & checking
└─────────────────┘   Borrow checker
       ↓
┌─────────────────┐
│ Automatic       │ ← Reverse-mode AD
│ Differentiation │   Gradient computation
└─────────────────┘   Chain rule application
       ↓
┌─────────────────┐
│ NS-IR           │ ← High-level IR
│ Generation      │   SSA form
└─────────────────┘   Control flow graph
       ↓
┌─────────────────┐
│ Optimization    │ ← Dead code elimination
│ Passes          │   Loop optimization
└─────────────────┘   Vectorization
       ↓
┌─────────────────┐
│ Code Generation │ ← LLVM IR generation
│ Backend         │   Machine code emission
└─────────────────┘   Debug info generation
       ↓
Native Binary/Object Files
```

### Key Components

#### 1. Lexer (UTF-8 Mathematical Tokenizer)
- **Purpose**: Convert source text into token stream
- **Special Features**:
  - Native Unicode mathematical symbol support (∑, ∂, ∇, etc.)
  - Context-sensitive keyword recognition
  - Dimensional unit parsing (5.0m/s², 3.2kg⋅m/s)
  - Complex number literals (3+4i)
- **Error Recovery**: Insert missing tokens, skip invalid characters
- **Performance**: Zero-copy tokenization where possible

#### 2. Parser (Pratt-Based Expression Parser)
- **Purpose**: Build Abstract Syntax Tree from token stream
- **Design**: Top-down operator precedence (Pratt parsing)
- **Features**:
  - Operator precedence for mathematical expressions
  - Rich AST nodes with source location information
  - Macro system integration
  - Error recovery with synchronization points
- **AST Design**: Strongly typed AST nodes with visitor pattern support

#### 3. Semantic Analyzer
- **Symbol Resolution**: Multi-pass resolver for forward declarations
- **Type System**:
  - Hindley-Milner type inference
  - Dependent types for array dimensions
  - Unit/dimensional analysis
  - Ownership and borrowing analysis
- **Error Reporting**: Rich diagnostics with fix-it suggestions

#### 4. Automatic Differentiation Pass
- **Algorithm**: Reverse-mode automatic differentiation
- **Integration**: Compiler pass, not library-based
- **Features**:
  - Source-to-source transformation
  - Gradient function generation
  - Memory-efficient tape-free implementation
  - Higher-order derivatives support

#### 5. Intermediate Representation (NS-IR)
- **Design**: Static Single Assignment (SSA) form
- **Optimization-Friendly**: Control flow graph representation
- **Domain-Specific Operations**:
  - Tensor operations (matmul, conv, pool)
  - Mathematical functions (sin, exp, log)
  - Parallel constructs (parallel_for, async_call)
  - GPU kernel dispatch

#### 6. Code Generation Backend
- **Primary Target**: LLVM IR for mature optimization pipeline
- **Future Targets**: Direct x86-64/ARM64 assembly generation
- **GPU Support**: PTX generation for CUDA, SPIR-V for OpenCL/Vulkan
- **Debug Information**: DWARF format for debugging support

## Runtime Architecture

### Memory Management System

```
┌─────────────────────────────────────────────────────────┐
│                  Memory Management                      │
├─────────────────────────────────────────────────────────┤
│  Stack Allocation (for small, short-lived objects)     │
├─────────────────────────────────────────────────────────┤
│  Ownership System (compile-time memory safety)         │
│  ┌─────────────────────┬─────────────────────────────┐ │
│  │  Owned Values       │    Borrowed References      │ │
│  │  (Move Semantics)   │    (&T, &mut T)            │ │
│  └─────────────────────┴─────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│  Garbage Collector (for complex ownership patterns)    │
│  ┌─────────────────────┬─────────────────────────────┐ │
│  │ Generational GC     │   Large Object Heap        │ │
│  │ (Small objects)     │   (Tensors, Arrays)        │ │
│  └─────────────────────┴─────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│  GPU Memory Management                                  │
│  ┌─────────────────────┬─────────────────────────────┐ │
│  │  Device Memory      │   Unified Memory           │ │
│  │  (Explicit)         │   (CUDA/OpenCL)           │ │
│  └─────────────────────┴─────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Concurrency Runtime

```
┌─────────────────────────────────────────────────────────┐
│                Concurrency Runtime                     │
├─────────────────────────────────────────────────────────┤
│  Green Thread Scheduler (M:N threading)               │
│  ┌─────────────────────┬─────────────────────────────┐ │
│  │  Work-Stealing      │    Priority Scheduling      │ │
│  │  Task Queue         │    (Real-time support)     │ │
│  └─────────────────────┴─────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│  Async/Await Implementation                            │
│  ┌─────────────────────┬─────────────────────────────┐ │
│  │  Future/Promise     │    Async I/O               │ │
│  │  State Machines     │    (epoll/kqueue/IOCP)    │ │
│  └─────────────────────┴─────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│  Actor System (Distributed Computing)                 │
│  ┌─────────────────────┬─────────────────────────────┐ │
│  │  Message Passing    │    Location Transparency   │ │
│  │  (Channels)         │    (Local + Remote)        │ │
│  └─────────────────────┴─────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│  Data Parallelism                                      │
│  ┌─────────────────────┬─────────────────────────────┐ │
│  │  SIMD Vectorization │    GPU Kernel Dispatch     │ │
│  │  (AVX, NEON)        │    (CUDA, OpenCL, Vulkan) │ │
│  └─────────────────────┴─────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Type System Architecture

### Core Type Categories

```
NeuralScript Type Hierarchy
├── Primitive Types
│   ├── Integers: i8, i16, i32, i64, i128, isize
│   ├── Unsigned: u8, u16, u32, u64, u128, usize  
│   ├── Floats: f16, f32, f64, f128
│   ├── Complex: c32, c64, c128
│   └── Others: bool, char, str
├── Composite Types
│   ├── Tuples: (T1, T2, ..., Tn)
│   ├── Arrays: Array<T, const N>
│   ├── Vectors: Vec<T>
│   ├── Tensors: Tensor<T, const DIMS>
│   └── Structs: struct Name { field: Type, ... }
├── Function Types
│   ├── Functions: fn(T1, T2) -> T3
│   ├── Closures: |T1, T2| -> T3
│   └── Async Functions: async fn(T1) -> Future<T2>
├── Generic Types
│   ├── Type Parameters: <T: Trait>
│   ├── Const Generics: <const N: usize>
│   └── Associated Types: trait Iterator { type Item; }
├── Unit Types (Dimensional Analysis)
│   ├── Base Units: Meter, Kilogram, Second, ...
│   ├── Derived Units: Newton, Joule, Pascal, ...
│   └── Compound Units: Meter/Second², Joule⋅Meter
└── Reference Types
    ├── Shared References: &T
    ├── Mutable References: &mut T
    └── Smart Pointers: Box<T>, Rc<T>, Arc<T>
```

### Type Inference Engine

The type inference system is based on Algorithm W (Hindley-Milner) extended with:

1. **Dimensional Constraints**: Track physical units through the type system
2. **Shape Constraints**: Tensor dimensions must be consistent
3. **Lifetime Constraints**: Ownership and borrowing rules
4. **Effect Constraints**: Async/sync, pure/impure function tracking

## Standard Library Architecture

### Module Organization

```
stdlib/
├── core/                    # Core language constructs
│   ├── types.ns            # Primitive type definitions
│   ├── ops.ns              # Operator implementations
│   ├── mem.ns              # Memory management utilities
│   └── error.ns            # Error handling (Result, Option)
├── collections/             # Data structures
│   ├── array.ns            # Fixed-size arrays
│   ├── vec.ns              # Dynamic arrays
│   ├── tensor.ns           # N-dimensional tensors
│   ├── map.ns              # Hash maps and tree maps
│   └── graph.ns            # Graph data structures
├── math/                    # Mathematical functions
│   ├── scalar.ns           # Scalar mathematical functions
│   ├── linalg.ns           # Linear algebra operations
│   ├── fft.ns              # Fast Fourier Transform
│   ├── special.ns          # Special functions (gamma, bessel)
│   └── stats.ns            # Statistical functions
├── ml/                      # Machine learning primitives
│   ├── tensor_ops.ns       # Tensor operations (conv, pool, etc.)
│   ├── neural.ns           # Neural network layers
│   ├── optimizers.ns       # Optimization algorithms
│   ├── losses.ns           # Loss functions
│   └── metrics.ns          # Evaluation metrics
├── scientific/              # Scientific computing
│   ├── integrate.ns        # Numerical integration
│   ├── solve.ns            # Equation solving
│   ├── optimize.ns         # Optimization algorithms
│   ├── signal.ns           # Signal processing
│   └── units.ns            # Physical units and constants
├── async/                   # Asynchronous programming
│   ├── runtime.ns          # Async runtime
│   ├── channels.ns         # Message passing
│   ├── actors.ns           # Actor system
│   └── parallel.ns         # Parallel algorithms
└── io/                      # Input/Output operations
    ├── fs.ns               # File system operations
    ├── network.ns          # Network programming
    ├── formats.ns          # Data format parsers (CSV, JSON, etc.)
    └── datasets.ns         # Dataset loading utilities
```

## Tooling Ecosystem Architecture

### Development Tools Integration

```
┌─────────────────────────────────────────────────────────┐
│                Development Environment                   │
├─────────────────────────────────────────────────────────┤
│  IDEs & Editors                                        │
│  ┌─────────────┬─────────────┬─────────────────────┐   │
│  │  VS Code    │   Neovim    │    IntelliJ IDEA    │   │
│  │ Extension   │   Plugin    │      Plugin         │   │
│  └─────────────┴─────────────┴─────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  Language Server Protocol (LSP)                        │
│  ┌─────────────┬─────────────┬─────────────────────┐   │
│  │ Semantic    │  Code       │   Inline            │   │
│  │ Highlighting│  Completion │   Diagnostics       │   │
│  └─────────────┴─────────────┴─────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  Interactive Development                               │
│  ┌─────────────┬─────────────┬─────────────────────┐   │
│  │   REPL      │  Jupyter    │   Notebooks         │   │
│  │             │ Integration │   (.nsnb format)    │   │
│  └─────────────┴─────────────┴─────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  Package Management                                    │
│  ┌─────────────┬─────────────┬─────────────────────┐   │
│  │ Dependency  │   Build     │    Registry         │   │
│  │ Resolution  │  Scripts    │   (ns-registry.org) │   │
│  └─────────────┴─────────────┴─────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  Debugging & Profiling                                │
│  ┌─────────────┬─────────────┬─────────────────────┐   │
│  │ Source-Level│ Performance │   Memory            │   │
│  │ Debugger    │  Profiler   │   Profiler          │   │
│  └─────────────┴─────────────┴─────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Performance Architecture

### Optimization Strategy

1. **Compile-Time Optimizations**
   - Constant folding and propagation
   - Dead code elimination
   - Loop optimizations (unrolling, fusion, vectorization)
   - Tensor operation fusion
   - Automatic parallelization

2. **Runtime Optimizations**
   - JIT compilation for dynamic workloads
   - Adaptive optimization based on runtime profiling
   - GPU kernel auto-tuning
   - Memory layout optimization

3. **Domain-Specific Optimizations**
   - BLAS/LAPACK integration for linear algebra
   - Specialized kernels for common ML operations
   - Automatic GPU/CPU dispatch based on data size
   - Vectorization using SIMD instructions (AVX, NEON)

### Performance Monitoring

```
┌─────────────────────────────────────────────────────────┐
│              Performance Monitoring                     │
├─────────────────────────────────────────────────────────┤
│  Compile-Time Metrics                                  │
│  ┌─────────────┬─────────────┬─────────────────────┐   │
│  │ Compilation │ Memory      │   Code Size         │   │
│  │    Time     │  Usage      │   Statistics        │   │
│  └─────────────┴─────────────┴─────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  Runtime Metrics                                       │
│  ┌─────────────┬─────────────┬─────────────────────┐   │
│  │ Execution   │   Memory    │    GPU              │   │
│  │   Time      │ Allocation  │   Utilization       │   │
│  └─────────────┴─────────────┴─────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  Benchmarking Framework                                │
│  ┌─────────────┬─────────────┬─────────────────────┐   │
│  │ Micro-      │  Macro-     │   Regression        │   │
│  │ benchmarks  │  benchmarks │   Testing           │   │
│  └─────────────┴─────────────┴─────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Security Architecture

### Memory Safety
- Ownership system prevents use-after-free and double-free
- Borrow checker prevents data races
- Array bounds checking (with optimization to remove redundant checks)
- Integer overflow checking in debug mode

### Type Safety
- Strong static typing prevents type confusion
- No null pointers (use Option<T> instead)
- Exhaustive pattern matching
- Dimensional analysis prevents unit errors

### Sandboxing
- Unsafe code blocks are clearly marked
- FFI calls have defined safe interfaces
- GPU kernels have memory access restrictions

## Extensibility Architecture

### Plugin System
```
┌─────────────────────────────────────────────────────────┐
│                  Plugin Architecture                    │
├─────────────────────────────────────────────────────────┤
│  Compiler Plugins                                      │
│  ┌─────────────┬─────────────┬─────────────────────┐   │
│  │  Custom     │   Linting   │   Code Generation   │   │
│  │  Passes     │   Rules     │     Plugins         │   │
│  └─────────────┴─────────────┴─────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  Runtime Extensions                                     │
│  ┌─────────────┬─────────────┬─────────────────────┐   │
│  │   Native    │    GPU      │    Distributed      │   │
│  │ Libraries   │  Backends   │    Computing        │   │
│  └─────────────┴─────────────┴─────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  Domain-Specific Languages                             │
│  ┌─────────────┬─────────────┬─────────────────────┐   │
│  │   Embedded  │   Query     │    Visualization    │   │
│  │    DSLs     │ Languages   │       DSLs          │   │
│  └─────────────┴─────────────┴─────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

This architecture provides the foundation for a robust, high-performance programming language that can grow and adapt to the evolving needs of the data science and scientific computing communities.
