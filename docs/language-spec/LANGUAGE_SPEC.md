# NeuralScript Language Specification v0.1.0

## 1. Overview

NeuralScript is a statically-typed, multi-paradigm programming language specifically designed for data science, machine learning, and scientific computing. It combines the mathematical expressiveness of Julia, the performance of Rust, the type safety of Haskell, and the ease of use of Python.

### 1.1 Design Principles

1. **Mathematical Naturalism**: Code should look like mathematical notation
2. **Performance by Default**: Generate optimized native code without manual tuning
3. **Type Safety**: Catch errors at compile time, especially dimensional mismatches
4. **Expressiveness**: Support multiple programming paradigms seamlessly
5. **Tooling Excellence**: Provide world-class development tools

### 1.2 Key Features

- **Multi-paradigm**: Functional, Object-Oriented, and Imperative styles
- **First-class tensors**: Built-in support for N-dimensional arrays with shape checking
- **Automatic differentiation**: Compiler-integrated gradient computation
- **Unicode mathematics**: Native support for mathematical symbols (∑, ∂, ∇, etc.)
- **Dimensional analysis**: Compile-time unit checking
- **Memory safety**: Ownership system with garbage collection fallback
- **Concurrency**: Built-in async/await and actor model support
- **GPU computing**: Seamless CUDA/OpenCL integration

## 2. Lexical Structure

### 2.1 Character Set

NeuralScript source code is encoded in UTF-8 and supports the full Unicode character set for identifiers and mathematical operators.

### 2.2 Identifiers

```ebnf
identifier = (letter | mathematical_symbol) (letter | digit | underscore | mathematical_symbol)*
letter = unicode_letter
mathematical_symbol = "α" | "β" | "γ" | "δ" | "ε" | "θ" | "λ" | "μ" | "ν" | "π" | "ρ" | "σ" | "τ" | "φ" | "χ" | "ψ" | "ω" | "Α" | "Β" | "Γ" | "Δ" | "Ε" | "Θ" | "Λ" | "Μ" | "Ν" | "Π" | "Ρ" | "Σ" | "Τ" | "Φ" | "Χ" | "Ψ" | "Ω"
```

Examples:
```neuralscript
let x = 5
let α = 3.14159
let θ₁ = rotation_angle
let Δt = time_step
```

### 2.3 Keywords

Reserved keywords cannot be used as identifiers:

```
async    await    break    const    continue  else      enum     false
fn       for      if       impl     import    in        let      loop
match    mod      mut      pub      return    self      Self     static
struct   super    trait    true     type      unsafe    use      where
while    yield

// Mathematical keywords
grad     div      curl     laplace  sum       prod      integral  diff
```

### 2.4 Operators

#### 2.4.1 Arithmetic Operators
```
+   -   *   /   %   **   //
⊙   ⊗   ⊕   ⊖   ⋅   ×    ÷
```

#### 2.4.2 Comparison Operators
```
==  !=  <   >   <=  >=
≡   ≠   ≤   ≥   ≈   ∈   ∉
```

#### 2.4.3 Logical Operators
```
&&  ||  !
∧   ∨   ¬   ⊕   ↔
```

#### 2.4.4 Mathematical Operators
```
∇   ∂   ∑   ∏   ∫   √   ∛
```

### 2.5 Literals

#### 2.5.1 Integer Literals
```
42          // decimal
0x2A        // hexadecimal
0o52        // octal
0b101010    // binary
1_000_000   // underscores for readability
```

#### 2.5.2 Floating-Point Literals
```
3.14
6.626e-34
1.23e+10
2.71828f32      // explicit f32 type
```

#### 2.5.3 Complex Literals
```
3 + 4i
2.5 - 1.8i
1e-3 + 2e-4i
```

#### 2.5.4 Unit Literals
```
5.0m        // meters
9.8m/s²     // acceleration
6.626e-34J⋅s // Planck constant
```

#### 2.5.5 String Literals
```
"Hello, world!"
r"Raw string with \backslashes"
f"Formatted {variable} string"
```

## 3. Type System

### 3.1 Primitive Types

```neuralscript
// Integer types
i8, i16, i32, i64, i128, isize
u8, u16, u32, u64, u128, usize

// Floating-point types
f16, f32, f64, f128

// Complex types
c32, c64, c128  // complex numbers

// Boolean
bool

// Character and string
char, str
```

### 3.2 Composite Types

#### 3.2.1 Arrays and Tensors
```neuralscript
// Fixed-size arrays
Array<T, const N: usize>
let arr: Array<f32, 10> = [1.0, 2.0, 3.0, ..., 10.0]

// Dynamic arrays
Vec<T>
let vec: Vec<f32> = vec![1.0, 2.0, 3.0]

// Tensors with compile-time shape checking
Tensor<T, const DIMS: &'static [usize]>
let matrix: Tensor<f32, [3, 4]> = tensor![[1, 2, 3, 4],
                                          [5, 6, 7, 8],
                                          [9, 10, 11, 12]]

// Mathematical aliases
Matrix<T, const M: usize, const N: usize> = Tensor<T, [M, N]>
Vector<T, const N: usize> = Tensor<T, [N]>
```

#### 3.2.2 Tuples
```neuralscript
let point: (f32, f32, f32) = (1.0, 2.0, 3.0)
let named_tuple: (x: f32, y: f32, z: f32) = (x: 1.0, y: 2.0, z: 3.0)
```

#### 3.2.3 Structs
```neuralscript
struct Point3D {
    x: f32,
    y: f32,
    z: f32,
}

// Generic structs with constraints
struct NeuralLayer<T: Numeric + Clone, const IN: usize, const OUT: usize> {
    weights: Matrix<T, IN, OUT>,
    biases: Vector<T, OUT>,
    activation: fn(T) -> T,
}
```

### 3.3 Unit Types and Dimensional Analysis

NeuralScript provides compile-time dimensional analysis to prevent unit-related errors:

```neuralscript
// Base SI units
struct Meter;      // Length
struct Kilogram;   // Mass  
struct Second;     // Time
struct Ampere;     // Electric current
struct Kelvin;     // Temperature
struct Mole;       // Amount of substance
struct Candela;    // Luminous intensity

// Derived units
type Newton = Kilogram * Meter / (Second * Second);
type Joule = Newton * Meter;
type Watt = Joule / Second;

// Unit-aware calculations
fn kinetic_energy(mass: Kilogram, velocity: Meter/Second) -> Joule {
    0.5 * mass * velocity²  // Compiler checks dimensional correctness
}
```

### 3.4 Generics and Traits

```neuralscript
// Generic functions
fn dot_product<T: Numeric>(a: Vector<T, N>, b: Vector<T, N>) -> T {
    sum(a ⊙ b)  // Element-wise multiplication and sum
}

// Trait definitions
trait Numeric {
    fn zero() -> Self;
    fn one() -> Self;
    fn sqrt(self) -> Self;
}

// Associated types and constants
trait TensorOp {
    type Item;
    const RANK: usize;
    
    fn shape(&self) -> [usize; Self::RANK];
    fn reshape<const NEW_SHAPE: &'static [usize]>(self) -> Tensor<Self::Item, NEW_SHAPE>;
}
```

## 4. Syntax and Semantics

### 4.1 Variable Declarations

```neuralscript
// Immutable by default
let x = 42;
let name = "NeuralScript";

// Mutable variables
let mut counter = 0;
counter += 1;

// Type annotations
let matrix: Matrix<f32, 3, 3> = zeros();
let λ: f64 = 0.001;  // Unicode identifiers

// Destructuring
let (x, y, z) = get_coordinates();
let Point { x, y } = point;
```

### 4.2 Functions

```neuralscript
// Basic function
fn add(a: i32, b: i32) -> i32 {
    a + b
}

// Generic functions
fn max<T: Ord>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

// Higher-order functions
fn map<T, U, const N: usize>(
    arr: Array<T, N>, 
    f: fn(T) -> U
) -> Array<U, N> {
    // Implementation
}

// Mathematical functions with Unicode
fn ∇f(f: fn(Vector<f64, N>) -> f64, x: Vector<f64, N>) -> Vector<f64, N> {
    // Automatic differentiation
    grad(f, x)
}
```

### 4.3 Control Flow

```neuralscript
// If expressions
let result = if condition { value1 } else { value2 };

// Pattern matching
match tensor_op {
    TensorOp::Add(a, b) => a + b,
    TensorOp::Multiply(a, b) => a ⊙ b,
    TensorOp::Transpose(t) => t.transpose(),
    _ => panic!("Unknown operation"),
}

// Loops
for i in 0..10 {
    println!("{i}");
}

// Parallel loops for tensors
for batch in dataset.parallel_batches(32) {
    // Parallel processing
}

// While loops
while convergence_criterion > tolerance {
    // Optimization iteration
}
```

### 4.4 Automatic Differentiation

```neuralscript
// Function marked for automatic differentiation
#[differentiable]
fn neural_network(
    x: Vector<f32, 784>, 
    weights: Matrix<f32, 784, 128>
) -> Vector<f32, 10> {
    let hidden = relu(weights ⊙ x);
    softmax(output_layer(hidden))
}

// Automatic gradient computation
let gradients = ∇neural_network(input, weights);

// Higher-order derivatives
let hessian = ∇²loss_function(parameters);
```

### 4.5 Concurrency and Parallelism

```neuralscript
// Async functions
async fn load_dataset(path: &str) -> Dataset {
    let file = async_read_file(path).await;
    parse_dataset(file)
}

// Parallel tensor operations
fn parallel_matrix_multiply(a: Matrix<f32, M, K>, b: Matrix<f32, K, N>) -> Matrix<f32, M, N> {
    // Automatically parallelized
    a ⊙ b
}

// Actor model for distributed computing
actor DataProcessor {
    state: ProcessorState,
    
    async fn process_batch(&mut self, batch: DataBatch) -> ProcessedBatch {
        // Process batch
    }
}
```

## 5. Standard Library Overview

### 5.1 Core Types
- Fundamental numeric types and operations
- Collection types (Array, Vec, HashMap)
- String manipulation
- Error handling (Result, Option)

### 5.2 Mathematics
- Linear algebra operations
- Statistical functions
- Probability distributions
- Special functions (gamma, bessel, etc.)

### 5.3 Machine Learning
- Tensor operations
- Neural network layers
- Optimization algorithms
- Loss functions

### 5.4 Scientific Computing
- Numerical integration and differentiation
- Solving differential equations
- Optimization and root finding
- Signal processing

## 6. Memory Management

NeuralScript uses a hybrid approach:

1. **Ownership System**: Inspired by Rust, for compile-time memory safety
2. **Garbage Collection**: For convenience when ownership is complex
3. **Manual Control**: For performance-critical sections

```neuralscript
// Ownership (zero-cost abstraction)
fn process_data(data: Vec<f32>) -> Vec<f32> {
    data.map(|x| x * 2.0)  // data is moved and consumed
}

// References
fn analyze_data(data: &Vec<f32>) -> Statistics {
    // data is borrowed, not moved
}

// Manual memory management for performance
unsafe {
    let raw_ptr = allocate_aligned(size, alignment);
    // Manual management
    deallocate(raw_ptr);
}
```

## 7. Compilation Model

NeuralScript compiles to native machine code through the following pipeline:

1. **Lexical Analysis**: Source code → Token stream
2. **Parsing**: Token stream → Abstract Syntax Tree (AST)
3. **Semantic Analysis**: AST → Type-checked AST with symbol tables
4. **Automatic Differentiation**: Insert gradient computation nodes
5. **IR Generation**: High-level AST → NeuralScript IR (NS-IR)
6. **Optimization**: Various optimization passes on NS-IR
7. **Code Generation**: NS-IR → Native machine code (via LLVM initially)

### 7.1 Just-In-Time Compilation

For interactive development:
```neuralscript
// JIT compilation for REPL and notebooks
ns> let x = tensor![1, 2, 3, 4]
ns> let y = x.map(|i| i * 2)
ns> println!("{y}")  // Compiled and executed immediately
```

## 8. Interoperability

### 8.1 C/C++ Integration
```neuralscript
extern "C" {
    fn cblas_dgemm(/* BLAS parameters */);
}

#[link(name = "openblas")]
extern "C" {
    // Link with external libraries
}
```

### 8.2 Python Integration
```neuralscript
use python::{numpy as np, torch};

fn call_python_function(data: Matrix<f32, 100, 50>) -> Matrix<f32, 100, 10> {
    let py_array = data.to_numpy();
    let result = python! {
        model = torch.load("model.pt")
        output = model(py_array)
        return output.numpy()
    };
    Matrix::from_numpy(result)
}
```

## 9. Error Handling

```neuralscript
// Result type for error handling
enum Result<T, E> {
    Ok(T),
    Err(E),
}

// Dimension mismatch errors (compile-time)
fn invalid_multiply() {
    let a: Matrix<f32, 3, 4> = zeros();
    let b: Matrix<f32, 5, 6> = zeros();
    let c = a ⊙ b;  // COMPILE ERROR: Dimension mismatch
}

// Runtime errors
fn safe_divide(a: f32, b: f32) -> Result<f32, &'static str> {
    if b == 0.0 {
        Err("Division by zero")
    } else {
        Ok(a / b)
    }
}
```

## 10. Grammar Specification (EBNF)

```ebnf
(* NeuralScript Grammar *)

program = item*

item = function_item
     | struct_item  
     | impl_item
     | trait_item
     | use_item
     | mod_item

function_item = visibility? "fn" identifier generic_params? "(" param_list? ")" return_type? block_expression

struct_item = visibility? "struct" identifier generic_params? struct_fields

generic_params = "<" generic_param ("," generic_param)* ">"
generic_param = identifier (":" trait_bound_list)?

expression = assignment_expression

assignment_expression = logical_or_expression (assignment_op logical_or_expression)*
assignment_op = "=" | "+=" | "-=" | "*=" | "/="

logical_or_expression = logical_and_expression ("||" logical_and_expression)*
logical_and_expression = equality_expression ("&&" equality_expression)*
equality_expression = relational_expression (equality_op relational_expression)*
equality_op = "==" | "!=" | "≡" | "≠"

relational_expression = additive_expression (relational_op additive_expression)*
relational_op = "<" | ">" | "<=" | ">=" | "≤" | "≥" | "∈" | "∉"

additive_expression = multiplicative_expression (additive_op multiplicative_expression)*
additive_op = "+" | "-" | "⊕" | "⊖"

multiplicative_expression = unary_expression (multiplicative_op unary_expression)*
multiplicative_op = "*" | "/" | "%" | "⊙" | "⊗" | "⋅" | "×" | "÷"

unary_expression = unary_op* postfix_expression
unary_op = "!" | "-" | "¬" | "∇" | "∂" | "√"

postfix_expression = primary_expression postfix_op*
postfix_op = "[" expression "]"
           | "." identifier
           | "(" argument_list? ")"

primary_expression = identifier
                  | literal
                  | "(" expression ")"
                  | array_expression
                  | tensor_expression

literal = integer_literal
        | float_literal  
        | complex_literal
        | string_literal
        | boolean_literal

integer_literal = decimal_literal | hex_literal | octal_literal | binary_literal
decimal_literal = digit (digit | "_")*
```

This specification will continue to evolve as the language develops. The next phase will involve implementing the lexer based on this specification.
