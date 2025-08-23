# NeuralScript Language Guide

*A comprehensive guide to NeuralScript's features and capabilities*

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Syntax](#basic-syntax)
3. [Type System](#type-system)
4. [Mathematical Notation](#mathematical-notation)
5. [Complex Numbers](#complex-numbers)
6. [Unit Literals & Dimensional Analysis](#unit-literals--dimensional-analysis)
7. [Functions and Control Flow](#functions-and-control-flow)
8. [Structs and Implementations](#structs-and-implementations)
9. [Advanced Features](#advanced-features)
10. [Standard Library](#standard-library)

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/xwest/neuralscript.git
cd neuralscript

# Set up development environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the interactive REPL
python repl.py

# Compile a NeuralScript program
python -m neuralscript compile program.ns
```

### Your First Program

Create a file `hello.ns`:

```neuralscript
fn main() -> i32 {
    println!("Hello, NeuralScript!")
    0
}
```

Compile and run:
```bash
python -m neuralscript run hello.ns
```

## Basic Syntax

### Variables and Constants

```neuralscript
// Variable declarations with type inference
let x = 42          // i32
let y = 3.14        // f64
let is_valid = true // bool
let name = "Alice"  // str

// Explicit type annotations
let count: i32 = 100
let ratio: f64 = 0.75

// Constants
const PI: f64 = 3.14159265359
const MAX_SIZE: usize = 1024
```

### Comments

```neuralscript
// Single-line comment

/*
 * Multi-line comment
 * with detailed explanations
 */
```

## Type System

NeuralScript has a rich type system with automatic type inference:

### Primitive Types

```neuralscript
// Integer types
let small: i8 = 127
let normal: i32 = 2147483647
let big: i64 = 9223372036854775807

// Floating-point types
let single: f32 = 3.14f32
let double: f64 = 2.718281828

// Boolean and character
let flag: bool = true
let letter: char = 'A'
```

### Composite Types

```neuralscript
// Arrays
let numbers: [i32; 5] = [1, 2, 3, 4, 5]
let matrix: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], 
                             [0.0, 1.0, 0.0], 
                             [0.0, 0.0, 1.0]]

// Tensors with compile-time shape checking
let tensor: Tensor<f32, [2, 3, 4]> = Tensor::zeros()

// Vectors
let vec: Vec<i32> = vec![1, 2, 3, 4, 5]
```

## Mathematical Notation

NeuralScript supports Unicode mathematical operators for natural expression:

### Basic Operators

```neuralscript
let a = 10
let b = 3

// Standard operators
let sum = a + b        // 13
let diff = a - b       // 7
let prod = a * b       // 30
let quot = a / b       // 3

// Unicode mathematical operators
let prod_unicode = a × b    // 30 (multiplication)
let quot_unicode = a ÷ b    // 3 (division)
let squared = a²            // 100 (superscript 2)
let cubed = b³             // 27 (superscript 3)
```

### Advanced Mathematical Operations

```neuralscript
// Square root and powers
let sqrt_val = √16.0       // 4.0
let power = 2^8           // 256

// Comparisons
let less_eq = a ≤ b       // false
let greater_eq = a ≥ b    // true
let not_equal = a ≠ b     // true

// Mathematical constants
let pi_val = π            // 3.14159...
let e_val = ℯ             // 2.71828...

// Trigonometric functions
let sine = sin(π/2)       // 1.0
let cosine = cos(0)       // 1.0
```

### Vector and Matrix Operations

```neuralscript
let v1 = vector![1.0, 2.0, 3.0]
let v2 = vector![4.0, 5.0, 6.0]

// Dot product
let dot = v1 • v2         // 32.0

// Cross product (3D vectors)
let cross = v1 × v2       // [-3.0, 6.0, -3.0]

// Matrix multiplication
let A = matrix![[1, 2], [3, 4]]
let B = matrix![[5, 6], [7, 8]]
let C = A ⊙ B            // Matrix multiplication
```

## Complex Numbers

Complex numbers are first-class citizens in NeuralScript:

### Complex Number Literals

```neuralscript
// Complex number literals
let z1 = 3.0+4.0i        // 3 + 4i
let z2 = 1.0-2.0i        // 1 - 2i
let z3 = 5.0+0.0i        // Real number as complex
let z4 = 0.0+1.0i        // Pure imaginary

// Complex arithmetic
let sum = z1 + z2        // 4 + 2i
let prod = z1 * z2       // 11 - 2i
let conj = z1*           // 3 - 4i (complex conjugate)
```

### Complex Functions

```neuralscript
fn mandelbrot(c: Complex<f64>, max_iter: i32) -> i32 {
    let mut z = 0.0+0.0i
    let mut iter = 0
    
    while iter < max_iter && |z| < 2.0 {
        z = z² + c
        iter += 1
    }
    
    iter
}
```

## Unit Literals & Dimensional Analysis

NeuralScript prevents dimensional analysis errors at compile time:

### Basic Units

```neuralscript
// SI base units
let mass = 50.0_kg           // kilograms
let length = 100.0_m         // meters
let time = 5.0_s             // seconds
let current = 2.0_A          // amperes
let temperature = 300.0_K    // kelvin
let amount = 1.0_mol         // moles
let luminosity = 1000.0_cd   // candelas
```

### Derived Units

```neuralscript
// Automatically derived units
let area = length²           // m²
let volume = length³         // m³
let velocity = length / time // m/s
let acceleration = velocity / time // m/s²
let force = mass × acceleration    // N (Newtons)
let energy = force × length       // J (Joules)
let power = energy / time         // W (Watts)
```

### Unit Safety

```neuralscript
fn calculate_kinetic_energy(mass: Kilogram, velocity: MeterPerSecond) -> Joule {
    0.5 × mass × velocity²
}

// Compile-time unit checking prevents errors
let ke = calculate_kinetic_energy(70.0_kg, 10.0_m/s)  // ✓ Valid
// let invalid = mass + velocity  // ✗ Compile error: Cannot add kg to m/s
```

### Complex Unit Expressions

```neuralscript
// Electromagnetic units
let voltage = 12.0_V
let current = 2.0_A
let resistance = voltage / current  // Ω (Ohms)
let power = voltage × current       // W (Watts)

// Physical constants with units
const LIGHT_SPEED: MeterPerSecond = 299_792_458_m/s
const PLANCK_CONSTANT: JouleSecond = 6.626e-34_J⋅s
const GRAVITATIONAL_CONSTANT: NewtonMeterSquaredPerKilogramSquared = 6.674e-11_N⋅m²/kg²
```

## Functions and Control Flow

### Function Definitions

```neuralscript
// Basic function
fn add(x: i32, y: i32) -> i32 {
    x + y
}

// Function with type inference
fn multiply(a, b) -> auto {
    a * b  // Return type inferred
}

// Generic functions
fn max<T: Comparable>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

// Higher-order functions
fn apply<T, U>(f: Fn(T) -> U, x: T) -> U {
    f(x)
}
```

### Control Flow

```neuralscript
// If expressions
let result = if condition {
    calculate_something()
} else {
    default_value()
}

// Pattern matching
match value {
    0 => println!("Zero"),
    1..=10 => println!("Small positive"),
    x if x < 0 => println!("Negative: {x}"),
    _ => println!("Other")
}

// Loops
for i in 0..10 {
    println!("Iteration {i}")
}

while condition {
    do_something()
}

loop {
    if break_condition {
        break
    }
}
```

## Structs and Implementations

### Struct Definitions

```neuralscript
// Basic struct
struct Point {
    x: f64,
    y: f64,
}

// Generic struct with constraints
struct Vector<T: Numeric, const N: usize> {
    data: [T; N],
}

// Struct with methods
impl Point {
    fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }
    
    fn distance(&self, other: &Point) -> f64 {
        √((self.x - other.x)² + (self.y - other.y)²)
    }
}
```

### Traits and Implementations

```neuralscript
// Define a trait
trait Drawable {
    fn draw(&self);
    fn area(&self) -> f64;
}

// Implement trait for struct
impl Drawable for Circle {
    fn draw(&self) {
        println!("Drawing circle at ({}, {}) with radius {}", 
                 self.center.x, self.center.y, self.radius)
    }
    
    fn area(&self) -> f64 {
        π × self.radius²
    }
}
```

## Advanced Features

### Automatic Differentiation

```neuralscript
// Functions are automatically differentiable
fn f(x: f64) -> f64 {
    x³ - 2.0×x² + x - 1.0
}

// Compute derivative using gradient operator
let x = 2.0
let df_dx = ∇f(x)  // Automatic differentiation

// Neural network with automatic gradients
fn loss_function(prediction: Vector<f64>, target: Vector<f64>) -> f64 {
    mse(prediction, target)
}

let gradients = ∇loss_function  // Compute gradients w.r.t. all parameters
```

### Tensor Operations

```neuralscript
// Define tensors with compile-time shape checking
fn matrix_multiply<const M: usize, const N: usize, const K: usize>(
    a: Tensor<f64, [M, N]>,
    b: Tensor<f64, [N, K]>
) -> Tensor<f64, [M, K]> {
    a ⊙ b  // Matrix multiplication
}

// Broadcasting operations
let a: Tensor<f32, [3, 1]> = tensor![[1.0], [2.0], [3.0]]
let b: Tensor<f32, [1, 4]> = tensor![[1.0, 2.0, 3.0, 4.0]]
let c = a + b  // Broadcasts to [3, 4]
```

### Parallel and Async Operations

```neuralscript
// Parallel computation
async fn parallel_sum(data: &[f64]) -> f64 {
    data.par_iter()
        .map(|&x| x * x)
        .sum()
}

// Async/await for I/O
async fn load_dataset(path: &str) -> Dataset<f32> {
    let file = File::open(path).await?
    parse_csv(file).await
}
```

### Physics Simulations

```neuralscript
// N-body gravitational simulation
fn update_bodies(bodies: &mut [Body], dt: Second) {
    for i in 0..bodies.len() {
        let mut force = Vector3::zero()
        
        for j in 0..bodies.len() {
            if i != j {
                let r_vec = bodies[j].position - bodies[i].position
                let r = r_vec.magnitude()
                let f_mag = G × bodies[i].mass × bodies[j].mass / r²
                force += f_mag × (r_vec / r)
            }
        }
        
        // Newton's second law: F = ma
        let acceleration = force / bodies[i].mass
        bodies[i].velocity += acceleration × dt
        bodies[i].position += bodies[i].velocity × dt
    }
}
```

## Standard Library

### Mathematical Functions

```neuralscript
// Basic math
let sqrt_val = sqrt(16.0)      // 4.0
let abs_val = abs(-5)          // 5
let max_val = max(10, 20)      // 20

// Trigonometric
let sin_val = sin(π/2)         // 1.0
let cos_val = cos(π)           // -1.0
let tan_val = tan(π/4)         // 1.0

// Logarithmic
let ln_val = ln(ℯ)             // 1.0
let log10_val = log10(100.0)   // 2.0
let log2_val = log2(8.0)       // 3.0
```

### Statistical Functions

```neuralscript
let data = [1.0, 2.0, 3.0, 4.0, 5.0]

let mean_val = mean(data)           // 3.0
let std_val = std(data)             // Standard deviation
let var_val = variance(data)        // Variance
let median_val = median(data)       // 3.0
```

### Linear Algebra

```neuralscript
let A = matrix![[1, 2], [3, 4]]
let b = vector![5, 6]

let det_A = det(A)                  // Determinant
let inv_A = inv(A)                  // Inverse matrix
let x = solve(A, b)                 // Solve Ax = b
let eigenvals = eigenvalues(A)      // Eigenvalues
```

### Machine Learning Utilities

```neuralscript
// Activation functions
let relu_val = relu(x)              // max(0, x)
let sigmoid_val = sigmoid(x)        // 1 / (1 + e^(-x))
let tanh_val = tanh(x)              // Hyperbolic tangent

// Loss functions
let mse_loss = mse(predictions, targets)
let ce_loss = cross_entropy(predictions, targets)

// Optimizers
let optimizer = Adam::new(learning_rate: 0.001)
optimizer.step(&mut parameters, &gradients)
```

## Error Handling

```neuralscript
// Result type for error handling
fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("Division by zero".to_string())
    } else {
        Ok(a / b)
    }
}

// Pattern matching on results
match divide(10.0, 2.0) {
    Ok(result) => println!("Result: {result}"),
    Err(error) => println!("Error: {error}")
}

// Option type for nullable values
fn find_root(f: Fn(f64) -> f64, guess: f64) -> Option<f64> {
    // Newton-Raphson implementation
    // Returns Some(root) or None if not found
}
```

## Best Practices

1. **Use descriptive names**: `velocity` instead of `v`
2. **Leverage type inference**: Let the compiler infer types when obvious
3. **Use units consistently**: Always specify units for physical quantities
4. **Document complex mathematics**: Add comments explaining formulas
5. **Break down complex expressions**: Use intermediate variables for clarity
6. **Prefer immutability**: Use `let` instead of `mut` when possible
7. **Handle errors explicitly**: Use `Result` and `Option` types appropriately

## Example: Complete Neural Network

```neuralscript
struct NeuralNetwork {
    layers: Vec<DenseLayer>,
    learning_rate: f64,
}

struct DenseLayer {
    weights: Matrix<f64>,
    biases: Vector<f64>,
}

impl NeuralNetwork {
    fn forward(&self, input: &Vector<f64>) -> Vector<f64> {
        let mut activation = input.clone()
        
        for layer in &self.layers {
            activation = layer.forward(&activation)
        }
        
        activation
    }
    
    fn train(&mut self, data: &[(Vector<f64>, Vector<f64>)], epochs: i32) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0
            
            for (input, target) in data {
                let prediction = self.forward(input)
                let loss = mse_loss(&prediction, target)
                
                // Automatic differentiation
                let gradients = ∇loss
                
                // Update weights
                for (layer, grad) in self.layers.iter_mut().zip(gradients) {
                    layer.weights -= self.learning_rate × grad.weights
                    layer.biases -= self.learning_rate × grad.biases
                }
                
                total_loss += loss
            }
            
            println!("Epoch {epoch}: Loss = {:.4}", total_loss / data.len() as f64)
        }
    }
}
```

This guide covers the essential features of NeuralScript. For more detailed examples and advanced topics, see the `examples/` directory and the full language specification.
