# Contributing to NeuralScript

Thank you for your interest in contributing to NeuralScript! This document provides guidelines and information for contributors.

## Code of Conduct

This project adheres to a Code of Conduct that we expect all contributors to follow. Please read the full text to understand what actions will and will not be tolerated.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the existing issues to see if the problem has already been reported. When you create a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Describe the exact steps to reproduce the problem**
* **Provide specific examples to demonstrate the steps**
* **Describe the behavior you observed and what behavior you expected**
* **Include screenshots if applicable**
* **Include NeuralScript version and platform details**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title**
* **Provide a detailed description of the suggested enhancement**
* **Explain why this enhancement would be useful**
* **List any similar features in other languages**
* **Include mockup code examples if applicable**

### Development Process

1. **Fork the repository** and create your feature branch from `main`
2. **Write clear, commented code** following the project's style guidelines
3. **Add tests** for any new functionality
4. **Update documentation** as needed
5. **Run the full test suite** to ensure nothing is broken
6. **Create a pull request** with a clear description of changes

### Setting Up Development Environment

```bash
# Clone your fork
git clone https://github.com/yourusername/neuralscript.git
cd neuralscript

# Set up Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests to verify setup
python -m pytest tests/
python run_tests.py
```

### Code Style Guidelines

#### Python Code (Compiler Implementation)

* Follow PEP 8 style guidelines
* Use type hints for all function parameters and return values
* Write docstrings for all classes and functions
* Keep line length under 100 characters
* Use meaningful variable and function names

```python
def parse_expression(self, precedence: int = 0) -> Optional[Expression]:
    """Parse an expression with the given precedence level.
    
    Args:
        precedence: Minimum precedence level for operators to include
        
    Returns:
        Parsed expression AST node or None if parsing fails
    """
```

#### NeuralScript Code (Examples and Tests)

* Use descriptive variable names, especially for physical quantities
* Include units for all physical quantities
* Add comments explaining complex mathematical operations
* Use Unicode operators where they improve readability
* Keep functions focused on a single responsibility

```neuralscript
// Good: Clear physics with proper units
fn gravitational_force(m1: Kilogram, m2: Kilogram, distance: Meter) -> Newton {
    GRAVITATIONAL_CONSTANT × m1 × m2 / distance²
}

// Good: Mathematical notation that matches textbooks
fn quantum_harmonic_oscillator(n: i32, x: Meter) -> Complex<f64> {
    let ψ_n = hermite_polynomial(n, x) × gaussian_envelope(x)
    normalize(ψ_n)
}
```

### Testing Guidelines

#### Unit Tests

* Write tests for all new functionality
* Use descriptive test names that explain what is being tested
* Include both positive and negative test cases
* Test edge cases and error conditions

```python
def test_unit_literal_parsing_with_underscores():
    """Test that unit literals with underscores parse correctly."""
    lexer = Lexer("100.0_m", "<test>")
    tokens = lexer.tokenize()
    
    assert len(tokens) == 2  # UNIT_LITERAL + EOF
    assert tokens[0].type == TokenType.UNIT_LITERAL
    assert tokens[0].value == (100.0, 'm')
```

#### Integration Tests

* Test complete compilation pipelines
* Verify that examples in documentation actually work
* Test interactions between different language features

### Documentation Guidelines

* Update relevant documentation for any changes
* Include code examples that demonstrate usage
* Explain the reasoning behind design decisions
* Keep language clear and accessible to developers of all backgrounds

### Commit Message Guidelines

Use clear and meaningful commit messages:

```
Add support for complex number literals with underscores

- Update lexer regex patterns to handle underscores in complex numbers
- Modify parsing logic to strip underscores before conversion
- Add comprehensive test cases for various underscore patterns
- Update documentation with examples

Fixes #123
```

### Pull Request Process

1. **Update documentation** as needed
2. **Add or update tests** to cover your changes  
3. **Ensure all tests pass** locally
4. **Provide a clear description** of what your PR does and why
5. **Reference any related issues** using GitHub keywords

#### PR Description Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update

## Testing
- [ ] Added new tests for the feature/fix
- [ ] All existing tests pass
- [ ] Manual testing performed

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
```

## Architecture Overview

Understanding the project structure helps when contributing:

### Compiler Pipeline

1. **Lexer** (`compiler/lexer/`): Tokenizes source code
2. **Parser** (`compiler/parser/`): Builds Abstract Syntax Tree  
3. **Semantic Analyzer** (`compiler/analyzer/`): Type checking and analysis
4. **IR Generator** (`compiler/ir/`): Generates intermediate representation
5. **Backend** (`compiler/backend/`): LLVM code generation

### Key Design Principles

* **Mathematical Expressiveness**: Enable natural mathematical notation
* **Type Safety**: Catch errors at compile time, especially dimensional analysis
* **Performance**: Generate efficient machine code
* **Developer Experience**: Clear error messages and good tooling

### Areas for Contribution

#### High Priority

* **Standard Library**: Mathematical functions, linear algebra, ML utilities
* **Error Messages**: Improve clarity and provide suggestions
* **Optimization**: Better code generation and performance
* **Documentation**: More examples and tutorials

#### Medium Priority  

* **IDE Integration**: VS Code extension, better LSP features
* **Package Manager**: Dependency management and distribution
* **GPU Backend**: CUDA/OpenCL code generation
* **Parallel Computing**: Built-in parallelization primitives

#### Advanced

* **JIT Compilation**: Runtime optimization for hot code paths
* **Distributed Computing**: Multi-node computation primitives
* **Quantum Computing**: Quantum circuit simulation and optimization
* **Advanced Type System**: Dependent types, effect systems

## Getting Help

* **GitHub Issues**: For bugs and feature requests
* **Discussions**: For general questions and ideas
* **Email**: For security issues or private concerns

## Recognition

Contributors will be recognized in:

* The project's AUTHORS file
* Release notes for significant contributions  
* Special recognition for major features or improvements

## License

By contributing to NeuralScript, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make NeuralScript better for everyone!
