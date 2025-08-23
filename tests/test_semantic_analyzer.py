"""
Test suite for the NeuralScript semantic analyzer.

Tests cover:
- Symbol resolution and scoping
- Type checking and inference
- Error detection and reporting
- Advanced language features (generics, tensors, units)

Author: xwest
"""

import unittest
from typing import List, Optional
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from compiler.lexer.lexer import Lexer
from compiler.parser.parser import Parser
from compiler.analyzer.semantic_analyzer import SemanticAnalyzer
from compiler.analyzer.errors import SemanticError


class TestSemanticAnalyzer(unittest.TestCase):
    """Test cases for the semantic analyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.lexer = Lexer()
        self.parser = Parser()
        self.analyzer = SemanticAnalyzer()
    
    def _analyze_code(self, code: str):
        """Helper to analyze a code snippet."""
        tokens = self.lexer.tokenize(code)
        ast = self.parser.parse(tokens)
        result = self.analyzer.analyze(ast)
        return result
    
    def test_basic_variable_declaration(self):
        """Test basic variable declaration and type inference."""
        code = """
        fn main() {
            let x = 42;
            let y: i32 = 100;
            let z: f64 = 3.14;
        }
        """
        
        result = self._analyze_code(code)
        
        # Should have no errors
        self.assertFalse(result.has_errors(), f"Unexpected errors: {result.errors}")
        
        # Check that variables were properly added to symbol table
        main_symbol = result.symbol_table.lookup_symbol("main", None)
        self.assertIsNotNone(main_symbol)
    
    def test_function_definition_and_call(self):
        """Test function definition and call type checking."""
        code = """
        fn add(a: i32, b: i32) -> i32 {
            return a + b;
        }
        
        fn main() {
            let result = add(5, 10);
        }
        """
        
        result = self._analyze_code(code)
        
        # Should have no errors
        self.assertFalse(result.has_errors(), f"Unexpected errors: {result.errors}")
        
        # Check function symbols
        add_symbol = result.symbol_table.lookup_symbol("add", None)
        self.assertIsNotNone(add_symbol)
        self.assertEqual(add_symbol.kind.value, "function")
    
    def test_type_mismatch_error(self):
        """Test detection of type mismatch errors."""
        code = """
        fn main() {
            let x: i32 = 42;
            let y: f64 = x + 3.14; // Type mismatch should be caught
        }
        """
        
        result = self._analyze_code(code)
        
        # Should have type mismatch errors
        self.assertTrue(result.has_errors())
        
        # Check that we get the right kind of error
        type_errors = [e for e in result.errors if "mismatch" in e.message.lower()]
        self.assertTrue(len(type_errors) > 0, "Expected type mismatch error")
    
    def test_undefined_variable_error(self):
        """Test detection of undefined variable errors."""
        code = """
        fn main() {
            let x = y + 5; // y is undefined
        }
        """
        
        result = self._analyze_code(code)
        
        # Should have undefined variable error
        self.assertTrue(result.has_errors())
        
        # Check for undefined symbol error
        undefined_errors = [e for e in result.errors if "undefined" in e.message.lower() or "not found" in e.message.lower()]
        self.assertTrue(len(undefined_errors) > 0, "Expected undefined variable error")
    
    def test_scoping_rules(self):
        """Test proper scoping behavior."""
        code = """
        fn main() {
            let x = 10;
            {
                let y = 20;
                let z = x + y; // x should be accessible
            }
            // y should not be accessible here
            let w = y + 5;
        }
        """
        
        result = self._analyze_code(code)
        
        # Should have error about y being undefined in outer scope
        self.assertTrue(result.has_errors())
        undefined_errors = [e for e in result.errors if "y" in str(e)]
        self.assertTrue(len(undefined_errors) > 0, "Expected undefined variable error for y")
    
    def test_struct_definition_and_usage(self):
        """Test struct definition and field access."""
        code = """
        struct Point {
            x: f64,
            y: f64,
        }
        
        fn main() {
            let p = Point { x: 1.0, y: 2.0 };
        }
        """
        
        result = self._analyze_code(code)
        
        # Should successfully define struct
        self.assertFalse(result.has_errors(), f"Unexpected errors: {result.errors}")
        
        # Check struct symbol
        point_symbol = result.symbol_table.lookup_symbol("Point", None)
        self.assertIsNotNone(point_symbol)
    
    def test_generic_function(self):
        """Test generic function definition."""
        code = """
        fn identity<T>(x: T) -> T {
            return x;
        }
        
        fn main() {
            let a = identity(42);
            let b = identity(3.14);
        }
        """
        
        result = self._analyze_code(code)
        
        # Should handle generics without errors
        self.assertFalse(result.has_errors(), f"Unexpected errors: {result.errors}")
    
    def test_tensor_operations(self):
        """Test tensor literal and operations."""
        code = """
        fn main() {
            let matrix = [[1, 2, 3], [4, 5, 6]];
            let vector = [1.0, 2.0, 3.0];
        }
        """
        
        result = self._analyze_code(code)
        
        # Should handle tensor literals
        self.assertFalse(result.has_errors(), f"Unexpected errors: {result.errors}")
    
    def test_unit_literals(self):
        """Test unit literal support."""
        code = """
        fn main() {
            let distance = 100.0_m;
            let time = 5.0_s;
            let speed = distance / time; // Should be m/s
        }
        """
        
        result = self._analyze_code(code)
        
        # Should handle unit literals
        self.assertFalse(result.has_errors(), f"Unexpected errors: {result.errors}")
    
    def test_complex_numbers(self):
        """Test complex number support."""
        code = """
        fn main() {
            let z1 = 3 + 4i;
            let z2 = 2.5 - 1.5i;
            let sum = z1 + z2;
        }
        """
        
        result = self._analyze_code(code)
        
        # Should handle complex literals
        self.assertFalse(result.has_errors(), f"Unexpected errors: {result.errors}")
    
    def test_control_flow_type_checking(self):
        """Test type checking in control flow statements."""
        code = """
        fn main() {
            let x = 10;
            
            if x > 5 {
                let y = "hello";
            }
            
            while x > 0 {
                x = x - 1;
            }
            
            for i in [1, 2, 3, 4, 5] {
                let square = i * i;
            }
        }
        """
        
        result = self._analyze_code(code)
        
        # Should handle control flow correctly
        self.assertFalse(result.has_errors(), f"Unexpected errors: {result.errors}")
    
    def test_return_type_checking(self):
        """Test return type validation."""
        code = """
        fn get_number() -> i32 {
            return "not a number"; // Type mismatch
        }
        
        fn get_string() -> str {
            return 42; // Type mismatch
        }
        """
        
        result = self._analyze_code(code)
        
        # Should detect return type mismatches
        self.assertTrue(result.has_errors())
        
        # Should have multiple type mismatch errors
        type_errors = [e for e in result.errors if "mismatch" in e.message.lower()]
        self.assertTrue(len(type_errors) >= 2, f"Expected at least 2 type errors, got {len(type_errors)}")
    
    def test_mathematical_operators(self):
        """Test Unicode mathematical operator support."""
        code = """
        fn main() {
            let a = 5;
            let b = 3;
            
            let sum = a + b;
            let product = a × b;  // Unicode multiplication
            let dot_product = a ⋅ b;  // Dot product
            let cross_product = a ⊗ b;  // Cross product
            
            let is_equal = a ≡ b;  // Unicode equality
            let not_equal = a ≠ b;  // Unicode inequality
        }
        """
        
        result = self._analyze_code(code)
        
        # Should handle Unicode operators
        self.assertFalse(result.has_errors(), f"Unexpected errors: {result.errors}")
    
    def test_trait_definition(self):
        """Test trait definition and method signatures."""
        code = """
        trait Drawable {
            fn draw(self) -> str;
            fn area(self) -> f64;
        }
        
        struct Circle {
            radius: f64,
        }
        """
        
        result = self._analyze_code(code)
        
        # Should handle trait definitions
        self.assertFalse(result.has_errors(), f"Unexpected errors: {result.errors}")
        
        # Check trait symbol
        drawable_symbol = result.symbol_table.lookup_symbol("Drawable", None)
        self.assertIsNotNone(drawable_symbol)
    
    def test_multiple_errors(self):
        """Test that analyzer reports multiple errors."""
        code = """
        fn main() {
            let x: i32 = "wrong type";  // Error 1: Type mismatch
            let y = undefined_var;      // Error 2: Undefined variable
            let z: bool = 42;          // Error 3: Type mismatch
        }
        """
        
        result = self._analyze_code(code)
        
        # Should have multiple errors
        self.assertTrue(result.has_errors())
        self.assertGreaterEqual(len(result.errors), 2, f"Expected multiple errors, got {len(result.errors)}")
    
    def test_nested_scopes(self):
        """Test nested scope handling."""
        code = """
        fn outer() {
            let a = 1;
            
            fn inner() {
                let b = 2;
                let c = a + b; // a should be accessible
            }
            
            // b should not be accessible here
            let d = b + 1;
        }
        """
        
        result = self._analyze_code(code)
        
        # Should have error about b being undefined in outer scope
        self.assertTrue(result.has_errors())


class TestSemanticAnalyzerIntegration(unittest.TestCase):
    """Integration tests combining lexer, parser, and semantic analyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.lexer = Lexer()
        self.parser = Parser()
        self.analyzer = SemanticAnalyzer()
    
    def test_full_pipeline_simple_program(self):
        """Test full compilation pipeline on a simple program."""
        code = """
        fn factorial(n: i32) -> i32 {
            if n <= 1 {
                return 1;
            } else {
                return n * factorial(n - 1);
            }
        }
        
        fn main() {
            let result = factorial(5);
            return 0;
        }
        """
        
        # Tokenize
        tokens = self.lexer.tokenize(code)
        self.assertGreater(len(tokens), 0)
        
        # Parse
        ast = self.parser.parse(tokens)
        self.assertIsNotNone(ast)
        
        # Analyze
        result = self.analyzer.analyze(ast)
        self.assertIsNotNone(result)
        
        # Should compile successfully
        if result.has_errors():
            for error in result.errors:
                print(f"Error: {error.message} at {error.location}")
        
        self.assertFalse(result.has_errors(), "Simple factorial program should compile without errors")
    
    def test_full_pipeline_with_errors(self):
        """Test full compilation pipeline with intentional errors."""
        code = """
        fn main() {
            let x: i32 = "string"; // Type error
            let y = undefined_var;  // Undefined variable
            return "not an integer"; // Return type error
        }
        """
        
        # Tokenize
        tokens = self.lexer.tokenize(code)
        
        # Parse
        ast = self.parser.parse(tokens)
        
        # Analyze
        result = self.analyzer.analyze(ast)
        
        # Should have errors
        self.assertTrue(result.has_errors())
        self.assertGreaterEqual(len(result.errors), 2)


def run_semantic_analyzer_tests():
    """Run all semantic analyzer tests."""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestSemanticAnalyzer))
    suite.addTest(unittest.makeSuite(TestSemanticAnalyzerIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    # Run the tests
    print("Running NeuralScript Semantic Analyzer Tests...")
    print("=" * 60)
    
    result = run_semantic_analyzer_tests()
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed.")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
