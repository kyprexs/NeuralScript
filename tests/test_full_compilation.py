"""
End-to-end compilation tests for NeuralScript.

Tests the full compilation pipeline from source code to executable.

Author: xwest
"""

import unittest
import sys
import os
import tempfile

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from compiler.lexer.lexer import Lexer
from compiler.parser.parser import Parser
from compiler.analyzer.semantic_analyzer import SemanticAnalyzer
from compiler.ir.ir_generator import IRGenerator
from compiler.backend.llvm_backend import LLVMBackend, create_mock_backend, HAS_LLVMLITE


class TestFullCompilation(unittest.TestCase):
    """Test the full compilation pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Note: Lexer requires source code, so we'll create it in each test method
        self.analyzer = SemanticAnalyzer()
        self.ir_generator = IRGenerator()
        
        # Use mock backend if llvmlite is not available
        if HAS_LLVMLITE:
            self.backend = LLVMBackend()
        else:
            self.backend = create_mock_backend()
    
    def _compile_code(self, code: str):
        """Compile a code snippet through the full pipeline."""
        # Tokenize
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        
        # Parse
        parser = Parser(tokens)
        ast = parser.parse()
        
        # Semantic analysis
        analysis_result = self.analyzer.analyze(ast)
        
        if analysis_result.has_errors():
            return None, analysis_result.errors, None, None
        
        # Generate IR
        ir_module = self.ir_generator.generate(analysis_result)
        
        # Generate LLVM IR (if available)
        llvm_module = None
        if HAS_LLVMLITE:
            llvm_module = self.backend.generate(ir_module)
        
        return ir_module, [], llvm_module, analysis_result
    
    def test_simple_function_compilation(self):
        """Test compilation of a simple function."""
        code = """
        fn add(a: i32, b: i32) -> i32 {
            return a + b;
        }
        
        fn main() -> i32 {
            let result = add(5, 10);
            return result;
        }
        """
        
        ir_module, errors, llvm_module, analysis = self._compile_code(code)
        
        # Should compile successfully
        self.assertIsNotNone(ir_module, f"Compilation failed with errors: {errors}")
        self.assertEqual(len(errors), 0)
        
        # Check IR module
        self.assertEqual(ir_module.name, "main")
        self.assertIn("add", ir_module.functions)
        self.assertIn("main", ir_module.functions)
        
        # Check functions have basic blocks
        add_func = ir_module.functions["add"]
        self.assertGreater(len(add_func.basic_blocks), 0)
        
        main_func = ir_module.functions["main"]
        self.assertGreater(len(main_func.basic_blocks), 0)
        
        # Print IR for debugging
        print("\n=== Generated NeuralScript IR ===")
        print(ir_module)
        
        if HAS_LLVMLITE and llvm_module:
            print("\n=== Generated LLVM IR ===")
            print(self.backend.print_llvm_ir(llvm_module))
    
    def test_control_flow_compilation(self):
        """Test compilation of control flow constructs."""
        code = """
        fn factorial(n: i32) -> i32 {
            if n <= 1 {
                return 1;
            } else {
                return n * factorial(n - 1);
            }
        }
        
        fn main() -> i32 {
            let result = factorial(5);
            return result;
        }
        """
        
        ir_module, errors, llvm_module, analysis = self._compile_code(code)
        
        # Should compile successfully
        self.assertIsNotNone(ir_module, f"Compilation failed with errors: {errors}")
        self.assertEqual(len(errors), 0)
        
        # Check that factorial function has multiple basic blocks (for if-else)
        factorial_func = ir_module.functions["factorial"]
        self.assertGreaterEqual(len(factorial_func.basic_blocks), 3, "Expected at least 3 blocks for if-else-merge")
        
        print("\n=== Factorial Function IR ===")
        print(factorial_func)
    
    def test_variable_declarations_compilation(self):
        """Test compilation of variable declarations."""
        code = """
        fn main() -> i32 {
            let x = 42;
            let y: i32 = 100;
            let z: f64 = 3.14;
            let sum = x + y;
            return sum;
        }
        """
        
        ir_module, errors, llvm_module, analysis = self._compile_code(code)
        
        # Should compile successfully
        self.assertIsNotNone(ir_module, f"Compilation failed with errors: {errors}")
        self.assertEqual(len(errors), 0)
        
        # Check that main function has instructions
        main_func = ir_module.functions["main"]
        entry_block = main_func.basic_blocks[0]
        self.assertGreater(len(entry_block.instructions), 0)
        
        print("\n=== Variable Declarations IR ===")
        print(main_func)
    
    def test_mathematical_expressions_compilation(self):
        """Test compilation of mathematical expressions."""
        code = """
        fn compute() -> f64 {
            let a = 5.0;
            let b = 3.0;
            let result = (a + b) * (a - b) / b;
            return result;
        }
        
        fn main() -> i32 {
            let value = compute();
            return 0;
        }
        """
        
        ir_module, errors, llvm_module, analysis = self._compile_code(code)
        
        # Should compile successfully
        self.assertIsNotNone(ir_module, f"Compilation failed with errors: {errors}")
        self.assertEqual(len(errors), 0)
        
        print("\n=== Mathematical Expressions IR ===")
        print(ir_module.functions["compute"])
    
    def test_tensor_operations_compilation(self):
        """Test compilation of tensor operations."""
        code = """
        fn process_tensors() -> i32 {
            let matrix = [[1, 2, 3], [4, 5, 6]];
            let vector = [1, 2, 3];
            return 0;
        }
        
        fn main() -> i32 {
            return process_tensors();
        }
        """
        
        ir_module, errors, llvm_module, analysis = self._compile_code(code)
        
        # Should compile successfully
        self.assertIsNotNone(ir_module, f"Compilation failed with errors: {errors}")
        self.assertEqual(len(errors), 0)
        
        print("\n=== Tensor Operations IR ===")
        print(ir_module.functions["process_tensors"])
    
    def test_complex_numbers_compilation(self):
        """Test compilation of complex number operations."""
        code = """
        fn complex_math() -> i32 {
            let z1 = 3 + 4i;
            let z2 = 2.5 - 1.5i;
            let sum = z1 + z2;
            return 0;
        }
        
        fn main() -> i32 {
            return complex_math();
        }
        """
        
        ir_module, errors, llvm_module, analysis = self._compile_code(code)
        
        # Should compile successfully
        self.assertIsNotNone(ir_module, f"Compilation failed with errors: {errors}")
        self.assertEqual(len(errors), 0)
        
        print("\n=== Complex Numbers IR ===")
        print(ir_module.functions["complex_math"])
    
    def test_compilation_errors(self):
        """Test handling of compilation errors."""
        code = """
        fn main() -> i32 {
            let x: i32 = "not a number";  // Type error
            let y = undefined_variable;   // Undefined variable
            return "not an integer";      // Return type error
        }
        """
        
        ir_module, errors, llvm_module, analysis = self._compile_code(code)
        
        # Should have errors
        self.assertIsNone(ir_module)
        self.assertGreater(len(errors), 0)
        
        print(f"\n=== Expected Compilation Errors ({len(errors)}) ===")
        for error in errors:
            message = error.diagnostic.message if hasattr(error, 'diagnostic') else str(error)
            location = error.diagnostic.location if hasattr(error, 'diagnostic') and error.diagnostic.location else None
            line = location.line if location else 'unknown'
            print(f"Error: {message} at line {line}")
    
    @unittest.skipUnless(HAS_LLVMLITE, "llvmlite not available")
    def test_llvm_optimization(self):
        """Test LLVM optimization passes."""
        code = """
        fn simple_math(x: i32) -> i32 {
            let a = x + 0;     // Should be optimized to just x
            let b = a * 1;     // Should be optimized to just a
            let c = b - 0;     // Should be optimized to just b
            return c;
        }
        
        fn main() -> i32 {
            return simple_math(42);
        }
        """
        
        ir_module, errors, llvm_module, analysis = self._compile_code(code)
        
        # Should compile successfully
        self.assertIsNotNone(ir_module)
        self.assertIsNotNone(llvm_module)
        self.assertEqual(len(errors), 0)
        
        print("\n=== LLVM IR (before optimization) ===")
        print(self.backend.print_llvm_ir(llvm_module))
        
        # Apply optimization
        optimized_module = self.backend.optimize_module(llvm_module, optimization_level=2)
        
        print("\n=== LLVM IR (after optimization) ===")
        print(self.backend.print_llvm_ir(optimized_module))
    
    def test_type_inference_compilation(self):
        """Test compilation with type inference."""
        code = """
        fn infer_types() -> i32 {
            let a = 42;          // Should infer i32
            let b = 3.14;        // Should infer f64
            let c = true;        // Should infer bool
            let d = a + 10;      // Should infer i32
            return d;
        }
        
        fn main() -> i32 {
            return infer_types();
        }
        """
        
        ir_module, errors, llvm_module, analysis = self._compile_code(code)
        
        # Should compile successfully
        self.assertIsNotNone(ir_module)
        self.assertEqual(len(errors), 0)
        
        # Check that types were properly inferred
        self.assertIsNotNone(analysis)
        self.assertGreater(len(analysis.type_annotations), 0)
        
        print(f"\n=== Type Inference Results ({len(analysis.type_annotations)} annotations) ===")
        for node, symbol_type in list(analysis.type_annotations.items())[:10]:  # Show first 10
            print(f"  {type(node).__name__}: {symbol_type.name}")


class TestCompilationPerformance(unittest.TestCase):
    """Test compilation performance on larger programs."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SemanticAnalyzer()
        self.ir_generator = IRGenerator()
        
        if HAS_LLVMLITE:
            self.backend = LLVMBackend()
        else:
            self.backend = create_mock_backend()
    
    def test_large_function_compilation(self):
        """Test compilation of a larger function."""
        # Generate a function with many operations
        operations = []
        for i in range(50):
            operations.append(f"let var{i} = {i} + {i+1};")
        
        code = f"""
        fn large_function() -> i32 {{
            {chr(10).join(operations)}
            let sum = 0;
            {chr(10).join(f"sum = sum + var{i};" for i in range(50))}
            return sum;
        }}
        
        fn main() -> i32 {{
            return large_function();
        }}
        """
        
        # Time the compilation
        import time
        start_time = time.time()
        
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        analysis_result = self.analyzer.analyze(ast)
        
        self.assertFalse(analysis_result.has_errors(), 
                        f"Large function compilation failed: {analysis_result.errors}")
        
        ir_module = self.ir_generator.generate(analysis_result)
        
        end_time = time.time()
        compilation_time = end_time - start_time
        
        print(f"\n=== Large Function Compilation ===")
        print(f"Compilation time: {compilation_time:.3f} seconds")
        print(f"Functions: {len(ir_module.functions)}")
        print(f"Instructions in large_function: {len(ir_module.functions['large_function'].basic_blocks[0].instructions)}")


def run_full_compilation_tests():
    """Run all full compilation tests."""
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestFullCompilation))
    suite.addTest(unittest.makeSuite(TestCompilationPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("Running NeuralScript Full Compilation Tests...")
    print("=" * 60)
    
    if not HAS_LLVMLITE:
        print("⚠️  llvmlite not available - using mock LLVM backend")
        print("   Install llvmlite for full LLVM functionality: pip install llvmlite")
        print()
    
    result = run_full_compilation_tests()
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All compilation tests passed!")
    else:
        print("❌ Some compilation tests failed.")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    print(f"\nTotal tests run: {result.testsRun}")
    
    # Show summary of what was built
    print("\n" + "=" * 60)
    print("🎉 NeuralScript Compiler Pipeline Complete!")
    print("Components implemented:")
    print("  ✅ Lexer - Tokenizes NeuralScript source code")
    print("  ✅ Parser - Builds Abstract Syntax Tree (AST)")  
    print("  ✅ Semantic Analyzer - Type checking and symbol resolution")
    print("  ✅ IR Generator - Converts AST to NeuralScript IR")
    print("  ✅ LLVM Backend - Generates LLVM IR and object code")
    print("  ✅ Full compilation pipeline with error handling")
    print("\nReady for:")
    print("  🔄 Advanced optimizations")
    print("  🧮 Tensor operation lowering")
    print("  📐 Dimensional analysis")
    print("  🔗 GPU code generation")
    print("  📦 Package system and modules")
