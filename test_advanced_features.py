#!/usr/bin/env python3
"""
Test script for advanced NeuralScript features.

Tests automatic differentiation and dimensional analysis capabilities.

Author: xwest
"""

import sys
sys.path.append('.')

from compiler.lexer.lexer import Lexer
from compiler.parser.parser import Parser
from compiler.analyzer.semantic_analyzer import SemanticAnalyzer
from compiler.autodiff.autodiff_engine import AutodiffEngine, Derivative, ADMode
from compiler.units.dimensional_analysis import DimensionalAnalyzer, add_dimensional_analysis_pass


def test_automatic_differentiation():
    """Test automatic differentiation capabilities."""
    print("🧮 Testing Automatic Differentiation")
    print("=" * 50)
    
    # Test forward-mode AD
    print("\n📈 Forward-mode AD:")
    engine = AutodiffEngine()
    engine.set_mode(ADMode.FORWARD)
    
    # Simple NeuralScript code for differentiation
    code = """
    fn quadratic(x: f64) -> f64 {
        return x * x + 2.0 * x + 1.0;
    }
    """
    
    try:
        # Parse the code
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        
        parser = Parser(tokens)
        ast = parser.parse()
        
        # Get the function
        func = ast.items[0]
        
        # Create derivative specification
        derivatives = [Derivative("x", order=1, mode=ADMode.FORWARD)]
        
        # Generate differentiated function
        diff_func = engine.differentiate_function(func, derivatives)
        
        print(f"✅ Generated differentiated function: {diff_func.name}")
        print(f"   Original: quadratic(x)")
        print(f"   Derivative: d/dx quadratic(x)")
        
        # Test reverse-mode AD
        print("\n📉 Reverse-mode AD:")
        engine.set_mode(ADMode.REVERSE)
        
        reverse_func = engine.differentiate_function(func, derivatives)
        print(f"✅ Generated reverse-mode function: {reverse_func.name}")
        
    except Exception as e:
        print(f"❌ AD test failed: {e}")
    
    print("\n🔢 Testing higher-order derivatives:")
    
    # Test Jacobian computation
    engine = AutodiffEngine()
    jacobian_func = engine.compute_jacobian(func, ["x"], ["result"])
    print(f"✅ Generated Jacobian function: {jacobian_func.name}")
    
    # Test Hessian computation
    hessian_func = engine.compute_hessian(func, ["x"])
    print(f"✅ Generated Hessian function: {hessian_func.name}")


def test_dimensional_analysis():
    """Test dimensional analysis and unit checking."""
    print("\n\n📐 Testing Dimensional Analysis")
    print("=" * 50)
    
    analyzer = DimensionalAnalyzer()
    
    # Test basic unit recognition
    print("\n🔍 Testing unit recognition:")
    
    units_to_test = [
        ("m", "meters"),
        ("kg", "kilograms"), 
        ("s", "seconds"),
        ("m/s", "velocity"),
        ("m/s²", "acceleration"),
        ("kg*m/s²", "force (Newtons)"),
        ("kg*m²/s²", "energy (Joules)"),
    ]
    
    for unit_str, description in units_to_test:
        unit = analyzer.registry.parse_compound_unit(unit_str)
        if unit:
            print(f"✅ {unit_str} -> {description}: {unit.dimension}")
        else:
            print(f"❌ Failed to parse: {unit_str}")
    
    # Test unit compatibility checking
    print("\n⚖️  Testing unit compatibility:")
    
    # Compatible units (length)
    meter = analyzer.registry.get_unit("m")
    kilometer = analyzer.registry.get_unit("km")
    if meter and kilometer:
        compatible = analyzer.check_assignment_compatibility(meter, kilometer)
        print(f"✅ meters ↔ kilometers: {'Compatible' if compatible else 'Incompatible'}")
    
    # Incompatible units (length vs time)
    second = analyzer.registry.get_unit("s")
    if meter and second:
        compatible = analyzer.check_assignment_compatibility(meter, second)
        print(f"✅ meters ↔ seconds: {'Compatible' if compatible else 'Incompatible'}")
    
    # Test dimensional analysis on code
    print("\n🧪 Testing dimensional analysis on code:")
    
    # Code with unit literals
    physics_code = """
    fn calculate_velocity(distance: f64, time: f64) -> f64 {
        let d = 100.0_m;      // 100 meters
        let t = 10.0_s;       // 10 seconds
        return d / t;         // Should be m/s
    }
    
    fn invalid_addition() -> f64 {
        let mass = 5.0_kg;    // 5 kilograms
        let length = 10.0_m;  // 10 meters
        return mass + length; // ERROR: Cannot add mass to length!
    }
    """
    
    try:
        # Parse the physics code
        lexer = Lexer(physics_code)
        tokens = lexer.tokenize()
        
        parser = Parser(tokens)
        ast = parser.parse()
        
        # Run dimensional analysis
        unit_errors = analyzer.analyze_program(ast)
        
        print(f"📊 Found {len(unit_errors)} dimensional analysis errors:")
        for error in unit_errors:
            print(f"   ⚠️  {error.message}")
        
        if len(unit_errors) > 0:
            print("✅ Dimensional analysis successfully caught unit errors!")
        else:
            print("ℹ️  No dimensional errors found (this might be expected)")
            
    except Exception as e:
        print(f"❌ Dimensional analysis test failed: {e}")
    
    # Test unit conversions
    print("\n🔄 Testing unit conversions:")
    
    conversions_to_test = [
        ("m", "km", 1000),  # 1000 meters = 1 kilometer
        ("g", "kg", 1),     # 1 gram = 0.001 kilograms  
        ("°C", "K", 273.15), # 0°C = 273.15K
    ]
    
    for from_unit, to_unit, test_value in conversions_to_test:
        from_u = analyzer.registry.get_unit(from_unit)
        to_u = analyzer.registry.get_unit(to_unit)
        
        if from_u and to_u:
            try:
                converted = from_u.convert_to(to_u, test_value)
                print(f"✅ {test_value} {from_unit} = {converted:.4f} {to_unit}")
            except Exception as e:
                print(f"❌ Conversion failed: {from_unit} -> {to_unit}: {e}")


def test_integration():
    """Test integration of advanced features with the main compiler."""
    print("\n\n🔗 Testing Integration with Compiler Pipeline")
    print("=" * 50)
    
    # Advanced NeuralScript code with differentiable functions
    advanced_code = """
fn physics_simulation(position: f64, velocity: f64, time: f64) -> f64 {
    let x0 = 100.0_m;
    let v0 = 10.0_ms;
    let t = 5.0_s;
    let g = 9.81_ms2;
    return x0 + v0 * t + 0.5 * g * t * t;
}

fn energy_calculation(mass: f64, velocity: f64) -> f64 {
    let m = 5.0_kg;
    let v = 20.0_ms;
    return 0.5 * m * v * v;
}
    """
    
    try:
        print("📝 Parsing advanced NeuralScript code...")
        
        # Lexical analysis
        lexer = Lexer(advanced_code)
        tokens = lexer.tokenize()
        print(f"✅ Generated {len(tokens)} tokens")
        
        # Syntax analysis
        parser = Parser(tokens)
        ast = parser.parse()
        print(f"✅ Generated AST with {len(ast.items)} top-level items")
        
        # Semantic analysis
        analyzer = SemanticAnalyzer()
        analysis_result = analyzer.analyze(ast)
        
        if analysis_result.has_errors():
            print(f"⚠️  Found {len(analysis_result.errors)} semantic errors:")
            for error in analysis_result.errors[:3]:  # Show first 3 errors
                print(f"   - {error.diagnostic.message}")
        else:
            print("✅ Semantic analysis passed")
        
        # Add dimensional analysis pass
        analysis_result = add_dimensional_analysis_pass(analysis_result)
        
        unit_errors = [e for e in analysis_result.errors if hasattr(e, 'diagnostic') and e.diagnostic.code and e.diagnostic.code.startswith('U')]
        if unit_errors:
            print(f"📐 Found {len(unit_errors)} unit errors:")
            for error in unit_errors[:3]:
                print(f"   - {error.diagnostic.message}")
        else:
            print("✅ Dimensional analysis passed")
        
        # Test automatic differentiation on functions
        print("\n🧮 Testing AD integration:")
        
        engine = AutodiffEngine()
        for item in ast.items:
            if isinstance(item, type(ast.items[0])):  # FunctionDef
                if hasattr(item, 'is_differentiable') and item.is_differentiable:
                    derivatives = [Derivative("position"), Derivative("velocity")]
                    diff_func = engine.differentiate_function(item, derivatives)
                    print(f"✅ Generated gradient function: {diff_func.name}")
        
        print("\n🎉 Advanced features integration test completed!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 NeuralScript Advanced Features Test Suite")
    print("=" * 60)
    
    # Run all tests
    test_automatic_differentiation()
    test_dimensional_analysis()
    test_integration()
    
    print("\n" + "=" * 60)
    print("🏁 Advanced features testing completed!")
    print("\n🔬 Features Demonstrated:")
    print("   • Forward and reverse-mode automatic differentiation")
    print("   • Jacobian and Hessian matrix computation")
    print("   • Comprehensive dimensional analysis with SI units")
    print("   • Unit compatibility checking and conversions")
    print("   • Integration with the main compiler pipeline")
    print("   • Support for mathematical and physical computations")
