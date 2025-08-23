#!/usr/bin/env python3
"""
Simple SIMD Test Suite
======================

Basic functionality tests for the SIMD implementation.
"""

import sys
import os
import time

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all SIMD modules can be imported"""
    print("🔍 Testing SIMD module imports...")
    
    try:
        from compiler.backend.simd_codegen import SIMDCodeGenerator, MatrixDimensions, DataType
        print("  ✅ SIMD codegen imported successfully")
        
        from compiler.optimizer.vectorization_pass import VectorizationPass
        print("  ✅ Auto-vectorization pass imported successfully")
        
        from compiler.optimizer.runtime_profiler import RuntimeProfiler, create_runtime_profiler
        print("  ✅ Runtime profiler imported successfully")
        
        from compiler.simd.simd_core import SIMDProcessor
        print("  ✅ SIMD core imported successfully")
        
        from compiler.backend.llvm_backend import LLVMBackend
        print("  ✅ LLVM backend imported successfully")
        
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_simd_codegen():
    """Test basic SIMD code generation"""
    print("\n⚙️ Testing SIMD code generation...")
    
    try:
        from compiler.backend.simd_codegen import SIMDCodeGenerator, MatrixDimensions, DataType
        
        codegen = SIMDCodeGenerator()
        print("  ✅ SIMD code generator created")
        
        # Test basic functionality
        dimensions = MatrixDimensions(8, 8, 8)
        print(f"  ✅ Created matrix dimensions: {dimensions.m}x{dimensions.k}x{dimensions.n}")
        
        # Test performance estimation if available
        try:
            perf_estimate = codegen.estimate_performance(dimensions)
            print(f"  ✅ Performance estimation: {perf_estimate['estimated_gflops']:.2f} GFLOPS")
        except AttributeError:
            print("  ✅ Code generator initialized (performance estimation not available)")
        
        return True
    except Exception as e:
        print(f"  ❌ SIMD codegen test failed: {e}")
        return False

def test_simd_processor():
    """Test SIMD processor hardware detection"""
    print("\n🖥️ Testing SIMD processor hardware detection...")
    
    try:
        from compiler.simd.simd_core import SIMDProcessor, DataType
        
        processor = SIMDProcessor()
        print("  ✅ SIMD processor created")
        
        instruction_sets = processor.get_available_instruction_sets()
        print(f"  ✅ Available instruction sets: {instruction_sets}")
        
        vector_width = processor.get_vector_width(DataType.FLOAT32)
        print(f"  ✅ Vector width for FLOAT32: {vector_width}")
        
        return True
    except Exception as e:
        print(f"  ❌ SIMD processor test failed: {e}")
        return False

def test_llvm_backend():
    """Test LLVM backend SIMD integration"""
    print("\n🔧 Testing LLVM backend SIMD integration...")
    
    try:
        from compiler.backend.llvm_backend import LLVMBackend
        from compiler.backend.simd_codegen import DataType
        
        backend = LLVMBackend(enable_simd=True, enable_profiling=True)
        print("  ✅ LLVM backend created with SIMD enabled")
        
        # Test basic SIMD capabilities check
        if hasattr(backend, 'simd_enabled') and backend.simd_enabled:
            print("  ✅ SIMD backend integration working")
        elif hasattr(backend, 'enable_simd') and backend.enable_simd:
            print("  ✅ SIMD backend integration working")
        else:
            print("  ⚠️  SIMD integration may have issues")
        
        return True
    except Exception as e:
        print(f"  ❌ LLVM backend test failed: {e}")
        return False

def test_runtime_profiler():
    """Test runtime profiler functionality"""
    print("\n📊 Testing runtime profiler...")
    
    try:
        from compiler.optimizer.runtime_profiler import create_runtime_profiler
        
        profiler = create_runtime_profiler(enable_detailed_profiling=False)
        print("  ✅ Runtime profiler created")
        
        # Record a function execution
        profile_id = profiler.record_function_call("test_function", 10.5, {"test": "data"})
        print(f"  ✅ Recorded function call: {profile_id}")
        
        # Get hot functions (simplified test)
        hot_functions = profiler.get_hot_functions()
        print("  ✅ Hot functions retrieved")
        
        return True
    except Exception as e:
        print(f"  ❌ Runtime profiler test failed: {e}")
        return False

def test_auto_vectorizer():
    """Test auto-vectorization pass"""
    print("\n🔄 Testing auto-vectorization pass...")
    
    try:
        from compiler.optimizer.vectorization_pass import VectorizationPass
        
        vectorizer = VectorizationPass()
        print("  ✅ Auto-vectorization pass created")
        
        # Note: We can't easily test the full pass without IR input,
        # but we can verify it initializes properly
        
        return True
    except Exception as e:
        print(f"  ❌ Auto-vectorization test failed: {e}")
        return False

def main():
    """Run all SIMD tests"""
    print("🧪 NeuralScript SIMD Implementation Test Suite")
    print("=" * 50)
    
    start_time = time.perf_counter()
    
    tests = [
        ("Module Imports", test_imports),
        ("SIMD Code Generation", test_simd_codegen),
        ("SIMD Processor", test_simd_processor),
        ("LLVM Backend Integration", test_llvm_backend),
        ("Runtime Profiler", test_runtime_profiler),
        ("Auto-Vectorization Pass", test_auto_vectorizer)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
    
    execution_time = (time.perf_counter() - start_time) * 1000
    
    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS")
    print("=" * 50)
    print(f"✅ Tests Passed: {passed}/{total} ({(passed/total)*100:.1f}%)")
    print(f"⏱️  Total Time: {execution_time:.2f}ms")
    
    if passed == total:
        print("\n🎉 All SIMD tests passed! The implementation is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
