#!/usr/bin/env python3
"""
Neural Network Training Validation Runner
=========================================

Main runner script to validate NeuralScript's neural network training
performance against the 2x faster than PyTorch target.

Usage:
    python run_neural_validation.py [options]

Options:
    --quick         Run only quick validation tests
    --comprehensive Run comprehensive benchmark comparison
    --integration   Run optimization system integration tests
    --unittest      Run formal unit tests
    --all           Run all validation types (default)
    --report        Generate detailed report file
    --json          Output results in JSON format
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.test_neural_training import (
    NeuralNetworkTrainingValidator, run_neural_network_validation,
    TestNeuralNetworkTraining
)
import unittest


class ValidationRunner:
    """Main runner for neural network training validation"""
    
    def __init__(self, output_json: bool = False, generate_report: bool = False):
        self.output_json = output_json
        self.generate_report = generate_report
        self.validator = NeuralNetworkTrainingValidator()
        self.run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    def run_quick_validation(self) -> Dict[str, Any]:
        """Run quick validation only"""
        print("🚀 Running Quick Neural Network Validation\n")
        
        result = self.validator.validate_quick_performance()
        
        return {
            'validation_type': 'quick',
            'result': result,
            'success': result.performance_target_met,
            'timestamp': self.run_timestamp
        }
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive benchmark validation"""
        print("🚀 Running Comprehensive Neural Network Validation\n")
        
        result = self.validator.validate_comprehensive_benchmark()
        
        return {
            'validation_type': 'comprehensive',
            'result': result,
            'success': result.performance_target_met,
            'timestamp': self.run_timestamp
        }
    
    def run_integration_validation(self) -> Dict[str, Any]:
        """Run optimization integration validation"""
        print("🚀 Running Optimization Integration Validation\n")
        
        result = self.validator.validate_optimization_integration()
        
        return {
            'validation_type': 'integration',
            'result': result,
            'success': result.performance_target_met,
            'timestamp': self.run_timestamp
        }
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run formal unit tests"""
        print("🚀 Running Neural Network Unit Tests\n")
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestNeuralNetworkTraining)
        
        # Run tests with proper runner
        runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
        test_result = runner.run(suite)
        
        return {
            'validation_type': 'unittest',
            'tests_run': test_result.testsRun,
            'failures': len(test_result.failures),
            'errors': len(test_result.errors),
            'success': test_result.wasSuccessful(),
            'timestamp': self.run_timestamp
        }
    
    def run_all_validation(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        print("🚀 Running Complete Neural Network Training Validation\n")
        
        # Run the comprehensive validation function
        full_results = run_neural_network_validation()
        
        # Also run unit tests
        unit_test_results = self.run_unit_tests()
        
        return {
            'validation_type': 'complete',
            'performance_results': full_results,
            'unit_test_results': unit_test_results,
            'overall_success': (
                full_results['target_achieved'] and 
                unit_test_results['success']
            ),
            'timestamp': self.run_timestamp
        }
    
    def save_report(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save validation report to file"""
        
        if not filename:
            filename = f"neural_validation_report_{self.run_timestamp}.txt"
        
        report_content = []
        
        # Header
        report_content.extend([
            "Neural Network Training Performance Validation Report",
            "=" * 55,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Validation Type: {results.get('validation_type', 'unknown')}",
            ""
        ])
        
        # Results summary
        if 'performance_results' in results:
            perf = results['performance_results']
            report_content.extend([
                "📊 Performance Validation Results:",
                f"   Target achieved: {'✅ YES' if perf['target_achieved'] else '❌ NO'}",
                f"   Average speedup: {perf['average_speedup']:.2f}x",
                f"   Recommendation: {perf['recommendation']}",
                ""
            ])
        
        if 'unit_test_results' in results:
            unit = results['unit_test_results']
            report_content.extend([
                "🧪 Unit Test Results:",
                f"   Tests run: {unit['tests_run']}",
                f"   Failures: {unit['failures']}",
                f"   Errors: {unit['errors']}",
                f"   Success: {'✅ YES' if unit['success'] else '❌ NO'}",
                ""
            ])
        
        # Overall verdict
        overall_success = results.get('overall_success', False)
        report_content.extend([
            "🎯 Overall Verdict:",
            f"   {'🎉 SUCCESS' if overall_success else '⚠️ NEEDS WORK'}: Neural network training {'meets' if overall_success else 'does not meet'} performance requirements",
            ""
        ])
        
        # System information
        report_content.extend([
            "🔧 System Information:",
            f"   Python version: {sys.version.split()[0]}",
            f"   Working directory: {os.getcwd()}",
            f"   Optimizations available: {self.validator.optimizations_available}",
        ])
        
        # Write report
        report_text = "\n".join(report_content)
        
        with open(filename, 'w') as f:
            f.write(report_text)
        
        print(f"\n📄 Report saved to: {filename}")
        return filename
    
    def output_results(self, results: Dict[str, Any]):
        """Output results in requested format"""
        
        if self.output_json:
            # Convert any non-serializable objects for JSON output
            json_results = self._make_json_serializable(results)
            print(json.dumps(json_results, indent=2))
        
        if self.generate_report:
            self.save_report(results)
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        
        if hasattr(obj, '__dict__'):
            return {k: self._make_json_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        else:
            return obj


def main():
    """Main entry point for validation runner"""
    
    parser = argparse.ArgumentParser(
        description="Neural Network Training Performance Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_neural_validation.py                    # Run all validations
    python run_neural_validation.py --quick            # Quick test only
    python run_neural_validation.py --comprehensive    # Comprehensive benchmark
    python run_neural_validation.py --report --json    # Generate report and JSON output
        """
    )
    
    # Validation type options
    parser.add_argument('--quick', action='store_true', 
                      help='Run only quick validation tests')
    parser.add_argument('--comprehensive', action='store_true',
                      help='Run comprehensive benchmark comparison')
    parser.add_argument('--integration', action='store_true',
                      help='Run optimization system integration tests')
    parser.add_argument('--unittest', action='store_true',
                      help='Run formal unit tests')
    parser.add_argument('--all', action='store_true',
                      help='Run all validation types (default)')
    
    # Output options
    parser.add_argument('--report', action='store_true',
                      help='Generate detailed report file')
    parser.add_argument('--json', action='store_true',
                      help='Output results in JSON format')
    
    args = parser.parse_args()
    
    # Default to all if no specific type selected
    if not any([args.quick, args.comprehensive, args.integration, args.unittest]):
        args.all = True
    
    # Create runner
    runner = ValidationRunner(
        output_json=args.json,
        generate_report=args.report
    )
    
    try:
        # Execute requested validation type
        if args.all:
            results = runner.run_all_validation()
        elif args.quick:
            results = runner.run_quick_validation()
        elif args.comprehensive:
            results = runner.run_comprehensive_validation()
        elif args.integration:
            results = runner.run_integration_validation()
        elif args.unittest:
            results = runner.run_unit_tests()
        
        # Output results
        runner.output_results(results)
        
        # Exit with appropriate code
        success = results.get('overall_success', results.get('success', False))
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Validation failed with error: {e}")
        if args.json:
            error_result = {
                'validation_type': 'error',
                'error': str(e),
                'success': False,
                'timestamp': time.strftime("%Y%m%d_%H%M%S")
            }
            print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
