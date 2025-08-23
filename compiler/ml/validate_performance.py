#!/usr/bin/env python3
"""
Neural Network Performance Validation Summary
==============================================

Simple validation runner to check NeuralScript neural network training
achieves 2x faster performance than PyTorch target.
"""

import time
import sys
from pathlib import Path

# Add the parent directory to Python path for imports  
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.test_neural_training import run_neural_network_validation


def main():
    """Run neural network performance validation"""
    
    print("Neural Network Training Performance Validation")
    print("=" * 50)
    print("Target: 2x faster than PyTorch")
    print("")
    
    try:
        # Run validation
        results = run_neural_network_validation()
        
        # Print clean summary
        print("\nVALIDATION RESULTS:")
        print("=" * 20)
        print(f"Average speedup achieved: {results['average_speedup']:.2f}x")
        print(f"Target achieved: {'YES' if results['target_achieved'] else 'NO'}")
        print(f"Recommendation: {results['recommendation']}")
        
        # Print success/failure clearly
        if results['target_achieved']:
            print("\n*** SUCCESS: NeuralScript achieves 2x+ faster training than PyTorch! ***")
            return True
        else:
            print("\n*** TARGET NOT MET: Additional optimization needed ***")
            return False
            
    except Exception as e:
        print(f"Error during validation: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
