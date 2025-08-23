"""
NeuralScript Backend Package.

Contains code generation backends for different targets.

Author: xwest
"""

from .llvm_backend import LLVMBackend, create_mock_backend

__all__ = ['LLVMBackend', 'create_mock_backend']
