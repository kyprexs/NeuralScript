"""
NeuralScript Memory Management System

A production-grade garbage collector and memory optimizer designed for
scientific computing and machine learning workloads.

Author: xwest
"""

from .gc_core import GarbageCollector, GCStats, GCConfiguration, GCMode
from .object_model import (
    GCObject, ObjectHeader, ReferenceTracker, ObjectType, GCFlags,
    ReferenceTracer, ObjectRelocator, ScalarObject, VectorObject,
    ComplexObject, StringObject, ArrayObject
)
from .heap_manager import (
    HeapManager, MemoryPool, AllocationStrategy, HeapRegionType,
    FreeBlock, HeapRegion
)
from .generational_gc import (
    GenerationalGC, Generation, AgeTracker, CollectionTrigger,
    GCMetrics, RememberedSet, WriteBarrier
)
from .memory_profiler import (
    MemoryProfiler, AllocationProfile, ProfilingLevel, AllocationSite,
    AllocationRecord, MemorySnapshot
)
from .optimizer import (
    MemoryOptimizer, OptimizationHint, OptimizationStrategy,
    AllocationPattern, OptimizationRecommendation, PatternAnalyzer
)

__all__ = [
    # Core GC classes
    'GarbageCollector', 'GCStats', 'GCConfiguration', 'GCMode',
    
    # Object model
    'GCObject', 'ObjectHeader', 'ReferenceTracker', 'ObjectType', 'GCFlags',
    'ReferenceTracer', 'ObjectRelocator', 'ScalarObject', 'VectorObject',
    'ComplexObject', 'StringObject', 'ArrayObject',
    
    # Heap management
    'HeapManager', 'MemoryPool', 'AllocationStrategy', 'HeapRegionType',
    'FreeBlock', 'HeapRegion',
    
    # Generational GC
    'GenerationalGC', 'Generation', 'AgeTracker', 'CollectionTrigger',
    'GCMetrics', 'RememberedSet', 'WriteBarrier',
    
    # Memory profiling
    'MemoryProfiler', 'AllocationProfile', 'ProfilingLevel', 'AllocationSite',
    'AllocationRecord', 'MemorySnapshot',
    
    # Memory optimization
    'MemoryOptimizer', 'OptimizationHint', 'OptimizationStrategy',
    'AllocationPattern', 'OptimizationRecommendation', 'PatternAnalyzer'
]
