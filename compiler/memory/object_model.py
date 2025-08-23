"""
GC Object Model for NeuralScript

Defines the object layout, headers, and reference tracking system
used by the garbage collector.
"""

import ctypes
import threading
import weakref
from typing import Dict, List, Optional, Set, Any, Callable
from enum import Enum, IntFlag
from dataclasses import dataclass
from abc import ABC, abstractmethod


class ObjectType(Enum):
    """Types of objects managed by the GC"""
    SCALAR = "scalar"
    VECTOR = "vector" 
    MATRIX = "matrix"
    TENSOR = "tensor"
    STRING = "string"
    FUNCTION = "function"
    CLOSURE = "closure"
    STRUCT = "struct"
    ARRAY = "array"
    COMPLEX = "complex"
    UNIT_VALUE = "unit_value"


class GCFlags(IntFlag):
    """Flags stored in object headers"""
    MARKED = 1 << 0          # Mark bit for mark-and-sweep
    REMEMBERED = 1 << 1      # In remembered set (generational)
    PINNED = 1 << 2         # Cannot be moved by copying GC
    FINALIZER = 1 << 3      # Has finalizer that needs calling
    WEAK_REFS = 1 << 4      # Has weak references pointing to it
    IMMUTABLE = 1 << 5      # Immutable object (optimization hint)
    LARGE_OBJECT = 1 << 6   # Allocated in large object heap
    CONCURRENT_SAFE = 1 << 7 # Safe for concurrent collection


@dataclass
class ObjectHeader:
    """
    Object header stored at the beginning of every GC-managed object.
    
    Layout (64-bit):
    - size: 32 bits (object size in bytes)
    - type_id: 16 bits (object type identifier) 
    - generation: 8 bits (generation number, 0-255)
    - flags: 8 bits (GC flags)
    """
    size: int                    # Size in bytes
    type_id: ObjectType         # Object type
    generation: int             # Generation number (0 = youngest)
    flags: GCFlags              # GC control flags
    allocation_id: int          # Unique allocation identifier
    thread_id: int              # Thread that allocated this object
    
    def __post_init__(self):
        self._lock = threading.RLock()
        self._weak_refs: Set[weakref.ref] = set()
    
    @property
    def is_marked(self) -> bool:
        return bool(self.flags & GCFlags.MARKED)
    
    @is_marked.setter 
    def is_marked(self, value: bool):
        with self._lock:
            if value:
                self.flags |= GCFlags.MARKED
            else:
                self.flags &= ~GCFlags.MARKED
    
    @property
    def is_pinned(self) -> bool:
        return bool(self.flags & GCFlags.PINNED)
    
    @property
    def has_finalizer(self) -> bool:
        return bool(self.flags & GCFlags.FINALIZER)
    
    def add_weak_ref(self, ref: weakref.ref):
        """Add a weak reference to this object"""
        with self._lock:
            self._weak_refs.add(ref)
            self.flags |= GCFlags.WEAK_REFS
    
    def remove_weak_ref(self, ref: weakref.ref):
        """Remove a weak reference"""
        with self._lock:
            self._weak_refs.discard(ref)
            if not self._weak_refs:
                self.flags &= ~GCFlags.WEAK_REFS
    
    def clear_weak_refs(self):
        """Clear all weak references (called during finalization)"""
        with self._lock:
            for ref in self._weak_refs:
                ref.clear()
            self._weak_refs.clear()
            self.flags &= ~GCFlags.WEAK_REFS


class GCObject(ABC):
    """
    Base class for all garbage-collected objects in NeuralScript.
    
    Provides the interface for GC operations like marking, tracing,
    and finalization.
    """
    
    def __init__(self, obj_type: ObjectType, size: int):
        self._header = ObjectHeader(
            size=size,
            type_id=obj_type,
            generation=0,  # Start in youngest generation
            flags=GCFlags(0),
            allocation_id=self._get_next_allocation_id(),
            thread_id=threading.get_ident()
        )
        self._finalizer: Optional[Callable] = None
    
    @classmethod
    def _get_next_allocation_id(cls) -> int:
        """Get next unique allocation ID"""
        if not hasattr(cls, '_allocation_counter'):
            cls._allocation_counter = 0
        cls._allocation_counter += 1
        return cls._allocation_counter
    
    @property
    def header(self) -> ObjectHeader:
        return self._header
    
    @abstractmethod
    def trace_references(self, tracer: 'ReferenceTracer'):
        """
        Trace all references from this object to other GC objects.
        Called during mark phase of GC.
        """
        pass
    
    @abstractmethod
    def relocate_references(self, relocator: 'ObjectRelocator'):
        """
        Update references when objects are moved by copying GC.
        """
        pass
    
    def set_finalizer(self, finalizer: Callable):
        """Set a finalizer to be called before object destruction"""
        self._finalizer = finalizer
        self._header.flags |= GCFlags.FINALIZER
    
    def finalize(self):
        """Call the finalizer if one exists"""
        if self._finalizer:
            try:
                self._finalizer()
            except Exception as e:
                # Log finalizer errors but don't propagate
                print(f"Finalizer error for {self}: {e}")
            finally:
                self._finalizer = None
                self._header.flags &= ~GCFlags.FINALIZER
    
    def pin(self):
        """Pin this object so it won't be moved by copying GC"""
        self._header.flags |= GCFlags.PINNED
    
    def unpin(self):
        """Unpin this object"""
        self._header.flags &= ~GCFlags.PINNED
    
    def __sizeof__(self) -> int:
        return self._header.size


class ReferenceTracer:
    """
    Interface for tracing object references during GC mark phase.
    
    Objects implement trace_references() and call tracer methods
    to report their references to other GC objects.
    """
    
    def __init__(self):
        self._traced_objects: Set[int] = set()
        self._work_stack: List[GCObject] = []
    
    def trace_object(self, obj: Optional[GCObject]):
        """Trace a reference to another GC object"""
        if obj is None:
            return
            
        obj_id = id(obj)
        if obj_id not in self._traced_objects:
            self._traced_objects.add(obj_id)
            self._work_stack.append(obj)
            obj.header.is_marked = True
    
    def trace_array(self, objects: List[Optional[GCObject]]):
        """Trace an array of object references"""
        for obj in objects:
            self.trace_object(obj)
    
    def process_work_stack(self):
        """Process all objects in the work stack"""
        while self._work_stack:
            obj = self._work_stack.pop()
            obj.trace_references(self)


class ObjectRelocator:
    """
    Handles updating object references when objects are moved
    by the copying garbage collector.
    """
    
    def __init__(self):
        self._forwarding_map: Dict[int, GCObject] = {}
    
    def add_forwarding(self, old_addr: int, new_obj: GCObject):
        """Record that an object has moved"""
        self._forwarding_map[old_addr] = new_obj
    
    def relocate_reference(self, obj_ref: GCObject) -> GCObject:
        """Update a reference to point to the new location"""
        old_addr = id(obj_ref)
        return self._forwarding_map.get(old_addr, obj_ref)


class ReferenceTracker:
    """
    Tracks object references for debugging and profiling.
    
    Can generate reference graphs and detect potential memory leaks.
    """
    
    def __init__(self):
        self._references: Dict[int, Set[int]] = {}
        self._reverse_refs: Dict[int, Set[int]] = {}
        self._object_info: Dict[int, ObjectHeader] = {}
        self._lock = threading.RLock()
    
    def add_reference(self, from_obj: GCObject, to_obj: GCObject):
        """Record a reference from one object to another"""
        with self._lock:
            from_id, to_id = id(from_obj), id(to_obj)
            
            # Forward reference
            if from_id not in self._references:
                self._references[from_id] = set()
            self._references[from_id].add(to_id)
            
            # Reverse reference  
            if to_id not in self._reverse_refs:
                self._reverse_refs[to_id] = set()
            self._reverse_refs[to_id].add(from_id)
            
            # Store object info
            self._object_info[from_id] = from_obj.header
            self._object_info[to_id] = to_obj.header
    
    def remove_reference(self, from_obj: GCObject, to_obj: GCObject):
        """Remove a reference"""
        with self._lock:
            from_id, to_id = id(from_obj), id(to_obj)
            
            if from_id in self._references:
                self._references[from_id].discard(to_id)
                
            if to_id in self._reverse_refs:
                self._reverse_refs[to_id].discard(from_id)
    
    def get_reachable_objects(self, root_obj: GCObject) -> Set[int]:
        """Find all objects reachable from a root object"""
        reachable = set()
        work_stack = [id(root_obj)]
        
        with self._lock:
            while work_stack:
                obj_id = work_stack.pop()
                if obj_id in reachable:
                    continue
                    
                reachable.add(obj_id)
                
                # Add all referenced objects to work stack
                if obj_id in self._references:
                    for ref_id in self._references[obj_id]:
                        if ref_id not in reachable:
                            work_stack.append(ref_id)
        
        return reachable
    
    def find_cycles(self) -> List[List[int]]:
        """Find reference cycles in the object graph"""
        cycles = []
        visited = set()
        
        def dfs(obj_id: int, path: List[int], path_set: Set[int]) -> bool:
            if obj_id in path_set:
                # Found a cycle
                cycle_start = path.index(obj_id)
                cycle = path[cycle_start:] + [obj_id]
                cycles.append(cycle)
                return True
                
            if obj_id in visited:
                return False
                
            visited.add(obj_id)
            path.append(obj_id)
            path_set.add(obj_id)
            
            # Explore references
            for ref_id in self._references.get(obj_id, set()):
                dfs(ref_id, path, path_set)
            
            path.pop()
            path_set.remove(obj_id)
            return False
        
        with self._lock:
            for obj_id in self._references:
                if obj_id not in visited:
                    dfs(obj_id, [], set())
        
        return cycles
    
    def get_memory_usage_by_type(self) -> Dict[ObjectType, int]:
        """Get memory usage breakdown by object type"""
        usage = {}
        
        with self._lock:
            for obj_id, header in self._object_info.items():
                obj_type = header.type_id
                if obj_type not in usage:
                    usage[obj_type] = 0
                usage[obj_type] += header.size
        
        return usage


# Concrete GC object implementations for NeuralScript types

class ScalarObject(GCObject):
    """A scalar numeric value"""
    
    def __init__(self, value: float):
        super().__init__(ObjectType.SCALAR, 8)  # 8 bytes for float64
        self.value = value
    
    def trace_references(self, tracer: ReferenceTracer):
        # Scalars have no references to other GC objects
        pass
    
    def relocate_references(self, relocator: ObjectRelocator):
        # Scalars have no references to relocate
        pass


class VectorObject(GCObject):
    """A vector of numeric values"""
    
    def __init__(self, values: List[float]):
        size = 8 * len(values) + 32  # Values + overhead
        super().__init__(ObjectType.VECTOR, size)
        self.values = values
        self.length = len(values)
    
    def trace_references(self, tracer: ReferenceTracer):
        # Vectors of primitives have no GC references
        pass
    
    def relocate_references(self, relocator: ObjectRelocator):
        pass


class ComplexObject(GCObject):
    """A complex number with real and imaginary parts"""
    
    def __init__(self, real: float, imag: float):
        super().__init__(ObjectType.COMPLEX, 16)  # 2 * 8 bytes
        self.real = real
        self.imag = imag
    
    def trace_references(self, tracer: ReferenceTracer):
        pass
    
    def relocate_references(self, relocator: ObjectRelocator):
        pass


class StringObject(GCObject):
    """A Unicode string object"""
    
    def __init__(self, text: str):
        # UTF-8 encoded size + overhead
        size = len(text.encode('utf-8')) + 32
        super().__init__(ObjectType.STRING, size)
        self.text = text
        self.length = len(text)
    
    def trace_references(self, tracer: ReferenceTracer):
        pass
    
    def relocate_references(self, relocator: ObjectRelocator):
        pass


class ArrayObject(GCObject):
    """An array of GC objects"""
    
    def __init__(self, elements: List[Optional[GCObject]]):
        size = len(elements) * 8 + 32  # Pointers + overhead
        super().__init__(ObjectType.ARRAY, size)
        self.elements = elements
        self.length = len(elements)
    
    def trace_references(self, tracer: ReferenceTracer):
        # Trace all elements
        tracer.trace_array(self.elements)
    
    def relocate_references(self, relocator: ObjectRelocator):
        # Update all element references
        for i, elem in enumerate(self.elements):
            if elem is not None:
                self.elements[i] = relocator.relocate_reference(elem)
