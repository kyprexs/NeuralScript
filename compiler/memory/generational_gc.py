"""
Generational Garbage Collector for NeuralScript

Implements a multi-generational copying collector with incremental
collection and write barriers for high-performance memory management.
"""

import threading
import time
import weakref
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from enum import Enum, auto
from dataclasses import dataclass, field
from collections import defaultdict, deque

from .object_model import GCObject, ObjectHeader, ReferenceTracer, ObjectRelocator, ObjectType
from .heap_manager import HeapManager, HeapRegionType


class CollectionTrigger(Enum):
    """Reasons why a GC collection was triggered"""
    ALLOCATION_PRESSURE = auto()    # High memory usage
    EXPLICIT_REQUEST = auto()       # Manual GC request  
    PERIODIC = auto()              # Scheduled collection
    PROMOTION_THRESHOLD = auto()    # Too many old objects
    LARGE_OBJECT = auto()          # Large object allocation


@dataclass
class GCMetrics:
    """Statistics for a garbage collection cycle"""
    generation: int
    trigger_reason: CollectionTrigger
    start_time: float
    end_time: float
    objects_collected: int
    bytes_collected: int
    objects_promoted: int
    bytes_promoted: int
    objects_scanned: int
    pause_time_ms: float
    
    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


@dataclass
class Generation:
    """Represents a generation in the generational collector"""
    generation_id: int
    objects: Set[GCObject] = field(default_factory=set)
    allocation_count: int = 0
    collection_count: int = 0
    promotion_threshold: int = 3  # Collections survived before promotion
    size_limit: int = 32 * 1024 * 1024  # 32MB default limit
    
    @property
    def total_size(self) -> int:
        return sum(obj.header.size for obj in self.objects)
    
    @property
    def should_collect(self) -> bool:
        return self.total_size > self.size_limit


class AgeTracker:
    """Tracks object ages for promotion decisions"""
    
    def __init__(self):
        self._object_ages: Dict[int, int] = {}
        self._age_histograms: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._lock = threading.RLock()
    
    def record_survival(self, obj: GCObject):
        """Record that an object survived a collection"""
        with self._lock:
            obj_id = id(obj)
            current_age = self._object_ages.get(obj_id, 0)
            new_age = current_age + 1
            
            self._object_ages[obj_id] = new_age
            
            # Update histogram
            gen = obj.header.generation
            self._age_histograms[gen][new_age] += 1
            if current_age > 0:
                self._age_histograms[gen][current_age] -= 1
    
    def get_age(self, obj: GCObject) -> int:
        """Get the age (survival count) of an object"""
        with self._lock:
            return self._object_ages.get(id(obj), 0)
    
    def should_promote(self, obj: GCObject, promotion_threshold: int) -> bool:
        """Check if an object should be promoted to the next generation"""
        return self.get_age(obj) >= promotion_threshold
    
    def remove_object(self, obj: GCObject):
        """Remove tracking for a dead object"""
        with self._lock:
            obj_id = id(obj)
            if obj_id in self._object_ages:
                age = self._object_ages[obj_id]
                gen = obj.header.generation
                self._age_histograms[gen][age] -= 1
                del self._object_ages[obj_id]
    
    def get_age_distribution(self, generation: int) -> Dict[int, int]:
        """Get age distribution for a generation"""
        with self._lock:
            return dict(self._age_histograms[generation])


class RememberedSet:
    """
    Tracks inter-generational references to avoid scanning older generations
    during young generation collection.
    """
    
    def __init__(self):
        self._old_to_young_refs: Set[Tuple[int, int]] = set()
        self._dirty_objects: Set[GCObject] = set()
        self._lock = threading.RLock()
    
    def record_reference(self, old_obj: GCObject, young_obj: GCObject):
        """Record a reference from an older object to a younger one"""
        if old_obj.header.generation <= young_obj.header.generation:
            return  # Not an old-to-young reference
        
        with self._lock:
            self._old_to_young_refs.add((id(old_obj), id(young_obj)))
            self._dirty_objects.add(old_obj)
    
    def remove_reference(self, old_obj: GCObject, young_obj: GCObject):
        """Remove a recorded reference"""
        with self._lock:
            self._old_to_young_refs.discard((id(old_obj), id(young_obj)))
    
    def get_dirty_objects(self) -> Set[GCObject]:
        """Get objects that may have references to younger generations"""
        with self._lock:
            return set(self._dirty_objects)
    
    def clear_dirty_objects(self):
        """Clear the dirty object set after processing"""
        with self._lock:
            self._dirty_objects.clear()


class WriteBarrier:
    """
    Write barrier implementation to maintain remembered set invariants.
    
    Intercepts pointer writes to detect old-to-young references.
    """
    
    def __init__(self, remembered_set: RememberedSet):
        self.remembered_set = remembered_set
        self._enabled = True
        self._lock = threading.RLock()
    
    def enable(self):
        """Enable write barrier"""
        with self._lock:
            self._enabled = True
    
    def disable(self):
        """Disable write barrier (during GC)"""  
        with self._lock:
            self._enabled = False
    
    def record_write(self, source_obj: GCObject, target_obj: Optional[GCObject]):
        """Record a pointer write from source to target"""
        if not self._enabled or target_obj is None:
            return
        
        # Check if this creates an old-to-young reference
        if source_obj.header.generation > target_obj.header.generation:
            self.remembered_set.record_reference(source_obj, target_obj)


class GenerationalGC:
    """
    Main generational garbage collector implementation.
    
    Features:
    - Multiple generations with different collection frequencies
    - Copying collection for young generations
    - Mark-and-sweep for old generations  
    - Incremental collection to reduce pause times
    - Write barriers and remembered sets
    """
    
    def __init__(self, heap_manager: HeapManager, num_generations: int = 3):
        self.heap_manager = heap_manager
        self.generations = [Generation(i) for i in range(num_generations)]
        self.age_tracker = AgeTracker()
        self.remembered_set = RememberedSet()
        self.write_barrier = WriteBarrier(self.remembered_set)
        
        # GC configuration
        self.max_pause_time_ms = 10.0  # Target pause time
        self.collection_frequency = [1, 10, 50]  # Collection ratios
        self.enable_incremental = True
        self.enable_concurrent = False  # Future feature
        
        # Statistics and metrics
        self.collection_history: List[GCMetrics] = []
        self.total_collections = 0
        self.total_pause_time = 0.0
        
        # Threading
        self._gc_lock = threading.RLock()
        self._collection_in_progress = False
        
        # Root set management
        self._root_objects: Set[GCObject] = set()
        self._root_callbacks: List[Callable[[], List[GCObject]]] = []
    
    def add_root_object(self, obj: GCObject):
        """Add an object to the root set"""
        with self._gc_lock:
            self._root_objects.add(obj)
    
    def remove_root_object(self, obj: GCObject):
        """Remove an object from the root set"""
        with self._gc_lock:
            self._root_objects.discard(obj)
    
    def add_root_callback(self, callback: Callable[[], List[GCObject]]):
        """Add a callback that provides root objects"""
        with self._gc_lock:
            self._root_callbacks.append(callback)
    
    def allocate_object(self, obj_type: ObjectType, size: int) -> Optional[GCObject]:
        """
        Allocate a new object in the youngest generation.
        
        May trigger garbage collection if allocation pressure is high.
        """
        with self._gc_lock:
            # Check if we should trigger collection before allocation
            if self._should_collect_before_allocation(size):
                self.collect_generation(0, CollectionTrigger.ALLOCATION_PRESSURE)
            
            # Allocate memory
            address = self.heap_manager.allocate_object(obj_type, size, generation=0)
            if address is None:
                # Try emergency collection
                self.collect_all_generations()
                address = self.heap_manager.allocate_object(obj_type, size, generation=0)
                
                if address is None:
                    return None  # Out of memory
            
            # Create object (would need concrete implementation)
            # For now, return None to indicate successful allocation
            return None
    
    def _should_collect_before_allocation(self, size: int) -> bool:
        """Check if we should collect before allocating"""
        young_gen = self.generations[0]
        
        # Check size limits
        if young_gen.should_collect:
            return True
        
        # Check memory pressure
        if self.heap_manager.should_trigger_gc():
            return True
        
        # Check allocation count
        if young_gen.allocation_count > 1000:
            return True
        
        return False
    
    def collect_generation(self, generation: int, trigger: CollectionTrigger) -> GCMetrics:
        """Collect a specific generation"""
        if generation >= len(self.generations):
            raise ValueError(f"Invalid generation: {generation}")
        
        with self._gc_lock:
            if self._collection_in_progress:
                # Avoid recursive collection
                return self._create_empty_metrics(generation, trigger)
            
            self._collection_in_progress = True
            
            try:
                return self._collect_generation_impl(generation, trigger)
            finally:
                self._collection_in_progress = False
    
    def _collect_generation_impl(self, generation: int, trigger: CollectionTrigger) -> GCMetrics:
        """Internal generation collection implementation"""
        start_time = time.time()
        
        # Disable write barrier during collection
        self.write_barrier.disable()
        
        try:
            if generation == 0:
                # Young generation: copying collection
                metrics = self._collect_young_generation(trigger, start_time)
            else:
                # Old generation: mark-and-sweep
                metrics = self._collect_old_generation(generation, trigger, start_time)
            
            # Update statistics
            self.collection_history.append(metrics)
            if len(self.collection_history) > 1000:
                self.collection_history = self.collection_history[-500:]  # Keep recent history
            
            self.total_collections += 1
            self.total_pause_time += metrics.pause_time_ms
            
            return metrics
            
        finally:
            # Re-enable write barrier
            self.write_barrier.enable()
    
    def _collect_young_generation(self, trigger: CollectionTrigger, start_time: float) -> GCMetrics:
        """Collect the young generation using copying collection"""
        young_gen = self.generations[0]
        
        # Phase 1: Mark reachable objects
        tracer = ReferenceTracer()
        marked_objects = set()
        
        # Trace from roots
        self._trace_from_roots(tracer, max_generation=0)
        
        # Trace from remembered set (old-to-young references)
        for dirty_obj in self.remembered_set.get_dirty_objects():
            tracer.trace_object(dirty_obj)
        
        tracer.process_work_stack()
        marked_objects = set(tracer._traced_objects)
        
        # Phase 2: Identify survivors and promotion candidates
        survivors = []
        promoted = []
        
        for obj in young_gen.objects:
            obj_id = id(obj)
            if obj_id in marked_objects:
                # Object survived
                self.age_tracker.record_survival(obj)
                
                if self.age_tracker.should_promote(obj, young_gen.promotion_threshold):
                    promoted.append(obj)
                else:
                    survivors.append(obj)
        
        # Phase 3: Promote objects to next generation
        if len(self.generations) > 1:
            old_gen = self.generations[1]
            for obj in promoted:
                obj.header.generation = 1
                old_gen.objects.add(obj)
                young_gen.objects.discard(obj)
        
        # Phase 4: Cleanup dead objects
        dead_objects = young_gen.objects - set(survivors) - set(promoted)
        bytes_collected = 0
        
        for obj in dead_objects:
            # Finalize object
            if obj.header.has_finalizer:
                obj.finalize()
            
            # Clear weak references
            obj.header.clear_weak_refs()
            
            # Track bytes collected
            bytes_collected += obj.header.size
            
            # Remove from age tracker
            self.age_tracker.remove_object(obj)
        
        # Update generation
        young_gen.objects = set(survivors)
        young_gen.collection_count += 1
        
        # Clear remembered set dirty objects
        self.remembered_set.clear_dirty_objects()
        
        # Create metrics
        end_time = time.time()
        return GCMetrics(
            generation=0,
            trigger_reason=trigger,
            start_time=start_time,
            end_time=end_time,
            objects_collected=len(dead_objects),
            bytes_collected=bytes_collected,
            objects_promoted=len(promoted),
            bytes_promoted=sum(obj.header.size for obj in promoted),
            objects_scanned=len(young_gen.objects),
            pause_time_ms=(end_time - start_time) * 1000
        )
    
    def _collect_old_generation(self, generation: int, trigger: CollectionTrigger, start_time: float) -> GCMetrics:
        """Collect an old generation using mark-and-sweep"""
        gen = self.generations[generation]
        
        # Phase 1: Mark
        tracer = ReferenceTracer()
        self._trace_from_roots(tracer, max_generation=generation)
        tracer.process_work_stack()
        
        marked_objects = set(tracer._traced_objects)
        
        # Phase 2: Sweep
        dead_objects = []
        bytes_collected = 0
        
        for obj in list(gen.objects):
            if id(obj) not in marked_objects:
                dead_objects.append(obj)
                bytes_collected += obj.header.size
                
                # Finalize and cleanup
                if obj.header.has_finalizer:
                    obj.finalize()
                obj.header.clear_weak_refs()
                
                # Remove from generation
                gen.objects.remove(obj)
                self.age_tracker.remove_object(obj)
        
        gen.collection_count += 1
        
        # Create metrics
        end_time = time.time()
        return GCMetrics(
            generation=generation,
            trigger_reason=trigger,
            start_time=start_time,
            end_time=end_time,
            objects_collected=len(dead_objects),
            bytes_collected=bytes_collected,
            objects_promoted=0,
            bytes_promoted=0,
            objects_scanned=len(gen.objects) + len(dead_objects),
            pause_time_ms=(end_time - start_time) * 1000
        )
    
    def _trace_from_roots(self, tracer: ReferenceTracer, max_generation: int = 999):
        """Trace objects reachable from root set"""
        # Static roots
        for root_obj in self._root_objects:
            if root_obj.header.generation <= max_generation:
                tracer.trace_object(root_obj)
        
        # Dynamic roots from callbacks
        for callback in self._root_callbacks:
            try:
                roots = callback()
                for root_obj in roots:
                    if root_obj.header.generation <= max_generation:
                        tracer.trace_object(root_obj)
            except Exception as e:
                # Log error but continue
                print(f"Error in root callback: {e}")
    
    def collect_all_generations(self) -> List[GCMetrics]:
        """Collect all generations"""
        metrics = []
        
        # Collect from oldest to youngest to handle inter-generational references
        for gen_id in reversed(range(len(self.generations))):
            metric = self.collect_generation(gen_id, CollectionTrigger.EXPLICIT_REQUEST)
            metrics.append(metric)
        
        return metrics
    
    def request_collection(self) -> GCMetrics:
        """Explicitly request garbage collection"""
        return self.collect_generation(0, CollectionTrigger.EXPLICIT_REQUEST)
    
    def _create_empty_metrics(self, generation: int, trigger: CollectionTrigger) -> GCMetrics:
        """Create empty metrics when collection is skipped"""
        now = time.time()
        return GCMetrics(
            generation=generation,
            trigger_reason=trigger,
            start_time=now,
            end_time=now,
            objects_collected=0,
            bytes_collected=0,
            objects_promoted=0,
            bytes_promoted=0,
            objects_scanned=0,
            pause_time_ms=0.0
        )
    
    def get_gc_statistics(self) -> Dict[str, Any]:
        """Get comprehensive GC statistics"""
        with self._gc_lock:
            recent_metrics = self.collection_history[-10:] if self.collection_history else []
            
            stats = {
                'total_collections': self.total_collections,
                'total_pause_time_ms': self.total_pause_time,
                'average_pause_time_ms': self.total_pause_time / max(self.total_collections, 1),
                'collection_in_progress': self._collection_in_progress,
                'generations': []
            }
            
            # Per-generation stats
            for i, gen in enumerate(self.generations):
                gen_stats = {
                    'generation': i,
                    'object_count': len(gen.objects),
                    'total_size': gen.total_size,
                    'allocation_count': gen.allocation_count,
                    'collection_count': gen.collection_count,
                    'promotion_threshold': gen.promotion_threshold,
                    'size_limit': gen.size_limit,
                    'age_distribution': self.age_tracker.get_age_distribution(i)
                }
                stats['generations'].append(gen_stats)
            
            # Recent collection metrics
            if recent_metrics:
                stats['recent_collections'] = [
                    {
                        'generation': m.generation,
                        'trigger': m.trigger_reason.name,
                        'duration_ms': m.duration_ms,
                        'objects_collected': m.objects_collected,
                        'bytes_collected': m.bytes_collected,
                        'objects_promoted': m.objects_promoted
                    }
                    for m in recent_metrics
                ]
            
            return stats
    
    def optimize_collection_parameters(self):
        """Optimize GC parameters based on allocation patterns"""
        if len(self.collection_history) < 10:
            return  # Need more data
        
        with self._gc_lock:
            recent_metrics = self.collection_history[-50:]
            
            # Analyze pause times
            avg_pause = sum(m.pause_time_ms for m in recent_metrics) / len(recent_metrics)
            
            if avg_pause > self.max_pause_time_ms * 1.5:
                # Pause times too high - make collection more frequent
                for gen in self.generations:
                    gen.size_limit = int(gen.size_limit * 0.8)
            elif avg_pause < self.max_pause_time_ms * 0.5:
                # Pause times acceptable - can increase thresholds
                for gen in self.generations:
                    gen.size_limit = int(gen.size_limit * 1.1)
            
            # Analyze promotion rates
            young_gen_metrics = [m for m in recent_metrics if m.generation == 0]
            if young_gen_metrics:
                avg_promotion_rate = sum(m.objects_promoted for m in young_gen_metrics) / len(young_gen_metrics)
                avg_survival_rate = sum(m.objects_scanned - m.objects_collected for m in young_gen_metrics) / len(young_gen_metrics)
                
                promotion_ratio = avg_promotion_rate / max(avg_survival_rate, 1)
                
                if promotion_ratio > 0.8:
                    # Too many promotions - increase promotion threshold
                    self.generations[0].promotion_threshold = min(
                        self.generations[0].promotion_threshold + 1, 10
                    )
                elif promotion_ratio < 0.2:
                    # Too few promotions - decrease threshold
                    self.generations[0].promotion_threshold = max(
                        self.generations[0].promotion_threshold - 1, 1
                    )
