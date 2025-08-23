"""
Core Garbage Collector for NeuralScript

Main interface and orchestration layer for the complete garbage
collection system including memory optimization and profiling.
"""

import threading
import time
import gc
import psutil
import os
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from enum import Enum, auto
from dataclasses import dataclass
from collections import deque

from .object_model import GCObject, ObjectType, ReferenceTracker
from .heap_manager import HeapManager, HeapRegionType
from .generational_gc import GenerationalGC, GCMetrics, CollectionTrigger
from .memory_profiler import MemoryProfiler, AllocationProfile
from .optimizer import MemoryOptimizer, OptimizationHint


class GCMode(Enum):
    """Garbage collection modes"""
    AUTOMATIC = auto()      # Automatic collection based on heuristics
    MANUAL = auto()         # Manual collection only
    INCREMENTAL = auto()    # Incremental collection to minimize pauses
    CONCURRENT = auto()     # Concurrent collection (future)
    ADAPTIVE = auto()       # Adaptive collection based on workload


@dataclass
class GCConfiguration:
    """Configuration parameters for the garbage collector"""
    
    # Generation parameters
    num_generations: int = 3
    young_gen_size_mb: int = 32
    old_gen_size_mb: int = 128
    large_object_threshold_kb: int = 32
    
    # Collection parameters
    mode: GCMode = GCMode.AUTOMATIC
    max_pause_time_ms: float = 10.0
    collection_ratio: List[int] = None  # [1, 10, 50] default
    enable_incremental: bool = True
    enable_write_barriers: bool = True
    
    # Memory optimization
    enable_compaction: bool = True
    compaction_threshold: float = 0.3  # Fragmentation threshold
    optimize_allocation_patterns: bool = True
    
    # Profiling and debugging
    enable_profiling: bool = False
    track_allocation_sites: bool = False
    debug_mode: bool = False
    
    def __post_init__(self):
        if self.collection_ratio is None:
            self.collection_ratio = [1, 10, 50]


@dataclass
class GCStats:
    """Comprehensive GC statistics"""
    
    # Collection statistics
    total_collections: int = 0
    collections_by_generation: List[int] = None
    total_pause_time_ms: float = 0.0
    average_pause_time_ms: float = 0.0
    max_pause_time_ms: float = 0.0
    
    # Memory statistics
    heap_size_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0
    fragmentation_ratio: float = 0.0
    
    # Object statistics
    total_objects: int = 0
    objects_by_type: Dict[str, int] = None
    objects_by_generation: List[int] = None
    
    # Performance metrics
    allocation_rate_mb_per_sec: float = 0.0
    collection_efficiency: float = 0.0  # Bytes freed per ms of pause time
    memory_pressure: float = 0.0
    
    def __post_init__(self):
        if self.collections_by_generation is None:
            self.collections_by_generation = [0, 0, 0]
        if self.objects_by_type is None:
            self.objects_by_type = {}
        if self.objects_by_generation is None:
            self.objects_by_generation = [0, 0, 0]


class GarbageCollector:
    """
    Main garbage collector class for NeuralScript.
    
    Orchestrates heap management, generational collection, memory optimization,
    and profiling to provide high-performance memory management for scientific
    computing workloads.
    """
    
    def __init__(self, config: Optional[GCConfiguration] = None):
        self.config = config or GCConfiguration()
        
        # Core components
        self.heap_manager = HeapManager()
        self.generational_gc = GenerationalGC(self.heap_manager, self.config.num_generations)
        self.reference_tracker = ReferenceTracker()
        self.memory_profiler = MemoryProfiler() if self.config.enable_profiling else None
        self.memory_optimizer = MemoryOptimizer(self.heap_manager)
        
        # Configuration
        self._configure_generational_gc()
        
        # Statistics and monitoring
        self.stats = GCStats()
        self._collection_history: deque = deque(maxlen=1000)
        self._last_stats_update = time.time()
        
        # Threading and synchronization
        self._gc_lock = threading.RLock()
        self._background_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Performance monitoring
        self._system_monitor = SystemMonitor()
        self._allocation_tracker = AllocationTracker()
        
        # Start background optimization thread
        if self.config.mode == GCMode.AUTOMATIC:
            self._start_background_thread()
    
    def _configure_generational_gc(self):
        """Configure the generational GC with our settings"""
        self.generational_gc.max_pause_time_ms = self.config.max_pause_time_ms
        self.generational_gc.enable_incremental = self.config.enable_incremental
        
        # Set generation size limits
        if len(self.generational_gc.generations) > 0:
            self.generational_gc.generations[0].size_limit = self.config.young_gen_size_mb * 1024 * 1024
        if len(self.generational_gc.generations) > 1:
            self.generational_gc.generations[1].size_limit = self.config.old_gen_size_mb * 1024 * 1024
    
    def _start_background_thread(self):
        """Start background thread for automatic GC and optimization"""
        if self._background_thread is not None:
            return
        
        def background_worker():
            while not self._shutdown_event.wait(timeout=1.0):
                try:
                    self._background_maintenance()
                except Exception as e:
                    print(f"GC background thread error: {e}")
        
        self._background_thread = threading.Thread(target=background_worker, daemon=True)
        self._background_thread.start()
    
    def _background_maintenance(self):
        """Periodic maintenance tasks run in background thread"""
        with self._gc_lock:
            # Check if GC is needed
            if self._should_collect():
                self.collect()
            
            # Run memory optimization
            if self.config.optimize_allocation_patterns:
                self.memory_optimizer.optimize()
            
            # Update statistics
            self._update_statistics()
            
            # Compact heap if needed
            if self.config.enable_compaction and self._should_compact():
                self._compact_fragmented_regions()
    
    def allocate(self, obj_type: ObjectType, size: int) -> Optional[int]:
        """
        Allocate memory for a new object.
        
        Returns the address of the allocated memory or None if allocation fails.
        """
        with self._gc_lock:
            # Track allocation
            self._allocation_tracker.record_allocation(obj_type, size)
            
            # Profile allocation if enabled
            if self.memory_profiler:
                self.memory_profiler.record_allocation(obj_type, size)
            
            # Delegate to generational GC
            result = self.generational_gc.allocate_object(obj_type, size)
            
            # Update statistics
            self._update_allocation_stats(obj_type, size, result is not None)
            
            return result
    
    def collect(self, generation: Optional[int] = None, 
                explicit: bool = False) -> List[GCMetrics]:
        """
        Trigger garbage collection.
        
        Args:
            generation: Specific generation to collect, or None for automatic
            explicit: Whether this is an explicit request
            
        Returns:
            List of collection metrics for each generation collected
        """
        with self._gc_lock:
            trigger = CollectionTrigger.EXPLICIT_REQUEST if explicit else CollectionTrigger.ALLOCATION_PRESSURE
            
            if generation is not None:
                # Collect specific generation
                metrics = [self.generational_gc.collect_generation(generation, trigger)]
            else:
                # Automatic collection strategy
                metrics = self._determine_collection_strategy(trigger)
            
            # Update history and statistics
            for metric in metrics:
                self._collection_history.append(metric)
                self._update_collection_stats(metric)
            
            return metrics
    
    def _determine_collection_strategy(self, trigger: CollectionTrigger) -> List[GCMetrics]:
        """Determine which generations to collect"""
        metrics = []
        
        # Check memory pressure
        pressure = self.heap_manager.get_memory_pressure()
        
        if pressure > 0.9:
            # High pressure - collect all generations
            metrics = self.generational_gc.collect_all_generations()
        elif pressure > 0.7:
            # Medium pressure - collect young and middle generations
            metrics.append(self.generational_gc.collect_generation(0, trigger))
            if len(self.generational_gc.generations) > 1:
                metrics.append(self.generational_gc.collect_generation(1, trigger))
        else:
            # Low pressure - collect young generation only
            metrics.append(self.generational_gc.collect_generation(0, trigger))
        
        return metrics
    
    def _should_collect(self) -> bool:
        """Determine if automatic collection should be triggered"""
        if self.config.mode != GCMode.AUTOMATIC:
            return False
        
        # Check heap manager recommendation
        if self.heap_manager.should_trigger_gc():
            return True
        
        # Check allocation rate
        if self._allocation_tracker.get_allocation_rate() > 100 * 1024 * 1024:  # 100MB/s
            return True
        
        # Check system memory pressure
        if self._system_monitor.get_memory_pressure() > 0.8:
            return True
        
        return False
    
    def _should_compact(self) -> bool:
        """Determine if heap compaction is needed"""
        heap_stats = self.heap_manager.get_heap_statistics()
        
        for pool_name, pool_stats in heap_stats['pools'].items():
            if pool_stats.get('fragmentation', 0) > self.config.compaction_threshold:
                return True
        
        return False
    
    def _compact_fragmented_regions(self):
        """Compact highly fragmented heap regions"""
        for region_type in [HeapRegionType.YOUNG_GENERATION, HeapRegionType.OLD_GENERATION]:
            self.heap_manager.compact_heap(region_type)
    
    def add_root(self, obj: GCObject):
        """Add an object to the root set"""
        self.generational_gc.add_root_object(obj)
        self.reference_tracker.add_reference(obj, obj)  # Self-reference to mark as root
    
    def remove_root(self, obj: GCObject):
        """Remove an object from the root set"""
        self.generational_gc.remove_root_object(obj)
    
    def add_root_callback(self, callback: Callable[[], List[GCObject]]):
        """Add a callback that dynamically provides root objects"""
        self.generational_gc.add_root_callback(callback)
    
    def get_statistics(self) -> GCStats:
        """Get comprehensive garbage collection statistics"""
        with self._gc_lock:
            self._update_statistics()
            return self.stats
    
    def get_detailed_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics including all subsystems"""
        with self._gc_lock:
            return {
                'gc_stats': self.get_statistics().__dict__,
                'heap_stats': self.heap_manager.get_heap_statistics(),
                'generational_stats': self.generational_gc.get_gc_statistics(),
                'reference_stats': self._get_reference_statistics(),
                'system_stats': self._system_monitor.get_system_statistics(),
                'allocation_stats': self._allocation_tracker.get_statistics(),
            }
    
    def _get_reference_statistics(self) -> Dict[str, Any]:
        """Get reference tracking statistics"""
        memory_usage = self.reference_tracker.get_memory_usage_by_type()
        cycles = self.reference_tracker.find_cycles()
        
        return {
            'memory_usage_by_type': {k.name: v for k, v in memory_usage.items()},
            'reference_cycles_detected': len(cycles),
            'largest_cycle_size': max(len(cycle) for cycle in cycles) if cycles else 0
        }
    
    def _update_statistics(self):
        """Update overall GC statistics"""
        current_time = time.time()
        
        # Only update every second to avoid overhead
        if current_time - self._last_stats_update < 1.0:
            return
        
        heap_stats = self.heap_manager.get_heap_statistics()
        gen_stats = self.generational_gc.get_gc_statistics()
        
        self.stats.heap_size_bytes = sum(pool['total_size'] for pool in heap_stats['pools'].values())
        self.stats.used_bytes = sum(pool['total_used'] for pool in heap_stats['pools'].values())
        self.stats.free_bytes = self.stats.heap_size_bytes - self.stats.used_bytes
        
        self.stats.total_collections = gen_stats['total_collections']
        self.stats.total_pause_time_ms = gen_stats['total_pause_time_ms']
        self.stats.average_pause_time_ms = gen_stats['average_pause_time_ms']
        
        # Update allocation rate
        self.stats.allocation_rate_mb_per_sec = self._allocation_tracker.get_allocation_rate() / (1024 * 1024)
        
        # Update memory pressure
        self.stats.memory_pressure = self.heap_manager.get_memory_pressure()
        
        self._last_stats_update = current_time
    
    def _update_allocation_stats(self, obj_type: ObjectType, size: int, success: bool):
        """Update allocation-related statistics"""
        if success:
            self.stats.objects_by_type[obj_type.name] = self.stats.objects_by_type.get(obj_type.name, 0) + 1
            self.stats.total_objects += 1
    
    def _update_collection_stats(self, metrics: GCMetrics):
        """Update collection-related statistics"""
        gen = metrics.generation
        if gen < len(self.stats.collections_by_generation):
            self.stats.collections_by_generation[gen] += 1
        
        self.stats.max_pause_time_ms = max(self.stats.max_pause_time_ms, metrics.pause_time_ms)
        
        # Update collection efficiency
        if metrics.pause_time_ms > 0:
            efficiency = metrics.bytes_collected / metrics.pause_time_ms
            self.stats.collection_efficiency = (self.stats.collection_efficiency + efficiency) / 2
    
    def optimize_for_workload(self, workload_hints: List[str]):
        """Optimize GC parameters for a specific workload"""
        with self._gc_lock:
            if 'scientific_computing' in workload_hints:
                # Optimize for large arrays and mathematical operations
                self.config.young_gen_size_mb = 64
                self.config.large_object_threshold_kb = 16
                self.config.max_pause_time_ms = 5.0
                
            elif 'machine_learning' in workload_hints:
                # Optimize for tensor operations and gradients
                self.config.young_gen_size_mb = 128
                self.config.old_gen_size_mb = 512
                self.config.enable_incremental = True
                
            elif 'real_time' in workload_hints:
                # Optimize for minimal pause times
                self.config.max_pause_time_ms = 2.0
                self.config.enable_incremental = True
                self.config.collection_ratio = [1, 5, 25]  # More frequent collection
            
            # Reconfigure components
            self._configure_generational_gc()
    
    def force_full_collection(self) -> List[GCMetrics]:
        """Force a full collection of all generations"""
        with self._gc_lock:
            return self.generational_gc.collect_all_generations()
    
    def shutdown(self):
        """Shutdown the garbage collector and cleanup resources"""
        # Stop background thread
        if self._background_thread:
            self._shutdown_event.set()
            self._background_thread.join(timeout=5.0)
            self._background_thread = None
        
        # Final cleanup
        with self._gc_lock:
            # Disable write barriers
            self.generational_gc.write_barrier.disable()
            
            # Clear all references
            self.generational_gc._root_objects.clear()
            self.generational_gc._root_callbacks.clear()


class SystemMonitor:
    """Monitors system-level memory and performance metrics"""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_memory_pressure(self) -> float:
        """Get system memory pressure (0.0 to 1.0)"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent / 100.0
        except:
            return 0.5  # Default to moderate pressure
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        try:
            memory = psutil.virtual_memory()
            process_memory = self.process.memory_info()
            
            return {
                'system_memory_total': memory.total,
                'system_memory_available': memory.available,
                'system_memory_percent': memory.percent,
                'process_memory_rss': process_memory.rss,
                'process_memory_vms': process_memory.vms,
                'cpu_percent': self.process.cpu_percent(),
                'cpu_count': psutil.cpu_count()
            }
        except Exception as e:
            return {'error': str(e)}


class AllocationTracker:
    """Tracks allocation patterns and rates"""
    
    def __init__(self):
        self._allocations: deque = deque(maxlen=10000)
        self._lock = threading.RLock()
    
    def record_allocation(self, obj_type: ObjectType, size: int):
        """Record an allocation"""
        with self._lock:
            self._allocations.append({
                'timestamp': time.time(),
                'type': obj_type,
                'size': size
            })
    
    def get_allocation_rate(self) -> float:
        """Get current allocation rate in bytes per second"""
        with self._lock:
            if len(self._allocations) < 2:
                return 0.0
            
            now = time.time()
            recent_allocations = [a for a in self._allocations if now - a['timestamp'] < 5.0]
            
            if len(recent_allocations) < 2:
                return 0.0
            
            total_bytes = sum(a['size'] for a in recent_allocations)
            time_span = recent_allocations[-1]['timestamp'] - recent_allocations[0]['timestamp']
            
            return total_bytes / max(time_span, 0.001)  # Avoid division by zero
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get allocation tracking statistics"""
        with self._lock:
            if not self._allocations:
                return {}
            
            recent = list(self._allocations)[-1000:]  # Last 1000 allocations
            
            total_size = sum(a['size'] for a in recent)
            type_counts = {}
            
            for alloc in recent:
                obj_type = alloc['type'].name
                type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
            
            return {
                'recent_allocations': len(recent),
                'total_bytes_allocated': total_size,
                'average_allocation_size': total_size / len(recent),
                'allocation_rate_bytes_per_sec': self.get_allocation_rate(),
                'allocations_by_type': type_counts
            }
