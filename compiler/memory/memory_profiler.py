"""
Memory Profiler for NeuralScript

Provides detailed allocation tracking, memory usage analysis,
and profiling capabilities for NeuralScript programs.
"""

import time
import threading
import traceback
from typing import Dict, List, Optional, Set, Tuple, Any, NamedTuple
from enum import Enum, auto
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq

from .object_model import ObjectType, GCObject


class ProfilingLevel(Enum):
    """Profiling detail levels"""
    NONE = 0           # No profiling
    BASIC = 1          # Basic allocation counts and sizes
    DETAILED = 2       # Include call sites and allocation patterns
    COMPREHENSIVE = 3  # Full call stacks and memory tracking
    
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented
    
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented
    
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


@dataclass
class AllocationSite:
    """Information about an allocation site"""
    function_name: str
    filename: str
    line_number: int
    call_stack: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash((self.function_name, self.filename, self.line_number))
    
    def __eq__(self, other):
        if not isinstance(other, AllocationSite):
            return False
        return (self.function_name == other.function_name and
                self.filename == other.filename and
                self.line_number == other.line_number)


@dataclass
class AllocationRecord:
    """Record of a single allocation"""
    timestamp: float
    obj_type: ObjectType
    size: int
    allocation_site: Optional[AllocationSite] = None
    thread_id: int = 0
    generation: int = 0
    
    def age_seconds(self) -> float:
        """Get age of allocation in seconds"""
        return time.time() - self.timestamp


@dataclass
class AllocationProfile:
    """Profile data for a specific allocation pattern"""
    site: Optional[AllocationSite] = None
    total_allocations: int = 0
    total_bytes: int = 0
    current_allocations: int = 0
    current_bytes: int = 0
    peak_allocations: int = 0
    peak_bytes: int = 0
    average_size: float = 0.0
    allocation_rate: float = 0.0  # allocations per second
    first_seen: float = 0.0
    last_seen: float = 0.0
    
    def update(self, size: int, timestamp: float):
        """Update profile with new allocation"""
        if self.first_seen == 0.0:
            self.first_seen = timestamp
        
        self.last_seen = timestamp
        self.total_allocations += 1
        self.total_bytes += size
        self.current_allocations += 1
        self.current_bytes += size
        
        self.peak_allocations = max(self.peak_allocations, self.current_allocations)
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        
        self.average_size = self.total_bytes / self.total_allocations
        
        # Calculate allocation rate over last 60 seconds
        time_span = max(timestamp - self.first_seen, 1.0)
        self.allocation_rate = self.total_allocations / time_span
    
    def deallocate(self, size: int):
        """Update profile when allocation is freed"""
        self.current_allocations = max(0, self.current_allocations - 1)
        self.current_bytes = max(0, self.current_bytes - size)


class MemorySnapshot:
    """Snapshot of memory state at a point in time"""
    
    def __init__(self, timestamp: float):
        self.timestamp = timestamp
        self.allocations_by_type: Dict[ObjectType, int] = defaultdict(int)
        self.bytes_by_type: Dict[ObjectType, int] = defaultdict(int)
        self.allocations_by_site: Dict[AllocationSite, AllocationProfile] = {}
        self.total_heap_size: int = 0
        self.total_used_bytes: int = 0
        self.fragmentation_ratio: float = 0.0


class MemoryProfiler:
    """
    Advanced memory profiler for NeuralScript.
    
    Tracks allocations, provides detailed analysis of memory usage patterns,
    and helps identify memory leaks and optimization opportunities.
    """
    
    def __init__(self, profiling_level: ProfilingLevel = ProfilingLevel.BASIC):
        self.profiling_level = profiling_level
        
        # Allocation tracking
        self._allocations: Dict[int, AllocationRecord] = {}  # address -> record
        self._allocation_profiles: Dict[AllocationSite, AllocationProfile] = defaultdict(AllocationProfile)
        
        # Historical data
        self._snapshots: deque = deque(maxlen=100)  # Keep last 100 snapshots
        self._allocation_history: deque = deque(maxlen=10000)  # Recent allocations
        
        # Performance tracking
        self._allocation_timeline: List[Tuple[float, int, int]] = []  # (timestamp, count, bytes)
        self._deallocation_timeline: List[Tuple[float, int, int]] = []
        
        # Leak detection
        self._long_lived_objects: Set[int] = set()
        self._potential_leaks: List[Tuple[AllocationSite, int]] = []
        
        # Synchronization
        self._lock = threading.RLock()
        
        # Configuration
        self._snapshot_interval = 60.0  # seconds
        self._last_snapshot_time = time.time()
        self._leak_detection_threshold = 300.0  # 5 minutes
    
    def record_allocation(self, obj_type: ObjectType, size: int, 
                         address: Optional[int] = None,
                         generation: int = 0) -> AllocationRecord:
        """Record a new allocation"""
        if self.profiling_level == ProfilingLevel.NONE:
            return None
        
        with self._lock:
            timestamp = time.time()
            thread_id = threading.get_ident()
            
            # Get allocation site if detailed profiling is enabled
            allocation_site = None
            if self.profiling_level >= ProfilingLevel.DETAILED:
                allocation_site = self._capture_allocation_site()
            
            # Create allocation record
            record = AllocationRecord(
                timestamp=timestamp,
                obj_type=obj_type,
                size=size,
                allocation_site=allocation_site,
                thread_id=thread_id,
                generation=generation
            )
            
            # Store record if we have an address
            if address is not None:
                self._allocations[address] = record
            
            # Update allocation history
            self._allocation_history.append(record)
            
            # Update allocation timeline
            self._allocation_timeline.append((timestamp, 1, size))
            
            # Update profiles if we have allocation site
            if allocation_site:
                profile = self._allocation_profiles[allocation_site]
                if profile.site is None:
                    profile.site = allocation_site
                profile.update(size, timestamp)
            
            # Periodic maintenance
            self._periodic_maintenance()
            
            return record
    
    def record_deallocation(self, address: int, size: int):
        """Record a deallocation"""
        if self.profiling_level == ProfilingLevel.NONE:
            return
        
        with self._lock:
            timestamp = time.time()
            
            # Remove from active allocations
            record = self._allocations.pop(address, None)
            
            # Update deallocation timeline
            self._deallocation_timeline.append((timestamp, 1, size))
            
            # Update allocation profile if we have the record
            if record and record.allocation_site:
                profile = self._allocation_profiles.get(record.allocation_site)
                if profile:
                    profile.deallocate(size)
            
            # Remove from long-lived tracking
            self._long_lived_objects.discard(address)
    
    def _capture_allocation_site(self) -> AllocationSite:
        """Capture the current allocation site from call stack"""
        try:
            stack = traceback.extract_stack()
            
            # Find the first frame outside the GC system
            for frame in reversed(stack[:-2]):  # Skip current frame and caller
                if not any(comp in frame.filename for comp in ['memory', 'gc_', 'heap_']):
                    call_stack = []
                    if self.profiling_level == ProfilingLevel.COMPREHENSIVE:
                        call_stack = [f"{f.filename}:{f.lineno} in {f.name}" 
                                    for f in stack[-10:]]  # Last 10 frames
                    
                    return AllocationSite(
                        function_name=frame.name,
                        filename=frame.filename,
                        line_number=frame.lineno,
                        call_stack=call_stack
                    )
            
            # Fallback if no suitable frame found
            frame = stack[-3] if len(stack) > 2 else stack[-1]
            return AllocationSite(
                function_name=frame.name,
                filename=frame.filename,
                line_number=frame.lineno
            )
        except:
            return AllocationSite(
                function_name="unknown",
                filename="unknown",
                line_number=0
            )
    
    def _periodic_maintenance(self):
        """Perform periodic maintenance tasks"""
        current_time = time.time()
        
        # Take snapshots periodically
        if current_time - self._last_snapshot_time > self._snapshot_interval:
            self._take_snapshot()
            self._last_snapshot_time = current_time
        
        # Detect potential memory leaks
        self._detect_memory_leaks(current_time)
        
        # Cleanup old timeline data
        self._cleanup_timeline_data(current_time)
    
    def _take_snapshot(self):
        """Take a snapshot of current memory state"""
        timestamp = time.time()
        snapshot = MemorySnapshot(timestamp)
        
        # Aggregate current allocations
        for record in self._allocations.values():
            snapshot.allocations_by_type[record.obj_type] += 1
            snapshot.bytes_by_type[record.obj_type] += record.size
        
        # Copy allocation profiles
        for site, profile in self._allocation_profiles.items():
            if profile.current_allocations > 0:
                snapshot.allocations_by_site[site] = AllocationProfile(
                    site=site,
                    total_allocations=profile.total_allocations,
                    total_bytes=profile.total_bytes,
                    current_allocations=profile.current_allocations,
                    current_bytes=profile.current_bytes,
                    peak_allocations=profile.peak_allocations,
                    peak_bytes=profile.peak_bytes,
                    average_size=profile.average_size,
                    allocation_rate=profile.allocation_rate,
                    first_seen=profile.first_seen,
                    last_seen=profile.last_seen
                )
        
        self._snapshots.append(snapshot)
    
    def _detect_memory_leaks(self, current_time: float):
        """Detect potential memory leaks"""
        # Check for long-lived objects
        leak_candidates = []
        
        for address, record in self._allocations.items():
            age = current_time - record.timestamp
            
            if age > self._leak_detection_threshold:
                self._long_lived_objects.add(address)
                
                if record.allocation_site:
                    leak_candidates.append((record.allocation_site, record.size))
        
        # Group potential leaks by allocation site
        leak_groups = defaultdict(int)
        for site, size in leak_candidates:
            leak_groups[site] += size
        
        # Update potential leaks list
        self._potential_leaks = [(site, total_size) 
                               for site, total_size in leak_groups.items()
                               if total_size > 1024 * 1024]  # > 1MB
        
        # Sort by total leaked bytes
        self._potential_leaks.sort(key=lambda x: x[1], reverse=True)
    
    def _cleanup_timeline_data(self, current_time: float):
        """Remove old timeline data to prevent unbounded growth"""
        cutoff_time = current_time - 3600.0  # Keep last hour
        
        self._allocation_timeline = [(t, c, s) for t, c, s in self._allocation_timeline 
                                   if t > cutoff_time]
        self._deallocation_timeline = [(t, c, s) for t, c, s in self._deallocation_timeline 
                                     if t > cutoff_time]
    
    def get_allocation_profiles(self) -> List[AllocationProfile]:
        """Get all allocation profiles sorted by total bytes"""
        with self._lock:
            profiles = list(self._allocation_profiles.values())
            profiles.sort(key=lambda p: p.total_bytes, reverse=True)
            return profiles
    
    def get_top_allocators(self, limit: int = 10) -> List[AllocationProfile]:
        """Get top allocation sites by total bytes allocated"""
        profiles = self.get_allocation_profiles()
        return profiles[:limit]
    
    def get_memory_leaks(self) -> List[Tuple[AllocationSite, int]]:
        """Get potential memory leaks"""
        with self._lock:
            return list(self._potential_leaks)
    
    def get_allocation_timeline(self, duration_seconds: float = 3600.0) -> Dict[str, List]:
        """Get allocation timeline data for visualization"""
        with self._lock:
            current_time = time.time()
            cutoff_time = current_time - duration_seconds
            
            # Filter recent data
            recent_allocs = [(t, c, s) for t, c, s in self._allocation_timeline 
                           if t > cutoff_time]
            recent_deallocs = [(t, c, s) for t, c, s in self._deallocation_timeline 
                             if t > cutoff_time]
            
            return {
                'allocations': recent_allocs,
                'deallocations': recent_deallocs,
                'start_time': cutoff_time,
                'end_time': current_time
            }
    
    def get_memory_usage_by_type(self) -> Dict[ObjectType, Tuple[int, int]]:
        """Get current memory usage by object type (count, bytes)"""
        with self._lock:
            usage = defaultdict(lambda: [0, 0])  # [count, bytes]
            
            for record in self._allocations.values():
                usage[record.obj_type][0] += 1
                usage[record.obj_type][1] += record.size
            
            return {obj_type: (count, bytes_used) 
                   for obj_type, (count, bytes_used) in usage.items()}
    
    def get_allocation_patterns(self) -> Dict[str, Any]:
        """Analyze allocation patterns and return insights"""
        with self._lock:
            if not self._allocation_history:
                return {}
            
            # Analyze allocation sizes
            sizes = [record.size for record in self._allocation_history]
            size_histogram = self._create_histogram(sizes, bins=20)
            
            # Analyze allocation frequency by type
            type_counts = defaultdict(int)
            for record in self._allocation_history:
                type_counts[record.obj_type.name] += 1
            
            # Analyze temporal patterns
            timestamps = [record.timestamp for record in self._allocation_history]
            time_gaps = []
            for i in range(1, len(timestamps)):
                time_gaps.append(timestamps[i] - timestamps[i-1])
            
            return {
                'allocation_size_histogram': size_histogram,
                'allocations_by_type': dict(type_counts),
                'average_allocation_interval': sum(time_gaps) / len(time_gaps) if time_gaps else 0,
                'allocation_burst_detection': self._detect_allocation_bursts(),
                'memory_growth_trend': self._analyze_memory_growth()
            }
    
    def _create_histogram(self, values: List[float], bins: int = 20) -> Dict[str, List]:
        """Create histogram of values"""
        if not values:
            return {'bins': [], 'counts': []}
        
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            return {'bins': [min_val], 'counts': [len(values)]}
        
        bin_width = (max_val - min_val) / bins
        bin_edges = [min_val + i * bin_width for i in range(bins + 1)]
        counts = [0] * bins
        
        for value in values:
            bin_idx = min(int((value - min_val) / bin_width), bins - 1)
            counts[bin_idx] += 1
        
        return {'bins': bin_edges, 'counts': counts}
    
    def _detect_allocation_bursts(self) -> List[Dict[str, Any]]:
        """Detect allocation bursts (sudden increases in allocation rate)"""
        if len(self._allocation_timeline) < 10:
            return []
        
        bursts = []
        window_size = 10
        
        for i in range(window_size, len(self._allocation_timeline)):
            # Calculate rate for current window
            window = self._allocation_timeline[i-window_size:i]
            time_span = window[-1][0] - window[0][0]
            
            if time_span > 0:
                rate = sum(alloc[2] for alloc in window) / time_span  # bytes per second
                
                # Compare with baseline (previous window)
                if i >= 2 * window_size:
                    baseline_window = self._allocation_timeline[i-2*window_size:i-window_size]
                    baseline_span = baseline_window[-1][0] - baseline_window[0][0]
                    baseline_rate = sum(alloc[2] for alloc in baseline_window) / baseline_span
                    
                    # Detect burst (3x increase)
                    if rate > 3 * baseline_rate and rate > 1024 * 1024:  # > 1MB/s
                        bursts.append({
                            'timestamp': window[-1][0],
                            'rate_bytes_per_sec': rate,
                            'baseline_rate_bytes_per_sec': baseline_rate,
                            'burst_ratio': rate / baseline_rate
                        })
        
        return bursts[-10:]  # Return last 10 bursts
    
    def _analyze_memory_growth(self) -> Dict[str, Any]:
        """Analyze memory growth trends"""
        if len(self._snapshots) < 2:
            return {}
        
        snapshots = list(self._snapshots)
        
        # Calculate growth rate
        first_snapshot = snapshots[0]
        last_snapshot = snapshots[-1]
        
        time_span = last_snapshot.timestamp - first_snapshot.timestamp
        if time_span <= 0:
            return {}
        
        memory_growth = last_snapshot.total_used_bytes - first_snapshot.total_used_bytes
        growth_rate = memory_growth / time_span  # bytes per second
        
        # Analyze growth by object type
        type_growth = {}
        for obj_type in ObjectType:
            first_bytes = first_snapshot.bytes_by_type.get(obj_type, 0)
            last_bytes = last_snapshot.bytes_by_type.get(obj_type, 0)
            type_growth[obj_type.name] = last_bytes - first_bytes
        
        return {
            'total_growth_bytes': memory_growth,
            'growth_rate_bytes_per_sec': growth_rate,
            'growth_by_type': type_growth,
            'time_span_seconds': time_span,
            'is_growing': memory_growth > 0
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive profiling report"""
        with self._lock:
            report_lines = []
            report_lines.append("NeuralScript Memory Profiling Report")
            report_lines.append("=" * 50)
            
            # Basic statistics
            total_allocs = len(self._allocations)
            total_bytes = sum(record.size for record in self._allocations.values())
            
            report_lines.append(f"Current allocations: {total_allocs}")
            report_lines.append(f"Current memory usage: {total_bytes / (1024*1024):.2f} MB")
            
            # Top allocators
            top_allocators = self.get_top_allocators(10)
            if top_allocators:
                report_lines.append("\nTop Allocation Sites:")
                report_lines.append("-" * 30)
                
                for profile in top_allocators:
                    site = profile.site
                    report_lines.append(
                        f"{site.function_name} ({site.filename}:{site.line_number})\n"
                        f"  Total: {profile.total_allocations} allocs, "
                        f"{profile.total_bytes / (1024*1024):.2f} MB\n"
                        f"  Current: {profile.current_allocations} allocs, "
                        f"{profile.current_bytes / (1024*1024):.2f} MB\n"
                        f"  Average size: {profile.average_size:.0f} bytes"
                    )
            
            # Memory leaks
            leaks = self.get_memory_leaks()
            if leaks:
                report_lines.append("\nPotential Memory Leaks:")
                report_lines.append("-" * 30)
                
                for site, leaked_bytes in leaks[:5]:  # Top 5 leaks
                    report_lines.append(
                        f"{site.function_name} ({site.filename}:{site.line_number})\n"
                        f"  Potentially leaked: {leaked_bytes / (1024*1024):.2f} MB"
                    )
            
            # Memory growth analysis
            growth_analysis = self._analyze_memory_growth()
            if growth_analysis:
                report_lines.append("\nMemory Growth Analysis:")
                report_lines.append("-" * 30)
                
                growth_mb = growth_analysis['total_growth_bytes'] / (1024*1024)
                rate_mb_s = growth_analysis['growth_rate_bytes_per_sec'] / (1024*1024)
                
                report_lines.append(f"Total growth: {growth_mb:.2f} MB")
                report_lines.append(f"Growth rate: {rate_mb_s:.2f} MB/s")
                
                if growth_analysis['is_growing']:
                    report_lines.append("⚠️  Memory usage is growing")
            
            return "\n".join(report_lines)
    
    def export_data(self, format: str = 'json') -> Any:
        """Export profiling data in specified format"""
        with self._lock:
            data = {
                'profiling_level': self.profiling_level.name,
                'timestamp': time.time(),
                'allocation_profiles': {
                    f"{site.function_name}@{site.filename}:{site.line_number}": {
                        'total_allocations': profile.total_allocations,
                        'total_bytes': profile.total_bytes,
                        'current_allocations': profile.current_allocations,
                        'current_bytes': profile.current_bytes,
                        'peak_allocations': profile.peak_allocations,
                        'peak_bytes': profile.peak_bytes,
                        'average_size': profile.average_size,
                        'allocation_rate': profile.allocation_rate
                    }
                    for site, profile in self._allocation_profiles.items()
                },
                'memory_usage_by_type': {
                    obj_type.name: {'count': count, 'bytes': bytes_used}
                    for obj_type, (count, bytes_used) in self.get_memory_usage_by_type().items()
                },
                'potential_leaks': [
                    {
                        'site': f"{site.function_name}@{site.filename}:{site.line_number}",
                        'leaked_bytes': leaked_bytes
                    }
                    for site, leaked_bytes in self._potential_leaks
                ],
                'allocation_patterns': self.get_allocation_patterns()
            }
            
            if format == 'json':
                import json
                return json.dumps(data, indent=2)
            else:
                return data
    
    def clear_history(self):
        """Clear allocation history and profiles"""
        with self._lock:
            self._allocations.clear()
            self._allocation_profiles.clear()
            self._allocation_history.clear()
            self._snapshots.clear()
            self._allocation_timeline.clear()
            self._deallocation_timeline.clear()
            self._long_lived_objects.clear()
            self._potential_leaks.clear()
    
    def set_profiling_level(self, level: ProfilingLevel):
        """Change the profiling level"""
        with self._lock:
            self.profiling_level = level
            
            # Clear detailed data if switching to basic profiling
            if level == ProfilingLevel.BASIC:
                for profile in self._allocation_profiles.values():
                    if profile.site and profile.site.call_stack:
                        profile.site.call_stack.clear()
            
            # Clear all data if disabling profiling
            elif level == ProfilingLevel.NONE:
                self.clear_history()
