"""
Heap Manager for NeuralScript GC

Manages memory pools, allocation strategies, and heap regions
for the garbage collector.
"""

import threading
import mmap
import os
import sys
import ctypes
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum, auto
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .object_model import GCObject, ObjectHeader, ObjectType


class AllocationStrategy(Enum):
    """Strategies for object allocation"""
    FIRST_FIT = auto()          # First available block that fits
    BEST_FIT = auto()           # Smallest block that fits
    WORST_FIT = auto()          # Largest available block
    NEXT_FIT = auto()           # First fit starting from last allocation
    SEGREGATED_FIT = auto()     # Separate free lists by size class


class HeapRegionType(Enum):
    """Types of heap regions"""
    YOUNG_GENERATION = auto()   # For newly allocated objects
    OLD_GENERATION = auto()     # For long-lived objects  
    PERMANENT = auto()          # For metadata and constants
    LARGE_OBJECT = auto()       # For objects > threshold size
    CODE = auto()               # For compiled code


@dataclass
class FreeBlock:
    """Represents a free block in the heap"""
    address: int
    size: int
    next_block: Optional['FreeBlock'] = None
    prev_block: Optional['FreeBlock'] = None
    
    def split(self, alloc_size: int) -> Optional['FreeBlock']:
        """Split this block if it's larger than needed"""
        if self.size <= alloc_size + 32:  # Minimum split size
            return None
            
        # Create new block for remainder
        remainder_addr = self.address + alloc_size
        remainder_size = self.size - alloc_size
        remainder = FreeBlock(remainder_addr, remainder_size)
        
        # Update this block
        self.size = alloc_size
        
        return remainder


@dataclass 
class HeapRegion:
    """A contiguous region of heap memory"""
    region_type: HeapRegionType
    start_address: int
    size: int
    used_bytes: int = 0
    free_blocks: List[FreeBlock] = None
    allocation_count: int = 0
    
    def __post_init__(self):
        if self.free_blocks is None:
            # Initially one big free block
            self.free_blocks = [FreeBlock(self.start_address, self.size)]
        self._lock = threading.RLock()
    
    @property
    def end_address(self) -> int:
        return self.start_address + self.size
    
    @property
    def free_bytes(self) -> int:
        return self.size - self.used_bytes
    
    @property
    def utilization(self) -> float:
        return self.used_bytes / self.size if self.size > 0 else 0.0
    
    def contains_address(self, address: int) -> bool:
        return self.start_address <= address < self.end_address
    
    def can_allocate(self, size: int) -> bool:
        """Check if this region can satisfy an allocation request"""
        with self._lock:
            for block in self.free_blocks:
                if block.size >= size:
                    return True
            return False


class MemoryPool:
    """
    Manages a pool of memory for a specific size class or region type.
    
    Uses segregated free lists and coalescing for efficient allocation
    and deallocation.
    """
    
    def __init__(self, name: str, region_type: HeapRegionType, 
                 initial_size: int = 64 * 1024 * 1024):  # 64MB default
        self.name = name
        self.region_type = region_type
        self.regions: List[HeapRegion] = []
        self.free_lists: Dict[int, List[FreeBlock]] = {}  # Size class -> blocks
        self.allocation_strategy = AllocationStrategy.SEGREGATED_FIT
        self._lock = threading.RLock()
        
        # Statistics
        self.total_allocated = 0
        self.total_freed = 0
        self.allocation_count = 0
        self.free_count = 0
        
        # Create initial region
        self._create_region(initial_size)
    
    def _create_region(self, size: int) -> HeapRegion:
        """Create a new heap region"""
        # Use mmap for virtual memory allocation (cross-platform)
        try:
            # Cross-platform memory mapping
            if sys.platform == 'win32':
                # Windows: anonymous memory mapping
                mem = mmap.mmap(-1, size)
            else:
                # Unix/Linux: use MAP_PRIVATE and MAP_ANONYMOUS
                mem = mmap.mmap(-1, size, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            
            # Simplified address handling for mock implementation
            address = id(mem)  # Use object id as mock address
            
            region = HeapRegion(
                region_type=self.region_type,
                start_address=address,
                size=size
            )
            
            # Store mmap object to keep memory alive
            region._mmap_obj = mem
            
            self.regions.append(region)
            
            # Add initial free block to appropriate size class
            free_block = region.free_blocks[0]
            self._add_to_free_list(free_block)
            
            return region
            
        except OSError as e:
            raise RuntimeError(f"Failed to allocate memory region: {e}")
    
    def _get_size_class(self, size: int) -> int:
        """Map allocation size to size class"""
        # Power-of-2 size classes: 16, 32, 64, 128, 256, 512, 1024, ...
        if size <= 16:
            return 16
        
        # Round up to next power of 2
        import math
        return 1 << math.ceil(math.log2(size))
    
    def _add_to_free_list(self, block: FreeBlock):
        """Add a free block to the appropriate size class"""
        size_class = self._get_size_class(block.size)
        
        if size_class not in self.free_lists:
            self.free_lists[size_class] = []
        
        self.free_lists[size_class].append(block)
        
        # Sort by size within class for best-fit
        self.free_lists[size_class].sort(key=lambda b: b.size)
    
    def _remove_from_free_list(self, block: FreeBlock):
        """Remove a block from the free list"""
        size_class = self._get_size_class(block.size)
        
        if size_class in self.free_lists:
            try:
                self.free_lists[size_class].remove(block)
            except ValueError:
                pass  # Block not in list
    
    def _find_best_fit_block(self, size: int) -> Optional[FreeBlock]:
        """Find the best fitting free block for allocation"""
        size_class = self._get_size_class(size)
        
        # Try exact size class first
        if size_class in self.free_lists:
            for block in self.free_lists[size_class]:
                if block.size >= size:
                    return block
        
        # Try larger size classes
        for cls in sorted(self.free_lists.keys()):
            if cls > size_class:
                for block in self.free_lists[cls]:
                    if block.size >= size:
                        return block
        
        return None
    
    def _coalesce_free_blocks(self):
        """Coalesce adjacent free blocks to reduce fragmentation"""
        with self._lock:
            # Collect all free blocks and sort by address
            all_blocks = []
            for block_list in self.free_lists.values():
                all_blocks.extend(block_list)
            
            all_blocks.sort(key=lambda b: b.address)
            
            # Clear existing free lists
            self.free_lists.clear()
            
            # Merge adjacent blocks
            i = 0
            while i < len(all_blocks):
                current = all_blocks[i]
                
                # Try to merge with next blocks
                while (i + 1 < len(all_blocks) and 
                       current.address + current.size == all_blocks[i + 1].address):
                    next_block = all_blocks[i + 1]
                    current.size += next_block.size
                    i += 1
                
                # Add merged block back to free list
                self._add_to_free_list(current)
                i += 1
    
    def allocate(self, size: int, alignment: int = 8) -> Optional[int]:
        """
        Allocate a block of memory.
        
        Returns the address of the allocated block or None if allocation fails.
        """
        if size <= 0:
            return None
        
        # Align size
        aligned_size = (size + alignment - 1) & ~(alignment - 1)
        
        with self._lock:
            # Find suitable free block
            block = self._find_best_fit_block(aligned_size)
            
            if block is None:
                # Try coalescing first
                self._coalesce_free_blocks()
                block = self._find_best_fit_block(aligned_size)
                
                if block is None:
                    # Need to expand heap
                    expansion_size = max(aligned_size * 2, 1024 * 1024)
                    try:
                        self._create_region(expansion_size)
                        block = self._find_best_fit_block(aligned_size)
                    except RuntimeError:
                        return None  # Out of memory
            
            if block is None:
                return None
            
            # Remove from free list
            self._remove_from_free_list(block)
            
            # Split block if it's much larger than needed
            remainder = block.split(aligned_size)
            if remainder:
                self._add_to_free_list(remainder)
            
            # Update statistics
            self.total_allocated += block.size
            self.allocation_count += 1
            
            # Update region stats
            for region in self.regions:
                if region.contains_address(block.address):
                    region.used_bytes += block.size
                    region.allocation_count += 1
                    break
            
            return block.address
    
    def deallocate(self, address: int, size: int):
        """Free a previously allocated block"""
        if address == 0 or size <= 0:
            return
        
        with self._lock:
            # Create free block
            free_block = FreeBlock(address, size)
            
            # Add to free list
            self._add_to_free_list(free_block)
            
            # Update statistics
            self.total_freed += size
            self.free_count += 1
            
            # Update region stats
            for region in self.regions:
                if region.contains_address(address):
                    region.used_bytes -= size
                    break
            
            # Periodic coalescing
            if self.free_count % 100 == 0:
                self._coalesce_free_blocks()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory pool statistics"""
        with self._lock:
            total_size = sum(r.size for r in self.regions)
            total_used = sum(r.used_bytes for r in self.regions)
            
            return {
                'name': self.name,
                'region_type': self.region_type.name,
                'total_size': total_size,
                'total_used': total_used,
                'total_free': total_size - total_used,
                'utilization': total_used / total_size if total_size > 0 else 0.0,
                'regions_count': len(self.regions),
                'allocations': self.allocation_count,
                'deallocations': self.free_count,
                'fragmentation': len(self.free_lists)
            }


class HeapManager:
    """
    Main heap manager that coordinates multiple memory pools
    and handles allocation requests from the GC.
    """
    
    def __init__(self):
        self.memory_pools: Dict[HeapRegionType, MemoryPool] = {}
        self.large_object_threshold = 32 * 1024  # 32KB
        self.allocation_strategy = AllocationStrategy.SEGREGATED_FIT
        self._lock = threading.RLock()
        
        # Create memory pools for different heap regions
        self._initialize_pools()
        
        # Statistics
        self.total_allocations = 0
        self.total_deallocations = 0
        self.bytes_allocated = 0
        self.bytes_freed = 0
    
    def _initialize_pools(self):
        """Initialize memory pools for different heap regions"""
        pool_configs = [
            (HeapRegionType.YOUNG_GENERATION, "Young Gen", 32 * 1024 * 1024),
            (HeapRegionType.OLD_GENERATION, "Old Gen", 128 * 1024 * 1024),
            (HeapRegionType.LARGE_OBJECT, "Large Objects", 64 * 1024 * 1024),
            (HeapRegionType.PERMANENT, "Permanent", 16 * 1024 * 1024),
        ]
        
        for region_type, name, size in pool_configs:
            self.memory_pools[region_type] = MemoryPool(name, region_type, size)
    
    def allocate_object(self, obj_type: ObjectType, size: int, 
                       generation: int = 0) -> Optional[int]:
        """
        Allocate memory for a GC object.
        
        Chooses the appropriate memory pool based on object characteristics.
        """
        if size <= 0:
            return None
        
        with self._lock:
            # Determine target pool
            if size >= self.large_object_threshold:
                pool = self.memory_pools[HeapRegionType.LARGE_OBJECT]
            elif generation == 0:
                pool = self.memory_pools[HeapRegionType.YOUNG_GENERATION]
            else:
                pool = self.memory_pools[HeapRegionType.OLD_GENERATION]
            
            # Attempt allocation
            address = pool.allocate(size)
            
            if address is not None:
                self.total_allocations += 1
                self.bytes_allocated += size
            
            return address
    
    def deallocate_object(self, address: int, size: int, region_type: HeapRegionType):
        """Deallocate memory for a GC object"""
        if address == 0 or size <= 0:
            return
        
        with self._lock:
            if region_type in self.memory_pools:
                self.memory_pools[region_type].deallocate(address, size)
                self.total_deallocations += 1
                self.bytes_freed += size
    
    def get_region_for_address(self, address: int) -> Optional[HeapRegionType]:
        """Find which heap region contains the given address"""
        with self._lock:
            for region_type, pool in self.memory_pools.items():
                for region in pool.regions:
                    if region.contains_address(address):
                        return region_type
            return None
    
    def compact_heap(self, region_type: HeapRegionType):
        """Compact a specific heap region to reduce fragmentation"""
        if region_type not in self.memory_pools:
            return
        
        with self._lock:
            pool = self.memory_pools[region_type]
            pool._coalesce_free_blocks()
    
    def get_memory_pressure(self) -> float:
        """
        Calculate memory pressure as a value between 0.0 and 1.0.
        
        Higher values indicate more memory pressure and suggest
        garbage collection should be triggered.
        """
        with self._lock:
            total_size = 0
            total_used = 0
            
            for pool in self.memory_pools.values():
                for region in pool.regions:
                    total_size += region.size
                    total_used += region.used_bytes
            
            return total_used / total_size if total_size > 0 else 0.0
    
    def should_trigger_gc(self) -> bool:
        """Check if GC should be triggered based on memory pressure"""
        pressure = self.get_memory_pressure()
        
        # Trigger GC if any region is > 80% full
        for pool in self.memory_pools.values():
            for region in pool.regions:
                if region.utilization > 0.8:
                    return True
        
        return pressure > 0.75
    
    def get_heap_statistics(self) -> Dict[str, Any]:
        """Get comprehensive heap statistics"""
        with self._lock:
            stats = {
                'total_allocations': self.total_allocations,
                'total_deallocations': self.total_deallocations,
                'bytes_allocated': self.bytes_allocated,
                'bytes_freed': self.bytes_freed,
                'net_bytes': self.bytes_allocated - self.bytes_freed,
                'memory_pressure': self.get_memory_pressure(),
                'pools': {}
            }
            
            for region_type, pool in self.memory_pools.items():
                stats['pools'][region_type.name] = pool.get_statistics()
            
            return stats
    
    def optimize_allocation_strategy(self):
        """
        Analyze allocation patterns and optimize strategy.
        
        This is called periodically to adapt to workload patterns.
        """
        with self._lock:
            # Analyze fragmentation levels
            high_fragmentation_pools = []
            
            for region_type, pool in self.memory_pools.items():
                stats = pool.get_statistics()
                fragmentation_ratio = stats['fragmentation'] / max(stats['regions_count'], 1)
                
                if fragmentation_ratio > 10:  # Threshold for high fragmentation
                    high_fragmentation_pools.append(region_type)
            
            # Trigger compaction for highly fragmented pools
            for region_type in high_fragmentation_pools:
                self.compact_heap(region_type)
    
    def set_large_object_threshold(self, threshold: int):
        """Set the large object threshold"""
        with self._lock:
            self.large_object_threshold = threshold
    
    def set_allocation_strategy(self, region_type: HeapRegionType, strategy: AllocationStrategy):
        """Set allocation strategy for a specific region type"""
        with self._lock:
            if region_type in self.memory_pools:
                self.memory_pools[region_type].allocation_strategy = strategy
    
    def create_specialized_pool(self, name: str, size: int, alignment: int = 64):
        """Create a specialized memory pool"""
        with self._lock:
            # Create a new pool with custom settings
            pool = MemoryPool(name, HeapRegionType.PERMANENT, size)
            # Note: alignment parameter stored for reference but not fully implemented
            pool._alignment = alignment
            return pool
    
    def resize_pool(self, pool_name: str, new_size: int):
        """Resize a memory pool (simplified implementation)"""
        with self._lock:
            # Find pool by name and try to resize
            for pool in self.memory_pools.values():
                if pool.name == pool_name:
                    # Add new region to expand the pool
                    try:
                        pool._create_region(new_size - sum(r.size for r in pool.regions))
                        return True
                    except RuntimeError:
                        return False
            return False
    
    def reset_statistics(self):
        """Reset allocation statistics"""
        with self._lock:
            self.total_allocations = 0
            self.total_deallocations = 0
            self.bytes_allocated = 0
            self.bytes_freed = 0
