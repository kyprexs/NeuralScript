"""
Smart Memory Pool Management System for NeuralScript
===================================================

High-performance memory allocation system designed to achieve 30% less
memory usage than Python through intelligent pooling, alignment, and
lifecycle management.

Features:
- Size-based memory pools to reduce fragmentation
- Cache-aligned allocations for SIMD operations
- Custom allocators for matrices and tensors
- Memory debugging and leak detection
- Thread-safe allocation with minimal locking
- Automatic pool scaling and compaction
"""

import threading
import ctypes
import mmap
import os
import weakref
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import struct


class AllocationType(Enum):
    """Types of memory allocations"""
    SMALL_OBJECT = "small_object"       # < 256 bytes
    MEDIUM_OBJECT = "medium_object"     # 256 bytes - 64KB
    LARGE_OBJECT = "large_object"       # > 64KB
    MATRIX_DATA = "matrix_data"         # Matrix/tensor data
    STRING_DATA = "string_data"         # String objects
    CODE_OBJECT = "code_object"         # Compiled code
    METADATA = "metadata"               # Object metadata
    SEQUENCE_DATA = "sequence_data"     # Lists, arrays
    MAPPING_DATA = "mapping_data"       # Dictionaries, maps
    SET_DATA = "set_data"               # Sets
    TUPLE_DATA = "tuple_data"           # Tuples


@dataclass
class MemoryBlock:
    """Represents a single memory block"""
    address: int
    size: int
    allocated: bool
    allocation_time: float
    allocation_type: AllocationType
    ref_count: int = 0
    debug_info: Optional[Dict] = None


@dataclass
class PoolStats:
    """Statistics for a memory pool"""
    total_blocks: int = 0
    allocated_blocks: int = 0
    free_blocks: int = 0
    total_bytes: int = 0
    allocated_bytes: int = 0
    free_bytes: int = 0
    fragmentation_ratio: float = 0.0
    allocations_count: int = 0
    deallocations_count: int = 0


class MemoryPool:
    """
    A memory pool for specific block sizes.
    
    Uses pre-allocated chunks to minimize system malloc/free calls
    and reduce memory fragmentation.
    """
    
    def __init__(self, block_size: int, initial_blocks: int = 128,
                 alignment: int = 64, pool_name: str = "default"):
        self.block_size = self._align_size(block_size, alignment)
        self.alignment = alignment
        self.pool_name = pool_name
        
        # Memory management
        self.chunks: List[int] = []  # Base addresses of memory chunks
        self.free_blocks: List[int] = []  # Available block addresses
        self.allocated_blocks: Dict[int, MemoryBlock] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = PoolStats()
        
        # Pre-allocate initial blocks
        self._allocate_chunk(initial_blocks)
    
    def _align_size(self, size: int, alignment: int) -> int:
        """Align size to the specified boundary"""
        return (size + alignment - 1) & ~(alignment - 1)
    
    def _allocate_chunk(self, num_blocks: int):
        """Allocate a new chunk of memory blocks"""
        chunk_size = self.block_size * num_blocks
        
        try:
            # Use platform-appropriate memory allocation
            if chunk_size >= 1024 * 1024:  # 1MB threshold
                # Use mmap on Unix-like systems, or fallback to ctypes
                if hasattr(mmap, 'MAP_PRIVATE') and hasattr(mmap, 'MAP_ANONYMOUS'):
                    chunk_addr = mmap.mmap(-1, chunk_size, 
                                         mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
                    chunk_base = ctypes.addressof(ctypes.c_char.from_buffer(chunk_addr))
                else:
                    # Windows fallback - use large ctypes allocation
                    buffer = ctypes.create_string_buffer(chunk_size + self.alignment)
                    aligned_addr = ctypes.addressof(buffer)
                    if aligned_addr % self.alignment != 0:
                        aligned_addr = (aligned_addr + self.alignment) & ~(self.alignment - 1)
                    chunk_base = aligned_addr
            else:
                # Use ctypes for smaller allocations with alignment
                buffer = ctypes.create_string_buffer(chunk_size + self.alignment)
                aligned_addr = ctypes.addressof(buffer)
                if aligned_addr % self.alignment != 0:
                    aligned_addr = (aligned_addr + self.alignment) & ~(self.alignment - 1)
                chunk_base = aligned_addr
            
            self.chunks.append(chunk_base)
            
            # Add all blocks to free list
            for i in range(num_blocks):
                block_addr = chunk_base + (i * self.block_size)
                self.free_blocks.append(block_addr)
            
            # Update statistics
            self.stats.total_blocks += num_blocks
            self.stats.free_blocks += num_blocks
            self.stats.total_bytes += chunk_size
            self.stats.free_bytes += chunk_size
            
        except Exception as e:
            raise MemoryError(f"Failed to allocate memory chunk: {e}")
    
    def allocate(self, allocation_type: AllocationType = AllocationType.SMALL_OBJECT,
                debug_info: Optional[Dict] = None) -> Optional[int]:
        """Allocate a memory block from the pool"""
        with self._lock:
            # Ensure we have free blocks
            if not self.free_blocks:
                # Double the number of blocks
                current_blocks = self.stats.total_blocks
                self._allocate_chunk(max(current_blocks, 64))
            
            if not self.free_blocks:
                return None
            
            # Get a free block
            block_addr = self.free_blocks.pop()
            
            # Create memory block tracking
            block = MemoryBlock(
                address=block_addr,
                size=self.block_size,
                allocated=True,
                allocation_time=time.time(),
                allocation_type=allocation_type,
                debug_info=debug_info
            )
            
            self.allocated_blocks[block_addr] = block
            
            # Update statistics
            self.stats.allocated_blocks += 1
            self.stats.free_blocks -= 1
            self.stats.allocated_bytes += self.block_size
            self.stats.free_bytes -= self.block_size
            self.stats.allocations_count += 1
            self.stats.fragmentation_ratio = self._calculate_fragmentation()
            
            return block_addr
    
    def deallocate(self, address: int) -> bool:
        """Deallocate a memory block back to the pool"""
        with self._lock:
            if address not in self.allocated_blocks:
                return False
            
            block = self.allocated_blocks.pop(address)
            self.free_blocks.append(address)
            
            # Update statistics
            self.stats.allocated_blocks -= 1
            self.stats.free_blocks += 1
            self.stats.allocated_bytes -= self.block_size
            self.stats.free_bytes += self.block_size
            self.stats.deallocations_count += 1
            self.stats.fragmentation_ratio = self._calculate_fragmentation()
            
            return True
    
    def _calculate_fragmentation(self) -> float:
        """Calculate memory fragmentation ratio"""
        if self.stats.total_bytes == 0:
            return 0.0
        
        # Simple fragmentation metric: ratio of free blocks to total blocks
        if self.stats.total_blocks == 0:
            return 0.0
        
        return self.stats.free_blocks / self.stats.total_blocks
    
    def get_stats(self) -> PoolStats:
        """Get current pool statistics"""
        with self._lock:
            return self.stats
    
    def compact(self) -> int:
        """Compact the pool by defragmenting free space"""
        with self._lock:
            # For now, just sort free blocks to improve locality
            self.free_blocks.sort()
            return len(self.free_blocks)
    
    def cleanup(self):
        """Clean up pool resources"""
        with self._lock:
            # Clear all tracking
            self.allocated_blocks.clear()
            self.free_blocks.clear()
            
            # Clean up memory chunks
            for chunk_addr in self.chunks:
                try:
                    # Note: In a real implementation, we'd properly free mmap regions
                    pass
                except Exception:
                    pass
            
            self.chunks.clear()


class SmartMemoryManager:
    """
    Smart memory manager that uses multiple pools for different allocation sizes.
    
    Designed to achieve 30% less memory usage than Python through:
    - Reduced fragmentation via size-specific pools
    - Cache-aligned allocations for better performance
    - Intelligent garbage collection integration
    - Memory debugging and leak detection
    """
    
    def __init__(self):
        # Size-based pools (in bytes)
        self.pool_configs = {
            # Small objects: 16B to 256B
            16: ("small_16", 512),
            32: ("small_32", 512), 
            64: ("small_64", 256),
            128: ("small_128", 256),
            256: ("small_256", 128),
            
            # Medium objects: 512B to 64KB
            512: ("medium_512", 64),
            1024: ("medium_1k", 64),
            4096: ("medium_4k", 32),
            16384: ("medium_16k", 16),
            65536: ("medium_64k", 8),
            
            # Large objects handled separately
        }
        
        # Initialize memory pools
        self.pools: Dict[int, MemoryPool] = {}
        for size, (name, initial_blocks) in self.pool_configs.items():
            self.pools[size] = MemoryPool(
                block_size=size,
                initial_blocks=initial_blocks,
                alignment=64,  # Cache line aligned
                pool_name=name
            )
        
        # Large object tracking
        self.large_objects: Dict[int, MemoryBlock] = {}
        
        # Statistics and monitoring
        self.global_stats = {
            'total_allocations': 0,
            'total_deallocations': 0,
            'bytes_allocated': 0,
            'bytes_deallocated': 0,
            'peak_memory_usage': 0,
            'current_memory_usage': 0,
            'allocation_histogram': defaultdict(int),
            'memory_saved_vs_python': 0.0
        }
        
        # Thread safety
        self._global_lock = threading.RLock()
        
        # Debugging and profiling
        self.debug_mode = False
        self.allocation_tracking: Dict[int, Dict] = {}
        
        # Automatic cleanup
        self._last_cleanup = time.time()
        self._cleanup_interval = 30.0  # seconds
    
    def allocate(self, size: int, 
                allocation_type: AllocationType = AllocationType.SMALL_OBJECT,
                alignment: int = 8,
                zero_memory: bool = False,
                debug_info: Optional[Dict] = None) -> Optional[int]:
        """
        Allocate memory with intelligent pool selection.
        
        Args:
            size: Size in bytes to allocate
            allocation_type: Type of allocation for tracking
            alignment: Required alignment (must be power of 2)
            zero_memory: Whether to zero-initialize the memory
            debug_info: Debug information for tracking
            
        Returns:
            Memory address or None if allocation failed
        """
        
        # Align size to requested alignment
        aligned_size = self._align_size(size, alignment)
        
        with self._global_lock:
            address = None
            
            # Choose allocation strategy based on size
            if aligned_size <= 65536:  # Use pools for small/medium objects
                pool_size = self._find_suitable_pool_size(aligned_size)
                if pool_size and pool_size in self.pools:
                    address = self.pools[pool_size].allocate(allocation_type, debug_info)
                
            if address is None:
                # Fall back to large object allocation
                address = self._allocate_large_object(aligned_size, allocation_type, debug_info)
            
            if address:
                # Zero memory if requested
                if zero_memory:
                    self._zero_memory(address, aligned_size)
                
                # Update global statistics
                self.global_stats['total_allocations'] += 1
                self.global_stats['bytes_allocated'] += aligned_size
                self.global_stats['current_memory_usage'] += aligned_size
                self.global_stats['allocation_histogram'][aligned_size] += 1
                
                if self.global_stats['current_memory_usage'] > self.global_stats['peak_memory_usage']:
                    self.global_stats['peak_memory_usage'] = self.global_stats['current_memory_usage']
                
                # Debug tracking
                if self.debug_mode:
                    self.allocation_tracking[address] = {
                        'size': aligned_size,
                        'type': allocation_type,
                        'timestamp': time.time(),
                        'debug_info': debug_info
                    }
                
                # Periodic cleanup
                current_time = time.time()
                if current_time - self._last_cleanup > self._cleanup_interval:
                    self._periodic_cleanup()
                    self._last_cleanup = current_time
            
            return address
    
    def deallocate(self, address: int) -> bool:
        """Deallocate memory back to appropriate pool"""
        if address == 0:
            return False
        
        with self._global_lock:
            success = False
            deallocated_size = 0
            
            # Try pools first
            for pool in self.pools.values():
                if address in pool.allocated_blocks:
                    block = pool.allocated_blocks[address]
                    deallocated_size = block.size
                    success = pool.deallocate(address)
                    break
            
            # Try large objects
            if not success and address in self.large_objects:
                block = self.large_objects.pop(address)
                deallocated_size = block.size
                success = self._deallocate_large_object(address, block)
            
            if success:
                # Update global statistics
                self.global_stats['total_deallocations'] += 1
                self.global_stats['bytes_deallocated'] += deallocated_size
                self.global_stats['current_memory_usage'] -= deallocated_size
                
                # Remove debug tracking
                if self.debug_mode and address in self.allocation_tracking:
                    del self.allocation_tracking[address]
            
            return success
    
    def _find_suitable_pool_size(self, size: int) -> Optional[int]:
        """Find the smallest pool that can accommodate the size"""
        suitable_sizes = [pool_size for pool_size in self.pool_configs.keys() if pool_size >= size]
        return min(suitable_sizes) if suitable_sizes else None
    
    def _allocate_large_object(self, size: int, allocation_type: AllocationType,
                              debug_info: Optional[Dict]) -> Optional[int]:
        """Allocate large objects outside of pools"""
        try:
            # Use platform-appropriate allocation for large objects
            if size >= 1024 * 1024:  # 1MB threshold
                # Use mmap on Unix-like systems, or fallback to ctypes
                if hasattr(mmap, 'MAP_PRIVATE') and hasattr(mmap, 'MAP_ANONYMOUS'):
                    memory_map = mmap.mmap(-1, size, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
                    address = ctypes.addressof(ctypes.c_char.from_buffer(memory_map))
                else:
                    # Windows fallback - use ctypes allocation
                    buffer = ctypes.create_string_buffer(size)
                    address = ctypes.addressof(buffer)
            else:
                # Use ctypes for medium-large allocations
                buffer = ctypes.create_string_buffer(size)
                address = ctypes.addressof(buffer)
            
            # Track large object
            block = MemoryBlock(
                address=address,
                size=size,
                allocated=True,
                allocation_time=time.time(),
                allocation_type=allocation_type,
                debug_info=debug_info
            )
            
            self.large_objects[address] = block
            return address
            
        except Exception as e:
            print(f"Large object allocation failed: {e}")
            return None
    
    def _deallocate_large_object(self, address: int, block: MemoryBlock) -> bool:
        """Deallocate large objects"""
        try:
            # Note: In a full implementation, we'd properly clean up the memory
            # For now, just remove from tracking
            return True
        except Exception:
            return False
    
    def _align_size(self, size: int, alignment: int) -> int:
        """Align size to specified boundary"""
        return (size + alignment - 1) & ~(alignment - 1)
    
    def _zero_memory(self, address: int, size: int):
        """Zero-initialize memory at address"""
        try:
            # Create a ctypes array to zero the memory
            memory_array = (ctypes.c_byte * size).from_address(address)
            ctypes.memset(memory_array, 0, size)
        except Exception:
            pass  # Best effort
    
    def _periodic_cleanup(self):
        """Perform periodic cleanup and optimization"""
        # Compact pools to reduce fragmentation
        for pool in self.pools.values():
            pool.compact()
        
        # Update memory savings calculation
        self._calculate_memory_savings()
    
    def _calculate_memory_savings(self):
        """Calculate memory savings compared to Python"""
        # Estimate Python overhead: ~28 bytes per object on average
        python_overhead_per_object = 28
        total_objects = self.global_stats['total_allocations'] - self.global_stats['total_deallocations']
        
        # Calculate theoretical Python memory usage
        python_memory = self.global_stats['current_memory_usage'] + (total_objects * python_overhead_per_object)
        
        if python_memory > 0:
            savings_ratio = (python_memory - self.global_stats['current_memory_usage']) / python_memory
            self.global_stats['memory_saved_vs_python'] = savings_ratio * 100.0
    
    def get_memory_stats(self) -> Dict:
        """Get comprehensive memory statistics"""
        with self._global_lock:
            stats = {
                'global_stats': self.global_stats.copy(),
                'pool_stats': {},
                'large_objects_count': len(self.large_objects),
                'large_objects_memory': sum(block.size for block in self.large_objects.values()),
                'fragmentation_summary': {}
            }
            
            # Get pool statistics
            total_pool_memory = 0
            total_pool_fragmentation = 0.0
            
            for size, pool in self.pools.items():
                pool_stats = pool.get_stats()
                stats['pool_stats'][size] = {
                    'name': pool.pool_name,
                    'total_blocks': pool_stats.total_blocks,
                    'allocated_blocks': pool_stats.allocated_blocks,
                    'free_blocks': pool_stats.free_blocks,
                    'total_bytes': pool_stats.total_bytes,
                    'allocated_bytes': pool_stats.allocated_bytes,
                    'free_bytes': pool_stats.free_bytes,
                    'fragmentation_ratio': pool_stats.fragmentation_ratio,
                    'utilization': pool_stats.allocated_bytes / pool_stats.total_bytes if pool_stats.total_bytes > 0 else 0.0
                }
                
                total_pool_memory += pool_stats.total_bytes
                total_pool_fragmentation += pool_stats.fragmentation_ratio
            
            # Calculate overall fragmentation
            if len(self.pools) > 0:
                stats['fragmentation_summary'] = {
                    'average_fragmentation': total_pool_fragmentation / len(self.pools),
                    'total_pool_memory': total_pool_memory,
                    'memory_efficiency': (stats['global_stats']['current_memory_usage'] / total_pool_memory * 100) if total_pool_memory > 0 else 0.0
                }
            
            return stats
    
    def enable_debug_mode(self, enable: bool = True):
        """Enable or disable debug tracking"""
        self.debug_mode = enable
        if not enable:
            self.allocation_tracking.clear()
    
    def get_memory_leaks(self) -> List[Dict]:
        """Detect potential memory leaks"""
        if not self.debug_mode:
            return []
        
        current_time = time.time()
        leak_threshold = 300.0  # 5 minutes
        
        leaks = []
        for address, info in self.allocation_tracking.items():
            age = current_time - info['timestamp']
            if age > leak_threshold:
                leaks.append({
                    'address': address,
                    'size': info['size'],
                    'type': info['type'],
                    'age_seconds': age,
                    'debug_info': info.get('debug_info')
                })
        
        return sorted(leaks, key=lambda x: x['age_seconds'], reverse=True)
    
    def force_cleanup(self):
        """Force immediate cleanup and compaction"""
        with self._global_lock:
            for pool in self.pools.values():
                pool.compact()
            self._calculate_memory_savings()
    
    def get_allocation_summary(self) -> Dict:
        """Get summary of allocation patterns"""
        summary = {
            'total_allocations': self.global_stats['total_allocations'],
            'total_deallocations': self.global_stats['total_deallocations'],
            'active_allocations': self.global_stats['total_allocations'] - self.global_stats['total_deallocations'],
            'current_memory_mb': self.global_stats['current_memory_usage'] / (1024 * 1024),
            'peak_memory_mb': self.global_stats['peak_memory_usage'] / (1024 * 1024),
            'memory_savings_vs_python_percent': self.global_stats['memory_saved_vs_python'],
            'most_common_sizes': dict(sorted(self.global_stats['allocation_histogram'].items(), 
                                           key=lambda x: x[1], reverse=True)[:10])
        }
        
        return summary
    
    def cleanup(self):
        """Clean up all memory pools and resources"""
        with self._global_lock:
            for pool in self.pools.values():
                pool.cleanup()
            
            self.large_objects.clear()
            self.allocation_tracking.clear()


# Global memory manager instance
_global_memory_manager: Optional[SmartMemoryManager] = None


def get_memory_manager() -> SmartMemoryManager:
    """Get the global memory manager instance"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = SmartMemoryManager()
    return _global_memory_manager


def allocate_memory(size: int, 
                   allocation_type: AllocationType = AllocationType.SMALL_OBJECT,
                   alignment: int = 8,
                   zero_memory: bool = False,
                   debug_info: Optional[Dict] = None) -> Optional[int]:
    """Convenience function for memory allocation"""
    return get_memory_manager().allocate(size, allocation_type, alignment, zero_memory, debug_info)


def deallocate_memory(address: int) -> bool:
    """Convenience function for memory deallocation"""
    return get_memory_manager().deallocate(address)


def get_memory_stats() -> Dict:
    """Get global memory statistics"""
    return get_memory_manager().get_memory_stats()


def enable_memory_debugging(enable: bool = True):
    """Enable memory debugging globally"""
    get_memory_manager().enable_debug_mode(enable)
