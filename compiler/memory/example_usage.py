"""
Example Usage of NeuralScript Memory Management System

This example demonstrates how to use the complete memory management
system including garbage collection, profiling, and optimization.
"""

import time
from compiler.memory import (
    GarbageCollector, GCConfiguration, GCMode,
    ObjectType, ProfilingLevel
)


def basic_usage_example():
    """Basic usage example showing core GC functionality"""
    print("=== Basic Memory Management Example ===")
    
    # Configure the garbage collector
    config = GCConfiguration(
        num_generations=3,
        young_gen_size_mb=64,
        old_gen_size_mb=256,
        enable_profiling=True,
        profiling_level=ProfilingLevel.DETAILED,
        mode=GCMode.AUTOMATIC
    )
    
    # Create the garbage collector
    gc = GarbageCollector(config)
    print(f"Created GC with {config.num_generations} generations")
    
    try:
        # Simulate some allocations
        print("\nSimulating allocations...")
        
        # Allocate different types of objects
        addresses = []
        
        # Small scalar objects (frequent, short-lived)
        for i in range(1000):
            addr = gc.allocate(ObjectType.SCALAR, 8)  # 8-byte scalar
            if addr:
                addresses.append(addr)
        
        # Medium vector objects
        for i in range(100):
            addr = gc.allocate(ObjectType.VECTOR, 1024)  # 1KB vector
            if addr:
                addresses.append(addr)
        
        # Large array objects
        for i in range(10):
            addr = gc.allocate(ObjectType.ARRAY, 1024 * 1024)  # 1MB array
            if addr:
                addresses.append(addr)
        
        print(f"Allocated {len(addresses)} objects")
        
        # Get statistics
        stats = gc.get_statistics()
        print(f"\nMemory Statistics:")
        print(f"  Total objects: {stats.total_objects}")
        print(f"  Heap size: {stats.heap_size_bytes / (1024*1024):.2f} MB")
        print(f"  Used memory: {stats.used_bytes / (1024*1024):.2f} MB")
        print(f"  Collections: {stats.total_collections}")
        print(f"  Average pause: {stats.average_pause_time_ms:.2f} ms")
        
        # Trigger explicit collection
        print("\nTriggering explicit collection...")
        metrics = gc.collect(explicit=True)
        
        for metric in metrics:
            print(f"  Gen {metric.generation}: {metric.bytes_collected} bytes collected "
                  f"in {metric.pause_time_ms:.2f} ms")
        
        # Get detailed statistics
        detailed_stats = gc.get_detailed_statistics()
        print(f"\nDetailed Statistics:")
        print(f"  System memory: {detailed_stats['system_stats'].get('system_memory_percent', 0):.1f}% used")
        print(f"  Allocation rate: {detailed_stats['allocation_stats'].get('allocation_rate_bytes_per_sec', 0) / (1024*1024):.2f} MB/s")
        
        # Generate profiling report if enabled
        if gc.memory_profiler:
            print("\nMemory Profiling Report:")
            report = gc.memory_profiler.generate_report()
            print(report[:500] + "..." if len(report) > 500 else report)
        
    finally:
        # Clean shutdown
        gc.shutdown()
        print("\nGC shutdown complete")


def optimization_example():
    """Example showing automatic memory optimization"""
    print("\n=== Memory Optimization Example ===")
    
    # Configure for optimization
    config = GCConfiguration(
        mode=GCMode.ADAPTIVE,
        enable_profiling=True,
        optimize_allocation_patterns=True,
        profiling_level=ProfilingLevel.COMPREHENSIVE
    )
    
    gc = GarbageCollector(config)
    
    try:
        print("Simulating workload patterns...")
        
        # Simulate streaming data pattern
        print("  - Streaming data pattern")
        for batch in range(50):
            batch_addresses = []
            for i in range(100):
                addr = gc.allocate(ObjectType.VECTOR, 512)
                if addr:
                    batch_addresses.append(addr)
                    # Record access pattern
                    gc.memory_optimizer.record_access(addr, 'write')
            
            # Process batch (sequential access)
            for addr in batch_addresses:
                gc.memory_optimizer.record_access(addr, 'read')
            
            time.sleep(0.001)  # Small delay to simulate processing
        
        # Simulate large object pattern
        print("  - Large object pattern")
        large_objects = []
        for i in range(20):
            addr = gc.allocate(ObjectType.ARRAY, 2 * 1024 * 1024)  # 2MB
            if addr:
                large_objects.append(addr)
                gc.memory_optimizer.record_access(addr, 'write')
        
        # Analyze patterns and get recommendations
        print("\nAnalyzing allocation patterns...")
        recommendations = gc.memory_optimizer.get_optimization_recommendations()
        
        print(f"Found {len(recommendations)} optimization opportunities:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"  {i}. {rec.strategy.name}")
            print(f"     Confidence: {rec.confidence:.1%}")
            print(f"     Est. improvement: {rec.estimated_improvement:.1f}x")
            print(f"     Reason: {rec.reasoning[:100]}...")
        
        # Apply optimizations
        print("\nApplying automatic optimizations...")
        applied = gc.memory_optimizer.optimize()
        
        if applied:
            print(f"Applied optimizations: {', '.join(applied)}")
        else:
            print("No optimizations applied (may need more data)")
        
        # Show optimization report
        opt_report = gc.memory_optimizer.get_optimization_report()
        print(f"\nOptimization Report:\n{opt_report}")
        
    finally:
        gc.shutdown()
        print("Optimization example complete")


def workload_specific_example():
    """Example showing workload-specific optimization"""
    print("\n=== Workload-Specific Optimization Example ===")
    
    # Configure for machine learning workload
    config = GCConfiguration(
        young_gen_size_mb=128,
        old_gen_size_mb=512,
        max_pause_time_ms=2.0,
        enable_incremental=True
    )
    
    gc = GarbageCollector(config)
    
    try:
        # Optimize for machine learning workload
        print("Optimizing for machine learning workload...")
        gc.optimize_for_workload(['machine_learning', 'scientific_computing'])
        
        # Simulate tensor operations
        print("Simulating tensor allocations...")
        
        tensors = []
        for epoch in range(10):
            # Allocate tensors for this epoch
            epoch_tensors = []
            
            # Input tensors
            for batch in range(32):
                addr = gc.allocate(ObjectType.ARRAY, 1024 * 1024)  # 1MB tensor
                if addr:
                    epoch_tensors.append(addr)
            
            # Intermediate tensors (short-lived)
            for layer in range(5):
                addr = gc.allocate(ObjectType.ARRAY, 512 * 1024)  # 512KB
                if addr:
                    epoch_tensors.append(addr)
            
            tensors.extend(epoch_tensors)
            
            # Simulate some processing time
            time.sleep(0.01)
        
        # Check memory efficiency
        stats = gc.get_statistics()
        print(f"\nTensor Training Statistics:")
        print(f"  Memory efficiency: {(stats.used_bytes/stats.heap_size_bytes)*100:.1f}%")
        print(f"  Average pause: {stats.average_pause_time_ms:.2f} ms")
        print(f"  Memory pressure: {stats.memory_pressure:.1%}")
        
        if stats.average_pause_time_ms > config.max_pause_time_ms:
            print("  ⚠️  Pause times exceed target - may need tuning")
        else:
            print("  ✓ Pause times within target")
        
    finally:
        gc.shutdown()
        print("Workload-specific example complete")


def performance_monitoring_example():
    """Example showing performance monitoring capabilities"""
    print("\n=== Performance Monitoring Example ===")
    
    config = GCConfiguration(
        enable_profiling=True,
        track_allocation_sites=True,
        profiling_level=ProfilingLevel.COMPREHENSIVE
    )
    
    gc = GarbageCollector(config)
    
    try:
        # Simulate a memory leak scenario
        print("Simulating memory allocation patterns...")
        
        persistent_objects = []  # Simulate memory leak
        
        for iteration in range(100):
            # Normal allocations (should be collected)
            temp_objects = []
            for i in range(50):
                addr = gc.allocate(ObjectType.SCALAR, 64)
                if addr:
                    temp_objects.append(addr)
            
            # Some objects that persist (simulate leak)
            if iteration % 10 == 0:
                addr = gc.allocate(ObjectType.STRING, 1024)
                if addr:
                    persistent_objects.append(addr)
            
            time.sleep(0.001)
        
        # Analyze for memory leaks
        print("Analyzing for memory leaks...")
        
        if gc.memory_profiler:
            leaks = gc.memory_profiler.get_memory_leaks()
            
            if leaks:
                print(f"Potential memory leaks detected: {len(leaks)}")
                for site, leaked_bytes in leaks[:3]:
                    print(f"  {site.function_name} ({site.filename}:{site.line_number})")
                    print(f"    Leaked: {leaked_bytes / 1024:.1f} KB")
            else:
                print("No significant memory leaks detected")
            
            # Get allocation patterns
            patterns = gc.memory_profiler.get_allocation_patterns()
            
            print(f"\nAllocation Patterns:")
            if 'allocations_by_type' in patterns:
                for obj_type, count in patterns['allocations_by_type'].items():
                    print(f"  {obj_type}: {count} allocations")
            
            # Export detailed data
            profiling_data = gc.memory_profiler.export_data('json')
            print(f"\nProfiling data size: {len(profiling_data)} bytes")
        
    finally:
        gc.shutdown()
        print("Performance monitoring example complete")


def main():
    """Run all examples"""
    print("NeuralScript Memory Management System Examples")
    print("=" * 50)
    
    try:
        basic_usage_example()
        optimization_example()
        workload_specific_example()
        performance_monitoring_example()
        
        print(f"\n{'='*50}")
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
