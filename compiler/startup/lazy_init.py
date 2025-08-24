"""
NeuralScript Lazy Initialization System
======================================

Implements lazy loading and deferred initialization for non-critical
components to achieve <100ms startup time target.

Features:
- Lazy loading for heavy components (JIT, SIMD, Memory systems)
- Smart component dependency resolution
- Parallel initialization for independent components
- Startup cache for frequently used configurations
"""

import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Callable, Type, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
import weakref
from enum import Enum, auto


class InitializationPriority(Enum):
    """Component initialization priority levels"""
    IMMEDIATE = auto()      # Must be initialized at startup
    CRITICAL = auto()       # Needed soon, initialize early
    NORMAL = auto()         # Standard priority
    LAZY = auto()           # Only initialize when needed
    BACKGROUND = auto()     # Can initialize in background


@dataclass
class ComponentMetadata:
    """Metadata for lazy-loadable components"""
    name: str
    priority: InitializationPriority
    dependencies: List[str] = field(default_factory=list)
    estimated_init_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    initialization_function: Optional[Callable] = None
    is_thread_safe: bool = True


class LazyComponent:
    """
    Wrapper for lazy-initialized components
    
    Provides transparent access to components that are initialized
    only when first accessed.
    """
    
    def __init__(self, component_name: str, init_function: Callable, 
                 priority: InitializationPriority = InitializationPriority.LAZY):
        self.component_name = component_name
        self.init_function = init_function
        self.priority = priority
        self._instance = None
        self._is_initialized = False
        self._initialization_time_ms = 0.0
        self._lock = threading.RLock()
        
        # Register with lazy initialization manager
        LazyInitManager.get_instance().register_component(self)
    
    @property
    def is_initialized(self) -> bool:
        """Check if component is initialized"""
        return self._is_initialized
    
    @property
    def initialization_time_ms(self) -> float:
        """Get time taken to initialize component"""
        return self._initialization_time_ms
    
    def get_instance(self):
        """Get component instance, initializing if necessary"""
        if not self._is_initialized:
            with self._lock:
                if not self._is_initialized:  # Double-checked locking
                    start_time = time.perf_counter()
                    
                    print(f"🔄 Lazy loading {self.component_name}...")
                    self._instance = self.init_function()
                    
                    self._initialization_time_ms = (time.perf_counter() - start_time) * 1000
                    self._is_initialized = True
                    
                    print(f"   ✅ {self.component_name} initialized in {self._initialization_time_ms:.1f}ms")
        
        return self._instance
    
    def force_initialize(self):
        """Force initialization without returning instance"""
        self.get_instance()
        
    def __getattr__(self, name):
        """Transparent access to component methods/attributes"""
        instance = self.get_instance()
        return getattr(instance, name)


class LazyInitManager:
    """
    Manager for lazy initialization system
    
    Coordinates component initialization, handles dependencies,
    and optimizes startup performance.
    """
    
    _instance: Optional['LazyInitManager'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        self.components: Dict[str, LazyComponent] = {}
        self.metadata: Dict[str, ComponentMetadata] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.initialization_order: List[str] = []
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="LazyInit")
        
        # Performance tracking
        self.total_saved_time_ms = 0.0
        self.components_lazy_loaded = 0
        
    @classmethod
    def get_instance(cls) -> 'LazyInitManager':
        """Get singleton instance of lazy init manager"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def register_component(self, component: LazyComponent, 
                          dependencies: List[str] = None,
                          estimated_init_time_ms: float = 10.0,
                          memory_usage_mb: float = 1.0):
        """Register a component for lazy initialization"""
        
        self.components[component.component_name] = component
        
        metadata = ComponentMetadata(
            name=component.component_name,
            priority=component.priority,
            dependencies=dependencies or [],
            estimated_init_time_ms=estimated_init_time_ms,
            memory_usage_mb=memory_usage_mb,
            initialization_function=component.init_function
        )
        self.metadata[component.component_name] = metadata
        
        # Update dependency graph
        self.dependency_graph[component.component_name] = dependencies or []
    
    def get_initialization_order(self) -> List[str]:
        """Get optimal initialization order based on dependencies"""
        if self.initialization_order:
            return self.initialization_order
        
        # Topological sort of dependency graph
        visited = set()
        temp_visited = set()
        order = []
        
        def dfs(component_name: str):
            if component_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {component_name}")
            if component_name in visited:
                return
            
            temp_visited.add(component_name)
            
            for dependency in self.dependency_graph.get(component_name, []):
                if dependency in self.components:
                    dfs(dependency)
            
            temp_visited.remove(component_name)
            visited.add(component_name)
            order.append(component_name)
        
        for component_name in self.components:
            if component_name not in visited:
                dfs(component_name)
        
        self.initialization_order = order
        return order
    
    def initialize_critical_components(self) -> float:
        """Initialize only critical components needed for startup"""
        
        start_time = time.perf_counter()
        critical_components = []
        
        # Find critical and immediate priority components
        for name, component in self.components.items():
            if component.priority in [InitializationPriority.IMMEDIATE, InitializationPriority.CRITICAL]:
                critical_components.append(name)
        
        # Initialize in dependency order
        order = self.get_initialization_order()
        for component_name in order:
            if component_name in critical_components:
                self.components[component_name].force_initialize()
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"⚡ Critical components initialized in {elapsed_ms:.1f}ms")
        
        return elapsed_ms
    
    def initialize_background_components(self):
        """Initialize background components in parallel"""
        
        background_components = []
        for name, component in self.components.items():
            if (component.priority == InitializationPriority.BACKGROUND and 
                not component.is_initialized):
                background_components.append(name)
        
        if not background_components:
            return
        
        print(f"🔄 Starting background initialization of {len(background_components)} components...")
        
        # Submit initialization tasks
        futures = []
        for component_name in background_components:
            future = self.thread_pool.submit(
                self.components[component_name].force_initialize
            )
            futures.append((component_name, future))
        
        # Don't wait for completion - they'll finish in background
        print("   Background initialization started")
    
    def get_startup_statistics(self) -> Dict[str, Any]:
        """Get startup performance statistics"""
        
        initialized_components = [c for c in self.components.values() if c.is_initialized]
        lazy_components = [c for c in self.components.values() if not c.is_initialized]
        
        total_init_time = sum(c.initialization_time_ms for c in initialized_components)
        estimated_saved_time = sum(
            self.metadata[c.component_name].estimated_init_time_ms 
            for c in lazy_components
        )
        
        return {
            'total_components': len(self.components),
            'initialized_components': len(initialized_components),
            'lazy_components': len(lazy_components),
            'total_initialization_time_ms': total_init_time,
            'estimated_time_saved_ms': estimated_saved_time,
            'lazy_loading_efficiency': estimated_saved_time / (total_init_time + estimated_saved_time) if total_init_time > 0 else 0
        }


# Lazy-loaded component instances for NeuralScript systems
class NeuralScriptLazyComponents:
    """Lazy-loaded NeuralScript system components"""
    
    def __init__(self):
        self._memory_manager = None
        self._simd_system = None  
        self._jit_compiler = None
        self._neural_network_framework = None
        
    @property
    def memory_manager(self):
        """Lazy-loaded memory management system"""
        if self._memory_manager is None:
            self._memory_manager = LazyComponent(
                "Memory Manager",
                self._init_memory_manager,
                InitializationPriority.NORMAL
            )
        return self._memory_manager.get_instance()
    
    @property
    def simd_system(self):
        """Lazy-loaded SIMD acceleration system"""
        if self._simd_system is None:
            self._simd_system = LazyComponent(
                "SIMD System",
                self._init_simd_system,
                InitializationPriority.LAZY
            )
        return self._simd_system.get_instance()
    
    @property 
    def jit_compiler(self):
        """Lazy-loaded JIT compilation system"""
        if self._jit_compiler is None:
            self._jit_compiler = LazyComponent(
                "JIT Compiler",
                self._init_jit_compiler,
                InitializationPriority.BACKGROUND
            )
        return self._jit_compiler.get_instance()
    
    @property
    def neural_network_framework(self):
        """Lazy-loaded neural network training framework"""
        if self._neural_network_framework is None:
            self._neural_network_framework = LazyComponent(
                "Neural Network Framework",
                self._init_neural_network_framework,
                InitializationPriority.LAZY
            )
        return self._neural_network_framework.get_instance()
    
    def _init_memory_manager(self):
        """Initialize memory management system"""
        try:
            from ..memory.memory_manager import get_memory_manager
            return get_memory_manager()
        except ImportError:
            return None
    
    def _init_simd_system(self):
        """Initialize SIMD acceleration system"""
        try:
            from ..simd.simd_core import SIMDEngine
            return SIMDEngine()
        except ImportError:
            return None
    
    def _init_jit_compiler(self):
        """Initialize JIT compilation system"""
        try:
            from ..jit.jit_integration import get_integrated_jit_compiler
            return get_integrated_jit_compiler()
        except ImportError:
            return None
    
    def _init_neural_network_framework(self):
        """Initialize neural network framework"""
        try:
            from ..ml.neural_network import NeuralNetwork
            return NeuralNetwork
        except ImportError:
            return None


# Global lazy components instance
_lazy_components: Optional[NeuralScriptLazyComponents] = None


def get_lazy_components() -> NeuralScriptLazyComponents:
    """Get global lazy components instance"""
    global _lazy_components
    if _lazy_components is None:
        _lazy_components = NeuralScriptLazyComponents()
    return _lazy_components


class FastStartup:
    """
    Fast startup system for NeuralScript
    
    Implements optimized startup path with lazy loading,
    caching, and parallel initialization.
    """
    
    def __init__(self, enable_profiling: bool = True):
        self.enable_profiling = enable_profiling
        
        # Import profiler if needed
        if enable_profiling:
            try:
                from .startup_profiler import get_startup_profiler
                self.profiler = get_startup_profiler()
            except ImportError:
                self.profiler = None
        else:
            self.profiler = None
            
        self.lazy_manager = LazyInitManager.get_instance()
        
    def fast_startup_sequence(self) -> Dict[str, Any]:
        """Execute optimized startup sequence"""
        
        if self.profiler:
            with self.profiler.profile_phase("Fast Startup", description="Optimized startup sequence"):
                return self._execute_fast_startup()
        else:
            return self._execute_fast_startup()
    
    def _execute_fast_startup(self) -> Dict[str, Any]:
        """Internal fast startup implementation"""
        
        startup_start = time.perf_counter()
        
        # Phase 1: Immediate essentials only
        if self.profiler:
            with self.profiler.profile_phase("Core System", critical_path=True, description="Essential system components"):
                self._init_core_system()
        else:
            self._init_core_system()
        
        # Phase 2: Critical components with dependency resolution
        critical_init_time = self.lazy_manager.initialize_critical_components()
        
        # Phase 3: Start background initialization
        self.lazy_manager.initialize_background_components()
        
        total_startup_time = (time.perf_counter() - startup_start) * 1000
        
        # Get statistics
        stats = self.lazy_manager.get_startup_statistics()
        
        result = {
            'total_startup_time_ms': total_startup_time,
            'critical_init_time_ms': critical_init_time,
            'target_achieved': total_startup_time < 100.0,
            'lazy_loading_stats': stats,
            'components_ready': stats['initialized_components'],
            'components_pending': stats['lazy_components']
        }
        
        print(f"🚀 Fast startup completed in {total_startup_time:.1f}ms")
        print(f"   Critical path: {critical_init_time:.1f}ms")
        print(f"   Components ready: {result['components_ready']}")
        print(f"   Components pending: {result['components_pending']}")
        print(f"   Target achieved: {'✅' if result['target_achieved'] else '❌'}")
        
        return result
    
    def _init_core_system(self):
        """Initialize only the most essential components"""
        
        # Register lazy components with the system
        lazy_components = get_lazy_components()
        
        # Register memory manager (normal priority - needed relatively soon)
        self.lazy_manager.register_component(
            LazyComponent("Memory Manager", lazy_components._init_memory_manager, InitializationPriority.NORMAL),
            dependencies=[],
            estimated_init_time_ms=20.0,
            memory_usage_mb=5.0
        )
        
        # Register SIMD system (lazy - only needed for compute-heavy operations)
        self.lazy_manager.register_component(
            LazyComponent("SIMD System", lazy_components._init_simd_system, InitializationPriority.LAZY),
            dependencies=[],
            estimated_init_time_ms=10.0,
            memory_usage_mb=2.0
        )
        
        # Register JIT compiler (background - can initialize while program runs)
        self.lazy_manager.register_component(
            LazyComponent("JIT Compiler", lazy_components._init_jit_compiler, InitializationPriority.BACKGROUND),
            dependencies=[],
            estimated_init_time_ms=30.0,
            memory_usage_mb=10.0
        )
        
        # Register neural network framework (lazy - only for ML workloads)
        self.lazy_manager.register_component(
            LazyComponent("Neural Network Framework", lazy_components._init_neural_network_framework, InitializationPriority.LAZY),
            dependencies=["Memory Manager", "SIMD System"],
            estimated_init_time_ms=15.0,
            memory_usage_mb=8.0
        )


def create_optimized_startup_system() -> FastStartup:
    """Create optimized startup system with lazy loading"""
    return FastStartup(enable_profiling=True)


# Convenience functions for accessing lazy-loaded systems
def get_memory_manager_lazy():
    """Get memory manager with lazy initialization"""
    return get_lazy_components().memory_manager


def get_simd_system_lazy():
    """Get SIMD system with lazy initialization"""
    return get_lazy_components().simd_system


def get_jit_compiler_lazy():
    """Get JIT compiler with lazy initialization"""
    return get_lazy_components().jit_compiler


def get_neural_network_framework_lazy():
    """Get neural network framework with lazy initialization"""
    return get_lazy_components().neural_network_framework


if __name__ == "__main__":
    # Test lazy initialization system
    print("🧪 Testing Lazy Initialization System")
    print("=" * 40)
    
    # Create fast startup system
    startup = create_optimized_startup_system()
    
    # Execute fast startup
    results = startup.fast_startup_sequence()
    
    # Test lazy loading
    print("\n🔄 Testing lazy component access...")
    
    # Access memory manager (should initialize)
    memory_mgr = get_memory_manager_lazy()
    print(f"Memory manager available: {memory_mgr is not None}")
    
    # Access neural network framework (should initialize with dependencies)
    nn_framework = get_neural_network_framework_lazy()  
    print(f"Neural network framework available: {nn_framework is not None}")
    
    # Get final statistics
    final_stats = LazyInitManager.get_instance().get_startup_statistics()
    
    print(f"\n📊 Final Statistics:")
    print(f"   Total components: {final_stats['total_components']}")
    print(f"   Initialized: {final_stats['initialized_components']}")
    print(f"   Still lazy: {final_stats['lazy_components']}")
    print(f"   Time saved by lazy loading: {final_stats['estimated_time_saved_ms']:.1f}ms")
    print(f"   Lazy loading efficiency: {final_stats['lazy_loading_efficiency']:.1%}")
    
    # Export profile
    if startup.profiler:
        startup.profiler.export_profile("fast_startup_profile.json")
