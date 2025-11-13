# mypy: allow-untyped-defs
"""Device-agnostic profiler backend support.

This module provides a pluggable profiler backend system that enables
out-of-tree device backends to integrate with PyTorch's profiler without
modifying core PyTorch code or resorting to monkey-patching.

Architecture:
    For **Kineto-supported devices** (CUDA, XPU, MTIA, HPU):
        - ProfilerBackend wrappers are auto-registered at import
        - They delegate entirely to torch.autograd.profiler (Kineto)
        - This guarantees 100% backward compatibility
        - Only synchronize() is called (to flush device operations)
    
    For **legacy ProfilerStubs devices** (e.g., OpenReg, existing NPU):
        - If a backend already registered ProfilerStubs via registerPrivateUse1Methods()
        - Automatically wrapped with ProfilerStubsAdapter in C++
        - Existing code continues to work without modification
        - Adapters expose ProfilerStubs through ProfilerBackend interface
    
    For **new custom PrivateUse1 devices**:
        - Backends implement full ProfilerBackend interface
        - prepare() initializes device-specific profiler
        - start() begins recording device events
        - stop() finalizes recording
        - get_results() returns custom profiling data
        - synchronize() flushes device operations

Backward Compatibility:
    The system supports THREE integration methods simultaneously:
    
    1. **ProfilerStubs (KINETO_PRIVATEUSE1_FALLBACK mode)**:
       - Legacy low-level event recording via registerPrivateUse1Methods()
       - Automatically wrapped with ProfilerStubsAdapter
       - Example: OpenReg backend
       
    2. **Kineto native backends**:
       - CUDA, XPU, MTIA, HPU via torch.autograd.profiler
       - Wrapped with _KinetoBackendWrapper for API consistency
       
    3. **New ProfilerBackend implementations**:
       - Modern Python or C++ ProfilerBackend implementations
       - Full control over profiling lifecycle
       - Example: New NPU backend with rich profiling features

The key components are:
    - ProfilerBackend: Abstract base class defining the profiler interface
    - DeviceProfilerRegistry: Global registry for backend registration
    - get_profiler_backend: Helper to select appropriate backend
    - ProfilerStubsAdapter: C++ adapter wrapping legacy ProfilerStubs

Example:
    Register a modern custom backend for PrivateUse1 device::

        from torch.profiler.backend import (
            ProfilerBackend,
            DeviceProfilerRegistry
        )

        class NPUProfiler(ProfilerBackend):
            def device_type(self):
                return "npu"  # torch._C._get_privateuse1_backend_name()

            def is_available(self):
                return torch.npu.is_available()

            def prepare(self, config):
                self.profiler = NPUProfilerImpl(config)

            def start(self):
                self.profiler.start_recording()

            def stop(self):
                self.profiler.stop_recording()

            def get_results(self):
                return {
                    "events": self.profiler.get_events(),
                    "kernel_count": self.profiler.get_kernel_count()
                }
            
            def synchronize(self):
                torch.npu.synchronize()

        # Register once at module initialization
        DeviceProfilerRegistry.register_backend("npu", NPUProfiler())

    Or continue using legacy ProfilerStubs (automatically wrapped)::

        // C++ code in your device extension
        struct MyDeviceMethods : public torch::profiler::impl::ProfilerStubs {
            void record(...) override { /* device event recording */ }
            float elapsed(...) override { /* elapsed time */ }
            void synchronize() override { /* device sync */ }
            // ... other methods
        };
        
        static MyDeviceMethods methods;
        torch::profiler::impl::registerPrivateUse1Methods(&methods);
        
        // Automatically wrapped with ProfilerStubsAdapter!
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Optional, Set

import torch
from torch._C import _get_privateuse1_backend_name
from torch.autograd import kineto_available, ProfilerActivity


__all__ = [
    "ProfilerBackend",
    "DeviceProfilerRegistry",
    "get_profiler_backend",
]


class ProfilerBackend(ABC):
    """Abstract base class for device-specific profiler backends.

    Out-of-tree hardware backends should subclass this interface and
    register an instance via :class:`DeviceProfilerRegistry` to provide
    custom profiling support for their devices.

    The lifecycle of a profiler backend is:
        1. prepare() - Initialize with configuration
        2. start() - Begin recording profiling events
        3. stop() - Finish recording
        4. get_results() - Retrieve profiling data
        5. (optional) export_trace() - Export to file format

    Example:
        Implementing a custom device profiler::

            class NPUProfiler(ProfilerBackend):
                def device_type(self):
                    return "npu"

                def is_available(self):
                    return torch.npu.is_available()

                def prepare(self, config):
                    self.npu_profiler = NPUProfilerImpl(
                        record_shapes=config.get("record_shapes", False)
                    )

                def start(self):
                    self.npu_profiler.start()

                def stop(self):
                    self.npu_profiler.stop()

                def get_results(self):
                    return {"events": self.npu_profiler.get_events()}

                def synchronize(self):
                    torch.npu.synchronize()

            # Register once at module import
            DeviceProfilerRegistry.register_backend("npu", NPUProfiler())
    """
    
    @abstractmethod
    def device_type(self) -> str:
        """Return the device type string this backend handles.

        Returns:
            str: Device type identifier (e.g., 'cuda', 'xpu', 'npu').
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend's device is available on the system.

        Returns:
            bool: True if the device is available and functional.
        """
        pass
    
    @abstractmethod
    def prepare(self, config: dict[str, Any]) -> None:
        """Prepare the profiler with the given configuration.
        
        Called once before start() to initialize the profiling session.

        Args:
            config (dict[str, Any]): Profiler configuration dictionary.
                Common keys include:
                
                - record_shapes (bool): Record tensor shapes
                - profile_memory (bool): Profile memory usage
                - with_stack (bool): Record Python stack traces
                - with_flops (bool): Estimate FLOPs
                - with_modules (bool): Record module hierarchy
                - activities (set of ProfilerActivity): Activities to profile
        """
        pass
    
    @abstractmethod
    def start(self) -> None:
        """Start profiling and begin recording events.

        Called after prepare() to begin the profiling session.
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop profiling and finalize event recording.

        Called to end the profiling session. After this, get_results()
        can be called to retrieve the profiling data.
        """
        pass
    
    @abstractmethod
    def get_results(self) -> dict[str, Any]:
        """Retrieve profiling results after profiling has stopped.

        Returns:
            dict[str, Any]: Profiling results. The structure is
                backend-specific but typically includes:
                
                - events (list): List of profiling events
                - metadata (dict): Additional profiling metadata
                - trace (dict or str): Trace data for export
        """
        pass
    
    def export_trace(self, path: str) -> bool:
        """Export profiling trace to a file.

        Override this method to provide custom trace export functionality
        (e.g., Chrome trace format, custom formats).

        Args:
            path (str): Filesystem path where trace should be written.
            
        Returns:
            bool: True if export succeeded, False otherwise.

        Note:
            Default implementation returns False (no export support).
        """
        return False
    
    def synchronize(self) -> None:
        """Synchronize device to ensure all operations complete.

        Override this method if your device requires explicit
        synchronization to flush pending operations before
        collecting profiling results.

        Note:
            Default implementation is a no-op.
        """
        pass


class DeviceProfilerRegistry:
    """Global registry for device-specific profiler backends.

    This registry enables out-of-tree device backends to integrate
    profiling support without modifying PyTorch core code. Backends
    register once (typically at module import) and are automatically
    used by the profiler when their device type is profiled.

    Thread Safety:
        The registry uses class-level storage and is not explicitly
        thread-safe. Register backends during module initialization
        before multi-threaded execution begins.

    Example:
        Register and use a custom backend::

            # In your device extension module __init__.py
            from torch.profiler.backend import (
                DeviceProfilerRegistry,
                ProfilerBackend
            )

            class MyBackend(ProfilerBackend):
                # ... implementation ...
                pass

            DeviceProfilerRegistry.register_backend(
                "my_device",
                MyBackend()
            )

            # Later, profiler automatically uses your backend
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.PrivateUse1]
            ) as prof:
                # Your device operations are profiled
                model(input.to("my_device"))
    """

    _backends: defaultdict[str, ProfilerBackend] = defaultdict(lambda: None)

    @classmethod
    def register_backend(
        cls,
        device_type: str,
        backend: ProfilerBackend
    ) -> None:
        """Register a profiler backend for a specific device type.

        Args:
            device_type (str): Device type identifier (e.g., 'cuda',
                'npu', 'my_device'). Should match the string returned by
                backend.device_type().
            backend (ProfilerBackend): Profiler backend instance to
                register. The instance will be reused across profiling
                sessions.

        Warning:
            If a backend is already registered for the device type, it
            will be replaced and a warning will be issued.
        """
        if device_type in cls._backends:
            import warnings
            warnings.warn(
                f"Profiler backend for '{device_type}' is being replaced. "
                f"This may indicate duplicate registration.",
                stacklevel=2
            )
        
        cls._backends[device_type] = backend
    
    @classmethod
    def get_backend(cls, device_type: str) -> Optional[ProfilerBackend]:
        """Retrieve the registered backend for a device type.

        Args:
            device_type (str): Device type identifier to look up.
            
        Returns:
            ProfilerBackend or None: The registered backend instance, or
                None if no backend is registered for this device type.
        """
        return cls._backends.get(device_type, None)
    
    @classmethod
    def has_backend(cls, device_type: str) -> bool:
        """Check if a backend is registered for a device type.

        Args:
            device_type (str): Device type identifier to check.

        Returns:
            bool: True if a backend is registered.
        """
        return device_type in cls._backends
    
    @classmethod
    def unregister_backend(cls, device_type: str) -> None:
        """Remove a registered backend.

        Args:
            device_type (str): Device type to unregister.

        Note:
            This is primarily useful for testing. In production,
            backends typically remain registered for the lifetime of
            the process.
        """
        cls._backends.pop(device_type, None)
    
    @classmethod
    def get_registered_devices(cls) -> Set[str]:
        """Get all currently registered device types.

        Returns:
            set of str: Set of device type identifiers that have
                registered backends.
        """
        return set(cls._backends.keys())


def get_profiler_backend(
    activities: Set[ProfilerActivity],
) -> Optional[ProfilerBackend]:
    """Select appropriate profiler backend based on profiling activities.

    Determines which device backend to use by examining the requested
    profiling activities. For PrivateUse1 devices, queries the registered
    backend name.

    Args:
        activities (set of ProfilerActivity): Set of activities to
            profile (e.g., CPU, CUDA, PrivateUse1).
        
    Returns:
        ProfilerBackend or None: The appropriate backend instance, or
            None if no backend is registered for the requested device.

    Note:
        Priority order when multiple device activities are specified:
        CUDA > XPU > MTIA > HPU > PrivateUse1
    """
    # Determine which device backend to use based on activities
    device_type = None
    
    if ProfilerActivity.CUDA in activities:
        device_type = "cuda"
    elif ProfilerActivity.XPU in activities:
        device_type = "xpu"
    elif ProfilerActivity.MTIA in activities:
        device_type = "mtia"
    elif ProfilerActivity.HPU in activities:
        device_type = "hpu"
    elif ProfilerActivity.PrivateUse1 in activities:
        device_type = _get_privateuse1_backend_name()
        if device_type == "privateuseone":
            device_type = "privateuse1"
    
    if device_type is None:
        return None
    
    return DeviceProfilerRegistry.get_backend(device_type)


# Default backend implementation for Kineto-supported devices
class _KinetoBackendWrapper(ProfilerBackend):
    """Profiler backend wrapper for Kineto-supported devices.

    This backend provides the ProfilerBackend interface for devices
    that are natively supported by Kineto (CUDA, XPU, MTIA, HPU).
    
    It acts as a **transparent pass-through** that delegates all
    profiling to the existing torch.autograd.profiler implementation.
    This ensures 100% backward compatibility while providing a
    consistent interface for all devices.

    Args:
        device_type (str): Device type this backend handles
            ('cuda', 'xpu', 'mtia', or 'hpu').

    Note:
        This backend does NOT implement custom profiling logic.
        It exists solely to provide the ProfilerBackend interface
        for built-in devices while delegating to torch.autograd.profiler.
        
        The actual profiling is performed by:
        - torch.autograd.profiler.profile (Python wrapper)
        - Kineto library (C++ implementation)
        - Device-specific CUPTI/similar libraries
    """
    
    def __init__(self, device_type: str):
        self._device_type = device_type
    
    def device_type(self) -> str:
        """Return the device type string.

        Returns:
            str: Device type identifier.
        """
        return self._device_type
    
    def is_available(self) -> bool:
        """Check if the device is available.

        Returns:
            bool: True if device is available on the system.
        """
        if self._device_type == "cuda":
            return torch.cuda.is_available()
        elif self._device_type == "xpu":
            return hasattr(torch, "xpu") and torch.xpu.is_available()
        elif self._device_type == "mtia":
            return hasattr(torch, "mtia") and torch.mtia.is_available()
        elif self._device_type == "hpu":
            return hasattr(torch, "hpu") and torch.hpu.is_available()
        return False
    
    def prepare(self, config: dict[str, Any]) -> None:
        """No-op: torch.autograd.profiler handles preparation.

        Args:
            config (dict[str, Any]): Profiler configuration (unused).
        """
        pass
    
    def start(self) -> None:
        """No-op: torch.autograd.profiler handles start."""
        pass
    
    def stop(self) -> None:
        """No-op: torch.autograd.profiler handles stop."""
        pass
    
    def get_results(self) -> dict[str, Any]:
        """Return empty results: torch.autograd.profiler owns the data.

        Returns:
            dict[str, Any]: Empty dict (profiler.py handles results).
        """
        return {}
    
    def synchronize(self) -> None:
        """Synchronize device operations before profiling stops.
        
        This ensures all device kernels have completed before the
        profiler collects timing data.
        """
        if self._device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif self._device_type == "xpu" and hasattr(torch, "xpu"):
            if hasattr(torch.xpu, "synchronize"):
                torch.xpu.synchronize()
        # Note: MTIA and HPU synchronization handled by torch.autograd.profiler


def _register_builtin_backends():
    """Register built-in profiler backends for Kineto-supported devices.

    This function registers wrapper backends for CUDA, XPU, MTIA, and HPU
    that delegate to torch.autograd.profiler. These backends exist to:
    
    1. Provide a consistent ProfilerBackend interface for all devices
    2. Guarantee 100% backward compatibility with existing profiling
    3. Enable device synchronization at the right time
    4. Allow future enhancement without breaking changes

    Note:
        Called automatically at module import. Do not call directly.
    """
    # First, try to register C++ ProfilerStubs adapters if available
    # This handles backends that already implemented ProfilerStubs
    # (e.g., OpenReg via registerPrivateUse1Methods)
    try:
        from torch._C._profiler import _register_profiler_stubs_adapters
        _register_profiler_stubs_adapters()
    except (ImportError, AttributeError):
        # C++ adapter not available, fall back to Python-only registration
        pass

    # Register wrapper for each Kineto-supported device if not already registered
    # This allows ProfilerStubs to take precedence over wrappers
    for device_type in ["cuda", "xpu", "mtia", "hpu"]:
        if not DeviceProfilerRegistry.has_backend(device_type):
            backend = _KinetoBackendWrapper(device_type)
            if backend.is_available():
                DeviceProfilerRegistry.register_backend(device_type, backend)


# Register built-in backends on module import
_register_builtin_backends()
