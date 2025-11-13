"""Example custom profiler backend for PrivateUse1 devices.

This module demonstrates how out-of-tree hardware backends can implement
their own profilers without modifying PyTorch core code. It provides a
complete working example that can be adapted for real device backends.

The key steps are:
    1. Subclass ProfilerBackend and implement required methods
    2. Register with DeviceProfilerRegistry
    3. Use torch.profiler normally - your backend is automatically used

Example:
    Implementing profiler support for a hypothetical NPU device::

        import torch
        from torch.profiler.backend import (
            ProfilerBackend,
            DeviceProfilerRegistry
        )

        class NPUProfilerBackend(ProfilerBackend):
            def __init__(self):
                self.events = []
                self.is_running = False
                
            def device_type(self):
                return "npu"
            
            def is_available(self):
                return hasattr(torch, "npu") and torch.npu.is_available()
            
            def prepare(self, config):
                import torch_npu
                torch_npu.profiler.init()
                self.config = config
            
            def start(self):
                import torch_npu
                torch_npu.profiler.start()
                self.is_running = True
            
            def stop(self):
                import torch_npu
                self.events = torch_npu.profiler.stop()
                self.is_running = False
            
            def get_results(self):
                return {
                    "events": self.events,
                    "device": "npu",
                    "num_events": len(self.events)
                }
            
            def synchronize(self):
                import torch_npu
                torch_npu.npu.synchronize()

        # Register once at module import
        DeviceProfilerRegistry.register_backend("npu", NPUProfilerBackend())

        # Profiling automatically uses your backend
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.PrivateUse1]
        ) as prof:
            x = torch.randn(10, 10, device="npu")
            y = x @ x.t()

        print(prof.key_averages().table())
"""

from typing import Any, Dict, List
from torch.profiler.backend import ProfilerBackend, DeviceProfilerRegistry
import torch
from torch._C import _get_privateuse1_backend_name


class ExampleCustomProfilerBackend(ProfilerBackend):
    """Example profiler backend for demonstration purposes.

    This is a simplified reference implementation showing the minimal
    requirements for a profiler backend. Real implementations should
    integrate with actual device profiling APIs.

    Attributes:
        device_name (str): Device type identifier.
        events (list of dict): Collected profiling events.
        config (dict): Current profiler configuration.
        is_running (bool): Whether profiling is currently active.
        start_time (float): Timestamp when profiling started.
    """
    
    def __init__(self, device_name: str = "custom"):
        """Initialize the example profiler backend.

        Args:
            device_name (str, optional): Device type name. Default: 'custom'.
        """
        self.device_name = device_name
        self.events: List[Dict[str, Any]] = []
        self.is_running = False
        self.start_time = 0
        self.config = {}
    
    def device_type(self) -> str:
        """Return device type string.

        Returns:
            str: Device type identifier.
        """
        return self.device_name
    
    def is_available(self) -> bool:
        """Check if device is available.

        Returns:
            bool: True if device is available.

        Note:
            Real implementations should check actual device availability.
        """
        return True  # For demonstration
    
    def prepare(self, config: Dict[str, Any]) -> None:
        """Initialize profiler with configuration.

        Args:
            config (dict[str, Any]): Profiler configuration dictionary.
        """
        self.config = config
        self.events = []
        print(f"[{self.device_name}] Profiler prepared with config: {config}")
    
    def start(self) -> None:
        """Start recording profiling events."""
        import time
        self.start_time = time.perf_counter()
        self.is_running = True
        print(f"[{self.device_name}] Profiler started")
    
    def stop(self) -> None:
        """Stop recording and finalize events."""
        import time
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        
        # In a real implementation:
        # 1. Call device-specific API to stop profiling
        # 2. Collect device events (kernels, memory transfers)
        # 3. Convert to standard format
        
        # For demonstration, create dummy event
        self.events.append({
            "name": "example_kernel",
            "device": self.device_name,
            "duration_us": duration * 1e6,
            "type": "kernel"
        })
        
        self.is_running = False
        print(f"[{self.device_name}] Profiler stopped. "
              f"Duration: {duration:.6f}s")
    
    def get_results(self) -> Dict[str, Any]:
        """Return collected profiling data.

        Returns:
            dict[str, Any]: Profiling results with device info and events.
        """
        return {
            "device": self.device_name,
            "num_events": str(len(self.events)),
            "events_summary": f"Collected {len(self.events)} events",
        }
    
    def export_trace(self, path: str) -> bool:
        """Export trace to file.

        Args:
            path (str): Filesystem path for trace file.

        Returns:
            bool: True if export succeeded, False otherwise.
        """
        try:
            import json
            with open(path, 'w') as f:
                json.dump({
                    "device": self.device_name,
                    "events": self.events,
                    "config": self.config
                }, f, indent=2)
            print(f"[{self.device_name}] Trace exported to {path}")
            return True
        except Exception as e:
            print(f"[{self.device_name}] Failed to export trace: {e}")
            return False
    
    def synchronize(self) -> None:
        """Synchronize device before finalizing.

        Note:
            Real implementations should call device synchronization API
            (e.g., torch.custom_device.synchronize()).
        """
        print(f"[{self.device_name}] Device synchronized")


def register_example_backend():
    """Register the example profiler backend.
    
    This function should be called during device extension initialization
    to register the profiler backend with PyTorch's profiler registry.

    Note:
        Only registers if PrivateUse1 backend is configured with a
        custom name (not the default 'privateuseone').
    """
    backend_name = _get_privateuse1_backend_name()
    
    # Only register if we're actually the PrivateUse1 backend
    if backend_name != "privateuseone":
        print(f"Registering custom profiler backend for: {backend_name}")
        backend = ExampleCustomProfilerBackend(backend_name)
        DeviceProfilerRegistry.register_backend(backend_name, backend)
        print(f"✓ Custom profiler backend registered for {backend_name}")
        print("  You can now use torch.profiler with "
              "ProfilerActivity.PrivateUse1")
    else:
        print("PrivateUse1 backend not configured. Skipping registration.")


# Example usage when imported
if __name__ == "__main__":
    print("=" * 70)
    print("Custom Profiler Backend Example")
    print("=" * 70)
    
    # Register our example backend
    # In production, this would be called from your device extension's __init__.py
    register_example_backend()
    
    print("\nTo use this in production:")
    print("1. Implement ProfilerBackend for your device")
    print("2. Call DeviceProfilerRegistry.register_backend() in your extension")
    print("3. Users can then profile with torch.profiler.ProfilerActivity.PrivateUse1")
    print("\nExample:")
    print("""
    from torch.profiler import profile, ProfilerActivity
    
    with profile(activities=[ProfilerActivity.PrivateUse1]) as prof:
        # Your device operations here
        pass
    
    print(prof.key_averages().table())
    """)
