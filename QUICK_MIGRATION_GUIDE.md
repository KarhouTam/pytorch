# Quick Migration Guide for Backend Developers

## 🚀 5-Minute Integration Guide

This guide shows how to add profiling support to your custom PyTorch device in just a few steps.

## Step 1: Create Your Backend Class (2 minutes)

```python
# File: your_device_extension/profiler.py

from torch.profiler.backend import ProfilerBackend, DeviceProfilerRegistry
from torch._C import _get_privateuse1_backend_name
import torch

class YourDeviceProfiler(ProfilerBackend):
    """Profiler backend for your custom device."""
    
    def __init__(self):
        self.events = []
        self.is_profiling = False
    
    def device_type(self) -> str:
        """Return your device type name."""
        return _get_privateuse1_backend_name()  # e.g., "npu", "xpu"
    
    def is_available(self) -> bool:
        """Check if your device is available."""
        return torch.your_device.is_available()  # Change to your device
    
    def prepare(self, config):
        """Initialize profiling with config."""
        # Example: Initialize your device profiler
        # your_device_profiler_init(
        #     record_shapes=config.get("record_shapes", False),
        #     profile_memory=config.get("profile_memory", False)
        # )
        pass
    
    def start(self):
        """Start recording profiling events."""
        # Example: Start your device profiler
        # your_device_profiler_start()
        self.is_profiling = True
    
    def stop(self):
        """Stop recording and collect events."""
        # Example: Stop and collect events
        # self.events = your_device_profiler_stop()
        self.is_profiling = False
    
    def get_results(self):
        """Return profiling results."""
        return {
            "device": self.device_type(),
            "num_events": str(len(self.events)),
            # Add more metadata as needed
        }
    
    def synchronize(self):
        """Synchronize device before collecting results."""
        # Example: Synchronize your device
        # torch.your_device.synchronize()
        pass
    
    def export_trace(self, path: str) -> bool:
        """Optional: Export trace to file."""
        try:
            # Example: Export trace in your format
            # your_device_export_trace(path, self.events)
            return True
        except Exception:
            return False
```

## Step 2: Register Your Backend (1 minute)

```python
# File: your_device_extension/__init__.py

from .profiler import YourDeviceProfiler
from torch.profiler.backend import DeviceProfilerRegistry
from torch._C import _get_privateuse1_backend_name

def _register_profiler():
    """Register profiler backend on import."""
    backend_name = _get_privateuse1_backend_name()
    if backend_name != "privateuseone":
        DeviceProfilerRegistry.register_backend(
            backend_name,
            YourDeviceProfiler()
        )
        print(f"✓ Profiler registered for {backend_name}")

# Auto-register when your extension is imported
_register_profiler()
```

## Step 3: Test It (2 minutes)

```python
# Test file or user code

import torch
import your_device_extension  # Registers profiler automatically
from torch.profiler import profile, ProfilerActivity

# Test profiling
with profile(activities=[ProfilerActivity.PrivateUse1]) as prof:
    x = torch.randn(100, 100, device="your_device")
    y = x @ x.t()

# Verify results
print(prof.key_averages().table())

# Should see your device events!
```

## Real-World Examples

### Example 1: NPU Device

```python
# torch_npu/profiler.py

from torch.profiler.backend import ProfilerBackend, DeviceProfilerRegistry
import torch_npu._C as npu_c

class NPUProfiler(ProfilerBackend):
    def device_type(self) -> str:
        return "npu"
    
    def prepare(self, config):
        npu_c.profiler_init(
            config.get("record_shapes", False),
            config.get("profile_memory", False)
        )
    
    def start(self):
        npu_c.profiler_start()
    
    def stop(self):
        self.events = npu_c.profiler_stop()
    
    def get_results(self):
        return {
            "device": "npu",
            "events": self.events,
            "num_kernels": len(self.events)
        }
    
    def synchronize(self):
        torch.npu.synchronize()
```

### Example 2: Custom AI Accelerator

```python
# my_accelerator/profiler.py

from torch.profiler.backend import ProfilerBackend
import my_accelerator_lib as accel

class AcceleratorProfiler(ProfilerBackend):
    def device_type(self) -> str:
        return "my_accelerator"
    
    def prepare(self, config):
        self.profiler = accel.Profiler(
            enable_shapes=config.get("record_shapes", False),
            enable_memory=config.get("profile_memory", False)
        )
    
    def start(self):
        self.profiler.begin()
    
    def stop(self):
        self.trace = self.profiler.end()
    
    def get_results(self):
        return {
            "device": "my_accelerator",
            "trace_size": str(len(self.trace)),
            "kernel_count": str(self.trace.num_kernels())
        }
    
    def export_trace(self, path: str) -> bool:
        self.trace.save(path)
        return True
```

## Common Patterns

### Pattern 1: Using C++ Backend

```python
# If your profiler is implemented in C++

import your_device._C as device_c

class MyProfiler(ProfilerBackend):
    def start(self):
        device_c.profiler_start()  # Call C++ function
    
    def stop(self):
        self.events = device_c.profiler_get_events()  # Get from C++
```

### Pattern 2: Lazy Initialization

```python
# If profiler needs lazy initialization

class MyProfiler(ProfilerBackend):
    def __init__(self):
        self._profiler = None
    
    def prepare(self, config):
        if self._profiler is None:
            self._profiler = initialize_profiler()
        self._profiler.configure(config)
```

### Pattern 3: Multiple Devices

```python
# If you have multiple device types

class MultiDeviceProfiler(ProfilerBackend):
    def __init__(self, device_id: int):
        self.device_id = device_id
    
    def synchronize(self):
        torch.my_device.synchronize(self.device_id)
```

## Checklist

Before releasing your profiler integration:

- [ ] Implemented all required `ProfilerBackend` methods
- [ ] Backend registered in `__init__.py`
- [ ] Tested with `torch.profiler.profile`
- [ ] Device synchronization working
- [ ] Results include useful metadata
- [ ] Documentation updated
- [ ] Example added for users

## Troubleshooting

### Issue: Backend not found

```python
# Check if registered
from torch.profiler.backend import DeviceProfilerRegistry
print(DeviceProfilerRegistry.get_registered_devices())
# Should see your device in the list
```

### Issue: Events not showing

```python
# Add debug logging
class MyProfiler(ProfilerBackend):
    def start(self):
        print(f"[{self.device_type()}] Starting profiler")
        # Your start code
    
    def stop(self):
        print(f"[{self.device_type()}] Stopping profiler")
        # Your stop code
```

### Issue: Wrong device timing

```python
# Make sure synchronize() is called
def synchronize(self):
    torch.my_device.synchronize()  # Critical for accurate timing!
```

## Advanced Features

### Custom Trace Export

```python
def export_trace(self, path: str) -> bool:
    """Export in Chrome Trace Format."""
    import json
    
    chrome_trace = {
        "traceEvents": [
            {
                "name": event.name,
                "cat": "kernel",
                "ph": "X",  # Complete event
                "ts": event.start_us,
                "dur": event.duration_us,
                "pid": 0,
                "tid": event.stream_id,
            }
            for event in self.events
        ]
    }
    
    with open(path, 'w') as f:
        json.dump(chrome_trace, f)
    return True
```

### Performance Counters

```python
def get_results(self):
    """Include performance counter data."""
    return {
        "device": self.device_type(),
        "events": self.events,
        "perf_counters": {
            "total_flops": sum(e.flops for e in self.events),
            "memory_bandwidth": self.calculate_bandwidth(),
        }
    }
```

## Complete Minimal Example

```python
# Complete working example (copy-paste ready!)

from torch.profiler.backend import ProfilerBackend, DeviceProfilerRegistry
from torch._C import _get_privateuse1_backend_name
import torch

class MinimalProfiler(ProfilerBackend):
    def device_type(self) -> str:
        return _get_privateuse1_backend_name()
    
    def is_available(self) -> bool:
        return True  # Adjust for your device
    
    def prepare(self, config):
        self.events = []
    
    def start(self):
        pass  # Add your start logic
    
    def stop(self):
        pass  # Add your stop logic
    
    def get_results(self):
        return {"device": self.device_type()}
    
    def synchronize(self):
        pass  # Add synchronization if needed

# Register
backend_name = _get_privateuse1_backend_name()
if backend_name != "privateuseone":
    DeviceProfilerRegistry.register_backend(backend_name, MinimalProfiler())
```

## Next Steps

1. ✅ Copy the minimal example
2. ✅ Add your device-specific profiler calls
3. ✅ Test with `torch.profiler`
4. ✅ Add to your documentation
5. ✅ Share with users!

## Need Help?

- 📖 Full docs: `docs/source/profiler_refactoring.md`
- 💻 Complete example: `torch/profiler/examples/custom_backend_example.py`
- 🧪 Tests: `test/profiler/test_profiler_backend.py`

---

**That's it!** Your device now has first-class profiling support! 🎉
