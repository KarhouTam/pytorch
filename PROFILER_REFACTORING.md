# PyTorch Profiler Refactoring - Experimental Branch

This experimental branch refactors the PyTorch profiler to be device-agnostic, enabling out-of-tree backends (especially PrivateUse1) to provide custom profiling implementations without monkey-patching PyTorch core code.

**Related Issue:** [pytorch/pytorch#166205](https://github.com/pytorch/pytorch/issues/166205)

## What Changed?

### Problems Solved

1. **No More Monkey-Patching**: Out-of-tree backends previously had to patch internal PyTorch code to integrate profilers (see [Ascend's approach](https://github.com/Ascend/pytorch/blob/e6cc3286ad9263ef36ff5e71f99cafc6efda46ea/torch_npu/contrib/transfer_to_npu.py#L304))

2. **Device-Agnostic Architecture**: The profiler is no longer tightly coupled to Kineto/CUDA

3. **Clean Extension Points**: Official APIs for backend registration via `ProfilerBackend` interface

4. **Better PrivateUse1 Support**: Custom backends can provide device-side timing instead of CPU-only fallback

## Quick Start

### For Users (No Changes!)

The user-facing API remains unchanged:

```python
import torch
from torch.profiler import profile, ProfilerActivity

# Works as before for CUDA
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    model(input)

# Now also works for custom devices!
with profile(activities=[ProfilerActivity.PrivateUse1]) as prof:
    model(input.to("custom_device"))

print(prof.key_averages().table())
```

### For Backend Developers

Register your custom profiler:

```python
from torch.profiler.backend import ProfilerBackend, DeviceProfilerRegistry

class MyDeviceProfiler(ProfilerBackend):
    def device_type(self) -> str:
        return "my_device"
    
    def is_available(self) -> bool:
        return torch.my_device.is_available()
    
    def prepare(self, config):
        # Initialize your device profiler
        my_device_profiler_init()
    
    def start(self):
        my_device_profiler_start()
    
    def stop(self):
        my_device_profiler_stop()
    
    def get_results(self):
        return {"events": my_device_profiler_get_events()}
    
    def synchronize(self):
        torch.my_device.synchronize()

# Register during extension initialization
DeviceProfilerRegistry.register_backend("my_device", MyDeviceProfiler())
```

That's it! No need to modify PyTorch core code.

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│         torch.profiler.profile (User API)            │
│                   (Unchanged)                        │
└───────────────────────┬──────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│      torch.profiler.profiler._KinetoProfile          │
│         (Orchestrates profiling)                     │
└─────────┬──────────────────────────┬─────────────────┘
          │                          │
          │ Dispatches to            │ Uses
          ▼                          ▼
┌─────────────────────┐    ┌────────────────────────────┐
│ ProfilerBackend     │    │ torch.autograd.profiler    │
│ (New System)        │    │ (Existing CPU/Kineto)      │
│                     │    │                            │
│ - CUDA (Kineto)     │    └────────────────────────────┘
│ - Custom Backends   │
│ - PrivateUse1       │
└─────────────────────┘
```

## New Files

### Python

- **`torch/profiler/backend.py`**: Core backend interface and registry
- **`torch/profiler/examples/custom_backend_example.py`**: Complete example implementation
- **`test/profiler/test_profiler_backend.py`**: Tests for new backend system

### C++ (Future Work)

- **`torch/csrc/profiler/backend_interface.h`**: C++ backend interface
- **`torch/csrc/profiler/backend_interface.cpp`**: Registry implementation
- **`torch/csrc/profiler/kineto_backend.h/cpp`**: Kineto wrapper for new interface

### Documentation

- **`docs/source/profiler_refactoring.md`**: Complete architecture documentation and migration guide

## Modified Files

- **`torch/profiler/profiler.py`**: 
  - Added `device_backend` member to `_KinetoProfile`
  - Integrated backend lifecycle calls (prepare, start, stop, synchronize)
  - Added warnings for missing custom backends
  - Maintained full backward compatibility

## Key Interfaces

### ProfilerBackend (Python ABC)

```python
class ProfilerBackend(ABC):
    @abstractmethod
    def device_type(self) -> str: ...
    @abstractmethod
    def is_available(self) -> bool: ...
    @abstractmethod
    def prepare(self, config: Dict[str, Any]) -> None: ...
    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...
    @abstractmethod
    def get_results(self) -> Dict[str, Any]: ...
    
    # Optional overrides
    def synchronize(self) -> None: ...
    def export_trace(self, path: str) -> bool: ...
```

### DeviceProfilerRegistry

```python
class DeviceProfilerRegistry:
    @classmethod
    def register_backend(cls, device_type: str, backend: ProfilerBackend): ...
    @classmethod
    def get_backend(cls, device_type: str) -> Optional[ProfilerBackend]: ...
    @classmethod
    def has_backend(cls, device_type: str) -> bool: ...
```

## Testing

Run the new tests:

```bash
# Test the backend system
python test/profiler/test_profiler_backend.py

# Test with the example backend
python -m torch.profiler.examples.custom_backend_example

# Verify backward compatibility
python test/profiler/test_profiler.py TestProfiler.test_kineto
```

## Benefits

### For Backend Developers

1. ✅ **No Monkey-Patching**: Clean extension points
2. ✅ **Better Maintainability**: PyTorch updates don't break your profiler
3. ✅ **Official API**: Stable interface for integration
4. ✅ **Device-Side Timing**: Provide accurate device profiling data
5. ✅ **Easy Integration**: Just implement the interface and register

### For PyTorch Core

1. ✅ **Cleaner Code**: Separation of concerns
2. ✅ **Extensible**: Easy to add new device types
3. ✅ **Backward Compatible**: No breaking changes
4. ✅ **Community Friendly**: Out-of-tree backends are first-class citizens

### For Users

1. ✅ **No Changes**: Existing code works unchanged
2. ✅ **Better Profiling**: Accurate device-side timing for custom backends
3. ✅ **Consistent API**: Same profiler API across all devices

## Backward Compatibility

✅ **100% Backward Compatible**

- All existing profiler code works unchanged
- CUDA/CPU profiling behavior preserved
- No breaking changes to user-facing APIs
- Kineto integration maintained for built-in devices

## Migration Examples

### Before: Monkey-Patching (Old Approach)

```python
# Had to patch PyTorch internals - fragile and unmaintainable
import torch.profiler.profiler as profiler_module

original_class = profiler_module._KinetoProfile

class PatchedProfiler(original_class):
    def start_trace(self):
        super().start_trace()
        my_device_start_profiling()  # Custom profiling

profiler_module._KinetoProfile = PatchedProfiler  # Monkey-patch!
```

### After: Clean Registration (New Approach)

```python
# Clean, official extension point
from torch.profiler.backend import ProfilerBackend, DeviceProfilerRegistry

class MyDeviceBackend(ProfilerBackend):
    # Implement interface
    ...

DeviceProfilerRegistry.register_backend("mydevice", MyDeviceBackend())
```

## Real-World Example: NPU Backend

```python
# File: torch_npu/profiler/__init__.py

from torch.profiler.backend import ProfilerBackend, DeviceProfilerRegistry
from torch._C import _get_privateuse1_backend_name
import torch_npu._C as npu_c

class NPUProfilerBackend(ProfilerBackend):
    def device_type(self) -> str:
        return _get_privateuse1_backend_name()  # "npu"
    
    def is_available(self) -> bool:
        return torch.npu.is_available()
    
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
            "events": self.events,
            "device": "npu",
            "num_kernels": len(self.events)
        }
    
    def synchronize(self):
        torch.npu.synchronize()

# Register on import
DeviceProfilerRegistry.register_backend(
    _get_privateuse1_backend_name(),
    NPUProfilerBackend()
)
```

Now NPU profiling "just works" with the standard PyTorch profiler API!

## Future Enhancements

- [ ] Standardized trace format for all backends
- [ ] Better CPU-device event correlation
- [ ] Multi-device profiling
- [ ] Performance counter interface
- [ ] Remote profiling for distributed training
- [ ] C++ backend integration (partially implemented)

## Documentation

- **Architecture Guide**: `docs/source/profiler_refactoring.md`
- **Example Implementation**: `torch/profiler/examples/custom_backend_example.py`
- **API Reference**: Docstrings in `torch/profiler/backend.py`

## Contributing

This is an **experimental branch**. We welcome feedback and contributions!

1. Try the new backend API with your device
2. Report issues or suggest improvements
3. Share your experience integrating custom profilers

## Questions?

- **Documentation**: See `docs/source/profiler_refactoring.md`
- **Example**: Run `python -m torch.profiler.examples.custom_backend_example`
- **Tests**: Check `test/profiler/test_profiler_backend.py`
- **GitHub**: Open issues with labels `module: PrivateUse1` and `oncall: profiler`

## Credits

This refactoring addresses [Issue #166205](https://github.com/pytorch/pytorch/issues/166205) raised by the community to improve profiler integration for out-of-tree backends.

---

**Status**: 🚧 Experimental - Ready for Testing

**Last Updated**: 2024-11-12
