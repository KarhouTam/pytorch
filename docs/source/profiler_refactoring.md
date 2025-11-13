# PyTorch Profiler Refactoring: Device-Agnostic Architecture

## Overview

This document describes the refactored PyTorch profiler architecture that enables out-of-tree device backends to provide custom profiling implementations without modifying PyTorch core code.

**Issue Reference:** [#166205](https://github.com/pytorch/pytorch/issues/166205)

## Motivation

### Problems with the Old Architecture

1. **Tight Coupling to Kineto**: The profiler was hard-wired to Kineto, making it difficult for custom backends (especially PrivateUse1) to provide their own profiling implementations.

2. **Monkey-Patching Required**: Out-of-tree backends (like Ascend NPU) had to monkey-patch internal PyTorch code to integrate their profilers, leading to maintenance burdens and fragility.

3. **Limited PrivateUse1 Support**: The `KINETO_PRIVATEUSE1_FALLBACK` mode only provided CPU-side timing, missing device-side kernel profiling information.

4. **No Clean Extension Point**: There was no official API for backends to register custom profilers.

### Goals of the Refactoring

1. **Device Agnostic**: Decouple profiler frontend from device-specific implementations
2. **Extensible**: Provide clean interfaces for backend registration
3. **Backward Compatible**: Maintain existing behavior for CUDA/built-in devices
4. **No Monkey-Patching**: Official extension points instead of code patching

## New Architecture

### Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    torch.profiler.profile                       │
│                  (User-facing API - unchanged)                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   torch.profiler.profiler                       │
│                    (_KinetoProfile class)                       │
│              Orchestrates profiling activities                  │
└────────────┬────────────────────────────────┬───────────────────┘
             │                                │
             │ Uses                           │ Uses
             ▼                                ▼
┌────────────────────────────┐  ┌────────────────────────────────┐
│  torch.profiler.backend    │  │ torch.autograd.profiler        │
│  (New Backend System)      │  │ (Existing CPU/Kineto)          │
│                            │  │                                │
│  - ProfilerBackend (ABC)   │  │  - prof.profile                │
│  - DeviceProfilerRegistry  │  │  - Kineto integration          │
└─────────────┬──────────────┘  └────────────────────────────────┘
              │
              │ Dispatches to
              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Device-Specific Backend Implementations             │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│ CUDA/Kineto  │ Custom NPU   │ Custom XPU   │ Custom PrivateUse1 │
│ (Built-in)   │ (Out-of-tree)│ (Out-of-tree)│ (Out-of-tree)      │
└──────────────┴──────────────┴──────────────┴────────────────────┘
```

### Key Interfaces

#### 1. ProfilerBackend (Python ABC)

Located in: `torch/profiler/backend.py`

```python
class ProfilerBackend(ABC):
    @abstractmethod
    def device_type(self) -> str:
        """Device type identifier (e.g., 'cuda', 'npu')"""
        
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available"""
        
    @abstractmethod
    def prepare(self, config: Dict[str, Any]) -> None:
        """Initialize profiler with config"""
        
    @abstractmethod
    def start(self) -> None:
        """Start recording events"""
        
    @abstractmethod
    def stop(self) -> None:
        """Stop recording and finalize"""
        
    @abstractmethod
    def get_results(self) -> Dict[str, Any]:
        """Get profiling results"""
```

#### 2. DeviceProfilerRegistry

Singleton registry for backend registration:

```python
class DeviceProfilerRegistry:
    @classmethod
    def register_backend(cls, device_type: str, backend: ProfilerBackend):
        """Register a custom backend"""
        
    @classmethod
    def get_backend(cls, device_type: str) -> Optional[ProfilerBackend]:
        """Retrieve registered backend"""
```

#### 3. ProfilerBackendInterface (C++ Interface)

Located in: `torch/csrc/profiler/backend_interface.h`

```cpp
class ProfilerBackendInterface {
 public:
  virtual c10::DeviceType deviceType() const = 0;
  virtual std::string name() const = 0;
  virtual void prepare(const ProfilerConfig& config, 
                       const std::set<ActivityType>& activities) = 0;
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual bool isAvailable() const = 0;
  virtual std::unordered_map<std::string, std::string> getResults() = 0;
};
```

#### 4. ProfilerBackendRegistry (C++)

C++ registry for backend registration:

```cpp
class ProfilerBackendRegistry {
 public:
  static void registerBackend(c10::DeviceType device_type, 
                              std::unique_ptr<ProfilerBackendInterface> backend);
  static ProfilerBackendInterface* getBackend(c10::DeviceType device_type);
};
```

## Migration Guide for Backend Developers

### For Out-of-Tree Backends (e.g., NPU, Custom Accelerators)

#### Option 1: Python-Only Backend (Simpler)

```python
# In your device extension's __init__.py

from torch.profiler.backend import ProfilerBackend, DeviceProfilerRegistry
from torch._C import _get_privateuse1_backend_name

class MyDeviceProfilerBackend(ProfilerBackend):
    def device_type(self) -> str:
        return _get_privateuse1_backend_name()
    
    def is_available(self) -> bool:
        return torch.my_device.is_available()
    
    def prepare(self, config):
        # Call your device's profiler initialization
        my_device_profiler.init(config)
    
    def start(self):
        my_device_profiler.start()
    
    def stop(self):
        my_device_profiler.stop()
    
    def get_results(self):
        events = my_device_profiler.get_events()
        return {"events": events, "device": self.device_type()}
    
    def synchronize(self):
        torch.my_device.synchronize()

# Register during extension initialization
backend_name = _get_privateuse1_backend_name()
DeviceProfilerRegistry.register_backend(
    backend_name,
    MyDeviceProfilerBackend()
)
```

#### Option 2: C++ Backend (More Control)

```cpp
// In your device extension's C++ code

#include <torch/csrc/profiler/backend_interface.h>

class MyDeviceProfilerBackend : public torch::profiler::impl::ProfilerBackendInterface {
 public:
  c10::DeviceType deviceType() const override {
    return c10::DeviceType::PrivateUse1;
  }
  
  std::string name() const override {
    return "MyDevice";
  }
  
  void prepare(const ProfilerConfig& config, 
               const std::set<ActivityType>& activities) override {
    // Initialize your device profiler
    my_device_profiler_init();
  }
  
  void start() override {
    my_device_profiler_start();
  }
  
  void stop() override {
    my_device_profiler_stop();
  }
  
  bool isAvailable() const override {
    return my_device_is_available();
  }
  
  std::unordered_map<std::string, std::string> getResults() override {
    return {{"device", "MyDevice"}, {"status", "success"}};
  }
};

// Register during extension initialization
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  torch::profiler::impl::ProfilerBackendRegistry::registerBackend(
    c10::DeviceType::PrivateUse1,
    std::make_unique<MyDeviceProfilerBackend>()
  );
}
```

### Example: Migrating from Monkey-Patching

**Before (Old Approach - Monkey Patching):**

```python
# Had to patch internal PyTorch code
import torch.profiler.profiler as profiler_module

original_kineto_profile = profiler_module._KinetoProfile

class PatchedKinetoProfile(original_kineto_profile):
    def start_trace(self):
        super().start_trace()
        # Add custom device profiling
        my_device_start_profiling()

profiler_module._KinetoProfile = PatchedKinetoProfile
```

**After (New Approach - Clean Registration):**

```python
# Clean integration via registry
from torch.profiler.backend import ProfilerBackend, DeviceProfilerRegistry

class MyDeviceBackend(ProfilerBackend):
    # Implement interface methods
    ...

DeviceProfilerRegistry.register_backend("mydevice", MyDeviceBackend())
```

## Usage Examples

### For Users (No Changes Required!)

The user-facing API remains unchanged:

```python
import torch
from torch.profiler import profile, ProfilerActivity

# Works the same as before for CUDA
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    model(input)

# Now also works seamlessly for custom backends!
with profile(activities=[ProfilerActivity.PrivateUse1]) as prof:
    model(input.to("mydevice"))

print(prof.key_averages().table())
```

### For Backend Developers

See `torch/profiler/examples/custom_backend_example.py` for a complete working example.

## Testing Your Backend

```python
import torch
from torch.profiler import profile, ProfilerActivity
from torch.profiler.backend import DeviceProfilerRegistry

# Check if your backend is registered
assert DeviceProfilerRegistry.has_backend("mydevice")

# Test profiling
with profile(activities=[ProfilerActivity.PrivateUse1]) as prof:
    x = torch.randn(10, 10, device="mydevice")
    y = x @ x.t()

# Verify events were captured
events = prof.key_averages()
assert len(events) > 0
print(events.table())
```

## Benefits

1. **No Monkey-Patching**: Clean, official extension points
2. **Better Maintainability**: Changes to PyTorch core don't break backends
3. **Easier Development**: Clear interfaces and examples
4. **Better Profiling**: Backends can provide device-specific timing
5. **Community Friendly**: Out-of-tree backends are first-class citizens

## Backward Compatibility

- Existing CUDA/CPU profiling works unchanged
- Old code continues to work without modifications
- Kineto integration preserved for built-in devices
- No breaking changes to user-facing APIs

## Future Enhancements

1. **Trace Format Standardization**: Common trace format for all backends
2. **Event Correlation**: Better correlation between CPU and device events
3. **Multi-Device Profiling**: Profile multiple device types simultaneously
4. **Performance Counters**: Standardized interface for hardware counters
5. **Remote Profiling**: Network-based profiling for distributed training

## References

- GitHub Issue: [#166205](https://github.com/pytorch/pytorch/issues/166205)
- Example Implementation: `torch/profiler/examples/custom_backend_example.py`
- Backend Interface: `torch/profiler/backend.py`
- C++ Interface: `torch/csrc/profiler/backend_interface.h`

## Questions?

For questions or issues with the new profiler architecture:
1. Check the example implementation
2. Review this documentation
3. Open an issue on GitHub with label `module: PrivateUse1` and `oncall: profiler`
