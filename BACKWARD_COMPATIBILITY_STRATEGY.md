# Profiler Backend System: Three-Tier Backward Compatibility

## Overview

The refactored PyTorch profiler backend system supports **THREE** different integration methods simultaneously, ensuring **100% backward compatibility** with existing vendor implementations while providing a clean path forward for new backends.

## The Three Integration Tiers

### Tier 1: Legacy ProfilerStubs (Existing Vendors)

**Who uses this**: Existing out-of-tree backends that already implemented `ProfilerStubs` via `registerPrivateUse1Methods()`.

**Example**: OpenReg backend in PyTorch's test extensions.

**How it works**:
```cpp
// In device extension C++ code (e.g., torch_openreg/csrc/profiler/openreg.cpp)
struct OpenRegMethods : public torch::profiler::impl::ProfilerStubs {
    void record(
        c10::DeviceIndex* device,
        ProfilerVoidEventStub* event,
        int64_t* cpu_ns) const override {
        // Device-specific event recording
        orEvent_t openreg_event;
        orEventCreate(&openreg_event);
        orEventRecord(openreg_event, stream);
        *event = std::shared_ptr<orEvent>(openreg_event, orEventDestroy);
    }
    
    float elapsed(
        const ProfilerVoidEventStub* e1,
        const ProfilerVoidEventStub* e2) const override {
        // Calculate elapsed time between events
        float ms = 0;
        orEventElapsedTime(&ms, e1->get(), e2->get());
        return ms * 1000.0; // Convert to microseconds
    }
    
    void synchronize() const override {
        orDeviceSynchronize();
    }
    
    // ... other methods
};

static OpenRegMethods methods;
torch::profiler::impl::registerPrivateUse1Methods(&methods);
```

**What happens**: 
1. Device registers ProfilerStubs via `registerPrivateUse1Methods()`
2. At profiler initialization, `registerProfilerStubsAdapters()` detects the registered stubs
3. Automatically wraps them with `ProfilerStubsAdapter`
4. Adapter exposes ProfilerStubs through the unified `ProfilerBackend` interface
5. **No code changes required** - existing implementations continue to work

**Profiling mode**: Uses `KINETO_PRIVATEUSE1_FALLBACK` - CPU-only timing with device event recording.

### Tier 2: Kineto Native Backends (Built-in Devices)

**Who uses this**: PyTorch's built-in accelerators - CUDA, XPU, MTIA, HPU.

**How it works**:
```python
# In torch/profiler/backend.py
class _KinetoBackendWrapper(ProfilerBackend):
    """Transparent wrapper delegating to torch.autograd.profiler (Kineto)"""
    
    def prepare(self, config): pass  # No-op
    def start(self): pass             # No-op
    def stop(self): pass              # No-op
    def get_results(self): return {}  # Empty dict
    def synchronize(self):            # Only active method
        torch.cuda.synchronize()      # Flush device operations
```

**What happens**:
1. Wrapper auto-registered for each Kineto-supported device at module import
2. All profiling logic delegated to `torch.autograd.profiler` (Kineto)
3. Only `synchronize()` is actively used to flush device operations
4. **Zero functional changes** - 100% backward compatible

**Profiling mode**: Full Kineto profiling with device-specific libraries (CUPTI for CUDA, etc.).

### Tier 3: Modern ProfilerBackend (New Implementations)

**Who uses this**: New out-of-tree backends wanting rich profiling features beyond ProfilerStubs.

**Example**: New NPU vendor wanting custom trace formats, metrics, etc.

**How it works**:
```python
# In vendor's Python extension
from torch.profiler.backend import ProfilerBackend, DeviceProfilerRegistry

class NPUProfiler(ProfilerBackend):
    """Modern NPU profiler with full feature set."""
    
    def device_type(self):
        return "npu"
    
    def is_available(self):
        return torch.npu.is_available()
    
    def prepare(self, config):
        # Initialize NPU profiler with config
        self.profiler = NPUProfilerImpl(
            record_shapes=config.get("record_shapes", False),
            profile_memory=config.get("profile_memory", False),
            with_stack=config.get("with_stack", False)
        )
    
    def start(self):
        # Start device-specific profiling
        self.profiler.start_recording()
    
    def stop(self):
        # Stop and collect profiling data
        self.profiler.stop_recording()
    
    def get_results(self):
        # Return rich profiling data
        return {
            "events": self.profiler.get_events(),
            "kernel_count": self.profiler.get_kernel_count(),
            "memory_usage": self.profiler.get_memory_stats(),
            "device_utilization": self.profiler.get_utilization()
        }
    
    def export_trace(self, path):
        # Custom trace export (e.g., NPU vendor format)
        return self.profiler.export_npu_trace(path)
    
    def synchronize(self):
        torch.npu.synchronize()

# Register at module init
DeviceProfilerRegistry.register_backend("npu", NPUProfiler())
```

**What happens**:
1. Backend explicitly implements `ProfilerBackend` interface
2. Full control over profiling lifecycle
3. Can provide custom results, trace formats, metrics
4. Integrates seamlessly with PyTorch profiler infrastructure

**Profiling mode**: Full custom profiling with device-specific capabilities.

## Migration Paths

### For Existing Vendors (Currently using ProfilerStubs)

**Option 1: No Changes (Recommended for stability)**
```cpp
// Keep existing ProfilerStubs implementation
// Automatically wrapped with ProfilerStubsAdapter
// Zero code changes required
```

**Option 2: Gradual Migration (Recommended for new features)**
```python
# Step 1: Keep ProfilerStubs for now
# Step 2: Add Python ProfilerBackend implementation
# Step 3: Register ProfilerBackend (takes precedence over adapter)
# Step 4: Eventually deprecate ProfilerStubs implementation
```

**Option 3: Complete Migration (For new capabilities)**
```python
# Implement full ProfilerBackend interface
# Remove old ProfilerStubs registration
# Get full control over profiling features
```

### For New Vendors

**Start with Tier 3** - Modern ProfilerBackend implementation:
```python
class MyDeviceProfiler(ProfilerBackend):
    # Implement all interface methods
    pass

DeviceProfilerRegistry.register_backend("my_device", MyDeviceProfiler())
```

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                   torch.profiler.profile                         │
│                   (_KinetoProfile class)                         │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         │ backend = get_profiler_backend(activities)
                         │ backend.prepare(config)
                         │ backend.start()
                         │ backend.synchronize()
                         │ backend.stop()
                         │
             ┌───────────▼────────────────┐
             │  DeviceProfilerRegistry    │
             │  (Unified backend lookup)  │
             └───┬───────────┬────────┬───┘
                 │           │        │
      ┏━━━━━━━━━━▼━━━━━┓  ┏━▼━━━━┓  ┏▼━━━━━━━━━━━━━━┓
      ┃ Tier 1:       ┃  ┃Tier 2 ┃  ┃ Tier 3:        ┃
      ┃ ProfilerStubs ┃  ┃Kineto ┃  ┃ Modern Backend ┃
      ┃ Adapter       ┃  ┃Wrapper┃  ┃                ┃
      ┗━━━━━━━┳━━━━━━━┛  ┗━━┳━━━━┛  ┗━━━━━━┳━━━━━━━━━┛
              │              │               │
              │              │               │
    ┌─────────▼──────────┐   │     ┌─────────▼──────────┐
    │ ProfilerStubs      │   │     │ Custom Profiler    │
    │ (C++ legacy impl)  │   │     │ (Full features)    │
    │                    │   │     │                    │
    │ - record()         │   │     │ - Rich events      │
    │ - elapsed()        │   │     │ - Custom formats   │
    │ - synchronize()    │   │     │ - Memory tracking  │
    │                    │   │     │ - Utilization      │
    │ Example: OpenReg   │   │     │                    │
    └────────────────────┘   │     │ Example: New NPU   │
                             │     └────────────────────┘
                             │
                  ┌──────────▼──────────┐
                  │ torch.autograd.     │
                  │ profiler (Kineto)   │
                  │                     │
                  │ - CUDA/CUPTI        │
                  │ - XPU               │
                  │ - MTIA              │
                  │ - HPU               │
                  └─────────────────────┘
```

## Compatibility Matrix

| Integration Method | Code Changes Required | Features Available | Performance | Use Case |
|-------------------|----------------------|-------------------|-------------|----------|
| **Tier 1: ProfilerStubs** | None (automatic wrap) | Basic (event timing) | Native | Existing backends |
| **Tier 2: Kineto Wrapper** | None (auto-registered) | Full Kineto features | Native | Built-in devices |
| **Tier 3: ProfilerBackend** | New implementation | Full custom features | Native | New rich backends |

## Detection and Priority

The system automatically detects and prioritizes backends:

1. **Check for explicit ProfilerBackend registration** (Tier 3)
   - If found: Use it (highest priority)
   
2. **Check for ProfilerStubs registration** (Tier 1)
   - If found: Wrap with ProfilerStubsAdapter
   - Auto-register adapter with ProfilerBackendRegistry
   
3. **Check for Kineto support** (Tier 2)
   - If device is CUDA/XPU/MTIA/HPU: Use _KinetoBackendWrapper
   
4. **No backend found**
   - Warning issued
   - Fallback to CPU-only profiling

**Example priority for PrivateUse1**:
```python
# Priority order:
1. DeviceProfilerRegistry.get_backend("npu")  # Tier 3 if registered
2. ProfilerStubsAdapter("npu")                 # Tier 1 if stubs registered  
3. None (warning + fallback)                   # No backend available
```

## Testing

### Test Tier 1 (ProfilerStubs Adapter)
```python
# Test that OpenReg's ProfilerStubs work through adapter
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.PrivateUse1]
) as prof:
    x = torch.randn(100, 100, device="openreg:0")
    y = x @ x
    torch.openreg.synchronize()

# OpenReg's ProfilerStubs automatically wrapped and used
assert len(prof.events()) > 0
```

### Test Tier 2 (Kineto Wrapper)
```python
# Test that CUDA profiling is unchanged
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA]
) as prof:
    x = torch.randn(100, 100, device="cuda")
    y = x @ x

# _KinetoBackendWrapper transparently delegates to Kineto
assert len(prof.events()) > 0
cuda_events = [e for e in prof.events() if "cuda" in e.name.lower()]
assert len(cuda_events) > 0
```

### Test Tier 3 (Modern Backend)
```python
# Test that custom ProfilerBackend works
class MockBackend(ProfilerBackend):
    def device_type(self): return "mock"
    def prepare(self, config): self.started = False
    def start(self): self.started = True
    def stop(self): self.started = False
    def get_results(self): return {"custom_metric": "42"}

DeviceProfilerRegistry.register_backend("mock", MockBackend())
backend = DeviceProfilerRegistry.get_backend("mock")
assert backend.device_type() == "mock"
```

## Summary

The three-tier system ensures:

1. ✅ **100% backward compatibility** - Existing ProfilerStubs implementations work unchanged
2. ✅ **Zero breaking changes** - Built-in CUDA/XPU profiling works unchanged
3. ✅ **Clean migration path** - Vendors can upgrade from Tier 1 → Tier 3 at their own pace
4. ✅ **Unified interface** - All backends exposed through consistent ProfilerBackend API
5. ✅ **Automatic detection** - System detects and wraps existing implementations
6. ✅ **Priority system** - Modern implementations take precedence over legacy adapters

**Result**: Existing vendors keep working, new vendors get clean APIs, PyTorch core remains stable.

## References

- **ProfilerStubs**: `torch/csrc/profiler/stubs/base.h`
- **ProfilerStubsAdapter**: `torch/csrc/profiler/profiler_stubs_adapter.h`
- **ProfilerBackend**: `torch/profiler/backend.py`
- **Example Tier 1**: `test/cpp_extensions/open_registration_extension/torch_openreg/csrc/profiler/openreg.cpp`
- **Example Tier 3**: `torch/profiler/examples/custom_backend_example.py`
