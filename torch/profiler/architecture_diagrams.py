"""
Visual Diagrams for Profiler Refactoring Architecture

This file contains ASCII diagrams explaining the refactored profiler system.
"""

ARCHITECTURE_OVERVIEW = """
═══════════════════════════════════════════════════════════════════════════════
                    PyTorch Profiler Refactoring Architecture
═══════════════════════════════════════════════════════════════════════════════

┌───────────────────────────────────────────────────────────────────────────┐
│                          USER CODE (Unchanged!)                           │
│                                                                           │
│   with torch.profiler.profile(                                            │
│       activities=[ProfilerActivity.PrivateUse1]                           │
│   ) as prof:                                                              │
│       model(input.to("custom_device"))                                    │
│                                                                           │
│   print(prof.key_averages().table())                                      │
└──────────────────────────────────┬────────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                   torch.profiler.profile (Context Manager)                │
│                        - Public API (stable)                              │
│                        - Handles scheduling                               │
│                        - Calls _KinetoProfile                             │
└──────────────────────────────────┬────────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                torch.profiler.profiler._KinetoProfile                     │
│                                                                           │
│  NEW: Device-agnostic orchestration layer                                 │
│                                                                           │
│  prepare_trace() ─────┬────→ Creates prof.profile()                       │
│  start_trace()   ─────┼────→ Starts CPU profiling                         │
│                       │                                                   │
│                       ├────→ IF device_backend registered:                │
│                       │         device_backend.prepare(config)            │
│                       │         device_backend.start()                    │
│                       │                                                   │
│  stop_trace()    ─────┼────→ device_backend.synchronize()                 │
│                       │      device_backend.stop()                        │
│                       │      device_backend.get_results()                 │
│                       │                                                   │
│                       └────→ Stops CPU profiling                          │
└──────────────┬────────────────────────────┬───────────────────────────────┘
               │                            │
               │ Dispatches to              │ Uses (CPU profiling)
               ▼                            ▼
┌────────────────────────────┐   ┌──────────────────────────────────────┐
│  DeviceProfilerRegistry    │   │  torch.autograd.profiler             │
│  (Backend Dispatcher)      │   │  (Existing Kineto/CPU)               │
│                            │   │                                      │
│  - register_backend()      │   │  - CPU event recording               │
│  - get_backend()           │   │  - Kineto integration                │
│  - has_backend()           │   │  - Event tree building               │
└──────────┬─────────────────┘   └──────────────────────────────────────┘
           │
           │ Returns appropriate backend
           ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      Device-Specific Backends                             │
│                      (Registered at Runtime)                              │
├───────────────┬───────────────────┬──────────────────┬───────────────┬────┤
│ CUDA/Kineto   │ Custom NPU        │ Custom XPU       │ Generic Fallback   │
│ (Built-in)    │ (Out-of-tree)     │ (Out-of-tree)    │ (CPU-only timing)  │
│               │                   │                  │                    │
│ - GPU kernels │ - NPU kernels     │ - XPU kernels    │ - CPU timing only  │
│ - Memory ops  │ - NPU memory      │ - XPU memory     │ - Warns user       │
│ - CUPTI trace │ - Device trace    │ - Device trace   │ - Simple impl      │
└───────────────┴───────────────────┴──────────────────┴──────────────────-─┘

═══════════════════════════════════════════════════════════════════════════════
"""

BACKEND_INTERFACE = """
═══════════════════════════════════════════════════════════════════════════════
                         ProfilerBackend Interface
═══════════════════════════════════════════════════════════════════════════════

                         ┌──────────────────────┐
                         │  ProfilerBackend     │
                         │  (Abstract Base)     │
                         └──────────┬───────────┘
                                    │
                                    │ implements
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
         ▼                          ▼                          ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  CUDA Backend    │    │   NPU Backend    │    │   XPU Backend    │
│  (Kineto)        │    │  (Custom)        │    │  (Custom)        │
├──────────────────┤    ├──────────────────┤    ├──────────────────┤
│ device_type()    │    │ device_type()    │    │ device_type()    │
│   → "cuda"       │    │   → "npu"        │    │   → "xpu"        │
│                  │    │                  │    │                  │
│ is_available()   │    │ is_available()   │    │ is_available()   │
│   → CUDA check   │    │   → NPU check    │    │   → XPU check    │
│                  │    │                  │    │                  │
│ prepare(config)  │    │ prepare(config)  │    │ prepare(config)  │
│   → kineto init  │    │   → npu_init()   │    │   → xpu_init()   │
│                  │    │                  │    │                  │
│ start()          │    │ start()          │    │ start()          │
│   → kineto start │    │   → npu_start()  │    │   → xpu_start()  │
│                  │    │                  │    │                  │
│ stop()           │    │ stop()           │    │ stop()           │
│   → kineto stop  │    │   → npu_stop()   │    │   → xpu_stop()   │
│                  │    │                  │    │                  │
│ get_results()    │    │ get_results()    │    │ get_results()    │
│   → events dict  │    │   → events dict  │    │   → events dict  │
│                  │    │                  │    │                  │
│ synchronize()    │    │ synchronize()    │    │ synchronize()    │
│   → cuda.sync()  │    │   → npu.sync()   │    │   → xpu.sync()   │
└──────────────────┘    └──────────────────┘    └──────────────────┘

═══════════════════════════════════════════════════════════════════════════════
"""

REGISTRATION_FLOW = """
═══════════════════════════════════════════════════════════════════════════════
                    Backend Registration and Usage Flow
═══════════════════════════════════════════════════════════════════════════════

Step 1: Backend Implementation (One-time, in device extension)
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  # In torch_npu/profiler/__init__.py (or your device extension)           │
│                                                                            │
│  from torch.profiler.backend import ProfilerBackend, DeviceProfilerRegistry│
│                                                                            │
│  class NPUProfilerBackend(ProfilerBackend):                               │
│      def device_type(self): return "npu"                                  │
│      def prepare(self, config): npu_profiler_init()                       │
│      def start(self): npu_profiler_start()                                │
│      def stop(self): npu_profiler_stop()                                  │
│      # ... implement other methods                                        │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ calls
                                   ▼
Step 2: Registration (Automatic on import)
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  DeviceProfilerRegistry.register_backend(                                 │
│      "npu",                                                                │
│      NPUProfilerBackend()                                                  │
│  )                                                                         │
│                                                                            │
│  ✓ Backend is now registered and ready!                                   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │
                                   ▼
Step 3: User Code (No changes needed!)
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  import torch                                                              │
│  import torch_npu  # Your device extension                                │
│  from torch.profiler import profile, ProfilerActivity                     │
│                                                                            │
│  with profile(activities=[ProfilerActivity.PrivateUse1]) as prof:        │
│      x = torch.randn(1000, 1000, device="npu")                            │
│      y = x @ x.t()                                                         │
│                                                                            │
│  print(prof.key_averages().table())                                       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │
                                   ▼
Step 4: Profiler Execution
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  _KinetoProfile.__init__()                                                │
│      └─→ Detects ProfilerActivity.PrivateUse1                            │
│          └─→ Gets backend name from _get_privateuse1_backend_name()      │
│              └─→ Looks up "npu" in DeviceProfilerRegistry                │
│                  └─→ Finds NPUProfilerBackend!                            │
│                                                                            │
│  _KinetoProfile.start_trace()                                             │
│      ├─→ Starts CPU profiling (always)                                    │
│      └─→ Calls backend.prepare(config)                                    │
│          └─→ Calls backend.start()                                        │
│                                                                            │
│  # User code runs with profiling active                                   │
│                                                                            │
│  _KinetoProfile.stop_trace()                                              │
│      ├─→ Calls backend.synchronize()                                      │
│      ├─→ Calls backend.stop()                                             │
│      ├─→ Gets results via backend.get_results()                           │
│      └─→ Stops CPU profiling                                              │
│                                                                            │
│  Results are combined and returned to user!                               │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
"""

COMPARISON = """
═══════════════════════════════════════════════════════════════════════════════
                    Before vs After Comparison
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────┬─────────────────────────────────────┐
│         BEFORE (Old Way)            │         AFTER (New Way)             │
├─────────────────────────────────────┼─────────────────────────────────────┤
│                                     │                                     │
│  ❌ Monkey-Patching Required        │  ✅ Clean Registration              │
│                                     │                                     │
│  import torch.profiler.profiler as p│  from torch.profiler.backend import │
│  original = p._KinetoProfile        │  ProfilerBackend,                   │
│                                     │  DeviceProfilerRegistry             │
│  class Patched(original):           │                                     │
│      def start_trace(self):         │  class MyBackend(ProfilerBackend):  │
│          super().start_trace()      │      def start(self):               │
│          my_device_start()          │          my_device_start()          │
│                                     │                                     │
│  p._KinetoProfile = Patched  # 💥   │  DeviceProfilerRegistry.            │
│                                     │    register_backend("my", MyBackend)│
│                                     │                                     │
├─────────────────────────────────────┼─────────────────────────────────────┤
│                                     │                                     │
│  ❌ Hard-Coded Device Types         │  ✅ Dynamic Device Support          │
│                                     │                                     │
│  # In PyTorch core:                 │  # In your extension:               │
│  if device == "cuda":               │  backend = MyDeviceBackend()        │
│      use_kineto()                   │  registry.register("mydev", backend)│
│  elif device == "npu":  # 🚫 Need   │                                     │
│      # to modify PyTorch!           │  # PyTorch automatically uses it!   │
│                                     │                                     │
├─────────────────────────────────────┼─────────────────────────────────────┤
│                                     │                                     │
│  ❌ Fallback Only (CPU timing)      │  ✅ Full Device Profiling           │
│                                     │                                     │
│  KINETO_PRIVATEUSE1_FALLBACK:       │  class MyBackend(ProfilerBackend):  │
│  - Only CPU-side timing             │      def start(self):               │
│  - No device kernel info            │          device_profiler_start()    │
│  - Inaccurate for GPU work          │      def stop(self):                │
│                                     │          events = get_kernels()     │
│                                     │          # Full kernel info! ✨      │
│                                     │                                     │
├─────────────────────────────────────┼─────────────────────────────────────┤
│                                     │                                     │
│  ❌ Fragile                          │  ✅ Stable                          │
│                                     │                                     │
│  # PyTorch update breaks patches    │  # Stable interface                 │
│  # _KinetoProfile internals change  │  # Backend interface versioned      │
│  # Your patching code breaks 💥     │  # Your backend still works ✅      │
│                                     │                                     │
├─────────────────────────────────────┼─────────────────────────────────────┤
│                                     │                                     │
│  ❌ Second-Class Citizen             │  ✅ First-Class Citizen             │
│                                     │                                     │
│  # Out-of-tree backends:            │  # Out-of-tree backends:            │
│  - Unofficial hacks                 │  - Official API                     │
│  - No documentation                 │  - Full documentation               │
│  - Community unfriendly             │  - Community first!                 │
│                                     │                                     │
└─────────────────────────────────────┴─────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(ARCHITECTURE_OVERVIEW)
    print("\n" * 2)
    print(BACKEND_INTERFACE)
    print("\n" * 2)
    print(REGISTRATION_FLOW)
    print("\n" * 2)
    print(COMPARISON)
