#include <torch/csrc/profiler/kineto_backend.h>

#include <ATen/Context.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/profiler/kineto_shim.h>
#include "ATen/xpu/XPUEvent.h"

namespace torch::profiler::impl {

// ============================================================================
// KinetoProfilerBackend Implementation
// ============================================================================
#if defined(USE_KINETO)
KinetoProfilerBackend::KinetoProfilerBackend(c10::DeviceType device_type)
    : device_type_(device_type) {
  // Set backend name based on device type
  backend_name_ = c10::DeviceTypeName(device_type);
}

c10::DeviceType KinetoProfilerBackend::deviceType() const {
  return device_type_;
}

std::string KinetoProfilerBackend::name() const {
  return backend_name_;
}

void KinetoProfilerBackend::prepare(
    const ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities) {
  TORCH_CHECK(!is_running_, "Cannot prepare while profiler is running");
  // Convert ActivityType to kineto activities
  std::set<libkineto::ActivityType> kineto_activities;
  for (const auto& activity : activities) {
    if (activity == torch::profiler::impl::ActivityType::CPU) {
      // CPU is handled separately
      continue;
    } else if (activity == torch::profiler::impl::ActivityType::CUDA && device_type_ == c10::DeviceType::CUDA) {
      kineto_activities.insert(libkineto::ActivityType::GPU_MEMCPY);
      kineto_activities.insert(libkineto::ActivityType::GPU_MEMSET);
      kineto_activities.insert(libkineto::ActivityType::CONCURRENT_KERNEL);
    } else if (activity == torch::profiler::impl::ActivityType::XPU && device_type_ == c10::DeviceType::XPU) {
      kineto_activities.insert(libkineto::ActivityType::GPU_MEMCPY);
      kineto_activities.insert(libkineto::ActivityType::CONCURRENT_KERNEL);
    }
    // Add more device types as needed
  }

  // Prepare kineto trace
  bool cpuOnly = kineto_activities.empty();
  kineto::prepareTrace(cpuOnly, activities, config.experimental_config, config.trace_id);
  
  is_prepared_ = true;
}

void KinetoProfilerBackend::start() {
  TORCH_CHECK(is_prepared_, "Must call prepare() before start()");
  TORCH_CHECK(!is_running_, "Profiler is already running");
  
  kineto::startTrace();
  is_running_ = true;
}

void KinetoProfilerBackend::stop() {
  TORCH_CHECK(is_running_, "Profiler is not running");
  
  // Synchronize device before stopping
  synchronize();
  
  trace_ = kineto::stopTrace();
  is_running_ = false;
  is_prepared_ = false;
}

bool KinetoProfilerBackend::isAvailable() const {
  switch (device_type_) {
    case c10::DeviceType::CUDA:
      return at::hasCUDA();
    case c10::DeviceType::XPU:
      return at::hasXPU();
    case c10::DeviceType::MTIA:
      return at::hasMTIA();
    default:
      return false;
  }
}

bool KinetoProfilerBackend::supportsActivity(ActivityType activity) const {
  switch (activity) {
    case ActivityType::CPU:
      // All backends support CPU activity
      return true;
    case ActivityType::CUDA:
      return device_type_ == c10::DeviceType::CUDA;
    case ActivityType::XPU:
      return device_type_ == c10::DeviceType::XPU;
    case ActivityType::MTIA:
      return device_type_ == c10::DeviceType::MTIA;
    case ActivityType::HPU:
      return device_type_ == c10::DeviceType::HPU;
    default:
      return false;
  }
}

std::unordered_map<std::string, std::string> KinetoProfilerBackend::getResults() {
  std::unordered_map<std::string, std::string> results;

  if (trace_.get()) {
    results["has_trace"] = "true";
    results["backend"] = backend_name_;
  } else {
    results["has_trace"] = "false";
  }
  
  return results;
}

bool KinetoProfilerBackend::exportTrace(const std::string& path) {
  if (!trace_.get()) {
    return false;
  }
  
  try {
    trace_.save(path);
    return true;
  } catch (const std::exception& e) {
    TORCH_WARN("Failed to export trace: ", e.what());
    return false;
  }
}

int64_t KinetoProfilerBackend::deviceElapsedUs(const void* event_ptr) const {
  // This would need to extract timing from the kineto event
  // For now, return -1 to indicate not available
  return -1;
}

void KinetoProfilerBackend::synchronize() {
  if (!isAvailable()) {
    return;
  }
  
  switch (device_type_) {
    case c10::DeviceType::CUDA:
      if (at::getNumGPUs() > 0) {
        c10::cuda::device_synchronize();
      }
      break;
    case c10::DeviceType::XPU:
      if (at::hasXPU()) {
        // XPU synchronization
        // at::xpu::synchronize();
      }
      break;
    default:
      // Other devices - no-op
      break;
  }
}
#endif // !defined(USE_KINETO)

// ============================================================================
// CPUProfilerBackend Implementation
// ============================================================================

CPUProfilerBackend::CPUProfilerBackend() = default;

c10::DeviceType CPUProfilerBackend::deviceType() const {
  return c10::DeviceType::CPU;
}

std::string CPUProfilerBackend::name() const {
  return "CPU";
}

void CPUProfilerBackend::prepare(
    const ProfilerConfig& config,
    const std::set<ActivityType>& activities) {
  // CPU profiling is always available, minimal preparation needed
}

void CPUProfilerBackend::start() {
  is_running_ = true;
}

void CPUProfilerBackend::stop() {
  is_running_ = false;
}

bool CPUProfilerBackend::isAvailable() const {
  return true; // CPU profiling always available
}

bool CPUProfilerBackend::supportsActivity(ActivityType activity) const {
  return activity == ActivityType::CPU;
}

std::unordered_map<std::string, std::string> CPUProfilerBackend::getResults() {
  std::unordered_map<std::string, std::string> results;
  results["backend"] = "CPU";
  return results;
}

// ============================================================================
// FallbackPrivateUse1Backend Implementation
// ============================================================================

FallbackPrivateUse1Backend::FallbackPrivateUse1Backend() {
  backend_name_ = c10::get_privateuse1_backend();
  if (backend_name_ == "privateuseone") {
    backend_name_ = "PrivateUse1";
  }
}

c10::DeviceType FallbackPrivateUse1Backend::deviceType() const {
  return c10::DeviceType::PrivateUse1;
}

std::string FallbackPrivateUse1Backend::name() const {
  return backend_name_;
}

void FallbackPrivateUse1Backend::prepare(
    const ProfilerConfig& config,
    const std::set<ActivityType>& activities) {
  TORCH_WARN_ONCE(
      "Using fallback profiler for ",
      backend_name_,
      ". Device-side timing may not be accurate. "
      "Consider implementing a custom ProfilerBackendInterface for better profiling support.");
}

void FallbackPrivateUse1Backend::start() {
  is_running_ = true;
}

void FallbackPrivateUse1Backend::stop() {
  is_running_ = false;
}

bool FallbackPrivateUse1Backend::isAvailable() const {
  return c10::get_privateuse1_backend() != "privateuseone";
}

bool FallbackPrivateUse1Backend::supportsActivity(ActivityType activity) const {
  // Fallback only supports CPU timing
  return activity == ActivityType::CPU;
}

std::unordered_map<std::string, std::string> FallbackPrivateUse1Backend::getResults() {
  std::unordered_map<std::string, std::string> results;
  results["backend"] = backend_name_;
  results["mode"] = "fallback";
  results["warning"] = "Using CPU-only timing for device operations";
  return results;
}

// ============================================================================
// Default Backend Registration
// ============================================================================

void registerDefaultProfilerBackends() {
  // Register CPU backend
  ProfilerBackendRegistry::registerBackend(
      c10::DeviceType::CPU,
      std::make_unique<CPUProfilerBackend>());

#if defined(USE_CUDA)
  // Register CUDA backend if available
  if (at::getNumGPUs() > 0) {
    ProfilerBackendRegistry::registerBackend(
        c10::DeviceType::CUDA,
        std::make_unique<KinetoProfilerBackend>(c10::DeviceType::CUDA));
  }
#endif

#if defined(USE_XPU)
  // Register XPU backend if available
  if (at::hasXPU()) {
    ProfilerBackendRegistry::registerBackend(
        c10::DeviceType::XPU,
        std::make_unique<KinetoProfilerBackend>(c10::DeviceType::XPU));
  }
#endif

  // Register fallback PrivateUse1 backend if a custom backend is registered
  if (c10::get_privateuse1_backend() != "privateuseone") {
    // Only register fallback if no custom backend was registered
    if (!ProfilerBackendRegistry::hasBackend(c10::DeviceType::PrivateUse1)) {
      ProfilerBackendRegistry::registerBackend(
          c10::DeviceType::PrivateUse1,
          std::make_unique<FallbackPrivateUse1Backend>());
    }
  }
}

} // namespace torch::profiler::impl
