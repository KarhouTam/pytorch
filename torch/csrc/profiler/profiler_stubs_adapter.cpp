#include <torch/csrc/profiler/profiler_stubs_adapter.h>
#include <torch/csrc/profiler/backend_interface.h>
#include <torch/csrc/profiler/stubs/base.h>
#include <c10/util/Exception.h>

namespace torch::profiler::impl {

ProfilerStubsAdapter::ProfilerStubsAdapter(
    const ProfilerStubs* stubs,
    c10::DeviceType device_type,
    std::string name)
    : stubs_(stubs), device_type_(device_type), name_(std::move(name)) {
  TORCH_CHECK(stubs != nullptr, "ProfilerStubs cannot be null");
}

void ProfilerStubsAdapter::prepare(
    const ProfilerConfig& config,
    const std::set<ActivityType>& activities) {
  config_ = config;
  activities_ = activities;
  // ProfilerStubs don't have an explicit prepare phase
  // They're initialized via registration
}

void ProfilerStubsAdapter::start() {
  is_recording_ = true;
  // ProfilerStubs are activated by the kineto profiler infrastructure
  // This adapter just tracks state
}

void ProfilerStubsAdapter::stop() {
  is_recording_ = false;
  // Synchronize device to ensure all events are recorded
  if (stubs_ && stubs_->enabled()) {
    stubs_->synchronize();
  }
}

bool ProfilerStubsAdapter::isAvailable() const {
  return stubs_ != nullptr && stubs_->enabled();
}

bool ProfilerStubsAdapter::supportsActivity(ActivityType activity) const {
  // ProfilerStubs support device activities (KINETO_PRIVATEUSE1_FALLBACK mode)
  // They don't support CPU-only activities
  switch (activity) {
    case ActivityType::CPU:
      return false; // CPU handled by Kineto directly
    case ActivityType::CUDA:
    case ActivityType::XPU:
    case ActivityType::MTIA:
    case ActivityType::PrivateUse1:
      return true; // Device activities supported via stubs
    default:
      return false;
  }
}

std::unordered_map<std::string, std::string> ProfilerStubsAdapter::getResults() {
  // ProfilerStubs don't provide structured results
  // Events are recorded via the stub's record() method and collected by Kineto
  std::unordered_map<std::string, std::string> results;
  results["backend_type"] = "ProfilerStubs";
  results["device_type"] = c10::DeviceTypeName(device_type_);
  results["enabled"] = stubs_->enabled() ? "true" : "false";
  return results;
}

void ProfilerStubsAdapter::synchronize() {
  if (stubs_ && stubs_->enabled()) {
    stubs_->synchronize();
  }
}

// Helper to check if a stub is not the default (disabled) implementation
static bool isStubEnabled(const ProfilerStubs* stub) {
  return stub != nullptr && stub->enabled();
}

void registerProfilerStubsAdapters() {
  // Check for CUDA stubs
  const ProfilerStubs* cuda_stubs = cudaStubs();
  if (isStubEnabled(cuda_stubs)) {
    ProfilerBackendRegistry::registerBackend(
        c10::DeviceType::CUDA,
        std::make_unique<ProfilerStubsAdapter>(
            cuda_stubs,
            c10::DeviceType::CUDA,
            "CUDA (via ProfilerStubs)"));
  }

  // Check for ITT stubs (Intel VTune)
  const ProfilerStubs* itt_stubs = ittStubs();
  if (isStubEnabled(itt_stubs)) {
    // ITT is an annotation backend, not a device backend
    // It could be registered separately if needed
  }

  // Check for PrivateUse1 stubs (custom devices like OpenReg, NPU, etc.)
  const ProfilerStubs* privateuse1_stubs = privateuse1Stubs();
  if (isStubEnabled(privateuse1_stubs)) {
    ProfilerBackendRegistry::registerBackend(
        c10::DeviceType::PrivateUse1,
        std::make_unique<ProfilerStubsAdapter>(
            privateuse1_stubs,
            c10::DeviceType::PrivateUse1,
            "PrivateUse1 (via ProfilerStubs)"));
  }
}

} // namespace torch::profiler::impl
