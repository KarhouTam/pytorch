#include <torch/csrc/profiler/backend_interface.h>

#include <c10/util/Exception.h>
#include <mutex>

namespace torch::profiler::impl {

namespace {
std::mutex& getRegistryMutex() {
  static std::mutex registry_mutex;
  return registry_mutex;
}
} // namespace

// Static registry storage
std::unordered_map<c10::DeviceType, std::unique_ptr<ProfilerBackendInterface>>&
ProfilerBackendRegistry::getRegistry() {
  static std::unordered_map<c10::DeviceType, std::unique_ptr<ProfilerBackendInterface>>
      registry;
  return registry;
}

void ProfilerBackendRegistry::registerBackend(
    c10::DeviceType device_type,
    std::unique_ptr<ProfilerBackendInterface> backend) {
  TORCH_CHECK(backend != nullptr, "Cannot register null profiler backend");
  TORCH_CHECK(
      backend->deviceType() == device_type,
      "Backend device type mismatch: expected ",
      c10::DeviceTypeName(device_type),
      " but got ",
      c10::DeviceTypeName(backend->deviceType()));

  std::lock_guard<std::mutex> lock(getRegistryMutex());
  auto& registry = getRegistry();

  if (registry.find(device_type) != registry.end()) {
    TORCH_WARN(
        "Profiler backend for ",
        c10::DeviceTypeName(device_type),
        " is being replaced. This may indicate multiple registrations.");
  }

  registry[device_type] = std::move(backend);
}

ProfilerBackendInterface* ProfilerBackendRegistry::getBackend(
    c10::DeviceType device_type) {
  std::lock_guard<std::mutex> lock(getRegistryMutex());
  auto& registry = getRegistry();
  auto it = registry.find(device_type);
  if (it != registry.end()) {
    return it->second.get();
  }
  return nullptr;
}

bool ProfilerBackendRegistry::hasBackend(c10::DeviceType device_type) {
  std::lock_guard<std::mutex> lock(getRegistryMutex());
  auto& registry = getRegistry();
  return registry.find(device_type) != registry.end();
}

void ProfilerBackendRegistry::unregisterBackend(c10::DeviceType device_type) {
  std::lock_guard<std::mutex> lock(getRegistryMutex());
  auto& registry = getRegistry();
  registry.erase(device_type);
}

std::vector<c10::DeviceType> ProfilerBackendRegistry::getRegisteredDevices() {
  std::lock_guard<std::mutex> lock(getRegistryMutex());
  auto& registry = getRegistry();
  std::vector<c10::DeviceType> devices;
  devices.reserve(registry.size());
  for (const auto& pair : registry) {
    devices.push_back(pair.first);
  }
  return devices;
}

// ProfilerBackendRegistrar implementation
ProfilerBackendRegistrar::ProfilerBackendRegistrar(
    c10::DeviceType device_type,
    BackendFactory factory)
    : device_type_(device_type) {
  ProfilerBackendRegistry::registerBackend(device_type, factory());
}

ProfilerBackendRegistrar::~ProfilerBackendRegistrar() {
  // Note: We don't automatically unregister in destructor to avoid
  // issues with static destruction order. Backends typically live
  // for the entire program lifetime.
}

} // namespace torch::profiler::impl
