#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <c10/core/Device.h>
#include <torch/csrc/profiler/orchestration/observer.h>

namespace torch::profiler::impl {

// Forward declarations
struct ProfilerResult;

/**
 * Abstract interface for device-specific profiler backends.
 * 
 * This interface allows different devices (CUDA, XPU, PrivateUse1, etc.) 
 * to provide their own profiling implementations without modifying 
 * PyTorch core code.
 * 
 * Out-of-tree backends can implement this interface and register 
 * it via ProfilerBackendRegistry.
 */
class TORCH_API ProfilerBackendInterface {
 public:
  virtual ~ProfilerBackendInterface() = default;

  /**
   * Returns the device type this backend is responsible for.
   */
  virtual c10::DeviceType deviceType() const = 0;

  /**
   * Returns a human-readable name for this backend (e.g., "CUDA", "NPU").
   */
  virtual std::string name() const = 0;

  /**
   * Called when profiling is being prepared.
   * This is where backends should initialize their profiling infrastructure.
   * 
   * @param config The profiler configuration
   * @param activities Set of activities to profile
   */
  virtual void prepare(
      const ProfilerConfig& config,
      const std::set<ActivityType>& activities) = 0;

  /**
   * Called when profiling should start recording.
   * This is where backends should start capturing events.
   */
  virtual void start() = 0;

  /**
   * Called when profiling should stop recording.
   * This is where backends should stop capturing events and prepare results.
   */
  virtual void stop() = 0;

  /**
   * Called to check if this backend is available on the current system.
   * For example, CUDA backend would check if CUDA is available.
   */
  virtual bool isAvailable() const = 0;

  /**
   * Called to check if this backend supports the given activity type.
   */
  virtual bool supportsActivity(ActivityType activity) const = 0;

  /**
   * Retrieves profiling results after stop() is called.
   * Results can include device-specific events, kernel traces, etc.
   * 
   * @return Map of metadata key-value pairs containing profiling results
   */
  virtual std::unordered_map<std::string, std::string> getResults() = 0;

  /**
   * Optional: Export trace to a file in backend-specific format.
   * Returns true if export was successful.
   */
  virtual bool exportTrace(const std::string& path) {
    // Default implementation - no-op
    return false;
  }

  /**
   * Optional: Get device-specific elapsed time for an event.
   * Returns time in microseconds, or -1 if not available.
   */
  virtual int64_t deviceElapsedUs(const void* event_ptr) const {
    return -1;
  }

  /**
   * Optional: Synchronize device to ensure all operations are complete.
   * This is useful before finalizing profiling results.
   */
  virtual void synchronize() {
    // Default implementation - no-op
  }
};

/**
 * Registry for profiler backends.
 * 
 * This allows out-of-tree devices to register their profiler 
 * implementations without modifying PyTorch core.
 * 
 * Usage example for a custom backend:
 * 
 * ```cpp
 * class MyDeviceProfilerBackend : public ProfilerBackendInterface {
 *   // Implement interface methods...
 * };
 * 
 * // In device initialization code:
 * ProfilerBackendRegistry::registerBackend(
 *     c10::DeviceType::PrivateUse1,
 *     std::make_unique<MyDeviceProfilerBackend>()
 * );
 * ```
 */
class TORCH_API ProfilerBackendRegistry {
 public:
  /**
   * Register a profiler backend for a specific device type.
   * 
   * @param device_type The device type (e.g., PrivateUse1)
   * @param backend Unique pointer to the backend implementation
   */
  static void registerBackend(
      c10::DeviceType device_type,
      std::unique_ptr<ProfilerBackendInterface> backend);

  /**
   * Get the profiler backend for a specific device type.
   * Returns nullptr if no backend is registered.
   */
  static ProfilerBackendInterface* getBackend(c10::DeviceType device_type);

  /**
   * Check if a backend is registered for a device type.
   */
  static bool hasBackend(c10::DeviceType device_type);

  /**
   * Unregister a backend (useful for testing or dynamic unloading).
   */
  static void unregisterBackend(c10::DeviceType device_type);

  /**
   * Get all registered backends.
   */
  static std::vector<c10::DeviceType> getRegisteredDevices();

 private:
  static std::unordered_map<c10::DeviceType, std::unique_ptr<ProfilerBackendInterface>>&
  getRegistry();
};

/**
 * Helper class for automatic backend registration using RAII.
 * 
 * Usage:
 * ```cpp
 * static ProfilerBackendRegistrar my_backend_registrar(
 *     c10::DeviceType::PrivateUse1,
 *     []() { return std::make_unique<MyDeviceProfilerBackend>(); }
 * );
 * ```
 */
class TORCH_API ProfilerBackendRegistrar {
 public:
  using BackendFactory = std::function<std::unique_ptr<ProfilerBackendInterface>()>;

  ProfilerBackendRegistrar(
      c10::DeviceType device_type,
      BackendFactory factory);

  ~ProfilerBackendRegistrar();

 private:
  c10::DeviceType device_type_;
};

} // namespace torch::profiler::impl
