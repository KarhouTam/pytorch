#pragma once

#include <torch/csrc/profiler/backend_interface.h>
#include <torch/csrc/profiler/kineto_shim.h>

namespace torch::profiler::impl {

/**
 * Kineto-based profiler backend for CUDA devices.
 * 
 * This wraps the existing Kineto infrastructure and implements
 * the ProfilerBackendInterface to work with the new architecture.
 */
class TORCH_API KinetoProfilerBackend : public ProfilerBackendInterface {
 public:
  explicit KinetoProfilerBackend(c10::DeviceType device_type);
  ~KinetoProfilerBackend() override = default;

  c10::DeviceType deviceType() const override;
  std::string name() const override;

  void prepare(
      const ProfilerConfig& config,
      const std::set<ActivityType>& activities) override;

  void start() override;
  void stop() override;

  bool isAvailable() const override;
  bool supportsActivity(ActivityType activity) const override;

  std::unordered_map<std::string, std::string> getResults() override;
  bool exportTrace(const std::string& path) override;
  int64_t deviceElapsedUs(const void* event_ptr) const override;
  void synchronize() override;

 private:
  c10::DeviceType device_type_;
  std::string backend_name_;
  bool is_prepared_{false};
  bool is_running_{false};
  
  // Store trace for export
  kineto::ActivityTraceWrapper trace_;
};

/**
 * CPU-only profiler backend.
 * 
 * This provides basic CPU profiling without requiring Kineto.
 */
class TORCH_API CPUProfilerBackend : public ProfilerBackendInterface {
 public:
  CPUProfilerBackend();
  ~CPUProfilerBackend() override = default;

  c10::DeviceType deviceType() const override;
  std::string name() const override;

  void prepare(
      const ProfilerConfig& config,
      const std::set<ActivityType>& activities) override;

  void start() override;
  void stop() override;

  bool isAvailable() const override;
  bool supportsActivity(ActivityType activity) const override;

  std::unordered_map<std::string, std::string> getResults() override;

 private:
  bool is_running_{false};
};

/**
 * Fallback profiler backend for PrivateUse1 devices.
 * 
 * This provides a simple implementation that uses CPU timestamps
 * for device operations, similar to KINETO_PRIVATEUSE1_FALLBACK.
 * 
 * Out-of-tree backends should replace this with their own implementation
 * that provides actual device-side profiling.
 */
class TORCH_API FallbackPrivateUse1Backend : public ProfilerBackendInterface {
 public:
  FallbackPrivateUse1Backend();
  ~FallbackPrivateUse1Backend() override = default;

  c10::DeviceType deviceType() const override;
  std::string name() const override;

  void prepare(
      const ProfilerConfig& config,
      const std::set<ActivityType>& activities) override;

  void start() override;
  void stop() override;

  bool isAvailable() const override;
  bool supportsActivity(ActivityType activity) const override;

  std::unordered_map<std::string, std::string> getResults() override;

 private:
  bool is_running_{false};
  std::string backend_name_;
};

// Register default backends
TORCH_API void registerDefaultProfilerBackends();

} // namespace torch::profiler::impl
