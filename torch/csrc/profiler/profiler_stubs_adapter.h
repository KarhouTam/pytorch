#pragma once

#include <torch/csrc/profiler/backend_interface.h>
#include <torch/csrc/profiler/stubs/base.h>
#include <memory>
#include <string>

namespace torch::profiler::impl {

/**
 * Adapter that bridges legacy ProfilerStubs to the new ProfilerBackend interface.
 * 
 * This allows existing out-of-tree backends that implemented ProfilerStubs
 * (e.g., via registerPrivateUse1Methods) to work with the new unified
 * profiler backend system without modification.
 * 
 * The adapter automatically wraps ProfilerStubs and exposes them through
 * the ProfilerBackend interface, maintaining backward compatibility.
 * 
 * Example: OpenReg implements ProfilerStubs and calls registerPrivateUse1Methods().
 * This adapter automatically wraps it so the new profiler system can use it.
 */
class TORCH_API ProfilerStubsAdapter : public ProfilerBackendInterface {
 public:
  /**
   * Create an adapter for the given ProfilerStubs implementation.
   * 
   * @param stubs Pointer to ProfilerStubs (must remain valid for adapter lifetime)
   * @param device_type Device type this backend handles
   * @param name Human-readable name for this backend
   */
  ProfilerStubsAdapter(
      const ProfilerStubs* stubs,
      c10::DeviceType device_type,
      std::string name);

  ~ProfilerStubsAdapter() override = default;

  // ProfilerBackendInterface implementation
  c10::DeviceType deviceType() const override { return device_type_; }
  std::string name() const override { return name_; }

  void prepare(
      const ProfilerConfig& config,
      const std::set<ActivityType>& activities) override;

  void start() override;
  void stop() override;

  bool isAvailable() const override;
  bool supportsActivity(ActivityType activity) const override;

  std::unordered_map<std::string, std::string> getResults() override;
  void synchronize() override;

 private:
  const ProfilerStubs* stubs_;
  c10::DeviceType device_type_;
  std::string name_;
  ProfilerConfig config_;
  std::set<ActivityType> activities_;
  bool is_recording_ = false;
};

/**
 * Automatically create ProfilerBackend adapters for any registered ProfilerStubs.
 * 
 * This function should be called during profiler initialization to detect
 * and wrap any legacy ProfilerStubs that were registered via:
 * - registerCUDAMethods()
 * - registerITTMethods()
 * - registerPrivateUse1Methods()
 * 
 * The adapters are automatically registered with ProfilerBackendRegistry.
 */
TORCH_API void registerProfilerStubsAdapters();

} // namespace torch::profiler::impl
