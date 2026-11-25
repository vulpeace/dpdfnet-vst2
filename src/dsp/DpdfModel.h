#pragma once
#include "DpdfMeta.h"
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <onnxruntime/onnxruntime_cxx_api.h>

namespace dpdfvst {

class DpdfModel {
public:
  explicit DpdfModel(const std::string& onnxPath, int32_t numThreads = 1);
  // Construct from memory buffer (embedded resource)
  DpdfModel(const void* data, size_t size, int32_t numThreads = 1);
  explicit DpdfModel(const std::filesystem::path& weightsDirectory);
  ~DpdfModel();

  bool isValid() const { return valid_; }
  bool usesNative() const { return native_ != nullptr; }
  const DpdfMeta& meta() const { return meta_; }
  const std::string& error() const { return error_; }

  // create zero state with erb/spec norm init
  std::vector<float> createInitState() const;

  // specIn size == freq_bins*2 (e.g. 962), state is mutable, specOut same size
  bool processFrame(const float* specIn, float* specOut,
                    std::vector<float>& state) const;

private:
  bool init(const std::string& path);
  bool initFromMemory(const void* data, size_t size);

  DpdfMeta meta_;
  bool valid_ = false;
  std::string error_;

  struct NativeModel;
  std::unique_ptr<NativeModel> native_;
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> sess_;
  std::vector<std::string> inputNames_;
  std::vector<const char*> inputNamesPtr_;
  std::vector<std::string> outputNames_;
  std::vector<const char*> outputNamesPtr_;
  int32_t numThreads_ = 1;
};

} // namespace dpdfvst
