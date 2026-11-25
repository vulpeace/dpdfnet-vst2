#pragma once
#include "DpdfModel.h"
#include "StftProcessor.h"
#include "resample.h"
#include <vector>
#include <memory>
#include <string>

namespace dpdfvst {

// High-level denoiser: resample mono audio to 48 kHz, denoise, and resample back.
// Single instance processes one mono stream (stateful).
class DpdfDenoiser {
public:
  explicit DpdfDenoiser(const std::string& onnxPath);
  DpdfDenoiser(const void* data, size_t size);
  explicit DpdfDenoiser(const std::filesystem::path& weightsDirectory);
  ~DpdfDenoiser();

  bool isReady() const { return model_ && model_->isValid() && stft_; }
  bool usesNative() const { return model_ && model_->usesNative(); }
  const std::string& error() const { return error_; }
  const DpdfMeta* meta() const { return model_ ? &model_->meta() : nullptr; }
  int32_t latencySamplesAtHostRate(double hostSampleRate) const;

  // Call when host sample rate changes
  void setHostSampleRate(int32_t sr);
  void reset();

  // Process interleaved? We process mono. Caller should downmix stereo to mono, then duplicate.
  // input: host-rate samples, n frames. Output appended to out (host-rate).
  void process(const float* in, int32_t n, std::vector<float>& out);
  void flush(std::vector<float>& out); // call on suspend

  // For VST: latency in host samples (resampled)
  int32_t hostLatency() const { return hostLatency_; }

private:
  std::unique_ptr<DpdfModel> model_;
  std::unique_ptr<StftProcessor> stft_;
  std::vector<float> state_;
  std::string error_;

  int32_t hostSr_ = 48000;
  std::unique_ptr<sherpa_onnx::LinearResample> inResampler_;  // host -> 48k
  std::unique_ptr<sherpa_onnx::LinearResample> outResampler_; // 48k -> host
  int32_t hostLatency_ = 0;
  void updateResamplers();
  void updateLatency();
  void prepare();
};

} // namespace dpdfvst
