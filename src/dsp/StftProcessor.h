#pragma once
#include <cstdint>
#include <vector>
#include <memory>
#include <string>
#include "DpdfMeta.h"
#include "rfft.h"

namespace dpdfvst {

// STFT processor at fixed model sample rate (48000). Handles analysis window, RFFT, overlap-add.
// Resampling is done outside this class by DpdfDenoiser.
class StftProcessor {
public:
  explicit StftProcessor(const DpdfMeta& meta);
  void reset();

  int32_t sampleRate() const { return meta_.sample_rate; }
  int32_t hopLength() const { return meta_.hop_length; }
  int32_t windowLength() const { return meta_.window_length; }
  int32_t nFft() const { return meta_.n_fft; }
  int32_t latencySamples() const { return meta_.window_length; } // first hop is discarded

  // Process pending input (at model SR) synchronously.
  // processFrame is called for each hop: specIn size = nFft+2 (interleaved re/im), specOut same
  template <typename ProcessFn>
  void processBlock(const float* input, int32_t n, ProcessFn processFrame, std::vector<float>& output) {
    pending_.insert(pending_.end(), input, input + n);
    int consumed = 0;
    while ((int32_t)pending_.size() - consumed >= meta_.hop_length) {
      processHop(pending_.data() + consumed, processFrame, output);
      consumed += meta_.hop_length;
    }
    if (consumed) pending_.erase(pending_.begin(), pending_.begin()+consumed);
  }

  template <typename ProcessFn>
  void flush(ProcessFn processFrame, std::vector<float>& output) {
    if (!pending_.empty()) {
      std::vector<float> padded(meta_.hop_length, 0.f);
      std::copy(pending_.begin(), pending_.end(), padded.begin());
      processHop(padded.data(), processFrame, output);
      pending_.clear();
    }
    if (started_) {
      std::vector<float> zero(meta_.hop_length, 0.f);
      processHop(zero.data(), processFrame, output);
    }
  }

private:
  template <typename ProcessFn>
  void processHop(const float* hop, ProcessFn processFrame, std::vector<float>& output) {
    // shift analysis buffer
    std::move(analysis_.begin() + meta_.hop_length, analysis_.end(), analysis_.begin());
    std::copy(hop, hop + meta_.hop_length, analysis_.end() - meta_.hop_length);
    for (int32_t i=0;i<meta_.window_length;++i) fftIn_[i]=analysis_[i]*window_[i];
    // zero pad if n_fft > window_length (not for 48k case where equal)
    if ((int32_t)fftIn_.size() > meta_.window_length) {
      std::fill(fftIn_.begin()+meta_.window_length, fftIn_.end(), 0.f);
    }
    fft_.Forward(fftIn_.data(), fftOut_.data());
    processFrame(fftOut_.data(), fftOut_.size(), enhOut_.data());
    fft_.Inverse(enhOut_.data(), ifftOut_.data());
    // overlap add
    std::move(ola_.begin()+meta_.hop_length, ola_.end(), ola_.begin());
    std::fill(ola_.end()-meta_.hop_length, ola_.end(), 0.f);
    for (int32_t i=0;i<meta_.window_length;++i) ola_[i] += ifftOut_[i]*window_[i];
    if (!started_) { started_=true; return; }
    output.insert(output.end(), ola_.begin(), ola_.begin()+meta_.hop_length);
  }

  static std::vector<float> makeWindow(const std::string& type, int32_t len);

  DpdfMeta meta_;
  sherpa_onnx::Rfft fft_;
  std::vector<float> window_;
  std::vector<float> analysis_;
  std::vector<float> ola_;
  std::vector<float> pending_;
  std::vector<float> fftIn_;
  std::vector<float> fftOut_;
  std::vector<float> enhOut_;
  std::vector<float> ifftOut_;
  bool started_ = false;
};

} // namespace dpdfvst
