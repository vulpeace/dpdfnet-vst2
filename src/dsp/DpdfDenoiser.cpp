#include "DpdfDenoiser.h"
#include <cmath>
#include <algorithm>

namespace dpdfvst {

DpdfDenoiser::DpdfDenoiser(const std::string& onnxPath) {
  model_ = std::make_unique<DpdfModel>(onnxPath, 1);
  prepare();
}

DpdfDenoiser::DpdfDenoiser(const void* data, size_t size) {
  model_ = std::make_unique<DpdfModel>(data, size, 1);
  prepare();
}

DpdfDenoiser::DpdfDenoiser(const std::filesystem::path& weightsDirectory) {
  model_ = std::make_unique<DpdfModel>(weightsDirectory);
  prepare();
}

void DpdfDenoiser::prepare() {
  if (!model_->isValid()) { error_ = model_->error(); return; }
  stft_ = std::make_unique<StftProcessor>(model_->meta());
  state_ = model_->createInitState();
  hostSr_ = 48000;
  updateResamplers();
  updateLatency();
}

DpdfDenoiser::~DpdfDenoiser() = default;

void DpdfDenoiser::updateResamplers(){
  if (!model_) return;
  int32_t modelSr = model_->meta().sample_rate;
  if (hostSr_ == modelSr) { inResampler_.reset(); outResampler_.reset(); return; }
  float minFreq = std::min<float>(hostSr_, modelSr);
  float cutoff = 0.99f * 0.5f * minFreq;
  int32_t width = 6;
  inResampler_ = std::make_unique<sherpa_onnx::LinearResample>(hostSr_, modelSr, cutoff, width);
  outResampler_ = std::make_unique<sherpa_onnx::LinearResample>(modelSr, hostSr_, cutoff, width);
}

void DpdfDenoiser::updateLatency(){
  if (!stft_) { hostLatency_=0; return; }
  int32_t modelLat = stft_->latencySamples(); // e.g. 960
  if (!outResampler_) hostLatency_ = modelLat;
  else {
    // latency at host rate = modelLat * hostSr / modelSr
    double r = double(hostSr_) / double(model_->meta().sample_rate);
    hostLatency_ = int32_t(std::round(modelLat * r));
  }
}

void DpdfDenoiser::setHostSampleRate(int32_t sr){
  if (sr==hostSr_) return;
  hostSr_ = sr;
  updateResamplers();
  updateLatency();
  reset();
}

void DpdfDenoiser::reset(){
  if (stft_) stft_->reset();
  if (model_) state_ = model_->createInitState();
  if (inResampler_) inResampler_->Reset();
  if (outResampler_) outResampler_->Reset();
}

void DpdfDenoiser::process(const float* in, int32_t n, std::vector<float>& out){
  if (!isReady() || n<=0) return;
  // 1. resample host->model
  std::vector<float> modelIn;
  const float* modelPtr = in;
  int32_t modelN = n;
  std::vector<float> resampledIn;
  if (inResampler_) {
    inResampler_->Resample(in, n, false, &resampledIn);
    modelPtr = resampledIn.data();
    modelN = (int32_t)resampledIn.size();
    // if no output yet (needs more), just return (latency)
    if (modelN==0) return;
  }

  // 2. Spectral denoising
  std::vector<float> modelOut;
  auto fn = [&](float* spec, size_t specSize, float* enhanced){
    // specSize == 962 for 48k
    if (!model_->processFrame(spec, enhanced, state_)) {
      // bypass on failure
      std::copy(spec, spec+specSize, enhanced);
    }
  };
  stft_->processBlock(modelPtr, modelN, fn, modelOut);
  if (modelOut.empty()) return;

  // 3. resample model->host
  if (outResampler_) {
    std::vector<float> hostOut;
    outResampler_->Resample(modelOut.data(), (int32_t)modelOut.size(), false, &hostOut);
    out.insert(out.end(), hostOut.begin(), hostOut.end());
  } else {
    out.insert(out.end(), modelOut.begin(), modelOut.end());
  }
}

void DpdfDenoiser::flush(std::vector<float>& out){
  if (!isReady()) return;
  // flush resampler tail first?
  std::vector<float> modelOut;
  auto fn = [&](float* spec, size_t specSize, float* enhanced){
    model_->processFrame(spec, enhanced, state_);
  };
  // flush pending STFT
  stft_->flush(fn, modelOut);

  // handle remaining resampler
  if (inResampler_) {
    std::vector<float> tail;
    float dummy=0;
    inResampler_->Resample(&dummy, 0, true, &tail);
    if (!tail.empty()) {
      // need to process tail through STFT as well? simple: append to modelOut via direct flush already handled?
      // For simplicity, process tail in one go
      std::vector<float> tailOut;
      stft_->processBlock(tail.data(), (int32_t)tail.size(), fn, tailOut);
      modelOut.insert(modelOut.end(), tailOut.begin(), tailOut.end());
      std::vector<float> tailOut2;
      stft_->flush(fn, tailOut2);
      modelOut.insert(modelOut.end(), tailOut2.begin(), tailOut2.end());
    }
  }

  if (modelOut.empty()) { reset(); return; }

  if (outResampler_) {
    std::vector<float> hostOut, tail;
    outResampler_->Resample(modelOut.data(), (int32_t)modelOut.size(), false, &hostOut);
    float dummy=0;
    outResampler_->Resample(&dummy,0,true,&tail);
    hostOut.insert(hostOut.end(), tail.begin(), tail.end());
    out.insert(out.end(), hostOut.begin(), hostOut.end());
  } else {
    out.insert(out.end(), modelOut.begin(), modelOut.end());
  }
  reset();
}

int32_t DpdfDenoiser::latencySamplesAtHostRate(double hostSampleRate) const {
  if (!model_) return 0;
  int32_t modelLat = stft_ ? stft_->latencySamples() : 0;
  double r = hostSampleRate / double(model_->meta().sample_rate);
  return int32_t(std::round(modelLat * r));
}

} // namespace dpdfvst
