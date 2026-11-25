#include "DpdfModel.h"
#include "dpdfnet8_48khz_hr/generated_model.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#endif

namespace dpdfvst {

struct DpdfModel::NativeModel {
  const dpdf_native_api_v1* api;
  dpdf_native_model* model = nullptr;
  ~NativeModel() { api->destroy(model); }
};

static bool hashMatches(std::vector<float>& weights, const char* expected) {
  BCRYPT_ALG_HANDLE algorithm = nullptr;
  if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
    return false;
  std::array<unsigned char, 32> hash{};
  const auto bytes = static_cast<ULONG>(weights.size() * sizeof(float));
  const auto status = BCryptHash(algorithm, nullptr, 0,
                                 reinterpret_cast<unsigned char*>(weights.data()), bytes,
                                 hash.data(), static_cast<ULONG>(hash.size()));
  BCryptCloseAlgorithmProvider(algorithm, 0);
  if (status < 0) return false;
  constexpr char hex[] = "0123456789abcdef";
  for (std::size_t i = 0; i < hash.size(); ++i) {
    if (expected[2 * i] != hex[hash[i] >> 4] || expected[2 * i + 1] != hex[hash[i] & 15])
      return false;
  }
  return true;
}

DpdfModel::DpdfModel(const std::filesystem::path& weightsDirectory) {
  const auto* api = dpdfnet8_48khz_hr_get_api(DPDF_NATIVE_ABI_VERSION);
  if (!api || !api->preset_supported(DPDF_PRESET_INT8_SELECTIVE)) return;
  const auto path = weightsDirectory / (std::string(api->model_name) + ".weights.f32");
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  const auto bytes = static_cast<std::streamsize>(api->weight_count * sizeof(float));
  if (!input || input.tellg() != bytes) return;
  std::vector<float> weights(api->weight_count);
  if (!input.seekg(0).read(reinterpret_cast<char*>(weights.data()), bytes) ||
      !hashMatches(weights, api->weights_sha256)) return;
  auto native = std::make_unique<NativeModel>();
  native->api = api;
  native->model = api->create(weights.data(), weights.size(), DPDF_PRESET_INT8_SELECTIVE);
  if (!native->model) return;
  meta_.sample_rate = static_cast<int32_t>(api->sample_rate);
  meta_.hop_length = static_cast<int32_t>(api->hop_size);
  meta_.state_size = static_cast<int32_t>(api->state_size);
  native_ = std::move(native);
  valid_ = true;
}

static Ort::SessionOptions sessionOptions(int32_t threads) {
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(threads);
  options.SetInterOpNumThreads(threads);
  options.SetExecutionMode(ORT_SEQUENTIAL);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  options.DisableProfiling();
  options.AddConfigEntry("session.intra_op.allow_spinning", "0");
  options.AddConfigEntry("session.set_denormal_as_zero", "1");
  return options;
}

static std::wstring ToWide(const std::string& s) {
#ifdef _WIN32
  int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
  std::wstring w(n-1, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
  return w;
#else
  return std::wstring(s.begin(), s.end());
#endif
}

static bool SplitToFloats(const std::string& s, std::vector<float>& out) {
  out.clear();
  std::stringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ',')) {
    // trim
    size_t a = token.find_first_not_of(" \t\r\n");
    size_t b = token.find_last_not_of(" \t\r\n");
    if (a == std::string::npos) continue;
    token = token.substr(a, b - a + 1);
    if (token.empty()) continue;
    try { out.push_back(std::stof(token)); } catch (...) { return false; }
  }
  return true;
}

static std::string GetMetaStr(Ort::Session* sess, const char* key) {
  Ort::ModelMetadata md = sess->GetModelMetadata();
  Ort::AllocatorWithDefaultOptions alloc;
  try {
    auto v = md.LookupCustomMetadataMapAllocated(key, alloc);
    if (v) return std::string(v.get());
  } catch (...) {}
  return {};
}

DpdfModel::DpdfModel(const std::string& onnxPath, int32_t numThreads)
    : numThreads_(numThreads) {
  valid_ = init(onnxPath);
}

DpdfModel::DpdfModel(const void* data, size_t size, int32_t numThreads)
    : numThreads_(numThreads) {
  valid_ = initFromMemory(data, size);
}

DpdfModel::~DpdfModel() = default;

bool DpdfModel::init(const std::string& path) {
  try {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING);
    auto options = sessionOptions(numThreads_);

#ifdef _WIN32
    auto wpath = ToWide(path);
    sess_ = std::make_unique<Ort::Session>(*env_, wpath.c_str(), options);
#else
    sess_ = std::make_unique<Ort::Session>(*env_, path.c_str(), options);
#endif
    // input/output names
    Ort::AllocatorWithDefaultOptions alloc;
    size_t nIn = sess_->GetInputCount();
    size_t nOut = sess_->GetOutputCount();
    for (size_t i = 0; i < nIn; ++i) {
      auto name = sess_->GetInputNameAllocated(i, alloc);
      inputNames_.push_back(name.get());
    }
    for (auto& s : inputNames_) inputNamesPtr_.push_back(s.c_str());
    for (size_t i = 0; i < nOut; ++i) {
      auto name = sess_->GetOutputNameAllocated(i, alloc);
      outputNames_.push_back(name.get());
    }
    for (auto& s : outputNames_) outputNamesPtr_.push_back(s.c_str());

    if (nIn != 2 || nOut != 2) {
      error_ = "Expected 2 inputs/2 outputs";
      return false;
    }

    // metadata
    auto getStr = [&](const char* k) { return GetMetaStr(sess_.get(), k); };
    auto getInt = [&](const char* k, int32_t def, bool required) -> int32_t {
      auto v = getStr(k);
      if (v.empty()) {
        if (required) throw std::runtime_error(std::string("missing meta ")+k);
        return def;
      }
      return std::atoi(v.c_str());
    };

    std::string v;
    v = getStr("model_type");
    if (v != "dpdfnet") { error_ = "model_type != dpdfnet: " + v; return false; }
    meta_.profile = getStr("profile");
    meta_.sample_rate = getInt("sample_rate", 0, true);
    meta_.n_fft = getInt("n_fft", 0, true);
    meta_.hop_length = getInt("hop_length", 0, true);
    meta_.window_length = getInt("window_length", 0, true);
    meta_.window_type = getStr("window_type");
    if (meta_.window_type.empty()) meta_.window_type = "vorbis";
    meta_.pad_mode = getStr("pad_mode");
    meta_.freq_bins = getInt("freq_bins", 0, true);
    meta_.erb_bins = getInt("erb_bins", 0, true);
    meta_.spec_bins = getInt("spec_bins", 0, true);
    meta_.state_size = getInt("state_size", 0, true);
    meta_.erb_norm_state_size = getInt("erb_norm_state_size", 0, true);
    meta_.spec_norm_state_size = getInt("spec_norm_state_size", 0, true);
    int32_t normalized = getInt("normalized", 0, false);
    int32_t center = getInt("center", 1, false);
    meta_.normalized = normalized != 0;
    meta_.center = center != 0;
    v = getStr("version");
    if (!v.empty()) meta_.version = std::atoi(v.c_str());

    std::string erbInit = getStr("erb_norm_init");
    std::string specInit = getStr("spec_norm_init");
    if (!SplitToFloats(erbInit, meta_.erb_norm_init)) { error_="bad erb_norm_init"; return false; }
    if (!SplitToFloats(specInit, meta_.spec_norm_init)) { error_="bad spec_norm_init"; return false; }

    // shapes
    auto specShape = sess_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto stateShape = sess_->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
    meta_.spec_shape = specShape;
    meta_.state_shape = stateShape;

    // basic validation
    if (meta_.sample_rate != 48000) {
      // allow but warn; we will resample
    }
    return true;
  } catch (const Ort::Exception& e) {
    error_ = std::string("ORT: ") + e.what();
    return false;
  } catch (const std::exception& e) {
    error_ = e.what();
    return false;
  }
}

bool DpdfModel::initFromMemory(const void* data, size_t size) {
  try {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING);
    sess_ = std::make_unique<Ort::Session>(*env_, data, size, sessionOptions(numThreads_));

    Ort::AllocatorWithDefaultOptions alloc;
    size_t nIn = sess_->GetInputCount();
    size_t nOut = sess_->GetOutputCount();
    for (size_t i = 0; i < nIn; ++i) {
      auto name = sess_->GetInputNameAllocated(i, alloc);
      inputNames_.push_back(name.get());
    }
    for (auto& s : inputNames_) inputNamesPtr_.push_back(s.c_str());
    for (size_t i = 0; i < nOut; ++i) {
      auto name = sess_->GetOutputNameAllocated(i, alloc);
      outputNames_.push_back(name.get());
    }
    for (auto& s : outputNames_) outputNamesPtr_.push_back(s.c_str());

    if (nIn != 2 || nOut != 2) {
      error_ = "Expected 2 inputs/2 outputs";
      return false;
    }

    auto getStr = [&](const char* k) { return GetMetaStr(sess_.get(), k); };
    auto getInt = [&](const char* k, int32_t def, bool required) -> int32_t {
      auto v = getStr(k);
      if (v.empty()) {
        if (required) throw std::runtime_error(std::string("missing meta ")+k);
        return def;
      }
      return std::atoi(v.c_str());
    };

    std::string v;
    v = getStr("model_type");
    if (v != "dpdfnet") { error_ = "model_type != dpdfnet: " + v; return false; }
    meta_.profile = getStr("profile");
    meta_.sample_rate = getInt("sample_rate", 0, true);
    meta_.n_fft = getInt("n_fft", 0, true);
    meta_.hop_length = getInt("hop_length", 0, true);
    meta_.window_length = getInt("window_length", 0, true);
    meta_.window_type = getStr("window_type");
    if (meta_.window_type.empty()) meta_.window_type = "vorbis";
    meta_.pad_mode = getStr("pad_mode");
    meta_.freq_bins = getInt("freq_bins", 0, true);
    meta_.erb_bins = getInt("erb_bins", 0, true);
    meta_.spec_bins = getInt("spec_bins", 0, true);
    meta_.state_size = getInt("state_size", 0, true);
    meta_.erb_norm_state_size = getInt("erb_norm_state_size", 0, true);
    meta_.spec_norm_state_size = getInt("spec_norm_state_size", 0, true);
    int32_t normalized = getInt("normalized", 0, false);
    int32_t center = getInt("center", 1, false);
    meta_.normalized = normalized != 0;
    meta_.center = center != 0;
    v = getStr("version");
    if (!v.empty()) meta_.version = std::atoi(v.c_str());

    std::string erbInit = getStr("erb_norm_init");
    std::string specInit = getStr("spec_norm_init");
    if (!SplitToFloats(erbInit, meta_.erb_norm_init)) { error_="bad erb_norm_init"; return false; }
    if (!SplitToFloats(specInit, meta_.spec_norm_init)) { error_="bad spec_norm_init"; return false; }

    auto specShape = sess_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto stateShape = sess_->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
    meta_.spec_shape = specShape;
    meta_.state_shape = stateShape;
    return true;
  } catch (const Ort::Exception& e) {
    error_ = std::string("ORT mem: ") + e.what();
    return false;
  } catch (const std::exception& e) {
    error_ = e.what();
    return false;
  }
}

std::vector<float> DpdfModel::createInitState() const {
  std::vector<float> s(meta_.state_size, 0.f);
  if (native_) {
    native_->api->init_state(s.data());
    return s;
  }
  size_t n1 = std::min<size_t>(meta_.erb_norm_init.size(), s.size());
  std::copy(meta_.erb_norm_init.begin(), meta_.erb_norm_init.begin()+n1, s.begin());
  size_t off = meta_.erb_norm_state_size;
  size_t n2 = std::min<size_t>(meta_.spec_norm_init.size(), s.size()>off ? s.size()-off : 0);
  if (off < s.size())
    std::copy(meta_.spec_norm_init.begin(), meta_.spec_norm_init.begin()+n2, s.begin()+off);
  return s;
}

bool DpdfModel::processFrame(const float* specIn, float* specOut,
                             std::vector<float>& state) const {
  if (native_) {
    const auto* end = specIn + native_->api->spectrum_size;
    if (!std::all_of(specIn, end, [](float x) { return std::isfinite(x); })) return false;
    return native_->api->process(native_->model, specIn, state.data(), specOut, state.data()) == 0;
  }
  if (!valid_ || !sess_) return false;
  try {
    size_t specSize = (size_t)meta_.spec_shape[2] * (size_t)meta_.spec_shape[3]; // 481*2=962 or generic
    // For ort tensor we need shape as stored: [1,1,481,2]
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value specTensor = Ort::Value::CreateTensor<float>(memInfo, const_cast<float*>(specIn), specSize, meta_.spec_shape.data(), meta_.spec_shape.size());
    Ort::Value stateTensor = Ort::Value::CreateTensor<float>(memInfo, state.data(), state.size(), meta_.state_shape.data(), meta_.state_shape.size());
    std::array<Ort::Value,2> inputs{std::move(specTensor), std::move(stateTensor)};
    auto outputs = sess_->Run(Ort::RunOptions{nullptr}, inputNamesPtr_.data(), inputs.data(), 2, outputNamesPtr_.data(), 2);
    // outputs[0] = enhanced spec, outputs[1]= new state
    const float* enh = outputs[0].GetTensorData<float>();
    std::copy(enh, enh + specSize, specOut);
    const float* newState = outputs[1].GetTensorData<float>();
    std::copy(newState, newState + state.size(), state.data());
    return true;
  } catch (const Ort::Exception& e) {
    // In audio thread avoid logging
    return false;
  }
}

} // namespace dpdfvst
