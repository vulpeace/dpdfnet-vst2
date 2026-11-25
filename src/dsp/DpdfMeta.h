#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace dpdfvst {

struct DpdfMeta {
  int32_t version = 1;
  int32_t sample_rate = 48000;
  int32_t n_fft = 960;
  int32_t hop_length = 480;
  int32_t window_length = 960;
  bool normalized = false;
  bool center = true;
  std::string window_type = "vorbis";
  std::string pad_mode = "reflect";
  int32_t freq_bins = 481;
  int32_t erb_bins = 32;
  int32_t spec_bins = 96;
  int32_t state_size = 0;
  int32_t erb_norm_state_size = 0;
  int32_t spec_norm_state_size = 0;
  std::string profile;
  std::vector<float> erb_norm_init;
  std::vector<float> spec_norm_init;
  std::vector<int64_t> spec_shape;   // [1,1,481,2]
  std::vector<int64_t> state_shape;
};

} // namespace dpdfvst
