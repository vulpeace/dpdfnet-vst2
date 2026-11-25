#include "StftProcessor.h"
#include <cmath>
#include <algorithm>

namespace dpdfvst {

static std::vector<float> VorbisWindow(int32_t n){
  constexpr float kPi = 3.14159265358979323846f;
  std::vector<float> w(n);
  float half=n/2.0f;
  for(int32_t i=0;i<n;++i){ float s=std::sin(0.5f*kPi*(i+0.5f)/half); w[i]=std::sin(0.5f*kPi*s*s); }
  return w;
}
static std::vector<float> HannWindow(int32_t n){
  std::vector<float> w(n);
  const double kPi=3.14159265358979323846;
  for(int32_t i=0;i<n;++i) w[i]=float(0.5*(1.0-std::cos(2.0*kPi*i/n)));
  return w;
}
static std::vector<float> HammingWindow(int32_t n){
  std::vector<float> w(n);
  const double kPi=3.14159265358979323846;
  for(int32_t i=0;i<n;++i) w[i]=float(0.54-0.46*std::cos(2.0*kPi*i/(n-1)));
  return w;
}
static std::vector<float> BlackmanWindow(int32_t n){
  std::vector<float> w(n);
  const double kPi=3.14159265358979323846;
  for(int32_t i=0;i<n;++i) w[i]=float(0.42-0.50*std::cos(2.0*kPi*i/(n-1))+0.08*std::cos(4.0*kPi*i/(n-1)));
  return w;
}

std::vector<float> StftProcessor::makeWindow(const std::string& type, int32_t len){
  if(type=="vorbis") return VorbisWindow(len);
  if(type=="hann_sqrt"){ auto w=HannWindow(len); for(auto &v:w) v=std::sqrt(v); return w; }
  if(type=="hann") return HannWindow(len);
  if(type=="hamming") return HammingWindow(len);
  if(type=="blackman") return BlackmanWindow(len);
  return VorbisWindow(len);
}

StftProcessor::StftProcessor(const DpdfMeta& meta)
 : meta_(meta), fft_(meta.n_fft),
   window_(makeWindow(meta.window_type, meta.window_length)),
   analysis_(meta.window_length, 0.f),
   ola_(meta.window_length, 0.f),
   fftIn_(std::max(meta.window_length, meta.n_fft), 0.f),
   fftOut_(2*(meta.n_fft/2+1), 0.f),
   enhOut_(2*(meta.n_fft/2+1), 0.f),
   ifftOut_(std::max(meta.window_length, meta.n_fft), 0.f) {}

void StftProcessor::reset(){
  std::fill(analysis_.begin(), analysis_.end(), 0.f);
  std::fill(ola_.begin(), ola_.end(), 0.f);
  pending_.clear();
  started_=false;
}

} // namespace dpdfvst
