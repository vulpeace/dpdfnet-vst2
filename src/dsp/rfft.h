#ifndef SHERPA_ONNX_CSRC_RFFT_H_
#define SHERPA_ONNX_CSRC_RFFT_H_

#include <cstdint>
#include <vector>

namespace sherpa_onnx {

struct RfftComplex {
  float re;
  float im;
};

class Rfft {
 public:
  explicit Rfft(int32_t n);

  void Forward(const float *in, float *out) const;
  void Inverse(const float *in, float *out) const;

  int32_t Size() const { return n_; }
  int32_t NumBins() const { return n_ / 2 + 1; }

 private:
  void CpxFft(const RfftComplex *in, RfftComplex *out, bool inverse) const;
  void Work(RfftComplex *out, const RfftComplex *in, int32_t fstride,
            int32_t in_stride, const int32_t *factors, bool inverse) const;
  void Butterfly2(RfftComplex *out, int32_t fstride, int32_t m,
                  bool inverse) const;
  void Butterfly4(RfftComplex *out, int32_t fstride, int32_t m,
                  bool inverse) const;
  void ButterflyGeneric(RfftComplex *out, int32_t fstride, int32_t m,
                        int32_t p, bool inverse) const;

  int32_t n_ = 0;
  int32_t m_ = 0;

  std::vector<int32_t> factors_;
  std::vector<RfftComplex> twiddles_;
  std::vector<RfftComplex> half_twiddles_;

  mutable std::vector<RfftComplex> packed_;
  mutable std::vector<RfftComplex> spectrum_;
  mutable std::vector<RfftComplex> bfly_;
};
}

#endif
